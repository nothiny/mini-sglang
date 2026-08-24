import functools
from typing import Dict, Tuple

import torch
from minisgl.moe import BaseMoeBackend
from minisgl.moe.config import MoeBackendConfig
from minisgl.moe.expert_parallel import ExpertParallelDispatcher
from minisgl.moe.weights import ExpertResidentCache, ResidentExpertWeights
from minisgl.moe.workspace import FusedMoeWorkspace, FusedMoeWorkspaceCache
from minisgl.utils import div_ceil


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
    out_weights: torch.Tensor | None = None,
    out_ids: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sgl_kernel import topk_softmax

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    M, _ = hidden_states.shape
    topk_weights = out_weights
    if topk_weights is None:
        topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = out_ids
    if topk_ids is None:
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    if topk_weights.shape != (M, topk) or topk_weights.dtype != torch.float32:
        raise ValueError("out_weights has an incompatible shape or dtype")
    if topk_ids.shape != (M, topk) or topk_ids.dtype != torch.int32:
        raise ValueError("out_ids has an incompatible shape or dtype")
    topk_softmax(topk_weights, topk_ids, gating_output.float(), renormalize)
    if renormalize:
        topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
    if num_token_non_padded is not None:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = -1
    return topk_weights, topk_ids


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    workspace: FusedMoeWorkspace | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = (
        torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
        if workspace is None
        else workspace.sorted_ids[:max_num_tokens_padded]
    )
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
    expert_ids = (
        torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
        if workspace is None
        else workspace.expert_ids[:max_num_m_blocks]
    )
    num_tokens_post_pad = (
        torch.empty((1), dtype=torch.int32, device=topk_ids.device)
        if workspace is None
        else workspace.num_tokens_post_pad
    )
    cumsum_buffer = (
        torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)
        if workspace is None
        else workspace.cumsum_buffer[: num_experts + 2]
    )
    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
) -> Dict[str, int]:
    if M <= E:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    if M <= 4 * E:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }


def get_direct_config(M: int, E: int, N: int, K: int, topk: int) -> Dict[str, int]:
    del M, E, topk
    return {
        "BLOCK_SIZE_N": 64 if N >= 64 else 32,
        "BLOCK_SIZE_K": 64 if K % 64 == 0 else 32,
    }


def get_moe_config_candidates(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
) -> Tuple[Dict[str, int], ...]:
    candidates = (
        get_default_config(M, E, N, K, topk),
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        },
        {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        },
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
    )
    unique: list[Dict[str, int]] = []
    seen: set[Tuple[Tuple[str, int], ...]] = set()
    for candidate in candidates:
        key = tuple(sorted(candidate.items()))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


@functools.lru_cache(maxsize=256)
def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
) -> Dict[str, int]:
    E, _, N = w2_shape
    config = get_default_config(M, E, N, w1_shape[2], top_k)
    return config


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    workspace: FusedMoeWorkspace | None = None,
    use_direct: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    grouped_config: Dict[str, int] | None = None,
) -> torch.Tensor:
    from minisgl.kernel import (
        direct_moe_gemv_triton,
        fused_moe_kernel_triton,
        moe_sum_reduce_triton,
    )
    from minisgl.layers import gelu_and_mul, silu_and_mul

    padded_size = 0
    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
    )
    if workspace is None:
        workspace = FusedMoeWorkspace.allocate(
            capacity_tokens=M,
            num_experts=E,
            top_k=topk_ids.shape[1],
            intermediate_size_x2=N,
            hidden_size=w2.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    _, _, intermediate_cache1, intermediate_cache2, intermediate_cache3 = workspace.views(M)
    compute_type = hidden_states.dtype

    out_hidden_states = hidden_states
    curr_hidden_states = hidden_states
    tokens_num, _ = curr_hidden_states.shape
    begin_token_idx, end_token_idx = 0, num_tokens

    intermediate_cache1 = intermediate_cache1[:tokens_num]
    intermediate_cache2 = intermediate_cache2[: tokens_num * topk_ids.shape[1]]
    intermediate_cache3 = intermediate_cache3[:tokens_num]
    config = grouped_config or get_config_func(tokens_num)

    curr_topk_ids = topk_ids[begin_token_idx:end_token_idx]
    curr_topk_weights = topk_weights[begin_token_idx:end_token_idx]

    FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
    if activation not in FN_MAP:
        raise ValueError(f"Unsupported activation function: {activation}")

    if use_direct:
        direct_config1 = get_direct_config(M, E, N, hidden_states.shape[1], topk_ids.shape[1])
        direct_moe_gemv_triton(
            curr_hidden_states,
            w1,
            intermediate_cache1,
            curr_topk_weights,
            curr_topk_ids,
            w1_scale,
            input_is_routed=False,
            mul_routed_weight=apply_router_weight_on_input,
            config=direct_config1,
        )
        FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
        direct_config2 = get_direct_config(
            M,
            E,
            w2.shape[1],
            w2.shape[2],
            topk_ids.shape[1],
        )
        direct_moe_gemv_triton(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            curr_topk_weights,
            curr_topk_ids,
            w2_scale,
            input_is_routed=True,
            mul_routed_weight=not apply_router_weight_on_input,
            config=direct_config2,
        )
        moe_sum_reduce_triton(intermediate_cache3, out_hidden_states)
        return out_hidden_states

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        curr_topk_ids,
        config["BLOCK_SIZE_M"],
        E,
        workspace=workspace,
    )

    fused_moe_kernel_triton(
        curr_hidden_states,
        w1,
        intermediate_cache1,
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk_ids.shape[1],
        config,
        compute_type=compute_type,
        B_scale=w1_scale,
    )
    FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
    fused_moe_kernel_triton(
        intermediate_cache2,
        w2,
        (intermediate_cache3),
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=compute_type,
        B_scale=w2_scale,
    )

    moe_sum_reduce_triton(
        intermediate_cache3,
        out_hidden_states[begin_token_idx:end_token_idx],
    )
    return out_hidden_states


class FusedMoe(BaseMoeBackend):
    def __init__(self, config: MoeBackendConfig | None = None) -> None:
        self.config = config or MoeBackendConfig()
        self.workspace_cache = FusedMoeWorkspaceCache()
        self._ep_dispatcher: ExpertParallelDispatcher | None = None
        self._resident_caches: Dict[int, ExpertResidentCache] = {}
        self._kernel_choices: Dict[Tuple[object, ...], bool] = {}
        self._grouped_config_choices: Dict[Tuple[object, ...], Dict[str, int]] = {}

    def _select_kernel(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        apply_router_weight_on_input: bool,
        workspace: FusedMoeWorkspace,
        w1_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
    ) -> Tuple[bool, Dict[str, int]]:
        num_tokens = hidden_states.shape[0]
        top_k = topk_ids.shape[1]
        default_config = try_get_optimal_moe_config(
            tuple(w1.shape),
            tuple(w2.shape),
            top_k,
            num_tokens,
        )
        heuristic_direct = num_tokens <= self.config.small_m_threshold
        if (
            not self.config.autotune
            or self.config.expert_offload
            or hidden_states.device.type != "cuda"
            or torch.cuda.is_current_stream_capturing()
        ):
            return heuristic_direct, default_config

        token_bucket = 1 << (num_tokens - 1).bit_length()
        key = (
            hidden_states.device,
            hidden_states.dtype,
            w1.dtype,
            w2.dtype,
            tuple(w1.shape),
            tuple(w2.shape),
            top_k,
            token_bucket,
            activation,
            apply_router_weight_on_input,
            w1_scale is not None,
            w2_scale is not None,
        )
        if key in self._kernel_choices:
            return self._kernel_choices[key], self._grouped_config_choices[key]

        grouped_candidates = get_moe_config_candidates(
            num_tokens,
            w1.shape[0],
            w1.shape[1],
            w1.shape[2],
            top_k,
        )
        trials = [(False, config) for config in grouped_candidates]
        if num_tokens <= max(32, self.config.small_m_threshold):
            trials.append((True, default_config))

        source = hidden_states.clone()
        trial = torch.empty_like(source)
        for use_direct, grouped_config in trials:
            for _ in range(2):
                trial.copy_(source)
                fused_experts_impl(
                    trial,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    activation,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    workspace=workspace,
                    use_direct=use_direct,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    grouped_config=grouped_config,
                )

        elapsed_ms: list[list[float]] = [[] for _ in trials]
        for round_idx in range(len(trials)):
            indices = list(range(len(trials)))
            indices = indices[round_idx:] + indices[:round_idx]
            for trial_idx in indices:
                use_direct, grouped_config = trials[trial_idx]
                started = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                started.record()
                for _ in range(2):
                    trial.copy_(source)
                    fused_experts_impl(
                        trial,
                        w1,
                        w2,
                        topk_weights,
                        topk_ids,
                        activation,
                        apply_router_weight_on_input=apply_router_weight_on_input,
                        workspace=workspace,
                        use_direct=use_direct,
                        w1_scale=w1_scale,
                        w2_scale=w2_scale,
                        grouped_config=grouped_config,
                    )
                finished.record()
                finished.synchronize()
                elapsed_ms[trial_idx].append(started.elapsed_time(finished) / 2)

        medians = []
        for samples in elapsed_ms:
            ordered = sorted(samples)
            midpoint = len(ordered) // 2
            if len(ordered) % 2:
                medians.append(ordered[midpoint])
            else:
                medians.append((ordered[midpoint - 1] + ordered[midpoint]) / 2)
        best_idx = min(range(len(trials)), key=medians.__getitem__)
        if trials[best_idx][0]:
            grouped_indices = [idx for idx, (direct, _) in enumerate(trials) if not direct]
            best_grouped_idx = min(grouped_indices, key=medians.__getitem__)
            if medians[best_idx] > 0.95 * medians[best_grouped_idx]:
                best_idx = best_grouped_idx
        use_direct, grouped_config = trials[best_idx]
        self._kernel_choices[key] = use_direct
        self._grouped_config_choices[key] = grouped_config
        return use_direct, grouped_config

    def _resolve_expert_weights(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
        expert_ids: torch.Tensor,
        device: torch.device,
    ) -> ResidentExpertWeights:
        if not self.config.expert_offload:
            return ResidentExpertWeights(w1, w2, w1_scale, w2_scale, expert_ids)
        cache_key = id(w1)
        cache = self._resident_caches.get(cache_key)
        if cache is None:
            cache = ExpertResidentCache(
                w1,
                w2,
                w1_scale,
                w2_scale,
                capacity=min(self.config.expert_cache_size, w1.shape[0]),
                device=device,
            )
            self._resident_caches[cache_key] = cache
        return cache.resolve(expert_ids)

    def _record_expert_usage(
        self,
        source_w1: torch.Tensor,
        resolved: ResidentExpertWeights,
    ) -> None:
        if not self.config.expert_offload:
            return
        self._resident_caches[id(source_w1)].record_usage(resolved.resident_slots)

    def _run_experts(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        apply_router_weight_on_input: bool,
        w1_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
        workspace: FusedMoeWorkspace | None = None,
        *,
        allow_chunking: bool = True,
    ) -> torch.Tensor:
        if self.config.expert_offload and allow_chunking:
            capacity = min(self.config.expert_cache_size, w1.shape[0])
            requested = torch.unique(topk_ids[topk_ids >= 0])
            if requested.numel() > capacity:
                return self._run_offloaded_experts_chunked(
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    activation,
                    apply_router_weight_on_input,
                    w1_scale,
                    w2_scale,
                    requested,
                    capacity,
                )
        resolved = self._resolve_expert_weights(
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_ids,
            hidden_states.device,
        )
        if workspace is None or workspace.num_experts != resolved.w1.shape[0]:
            workspace = self.workspace_cache.get(
                num_tokens=hidden_states.shape[0],
                num_experts=resolved.w1.shape[0],
                top_k=topk_ids.shape[1],
                intermediate_size_x2=resolved.w1.shape[1],
                hidden_size=resolved.w2.shape[1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                cache=self.config.enable_workspace_cache,
            )
        use_direct, grouped_config = self._select_kernel(
            hidden_states,
            resolved.w1,
            resolved.w2,
            topk_weights,
            resolved.expert_ids,
            activation,
            apply_router_weight_on_input,
            workspace,
            resolved.w1_scale,
            resolved.w2_scale,
        )
        output = fused_experts_impl(
            hidden_states,
            resolved.w1,
            resolved.w2,
            topk_weights,
            resolved.expert_ids,
            activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            workspace=workspace,
            use_direct=use_direct,
            w1_scale=resolved.w1_scale,
            w2_scale=resolved.w2_scale,
            grouped_config=grouped_config,
        )
        self._record_expert_usage(w1, resolved)
        return output

    def _run_offloaded_experts_chunked(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        apply_router_weight_on_input: bool,
        w1_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
        requested: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        """Process more active experts than fit in the resident cache.

        Routes are grouped into expert-sized waves. Each wave uses top-k=1
        execution and is accumulated back into its original token row.
        """

        source = hidden_states
        output = torch.zeros_like(source)
        flat_ids = topk_ids.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        top_k = topk_ids.shape[1]
        requested_ids = tuple(int(value) for value in requested.cpu().tolist())
        for offset in range(0, len(requested_ids), capacity):
            expert_wave = requested_ids[offset : offset + capacity]
            selected = torch.zeros_like(flat_ids, dtype=torch.bool)
            for expert_id in expert_wave:
                selected.logical_or_(flat_ids == expert_id)
            route_ids = torch.nonzero(selected, as_tuple=False).flatten()
            token_ids = torch.div(route_ids, top_k, rounding_mode="floor")
            wave_output = self._run_experts(
                source[token_ids].contiguous(),
                w1,
                w2,
                flat_weights[route_ids].view(-1, 1),
                flat_ids[route_ids].view(-1, 1),
                activation,
                apply_router_weight_on_input,
                w1_scale,
                w2_scale,
                allow_chunking=False,
            )
            output.index_add_(0, token_ids, wave_output)
        return output

    def _run_expert_parallel(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        apply_router_weight_on_input: bool,
        w1_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._ep_dispatcher is None:
            self._ep_dispatcher = ExpertParallelDispatcher(
                num_experts=self.config.expert_parallel_size * w1.shape[0],
                rank=self.config.expert_parallel_rank,
                world_size=self.config.expert_parallel_size,
            )
        dispatch = self._ep_dispatcher.dispatch(hidden_states, topk_weights, topk_ids)
        local_outputs: torch.Tensor | None = None
        if self.config.expert_parallel_overlap and dispatch.local_hidden_states.shape[0]:
            local_outputs = self._run_experts(
                dispatch.local_hidden_states,
                w1,
                w2,
                dispatch.local_routed_weights.view(-1, 1),
                dispatch.local_expert_ids.view(-1, 1),
                activation,
                apply_router_weight_on_input,
                w1_scale,
                w2_scale,
            )

        dispatch.wait()
        num_received = dispatch.hidden_states.shape[0]
        if not self.config.expert_parallel_overlap:
            if num_received:
                expert_outputs = self._run_experts(
                    dispatch.hidden_states,
                    w1,
                    w2,
                    dispatch.routed_weights.view(-1, 1),
                    dispatch.expert_ids.view(-1, 1),
                    activation,
                    apply_router_weight_on_input,
                    w1_scale,
                    w2_scale,
                )
            else:
                expert_outputs = hidden_states.new_empty((0, hidden_states.shape[1]))
        else:
            expert_outputs = hidden_states.new_empty((num_received, hidden_states.shape[1]))
            local_start = dispatch.local_recv_start
            local_stop = dispatch.local_recv_stop
            if local_outputs is not None:
                expert_outputs[local_start:local_stop].copy_(local_outputs)

            remote_hidden_parts = (
                dispatch.hidden_states[:local_start],
                dispatch.hidden_states[local_stop:],
            )
            remote_id_parts = (
                dispatch.expert_ids[:local_start],
                dispatch.expert_ids[local_stop:],
            )
            remote_weight_parts = (
                dispatch.routed_weights[:local_start],
                dispatch.routed_weights[local_stop:],
            )
            remote_count = num_received - (local_stop - local_start)
            if remote_count:
                remote_hidden = torch.cat(remote_hidden_parts)
                remote_ids = torch.cat(remote_id_parts)
                remote_weights = torch.cat(remote_weight_parts)
                remote_outputs = self._run_experts(
                    remote_hidden,
                    w1,
                    w2,
                    remote_weights.view(-1, 1),
                    remote_ids.view(-1, 1),
                    activation,
                    apply_router_weight_on_input,
                    w1_scale,
                    w2_scale,
                )
                expert_outputs[:local_start].copy_(remote_outputs[:local_start])
                expert_outputs[local_stop:].copy_(remote_outputs[local_start:])

        return self._ep_dispatcher.combine(
            expert_outputs,
            dispatch,
            num_tokens=hidden_states.shape[0],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        expected_experts = w1.shape[0] * self.config.expert_parallel_size
        if gating_output.shape[1] != expected_experts:
            raise ValueError(
                f"router has {gating_output.shape[1]} experts, but weights/config expect "
                f"{expected_experts}"
            )
        if not 0 < topk <= gating_output.shape[1]:
            raise ValueError("topk must be within the router expert count")
        workspace = self.workspace_cache.get(
            num_tokens=num_tokens,
            num_experts=w1.shape[0],
            top_k=topk,
            intermediate_size_x2=w1.shape[1],
            hidden_size=w2.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            cache=self.config.enable_workspace_cache,
        )
        topk_weights_out, topk_ids_out, _, _, _ = workspace.views(num_tokens)
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            out_weights=topk_weights_out,
            out_ids=topk_ids_out,
        )
        if self.config.expert_parallel_size > 1:
            return self._run_expert_parallel(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                apply_router_weight_on_input,
                w1_scale,
                w2_scale,
            )
        return self._run_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input,
            w1_scale,
            w2_scale,
            workspace,
        )
