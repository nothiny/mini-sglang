from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.moe import get_moe_backend_config
from minisgl.moe.weights import quantize_expert_weight, quantize_expert_weight_fp8
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _shard_tensor(
    key: str,
    value: torch.Tensor,
    r: int,
    n: int,
    num_kv_heads: int,
    *,
    expert_parallel: bool = False,
):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if expert_parallel and _EXPERT_PATTERN.match(key):
        return value
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def _local_expert_index(
    expert_idx: int,
    *,
    ep_size: int,
    ep_rank: int,
    local_experts: int,
    placement: str,
) -> int | None:
    if ep_size == 1:
        return expert_idx
    if placement == "contiguous":
        expert_start = ep_rank * local_experts
        if expert_start <= expert_idx < expert_start + local_experts:
            return expert_idx - expert_start
        return None
    if placement == "round-robin":
        if expert_idx % ep_size == ep_rank:
            return expert_idx // ep_size
        return None
    raise ValueError(f"Unsupported expert placement: {placement}")


def _local_expert_indices(
    expert_idx: int,
    *,
    ep_size: int,
    ep_rank: int,
    local_experts: int,
    placement: str,
    replicated_experts: Tuple[int, ...],
) -> Tuple[int, ...]:
    indices: list[int] = []
    base_index = _local_expert_index(
        expert_idx,
        ep_size=ep_size,
        ep_rank=ep_rank,
        local_experts=local_experts,
        placement=placement,
    )
    if base_index is not None:
        indices.append(base_index)
    if expert_idx in replicated_experts:
        indices.append(local_experts + replicated_experts.index(expert_idx))
    return tuple(indices)


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()
    moe_config = get_moe_backend_config()
    ep_size = moe_config.expert_parallel_size
    local_experts = config.num_experts // ep_size if ep_size > 1 else config.num_experts
    packed_local_experts = local_experts + (
        len(moe_config.replicated_experts) if ep_size > 1 else 0
    )
    stage_experts_on_cpu = moe_config.expert_offload or moe_config.expert_quantization != "none"
    checkpoint_device = torch.device("cpu") if stage_experts_on_cpu else device

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(checkpoint_device)) as f:
            for checkpoint_name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if checkpoint_name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                name = checkpoint_name.removeprefix("language_model.")
                expert_info = _get_expert_stack_info(name) if config.is_moe else None
                if expert_info is not None and ep_size > 1:
                    _, expert_idx = expert_info
                    if not _local_expert_indices(
                        expert_idx,
                        ep_size=ep_size,
                        ep_rank=moe_config.expert_parallel_rank,
                        local_experts=local_experts,
                        placement=moe_config.expert_placement,
                        replicated_experts=moe_config.replicated_experts,
                    ):
                        continue
                raw = f.get_tensor(checkpoint_name)
                tensor = _shard_tensor(
                    name,
                    raw,
                    tp_info.rank,
                    tp_info.size,
                    config.num_kv_heads,
                    expert_parallel=ep_size > 1,
                )
                del raw
                if not (stage_experts_on_cpu and expert_info is not None):
                    tensor = tensor.to(device)

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (stack_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = stack_info
                    local_expert_indices = _local_expert_indices(
                        expert_idx,
                        ep_size=ep_size,
                        ep_rank=moe_config.expert_parallel_rank,
                        local_experts=local_experts,
                        placement=moe_config.expert_placement,
                        replicated_experts=moe_config.replicated_experts,
                    )
                    assert local_expert_indices
                    slots = expert_buf.setdefault(packed_key, {})
                    for local_expert_idx in local_expert_indices:
                        slots[local_expert_idx] = out[1]
                    if len(slots) != packed_local_experts:
                        continue
                    experts = [slots[idx] for idx in range(packed_local_experts)]
                    del expert_buf[packed_key]
                    packed = torch.stack(experts, dim=0)
                    if moe_config.expert_quantization == "int8":
                        packed, scale = quantize_expert_weight(packed)
                    elif moe_config.expert_quantization == "fp8":
                        packed, scale = quantize_expert_weight_fp8(packed)
                    else:
                        scale = None
                    yield packed_key, packed
                    if scale is not None:
                        yield f"{packed_key}_scale", scale
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
