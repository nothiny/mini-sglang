from __future__ import annotations

import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from minisgl.moe import MoeBackendConfig
from minisgl.moe.expert_parallel import ExpertParallelDispatcher
from minisgl.moe.fused import FusedMoe
from minisgl.moe.workspace import FusedMoeWorkspace


class _ReferenceExpertParallelMoe(FusedMoe):
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
        del w2, activation, apply_router_weight_on_input, w1_scale, w2_scale
        del workspace, allow_chunking
        global_ids = topk_ids.to(torch.int64) + self.config.expert_parallel_rank * w1.shape[0]
        return (
            hidden_states
            * (global_ids + 1).to(hidden_states.dtype)
            * topk_weights.to(hidden_states.dtype)
        )


def _run_expert_parallel_rank(rank: int, world_size: int, store_path: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        hidden = torch.arange(6 * 3, dtype=torch.float32).view(6, 3)
        topk_ids = torch.tensor(
            [[0, 2], [1, 3], [2, 0], [3, 1], [0, 3], [2, 1]],
            dtype=torch.int32,
        )
        topk_weights = torch.tensor(
            [[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.5, 0.5], [0.9, 0.1], [0.6, 0.4]],
            dtype=torch.float32,
        )
        dispatcher = ExpertParallelDispatcher(num_experts=4, rank=rank, world_size=world_size)
        dispatch = dispatcher.dispatch(hidden, topk_weights, topk_ids)
        dispatch.wait()

        local_expert_count = 2
        global_expert_ids = dispatch.expert_ids.to(torch.int64) + rank * local_expert_count
        expert_outputs = (
            dispatch.hidden_states
            * (global_expert_ids + 1).to(torch.float32).unsqueeze(1)
            * dispatch.routed_weights.unsqueeze(1)
        )
        actual = dispatcher.combine(expert_outputs, dispatch, num_tokens=hidden.shape[0])
        factors = ((topk_ids.to(torch.float32) + 1) * topk_weights).sum(dim=1, keepdim=True)
        expected = hidden * factors
        torch.testing.assert_close(actual, expected)

        w1 = torch.empty(2, 4, 3)
        w2 = torch.empty(2, 3, 2)
        for overlap in (False, True):
            backend = _ReferenceExpertParallelMoe(
                MoeBackendConfig(
                    expert_parallel_size=world_size,
                    expert_parallel_rank=rank,
                    expert_parallel_overlap=overlap,
                )
            )
            actual_backend = backend._run_expert_parallel(
                hidden.clone(),
                w1,
                w2,
                topk_weights,
                topk_ids,
                "silu",
                False,
                None,
                None,
            )
            torch.testing.assert_close(actual_backend, expected)
    finally:
        dist.destroy_process_group()


def test_expert_parallel_dispatch_and_combine_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        mp.spawn(
            _run_expert_parallel_rank,
            args=(2, f"{tmpdir}/store"),
            nprocs=2,
            join=True,
        )
