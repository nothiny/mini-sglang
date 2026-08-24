from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("workspace capacity must be positive")
    return 1 << (value - 1).bit_length()


@dataclass
class FusedMoeWorkspace:
    """Reusable temporary storage for one fused MoE shape."""

    capacity_tokens: int
    num_experts: int
    top_k: int
    intermediate_size_x2: int
    hidden_size: int
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    intermediate: torch.Tensor
    activated: torch.Tensor
    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_pad: torch.Tensor
    cumsum_buffer: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        capacity_tokens: int,
        num_experts: int,
        top_k: int,
        intermediate_size_x2: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> FusedMoeWorkspace:
        routed_capacity = capacity_tokens * top_k
        max_block_size = 64
        min_block_size = 16
        max_sorted = routed_capacity + (num_experts + 1) * (max_block_size - 1)
        max_blocks = (max_sorted + min_block_size - 1) // min_block_size
        return cls(
            capacity_tokens=capacity_tokens,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size_x2=intermediate_size_x2,
            hidden_size=hidden_size,
            topk_weights=torch.empty((capacity_tokens, top_k), dtype=torch.float32, device=device),
            topk_ids=torch.empty((capacity_tokens, top_k), dtype=torch.int32, device=device),
            intermediate=torch.empty(
                routed_capacity * max(intermediate_size_x2, hidden_size),
                dtype=dtype,
                device=device,
            ),
            activated=torch.empty(
                (routed_capacity, intermediate_size_x2 // 2),
                dtype=dtype,
                device=device,
            ),
            sorted_ids=torch.empty(max_sorted, dtype=torch.int32, device=device),
            expert_ids=torch.empty(max_blocks, dtype=torch.int32, device=device),
            num_tokens_post_pad=torch.empty(1, dtype=torch.int32, device=device),
            cumsum_buffer=torch.empty(num_experts + 2, dtype=torch.int32, device=device),
        )

    def views(
        self, num_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_tokens > self.capacity_tokens:
            raise ValueError("workspace is smaller than the requested token count")
        routed_tokens = num_tokens * self.top_k
        topk_weights = self.topk_weights[:num_tokens]
        topk_ids = self.topk_ids[:num_tokens]
        intermediate1 = self.intermediate[: routed_tokens * self.intermediate_size_x2].view(
            num_tokens, self.top_k, self.intermediate_size_x2
        )
        intermediate2 = self.activated[:routed_tokens]
        intermediate3 = self.intermediate[: routed_tokens * self.hidden_size].view(
            num_tokens, self.top_k, self.hidden_size
        )
        return topk_weights, topk_ids, intermediate1, intermediate2, intermediate3


_WorkspaceKey = Tuple[torch.device, torch.dtype, int, int, int, int]


class FusedMoeWorkspaceCache:
    """Grow-only workspace cache safe for CUDA graph captures.

    Superseded allocations are retained because an already captured CUDA graph
    may still refer to their addresses.
    """

    def __init__(self) -> None:
        self._current: Dict[_WorkspaceKey, FusedMoeWorkspace] = {}
        self._retired: list[FusedMoeWorkspace] = []
        self.allocation_count = 0

    def get(
        self,
        *,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        intermediate_size_x2: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        cache: bool,
    ) -> FusedMoeWorkspace:
        capacity = _next_power_of_two(num_tokens) if cache else num_tokens
        key = (
            device,
            dtype,
            num_experts,
            top_k,
            intermediate_size_x2,
            hidden_size,
        )
        workspace = self._current.get(key) if cache else None
        if workspace is not None and workspace.capacity_tokens >= num_tokens:
            return workspace

        new_workspace = FusedMoeWorkspace.allocate(
            capacity_tokens=capacity,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size_x2=intermediate_size_x2,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
        )
        self.allocation_count += 1
        if cache:
            if workspace is not None:
                self._retired.append(workspace)
            self._current[key] = new_workspace
        return new_workspace


__all__ = ["FusedMoeWorkspace", "FusedMoeWorkspaceCache"]
