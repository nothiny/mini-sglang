from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist


@dataclass
class ExpertParallelDispatch:
    hidden_states: torch.Tensor
    expert_ids: torch.Tensor
    routed_weights: torch.Tensor
    origin_token_ids: torch.Tensor
    send_splits: Tuple[int, ...]
    recv_splits: Tuple[int, ...]
    local_hidden_states: torch.Tensor
    local_expert_ids: torch.Tensor
    local_routed_weights: torch.Tensor
    local_recv_start: int
    _works: Tuple[dist.Work, ...]

    @property
    def local_recv_stop(self) -> int:
        return self.local_recv_start + self.local_hidden_states.shape[0]

    def wait(self) -> None:
        for work in self._works:
            work.wait()
        self._works = ()


def build_replicated_dispatch_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one rank's routes when all EP ranks hold the same input tokens.

    Token ``i`` is assigned to origin rank ``i % world_size``. This ensures
    every token/expert route is dispatched exactly once rather than once per
    replicated tensor-parallel rank.
    """

    if num_experts % world_size:
        raise ValueError("num_experts must be divisible by expert parallel size")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("top-k weights and ids must have identical shapes")
    if hidden_states.shape[0] != topk_ids.shape[0]:
        raise ValueError("hidden states and routing tensors must have the same token count")

    local_experts = num_experts // world_size
    token_ids = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    origin_token_ids = token_ids[token_ids.remainder(world_size) == rank]
    top_k = topk_ids.shape[1]
    route_token_ids = origin_token_ids.repeat_interleave(top_k)
    route_expert_ids = topk_ids[origin_token_ids].reshape(-1).to(torch.int64)
    route_weights = topk_weights[origin_token_ids].reshape(-1)
    owners = torch.div(route_expert_ids, local_experts, rounding_mode="floor")
    if bool(torch.any((owners < 0) | (owners >= world_size))):
        raise ValueError("router selected an expert outside the configured range")

    order = torch.argsort(owners, stable=True)
    owners = owners[order]
    route_token_ids = route_token_ids[order]
    route_expert_ids = route_expert_ids[order]
    route_weights = route_weights[order]
    send_hidden = hidden_states[route_token_ids]
    local_expert_ids = (route_expert_ids - owners * local_experts).to(torch.int32)
    send_counts = torch.bincount(owners, minlength=world_size).to(torch.int64)
    return send_hidden, local_expert_ids, route_weights, route_token_ids, send_counts


class ExpertParallelDispatcher:
    """Variable-size token dispatch/combine for replicated TP inputs."""

    def __init__(self, num_experts: int, rank: int, world_size: int) -> None:
        if world_size < 2:
            raise ValueError("ExpertParallelDispatcher requires at least two ranks")
        if num_experts % world_size:
            raise ValueError("num_experts must be divisible by expert parallel size")
        self.num_experts = num_experts
        self.rank = rank
        self.world_size = world_size

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertParallelDispatch:
        send_hidden, send_ids, send_weights, origin_ids, send_counts = (
            build_replicated_dispatch_inputs(
                hidden_states,
                topk_weights,
                topk_ids,
                rank=self.rank,
                world_size=self.world_size,
                num_experts=self.num_experts,
            )
        )

        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        send_splits = tuple(int(value) for value in send_counts.cpu().tolist())
        recv_splits = tuple(int(value) for value in recv_counts.cpu().tolist())
        recv_routes = sum(recv_splits)
        recv_hidden = hidden_states.new_empty((recv_routes, hidden_states.shape[1]))
        recv_ids = topk_ids.new_empty((recv_routes,), dtype=torch.int32)
        recv_weights = topk_weights.new_empty((recv_routes,))

        local_send_start = sum(send_splits[: self.rank])
        local_send_stop = local_send_start + send_splits[self.rank]
        # The fused expert kernel writes its input in place, so keep a private
        # copy that can be computed while collectives still read send_hidden.
        local_hidden = send_hidden[local_send_start:local_send_stop].clone()
        local_ids = send_ids[local_send_start:local_send_stop]
        local_weights = send_weights[local_send_start:local_send_stop]

        works = (
            dist.all_to_all_single(
                recv_hidden,
                send_hidden.contiguous(),
                output_split_sizes=list(recv_splits),
                input_split_sizes=list(send_splits),
                async_op=True,
            ),
            dist.all_to_all_single(
                recv_ids,
                send_ids.contiguous(),
                output_split_sizes=list(recv_splits),
                input_split_sizes=list(send_splits),
                async_op=True,
            ),
            dist.all_to_all_single(
                recv_weights,
                send_weights.contiguous(),
                output_split_sizes=list(recv_splits),
                input_split_sizes=list(send_splits),
                async_op=True,
            ),
        )

        return ExpertParallelDispatch(
            hidden_states=recv_hidden,
            expert_ids=recv_ids,
            routed_weights=recv_weights,
            origin_token_ids=origin_ids,
            send_splits=send_splits,
            recv_splits=recv_splits,
            local_hidden_states=local_hidden,
            local_expert_ids=local_ids,
            local_routed_weights=local_weights,
            local_recv_start=sum(recv_splits[: self.rank]),
            _works=works,
        )

    def combine(
        self,
        expert_outputs: torch.Tensor,
        dispatch: ExpertParallelDispatch,
        *,
        num_tokens: int,
    ) -> torch.Tensor:
        dispatch.wait()
        returned = expert_outputs.new_empty((sum(dispatch.send_splits), expert_outputs.shape[1]))
        work = dist.all_to_all_single(
            returned,
            expert_outputs.contiguous(),
            output_split_sizes=list(dispatch.send_splits),
            input_split_sizes=list(dispatch.recv_splits),
            async_op=True,
        )
        work.wait()

        output = expert_outputs.new_zeros((num_tokens, expert_outputs.shape[1]))
        output.index_add_(0, dispatch.origin_token_ids, returned)
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output


__all__ = [
    "ExpertParallelDispatch",
    "ExpertParallelDispatcher",
    "build_replicated_dispatch_inputs",
]
