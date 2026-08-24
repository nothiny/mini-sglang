from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

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
    _received_metadata: torch.Tensor | None

    @property
    def local_recv_stop(self) -> int:
        return self.local_recv_start + self.local_hidden_states.shape[0]

    def wait(self) -> None:
        for work in self._works:
            work.wait()
        self._works = ()
        if self._received_metadata is not None:
            self.expert_ids.copy_(self._received_metadata[:, 0])
            self.routed_weights.copy_(self._received_metadata[:, 1])
            self._received_metadata = None


def _expert_owners_and_local_ids(
    route_expert_ids: torch.Tensor,
    route_token_ids: torch.Tensor,
    *,
    world_size: int,
    num_experts: int,
    placement: Literal["contiguous", "round-robin"],
    replicated_experts: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    local_experts = num_experts // world_size
    if placement == "contiguous":
        owners = torch.div(route_expert_ids, local_experts, rounding_mode="floor")
        local_ids = route_expert_ids - owners * local_experts
    elif placement == "round-robin":
        owners = route_expert_ids.remainder(world_size)
        local_ids = torch.div(route_expert_ids, world_size, rounding_mode="floor")
    else:
        raise ValueError(f"Unsupported expert placement: {placement}")
    for replica_idx, expert_id in enumerate(replicated_experts):
        replicated = route_expert_ids == expert_id
        owners = torch.where(replicated, route_token_ids.remainder(world_size), owners)
        local_ids = torch.where(
            replicated,
            torch.full_like(local_ids, local_experts + replica_idx),
            local_ids,
        )
    return owners, local_ids.to(torch.int32)


def build_replicated_dispatch_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_experts: int,
    placement: Literal["contiguous", "round-robin"] = "contiguous",
    replicated_experts: Tuple[int, ...] = (),
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

    origin_token_ids = torch.arange(
        rank,
        hidden_states.shape[0],
        world_size,
        device=hidden_states.device,
    )
    top_k = topk_ids.shape[1]
    route_token_ids = origin_token_ids.repeat_interleave(top_k)
    route_expert_ids = topk_ids[origin_token_ids].reshape(-1).to(torch.int64)
    route_weights = topk_weights[origin_token_ids].reshape(-1)
    owners, local_expert_ids = _expert_owners_and_local_ids(
        route_expert_ids,
        route_token_ids,
        world_size=world_size,
        num_experts=num_experts,
        placement=placement,
        replicated_experts=replicated_experts,
    )
    if owners.device.type == "cpu" and bool(torch.any((owners < 0) | (owners >= world_size))):
        raise ValueError("router selected an expert outside the configured range")

    order = torch.argsort(owners, stable=True)
    owners = owners[order]
    route_token_ids = route_token_ids[order]
    route_expert_ids = route_expert_ids[order]
    route_weights = route_weights[order]
    send_hidden = hidden_states[route_token_ids]
    local_expert_ids = local_expert_ids[order]
    send_counts = torch.bincount(owners, minlength=world_size).to(torch.int64)
    return send_hidden, local_expert_ids, route_weights, route_token_ids, send_counts


def build_static_replicated_dispatch_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_experts: int,
    placement: Literal["contiguous", "round-robin"] = "contiguous",
    replicated_experts: Tuple[int, ...] = (),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack routes into fixed destination buckets without host-visible counts."""

    if num_experts % world_size:
        raise ValueError("num_experts must be divisible by expert parallel size")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("top-k weights and ids must have identical shapes")
    if hidden_states.shape[0] != topk_ids.shape[0]:
        raise ValueError("hidden states and routing tensors must have the same token count")

    origin_token_ids = torch.arange(
        rank,
        hidden_states.shape[0],
        world_size,
        device=hidden_states.device,
    )
    top_k = topk_ids.shape[1]
    route_token_ids = origin_token_ids.repeat_interleave(top_k)
    route_expert_ids = topk_ids[origin_token_ids].reshape(-1).to(torch.int64)
    route_weights = topk_weights[origin_token_ids].reshape(-1)
    owners, local_ids = _expert_owners_and_local_ids(
        route_expert_ids,
        route_token_ids,
        world_size=world_size,
        num_experts=num_experts,
        placement=placement,
        replicated_experts=replicated_experts,
    )
    if owners.device.type == "cpu" and bool(torch.any((owners < 0) | (owners >= world_size))):
        raise ValueError("router selected an expert outside the configured range")

    capacity = max(1, ((hidden_states.shape[0] + world_size - 1) // world_size) * top_k)
    order = torch.argsort(owners, stable=True)
    sorted_owners = owners[order]
    counts = torch.bincount(sorted_owners, minlength=world_size)
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(order.numel(), device=order.device) - starts[sorted_owners]
    slots = sorted_owners * capacity + positions
    total_slots = world_size * capacity

    send_hidden = hidden_states.new_zeros((total_slots, hidden_states.shape[1]))
    send_ids = topk_ids.new_full((total_slots,), -1, dtype=torch.int32)
    send_weights = topk_weights.new_zeros((total_slots,))
    send_origins = topk_ids.new_full((total_slots,), -1, dtype=torch.int64)
    send_hidden[slots] = hidden_states[route_token_ids[order]]
    send_ids[slots] = local_ids[order]
    send_weights[slots] = route_weights[order]
    send_origins[slots] = route_token_ids[order]
    return send_hidden, send_ids, send_weights, send_origins, capacity


class ExpertParallelDispatcher:
    """Variable-size token dispatch/combine for replicated TP inputs."""

    def __init__(
        self,
        num_experts: int,
        rank: int,
        world_size: int,
        *,
        dispatch_mode: Literal["dynamic", "static"] = "dynamic",
        placement: Literal["contiguous", "round-robin"] = "contiguous",
        replicated_experts: Tuple[int, ...] = (),
    ) -> None:
        if world_size < 2:
            raise ValueError("ExpertParallelDispatcher requires at least two ranks")
        if num_experts % world_size:
            raise ValueError("num_experts must be divisible by expert parallel size")
        self.num_experts = num_experts
        self.rank = rank
        self.world_size = world_size
        self.dispatch_mode = dispatch_mode
        self.placement = placement
        self.replicated_experts = replicated_experts

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertParallelDispatch:
        if self.dispatch_mode == "static":
            send_hidden, send_ids, send_weights, origin_ids, capacity = (
                build_static_replicated_dispatch_inputs(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    rank=self.rank,
                    world_size=self.world_size,
                    num_experts=self.num_experts,
                    placement=self.placement,
                    replicated_experts=self.replicated_experts,
                )
            )
            send_splits = recv_splits = (capacity,) * self.world_size
        else:
            send_hidden, send_ids, send_weights, origin_ids, send_counts = (
                build_replicated_dispatch_inputs(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    rank=self.rank,
                    world_size=self.world_size,
                    num_experts=self.num_experts,
                    placement=self.placement,
                    replicated_experts=self.replicated_experts,
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
        send_metadata = torch.stack((send_ids.float(), send_weights.float()), dim=1)
        recv_metadata = send_metadata.new_empty((recv_routes, 2))

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
                output_split_sizes=None if self.dispatch_mode == "static" else list(recv_splits),
                input_split_sizes=None if self.dispatch_mode == "static" else list(send_splits),
                async_op=True,
            ),
            dist.all_to_all_single(
                recv_metadata,
                send_metadata,
                output_split_sizes=None if self.dispatch_mode == "static" else list(recv_splits),
                input_split_sizes=None if self.dispatch_mode == "static" else list(send_splits),
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
            _received_metadata=recv_metadata,
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
            output_split_sizes=(
                None if self.dispatch_mode == "static" else list(dispatch.send_splits)
            ),
            input_split_sizes=(
                None if self.dispatch_mode == "static" else list(dispatch.recv_splits)
            ),
            async_op=True,
        )
        work.wait()

        output = expert_outputs.new_zeros((num_tokens, expert_outputs.shape[1]))
        valid = dispatch.origin_token_ids >= 0
        safe_origins = dispatch.origin_token_ids.clamp_min(0)
        returned.mul_(valid.unsqueeze(1))
        output.index_add_(0, safe_origins, returned)
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output


__all__ = [
    "ExpertParallelDispatch",
    "ExpertParallelDispatcher",
    "build_replicated_dispatch_inputs",
    "build_static_replicated_dispatch_inputs",
]
