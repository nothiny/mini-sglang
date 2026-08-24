from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import torch


def quantize_expert_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8 quantization for ``[E, N, K]`` weights."""

    if weight.ndim != 3 or not weight.is_floating_point():
        raise ValueError("expert weight must be a floating-point [E, N, K] tensor")
    scale = weight.abs().amax(dim=-1).float().clamp_min_(1e-8).div_(127.0)
    normalized = weight.to(dtype=torch.float32, copy=True)
    normalized.div_(scale.unsqueeze(-1)).round_().clamp_(-127, 127)
    return normalized.to(torch.int8), scale


def quantize_expert_weight_fp8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel E4M3 quantization for native FP8 grouped GEMM."""

    if weight.ndim != 3 or not weight.is_floating_point():
        raise ValueError("expert weight must be a floating-point [E, N, K] tensor")
    dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(dtype).max
    scale = weight.abs().amax(dim=-1).float().clamp_min_(1e-8).div_(fp8_max)
    normalized = weight.to(dtype=torch.float32, copy=True)
    normalized.div_(scale.unsqueeze(-1)).clamp_(-fp8_max, fp8_max)
    return normalized.to(dtype), scale


@dataclass(frozen=True)
class ResidentExpertWeights:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor | None
    w2_scale: torch.Tensor | None
    expert_ids: torch.Tensor
    resident_slots: Tuple[int, ...] = ()


@dataclass(frozen=True)
class ExpertTransfer:
    """One expert copy queued on the cache transfer stream."""

    expert_id: int
    resident_slot: int
    ready_event: torch.cuda.Event


@dataclass(frozen=True)
class ExpertLoadPlan:
    """Resident weights plus enough state to pipeline route execution."""

    resident: ResidentExpertWeights
    hit_expert_ids: Tuple[int, ...]
    hit_slots: Tuple[int, ...]
    transfers: Tuple[ExpertTransfer, ...]


class ExpertResidentCache:
    """Fixed-size GPU LRU over packed CPU expert weights."""

    def __init__(
        self,
        w1_host: torch.Tensor,
        w2_host: torch.Tensor,
        w1_scale_host: torch.Tensor | None,
        w2_scale_host: torch.Tensor | None,
        *,
        capacity: int,
        device: torch.device,
    ) -> None:
        if w1_host.device.type != "cpu" or w2_host.device.type != "cpu":
            raise ValueError("offloaded expert weights must reside on CPU")
        if w1_host.shape[0] != w2_host.shape[0]:
            raise ValueError("w1 and w2 must contain the same number of experts")
        if not 0 < capacity <= w1_host.shape[0]:
            raise ValueError("expert cache capacity must be within the expert count")
        self.num_experts = w1_host.shape[0]
        self.capacity = capacity
        self.device = device
        self.w1_host = w1_host
        self.w2_host = w2_host
        self.w1_scale_host = w1_scale_host
        self.w2_scale_host = w2_scale_host
        host_tensors = tuple(
            tensor
            for tensor in (w1_host, w2_host, w1_scale_host, w2_scale_host)
            if tensor is not None
        )
        self._direct_pinned_copy = all(tensor.is_pinned() for tensor in host_tensors)
        self._w1_staging = self._make_staging(w1_host)
        self._w2_staging = self._make_staging(w2_host)
        self._w1_scale_staging = self._make_staging(w1_scale_host)
        self._w2_scale_staging = self._make_staging(w2_scale_host)
        self.w1 = torch.empty((capacity, *w1_host.shape[1:]), dtype=w1_host.dtype, device=device)
        self.w2 = torch.empty((capacity, *w2_host.shape[1:]), dtype=w2_host.dtype, device=device)
        self.w1_scale = (
            None
            if w1_scale_host is None
            else torch.empty(
                (capacity, *w1_scale_host.shape[1:]),
                dtype=w1_scale_host.dtype,
                device=device,
            )
        )
        self.w2_scale = (
            None
            if w2_scale_host is None
            else torch.empty(
                (capacity, *w2_scale_host.shape[1:]),
                dtype=w2_scale_host.dtype,
                device=device,
            )
        )
        self._expert_to_slot: OrderedDict[int, int] = OrderedDict()
        self._free_slots = list(range(capacity))
        self._mapping = torch.full((self.num_experts,), -1, dtype=torch.int64, device=device)
        self._transfer_stream = torch.cuda.Stream(device=device)
        self._slot_last_use: list[torch.cuda.Event | None] = [None] * capacity
        self._slot_transfer_done: list[torch.cuda.Event | None] = [None] * capacity
        self.hits = 0
        self.misses = 0
        self.bytes_transferred = 0

    def requested_experts(self, expert_ids: torch.Tensor) -> Tuple[int, ...]:
        """Return the small host-side control-plane set for an offloaded layer.

        Fully resident execution never calls this method. CPU offload needs the
        concrete IDs in order to issue pinned-host CUDA copies; route filtering
        and remapping remain device-side.
        """

        valid_ids = expert_ids[expert_ids >= 0]
        if not valid_ids.numel():
            return ()
        return tuple(int(value) for value in torch.unique(valid_ids).cpu().tolist())

    def is_resident(self, expert_id: int) -> bool:
        return expert_id in self._expert_to_slot

    def prepare(
        self,
        expert_ids: torch.Tensor,
        requested: Tuple[int, ...] | None = None,
    ) -> ExpertLoadPlan:
        """Queue missing copies without stalling the current compute stream."""

        if requested is None:
            requested = self.requested_experts(expert_ids)
        if len(requested) > self.capacity:
            raise RuntimeError(
                f"A batch routes to {len(requested)} experts, but the GPU expert cache "
                f"holds only {self.capacity}; increase --moe-expert-cache-size"
            )
        requested_set = set(requested)
        missing: list[tuple[int, int]] = []
        hit_expert_ids: list[int] = []
        hit_slots: list[int] = []
        for expert_id in requested:
            slot = self._expert_to_slot.get(expert_id)
            if slot is not None:
                self.hits += 1
                self._expert_to_slot.move_to_end(expert_id)
                hit_expert_ids.append(expert_id)
                hit_slots.append(slot)
                continue
            self.misses += 1
            slot = self._claim_slot(requested_set)
            self._expert_to_slot[expert_id] = slot
            self._mapping[expert_id] = slot
            missing.append((expert_id, slot))

        transfers: list[ExpertTransfer] = []
        if missing:
            with torch.cuda.stream(self._transfer_stream):
                for expert_id, slot in missing:
                    sources = self._stage_host_expert(expert_id, slot)
                    last_use = self._slot_last_use[slot]
                    if last_use is not None:
                        self._transfer_stream.wait_event(last_use)
                    self.w1[slot].copy_(sources[0], non_blocking=True)
                    self.w2[slot].copy_(sources[1], non_blocking=True)
                    self.bytes_transferred += (
                        self.w1_host[expert_id].nbytes + self.w2_host[expert_id].nbytes
                    )
                    if self.w1_scale is not None and self.w1_scale_host is not None:
                        assert sources[2] is not None
                        self.w1_scale[slot].copy_(sources[2], non_blocking=True)
                        self.bytes_transferred += self.w1_scale_host[expert_id].nbytes
                    if self.w2_scale is not None and self.w2_scale_host is not None:
                        assert sources[3] is not None
                        self.w2_scale[slot].copy_(sources[3], non_blocking=True)
                        self.bytes_transferred += self.w2_scale_host[expert_id].nbytes
                    ready_event = torch.cuda.Event()
                    ready_event.record(self._transfer_stream)
                    self._slot_transfer_done[slot] = ready_event
                    transfers.append(ExpertTransfer(expert_id, slot, ready_event))

        resident_slots = tuple(self._expert_to_slot[expert_id] for expert_id in requested)

        valid = expert_ids >= 0
        remapped_ids = torch.where(
            valid,
            self._mapping[expert_ids.clamp_min(0).to(torch.int64)],
            torch.full_like(expert_ids, -1, dtype=torch.int64),
        ).to(torch.int32)
        return ExpertLoadPlan(
            resident=ResidentExpertWeights(
                self.w1,
                self.w2,
                self.w1_scale,
                self.w2_scale,
                remapped_ids,
                resident_slots,
            ),
            hit_expert_ids=tuple(hit_expert_ids),
            hit_slots=tuple(hit_slots),
            transfers=tuple(transfers),
        )

    def resolve(self, expert_ids: torch.Tensor) -> ResidentExpertWeights:
        """Compatibility path that waits for all requested experts."""

        plan = self.prepare(expert_ids)
        current_stream = torch.cuda.current_stream(self.device)
        if plan.transfers:
            current_stream.wait_event(plan.transfers[-1].ready_event)
        self.record_usage(plan.resident.resident_slots)
        return plan.resident

    def record_usage(self, resident_slots: Tuple[int, ...]) -> None:
        """Protect slots until kernels already queued on the current stream finish."""

        if not resident_slots:
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        for slot in resident_slots:
            self._slot_last_use[slot] = event

    def _claim_slot(self, requested: set[int]) -> int:
        if self._free_slots:
            return self._free_slots.pop(0)
        for evicted_expert, slot in list(self._expert_to_slot.items()):
            if evicted_expert in requested:
                continue
            del self._expert_to_slot[evicted_expert]
            self._mapping[evicted_expert] = -1
            return slot
        raise RuntimeError("No evictable expert cache slot is available")

    def _make_staging(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None or self._direct_pinned_copy:
            return None
        return torch.empty(
            (self.capacity, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )

    def _stage_host_expert(
        self,
        expert_id: int,
        slot: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self._direct_pinned_copy:
            return (
                self.w1_host[expert_id],
                self.w2_host[expert_id],
                None if self.w1_scale_host is None else self.w1_scale_host[expert_id],
                None if self.w2_scale_host is None else self.w2_scale_host[expert_id],
            )

        previous_transfer = self._slot_transfer_done[slot]
        if previous_transfer is not None:
            previous_transfer.synchronize()
        assert self._w1_staging is not None and self._w2_staging is not None
        self._w1_staging[slot].copy_(self.w1_host[expert_id])
        self._w2_staging[slot].copy_(self.w2_host[expert_id])
        w1_scale = None
        w2_scale = None
        if self.w1_scale_host is not None:
            assert self._w1_scale_staging is not None
            self._w1_scale_staging[slot].copy_(self.w1_scale_host[expert_id])
            w1_scale = self._w1_scale_staging[slot]
        if self.w2_scale_host is not None:
            assert self._w2_scale_staging is not None
            self._w2_scale_staging[slot].copy_(self.w2_scale_host[expert_id])
            w2_scale = self._w2_scale_staging[slot]
        return self._w1_staging[slot], self._w2_staging[slot], w1_scale, w2_scale


__all__ = [
    "ExpertLoadPlan",
    "ExpertResidentCache",
    "ExpertTransfer",
    "ResidentExpertWeights",
    "quantize_expert_weight",
    "quantize_expert_weight_fp8",
]
