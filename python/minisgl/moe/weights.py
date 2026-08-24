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


@dataclass(frozen=True)
class ResidentExpertWeights:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor | None
    w2_scale: torch.Tensor | None
    expert_ids: torch.Tensor
    resident_slots: Tuple[int, ...] = ()


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
        self.w1_host = _ensure_pinned(w1_host)
        self.w2_host = _ensure_pinned(w2_host)
        self.w1_scale_host = _ensure_pinned(w1_scale_host)
        self.w2_scale_host = _ensure_pinned(w2_scale_host)
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
        self.hits = 0
        self.misses = 0
        self.bytes_transferred = 0

    def resolve(self, expert_ids: torch.Tensor) -> ResidentExpertWeights:
        requested = tuple(
            int(expert_id) for expert_id in torch.unique(expert_ids[expert_ids >= 0]).cpu().tolist()
        )
        if len(requested) > self.capacity:
            raise RuntimeError(
                f"A batch routes to {len(requested)} experts, but the GPU expert cache "
                f"holds only {self.capacity}; increase --moe-expert-cache-size"
            )
        requested_set = set(requested)
        missing: list[tuple[int, int]] = []
        for expert_id in requested:
            slot = self._expert_to_slot.get(expert_id)
            if slot is not None:
                self.hits += 1
                self._expert_to_slot.move_to_end(expert_id)
                continue
            self.misses += 1
            slot = self._claim_slot(requested_set)
            self._expert_to_slot[expert_id] = slot
            self._mapping[expert_id] = slot
            missing.append((expert_id, slot))

        if missing:
            current_stream = torch.cuda.current_stream(self.device)
            with torch.cuda.stream(self._transfer_stream):
                for expert_id, slot in missing:
                    last_use = self._slot_last_use[slot]
                    if last_use is not None:
                        self._transfer_stream.wait_event(last_use)
                    self.w1[slot].copy_(self.w1_host[expert_id], non_blocking=True)
                    self.w2[slot].copy_(self.w2_host[expert_id], non_blocking=True)
                    self.bytes_transferred += (
                        self.w1_host[expert_id].nbytes + self.w2_host[expert_id].nbytes
                    )
                    if self.w1_scale is not None and self.w1_scale_host is not None:
                        self.w1_scale[slot].copy_(self.w1_scale_host[expert_id], non_blocking=True)
                        self.bytes_transferred += self.w1_scale_host[expert_id].nbytes
                    if self.w2_scale is not None and self.w2_scale_host is not None:
                        self.w2_scale[slot].copy_(self.w2_scale_host[expert_id], non_blocking=True)
                        self.bytes_transferred += self.w2_scale_host[expert_id].nbytes
            current_stream.wait_stream(self._transfer_stream)

        resident_slots = tuple(self._expert_to_slot[expert_id] for expert_id in requested)

        valid = expert_ids >= 0
        remapped_ids = torch.where(
            valid,
            self._mapping[expert_ids.clamp_min(0).to(torch.int64)],
            torch.full_like(expert_ids, -1, dtype=torch.int64),
        ).to(torch.int32)
        return ResidentExpertWeights(
            self.w1,
            self.w2,
            self.w1_scale,
            self.w2_scale,
            remapped_ids,
            resident_slots,
        )

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


def _ensure_pinned(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None or tensor.is_pinned():
        return tensor
    return tensor.pin_memory()


__all__ = [
    "ExpertResidentCache",
    "ResidentExpertWeights",
    "quantize_expert_weight",
]
