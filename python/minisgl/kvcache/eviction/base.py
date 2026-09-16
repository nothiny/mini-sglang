from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EvictionContext:
    """Cache-wide information available for one victim selection."""

    now_ns: int
    requested_tokens: int
    free_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class EvictionCandidate:
    """Read-only Radix node metadata exposed to an eviction policy."""

    node_id: int
    segment_tokens: int
    prefix_depth: int
    kv_bytes: int
    ref_count: int


def estimate_recompute_cost(prefix_depth: int, segment_tokens: int) -> float:
    """Estimate prefill work for a segment following ``prefix_depth`` tokens."""

    return float(segment_tokens) * (prefix_depth + segment_tokens / 2)


class BaseEvictionPolicy(ABC):
    """Event-driven victim-selection interface for Radix prefix caches.

    Implementations own replacement metadata only. They must not mutate the
    Radix tree, reference counts, or KV page allocations.
    """

    name = "base"
    uses_ghost_feedback = False

    def on_insert(self, node_id: int, now_ns: int) -> None:
        pass

    def on_access(self, node_id: int, now_ns: int) -> None:
        pass

    def on_split(
        self,
        old_node_id: int,
        prefix_node_id: int,
        suffix_node_id: int,
    ) -> None:
        pass

    def on_evict(
        self,
        node_id: int,
        prefix_hash: int,
        recompute_cost: float,
        now_ns: int,
    ) -> None:
        pass

    def on_ghost_hit(
        self,
        prefix_hash: int,
        recompute_cost: float,
        now_ns: int,
    ) -> None:
        pass

    @abstractmethod
    def select_victim(
        self,
        candidates: Sequence[EvictionCandidate],
        context: EvictionContext,
    ) -> int:
        """Return the ``node_id`` of the next candidate to evict."""


__all__ = [
    "BaseEvictionPolicy",
    "EvictionCandidate",
    "EvictionContext",
    "estimate_recompute_cost",
]
