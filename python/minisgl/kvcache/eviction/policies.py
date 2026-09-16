from __future__ import annotations

import math
import random
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Mapping, Sequence, Tuple, TypeVar

from .base import (
    BaseEvictionPolicy,
    EvictionCandidate,
    EvictionContext,
    estimate_recompute_cost,
)

T = TypeVar("T")


def _require_candidates(candidates: Sequence[EvictionCandidate]) -> None:
    if not candidates:
        raise ValueError("select_victim requires at least one candidate")


def _inherit_split_state(
    state: Dict[int, T],
    old_node_id: int,
    prefix_node_id: int,
    suffix_node_id: int,
) -> None:
    value = state.pop(old_node_id, None)
    if value is not None:
        state[prefix_node_id] = value
        state[suffix_node_id] = value


class LRUPolicy(BaseEvictionPolicy):
    name = "lru"

    def __init__(self) -> None:
        self.last_access_ns: Dict[int, int] = {}

    def on_insert(self, node_id: int, now_ns: int) -> None:
        self.last_access_ns[node_id] = now_ns

    def on_access(self, node_id: int, now_ns: int) -> None:
        self.last_access_ns[node_id] = now_ns

    def on_split(self, old_node_id: int, prefix_node_id: int, suffix_node_id: int) -> None:
        _inherit_split_state(self.last_access_ns, old_node_id, prefix_node_id, suffix_node_id)

    def on_evict(
        self,
        node_id: int,
        prefix_hash: int,
        recompute_cost: float,
        now_ns: int,
    ) -> None:
        self.last_access_ns.pop(node_id, None)

    def select_victim(
        self,
        candidates: Sequence[EvictionCandidate],
        context: EvictionContext,
    ) -> int:
        _require_candidates(candidates)
        return min(
            candidates,
            key=lambda candidate: (
                self.last_access_ns.get(candidate.node_id, 0),
                candidate.node_id,
            ),
        ).node_id


class LFUPolicy(BaseEvictionPolicy):
    name = "lfu"

    def __init__(self) -> None:
        self.access_count: Dict[int, int] = {}
        self.last_access_ns: Dict[int, int] = {}

    def on_insert(self, node_id: int, now_ns: int) -> None:
        self.access_count[node_id] = 1
        self.last_access_ns[node_id] = now_ns

    def on_access(self, node_id: int, now_ns: int) -> None:
        self.access_count[node_id] = self.access_count.get(node_id, 0) + 1
        self.last_access_ns[node_id] = now_ns

    def on_split(self, old_node_id: int, prefix_node_id: int, suffix_node_id: int) -> None:
        _inherit_split_state(self.access_count, old_node_id, prefix_node_id, suffix_node_id)
        _inherit_split_state(self.last_access_ns, old_node_id, prefix_node_id, suffix_node_id)

    def on_evict(
        self,
        node_id: int,
        prefix_hash: int,
        recompute_cost: float,
        now_ns: int,
    ) -> None:
        self.access_count.pop(node_id, None)
        self.last_access_ns.pop(node_id, None)

    def select_victim(
        self,
        candidates: Sequence[EvictionCandidate],
        context: EvictionContext,
    ) -> int:
        _require_candidates(candidates)
        return min(
            candidates,
            key=lambda candidate: (
                self.access_count.get(candidate.node_id, 0),
                self.last_access_ns.get(candidate.node_id, 0),
                candidate.node_id,
            ),
        ).node_id


class LRUKPolicy(BaseEvictionPolicy):
    name = "lru-k"

    def __init__(self, k: int = 2) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.access_history: Dict[int, Deque[int]] = {}

    def on_insert(self, node_id: int, now_ns: int) -> None:
        self.access_history[node_id] = deque([now_ns], maxlen=self.k)

    def on_access(self, node_id: int, now_ns: int) -> None:
        history = self.access_history.setdefault(node_id, deque(maxlen=self.k))
        history.append(now_ns)

    def on_split(self, old_node_id: int, prefix_node_id: int, suffix_node_id: int) -> None:
        history = self.access_history.pop(old_node_id, None)
        if history is not None:
            self.access_history[prefix_node_id] = deque(history, maxlen=self.k)
            self.access_history[suffix_node_id] = deque(history, maxlen=self.k)

    def on_evict(
        self,
        node_id: int,
        prefix_hash: int,
        recompute_cost: float,
        now_ns: int,
    ) -> None:
        self.access_history.pop(node_id, None)

    def _priority(self, node_id: int) -> Tuple[bool, int, int]:
        history = self.access_history.get(node_id)
        if not history:
            return False, 0, node_id
        if len(history) < self.k:
            return False, history[-1], node_id
        return True, history[0], node_id

    def select_victim(
        self,
        candidates: Sequence[EvictionCandidate],
        context: EvictionContext,
    ) -> int:
        _require_candidates(candidates)
        return min(candidates, key=lambda candidate: self._priority(candidate.node_id)).node_id


@dataclass
class _DecayState:
    frequency: float
    last_update_ns: int
    last_access_ns: int


class FrequencyDecayPolicy(BaseEvictionPolicy):
    name = "frequency-decay"

    def __init__(self, half_life_seconds: float = 60.0) -> None:
        if not math.isfinite(half_life_seconds) or half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be greater than 0")
        self.half_life_ns = half_life_seconds * 1_000_000_000
        self.state: Dict[int, _DecayState] = {}

    def _decay(self, state: _DecayState, now_ns: int) -> float:
        elapsed_ns = max(now_ns - state.last_update_ns, 0)
        return state.frequency * math.exp(-math.log(2) * elapsed_ns / self.half_life_ns)

    def on_insert(self, node_id: int, now_ns: int) -> None:
        self.state[node_id] = _DecayState(1.0, now_ns, now_ns)

    def on_access(self, node_id: int, now_ns: int) -> None:
        state = self.state.get(node_id)
        frequency = 0.0 if state is None else self._decay(state, now_ns)
        self.state[node_id] = _DecayState(frequency + 1.0, now_ns, now_ns)

    def on_split(self, old_node_id: int, prefix_node_id: int, suffix_node_id: int) -> None:
        state = self.state.pop(old_node_id, None)
        if state is not None:
            self.state[prefix_node_id] = _DecayState(**vars(state))
            self.state[suffix_node_id] = _DecayState(**vars(state))

    def on_evict(
        self,
        node_id: int,
        prefix_hash: int,
        recompute_cost: float,
        now_ns: int,
    ) -> None:
        self.state.pop(node_id, None)

    def select_victim(
        self,
        candidates: Sequence[EvictionCandidate],
        context: EvictionContext,
    ) -> int:
        _require_candidates(candidates)

        def priority(candidate: EvictionCandidate) -> Tuple[float, int, int]:
            state = self.state.get(candidate.node_id)
            if state is None:
                return 0.0, 0, candidate.node_id
            return self._decay(state, context.now_ns), state.last_access_ns, candidate.node_id

        return min(candidates, key=priority).node_id


class CostAwarePolicy(BaseEvictionPolicy):
    name = "cost-aware"

    def __init__(self, half_life_seconds: float = 60.0) -> None:
        if not math.isfinite(half_life_seconds) or half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be greater than 0")
        self.half_life_ns = half_life_seconds * 1_000_000_000
        self.access_count: Dict[int, int] = {}
        self.last_access_ns: Dict[int, int] = {}

    def on_insert(self, node_id: int, now_ns: int) -> None:
        self.access_count[node_id] = 1
        self.last_access_ns[node_id] = now_ns

    def on_access(self, node_id: int, now_ns: int) -> None:
        self.access_count[node_id] = self.access_count.get(node_id, 0) + 1
        self.last_access_ns[node_id] = now_ns

    def on_split(self, old_node_id: int, prefix_node_id: int, suffix_node_id: int) -> None:
        _inherit_split_state(self.access_count, old_node_id, prefix_node_id, suffix_node_id)
        _inherit_split_state(self.last_access_ns, old_node_id, prefix_node_id, suffix_node_id)

    def on_evict(
        self,
        node_id: int,
        prefix_hash: int,
        recompute_cost: float,
        now_ns: int,
    ) -> None:
        self.access_count.pop(node_id, None)
        self.last_access_ns.pop(node_id, None)

    def retention_value(self, candidate: EvictionCandidate, now_ns: int) -> float:
        count = self.access_count.get(candidate.node_id, 0)
        last_access = self.last_access_ns.get(candidate.node_id, 0)
        age_ns = max(now_ns - last_access, 0)
        frequency_probability = 1 - math.exp(-count)
        recency_probability = math.exp(-math.log(2) * age_ns / self.half_life_ns)
        reuse_probability = frequency_probability * recency_probability
        recompute_cost = estimate_recompute_cost(candidate.prefix_depth, candidate.segment_tokens)
        return reuse_probability * recompute_cost / max(candidate.kv_bytes, 1)

    def select_victim(
        self,
        candidates: Sequence[EvictionCandidate],
        context: EvictionContext,
    ) -> int:
        _require_candidates(candidates)
        return min(
            candidates,
            key=lambda candidate: (
                self.retention_value(candidate, context.now_ns),
                self.last_access_ns.get(candidate.node_id, 0),
                candidate.node_id,
            ),
        ).node_id


class AdaptivePolicy(BaseEvictionPolicy):
    """Online regret learner over a collection of eviction experts."""

    name = "adaptive"
    uses_ghost_feedback = True

    def __init__(
        self,
        experts: Sequence[BaseEvictionPolicy],
        learning_rate: float = 0.05,
        ghost_capacity: int = 4096,
        seed: int = 0,
    ) -> None:
        if not experts:
            raise ValueError("AdaptivePolicy requires at least one expert")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if ghost_capacity < 0:
            raise ValueError("ghost_capacity must be non-negative")
        names = [expert.name for expert in experts]
        if len(set(names)) != len(names):
            raise ValueError("AdaptivePolicy expert names must be unique")
        if self.name in names:
            raise ValueError("AdaptivePolicy cannot contain itself as an expert")

        self.experts = tuple(experts)
        self.learning_rate = learning_rate
        self.ghost_capacity = ghost_capacity
        self._weights = [1.0 / len(experts)] * len(experts)
        self._random = random.Random(seed)
        self._pending_decisions: Dict[int, int] = {}
        self._ghost_decisions: OrderedDict[int, Tuple[int, float]] = OrderedDict()
        self.last_selected_expert: str | None = None

    @property
    def expert_weights(self) -> Mapping[str, float]:
        return {expert.name: weight for expert, weight in zip(self.experts, self._weights)}

    def on_insert(self, node_id: int, now_ns: int) -> None:
        for expert in self.experts:
            expert.on_insert(node_id, now_ns)

    def on_access(self, node_id: int, now_ns: int) -> None:
        for expert in self.experts:
            expert.on_access(node_id, now_ns)

    def on_split(self, old_node_id: int, prefix_node_id: int, suffix_node_id: int) -> None:
        for expert in self.experts:
            expert.on_split(old_node_id, prefix_node_id, suffix_node_id)

    def on_evict(
        self,
        node_id: int,
        prefix_hash: int,
        recompute_cost: float,
        now_ns: int,
    ) -> None:
        for expert in self.experts:
            expert.on_evict(node_id, prefix_hash, recompute_cost, now_ns)

        expert_index = self._pending_decisions.pop(node_id, None)
        if expert_index is None or self.ghost_capacity == 0:
            return
        self._ghost_decisions[prefix_hash] = (expert_index, recompute_cost)
        self._ghost_decisions.move_to_end(prefix_hash)
        while len(self._ghost_decisions) > self.ghost_capacity:
            self._ghost_decisions.popitem(last=False)

    def on_ghost_hit(
        self,
        prefix_hash: int,
        recompute_cost: float,
        now_ns: int,
    ) -> None:
        for expert in self.experts:
            expert.on_ghost_hit(prefix_hash, recompute_cost, now_ns)

        decision = self._ghost_decisions.pop(prefix_hash, None)
        if decision is None:
            return
        expert_index, recorded_cost = decision
        regret = max(recompute_cost, recorded_cost, 0.0)
        stable_regret = min(math.log1p(regret), 50.0)
        self._weights[expert_index] *= math.exp(-self.learning_rate * stable_regret)
        self._renormalize_weights()

    def select_victim(
        self,
        candidates: Sequence[EvictionCandidate],
        context: EvictionContext,
    ) -> int:
        _require_candidates(candidates)
        candidate_ids = {candidate.node_id for candidate in candidates}
        proposals: List[int] = []
        for expert in self.experts:
            proposal = expert.select_victim(candidates, context)
            if proposal not in candidate_ids:
                raise RuntimeError(
                    f"Eviction expert '{expert.name}' selected ineligible node {proposal}"
                )
            proposals.append(proposal)

        expert_index = self._random.choices(range(len(self.experts)), weights=self._weights, k=1)[0]
        victim_id = proposals[expert_index]
        self._pending_decisions[victim_id] = expert_index
        self.last_selected_expert = self.experts[expert_index].name
        return victim_id

    def _renormalize_weights(self) -> None:
        total = sum(self._weights)
        if not math.isfinite(total) or total <= 0:
            self._weights = [1.0 / len(self.experts)] * len(self.experts)
            return
        self._weights = [max(weight / total, 1e-12) for weight in self._weights]
        total = sum(self._weights)
        self._weights = [weight / total for weight in self._weights]


__all__ = [
    "AdaptivePolicy",
    "CostAwarePolicy",
    "FrequencyDecayPolicy",
    "LFUPolicy",
    "LRUKPolicy",
    "LRUPolicy",
]
