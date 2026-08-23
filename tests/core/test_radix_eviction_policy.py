from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import minisgl.core as core
import pytest
import torch
from minisgl.kvcache.eviction import (
    BaseEvictionPolicy,
    EvictionCandidate,
    EvictionContext,
    EvictionPolicyConfig,
    create_eviction_policy,
)
from minisgl.kvcache.radix_cache import RadixCacheHandle, RadixPrefixCache


class RecordingPolicy(BaseEvictionPolicy):
    name = "recording"
    uses_ghost_feedback = True

    def __init__(self) -> None:
        self.inserted: List[int] = []
        self.accessed: List[int] = []
        self.splits: List[Tuple[int, int, int]] = []
        self.evicted: List[Tuple[int, int, float]] = []
        self.ghost_hits: List[Tuple[int, float]] = []
        self.candidate_batches: List[Tuple[EvictionCandidate, ...]] = []

    def on_insert(self, node_id: int, now_ns: int) -> None:
        self.inserted.append(node_id)

    def on_access(self, node_id: int, now_ns: int) -> None:
        self.accessed.append(node_id)

    def on_split(self, old_node_id: int, prefix_node_id: int, suffix_node_id: int) -> None:
        self.splits.append((old_node_id, prefix_node_id, suffix_node_id))

    def on_evict(
        self,
        node_id: int,
        prefix_hash: int,
        recompute_cost: float,
        now_ns: int,
    ) -> None:
        self.evicted.append((node_id, prefix_hash, recompute_cost))

    def on_ghost_hit(
        self,
        prefix_hash: int,
        recompute_cost: float,
        now_ns: int,
    ) -> None:
        self.ghost_hits.append((prefix_hash, recompute_cost))

    def select_victim(
        self,
        candidates: Sequence[EvictionCandidate],
        context: EvictionContext,
    ) -> int:
        self.candidate_batches.append(tuple(candidates))
        return min(candidate.node_id for candidate in candidates)


@pytest.fixture(autouse=True)
def reset_global_ctx() -> Iterator[None]:
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def _make_cache(policy: BaseEvictionPolicy | None, page_size: int = 1) -> RadixPrefixCache:
    core.set_global_ctx(core.Context(page_size=page_size))
    return RadixPrefixCache(
        device=torch.device("cpu"),
        eviction_policy=policy,
        total_tokens=32,
        kv_bytes_per_token=128,
        ghost_capacity=16,
    )


def _insert(cache: RadixPrefixCache, tokens: List[int], start: int) -> RadixCacheHandle:
    input_ids = torch.tensor(tokens, dtype=torch.int32)
    indices = torch.arange(start, start + len(tokens), dtype=torch.int32)
    handle = cache.insert_prefix(input_ids, indices).handle
    assert isinstance(handle, RadixCacheHandle)
    return handle


def test_class_injection_and_radix_candidate_protection() -> None:
    policy = RecordingPolicy()
    cache = _make_cache(policy)
    first = _insert(cache, [1, 2], 0)
    second = _insert(cache, [3, 4], 2)
    cache.lock_handle(first)

    evicted = cache.evict(1)

    assert evicted.tolist() == [2, 3]
    assert policy.evicted[0][0] == second.node.uuid
    assert all(candidate.ref_count == 0 for candidate in policy.candidate_batches[0])
    assert first.node.uuid not in {candidate.node_id for candidate in policy.candidate_batches[0]}
    cache.lock_handle(first, unlock=True)
    cache.check_integrity()


def test_invalid_eviction_requests_are_rejected() -> None:
    cache = _make_cache(RecordingPolicy())
    _insert(cache, [1, 2], 0)

    with pytest.raises(ValueError, match="non-negative"):
        cache.evict(-1)
    with pytest.raises(RuntimeError, match="only 2 is evictable"):
        cache.evict(3)


def test_split_access_evict_and_ghost_feedback_lifecycle() -> None:
    policy = RecordingPolicy()
    cache = _make_cache(policy, page_size=2)
    original = _insert(cache, [1, 2, 3, 4], 0)

    match = cache.match_prefix(torch.tensor([1, 2, 9, 9], dtype=torch.int32)).cuda_handle
    assert isinstance(match, RadixCacheHandle)
    assert match.cached_len == 2
    assert len(policy.splits) == 1
    old_node_id, prefix_node_id, suffix_node_id = policy.splits[0]
    assert old_node_id == original.node.uuid == suffix_node_id
    assert match.node.uuid == prefix_node_id
    assert prefix_node_id in policy.accessed
    cache.check_integrity()

    evicted = cache.evict(2)
    assert evicted.tolist() == [2, 3]
    assert policy.evicted[-1][0] == original.node.uuid

    rematch = cache.match_prefix(torch.tensor([1, 2, 3, 4], dtype=torch.int32)).cuda_handle
    assert rematch.cached_len == 2
    assert len(policy.ghost_hits) == 1
    assert policy.ghost_hits[0][0] == policy.evicted[-1][1]
    assert policy.ghost_hits[0][1] == policy.evicted[-1][2]
    cache.check_integrity()


def test_default_lru_behavior_is_preserved() -> None:
    cache = _make_cache(policy=None)
    first = _insert(cache, [1, 2], 0)
    second = _insert(cache, [3, 4], 2)
    cache.match_prefix(torch.tensor([1, 2], dtype=torch.int32))

    evicted = cache.evict(1)

    assert evicted.tolist() == [2, 3]
    assert first.node.uuid != second.node.uuid
    cache.check_integrity()


@pytest.mark.parametrize(
    "policy_name",
    ["lru", "lfu", "lru-k", "frequency-decay", "cost-aware", "adaptive"],
)
def test_all_registered_policies_preserve_radix_integrity(policy_name: str) -> None:
    config = EvictionPolicyConfig(policy=policy_name, adaptive_seed=0)
    cache = _make_cache(create_eviction_policy(config=config))
    _insert(cache, [1, 2, 3, 4], 0)
    _insert(cache, [1, 2, 5, 6], 4)
    _insert(cache, [7, 8], 8)
    _insert(cache, [9, 10], 10)
    cache.match_prefix(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    cache.check_integrity()

    while cache.size_info.evictable_size > 0:
        cache.evict(1)
        cache.check_integrity()

    assert cache.size_info.total_size == 0
