from __future__ import annotations

from typing import Sequence

import pytest
from minisgl.kvcache.eviction import (
    SUPPORTED_EVICTION_POLICIES,
    AdaptivePolicy,
    BaseEvictionPolicy,
    CostAwarePolicy,
    EvictionCandidate,
    EvictionContext,
    EvictionPolicyConfig,
    FrequencyDecayPolicy,
    LFUPolicy,
    LRUKPolicy,
    LRUPolicy,
    create_eviction_policy,
)
from minisgl.server.args import parse_args


def _candidate(
    node_id: int,
    *,
    segment_tokens: int = 16,
    prefix_depth: int = 0,
    kv_bytes: int = 1024,
) -> EvictionCandidate:
    return EvictionCandidate(
        node_id=node_id,
        segment_tokens=segment_tokens,
        prefix_depth=prefix_depth,
        kv_bytes=kv_bytes,
        ref_count=0,
    )


def _context(now_ns: int = 100) -> EvictionContext:
    return EvictionContext(
        now_ns=now_ns,
        requested_tokens=16,
        free_tokens=0,
        total_tokens=128,
    )


def test_lru_and_lfu_choose_expected_victims() -> None:
    candidates = [_candidate(1), _candidate(2)]

    lru = LRUPolicy()
    lru.on_insert(1, 10)
    lru.on_insert(2, 20)
    lru.on_access(1, 30)
    assert lru.select_victim(candidates, _context()) == 2

    lfu = LFUPolicy()
    lfu.on_insert(1, 10)
    lfu.on_insert(2, 20)
    lfu.on_access(1, 30)
    lfu.on_access(1, 40)
    assert lfu.select_victim(candidates, _context()) == 2


def test_lru_k_prioritizes_nodes_with_fewer_than_k_accesses() -> None:
    candidates = [_candidate(1), _candidate(2)]
    policy = LRUKPolicy(k=2)
    policy.on_insert(1, 10)
    policy.on_access(1, 30)
    policy.on_insert(2, 20)

    assert policy.select_victim(candidates, _context()) == 2

    policy.on_access(2, 40)
    assert policy.select_victim(candidates, _context()) == 1


def test_policy_state_is_inherited_on_radix_split() -> None:
    policy = LFUPolicy()
    policy.on_insert(10, 10)
    policy.on_access(10, 20)
    policy.on_split(old_node_id=10, prefix_node_id=11, suffix_node_id=10)

    assert policy.access_count[10] == 2
    assert policy.access_count[11] == 2
    assert policy.last_access_ns[10] == 20
    assert policy.last_access_ns[11] == 20


def test_frequency_decay_forgets_stale_hot_nodes() -> None:
    second = 1_000_000_000
    policy = FrequencyDecayPolicy(half_life_seconds=1)
    policy.on_insert(1, 0)
    for _ in range(3):
        policy.on_access(1, 0)
    policy.on_insert(2, 10 * second)

    candidates = [_candidate(1), _candidate(2)]
    assert policy.select_victim(candidates, _context(now_ns=20 * second)) == 1


def test_cost_aware_retains_deeper_expensive_prefixes() -> None:
    now_ns = 100
    policy = CostAwarePolicy(half_life_seconds=60)
    policy.on_insert(1, now_ns)
    policy.on_insert(2, now_ns)
    shallow = _candidate(1, segment_tokens=16, prefix_depth=0, kv_bytes=1024)
    deep = _candidate(2, segment_tokens=16, prefix_depth=256, kv_bytes=1024)

    assert policy.retention_value(deep, now_ns) > policy.retention_value(shallow, now_ns)
    assert policy.select_victim([shallow, deep], _context(now_ns)) == shallow.node_id


class _FirstCandidatePolicy(BaseEvictionPolicy):
    name = "first"

    def select_victim(
        self,
        candidates: Sequence[EvictionCandidate],
        context: EvictionContext,
    ) -> int:
        return candidates[0].node_id


class _LastCandidatePolicy(BaseEvictionPolicy):
    name = "last"

    def select_victim(
        self,
        candidates: Sequence[EvictionCandidate],
        context: EvictionContext,
    ) -> int:
        return candidates[-1].node_id


def test_adaptive_policy_penalizes_expert_after_ghost_hit() -> None:
    policy = AdaptivePolicy(
        [_FirstCandidatePolicy(), _LastCandidatePolicy()],
        learning_rate=0.2,
        seed=0,
    )
    candidates = [_candidate(1), _candidate(2)]
    victim = policy.select_victim(candidates, _context())
    selected_expert = policy.last_selected_expert
    assert selected_expert is not None

    policy.on_evict(victim, prefix_hash=1234, recompute_cost=1000, now_ns=200)
    before = policy.expert_weights[selected_expert]
    policy.on_ghost_hit(prefix_hash=1234, recompute_cost=1000, now_ns=300)
    after = policy.expert_weights[selected_expert]

    assert after < before
    assert sum(policy.expert_weights.values()) == pytest.approx(1.0)


def test_policy_factory_and_cli_configuration() -> None:
    expected = {"lru", "lfu", "lru-k", "frequency-decay", "cost-aware", "adaptive"}
    assert set(SUPPORTED_EVICTION_POLICIES.supported_names()) == expected
    for policy_name in expected:
        assert create_eviction_policy(policy_name).name == policy_name

    config = EvictionPolicyConfig(policy="lru-k", lru_k=3)
    policy = create_eviction_policy(config=config)
    assert isinstance(policy, LRUKPolicy)
    assert policy.k == 3

    args, run_shell = parse_args(
        [
            "--model",
            "unused/model",
            "--dtype",
            "float16",
            "--cache",
            "radix",
            "--cache-eviction-policy",
            "adaptive",
            "--adaptive-experts",
            "lru,lfu,cost-aware",
            "--adaptive-learning-rate",
            "0.1",
            "--eviction-ghost-capacity",
            "128",
        ]
    )
    assert not run_shell
    assert args.cache_eviction_policy == "adaptive"
    assert args.adaptive_experts == ("lru", "lfu", "cost-aware")
    assert args.adaptive_learning_rate == 0.1
    assert args.eviction_ghost_capacity == 128
