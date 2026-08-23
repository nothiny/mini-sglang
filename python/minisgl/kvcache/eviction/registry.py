from __future__ import annotations

from typing import Protocol

from minisgl.utils import Registry

from .base import BaseEvictionPolicy
from .config import EvictionPolicyConfig
from .policies import (
    AdaptivePolicy,
    CostAwarePolicy,
    FrequencyDecayPolicy,
    LFUPolicy,
    LRUKPolicy,
    LRUPolicy,
)


class EvictionPolicyCreator(Protocol):
    def __call__(self, config: EvictionPolicyConfig) -> BaseEvictionPolicy: ...


SUPPORTED_EVICTION_POLICIES = Registry[EvictionPolicyCreator]("KV Cache Eviction Policy")


@SUPPORTED_EVICTION_POLICIES.register("lru")
def _create_lru_policy(config: EvictionPolicyConfig) -> BaseEvictionPolicy:
    return LRUPolicy()


@SUPPORTED_EVICTION_POLICIES.register("lfu")
def _create_lfu_policy(config: EvictionPolicyConfig) -> BaseEvictionPolicy:
    return LFUPolicy()


@SUPPORTED_EVICTION_POLICIES.register("lru-k")
def _create_lru_k_policy(config: EvictionPolicyConfig) -> BaseEvictionPolicy:
    return LRUKPolicy(k=config.lru_k)


@SUPPORTED_EVICTION_POLICIES.register("frequency-decay")
def _create_frequency_decay_policy(config: EvictionPolicyConfig) -> BaseEvictionPolicy:
    return FrequencyDecayPolicy(half_life_seconds=config.frequency_half_life)


@SUPPORTED_EVICTION_POLICIES.register("cost-aware")
def _create_cost_aware_policy(config: EvictionPolicyConfig) -> BaseEvictionPolicy:
    return CostAwarePolicy(half_life_seconds=config.frequency_half_life)


@SUPPORTED_EVICTION_POLICIES.register("adaptive")
def _create_adaptive_policy(config: EvictionPolicyConfig) -> BaseEvictionPolicy:
    if "adaptive" in config.adaptive_experts:
        raise ValueError("adaptive cannot be nested inside adaptive_experts")
    SUPPORTED_EVICTION_POLICIES.assert_supported(config.adaptive_experts)
    experts = [create_eviction_policy(name, config) for name in config.adaptive_experts]
    return AdaptivePolicy(
        experts=experts,
        learning_rate=config.adaptive_learning_rate,
        ghost_capacity=config.ghost_capacity,
        seed=config.adaptive_seed,
    )


def create_eviction_policy(
    name: str | None = None,
    config: EvictionPolicyConfig | None = None,
) -> BaseEvictionPolicy:
    if config is None:
        config = EvictionPolicyConfig(policy=name or "lru")
    policy_name = name or config.policy
    return SUPPORTED_EVICTION_POLICIES[policy_name](config)


__all__ = ["SUPPORTED_EVICTION_POLICIES", "create_eviction_policy"]
