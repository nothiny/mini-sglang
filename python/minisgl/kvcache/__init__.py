from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)
from .eviction import (
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


class CacheManagerCreator(Protocol):
    def __call__(
        self,
        device: torch.device,
        *,
        eviction_policy: BaseEvictionPolicy | None = None,
        eviction_policy_config: EvictionPolicyConfig | None = None,
        total_tokens: int = 0,
        kv_bytes_per_token: int = 1,
    ) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
    if model_config.is_mla:
        from .mla_pool import MLAKVCache

        return MLAKVCache(
            num_layers=model_config.num_layers,
            num_pages=num_pages,
            page_size=page_size,
            kv_lora_rank=model_config.kv_lora_rank,
            qk_rope_head_dim=model_config.qk_rope_head_dim,
            device=device,
            dtype=dtype,
        )

    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(
    device: torch.device,
    *,
    eviction_policy: BaseEvictionPolicy | None = None,
    eviction_policy_config: EvictionPolicyConfig | None = None,
    total_tokens: int = 0,
    kv_bytes_per_token: int = 1,
) -> BasePrefixCache:
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(
    device: torch.device,
    *,
    eviction_policy: BaseEvictionPolicy | None = None,
    eviction_policy_config: EvictionPolicyConfig | None = None,
    total_tokens: int = 0,
    kv_bytes_per_token: int = 1,
) -> BasePrefixCache:
    from .radix_cache import RadixPrefixCache

    policy_config = eviction_policy_config or EvictionPolicyConfig()
    policy = eviction_policy or create_eviction_policy(config=policy_config)
    return RadixPrefixCache(
        device=device,
        eviction_policy=policy,
        total_tokens=total_tokens,
        kv_bytes_per_token=kv_bytes_per_token,
        ghost_capacity=policy_config.ghost_capacity,
    )


def create_prefix_cache(
    device: torch.device,
    type: str,
    *,
    eviction_policy: BaseEvictionPolicy | None = None,
    eviction_policy_config: EvictionPolicyConfig | None = None,
    total_tokens: int = 0,
    kv_bytes_per_token: int = 1,
) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](
        device,
        eviction_policy=eviction_policy,
        eviction_policy_config=eviction_policy_config,
        total_tokens=total_tokens,
        kv_bytes_per_token=kv_bytes_per_token,
    )


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "AdaptivePolicy",
    "BaseEvictionPolicy",
    "CostAwarePolicy",
    "EvictionCandidate",
    "EvictionContext",
    "EvictionPolicyConfig",
    "FrequencyDecayPolicy",
    "LFUPolicy",
    "LRUKPolicy",
    "LRUPolicy",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
    "SUPPORTED_EVICTION_POLICIES",
    "create_eviction_policy",
]
