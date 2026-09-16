from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from minisgl.engine import EngineConfig
from minisgl.kvcache import EvictionPolicyConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    cache_eviction_policy: str = "lru"
    eviction_k: int = 2
    eviction_half_life: float = 60.0
    eviction_ghost_capacity: int = 4096
    adaptive_experts: Tuple[str, ...] = ("lru", "lfu", "cost-aware")
    adaptive_learning_rate: float = 0.05
    adaptive_seed: int = 0
    offline_mode: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def eviction_policy_config(self) -> EvictionPolicyConfig:
        return EvictionPolicyConfig(
            policy=self.cache_eviction_policy,
            lru_k=self.eviction_k,
            frequency_half_life=self.eviction_half_life,
            ghost_capacity=self.eviction_ghost_capacity,
            adaptive_experts=self.adaptive_experts,
            adaptive_learning_rate=self.adaptive_learning_rate,
            adaptive_seed=self.adaptive_seed,
        )

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
