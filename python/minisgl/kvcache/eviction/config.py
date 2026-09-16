from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class EvictionPolicyConfig:
    policy: str = "lru"
    lru_k: int = 2
    frequency_half_life: float = 60.0
    ghost_capacity: int = 4096
    adaptive_experts: Tuple[str, ...] = ("lru", "lfu", "cost-aware")
    adaptive_learning_rate: float = 0.05
    adaptive_seed: int = 0

    def __post_init__(self) -> None:
        if self.lru_k < 1:
            raise ValueError("lru_k must be at least 1")
        if not math.isfinite(self.frequency_half_life) or self.frequency_half_life <= 0:
            raise ValueError("frequency_half_life must be greater than 0")
        if self.ghost_capacity < 0:
            raise ValueError("ghost_capacity must be non-negative")
        if not self.adaptive_experts:
            raise ValueError("adaptive_experts must contain at least one policy")
        if len(set(self.adaptive_experts)) != len(self.adaptive_experts):
            raise ValueError("adaptive expert names must be unique")
        if "adaptive" in self.adaptive_experts:
            raise ValueError("adaptive cannot contain itself as an expert")
        if not math.isfinite(self.adaptive_learning_rate) or self.adaptive_learning_rate <= 0:
            raise ValueError("adaptive_learning_rate must be greater than 0")


__all__ = ["EvictionPolicyConfig"]
