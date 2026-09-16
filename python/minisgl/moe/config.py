from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

ExpertQuantization = Literal["none", "int8", "fp8"]
ExpertParallelDispatchMode = Literal["dynamic", "static"]
ExpertPlacement = Literal["contiguous", "round-robin"]


@dataclass(frozen=True)
class MoeBackendConfig:
    """Runtime options shared by MoE layers and backends."""

    enable_workspace_cache: bool = True
    small_m_threshold: int = 0
    autotune: bool = False
    expert_parallel_size: int = 1
    expert_parallel_rank: int = 0
    expert_parallel_overlap: bool = True
    expert_parallel_dispatch: ExpertParallelDispatchMode = "dynamic"
    expert_placement: ExpertPlacement = "contiguous"
    replicated_experts: Tuple[int, ...] = ()
    expert_quantization: ExpertQuantization = "none"
    expert_offload: bool = False
    expert_cache_size: int = 0
    expert_offload_overlap: bool = True
    expert_offload_wave_size: int = 8

    def __post_init__(self) -> None:
        if self.small_m_threshold < 0:
            raise ValueError("small_m_threshold must be non-negative")
        if self.expert_parallel_size < 1:
            raise ValueError("expert_parallel_size must be at least 1")
        if not 0 <= self.expert_parallel_rank < self.expert_parallel_size:
            raise ValueError("expert_parallel_rank must be within expert_parallel_size")
        if self.expert_parallel_dispatch not in ("dynamic", "static"):
            raise ValueError("expert_parallel_dispatch must be 'dynamic' or 'static'")
        if self.expert_placement not in ("contiguous", "round-robin"):
            raise ValueError("expert_placement must be 'contiguous' or 'round-robin'")
        if any(expert_id < 0 for expert_id in self.replicated_experts):
            raise ValueError("replicated_experts must contain non-negative IDs")
        if len(set(self.replicated_experts)) != len(self.replicated_experts):
            raise ValueError("replicated_experts must contain unique IDs")
        if self.expert_quantization not in ("none", "int8", "fp8"):
            raise ValueError("expert_quantization must be 'none', 'int8', or 'fp8'")
        if self.expert_cache_size < 0:
            raise ValueError("expert_cache_size must be non-negative")
        if self.expert_offload_wave_size < 1:
            raise ValueError("expert_offload_wave_size must be positive")
        if self.expert_offload and self.expert_cache_size < 1:
            raise ValueError("expert_cache_size must be positive when expert_offload is enabled")


_MOE_BACKEND_CONFIG = MoeBackendConfig()
_MOE_BACKEND_CONFIG_SET = False


def set_moe_backend_config(config: MoeBackendConfig) -> None:
    global _MOE_BACKEND_CONFIG, _MOE_BACKEND_CONFIG_SET
    if _MOE_BACKEND_CONFIG_SET:
        raise RuntimeError("MoE backend config has been set")
    _MOE_BACKEND_CONFIG = config
    _MOE_BACKEND_CONFIG_SET = True


def get_moe_backend_config() -> MoeBackendConfig:
    return _MOE_BACKEND_CONFIG


__all__ = [
    "ExpertParallelDispatchMode",
    "ExpertPlacement",
    "ExpertQuantization",
    "MoeBackendConfig",
    "get_moe_backend_config",
    "set_moe_backend_config",
]
