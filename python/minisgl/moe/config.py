from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ExpertQuantization = Literal["none", "int8"]


@dataclass(frozen=True)
class MoeBackendConfig:
    """Runtime options shared by MoE layers and backends."""

    enable_workspace_cache: bool = True
    small_m_threshold: int = 0
    autotune: bool = False
    expert_parallel_size: int = 1
    expert_parallel_rank: int = 0
    expert_parallel_overlap: bool = True
    expert_quantization: ExpertQuantization = "none"
    expert_offload: bool = False
    expert_cache_size: int = 0

    def __post_init__(self) -> None:
        if self.small_m_threshold < 0:
            raise ValueError("small_m_threshold must be non-negative")
        if self.expert_parallel_size < 1:
            raise ValueError("expert_parallel_size must be at least 1")
        if not 0 <= self.expert_parallel_rank < self.expert_parallel_size:
            raise ValueError("expert_parallel_rank must be within expert_parallel_size")
        if self.expert_quantization not in ("none", "int8"):
            raise ValueError("expert_quantization must be 'none' or 'int8'")
        if self.expert_cache_size < 0:
            raise ValueError("expert_cache_size must be non-negative")
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
    "ExpertQuantization",
    "MoeBackendConfig",
    "get_moe_backend_config",
    "set_moe_backend_config",
]
