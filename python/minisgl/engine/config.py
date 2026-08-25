from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    # Hierarchical KV cache (L1 GPU -> L2 pinned RAM -> L3 local storage).
    enable_hicache: bool = False
    hicache_size_gb: float | None = None
    hicache_ratio: float = 1.0
    hicache_storage_size_gb: float | None = None
    hicache_storage_ratio: float = 0.0
    hicache_storage_path: str | None = None
    hicache_io_workers: int = 2
    hicache_staging_pages: int = 8
    hicache_promote_storage: bool = True
    hicache_policy: str = "cost"
    hicache_recompute_us_per_token: float = 50.0
    hicache_host_bandwidth_gib_s: float = 12.0
    hicache_storage_bandwidth_gib_s: float = 3.0
    hicache_cost_margin: float = 1.1
    hicache_prefetch: bool = True
    hicache_transfer_backend: str = "auto"

    @cached_property
    def hf_config(self) -> Any:
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
