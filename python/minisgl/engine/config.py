from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig
    from minisgl.moe import MoeBackendConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    moe_enable_workspace_cache: bool = True
    moe_small_m_threshold: int = 0
    moe_autotune: bool = False
    expert_parallel_size: int = 1
    moe_expert_parallel_overlap: bool = True
    moe_expert_quantization: str = "none"
    moe_expert_offload: bool = False
    moe_expert_cache_size: int = 0
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    @cached_property
    def hf_config(self):
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
    def moe_backend_config(self) -> MoeBackendConfig:
        from minisgl.moe import MoeBackendConfig

        ep_rank = self.tp_info.rank if self.expert_parallel_size > 1 else 0
        return MoeBackendConfig(
            enable_workspace_cache=self.moe_enable_workspace_cache,
            small_m_threshold=self.moe_small_m_threshold,
            autotune=self.moe_autotune,
            expert_parallel_size=self.expert_parallel_size,
            expert_parallel_rank=ep_rank,
            expert_parallel_overlap=self.moe_expert_parallel_overlap,
            expert_quantization=self.moe_expert_quantization,  # type: ignore[arg-type]
            expert_offload=self.moe_expert_offload,
            expert_cache_size=self.moe_expert_cache_size,
        )

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
