from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.mla import BatchMLAPagedAttentionWrapper  # type: ignore
    from minisgl.models import ModelConfig


@dataclass
class MLAMetadata(BaseAttnMetadata):
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_len_arr: torch.Tensor
    num_qo_heads: int
    head_dim_ckv: int
    head_dim_kpe: int
    sm_scale: float
    dtype: torch.dtype
    wrapper: "BatchMLAPagedAttentionWrapper"
    initialized: bool = False

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.qo_indptr[1 : 1 + bs] - 1


class MLAttentionBackend(BaseAttnBackend):
    """Paged MLA attention on the absorbed latent cache.

    The layer does the absorption (``q_abs = W_UK^T q_nope``) and the output
    projection (``o = W_UV z``); this backend only stores ``(c_KV, k_pe)`` and
    runs ``BatchMLAPagedAttentionWrapper`` over them.
    """

    def __init__(self, config: ModelConfig) -> None:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.num_qo_heads = config.num_qo_heads // get_tp_info().size
        self.head_dim_ckv = config.kv_lora_rank
        self.head_dim_kpe = config.qk_rope_head_dim
        assert self.head_dim_ckv is not None and self.head_dim_kpe is not None
        self.sm_scale = (config.qk_nope_head_dim + config.qk_rope_head_dim) ** -0.5
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=self.device)
        self.wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    def _plan(self, metadata: MLAMetadata) -> None:
        if metadata.initialized:
            return
        # The planner stages host data and launches an async H2D copy; wait before
        # the next plan mutates that staging buffer.
        self.last_event.synchronize()
        metadata.initialized = True
        metadata.wrapper.plan(
            metadata.qo_indptr,
            metadata.kv_indptr,
            metadata.kv_indices,
            metadata.kv_len_arr,
            metadata.num_qo_heads,
            metadata.head_dim_ckv,
            metadata.head_dim_kpe,
            1,
            True,
            metadata.sm_scale,
            metadata.dtype,
            metadata.dtype,
        )
        self.last_event.record()

    def forward_mla(
        self,
        q_abs: torch.Tensor,
        q_pe: torch.Tensor,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        self.kvcache.store_mla(c_kv, k_pe, batch.out_loc, layer_id)
        metadata = batch.attn_metadata
        assert isinstance(metadata, MLAMetadata)
        self._plan(metadata)
        ckv = self.kvcache.ckv_cache(layer_id).view(-1, 1, self.head_dim_ckv)
        kpe = self.kvcache.kpe_cache(layer_id).view(-1, 1, self.head_dim_kpe)
        return metadata.wrapper.run(q_abs, q_pe, ckv, kpe)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        host = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        q_lens = [req.extend_len for req in reqs]
        kv_lens = [req.device_len for req in reqs]
        qo_indptr = torch.tensor([0] + q_lens, **host).cumsum_(0)
        kv_indptr = torch.tensor([0] + kv_lens, **host).cumsum_(0)
        page_table = get_global_ctx().page_table
        kv_indices = torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])
        batch.attn_metadata = MLAMetadata(
            qo_indptr=qo_indptr.to(self.device, non_blocking=True),
            kv_indptr=kv_indptr.to(self.device, non_blocking=True),
            kv_indices=kv_indices.to(torch.int32),
            kv_len_arr=torch.tensor(kv_lens, **host).to(self.device, non_blocking=True),
            num_qo_heads=self.num_qo_heads,
            head_dim_ckv=self.head_dim_ckv,
            head_dim_kpe=self.head_dim_kpe,
            sm_scale=self.sm_scale,
            dtype=self.kvcache.dtype,
            wrapper=self.wrapper,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        raise NotImplementedError("MLA attention uses forward_mla")

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        raise NotImplementedError("CUDA graph capture is not supported for MLA yet")

    def prepare_for_capture(self, batch: Batch) -> None:
        raise NotImplementedError("CUDA graph capture is not supported for MLA yet")

    def prepare_for_replay(self, batch: Batch) -> None:
        raise NotImplementedError("CUDA graph capture is not supported for MLA yet")


__all__ = ["MLAMetadata", "MLAttentionBackend"]
