from __future__ import annotations

import torch

from .base import BaseKVCachePool


class MLAKVCache(BaseKVCachePool):
    """DeepSeek MLA latent KV cache.

    One record per physical token per layer:
      - ``ckv``: the compressed latent ``c_KV`` (``kv_lora_rank``)
      - ``kpe``: the decoupled RoPE key   (``qk_rope_head_dim``)

    Unlike the MHA pool this is not ``[K, V, ...]`` and K/V do not share a head
    dimension, so it exposes ``ckv_cache`` / ``kpe_cache`` instead. The layout
    matches ``flashinfer.mla.BatchMLAPagedAttentionWrapper``
    (``[layers, tokens, dim]`` with ``page_size == 1``).
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_pages: int,
        page_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._num_layers = num_layers
        self._num_pages = num_pages
        self._page_size = page_size
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
        self._dtype = dtype
        self._device = device
        self._tokens = num_pages * page_size
        self._ckv = torch.empty(
            (num_layers, self._tokens, kv_lora_rank), dtype=dtype, device=device
        )
        self._kpe = torch.empty(
            (num_layers, self._tokens, qk_rope_head_dim), dtype=dtype, device=device
        )

    def store_mla(
        self,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        self._ckv[layer_id, out_loc] = c_kv
        self._kpe[layer_id, out_loc] = k_pe

    def ckv_cache(self, layer_id: int) -> torch.Tensor:
        return self._ckv[layer_id]

    def kpe_cache(self, layer_id: int) -> torch.Tensor:
        return self._kpe[layer_id]

    @property
    def kv_lora_rank(self) -> int:
        return self._kv_lora_rank

    @property
    def qk_rope_head_dim(self) -> int:
        return self._qk_rope_head_dim

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def bytes_per_page(self) -> int:
        return (
            (self._kv_lora_rank + self._qk_rope_head_dim) * self._dtype.itemsize * self._num_layers
        ) * self._page_size

    # The MHA pool interface does not describe MLA; the MLA backend uses store_mla.
    def k_cache(self, index: int) -> torch.Tensor:
        return self._ckv[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._kpe[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        raise NotImplementedError("MLAKVCache stores (c_KV, k_pe) via store_mla")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers


__all__ = ["MLAKVCache"]
