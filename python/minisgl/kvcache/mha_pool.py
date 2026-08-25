from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._num_pages = num_pages
        self._page_size = page_size
        self._local_kv_heads = local_kv_heads
        self._head_dim = head_dim
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def local_kv_heads(self) -> int:
        return self._local_kv_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def bytes_per_page(self) -> int:
        return (
            2
            * self.num_layers
            * self.page_size
            * self.local_kv_heads
            * self.head_dim
            * self.dtype.itemsize
        )

    @property
    def buffer(self) -> torch.Tensor:
        """Return the physical ``[K/V, layer, page, token, head, dim]`` buffer."""
        return self._kv_buffer


class HostMHAKVCache(BaseKVCachePool):
    """Page-major MHA KV storage backed by pinned host memory.

    Physical pages are contiguous as ``[page, K/V, layer, token, head, dim]``. The public
    ``buffer`` property exposes a layer-major compatibility view matching ``MHAKVCache``.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_pages: int,
        page_size: int,
        local_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        pin_memory: bool = True,
    ) -> None:
        self._page_buffer = torch.empty(
            (num_pages, 2, num_layers, page_size, local_kv_heads, head_dim),
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        self._num_layers = num_layers
        self._num_pages = num_pages
        self._page_size = page_size
        self._local_kv_heads = local_kv_heads
        self._head_dim = head_dim
        self._k_buffer = self._page_buffer[:, 0]
        self._v_buffer = self._page_buffer[:, 1]

    @classmethod
    def from_device_pool(
        cls,
        pool: MHAKVCache,
        num_pages: int,
        *,
        pin_memory: bool = True,
    ) -> HostMHAKVCache:
        return cls(
            num_layers=pool.num_layers,
            num_pages=num_pages,
            page_size=pool.page_size,
            local_kv_heads=pool.local_kv_heads,
            head_dim=pool.head_dim,
            dtype=pool.dtype,
            pin_memory=pin_memory,
        )

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[:, index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[:, index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        indices = out_loc.to(device="cpu", dtype=torch.int64)
        pages = indices // self.page_size
        offsets = indices % self.page_size
        self._page_buffer[pages, 0, layer_id, offsets] = k.cpu()
        self._page_buffer[pages, 1, layer_id, offsets] = v.cpu()

    @property
    def device(self) -> torch.device:
        return self._page_buffer.device

    @property
    def dtype(self) -> torch.dtype:
        return self._page_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def local_kv_heads(self) -> int:
        return self._local_kv_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def bytes_per_page(self) -> int:
        return (
            2
            * self.num_layers
            * self.page_size
            * self.local_kv_heads
            * self.head_dim
            * self.dtype.itemsize
        )

    @property
    def buffer(self) -> torch.Tensor:
        return self._page_buffer.permute(1, 2, 0, 3, 4, 5)

    @property
    def page_buffer(self) -> torch.Tensor:
        """Return contiguous ``[page, K/V, layer, token, head, dim]`` storage."""
        return self._page_buffer
