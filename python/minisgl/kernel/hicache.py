from __future__ import annotations

import torch


def gather_kv_pages(source: torch.Tensor, page_ids: torch.Tensor, packed: torch.Tensor) -> None:
    """Gather page dimension 2 across every K/V layer with one Triton launch."""
    import triton

    from .triton.hicache import hicache_gather_pages_kernel

    _validate(source, page_ids, packed)
    page_elements = source.shape[3] * source.shape[4] * source.shape[5]
    num_elements = packed.numel()
    hicache_gather_pages_kernel[(triton.cdiv(num_elements, 256),)](
        source,
        page_ids,
        packed,
        num_elements,
        source.shape[1],
        source.shape[2],
        page_elements,
        BLOCK_SIZE=256,
        num_warps=4,
    )


def scatter_kv_pages(
    packed: torch.Tensor, page_ids: torch.Tensor, destination: torch.Tensor
) -> None:
    """Scatter packed pages into page dimension 2 with one Triton launch."""
    import triton

    from .triton.hicache import hicache_scatter_pages_kernel

    _validate(destination, page_ids, packed)
    page_elements = destination.shape[3] * destination.shape[4] * destination.shape[5]
    num_elements = packed.numel()
    hicache_scatter_pages_kernel[(triton.cdiv(num_elements, 256),)](
        packed,
        page_ids,
        destination,
        num_elements,
        destination.shape[1],
        destination.shape[2],
        page_elements,
        BLOCK_SIZE=256,
        num_warps=4,
    )


def _validate(pool: torch.Tensor, page_ids: torch.Tensor, packed: torch.Tensor) -> None:
    if not pool.is_cuda or not page_ids.is_cuda or not packed.is_cuda:
        raise ValueError("Triton HiCache packing requires CUDA tensors")
    if not pool.is_contiguous() or not packed.is_contiguous():
        raise ValueError("Triton HiCache tensors must be contiguous")
    if pool.ndim != 6 or packed.ndim != 6:
        raise ValueError("HiCache tensors must use [K/V, layer, page, token, head, dim]")
    if (
        packed.shape[1] != pool.shape[0]
        or packed.shape[2] != pool.shape[1]
        or packed.shape[3:] != pool.shape[3:]
    ):
        raise ValueError("Packed and pool KV geometry differ")
    if packed.shape[0] != len(page_ids):
        raise ValueError("Packed page count differs from page id count")
