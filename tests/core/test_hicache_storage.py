from __future__ import annotations

import os
from concurrent.futures import Future

import pytest
import torch
from minisgl.kvcache.mha_pool import HostMHAKVCache
from minisgl.kvcache.tiered_pool import (
    CacheTier,
    StorageMHAKVCache,
    TransferDirection,
    TransferState,
    TransferTicket,
    page_ids_from_token_indices,
)


def _make_host_pool(dtype: torch.dtype = torch.bfloat16) -> HostMHAKVCache:
    return HostMHAKVCache(
        num_layers=2,
        num_pages=4,
        page_size=2,
        local_kv_heads=1,
        head_dim=4,
        dtype=dtype,
        pin_memory=False,
    )


def test_page_ids_validate_complete_aligned_pages():
    indices = torch.tensor([0, 1, 4, 5], dtype=torch.int32)
    assert page_ids_from_token_indices(indices, page_size=2) == [0, 2]

    with pytest.raises(ValueError, match="not page aligned"):
        page_ids_from_token_indices(torch.tensor([0, 1, 2]), page_size=2)
    with pytest.raises(ValueError, match="complete, consecutive"):
        page_ids_from_token_indices(torch.tensor([0, 2]), page_size=2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_storage_pool_round_trip_and_slot_remapping(dtype: torch.dtype):
    source = _make_host_pool(dtype)
    restored = _make_host_pool(dtype)
    source.buffer.copy_(torch.arange(source.buffer.numel()).view(source.buffer.shape))
    storage = StorageMHAKVCache(
        num_layers=source.num_layers,
        num_pages=4,
        page_size=source.page_size,
        local_kv_heads=source.local_kv_heads,
        head_dim=source.head_dim,
        dtype=dtype,
    )
    temporary_path = storage.path

    source_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int32)
    storage_indices = torch.tensor([2, 3, 6, 7], dtype=torch.int32)
    assert storage.write_pages_from_host(source, source_indices, storage_indices) == 2
    assert storage.read_pages_to_host(storage_indices, restored, source_indices) == 2

    assert torch.equal(source.buffer[:, :, 0], restored.buffer[:, :, 0])
    assert torch.equal(source.buffer[:, :, 2], restored.buffer[:, :, 2])
    storage.close()
    assert not temporary_path.exists()


def test_explicit_storage_file_is_preallocated_and_preserved(tmp_path):
    path = tmp_path / "cache.kv"
    storage = StorageMHAKVCache(
        num_layers=1,
        num_pages=3,
        page_size=2,
        local_kv_heads=1,
        head_dim=4,
        dtype=torch.float16,
        path=str(path),
    )
    assert os.path.getsize(path) == storage.num_pages * storage.bytes_per_page
    storage.close()
    assert path.exists()


def test_storage_coalesces_adjacent_pages_into_one_extent():
    source = _make_host_pool(torch.float16)
    restored = _make_host_pool(torch.float16)
    source.buffer.copy_(torch.arange(source.buffer.numel()).view(source.buffer.shape))
    storage = StorageMHAKVCache(
        num_layers=source.num_layers,
        num_pages=4,
        page_size=source.page_size,
        local_kv_heads=source.local_kv_heads,
        head_dim=source.head_dim,
        dtype=source.dtype,
    )
    adjacent = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

    assert storage.write_pages_from_host(source, adjacent, adjacent) == 1
    assert storage.read_pages_to_host(adjacent, restored, adjacent) == 1
    assert torch.equal(source.buffer[:, :, :2], restored.buffer[:, :, :2])
    storage.close()


def test_storage_rejects_out_of_range_slots_and_short_reads(tmp_path):
    host = _make_host_pool(torch.float16)
    path = tmp_path / "corrupt.kv"
    storage = StorageMHAKVCache(
        num_layers=host.num_layers,
        num_pages=2,
        page_size=host.page_size,
        local_kv_heads=host.local_kv_heads,
        head_dim=host.head_dim,
        dtype=host.dtype,
        path=str(path),
    )
    one_page = torch.tensor([0, 1], dtype=torch.int32)
    outside_storage = torch.tensor([4, 5], dtype=torch.int32)
    with pytest.raises(IndexError, match="outside capacity"):
        storage.write_pages_from_host(host, one_page, outside_storage)

    storage.write_pages_from_host(host, one_page, one_page)
    os.truncate(path, 0)
    with pytest.raises(OSError, match="Short read"):
        storage.read_pages_to_host(one_page, host, one_page)
    storage.close()


def test_transfer_ticket_records_async_failure():
    future: Future[None] = Future()
    ticket = TransferTicket(
        direction=TransferDirection.H2S,
        source=CacheTier.HOST,
        destination=CacheTier.STORAGE,
        pages=1,
        num_bytes=128,
        future=future,
    )
    future.set_exception(OSError("simulated storage failure"))

    with pytest.raises(OSError, match="simulated storage failure"):
        ticket.wait()
    assert ticket.done
    assert ticket.state == TransferState.FAILED
    assert ticket.error == "simulated storage failure"
