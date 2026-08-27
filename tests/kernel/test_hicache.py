from __future__ import annotations

import minisgl.core as core
import minisgl.distributed.info as distributed_info
import pytest
import torch
from minisgl.core import Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.kvcache.mha_pool import HostMHAKVCache, MHAKVCache
from minisgl.kvcache.tiered_pool import CacheTransferManager, StorageMHAKVCache
from minisgl.scheduler.cache import CacheManager, PendingMaterialization
from minisgl.scheduler.utils import PendingReq

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(autouse=True)
def isolated_global_state(monkeypatch):
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 1))
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    yield
    torch.cuda.synchronize()


def _make_device_pool(num_pages: int = 8, page_size: int = 2) -> MHAKVCache:
    return MHAKVCache(
        num_kv_heads=2,
        num_layers=2,
        head_dim=8,
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )


@pytest.mark.parametrize("copy_backend", ["auto", "triton", "torch"])
def test_transfer_round_trip_across_all_tiers(tmp_path, copy_backend: str):
    device = _make_device_pool()
    host = HostMHAKVCache.from_device_pool(device, device.num_pages)
    storage = StorageMHAKVCache.from_device_pool(
        device, device.num_pages, path=str(tmp_path / "roundtrip.kv")
    )
    transfers = CacheTransferManager(
        device, host, storage, staging_pages=2, copy_backend=copy_backend
    )
    if copy_backend == "auto":
        transfers.warmup()
    expected = torch.arange(device.buffer.numel(), device="cuda", dtype=device.dtype).view(
        device.buffer.shape
    )
    device.buffer.copy_(expected)
    device_indices = torch.arange(device.num_pages * device.page_size, device="cuda")
    host_indices = device_indices.to(device="cpu", dtype=torch.int32)

    transfers.device_to_host(device_indices, host_indices).wait()
    assert torch.equal(host.buffer, expected.cpu())
    device.buffer.zero_()
    transfers.host_to_device(host_indices, device_indices).wait()
    assert torch.equal(device.buffer, expected)

    transfers.host_to_storage(host_indices, host_indices).wait()
    host.buffer.zero_()
    transfers.storage_to_host(host_indices, host_indices).wait()
    assert torch.equal(host.buffer, expected.cpu())
    device.buffer.zero_()
    transfers.storage_to_device(host_indices, device_indices)
    assert torch.equal(device.buffer, expected)
    if copy_backend == "auto":
        assert transfers.gather_backend in {"triton", "torch"}
        assert transfers.scatter_backend in {"triton", "torch"}
        assert transfers.autotune_pages > 0
        assert set(transfers.autotune_timings_ms) == {
            "gather_triton",
            "gather_torch",
            "scatter_triton",
            "scatter_torch",
        }
    else:
        assert transfers.copy_backend == copy_backend
    assert len(transfers._workspaces) == 1

    transfers.shutdown()
    storage.close()


def test_triton_scatter_waits_for_page_ids_producer_stream():
    device = _make_device_pool(num_pages=64, page_size=1)
    host = HostMHAKVCache.from_device_pool(device, device.num_pages)
    transfers = CacheTransferManager(device, host, None, staging_pages=2, copy_backend="triton")
    host.buffer.copy_(torch.arange(host.buffer.numel(), dtype=host.dtype).view(host.buffer.shape))
    device.buffer.zero_()
    host_indices = torch.arange(device.num_pages, dtype=torch.int32)
    device_indices = host_indices.to(device="cuda")

    # Keep the producer stream busy immediately before host_to_device creates its
    # private CUDA page-id tensor.  The transfer stream must not consume that tensor
    # until the producer stream has initialized it.
    torch.cuda._sleep(10_000_000)
    transfers.host_to_device(host_indices, device_indices).wait()

    assert torch.equal(device.buffer.cpu(), host.buffer)
    transfers.shutdown()


@pytest.mark.parametrize("page_size", [1, 2, 4])
def test_cache_manager_restores_l3_after_l1_and_l2_eviction(tmp_path, page_size: int):
    num_pages = 6
    core.set_global_ctx(core.Context(page_size=page_size))
    device_pool = _make_device_pool(num_pages=num_pages + 1, page_size=page_size)
    page_table = torch.zeros((2, 16), dtype=torch.int32, device="cuda")
    manager = CacheManager(
        num_pages,
        page_size,
        page_table,
        type="radix",
        kv_cache=device_pool,
        enable_hicache=True,
        hicache_ratio=1.0,
        hicache_storage_ratio=2.0,
        hicache_storage_path=str(tmp_path / "tiered.kv"),
        hicache_staging_pages=2,
        hicache_policy="always",
        hicache_transfer_backend="triton",
    )

    input_ids = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
    pending = PendingReq(0, input_ids, SamplingParams(max_tokens=1))
    initial_match = manager.match_req(pending)
    manager.lock(initial_match.cuda_handle)
    initial_handle = manager.materialize_match(pending, initial_match, table_idx=0)
    req = Req(
        input_ids=input_ids[:4],
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=0,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=initial_handle,
    )
    manager.allocate_paged([req])
    original_indices = page_table[0, :4]
    original_pages = (original_indices[::page_size] // page_size).tolist()
    for logical_page, physical_page in enumerate(original_pages):
        device_pool.buffer[:, :, physical_page].fill_(logical_page + 1)
    req.cached_len = 4
    manager.cache_req(req, finished=True)
    assert manager.has_pending_transfers
    manager.check_integrity()
    assert not manager.has_pending_transfers

    # Force the complete L1 leaf out, return the temporary allocation, then erase GPU data.
    cached_pages = 4 // page_size
    pressure_pages = num_pages - cached_pages + 1
    temporary_device = manager._allocate(pressure_pages)
    manager._free(manager._page_to_token(temporary_device))
    device_pool.buffer.zero_()

    # Force the same prefix out of L2 while retaining its write-through L3 copy.
    assert manager.host_tier is not None
    temporary_host = manager.host_tier.allocate_tokens(pressure_pages * page_size)
    assert temporary_host is not None
    manager.host_tier.free(temporary_host)

    restored_match = manager.match_req(pending)
    assert restored_match.cuda_handle.cached_len == 0
    assert restored_match.host_handle is not None
    assert restored_match.host_handle.cached_len == 0
    assert restored_match.storage_handle is not None
    assert restored_match.storage_handle.cached_len == 4

    manager.lock(restored_match.cuda_handle)
    materialized = manager.begin_materialize_match(
        pending, restored_match, table_idx=1, allow_async=True
    )
    assert isinstance(materialized, PendingMaterialization)
    manager.mark_materializations_overlapped()
    restored_handle = manager.progress_materialization(materialized, blocking=True)
    assert restored_handle is not None
    assert restored_handle.cached_len == 4
    restored_indices = restored_handle.get_matched_indices()
    restored_pages = (restored_indices[::page_size] // page_size).tolist()
    for logical_page, physical_page in enumerate(restored_pages):
        expected = torch.full_like(device_pool.buffer[:, :, physical_page], logical_page + 1)
        assert torch.equal(device_pool.buffer[:, :, physical_page], expected)

    assert manager.metrics.l3_hit_tokens == 4
    assert manager.metrics.storage_promotions > 0
    assert manager.metrics.storage_read_bytes > 0
    assert manager.metrics.h2d_bytes > 0
    assert manager.metrics.storage_read_extents > 0
    assert manager.metrics.transfer_enqueue_seconds > 0
    assert manager.metrics.restore_overlapped_seconds > 0
    assert manager.metrics.restore_overlapped_seconds <= manager.metrics.restore_e2e_seconds
    assert manager.transfer_manager is not None
    assert manager.transfer_manager.copy_backend == "triton"
    manager.unlock(restored_handle)
    manager.check_integrity()
    manager.shutdown()
