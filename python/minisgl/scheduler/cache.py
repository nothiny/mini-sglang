from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Literal, Tuple

import torch
from minisgl.core import Req
from minisgl.distributed import try_get_tp_info
from minisgl.kvcache import (
    BaseCacheHandle,
    BasePrefixCache,
    CacheTier,
    CacheTransferManager,
    HostMHAKVCache,
    MatchResult,
    MHAKVCache,
    StorageMHAKVCache,
    TransferDirection,
    TransferTicket,
    create_prefix_cache,
)
from minisgl.utils import div_ceil, init_logger

if TYPE_CHECKING:
    from .utils import PendingReq


logger = init_logger(__name__)


@dataclass
class HiCacheMetrics:
    l1_hit_tokens: int = 0
    l2_hit_tokens: int = 0
    l3_hit_tokens: int = 0
    recomputed_tokens: int = 0
    device_evictions: int = 0
    host_evictions: int = 0
    storage_evictions: int = 0
    host_promotions: int = 0
    storage_promotions: int = 0
    backup_failures: int = 0
    restore_fallbacks: int = 0
    d2h_bytes: int = 0
    h2d_bytes: int = 0
    storage_write_bytes: int = 0
    storage_read_bytes: int = 0
    d2h_transfers: int = 0
    h2d_transfers: int = 0
    storage_write_transfers: int = 0
    storage_read_transfers: int = 0
    storage_write_extents: int = 0
    storage_read_extents: int = 0
    d2h_seconds: float = 0.0
    h2d_seconds: float = 0.0
    storage_write_seconds: float = 0.0
    storage_read_seconds: float = 0.0
    transfer_enqueue_seconds: float = 0.0
    transfer_workspace_setup_seconds: float = 0.0
    transfer_queue_wait_seconds: float = 0.0
    transfer_service_seconds: float = 0.0
    transfer_seconds: float = 0.0
    restore_e2e_seconds: float = 0.0
    restore_overlapped_seconds: float = 0.0
    restore_requests: int = 0
    overlapped_restore_requests: int = 0
    policy_restore_skips: int = 0
    policy_backup_skips: int = 0

    def record_transfer(self, ticket: TransferTicket) -> None:
        ticket.wait()
        self.transfer_seconds += ticket.latency_seconds
        self.transfer_enqueue_seconds += ticket.enqueue_seconds
        self.transfer_workspace_setup_seconds += ticket.workspace_setup_seconds
        self.transfer_queue_wait_seconds += ticket.queue_wait_seconds
        self.transfer_service_seconds += ticket.service_seconds
        if ticket.direction == TransferDirection.D2H:
            self.d2h_bytes += ticket.num_bytes
            self.d2h_transfers += 1
            self.d2h_seconds += ticket.recurring_seconds
        elif ticket.direction == TransferDirection.H2D:
            self.h2d_bytes += ticket.num_bytes
            self.h2d_transfers += 1
            self.h2d_seconds += ticket.recurring_seconds
        elif ticket.direction == TransferDirection.H2S:
            self.storage_write_bytes += ticket.num_bytes
            self.storage_write_transfers += 1
            self.storage_write_extents += ticket.io_operations
            self.storage_write_seconds += ticket.recurring_seconds
        elif ticket.direction == TransferDirection.S2H:
            self.storage_read_bytes += ticket.num_bytes
            self.storage_read_transfers += 1
            self.storage_read_extents += ticket.io_operations
            self.storage_read_seconds += ticket.recurring_seconds

    def snapshot(self) -> Dict[str, int | float]:
        result = asdict(self)
        for name in ("d2h", "h2d", "storage_write", "storage_read"):
            num_bytes = float(result[f"{name}_bytes"])
            seconds = float(result[f"{name}_seconds"])
            result[f"{name}_gib_s"] = num_bytes / seconds / (1 << 30) if seconds > 0 else 0.0
        return result


class HiCacheCostModel:
    """Online cost model for restore and write-through admission decisions."""

    def __init__(
        self,
        *,
        policy: str,
        recompute_us_per_token: float,
        host_bandwidth_gib_s: float,
        storage_bandwidth_gib_s: float,
        margin: float,
    ) -> None:
        if policy not in {"always", "cost"}:
            raise ValueError("HiCache policy must be 'always' or 'cost'")
        if (
            min(
                recompute_us_per_token,
                host_bandwidth_gib_s,
                storage_bandwidth_gib_s,
                margin,
            )
            <= 0
        ):
            raise ValueError("HiCache cost-model parameters must be positive")
        self.policy = policy
        self.margin = margin
        self.recompute_seconds_per_token = recompute_us_per_token / 1_000_000
        self._default_seconds_per_byte = {
            TransferDirection.D2H: 1 / (host_bandwidth_gib_s * (1 << 30)),
            TransferDirection.H2D: 1 / (host_bandwidth_gib_s * (1 << 30)),
            TransferDirection.H2S: 1 / (storage_bandwidth_gib_s * (1 << 30)),
            TransferDirection.S2H: 1 / (storage_bandwidth_gib_s * (1 << 30)),
        }
        self._observed_seconds_per_byte: Dict[TransferDirection, float] = {}
        self.prefill_observations = 0
        self.transfer_observations = 0
        self.estimated_saved_seconds = 0.0

    def observe_prefill(self, tokens: int, seconds: float) -> None:
        # Tiny extensions are dominated by fixed scheduling and sampling overhead.
        if tokens < 16 or seconds <= 0:
            return
        sample = seconds / tokens
        self.recompute_seconds_per_token = self._ewma(self.recompute_seconds_per_token, sample)
        self.prefill_observations += 1

    def observe_transfer(self, ticket: TransferTicket) -> None:
        if ticket.num_bytes <= 0 or ticket.state.value != "completed":
            return
        sample = ticket.recurring_seconds / ticket.num_bytes
        previous = self._observed_seconds_per_byte.get(
            ticket.direction, self._default_seconds_per_byte[ticket.direction]
        )
        self._observed_seconds_per_byte[ticket.direction] = self._ewma(previous, sample)
        self.transfer_observations += 1

    def select_restore(
        self,
        *,
        cuda_len: int,
        host_len: int,
        storage_len: int,
        bytes_per_token: int,
    ) -> tuple[CacheTier, int]:
        candidates: List[tuple[float, CacheTier, int]] = [(0.0, CacheTier.GPU, cuda_len)]
        if host_len > cuda_len:
            candidates.append(
                (
                    self._restore_benefit(
                        host_len - cuda_len,
                        bytes_per_token,
                        (TransferDirection.H2D,),
                    ),
                    CacheTier.HOST,
                    host_len,
                )
            )
        if storage_len > cuda_len:
            candidates.append(
                (
                    self._restore_benefit(
                        storage_len - cuda_len,
                        bytes_per_token,
                        (TransferDirection.S2H, TransferDirection.H2D),
                    ),
                    CacheTier.STORAGE,
                    storage_len,
                )
            )
        if self.policy == "always":
            _, tier, target = max(
                candidates, key=lambda item: (item[2], -list(CacheTier).index(item[1]))
            )
            return tier, target
        benefit, tier, target = max(candidates, key=lambda item: item[0])
        if benefit > 0:
            self.estimated_saved_seconds += benefit
            return tier, target
        return CacheTier.GPU, cuda_len

    def should_backup(self, *, tier: CacheTier, tokens: int, bytes_per_token: int) -> bool:
        """Admit a prefix when a future restore can beat recomputation.

        Write-through runs in the background, so charging its D2H/H2S cost to the
        request critical path makes the policy unnecessarily conservative.  The
        online transfer observations still capture contention in the restore path.
        """
        if self.policy == "always":
            return True
        num_bytes = tokens * bytes_per_token
        recompute = tokens * self.recompute_seconds_per_token
        directions: tuple[TransferDirection, ...]
        if tier == CacheTier.HOST:
            directions = (TransferDirection.H2D,)
        else:
            directions = (
                TransferDirection.S2H,
                TransferDirection.H2D,
            )
        total = sum(self._estimate_transfer(direction, num_bytes) for direction in directions)
        return total * self.margin < recompute

    def snapshot(self) -> Dict[str, int | float | str]:
        result: Dict[str, int | float | str] = {
            "policy": self.policy,
            "margin": self.margin,
            "recompute_us_per_token": self.recompute_seconds_per_token * 1_000_000,
            "prefill_observations": self.prefill_observations,
            "transfer_observations": self.transfer_observations,
            "estimated_saved_seconds": self.estimated_saved_seconds,
        }
        for direction in TransferDirection:
            seconds_per_byte = self._observed_seconds_per_byte.get(
                direction, self._default_seconds_per_byte[direction]
            )
            result[f"estimated_{direction.value}_gib_s"] = 1 / (seconds_per_byte * (1 << 30))
        return result

    def _restore_benefit(
        self,
        tokens: int,
        bytes_per_token: int,
        directions: tuple[TransferDirection, ...],
    ) -> float:
        if self.policy == "always":
            return float(tokens)
        recompute = tokens * self.recompute_seconds_per_token
        num_bytes = tokens * bytes_per_token
        transfer = sum(self._estimate_transfer(direction, num_bytes) for direction in directions)
        return recompute - transfer * self.margin

    def _estimate_transfer(self, direction: TransferDirection, num_bytes: int) -> float:
        rate = self._observed_seconds_per_byte.get(direction)
        if rate is None and direction in {
            TransferDirection.D2H,
            TransferDirection.H2D,
        }:
            reverse = (
                TransferDirection.H2D
                if direction == TransferDirection.D2H
                else TransferDirection.D2H
            )
            rate = self._observed_seconds_per_byte.get(
                reverse, self._default_seconds_per_byte[direction]
            )
        elif rate is None:
            # Storage read and write throughput is commonly asymmetric.  A slow
            # write-through observation must not suppress a potentially fast read
            # before the first restore has supplied an S2H sample.
            rate = self._default_seconds_per_byte[direction]
        return num_bytes * rate

    @staticmethod
    def _ewma(previous: float, sample: float, alpha: float = 0.2) -> float:
        return previous * (1 - alpha) + sample * alpha


@dataclass
class _PendingHostBackup:
    input_ids: torch.Tensor
    device_handle: BaseCacheHandle
    existing_host_handle: BaseCacheHandle
    new_host_indices: torch.Tensor
    target_len: int
    ticket: TransferTicket
    backup_storage: bool


@dataclass
class _PendingStorageBackup:
    input_ids: torch.Tensor
    source_host_handle: BaseCacheHandle
    existing_storage_handle: BaseCacheHandle
    new_storage_indices: torch.Tensor
    target_len: int
    ticket: TransferTicket


@dataclass
class PendingMaterialization:
    """Private L1 allocation and transfer chain for one lower-tier prefix hit."""

    input_ids: torch.Tensor
    input_len: int
    table_idx: int
    cuda_handle: BaseCacheHandle
    cuda_len: int
    host_len: int
    storage_len: int
    target_len: int
    restore_tier: CacheTier
    allocated_pages: torch.Tensor
    device_indices: torch.Tensor
    ticket: TransferTicket
    phase: Literal["s2h", "h2d"]
    source_handle: BaseCacheHandle | None
    started_at: float
    host_match: BaseCacheHandle | None = None
    host_new_indices: torch.Tensor | None = None
    storage_handle: BaseCacheHandle | None = None
    overlap_confirmed: bool = False
    overlap_started_at: float | None = None
    device_pages_private: bool = True
    cuda_handle_locked: bool = True
    published_handle: BaseCacheHandle | None = None
    completed_handle: BaseCacheHandle | None = None
    finished: bool = False


class _LowerTierManager:
    """Page allocator plus Radix Tree metadata for one non-GPU tier."""

    def __init__(self, name: str, num_pages: int, page_size: int) -> None:
        self.name = name
        self.num_pages = num_pages
        self.page_size = page_size
        self.prefix_cache: BasePrefixCache = create_prefix_cache(
            device=torch.device("cpu"), type="radix"
        )
        self.free_slots = torch.arange(num_pages, dtype=torch.int32) * page_size
        self.evicted_tokens = 0

    @property
    def capacity_tokens(self) -> int:
        return self.num_pages * self.page_size

    @property
    def available_size(self) -> int:
        return (
            int(self.prefix_cache.size_info.evictable_size) + len(self.free_slots) * self.page_size
        )

    def allocate_tokens(self, num_tokens: int) -> torch.Tensor | None:
        if num_tokens == 0:
            return torch.empty(0, dtype=torch.int32)
        if num_tokens % self.page_size != 0:
            raise ValueError(f"{self.name} allocation is not page aligned")
        needed_pages = num_tokens // self.page_size
        if num_tokens > self.available_size:
            return None
        if needed_pages > len(self.free_slots):
            evict_tokens = (needed_pages - len(self.free_slots)) * self.page_size
            evicted = self.prefix_cache.evict(evict_tokens)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size].cpu()])
            self.free_slots = torch.sort(self.free_slots).values
            self.evicted_tokens += len(evicted)
        pages = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return self.page_to_token(pages)

    def free(self, indices: torch.Tensor) -> None:
        if len(indices) == 0:
            return
        if len(indices) % self.page_size != 0:
            raise ValueError(f"{self.name} free is not page aligned")
        starts = indices.detach().cpu()[:: self.page_size]
        self.free_slots = torch.cat([self.free_slots, starts])
        self.free_slots = torch.sort(self.free_slots).values

    def page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages.to(torch.int32)
        offsets = torch.arange(self.page_size, dtype=torch.int32)
        return (pages.to(torch.int32).unsqueeze(1) + offsets).flatten()

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                f"{self.name} integrity check failed: free_pages({len(self.free_slots)}) + "
                f"cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1 and not torch.all(self.free_slots % self.page_size == 0):
            raise RuntimeError(f"{self.name} free list contains unaligned pages")
        if len(self.free_slots) != len(set(self.free_slots.tolist())):
            raise RuntimeError(f"{self.name} free list contains duplicate pages")


class CacheManager:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        type: str,
        *,
        kv_cache: MHAKVCache | None = None,
        enable_hicache: bool = False,
        hicache_size_gb: float | None = None,
        hicache_ratio: float = 1.0,
        hicache_storage_size_gb: float | None = None,
        hicache_storage_ratio: float = 0.0,
        hicache_storage_path: str | None = None,
        hicache_io_workers: int = 2,
        hicache_staging_pages: int = 8,
        hicache_promote_storage: bool = True,
        hicache_policy: str = "cost",
        hicache_recompute_us_per_token: float = 50.0,
        hicache_host_bandwidth_gib_s: float = 12.0,
        hicache_storage_bandwidth_gib_s: float = 3.0,
        hicache_cost_margin: float = 1.1,
        hicache_transfer_backend: str = "auto",
    ) -> None:
        # Free slots are page starts. For page_size=2: [0, 2, 4, ...].
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache: BasePrefixCache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        self.metrics = HiCacheMetrics()
        self.bytes_per_token = (
            kv_cache.bytes_per_page // page_size if isinstance(kv_cache, MHAKVCache) else 0
        )
        self.cost_model = HiCacheCostModel(
            policy=hicache_policy,
            recompute_us_per_token=hicache_recompute_us_per_token,
            host_bandwidth_gib_s=hicache_host_bandwidth_gib_s,
            storage_bandwidth_gib_s=hicache_storage_bandwidth_gib_s,
            margin=hicache_cost_margin,
        )

        self.hicache_enabled = enable_hicache
        self.hicache_promote_storage = hicache_promote_storage
        self.host_pool: HostMHAKVCache | None = None
        self.storage_pool: StorageMHAKVCache | None = None
        self.host_tier: _LowerTierManager | None = None
        self.storage_tier: _LowerTierManager | None = None
        self.transfer_manager: CacheTransferManager | None = None
        self._pending_host_backups: List[_PendingHostBackup] = []
        self._pending_storage_backups: List[_PendingStorageBackup] = []
        self._pending_materializations: List[PendingMaterialization] = []
        self._polling_transfers = False

        if not enable_hicache:
            return
        if type != "radix":
            raise ValueError("HiCache requires cache_type='radix'")
        if kv_cache is None or not isinstance(kv_cache, MHAKVCache):
            raise TypeError("HiCache currently requires an MHAKVCache device pool")
        if hicache_io_workers < 1 or hicache_staging_pages < 1:
            raise ValueError("HiCache I/O workers and staging pages must be positive")

        host_pages = self._capacity_pages(
            explicit_gb=hicache_size_gb,
            ratio=hicache_ratio,
            device_pages=num_pages,
            bytes_per_page=kv_cache.bytes_per_page,
            name="host",
        )
        if host_pages < 1:
            raise ValueError("HiCache requires at least one pinned-host page")
        storage_pages = self._capacity_pages(
            explicit_gb=hicache_storage_size_gb,
            ratio=hicache_storage_ratio,
            device_pages=num_pages,
            bytes_per_page=kv_cache.bytes_per_page,
            name="storage",
            allow_zero=True,
        )

        self.host_pool = HostMHAKVCache.from_device_pool(kv_cache, host_pages)
        self.host_tier = _LowerTierManager("HiCache L2", host_pages, page_size)
        if storage_pages > 0:
            storage_path = self._resolve_storage_path(hicache_storage_path)
            self.storage_pool = StorageMHAKVCache.from_device_pool(
                kv_cache, storage_pages, path=storage_path
            )
            self.storage_tier = _LowerTierManager("HiCache L3", storage_pages, page_size)

        self.transfer_manager = CacheTransferManager(
            device_pool=kv_cache,
            host_pool=self.host_pool,
            storage_pool=self.storage_pool,
            io_workers=hicache_io_workers,
            staging_pages=hicache_staging_pages,
            copy_backend=hicache_transfer_backend,
        )
        self.transfer_manager.warmup()
        host_gib = host_pages * kv_cache.bytes_per_page / (1 << 30)
        storage_gib = storage_pages * kv_cache.bytes_per_page / (1 << 30)
        logger.info_rank0(
            "HiCache enabled: L1=%d pages, L2=%d pages (%.2f GiB), "
            "L3=%d pages (%.2f GiB), copies=%s",
            num_pages,
            host_pages,
            host_gib,
            storage_pages,
            storage_gib,
            self.transfer_manager.copy_backend,
        )

    def match_req(self, req: PendingReq) -> MatchResult:
        self.poll_transfers(blocking=False)
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        input_ids = req.input_ids[: input_len - 1]
        result = self._match_prefixes(input_ids)
        if result.cuda_handle.cached_len == 0 and self.has_pending_transfers:
            # A lower-tier-only hit may still be waiting for metadata publication.
            self.poll_transfers(blocking=True)
            result = self._match_prefixes(input_ids)
        return result

    @property
    def has_pending_transfers(self) -> bool:
        return bool(self._pending_host_backups or self._pending_storage_backups)

    def hicache_status(self) -> Dict[str, object]:
        tiers: Dict[str, object] = {
            "l1": {
                "capacity_pages": self.num_pages,
                "free_pages": len(self.free_slots),
                "evictable_tokens": self.prefix_cache.size_info.evictable_size,
                "protected_tokens": self.prefix_cache.size_info.protected_size,
            }
        }
        if self.host_tier is not None:
            tiers["l2"] = {
                "capacity_pages": self.host_tier.num_pages,
                "free_pages": len(self.host_tier.free_slots),
                "evictable_tokens": self.host_tier.prefix_cache.size_info.evictable_size,
                "protected_tokens": self.host_tier.prefix_cache.size_info.protected_size,
            }
        if self.storage_tier is not None:
            tiers["l3"] = {
                "capacity_pages": self.storage_tier.num_pages,
                "free_pages": len(self.storage_tier.free_slots),
                "evictable_tokens": self.storage_tier.prefix_cache.size_info.evictable_size,
                "protected_tokens": self.storage_tier.prefix_cache.size_info.protected_size,
                "path": str(self.storage_pool.path) if self.storage_pool is not None else None,
            }
        return {
            "enabled": self.hicache_enabled,
            "tiers": tiers,
            "pending_host_backups": len(self._pending_host_backups),
            "pending_storage_backups": len(self._pending_storage_backups),
            "pending_materializations": len(self._pending_materializations),
            "metrics": self.metrics.snapshot(),
            "cost_model": self.cost_model.snapshot(),
            "transfer_backend": (
                self.transfer_manager.copy_backend if self.transfer_manager is not None else None
            ),
            "transfer_backends": (
                {
                    "gather": self.transfer_manager.gather_backend,
                    "scatter": self.transfer_manager.scatter_backend,
                }
                if self.transfer_manager is not None
                else None
            ),
            "transfer_autotune_ms": (
                self.transfer_manager.autotune_timings_ms
                if self.transfer_manager is not None
                else None
            ),
            "transfer_autotune_pages": (
                self.transfer_manager.autotune_pages if self.transfer_manager is not None else 0
            ),
        }

    def poll_transfers(self, *, blocking: bool) -> None:
        if self._polling_transfers or not self.has_pending_transfers:
            return
        self._polling_transfers = True
        try:
            while True:
                progressed = False
                for storage_pending in list(self._pending_storage_backups):
                    if blocking or storage_pending.ticket.done:
                        self._pending_storage_backups.remove(storage_pending)
                        self._finalize_storage_backup(storage_pending)
                        progressed = True
                for host_pending in list(self._pending_host_backups):
                    if blocking or host_pending.ticket.done:
                        self._pending_host_backups.remove(host_pending)
                        self._finalize_host_backup(host_pending)
                        progressed = True
                if not blocking or not self.has_pending_transfers:
                    break
                if not progressed:
                    # With blocking=True every ticket is eligible, so this is defensive only.
                    raise RuntimeError("HiCache transfer queue made no progress")
        finally:
            self._polling_transfers = False

    def observe_prefill(self, tokens: int, seconds: float) -> None:
        self.cost_model.observe_prefill(tokens, seconds)

    def _record_transfer(self, ticket: TransferTicket) -> None:
        self.metrics.record_transfer(ticket)
        self.cost_model.observe_transfer(ticket)

    def _record_match_metrics(
        self, *, cuda_len: int, host_len: int, actual_len: int, input_len: int
    ) -> None:
        self.metrics.l1_hit_tokens += cuda_len
        self.metrics.l2_hit_tokens += max(0, min(actual_len, host_len) - cuda_len)
        self.metrics.l3_hit_tokens += max(0, actual_len - max(cuda_len, host_len))
        self.metrics.recomputed_tokens += input_len - actual_len

    @property
    def available_size(self) -> int:
        return (
            int(self.prefix_cache.size_info.evictable_size) + len(self.free_slots) * self.page_size
        )

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def materialize_match(
        self,
        req: PendingReq,
        match: MatchResult,
        table_idx: int,
        *,
        _selection: tuple[CacheTier, int] | None = None,
    ) -> BaseCacheHandle:
        """Make the longest matched prefix GPU resident and return its locked L1 handle.

        The caller must lock ``match.cuda_handle`` before this method. On a successful lower-tier
        restore, that lock is atomically replaced with a lock on the promoted L1 handle.
        """
        cuda_handle = match.cuda_handle
        cuda_len = cuda_handle.cached_len
        self._write_matched_page_table(table_idx, cuda_handle)

        host_len = match.host_handle.cached_len if match.host_handle is not None else 0
        storage_len = match.storage_handle.cached_len if match.storage_handle is not None else 0
        restore_tier, target_len = _selection or self.cost_model.select_restore(
            cuda_len=cuda_len,
            host_len=host_len,
            storage_len=storage_len,
            bytes_per_token=self.bytes_per_token,
        )
        actual_len = cuda_len
        restore_started_at = time.perf_counter() if target_len > cuda_len else None
        if restore_started_at is not None:
            self.metrics.restore_requests += 1

        if target_len == cuda_len and max(host_len, storage_len) > cuda_len:
            self.metrics.policy_restore_skips += max(host_len, storage_len) - cuda_len

        if target_len > cuda_len and self.transfer_manager is not None:
            needed_tokens = target_len - cuda_len
            allocated_pages: torch.Tensor | None = None
            allocation_published = False
            try:
                allocated_pages = self._allocate(needed_tokens // self.page_size)
                device_indices = self._page_to_token(allocated_pages)
                if restore_tier == CacheTier.STORAGE:
                    assert match.storage_handle is not None and self.storage_tier is not None
                    self._restore_from_storage(
                        req.input_ids,
                        match.storage_handle,
                        device_indices,
                        cuda_len,
                        target_len,
                    )
                elif restore_tier == CacheTier.HOST:
                    assert match.host_handle is not None and self.host_tier is not None
                    self._restore_from_host(match.host_handle, device_indices, cuda_len, target_len)
                else:
                    raise RuntimeError(f"Unsupported HiCache restore tier: {restore_tier}")

                cuda_indices = cuda_handle.get_matched_indices()
                proposed_indices = torch.cat([cuda_indices, device_indices])
                inserted = self.prefix_cache.insert_prefix(
                    req.input_ids[:target_len], proposed_indices
                )
                duplicate_len = inserted.cached_len - cuda_len
                if duplicate_len > 0:
                    self._free(device_indices[:duplicate_len])
                allocation_published = True
                self.unlock(cuda_handle)
                cuda_handle = inserted.handle
                self.lock(cuda_handle)
                self._write_matched_page_table(table_idx, cuda_handle)
                actual_len = cuda_handle.cached_len
            except Exception as exc:
                if allocated_pages is not None and not allocation_published:
                    self._free(self._page_to_token(allocated_pages))
                self.metrics.restore_fallbacks += 1
                logger.warning_rank0("HiCache restore failed; recomputing from L1 prefix: %s", exc)

        if restore_started_at is not None:
            self.metrics.restore_e2e_seconds += time.perf_counter() - restore_started_at
        self._record_match_metrics(
            cuda_len=cuda_len,
            host_len=host_len,
            actual_len=actual_len,
            input_len=req.input_len,
        )
        return cuda_handle

    def begin_materialize_match(
        self,
        req: PendingReq,
        match: MatchResult,
        table_idx: int,
        *,
        allow_async: bool,
    ) -> BaseCacheHandle | PendingMaterialization:
        """Start a lower-tier restore without stalling a schedulable decode batch."""
        if not allow_async or self.transfer_manager is None:
            return self.materialize_match(req, match, table_idx)

        cuda_handle = match.cuda_handle
        cuda_len = cuda_handle.cached_len
        host_len = match.host_handle.cached_len if match.host_handle is not None else 0
        storage_len = match.storage_handle.cached_len if match.storage_handle is not None else 0
        selection = self.cost_model.select_restore(
            cuda_len=cuda_len,
            host_len=host_len,
            storage_len=storage_len,
            bytes_per_token=self.bytes_per_token,
        )
        restore_tier, target_len = selection
        if target_len <= cuda_len:
            return self.materialize_match(req, match, table_idx, _selection=selection)

        self._write_matched_page_table(table_idx, cuda_handle)
        started_at = time.perf_counter()
        needed_pages = (target_len - cuda_len) // self.page_size

        if restore_tier == CacheTier.HOST:
            assert match.host_handle is not None and self.host_tier is not None
            host_handle = match.host_handle
            self.host_tier.prefix_cache.lock_handle(host_handle)
            allocated_pages: torch.Tensor | None = None
            try:
                allocated_pages = self._allocate(needed_pages)
                device_indices = self._page_to_token(allocated_pages)
                host_indices = host_handle.get_matched_indices()[cuda_len:target_len]
                ticket = self.transfer_manager.host_to_device(host_indices, device_indices)
                pending = PendingMaterialization(
                    input_ids=req.input_ids,
                    input_len=req.input_len,
                    table_idx=table_idx,
                    cuda_handle=cuda_handle,
                    cuda_len=cuda_len,
                    host_len=host_len,
                    storage_len=storage_len,
                    target_len=target_len,
                    restore_tier=restore_tier,
                    allocated_pages=allocated_pages,
                    device_indices=device_indices,
                    ticket=ticket,
                    phase="h2d",
                    source_handle=host_handle,
                    started_at=started_at,
                )
                self._pending_materializations.append(pending)
                self.metrics.restore_requests += 1
                return pending
            except Exception as exc:
                if allocated_pages is not None:
                    self._free(self._page_to_token(allocated_pages))
                self.host_tier.prefix_cache.lock_handle(host_handle, unlock=True)
                return self._fallback_materialization_start(
                    req, cuda_handle, cuda_len, host_len, table_idx, started_at, exc
                )

        if restore_tier == CacheTier.STORAGE:
            assert match.storage_handle is not None and self.storage_tier is not None
            if (
                not self.hicache_promote_storage
                or self.host_tier is None
                or target_len > self.host_tier.capacity_tokens
            ):
                return self.materialize_match(req, match, table_idx, _selection=selection)

            storage_handle: BaseCacheHandle | None = match.storage_handle
            assert storage_handle is not None
            self.storage_tier.prefix_cache.lock_handle(storage_handle)
            allocated_pages = None
            host_match: BaseCacheHandle | None = None
            host_new_indices: torch.Tensor | None = None
            host_locked = False
            try:
                allocated_pages = self._allocate(needed_pages)
                device_indices = self._page_to_token(allocated_pages)
                host_match = self.host_tier.prefix_cache.match_prefix(
                    req.input_ids[:target_len]
                ).cuda_handle
                self.host_tier.prefix_cache.lock_handle(host_match)
                host_locked = True
                if host_match.cached_len >= target_len:
                    self.storage_tier.prefix_cache.lock_handle(storage_handle, unlock=True)
                    storage_handle = None
                    host_indices = host_match.get_matched_indices()[cuda_len:target_len]
                    ticket = self.transfer_manager.host_to_device(host_indices, device_indices)
                    pending = PendingMaterialization(
                        input_ids=req.input_ids,
                        input_len=req.input_len,
                        table_idx=table_idx,
                        cuda_handle=cuda_handle,
                        cuda_len=cuda_len,
                        host_len=host_len,
                        storage_len=storage_len,
                        target_len=target_len,
                        restore_tier=restore_tier,
                        allocated_pages=allocated_pages,
                        device_indices=device_indices,
                        ticket=ticket,
                        phase="h2d",
                        source_handle=host_match,
                        started_at=started_at,
                    )
                else:
                    host_new_indices = self.host_tier.allocate_tokens(
                        target_len - host_match.cached_len
                    )
                    if host_new_indices is None:
                        raise RuntimeError("No L2 capacity for asynchronous L3 promotion")
                    storage_indices = storage_handle.get_matched_indices()[
                        host_match.cached_len : target_len
                    ]
                    ticket = self.transfer_manager.storage_to_host(
                        storage_indices, host_new_indices
                    )
                    pending = PendingMaterialization(
                        input_ids=req.input_ids,
                        input_len=req.input_len,
                        table_idx=table_idx,
                        cuda_handle=cuda_handle,
                        cuda_len=cuda_len,
                        host_len=host_len,
                        storage_len=storage_len,
                        target_len=target_len,
                        restore_tier=restore_tier,
                        allocated_pages=allocated_pages,
                        device_indices=device_indices,
                        ticket=ticket,
                        phase="s2h",
                        source_handle=None,
                        started_at=started_at,
                        host_match=host_match,
                        host_new_indices=host_new_indices,
                        storage_handle=storage_handle,
                    )
                self._pending_materializations.append(pending)
                self.metrics.restore_requests += 1
                return pending
            except Exception as exc:
                if host_new_indices is not None:
                    self.host_tier.free(host_new_indices)
                if host_locked and host_match is not None:
                    self.host_tier.prefix_cache.lock_handle(host_match, unlock=True)
                if storage_handle is not None:
                    self.storage_tier.prefix_cache.lock_handle(storage_handle, unlock=True)
                if allocated_pages is not None:
                    self._free(self._page_to_token(allocated_pages))
                return self._fallback_materialization_start(
                    req, cuda_handle, cuda_len, host_len, table_idx, started_at, exc
                )

        return self.materialize_match(req, match, table_idx, _selection=selection)

    def progress_materialization(
        self, pending: PendingMaterialization, *, blocking: bool
    ) -> BaseCacheHandle | None:
        if pending.finished:
            return pending.completed_handle
        assert self.transfer_manager is not None
        if not blocking and not pending.ticket.done:
            return None

        try:
            if pending.phase == "s2h":
                self._record_transfer(pending.ticket)
                assert self.host_tier is not None
                assert pending.host_match is not None
                assert pending.host_new_indices is not None
                combined = torch.cat(
                    [
                        pending.host_match.get_matched_indices(),
                        pending.host_new_indices,
                    ]
                )
                inserted = self.host_tier.prefix_cache.insert_prefix(
                    pending.input_ids[: pending.target_len], combined
                )
                duplicate_len = inserted.cached_len - pending.host_match.cached_len
                # The radix tree owns every non-duplicate page after insertion.  Clear
                # private ownership before launching H2D so an allocation failure cannot
                # return published L2 pages to the free list.
                pending.host_new_indices = None
                if duplicate_len > 0:
                    self.host_tier.free(combined[pending.host_match.cached_len :][:duplicate_len])
                self.host_tier.prefix_cache.lock_handle(inserted.handle)
                pending.source_handle = inserted.handle
                self.host_tier.prefix_cache.lock_handle(pending.host_match, unlock=True)
                pending.host_match = None
                assert self.storage_tier is not None and pending.storage_handle is not None
                self.storage_tier.prefix_cache.lock_handle(pending.storage_handle, unlock=True)
                pending.storage_handle = None
                pending.phase = "h2d"
                host_indices = inserted.handle.get_matched_indices()[
                    pending.cuda_len : pending.target_len
                ]
                pending.ticket = self.transfer_manager.host_to_device(
                    host_indices, pending.device_indices
                )
                if not blocking and not pending.ticket.done:
                    return None

            self._record_transfer(pending.ticket)
            assert self.host_tier is not None and pending.source_handle is not None
            self.host_tier.prefix_cache.lock_handle(pending.source_handle, unlock=True)
            pending.source_handle = None
            handle = self._publish_materialized_l1(pending)
            restored_pages = len(pending.device_indices) // self.page_size
            if pending.restore_tier == CacheTier.HOST:
                self.metrics.host_promotions += restored_pages
            else:
                self.metrics.storage_promotions += restored_pages
            pending.completed_handle = handle
            pending.finished = True
            if pending in self._pending_materializations:
                self._pending_materializations.remove(pending)
            self._record_materialization_latency(pending)
            self._record_match_metrics(
                cuda_len=pending.cuda_len,
                host_len=pending.host_len,
                actual_len=handle.cached_len,
                input_len=pending.input_len,
            )
            return handle
        except Exception as exc:
            return self._fail_materialization(pending, exc)

    def mark_materializations_overlapped(self) -> None:
        now = time.perf_counter()
        for pending in self._pending_materializations:
            pending.overlap_confirmed = True
            if pending.overlap_started_at is None:
                pending.overlap_started_at = now

    def cancel_materialization(self, pending: PendingMaterialization) -> None:
        handle = self.progress_materialization(pending, blocking=True)
        if handle is not None:
            self.unlock(handle)

    def _publish_materialized_l1(self, pending: PendingMaterialization) -> BaseCacheHandle:
        cuda_indices = pending.cuda_handle.get_matched_indices()
        proposed_indices = torch.cat([cuda_indices, pending.device_indices])
        inserted = self.prefix_cache.insert_prefix(
            pending.input_ids[: pending.target_len], proposed_indices
        )
        duplicate_len = inserted.cached_len - pending.cuda_len
        pending.device_pages_private = False
        if duplicate_len > 0:
            self._free(pending.device_indices[:duplicate_len])
        handle = inserted.handle
        self.lock(handle)
        pending.published_handle = handle
        self.unlock(pending.cuda_handle)
        pending.cuda_handle_locked = False
        self._write_matched_page_table(pending.table_idx, handle)
        return handle

    def _fallback_materialization_start(
        self,
        req: PendingReq,
        cuda_handle: BaseCacheHandle,
        cuda_len: int,
        host_len: int,
        table_idx: int,
        started_at: float,
        exc: Exception,
    ) -> BaseCacheHandle:
        self.metrics.restore_fallbacks += 1
        self.metrics.restore_e2e_seconds += time.perf_counter() - started_at
        logger.warning_rank0("HiCache async restore failed; recomputing from L1: %s", exc)
        self._write_matched_page_table(table_idx, cuda_handle)
        self._record_match_metrics(
            cuda_len=cuda_len,
            host_len=host_len,
            actual_len=cuda_len,
            input_len=req.input_len,
        )
        return cuda_handle

    def _fail_materialization(
        self, pending: PendingMaterialization, exc: Exception
    ) -> BaseCacheHandle:
        assert self.host_tier is not None
        if pending.host_new_indices is not None:
            self.host_tier.free(pending.host_new_indices)
            pending.host_new_indices = None
        if pending.host_match is not None:
            self.host_tier.prefix_cache.lock_handle(pending.host_match, unlock=True)
            pending.host_match = None
        if pending.storage_handle is not None:
            assert self.storage_tier is not None
            self.storage_tier.prefix_cache.lock_handle(pending.storage_handle, unlock=True)
            pending.storage_handle = None
        if pending.source_handle is not None:
            self.host_tier.prefix_cache.lock_handle(pending.source_handle, unlock=True)
            pending.source_handle = None
        if pending.published_handle is not None:
            self.unlock(pending.published_handle)
            pending.published_handle = None
        if not pending.cuda_handle_locked:
            self.lock(pending.cuda_handle)
            pending.cuda_handle_locked = True
        if pending.device_pages_private:
            self._free(self._page_to_token(pending.allocated_pages))
        self.metrics.restore_fallbacks += 1
        self._record_materialization_latency(pending)
        logger.warning_rank0("HiCache async restore failed; recomputing from L1: %s", exc)
        self._write_matched_page_table(pending.table_idx, pending.cuda_handle)
        pending.completed_handle = pending.cuda_handle
        pending.finished = True
        if pending in self._pending_materializations:
            self._pending_materializations.remove(pending)
        self._record_match_metrics(
            cuda_len=pending.cuda_len,
            host_len=pending.host_len,
            actual_len=pending.cuda_len,
            input_len=pending.input_len,
        )
        return pending.cuda_handle

    def _record_materialization_latency(self, pending: PendingMaterialization) -> None:
        finished_at = time.perf_counter()
        self.metrics.restore_e2e_seconds += finished_at - pending.started_at
        if pending.overlap_confirmed and pending.overlap_started_at is not None:
            self.metrics.restore_overlapped_seconds += max(
                0.0, finished_at - pending.overlap_started_at
            )
            self.metrics.overlapped_restore_requests += 1

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        # valid: [0, req.cached_len); request-private: [old_handle.cached_len, req.cached_len)
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)

        self.poll_transfers(blocking=False)
        backup_host = self.host_tier is not None and self.cost_model.should_backup(
            tier=CacheTier.HOST,
            tokens=new_handle.cached_len,
            bytes_per_token=self.bytes_per_token,
        )
        backup_storage = self.storage_tier is not None and self.cost_model.should_backup(
            tier=CacheTier.STORAGE,
            tokens=new_handle.cached_len,
            bytes_per_token=self.bytes_per_token,
        )
        if self.host_tier is not None and not backup_host:
            self.metrics.policy_backup_skips += new_handle.cached_len
        if self.storage_tier is not None and not backup_storage:
            self.metrics.policy_backup_skips += new_handle.cached_len

        if self.hicache_enabled and new_handle.cached_len > 0 and (backup_host or backup_storage):
            self.lock(new_handle)  # transfer ownership; released after L2 publication
            try:
                self._schedule_host_backup(
                    insert_ids[: new_handle.cached_len].clone(),
                    new_handle,
                    backup_host=backup_host,
                    backup_storage=backup_storage,
                )
            except Exception as exc:
                self.unlock(new_handle)
                self.metrics.backup_failures += 1
                logger.warning_rank0("HiCache backup failed; keeping the L1 copy only: %s", exc)

        # Unlock only after insertion and lower-tier backup have stopped reading request pages.
        self.unlock(old_handle)
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:
            self._free(page_indices[new_handle.cached_len :])
        else:
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        for pending in list(self._pending_materializations):
            self.progress_materialization(pending, blocking=True)
        self.poll_transfers(blocking=True)
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1 and not torch.all(self.free_slots % self.page_size == 0):
            raise RuntimeError("Device free list contains unaligned pages")
        if len(self.free_slots) != len(set(self.free_slots.detach().cpu().tolist())):
            raise RuntimeError("Device free list contains duplicate pages")

        if self.host_tier is not None:
            self.host_tier.check_integrity()
            self.metrics.host_evictions = self.host_tier.evicted_tokens // self.page_size
        if self.storage_tier is not None:
            self.storage_tier.check_integrity()
            self.metrics.storage_evictions = self.storage_tier.evicted_tokens // self.page_size

    def shutdown(self) -> None:
        for pending in list(self._pending_materializations):
            self.cancel_materialization(pending)
        self.poll_transfers(blocking=True)
        if self.transfer_manager is not None:
            self.transfer_manager.shutdown()
        if self.storage_pool is not None:
            self.storage_pool.close()

    @contextmanager
    def lazy_free_region(self) -> Iterator[None]:
        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free  # type: ignore[method-assign]
            yield
        finally:
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _restore_from_host(
        self,
        host_handle: BaseCacheHandle,
        device_indices: torch.Tensor,
        start: int,
        end: int,
    ) -> None:
        assert self.host_tier is not None and self.transfer_manager is not None
        self.host_tier.prefix_cache.lock_handle(host_handle)
        try:
            host_indices = host_handle.get_matched_indices()[start:end]
            ticket = self.transfer_manager.host_to_device(host_indices, device_indices)
            self._record_transfer(ticket)
            self.metrics.host_promotions += len(device_indices) // self.page_size
        finally:
            self.host_tier.prefix_cache.lock_handle(host_handle, unlock=True)

    def _restore_from_storage(
        self,
        input_ids: torch.Tensor,
        storage_handle: BaseCacheHandle,
        device_indices: torch.Tensor,
        start: int,
        end: int,
    ) -> None:
        assert self.storage_tier is not None and self.transfer_manager is not None
        self.storage_tier.prefix_cache.lock_handle(storage_handle)
        try:
            promoted_host = None
            if self.hicache_promote_storage and self.host_tier is not None:
                promoted_host = self._promote_storage_to_host(input_ids[:end], storage_handle, end)
            if promoted_host is not None and promoted_host.cached_len >= end:
                assert self.host_tier is not None
                self.host_tier.prefix_cache.lock_handle(promoted_host)
                try:
                    host_indices = promoted_host.get_matched_indices()[start:end]
                    ticket = self.transfer_manager.host_to_device(host_indices, device_indices)
                    self._record_transfer(ticket)
                finally:
                    self.host_tier.prefix_cache.lock_handle(promoted_host, unlock=True)
            else:
                storage_indices = storage_handle.get_matched_indices()[start:end]
                tickets = self.transfer_manager.storage_to_device(storage_indices, device_indices)
                for ticket in tickets:
                    self._record_transfer(ticket)
            self.metrics.storage_promotions += len(device_indices) // self.page_size
        finally:
            self.storage_tier.prefix_cache.lock_handle(storage_handle, unlock=True)

    def _promote_storage_to_host(
        self,
        input_ids: torch.Tensor,
        storage_handle: BaseCacheHandle,
        target_len: int,
    ) -> BaseCacheHandle | None:
        assert self.host_tier is not None and self.transfer_manager is not None
        if target_len > self.host_tier.capacity_tokens:
            return None
        match = self.host_tier.prefix_cache.match_prefix(input_ids[:target_len]).cuda_handle
        if match.cached_len >= target_len:
            return match

        self.host_tier.prefix_cache.lock_handle(match)
        new_indices: torch.Tensor | None = None
        published = False
        try:
            new_indices = self.host_tier.allocate_tokens(target_len - match.cached_len)
            if new_indices is None:
                return None
            storage_indices = storage_handle.get_matched_indices()[match.cached_len : target_len]
            ticket = self.transfer_manager.storage_to_host(storage_indices, new_indices)
            self._record_transfer(ticket)
            combined = torch.cat([match.get_matched_indices(), new_indices])
            inserted = self.host_tier.prefix_cache.insert_prefix(input_ids[:target_len], combined)
            published = True
            duplicate_len = inserted.cached_len - match.cached_len
            if duplicate_len > 0:
                self.host_tier.free(new_indices[:duplicate_len])
            return inserted.handle
        except Exception:
            if new_indices is not None and not published:
                self.host_tier.free(new_indices)
            raise
        finally:
            self.host_tier.prefix_cache.lock_handle(match, unlock=True)

    def _schedule_host_backup(
        self,
        input_ids: torch.Tensor,
        device_handle: BaseCacheHandle,
        *,
        backup_host: bool,
        backup_storage: bool,
    ) -> None:
        """Launch D2H write-through without blocking the next scheduling iteration."""
        assert self.host_tier is not None and self.transfer_manager is not None
        if not backup_host:
            if backup_storage:
                self._schedule_storage_backup(input_ids, device_handle, None)
            self.unlock(device_handle)
            return
        target_len = min(len(input_ids), self.host_tier.capacity_tokens)
        target_len -= target_len % self.page_size
        if target_len == 0:
            if backup_storage:
                self._schedule_storage_backup(input_ids, device_handle, None)
            self.unlock(device_handle)
            return

        match = self.host_tier.prefix_cache.match_prefix(input_ids[:target_len]).cuda_handle
        if match.cached_len >= target_len:
            if backup_storage:
                self._schedule_storage_backup(input_ids, device_handle, match)
            self.unlock(device_handle)
            return

        self.host_tier.prefix_cache.lock_handle(match)
        new_indices = self.host_tier.allocate_tokens(target_len - match.cached_len)
        if new_indices is None:
            self.host_tier.prefix_cache.lock_handle(match, unlock=True)
            if backup_storage:
                self._schedule_storage_backup(input_ids, device_handle, match)
            self.unlock(device_handle)
            return

        try:
            source = device_handle.get_matched_indices()[match.cached_len : target_len]
            ticket = self.transfer_manager.device_to_host(source, new_indices)
            self._pending_host_backups.append(
                _PendingHostBackup(
                    input_ids=input_ids,
                    device_handle=device_handle,
                    existing_host_handle=match,
                    new_host_indices=new_indices,
                    target_len=target_len,
                    ticket=ticket,
                    backup_storage=backup_storage,
                )
            )
        except Exception:
            self.host_tier.free(new_indices)
            self.host_tier.prefix_cache.lock_handle(match, unlock=True)
            raise

    def _finalize_host_backup(self, pending: _PendingHostBackup) -> None:
        assert self.host_tier is not None
        published = False
        try:
            self._record_transfer(pending.ticket)
            combined = torch.cat(
                [
                    pending.existing_host_handle.get_matched_indices(),
                    pending.new_host_indices,
                ]
            )
            inserted = self.host_tier.prefix_cache.insert_prefix(
                pending.input_ids[: pending.target_len], combined
            )
            published = True
            duplicate_len = inserted.cached_len - pending.existing_host_handle.cached_len
            if duplicate_len > 0:
                self.host_tier.free(pending.new_host_indices[:duplicate_len])
            if pending.backup_storage:
                self._schedule_storage_backup(
                    pending.input_ids, pending.device_handle, inserted.handle
                )
        except Exception as exc:
            self.metrics.backup_failures += 1
            logger.warning_rank0("Asynchronous L2 backup failed: %s", exc)
            if not published:
                self.host_tier.free(pending.new_host_indices)
        finally:
            self.host_tier.prefix_cache.lock_handle(pending.existing_host_handle, unlock=True)
            self.unlock(pending.device_handle)

    def _schedule_storage_backup(
        self,
        input_ids: torch.Tensor,
        device_handle: BaseCacheHandle,
        host_handle: BaseCacheHandle | None,
    ) -> None:
        if self.storage_tier is None or self.transfer_manager is None:
            return
        target_len = min(len(input_ids), self.storage_tier.capacity_tokens)
        target_len -= target_len % self.page_size
        if target_len == 0:
            return

        match = self.storage_tier.prefix_cache.match_prefix(input_ids[:target_len]).cuda_handle
        if match.cached_len >= target_len:
            return
        self.storage_tier.prefix_cache.lock_handle(match)
        new_indices = self.storage_tier.allocate_tokens(target_len - match.cached_len)
        if new_indices is None:
            self.storage_tier.prefix_cache.lock_handle(match, unlock=True)
            return

        source_host_locked = False
        published = False
        try:
            if host_handle is not None and host_handle.cached_len >= target_len:
                assert self.host_tier is not None
                self.host_tier.prefix_cache.lock_handle(host_handle)
                source_host_locked = True
                source = host_handle.get_matched_indices()[match.cached_len : target_len]
                ticket = self.transfer_manager.host_to_storage(source, new_indices)
                self._pending_storage_backups.append(
                    _PendingStorageBackup(
                        input_ids=input_ids,
                        source_host_handle=host_handle,
                        existing_storage_handle=match,
                        new_storage_indices=new_indices,
                        target_len=target_len,
                        ticket=ticket,
                    )
                )
                return

            source = device_handle.get_matched_indices()[match.cached_len : target_len]
            tickets = self.transfer_manager.device_to_storage(source, new_indices)
            for ticket in tickets:
                self._record_transfer(ticket)
            combined = torch.cat([match.get_matched_indices(), new_indices])
            inserted = self.storage_tier.prefix_cache.insert_prefix(
                input_ids[:target_len], combined
            )
            published = True
            duplicate_len = inserted.cached_len - match.cached_len
            if duplicate_len > 0:
                self.storage_tier.free(new_indices[:duplicate_len])
        except Exception:
            if not published:
                self.storage_tier.free(new_indices)
            if source_host_locked:
                assert self.host_tier is not None and host_handle is not None
                self.host_tier.prefix_cache.lock_handle(host_handle, unlock=True)
                source_host_locked = False
            raise
        finally:
            if not source_host_locked:
                self.storage_tier.prefix_cache.lock_handle(match, unlock=True)

    def _finalize_storage_backup(self, pending: _PendingStorageBackup) -> None:
        assert self.host_tier is not None and self.storage_tier is not None
        published = False
        try:
            self._record_transfer(pending.ticket)
            combined = torch.cat(
                [
                    pending.existing_storage_handle.get_matched_indices(),
                    pending.new_storage_indices,
                ]
            )
            inserted = self.storage_tier.prefix_cache.insert_prefix(
                pending.input_ids[: pending.target_len], combined
            )
            published = True
            duplicate_len = inserted.cached_len - pending.existing_storage_handle.cached_len
            if duplicate_len > 0:
                self.storage_tier.free(pending.new_storage_indices[:duplicate_len])
        except Exception as exc:
            self.metrics.backup_failures += 1
            logger.warning_rank0("Asynchronous L3 backup failed: %s", exc)
            if not published:
                self.storage_tier.free(pending.new_storage_indices)
        finally:
            self.storage_tier.prefix_cache.lock_handle(pending.existing_storage_handle, unlock=True)
            self.host_tier.prefix_cache.lock_handle(pending.source_host_handle, unlock=True)

    def _match_prefixes(self, input_ids: torch.Tensor) -> MatchResult:
        cuda_handle = self.prefix_cache.match_prefix(input_ids).cuda_handle
        host_handle = None
        storage_handle = None
        if self.host_tier is not None:
            host_handle = self.host_tier.prefix_cache.match_prefix(input_ids).cuda_handle
        if self.storage_tier is not None:
            storage_handle = self.storage_tier.prefix_cache.match_prefix(input_ids).cuda_handle
        return MatchResult(cuda_handle, host_handle, storage_handle)

    def _write_matched_page_table(self, table_idx: int, handle: BaseCacheHandle) -> None:
        if handle.cached_len > 0:
            self.page_table[table_idx, : handle.cached_len].copy_(handle.get_matched_indices())

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        needed_tokens = needed_pages * self.page_size
        if needed_tokens > self.available_size and self.has_pending_transfers:
            self.poll_transfers(blocking=True)
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            self.metrics.device_evictions += len(evicted) // self.page_size
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()

    @staticmethod
    def _capacity_pages(
        *,
        explicit_gb: float | None,
        ratio: float,
        device_pages: int,
        bytes_per_page: int,
        name: str,
        allow_zero: bool = False,
    ) -> int:
        if explicit_gb is not None:
            if explicit_gb < 0 or (explicit_gb == 0 and not allow_zero):
                raise ValueError(f"HiCache {name} size must be positive")
            pages = int(explicit_gb * (1 << 30)) // bytes_per_page
        else:
            if ratio < 0 or (ratio == 0 and not allow_zero):
                raise ValueError(f"HiCache {name} ratio must be positive")
            pages = int(device_pages * ratio)
        if not allow_zero and pages < 1:
            raise ValueError(f"HiCache {name} capacity is smaller than one KV page")
        return pages

    @staticmethod
    def _resolve_storage_path(path: str | None) -> str | None:
        if path is None:
            return None
        tp_info = try_get_tp_info()
        if tp_info is None or tp_info.size == 1:
            return path
        resolved = Path(path).expanduser()
        return str(resolved.with_name(f"{resolved.stem}.rank{tp_info.rank}{resolved.suffix}"))


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated


__all__ = [
    "CacheManager",
    "HiCacheCostModel",
    "HiCacheMetrics",
    "PendingMaterialization",
]
