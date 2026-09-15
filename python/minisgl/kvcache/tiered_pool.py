from __future__ import annotations

import os
import tempfile
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Callable, List

import torch

from .mha_pool import HostMHAKVCache, MHAKVCache


class CacheTier(str, Enum):
    GPU = "gpu"
    HOST = "host"
    STORAGE = "storage"


class TransferDirection(str, Enum):
    D2H = "d2h"
    H2D = "h2d"
    H2S = "h2s"
    S2H = "s2h"


class TransferState(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TransferTicket:
    """Own the resources and completion primitive for one cache transfer."""

    direction: TransferDirection
    source: CacheTier
    destination: CacheTier
    pages: int
    num_bytes: int
    start_event: torch.cuda.Event | None = None
    event: torch.cuda.Event | None = None
    future: Future[int | None] | None = None
    on_complete: Callable[[], None] | None = None
    on_finalize: Callable[[], None] | None = None
    created_at: float = field(default_factory=time.perf_counter)
    submitted_at: float | None = None
    started_at: float | None = None
    work_completed_at: float | None = None
    completed_at: float | None = None
    state: TransferState = TransferState.PENDING
    error: str | None = None
    io_operations: int = 0
    device_seconds: float = 0.0
    completion_seconds: float = 0.0
    workspace_setup_seconds: float = 0.0
    _finalized: bool = False

    @property
    def done(self) -> bool:
        if self.state != TransferState.PENDING:
            return True
        event_done = self.event is None or self.event.query()
        future_done = self.future is None or self.future.done()
        return event_done and future_done

    def wait(self) -> None:
        if self.state == TransferState.COMPLETED:
            return
        try:
            if self.future is not None:
                operations = self.future.result()
                if operations is not None:
                    self.io_operations = operations
            if self.event is not None:
                self.event.synchronize()
                if self.start_event is not None:
                    self.device_seconds = self.start_event.elapsed_time(self.event) / 1000
            if self.on_complete is not None:
                completion_started_at = time.perf_counter()
                self.on_complete()
                self.completion_seconds = time.perf_counter() - completion_started_at
        except Exception as exc:
            self.state = TransferState.FAILED
            self.error = str(exc)
            self.completed_at = time.perf_counter()
            raise
        else:
            self.state = TransferState.COMPLETED
            self.completed_at = time.perf_counter()
        finally:
            if not self._finalized:
                self._finalized = True
                if self.on_finalize is not None:
                    self.on_finalize()

    @property
    def latency_seconds(self) -> float:
        end = self.completed_at if self.completed_at is not None else time.perf_counter()
        return end - self.created_at

    @property
    def enqueue_seconds(self) -> float:
        end = self.submitted_at if self.submitted_at is not None else time.perf_counter()
        return max(0.0, end - self.created_at)

    @property
    def queue_wait_seconds(self) -> float:
        if self.started_at is None or self.submitted_at is None:
            return 0.0
        return max(0.0, self.started_at - self.submitted_at)

    @property
    def service_seconds(self) -> float:
        if self.event is not None:
            return self.device_seconds + self.completion_seconds
        if self.started_at is None:
            return 0.0
        end = self.work_completed_at or self.completed_at or time.perf_counter()
        return max(0.0, end - self.started_at) + self.completion_seconds

    @property
    def active_seconds(self) -> float:
        return self.enqueue_seconds + self.queue_wait_seconds + self.service_seconds

    @property
    def recurring_seconds(self) -> float:
        return max(0.0, self.active_seconds - self.workspace_setup_seconds)


def page_ids_from_token_indices(indices: torch.Tensor, page_size: int) -> List[int]:
    """Validate expanded token indices and return their physical page ids."""
    if len(indices) == 0:
        return []
    if len(indices) % page_size != 0:
        raise ValueError(f"Transfer length {len(indices)} is not page aligned ({page_size=})")

    values = indices.detach().to(device="cpu", dtype=torch.int64).view(-1, page_size)
    starts = values[:, 0]
    offsets = torch.arange(page_size, dtype=torch.int64).view(1, -1)
    if not torch.equal(values, starts.view(-1, 1) + offsets):
        raise ValueError("Transfer indices must contain complete, consecutive pages")
    if not torch.all(starts % page_size == 0):
        raise ValueError("Transfer page starts must be page aligned")
    return [int(page) for page in (starts // page_size).tolist()]


class StorageMHAKVCache:
    """Fixed-slot, file-backed KV page storage used as the L3 cache tier.

    Prefix metadata stays in memory. The data file is process-scoped and uses fixed-size
    offsets, which makes allocation and asynchronous pread/pwrite deterministic.
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
        path: str | None = None,
    ) -> None:
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.local_kv_heads = local_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.page_shape = (2, num_layers, page_size, local_kv_heads, head_dim)
        self.page_elements = 2 * num_layers * page_size * local_kv_heads * head_dim
        self.bytes_per_page = self.page_elements * dtype.itemsize

        self._owned_path = path is None
        if path is None:
            fd, resolved_path = tempfile.mkstemp(prefix="minisgl-hicache-", suffix=".kv")
            self.path = Path(resolved_path)
            self._fd = fd
        else:
            self.path = Path(path).expanduser().resolve()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        os.fchmod(self._fd, 0o600)
        os.ftruncate(self._fd, self.num_pages * self.bytes_per_page)
        self._closed = False

    @classmethod
    def from_device_pool(
        cls, pool: MHAKVCache, num_pages: int, *, path: str | None = None
    ) -> StorageMHAKVCache:
        return cls(
            num_layers=pool.num_layers,
            num_pages=num_pages,
            page_size=pool.page_size,
            local_kv_heads=pool.local_kv_heads,
            head_dim=pool.head_dim,
            dtype=pool.dtype,
            path=path,
        )

    def write_pages_from_host(
        self,
        host_pool: HostMHAKVCache,
        host_indices: torch.Tensor,
        storage_indices: torch.Tensor,
    ) -> int:
        return self.write_pages(
            host_pool,
            page_ids_from_token_indices(host_indices, self.page_size),
            page_ids_from_token_indices(storage_indices, self.page_size),
        )

    def read_pages_to_host(
        self,
        storage_indices: torch.Tensor,
        host_pool: HostMHAKVCache,
        host_indices: torch.Tensor,
    ) -> int:
        return self.read_pages(
            page_ids_from_token_indices(storage_indices, self.page_size),
            host_pool,
            page_ids_from_token_indices(host_indices, self.page_size),
        )

    def write_pages(
        self,
        host_pool: HostMHAKVCache,
        host_pages: List[int],
        storage_pages: List[int],
    ) -> int:
        """Store pre-parsed complete pages, coalescing adjacent runs into extents."""
        if len(host_pages) != len(storage_pages):
            raise ValueError("Source and destination storage page counts differ")
        self._validate_page_range(storage_pages)
        self._validate_page_range(host_pages, capacity=host_pool.num_pages, tier="host")
        if not host_pages:
            return 0

        extents = self._coalesce_mappings(host_pages, storage_pages)
        for first, last in extents:
            pages = host_pool.page_buffer[host_pages[first] : host_pages[first] + last - first]
            data = memoryview(pages.view(torch.uint8).numpy().reshape(-1))
            self._pwrite_all(data, storage_pages[first] * self.bytes_per_page)
        return len(extents)

    def read_pages(
        self,
        storage_pages: List[int],
        host_pool: HostMHAKVCache,
        host_pages: List[int],
    ) -> int:
        """Restore pre-parsed complete pages, coalescing adjacent runs into extents."""
        if len(host_pages) != len(storage_pages):
            raise ValueError("Source and destination storage page counts differ")
        self._validate_page_range(storage_pages)
        self._validate_page_range(host_pages, capacity=host_pool.num_pages, tier="host")
        if not storage_pages:
            return 0

        extents = self._coalesce_mappings(storage_pages, host_pages)
        for first, last in extents:
            pages = host_pool.page_buffer[host_pages[first] : host_pages[first] + last - first]
            destination = memoryview(pages.view(torch.uint8).numpy().reshape(-1))
            self._pread_into(
                destination,
                storage_pages[first] * self.bytes_per_page,
            )
        return len(extents)

    @staticmethod
    def _coalesce_mappings(
        source_pages: List[int], destination_pages: List[int]
    ) -> List[tuple[int, int]]:
        """Return runs that are physically adjacent in both pools."""
        if not source_pages:
            return []
        extents: List[tuple[int, int]] = []
        first = 0
        for index in range(1, len(source_pages)):
            if (
                source_pages[index] != source_pages[index - 1] + 1
                or destination_pages[index] != destination_pages[index - 1] + 1
            ):
                extents.append((first, index))
                first = index
        extents.append((first, len(source_pages)))
        return extents

    def _pwrite_all(self, data: bytes | memoryview, offset: int) -> None:
        view = memoryview(data)
        written = 0
        while written < len(view):
            count = os.pwrite(self._fd, view[written:], offset + written)
            if count <= 0:
                raise OSError("Short write while storing HiCache page")
            written += count

    def _pread_into(self, destination: memoryview, offset: int) -> None:
        read = 0
        while read < len(destination):
            count = os.preadv(self._fd, [destination[read:]], offset + read)
            if count <= 0:
                raise OSError("Short read while restoring HiCache page")
            read += count

    def _validate_page_range(
        self,
        pages: List[int],
        *,
        capacity: int | None = None,
        tier: str = "storage",
    ) -> None:
        capacity = self.num_pages if capacity is None else capacity
        if any(page < 0 or page >= capacity for page in pages):
            raise IndexError(f"{tier} transfer page is outside capacity {capacity}")

    def close(self) -> None:
        if self._closed:
            return
        os.close(self._fd)
        self._closed = True
        if self._owned_path:
            self.path.unlink(missing_ok=True)


@dataclass
class _PackedPageWorkspace:
    """Reusable page-packed buffers for one fused gather/DMA/scatter transfer.

    ``host`` is only needed when the host side is physically fragmented; the common
    contiguous case reads or writes the pinned L2 pool directly. Pinned allocations
    are by far the most expensive part, so they are kept lazy.
    """

    pages: int
    device: torch.Tensor
    busy: bool = False
    host: torch.Tensor | None = None


class CacheTransferManager:
    """Move complete KV pages between GPU, pinned RAM, and local storage."""

    def __init__(
        self,
        device_pool: MHAKVCache,
        host_pool: HostMHAKVCache,
        storage_pool: StorageMHAKVCache | None,
        *,
        io_workers: int = 2,
        staging_pages: int = 8,
        copy_backend: str = "auto",
    ) -> None:
        if device_pool.page_size != host_pool.page_size:
            raise ValueError("Device and host cache page sizes must match")
        self.device_pool = device_pool
        self.host_pool = host_pool
        self.storage_pool = storage_pool
        if copy_backend not in {"auto", "triton", "torch"}:
            raise ValueError("HiCache copy backend must be auto, triton, or torch")
        self.requested_copy_backend = copy_backend
        self.copy_backend = copy_backend
        initial_backend = "triton" if copy_backend == "auto" else copy_backend
        self.gather_backend = initial_backend
        self.scatter_backend = initial_backend
        self._copy_backends_tuned = copy_backend != "auto"
        self.autotune_timings_ms: dict[str, float] = {}
        self.autotune_pages = 0
        self.stream = torch.cuda.Stream(device=device_pool.device)
        self.executor = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="hicache-io")
        self.staging_pool = HostMHAKVCache.from_device_pool(
            device_pool, max(1, staging_pages), pin_memory=True
        )
        self._workspaces: List[_PackedPageWorkspace] = []
        self._workspace_lock = Lock()

    @property
    def page_size(self) -> int:
        return self.device_pool.page_size

    def device_to_host(
        self, device_indices: torch.Tensor, host_indices: torch.Tensor
    ) -> TransferTicket:
        return self._fused_device_to_host(self.host_pool, device_indices, host_indices)

    def host_to_device(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> TransferTicket:
        return self._fused_host_to_device(self.host_pool, host_indices, device_indices)

    def host_to_storage(
        self, host_indices: torch.Tensor, storage_indices: torch.Tensor
    ) -> TransferTicket:
        storage = self._require_storage()
        host_pages = page_ids_from_token_indices(host_indices, self.page_size)
        storage_pages = page_ids_from_token_indices(storage_indices, self.page_size)
        return self._io_ticket(
            TransferDirection.H2S,
            len(host_pages),
            lambda: storage.write_pages(self.host_pool, host_pages, storage_pages),
        )

    def storage_to_host(
        self, storage_indices: torch.Tensor, host_indices: torch.Tensor
    ) -> TransferTicket:
        storage = self._require_storage()
        storage_pages = page_ids_from_token_indices(storage_indices, self.page_size)
        host_pages = page_ids_from_token_indices(host_indices, self.page_size)
        return self._io_ticket(
            TransferDirection.S2H,
            len(storage_pages),
            lambda: storage.read_pages(storage_pages, self.host_pool, host_pages),
        )

    def device_to_storage(
        self, device_indices: torch.Tensor, storage_indices: torch.Tensor
    ) -> List[TransferTicket]:
        tickets: List[TransferTicket] = []
        for device_chunk, storage_chunk, staging_indices in self._staging_chunks(
            device_indices, storage_indices
        ):
            d2h = self._device_to_staging(device_chunk, staging_indices)
            d2h.wait()
            tickets.append(d2h)
            h2s = self._staging_to_storage(staging_indices, storage_chunk)
            h2s.wait()
            tickets.append(h2s)
        return tickets

    def storage_to_device(
        self, storage_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> List[TransferTicket]:
        tickets: List[TransferTicket] = []
        for storage_chunk, device_chunk, staging_indices in self._staging_chunks(
            storage_indices, device_indices
        ):
            s2h = self._storage_to_staging(storage_chunk, staging_indices)
            s2h.wait()
            tickets.append(s2h)
            h2d = self._staging_to_device(staging_indices, device_chunk)
            h2d.wait()
            tickets.append(h2d)
        return tickets

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.stream.synchronize()
        self._workspaces.clear()

    def warmup(self) -> None:
        """Tune page packing, then validate DMA using the unmanaged dummy page."""
        if not self._copy_backends_tuned:
            self._autotune_copy_backends()
        device_start = (self.device_pool.num_pages - 1) * self.page_size
        device_indices = torch.arange(
            device_start,
            device_start + self.page_size,
            dtype=torch.int32,
            device=self.device_pool.device,
        )
        host_indices = torch.arange(self.page_size, dtype=torch.int32)
        self.device_to_host(device_indices, host_indices).wait()
        self.host_to_device(host_indices, device_indices).wait()

    def _device_to_staging(
        self, device_indices: torch.Tensor, staging_indices: torch.Tensor
    ) -> TransferTicket:
        return self._fused_device_to_host(self.staging_pool, device_indices, staging_indices)

    def _staging_to_device(
        self, staging_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> TransferTicket:
        return self._fused_host_to_device(self.staging_pool, staging_indices, device_indices)

    def _staging_to_storage(
        self, staging_indices: torch.Tensor, storage_indices: torch.Tensor
    ) -> TransferTicket:
        storage = self._require_storage()
        staging_pages = page_ids_from_token_indices(staging_indices, self.page_size)
        storage_pages = page_ids_from_token_indices(storage_indices, self.page_size)
        return self._io_ticket(
            TransferDirection.H2S,
            len(staging_pages),
            lambda: storage.write_pages(self.staging_pool, staging_pages, storage_pages),
        )

    def _storage_to_staging(
        self, storage_indices: torch.Tensor, staging_indices: torch.Tensor
    ) -> TransferTicket:
        storage = self._require_storage()
        storage_pages = page_ids_from_token_indices(storage_indices, self.page_size)
        staging_pages = page_ids_from_token_indices(staging_indices, self.page_size)
        return self._io_ticket(
            TransferDirection.S2H,
            len(storage_pages),
            lambda: storage.read_pages(storage_pages, self.staging_pool, staging_pages),
        )

    def _staging_chunks(
        self, source_indices: torch.Tensor, destination_indices: torch.Tensor
    ) -> List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        source_pages = page_ids_from_token_indices(source_indices, self.page_size)
        destination_pages = page_ids_from_token_indices(destination_indices, self.page_size)
        self._check_page_counts(source_pages, destination_pages)
        chunks: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        chunk_tokens = self.staging_pool.num_pages * self.page_size
        for start in range(0, len(source_indices), chunk_tokens):
            end = min(start + chunk_tokens, len(source_indices))
            length = end - start
            staging = torch.arange(length, dtype=torch.int32)
            chunks.append((source_indices[start:end], destination_indices[start:end], staging))
        return chunks

    def _fused_device_to_host(
        self,
        host_pool: HostMHAKVCache,
        device_indices: torch.Tensor,
        host_indices: torch.Tensor,
    ) -> TransferTicket:
        """Gather all selected GPU pages, perform one DMA, then scatter once on CPU."""
        created_at = time.perf_counter()
        device_pages = page_ids_from_token_indices(device_indices, self.page_size)
        host_pages = page_ids_from_token_indices(host_indices, self.page_size)
        self._check_page_counts(device_pages, host_pages)
        self._validate_pool_pages(device_pages, self.device_pool, "device")
        self._validate_pool_pages(host_pages, host_pool, "host")
        if not device_pages:
            return self._completed_cuda_ticket(TransferDirection.D2H, created_at)

        workspace_started_at = time.perf_counter()
        transfer_pages = len(device_pages)
        host_is_consecutive = self._are_consecutive(host_pages)
        workspace, workspace_created = self._acquire_workspace(
            transfer_pages, needs_host=not host_is_consecutive
        )
        workspace_setup_seconds = (
            time.perf_counter() - workspace_started_at if workspace_created else 0.0
        )
        packed_device = workspace.device[:transfer_pages]
        device_page_ids = torch.tensor(
            device_pages, dtype=torch.int64, device=self.device_pool.device
        )
        try:
            with torch.cuda.stream(self.stream):
                self.stream.wait_stream(torch.cuda.current_stream(self.device_pool.device))
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record(self.stream)
                self._gather_device_pages(device_page_ids, packed_device)
                if host_is_consecutive:
                    host_destination = host_pool.page_buffer[
                        host_pages[0] : host_pages[0] + len(host_pages)
                    ]
                    host_destination.copy_(packed_device, non_blocking=True)
                else:
                    assert workspace.host is not None
                    workspace.host[:transfer_pages].copy_(packed_device, non_blocking=True)
                event = torch.cuda.Event(enable_timing=True)
                event.record(self.stream)
            submitted_at = time.perf_counter()

            publish = None
            if not host_is_consecutive:
                assert workspace.host is not None
                host_buffer = workspace.host[:transfer_pages]
                host_page_ids = torch.tensor(host_pages, dtype=torch.int64)

                def publish() -> None:
                    host_pool.page_buffer.index_copy_(0, host_page_ids, host_buffer)

            return self._cuda_ticket(
                TransferDirection.D2H,
                len(device_pages),
                event,
                start_event=start_event,
                created_at=created_at,
                submitted_at=submitted_at,
                on_complete=publish,
                on_finalize=lambda: self._release_workspace(workspace),
                workspace_setup_seconds=workspace_setup_seconds,
            )
        except Exception:
            self._release_workspace(workspace)
            raise

    def _fused_host_to_device(
        self,
        host_pool: HostMHAKVCache,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
    ) -> TransferTicket:
        """Gather all selected host pages, perform one DMA, then scatter once on GPU."""
        created_at = time.perf_counter()
        host_pages = page_ids_from_token_indices(host_indices, self.page_size)
        device_pages = page_ids_from_token_indices(device_indices, self.page_size)
        self._check_page_counts(host_pages, device_pages)
        self._validate_pool_pages(host_pages, host_pool, "host")
        self._validate_pool_pages(device_pages, self.device_pool, "device")
        if not host_pages:
            return self._completed_cuda_ticket(TransferDirection.H2D, created_at)

        workspace_started_at = time.perf_counter()
        transfer_pages = len(host_pages)
        host_is_consecutive = self._are_consecutive(host_pages)
        workspace, workspace_created = self._acquire_workspace(
            transfer_pages, needs_host=not host_is_consecutive
        )
        workspace_setup_seconds = (
            time.perf_counter() - workspace_started_at if workspace_created else 0.0
        )
        packed_device = workspace.device[:transfer_pages]
        device_page_ids = torch.tensor(
            device_pages, dtype=torch.int64, device=self.device_pool.device
        )
        try:
            if host_is_consecutive:
                host_source = host_pool.page_buffer[host_pages[0] : host_pages[0] + len(host_pages)]
            else:
                assert workspace.host is not None
                host_buffer = workspace.host[:transfer_pages]
                host_page_ids = torch.tensor(host_pages, dtype=torch.int64)
                torch.index_select(host_pool.page_buffer, 0, host_page_ids, out=host_buffer)
                host_source = host_buffer
            with torch.cuda.stream(self.stream):
                # device_page_ids is produced on the caller's current stream.  Triton
                # consumes it directly, so make the transfer stream wait for that
                # initialization before launching the scatter kernel.  PyTorch's
                # index_copy_ happened to hide this race in many small workloads.
                self.stream.wait_stream(torch.cuda.current_stream(self.device_pool.device))
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record(self.stream)
                packed_device.copy_(host_source, non_blocking=True)
                self._scatter_device_pages(packed_device, device_page_ids)
                event = torch.cuda.Event(enable_timing=True)
                event.record(self.stream)
            submitted_at = time.perf_counter()
            return self._cuda_ticket(
                TransferDirection.H2D,
                len(host_pages),
                event,
                start_event=start_event,
                created_at=created_at,
                submitted_at=submitted_at,
                on_finalize=lambda: self._release_workspace(workspace),
                workspace_setup_seconds=workspace_setup_seconds,
            )
        except Exception:
            self._release_workspace(workspace)
            raise

    def _io_ticket(
        self,
        direction: TransferDirection,
        pages: int,
        operation: Callable[[], int],
    ) -> TransferTicket:
        source = CacheTier.HOST if direction == TransferDirection.H2S else CacheTier.STORAGE
        destination = CacheTier.STORAGE if direction == TransferDirection.H2S else CacheTier.HOST
        ticket = TransferTicket(
            direction=direction,
            source=source,
            destination=destination,
            pages=pages,
            num_bytes=pages * self.device_pool.bytes_per_page,
        )

        def run() -> int:
            ticket.started_at = time.perf_counter()
            try:
                return operation()
            finally:
                ticket.work_completed_at = time.perf_counter()

        ticket.submitted_at = time.perf_counter()
        ticket.future = self.executor.submit(run)
        return ticket

    def _cuda_ticket(
        self,
        direction: TransferDirection,
        pages: int,
        event: torch.cuda.Event,
        *,
        start_event: torch.cuda.Event,
        created_at: float,
        submitted_at: float,
        on_complete: Callable[[], None] | None = None,
        on_finalize: Callable[[], None] | None = None,
        workspace_setup_seconds: float = 0.0,
    ) -> TransferTicket:
        source = CacheTier.GPU if direction == TransferDirection.D2H else CacheTier.HOST
        destination = CacheTier.HOST if direction == TransferDirection.D2H else CacheTier.GPU
        return TransferTicket(
            direction=direction,
            source=source,
            destination=destination,
            pages=pages,
            num_bytes=pages * self.device_pool.bytes_per_page,
            start_event=start_event,
            event=event,
            on_complete=on_complete,
            on_finalize=on_finalize,
            created_at=created_at,
            submitted_at=submitted_at,
            started_at=submitted_at,
            workspace_setup_seconds=workspace_setup_seconds,
        )

    def _completed_cuda_ticket(
        self, direction: TransferDirection, created_at: float
    ) -> TransferTicket:
        now = time.perf_counter()
        source = CacheTier.GPU if direction == TransferDirection.D2H else CacheTier.HOST
        destination = CacheTier.HOST if direction == TransferDirection.D2H else CacheTier.GPU
        return TransferTicket(
            direction=direction,
            source=source,
            destination=destination,
            pages=0,
            num_bytes=0,
            created_at=created_at,
            submitted_at=now,
            started_at=now,
            completed_at=now,
            state=TransferState.COMPLETED,
        )

    def _acquire_workspace(
        self, pages: int, *, needs_host: bool
    ) -> tuple[_PackedPageWorkspace, bool]:
        with self._workspace_lock:
            candidates = [
                workspace
                for workspace in self._workspaces
                if workspace.pages >= pages and not workspace.busy
            ]
            if candidates:
                workspace = min(candidates, key=lambda item: item.pages)
                workspace.busy = True
                host_created = needs_host and workspace.host is None
                if host_created:
                    workspace.host = self._empty_host(workspace.pages)
                return workspace, host_created
            workspace = _PackedPageWorkspace(
                pages=pages,
                device=self._empty_device(pages),
                busy=True,
                host=self._empty_host(pages) if needs_host else None,
            )
            self._workspaces.append(workspace)
            return workspace, True

    def _packed_shape(self, pages: int) -> tuple[int, ...]:
        return (
            pages,
            2,
            self.device_pool.num_layers,
            self.page_size,
            self.device_pool.local_kv_heads,
            self.device_pool.head_dim,
        )

    def _empty_device(self, pages: int) -> torch.Tensor:
        return torch.empty(
            self._packed_shape(pages),
            dtype=self.device_pool.dtype,
            device=self.device_pool.device,
        )

    def _empty_host(self, pages: int) -> torch.Tensor:
        return torch.empty(
            self._packed_shape(pages),
            dtype=self.device_pool.dtype,
            device="cpu",
            pin_memory=True,
        )

    def _release_workspace(self, workspace: _PackedPageWorkspace) -> None:
        with self._workspace_lock:
            workspace.busy = False
            idle = [item for item in self._workspaces if not item.busy]
            if len(idle) > 1:
                # Keep the largest idle workspace, preferring one that already owns a
                # pinned page-packed buffer so its allocation is not thrown away.
                keep = max(idle, key=lambda item: (item.pages, item.host is not None))
                self._workspaces = [item for item in self._workspaces if item.busy or item is keep]

    def _autotune_copy_backends(self) -> None:
        """Select gather and scatter independently for the current KV geometry."""
        max_tune_bytes = 64 << 20
        pages_by_bytes = max(1, max_tune_bytes // self.device_pool.bytes_per_page)
        tune_pages = min(256, pages_by_bytes, self.device_pool.num_pages - 1)
        self.autotune_pages = tune_pages
        if tune_pages < 1:
            self.gather_backend = self.scatter_backend = "torch"
            self._refresh_copy_backend_label()
            self._copy_backends_tuned = True
            return

        shape = (
            tune_pages,
            2,
            self.device_pool.num_layers,
            self.page_size,
            self.device_pool.local_kv_heads,
            self.device_pool.head_dim,
        )
        try:
            page_ids = torch.arange(tune_pages, dtype=torch.int64, device=self.device_pool.device)
            packed = torch.empty(
                shape, dtype=self.device_pool.dtype, device=self.device_pool.device
            )
        except Exception as exc:
            self.gather_backend = self.scatter_backend = "triton"
            self._refresh_copy_backend_label()
            self._copy_backends_tuned = True
            warnings.warn(
                f"HiCache autotune workspace unavailable; using fused copies: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        def torch_gather() -> None:
            gathered = torch.index_select(self.device_pool.buffer, 2, page_ids)
            packed.copy_(gathered.permute(2, 0, 1, 3, 4, 5))

        def torch_scatter() -> None:
            self.device_pool.buffer.index_copy_(2, page_ids, packed.permute(1, 2, 0, 3, 4, 5))

        try:
            from minisgl.kernel.hicache import gather_kv_pages, scatter_kv_pages

            def triton_gather() -> None:
                gather_kv_pages(self.device_pool.buffer, page_ids, packed)

            def triton_scatter() -> None:
                scatter_kv_pages(packed, page_ids, self.device_pool.buffer)

            self.stream.wait_stream(torch.cuda.current_stream(self.device_pool.device))
            with torch.cuda.stream(self.stream):
                torch_gather()
            self.stream.synchronize()
            gather_times = {
                "triton": self._measure_cuda_operation(triton_gather),
                "torch": self._measure_cuda_operation(torch_gather),
            }
            scatter_times = {
                "triton": self._measure_cuda_operation(triton_scatter),
                "torch": self._measure_cuda_operation(torch_scatter),
            }
            self.autotune_timings_ms = {
                "gather_triton": gather_times["triton"] * 1000,
                "gather_torch": gather_times["torch"] * 1000,
                "scatter_triton": scatter_times["triton"] * 1000,
                "scatter_torch": scatter_times["torch"] * 1000,
            }
            # Prefer the fused implementation inside a small noise band: it avoids
            # PyTorch's temporary gather allocation and is more stable under pressure.
            self.gather_backend = (
                "triton" if gather_times["triton"] <= gather_times["torch"] * 1.05 else "torch"
            )
            self.scatter_backend = (
                "triton" if scatter_times["triton"] <= scatter_times["torch"] * 1.02 else "torch"
            )
        except Exception as exc:
            self.gather_backend = self.scatter_backend = "torch"
            warnings.warn(
                f"Triton HiCache autotuning unavailable; using torch: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        self._refresh_copy_backend_label()
        self._copy_backends_tuned = True

    def _measure_cuda_operation(self, operation: Callable[[], None]) -> float:
        warmup_iterations = 2
        measure_iterations = 10
        with torch.cuda.stream(self.stream):
            for _ in range(warmup_iterations):
                operation()
        self.stream.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            start_event.record(self.stream)
            for _ in range(measure_iterations):
                operation()
            end_event.record(self.stream)
        end_event.synchronize()
        elapsed_ms = float(start_event.elapsed_time(end_event))
        return elapsed_ms / 1000 / measure_iterations

    def _refresh_copy_backend_label(self) -> None:
        if self.gather_backend == self.scatter_backend:
            self.copy_backend = self.gather_backend
        else:
            self.copy_backend = f"gather={self.gather_backend},scatter={self.scatter_backend}"

    def _gather_device_pages(self, page_ids: torch.Tensor, destination: torch.Tensor) -> None:
        if self.gather_backend == "triton":
            try:
                from minisgl.kernel.hicache import gather_kv_pages

                gather_kv_pages(self.device_pool.buffer, page_ids, destination)
                return
            except Exception as exc:
                if self.requested_copy_backend == "triton":
                    raise
                self.gather_backend = "torch"
                self._refresh_copy_backend_label()
                warnings.warn(
                    f"Triton HiCache gather unavailable; falling back to torch: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        gathered = torch.index_select(self.device_pool.buffer, 2, page_ids)
        destination.copy_(gathered.permute(2, 0, 1, 3, 4, 5))

    def _scatter_device_pages(self, source: torch.Tensor, page_ids: torch.Tensor) -> None:
        if self.scatter_backend == "triton":
            try:
                from minisgl.kernel.hicache import scatter_kv_pages

                scatter_kv_pages(source, page_ids, self.device_pool.buffer)
                return
            except Exception as exc:
                if self.requested_copy_backend == "triton":
                    raise
                self.scatter_backend = "torch"
                self._refresh_copy_backend_label()
                warnings.warn(
                    f"Triton HiCache scatter unavailable; falling back to torch: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.device_pool.buffer.index_copy_(2, page_ids, source.permute(1, 2, 0, 3, 4, 5))

    @staticmethod
    def _validate_pool_pages(
        pages: List[int], pool: MHAKVCache | HostMHAKVCache, tier: str
    ) -> None:
        if any(page < 0 or page >= pool.num_pages for page in pages):
            raise IndexError(f"{tier} transfer page is outside the KV pool")

    @staticmethod
    def _are_consecutive(pages: List[int]) -> bool:
        return all(page == pages[0] + offset for offset, page in enumerate(pages))

    @staticmethod
    def _check_page_counts(source_pages: List[int], destination_pages: List[int]) -> None:
        if len(source_pages) != len(destination_pages):
            raise ValueError("Source and destination transfer page counts differ")

    def _require_storage(self) -> StorageMHAKVCache:
        if self.storage_pool is None:
            raise RuntimeError("The storage cache tier is not configured")
        return self.storage_pool


__all__ = [
    "CacheTier",
    "CacheTransferManager",
    "HostMHAKVCache",
    "StorageMHAKVCache",
    "TransferDirection",
    "TransferState",
    "TransferTicket",
    "page_ids_from_token_indices",
]
