# SPDX-License-Identifier: Apache-2.0
"""KV offload connector (vLLM `v1/kv_offload/cpu/gpu_worker.py`): the copy path off the runner's main thread.

Installed next to gpu_worker.py as `vllm.v1.kv_offload.cpu.kv_swap_async` by
tools/dsv41_sm120/kv_offload/patch_vllm_kv_swap_async_sm120.py (session K-10, K/REPORT.md 13.2).

What is wrong today (measured, K-9 / K-10 on the test host, TP4, connector kv16+FS):
  * The connector's CPU side is one /dev/shm mmap region shared by the four TP workers (SharedOffloadRegion). Every
    worker `cudaHostRegister`s the whole region at start-up, all four at the same moment; in 14 of 14 server starts
    exactly one (twice: two) of them succeeded and the others failed with cudaErrorInvalidValue (a race on the long-term
    pin of the same pages) -> "the offload region stays pageable" on three ranks. Whether rank 0 - the only store
    writer of the TP-replicated MLA KV - gets the pinned region is luck per server start.
  * A `cuMemcpyBatchAsync` whose host side is pageable is host-synchronous: device -> pageable returns only when the
    copy is complete, pageable -> device does a stream sync before it starts. The store is issued on its own stream
    that waits for the compute stream, so the call returns only when the previous step's GPU work AND the copy are
    done: `_C_cache_ops::swap_blocks_batch` blocks rank 0's main thread 85-220 ms per 8 192-token step for 7.5 MB
    (K-8 traces: 211 / 219 ms; K-9 HOST_TIMING head 86 ms), ranks 1-3 wait for rank 0 in the first collective.

What this module does (`VLLM_MOET_KV_SWAP_ASYNC=1`; default off = vLLM's behaviour):
  1. `pin_lock` / `pin_retries`: the host registration of the shared region runs under a cross-process file lock
     (flock on <mmap_path>.pinlock) and is retried (VLLM_MOET_KV_SWAP_PIN_RETRIES, default 3), so the four workers
     register one after another instead of racing; `VLLM_MOET_KV_SWAP_PIN=0` skips the registration (every rank
     pageable: the staging path below on all ranks - the robustness case).
  2. If the region is pageable anyway, the store direction copies device -> a pinned staging buffer on the transfer
     stream (truly asynchronous) and a drain thread waits for that copy's event and memmoves staging -> the region;
     the load direction has a feeder thread memmove region -> pinned staging and issue the staging -> device copy
     on the transfer stream. The main thread only builds the descriptors and hands the job over.

The connector's save-before-evict invariant (offloading/scheduler.py, kept unchanged):
  * GPU source blocks of a store: a running request's blocks are protected by their ref_cnt; when the request finishes
    (deferred_fence_block_ids) or for sliding-window blocks from the store's creation (fenced_block_ids) the block ids
    are registered in `_block_id_to_pending_jobs`; when the KV cache manager re-allocates such a block in a step
    (`_current_batch_allocated_block_ids`) - or a request is preempted - the pending store jobs go into that step's
    `jobs_to_flush`, and the worker's `handle_preemptions` (before the forward) submits the deferred stores and
    `wait()`s on them: the model may overwrite the block only after the store's read of it is complete.
    -> here: `wait()` blocks until the job's device -> staging copy AND the staging -> region memmove are complete.
  * CPU destination chunks: the manager hands out a chunk to a store (prepare_store) and makes it readable / evictable
    only on `complete_store`, which the scheduler calls when every worker reported the job in `completed_jobs`
    (worker `get_finished` -> `build_connector_worker_meta` -> `update_connector_output`). A load of a chunk is only
    scheduled after that. -> here: a store is reported finished only after the memmove into the region is done.
  * Order: transfers of one direction execute in submission order (each stream waits for the previous transfer's
    end event); the drain / feeder threads are FIFO, so completions keep the order.
  * A store reads the GPU KV only after the compute stream's work that wrote it (`stream.wait_stream(compute)`); a
    load's destination blocks are written only after pending writes / zeroing to them on the compute stream
    (the load waits for an event recorded on the compute stream at submission).
"""
from __future__ import annotations

import ctypes
import fcntl
import os
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MB = 1024 * 1024


def enabled() -> bool:
    return os.environ.get("VLLM_MOET_KV_SWAP_ASYNC", "0") == "1"


def pin_enabled() -> bool:
    """False = skip the host registration of the shared region (every rank pageable -> the staging paths)."""
    return os.environ.get("VLLM_MOET_KV_SWAP_PIN", "1") != "0"


def pin_retries() -> int:
    try:
        return max(1, int(os.environ.get("VLLM_MOET_KV_SWAP_PIN_RETRIES", "3")))
    except ValueError:
        return 3


def staging_budget_bytes() -> int:
    """Pinned staging memory per direction and worker (default 256 MB; a transfer above it is split into chunks)."""
    try:
        return max(1, int(os.environ.get("VLLM_MOET_KV_SWAP_STAGING_MB", "256"))) * _MB
    except ValueError:
        return 256 * _MB


def staging_chunk_bytes() -> int:
    """Largest single staging buffer (default 32 MB = ~4 pipelined steps of 8 192 tokens of this model's KV)."""
    try:
        return max(_MB, int(os.environ.get("VLLM_MOET_KV_SWAP_CHUNK_MB", "32")) * _MB)
    except ValueError:
        return 32 * _MB


def _tp_rank() -> int:
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - outside a TP group (tests, single process)
        return 0


@contextmanager
def pin_lock(region: Any) -> Iterator[None]:
    """Cross-process exclusion for the host registration of one shared region (flock on <mmap_path>.pinlock)."""
    path = getattr(region, "mmap_path", None)
    if not enabled() or not path:
        yield
        return
    # rank order: the store writer (TP rank 0) registers first, the others queue behind it
    time.sleep(0.2 * _tp_rank())
    fd = os.open(f"{path}.pinlock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        t0 = time.monotonic()
        fcntl.flock(fd, fcntl.LOCK_EX)
        waited = time.monotonic() - t0
        if waited > 0.5:
            logger.info("KV offload: host registration lock acquired after %.1f s (another rank was registering)", waited)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def register_with_retries(host_register: Callable[[int, int], bool], address: int, size: int) -> bool:
    """host_register with retries (the registration race leaves a transient failure; a later attempt succeeds)."""
    n = pin_retries() if enabled() else 1
    for attempt in range(1, n + 1):
        if host_register(address, size):
            if attempt > 1:
                logger.info("KV offload: host registration succeeded on attempt %d", attempt)
            return True
        if attempt < n:
            time.sleep(0.5 * attempt)
    return False


# ----------------------------------------------------------------------------------------------------------------------
# staging buffers


class StagingPool:
    """Pinned host buffers for one direction; bounded by a byte budget, blocking acquire when the budget is used up."""

    def __init__(self, budget: int, chunk: int, name: str):
        self.budget = budget
        self.chunk = chunk
        self.name = name
        self._free: list[torch.Tensor] = []
        self._allocated = 0
        self._cv = threading.Condition()
        self.waits = 0
        self.max_allocated = 0

    def _alloc(self, nbytes: int) -> torch.Tensor:
        buf = torch.empty(nbytes, dtype=torch.int8, pin_memory=True)
        self._allocated += nbytes
        self.max_allocated = max(self.max_allocated, self._allocated)
        return buf

    def acquire(self, nbytes: int) -> torch.Tensor:
        assert nbytes <= self.chunk
        with self._cv:
            while True:
                for i, b in enumerate(self._free):
                    if b.numel() >= nbytes:
                        return self._free.pop(i)
                if self._allocated + nbytes <= self.budget:
                    return self._alloc(nbytes)
                # over budget: drop a too-small free buffer and re-allocate, else wait for a release
                if self._free:
                    small = self._free.pop(0)
                    self._allocated -= small.numel()
                    del small
                    continue
                self.waits += 1
                self._cv.wait()

    def release(self, buf: torch.Tensor) -> None:
        with self._cv:
            self._free.append(buf)
            self._cv.notify_all()


@dataclass
class AsyncState:
    """Completion state of one transfer on the asynchronous paths."""

    is_store: bool
    pending_chunks: int = 0
    done: threading.Event = field(default_factory=threading.Event)   # store: every chunk memmoved into the region
    issued: threading.Event = field(default_factory=threading.Event)  # load: the device copy issued, end_event recorded
    end_event: Any = None
    error: BaseException | None = None
    host_copy_s: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def chunk_done(self) -> None:
        with self._lock:
            self.pending_chunks -= 1
            last = self.pending_chunks == 0
        if last:
            self.done.set()

    def fail(self, exc: BaseException) -> None:
        self.error = exc
        self.issued.set()
        self.done.set()

    def is_done(self) -> bool:
        if self.error is not None:
            return True
        if self.is_store:
            return self.done.is_set()
        return self.issued.is_set() and bool(self.end_event.query())

    def wait(self) -> None:
        if self.is_store:
            self.done.wait()
        else:
            self.issued.wait()
            if self.error is None:
                self.end_event.synchronize()


@dataclass
class _Chunk:
    state: AsyncState
    stream: Any
    start_event: Any
    end_event: Any
    stg: torch.Tensor
    stg_base: int
    ptrs: np.ndarray      # store: the final host destinations; load: the pageable host sources
    stg_ptrs: np.ndarray  # staging addresses per op
    sizes: np.ndarray
    dev_ptrs: torch.Tensor | None = None  # load: the device destinations (int64 CPU tensor)
    ready_event: Any = None               # load: recorded on the compute stream at submission
    prev_end: Any = None                  # load: the previous load's end event (in-order execution)
    swap_fn: Callable | None = None
    first: bool = False
    last: bool = False


def _memmove_ops(dst: np.ndarray, src: np.ndarray, sizes: np.ndarray) -> None:
    for d, s, n in zip(dst.tolist(), src.tolist(), sizes.tolist()):
        ctypes.memmove(d, s, n)


class _Worker(threading.Thread):
    def __init__(self, name: str, device: torch.device, pool: StagingPool):
        super().__init__(name=name, daemon=True)
        self.device = device
        self.pool = pool
        self.q: queue.Queue[_Chunk | None] = queue.Queue()
        self.chunks = 0
        self.bytes = 0
        self.busy_s = 0.0
        self.start()

    def submit(self, chunk: _Chunk) -> None:
        self.q.put(chunk)

    def stop(self) -> None:
        self.q.put(None)
        self.join(timeout=60.0)

    def run(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        while True:
            chunk = self.q.get()
            if chunk is None:
                return
            t0 = time.monotonic()
            try:
                self._process(chunk)
            except BaseException as exc:  # noqa: BLE001 - reported through the transfer's result
                logger.exception("KV offload %s: asynchronous copy failed", self.name)
                chunk.state.fail(exc)
                self.pool.release(chunk.stg)
            self.busy_s += time.monotonic() - t0
            self.chunks += 1
            self.bytes += int(chunk.sizes.sum())

    def _process(self, chunk: _Chunk) -> None:
        raise NotImplementedError


class StoreDrain(_Worker):
    """GPU -> pinned staging was issued by the main thread; here: wait for it, memmove staging -> the pageable region."""

    def _process(self, chunk: _Chunk) -> None:
        chunk.end_event.synchronize()
        t0 = time.monotonic()
        _memmove_ops(chunk.ptrs, chunk.stg_ptrs, chunk.sizes)
        chunk.state.host_copy_s += time.monotonic() - t0
        self.pool.release(chunk.stg)
        chunk.state.chunk_done()


class LoadFeeder(_Worker):
    """memmove the pageable region -> pinned staging, then issue staging -> GPU on the transfer stream (in order)."""

    def _process(self, chunk: _Chunk) -> None:
        t0 = time.monotonic()
        _memmove_ops(chunk.stg_ptrs, chunk.ptrs, chunk.sizes)
        chunk.state.host_copy_s += time.monotonic() - t0
        stream = chunk.stream
        with torch.cuda.stream(stream):
            stream.wait_event(chunk.ready_event)
            if chunk.prev_end is not None:
                stream.wait_event(chunk.prev_end)
            if chunk.first:
                chunk.start_event.record(stream)
            stg_t = torch.from_numpy(chunk.stg_ptrs.view(np.int64))
            sizes_t = torch.from_numpy(chunk.sizes)
            assert chunk.swap_fn is not None and chunk.dev_ptrs is not None
            chunk.swap_fn(stg_t, chunk.dev_ptrs, sizes_t, is_src_access_order_any=True)
            if chunk.last:
                chunk.end_event.record(stream)
                chunk.state.issued.set()
            # the staging buffer is free once the device has read it; the loads are in-order anyway
            ev = torch.cuda.Event()
            ev.record(stream)
        ev.synchronize()
        self.pool.release(chunk.stg)
        chunk.state.chunk_done()


class HandlerState:
    """Per SingleDirectionOffloadingHandler: the staging pool + the drain / feeder thread of its direction."""

    def __init__(self, gpu_to_cpu: bool, device: torch.device):
        self.gpu_to_cpu = gpu_to_cpu
        self.device = device
        tag = "store" if gpu_to_cpu else "load"
        self.pool = StagingPool(staging_budget_bytes(), staging_chunk_bytes(), tag)
        self.worker: _Worker = (StoreDrain if gpu_to_cpu else LoadFeeder)(f"kv-swap-{tag}", device, self.pool)
        self.transfers = 0
        self.chunks = 0
        self.bytes = 0
        self.sync_fallbacks = 0
        logger.info(
            "KV offload %s: asynchronous copies through pinned staging (VLLM_MOET_KV_SWAP_ASYNC=1; budget %d MB, chunk %d MB) - the region is pageable",
            tag, self.pool.budget // _MB, self.pool.chunk // _MB,
        )

    def _chunks(self, sizes: np.ndarray) -> list[tuple[int, int]]:
        """Split the op list into consecutive ranges whose bytes fit one staging buffer."""
        out: list[tuple[int, int]] = []
        start, acc = 0, 0
        cap = self.pool.chunk
        for i, n in enumerate(sizes.tolist()):
            if n > cap:
                raise ValueError(f"KV offload: one copy of {n} bytes exceeds the staging chunk of {cap} bytes (raise VLLM_MOET_KV_SWAP_CHUNK_MB)")
            if acc + n > cap:
                out.append((start, i))
                start, acc = i, 0
            acc += n
        out.append((start, len(sizes)))
        return out

    def submit(
        self,
        state: AsyncState,
        stream: Any,
        start_event: Any,
        end_event: Any,
        swap_fn: Callable,
        src: torch.Tensor,
        dst: torch.Tensor,
        sizes: torch.Tensor,
        compute_stream: Any,
        prev_end: Any,
    ) -> None:
        """Issue one transfer on the asynchronous path. src / dst / sizes: the int64 descriptor tensors (CPU)."""
        sizes_np = sizes.numpy().astype(np.int64, copy=True)
        host_np = (dst if self.gpu_to_cpu else src).numpy().astype(np.int64, copy=True)
        dev_t = (src if self.gpu_to_cpu else dst)
        ranges = self._chunks(sizes_np)
        state.pending_chunks = len(ranges)
        state.end_event = end_event
        self.transfers += 1
        self.chunks += len(ranges)
        self.bytes += int(sizes_np.sum())
        if self.gpu_to_cpu:
            # device -> pinned staging on the transfer stream now (asynchronous: pinned destination), the drain does the rest
            with torch.cuda.stream(stream):
                stream.wait_stream(compute_stream)
                if prev_end is not None:
                    stream.wait_event(prev_end)
                start_event.record(stream)
                for ci, (lo, hi) in enumerate(ranges):
                    n = int(sizes_np[lo:hi].sum())
                    stg = self.pool.acquire(n)
                    base = stg.data_ptr()
                    offs = np.concatenate(([0], np.cumsum(sizes_np[lo:hi])[:-1])).astype(np.int64)
                    stg_ptrs = base + offs
                    swap_fn(dev_t[lo:hi], torch.from_numpy(stg_ptrs), sizes[lo:hi], is_src_access_order_any=False)
                    ev = end_event if ci == len(ranges) - 1 else torch.cuda.Event()
                    ev.record(stream)
                    self.worker.submit(_Chunk(state, stream, start_event, ev, stg, base, host_np[lo:hi], stg_ptrs, sizes_np[lo:hi], last=ci == len(ranges) - 1))
        else:
            ready = torch.cuda.Event()
            ready.record(compute_stream)
            for ci, (lo, hi) in enumerate(ranges):
                n = int(sizes_np[lo:hi].sum())
                stg = self.pool.acquire(n)
                base = stg.data_ptr()
                offs = np.concatenate(([0], np.cumsum(sizes_np[lo:hi])[:-1])).astype(np.int64)
                stg_ptrs = base + offs
                self.worker.submit(_Chunk(state, stream, start_event, end_event, stg, base, host_np[lo:hi], stg_ptrs, sizes_np[lo:hi],
                                          dev_ptrs=dev_t[lo:hi].clone(), ready_event=ready, prev_end=prev_end if ci == 0 else None,
                                          swap_fn=swap_fn, first=ci == 0, last=ci == len(ranges) - 1))

    def stats(self) -> dict[str, Any]:
        return {
            "direction": "store" if self.gpu_to_cpu else "load",
            "transfers": self.transfers,
            "chunks": self.chunks,
            "bytes": self.bytes,
            "thread_chunks": self.worker.chunks,
            "thread_busy_s": round(self.worker.busy_s, 3),
            "staging_max_mb": self.pool.max_allocated // _MB,
            "staging_waits": self.pool.waits,
        }

    def shutdown(self) -> None:
        self.worker.stop()
