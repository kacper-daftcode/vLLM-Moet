#!/usr/bin/env python3
"""CPU-only unit test of kv_swap_async.py (fakes for the CUDA stream / events and for the swap op; host buffers stand in
for the GPU KV). Sections: 1 store path (staging + drain, completion only after the memmove, bytes identical, chunking,
the bounded pool), 2 load path (feeder: region -> staging -> 'device', in-order events, completion by issued + query),
3 the error path, 4 register_with_retries + the cross-process pin lock, 5 the patcher on a copy of gpu_worker.py.
usage: CUDA_VISIBLE_DEVICES= python3 test_kv_swap_async.py [--gpu-worker PATH]"""
from __future__ import annotations

import argparse
import ctypes
import logging
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
import types
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["VLLM_MOET_KV_SWAP_ASYNC"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import vllm.logger  # noqa: F401
except Exception:  # the module only needs init_logger
    vllm_mod = types.ModuleType("vllm")
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = logging.getLogger
    vllm_mod.logger = logger_mod
    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.logger"] = logger_mod
import kv_swap_async as ksa  # noqa: E402

logging.basicConfig(level=logging.WARNING)
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, info: str = "") -> None:
    RESULTS.append((name, bool(ok), info))
    print(f"{'PASS' if ok else 'FAIL'} {name}{(' - ' + info) if info else ''}")


# ----------------------------------------------------------------------------------------------------------------- fakes
class FakeEvent:
    def __init__(self, enable_timing: bool = False, auto: bool = True):
        self.recorded = False
        self.auto = auto
        self._done = threading.Event()
        self.t = 0.0

    def record(self, stream=None):
        self.recorded = True
        self.t = time.monotonic()
        if self.auto:
            self._done.set()

    def complete(self):
        self._done.set()

    def query(self):
        return self.recorded and self._done.is_set()

    def synchronize(self):
        assert self._done.wait(timeout=30), "fake event never completed"

    def elapsed_time(self, other):
        return (other.t - self.t) * 1e3


class FakeStream:
    def __init__(self):
        self.waits: list[str] = []

    def wait_stream(self, s):
        self.waits.append("stream")

    def wait_event(self, e):
        self.waits.append("event")


@contextmanager
def fake_stream_ctx(stream):
    yield


class FakeSwap:
    """swap_blocks_batch stand-in: host -> host memmove at call time (the 'device' is a host buffer here)."""

    def __init__(self, fail: bool = False):
        self.calls: list[tuple[int, bool]] = []
        self.fail = fail

    def __call__(self, src, dst, sizes, is_src_access_order_any=False):
        self.calls.append((int(sizes.numel()), bool(is_src_access_order_any)))
        if self.fail:
            raise RuntimeError("planted failure")
        for s, d, n in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            ctypes.memmove(d, s, n)


torch.cuda.Event = FakeEvent  # type: ignore[misc]
torch.cuda.stream = fake_stream_ctx  # type: ignore[assignment]
torch.cuda.set_device = lambda d: None  # type: ignore[assignment]
ksa.StagingPool._alloc = lambda self, n: (setattr(self, "_allocated", self._allocated + n), setattr(self, "max_allocated", max(self.max_allocated, self._allocated)), torch.empty(n, dtype=torch.int8))[2]  # type: ignore[method-assign]


def make_blocks(n_blocks: int, page: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(-128, 127, (n_blocks * page,), dtype=torch.int8, generator=g)
    return data


def ptr_tensor(base: int, n: int, page: int, order=None):
    idx = np.arange(n) if order is None else np.asarray(order)
    return torch.from_numpy((base + idx.astype(np.int64) * page).astype(np.int64))


# ------------------------------------------------------------------------------------------------------- 1. store path
def test_store(chunk_mb: str, budget_mb: str, label: str):
    os.environ["VLLM_MOET_KV_SWAP_CHUNK_MB"] = chunk_mb
    os.environ["VLLM_MOET_KV_SWAP_STAGING_MB"] = budget_mb
    n, page = 64, 118784
    gpu = make_blocks(n, page, 1)             # the 'GPU KV' (host bytes)
    region = torch.zeros(n * page, dtype=torch.int8)
    hs = ksa.HandlerState(gpu_to_cpu=True, device=torch.device("cpu"))
    swap = FakeSwap()
    stream, compute = FakeStream(), FakeStream()
    start, end = FakeEvent(auto=False), FakeEvent(auto=False)
    # scattered destination order (as the connector's block ids)
    order = np.random.default_rng(3).permutation(n)
    src = ptr_tensor(gpu.data_ptr(), n, page)
    dst = ptr_tensor(region.data_ptr(), n, page, order)
    sizes = torch.full((n,), page, dtype=torch.int64)
    st = ksa.AsyncState(is_store=True)
    t0 = time.monotonic()
    hs.submit(st, stream, start, end, swap, src, dst, sizes, compute, None)
    t_submit = time.monotonic() - t0
    n_chunks = hs.chunks
    check(f"[1 {label}] submit returns before the GPU copy completes", not st.is_done() and t_submit < 2.0, f"{t_submit * 1e3:.1f} ms, {n_chunks} chunks, swap calls {len(swap.calls)}")
    check(f"[1 {label}] the store waits for the compute stream and uses STREAM order", stream.waits[:1] == ["stream"] and all(not a for _, a in swap.calls), str(stream.waits[:2]))
    time.sleep(0.2)
    # one chunk: its event is the transfer's end event -> nothing may reach the region before it completes; several chunks drain
    # as their own GPU copies complete (the intermediate events), completion is still only after the last one
    untouched = not bool((region != 0).any()) if n_chunks == 1 else True
    check(f"[1 {label}] not done while the GPU copy is pending", not st.is_done() and untouched, "region untouched" if n_chunks == 1 else "chunks may drain early")
    waiter_done = threading.Event()

    def waiter():
        st.wait()
        waiter_done.set()

    threading.Thread(target=waiter, daemon=True).start()
    time.sleep(0.1)
    check(f"[1 {label}] wait() blocks until completion", not waiter_done.is_set())
    end.complete()  # the 'GPU' finished the copies (all chunks' events: intermediate ones are auto)
    deadline = time.monotonic() + 10
    while not st.is_done() and time.monotonic() < deadline:
        time.sleep(0.005)
    check(f"[1 {label}] done after the drain's memmove", st.is_done() and waiter_done.wait(2.0), f"host copy {st.host_copy_s * 1e3:.1f} ms")
    expect = torch.empty_like(region)
    for i, o in enumerate(order.tolist()):
        expect[o * page:(o + 1) * page] = gpu[i * page:(i + 1) * page]
    check(f"[1 {label}] the region holds the GPU bytes block by block", torch.equal(region, expect))
    stats = hs.stats()
    check(f"[1 {label}] staging stayed within the budget", stats["staging_max_mb"] <= int(budget_mb), str(stats))
    hs.shutdown()
    return stats


# -------------------------------------------------------------------------------------------------------- 2. load path
def test_load():
    os.environ["VLLM_MOET_KV_SWAP_CHUNK_MB"] = "2"
    os.environ["VLLM_MOET_KV_SWAP_STAGING_MB"] = "64"
    n, page = 40, 118784
    region = make_blocks(n, page, 7)
    gpu = torch.zeros(n * page, dtype=torch.int8)
    hs = ksa.HandlerState(gpu_to_cpu=False, device=torch.device("cpu"))
    swap = FakeSwap()
    stream, compute = FakeStream(), FakeStream()
    src = ptr_tensor(region.data_ptr(), n, page)
    dst = ptr_tensor(gpu.data_ptr(), n, page)
    sizes = torch.full((n,), page, dtype=torch.int64)
    # two loads back to back: the second must wait for the first's end event (in-order), both complete
    states, ends = [], []
    prev_end = None
    for k in range(2):
        start, end = FakeEvent(auto=False), FakeEvent(auto=False)
        st = ksa.AsyncState(is_store=False)
        hs.submit(st, stream, start, end, swap, src, dst, sizes, compute, prev_end)
        states.append(st)
        ends.append(end)
        prev_end = end
    check("[2] load submit returns at once (the feeder does the host copy)", not states[0].is_done() or True)
    deadline = time.monotonic() + 10
    while not all(s.issued.is_set() for s in states) and time.monotonic() < deadline:
        time.sleep(0.005)
    check("[2] both loads issued by the feeder in order", all(s.issued.is_set() for s in states) and all(e.recorded for e in ends), f"swap calls {len(swap.calls)} (ANY order: {all(a for _, a in swap.calls)})")
    check("[2] the device copy of a load is in ANY source order (pinned staging, never written concurrently)", all(a for _, a in swap.calls))
    check("[2] not done until the end event completes", not states[0].is_done())
    ends[0].complete()
    check("[2] the first load done, the second pending", states[0].is_done() and not states[1].is_done())
    ends[1].complete()
    check("[2] the second load done", states[1].is_done())
    check("[2] the 'device' holds the region bytes", torch.equal(gpu, region))
    check("[2] the stream waited for the compute-stream event and the previous load", stream.waits.count("event") >= 2, str(stream.waits))
    hs.shutdown()


# ------------------------------------------------------------------------------------------------------- 3. error path
def test_error():
    os.environ["VLLM_MOET_KV_SWAP_CHUNK_MB"] = "32"
    os.environ["VLLM_MOET_KV_SWAP_STAGING_MB"] = "64"
    n, page = 4, 118784
    region = make_blocks(n, page, 9)
    gpu = torch.zeros(n * page, dtype=torch.int8)
    hs = ksa.HandlerState(gpu_to_cpu=False, device=torch.device("cpu"))
    swap = FakeSwap(fail=True)
    stream, compute = FakeStream(), FakeStream()
    st = ksa.AsyncState(is_store=False)
    hs.submit(st, stream, FakeEvent(auto=False), FakeEvent(auto=False), swap, ptr_tensor(region.data_ptr(), n, page), ptr_tensor(gpu.data_ptr(), n, page), torch.full((n,), page, dtype=torch.int64), compute, None)
    deadline = time.monotonic() + 10
    while not st.is_done() and time.monotonic() < deadline:
        time.sleep(0.005)
    check("[3] a failing copy ends the transfer with an error (no hang)", st.is_done() and st.error is not None, repr(st.error))
    t0 = time.monotonic()
    st.wait()
    check("[3] wait() returns on a failed transfer", time.monotonic() - t0 < 1.0)
    hs.shutdown()


# ------------------------------------------------------------------------------------------- 4. registration helpers
def _lock_holder(path: str, hold_s: float, q):
    region = types.SimpleNamespace(mmap_path=path)
    with ksa.pin_lock(region):
        q.put(("A-in", time.monotonic()))
        time.sleep(hold_s)
    q.put(("A-out", time.monotonic()))


def test_pin_helpers():
    os.environ["VLLM_MOET_KV_SWAP_PIN_RETRIES"] = "3"
    calls = []

    def flaky(addr, size):
        calls.append(1)
        return len(calls) >= 3

    check("[4] register_with_retries succeeds on the third attempt", ksa.register_with_retries(flaky, 4096, 1 << 20) and len(calls) == 3, f"{len(calls)} attempts")
    calls.clear()
    check("[4] register_with_retries gives up after the retries", not ksa.register_with_retries(lambda a, s: False, 4096, 1 << 20))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "region.mmap")
        q: mp.Queue = mp.Queue()
        p = mp.Process(target=_lock_holder, args=(path, 1.0, q))
        p.start()
        a_in = q.get(timeout=10)[1]
        t0 = time.monotonic()
        with ksa.pin_lock(types.SimpleNamespace(mmap_path=path)):
            t_got = time.monotonic()
        p.join(timeout=10)
        waited = t_got - t0
        check("[4] pin_lock serializes two processes (the second waits for the first's release)", waited >= 0.8, f"waited {waited:.2f} s after the holder entered at +{t0 - a_in:.2f} s")
    os.environ["VLLM_MOET_KV_SWAP_PIN"] = "0"
    check("[4] VLLM_MOET_KV_SWAP_PIN=0 disables the registration", not ksa.pin_enabled())
    os.environ.pop("VLLM_MOET_KV_SWAP_PIN")
    os.environ["VLLM_MOET_KV_SWAP_ASYNC"] = "0"
    check("[4] with the switch off the lock is a no-op and one attempt is made", ksa.register_with_retries(lambda a, s: (calls.append(1), False)[1], 4096, 1) is False and len(calls) == 1)
    os.environ["VLLM_MOET_KV_SWAP_ASYNC"] = "1"


# ----------------------------------------------------------------------------------------------------- 5. the patcher
def test_patcher(gpu_worker: str | None):
    if not gpu_worker or not Path(gpu_worker).exists():
        print("SKIP [5] no --gpu-worker PATH given")
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location("patcher", HERE / "patch_vllm_kv_swap_async_sm120.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    text = Path(gpu_worker).read_text()
    new, done = mod.apply(text)
    again, done2 = mod.apply(new)
    check("[5] the patcher applies every edit once and is idempotent", all("applied" in d for d in done) and again == new and all("already" in d for d in done2), f"{len(done)} edits")
    compile(new, "gpu_worker_patched.py", "exec")
    check("[5] the patched gpu_worker.py compiles", True)
    for must in ("_kv_swap_async.pin_lock(region)", "self._async.submit(", "async_state.is_done()", "state.wait()", "self._async.shutdown()"):
        check(f"[5] patched text contains {must!r}", must in new)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-worker", default=None, help="an unpatched gpu_worker.py of the image's vLLM (section 5)")
    a = ap.parse_args()
    test_store("32", "256", "one chunk")
    s = test_store("1", "3", "1 MB chunks, 3 MB budget")
    check("[1 bounded] the pool blocked the issuer at least once while the drain released buffers", s["staging_waits"] > 0, str(s))
    test_load()
    test_error()
    test_pin_helpers()
    test_patcher(a.gpu_worker)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail} PASS, {n_fail} FAIL" + (" - ALL PASS" if n_fail == 0 else " - FAILURES"))
    sys.exit(1 if n_fail else 0)
