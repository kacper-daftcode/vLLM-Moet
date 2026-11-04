"""Engram v2, Python side: the host fetcher (libengram_v2.so: DRAM row cache + io_uring NVMe reads + the mailbox
protocol), the capturable GPU ops (engram_v2_cuda.cu: publish, bounded wait + landing copy), the per-layer mapped
mailboxes / pinned staging / device landing arrays, and the prefill lookahead (the prompt's hashes on the host,
the next chunks' rows prefetched into the row cache).

    eng = EngramV2([dict(path=..., values_off=..., scales_off=..., n_rows=..., vocab_start=..., vocab_end=..., max_rows=...), ...],
                   cache_bytes=2 << 30, fetch_cpu=33, queue_depth=64, direct=True, device=torch.device("cuda", 0))
    eng.start()
    L = eng.layers[i]   # req_d / ack_d / st_values_d / st_scales_d (device pointers), land_values / land_scales, seq_dev, arange
    eng.ext.publish(L.req_d, L.seq_dev, hash_ids, head_start, local_heads, n_tokens, L.max_rows)
    eng.ext.wait_copy(L.ack_d, L.req_d, L.seq_dev, L.st_values_d, L.st_scales_d, L.land_values, L.land_scales, eng.err_d, eng.timeout_ns, rows)
    ... vLLM's _engram_lookup_kernel over L.land_values / L.land_scales ...
    eng.error()   # nonzero after a step failed closed
"""
from __future__ import annotations

import ctypes
import functools
import os
import threading

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROW_VALUES, ROW_SCALES, ROW_BYTES = 256, 8, 264
STATS_N = 35


@functools.lru_cache(maxsize=None)
def cuda_ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("MOET_CPU_TIER_ARCH", "12.0")
    return load(name="moet_engram_v2_cuda", sources=[os.path.join(_HERE, "engram_v2_cuda.cu")], extra_include_paths=[_HERE],
                extra_cuda_cflags=["-O3", "-std=c++20"], extra_cflags=["-O3", "-std=c++20"], verbose=False)


class _LayerCfg(ctypes.Structure):
    _fields_ = [("path", ctypes.c_char_p), ("values_off", ctypes.c_int64), ("scales_off", ctypes.c_int64), ("n_rows", ctypes.c_int64),
                ("vocab_start", ctypes.c_int64), ("vocab_end", ctypes.c_int64)]


class _Config(ctypes.Structure):
    _fields_ = [("n_layers", ctypes.c_int), ("layers", ctypes.POINTER(_LayerCfg)), ("cache_bytes", ctypes.c_int64),
                ("fetch_cpu", ctypes.c_int), ("queue_depth", ctypes.c_int), ("direct", ctypes.c_int), ("max_rows", ctypes.c_int),
                ("prefetch_batch", ctypes.c_int), ("resident_scales", ctypes.c_int)]


@functools.lru_cache(maxsize=None)
def host_lib(path: str | None = None):
    lib = ctypes.CDLL(path or os.environ.get("ENGRAM_V2_LIB") or os.path.join(_HERE, "libengram_v2.so"))
    lib.engram_v2_create.restype = ctypes.c_void_p
    lib.engram_v2_create.argtypes = [ctypes.POINTER(_Config)]
    lib.engram_v2_set_mailbox.restype = ctypes.c_int
    lib.engram_v2_set_mailbox.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.engram_v2_start.restype = ctypes.c_int
    lib.engram_v2_start.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.engram_v2_stop.restype = ctypes.c_int
    lib.engram_v2_stop.argtypes = [ctypes.c_void_p]
    lib.engram_v2_start_polled.restype = ctypes.c_int
    lib.engram_v2_start_polled.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.engram_v2_poll.restype = None
    lib.engram_v2_poll.argtypes = [ctypes.c_void_p]
    lib.engram_v2_destroy.argtypes = [ctypes.c_void_p]
    lib.engram_v2_request_bytes.restype = ctypes.c_int64
    lib.engram_v2_request_bytes.argtypes = [ctypes.c_int]
    lib.engram_v2_ack_bytes.restype = ctypes.c_int64
    lib.engram_v2_prefetch.restype = ctypes.c_int
    lib.engram_v2_prefetch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.engram_v2_prefetch_clear.argtypes = [ctypes.c_void_p]
    lib.engram_v2_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
    lib.engram_v2_now_ns.restype = ctypes.c_int64
    lib.engram_v2_read_row_direct.restype = ctypes.c_int
    lib.engram_v2_read_row_direct.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int32, ctypes.c_void_p]
    return lib


STAT_NAMES = ["requests", "rows", "hits", "misses", "reads", "bytes", "io_errors", "serve_ns", "serve_max_ns", "small_requests",
              "small_serve_ns", "small_serve_max_ns", "big_requests", "big_serve_ns", "big_serve_max_ns", "prefetch_rows",
              "prefetch_reads", "prefetch_hits", "resyncs", "zero_rows", "dup_rows", "prefetch_queued",
              "h50", "h100", "h200", "h300", "h500", "h1000", "h5000", "hmore", "cache_filled", "cache_evictions", "cache_capacity",
              "ring_err", "resident_scale_bytes"]


class HostFetcher:
    """The fetcher of one rank without any CUDA (tests, benches): plain host memory for the mailboxes and staging."""

    def __init__(self, layers: list[dict], cache_bytes: int, fetch_cpu: int = -1, queue_depth: int = 64, direct: bool = True,
                 prefetch_batch: int = 0, lib=None, resident_scales: bool = False):
        self.lib = lib or host_lib()
        self.layers_cfg = layers
        self.max_rows = max(int(l["max_rows"]) for l in layers)
        self._paths = [l["path"].encode() for l in layers]
        arr = (_LayerCfg * len(layers))()
        for i, l in enumerate(layers):
            arr[i] = _LayerCfg(self._paths[i], int(l["values_off"]), int(l["scales_off"]), int(l["n_rows"]), int(l["vocab_start"]),
                               int(l["vocab_end"]))
        self._cfg = _Config(len(layers), arr, int(cache_bytes), int(fetch_cpu), int(queue_depth), 1 if direct else 0, self.max_rows,
                            int(prefetch_batch), 1 if resident_scales else 0)
        self._arr = arr
        self.handle = self.lib.engram_v2_create(ctypes.byref(self._cfg))
        if not self.handle:
            raise RuntimeError("engram_v2_create failed (io_uring denied by seccomp? a table file missing?)")
        self.started = False

    def request_bytes(self) -> int:
        return int(self.lib.engram_v2_request_bytes(self.max_rows))

    def ack_bytes(self) -> int:
        return int(self.lib.engram_v2_ack_bytes())

    def set_mailbox(self, layer: int, req: int, ack: int, st_values: int, st_scales: int) -> None:
        assert self.lib.engram_v2_set_mailbox(self.handle, layer, req, ack, st_values, st_scales) == 0

    def start(self, first_seq: int = 1) -> None:
        rc = self.lib.engram_v2_start(self.handle, first_seq)
        assert rc == 0, f"engram_v2_start rc {rc}"
        self.started = True

    def stop(self) -> None:
        if self.started:
            self.lib.engram_v2_stop(self.handle)
            self.started = False

    def start_polled(self) -> None:
        """The polled mode (no thread): the owner calls lib.engram_v2_poll(handle) from its own loop - the CPU tier's
        dispatcher through CpuTier.set_poll_hook. Call after stop() when switching from the thread mode."""
        rc = self.lib.engram_v2_start_polled(self.handle, 0)
        assert rc == 0, f"engram_v2_start_polled rc {rc}"
        self.started = True

    def close(self) -> None:
        self.stop()
        if self.handle:
            self.lib.engram_v2_destroy(self.handle)
            self.handle = None

    def prefetch(self, layer: int, ids: np.ndarray) -> None:
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        if ids.size:
            assert self.lib.engram_v2_prefetch(self.handle, layer, ids.ctypes.data, int(ids.size)) == 0

    def prefetch_clear(self) -> None:
        self.lib.engram_v2_prefetch_clear(self.handle)

    def stats(self, reset: bool = False) -> dict:
        buf = (ctypes.c_int64 * STATS_N)()
        self.lib.engram_v2_stats(self.handle, buf, 1 if reset else 0)
        return dict(zip(STAT_NAMES, [int(x) for x in buf]))

    def cache_rows(self) -> int:
        return self.stats()["cache_capacity"]

    def read_row_direct(self, layer: int, row: int) -> np.ndarray:
        """Test helper: the row's 264 bytes straight from the file (pread), bypassing the cache."""
        out = np.empty(ROW_BYTES, dtype=np.uint8)
        rc = self.lib.engram_v2_read_row_direct(self.handle, layer, row, out.ctypes.data)
        assert rc == 0, rc
        return out

    def now_ns(self) -> int:
        return int(self.lib.engram_v2_now_ns())

    @staticmethod
    def stats_text(s: dict) -> str:
        req = max(1, s["requests"])
        small = max(1, s["small_requests"])
        big = max(1, s["big_requests"])
        rows = max(1, s["rows"])
        hist = "/".join(str(s[k]) for k in ("h50", "h100", "h200", "h300", "h500", "h1000", "h5000", "hmore"))
        return (f"requests {s['requests']} rows {s['rows']} hit {s['hits'] / rows:.3f} (hits {s['hits']} misses {s['misses']} dup {s['dup_rows']} "
                f"zero {s['zero_rows']}) reads {s['reads']} ({s['bytes'] / 2**20:.1f} MiB) io_errors {s['io_errors']}; "
                f"serve small ({s['small_requests']}) mean {s['small_serve_ns'] / small / 1e3:.0f} us max {s['small_serve_max_ns'] / 1e3:.0f} us "
                f"hist(<=50/100/200/300/500/1000/5000/>us) {hist}; big ({s['big_requests']}) mean {s['big_serve_ns'] / big / 1e6:.1f} ms "
                f"max {s['big_serve_max_ns'] / 1e6:.1f} ms; prefetch rows {s['prefetch_rows']} reads {s['prefetch_reads']} "
                f"hits {s['prefetch_hits']} queued {s['prefetch_queued']}; cache {s['cache_filled']}/{s['cache_capacity']} rows, "
                f"evictions {s['cache_evictions']}; resident scales {s['resident_scale_bytes'] / 2**20:.0f} MiB; resyncs {s['resyncs']}"
                + (f"; RING ERROR {s['ring_err']}" if s["ring_err"] else ""))


class _Layer:
    pass


class EngramV2:
    """The fetcher + the GPU-visible mailboxes, staging and landing of every Engram layer of this rank."""

    def __init__(self, layers: list[dict], cache_bytes: int, fetch_cpu: int, queue_depth: int, direct: bool, device: torch.device,
                 timeout_ms: float = 2000.0, lib_path: str | None = None, resident_scales: bool = False):
        self.ext = cuda_ext()
        self.lib = host_lib(lib_path)
        self.device = device
        self.timeout_ns = int(timeout_ms * 1e6)
        self.fetcher = HostFetcher(layers, cache_bytes, fetch_cpu, queue_depth, direct, lib=self.lib, resident_scales=resident_scales)
        assert self.fetcher.request_bytes() == self.ext.request_bytes(self.fetcher.max_rows)
        assert self.fetcher.ack_bytes() == self.ext.ack_bytes()
        self.layers: list[_Layer] = []
        self._mapped: list[int] = []
        with torch.cuda.device(device):
            self.err = self.ext.alloc_mapped(64)
            self.err_d = self.ext.device_ptr(self.err)
            self._mapped.append(self.err)
            for i, cfg in enumerate(layers):
                L = _Layer()
                L.max_rows = int(cfg["max_rows"])
                L.req = self.ext.alloc_mapped(self.ext.request_bytes(L.max_rows))
                L.ack = self.ext.alloc_mapped(self.ext.ack_bytes())
                L.st_values = self.ext.alloc_mapped(L.max_rows * ROW_VALUES)
                L.st_scales = self.ext.alloc_mapped(L.max_rows * ROW_SCALES)
                self._mapped += [L.req, L.ack, L.st_values, L.st_scales]
                L.req_d, L.ack_d = self.ext.device_ptr(L.req), self.ext.device_ptr(L.ack)
                L.st_values_d, L.st_scales_d = self.ext.device_ptr(L.st_values), self.ext.device_ptr(L.st_scales)
                L.land_values = torch.zeros(L.max_rows, ROW_VALUES, dtype=torch.uint8, device=device)
                L.land_scales = torch.zeros(L.max_rows, ROW_SCALES, dtype=torch.uint8, device=device)
                L.seq_dev = torch.zeros(1, dtype=torch.int32, device=device)
                L.arange = torch.arange(L.max_rows, dtype=torch.int32, device=device)
                self.fetcher.set_mailbox(i, L.req, L.ack, L.st_values, L.st_scales)
                self.layers.append(L)
        self._err_view = (ctypes.c_uint32 * 16).from_address(self.err)

    def start(self) -> None:
        self.fetcher.start(first_seq=1)

    def attach_dispatcher(self, tier) -> None:
        """Hand the fetcher to the CPU tier's dispatcher thread (cpu_tier.CpuTier): its own thread stops, the dispatcher
        polls the mailboxes / completions from its dedicated core."""
        self.fetcher.stop()
        self.fetcher.start_polled()
        tier.set_poll_hook(self.lib.engram_v2_poll, self.fetcher.handle)
        self.attached = tier

    def detach_dispatcher(self) -> None:
        tier = getattr(self, "attached", None)
        if tier is not None:
            try:
                tier.set_poll_hook(None, None)
            except Exception:
                pass
            self.attached = None

    def error(self) -> int:
        return int(self._err_view[0])

    def clear_error(self) -> None:
        self._err_view[0] = 0

    def stats(self, reset: bool = False) -> dict:
        return self.fetcher.stats(reset)

    def stats_line(self, reset: bool = False) -> str:
        return HostFetcher.stats_text(self.fetcher.stats(reset))

    def cache_rows(self) -> int:
        return self.fetcher.cache_rows()

    def prefetch(self, layer: int, ids: np.ndarray) -> None:
        self.fetcher.prefetch(layer, ids)

    def prefetch_clear(self) -> None:
        self.fetcher.prefetch_clear()

    def close(self) -> None:
        self.detach_dispatcher()
        self.fetcher.close()
        for p in self._mapped:
            self.ext.free_mapped(p)
        self._mapped = []


# ---------------------------------------------------------------------------------------------------------------
# the host-side hash (vLLM's _hash_ids_kernel in numpy, text-only: no dead tokens) and the prefill lookahead
# ---------------------------------------------------------------------------------------------------------------
class HostHash:
    """Row ids of the n-grams ending at every position of a token stream, per Engram layer, for this rank's hash
    columns - the same arithmetic as vLLM's _hash_ids_kernel (the compressed token map, the per-layer multipliers,
    the prime bucket ranges and offsets all come from vLLM's own code). Positions before the stream start hash the
    pad id, like the kernel's `lookback < 0` rule. Text only: image-span (dead) tokens are not modelled - a wrong
    guess there costs a cache miss, never a wrong row (the rows a step uses always come from the GPU's hashes)."""

    def __init__(self, model_dir: str, layout=None, token_map=None, pad_id: int | None = None):
        import json
        from vllm.models.deepseek_v41.common.engram import EngramLayout, build_compressed_token_map, compute_hash_multipliers

        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        cfg = cfg.get("text_config", cfg)

        class _C:   # EngramLayout reads attributes
            pass
        c = _C()
        for k, v in cfg.items():
            setattr(c, k, v)
        self.layout = layout or EngramLayout(c)
        if token_map is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            token_map, vocab = build_compressed_token_map(tok)
            assert vocab == self.layout.compressed_vocab_size, (vocab, self.layout.compressed_vocab_size)
        self.token_map = np.asarray(token_map, dtype=np.int64)
        self.pad_id = int(self.token_map[self.layout.pad_token_id]) if pad_id is None else pad_id
        self.mult = compute_hash_multipliers(self.layout.layer_ids, self.layout.max_ngram_size, self.layout.compressed_vocab_size).numpy()
        self.primes = np.array([[p for per in layer for p in per] for layer in self.layout.primes], dtype=np.int64)
        self.offsets = self.layout.offsets.numpy().astype(np.int64)
        self.n_heads = self.layout.n_heads
        self.max_ngram = self.layout.max_ngram_size

    def hash_positions(self, token_ids: np.ndarray, start: int, end: int, layer_index: int, cols: slice | None = None) -> np.ndarray:
        """[end - start, n_cols] int32 row ids of layer `layer_index` for positions start..end-1 of the stream."""
        ids = np.asarray(token_ids, dtype=np.int64)
        T = end - start
        comp = self.token_map[ids]
        rolling = np.zeros(T, dtype=np.int64)
        pos = np.arange(start, end)
        n_cols = (self.max_ngram - 1) * self.n_heads
        out = np.empty((T, n_cols), dtype=np.int64)
        for shift in range(self.max_ngram):
            lb = pos - shift
            value = np.where(lb >= 0, comp[np.clip(lb, 0, None)], self.pad_id)
            rolling ^= value * self.mult[layer_index, shift]
            if shift > 0:
                for h in range(self.n_heads):
                    col = (shift - 1) * self.n_heads + h
                    out[:, col] = rolling % self.primes[layer_index, col] + self.offsets[layer_index, col]
        if cols is not None:
            out = out[:, cols]
        return out.astype(np.int32)


class Lookahead:
    """The prefill lookahead: when a request is scheduled, its prompt's rows for the chunks after the current one are
    queued for background reads into the row cache (this rank's hash columns, both Engram layers), up to `chunks`
    chunks ahead; the queue is cleared when the prompt is through. The hashing runs in a helper thread so the step
    hook never waits for it."""

    def __init__(self, eng: EngramV2, tables: list, model_dir: str, chunk_tokens: int, chunks: int):
        self.eng, self.tables, self.chunk, self.ahead = eng, tables, chunk_tokens, chunks
        self.hh = None
        self.model_dir = model_dir
        self.reqs: dict[str, dict] = {}    # req_id -> dict(prompt=np.ndarray, queued_to=int)
        self.stats = dict(requests=0, rows_queued=0, hash_s=0.0, skipped=0)
        self.lock = threading.Lock()
        self._jobs: list = []
        self._cv = threading.Condition(self.lock)
        self._thread = threading.Thread(target=self._worker, name="engram-v2-lookahead", daemon=True)
        self._thread.start()

    def _ensure_hash(self):
        if self.hh is None:
            import time
            t0 = time.perf_counter()
            self.hh = HostHash(self.model_dir, layout=None)
            self.stats["hash_s"] = time.perf_counter() - t0

    def _worker(self):
        while True:
            with self._cv:
                while not self._jobs:
                    self._cv.wait()
                job = self._jobs.pop(0)
            try:
                self._ensure_hash()
                prompt, start, end = job
                for T in self.tables:
                    emb = T.emb
                    cols = slice(int(emb.head_start), int(emb.head_start + emb.part_n_hash_cols))
                    ids = self.hh.hash_positions(prompt, start, end, T.idx, cols)
                    self.eng.prefetch(T.eng_index, np.unique(ids.reshape(-1)))
                    with self.lock:
                        self.stats["rows_queued"] += int(ids.size)
            except Exception:
                with self.lock:
                    self.stats["skipped"] += 1

    def on_schedule(self, so) -> None:
        """scheduler_output of the step about to run (the V2 runner's execute_model argument)."""
        new = getattr(so, "scheduled_new_reqs", None) or []
        for r in new:
            ids = getattr(r, "prompt_token_ids", None)
            if ids is None or len(ids) <= self.chunk:
                continue
            with self.lock:
                self.reqs[r.req_id] = dict(prompt=np.asarray(ids, dtype=np.int64), queued_to=0)
                self.stats["requests"] += 1
        sched = getattr(so, "num_scheduled_tokens", {}) or {}
        cached = getattr(so, "scheduled_cached_reqs", None)
        computed: dict[str, int] = {}
        for r in new:
            computed[r.req_id] = int(getattr(r, "num_computed_tokens", 0) or 0)
        if cached is not None:
            for rid, nc in zip(getattr(cached, "req_ids", []), getattr(cached, "num_computed_tokens", [])):
                computed[rid] = int(nc)
        for rid, n in sched.items():
            info = self.reqs.get(rid)
            if info is None:
                continue
            prompt = info["prompt"]
            pos = computed.get(rid, 0) + int(n)          # the first position after this step's chunk
            if pos >= len(prompt):                        # the prompt is through: nothing left to prefetch
                with self.lock:
                    self.reqs.pop(rid, None)
                self.eng.prefetch_clear()
                continue
            end = min(len(prompt), pos + self.ahead * self.chunk)
            start = max(pos, info["queued_to"])
            if start < end:
                info["queued_to"] = end
                with self._cv:
                    self._jobs.append((prompt, start, end))
                    self._cv.notify()
        for rid in getattr(so, "finished_req_ids", []) or []:
            with self.lock:
                if self.reqs.pop(rid, None) is not None:
                    self.eng.prefetch_clear()

    def stats_line(self) -> str:
        s = self.stats
        return f"requests {s['requests']} rows queued {s['rows_queued']} hash setup {s['hash_s']:.1f} s skipped {s['skipped']}"
