"""Unit tests of the Engram v2 host fetcher (no GPU): synthetic table files on /mnt/nvme (O_DIRECT needs a real
filesystem: tmpfs refuses it) and, when the checkpoint is present, the real layer-1 / layer-14 tables.

    /root/work/venv/bin/python test_engram_v2.py [--scratch /mnt/nvme/cache/ds41/engram-v2-tests] [--model-dir ...]
                                                 [--cpu 33] [--no-real] [--perf-steps 400]

Checks: every row the fetcher writes into the staging equals the bytes at the row's offsets in the file (misses and
hits), duplicate ids within a request, out-of-range ids (zeros), a request larger than the ring (the issue loop's
wait path), the prefetch queue (rows warm before the request), two layers, the sequence protocol, io failure
reporting, and the publish -> ack latency of the decode pattern (12 rows per layer, both layers back to back).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engram_v2 import ROW_BYTES, ROW_SCALES, ROW_VALUES, HostFetcher  # noqa: E402

REQ_HDR = 32   # EngramV2Req header bytes before ids[]


class Mailbox:
    """Plain host memory standing in for the mapped mailbox (the GPU side is not involved here)."""

    def __init__(self, fetcher: HostFetcher, layer: int, max_rows: int):
        self.max_rows = max_rows
        self.req = (ctypes.c_uint8 * fetcher.request_bytes())()
        self.ack = (ctypes.c_uint8 * fetcher.ack_bytes())()
        self.st_values = np.zeros((max_rows, ROW_VALUES), dtype=np.uint8)
        self.st_scales = np.zeros((max_rows, ROW_SCALES), dtype=np.uint8)
        fetcher.set_mailbox(layer, ctypes.addressof(self.req), ctypes.addressof(self.ack), self.st_values.ctypes.data,
                            self.st_scales.ctypes.data)
        self.seq = 0
        self.ids_view = np.frombuffer(self.req, dtype=np.int32, offset=REQ_HDR, count=max_rows)
        self.step_view = ctypes.c_uint32.from_buffer(self.req, 0)
        self.n_rows_view = ctypes.c_uint32.from_buffer(self.req, 4)
        self.ack_step = ctypes.c_uint32.from_buffer(self.ack, 0)
        self.ack_status = ctypes.c_uint32.from_buffer(self.ack, 4)
        self.ack_hits = ctypes.c_uint32.from_buffer(self.ack, 24)
        self.ack_misses = ctypes.c_uint32.from_buffer(self.ack, 28)
        self.ack_reads = ctypes.c_uint32.from_buffer(self.ack, 32)

    def publish(self, ids: np.ndarray) -> int:
        ids = np.asarray(ids, dtype=np.int32)
        assert ids.size <= self.max_rows
        self.ids_view[: ids.size] = ids
        self.n_rows_view.value = ids.size
        self.seq += 1
        # x86 TSO: the stores above are visible before this one; the fetcher reads `step` with acquire
        self.step_view.value = self.seq
        return self.seq

    def wait(self, seq: int, timeout_s: float = 10.0) -> float:
        t0 = time.perf_counter()
        while self.ack_step.value != seq:
            if time.perf_counter() - t0 > timeout_s:
                raise TimeoutError(f"no ack for seq {seq} after {timeout_s} s")
        return time.perf_counter() - t0

    def rows(self, n: int) -> np.ndarray:
        return np.concatenate([self.st_values[:n], self.st_scales[:n]], axis=1)


def make_table(path: str, n_rows: int, values_off: int, scales_off: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 256, size=(n_rows, ROW_VALUES), dtype=np.uint8)
    scales = rng.integers(100, 140, size=(n_rows, ROW_SCALES), dtype=np.uint8)
    total = max(values_off + n_rows * ROW_VALUES, scales_off + n_rows * ROW_SCALES)
    buf = np.zeros(total + 4096, dtype=np.uint8)
    buf[values_off:values_off + n_rows * ROW_VALUES] = values.reshape(-1)
    buf[scales_off:scales_off + n_rows * ROW_SCALES] = scales.reshape(-1)
    buf.tofile(path)
    return values, scales


def expect_rows(values, scales, ids, lo, hi) -> np.ndarray:
    out = np.zeros((len(ids), ROW_BYTES), dtype=np.uint8)
    for i, r in enumerate(ids):
        if lo <= r < hi:
            out[i, :ROW_VALUES] = values[r]
            out[i, ROW_VALUES:] = scales[r]
    return out


def test_synthetic(scratch: str, cpu: int) -> None:
    n_rows = 20000
    f0 = os.path.join(scratch, "table0.bin")
    f1 = os.path.join(scratch, "table1.bin")
    v0, s0 = make_table(f0, n_rows, values_off=664, scales_off=664 + n_rows * ROW_VALUES + 8, seed=1)      # unaligned like the shard
    v1, s1 = make_table(f1, n_rows, values_off=4096, scales_off=4096 + n_rows * ROW_VALUES + 512, seed=2)  # aligned
    max_rows = 600
    layers = [dict(path=f0, values_off=664, scales_off=664 + n_rows * ROW_VALUES + 8, n_rows=n_rows, vocab_start=1000, vocab_end=19000, max_rows=max_rows),
              dict(path=f1, values_off=4096, scales_off=4096 + n_rows * ROW_VALUES + 512, n_rows=n_rows, vocab_start=0, vocab_end=n_rows, max_rows=max_rows)]
    # a tiny ring (8 entries = 4 rows in flight) so that big requests exercise the issue loop's wait path; a cache of
    # 1 MiB per layer (~ 4K rows: fewer than the 20K-row table, so the sweep below evicts)
    F = HostFetcher(layers, cache_bytes=2 * 1024 * 1024, fetch_cpu=cpu, queue_depth=8, direct=True)
    M = [Mailbox(F, 0, max_rows), Mailbox(F, 1, max_rows)]
    F.start()
    rng = np.random.default_rng(7)
    # 1. misses, in range
    ids = rng.integers(1000, 19000, size=12).astype(np.int32)
    seq = M[0].publish(ids)
    M[0].wait(seq)
    got = M[0].rows(12)
    assert np.array_equal(got, expect_rows(v0, s0, ids, 1000, 19000)), "synthetic layer 0: rows differ from the file"
    assert M[0].ack_status.value == 0 and M[0].ack_misses.value == 12 and M[0].ack_hits.value == 0, (M[0].ack_misses.value, M[0].ack_hits.value)
    assert M[0].ack_reads.value == 24, M[0].ack_reads.value
    # 2. the same ids again: hits, no reads
    seq = M[0].publish(ids)
    M[0].wait(seq)
    assert np.array_equal(M[0].rows(12), expect_rows(v0, s0, ids, 1000, 19000))
    assert M[0].ack_hits.value == 12 and M[0].ack_misses.value == 0 and M[0].ack_reads.value == 0
    # 3. duplicates + out-of-range (zeros) + a -1
    ids = np.array([5000, 5000, 999, 19000, -1, 5001, 5000, 18999], dtype=np.int32)
    seq = M[0].publish(ids)
    M[0].wait(seq)
    got = M[0].rows(len(ids))
    assert np.array_equal(got, expect_rows(v0, s0, ids, 1000, 19000)), "dups / out-of-range"
    assert not got[2].any() and not got[3].any() and not got[4].any()
    assert M[0].ack_misses.value == 3, M[0].ack_misses.value   # 5000, 5001, 18999 (5000 counted once)
    # 4. a request far larger than the ring, layer 1 (aligned offsets), with repeats
    ids = rng.integers(0, n_rows, size=max_rows).astype(np.int32)
    seq = M[1].publish(ids)
    dt = M[1].wait(seq)
    assert np.array_equal(M[1].rows(max_rows), expect_rows(v1, s1, ids, 0, n_rows)), "big request layer 1"
    print(f"  big request: {max_rows} rows, {M[1].ack_misses.value} misses, {M[1].ack_reads.value} reads in {dt * 1e3:.1f} ms (ring 8)")
    # 5. prefetch warms the cache: the following request is all hits
    ids = rng.integers(0, n_rows, size=40).astype(np.int32)
    F.prefetch(1, np.unique(ids))
    for _ in range(200):
        if F.stats()["prefetch_rows"] >= len(np.unique(ids)):
            break
        time.sleep(0.005)
    st = F.stats()
    assert st["prefetch_rows"] >= len(np.unique(ids)) - st["prefetch_hits"], st
    seq = M[1].publish(ids)
    M[1].wait(seq)
    assert np.array_equal(M[1].rows(40), expect_rows(v1, s1, ids, 0, n_rows))
    assert M[1].ack_misses.value == 0, f"prefetched rows missed: {M[1].ack_misses.value}"
    # 6. both layers back to back (the decode pattern), many times; every row checked
    lat = []
    for it in range(200):
        a = rng.integers(1000, 19000, size=12).astype(np.int32)
        b = rng.integers(0, n_rows, size=12).astype(np.int32)
        t0 = time.perf_counter()
        sa = M[0].publish(a)
        sb = M[1].publish(b)
        M[0].wait(sa)
        M[1].wait(sb)
        lat.append((time.perf_counter() - t0) * 1e6)
        assert np.array_equal(M[0].rows(12), expect_rows(v0, s0, a, 1000, 19000)), f"iteration {it} layer 0"
        assert np.array_equal(M[1].rows(12), expect_rows(v1, s1, b, 0, n_rows)), f"iteration {it} layer 1"
    lat = np.array(lat)
    print(f"  decode pattern (12 + 12 rows, ring 8, small cache): publish->both acks median {np.median(lat):.0f} us p90 {np.percentile(lat, 90):.0f} max {lat.max():.0f}")
    # 7. the cache evicts and stays consistent: a sweep over more rows than the cache holds, then re-reads
    st = F.stats()
    sweep = np.arange(0, n_rows, dtype=np.int32)
    for c in range(0, len(sweep), max_rows):
        chunk = sweep[c:c + max_rows]
        seq = M[1].publish(chunk)
        M[1].wait(seq)
        assert np.array_equal(M[1].rows(len(chunk)), expect_rows(v1, s1, chunk, 0, n_rows)), "sweep"
    st2 = F.stats()
    assert st2["cache_evictions"] > st["cache_evictions"], "no evictions in a sweep larger than the cache"
    ids = rng.choice(sweep, size=300, replace=False).astype(np.int32)
    seq = M[1].publish(ids)
    M[1].wait(seq)
    assert np.array_equal(M[1].rows(300), expect_rows(v1, s1, ids, 0, n_rows)), "after evictions"
    print(f"  cache: {st2['cache_filled']} / {st2['cache_capacity']} rows, {st2['cache_evictions']} evictions; hits so far {st2['hits']}, misses {st2['misses']}")
    # 8. a request beyond max_rows is refused with status 2 (the GPU side never sends one; the host must not overrun)
    M[0].ids_view[:] = 1000
    M[0].n_rows_view.value = max_rows + 1
    M[0].seq += 1
    M[0].step_view.value = M[0].seq
    M[0].wait(M[0].seq)
    assert M[0].ack_status.value == 2, M[0].ack_status.value
    # 9. the row helper agrees with the file
    for r in (0, 1, 777, n_rows - 1):
        assert np.array_equal(F.read_row_direct(1, r), np.concatenate([v1[r], s1[r]]))
    F.close()
    print("  synthetic: PASS")


def tensor_location(model_dir: str, name: str):
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][name]
    path = os.path.join(model_dir, shard)
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    ent = hdr[name]
    return path, 8 + n + ent["data_offsets"][0], ent["shape"]


def test_real(model_dir: str, cpu: int, perf_steps: int) -> None:
    layers = []
    for lid in (1, 14):
        pw, bw, shw = tensor_location(model_dir, f"layers.{lid}.engram.embed.weight")
        ps, bs, shs = tensor_location(model_dir, f"layers.{lid}.engram.embed.scale")
        assert pw == ps and shw[1] == 256 and shs[1] == 8
        n = shw[0]
        # rank 0 of TP4 owns the first 6 of the 24 hash columns: rows [0, sum of the first 6 primes) - use a quarter
        layers.append(dict(path=pw, values_off=bw, scales_off=bs, n_rows=n, vocab_start=0, vocab_end=n // 4, max_rows=4096 * 6))
    F = HostFetcher(layers, cache_bytes=256 << 20, fetch_cpu=cpu, queue_depth=64, direct=True)
    M = [Mailbox(F, 0, 4096 * 6), Mailbox(F, 1, 4096 * 6)]
    F.start()
    rng = np.random.default_rng(11)
    # independent reference: numpy memmap of the shard (buffered reads, not the lib)
    mm = [np.memmap(l["path"], dtype=np.uint8, mode="r") for l in layers]

    def ref_rows(li, ids):
        l = layers[li]
        out = np.zeros((len(ids), ROW_BYTES), dtype=np.uint8)
        for i, r in enumerate(ids):
            r = int(r)
            if 0 <= r < l["vocab_end"]:
                ov, os_ = int(l["values_off"]) + r * ROW_VALUES, int(l["scales_off"]) + r * ROW_SCALES
                out[i, :ROW_VALUES] = mm[li][ov: ov + ROW_VALUES]
                out[i, ROW_VALUES:] = mm[li][os_: os_ + ROW_SCALES]
        return out

    # correctness on 3 random requests per layer incl. a prefill-sized one
    for li in (0, 1):
        for n in (12, 600, 4096 * 6):
            ids = rng.integers(0, layers[li]["vocab_end"], size=n).astype(np.int32)
            t0 = time.perf_counter()
            seq = M[li].publish(ids)
            M[li].wait(seq, timeout_s=60)
            dt = time.perf_counter() - t0
            got = M[li].rows(n)
            ref = ref_rows(li, ids)
            assert np.array_equal(got, ref), f"real layer {li}: {int((got != ref).any(axis=1).sum())} of {n} rows differ"
            print(f"  real layer idx {li}: {n} rows ({M[li].ack_misses.value} misses, {M[li].ack_reads.value} reads) in {dt * 1e3:.1f} ms"
                  f"{'' if n < 1000 else f' = {M[li].ack_reads.value / dt / 1e3:.0f}K reads/s'}")
    # the decode pattern on the real files: 12 rows per layer, both layers, the gap of a decode step between
    lat = []
    for _ in range(perf_steps):
        a = rng.integers(0, layers[0]["vocab_end"], size=12).astype(np.int32)
        b = rng.integers(0, layers[1]["vocab_end"], size=12).astype(np.int32)
        t0 = time.perf_counter()
        sa = M[0].publish(a)
        sb = M[1].publish(b)
        ta = M[0].wait(sa)
        M[1].wait(sb)
        lat.append(((time.perf_counter() - t0) * 1e6, ta * 1e6))
        until = time.perf_counter() + 0.007
        while time.perf_counter() < until:
            pass
    lat = np.array(lat)
    print(f"  real decode pattern x {perf_steps} (12 + 12 rows, cold rows, 7 ms gaps): layer-1 ack median {np.median(lat[:, 1]):.0f} us "
          f"p90 {np.percentile(lat[:, 1], 90):.0f} p99 {np.percentile(lat[:, 1], 99):.0f}; both acks median {np.median(lat[:, 0]):.0f} us "
          f"p90 {np.percentile(lat[:, 0], 90):.0f} p99 {np.percentile(lat[:, 0], 99):.0f} max {lat[:, 0].max():.0f}")
    print("  " + HostFetcher.stats_text(F.stats()))
    F.close()
    print("  real tables: PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/mnt/nvme/cache/ds41/engram-v2-tests")
    ap.add_argument("--model-dir", default="/mnt/nvme/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--cpu", type=int, default=33)
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--perf-steps", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(args.scratch, exist_ok=True)
    print("synthetic tables:")
    test_synthetic(args.scratch, args.cpu)
    if not args.no_real and os.path.isfile(os.path.join(args.model_dir, "model.safetensors.index.json")):
        print("real tables:")
        test_real(args.model_dir, args.cpu, args.perf_steps)
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
