#!/usr/bin/env python3
"""Op-level test of the sm_120 cuBLAS-replica small-M BF16 GEMM on the DeepSeek-V4.1 decode shapes.

For each shape: calibrate against cuBLAS on this GPU (bf16_replica_gemm_sm120.calibrate), then check every
calibrated token count on fresh activations (unit-range and wide-range, >= 60k outputs, bit patterns), eagerly and
replayed from a CUDA graph captured on a side stream; the spread kernel's tile counters must be back at zero.
--bench times cuBLAS against the kernel in a CUDA graph with cold weights (copies cycled through > 3x L2).

usage: test_bf16_replica_gemm_sm120.py [--shapes all|name,...] [--bench] [--ms 6,12]
Exit status 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bf16_replica_gemm_sm120 as R  # noqa: E402

SHAPES = {  # name: (N, K, fp32 out, site)
    "weights_proj": (32, 5120, False, "indexer"),
    "wk": (128, 512, False, "wk"),
    "compressor_r2": (1024, 5120, True, "compressor"),
    "compressor_r1": (512, 5120, True, "compressor"),
    "router": (384, 5120, True, "router"),
    "drafter_router": (128, 5120, True, "router"),
}
dev = torch.device("cuda")


def unit_x(m, k, g):
    return torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)


def check_shape(name, n, k, f32, site, gen) -> bool:
    w = (torch.randn(n, k, device=dev, generator=gen) * 0.05).to(torch.bfloat16)
    shape = R.calibrate(w, f32, k, site)
    print(f"{name:15s} {k}->{n} {'fp32' if f32 else 'bf16'}: {shape.describe()}", flush=True)
    ok = True
    for m, case in sorted(shape.cases.items()):
        trials = max(2, -(-R.MIN_OUTPUTS // (m * n)))
        bad = 0
        for i in range(trials):
            x = unit_x(m, k, gen) if i % 2 else R._random_x(m, k, k, dev, 1000 + i * 37 + m)
            out = torch.empty(m, n, device=dev, dtype=torch.float32 if f32 else torch.bfloat16)
            R._run(shape, case, x, w, out)
            bad += int((R._bits(out) != R._bits(R.cublas_mm(x, w, f32))).sum())
        if bad:
            print(f"  M={m} {case}: {bad} of {trials * m * n} outputs differ from cuBLAS")
            ok = False
    for slices, (_, cnt) in shape.ws.items():
        if int(cnt.abs().sum()):
            print(f"  spread counters for {slices} slices not reset: {cnt.tolist()}")
            ok = False
    # graph capture on a side stream, as the indexer / compressor closures run on vLLM's aux streams
    if shape.cases:
        ms = sorted(shape.cases)
        xs = {m: unit_x(m, k, gen) for m in ms}
        outs = {}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fork = torch.cuda.Event()
            fork.record()
            side.wait_event(fork)
            with torch.cuda.stream(side):
                for m in ms[::2]:
                    outs[m] = torch.empty(m, n, device=dev, dtype=torch.float32 if f32 else torch.bfloat16)
                    R._run(shape, shape.cases[m], xs[m], w, outs[m])
            for m in ms[1::2]:
                outs[m] = torch.empty(m, n, device=dev, dtype=torch.float32 if f32 else torch.bfloat16)
                R._run(shape, shape.cases[m], xs[m], w, outs[m])
            join = torch.cuda.Event()
            join.record(side)
            torch.cuda.current_stream().wait_event(join)
        for rep in range(3):
            for m in ms:  # new inputs between replays
                xs[m].copy_(unit_x(m, k, gen))
            graph.replay()
            torch.cuda.synchronize()
            for m in ms:
                if not torch.equal(R._bits(outs[m]), R._bits(R.cublas_mm(xs[m], w, f32))):
                    print(f"  graph replay {rep}: M={m} differs from cuBLAS")
                    ok = False
        for slices, (_, cnt) in shape.ws.items():
            if int(cnt.abs().sum()):
                print(f"  spread counters not reset after graph replays: {cnt.tolist()}")
                ok = False
    return ok


def bench_graph(fns, iters=20, reps=20):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for f in fns[:3]:
            f()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(iters):
            fns[i % len(fns)]()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        graph.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / reps / iters


def bench_shape(name, n, k, f32, site, ms, gen):
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    copies = min(64, max(4, -(-(3 * l2) // (n * k * 2))))
    ws = [(torch.randn(n, k, device=dev, generator=gen) * 0.05).to(torch.bfloat16) for _ in range(copies)]
    shape = R.calibrate(ws[0], f32, k, site)
    for m in ms:
        case = shape.cases.get(m)
        x = unit_x(m, k, gen)
        t_cb = bench_graph([(lambda w=w: R.cublas_mm(x, w, f32)) for w in ws])
        line = f"{name:15s} M={m:2d} cuBLAS {t_cb:6.2f} us"
        if case is not None:
            outs = [torch.empty(m, n, device=dev, dtype=torch.float32 if f32 else torch.bfloat16) for _ in ws]
            t_rp = bench_graph([(lambda w=w, o=o: R._run(shape, case, x, w, o)) for w, o in zip(ws, outs)])
            line += f"  {case.kind:6s} {t_rp:6.2f} us  ({t_cb / t_rp:4.2f}x)"
        else:
            line += "  (cuBLAS stays)"
        print(line, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--ms", default="6,12")
    a = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    print(f"GPU {p.name}, {p.multi_processor_count} SMs, L2 {p.L2_cache_size >> 20} MB, torch {torch.__version__}")
    names = list(SHAPES) if a.shapes == "all" else a.shapes.split(",")
    gen = torch.Generator(device=dev).manual_seed(20261001)
    ok = all([check_shape(nm, *SHAPES[nm], gen) for nm in names])
    if a.bench:
        for nm in names:
            bench_shape(nm, *SHAPES[nm], [int(m) for m in a.ms.split(",")], gen)
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
