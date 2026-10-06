#!/usr/bin/env python3
"""Op-level test of the sm_120 cuBLAS-replica small-M BF16 GEMM on the DeepSeek-V4.1 decode shapes.

For each shape: calibrate against cuBLAS on this GPU (bf16_replica_gemm_sm120.calibrate), then check every
calibrated token count on fresh activations (unit-range and wide-range, >= 60k outputs, bit patterns), eagerly and
replayed from a CUDA graph captured on a side stream; the spread kernel's tile counters must be back at zero.
The checks run with the launch and chain variant the environment selects (VLLM_MOET_BF16_GEMM_PDL,
VLLM_MOET_BF16_CHAIN). Then, whatever the environment: both one-slice chains (v1, v2) against cuBLAS for every
token count calibrated onto a chain, and the PDL ordering of every kernel the shape uses - behind a predecessor that
releases its dependents at once and only ~20 us later writes the activations (or reads the output buffer), eagerly,
in a graph and across a fork, and back-to-back spread launches sharing one workspace in a graph.
--bench times cuBLAS against the kernel in a CUDA graph with cold weights (copies cycled through > 3x L2), for the
chains both variants, with and without PDL.

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

# A predecessor for the PDL checks: releases its programmatic dependents at once, spins, then copies src -> dst.
RACER_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void racer_kernel(const int* src, int* dst, int n, long long spin_ns) {
  asm volatile("griddepcontrol.launch_dependents;");
  long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < spin_ns);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}
void racer(torch::Tensor src, torch::Tensor dst, int64_t spin_ns) {
  TORCH_CHECK(src.is_contiguous() && dst.is_contiguous() && src.nbytes() == dst.nbytes() && src.nbytes() % 4 == 0);
  racer_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      (const int*)src.data_ptr(), (int*)dst.data_ptr(), (int)(src.nbytes() / 4), spin_ns);
}
"""
SPIN_NS = 20_000


def racer():
    if not hasattr(racer, "ext"):
        from torch.utils.cpp_extension import load_inline

        racer.ext = load_inline("vllm_moet_test_pdl_racer", cpp_sources="void racer(torch::Tensor, torch::Tensor, int64_t);",
                                cuda_sources=RACER_SRC, functions=["racer"], extra_cuda_cflags=["-O3"])
    return racer.ext.racer


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


def variants(case):
    """The kernels that may compute a calibrated case: both chains for a one-slice case."""
    if case.kind in ("chain", "ring"):
        return [R.Case(kd, case.step, case.pbf16) for kd in ("chain", "ring")]
    return [case]


def out_like(m, n, f32):
    return torch.empty(m, n, device=dev, dtype=torch.float32 if f32 else torch.bfloat16)


def check_chain_variants(name, n, k, f32, site, gen) -> bool:
    """chain v1 and v2 reproduce cuBLAS for every token count the calibration put on a one-slice chain."""
    w = (torch.randn(n, k, device=dev, generator=gen) * 0.05).to(torch.bfloat16)
    shape = R.calibrate(w, f32, k, site)
    ok, n_cases = True, 0
    for m, case in sorted(shape.cases.items()):
        if case.kind not in ("chain", "ring"):
            continue
        trials = max(2, -(-R.MIN_OUTPUTS // (m * n)))
        for c in variants(case):
            bad = 0
            for i in range(trials):
                x = unit_x(m, k, gen) if i % 2 else R._random_x(m, k, k, dev, 2000 + i * 31 + m)
                out = out_like(m, n, f32)
                R._run(shape, c, x, w, out)
                bad += int((R._bits(out) != R._bits(R.cublas_mm(x, w, f32))).sum())
            n_cases += 1
            if bad:
                print(f"  {name} M={m} {c.kind}: {bad} of {trials * m * n} outputs differ from cuBLAS")
                ok = False
    if n_cases:
        print(f"{name:15s} chain v1 and v2: {n_cases} (token count, variant) cases {'identical to cuBLAS' if ok else 'DIFFER'}")
    return ok


def check_pdl(name, n, k, f32, site, gen) -> bool:
    """The PDL launch of every kernel the calibration chose for the shape behind a predecessor that releases its
    dependents at once: the activations it writes ~20 us later are the ones used (RAW), the output buffer it reads
    ~20 us later still holds the old contents (WAR), and for spread the counters it zeroes late are the ones counted -
    eagerly, inside a graph, and across a fork to a side stream; then back-to-back spread launches sharing one
    workspace inside a graph (the second may start while the first runs)."""
    w = (torch.randn(n, k, device=dev, generator=gen) * 0.05).to(torch.bfloat16)
    shape = R.calibrate(w, f32, k, site)
    race = racer()
    saved, R.PDL = R.PDL, True
    ok, n_checks = True, 0

    def bad(tag, m, c, msg):
        nonlocal ok
        print(f"  {name} PDL {tag} M={m} {c.kind}: {msg}")
        ok = False

    try:
        work = [(m, c) for m, case in sorted(shape.cases.items()) for c in variants(case)]
        for m, c in work:  # eager
            x_new, xbuf = unit_x(m, k, gen), unit_x(m, k, gen)
            out = out_like(m, n, f32)
            race(x_new, xbuf, SPIN_NS)
            R._run(shape, c, xbuf, w, out)
            torch.cuda.synchronize()
            if not torch.equal(R._bits(out), R._bits(R.cublas_mm(x_new, w, f32))):
                bad("RAW", m, c, "the kernel did not see the activations its predecessor wrote")
            x = unit_x(m, k, gen)
            pattern = torch.randn(m, n, device=dev, generator=gen).to(out.dtype)
            out = pattern.clone()
            seen = torch.empty_like(out)
            race(out, seen, SPIN_NS)
            R._run(shape, c, x, w, out)
            torch.cuda.synchronize()
            if not torch.equal(R._bits(seen), R._bits(pattern)):
                bad("WAR", m, c, "the kernel wrote its output before its predecessor read the buffer")
            if not torch.equal(R._bits(out), R._bits(R.cublas_mm(x, w, f32))):
                bad("WAR", m, c, "wrong output")
            if c.kind == "spread":
                _, cnt = shape.workspace(-(-k // c.step))
                zeros = torch.zeros_like(cnt)
                cnt.fill_(12345)
                out.copy_(pattern)  # a missed reduction must show
                race(zeros, cnt, SPIN_NS)
                R._run(shape, c, x, w, out)
                torch.cuda.synchronize()
                if int(cnt.abs().sum()) or not torch.equal(R._bits(out), R._bits(R.cublas_mm(x, w, f32))):
                    bad("counters", m, c, "the kernel touched its counters before its predecessor zeroed them")
            n_checks += 1
        # inside a graph (programmatic edges), and across a fork (the kernel first on a side stream)
        for fork in (False, True):
            src = {i: unit_x(m, k, gen) for i, (m, _) in enumerate(work)}
            xbuf = {i: unit_x(m, k, gen) for i, (m, _) in enumerate(work)}
            outs = {}
            side = torch.cuda.Stream()
            cap = torch.cuda.Stream()
            cap.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(cap):
                with torch.cuda.graph(graph, stream=cap):
                    for i, (m, c) in enumerate(work):
                        race(src[i], xbuf[i], SPIN_NS)
                        outs[i] = out_like(m, n, f32)
                        if fork:
                            side.wait_stream(cap)
                            with torch.cuda.stream(side):
                                R._run(shape, c, xbuf[i], w, outs[i])
                            cap.wait_stream(side)
                        else:
                            R._run(shape, c, xbuf[i], w, outs[i])
            torch.cuda.synchronize()
            for rep in range(2):
                for i, (m, _) in enumerate(work):
                    src[i].copy_(unit_x(m, k, gen))
                graph.replay()
                torch.cuda.synchronize()
                for i, (m, c) in enumerate(work):
                    if not torch.equal(R._bits(outs[i]), R._bits(R.cublas_mm(src[i], w, f32))):
                        bad("graph" + (" fork" if fork else ""), m, c, f"replay {rep}: stale activations")
            n_checks += len(work)
        # back-to-back spread launches of one case sharing the stream's workspace and counters
        for m, c in work:
            if c.kind != "spread":
                continue
            xs = [unit_x(m, k, gen) for _ in range(8)]
            outs = [out_like(m, n, f32) for _ in xs]
            cap = torch.cuda.Stream()
            cap.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(cap):
                with torch.cuda.graph(graph, stream=cap):
                    for xx, oo in zip(xs, outs):
                        R._run(shape, c, xx, w, oo)
            for rep in range(3):
                for xx in xs:
                    xx.copy_(unit_x(m, k, gen))
                graph.replay()
                torch.cuda.synchronize()
                for j, (xx, oo) in enumerate(zip(xs, outs)):
                    if not torch.equal(R._bits(oo), R._bits(R.cublas_mm(xx, w, f32))):
                        bad("back-to-back", m, c, f"replay {rep} launch {j} differs from cuBLAS")
            n_checks += 1
        for _, (_, cnt) in shape.ws.items():
            if int(cnt.abs().sum()):
                print(f"  {name} PDL: spread counters not reset: {cnt.tolist()}")
                ok = False
    finally:
        R.PDL = saved
    print(f"{name:15s} PDL ordering: {n_checks} checks {'OK' if ok else 'FAILED'}")
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
            # the other launch / chain variants (bit-identical); back to back in one graph, so a PDL launch overlaps
            # the previous call's tail
            saved, cur = R.PDL, R.pdl_for(site)
            for c in variants(case):
                for pdl in (False, True):
                    if c == case and pdl == cur:
                        continue
                    R.PDL = pdl
                    t = bench_graph([(lambda w=w, o=o: R._run(shape, c, x, w, o)) for w, o in zip(ws, outs)])
                    label = {"chain": "v1", "ring": "v2"}.get(c.kind, c.kind)
                    line += f" | {label}{'+PDL' if pdl else ''} {t:6.2f}"
            R.PDL = saved
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
    pdl = "every site" if R.PDL is True else "no site" if R.PDL is False else ",".join(sorted(R.PDL))
    print(f"PDL launch: {pdl}, one-slice chain {'v2' if R.CHAIN == 'ring' else 'v1'} (environment)")
    names = list(SHAPES) if a.shapes == "all" else a.shapes.split(",")
    gen = torch.Generator(device=dev).manual_seed(20261001)
    ok = all([check_shape(nm, *SHAPES[nm], gen) for nm in names])
    ok &= all([check_chain_variants(nm, *SHAPES[nm], gen) for nm in names])
    ok &= all([check_pdl(nm, *SHAPES[nm], gen) for nm in names])
    if a.bench:
        for nm in names:
            bench_shape(nm, *SHAPES[nm], [int(m) for m in a.ms.split(",")], gen)
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
