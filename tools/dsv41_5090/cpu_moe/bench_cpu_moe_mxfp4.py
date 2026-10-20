"""G1 benchmark of the CPU MXFP4 expert kernel: GB/s of expert bytes per CCD at m = 1..8 tokens per expert.

    python bench_cpu_moe_mxfp4.py [--experts 96] [--per-job 48] [--ms 1,2,3,4,6,8] [--ccds 0] [--smt]
                                  [--layer 20] [--repeat 5] [--json out.json]

One pool per CCD (8 cores; --smt adds the sibling hyperthreads). Every job = `per-job` distinct experts in a
random order (the packs live in a hugepage arena far larger than the 32 MB L3, so every expert streams from
DRAM) with m tokens each. Reports the median per-expert time, the bandwidth in pack bytes (4.81 MB per
expert slice) and in checkpoint-record bytes (4.70 MB), the read-only ceiling of the same striping, and
with --ccds 0,1,2,3 the aggregate of four pools running at once (the four TP4 ranks). Gate G1 of the plan:
>= 45 GB/s per CCD at m = 1 (>= 180 GB/s for four), and m = 6 within 2x of m = 1.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv41_mxfp4 as D  # noqa: E402

CCD_CPUS = {0: list(range(0, 8)), 1: list(range(8, 16)), 2: list(range(16, 24)), 3: list(range(24, 32))}
SMT_SIBLING = 32


def load_experts(layer: int, experts: list[int], rank: int, arena: D.PackArena) -> None:
    t0 = time.time()
    for i, e in enumerate(experts):
        arena.slot(i)[:] = D.build_pack(D.expert_slice_raw(layer, e, rank))
    print(f"  loaded + packed {len(experts)} experts of layer {layer} rank {rank} in {time.time() - t0:.1f}s", flush=True)


def make_inputs(T: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(T, D.HID, generator=gen).to(torch.bfloat16)
    xq, xe = D.quant_fp8_ue8m0(x.float())
    return D.to_kernel_inputs(xq, xe)


def run_jobs(pool: D.Pool, arena: D.PackArena, n_exp: int, per_job: int, m: int, repeat: int, seed: int,
             xq: np.ndarray, xs: np.ndarray, barrier: threading.Barrier | None = None) -> list[float]:
    rng = np.random.default_rng(seed)
    T = xq.shape[0]
    times = []
    out = np.zeros((per_job * m, D.HID), dtype=np.uint16)
    for r in range(repeat + 1):  # the first run is a warm-up (page faults on out)
        order = rng.permutation(n_exp)[:per_job]
        ptrs = [arena.ptr(int(i)) for i in order]
        toks = [int(t) for _ in range(per_job) for t in rng.permutation(T)[:m]]
        if barrier is not None:
            barrier.wait()
        t0 = time.perf_counter()
        pool.experts_forward(ptrs, [m] * per_job, toks, xq, xs, out)
        dt = time.perf_counter() - t0
        if r > 0:
            times.append(dt / per_job)
    return times


def run_read(pool: D.Pool, arena: D.PackArena, n_exp: int, per_job: int, repeat: int, seed: int,
             barrier: threading.Barrier | None = None) -> list[float]:
    rng = np.random.default_rng(seed)
    times = []
    for r in range(repeat + 1):
        order = rng.permutation(n_exp)[:per_job]
        ptrs = [arena.ptr(int(i)) for i in order]
        if barrier is not None:
            barrier.wait()
        t0 = time.perf_counter()
        pool.read_bench(ptrs)
        dt = time.perf_counter() - t0
        if r > 0:
            times.append(dt / per_job)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=96)
    ap.add_argument("--per-job", type=int, default=48)
    ap.add_argument("--ms", default="1,2,3,4,6,8")
    ap.add_argument("--ccds", default="0")
    ap.add_argument("--smt", action="store_true", help="16 threads per CCD (cores + siblings)")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    ccds = [int(c) for c in args.ccds.split(",")]
    ms = [int(m) for m in args.ms.split(",")]
    lib = D.load_lib()
    torch.set_num_threads(16)

    print(f"experts per arena {args.experts} ({args.experts * D.PACK_BYTES / 1e6:.0f} MB), per job {args.per_job}, "
          f"CCDs {ccds}, threads/CCD {16 if args.smt else 8}")
    arenas, pools = [], []
    for i, c in enumerate(ccds):
        arena = D.PackArena(args.experts, lib)
        # each CCD gets a different rank's slices (distinct bytes, like the four TP ranks)
        load_experts(args.layer, list(range(args.experts)), c % D.TP, arena)
        cpus = CCD_CPUS[c] + ([x + SMT_SIBLING for x in CCD_CPUS[c]] if args.smt else [])
        pools.append(D.Pool(cpus, lib))
        arenas.append(arena)
    xq, xs = make_inputs(args.tokens, 1)

    results = {"pack_bytes": D.PACK_BYTES, "record_bytes": D.RAW_BYTES, "ccds": ccds, "smt": args.smt, "rows": []}

    def report(label: str, per_ccd_times: list[list[float]], m: int | None):
        med = [statistics.median(t) for t in per_ccd_times]
        us = [1e6 * x for x in med]
        gbs_pack = [D.PACK_BYTES / x / 1e9 for x in med]
        gbs_rec = [D.RAW_BYTES / x / 1e9 for x in med]
        agg = sum(gbs_pack)
        print(f"  {label:22s} per-expert us " + " ".join(f"{u:7.1f}" for u in us) +
              f"   GB/s(pack) " + " ".join(f"{g:5.1f}" for g in gbs_pack) +
              f"   GB/s(record) " + " ".join(f"{g:5.1f}" for g in gbs_rec) +
              (f"   aggregate {agg:6.1f} GB/s" if len(ccds) > 1 else ""))
        results["rows"].append(dict(label=label, m=m, per_expert_us=us, gbs_pack=gbs_pack, gbs_record=gbs_rec,
                                    aggregate_gbs_pack=agg))

    # read-only ceiling
    print("read-only streaming of the packs (DRAM ceiling of this striping):")
    bar = threading.Barrier(len(ccds)) if len(ccds) > 1 else None
    res = [None] * len(ccds)

    def read_worker(i):
        res[i] = run_read(pools[i], arenas[i], args.experts, args.per_job, args.repeat, 100 + i, bar)

    ths = [threading.Thread(target=read_worker, args=(i,)) for i in range(len(ccds))]
    [t.start() for t in ths]
    [t.join() for t in ths]
    report("read-only", res, None)

    print("expert forward (FC1 -> SwiGLU/requant -> FC2), m tokens per expert:")
    for m in ms:
        res = [None] * len(ccds)

        def fwd_worker(i, m=m):
            res[i] = run_jobs(pools[i], arenas[i], args.experts, args.per_job, m, args.repeat, 200 + i, xq, xs, bar)

        ths = [threading.Thread(target=fwd_worker, args=(i,)) for i in range(len(ccds))]
        [t.start() for t in ths]
        [t.join() for t in ths]
        report(f"m={m}", res, m)

    for p in pools:
        p.close()
    for a in arenas:
        a.close()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
