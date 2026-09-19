#!/usr/bin/env python3
"""Per-CUDA-graph kernel breakdown of a torch profiler trace.

Kernels launched from one `cudaGraphLaunch` share its correlation id; this groups the GPU
kernels by launch, clusters the launches by kernel count (the decode graph, the drafter graph,
prefill pieces, ...) and prints, for each cluster, the kernel composition of one typical
launch: count, total and average time per kernel name, in launch order when --order is given.

usage: trace_graphs.py TRACE.json.gz [--min-kernels 20] [--top 40] [--order] [--graph N]
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--min-kernels", type=int, default=20)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--order", action="store_true", help="print the kernels of one launch in order")
    ap.add_argument("--graph", type=int, default=None, help="only the cluster with this kernel count")
    args = ap.parse_args()
    with gzip.open(args.trace, "rt") as f:
        ev = json.load(f)["traceEvents"]
    kernels = [e for e in ev if e.get("cat") == "kernel"]
    launches = {e["args"]["correlation"]: e for e in ev
                if e.get("cat") == "cuda_runtime" and "GraphLaunch" in e.get("name", "")}
    by_corr: dict[int, list] = defaultdict(list)
    for k in kernels:
        c = k.get("args", {}).get("correlation")
        if c in launches:
            by_corr[c].append(k)
    print(f"{len(kernels)} kernels, {len(launches)} graph launches, {sum(len(v) for v in by_corr.values())} kernels inside graphs")
    clusters: dict[int, list[int]] = defaultdict(list)
    for c, ks in by_corr.items():
        clusters[len(ks)].append(c)
    rows = sorted(clusters.items(), key=lambda kv: -len(kv[1]) * kv[0])
    print("\nclusters (kernels per launch -> launches, median span us):")
    for n, cs in rows:
        if n < args.min_kernels:
            continue
        spans = sorted(max(k["ts"] + k["dur"] for k in by_corr[c]) - min(k["ts"] for k in by_corr[c]) for c in cs)
        tot = sorted(sum(k["dur"] for k in by_corr[c]) for c in cs)
        print(f"  {n:5d} kernels x {len(cs):4d} launches   span {spans[len(spans)//2]:8.1f} us   kernel sum {tot[len(tot)//2]:8.1f} us")
    for n, cs in rows:
        if n < args.min_kernels or (args.graph is not None and n != args.graph):
            continue
        if args.graph is None and len(cs) < 3:
            continue
        # pick the median-span launch as the representative
        cs_sorted = sorted(cs, key=lambda c: max(k["ts"] + k["dur"] for k in by_corr[c]) - min(k["ts"] for k in by_corr[c]))
        rep = cs_sorted[len(cs_sorted) // 2]
        ks = sorted(by_corr[rep], key=lambda k: k["ts"])
        span = ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]
        ksum = sum(k["dur"] for k in ks)
        gaps = [ks[i + 1]["ts"] - (ks[i]["ts"] + ks[i]["dur"]) for i in range(len(ks) - 1)]
        print(f"\n=== graph with {n} kernels ({len(cs)} launches): representative span {span:.1f} us, kernel sum {ksum:.1f} us, "
              f"gaps sum {sum(g for g in gaps if g > 0):.1f} us (median gap {sorted(gaps)[len(gaps)//2]:.2f} us)")
        agg: dict[str, list[float]] = defaultdict(list)
        for k in ks:
            agg[k["name"]].append(k["dur"])
        print(f"  {'count':>5} {'total us':>9} {'avg us':>7}  kernel")
        for name, ds in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[: args.top]:
            print(f"  {len(ds):5d} {sum(ds):9.1f} {sum(ds)/len(ds):7.1f}  {name[:120]}")
        if args.order:
            print("  --- launch order (dur us, grid, name) ---")
            for k in ks:
                a = k.get("args", {})
                print(f"  {k['dur']:7.1f} {str(a.get('grid')):>16} {str(a.get('block')):>14}  {k['name'][:100]}")


if __name__ == "__main__":
    main()
