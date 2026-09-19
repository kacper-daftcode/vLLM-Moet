#!/usr/bin/env python3
"""Allreduce duration distribution per rank trace: median/p90/p99, count and sum of the long
(>100 us) waits, what ran before them, and the NCCL sum per 116 consecutive allreduces (one qwen38
step) in the middle of the window.

usage: trace_nccl_dist.py label=GLOB [label=GLOB ...]
"""
import glob, gzip, json, statistics as st, sys
from collections import Counter
for label, pat in [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[1:]]:
    p = glob.glob(pat)[0]
    ev = json.load(gzip.open(p, "rt"))["traceEvents"]
    kern = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"), key=lambda e: e["ts"])
    ar = [e for e in kern if "AllReduce" in e["name"]]
    d = sorted(e["dur"] for e in ar)
    n = len(d)
    big = [x for x in d if x > 100]
    print(f"{label}: {n} allreduces; median {st.median(d):.1f} p75 {d[int(.75*n)]:.1f} p90 {d[int(.9*n)]:.1f} p99 {d[int(.99*n)]:.1f} max {d[-1]:.0f} us; "
          f">100us: {len(big)} (sum {sum(big)/1000:.1f} ms of total {sum(d)/1000:.1f} ms); >40us: {len([x for x in d if x>40])}")
    # what precedes the long ones?
    prev = Counter()
    for i, e in enumerate(kern):
        if "AllReduce" in e["name"] and e["dur"] > 100 and i > 0:
            prev[kern[i-1]["name"][:60]] += 1
    print("   kernels before >100us allreduces:", prev.most_common(5))
    # per-step (116 AR) sums for a middle window of 20 steps
    if n > 116 * 30:
        s = n // 2 - 116 * 10
        sums = [sum(e["dur"] for e in ar[s + i*116: s + (i+1)*116]) for i in range(20)]
        print(f"   NCCL sum per 116 allreduces (20 mid steps): median {st.median(sums)/1000:.2f} ms, min {min(sums)/1000:.2f}, max {max(sums)/1000:.2f}")
