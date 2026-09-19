#!/usr/bin/env python3
"""NCCL allreduce anatomy from torch.profiler traces of all TP ranks.

usage: ar_skew.py DIR [gap_us] [min_kernels]

Steps = GPU kernel runs separated by idle gaps > gap_us and holding >= min_kernels
kernels. Per rank: first-NCCL-of-step duration vs the rest, idle gap before the step.
Across ranks: steps aligned by start time; per step the launch skew (max-min of the
first kernel start) and the first-NCCL duration on each rank.
"""
import glob
import gzip
import json
import os
import re
import statistics as st
import sys

d = sys.argv[1]
gap_us = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
min_k = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
files = sorted(glob.glob(os.path.join(d, "dp0_pp0_tp*_rank*.pt.trace.json.gz")))
assert files, "no traces"
NCCL = re.compile(r"nccl|AllReduce|ncclDevKernel", re.I)


def load(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    ev = data["traceEvents"] if isinstance(data, dict) else data
    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}]
    kern.sort(key=lambda e: e["ts"])
    return kern


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def segments(kern):
    segs, cur, last_end = [], [], None
    for e in kern:
        if last_end is not None and e["ts"] - last_end > gap_us:
            if cur:
                segs.append(cur)
            cur = []
        cur.append(e)
        last_end = max(last_end or 0, e["ts"] + e["dur"])
    if cur:
        segs.append(cur)
    return segs


rank_steps = []
for path in files:
    kern = load(path)
    segs = segments(kern)
    counts = sorted(len(s) for s in segs)
    big = [s for s in segs if len(s) >= min_k]
    name = os.path.basename(path).split(".")[0]
    print(f"{name}: {len(segs)} segments (kernel-count p50 {counts[len(counts)//2]}, max {counts[-1]}), {len(big)} steps >= {min_k} kernels")
    info = []
    prev_end = None
    for s in big:
        ars = [e for e in s if NCCL.search(e["name"])]
        start = s[0]["ts"]
        end = max(e["ts"] + e["dur"] for e in s)
        info.append(dict(start=start, end=end, n=len(s), n_ar=len(ars),
                         first_ar=ars[0]["dur"] if ars else float("nan"),
                         first_ar_ts=ars[0]["ts"] if ars else float("nan"),
                         rest=[e["dur"] for e in ars[1:]],
                         ar_sum=sum(e["dur"] for e in ars),
                         idle_before=(start - prev_end) if prev_end is not None else float("nan"),
                         busy=sum(e["dur"] for e in s)))
        prev_end = end
    if info:
        rest = [x for i in info for x in i["rest"]]
        idle = [i["idle_before"] for i in info[1:]]
        print(f"   NCCL/step {st.median([i['n_ar'] for i in info]):.0f}; step span median {st.median([i['end']-i['start'] for i in info]):.0f} us, "
              f"kernel busy median {st.median([i['busy'] for i in info]):.0f} us, idle before step median {st.median(idle):.0f} us (p90 {pct(idle,90):.0f})")
        print(f"   first NCCL of step: median {st.median([i['first_ar'] for i in info]):.1f} us, p10 {pct([i['first_ar'] for i in info],10):.1f}, p90 {pct([i['first_ar'] for i in info],90):.1f}")
        print(f"   rest NCCL: median {st.median(rest):.1f} us, mean {st.mean(rest):.1f}, p90 {pct(rest,90):.1f}, p99 {pct(rest,99):.1f}; NCCL sum/step median {st.median([i['ar_sum'] for i in info]):.0f} us")
    rank_steps.append(info)

# align steps across ranks by start time (within half a step)
if len(rank_steps) > 1 and all(rank_steps):
    base = rank_steps[0]
    span = st.median([i["end"] - i["start"] for i in base])
    aligned = []
    for i in base:
        row = [i]
        ok = True
        for other in rank_steps[1:]:
            m = min(other, key=lambda j: abs(j["start"] - i["start"]))
            if abs(m["start"] - i["start"]) > span / 2:
                ok = False
                break
            row.append(m)
        if ok:
            aligned.append(row)
    print(f"\ncross-rank: {len(aligned)} aligned steps")
    launch_skew = [max(r["start"] for r in row) - min(r["start"] for r in row) for row in aligned]
    first_ar_min = [min(r["first_ar"] for r in row) for row in aligned]
    first_ar_max = [max(r["first_ar"] for r in row) for row in aligned]
    end_skew = [max(r["end"] for r in row) - min(r["end"] for r in row) for row in aligned]
    print(f"   step start skew (first kernel): median {st.median(launch_skew):.0f} us, p90 {pct(launch_skew,90):.0f}, max {max(launch_skew):.0f}")
    print(f"   first NCCL dur: min-over-ranks median {st.median(first_ar_min):.1f} us; max-over-ranks median {st.median(first_ar_max):.1f} us")
    print(f"   step end skew: median {st.median(end_skew):.0f} us")
    from collections import Counter
    last = Counter(max(range(len(row)), key=lambda r: row[r]["start"]) for row in aligned)
    first = Counter(min(range(len(row)), key=lambda r: row[r]["start"]) for row in aligned)
    print(f"   last-to-start rank histogram {dict(sorted(last.items()))}; first-to-start {dict(sorted(first.items()))}")
    print("   sample steps (start offset us per rank | first NCCL dur per rank):")
    for row in aligned[5:13]:
        t0 = min(r["start"] for r in row)
        print("     " + " ".join(f"{r['start']-t0:6.0f}" for r in row) + " | " + " ".join(f"{r['first_ar']:7.1f}" for r in row))
