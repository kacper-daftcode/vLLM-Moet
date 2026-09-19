#!/usr/bin/env python3
"""Detail of GPU idle gaps (>gap_us) in one rank trace: what ran before/after, correlation ids,
whether the gap is inside a graph launch, and per-graph internal gaps.

usage: gap_detail.py TRACE.json.gz [gap_us] [n]
"""
import gzip
import json
import statistics as st
import sys
from collections import defaultdict

path = sys.argv[1]
gap_us = float(sys.argv[2]) if len(sys.argv) > 2 else 150
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
with gzip.open(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"] if isinstance(data, dict) else data
X = [e for e in ev if e.get("ph") == "X"]
kern = sorted((e for e in X if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}), key=lambda e: e["ts"])
rt = {(e.get("args") or {}).get("correlation"): e for e in X if e.get("cat") == "cuda_runtime"}
T0 = kern[0]["ts"]
streams = defaultdict(int)
for e in kern:
    streams[(e.get("args") or {}).get("stream")] += 1
print("kernel streams:", dict(streams))

gaps = []
last = kern[0]
last_end = last["ts"] + last["dur"]
for e in kern[1:]:
    if e["ts"] - last_end > gap_us:
        gaps.append((last_end - T0, e["ts"] - last_end, last, e))
    if e["ts"] + e["dur"] > last_end:
        last_end = e["ts"] + e["dur"]
        last = e
print(f"{len(gaps)} gaps > {gap_us} us; median {st.median([g[1] for g in gaps]):.0f} us")
# classify gaps by (before-name, after-name)
cls = defaultdict(list)
for at, g, a, b in gaps:
    cls[(a["name"][:50], b["name"][:50])].append(g)
print("--- gap classes (before -> after): count, median us ---")
for k, v in sorted(cls.items(), key=lambda x: -sum(x[1]))[:12]:
    print(f"  {len(v):4d}x med {st.median(v):7.0f} us  sum {sum(v)/1000:7.1f} ms  {k[0]}  ->  {k[1]}")
print(f"--- first {n} gaps ---")
for at, g, a, b in gaps[3:3 + n]:
    ca = (a.get("args") or {}).get("correlation")
    cb = (b.get("args") or {}).get("correlation")
    la = rt.get(ca, {}).get("name")
    lb = rt.get(cb, {}).get("name")
    print(f"at {at/1000:8.2f} ms gap {g:6.0f} us | before: {a['name'][:45]} (corr {ca}, {la}, stream {(a.get('args') or {}).get('stream')}) | after: {b['name'][:45]} (corr {cb}, {lb}, stream {(b.get('args') or {}).get('stream')})")
    if ca == cb:
        print("     -> same launch (gap inside a graph)")
    else:
        # CPU launch times of both
        for c, l in ((ca, "before"), (cb, "after")):
            r = rt.get(c)
            if r:
                print(f"     {l}: launched on CPU at {(r['ts']-T0)/1000:8.2f} ms ({r['name']}, cpu dur {r['dur']:.0f} us)")
