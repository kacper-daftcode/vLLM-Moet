#!/usr/bin/env python3
"""Context around GPU idle gaps of a given class (before-kernel name prefix).

usage: gap_ctx.py TRACE.json.gz BEFORE_PREFIX [gap_us] [n_gaps] [n_ctx]
"""
import gzip
import json
import sys
from collections import defaultdict

path, before_prefix = sys.argv[1], sys.argv[2]
gap_us = float(sys.argv[3]) if len(sys.argv) > 3 else 150
n_gaps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
n_ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 6
with gzip.open(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"] if isinstance(data, dict) else data
X = [e for e in ev if e.get("ph") == "X"]
kern = sorted((e for e in X if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}), key=lambda e: e["ts"])
rt_all = sorted((e for e in X if e.get("cat") == "cuda_runtime"), key=lambda e: e["ts"])
rt = {(e.get("args") or {}).get("correlation"): e for e in rt_all}
cpu = sorted((e for e in X if e.get("cat") in {"cpu_op", "user_annotation"}), key=lambda e: e["ts"])
gpu_ann = sorted((e for e in X if e.get("cat") == "gpu_user_annotation"), key=lambda e: e["ts"])
T0 = kern[0]["ts"]


def corr(e):
    return (e.get("args") or {}).get("correlation")


def strm(e):
    return (e.get("args") or {}).get("stream")


found = 0
last_end = kern[0]["ts"] + kern[0]["dur"]
last_i = 0
for i in range(1, len(kern)):
    e = kern[i]
    if e["ts"] - last_end > gap_us and kern[last_i]["name"].startswith(before_prefix):
        found += 1
        if found > 3 + n_gaps:
            break
        if found > 3:
            g0, g1 = last_end, e["ts"]
            print(f"\n===== gap {g1-g0:.0f} us at {(g0-T0)/1000:.2f} ms =====")
            print("GPU kernels around the gap (ts rel gap start, dur, stream, corr, launch api @cpu ts):")
            for k in kern[max(0, last_i - n_ctx):last_i + 1] + kern[i:i + n_ctx]:
                r = rt.get(corr(k))
                print(f"  {k['ts']-g0:9.0f} +{k['dur']:6.0f}  s{strm(k):<6} c{corr(k):<7} {k['name'][:58]:58s} {r['name'] if r else '?':18s} @{(r['ts']-g0)/1000 if r else float('nan'):9.2f} ms")
            print("CPU runtime calls in [gap-0.2ms, gap end+0.1ms]:")
            for r in rt_all:
                if g0 - 200 <= r["ts"] <= g1 + 100:
                    print(f"  {r['ts']-g0:9.0f} +{r['dur']:6.0f}  tid {r['tid']} {r['name']} c{corr(r)}")
            print("top-level CPU ops overlapping the gap window (dur >= 5 us):")
            cur_end = defaultdict(lambda: -1)
            for o in cpu:
                if o["ts"] + o["dur"] < g0 - 200 or o["ts"] > g1 + 100:
                    continue
                if o["ts"] < cur_end[o["tid"]]:
                    continue
                cur_end[o["tid"]] = o["ts"] + o["dur"]
                if o["dur"] >= 5:
                    print(f"  {o['ts']-g0:9.0f} +{o['dur']:6.0f}  tid {o['tid']} {o['name'][:100]}")
            print("GPU annotations overlapping:")
            for a in gpu_ann:
                if a["ts"] <= g1 and a["ts"] + a["dur"] >= g0:
                    print(f"  {a['ts']-g0:9.0f} +{a['dur']:6.0f}  s{a.get('tid')} {a['name'][:80]}")
    if e["ts"] + e["dur"] > last_end:
        last_end = e["ts"] + e["dur"]
        last_i = i
