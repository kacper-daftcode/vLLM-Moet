#!/usr/bin/env python3
"""Per-step anatomy of a Ripple decode trace (torch profiler, one rank). Steps are cut at the last kernel of
the step (default `_post_update_kernel`, the DFlash2 drafter's tail); for the middle steps it prints the mean per-step
time by class, the GPU idle inside the step, and (with --dump N) the ordered kernel list of step N collapsed into runs.

usage: step_anatomy.py TRACE.json.gz   (a rank trace from PROFILER=1 / /start_profile .. /stop_profile) [--marker NAME] [--dump N] [--classes]
"""
import argparse
import gzip
import json
import re
from collections import defaultdict

CLASSES = [
    ("P2P all-reduce (ours)", r"push_twoshot|reduce_slice|assemble_kernel|push_kernel|wait_reduce|push_oneshot|consume"),
    ("NCCL", r"nccl"),
    ("NVFP4 GEMM (CUTLASS)", r"cutlass13device_kernel|GemmUniversal|Sm120|fp4_mm|nvfp4"),
    ("BF16 GEMM (cuBLAS/cutlass_80)", r"cutlass_80|nvjet|splitKreduce|gemv|cublas|Cijk"),
    ("attention (FA2 splitkv)", r"flash_fwd|flash::"),
    ("KV write", r"reshape_and_cache"),
    ("fp4 input quant", r"cvt_fp16_to_fp4|scaled_fp4_quant"),
    ("norm / gating / residual (triton compiled)", r"triton_red|triton_per|triton_poi|rms_norm|layernorm"),
    ("drafter glue (dflash2)", r"dflash|selector|resample|rejection|post_update|local_logits|grouped_conv"),
    ("sampler / topk", r"sampl|RadixTopK|topk|TopK|argmax|softmax"),
    ("memcpy / memset", r"Memcpy|Memset|memcpy|memset"),
    ("elementwise (aten)", r"elementwise|vectorized|index|copy_|fill|cat_|reduce_kernel|Functor|unrolled"),
]


def classify(name):
    for c, pat in CLASSES:
        if re.search(pat, name):
            return c
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--marker", default="_post_update_kernel")
    ap.add_argument("--dump", type=int, default=-1)
    ap.add_argument("--classes", action="store_true")
    a = ap.parse_args()
    with gzip.open(a.trace, "rt") as f:
        ev = json.load(f)["traceEvents"]
    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    kern.sort(key=lambda e: e["ts"])
    marks = [i for i, e in enumerate(kern) if a.marker in e["name"]]
    if len(marks) < 4:
        print(f"only {len(marks)} markers '{a.marker}'"); return
    steps = []
    for s, e in zip(marks[:-1], marks[1:]):
        steps.append(kern[s + 1:e + 1])
    print(f"{len(steps)} steps between markers (window {kern[0]['ts']:.0f}..{kern[-1]['ts'] + kern[-1]['dur']:.0f} us)")
    # middle steps only (skip first / last which include prefill or tail)
    mid = steps[2:-1] if len(steps) > 4 else steps
    tot = defaultdict(float); cnt = defaultdict(int); walls = []; idles = []; busies = []
    for st in mid:
        t0 = st[0]["ts"]; t1 = max(e["ts"] + e["dur"] for e in st)
        walls.append(t1 - t0)
        # busy = union of intervals
        iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in st)
        busy = 0.0; cs, ce = iv[0]
        for s_, e_ in iv[1:]:
            if s_ > ce:
                busy += ce - cs; cs, ce = s_, e_
            else:
                ce = max(ce, e_)
        busy += ce - cs
        busies.append(busy); idles.append((t1 - t0) - busy)
        for e in st:
            c = classify(e["name"]); tot[c] += e["dur"]; cnt[c] += 1
    n = len(mid)
    print(f"mean step wall {sum(walls)/n:.0f} us (min {min(walls):.0f}, max {max(walls):.0f}); GPU busy {sum(busies)/n:.0f} us; idle {sum(idles)/n:.0f} us; kernels/step {sum(cnt.values())/n:.0f}")
    print("--- per-step mean by class (us, launches) ---")
    for c, t in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"  {t/n:8.0f} us  {cnt[c]/n:6.0f} launches  avg {t/cnt[c]:6.1f} us  {c}")
    if a.dump >= 0:
        st = steps[a.dump]
        print(f"--- step {a.dump}: {len(st)} kernels, ordered, collapsed into runs ---")
        runs = []
        for e in st:
            key = re.sub(r"\(.*", "", e["name"])[:110]
            if runs and runs[-1][0] == key:
                runs[-1][1] += 1; runs[-1][2] += e["dur"]
            else:
                runs.append([key, 1, e["dur"], e["ts"]])
        prev_end = st[0]["ts"]
        for key, c, d, ts in runs:
            gap = ts - prev_end
            print(f"  +{ts - st[0]['ts']:8.0f} us  gap {gap:6.1f}  {c:3d}x {d:8.1f} us  {key}")
            prev_end = ts + d  # approximate (runs are consecutive)
    if a.classes:
        names = defaultdict(lambda: [0.0, 0])
        for st in mid:
            for e in st:
                names[e["name"][:150]][0] += e["dur"]; names[e["name"][:150]][1] += 1
        print("--- per-step mean by kernel name ---")
        for k, (t, c) in sorted(names.items(), key=lambda x: -x[1][0])[:40]:
            print(f"  {t/n:8.1f} us  {c/n:6.1f}x  avg {t/c:6.1f}  {k}")


if __name__ == "__main__":
    main()
