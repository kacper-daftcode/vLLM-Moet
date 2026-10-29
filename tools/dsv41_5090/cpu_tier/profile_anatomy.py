"""The anatomy of the hybrid's decode step from a torch-profiler chrome trace of one TP rank (vLLM's
--profiler-config.profiler=torch, /start_profile {"max_iterations": N} ... /stop_profile; the worker writes
<prefix>_dp0_pp0_tp<r>.pt.trace.json[.gz]). Prints: the GPU-busy share of the profiled span (the union of the kernel
/ memcpy intervals over all streams), the per-step wall and GPU time when the step count is given, the top kernels by
total time, and the groups the hybrid cares about: our publish / wait-inject kernels, b12x's MoE kernels, NCCL, the
attention / MLA side, the Engram, GEMVs. Steps are found from the publish kernels (one per MoE layer per step).

    python profile_anatomy.py TRACE[.gz] [--steps N] [--top 25] [--layers 40]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict

GROUPS = [
    ("hybrid publish", re.compile(r"publish", re.I)),
    ("hybrid wait+inject", re.compile(r"wait_inject|inject", re.I)),
    ("hybrid repack/promote", re.compile(r"repack|promote|table_write", re.I)),
    ("MoE b12x", re.compile(r"b12x|w4a8|moe|TopKSum|topk_sum|route|expert", re.I)),
    ("NCCL", re.compile(r"nccl", re.I)),
    # vLLM's other all-reduce backends: FlashInfer PCIe IPC (ipc_rsag_* / ipc_topo_* / ipc_tp2_* / push_oneshot_* /
    # ce_add / ce_publish_flag / ce_wait_flag / kernel_pcie_ipc_ce_*), the image's sm_120 P2P push (push_phased_* /
    # wait_reduce / push_twoshot / reduce_slice / assemble / wait_gather), vLLM's custom one-shot (cross_device_reduce)
    ("all-reduce (non-NCCL)", re.compile(r"all_?reduce|pcie_?ipc|one_?shot|two_?shot|cross_device_reduce|ipc_rsag|ipc_topo|"
                                         r"ipc_tp2|ce_add_kernel|ce_publish_flag|ce_wait_flag|push_phased|wait_reduce|"
                                         r"reduce_slice_kernel|assemble_kernel|wait_gather", re.I)),
    ("attention / MLA / indexer", re.compile(r"mla|attn|attention|indexer|sparse|flash|paged|fmha|topk|dsa", re.I)),
    ("Engram", re.compile(r"engram|ngram|hash", re.I)),
    ("GEMV / GEMM", re.compile(r"gemv|gemm|cutlass|cublas|Kernel.*sm120|matmul", re.I)),
    ("norm / elementwise", re.compile(r"norm|elementwise|vectorized|fill|copy|cast|silu|add|mul|scale|quant|rotary|rope", re.I)),
    ("memcpy / memset", re.compile(r"memcpy|memset", re.I)),
]


def load(path: str) -> list[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def union_len(intervals: list[tuple[float, float]]) -> float:
    intervals.sort()
    total, cur_s, cur_e = 0.0, None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--steps", type=int, default=0, help="profiled decode steps (max_iterations); 0 = from the publish kernels")
    ap.add_argument("--layers", type=int, default=40, help="MoE layers per step (publish kernels per step)")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    ev = load(args.trace)
    gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation")
           and e.get("cat") != "gpu_user_annotation"]
    cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "python_function")]
    if not gpu:
        print("no GPU events in the trace (CUPTI activities missing?)"); return 1
    t0 = min(e["ts"] for e in gpu)
    t1 = max(e["ts"] + e["dur"] for e in gpu)
    span = t1 - t0
    busy = union_len([(e["ts"], e["ts"] + e["dur"]) for e in gpu])
    streams = defaultdict(float)
    for e in gpu:
        streams[e.get("args", {}).get("stream", "?")] += e["dur"]
    print(f"trace {args.trace}: {len(gpu)} GPU events, {len(cpu)} CPU events; GPU span {span / 1e3:.1f} ms, GPU busy (union over "
          f"streams) {busy / 1e3:.1f} ms = {100 * busy / max(span, 1e-9):.1f} %; per stream (sum of durations): "
          + ", ".join(f"s{k}: {v / 1e3:.1f} ms" for k, v in sorted(streams.items(), key=lambda kv: -kv[1])[:6]))
    # steps from the publish kernels
    pubs = sorted(e["ts"] for e in gpu if re.search(r"publish", e["name"], re.I))
    n_steps = args.steps or (len(pubs) // args.layers if pubs else 0)
    if pubs and n_steps:
        # step boundaries: the first publish of every group of `layers` publishes
        starts = [pubs[i * args.layers] for i in range(len(pubs) // args.layers)]
        if len(starts) >= 2:
            periods = [b - a for a, b in zip(starts, starts[1:])]
            periods.sort()
            med = periods[len(periods) // 2]
            print(f"steps: {len(pubs)} publish kernels = {len(pubs) / args.layers:.1f} steps; step period median {med / 1e3:.2f} ms "
                  f"(min {periods[0] / 1e3:.2f}, max {periods[-1] / 1e3:.2f}); GPU busy per step ~{busy / max(1, len(starts)) / 1e3:.2f} ms "
                  f"-> GPU idle ~{(med - busy / max(1, len(starts))) / 1e3:.2f} ms per step (if the busy time is spread evenly)")
            # a typical step: the GPU busy inside [start_k, start_k+1) for the median step
            k = len(periods) // 2
            seg = [(max(e["ts"], starts[k]), min(e["ts"] + e["dur"], starts[k + 1])) for e in gpu
                   if e["ts"] < starts[k + 1] and e["ts"] + e["dur"] > starts[k]]
            seg = [(s, e) for s, e in seg if e > s]
            print(f"  step {k}: wall {(starts[k + 1] - starts[k]) / 1e3:.2f} ms, GPU busy {union_len(seg) / 1e3:.2f} ms, "
                  f"{len(seg)} kernels")
            # the biggest idle gaps inside that step (the GPU waits for the host: our wait kernel spins count as busy)
            seg.sort()
            gaps = []
            cur = starts[k]
            for s, e in seg:
                if s > cur:
                    gaps.append((s - cur, cur - starts[k]))
                cur = max(cur, e)
            if starts[k + 1] > cur:
                gaps.append((starts[k + 1] - cur, cur - starts[k]))   # the tail: from the last kernel to the next step
            gaps.sort(reverse=True)
            print("  largest GPU gaps in that step (ms, at ms into the step): " + ", ".join(f"{g / 1e3:.2f}@{at / 1e3:.2f}" for g, at in gaps[:6]))
    else:
        print("no publish kernels found (not a hybrid trace, or the names differ) - no step split")
    # kernels by name
    by_name = defaultdict(lambda: [0.0, 0])
    for e in gpu:
        by_name[e["name"]][0] += e["dur"]
        by_name[e["name"]][1] += 1
    per_step = max(1, n_steps)
    print(f"\ntop {args.top} GPU kernels by total time (per step = total / {per_step}):")
    for name, (dur, cnt) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"  {dur / 1e3 / per_step:7.3f} ms/step  {cnt / per_step:6.1f}/step  {dur / cnt:7.1f} us each  {name[:110]}")
    print("\ngroups (first matching group wins; per step):")
    grp = defaultdict(lambda: [0.0, 0])
    for name, (dur, cnt) in by_name.items():
        g = next((gname for gname, rx in GROUPS if rx.search(name)), "other")
        grp[g][0] += dur
        grp[g][1] += cnt
    for g, (dur, cnt) in sorted(grp.items(), key=lambda kv: -kv[1][0]):
        print(f"  {dur / 1e3 / per_step:7.3f} ms/step  {cnt / per_step:7.1f} kernels/step  {g}")
    # CPU side: the biggest cpu ops / annotations per step
    if cpu:
        by_cpu = defaultdict(lambda: [0.0, 0])
        for e in cpu:
            by_cpu[e["name"]][0] += e["dur"]
            by_cpu[e["name"]][1] += 1
        print(f"\ntop CPU ops / annotations by total time (per step):")
        for name, (dur, cnt) in sorted(by_cpu.items(), key=lambda kv: -kv[1][0])[:15]:
            print(f"  {dur / 1e3 / per_step:7.3f} ms/step  {cnt / per_step:6.1f}/step  {name[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
