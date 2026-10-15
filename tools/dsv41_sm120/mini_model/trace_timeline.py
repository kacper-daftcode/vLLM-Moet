#!/usr/bin/env python3
"""Where a torch.profiler trace of a prefill spends its wall time: GPU busy (union over streams), NCCL time and how
much of it ran concurrently with other kernels, the CPU side (op time per thread, launches, how far the GPU runs
behind the host launches), and the handoffs between microbatch threads.

usage: trace_timeline.py TRACE.json[.gz] [--nccl-hist]
Written for the prefill DBO work (docs/dsv41-sm120-port.md, "Dual-batch overlap of the prefill all-reduces"): on the
dummy-weight model the two-microbatch step showed 0-29 ms of 165-176 ms of NCCL overlapped with compute and the
all-reduces spinning for peers whose host threads had not launched yet."""

import gzip
import json
import statistics
import sys
from collections import defaultdict


def union(iv):
    tot = 0
    cs = ce = None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if ce is not None else 0)


def merged(iv):
    out: list[list[float]] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def intersection(a, b):
    a, b = merged(a), merged(b)
    i = j = 0
    tot = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def quantiles(v):
    v = sorted(v)
    if not v:
        return "n=0"
    return f"p10 {v[len(v) // 10] / 1e3:.2f} median {statistics.median(v) / 1e3:.2f} p90 {v[int(len(v) * 0.9)] / 1e3:.2f} ms (n={len(v)})"


def main() -> int:
    path = sys.argv[1]
    with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
        data = json.load(f)
    ev = data["traceEvents"] if isinstance(data, dict) else data
    kern = sorted(
        [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")],
        key=lambda e: e["ts"],
    )
    t0 = min(e["ts"] for e in kern)
    t1 = max(e["ts"] + e["dur"] for e in kern)
    wall = t1 - t0
    span = lambda e: (e["ts"], e["ts"] + e["dur"])  # noqa: E731
    nccl = [e for e in kern if "nccl" in e["name"].lower()]
    other = [e for e in kern if "nccl" not in e["name"].lower()]
    streams: dict = defaultdict(float)
    for e in kern:
        streams[e.get("args", {}).get("stream", "?")] += e["dur"]
    print(
        f"GPU wall {wall / 1e3:.1f} ms; busy (union) {union(map(span, kern)) / 1e3:.1f} ms = "
        f"{union(map(span, kern)) / wall * 100:.0f} %; NCCL {union(map(span, nccl)) / 1e3:.1f} ms of which "
        f"{intersection(list(map(span, nccl)), list(map(span, other))) / 1e3:.1f} ms concurrent with other kernels; "
        f"other kernels {union(map(span, other)) / 1e3:.1f} ms; {len(kern)} kernels; streams: "
        + ", ".join(f"{k}: {v / 1e3:.0f} ms" for k, v in sorted(streams.items(), key=lambda x: -x[1])[:5])
    )
    if "--nccl-hist" in sys.argv and nccl:
        d = sorted(e["dur"] for e in nccl)
        names: dict = defaultdict(int)
        for e in nccl:
            names[e["name"].split("(")[0]] += 1
        print(f"  NCCL kernels: {quantiles(d)}; min {d[0]:.0f} us max {d[-1]:.0f} us; {dict(names)}")
    # CPU side within the GPU window
    cats: dict = defaultdict(list)
    threads: dict = defaultdict(list)
    for e in ev:
        if e.get("ph") != "X" or e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"):
            continue
        s = e["ts"]
        if s < t0 or s > t1:
            continue
        cats[e.get("cat")].append((s, s + e.get("dur", 0)))
        if e.get("cat") in ("cpu_op", "cuda_runtime"):
            threads[(e.get("pid"), e.get("tid"))].append((s, s + e.get("dur", 0)))
    print(
        "CPU events in the window: "
        + ", ".join(f"{k} {union(v) / 1e3:.0f} ms / {len(v)}" for k, v in sorted(cats.items(), key=lambda x: -union(x[1]))[:6])
    )
    for (pid, tid), iv in sorted(threads.items(), key=lambda x: -union(x[1]))[:4]:
        print(f"  thread {pid}/{tid}: ops + runtime busy {union(iv) / 1e3:.0f} ms ({union(iv) / wall * 100:.0f} % of wall), {len(iv)} events")
    rt = {
        e["args"].get("correlation"): e
        for e in ev
        if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})
    }
    lead_nccl, lead_other = [], []
    for e in kern:
        r = rt.get(e.get("args", {}).get("correlation"))
        if r is None:
            continue
        (lead_nccl if "nccl" in e["name"].lower() else lead_other).append(e["ts"] - (r["ts"] + r["dur"]))
    print(f"GPU start after the host launch (the GPU running behind the CPU): other kernels {quantiles(lead_other)}; NCCL {quantiles(lead_nccl)}")
    launches = sorted(
        [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "Launch" in e.get("name", "") and t0 <= e["ts"] <= t1],
        key=lambda e: e["ts"],
    )
    gaps = [b["ts"] - (a["ts"] + a["dur"]) for a, b in zip(launches, launches[1:]) if a["tid"] != b["tid"]]
    if gaps:
        print(f"host thread handoffs: {len(gaps)}, last launch of one thread to the first of the next {quantiles(gaps)}, total {sum(gaps) / 1e3:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
