#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kernel budget of b12x's prefill MoE (`integration.tp_moe.dynamic`, deterministic) on the served dumps, one SM120 GPU:
per-kernel GPU times from torch.profiler (dynamic front-end / routing, Phase1, Silu, Phase2, TopKSum, rows quant...)
and the DRAM floor of the weights touched (bytes of the active experts' FC1 + FC2 planes / 1.8 TB/s). Optionally runs
the spec kernels as the correctness reference. Under `ncu` (see L/win/moe_ncu.sh) run with --iters 1 --no-profile.

  python3 bench_b12x_prefill_sm120.py [--dumps DIR] [--iters 5] [--no-profile] [--spec]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

E, H, N, TOPK, LIMIT = 384, 5120, 576, 6, 10.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="/workspace/tmp/par31/L/win/dumps")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--spec", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
    import torch
    from test_b12x_moe_prefill_spec_sm120 import _served

    dev = torch.device("cuda")
    d = Path(a.dumps)
    raw = torch.load(d / "weights_layer2.pt", map_location="cpu", weights_only=False)
    c = torch.load(d / "call1_layer2.pt", map_location="cpu", weights_only=False)
    x, ids, w = c["hidden_states"].to(dev), c["topk_ids"].to(dev), c["topk_weights"].to(dev)
    served = _served(torch, dev, raw)
    T = int(x.shape[0])
    out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    active = int(torch.unique(ids[ids >= 0]).numel())
    w_bytes = active * (2 * N * H // 2 + 2 * N * H // 32 + H * N // 2 + H * N // 32)
    print(f"rows {T} x top-{TOPK} = {T * TOPK} pairs, active experts {active}/{E}; weight bytes touched {w_bytes / 1e6:.0f} MB"
          f" -> DRAM floor {w_bytes / 1.8e12 * 1e6:.0f} us at 1.8 TB/s; FLOP {2 * T * TOPK * (2 * N * H + H * N) / 1e9:.0f} G")
    for _ in range(2):
        served.run(x, ids, w, out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        served.run(x, ids, w, out)
    torch.cuda.synchronize()
    print(f"b12x dynamic (eager, host included): {(time.perf_counter() - t0) / a.iters * 1e3:.2f} ms per call")
    if not a.no_profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(a.iters):
                served.run(x, ids, w, out)
            torch.cuda.synchronize()
        agg = {}
        for ev in prof.events():
            if ev.device_type.name != "CUDA":
                continue
            k = ev.name[:90]
            t_, n_ = agg.get(k, (0.0, 0))
            agg[k] = (t_ + ev.device_time, n_ + 1)
        tot = sum(t_ for t_, _ in agg.values()) / a.iters
        print(f"GPU kernels per call: {tot / 1e3:.2f} ms")
        for k, (t_, n_) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            print(f"  {t_ / a.iters / 1e3:7.3f} ms  {n_ // a.iters:3d}x  {k}")
    if a.spec:
        from b12x_moe_prefill_spec_sm120 import moe_prefill_spec
        from test_b12x_moe_prefill_spec_sm120 import _planes
        import b12x.moe.fused_moe._impl as impl

        spied = {}
        orig = impl._launch_dynamic

        def spy(**kw):
            spied.update(kw)
            return orig(**kw)

        impl._launch_dynamic = spy
        served.run(x, ids, w, out)
        planes = _planes(torch, served.prepared, spied)
        res = moe_prefill_spec(x, ids, w, planes, LIMIT)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max(1, a.iters // 2)):
            res = moe_prefill_spec(x, ids, w, planes, LIMIT)
        torch.cuda.synchronize()
        print(f"spec kernels: {(time.perf_counter() - t0) / max(1, a.iters // 2) * 1e3:.1f} ms per call; output == b12x: "
              f"{torch.equal(res[0].view(torch.int16), out.view(torch.int16))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
