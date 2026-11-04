#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kernel budget of the fast prefill MoE vs b12x's `dynamic` (deterministic) path and the spec on the served dumps
(call1: 4096 rows, 319 experts), one SM120 GPU: per-kernel GPU times from torch.profiler (front-end, FC1, FC2, top-k
sum) for the configurations asked for, b12x's Phase1 / Silu / Phase2 / TopKSum for the same call, the spec's kernels,
and the DRAM floor. Under `ncu` run one config with --iters 1 --no-b12x --no-profile.

  python3 bench_b12x_moe_prefill_fast_sm120.py [--dumps DIR] [--iters 10] [--cfgs 0,0,1 1,0,1 ...] [--no-b12x] [--no-spec]
  (--cfgs = cfg1,cfg2,dynamic tuples; the default list = (3,2)/(4,2) stage configs x dynamic, plus static)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b12x_moe_prefill_spec"))

E, H, N, TOPK, LIMIT = 384, 5120, 576, 6, 10.0


def _profile(torch, fn, iters, tag):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / iters * 1e3
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    agg = {}
    for ev in prof.events():
        if ev.device_type.name != "CUDA":
            continue
        k = ev.name[:100]
        t_, n_ = agg.get(k, (0.0, 0))
        agg[k] = (t_ + ev.device_time, n_ + 1)
    tot = sum(t_ for t_, _ in agg.values()) / iters
    print(f"{tag}: {wall:.3f} ms per call eager (host included), GPU kernels {tot / 1e3:.3f} ms:")
    for k, (t_, n_) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        if t_ / iters >= 1.0:
            print(f"  {t_ / iters / 1e3:7.3f} ms  {n_ / iters:5.1f}x  {k}")
    sys.stdout.flush()
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=os.environ.get("VLLM_MOET_DUMPS", "dumps"),
                    help="directory with the served model's MoE dumps (patch_vllm_b12x_moe_dump_sm120.py); env VLLM_MOET_DUMPS")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cfgs", nargs="*", default=["0,0,1", "1,0,1", "0,1,1", "1,1,1", "0,0,0"],
                    help="cfg1,cfg2,dynamic tuples")
    ap.add_argument("--no-b12x", action="store_true")
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
    import torch
    from b12x_moe_prefill_fast_sm120 import moe_prefill_fast
    from test_b12x_moe_prefill_spec_sm120 import _planes, _served
    import b12x.moe.fused_moe._impl as impl

    dev = torch.device("cuda")
    d = Path(a.dumps)
    raw = torch.load(d / "weights_layer2.pt", map_location="cpu", weights_only=False)
    c = torch.load(d / "call1_layer2.pt", map_location="cpu", weights_only=False)
    x, ids, w = c["hidden_states"].to(dev), c["topk_ids"].to(dev), c["topk_weights"].to(dev)
    out_ref = c["output"].to(dev)
    served = _served(torch, dev, raw)
    T = int(x.shape[0])
    out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    active = int(torch.unique(ids[ids >= 0]).numel())
    w_bytes = active * (2 * N * H // 2 + 2 * N * H // 32 + H * N // 2 + H * N // 32)
    print(f"rows {T} x top-{TOPK} = {T * TOPK} pairs, active experts {active}/{E}; weight bytes touched {w_bytes / 1e6:.0f} MB"
          f" -> DRAM floor {w_bytes / 1.8e12 * 1e6:.0f} us at 1.8 TB/s (FC1 {active * (N * H + N * H // 16) / 1.8e12 * 1e6:.0f},"
          f" FC2 {active * (H * N // 2 + H * N // 32) / 1.8e12 * 1e6:.0f}); route rows {T * TOPK * H * 2 / 1e6:.0f} MB;"
          f" FLOP {2 * T * TOPK * (2 * N * H + H * N) / 1e9:.0f} G")
    spied = {}
    orig = impl._launch_dynamic

    def spy(**kw):
        spied.update(kw)
        return orig(**kw)

    impl._launch_dynamic = spy
    served.run(x, ids, w, out)
    torch.cuda.synchronize()
    planes = _planes(torch, served.prepared, spied)
    print(f"b12x output == served dump: {torch.equal(out.view(torch.int16), out_ref.view(torch.int16))}")
    if not a.no_b12x:
        if a.no_profile:
            pass
        else:
            _profile(torch, lambda: served.run(x, ids, w, out), a.iters, "b12x dynamic (det)")
    if not a.no_spec:
        from b12x_moe_prefill_spec_sm120 import moe_prefill_spec

        if not a.no_profile:
            _profile(torch, lambda: moe_prefill_spec(x, ids, w, planes, LIMIT), max(2, a.iters // 2), "spec B")
    for cfg in a.cfgs:
        c1, c2, dyn = [int(v) for v in cfg.split(",")][:3]
        kw = dict(cfg1=c1, cfg2=c2, dynamic=dyn)
        res = moe_prefill_fast(x, ids, w, planes, LIMIT, **kw)
        torch.cuda.synchronize()
        okb = torch.equal(res[0].view(torch.int16), out.view(torch.int16))
        if a.no_profile:
            for _ in range(2):
                moe_prefill_fast(x, ids, w, planes, LIMIT, **kw)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.iters):
                moe_prefill_fast(x, ids, w, planes, LIMIT, **kw)
            torch.cuda.synchronize()
            print(f"fast cfg1={c1} cfg2={c2} dyn={dyn}: {(time.perf_counter() - t0) / a.iters * 1e3:.3f} ms per call; == b12x: {okb}")
        else:
            _profile(torch, lambda: moe_prefill_fast(x, ids, w, planes, LIMIT, **kw), a.iters,
                     f"fast cfg1={c1} cfg2={c2} dyn={dyn} (output == b12x: {okb})")
    if not a.no_profile:  # the front-end alone: full (quant + routing) and the routing block only
        from b12x_moe_prefill_fast_sm120 import _ext

        ids32 = ids.to(torch.int32).contiguous()
        _profile(torch, lambda: _ext().moe_prefill_fast_frontend(x, ids32, E, False), a.iters, "front-end (quant + routing)")
        _profile(torch, lambda: _ext().moe_prefill_fast_frontend(x, ids32, E, True), a.iters, "front-end (routing block only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
