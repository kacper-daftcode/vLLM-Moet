#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bit test + timing: FlashInfer mm_mxfp8 backend="b12x" vs backend="cutlass" (the served sm_120 path) on the served GEMM
shapes and a sweep of row counts (decode and prefill sizes, multiples and non-multiples of the 128-row tile, odd M).
PASS = torch.equal on every (shape, M). usage: python3 test_mxfp8_gemm_b12x_sm120.py [--json out] [--quick]"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch


def time_us(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); t.append(s.elapsed_time(e) * 1000.0)
    t.sort()
    return t[len(t) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    from flashinfer import mm_mxfp8, mxfp8_quantize
    dev = torch.device("cuda")
    torch.manual_seed(1234)
    # (K, N): the two MXFP8 sites of the served rank (autotune log) + the per-rank projections of DeepSeek-V4.1-Flash at TP4
    shapes = [(6144, 6400), (1280, 4096), (5120, 5120), (2048, 5120), (5120, 1792), (1024, 5120), (5120, 4096)]
    ms = [1, 2, 8, 16, 17, 56, 64, 100, 127, 128, 129, 255, 256, 1000, 1663, 4092, 4096]
    if a.quick:
        shapes, ms = shapes[:2], [8, 129, 4092, 4096]
    rows, fails, errors = [], 0, 0
    for k, n in shapes:
        w = (torch.randn(n, k, device=dev, dtype=torch.bfloat16) / 8).clamp(-2, 2)
        w_q, w_s = mxfp8_quantize(w, is_sf_swizzled_layout=True)
        for m in ms:
            x = (torch.randn(m, k, device=dev, dtype=torch.bfloat16) / 8).clamp(-2, 2)
            if m % 7 == 3:  # some rows with large magnitudes / zeros / exact powers of two
                x[0] = 0; x[-1] = 2.0; x[m // 2] = x[m // 2] * 64
            a_q, a_s = mxfp8_quantize(x, is_sf_swizzled_layout=True)
            r = dict(k=k, n=n, m=m)
            try:
                ref = mm_mxfp8(a_q, w_q.t(), a_s, w_s, out_dtype=torch.bfloat16, backend="cutlass")
                out = mm_mxfp8(a_q, w_q.t(), a_s, w_s, out_dtype=torch.bfloat16, backend="b12x")
                torch.cuda.synchronize()
                r["equal"] = bool(torch.equal(out, ref))
                if not r["equal"]:
                    d = (out.float() - ref.float()).abs()
                    r["maxabs"] = d.max().item(); r["n_diff"] = int((d > 0).sum().item()); fails += 1
                if m in (8, 128, 1663, 4096):
                    r["cutlass_us"] = time_us(lambda: mm_mxfp8(a_q, w_q.t(), a_s, w_s, out_dtype=torch.bfloat16, backend="cutlass"))
                    r["b12x_us"] = time_us(lambda: mm_mxfp8(a_q, w_q.t(), a_s, w_s, out_dtype=torch.bfloat16, backend="b12x"))
            except Exception as exc:  # noqa: BLE001
                r["error"] = repr(exc)[:200]; errors += 1
            rows.append(r)
            print("  " + json.dumps(r)); sys.stdout.flush()
    n_eq = sum(1 for r in rows if r.get("equal"))
    print(f"{'PASS' if fails == 0 and errors == 0 else 'FAIL'}: {n_eq}/{len(rows)} (shape, M) cases bit-identical, {fails} differ, {errors} errors")
    if a.json:
        json.dump(dict(device=torch.cuda.get_device_name(0), time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), rows=rows,
                       equal=n_eq, differ=fails, errors=errors), open(a.json, "w"), indent=1)
    return 0 if fails == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
