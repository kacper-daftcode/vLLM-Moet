#!/usr/bin/env python3
"""Bit test of step 35: spread_topk_out (the router's spread GEMM + DeepSeek-V4's routing top-6 in one launch) against
spread_out followed by dsv4_topk_sm120's kernel, on random operands: tokens 1-8, every supported slice width, logits
in a tame and in a wide range (the logf / x > 20 branches), int32 and int64 ids, with and without padding rows and
the image-token bias, plain and PDL launches. Logits, weights and ids must be bit-identical. Then the served shape
(8 tokens, 5120 -> 384) is timed in a CUDA graph: two launches vs one.
usage: python3 test_bf16_spread_topk_sm120.py [--cases N]   (one sm_120 GPU; both extensions are built on first use)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dsv4_topk"))
import bf16_replica_gemm_sm120 as gemm  # noqa: E402
import dsv4_topk_sm120 as topk  # noqa: E402

dev = torch.device("cuda")
E, K = 384, 5120
STEPS = [192, 256, 320, 384, 448, 512, 576, 640]  # steps/16 = 12, 16, ..., 40 (<= 32 slices of K = 5120)


def workspace(n, slices):
    return (torch.empty(n // 8 * slices * 128 + 2 * 16 * n, device=dev, dtype=torch.float32),
            torch.zeros(n // 8 + 1, device=dev, dtype=torch.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=72)
    a = ap.parse_args()
    ext, text = gemm._ext(), topk._ext()
    gen = torch.Generator(device="cpu").manual_seed(35)
    fails = 0
    for i in range(a.cases):
        m = 1 + i % 8
        step = STEPS[(i // 8) % len(STEPS)]
        wide = (i // 3) % 2 == 1
        pdl = (i // 2) % 2 == 1
        ids64 = (i // 5) % 2 == 1
        pad = (i // 7) % 3 == 1
        vl = (i // 7) % 3 == 2
        x = (torch.randn((m, K), generator=gen) * (0.5 if not wide else 4.0)).to(torch.bfloat16).to(dev)
        w = (torch.randn((E, K), generator=gen) * (0.02 if not wide else 0.4)).to(torch.bfloat16).to(dev)
        if wide and i % 6 == 5:
            w[0, :64] = 3.0e4  # inf / NaN logits for a few experts
        bias = (torch.randn(E, generator=gen) * 0.1).to(dev)
        scale = 1.0 + (i % 4) * 0.5
        is_padding = None
        if pad:
            is_padding = torch.zeros(m, dtype=torch.bool, device=dev)
            is_padding[m - 1] = True
        input_ids, bias_vl, image_lo = None, None, 0
        if vl:
            image_lo = 1000
            input_ids = torch.randint(0, 2000, (m,), generator=gen).to(torch.int64 if ids64 else torch.int32).to(dev)
            input_ids[0] = 1002
            bias_vl = (torch.randn(E, generator=gen) * 0.1).to(dev)
        idt = torch.int64 if ids64 else torch.int32
        slices = -(-K // step)
        ws, cnt = workspace(E, slices)
        # reference: the spread GEMM, then the top-k kernel
        out_ref = torch.empty((m, E), device=dev, dtype=torch.float32)
        ext.spread_out(x, w, out_ref, step, False, ws, cnt, 4, pdl)
        w_ref = torch.empty((m, 6), device=dev, dtype=torch.float32)
        i_ref = torch.empty((m, 6), device=dev, dtype=idt)
        text.dsv4_topk_out(out_ref, bias, is_padding, w_ref, i_ref, scale, input_ids, bias_vl, image_lo, False)
        # fused
        out_new = torch.full((m, E), -7.0, device=dev, dtype=torch.float32)
        w_new = torch.full((m, 6), -7.0, device=dev, dtype=torch.float32)
        i_new = torch.full((m, 6), -7, device=dev, dtype=idt)
        ext.spread_topk_out(x, w, out_new, step, ws, cnt, 4, pdl, bias, is_padding, w_new, i_new, scale, input_ids, bias_vl, image_lo)
        torch.cuda.synchronize()
        bad = []
        if not torch.equal(out_ref.view(torch.int32), out_new.view(torch.int32)):
            bad.append("logits")
        if not torch.equal(w_ref.view(torch.int32), w_new.view(torch.int32)):
            bad.append("weights")
        if not torch.equal(i_ref, i_new):
            bad.append("ids")
        if int(cnt.abs().sum().item()) != 0:
            bad.append("counters not reset")
        if bad:
            fails += 1
            print(f"case {i} m={m} step={step} wide={wide} pdl={pdl} ids64={ids64} pad={pad} vl={vl}: MISMATCH {bad}")
    print(f"{a.cases - fails}/{a.cases} cases bit-identical (logits, top-6 weights and ids)")
    # timing at the served shape in a CUDA graph (the calibrated router case on this GPU is a spread of 8 x 640-wide slices
    # at 8 tokens; take step 640)
    m, step = 8, 640
    slices = -(-K // step)
    x = torch.randn((m, K), device=dev).to(torch.bfloat16)
    w = (torch.randn((E, K), device=dev) * 0.02).to(torch.bfloat16)
    bias = torch.randn(E, device=dev) * 0.1
    ws, cnt = workspace(E, slices)
    out = torch.empty((m, E), device=dev, dtype=torch.float32)
    tw = torch.empty((m, 6), device=dev, dtype=torch.float32)
    ti = torch.empty((m, 6), device=dev, dtype=torch.int32)
    for pdl in (False, True):
        def two():
            ext.spread_out(x, w, out, step, False, ws, cnt, 4, pdl)
            text.dsv4_topk_out(out, bias, None, tw, ti, 1.5, None, None, 0, False)
        def one():
            ext.spread_topk_out(x, w, out, step, ws, cnt, 4, pdl, bias, None, tw, ti, 1.5, None, None, 0)
        for name, fn in ((f"spread + top-k (pdl={pdl})", two), (f"spread_topk (pdl={pdl})", one)):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                fn()
                torch.cuda.synchronize()
                with torch.cuda.graph(g, stream=s):
                    for _ in range(40):
                        fn()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                g.replay()
            torch.cuda.synchronize()
            print(f"   {name}: {(time.perf_counter() - t0) / 50 / 40 * 1e6:.2f} us per layer in a graph")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
