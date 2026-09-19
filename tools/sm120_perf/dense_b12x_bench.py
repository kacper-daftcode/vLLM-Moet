#!/usr/bin/env python3
"""Dense MXFP8 GEMM at DeepSeek-V4.1 decode shapes: this repo's mxfp8_gemv (production) vs
b12x.gemm.blockscaled.mm (CuTe-DSL MXFP8 GEMM for sm_120, `pip install b12x`) on one GPU, cold L2
(rotating weights). Run inside the ds41 image with the repo mounted (the GEMV extension is JIT-built
from ../dsv41_sm120/sm120_gemv if the image does not carry it).
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dsv41_sm120", "sm120_gemv"))
from mxfp8_gemv_sm120 import mxfp8_gemv  # noqa: E402

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize  # noqa: E402

dev = torch.device("cuda:0")
SHAPES = [(1280, 5120), (4096, 1280), (576, 5120), (5120, 2048), (1152, 5120), (5120, 576)]  # (N, K)
ROT = 8


def timeit(fns, iters=40):
    for f in fns:
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for f in fns:
            f()
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for f in fns:
                f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / len(fns))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    import b12x.gemm.blockscaled as bs
    print("device", torch.cuda.get_device_name(0), "b12x supported:", bs.is_supported())
    for M in (1, 6, 16):
        for N, K in SHAPES:
            x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
            xq_sw, xsf_sw = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            xq, xsf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
            ws, ws_b12 = [], []
            ref = None
            for r in range(ROT):
                w = (torch.randn(N, K, device=dev) * 0.05).to(torch.bfloat16)
                wq_sw, wsf_sw = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=True)
                wq, wsf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
                ws.append((wq_sw, wsf_sw))
                try:
                    ws_b12.append(bs.pack_weight(wq, wsf.view(torch.uint8).reshape(N, K // 32)))
                except Exception as e:  # noqa: BLE001
                    print("pack_weight failed:", repr(e)[:300])
                    return
                if r == 0:
                    ref = (x.float() @ w.float().t())
            out_ours = mxfp8_gemv(xq_sw, xsf_sw, ws[0][0], ws[0][1])
            err_ours = ((out_ours.float() - ref).norm() / ref.norm()).item()
            t_ours = timeit([lambda w=w: mxfp8_gemv(xq_sw, xsf_sw, w[0], w[1]) for w in ws])
            line = f"M={M:2d} {N:5d}<-{K:5d}: ours {t_ours:6.1f} us (relerr {err_ours:.1e})"
            for label, lhs in (("b12x fp8-in", (xq, xsf.view(torch.uint8).reshape(M, K // 32))), ("b12x bf16-in", x)):
                try:
                    t0 = time.time()
                    out_b = bs.mm(lhs, ws_b12[0], expected_m=M)
                    torch.cuda.synchronize()
                    jit = time.time() - t0
                    err_b = ((out_b.float() - ref).norm() / ref.norm()).item()
                    t_b = timeit([lambda w=w: bs.mm(lhs, w, expected_m=M) for w in ws_b12])
                    line += f" | {label} {t_b:6.1f} us (relerr {err_b:.1e}, jit {jit:.0f}s)"
                except Exception as e:  # noqa: BLE001
                    line += f" | {label} failed: {repr(e)[:160]}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
