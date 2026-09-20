#!/usr/bin/env python3
"""Correctness + timing of the sm_120 small-M MXFP8 GEMV against FlashInfer's CUTLASS
mm_mxfp8 (the kernel vLLM uses today) and a fp32 reference on dequantized operands.

Run inside the serving image on one sm_120 GPU:
    python3 test_mxfp8_gemv_sm120.py [--shapes decode|all] [--iters 50]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mxfp8_gemv_sm120 import mxfp8_gemv  # noqa: E402

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.utils import flashinfer as vllm_flashinfer  # noqa: E402

# DeepSeek-V4.1-Flash dense shapes at TP4 (N, K): the fused projections the server launches
# (grid sizes read from the 2026-09-18 profile: N = 1792, 8192, 5120 x2, 1152, 25600, 4096)
# plus the unfused component shapes and a few generic ones.
SHAPES = {
    "q_a+kv_a 5120->1792": (1792, 5120),
    "q_b+idx 1280->8192": (8192, 1280),
    "wo_b 2048->5120": (5120, 2048),
    "shared w13 5120->1152": (1152, 5120),
    "shared w2 576->5120": (5120, 576),
    "idx wq_b 1280->4096": (4096, 1280),
    "big 5120->25600": (25600, 5120),
    "q_a 5120->1280": (1280, 5120),
    "kv_a 5120->576": (576, 5120),
    "generic 4096x4096": (4096, 4096),
}
DECODE_SHAPES = [k for k in SHAPES if not k.startswith(("q_a 5120", "kv_a", "generic"))]


def dequant(x_fp8: torch.Tensor, sf_2d: torch.Tensor) -> torch.Tensor:
    x = x_fp8.to(torch.float32)
    nb = x.shape[-1] // MXFP8_BLOCK_SIZE
    xb = x.view(*x.shape[:-1], nb, MXFP8_BLOCK_SIZE)
    return (xb * torch.exp2(sf_2d.to(torch.float32) - 127.0).unsqueeze(-1)).view(x.shape)


def quant_rowmajor(x: torch.Tensor):
    """Quantize with vLLM's op in row-major scale layout, return (fp8, sf_2d)."""
    q, sf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    return q, sf.view(x.shape[0], -1)


def bench(fn, iters: int, per_graph: int = 20, rotate=None) -> float:
    """GPU time per call in us: `per_graph` back-to-back calls captured in one CUDA
    graph (removes Python/launch overhead, like the server's cudagraph replay).
    `rotate`: optional list of argument tuples cycled through the captured calls so
    the weights are not L2-resident (GB202 has a 128 MB L2; a decode step streams
    ~2 GB of weights, so the server always reads them from HBM)."""
    def call(i):
        if rotate is None:
            return fn()
        return fn(*rotate[i % len(rotate)])

    for i in range(3):
        call(i)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for i in range(2):
            call(i)
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for i in range(per_graph):
                call(i)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(max(3, iters // 5)):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) * 1000 / per_graph)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--ms", default="1,6,8,16", help="comma list of M values")
    ap.add_argument("--shapes", default="all", help="all | decode | comma list of names")
    ap.add_argument("--no-cutlass-timing", action="store_true", help="skip the CUTLASS timing (correctness still checked)")
    args = ap.parse_args()
    torch.manual_seed(3)
    dev = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}  "
          f"VLLM_MOET_GEMV_IMPL={__import__('os').environ.get('VLLM_MOET_GEMV_IMPL', '(default: v3)')}")
    t0 = time.time()
    mxfp8_gemv(*[t for t in _dummy(dev)])  # trigger JIT build
    print(f"extension ready in {time.time()-t0:.0f}s")

    if args.shapes == "all":
        names = list(SHAPES)
    elif args.shapes == "decode":
        names = DECODE_SHAPES
    else:
        names = [s.strip() for s in args.shapes.split(",")]
    fails = 0
    for name in names:
        N, K = SHAPES[name]
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
        w_q, w_sf2d = quant_rowmajor(w)
        w_sf_sw = swizzle_mxfp8_scale(w_sf2d, M=N, K=K)
        w_deq = dequant(w_q, w_sf2d)
        # Cold-L2 timing: cycle through enough distinct weight (and scale) copies to
        # exceed the 128 MB L2 (>= 160 MB total), so every captured call streams from HBM
        # -- in the server both the weights and their scales are cold.
        n_copies = max(2, (160 << 20) // max(N * K, 1) + 1)
        n_copies = min(n_copies, 64)
        w_copies = [(w_q.clone(), w_sf_sw.clone()) for _ in range(n_copies)]
        for M in [int(m) for m in args.ms.split(",")]:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            a_q, a_sf_sw = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            a_q2, a_sf2d = quant_rowmajor(x)
            assert torch.equal(a_q, a_q2)
            ref = dequant(a_q2, a_sf2d) @ w_deq.t()  # fp32 exact on dequantized operands
            cutlass = vllm_flashinfer.mm_mxfp8(
                a_q, w_q.t(), a_sf_sw, w_sf_sw, out_dtype=torch.bfloat16, backend="cutlass"
            )
            ours = mxfp8_gemv(a_q, a_sf_sw, w_q, w_sf_sw)
            err_ours = ((ours.float() - ref).abs() / (ref.abs() + 1e-2)).max().item()
            err_cut = ((cutlass.float() - ref).abs() / (ref.abs() + 1e-2)).max().item()
            close = torch.allclose(ours.float(), cutlass.float(), rtol=2e-2, atol=1e-2)
            rot = w_copies
            if args.no_cutlass_timing:
                t_cut = float("nan")
            else:
                t_cut = bench(
                    lambda wc, sc: vllm_flashinfer.mm_mxfp8(
                        a_q, wc.t(), a_sf_sw, sc, out_dtype=torch.bfloat16, backend="cutlass"
                    ),
                    args.iters, per_graph=len(rot), rotate=rot,
                )
            t_ours = bench(lambda wc, sc: mxfp8_gemv(a_q, a_sf_sw, wc, sc), args.iters,
                           per_graph=len(rot), rotate=rot)
            gbps = (N * K + M * K) / t_ours / 1e3  # GB/s of fp8 bytes streamed
            ok = close and err_ours < 2e-2
            fails += 0 if ok else 1
            print(
                f"[{'ok ' if ok else 'BAD'}] {name:<24} M={M:2d}  maxrel ours={err_ours:.2e} cutlass={err_cut:.2e}  "
                f"cutlass {t_cut:6.1f} us  gemv {t_ours:6.1f} us  x{t_cut/t_ours:4.1f}  ({gbps:5.0f} GB/s)"
            )
    print(f"\n{'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


def _dummy(dev):
    x = torch.randn(4, 128, device=dev, dtype=torch.bfloat16)
    w = torch.randn(128, 128, device=dev, dtype=torch.bfloat16)
    a_q, a_sf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    w_q, w_sf2d = quant_rowmajor(w)
    return a_q, a_sf, w_q, swizzle_mxfp8_scale(w_sf2d, M=128, K=128)


if __name__ == "__main__":
    raise SystemExit(main())
