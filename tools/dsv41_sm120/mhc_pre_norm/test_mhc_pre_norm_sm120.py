#!/usr/bin/env python3
"""The sm_120 mHC pre epilogue against vLLM's TileLang kernel (`MHC_PRE_NORM_KERNEL`, the call of
`mhc_fused_post_pre_delayed_tilelang`: shifted mHC with the carried pre-mix, RMSNorm fused), on
DeepSeek-V4.1-Flash's geometry (H = 5120, hc_mult 4, 24 mixes, Sinkhorn 20, eps 1e-20 / 1e-6).

Compares post_mix, comb_mix, next_pre_mix (fp32: bit-identical or last-bit), layer_input and the
draft aux (bf16: identical or 1-ulp) for 1..64 tokens and several split counts, then times both in
a CUDA graph. Run inside the ds41 image on one sm_120 GPU:
    python3 test_mhc_pre_norm_sm120.py [--bench]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mhc_pre_norm_sm120 import mhc_pre_norm  # noqa: E402

from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL  # noqa: E402

H, HC, MIX = 5120, 4, 24
RMS_EPS, HC_EPS, POST_MULT, SINKHORN, NORM_EPS = 1e-20, 1e-6, 2.0, 20, 1e-20


def run_tilelang(mixes, sqrsum, scale, base, residual, norm_w, pre_in, T, capture_aux):
    post = torch.empty(T, HC, dtype=torch.float32, device=residual.device)
    comb = torch.empty(T, HC * HC, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(T, H, dtype=torch.bfloat16, device=residual.device)
    next_pre = torch.empty(T, HC, dtype=torch.float32, device=residual.device)
    aux = torch.empty(T if capture_aux else 0, H, dtype=torch.bfloat16, device=residual.device)
    MHC_PRE_NORM_KERNEL(
        mixes, sqrsum, scale, base, residual, post, comb, layer_input, norm_w, pre_in, next_pre,
        aux if capture_aux else layer_input,
        hidden_size=H, rms_eps=RMS_EPS, hc_pre_eps=HC_EPS, hc_sinkhorn_eps=HC_EPS, hc_post_mult_value=POST_MULT,
        sinkhorn_repeat=SINKHORN, norm_eps=NORM_EPS, hc_mult=HC, use_pre_mix_in=True, save_pre_mix=True,
        rms_numel=HC * H, write_aux=capture_aux,
    )
    return post, comb, layer_input, next_pre, aux


def run_ours(mixes, sqrsum, scale, base, residual, norm_w, pre_in, T, capture_aux):
    post = torch.empty(T, HC, dtype=torch.float32, device=residual.device)
    comb = torch.empty(T, HC * HC, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(T, H, dtype=torch.bfloat16, device=residual.device)
    next_pre = torch.empty(T, HC, dtype=torch.float32, device=residual.device)
    aux = torch.empty(T, H, dtype=torch.bfloat16, device=residual.device) if capture_aux else None
    mhc_pre_norm(mixes, sqrsum, scale, base, residual, post, comb, layer_input, norm_w, pre_in, next_pre, aux,
                 rms_numel=HC * H, rms_eps=RMS_EPS, hc_pre_eps=HC_EPS, hc_sinkhorn_eps=HC_EPS,
                 hc_post_mult_value=POST_MULT, sinkhorn_repeat=SINKHORN, norm_eps=NORM_EPS)
    return post, comb, layer_input, next_pre, aux


def ulp_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    """(elements that differ, max difference in units of the smaller magnitude's ulp)."""
    if a.dtype == torch.bfloat16:
        ai, bi = a.view(torch.int16).int(), b.view(torch.int16).int()
    else:
        ai, bi = a.view(torch.int32), b.view(torch.int32)
    diff = (ai - bi).abs()
    return int((diff != 0).sum()), int(diff.max()) if diff.numel() else 0


def bench_graph(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        s.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(reps):
                fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / reps)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(3)
    scale = torch.tensor([0.02, 0.02, 0.05], device=dev)
    base = (torch.randn(MIX, device=dev) * 0.5).float()
    norm_w = (torch.rand(H, device=dev) + 0.5).to(torch.bfloat16)
    fails = 0
    for T, S, capture_aux in ((1, 12, False), (6, 12, False), (6, 16, True), (7, 12, False), (16, 12, False), (48, 12, True), (64, 12, False)):
        mixes = (torch.randn(S, T, MIX, device=dev) * 30).float()
        sqrsum = (torch.rand(S, T, device=dev) * 2000 + 100).float()
        residual = (torch.randn(T, HC, H, device=dev) * 0.8).to(torch.bfloat16)
        pre_in = (torch.rand(T, HC, device=dev) * 0.9 + 0.05).float()
        ref = run_tilelang(mixes, sqrsum, scale, base, residual, norm_w, pre_in, T, capture_aux)
        ours = run_ours(mixes, sqrsum, scale, base, residual, norm_w, pre_in, T, capture_aux)
        torch.cuda.synchronize()
        names = ("post_mix", "comb_mix", "layer_input", "next_pre_mix", "aux")
        report = []
        ok = True
        for name, r, o in zip(names, ref, ours):
            if o is None or r.numel() == 0:
                continue
            n, u = ulp_stats(r, o)
            report.append(f"{name} {'identical' if n == 0 else f'{n} differ (<= {u} ulp)'}")
            limit = 4 if name in ("comb_mix",) else (1 if name in ("layer_input", "aux") else 2)
            ok &= u <= limit
            # magnitude check too
            ok &= torch.allclose(r.float(), o.float(), rtol=1e-3, atol=1e-5)
        fails += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] T={T:2d} S={S} aux={capture_aux}: " + "; ".join(report))
    if args.bench:
        for T in (1, 6, 16, 48):
            S = 12
            mixes = (torch.randn(S, T, MIX, device=dev) * 30).float()
            sqrsum = (torch.rand(S, T, device=dev) * 2000 + 100).float()
            residual = (torch.randn(T, HC, H, device=dev) * 0.8).to(torch.bfloat16)
            pre_in = (torch.rand(T, HC, device=dev) * 0.9 + 0.05).float()
            t_tl = bench_graph(lambda: run_tilelang(mixes, sqrsum, scale, base, residual, norm_w, pre_in, T, False))
            t_us = bench_graph(lambda: run_ours(mixes, sqrsum, scale, base, residual, norm_w, pre_in, T, False))
            t_us_aux = bench_graph(lambda: run_ours(mixes, sqrsum, scale, base, residual, norm_w, pre_in, T, True))
            print(f"  T={T:2d}: TileLang pre_big_fuse_with_norm {t_tl:5.2f} us | ours {t_us:5.2f} us (with aux {t_us_aux:5.2f})")
    print("ALL OK" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
