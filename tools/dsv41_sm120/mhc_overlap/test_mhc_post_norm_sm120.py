#!/usr/bin/env python3
"""The sm_120 shifted-mHC critical-path kernel against vLLM's TileLang kernels on DeepSeek-V4.1-Flash's
geometry (H = 5120, hc_mult 4): the post-mapped bf16 streams against `mhc_fused_tilelang` (the served
decode kernel, tokens <= 32) and `mhc_post_tilelang` (the separate post kernel), the collapsed +
RMSNorm'd layer input and the draft aux against `mhc_pre_big_fuse_with_norm` fed the TileLang outputs.
Then the timing in a CUDA graph: ours alone (the new critical path) against the served pair and against
upstream's overlap critical path (post + pre_norm[input]).

Run inside the ds41 image on one sm_120 GPU:
    python3 test_mhc_post_norm_sm120.py [--bench] [--modes]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mhc_post_norm_sm120 import POST_MIX_MODE, mhc_post_norm, mhc_proj  # noqa: E402

from vllm.model_executor.kernels.mhc.tilelang import (  # noqa: E402
    mhc_fused_post_pre_delayed_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.kernels.mhc.tilelang_kernels import _MHC_FUSED_TILELANG_KERNEL  # noqa: E402
from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL  # noqa: E402

H, HC, MIX = 5120, 4, 24
RMS_EPS, HC_EPS, POST_MULT, SINKHORN, NORM_EPS = 1e-20, 1e-6, 2.0, 20, 1e-20


def inputs(T, dev):
    x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
    residual = (torch.randn(T, HC, H, device=dev) * 0.8).to(torch.bfloat16)
    post_layer_mix = (torch.rand(T, HC, 1, device=dev) * 2).float()
    comb_res_mix = torch.softmax(torch.randn(T, HC, HC, device=dev), dim=-1).float().contiguous()
    fn = (torch.randn(MIX, HC * H, device=dev) * 0.01).float()
    scale = torch.tensor([0.02, 0.02, 0.05], device=dev)
    base = (torch.randn(MIX, device=dev) * 0.5).float()
    norm_w = (torch.rand(H, device=dev) + 0.5).to(torch.bfloat16)
    pre_mix = (torch.rand(T, HC, device=dev) * 0.9 + 0.05).float()
    return x, residual, post_layer_mix, comb_res_mix, fn, scale, base, norm_w, pre_mix


def tl_pre_norm(mode, mixes, sqrsum, scale, base, residual_cur, norm_w, pre_mix, T, write_aux):
    post = torch.empty(T, HC, dtype=torch.float32, device=residual_cur.device)
    comb = torch.empty(T, HC * HC, dtype=torch.float32, device=residual_cur.device)
    layer_input = torch.empty(T, H, dtype=torch.bfloat16, device=residual_cur.device)
    next_pre = torch.empty(T, HC, dtype=torch.float32, device=residual_cur.device)
    aux = torch.empty(T if write_aux else 0, H, dtype=torch.bfloat16, device=residual_cur.device)
    MHC_PRE_NORM_KERNEL(
        mixes, sqrsum, scale, base, residual_cur, post, comb, layer_input, norm_w, pre_mix, next_pre,
        aux if write_aux else layer_input,
        hidden_size=H, rms_eps=RMS_EPS, hc_pre_eps=HC_EPS, hc_sinkhorn_eps=HC_EPS, hc_post_mult_value=POST_MULT,
        sinkhorn_repeat=SINKHORN, norm_eps=NORM_EPS, hc_mult=HC, use_pre_mix_in=True, save_pre_mix=True,
        rms_numel=HC * H, split_mode=mode, write_aux=write_aux,
    )
    return post, comb, layer_input, next_pre, aux


def ulp_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    ai, bi = a.view(torch.int16).int(), b.view(torch.int16).int()
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
    ap.add_argument("--modes", action="store_true", help="show which post-mix contraction matches TileLang")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(3)
    fails = 0
    for T, capture_aux in ((1, False), (6, False), (6, True), (7, False), (16, True), (32, False), (48, True), (64, False)):
        x, residual, post_layer_mix, comb_res_mix, fn, scale, base, norm_w, pre_mix = inputs(T, dev)
        fused = T <= 32
        if fused:
            mixes, sqrsum, res_ref = _MHC_FUSED_TILELANG_KERNEL(
                comb_res_mix, residual, post_layer_mix.view(T, HC), x, fn.view(MIX, HC, H), HC, H, MIX)
        res_post = mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
        if not fused:
            res_ref = res_post
            mixes = torch.zeros(1, T, MIX, device=dev)
            sqrsum = torch.ones(1, T, device=dev)
        ref = tl_pre_norm("fused", mixes, sqrsum, scale, base, res_ref, norm_w, pre_mix, T, capture_aux)
        modes = (0, 1, 2) if args.modes else (POST_MIX_MODE,)
        for mode in modes:
            res_out, layer_input, aux = mhc_post_norm(x, residual, post_layer_mix, comb_res_mix, pre_mix, norm_w, NORM_EPS,
                                                      capture_aux=capture_aux, mode=mode)
            torch.cuda.synchronize()
            report, ok = [], True
            for name, r, o, limit in (("residual_out", res_ref, res_out, 0), ("residual_out vs post kernel", res_post, res_out, 0),
                                      ("layer_input", ref[2], layer_input, 1), ("aux", ref[4], aux, 1)):
                if r.numel() == 0:
                    continue
                n, u = ulp_stats(r, o)
                report.append(f"{name} {'identical' if n == 0 else f'{n} differ (<= {u} ulp)'}")
                ok &= u <= limit
                ok &= torch.allclose(r.float(), o.float(), rtol=1e-2, atol=1e-3)
            if mode == POST_MIX_MODE:
                fails += 0 if ok else 1
            tag = f" mode={mode}" if args.modes else ""
            print(f"[{'ok ' if ok else 'BAD'}] T={T:2d} aux={capture_aux}{tag} ({'fused' if fused else 'post'} reference): "
                  + "; ".join(report))
        if fused:
            # the side-stream projection against mhc_fused_tilelang's split partials (fp32: bit-identical)
            for tok_block in (1, 2, 4):
                mixes_o, sqrsum_o = mhc_proj(x, residual, post_layer_mix, comb_res_mix, fn, tok_block=tok_block)
                torch.cuda.synchronize()
                ok = torch.equal(mixes, mixes_o) and torch.equal(sqrsum, sqrsum_o)
                fails += 0 if ok else 1
                dm = (mixes.view(torch.int32) - mixes_o.view(torch.int32)).abs().max().item()
                ds = (sqrsum.view(torch.int32) - sqrsum_o.view(torch.int32)).abs().max().item()
                print(f"[{'ok ' if ok else 'BAD'}] T={T:2d} projection tok_block={tok_block}: mixes "
                      f"{'identical' if dm == 0 else f'<= {dm} ulp'}, sqrsum {'identical' if ds == 0 else f'<= {ds} ulp'}")
    if args.bench:
        for T in (1, 6, 16, 32):
            x, residual, post_layer_mix, comb_res_mix, fn, scale, base, norm_w, pre_mix = inputs(T, dev)
            mixes, sqrsum, res_ref = _MHC_FUSED_TILELANG_KERNEL(
                comb_res_mix, residual, post_layer_mix.view(T, HC), x, fn.view(MIX, HC, H), HC, H, MIX)
            torch.cuda.synchronize()
            t_pair = bench_graph(lambda: mhc_fused_post_pre_delayed_tilelang(
                x, residual, post_layer_mix, comb_res_mix, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
                pre_mix=pre_mix, norm_weight=norm_w, norm_eps=NORM_EPS, capture_aux=False))
            t_upstream = bench_graph(lambda: (mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix),
                                              tl_pre_norm("input", mixes, sqrsum, scale, base, res_ref, norm_w, pre_mix, T, False)))
            t_ours = bench_graph(lambda: mhc_post_norm(x, residual, post_layer_mix, comb_res_mix, pre_mix, norm_w, NORM_EPS, pdl=False))
            t_ours_pdl = bench_graph(lambda: mhc_post_norm(x, residual, post_layer_mix, comb_res_mix, pre_mix, norm_w, NORM_EPS,
                                                           pdl=True))
            t_ours_aux = bench_graph(lambda: mhc_post_norm(x, residual, post_layer_mix, comb_res_mix, pre_mix, norm_w, NORM_EPS,
                                                           capture_aux=True))
            t_side = bench_graph(lambda: (_MHC_FUSED_TILELANG_KERNEL(
                comb_res_mix, residual, post_layer_mix.view(T, HC), x, fn.view(MIX, HC, H), HC, H, MIX),
                tl_pre_norm("stats", mixes, sqrsum, scale, base, res_ref, norm_w, pre_mix, T, False)))
            t_tl_proj = bench_graph(lambda: _MHC_FUSED_TILELANG_KERNEL(
                comb_res_mix, residual, post_layer_mix.view(T, HC), x, fn.view(MIX, HC, H), HC, H, MIX))
            t_proj = {b: bench_graph(lambda: mhc_proj(x, residual, post_layer_mix, comb_res_mix, fn, tok_block=b)) for b in (1, 2, 4)}
            print(f"  T={T:2d}: projection: TileLang fused {t_tl_proj:5.2f} us ({T * 96} CTAs) | ours "
                  + " | ".join(f"tok_block={b} {t:5.2f} ({8 * -(-T // b)} CTAs x 256)" for b, t in t_proj.items()))
            # behind a kernel that does not trigger early (like the all-reduce that produces x)
            z = torch.zeros(1, device=dev)
            t_fill = bench_graph(lambda: z.fill_(1.0))
            t_chain = bench_graph(lambda: (z.fill_(1.0), mhc_post_norm(x, residual, post_layer_mix, comb_res_mix, pre_mix, norm_w,
                                                                       NORM_EPS, pdl=False)))
            t_chain_pdl = bench_graph(lambda: (z.fill_(1.0), mhc_post_norm(x, residual, post_layer_mix, comb_res_mix, pre_mix,
                                                                           norm_w, NORM_EPS, pdl=True)))
            print(f"  T={T:2d}: served pair {t_pair:5.2f} us | upstream overlap critical path (post + pre_norm[input]) "
                  f"{t_upstream:5.2f} | ours {t_ours:5.2f} (with PDL {t_ours_pdl:5.2f}; with aux {t_ours_aux:5.2f}) | "
                  f"side stream (fused GEMM + stats) {t_side:5.2f} | fill {t_fill:4.2f} -> fill+ours {t_chain:5.2f} "
                  f"(with PDL {t_chain_pdl:5.2f})")
    print("ALL OK" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
