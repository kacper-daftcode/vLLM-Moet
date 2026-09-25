#!/usr/bin/env python3
"""The patched `mhc_shifted_post_pre` (patch_vllm_mhc_overlap_sm120.py) against the served path, driven
like the decoder layer does: with a side stream (the critical path in one kernel, the projection and the
coefficients on the side stream, joined afterwards) against `stream=None` (the TileLang pair), eagerly and
inside a CUDA graph, for 1-64 tokens with and without the draft aux; every output bit-identical. Then the
timing that matters: one boundary followed by a stand-in for the sublayer (a GEMM), with the join where
the decoder layer puts it.

Run inside the ds41 image on one sm_120 GPU after the patch has been applied (the image does that at
build; on an unpatched image pass --patch to apply it in place first):
    python3 test_mhc_overlap_integration.py [--patch] [--bench]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

H, HC, MIX = 5120, 4, 24
RMS_EPS, HC_EPS, POST_MULT, SINKHORN, NORM_EPS = 1e-20, 1e-6, 2.0, 20, 1e-20


def apply_patch_in_place() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import patch_vllm_mhc_overlap_sm120 as p

    src = p.DEFAULT_PATH.read_text()
    if p.MARKER not in src:
        p.DEFAULT_PATH.write_text(p.patch_text(src, str(Path(__file__).resolve().parent)))
        print(f"patched {p.DEFAULT_PATH}")


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
    ap.add_argument("--patch", action="store_true", help="apply patch_vllm_mhc_overlap_sm120.py in place first")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if args.patch:
        apply_patch_in_place()
    import os

    os.environ.setdefault("VLLM_MOET_MHC_OVERLAP_DIR", str(Path(__file__).resolve().parent))
    from vllm.models.deepseek_v41.nvidia.ops import mhc as ops

    assert hasattr(ops, "_moet_mhc_shifted_post_pre_overlap"), "ops/mhc.py is not patched (use --patch)"
    assert ops._moet_mhc_post_norm() is not None, "the sm_120 kernel did not load"
    print(f"MHC_OVERLAP_MAX_TOKENS = {ops.MHC_OVERLAP_MAX_TOKENS}")
    dev = torch.device("cuda")
    torch.manual_seed(5)
    side = torch.cuda.Stream()

    def run(stream, T, capture_aux, ins):
        x, residual, post_layer_mix, comb_res_mix, fn, scale, base, norm_w, pre_mix = ins
        outs = ops.mhc_shifted_post_pre(
            x, residual, post_layer_mix, comb_res_mix, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
            pre_mix=pre_mix, norm_weight=norm_w, norm_eps=NORM_EPS, capture_aux=capture_aux, stream=stream,
            reduce_results=False)
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
        return outs

    names = ("residual", "post_mix", "comb_mix", "layer_input", "next_pre_mix", "aux")
    fails = 0
    for proj in ("fused", "ours", "tf32"):
        os.environ["VLLM_MOET_MHC_PROJ"] = proj
        for T, capture_aux in ((1, False), (6, False), (6, True), (16, True), (32, False), (33, False), (48, True), (64, False)):
            ins = inputs(T, dev)
            ref = run(None, T, capture_aux, ins)
            eager = run(side, T, capture_aux, ins)
            torch.cuda.synchronize()
            # the same inside a CUDA graph (fork / join / record_stream under capture), replayed twice
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                run(side, T, capture_aux, ins)
                s.synchronize()
                with torch.cuda.graph(g, stream=s):
                    graphed = run(side, T, capture_aux, ins)
            g.replay()
            g.replay()
            torch.cuda.synchronize()
            report, ok = [], True
            for name, r, e, gr in zip(names, ref, eager, graphed):
                same_e = torch.equal(r, e)
                same_g = torch.equal(r, gr)
                if not same_e or not same_g:
                    # tf32 below 33 tokens is the served path's >32-token numerics: close, not identical
                    close = torch.allclose(r.float(), e.float(), rtol=2e-2, atol=1e-3) and torch.equal(e, gr)
                    ok &= close and proj == "tf32" and T <= 32
                    report.append(f"{name} {'~' if close else '!='}")
                else:
                    report.append(f"{name} =")
            fails += 0 if ok else 1
            path = "post+tf32" if (T > 32 or proj == "tf32") else proj
            print(f"[{'ok ' if ok else 'BAD'}] T={T:2d} aux={capture_aux} proj={proj:5s} ({path}) eager+graph vs served: "
                  + ", ".join(report))
    os.environ["VLLM_MOET_MHC_PROJ"] = "ours"
    if args.bench:
        # one boundary + a stand-in for the sublayer, the join after it as in the decoder layer. Two stand-ins:
        # a compute-bound GEMM that fills every SM (the pessimistic case: the side stream can only share) and
        # a weight-streaming small-M GEMM like the decode GEMVs (bandwidth-bound, SM slots free).
        w_big = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        a_big = torch.randn(64, 8192, device=dev, dtype=torch.bfloat16)
        w_gemv = [torch.randn(5120, 8192, device=dev, dtype=torch.bfloat16) for _ in range(4)]  # 4 x 84 MB, cold
        for T in (1, 6, 16, 32, 48):
            ins = inputs(T, dev)
            a_gemv = torch.randn(T, 5120, device=dev, dtype=torch.bfloat16)
            for label, work in (("compute-bound GEMM", lambda: torch.mm(a_big, w_big)),
                                ("weight-streaming GEMVs", lambda: [torch.mm(a_gemv, w) for w in w_gemv])):

                def served():
                    ops.mhc_shifted_post_pre(*ins[:7], RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN, pre_mix=ins[8],
                                             norm_weight=ins[7], norm_eps=NORM_EPS, stream=None)
                    work()

                def overlap():
                    ops.mhc_shifted_post_pre(*ins[:7], RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN, pre_mix=ins[8],
                                             norm_weight=ins[7], norm_eps=NORM_EPS, stream=side)
                    work()
                    torch.cuda.current_stream().wait_stream(side)

                t_work = bench_graph(work)
                t_served = bench_graph(served)
                line = (f"  T={T:2d} {label:22s} {t_work:6.2f} us | served boundary +{t_served - t_work:5.2f} | overlapped:")
                for proj in ("fused", "ours", "tf32"):
                    os.environ["VLLM_MOET_MHC_PROJ"] = proj
                    t_overlap = bench_graph(overlap)
                    line += f" {proj} +{t_overlap - t_work:5.2f}"
                os.environ["VLLM_MOET_MHC_PROJ"] = "ours"
                print(line)
    print("ALL OK" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
