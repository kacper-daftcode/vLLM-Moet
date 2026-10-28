#!/usr/bin/env python3
"""The mHC boundary kernel's MXFP8 copy of layer_input (step 38) against the served quantization.

1. bits: with quant_out the bf16 outputs (residual_out, layer_input, aux) are unchanged, the e4m3 bytes equal
   mxfp8_e4m3_quantize(layer_input, is_sf_swizzled_layout=True)[0] and the scale bytes of every real (token, 32-block)
   equal the served F8_128x4 bytes - at the served H = 5120 and the other H the kernel takes, T = 1..170 including
   the 128-row tile boundary, with zero rows (scale 0), saturating magnitudes and tiny values.
2. end to end: the sm_120 GEMV (and FlashInfer's CUTLASS mm_mxfp8 above 16 rows) on the kernel's pair gives the
   same bf16 output as on the served quantization's pair.
3. the consumer hook: mhc_post_norm(..., quant_out=True) returns a layer_input with `_moet_mxfp8` the patched
   apply_weights reads (checked here by reproducing its test), and nothing when quant_out is off.
--bench: the kernel with / without the quant tail and the chain kernel -> quant -> GEMV vs kernel(+quant) -> GEMV
in a CUDA graph with PDL at T = 1, 6, 8, 16 (what the served graph sees per boundary).

    python3 test_mhc_post_norm_mxfp8_sm120.py [--bench]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mhc_post_norm_sm120 import MXFP8_ATTR, mhc_post_norm, mxfp8_buffers  # noqa: E402

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize  # noqa: E402

HC, NORM_EPS = 4, 1e-20


def inputs(T, H, dev, kind="normal"):
    x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
    residual = (torch.randn(T, HC, H, device=dev) * 0.8).to(torch.bfloat16)
    post_layer_mix = (torch.rand(T, HC, 1, device=dev) * 2).float()
    comb_res_mix = torch.softmax(torch.randn(T, HC, HC, device=dev), dim=-1).float().contiguous()
    norm_w = (torch.rand(H, device=dev) + 0.5).to(torch.bfloat16)
    pre_mix = (torch.rand(T, HC, device=dev) * 0.9 + 0.05).float()
    if kind == "edges":
        # The RMSNorm makes every row O(1) x w, so the magnitudes come from the weight: columns 0..1023 at ~1e3
        # (large exponents, values at the e4m3 saturation edge after the block scaling), 1024..2047 at 1e-36
        # (the normalized amax in fp32's subnormal range - the recipe's no-bump branch), zero rows (sum of squares 0:
        # all-zero blocks, scale 0) and a zero block inside a normal row.
        norm_w[:1024] = (norm_w[:1024].float() * 1000.0).to(torch.bfloat16)
        norm_w[1024:2048] = 1e-36
        if T >= 1:
            x[0].zero_(); residual[0].zero_()
        if T >= 4:
            x[3, 64:96].zero_(); residual[3, :, 64:96].zero_()
    return x, residual, post_layer_mix, comb_res_mix, norm_w, pre_mix


def swizzled_offsets(tokens: int, groups: int, dev) -> torch.Tensor:
    """F8_128x4 offsets of the real (token, group) scale bytes (groups % 4 == 0 here)."""
    row = torch.arange(tokens, device=dev).view(-1, 1)
    g = torch.arange(groups, device=dev).view(1, -1)
    return (row // 128 * (128 * groups) + g // 4 * 512 + row % 32 * 16 + row % 128 // 32 * 4 + g % 4).flatten()


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
    torch.manual_seed(38)
    fails = 0
    cases = 0

    gemv = None
    try:
        sys.path.insert(0, "/opt/vllm-moet/dsv41_sm120/sm120_gemv")
        from mxfp8_gemv_sm120 import mxfp8_gemv as gemv
    except Exception as exc:  # noqa: BLE001
        print(f"GEMV unavailable ({exc!r}): the end-to-end check uses CUTLASS only")
    from vllm.utils import flashinfer as vllm_flashinfer

    for H in (5120, 4096, 2048, 1024):
        for T in (1, 2, 3, 4, 6, 7, 8, 16, 17, 31, 32, 64, 127, 128, 129, 170):
            for kind in ("normal", "edges"):
                x, residual, plm, crm, norm_w, pre_mix = inputs(T, H, dev, kind)
                ref = mhc_post_norm(x, residual, plm, crm, pre_mix, norm_w, NORM_EPS, capture_aux=True, quant_out=False)
                q = mxfp8_buffers(T, H, dev)
                got = mhc_post_norm(x, residual, plm, crm, pre_mix, norm_w, NORM_EPS, capture_aux=True, quant_out=q)
                cases += 1
                same_bf16 = all(torch.equal(a, b) for a, b in zip(ref, got))
                s_data, s_scale = mxfp8_e4m3_quantize(ref[1], is_sf_swizzled_layout=True)
                offs = swizzled_offsets(T, H // 32, dev)
                same_data = torch.equal(q[0].view(torch.uint8), s_data.view(torch.uint8))
                same_scale = torch.equal(q[1].flatten()[offs], s_scale.flatten()[offs])
                has_attr = getattr(got[1], MXFP8_ATTR, None) is not None and getattr(ref[1], MXFP8_ATTR, None) is None
                # end to end on a weight of the served kind
                N = 1536 if H == 5120 else 512
                w = (torch.randn(N, H, device=dev) * 0.02).to(torch.bfloat16)
                w_q, w_sf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=True)
                if T <= 16 and gemv is not None:
                    out_ref = gemv(s_data, s_scale, w_q, w_sf)
                    out_got = gemv(q[0], q[1], w_q, w_sf)
                else:
                    out_ref = vllm_flashinfer.mm_mxfp8(s_data, w_q.t(), s_scale, w_sf, out_dtype=torch.bfloat16, backend="cutlass")
                    out_got = vllm_flashinfer.mm_mxfp8(q[0], w_q.t(), q[1], w_sf, out_dtype=torch.bfloat16, backend="cutlass")
                same_out = torch.equal(out_ref, out_got)
                ok = same_bf16 and same_data and same_scale and has_attr and same_out
                if not ok:
                    fails += 1
                    nd = int((q[0].view(torch.uint8) != s_data.view(torch.uint8)).sum())
                    ns = int((q[1].flatten()[offs] != s_scale.flatten()[offs]).sum())
                    print(f"FAIL H={H} T={T} {kind}: bf16 {same_bf16} data {same_data} ({nd} bytes) scale {same_scale} ({ns} bytes) "
                          f"attr {has_attr} out {same_out}")
    print(f"bits: {cases - fails}/{cases} cases identical (bf16 outputs, e4m3 bytes, real scale bytes, GEMV/CUTLASS output)")

    # the consumer hook as the patched apply_weights uses it
    x, residual, plm, crm, norm_w, pre_mix = inputs(6, 5120, dev)
    _, li, _ = mhc_post_norm(x, residual, plm, crm, pre_mix, norm_w, NORM_EPS, quant_out=True)
    pre = getattr(li, MXFP8_ATTR, None)
    hook_ok = pre is not None and pre[0].shape == li.shape and pre[0].dtype == torch.float8_e4m3fn and pre[1].dtype == torch.uint8
    print(f"hook: quant_out=True attaches {MXFP8_ATTR} = (e4m3 {tuple(pre[0].shape) if pre else None}, uint8 {pre[1].numel() if pre else None}) -> {'ok' if hook_ok else 'FAIL'}")
    fails += 0 if hook_ok else 1

    if args.bench:
        H = 5120
        N = 1536
        w = (torch.randn(N, H, device=dev) * 0.02).to(torch.bfloat16)
        w_q, w_sf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=True)
        for T in (1, 6, 8, 16):
            x, residual, plm, crm, norm_w, pre_mix = inputs(T, H, dev)
            z = torch.empty(64 << 20, device=dev, dtype=torch.float32)  # a producer in front (PDL overlap, cold L2)
            q = mxfp8_buffers(T, H, dev)
            t_k = bench_graph(lambda: (z.fill_(1.0), mhc_post_norm(x, residual, plm, crm, pre_mix, norm_w, NORM_EPS, quant_out=False)))
            t_kq = bench_graph(lambda: (z.fill_(1.0), mhc_post_norm(x, residual, plm, crm, pre_mix, norm_w, NORM_EPS, quant_out=q)))
            t_fill = bench_graph(lambda: z.fill_(1.0))

            def chain_sep():
                z.fill_(1.0)
                _, li, _ = mhc_post_norm(x, residual, plm, crm, pre_mix, norm_w, NORM_EPS, quant_out=False)
                a, s = mxfp8_e4m3_quantize(li, is_sf_swizzled_layout=True)
                return gemv(a, s, w_q, w_sf) if gemv is not None else None

            def chain_fused():
                z.fill_(1.0)
                mhc_post_norm(x, residual, plm, crm, pre_mix, norm_w, NORM_EPS, quant_out=q)
                return gemv(q[0], q[1], w_q, w_sf) if gemv is not None else None

            t_sep = bench_graph(chain_sep) if gemv is not None else float("nan")
            t_fused = bench_graph(chain_fused) if gemv is not None else float("nan")
            print(f"T={T:2d}: kernel {t_k - t_fill:5.2f} us, +quant tail {t_kq - t_fill:5.2f} (+{t_kq - t_k:4.2f}); "
                  f"chain kernel->quant->GEMV {t_sep - t_fill:6.2f} vs kernel(+quant)->GEMV {t_fused - t_fill:6.2f} "
                  f"({t_fused - t_sep:+5.2f} us per site)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
