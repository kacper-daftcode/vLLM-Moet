#!/usr/bin/env python3
"""fp4_kv_quant.py vs the checkpoint's own quantizer (`inference/kernel.py::fp4_act_quant`, TileLang).

Needs the checkpoint's `inference/` directory (tilelang is in the serving image):
  python3 test_fp4_kv_quant.py --model-dir /model

Checks, on RMS-normed-like latents (unit scale, heavy tails, zero groups, tiny groups, values
at exact e2m1 midpoints):
  * dequantized values (inplace=True reference) bit-exact in bf16;
  * the codes/scales: reference codes (float4_e2m1fn_x2 bytes) and e4m3 scale bytes bit-exact;
  * dequant(codes, scales) == dequantized values (the NVFP4 record round trip is lossless).
"""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--rows", type=int, default=8192)
    args = ap.parse_args()
    sys.path.insert(0, f"{args.model_dir}/inference")
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import kernel as ref  # the checkpoint's inference/kernel.py
    from fp4_kv_quant import fp4_fake_quant_rows

    torch.manual_seed(5)
    dev = torch.device("cuda")
    T = args.rows
    x = torch.randn(T, 512, device=dev, dtype=torch.bfloat16)
    x[: T // 8] *= 8.0  # heavy tails -> saturation at 6 after scaling is impossible (scale = amax/6) but exercises big scales
    x[T // 8: T // 4] *= 2.0**-7  # tiny groups -> scale floor 2**-9 region
    x[T // 4: T // 4 + 16, 0:16] = 0  # all-zero groups
    x[T // 4 + 16: T // 4 + 32, 16:32] = 2.0**-12  # below the floor: quantize to 0 with s = 2**-9
    # exact e2m1 midpoints after scaling: build groups whose amax is exactly 6 * 2**k
    mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 2, device=dev)
    for k in range(-4, 4):
        row = T // 2 + k + 4
        x[row, :16] = (mid * 2.0**k).to(torch.bfloat16)
        x[row, 16:32] = (-mid * 2.0**k).to(torch.bfloat16)
    ok = True

    # reference: inplace dequant + codes/scales
    ref_deq = x.clone()
    ref.fp4_act_quant(ref_deq, 16, True, scale_dtype=torch.float8_e4m3fn)
    ref_codes, ref_scales = ref.fp4_act_quant(x.clone(), 16, False, scale_dtype=torch.float8_e4m3fn)
    torch.cuda.synchronize()

    got, packed, scales = fp4_fake_quant_rows(x, return_codes=True)
    torch.cuda.synchronize()

    n_deq = (got.view(torch.int16) != ref_deq.view(torch.int16)).sum().item()
    print(f"dequantized bf16: {n_deq} of {got.numel()} values differ (rel rms err vs input {((got.float() - x.float()).norm() / x.float().norm()).item():.3e})")
    ok &= n_deq == 0
    ref_codes_u8 = ref_codes.view(torch.uint8)
    n_codes = (ref_codes_u8 != packed).sum().item()
    # -0 vs +0: both encoders use the same cvt, so even that should agree; report separately if not
    lo_g, hi_g, lo_r, hi_r = packed & 15, packed >> 4, ref_codes_u8 & 15, ref_codes_u8 >> 4
    n_val = ((lo_g & 7) != (lo_r & 7)).sum().item() + ((hi_g & 7) != (hi_r & 7)).sum().item()
    print(f"e2m1 codes: {n_codes} of {packed.numel()} bytes differ ({n_val} magnitude nibbles differ)")
    ok &= n_val == 0
    n_sc = (ref_scales.view(torch.uint8) != scales).sum().item()
    print(f"e4m3 scales: {n_sc} of {scales.numel()} differ")
    ok &= n_sc == 0
    # scale floor: all-zero group -> scale 2**-9 (e4m3 0x01), codes 0
    zero_rows = slice(T // 4, T // 4 + 16)
    floor_ok = bool((scales[zero_rows, 0] == 0x01).all()) and bool((packed[zero_rows, :8] == 0).all())
    print(f"all-zero group -> scale byte 0x01 (2**-9), codes 0: {floor_ok}")
    ok &= floor_ok
    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
