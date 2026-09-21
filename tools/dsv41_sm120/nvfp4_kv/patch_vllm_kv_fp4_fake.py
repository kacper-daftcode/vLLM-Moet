#!/usr/bin/env python3
"""Experiment "A0": store the compressed KV as fp8_ds_mla of the FP4-quantized latent.

DeepSeek's own inference keeps the compressed KV as bf16 of `fp4_act_quant(latent, 16,
inplace=True, scale_dtype=e4m3)` -- FP4 e2m1 values with one E4M3 scale per 16 dims, applied after
RoPE (the format the model was trained with). vLLM's `_rope_quant_insert_kernel` quantizes the
bf16 latent straight to the fp8_ds_mla record (448 fp8 with a UE8M0 scale per 64 + 64 bf16 RoPE
dims). Variant A of the NVFP4 main-KV plan stores the FP4 record (288 B) and re-quantizes it into
an fp8_ds_mla scratch for the SM120 attention kernels, so the values the kernel sees are
fp8(dequant(fp4(latent))) -- a double quantization. This patch produces exactly those values with
today's storage (no allocator change): when VLLM_MOET_KV_FP4_FAKE=1 the insert kernel runs the
reference FP4 quantizer (bit-exact port, tools/dsv41_sm120/nvfp4_kv/fp4_kv_quant.py) on the
RoPE'd latent before the fp8_ds_mla quantization. Off by default; SWA cache untouched.

Purpose: measure the quality of variant A's numerics end to end (GSM8K-200, needle, greedy
agreement) before building the storage side. Not applied in the image.

Usage: python3 patch_vllm_kv_fp4_fake.py [--file PATH] [--out PATH] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py"
)
MARKER = "[vllm-moet] A0: FP4(e2m1, e4m3/16) fake quant of the compressed latent"

HELPERS = f'''

# {MARKER} -- bit-exact port of the checkpoint's fp4_act_quant(x, 16, inplace, e4m3)
import os as _os

_KV_FP4_FAKE = _os.environ.get("VLLM_MOET_KV_FP4_FAKE", "0") == "1"


@triton.jit
def _moet_fp32x2_to_fp4x2(x_lo, x_hi):
    return tl.inline_asm_elementwise(
        """
        {{
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }}
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _moet_e2m1_code_to_f32(code):
    mag = code & 7
    sign = (code >> 3) & 1
    exp = (mag >> 1).to(tl.int32) - 1
    frac = 1.0 + 0.5 * (mag & 1).to(tl.float32)
    val = tl.exp2(exp.to(tl.float32)) * frac
    val = tl.where(mag == 1, 0.5, val)
    val = tl.where(mag == 0, 0.0, val)
    bits = val.to(tl.int32, bitcast=True) | (sign.to(tl.int32) << 31)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _moet_fp4_fake_quant_512(x):
    g = tl.reshape(x, (32, 16))
    amax = tl.maximum(tl.max(tl.abs(g), 1), 6.0 * (2.0**-9))
    s = (amax * (1.0 / 6.0)).to(tl.float8e4nv).to(tl.float32)
    scaled = tl.clamp(tl.math.div_rn(g, tl.broadcast_to(tl.reshape(s, (32, 1)), (32, 16))), -6.0, 6.0)
    even, odd = tl.split(tl.reshape(tl.reshape(scaled, (512,)), (256, 2)))
    packed = _moet_fp32x2_to_fp4x2(even, odd)
    lo = _moet_e2m1_code_to_f32(packed & 15)
    hi = _moet_e2m1_code_to_f32((packed >> 4) & 15)
    deq = tl.reshape(tl.interleave(lo, hi), (32, 16)) * tl.reshape(s, (32, 1))
    return tl.reshape(deq, (512,))
'''

# 1. launcher passes the switch
OLD_LAUNCH = (
    "            COMPRESS_RATIO=compress_ratio,\n"
    "            SANITIZE_CACHE_NANS=_ON_GFX950,\n"
    "            num_warps=4,\n"
    "            **launch_kwargs,\n"
    "        )\n"
    "        return\n"
)
NEW_LAUNCH = (
    "            COMPRESS_RATIO=compress_ratio,\n"
    "            SANITIZE_CACHE_NANS=_ON_GFX950,\n"
    "            FP4_FAKE=_KV_FP4_FAKE,\n"
    "            num_warps=4,\n"
    "            **launch_kwargs,\n"
    "        )\n"
    "        return\n"
)
# 2. kernel signature
OLD_SIG = (
    "    COMPRESS_RATIO: tl.constexpr,\n"
    "    SANITIZE_CACHE_NANS: tl.constexpr,\n"
    "):\n"
    "    t = tl.program_id(0)\n"
    "    slot = tl.load(cache_slots + t)\n"
    "    if slot < 0:\n"
    "        return\n"
    "    position = tl.load(positions + t)\n"
    "    if (position + 1) % COMPRESS_RATIO != 0:\n"
    "        return\n"
    "    d = tl.arange(0, 512)\n"
    "    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)\n"
    "    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE\n"
)
NEW_SIG = (
    "    COMPRESS_RATIO: tl.constexpr,\n"
    "    SANITIZE_CACHE_NANS: tl.constexpr,\n"
    "    FP4_FAKE: tl.constexpr = False,\n"
    "):\n"
    "    t = tl.program_id(0)\n"
    "    slot = tl.load(cache_slots + t)\n"
    "    if slot < 0:\n"
    "        return\n"
    "    position = tl.load(positions + t)\n"
    "    if (position + 1) % COMPRESS_RATIO != 0:\n"
    "        return\n"
    "    d = tl.arange(0, 512)\n"
    "    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)\n"
    "    if FP4_FAKE:\n"
    "        # reference order: RoPE on the bf16 latent (bf16 result), then fp4_act_quant over all 512 dims\n"
    "        even0, odd0 = tl.split(tl.reshape(normed, (256, 2)))\n"
    "        pair0 = tl.arange(0, 256) - 224\n"
    "        cs0 = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE\n"
    "        c0 = tl.load(cs0 + tl.maximum(pair0, 0), pair0 >= 0, other=1.0).to(tl.float32)\n"
    "        s0 = tl.load(cs0 + 32 + tl.maximum(pair0, 0), pair0 >= 0, other=0.0).to(tl.float32)\n"
    "        rot0 = tl.interleave(even0 * c0 - odd0 * s0, odd0 * c0 + even0 * s0)\n"
    "        rot0 = rot0.to(tl.bfloat16).to(tl.float32)\n"
    "        normed = _moet_fp4_fake_quant_512(tl.where(d < 448, normed, rot0))\n"
    "    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE\n"
)
# 3. RoPE part: with the fake quant the rotated dims are already in `normed`
OLD_ROPE = (
    "    even, odd = tl.split(tl.reshape(normed, (256, 2)))\n"
    "    pair = tl.arange(0, 256) - 224\n"
    "    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE\n"
    "    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)\n"
    "    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)\n"
    "    rotated = tl.interleave(even * c - odd * s, odd * c + even * s)\n"
    "    if SANITIZE_CACHE_NANS:\n"
    "        rotated = tl.where(rotated == rotated, rotated, 0.0)\n"
    "    rope_dst = (values + 448).to(tl.pointer_type(tl.bfloat16))\n"
    "    tl.store(rope_dst + d - 448, rotated.to(tl.bfloat16), d >= 448)\n"
)
NEW_ROPE = (
    "    if FP4_FAKE:\n"
    "        rotated = normed\n"
    "    else:\n"
    "        even, odd = tl.split(tl.reshape(normed, (256, 2)))\n"
    "        pair = tl.arange(0, 256) - 224\n"
    "        cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE\n"
    "        c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)\n"
    "        s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)\n"
    "        rotated = tl.interleave(even * c - odd * s, odd * c + even * s)\n"
    "    if SANITIZE_CACHE_NANS:\n"
    "        rotated = tl.where(rotated == rotated, rotated, 0.0)\n"
    "    rope_dst = (values + 448).to(tl.pointer_type(tl.bfloat16))\n"
    "    tl.store(rope_dst + d - 448, rotated.to(tl.bfloat16), d >= 448)\n"
)


def patch_text(src: str) -> str:
    if MARKER in src:
        return src
    for old, new in ((OLD_LAUNCH, NEW_LAUNCH), (OLD_SIG, NEW_SIG), (OLD_ROPE, NEW_ROPE)):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"anchor found {n} times (expected 1):\n{old}")
        src = src.replace(old, new, 1)
    return src + HELPERS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=None, help="write here instead of in place")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    if MARKER in src:
        print(f"{args.file}: already patched")
        return 0
    patched = patch_text(src)
    if args.check:
        print(f"{args.file}: patch applies cleanly (not written)")
        return 0
    compile(patched, str(args.file), "exec")
    out = args.out or args.file
    out.write_text(patched)
    print(f"{out}: patched (VLLM_MOET_KV_FP4_FAKE=1 enables the FP4 fake quant of the compressed KV)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
