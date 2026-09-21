# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1's compressed-KV quantizer (FP4 e2m1 values, one E4M3 scale per 16) in Triton.

Reference: `inference/kernel.py::fp4_act_quant(x, 16, inplace, scale_dtype=e4m3)` of the released
checkpoint (`fp4_quant_kernel`, TileLang):

    amax = max(max|x_group|, 6 * 2**-9)      # even an all-zero group keeps a nonzero scale
    s    = fp32(e4m3(amax / 6))              # round-to-nearest e4m3, 2**-9 is e4m3's smallest subnormal
    y    = e2m1(clamp(x / s, -6, 6))         # cvt.rn.satfinite.e2m1x2.f32 (RNE)
    x'   = bf16(y * s)                       # inplace=True: what the reference keeps in its KV cache

`fp4_quant_group16` returns the codes and scales the way an NVFP4 record stores them (vLLM main's
`nvfp4_ds_mla`: 256 B of packed e2m1 pairs + 32 e4m3 bytes per 512-dim state);
`fp4_dequant_group16` inverts them exactly (y * s has <= 6 significant bits, so the bf16 round trip
is lossless); `fp4_fake_quant_512` is the in-place quant -> dequant the reference applies to the
latent before writing its cache -- used to measure variant A's numerics with today's fp8_ds_mla
storage (the "A0" experiment) without touching the allocator.

All helpers are `@triton.jit` device functions over a [512] fp32 vector (one compressed state);
`fp4_fake_quant_rows` is a host-callable kernel for tests.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP4_MAX = 6.0
FP4_GROUP = 16
FP4_SCALE_FLOOR = 6.0 * 2.0**-9  # reference: amax floor for the e4m3-scaled compressed KV


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    """Two fp32 -> one byte of e2m1 codes (low nibble = x_lo), cvt.rn.satfinite (RNE, saturate at 6)."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _e2m1_code_to_f32(code):
    """4-bit e2m1 code (sign<<3 | mag) -> fp32 value; mag: 0 .5 1 1.5 2 3 4 6."""
    mag = code & 7
    sign = (code >> 3) & 1
    exp = (mag >> 1).to(tl.int32) - 1  # mag 2,3 -> 0; 4,5 -> 1; 6,7 -> 2
    frac = 1.0 + 0.5 * (mag & 1).to(tl.float32)
    val = tl.exp2(exp.to(tl.float32)) * frac
    val = tl.where(mag == 1, 0.5, val)
    val = tl.where(mag == 0, 0.0, val)
    # set the sign bit directly so code 8 decodes to -0.0 like the reference (fsub 0-x would give +0)
    bits = val.to(tl.int32, bitcast=True) | (sign.to(tl.int32) << 31)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def fp4_quant_group16(x):
    """x: fp32 [512] (one state). Returns (packed uint8 [256], scale_bytes uint8 [32], dequant fp32 [512]).

    packed[i] holds x[2i] (low nibble) and x[2i+1] (high nibble); scale_bytes[g] is the e4m3
    bit pattern of the scale of group g = dims 16g..16g+15."""
    g = tl.reshape(x, (32, 16))
    amax = tl.maximum(tl.max(tl.abs(g), 1), 6.0 * (2.0**-9))
    s_fp8 = (amax * (1.0 / 6.0)).to(tl.float8e4nv)
    s = s_fp8.to(tl.float32)
    # IEEE division like the reference's CUDA `x / s` (div.rn.f32); Triton's `/` is div.full.f32 and
    # x * (1/s) is off by an ulp as well -- either flips ~0.04 % of the codes at e2m1 midpoints
    scaled = tl.clamp(tl.math.div_rn(g, tl.broadcast_to(tl.reshape(s, (32, 1)), (32, 16))), -6.0, 6.0)
    flat = tl.reshape(scaled, (512,))
    even, odd = tl.split(tl.reshape(flat, (256, 2)))
    packed = _fp32x2_to_fp4x2(even, odd)
    lo = _e2m1_code_to_f32(packed & 15)
    hi = _e2m1_code_to_f32((packed >> 4) & 15)
    deq = tl.reshape(tl.interleave(lo, hi), (32, 16)) * tl.reshape(s, (32, 1))
    return packed, s_fp8.to(tl.uint8, bitcast=True), tl.reshape(deq, (512,))


@triton.jit
def fp4_dequant_group16(packed, scale_bytes):
    """Inverse of fp4_quant_group16 without the values: packed uint8 [256], scale_bytes uint8 [32] -> fp32 [512]."""
    lo = _e2m1_code_to_f32(packed & 15)
    hi = _e2m1_code_to_f32((packed >> 4) & 15)
    s = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    vals = tl.reshape(tl.interleave(lo, hi), (32, 16)) * tl.reshape(s, (32, 1))
    return tl.reshape(vals, (512,))


@triton.jit
def fp4_fake_quant_512(x):
    """quant -> dequant of one 512-dim state (fp32 in, fp32 out; values are exactly bf16-representable)."""
    _, _, deq = fp4_quant_group16(x)
    return deq


@triton.jit
def _fp4_fake_quant_rows_kernel(x_ptr, y_ptr, packed_ptr, scale_ptr, WRITE_CODES: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 512)
    x = tl.load(x_ptr + t * 512 + d).to(tl.float32)
    packed, sb, deq = fp4_quant_group16(x)
    tl.store(y_ptr + t * 512 + d, deq.to(tl.bfloat16))
    if WRITE_CODES:
        tl.store(packed_ptr + t * 256 + tl.arange(0, 256), packed)
        tl.store(scale_ptr + t * 32 + tl.arange(0, 32), sb)


def fp4_fake_quant_rows(x: torch.Tensor, return_codes: bool = False):
    """Host wrapper: x bf16 [T, 512] -> dequantized bf16 [T, 512] (and the NVFP4 codes/scales)."""
    assert x.ndim == 2 and x.shape[1] == 512 and x.dtype == torch.bfloat16 and x.is_contiguous()
    y = torch.empty_like(x)
    packed = torch.empty((x.shape[0], 256), dtype=torch.uint8, device=x.device) if return_codes else y
    scales = torch.empty((x.shape[0], 32), dtype=torch.uint8, device=x.device) if return_codes else y
    _fp4_fake_quant_rows_kernel[(x.shape[0],)](x, y, packed, scales, WRITE_CODES=return_codes, num_warps=4)
    return (y, packed, scales) if return_codes else y
