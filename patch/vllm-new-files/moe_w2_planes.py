# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""2-bit tensor-sym expert planes for the 1-GPU DeepSeek-V4 plan.

Load-time GPU quantizer + fragment-major plane packer for the cubit
`moe_w2` decode kernel. The quantization is the QUANT_PROBE-validated
K=4 sign-symmetric codebook {-4, -1, 1, 4} (acceptance 2.73 vs 2.68
baseline): every mxfp4 e2m1 value maps to the nearest level with
odd-symmetric tie-breaking (zeros map sign-preservingly to +-1).

Mapping (e2m1 nibble -> 2-bit code), code order {0:-4, 1:-1, 2:+1, 3:+4}:
  +vals [0, .5, 1, 1.5, 2, 3, 4, 6] -> [+1 x5, +4 x3] -> codes [2,2,2,2,2,3,3,3]
  -vals (nibble | 8)                -> [-1 x5, -4 x3] -> codes [1,1,1,1,1,0,0,0]
Scales: the checkpoint's block-32 UE8M0 bytes are kept VERBATIM (the
kernel feeds them straight into QMMA.SF per k32).

Plane layout (fragment-major, per expert weight matrix [N, K]):
  for each 16-row block nb (N/16), for each k64 block kb (K/64),
  for each lane (g, t) in (8, 4):
    8 bytes = codes for the lane's QMMA fragment chunks, in order:
      [t0 k32a lo, t0 k32a hi, t0 k32b lo, t0 k32b hi,
       t1 k32a lo, t1 k32a hi, t1 k32b lo, t1 k32b hi]
    where t0 row = nb*16 + g, t1 row = nb*16 + g + 8,
          k32a = kb*64, k32b = kb*64 + 32,
          lo = weights [k + 4t .. 4t+3], hi = [k + 16 + 4t .. +3],
          each 4-weight chunk packs little-endian: code(k+4t) in bits 0-1.
  => plane bytes = N/16 * K/64 * 32 lanes * 8 = N*K/4.
"""

import torch

# e2m1 nibble -> 2-bit code (tensor-sym {-4,-1,1,4}), validated against
# tools/repack_expert_bits.py in tools/test_moe_w2_planes.py
_NIBBLE_TO_CODE = torch.tensor(
    [2, 2, 2, 2, 2, 3, 3, 3,   # +0,.5,1,1.5,2,3,4,6
     1, 1, 1, 1, 1, 0, 0, 0],  # -0,-.5,-1,-1.5,-2,-3,-4,-6
    dtype=torch.uint8)

# 2-bit code -> e2m1 nibble of the reconstructed level (for golden tests)
_CODE_TO_NIBBLE = torch.tensor([0xE, 0xA, 0x2, 0x6], dtype=torch.uint8)

# 2-bit code -> e4m3 byte (the kernel's PRMT LUT): -4,-1,1,4
PRMT_LUT_WORD = 0x4838B8C8


def mxfp4_to_codes(w_packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] u8 packed e2m1 pairs -> [..., K] u8 2-bit codes (0..3).

    Nibble order: low nibble = even k (matches mxfp4 packing).
    """
    lut = _NIBBLE_TO_CODE.to(w_packed.device)
    lo = lut[(w_packed & 0xF).long()]
    hi = lut[(w_packed >> 4).long()]
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def pack_fragment_major(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] u8 codes (0..3) -> fragment-major plane [N*K/4] u8."""
    N, K = codes.shape
    assert N % 16 == 0 and K % 64 == 0
    c = codes.view(N // 16, 2, 8, K // 64, 2, 2, 4, 4)
    # dims: nb, tile(g|g+8), g, kb, k32(a|b), half(lo|hi), t, k4
    #   row = nb*16 + tile*8 + g ; k = kb*64 + k32*32 + half*16 + t*4 + k4
    # target order: [nb, kb, g, t, tile, k32, half, k4]
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous()
    # pack 4 codes (k4) little-endian into one byte
    c = c.view(-1, 4).to(torch.int32)
    packed = (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6))
    return packed.to(torch.uint8).flatten()


def quantize_expert(w_packed: torch.Tensor) -> torch.Tensor:
    """mxfp4 [N, K/2] u8 -> fragment-major 2-bit plane [N*K/4] u8 (GPU)."""
    return pack_fragment_major(mxfp4_to_codes(w_packed))


def mxfp4_to_nibbles(w_packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] u8 packed e2m1 pairs -> [..., K] u8 raw nibbles (0..15)."""
    lo = w_packed & 0xF
    hi = w_packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def pack_fp4_fragment_major(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] u8 e2m1 nibbles -> fragment-major FP4 plane [N*K/2] u8.

    moe_w4_mm layout: per (nb, kb64, lane) 16 bytes = 4 words in order
    [t0 k32a, t0 k32b, t1 k32a, t1 k32b]; word nibbles 0-3 = lo quad
    (k = 4t+j), 4-7 = hi quad (k = 16+4t+j), little-endian.
    """
    N, K = codes.shape
    assert N % 16 == 0 and K % 64 == 0
    c = codes.view(N // 16, 2, 8, K // 64, 2, 2, 4, 4)
    # [nb, tile, g, kb, k32, half, t, j] -> [nb, kb, g, t, tile, k32, half, j]
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous()
    c = c.view(-1, 2).to(torch.int16)
    return (c[:, 0] | (c[:, 1] << 4)).to(torch.uint8).flatten()


def pack_scales(scales: torch.Tensor) -> torch.Tensor:
    """[N, K/32] u8 e8m0 -> kernel scale plane [N*K/32] u8.

    Layout: sbyte[nb, ks, r] at (nb*(K/32) + ks)*16 + r  (r = row in the
    16-row block); kernel lane (g,t) reads r=g (tile0) / r=8+g (tile1).
    """
    N, KS = scales.shape
    assert N % 16 == 0
    return scales.view(N // 16, 16, KS).transpose(1, 2).contiguous().flatten()


def reference_dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[N, K] codes + [N, K/32] e8m0 scale bytes -> f32 weights (golden ref)."""
    levels = torch.tensor([-4.0, -1.0, 1.0, 4.0], device=codes.device)
    vals = levels[codes.long()]
    s = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)
    return vals * s
