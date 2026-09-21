# SPDX-License-Identifier: Apache-2.0
"""Variant A of the FP4 main-KV plan for sm_120: the compressed KV lives in the packed record the model
was trained with, and the SM120 sparse-MLA kernels (which read fp8_ds_mla only) get an fp8_ds_mla
scratch of exactly the states a step attends to.

Records (per compressed state, paged; page = [PBS x values][PBS x scales], same segregation as
vLLM's fp8_ds_mla and vLLM main's nvfp4_ds_mla / V4.1 mxfp8 records):

  288 B "nvfp4"   : 256 B of e2m1 pairs (all 512 dims, RoPE'd) + 32 e4m3 scales, one per 16 dims --
                    DeepSeek's `fp4_act_quant(latent, 16, e4m3)` after RoPE (inference/model.py::_compress_kv)
  528 B "fp8_v41" : 512 fp8 e4m3 (all 512 dims, RoPE'd) + 16 UE8M0 scales, one per 32 dims --
                    DeepSeek's `act_quant(kv, 32, "ue8m0")` (the V4.1 sliding-window record)
  584 B fp8_ds_mla: 448 fp8 e4m3 NoPE with 7 UE8M0 scales per 64 (+1 pad) and 64 bf16 RoPE dims --
                    what vLLM 0909 stores today and what FlashInfer's SM120 DSV4 kernels read

Kernels:
  rope_quant_insert_packed(latent, positions, cos_sin, kv_cache, slot_mapping, compress_ratio)
      RoPE (GPT-J, last 64 dims, bf16 result like the reference) + quantize + store one record;
      the record width is taken from kv_cache.shape[-1] (288 or 528).
  gather_requant_to_ds_mla(src_cache, indices, scratch) -> remapped indices
      For every (row t, slot j) with indices[t, j] >= 0: read record indices[t, j] from the packed
      cache, dequantize to fp32, re-quantize to fp8_ds_mla and write it to scratch state t*topk+j
      (scratch is a paged fp8_ds_mla cache of 128-state pages, contiguous). Returns the index
      tensor the attention kernel should use instead (t*topk+j, -1 kept). The re-quantization is
      vLLM's own `_rope_quant_insert_kernel` recipe applied to the dequantized values, so the scratch
      holds exactly what experiment A0 (patch_vllm_kv_fp4_fake.py) writes for the same latent.
      Used for decode (rows x 512 records; one gather per (kv source, index set) per step).
  dequant_context_to_ds_mla(src_cache, block_table_row, num_states, scratch)
      One request's whole compressed context in logical order (state i -> scratch slot i), same
      re-quantization. Used for prefill, where every query row of a 4096-token chunk would otherwise
      gather its own 512 records (38 layers x 1.8 GB per chunk); the context is dequantized once per
      kv source and addressed with the request-local top-k indices.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fp4_kv_quant import _e2m1_code_to_f32, _fp32x2_to_fp4x2

DS_MLA_BYTES = 584
DS_MLA_PAGE = 128


# --------------------------------------------------------------------------------------- insert
@triton.jit
def _rope_bf16_512(latent_ptr, t, positions_ptr, cos_sin, COS_STRIDE: tl.constexpr, COMPRESS_RATIO: tl.constexpr):
    """RoPE'd latent as fp32 [512] with bf16-rounded RoPE dims (reference rotates a bf16 tensor in place)."""
    d = tl.arange(0, 512)
    normed = tl.load(latent_ptr + t.to(tl.int64) * 512 + d).to(tl.float32)
    position = tl.load(positions_ptr + t)
    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    rot = tl.interleave(even * c - odd * s, odd * c + even * s).to(tl.bfloat16).to(tl.float32)
    return tl.where(d < 448, normed, rot)


@triton.jit
def _rope_quant_insert_packed_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    RECORD: tl.constexpr,  # 288 or 528
):
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    x = _rope_bf16_512(latent, t, positions, cos_sin, COS_STRIDE, COMPRESS_RATIO)
    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    pos = slot % CACHE_BLOCK
    if RECORD == 288:
        g = tl.reshape(x, (32, 16))
        amax = tl.maximum(tl.max(tl.abs(g), 1), 6.0 * (2.0**-9))
        s_fp8 = (amax * (1.0 / 6.0)).to(tl.float8e4nv)
        scaled = tl.clamp(tl.math.div_rn(g, tl.broadcast_to(tl.reshape(s_fp8.to(tl.float32), (32, 1)), (32, 16))), -6.0, 6.0)
        even, odd = tl.split(tl.reshape(tl.reshape(scaled, (512,)), (256, 2)))
        tl.store(page + pos * 256 + tl.arange(0, 256), _fp32x2_to_fp4x2(even, odd))
        tl.store(page + CACHE_BLOCK * 256 + pos * 32 + tl.arange(0, 32), s_fp8.to(tl.uint8, bitcast=True))
    else:
        # act_quant(kv, 32, scale_fmt="ue8m0"): amax floor 1e-4, scale 2^ceil(log2(amax/448)), fp8 RN
        g = tl.reshape(x, (16, 32))
        amax = tl.maximum(tl.max(tl.abs(g), 1), 1e-4)
        exponent = tl.ceil(tl.log2(amax * (1.0 / 448.0)))
        scaled = tl.clamp(g * tl.reshape(tl.exp2(-exponent), (16, 1)), -448.0, 448.0)
        fp8 = tl.reshape(scaled.to(tl.float8e4nv).to(tl.uint8, bitcast=True), (512,))
        tl.store(page + pos * 512 + tl.arange(0, 512), fp8)
        encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0)
        tl.store(page + CACHE_BLOCK * 512 + pos * 16 + tl.arange(0, 16), encoded.to(tl.uint8))


def rope_quant_insert_packed(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
) -> None:
    assert compress_ratio in (1, 2)
    assert latent.shape[1] == 512 and latent.dtype == torch.bfloat16 and latent.is_contiguous()
    assert kv_cache.dtype == torch.uint8 and kv_cache.ndim == 3 and kv_cache.shape[-1] in (288, 528)
    num_tokens = slot_mapping.numel()
    assert num_tokens <= min(latent.shape[0], positions.numel())
    if num_tokens == 0:
        return
    _rope_quant_insert_packed_kernel[(num_tokens,)](
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        COS_STRIDE=cos_sin_cache.stride(0),
        CACHE_STRIDE=kv_cache.stride(0),
        CACHE_BLOCK=kv_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        RECORD=kv_cache.shape[-1],
        num_warps=4,
    )


# --------------------------------------------------------------------------------------- gather
@triton.jit
def _dequant_record_512(page, pos, CACHE_BLOCK: tl.constexpr, RECORD: tl.constexpr):
    """One packed record -> fp32 [512]."""
    if RECORD == 288:
        packed = tl.load(page + pos * 256 + tl.arange(0, 256))
        sb = tl.load(page + CACHE_BLOCK * 256 + pos * 32 + tl.arange(0, 32))
        lo = _e2m1_code_to_f32(packed & 15)
        hi = _e2m1_code_to_f32((packed >> 4) & 15)
        s = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        vals = tl.reshape(tl.interleave(lo, hi), (32, 16)) * tl.reshape(s, (32, 1))
        return tl.reshape(vals, (512,))
    else:
        fp8 = tl.load(page + pos * 512 + tl.arange(0, 512)).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        e = tl.load(page + CACHE_BLOCK * 512 + pos * 16 + tl.arange(0, 16)).to(tl.float32) - 127.0
        vals = tl.reshape(fp8, (16, 32)) * tl.reshape(tl.exp2(e), (16, 1))
        return tl.reshape(vals, (512,))


@triton.jit
def _store_ds_mla(x, page, pos, CACHE_BLOCK: tl.constexpr):
    """vLLM's fp8_ds_mla record of fp32 [512] (RoPE dims already bf16-valued): 448 fp8/UE8M0-64 + 64 bf16."""
    d = tl.arange(0, 512)
    values = page + pos * 576
    scales = page + CACHE_BLOCK * 576 + pos * 8
    quant = tl.reshape(x, (8, 64))
    amax = tl.maximum(tl.max(tl.abs(quant), 1), 1e-4)
    exponent = tl.ceil(tl.log2(amax * (1.0 / 448.0)))
    scaled = quant * tl.reshape(tl.exp2(-exponent), (8, 1))
    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
    packed = tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,))
    tl.store(values + d, packed, d < 448)
    s = tl.arange(0, 8)
    encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0)
    tl.store(scales + s, encoded.to(tl.uint8), s < 7)
    tl.store(scales + 7, tl.full((), 0, tl.uint8))
    rope_dst = (values + 448).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_dst + d - 448, x.to(tl.bfloat16), d >= 448)


@triton.jit
def _gather_requant_kernel(
    src,
    indices,
    out_indices,
    scratch,
    TOPK: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    SRC_BLOCK: tl.constexpr,
    RECORD: tl.constexpr,
    SCRATCH_STRIDE: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)  # = t * TOPK + j
    idx = tl.load(indices + r)
    if idx < 0:
        tl.store(out_indices + r, -1)
        return
    tl.store(out_indices + r, r.to(tl.int32))
    src_page = src + (idx // SRC_BLOCK).to(tl.int64) * SRC_STRIDE
    x = _dequant_record_512(src_page, idx % SRC_BLOCK, SRC_BLOCK, RECORD)
    dst_page = scratch + (r // 128) * SCRATCH_STRIDE
    _store_ds_mla(x, dst_page, (r % 128).to(tl.int32), 128)


@triton.jit
def _dequant_context_kernel(
    src,
    block_table_row,
    scratch,
    num_states,
    SRC_STRIDE: tl.constexpr,
    SRC_BLOCK: tl.constexpr,
    RECORD: tl.constexpr,
    SCRATCH_STRIDE: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)  # request-local compressed position = scratch state
    if i >= num_states:
        return
    blk = tl.load(block_table_row + i // SRC_BLOCK)
    src_page = src + blk.to(tl.int64) * SRC_STRIDE
    x = _dequant_record_512(src_page, (i % SRC_BLOCK).to(tl.int32), SRC_BLOCK, RECORD)
    dst_page = scratch + (i // 128) * SCRATCH_STRIDE
    _store_ds_mla(x, dst_page, (i % 128).to(tl.int32), 128)


def dequant_context_to_ds_mla(
    src_cache: torch.Tensor,  # [P, PBS, 288|528] uint8
    block_table_row: torch.Tensor,  # [max_blocks] int32, one request's block table
    num_states: int,  # compressed states of the request present in the cache
    scratch: torch.Tensor,  # [>= ceil(num_states/128), 128, 584] uint8
) -> None:
    """Prefill path: one request's whole compressed context, dequantized + re-quantized into an
    fp8_ds_mla scratch in logical order (state i -> scratch slot i), so the request-local top-k
    indices address the scratch directly."""
    assert src_cache.dtype == torch.uint8 and src_cache.ndim == 3 and src_cache.shape[-1] in (288, 528)
    assert block_table_row.dtype == torch.int32 and block_table_row.ndim == 1 and block_table_row.is_contiguous()
    assert scratch.dtype == torch.uint8 and scratch.shape[1:] == (DS_MLA_PAGE, DS_MLA_BYTES) and scratch.is_contiguous()
    assert scratch.shape[0] * DS_MLA_PAGE >= num_states, (scratch.shape, num_states)
    if num_states <= 0:
        return
    assert block_table_row.numel() * src_cache.shape[1] >= num_states
    _dequant_context_kernel[(num_states,)](
        src_cache,
        block_table_row,
        scratch,
        num_states,
        SRC_STRIDE=src_cache.stride(0),
        SRC_BLOCK=src_cache.shape[1],
        RECORD=src_cache.shape[-1],
        SCRATCH_STRIDE=scratch.stride(0),
        num_warps=4,
    )


def scratch_pages(rows: int, topk: int) -> int:
    return (rows * topk + DS_MLA_PAGE - 1) // DS_MLA_PAGE


def gather_requant_to_ds_mla(
    src_cache: torch.Tensor,  # [P, PBS, 288|528] uint8 (block stride from .stride(0))
    indices: torch.Tensor,  # [rows, topk] int32 global slots into src_cache, -1 = none
    scratch: torch.Tensor,  # [>= scratch_pages(rows, topk), 128, 584] uint8, contiguous
) -> torch.Tensor:
    assert src_cache.dtype == torch.uint8 and src_cache.ndim == 3 and src_cache.shape[-1] in (288, 528)
    assert indices.dtype == torch.int32 and indices.ndim == 2 and indices.is_contiguous()
    rows, topk = indices.shape
    assert scratch.dtype == torch.uint8 and scratch.shape[1:] == (DS_MLA_PAGE, DS_MLA_BYTES) and scratch.is_contiguous()
    assert scratch.shape[0] >= scratch_pages(rows, topk), (scratch.shape, rows, topk)
    out = torch.empty_like(indices)
    n = rows * topk
    if n == 0:
        return out
    _gather_requant_kernel[(n,)](
        src_cache,
        indices,
        out,
        scratch,
        TOPK=topk,
        SRC_STRIDE=src_cache.stride(0),
        SRC_BLOCK=src_cache.shape[1],
        RECORD=src_cache.shape[-1],
        SCRATCH_STRIDE=scratch.stride(0),
        num_warps=4,
    )
    return out
