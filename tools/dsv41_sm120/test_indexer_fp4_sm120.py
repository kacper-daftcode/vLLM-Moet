#!/usr/bin/env python3
"""Op-level validation of vLLM's MXFP4 indexer path on sm_120 (DeepSeek-V4.1).

vLLM 0909 gates `--attention-config '{"indexer_kv_dtype":"mxfp4"}'` on sm_10x
(patch_vllm_indexer_fp4_sm120.py lifts it). Everything the path runs is checked
here on one SM120 GPU, against fp32 torch references written from the kernels'
documented semantics and DeepSeek's `fp4` quantizer (UE8M0 block scale
2^ceil(log2(amax/6)) per 32 values, e2m1 round-to-nearest-even, saturate at 6):

  Q      fused_indexer_q_rope_quant(use_fp4=True) (CuTe DSL or Triton, whichever
         the image picks): packed e2m1 nibbles, UE8M0 scales and the weight fold
         (weights * softmax_scale * head_scale, no per-token q scale) match the
         reference bit-for-bit, RoPE on the last 64 dims (GPT-J interleaved,
         bf16 round trip), NoPE blocks straight from bf16;
  K      indexer_k_norm_rope_store(use_fp4_cache=True) (Triton): k_norm -> RoPE at
         the group position -> MXFP4 -> segregated paged store (values, then
         scales), only group-boundary tokens, compress_ratio 1 and 2;
  LOGITS DeepGEMM fp8_fp4_paged_mqa_logits on the kernel-written cache + Q with
         128-key pages (patch_deepgemm.py): vs the dequantized reference < 1e-3
         (bit-exact re-page 64 vs 128), and vs the unquantized bf16 logits,
         side by side with the FP8 indexer path on the same inputs.

e2m1 ties (|x/s| exactly on a midpoint) round to even in hardware and are
reported separately; -0 (the cvt keeps the sign of values that round to zero)
counts as 0; everything else must match.

Run inside the serving image on one SM120 GPU:
  python3 test_indexer_fp4_sm120.py [--dg-module vllm.third_party.deep_gemm]
"""
from __future__ import annotations

import argparse
import importlib
import math
import sys

import torch

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def e2m1_codes(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 -> 4-bit e2m1 codes (sign<<3 | magnitude), RNE, satfinite. Also returns the tie mask."""
    ax = x.abs().clamp(max=6.0)
    mids = E2M1_MIDPOINTS.to(x.device)
    # bucketize(right=True) = number of midpoints <= |x|: the nearest code, with a value exactly on a
    # midpoint going up; round-to-nearest-even wants the even code there instead.
    code = torch.bucketize(ax, mids, right=True)
    tie = torch.isin(ax, mids)
    upper = code.clone()
    code = torch.where(tie & (upper % 2 == 1), upper - 1, upper)
    code = code.to(torch.uint8)
    neg = (x < 0) & (code != 0)
    return code | (neg.to(torch.uint8) << 3), tie


def ue8m0_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = torch.maximum(amax, torch.full_like(amax, 6.0 * 2.0**-126))
    log2r = torch.ceil(torch.log2(amax * (1.0 / 6.0))).clamp(-127.0, 127.0)
    return torch.exp2(log2r), (log2r + 127.0).to(torch.uint8)


def quantize_block32(x: torch.Tensor):
    """x fp32 [..., 32] -> (packed uint8 [..., 16], ue8m0 uint8 [...], tie mask [..., 32])."""
    amax = x.abs().amax(dim=-1)
    scale, ue8m0 = ue8m0_scale(amax)
    codes, tie = e2m1_codes(x * (1.0 / scale).unsqueeze(-1))
    lo, hi = codes[..., 0::2], codes[..., 1::2]
    return (lo | (hi << 4)), ue8m0, tie


def dequant(packed: torch.Tensor, ue8m0: torch.Tensor) -> torch.Tensor:
    """packed uint8 [..., 64] + ue8m0 [..., 4] -> fp32 [..., 128]."""
    lo, hi = packed & 0x0F, (packed >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).flatten(-2)  # [..., 128]
    vals = E2M1_VALUES.to(packed.device)[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -vals, vals)
    scales = torch.exp2(ue8m0.float() - 127.0)  # [..., 4]
    return vals * scales.repeat_interleave(32, dim=-1)


def rope_gptj(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [..., 64] fp32 interleaved pairs; cos/sin [..., 32]. bf16 round trip like the kernels."""
    even, odd = x[..., 0::2], x[..., 1::2]
    r_even = (even * cos - odd * sin).to(torch.bfloat16).float()
    r_odd = (odd * cos + even * sin).to(torch.bfloat16).float()
    return torch.stack([r_even, r_odd], dim=-1).flatten(-2)


def ref_q(q: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, weights: torch.Tensor, sm_scale: float, head_scale: float):
    """fused_indexer_q_rope_quant(use_fp4=True) reference. q bf16 [T, H, 128]."""
    x = q.float()
    cs = cos_sin[positions.long()].float()  # [T, 64]
    cos, sin = cs[:, :32].unsqueeze(1), cs[:, 32:].unsqueeze(1)
    rot = rope_gptj(x[..., 64:], cos, sin)
    full = torch.cat([x[..., :64], rot], dim=-1)  # [T, H, 128]
    packed, ue8m0, tie = quantize_block32(full.view(*full.shape[:-1], 4, 32))
    packed = packed.flatten(-2)  # [T, H, 64]
    w_out = (weights.float() * sm_scale) * head_scale
    return packed, ue8m0, tie.flatten(-2), w_out, full


def ref_k(k_pre: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, w: torch.Tensor, eps: float, cr: int):
    """indexer_k_norm_rope_store(use_fp4_cache=True) reference. Returns per-token packed/scales/tie and the fp32 keys."""
    k = k_pre.float()
    var = (k * k).mean(dim=-1, keepdim=True)
    k = (k * torch.rsqrt(var + eps) * w.float()).to(torch.bfloat16).float()
    cpos = (positions // cr) * cr
    cs = cos_sin[cpos.long()].float()
    rot = rope_gptj(k[..., 64:], cs[:, :32], cs[:, 32:])
    full = torch.cat([k[..., :64], rot], dim=-1)
    packed, ue8m0, tie = quantize_block32(full.view(-1, 4, 32))
    return packed.flatten(-2), ue8m0, tie.flatten(-2), full


def ref_logits(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor, ctx: int) -> torch.Tensor:
    """q [next_n, H, 128], k [ctx, 128] fp32, weights [next_n, H] -> [next_n, ctx] (2-D context lens: all rows see ctx keys)."""
    s = torch.einsum("nhd,kd->nhk", q, k)
    return (torch.relu(s) * weights.unsqueeze(-1)).sum(dim=1)


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    return 1 - (2 * (x * y).sum() / denominator).item()


def compare_packed(name: str, got_p, got_s, ref_p, ref_s, tie) -> bool:
    ok = True
    if not torch.equal(got_s, ref_s):
        n = (got_s != ref_s).sum().item()
        print(f"  [BAD] {name}: {n} UE8M0 scale mismatches")
        ok = False
    def nibbles(p):
        n = torch.stack([p & 0x0F, (p >> 4) & 0x0F], dim=-1).flatten(-2)
        # the hardware cvt keeps the sign of values that round to zero (-0 = code 8); same value
        return torch.where((n & 7) == 0, torch.zeros_like(n), n)

    mism = nibbles(got_p) != nibbles(ref_p)  # per element
    n_mism = mism.sum().item()
    n_tie = tie.sum().item()
    n_bad = (mism & ~tie).sum().item()
    print(f"  {'[ok ]' if n_bad == 0 else '[BAD]'} {name}: {mism.numel()} e2m1 values, {n_mism} differ ({n_tie} exact midpoints), {n_bad} off-tie mismatches")
    return ok and n_bad == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dg-module", default="vllm.third_party.deep_gemm")
    ap.add_argument("--tokens", type=int, default=4096)
    args = ap.parse_args()
    torch.manual_seed(3)
    dev = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")

    from vllm.models.deepseek_v4_1.common.ops import (
        MXFP4_BLOCK_SIZE,
        fused_indexer_q_rope_quant,
        indexer_k_norm_rope_store,
    )
    from vllm.utils.import_utils import has_cutedsl

    assert MXFP4_BLOCK_SIZE == 32
    T, H, D, ROPE = args.tokens, 32, 128, 64
    # GPT-J cos/sin cache [max_pos, 64] = cos(32) | sin(32), rope theta of the compress RoPE
    max_pos = T + 16
    inv_freq = 1.0 / (160000.0 ** (torch.arange(0, ROPE, 2, device=dev).float() / ROPE))
    freqs = torch.outer(torch.arange(max_pos, device=dev).float(), inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)
    positions = torch.arange(T, device=dev, dtype=torch.int64)
    all_ok = True

    # ---------------- Q ----------------
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16) * 3
    weights = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    sm_scale, head_scale = D ** -0.5, H ** -0.5
    (q_packed, q_scale_i32), w_out = fused_indexer_q_rope_quant(positions, q, cos_sin, weights, sm_scale, head_scale, use_fp4=True)
    torch.cuda.synchronize()
    q_scale = q_scale_i32.view(torch.uint8).view(T, H, 4) if q_scale_i32.dtype == torch.int32 else q_scale_i32
    rq_packed, rq_scale, rq_tie, rw_out, q_full = ref_q(q, positions, cos_sin, weights, sm_scale, head_scale)
    print(f"Q kernel backend: {'CuTe DSL' if has_cutedsl() else 'Triton'}; packed {tuple(q_packed.shape)} {q_packed.dtype}, scales {tuple(q_scale_i32.shape)} {q_scale_i32.dtype}")
    all_ok &= compare_packed("Q e2m1 + UE8M0", q_packed, q_scale, rq_packed, rq_scale, rq_tie)
    w_ok = torch.allclose(w_out, rw_out, rtol=0, atol=0)
    if not w_ok:
        w_ok = torch.allclose(w_out, rw_out, rtol=1e-6, atol=0)
        print(f"  weights fold: not bit-exact, max rel {((w_out - rw_out).abs() / rw_out.abs().clamp_min(1e-30)).max().item():.2e}")
    print(f"  {'[ok ]' if w_ok else '[BAD]'} weights_out = weights * softmax_scale * head_scale")
    all_ok &= w_ok
    # dequantized Q vs fp32 RoPE'd Q: FP4 block quantization error (information only)
    q_deq = dequant(q_packed, q_scale)
    print(f"  info: Q dequant rel err (rms) {((q_deq - q_full).norm() / q_full.norm()).item():.3e}")

    # ---------------- K store ----------------
    k_w = (1 + 0.1 * torch.randn(D, device=dev)).to(torch.bfloat16)
    eps = 1e-20
    dg = importlib.import_module(args.dg_module)
    for cr in (1, 2):
        k_pre = torch.randn(T, D, device=dev, dtype=torch.bfloat16) * 2
        num_states = T // cr
        page = 128
        num_pages = (num_states + page - 1) // page + 2
        row_bytes = D // 2 + D // 32  # 68
        k_cache = torch.randint(0, 256, (num_pages, page, row_bytes), device=dev, dtype=torch.uint8)  # noise to catch missing writes
        # shuffled page table; slot of token t (group boundary) = page_of(t // cr) * 128 + (t // cr) % 128; others -1
        perm = torch.randperm(num_pages, device=dev)[: (num_states + page - 1) // page]
        state = positions // cr
        slots = perm[(state // page).long()] * page + state % page
        is_boundary = (positions + 1) % cr == 0
        slot_mapping = torch.where(is_boundary, slots, torch.full_like(slots, -1))
        slot_mapping[7] = -1  # a skipped token (padding) even at a boundary
        indexer_k_norm_rope_store(k_pre, positions, cos_sin, k_w, eps, k_cache, slot_mapping, cr, True)
        torch.cuda.synchronize()
        rk_packed, rk_scale, rk_tie, k_full = ref_k(k_pre, positions, cos_sin, k_w, eps, cr)
        written = slot_mapping >= 0
        sl = slot_mapping[written]
        pg, pos_in = sl // page, sl % page
        flat = k_cache.view(num_pages, -1)
        got_vals = flat[pg, :page * 64].view(-1, page, 64)[torch.arange(sl.numel(), device=dev), pos_in]
        got_sf = flat[pg, page * 64:].view(-1, page, 4)[torch.arange(sl.numel(), device=dev), pos_in]
        all_ok &= compare_packed(f"K store cr={cr} (segregated page layout)", got_vals, got_sf, rk_packed[written], rk_scale[written], rk_tie[written])

        # fill the hole left by the skipped token so the cache holds every state of the request
        fill = torch.full_like(slot_mapping, -1)
        fill[7] = slots[7]
        indexer_k_norm_rope_store(k_pre, positions, cos_sin, k_w, eps, k_cache, fill, cr, True)
        torch.cuda.synchronize()
        written = is_boundary
        sl = slot_mapping.clone()
        sl[7] = slots[7]
        sl = sl[written]
        pg, pos_in = sl // page, sl % page
        got_vals = flat[pg, :page * 64].view(-1, page, 64)[torch.arange(sl.numel(), device=dev), pos_in]
        got_sf = flat[pg, page * 64:].view(-1, page, 4)[torch.arange(sl.numel(), device=dev), pos_in]

        # ---------------- logits through DeepGEMM on the kernel-written cache ----------------
        # one request, ctx = number of states of this request; keys in state order
        state_ids = state[written]
        order = torch.argsort(state_ids)
        ctx = int(state_ids.numel())
        assert ctx == num_states
        next_n = 6
        q_rows = q_packed[-next_n:].contiguous()  # [next_n, H, 64] uint8
        q_sf_rows = q_scale_i32[-next_n:].contiguous().view(1, next_n, H)
        w_rows = w_out[-next_n:].contiguous()
        # block table: logical page j -> physical perm[j] (the pages the store kernel used)
        block_table = perm.to(torch.int32).view(1, -1).contiguous()
        ctx2d = torch.full((1, next_n), ctx, device=dev, dtype=torch.int32)
        kv_view = torch.as_strided(k_cache, size=(num_pages, page, 1, row_bytes), stride=(page * row_bytes, row_bytes, row_bytes, 1))
        meta = dg.get_paged_mqa_logits_metadata(ctx2d, page, dg.get_num_sms())
        logits = dg.fp8_fp4_paged_mqa_logits(
            q=(q_rows.view(torch.int8).view(1, next_n, H, 64), q_sf_rows), kv_cache=kv_view, weights=w_rows,
            context_lens=ctx2d, block_table=block_table, schedule_meta=meta,
            max_context_len=num_pages * page, clean_logits=False, logits_dtype=torch.float)
        torch.cuda.synchronize()
        got = logits[:, :ctx]
        # dequantized reference (what the kernel should compute exactly, up to fp32 summation order)
        k_deq = dequant(got_vals, got_sf)[order]  # keys sorted by state -> logical position
        q_deq_rows = q_deq[-next_n:]
        ref_deq = ref_logits(q_deq_rows, k_deq, w_rows, ctx)
        # unquantized reference: fp32 RoPE'd q and k (what the indexer "means")
        ref_exact = ref_logits(q_full[-next_n:], k_full[written][order], w_rows, ctx)
        d_deq, d_exact = calc_diff(got, ref_deq), calc_diff(got, ref_exact)
        # FP8 indexer path on the same inputs for comparison (per-token fp8 K with fp32 scale, fp8 Q folded into weights)
        k_fp8_in = k_full[written][order]
        k_sf = k_fp8_in.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448.0
        k_fp8 = (k_fp8_in / k_sf).to(torch.float8_e4m3fn).float() * k_sf
        q_fp8_in = q_full[-next_n:]
        q_sf = q_fp8_in.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448.0
        q_fp8 = (q_fp8_in / q_sf).to(torch.float8_e4m3fn).float() * q_sf
        d_fp8 = calc_diff(ref_logits(q_fp8, k_fp8, w_rows, ctx), ref_exact)
        ok = d_deq < 1e-3
        all_ok &= ok
        print(f"  {'[ok ]' if ok else '[BAD]'} DeepGEMM logits cr={cr} ctx={ctx} next_n={next_n} page=128: vs dequantized ref {d_deq:.2e}; "
              f"vs unquantized fp32 ref {d_exact:.2e} (FP8 indexer on the same inputs: {d_fp8:.2e})")
        # 64-key re-page of the same cache must be bit-exact
        vals_sorted, sf_sorted = got_vals[order], got_sf[order]
        pages64 = (ctx + 63) // 64 + 1
        cache64 = torch.randint(0, 256, (pages64, 64 * 64 + 64 * 4), device=dev, dtype=torch.uint8)
        n_full = (ctx + 63) // 64
        padv = torch.zeros(n_full * 64, 64, device=dev, dtype=torch.uint8)
        pads = torch.zeros(n_full * 64, 4, device=dev, dtype=torch.uint8)
        padv[:ctx], pads[:ctx] = vals_sorted, sf_sorted
        perm64 = torch.randperm(pages64, device=dev)[:n_full]
        cache64[perm64, : 64 * 64] = padv.view(n_full, 64 * 64)
        cache64[perm64, 64 * 64:] = pads.view(n_full, 64 * 4)
        kv64 = torch.as_strided(cache64, size=(pages64, 64, 1, row_bytes), stride=(64 * row_bytes, row_bytes, row_bytes, 1))
        meta64 = dg.get_paged_mqa_logits_metadata(ctx2d, 64, dg.get_num_sms())
        logits64 = dg.fp8_fp4_paged_mqa_logits(
            q=(q_rows.view(torch.int8).view(1, next_n, H, 64), q_sf_rows), kv_cache=kv64, weights=w_rows,
            context_lens=ctx2d, block_table=perm64.to(torch.int32).view(1, -1).contiguous(), schedule_meta=meta64,
            max_context_len=pages64 * 64, clean_logits=False, logits_dtype=torch.float)
        torch.cuda.synchronize()
        parity = torch.equal(logits64[:, :ctx], got)
        all_ok &= parity
        print(f"  {'[ok ]' if parity else '[BAD]'} re-paged 64 vs 128 keys/page: {'BIT-EXACT' if parity else (logits64[:, :ctx] - got).abs().max().item()}")

    print("\nALL OK" if all_ok else "\nFAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
