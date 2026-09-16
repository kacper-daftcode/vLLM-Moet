#!/usr/bin/env python3
"""Op-level validation of the DeepSeek-V4.1 SM120 sparse-MLA instantiations.

Two checks per shape (DSV4 fp8_ds_mla layout, d_qk = d_v = 512):
  * REF    - output/LSE vs the dense torch reference over the gathered KV
             (same tolerances as FlashInfer's own tests: atol = rtol = 5e-2).
  * PARITY - the same logical tokens re-paged: SWA cache at 32 tokens/page
             (new instantiations) vs 64 tokens/page (stock instantiations),
             identical flat indices. The kernels differ only in the
             page-address arithmetic, so outputs must be bit-identical.
             Where stock has no equivalent shape (dual TOPK 192/1152,
             extra page 128) the parity partner is the widest stock shape
             (TOPK 2048 FP8 with topk_length) or the extra cache re-paged
             to 64 states/page.

Run inside the serving image on one SM120 GPU with the patched flashinfer
(and the AOT sparse_mla_sm120.so removed):
  python3 test_sparse_mla_sm120_dsv41.py [--quick]

Helpers quantize_kv_dsv4 / dequantize_kv_dsv4 / _ref_sparse_attn /
_make_decode_scratch are vendored from FlashInfer tests/attention/
test_sparse_mla_sm120.py (BSD-3-Clause, NVIDIA CORPORATION & AFFILIATES).
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time

import torch

from flashinfer.mla._sparse_mla_sm120 import (
    _sparse_mla_sm120_paged_attention as sparse_mla_sm120_paged_attention,
)

D_QK = D_V = 512
SWA_PBS = 32  # vLLM DeepseekV4SWACache block_size
VISION_TOPK = 1152  # 128 window + 1024 image tokens


# ----------------------------------------------------------------- vendored
def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def _fp32_to_ue8m0_bytes(scale_fp32: torch.Tensor) -> torch.Tensor:
    bits = scale_fp32.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def quantize_kv_dsv4(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV [nb, bs, 1, 512] into the DSv4 FP8 FOOTER format [nb, bs, 1, 584]."""
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2  # 576
    scale_bytes = num_tiles + 1  # 8
    bpt = data_stride + scale_bytes  # 584
    nb, bs, hk, d = kv_bf16.shape
    assert d == 512 and hk == 1
    kv = kv_bf16.squeeze(2)
    result_flat = torch.zeros(nb, bs * bpt, dtype=torch.uint8, device=kv.device)
    for ti in range(num_tiles):
        tile = kv[..., ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        ue8m0 = _fp32_to_ue8m0_bytes(scale)
        for tok in range(bs):
            data_off = tok * data_stride + ti * tile_size
            result_flat[:, data_off : data_off + tile_size] = fp8[:, tok].view(torch.uint8)
            scale_off = bs * data_stride + tok * scale_bytes + ti
            result_flat[:, scale_off] = ue8m0[:, tok]
    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    rope = rope.reshape(nb, bs, d_rope * 2)
    for tok in range(bs):
        rope_off = tok * data_stride + d_nope
        result_flat[:, rope_off : rope_off + d_rope * 2] = rope[:, tok]
    return result_flat.view(nb, bs, 1, bpt)


def dequantize_kv_dsv4(packed: torch.Tensor) -> torch.Tensor:
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2
    scale_bytes = num_tiles + 1
    bpt = data_stride + scale_bytes
    nb, bs, _, _ = packed.shape
    result = torch.zeros(nb, bs, 512, dtype=torch.bfloat16, device=packed.device)
    p = packed.view(nb, bs * bpt)
    for tok in range(bs):
        data_off = tok * data_stride
        scale_off = bs * data_stride + tok * scale_bytes
        for ti in range(num_tiles):
            fp8_off = data_off + ti * tile_size
            fp8 = p[:, fp8_off : fp8_off + tile_size].view(torch.float8_e4m3fn).float()
            ue8m0 = p[:, scale_off + ti]
            scale = torch.pow(2.0, ue8m0.float() - 127.0)
            result[:, tok, ti * tile_size : (ti + 1) * tile_size] = (
                fp8 * scale.unsqueeze(-1)
            ).to(torch.bfloat16)
        rope_off = data_off + d_nope
        rope_bytes = p[:, rope_off : rope_off + d_rope * 2].contiguous()
        result[:, tok, d_nope:] = rope_bytes.view(torch.bfloat16).reshape(nb, d_rope)
    return result.view(nb, bs, 1, 512)


def _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v, attn_sink=None, topk_length=None):
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]
    kv_flat = kv_dequant.view(-1, d_qk).float()
    q_f = q.float()
    idx_fixed = indices.clamp(min=0)
    invalid = indices < 0
    if topk_length is not None:
        ar = torch.arange(topk, device=q.device).unsqueeze(0)
        invalid = invalid | (ar >= topk_length.unsqueeze(-1))
    gathered = kv_flat.index_select(0, idx_fixed.view(-1)).view(num_tokens, topk, d_qk)
    P = torch.einsum("thd,tkd->thk", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")
    lse_e = torch.logsumexp(P, dim=-1)
    lse_safe = lse_e.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out_f = torch.einsum("thk,tkd->thd", weights, gathered[..., :d_v])
    LN2 = float(torch.log(torch.tensor(2.0)).item())
    lse_log2 = lse_e / LN2
    if attn_sink is not None:
        sink = attn_sink.float()
        sink_log2 = sink / LN2
        factor = torch.sigmoid(lse_e.float() - sink.unsqueeze(0))
        out_f = out_f * factor.unsqueeze(-1)
        lse_log2 = torch.where(
            lse_log2 == float("-inf"),
            sink_log2.unsqueeze(0).expand_as(lse_log2),
            lse_log2 + torch.log2(1.0 + torch.exp2(sink_log2.unsqueeze(0) - lse_log2)),
        )
    return out_f.to(torch.bfloat16), lse_log2


def _make_decode_scratch(num_tokens, num_heads, topk, d_v, device, *, extra_topk=0):
    num_splits = (topk + 63) // 64 + (extra_topk + 63) // 64
    return (
        torch.empty((num_tokens, num_heads, num_splits, d_v), dtype=torch.bfloat16, device=device),
        torch.empty((num_tokens, num_heads, num_splits), dtype=torch.float32, device=device),
    )


# ------------------------------------------------------------------ harness
class Case:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        keys = ("nt", "nh", "topk", "dual", "extra_pbs", "extra_topk", "lengths", "sink")
        return " ".join(f"{k}={getattr(self, k)}" for k in keys if hasattr(self, k))


def make_kv(num_tokens_total: int, device, pbs_list=(32, 64)):
    """Random logical KV of S tokens packed at each page size in pbs_list.

    Returns (dequantized [S,1,1,512], {pbs: packed [S/pbs, pbs, 1, 584]}). The
    quantization is per token/tile, so every paging holds identical values and
    only the page layout differs - the premise of the PARITY check.
    """
    kv = (torch.randn(num_tokens_total, 1, D_QK, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    packed = {pbs: quantize_kv_dsv4(kv.view(-1, pbs, 1, D_QK)) for pbs in pbs_list}
    deqs = {pbs: dequantize_kv_dsv4(p).reshape(-1, D_QK) for pbs, p in packed.items()}
    ref = deqs[pbs_list[0]]
    for pbs, d in deqs.items():
        assert torch.equal(ref, d), f"re-paging changed values at pbs={pbs}"
    return ref.reshape(-1, 1, 1, D_QK), packed


def run_kernel(q, kv_packed, idx, sm_scale, *, sink, topk_length, extra_kv, extra_idx,
               extra_topk_length, decode_scratch):
    nt, nh, _ = q.shape
    out = torch.zeros((nt, nh, D_V), dtype=torch.bfloat16, device=q.device)
    lse = torch.zeros((nt, nh), dtype=torch.float32, device=q.device)
    kw = {}
    if decode_scratch:
        mid_out, mid_lse = _make_decode_scratch(
            nt, nh, idx.shape[-1], D_V, q.device,
            extra_topk=extra_idx.shape[-1] if extra_idx is not None else 0,
        )
        kw = dict(mid_out=mid_out, mid_lse=mid_lse)
    sparse_mla_sm120_paged_attention(
        q, kv_packed, idx, out, lse, sm_scale, d_v=D_V, attn_sink=sink,
        topk_length=topk_length, extra_kv_cache=extra_kv, extra_indices=extra_idx,
        extra_topk_length=extra_topk_length, **kw,
    )
    torch.cuda.synchronize()
    return out, lse


def run_case(c: Case, device) -> dict:
    torch.manual_seed(1234)
    s_main = 64 * 64  # 4096 SWA tokens
    main_deq, main_packed = make_kv(s_main, device, (32, 64))
    main32, main64 = main_packed[32], main_packed[64]
    q = (torch.randn(c.nt, c.nh, D_QK, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    idx = torch.randint(0, s_main, (c.nt, c.topk), device=device, dtype=torch.int32)
    idx[:, (c.topk * 3) // 4 :] = -1  # -1 padding like vLLM's left-aligned rows
    sink = (torch.randn(c.nh, device=device, dtype=torch.float32) * 2.0) if c.sink else None
    topk_length = None
    if c.lengths:
        topk_length = torch.randint(1, c.topk + 1, (c.nt,), device=device, dtype=torch.int32)
    sm_scale = D_QK**-0.5

    extra_deq = extra_packed = extra_packed64 = extra_idx = extra_topk_length = None
    if c.dual:
        s_extra = 32 * 128  # 4096 compressed states
        extra_deq, extra_all = make_kv(s_extra, device, (c.extra_pbs, 64))
        extra_packed, extra_packed64 = extra_all[c.extra_pbs], extra_all[64]
        extra_idx = torch.randint(0, s_extra, (c.nt, c.extra_topk), device=device, dtype=torch.int32)
        extra_idx[:, (c.extra_topk * 3) // 4 :] = -1
        if c.lengths:
            extra_topk_length = torch.randint(1, c.extra_topk + 1, (c.nt,), device=device, dtype=torch.int32)

    # ---- reference over the virtual concatenation [main | extra]
    if c.dual:
        virtual_kv = torch.cat([main_deq.reshape(-1, D_QK), extra_deq.reshape(-1, D_QK)], 0).reshape(-1, 1, 1, D_QK)
        e_shift = torch.where(extra_idx < 0, extra_idx, extra_idx + s_main)
        if c.lengths:
            # apply both length masks by -1 (reference handles a single topk_length)
            ar_m = torch.arange(c.topk, device=device).unsqueeze(0)
            ar_e = torch.arange(c.extra_topk, device=device).unsqueeze(0)
            idx_m = torch.where(ar_m < topk_length.unsqueeze(-1), idx, torch.full_like(idx, -1))
            e_shift = torch.where(ar_e < extra_topk_length.unsqueeze(-1), e_shift, torch.full_like(e_shift, -1))
        else:
            idx_m = idx
        ref_out, ref_lse = _ref_sparse_attn(q, virtual_kv, torch.cat([idx_m, e_shift], -1), sm_scale, D_V, attn_sink=sink)
    else:
        ref_out, ref_lse = _ref_sparse_attn(q, main_deq, idx, sm_scale, D_V, attn_sink=sink, topk_length=topk_length)

    decode_scratch = c.nt <= 64
    out, lse = run_kernel(q, main32, idx, sm_scale, sink=sink, topk_length=topk_length,
                          extra_kv=extra_packed, extra_idx=extra_idx,
                          extra_topk_length=extra_topk_length, decode_scratch=decode_scratch)
    res = dict(case=repr(c))
    res["ref_out_maxdiff"] = (out.float() - ref_out.float()).abs().max().item()
    res["ref_lse_maxdiff"] = (lse - ref_lse).abs().max().item()
    try:
        torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(lse, ref_lse, atol=5e-2, rtol=5e-2)
        res["ref"] = "PASS"
    except AssertionError as e:
        res["ref"] = "FAIL: " + str(e).splitlines()[0][:120]

    # ---- parity partner: stock instantiation on the same logical data
    partner = None
    if not c.dual and c.topk in (128, 192, 256, 512, 1024, 2048):
        partner = dict(kv=main64, idx=idx, tl=topk_length)
    elif not c.dual and c.topk == VISION_TOPK:
        pad = torch.full((c.nt, 2048 - c.topk), -1, device=device, dtype=torch.int32)
        tl = topk_length if topk_length is not None else torch.full((c.nt,), c.topk, device=device, dtype=torch.int32)
        partner = dict(kv=main64, idx=torch.cat([idx, pad], -1), tl=tl)
    elif c.dual and c.topk == 128:
        partner = dict(kv=main64, idx=idx, tl=topk_length, ekv=extra_packed64, eidx=extra_idx, etl=extra_topk_length)
    if partner is not None:
        try:
            p_out, p_lse = run_kernel(q, partner["kv"], partner["idx"], sm_scale, sink=sink,
                                      topk_length=partner["tl"], extra_kv=partner.get("ekv"),
                                      extra_idx=partner.get("eidx"), extra_topk_length=partner.get("etl"),
                                      decode_scratch=decode_scratch)
            res["parity_out_maxdiff"] = (out.float() - p_out.float()).abs().max().item()
            res["parity_lse_maxdiff"] = (lse - p_lse).abs().max().item()
            res["parity"] = "BIT-EXACT" if (torch.equal(out, p_out) and torch.equal(lse, p_lse)) else (
                "CLOSE" if res["parity_out_maxdiff"] <= 2e-2 and res["parity_lse_maxdiff"] <= 2e-2 else "FAIL")
        except Exception as e:  # stock envelope may reject (report, do not fail REF)
            res["parity"] = "n/a (" + repr(e)[:80] + ")"
    else:
        res["parity"] = "n/a"
    return res


def build_cases(quick: bool):
    cases = []
    nts = [1, 16, 64, 65, 200] if not quick else [16, 200]
    nhs = [8, 16, 32, 64] if not quick else [8]
    for nt, nh in itertools.product(nts, nhs):
        for topk in (128, 192, VISION_TOPK):
            # VISION_TOPK at nt <= 64 is a small prefill batch that vLLM sends
            # through the decode entry (decode kernel TOPK=1152 instantiation).
            for sink, lengths in ((False, False), (True, True)):
                cases.append(Case(nt=nt, nh=nh, topk=topk, dual=False, sink=sink, lengths=lengths))
                for extra_pbs in (128, 64):
                    cases.append(Case(nt=nt, nh=nh, topk=topk, dual=True, extra_pbs=extra_pbs,
                                      extra_topk=512, sink=sink, lengths=lengths))
        if nt <= 64:
            cases.append(Case(nt=nt, nh=nh, topk=256, dual=False, sink=False, lengths=False))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    cases = build_cases(args.quick)
    print(f"device={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} cases={len(cases)}")
    fails = 0
    t0 = time.time()
    for c in cases:
        try:
            r = run_case(c, device)
        except Exception as e:  # noqa: BLE001
            r = dict(case=repr(c), ref="ERROR: " + repr(e)[:160], parity="-")
        ok = r["ref"] == "PASS" and not str(r.get("parity", "")).startswith("FAIL")
        fails += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] {r['case']:<75} ref={r['ref']:<8} "
              f"d_out={r.get('ref_out_maxdiff', float('nan')):.4f} d_lse={r.get('ref_lse_maxdiff', float('nan')):.4f} "
              f"parity={r.get('parity')}"
              + (f" (d_out={r['parity_out_maxdiff']:.2e} d_lse={r['parity_lse_maxdiff']:.2e})" if "parity_out_maxdiff" in r else ""))
        sys.stdout.flush()
    print(f"\n{len(cases) - fails}/{len(cases)} passed in {time.time() - t0:.0f}s")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
