#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer main's DeepSeek-V4.1 mixed-cache sparse MLA (flashinfer-ai/flashinfer#5197, merged 2026-09-18)
on vLLM's V4.1 geometry and against our record writers and references.

Upstream reads the compressed cache in its training-format record directly:
``kv_cache_format="fp8_dsv41_fp4_ca"`` = a 528 B V4.1 fp8 sliding-window cache (512 fp8 + 16 UE8M0/32) plus a
288 B FP4 compressed cache (256 B e2m1 pairs + 32 e4m3/16, vLLM main's ``nvfp4_ds_mla``), the FP4 rows converted
to fp8/UE8M0-32 tiles on chip (``kernels/dsv41_fp8/convert.cuh``). This is the kernel the section-5 project of
HANDOFF_2026-09-21 set out to write. This test asks whether it serves our deployment:

  BYTES  our ``rope_quant_insert_packed`` (288 B record, bit-exact with the checkpoint's fp4_act_quant) writes the
         same bytes as upstream's ``dsv41_fp4_quantize_pack_sparse_mla_cache``; our 528 B record == the torch
         reference of the V4.1 sliding-window record.
  REF    attention output / LSE vs the fp32 reference over the dequantized rows (528 B SWA + exactly dequantized
         FP4 compressed rows), FlashInfer's own tolerance (atol = rtol = 5e-2).
  REQ    the same vs the double-quantization model the FP8 route computes (FP4 rows -> fp8/UE8M0-32 -> attention).
  TODAY  today's production numerics on the same data: FlashInfer's DSV4 kernel (584 B records) on a 584 B SWA cache
         and on the fp8_ds_mla scratch our ``dequant_context_to_ds_mla`` builds from the FP4 cache -- its distance
         from the same reference, next to the mixed kernel's.
  GEOM   vLLM's V4.1 geometry: SWA page 32, compressed page 128 (cr = 1) / 64 (cr = 2), 8 / 16 / 64 query heads
         (TP8 / TP4 / TP1), SWA top-k 128 / 192 (DSpark rows) / 1152 (vision rows), compressed top-k 512, -1 padding
         and per-row top-k lengths, attention sinks, decode (T <= 64) and prefill (T > 64) entries.
  PERF   fp8_dsv41_fp4_ca vs fp8_dsv41 (all-fp8 528 B) vs fp8 (584 B DSV4, today's kernel) vs fp8 + our gather /
         context dequantization (today's production path) on the served shapes, cold caches (256K-token pools).

Run inside a container with flashinfer >= 0.7.0 built from main >= eb5f05be on one SM120 GPU:
  cd tools/dsv41_sm120/nvfp4_kv && python3 test_flashinfer_dsv41_fp4_extra.py [--quick] [--perf-only] [--json out]

``_ref_sparse_attn`` and the footer quantizers follow FlashInfer's tests/attention/sparse_mla_test_utils.py
(BSD-3-Clause, NVIDIA CORPORATION & AFFILIATES), vectorized here.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nvfp4_kv_kernels import (  # noqa: E402
    DS_MLA_BYTES,
    DS_MLA_PAGE,
    dequant_context_to_ds_mla,
    gather_requant_to_ds_mla,
    rope_quant_insert_packed,
    scratch_pages,
)

D_QK = D_V = 512
SWA_PBS = 32
EXTRA_TOPK = 512
FMT_MIXED = "fp8_dsv41_fp4_ca"
FMT_V41 = "fp8_dsv41"
FMT_V4 = "fp8"


# ------------------------------------------------------------------------------------------ records
def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(x)) for x > 0, exact (frexp instead of a rounded log2)."""
    m, e = torch.frexp(x.float())
    e = torch.where(m == 0.5, e - 1, e)
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), e)


def _ue8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    return ((scale.float().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def quantize_v41_528(kv: torch.Tensor, pbs: int) -> torch.Tensor:
    """bf16 [S, 512] -> V4.1 sliding-window record [S/pbs, pbs, 1, 528]: 512 fp8 + 16 UE8M0 scales of 32
    (act_quant(kv, 32, "ue8m0"): amax floor 1e-4, scale 2^ceil(log2(amax/448)))."""
    s = kv.shape[0]
    nb = s // pbs
    x = kv.float().view(s, 16, 32)
    scale = _pow2_ceil(x.abs().amax(-1).clamp(min=1e-4) / 448.0)
    fp8 = (x / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    data = fp8.view(torch.uint8).reshape(nb, pbs * 512)
    sc = _ue8m0_bytes(scale).reshape(nb, pbs * 16)
    return torch.cat([data, sc], 1).view(nb, pbs, 1, 528)


def dequantize_v41_528(packed: torch.Tensor) -> torch.Tensor:
    nb, pbs = packed.shape[0], packed.shape[1]
    flat = packed.reshape(nb, pbs * 528)
    fp8 = flat[:, : pbs * 512].reshape(nb * pbs, 16, 32).view(torch.float8_e4m3fn).float()
    e = flat[:, pbs * 512 :].reshape(nb * pbs, 16).float() - 127.0
    return (fp8 * torch.exp2(e).unsqueeze(-1)).reshape(nb * pbs, 512).to(torch.bfloat16)


def quantize_v4_584(kv: torch.Tensor, pbs: int) -> torch.Tensor:
    """bf16 [S, 512] -> vLLM's fp8_ds_mla record [S/pbs, pbs, 1, 584]:
    448 fp8 (7 UE8M0 scales of 64 + a pad byte) + 64 bf16 RoPE dims."""
    s = kv.shape[0]
    nb = s // pbs
    nope = kv[:, :448].float().view(s, 7, 64)
    scale = _pow2_ceil(nope.abs().amax(-1).clamp(min=1e-4) / 448.0)
    fp8 = (nope / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).reshape(s, 448)
    rope = kv[:, 448:].contiguous().view(torch.uint8)
    data = torch.cat([fp8, rope], 1).reshape(nb, pbs * 576)
    sc = torch.cat([_ue8m0_bytes(scale), torch.zeros(s, 1, dtype=torch.uint8, device=kv.device)], 1)
    return torch.cat([data, sc.reshape(nb, pbs * 8)], 1).view(nb, pbs, 1, 584)


def dequantize_v4_584(packed: torch.Tensor) -> torch.Tensor:
    nb, pbs = packed.shape[0], packed.shape[1]
    flat = packed.reshape(nb, pbs * 584)
    rows = flat[:, : pbs * 576].reshape(nb * pbs, 576)
    fp8 = rows[:, :448].reshape(-1, 7, 64).view(torch.float8_e4m3fn).float()
    e = flat[:, pbs * 576 :].reshape(nb * pbs, 8)[:, :7].float() - 127.0
    nope = (fp8 * torch.exp2(e).unsqueeze(-1)).reshape(-1, 448)
    rope = rows[:, 448:].contiguous().view(torch.bfloat16).float()
    return torch.cat([nope, rope], 1).to(torch.bfloat16)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequantize_fp4_288(packed: torch.Tensor) -> torch.Tensor:
    """V41_FP4 [nb, pbs, 1, 288] -> bf16 [nb*pbs, 512], exact (e2m1 x e4m3 has <= 6 significant bits)."""
    nb, pbs = packed.shape[0], packed.shape[1]
    flat = packed.reshape(nb, pbs * 288)
    data = flat[:, : pbs * 256].reshape(nb * pbs, 256)
    sc = flat[:, pbs * 256 :].reshape(nb * pbs, 32).view(torch.float8_e4m3fn).float()
    codes = torch.empty(nb * pbs, 512, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = data & 0xF
    codes[:, 1::2] = data >> 4
    mags = torch.tensor(_E2M1, device=packed.device)[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mags, mags)
    return (vals.view(nb * pbs, 32, 16) * sc.unsqueeze(-1)).reshape(nb * pbs, 512).to(torch.bfloat16)


def our_insert(kv: torch.Tensor, pbs: int, record: int) -> torch.Tensor:
    """Our record writer at RoPE identity (position 0: cos 1, sin 0) -> [S/pbs, pbs, record] uint8."""
    s = kv.shape[0]
    cache = torch.zeros(s // pbs, pbs, record, dtype=torch.uint8, device=kv.device)
    positions = torch.zeros(s, dtype=torch.int64, device=kv.device)
    cos_sin = torch.cat([torch.ones(1, 32), torch.zeros(1, 32)], 1).to(device=kv.device, dtype=torch.bfloat16)
    slots = torch.arange(s, dtype=torch.int64, device=kv.device)
    rope_quant_insert_packed(kv.contiguous(), positions, cos_sin, cache, slots, 1)
    return cache


def strided_view(cache: torch.Tensor, slots: int, slot: int) -> torch.Tensor:
    """The cache as one page of a vLLM-style shared block: [nb, slots x page] bytes, our pages at `slot`
    (vLLM packs the four kv sources' compressed + indexer pages into one block; SWA groups pack ~11 layers)."""
    nb, pbs, _, bpt = cache.shape
    page = pbs * bpt
    buf = torch.zeros(nb, slots * page, dtype=torch.uint8, device=cache.device)
    buf[:, slot * page : (slot + 1) * page] = cache.reshape(nb, page)
    return buf[:, slot * page : (slot + 1) * page].view(nb, pbs, 1, bpt)


def scratch_from_fp4(extra288: torch.Tensor) -> torch.Tensor:
    """Today's prefill pool: the whole compressed context as fp8_ds_mla, state i at slot i (128-state pages)."""
    nb, pbs = extra288.shape[0], extra288.shape[1]
    src = extra288.reshape(nb, pbs, 288)  # keeps a block-strided view's stride(0)
    n = nb * pbs
    pages = (n + DS_MLA_PAGE - 1) // DS_MLA_PAGE
    scratch = torch.zeros(pages, DS_MLA_PAGE, DS_MLA_BYTES, dtype=torch.uint8, device=src.device)
    block_table = torch.arange(nb, dtype=torch.int32, device=src.device)
    dequant_context_to_ds_mla(src, block_table, n, scratch)
    return scratch.view(-1, DS_MLA_PAGE, 1, DS_MLA_BYTES)


# ---------------------------------------------------------------------------------------- reference
def _ref_sparse_attn(q, kv_rows, indices, sm_scale, attn_sink=None):
    """fp32 attention over gathered rows; -1 = masked. Returns (bf16 out [T,H,512], base-2 LSE [T,H])."""
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]
    kv_flat = kv_rows.reshape(-1, d_qk).float()
    invalid = indices < 0
    gathered = kv_flat.index_select(0, indices.clamp(min=0).view(-1)).view(num_tokens, topk, d_qk)
    P = torch.einsum("thd,tkd->thk", q.float(), gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")
    lse_e = torch.logsumexp(P, dim=-1)
    lse_safe = lse_e.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out_f = torch.einsum("thk,tkd->thd", weights, gathered[..., :D_V])
    ln2 = float(torch.log(torch.tensor(2.0)).item())
    lse_log2 = lse_e / ln2
    if attn_sink is not None:
        sink = attn_sink.float()
        out_f = out_f * torch.sigmoid(lse_e - sink.unsqueeze(0)).unsqueeze(-1)
        sink_log2 = sink / ln2
        lse_log2 = torch.where(
            lse_log2 == float("-inf"),
            sink_log2.unsqueeze(0).expand_as(lse_log2),
            lse_log2 + torch.log2(1.0 + torch.exp2(sink_log2.unsqueeze(0) - lse_log2)),
        )
    return out_f.to(torch.bfloat16), lse_log2


def _mask_by_len(idx: torch.Tensor, lens: torch.Tensor | None) -> torch.Tensor:
    if lens is None:
        return idx
    ar = torch.arange(idx.shape[-1], device=idx.device).unsqueeze(0)
    return torch.where(ar < lens.unsqueeze(-1), idx, torch.full_like(idx, -1))


# ------------------------------------------------------------------------------------------- kernel
def run_flashinfer(q, swa_cache, swa_idx, swa_lens, extra_cache, extra_idx, extra_lens, fmt, sink, workspace):
    import flashinfer.mla

    out = torch.zeros(q.shape[0], q.shape[1], D_V, dtype=torch.bfloat16, device=q.device)
    flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q,
        swa_kv_cache=swa_cache,
        workspace_buffer=workspace,
        sparse_indices=swa_idx,
        compressed_kv_cache=extra_cache,
        out=out,
        bmm1_scale=D_QK**-0.5,
        sinks=sink,
        kv_layout="NHD",
        swa_topk_lens=swa_lens,
        extra_sparse_indices=extra_idx,
        extra_sparse_topk_lens=extra_lens,
        kv_cache_format=fmt,
    )
    return out


def _lens_or_full(lens, n, topk):
    if lens is not None:
        return lens
    return torch.full((n,), topk, dtype=torch.int32, device="cuda")


# -------------------------------------------------------------------------------------------- BYTES
def check_bytes(device) -> dict:
    import flashinfer.mla

    torch.manual_seed(7)
    s = 4096
    kv = (torch.randn(s, D_QK, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    kv[:64] *= 300.0  # groups at the e4m3 scale ceiling (amax/6 -> 448)
    kv[64:128] = 1e-5  # groups at the 2^-9 scale floor
    kv[128:192] = 0.0
    res = {}
    for pbs in (128, 64):
        ours = our_insert(kv, pbs, 288)
        theirs = flashinfer.mla.dsv41_fp4_quantize_pack_sparse_mla_cache(kv.view(s // pbs, pbs, D_QK), kv_layout="NHD")
        theirs = theirs.reshape(s // pbs, pbs, 288)
        o, t = ours.view(-1, pbs * 288), theirs.view(-1, pbs * 288)
        diff_data = (o[:, : pbs * 256] != t[:, : pbs * 256]).sum().item()
        diff_sc = (o[:, pbs * 256 :] != t[:, pbs * 256 :]).sum().item()
        res[f"fp4_288_pbs{pbs}"] = dict(data_bytes_differ=diff_data, scale_bytes_differ=diff_sc)
    ours528 = our_insert(kv, SWA_PBS, 528).view(-1, SWA_PBS, 1, 528)
    ref528 = quantize_v41_528(kv, SWA_PBS)
    d = (ours528 != ref528).view(-1, SWA_PBS * 528)
    res["fp8_528_pbs32"] = dict(
        data_bytes_differ=d[:, : SWA_PBS * 512].sum().item(),
        scale_bytes_differ=d[:, SWA_PBS * 512 :].sum().item(),
        values_equal_after_dequant=torch.equal(dequantize_v41_528(ours528), dequantize_v41_528(ref528)),
    )
    fp4 = [v for k, v in res.items() if k.startswith("fp4")]
    ok = all(v["data_bytes_differ"] == 0 and v["scale_bytes_differ"] == 0 for v in fp4)
    ok = ok and res["fp8_528_pbs32"]["values_equal_after_dequant"]
    res["ok"] = ok
    return res


# --------------------------------------------------------------------------------------------- GEOM
class Case:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        s = " ".join(f"{k}={getattr(self, k)}" for k in ("nt", "nh", "topk", "extra_pbs", "sink", "lengths"))
        return s + (" strided" if getattr(self, "strided", False) else "")


def run_case(c: Case, device, workspace, today: bool) -> dict:
    torch.manual_seed(1234 + c.nt * 7 + c.nh)
    s_main, s_extra = 4096, 4096
    main_bf16 = (torch.randn(s_main, D_QK, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    extra_bf16 = (torch.randn(s_extra, D_QK, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    swa528 = our_insert(main_bf16, SWA_PBS, 528).view(-1, SWA_PBS, 1, 528)
    extra288 = our_insert(extra_bf16, c.extra_pbs, 288).view(-1, c.extra_pbs, 1, 288)
    swa_rows = dequantize_v41_528(swa528)
    extra_rows = dequantize_fp4_288(extra288)
    if getattr(c, "strided", False):
        swa528 = strided_view(swa528, 11, 3)  # SWA group of 11 layers, ours the 4th
        extra288 = strided_view(extra288, 8, 4)  # 4 x (compressed page | indexer page), ours the 3rd source

    q = (torch.randn(c.nt, c.nh, D_QK, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    idx = torch.randint(0, s_main, (c.nt, c.topk), device=device, dtype=torch.int32)
    idx[:, (c.topk * 3) // 4 :] = -1  # left-aligned rows with -1 padding, as vLLM builds them
    eidx = torch.randint(0, s_extra, (c.nt, EXTRA_TOPK), device=device, dtype=torch.int32)
    eidx[:, (EXTRA_TOPK * 3) // 4 :] = -1
    sink = (torch.randn(c.nh, device=device, dtype=torch.float32) * 2.0) if c.sink else None
    lens = elens = None
    if c.lengths:
        lens = torch.randint(1, c.topk + 1, (c.nt,), device=device, dtype=torch.int32)
        elens = torch.randint(1, EXTRA_TOPK + 1, (c.nt,), device=device, dtype=torch.int32)
    sm_scale = D_QK**-0.5

    # references over the virtual concatenation [swa rows | extra rows]
    e_shift = torch.where(eidx < 0, eidx, eidx + s_main)
    v_idx = torch.cat([_mask_by_len(idx, lens), _mask_by_len(e_shift, elens)], -1)
    ref_out, ref_lse = _ref_sparse_attn(q, torch.cat([swa_rows, extra_rows], 0), v_idx, sm_scale, sink)
    extra_requant = dequantize_v41_528(quantize_v41_528(extra_rows, c.extra_pbs))  # what the FP8 route feeds the MMA
    req_out, _ = _ref_sparse_attn(q, torch.cat([swa_rows, extra_requant], 0), v_idx, sm_scale, sink)

    out = run_flashinfer(q, swa528, idx, _lens_or_full(lens, c.nt, c.topk), extra288, eidx,
                         _lens_or_full(elens, c.nt, EXTRA_TOPK), FMT_MIXED, sink, workspace)
    torch.cuda.synchronize()
    res = dict(case=repr(c))
    res["ref_maxdiff"] = (out.float() - ref_out.float()).abs().max().item()
    res["req_maxdiff"] = (out.float() - req_out.float()).abs().max().item()
    try:
        torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)
        res["ref"] = "PASS"
    except AssertionError as e:
        res["ref"] = "FAIL: " + str(e).splitlines()[0][:100]

    if today:
        # today's path: DSV4 kernel on a 584 B SWA cache and on the fp8_ds_mla scratch built from the FP4 cache
        swa584 = quantize_v4_584(main_bf16, SWA_PBS)
        scratch = scratch_from_fp4(extra288)
        t_out = run_flashinfer(q, swa584, idx, _lens_or_full(lens, c.nt, c.topk), scratch, eidx,
                               _lens_or_full(elens, c.nt, EXTRA_TOPK), FMT_V4, sink, workspace)
        today_rows = torch.cat([dequantize_v4_584(swa584), dequantize_v4_584(scratch)], 0)
        t_ref, _ = _ref_sparse_attn(q, today_rows, v_idx, sm_scale, sink)  # exact rows of today's storage
        res["today_vs_exact_ref"] = (t_out.float() - ref_out.float()).abs().max().item()
        res["today_vs_own_rows"] = (t_out.float() - t_ref.float()).abs().max().item()
        res["mixed_vs_today"] = (out.float() - t_out.float()).abs().max().item()
    return res


def build_cases(quick: bool):
    nts = [6, 65] if quick else [1, 6, 16, 48, 64, 65, 200]
    nhs = [16] if quick else [16, 8, 64]
    cases = []
    for nt, nh in itertools.product(nts, nhs):
        for topk in (128, 192, 1152):
            for extra_pbs in (128, 64):
                for sink, lengths in ((False, False), (True, True)):
                    cases.append(Case(nt=nt, nh=nh, topk=topk, extra_pbs=extra_pbs, sink=sink, lengths=lengths))
                    if nt in (6, 65):  # vLLM's block-strided cache views, decode and prefill entry
                        cases.append(Case(nt=nt, nh=nh, topk=topk, extra_pbs=extra_pbs, sink=sink, lengths=lengths,
                                          strided=True))
    return cases


# --------------------------------------------------------------------------------------------- PERF
def _time_us(fn, iters=20, warmup=3):
    """Median GPU time of one CUDA-graph replay of fn (the serving path replays graphs, so Python launch
    overhead -- 30-50 us for the trtllm entry -- must not count); eager timing if capture fails."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = None
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        graph = None
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        if graph is not None:
            graph.replay()
        else:
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def perf(device, workspace, quick: bool) -> list[dict]:
    torch.manual_seed(99)
    s_main = 1 << 18  # 256K SWA tokens: 138 MB at 528 B, 153 MB at 584 B -- beyond L2
    s_extra = 1 << 18  # 256K compressed states: 75 MB at 288 B
    main_bf16 = (torch.randn(s_main, D_QK, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    extra_bf16 = (torch.randn(s_extra, D_QK, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    swa528 = our_insert(main_bf16, SWA_PBS, 528).view(-1, SWA_PBS, 1, 528)
    swa584 = quantize_v4_584(main_bf16, SWA_PBS)
    del main_bf16
    caches = {}
    for pbs in (128, 64):
        e288 = our_insert(extra_bf16, pbs, 288).view(-1, pbs, 1, 288)
        e528 = our_insert(extra_bf16, pbs, 528).view(-1, pbs, 1, 528)
        e584 = quantize_v4_584(extra_bf16, pbs)
        caches[pbs] = (e288, e528, e584)
    del extra_bf16
    torch.cuda.synchronize()

    shapes = [  # (label, T, H, swa topk, extra pbs)
        ("decode C1 (DSpark 6 rows)", 6, 16, 192, 128),
        ("decode C4 (24 rows)", 24, 16, 192, 128),
        ("decode C8 (48 rows)", 48, 16, 192, 128),
        ("decode C1, cr=2 layer", 6, 16, 192, 64),
        ("prefill chunk 4096, text", 4096, 16, 128, 128),
        ("prefill chunk 4096, vision rows", 4096, 16, 1152, 128),
    ]
    if quick:
        shapes = [shapes[0], shapes[4]]
    rows = []
    sink = torch.randn(16, device=device, dtype=torch.float32)
    for label, nt, nh, topk, epbs in shapes:
        e288, e528, e584 = caches[epbs]
        q = (torch.randn(nt, nh, D_QK, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
        idx = torch.randint(0, s_main, (nt, topk), device=device, dtype=torch.int32)
        eidx = torch.randint(0, s_extra, (nt, EXTRA_TOPK), device=device, dtype=torch.int32)
        lens = torch.full((nt,), topk, dtype=torch.int32, device=device)
        elens = torch.full((nt,), EXTRA_TOPK, dtype=torch.int32, device=device)
        row = dict(shape=label, nt=nt, nh=nh, topk=topk, extra_pbs=epbs)
        def timed(swa, extra, fmt):
            return _time_us(lambda: run_flashinfer(q, swa, idx, lens, extra, eidx, elens, fmt, sink, workspace))

        row["mixed_us"] = timed(swa528, e288, FMT_MIXED)
        row["v41_fp8_us"] = timed(swa528, e528, FMT_V41)
        row["v4_fp8_us"] = timed(swa584, e584, FMT_V4)
        if nt <= 64:
            scratch = torch.empty(scratch_pages(nt, EXTRA_TOPK), DS_MLA_PAGE, DS_MLA_BYTES, dtype=torch.uint8,
                                  device=device)
            src = e288.view(e288.shape[0], epbs, 288)
            row["today_gather_us"] = _time_us(lambda: gather_requant_to_ds_mla(src, eidx, scratch))
            row["today_total_us"] = row["v4_fp8_us"] + row["today_gather_us"]
        else:
            # prefill: today dequantizes each request's whole context once per kv source; per 4096 states here,
            # plus the per-131072-state cost for a long context (both O(context), not O(rows x 512))
            src = e288.view(e288.shape[0], epbs, 288)
            bt = torch.arange(src.shape[0], dtype=torch.int32, device=device)
            for n_states in (4096, 131072):
                pages = (n_states + DS_MLA_PAGE - 1) // DS_MLA_PAGE
                scratch = torch.empty(pages, DS_MLA_PAGE, DS_MLA_BYTES, dtype=torch.uint8, device=device)
                row[f"today_dequant_ctx{n_states}_us"] = _time_us(
                    lambda: dequant_context_to_ds_mla(src, bt, n_states, scratch)
                )
            row["today_total_us(ctx4096)"] = row["v4_fp8_us"] + row["today_dequant_ctx4096_us"]
        rows.append(row)
        print("  " + json.dumps(row))
        sys.stdout.flush()
    return rows


def precision_sweep(device, quick: bool) -> list[dict]:
    """The mixed cache through the wrapper API at each compute precision: ``default`` (FP8 QK/PV, the trtllm
    entry's route), explicit ``fp8``, and ``bf16`` (bf16 Q x exactly dequantized rows -- DeepSeek's reference
    arithmetic, no double quantization). Same cold pools and shapes as PERF, plus the deviation from the exact
    reference on a 4096-row slice of each shape."""
    from flashinfer.mla import SparseMLASm120Wrapper

    torch.manual_seed(99)
    s_main = s_extra = 1 << 18
    main_bf16 = (torch.randn(s_main, D_QK, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    extra_bf16 = (torch.randn(s_extra, D_QK, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    swa528 = our_insert(main_bf16, SWA_PBS, 528).view(-1, SWA_PBS, 1, 528)
    e288 = our_insert(extra_bf16, 128, 288).view(-1, 128, 1, 288)
    swa_rows = dequantize_v41_528(swa528)  # bf16 [S, 512]
    extra_rows = dequantize_fp4_288(e288)
    del main_bf16, extra_bf16
    shapes = [(6, 16, 192), (48, 16, 192), (4096, 16, 128), (4096, 16, 1152)]
    if quick:
        shapes = [shapes[0], shapes[2]]
    sink = torch.randn(16, device=device, dtype=torch.float32)
    rows = []
    for nt, nh, topk in shapes:
        q = (torch.randn(nt, nh, D_QK, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
        idx = torch.randint(0, s_main, (nt, topk), device=device, dtype=torch.int32)
        eidx = torch.randint(0, s_extra, (nt, EXTRA_TOPK), device=device, dtype=torch.int32)
        lens = torch.full((nt,), topk, dtype=torch.int32, device=device)
        elens = torch.full((nt,), EXTRA_TOPK, dtype=torch.int32, device=device)
        n_ref = min(nt, 256)
        v_idx = torch.cat([idx[:n_ref], eidx[:n_ref] + s_main], -1)
        ref_out, _ = _ref_sparse_attn(q[:n_ref], torch.cat([swa_rows, extra_rows], 0), v_idx, D_QK**-0.5, sink)
        row = dict(nt=nt, nh=nh, topk=topk, extra_pbs=128)
        for precision in ("default", "fp8", "bf16"):
            try:
                w = SparseMLASm120Wrapper(kv_scale_format="ue8m0_g32", extra_kv_fp4=True, compute_precision=precision)
                out = torch.zeros(nt, nh, D_V, dtype=torch.bfloat16, device=device)

                def call():
                    w.run(q, swa528, idx, out, D_QK**-0.5, topk_length=lens, attn_sink=sink,
                          extra_kv_cache=e288, extra_indices=eidx, extra_topk_length=elens)

                call()
                torch.cuda.synchronize()
                row[f"{precision}_us"] = _time_us(call)
                row[f"{precision}_ref_maxdiff"] = (out[:n_ref].float() - ref_out.float()).abs().max().item()
            except Exception as e:  # noqa: BLE001
                row[f"{precision}_error"] = repr(e)[:160]
        rows.append(row)
        print("  " + json.dumps(row))
        sys.stdout.flush()
    return rows


# --------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--perf-only", action="store_true")
    ap.add_argument("--no-perf", action="store_true")
    ap.add_argument("--no-today", action="store_true", help="skip the DSV4-kernel comparison in GEOM cases")
    ap.add_argument("--precisions", action="store_true", help="also time default / fp8 / bf16 compute precisions")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    import flashinfer.mla

    device = torch.device("cuda")
    workspace = torch.empty(128 << 20, dtype=torch.int8, device=device)
    report = dict(device=torch.cuda.get_device_name(0), flashinfer=flashinfer.mla.__file__)
    print(f"device={report['device']} cc={torch.cuda.get_device_capability(0)} flashinfer={report['flashinfer']}")
    fails = 0
    if not args.perf_only:
        b = check_bytes(device)
        report["bytes"] = b
        print("BYTES", json.dumps(b))
        fails += 0 if b["ok"] else 1

        cases = build_cases(args.quick)
        print(f"GEOM cases={len(cases)}")
        t0 = time.time()
        geom = []
        for c in cases:
            try:
                r = run_case(c, device, workspace, today=not args.no_today)
            except Exception as e:  # noqa: BLE001
                r = dict(case=repr(c), ref="ERROR: " + repr(e)[:200])
            geom.append(r)
            ok = r["ref"] == "PASS"
            fails += 0 if ok else 1
            extra = ""
            if "today_vs_exact_ref" in r:
                extra = (f" today: ref={r['today_vs_exact_ref']:.4f} own={r['today_vs_own_rows']:.4f}"
                         f" mixed-today={r['mixed_vs_today']:.4f}")
            tag = "ok " if ok else "BAD"
            print(f"[{tag}] {r['case']:<52} {r['ref']:<8} ref={r.get('ref_maxdiff', float('nan')):.4f}"
                  f" req={r.get('req_maxdiff', float('nan')):.4f}{extra}")
            sys.stdout.flush()
        report["geom"] = geom
        passed = sum(1 for r in geom if r["ref"] == "PASS")
        print(f"GEOM {passed}/{len(cases)} passed in {time.time() - t0:.0f}s")
    if not args.no_perf:
        print("PERF (median us; cold 256K-token pools)")
        report["perf"] = perf(device, workspace, args.quick)
        if args.precisions:
            print("PRECISION SWEEP (wrapper API, mixed cache; median us; maxdiff vs exact reference on <= 256 rows)")
            report["precisions"] = precision_sweep(device, args.quick)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    print("FAILURES:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
