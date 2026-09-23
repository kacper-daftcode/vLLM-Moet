#!/usr/bin/env python3
"""Bit-exactness of the fused MoE quant+scatter against vLLM's own chain, and its timing.

Reference (what the served image runs per MoE layer at decode):
    per_token_group_quant_fp8(x, 128)  ->  deepgemm_moe_permute(...)  ->  FC1 (DeepGEMM packs the
    fp32 UE8M0 scales into int32 inside the call)  ->  silu*up + quant  ->  FC2  ->  gather
Fused:
    fused_quant_permute(x, topk_ids, ...)  ->  FC1 (packed scales taken as is)  ->  ...

Checks, per (token, expert) pair through each path's own inverse permutation: identical fp8 row
bytes, identical UE8M0 scale bytes, the pair's expert id in m_indices at its slot; m_indices
identical as a whole (expert-ordered BLOCK_M-aligned regions, -1 padding); padding rows' packed
scales zero in both; the gathered MoE output bit-identical. Shapes: the served decode shapes
(E=384, K=5120, I=640, top-6, 1..64 tokens) plus edge cases (one expert for everything, invalid
-1 routing slots, K/128 not a multiple of 4, fp16, zero rows, huge rows).

Run inside the ds41 image on one sm_120 GPU:
    python3 test_moe_quant_scatter_sm120.py            # correctness
    python3 test_moe_quant_scatter_sm120.py --bench    # + per-stage CUDA-graph timings / profile
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moe_quant_scatter_sm120 import GROUP_SIZE, fused_quant_permute, fused_quant_permute_applicable  # noqa: E402

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (  # noqa: E402
    compute_aligned_M_and_alignment,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8,
    silu_mul_quant_fp8_packed_triton,
)
from vllm.utils.deep_gemm import (  # noqa: E402
    get_mk_alignment_for_contiguous_layout,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    mk_alignment_scope,
)


def ue8m0_bytes_from_fp32(s: torch.Tensor) -> torch.Tensor:
    return ((s.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def unpack_scales(packed: torch.Tensor, sf_k: int) -> torch.Tensor:
    """int32 [M_sum, ceil(sf_k/4)] (any strides) -> uint8 [M_sum, sf_k]."""
    p = packed.contiguous()
    b = torch.stack([(p >> (8 * j)) & 0xFF for j in range(4)], dim=-1).to(torch.uint8)
    return b.reshape(p.shape[0], -1)[:, :sf_k]


def reference(x, topk_ids, E, M_sum, aq_out):
    aq, s = per_token_group_quant_fp8(x, GROUP_SIZE)
    aq_p, s_p, expert_ids, inv_perm, align_used = deepgemm_moe_permute(
        aq=aq, aq_scale=s, topk_ids=topk_ids, local_num_experts=E, expert_map=None,
        expert_tokens_meta=None, aq_out=aq_out)
    assert aq_p.size(0) == M_sum
    return aq, s, aq_p, s_p, expert_ids, inv_perm, align_used


def check_case(name, x, topk_ids, E, verbose=True, software_cvt=False) -> tuple[int, int]:
    M, K = x.shape
    TOPK = topk_ids.size(1)
    sf_k = K // GROUP_SIZE
    dev = x.device
    assert fused_quant_permute_applicable(x, topk_ids, E, None, None), name
    block_m = get_mk_alignment_for_contiguous_layout()[0]
    M_sum, align_used = compute_aligned_M_and_alignment(M, TOPK, E, block_m, None)

    aq_out_ref = torch.full((M_sum, K), 0x55, device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    aq, s, aq_p_ref, s_p_ref, eid_ref, inv_ref, align_ref = reference(x, topk_ids, E, M_sum, aq_out_ref)
    assert align_ref == align_used
    aq_out_f = torch.full((M_sum, K), 0x55, device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    aq_p_f, s_p_f, eid_f, inv_f = fused_quant_permute(x, topk_ids, E, M_sum, align_used, aq_out=aq_out_f,
                                                      software_cvt=software_cvt)
    torch.cuda.synchronize()

    errors = []
    # m_indices identical (regions in expert order, first count rows = expert, rest -1)
    if not torch.equal(eid_f, eid_ref):
        errors.append(f"m_indices differ in {(eid_f != eid_ref).sum().item()} rows")
    valid = topk_ids >= 0
    P_valid = int(valid.sum())
    # every valid pair: a distinct slot, expert id there, same fp8 bytes, same scale bytes
    dest_f = inv_f[valid].long()
    dest_r = inv_ref[valid].long()
    if dest_f.unique().numel() != P_valid:
        errors.append("fused slots not distinct")
    tok = torch.arange(M, device=dev).unsqueeze(1).expand(M, TOPK)[valid]
    exp = topk_ids[valid].long()
    if not torch.equal(eid_f[dest_f], exp.int()):
        errors.append("fused slot has the wrong expert id")
    rows_f = aq_p_f.view(torch.uint8)[dest_f]
    rows_r = aq_p_ref.view(torch.uint8)[dest_r]
    rows_q = aq.view(torch.uint8)[tok]
    if not torch.equal(rows_f, rows_q):
        errors.append(f"fp8 rows differ: {(rows_f != rows_q).any(dim=1).sum().item()} of {P_valid} pairs, "
                      f"{(rows_f != rows_q).sum().item()} bytes")
    if not torch.equal(rows_r, rows_q):
        errors.append("reference permute inconsistent with its own quantization (?)")
    sb_f = unpack_scales(s_p_f, sf_k)[dest_f]
    sb_q = ue8m0_bytes_from_fp32(s)[tok]
    if not torch.equal(sb_f, sb_q):
        errors.append(f"scale bytes differ in {(sb_f != sb_q).sum().item()} of {sb_q.numel()}")
    # padding rows inside the expert regions: zero packed scales in both (the fused kernel leaves
    # the tail after the last region alone: DeepGEMM skips those blocks)
    pad = eid_f < 0
    last_real = int((eid_f >= 0).nonzero().max()) if P_valid else -1
    total = (last_real + 1 + align_used - 1) // align_used * align_used
    in_region = torch.arange(M_sum, device=dev) < total
    if s_p_f.contiguous()[pad & in_region].any():
        errors.append("fused padding rows have non-zero packed scales")
    if s_p_ref[pad].any():
        errors.append("reference padding rows have non-zero scales (?)")
    # untouched fp8 rows keep the workspace pattern in both (nothing written outside the slots)
    untouched_f = aq_p_f.view(torch.uint8)[pad]
    if not bool((untouched_f == 0x55).all()):
        errors.append("fused kernel wrote fp8 bytes into padding rows")
    if verbose:
        print(f"  {name}: M={M} topk={TOPK} K={K} E={E} pairs={P_valid} M_sum={M_sum} align={align_used} "
              f"-> {'OK' if not errors else 'FAIL: ' + '; '.join(errors)}")
    return len(errors), P_valid


def make_experts(E, N1, K, I, dev):
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _pack_deepgemm_mxfp4_scales

    w13 = torch.randint(0, 256, (E, N1, K // 2), device=dev, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, K, I // 2), device=dev, dtype=torch.uint8)
    w13_s = torch.randint(118, 126, (E, N1, K // 32), device=dev, dtype=torch.uint8)
    w2_s = torch.randint(118, 126, (E, K, I // 32), device=dev, dtype=torch.uint8)
    w13_s, w2_s = _pack_deepgemm_mxfp4_scales(w13, w2, w13_s, w2_s)
    return w13, w2, w13_s, w2_s


def chain(a1q, a1q_s, expert_ids, inv_perm, topk_ids, topk_w, w, M_sum, N1, I, K, M, align_used):
    dev = a1q.device
    w13, w2, w13_s, w2_s = w
    mm1 = torch.empty((M_sum, N1), device=dev, dtype=torch.bfloat16)
    a2q_buf = torch.empty((M_sum, I), device=dev, dtype=torch.float8_e4m3fn)
    mm2 = torch.empty((M_sum, K), device=dev, dtype=torch.bfloat16)
    out = torch.empty((M, K), device=dev, dtype=torch.bfloat16)
    with mk_alignment_scope(align_used):
        m_grouped_fp8_fp4_gemm_nt_contiguous((a1q, a1q_s), (w13.view(torch.int8), w13_s), mm1, expert_ids,
                                             recipe_a=(1, 128), recipe_b=(1, 32))
        a2q, a2q_s = silu_mul_quant_fp8_packed_triton(mm1.view(-1, N1), group_size=128, output_q=a2q_buf,
                                                      m_indices=expert_ids)
        m_grouped_fp8_fp4_gemm_nt_contiguous((a2q, a2q_s), (w2.view(torch.int8), w2_s), mm2, expert_ids,
                                             recipe_a=(1, 128), recipe_b=(1, 32))
    deepgemm_unpermute_and_reduce(a=mm2, topk_ids=topk_ids, topk_weights=topk_w, inv_perm=inv_perm,
                                  expert_map=None, output=out)
    return out


def check_chain(M, TOPK, E, K, I, dev, seed) -> int:
    """Full MoE output, reference vs fused, bit-identical."""
    torch.manual_seed(seed)
    N1 = 2 * I
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.7
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    topk_w = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)
    w = make_experts(E, N1, K, I, dev)
    block_m = get_mk_alignment_for_contiguous_layout()[0]
    M_sum, align_used = compute_aligned_M_and_alignment(M, TOPK, E, block_m, None)
    aq_out = torch.empty((M_sum, K), device=dev, dtype=torch.float8_e4m3fn)
    _, _, aq_p, s_p, eid, inv, _ = reference(x, topk_ids, E, M_sum, aq_out)
    out_ref = chain(aq_p, s_p, eid, inv, topk_ids, topk_w, w, M_sum, N1, I, K, M, align_used)
    aq_out2 = torch.empty((M_sum, K), device=dev, dtype=torch.float8_e4m3fn)
    aq_p2, s_p2, eid2, inv2 = fused_quant_permute(x, topk_ids, E, M_sum, align_used, aq_out=aq_out2)
    out_f = chain(aq_p2, s_p2, eid2, inv2, topk_ids, topk_w, w, M_sum, N1, I, K, M, align_used)
    torch.cuda.synchronize()
    same = torch.equal(out_ref, out_f)
    d = (out_ref.float() - out_f.float()).abs().max().item()
    print(f"  chain M={M} topk={TOPK} E={E} K={K} I={I}: MoE output {'bit-identical' if same else f'DIFFERS max {d:.3e}'}"
          f" (|out| max {out_ref.float().abs().max().item():.3f})")
    return 0 if same else 1


def correctness(dev) -> int:
    fails = 0
    E, K, TOPK = 384, 5120, 6
    torch.manual_seed(1)
    for M in (1, 2, 6, 7, 12, 48, 64):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.7
        topk = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        fails += check_case(f"served shape", x, topk, E)[0]
    # heavy collisions: few experts for many pairs (ranks > 0, expert regions of several blocks)
    x = torch.randn(64, K, device=dev, dtype=torch.bfloat16)
    topk = torch.randint(0, 3, (64, TOPK), device=dev, dtype=torch.int32)
    fails += check_case("3 experts for 384 pairs", x, topk, E)[0]
    topk = torch.zeros((64, TOPK), device=dev, dtype=torch.int32)
    fails += check_case("one expert for everything", x, topk, E)[0]
    topk = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(64)]).to(torch.int32)
    topk[3, 2] = -1
    topk[10, :] = -1
    fails += check_case("invalid -1 routing slots", x, topk, E)[0]
    # last experts / first experts only (scan boundaries)
    topk = torch.randint(E - 4, E, (6, TOPK), device=dev, dtype=torch.int32)
    fails += check_case("only the last experts", x[:6], topk, E)[0]
    topk = torch.randint(0, 4, (6, TOPK), device=dev, dtype=torch.int32)
    fails += check_case("only the first experts", x[:6], topk, E)[0]
    # values: zero rows (eps scale), huge rows, exact powers of two around the fp8 max
    x = torch.zeros(6, K, device=dev, dtype=torch.bfloat16)
    topk = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(6)]).to(torch.int32)
    fails += check_case("zero rows", x, topk, E)[0]
    x = (torch.randn(6, K, device=dev) * 1e30).to(torch.bfloat16)
    fails += check_case("huge rows", x, topk, E)[0]
    x = (torch.randint(-8, 9, (6, K), device=dev).float() * 448.0 * torch.exp2(torch.randint(-20, 20, (6, 1), device=dev).float())).to(torch.bfloat16)
    fails += check_case("multiples of 448 x 2^n", x, topk, E)[0]
    x = torch.randn(6, K, device=dev, dtype=torch.bfloat16) * torch.exp2(torch.randint(-60, 60, (6, 1), device=dev).float()).to(torch.bfloat16)
    fails += check_case("rows spanning 2^-60..2^60", x, topk, E)[0]
    x = torch.randn(6, K, device=dev, dtype=torch.bfloat16) * 1e-30
    fails += check_case("tiny rows", x, topk, E)[0]
    # every finite bf16 bit pattern under every reachable scale: the hardware e4m3 conversion
    # (served) and c10's software one (vLLM's kernel) against vLLM's kernel itself. Row r has
    # absmax 2^(r-40) at element 0 of each 128-group, so each group's scale is fixed and the other
    # 127 elements are the finite bf16 values with |v| <= absmax (others get masked to 0).
    bits = torch.arange(65536, device=dev, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    vals = torch.where(torch.isfinite(bits), bits, torch.zeros_like(bits))
    n_exp, n_groups = 81, 520  # 520 x 127 = 66040 >= 65536 values per row
    rows = []
    for r in range(n_exp):
        amax = 2.0 ** (r - 40)
        v = torch.where(vals.float().abs() <= amax, vals, torch.zeros_like(vals))
        v = torch.cat([v, torch.zeros(n_groups * 127 - 65536, device=dev, dtype=torch.bfloat16)])
        row = torch.zeros(n_groups, 128, device=dev, dtype=torch.bfloat16)
        row[:, 1:] = v.view(n_groups, 127)
        row[:, 0] = amax
        rows.append(row.view(-1))
    xe = torch.stack(rows)  # [81, 520 * 128]
    Ee = 8
    topk = (torch.arange(n_exp, device=dev) % Ee).view(n_exp, 1).to(torch.int32)
    exhaustive_fails = 0
    for sl in range(0, n_groups, 64):  # K = 8192 per slice (the kernel's maximum)
        x_sl = xe[:, 128 * sl: 128 * min(sl + 64, n_groups)].contiguous()
        for cvt in (False, True):
            exhaustive_fails += check_case("exhaustive", x_sl, topk, Ee, verbose=False, software_cvt=cvt)[0]
    fails += exhaustive_fails
    print(f"  all finite bf16 values x 81 scale exponents, hardware and c10 e4m3 conversion vs vLLM's kernel: "
          f"{'identical' if exhaustive_fails == 0 else 'DIFFER'}")
    # other geometries: K/128 not a multiple of 4, fp16, other expert counts / top-k
    for (M, TOPK2, E2, K2) in ((6, 6, 384, 640), (6, 6, 384, 384), (5, 8, 256, 2048), (3, 4, 1024, 1152), (1, 1, 1, 128), (64, 8, 512, 5120)):
        x = torch.randn(M, K2, device=dev, dtype=torch.bfloat16)
        topk = torch.stack([torch.randperm(E2, device=dev)[:TOPK2] for _ in range(M)]).to(torch.int32)
        fails += check_case(f"geometry", x, topk, E2)[0]
    x = torch.randn(6, K, device=dev, dtype=torch.float16) * 0.7
    topk = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(6)]).to(torch.int32)
    fails += check_case("fp16 rows", x, topk, E)[0]
    # strided x (a row-slice of a wider tensor)
    xw = torch.randn(6, K + 256, device=dev, dtype=torch.bfloat16)
    fails += check_case("row stride > K", xw[:, :K], topk, E)[0]
    # many random seeds on the served shape
    n_pairs = 0
    for seed in range(40):
        torch.manual_seed(100 + seed)
        M = int(torch.randint(1, 65, (1,)).item())
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * float(torch.rand(1).item() * 4 + 0.05)
        topk = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        f, p = check_case("random", x, topk, E, verbose=False)
        fails += f
        n_pairs += p
    print(f"  40 random served-shape cases: {n_pairs} pairs, {'all OK' if fails == 0 else 'FAILURES'}")
    # full chain (DeepGEMM FC1 / act / FC2 / gather) bit-identical
    for (M, seed) in ((6, 7), (48, 8), (64, 9), (1, 10)):
        fails += check_chain(M, TOPK, E, K, 640, dev, seed)
    return fails


def bench_graph(fns, iters: int) -> float:
    for f in fns[:2]:
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for f in fns:
            f()
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for f in fns:
                f()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(max(3, iters // 4)):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) * 1000 / len(fns))
    return statistics.median(ts)


def bench(dev, tokens: list[int], iters: int, rot: int, profile: bool) -> None:
    E, K, I, TOPK = 384, 5120, 640, 6
    N1 = 2 * I
    torch.manual_seed(5)
    weights = [make_experts(E, N1, K, I, dev) for _ in range(rot)]
    for M in tokens:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.7
        topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        topk_w = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)
        block_m = get_mk_alignment_for_contiguous_layout()[0]
        M_sum, align_used = compute_aligned_M_and_alignment(M, TOPK, E, block_m, None)
        aq_out = torch.empty((M_sum, K), device=dev, dtype=torch.float8_e4m3fn)
        mm1 = torch.empty((M_sum, N1), device=dev, dtype=torch.bfloat16)
        a2q_buf = torch.empty((M_sum, I), device=dev, dtype=torch.float8_e4m3fn)
        mm2 = torch.empty((M_sum, K), device=dev, dtype=torch.bfloat16)
        out = torch.empty((M, K), device=dev, dtype=torch.bfloat16)

        state: dict = {}

        def ref_quant_permute():
            aq, s = per_token_group_quant_fp8(x, GROUP_SIZE)
            state["ref"] = deepgemm_moe_permute(aq=aq, aq_scale=s, topk_ids=topk_ids, local_num_experts=E,
                                                expert_map=None, expert_tokens_meta=None, aq_out=aq_out)

        def fused():
            state["fused"] = fused_quant_permute(x, topk_ids, E, M_sum, align_used, aq_out=aq_out)

        ref_quant_permute()
        fused()
        torch.cuda.synchronize()
        aq_r, s_r, eid_r, inv_r, _ = state["ref"]
        aq_f, s_f, eid_f, inv_f = state["fused"]

        def fc1(a, s, eid, w):
            with mk_alignment_scope(align_used):
                m_grouped_fp8_fp4_gemm_nt_contiguous((a, s), (w[0].view(torch.int8), w[2]), mm1, eid,
                                                     recipe_a=(1, 128), recipe_b=(1, 32))

        def act(eid):
            return silu_mul_quant_fp8_packed_triton(mm1.view(-1, N1), group_size=128, output_q=a2q_buf, m_indices=eid)

        a2q, a2q_s = act(eid_r)

        def fc2(eid, w):
            with mk_alignment_scope(align_used):
                m_grouped_fp8_fp4_gemm_nt_contiguous((a2q, a2q_s), (w[1].view(torch.int8), w[3]), mm2, eid,
                                                     recipe_a=(1, 128), recipe_b=(1, 32))

        def gather(inv):
            deepgemm_unpermute_and_reduce(a=mm2, topk_ids=topk_ids, topk_weights=topk_w, inv_perm=inv,
                                          expert_map=None, output=out)

        t = {}
        t["ref quant+permute"] = bench_graph([ref_quant_permute] * rot, iters)
        t["ref FC1 (+pack)"] = bench_graph([lambda w=w: fc1(aq_r, s_r, eid_r, w) for w in weights], iters)
        t["fused quant+permute"] = bench_graph([fused] * rot, iters)
        t["fused FC1"] = bench_graph([lambda w=w: fc1(aq_f, s_f, eid_f, w) for w in weights], iters)
        t["act+quant"] = bench_graph([lambda: act(eid_r)] * rot, iters)
        t["FC2"] = bench_graph([lambda w=w: fc2(eid_r, w) for w in weights], iters)
        t["gather"] = bench_graph([lambda: gather(inv_r)] * rot, iters)
        ref_total = t["ref quant+permute"] + t["ref FC1 (+pack)"] + t["act+quant"] + t["FC2"] + t["gather"]
        fused_total = t["fused quant+permute"] + t["fused FC1"] + t["act+quant"] + t["FC2"] + t["gather"]
        print(f"M={M:2d} pairs={M * TOPK:3d} M_sum={M_sum:5d} align={align_used}: "
              + "  ".join(f"{k} {v:5.1f}" for k, v in t.items())
              + f"  | layer ref {ref_total:6.1f} us -> fused {fused_total:6.1f} us ({ref_total - fused_total:+.1f})")

        if profile:
            from torch.profiler import ProfilerActivity, profile

            def run_ref():
                ref_quant_permute()
                fc1(aq_r, s_r, eid_r, weights[0])

            def run_fused():
                fused()
                fc1(aq_f, s_f, eid_f, weights[0])

            for label, fn in (("reference", run_ref), ("fused", run_fused)):
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    fn()
                    s.synchronize()
                    with torch.cuda.graph(g, stream=s):
                        fn()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(20):
                        g.replay()
                    torch.cuda.synchronize()
                rows = [(e.key, e.count, e.self_device_time_total / max(e.count, 1)) for e in prof.key_averages()
                        if e.self_device_time_total > 0 and e.count >= 20]
                rows.sort(key=lambda r: -r[1] * r[2])
                print(f"  {label} quant+permute+FC1 kernels (per graph replay):")
                for k, c, us in rows:
                    print(f"    {c // 20:2d}x {us:6.1f} us  {k[:110]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--tokens", default="1,6,12,48,64")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--rot", type=int, default=6)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--skip-correctness", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)}  default mk alignment {get_mk_alignment_for_contiguous_layout()}")
    fails = 0
    if not args.skip_correctness:
        fails = correctness(dev)
        print("ALL OK" if fails == 0 else f"{fails} FAILURES")
    if args.bench:
        bench(dev, [int(v) for v in args.tokens.split(",")], args.iters, args.rot, args.profile)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
