#!/usr/bin/env python3
"""DeepGEMM FP8xFP4 grouped MoE (DeepSeek-V4.1-Flash decode shape) vs the M alignment
(= DeepGEMM BLOCK_M = rows of padding per expert).

vLLM pads every touched expert's rows to the contiguous-layout alignment DeepGEMM picks
(64 on sm_120 for 6 tokens x top-6 = 36 pairs -> 36 experts x 64 rows = 2304 rows, of
which 36 are real). FC1 writes, silu*up+quant reads/writes and FC2 writes all run over the
padded rows. This script runs vLLM's own permute -> FC1 -> act+quant -> FC2 chain under
`mk_alignment_scope(align)` for several alignments, times each kernel in a CUDA graph
(cold L2: rotating expert weights) and checks that the real rows are bit-identical.

Run inside the ds41 image on one sm_120 GPU:
    python3 dg_moe_blockm_bench.py [--aligns 64,32,16] [--tokens 6] [--topk 6] [--experts 384]
"""

from __future__ import annotations

import argparse
import statistics

import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M_and_alignment,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _pack_deepgemm_mxfp4_scales
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_quant_fp8_packed_triton,
)
from vllm.utils.deep_gemm import (
    get_mk_alignment_for_contiguous_layout,
    get_theoretical_mk_alignment_for_contiguous_layout,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    mk_alignment_scope,
)


def bench(fns, iters: int) -> float:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligns", default="64:64,64:-1,128:128,32:32,16:16",
                    help="comma list of align[:scope]; scope 0/omitted = align_used, -1 = no scope")
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--inter", type=int, default=640)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--rot", type=int, default=6, help="weight copies rotated inside the graph (cold L2)")
    ap.add_argument("--scope", type=int, default=0, help="force the mk_alignment_scope value around the GEMMs (0 = align_used, -1 = no scope)")
    ap.add_argument("--save-out", default=None, help="save the first configuration's MoE output (torch.save)")
    ap.add_argument("--compare-out", default=None, help="compare every configuration's output with this saved tensor")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(5)
    E, M, TOPK, K, I = args.experts, args.tokens, args.topk, args.hidden, args.inter
    N1 = 2 * I
    print(f"device={torch.cuda.get_device_name(0)}  E={E} tokens={M} topk={TOPK} K={K} inter={I}")
    print(f"default mk alignment {get_mk_alignment_for_contiguous_layout()}; theoretical for "
          f"expected_m={M * TOPK}, groups={E}: {get_theoretical_mk_alignment_for_contiguous_layout(M * TOPK, E)}")

    # FP4 experts (random bits) + UE8M0 scales, packed as vLLM does for DeepGEMM
    def make_experts():
        w13 = torch.randint(0, 256, (E, N1, K // 2), device=dev, dtype=torch.uint8)
        w2 = torch.randint(0, 256, (E, K, I // 2), device=dev, dtype=torch.uint8)
        w13_s = torch.randint(118, 126, (E, N1, K // 32), device=dev, dtype=torch.uint8)
        w2_s = torch.randint(118, 126, (E, K, I // 32), device=dev, dtype=torch.uint8)
        w13_s, w2_s = _pack_deepgemm_mxfp4_scales(w13, w2, w13_s, w2_s)
        return w13, w2, w13_s, w2_s

    weights = [make_experts() for _ in range(args.rot)]
    # activations: fp8 + fp32 per-(token, 128-group) scales, top-k routing without repeats
    x = (torch.randn(M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    x_s = torch.exp2(torch.randint(-8, -4, (M, K // 128), device=dev).float())
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    topk_w = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)

    ref_out = None
    import vllm.model_executor.layers.fused_moe.deep_gemm_utils as dgu

    orig_compute = dgu.compute_aligned_M_and_alignment
    for spec in args.aligns.split(","):
        align, _, sc = spec.partition(":")
        align = int(align)
        args.scope = int(sc) if sc else 0
        # what the production patch does: cap the per-call alignment (padding rows per expert)
        def capped(M, num_topk, local_num_experts, alignment, expert_tokens_meta, _cap=align):
            M_sum_, used = orig_compute(M, num_topk, local_num_experts, alignment, expert_tokens_meta)
            if used > _cap:
                used = _cap
                max_active = min(M * num_topk, local_num_experts)
                M_sum_ = M * num_topk + max_active * (used - 1)
                M_sum_ = (M_sum_ + used - 1) // used * used
            return M_sum_, used

        dgu.compute_aligned_M_and_alignment = capped
        M_sum, align_used = capped(M, TOPK, E, get_mk_alignment_for_contiguous_layout()[0], None)
        a1q_perm = torch.empty((M_sum, K), device=dev, dtype=torch.float8_e4m3fn)
        a1q, a1q_s, expert_ids, inv_perm, align_used2 = deepgemm_moe_permute(
            aq=x, aq_scale=x_s, topk_ids=topk_ids, local_num_experts=E, expert_map=None,
            expert_tokens_meta=None, aq_out=a1q_perm)
        dgu.compute_aligned_M_and_alignment = orig_compute
        assert a1q.size(0) == M_sum, (a1q.shape, M_sum)
        import contextlib
        scope_val = align_used if args.scope == 0 else args.scope
        with (contextlib.nullcontext() if scope_val < 0 else mk_alignment_scope(scope_val)):
            print(f"  GEMMs under mk alignment {get_mk_alignment_for_contiguous_layout()}")
            mm1 = torch.empty((M_sum, N1), device=dev, dtype=torch.bfloat16)
            a2q_buf = torch.empty((M_sum, I), device=dev, dtype=torch.float8_e4m3fn)
            mm2 = torch.empty((M_sum, K), device=dev, dtype=torch.bfloat16)
            out = torch.empty((M, K), device=dev, dtype=torch.bfloat16)

            def fc1(w13, w13_s):
                m_grouped_fp8_fp4_gemm_nt_contiguous((a1q, a1q_s), (w13.view(torch.int8), w13_s), mm1, expert_ids,
                                                     recipe_a=(1, 128), recipe_b=(1, 32))

            import inspect
            act_kw = {"m_indices": expert_ids} if "m_indices" in inspect.signature(silu_mul_quant_fp8_packed_triton).parameters else {}

            def act():
                return silu_mul_quant_fp8_packed_triton(mm1.view(-1, N1), group_size=128, output_q=a2q_buf, **act_kw)

            def fc2(w2, w2_s, a2q, a2q_s):
                m_grouped_fp8_fp4_gemm_nt_contiguous((a2q, a2q_s), (w2.view(torch.int8), w2_s), mm2, expert_ids,
                                                     recipe_a=(1, 128), recipe_b=(1, 32))

            # correctness (weights set 0): full chain, gather the real rows
            print(f"  operands: a1q {tuple(a1q.shape)} {a1q.dtype}, a1q_s {tuple(a1q_s.shape)} {a1q_s.dtype} "
                  f"strides {a1q_s.stride()}, w13 {tuple(weights[0][0].shape)}, w13_s {tuple(weights[0][2].shape)} "
                  f"{weights[0][2].dtype} strides {weights[0][2].stride()}, expert_ids {tuple(expert_ids.shape)}")
            fc1(weights[0][0], weights[0][2])
            a2q, a2q_s = act()
            fc2(weights[0][1], weights[0][3], a2q, a2q_s)
            deepgemm_unpermute_and_reduce(a=mm2, topk_ids=topk_ids, topk_weights=topk_w, inv_perm=inv_perm,
                                          expert_map=None, output=out)
            torch.cuda.synchronize()
            if ref_out is None:
                ref_out = out.clone()
                same = "reference"
                if args.save_out:
                    torch.save(ref_out.cpu(), args.save_out)
                if args.compare_out:
                    saved = torch.load(args.compare_out).to(dev)
                    same = f"vs saved: {(out != saved).sum().item()} of {out.numel()} differ (max {(out.float() - saved.float()).abs().max().item():.3e})"
            else:
                d = (out.float() - ref_out.float()).abs().max().item()
                same = f"max|out - out@{args.aligns.split(',')[0]}| = {d:.3e} ({(out != ref_out).sum().item()} of {out.numel()} differ)"

            def permute():
                dgu.compute_aligned_M_and_alignment = capped
                try:
                    deepgemm_moe_permute(aq=x, aq_scale=x_s, topk_ids=topk_ids, local_num_experts=E, expert_map=None,
                                         expert_tokens_meta=None, aq_out=a1q_perm)
                finally:
                    dgu.compute_aligned_M_and_alignment = orig_compute

            def gather():
                deepgemm_unpermute_and_reduce(a=mm2, topk_ids=topk_ids, topk_weights=topk_w, inv_perm=inv_perm,
                                              expert_map=None, output=out)

            t_perm = bench([permute] * args.rot, args.iters)
            t_gather = bench([gather] * args.rot, args.iters)
            t_fc1 = bench([lambda w=w: fc1(w[0], w[2]) for w in weights], args.iters)
            t_act = bench([act] * args.rot, args.iters)
            t_fc2 = bench([lambda w=w: fc2(w[1], w[3], a2q, a2q_s) for w in weights], args.iters)
            print(f"align {align:3d} scope {scope_val:4d} -> used {align_used}/{align_used2}, M_sum {M_sum:5d} rows: "
                  f"permute {t_perm:5.1f} us  FC1 {t_fc1:6.1f} us  act+quant {t_act:5.1f} us  FC2 {t_fc2:6.1f} us  "
                  f"gather {t_gather:5.1f} us  sum {t_perm + t_fc1 + t_act + t_fc2 + t_gather:6.1f} us   {same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
