#!/usr/bin/env python3
"""Per-layer routed-MoE time at decode shapes on one sm_120 GPU: vLLM's DeepGEMM FP8xFP4 chain
(what the ds41 image runs) vs FlashInfer CUTLASS W4A8 fused MoE (MXFP4 weights x MXFP8 activations,
`--moe-backend flashinfer_cutlass_afp8` in vLLM, the kernel 0xSero's SGLang stack uses on sm_120)
[vs B12x fused MoE if the package is installed].

DeepSeek-V4.1-Flash geometry per TP4 rank: 384 experts, hidden 5120, intermediate 640, top-6.
Cold L2: `--rot` weight copies rotated inside one CUDA graph. Correctness: both paths vs an fp32
reference on the dequantized FP4 weights and the *unquantized* bf16 activations, so the number also
shows what each activation quantization (per-128 fp32 scales vs per-32 UE8M0) costs.

Run inside the ds41 image (`pip install b12x` for the third column):
    python3 moe_backends_bench.py [--tokens 1,6,16,48,64] [--rot 4] [--skip fi,b12x]
The relerr column is only meaningful for the FlashInfer and B12x paths: the synthetic FP4 bytes are
interpreted through this script's plain low-nibble-first dequantization, which is not DeepGEMM's
layout (production correctness of that path is established end to end, GSM8K parity).
"""
from __future__ import annotations

import argparse
import contextlib
import time

import torch

dev = torch.device("cuda:0")


def bench_graph(fns, iters=30):
    for f in fns[:1]:
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
    for _ in range(iters):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) * 1000 / len(fns))
    ts.sort()
    return ts[len(ts) // 2]


def fp4_lut():
    # e2m1 values
    return torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], device=dev)


NIBBLE_SWAP = False


def dequant_fp4(w_u8, s_u8, lut):
    # w_u8 [E, N, K/2] packed low nibble first (high first with NIBBLE_SWAP); s_u8 [E, N, K/32] ue8m0
    lo = (w_u8 & 0xF).long()
    hi = (w_u8 >> 4).long()
    pair = [lut[hi], lut[lo]] if NIBBLE_SWAP else [lut[lo], lut[hi]]
    vals = torch.stack(pair, dim=-1).flatten(-2)  # [E, N, K]
    scale = torch.exp2(s_u8.float() - 127.0)  # [E, N, K/32]
    return (vals.view(*vals.shape[:-1], -1, 32) * scale.unsqueeze(-1)).flatten(-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,6,16,48,64")
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--inter", type=int, default=640)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--rot", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--skip", default="")
    ap.add_argument("--b12x-w13-layout", default="w31")
    ap.add_argument("--nibble-swap", action="store_true")
    args = ap.parse_args()
    E, K, I, TOPK = args.experts, args.hidden, args.inter, args.topk
    global NIBBLE_SWAP
    NIBBLE_SWAP = args.nibble_swap
    N1 = 2 * I
    skip = set(args.skip.split(","))
    print(f"device={torch.cuda.get_device_name(0)} E={E} K={K} I={I} topk={TOPK} rot={args.rot}")

    torch.manual_seed(0)
    lut = fp4_lut()

    def make_experts():
        w13 = torch.randint(0, 256, (E, N1, K // 2), device=dev, dtype=torch.uint8)
        w2 = torch.randint(0, 256, (E, K, I // 2), device=dev, dtype=torch.uint8)
        w13_s = torch.randint(118, 126, (E, N1, K // 32), device=dev, dtype=torch.uint8)
        w2_s = torch.randint(118, 126, (E, K, I // 32), device=dev, dtype=torch.uint8)
        return w13, w2, w13_s, w2_s

    raw = [make_experts() for _ in range(args.rot)]

    # ---------------- DeepGEMM path (vLLM chain) ----------------
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
        deepgemm_moe_permute,
        deepgemm_unpermute_and_reduce,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _pack_deepgemm_mxfp4_scales
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
        silu_mul_quant_fp8_packed_triton,
    )

    def quant_a1(t):  # production: per-128 fp32 scales rounded up to powers of two (DeepGEMM E8M0 mode)
        return per_token_group_quant_fp8(t, 128, use_ue8m0=True)
    from vllm.utils.deep_gemm import m_grouped_fp8_fp4_gemm_nt_contiguous

    dg_weights = []
    for w13, w2, w13_s, w2_s in raw:
        p13, p2 = _pack_deepgemm_mxfp4_scales(w13, w2, w13_s, w2_s)
        dg_weights.append((w13, w2, p13, p2))

    # ---------------- FlashInfer CUTLASS path ----------------
    fi_ok = "fi" not in skip
    if fi_ok:
        try:
            from flashinfer import block_scale_interleave, mxfp8_quantize
            from flashinfer.fused_moe import cutlass_fused_moe
        except Exception as e:  # noqa: BLE001
            print("flashinfer path unavailable:", repr(e))
            fi_ok = False
    fi_weights = []
    if fi_ok:
        for w13, w2, w13_s, w2_s in raw:
            # FlashInfer swiglu expects [up; gate] halves (vLLM swaps w1/w3 for this backend)
            w13_sw = torch.cat([w13[:, I:], w13[:, :I]], dim=1).contiguous()
            s13_sw = torch.cat([w13_s[:, I:], w13_s[:, :I]], dim=1).contiguous()
            s13_il = block_scale_interleave(s13_sw.view(torch.uint8)).reshape(s13_sw.shape)
            s2_il = block_scale_interleave(w2_s.view(torch.uint8)).reshape(w2_s.shape)
            fi_weights.append((w13_sw.view(torch.long), w2.view(torch.long), s13_il.view(torch.int32), s2_il.view(torch.int32)))
        fake = torch.ones(E, device=dev, dtype=torch.float32)

    # ---------------- B12x path (optional) ----------------
    b12_ok = "b12x" not in skip
    b12_prepared = []
    if b12_ok:
        try:
            import b12x.moe.fused_moe as b12moe
            print("b12x present:", b12moe.__file__, "supported:", b12moe.is_supported())
            w13_layout = args.b12x_w13_layout
            for w13, w2, w13_s, w2_s in raw:
                wp = b12moe.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
                                         params_dtype=torch.bfloat16, num_experts=E, hidden_size=K,
                                         intermediate_size=I, w13_layout=w13_layout)
                ones = torch.ones(E, device=dev, dtype=torch.float32)
                prep = b12moe.prepare_weights(plan=wp, w1_fp4=w13.clone(), w1_blockscale=w13_s.clone(), w1_global_scale=ones,
                                              a1_gscale=ones, w2_fp4=w2.clone(), w2_blockscale=w2_s.clone(),
                                              w2_global_scale=ones, a2_gscale=ones, params_dtype=torch.bfloat16)
                b12_prepared.append(prep)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print("b12x not available:", repr(e))
            b12_ok = False

    for M in [int(x) for x in args.tokens.split(",")]:
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        topk_w = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1).to(torch.float32)

        # fp32 reference on dequantized weights (weights set 0), bf16 activations unquantized;
        # two references: gate-first and up-first halves of w13 (kernels differ in convention)
        w13, w2, w13_s, w2_s = raw[0]
        refs = {"gate|up": torch.zeros(M, K, device=dev, dtype=torch.float32),
                "up|gate": torch.zeros(M, K, device=dev, dtype=torch.float32)}
        for t in range(M):
            for j in range(TOPK):
                e = int(topk_ids[t, j])
                W13 = dequant_fp4(w13[e:e + 1], w13_s[e:e + 1], lut)[0]  # [2I, K]
                W2 = dequant_fp4(w2[e:e + 1], w2_s[e:e + 1], lut)[0]  # [K, I]
                h = x[t].float() @ W13.t()
                for name, (g, u) in {"gate|up": (h[:I], h[I:]), "up|gate": (h[I:], h[:I])}.items():
                    a = torch.nn.functional.silu(g) * u
                    refs[name][t] += topk_w[t, j] * (a @ W2.t())

        def best_err(out):
            errs = {n: ((out.float() - r).norm() / r.norm()).item() for n, r in refs.items()}
            n = min(errs, key=errs.get)
            return f"{errs[n]:.2e} ({n})"

        line = f"M={M:3d}"
        # DeepGEMM chain
        xq, xq_s = quant_a1(x)
        a1q, a1q_s, expert_ids, inv_perm, align_used = deepgemm_moe_permute(
            aq=xq, aq_scale=xq_s, topk_ids=topk_ids, local_num_experts=E, expert_map=None, expert_tokens_meta=None)
        M_sum = a1q.size(0)
        mm1 = torch.empty((M_sum, N1), device=dev, dtype=torch.bfloat16)
        a2q_buf = torch.empty((M_sum, I), device=dev, dtype=torch.float8_e4m3fn)
        mm2 = torch.empty((M_sum, K), device=dev, dtype=torch.bfloat16)
        out_dg = torch.empty((M, K), device=dev, dtype=torch.bfloat16)
        import inspect
        act_kw = {"m_indices": expert_ids} if "m_indices" in inspect.signature(silu_mul_quant_fp8_packed_triton).parameters else {}
        state = {}

        def dg_chain(w):
            xq_, xq_s_ = quant_a1(x)
            a1q_, a1q_s_, expert_ids_, inv_perm_, _ = deepgemm_moe_permute(
                aq=xq_, aq_scale=xq_s_, topk_ids=topk_ids, local_num_experts=E, expert_map=None,
                expert_tokens_meta=None, aq_out=a1q)
            m_grouped_fp8_fp4_gemm_nt_contiguous((a1q_, a1q_s_), (w[0].view(torch.int8), w[2]), mm1, expert_ids_,
                                                 recipe_a=(1, 128), recipe_b=(1, 32))
            a2q, a2q_s = silu_mul_quant_fp8_packed_triton(mm1.view(-1, N1), group_size=128, output_q=a2q_buf, **act_kw)
            m_grouped_fp8_fp4_gemm_nt_contiguous((a2q, a2q_s), (w[1].view(torch.int8), w[3]), mm2, expert_ids_,
                                                 recipe_a=(1, 128), recipe_b=(1, 32))
            deepgemm_unpermute_and_reduce(a=mm2, topk_ids=topk_ids, topk_weights=topk_w, inv_perm=inv_perm_,
                                          expert_map=None, output=out_dg)

        dg_chain(dg_weights[0])
        torch.cuda.synchronize()
        err_dg = best_err(out_dg)
        t_dg = bench_graph([lambda w=w: dg_chain(w) for w in dg_weights], args.iters)
        line += f"  DeepGEMM chain {t_dg:7.1f} us (rows {M_sum}, align {align_used}, relerr {err_dg})"

        if fi_ok:
            out_fi = torch.empty((M, K), device=dev, dtype=torch.bfloat16)

            def fi_chain(w):
                xq8, sf = mxfp8_quantize(x, True)
                cutlass_fused_moe(xq8, topk_ids.to(torch.int), topk_w, w[0], w[1], torch.bfloat16,
                                  quant_scales=[w[2], fake, w[3], fake], input_sf=sf, output=out_fi,
                                  use_mxfp8_act_scaling=True)

            try:
                t0 = time.time()
                fi_chain(fi_weights[0])
                torch.cuda.synchronize()
                jit = time.time() - t0
                err_fi = best_err(out_fi)
                t_fi = bench_graph([lambda w=w: fi_chain(w) for w in fi_weights], args.iters)
                line += f"  | FlashInfer CUTLASS W4A8 {t_fi:7.1f} us (relerr {err_fi}, first call {jit:.0f}s)"
            except Exception as e:  # noqa: BLE001
                line += f"  | FlashInfer CUTLASS failed: {repr(e)[:200]}"
                fi_ok = False
        if b12_ok:
            try:
                out_b = torch.empty((M, K), device=dev, dtype=torch.bfloat16)
                plan = b12moe.plan(b12moe.Caps(max_tokens=M, num_topk=TOPK, device=dev, weight_plan=b12_prepared[0].plan,
                                               core_token_counts=(M,), route_num_experts=0, quant_mode="w4a8_mx",
                                               apply_router_weight_on_input=False, swiglu_limit=None, swiglu_alpha=None,
                                               swiglu_beta=None, frozen=True))
                spec = plan.scratch_specs()[0]
                scratch = torch.empty(int(spec.shape[0]), device=dev, dtype=torch.uint8)
                tid32 = topk_ids.to(torch.int32)

                def b12_chain(prep):
                    binding = b12moe.bind(plan, scratch=scratch, a=x, experts=prep, topk_weights=topk_w, topk_ids=tid32,
                                          output=out_b, input_scales_static=True, unit_scale_contract=False)
                    b12moe.run(binding=binding)

                t0 = time.time()
                b12_chain(b12_prepared[0])
                torch.cuda.synchronize()
                jit = time.time() - t0
                err_b = best_err(out_b)
                t_b = bench_graph([lambda pp=pp: b12_chain(pp) for pp in b12_prepared], args.iters)
                line += f"  | B12x w4a8_mx {t_b:7.1f} us (relerr {err_b}, first call {jit:.0f}s)"
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                line += f"  | B12x failed: {repr(e)[:200]}"
                b12_ok = False
        print(line, flush=True)


if __name__ == "__main__":
    main()
