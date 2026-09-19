#!/usr/bin/env python3
"""Correctness + cold-L2 timing of the sm_120 FP8 [32,32] MoE GEMV against vLLM's Triton
`fused_moe_kernel` (the production path of Qwen3.8-Flash-Next-FP8 at TP4) and an fp32
reference on the same quantized operands.

Per-rank shapes: E=512 experts, w13 [E, 320, 2560], w2 [E, 2560, 160], topk 10, block
scales [32, 32]. Both GEMMs of the expert MLP are tested for decode token counts; the
Triton side uses the config vLLM would pick (VLLM_TUNED_CONFIG_FOLDER honoured).

Run inside the qwen38 image on one sm_120 GPU:
    python3 test_fused_moe_gemv_sm120.py [--ms 1,2,4,8,12,16,32] [--cfgs 8x5,8x4,4x5] [--iters 40]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fused_moe_gemv_sm120 import fused_moe_gemv, fused_moe_gemv_act, kernel_cfg, moe_gemv_applicable  # noqa: E402

import vllm._custom_ops  # noqa: E402,F401  (registers torch.ops._C)
from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402
    _prepare_expert_assignment,
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8,
)
from vllm.triton_utils import tl  # noqa: E402

E, INTER, K, TOPK = 512, 160, 2560, 10
N1 = 2 * INTER
BLOCK = [32, 32]


def quant_weights(n: int, k: int, gen: torch.Generator, dev: torch.device):
    """Random fp8 experts with log-uniform fp32 block scales."""
    w = torch.randn(E, n, k, device=dev, dtype=torch.float32, generator=gen)
    w_q = (w * 32.0).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.exp2(torch.empty(E, n // 32, k // 32, device=dev).uniform_(-9, -5, generator=gen))
    return w_q, s.contiguous()


def dequant_w(w_q: torch.Tensor, s: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """fp32 dequantized experts idx (1-D long) -> [len(idx), N, K]."""
    w_q, s = w_q[idx], s[idx]
    e, n, k = w_q.shape
    return (w_q.float().view(e, n // 32, 32, k // 32, 32) * s[:, :, None, :, None]).view(e, n, k)


def dequant_a(a_q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    m, k = a_q.shape
    return (a_q.float().view(m, k // 32, 32) * s[:, :, None]).view(m, k)


def make_routing(M: int, gen: torch.Generator, dev: torch.device):
    ids = torch.stack([torch.randperm(E, device=dev, generator=gen)[:TOPK] for _ in range(M)]).to(torch.int32)
    w = torch.softmax(torch.randn(M, TOPK, device=dev, generator=gen), dim=-1).float().contiguous()
    return ids, w


def bench_graph(fns, iters: int) -> float:
    """Median us per call of a CUDA graph that runs every fn once (rotating operands)."""
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
    ap.add_argument("--ms", default="1,2,4,8,12,16,32")
    ap.add_argument("--cfgs", default="", help="extra (ksplit x unroll) variants to time, e.g. 8x4,4x5")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--rot", type=int, default=12, help="routing sets per timing graph (cold L2)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(11)
    print(f"device={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")

    w1, w1_s = quant_weights(N1, K, gen, dev)
    w2, w2_s = quant_weights(K, INTER, gen, dev)
    fails = 0
    rows = []
    for M in [int(m) for m in args.ms.split(",")]:
        config = try_get_optimal_moe_config(w1.shape, w2.shape, TOPK, "fp8_w8a8", M, block_shape=BLOCK)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=gen)
        a1, a1_s = per_token_group_quant_fp8(x, 32)
        topk_ids, topk_w = make_routing(M, gen, dev)
        sorted_ids, expert_ids, npp = _prepare_expert_assignment(topk_ids, config, M, TOPK, E, None, block_shape=BLOCK)
        naive = sorted_ids is None
        assert moe_gemv_applicable(a1, w1, torch.empty(M, TOPK, N1, device=dev, dtype=torch.bfloat16), a1_s, w1_s, None,
                                   sorted_ids, expert_ids, TOPK, config, True, False, BLOCK) or M * TOPK > 320

        def run_triton(A, B, C, As, Bs, tw, mul, tk):
            invoke_fused_moe_triton_kernel(A, B, C, As, Bs, tw, sorted_ids, expert_ids, npp, mul, tk, config,
                                           compute_type=tl.bfloat16, use_fp8_w8a8=True, use_int8_w8a8=False,
                                           use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False,
                                           block_shape=BLOCK, B_bias=None)

        # ---- w13: [M, K] x [E, 2N, K]^T -> [M, topk, 2N]
        c1_t = torch.zeros(M, TOPK, N1, device=dev, dtype=torch.bfloat16)
        c1_o = torch.zeros_like(c1_t)
        run_triton(a1, w1, c1_t, a1_s, w1_s, None, False, TOPK)
        fused_moe_gemv(a1, w1, c1_o, a1_s, w1_s, None, sorted_ids, expert_ids, npp, False, TOPK, config)
        a1_d = dequant_a(a1, a1_s)
        ref1 = torch.stack([torch.einsum("k,enk->en", a1_d[m], dequant_w(w1, w1_s, topk_ids[m].long()))
                            for m in range(M)])  # [M, topk, 2N]
        torch.cuda.synchronize()
        e1_o = ((c1_o.float() - ref1).abs() / (ref1.abs() + 1e-1)).max().item()
        e1_t = ((c1_t.float() - ref1).abs() / (ref1.abs() + 1e-1)).max().item()
        d1 = (c1_o.float() - c1_t.float()).abs().max().item()
        # ---- silu(gate) * up -> fp8 [M*topk, N] -> w2 with routed weights
        h = torch.empty(M * TOPK, INTER, device=dev, dtype=torch.bfloat16)
        torch.ops._C.silu_and_mul(h, c1_t.view(-1, N1))
        a2, a2_s = per_token_group_quant_fp8(h, 32)
        c3_t = torch.zeros(M, TOPK, K, device=dev, dtype=torch.bfloat16)
        c3_o = torch.zeros_like(c3_t)
        run_triton(a2, w2, c3_t, a2_s, w2_s, topk_w, True, 1)
        fused_moe_gemv(a2, w2, c3_o, a2_s, w2_s, topk_w, sorted_ids, expert_ids, npp, True, 1, config)
        a2_d = dequant_a(a2, a2_s).view(M, TOPK, INTER)
        ref3 = torch.stack([torch.einsum("tn,tkn->tk", a2_d[m], dequant_w(w2, w2_s, topk_ids[m].long()))
                            for m in range(M)]) * topk_w[:, :, None]
        torch.cuda.synchronize()
        e3_o = ((c3_o.float() - ref3).abs() / (ref3.abs() + 1e-1)).max().item()
        e3_t = ((c3_t.float() - ref3).abs() / (ref3.abs() + 1e-1)).max().item()
        d3 = (c3_o.float() - c3_t.float()).abs().max().item()
        ok = e1_o < 2e-2 and e3_o < 2e-2 and e1_o < 3 * max(e1_t, 1e-3) and e3_o < 3 * max(e3_t, 1e-3)
        fused_line = ""
        if naive:
            # down GEMM with silu*up + quant fused (naive path only): vs the 3-kernel path
            c3_f = torch.zeros_like(c3_t)
            fused_moe_gemv_act(c1_t.view(-1, N1), w2, c3_f, w2_s, topk_w, expert_ids, npp, True, config)
            torch.cuda.synchronize()
            d3f = (c3_f.float() - c3_t.float()).abs().max().item()
            e3_f = ((c3_f.float() - ref3).abs() / (ref3.abs() + 1e-1)).max().item()
            nz = (c3_f != c3_t).sum().item()
            ok = ok and e3_f < 2e-2 and e3_f < 3 * max(e3_t, 1e-3)
            fused_line = f"\n      w2 fused act+quant: maxrel vs fp32 {e3_f:.2e}  max|fused-triton| {d3f:.3e} ({nz} of {c3_t.numel()} elements differ)"
        fails += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] M={M:2d} {'naive ' if naive else 'aligned'} cfg={config}\n"
              f"      w13: maxrel vs fp32 ours {e1_o:.2e} triton {e1_t:.2e}  max|ours-triton| {d1:.3e}\n"
              f"      w2 : maxrel vs fp32 ours {e3_o:.2e} triton {e3_t:.2e}  max|ours-triton| {d3:.3e}" + fused_line)

        # ---- timing, rotating routing sets so the touched experts are cold in L2
        sets = []
        for _ in range(args.rot):
            ids_r, w_r = make_routing(M, gen, dev)
            s_r, e_r, n_r = _prepare_expert_assignment(ids_r, config, M, TOPK, E, None, block_shape=BLOCK)
            sets.append((s_r, e_r, n_r, w_r))

        def triton_fns(A, B, C, As, Bs, mul, tk):
            out = []
            for s_r, e_r, n_r, w_r in sets:
                out.append(lambda s_r=s_r, e_r=e_r, n_r=n_r, w_r=w_r: invoke_fused_moe_triton_kernel(
                    A, B, C, As, Bs, w_r if mul else None, s_r, e_r, n_r, mul, tk, config, compute_type=tl.bfloat16,
                    use_fp8_w8a8=True, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
                    per_channel_quant=False, block_shape=BLOCK, B_bias=None))
            return out

        def ours_fns(A, B, C, As, Bs, mul, tk, cfg):
            return [lambda s_r=s_r, e_r=e_r, n_r=n_r, w_r=w_r: fused_moe_gemv(
                A, B, C, As, Bs, w_r if mul else None, s_r, e_r, n_r, mul, tk, config, cfg=cfg) for s_r, e_r, n_r, w_r in sets]

        t1_t = bench_graph(triton_fns(a1, w1, c1_t, a1_s, w1_s, False, TOPK), args.iters)
        t3_t = bench_graph(triton_fns(a2, w2, c3_t, a2_s, w2_s, True, 1), args.iters)
        cfgs1 = [kernel_cfg(K, not naive)] + [tuple(int(v) for v in c.split("x")) for c in filter(None, args.cfgs.split(","))]
        cfgs2 = [kernel_cfg(INTER, not naive)] + [tuple(int(v) for v in c.split("x")) for c in filter(None, args.cfgs.split(","))]
        res1 = {c: bench_graph(ours_fns(a1, w1, c1_o, a1_s, w1_s, False, TOPK, c), args.iters) for c in dict.fromkeys(cfgs1)}
        res3 = {c: bench_graph(ours_fns(a2, w2, c3_o, a2_s, w2_s, True, 1, c), args.iters) for c in dict.fromkeys(cfgs2)}
        b1 = min(res1, key=res1.get)
        b3 = min(res3, key=res3.get)
        if naive:
            # act_and_mul + per-group quant + down GEMV (3 launches) vs the fused kernel
            h_b = torch.empty(M * TOPK, INTER, device=dev, dtype=torch.bfloat16)

            def three(s_r, e_r, n_r, w_r):
                torch.ops._C.silu_and_mul(h_b, c1_t.view(-1, N1))
                q, qs = per_token_group_quant_fp8(h_b, 32)
                fused_moe_gemv(q, w2, c3_o, qs, w2_s, w_r, s_r, e_r, n_r, True, 1, config, cfg=b3)

            t_three = bench_graph([lambda s_r=s_r, e_r=e_r, n_r=n_r, w_r=w_r: three(s_r, e_r, n_r, w_r)
                                   for s_r, e_r, n_r, w_r in sets], args.iters)
            t_fused = bench_graph([lambda e_r=e_r, n_r=n_r, w_r=w_r: fused_moe_gemv_act(
                c1_t.view(-1, N1), w2, c3_o, w2_s, w_r, e_r, n_r, True, config) for _, e_r, n_r, w_r in sets], args.iters)
            print(f"      down path: act_and_mul + quant + GEMV {t_three:6.1f} us  ->  fused GEMV {t_fused:6.1f} us")
        # bytes of expert weights touched (upper bound: distinct experts x both matrices)
        mb = M * TOPK * (N1 * K + K * INTER) / 1e6
        print(f"      time: w13 triton {t1_t:6.1f} us  ours {res1[cfgs1[0]]:6.1f} us ({cfgs1[0][0]}x{cfgs1[0][1]})"
              + (f"  best {res1[b1]:6.1f} ({b1[0]}x{b1[1]})" if len(res1) > 1 else "")
              + f" | w2 triton {t3_t:6.1f} us  ours {res3[cfgs2[0]]:6.1f} us ({cfgs2[0][0]}x{cfgs2[0][1]})"
              + (f"  best {res3[b3]:6.1f} ({b3[0]}x{b3[1]})" if len(res3) > 1 else "")
              + f" | sum {t1_t + t3_t:6.1f} -> {res1[b1] + res3[b3]:6.1f} us  (<= {mb:.1f} MB weights)")
        if len(res1) > 1:
            print("      w13 variants: " + ", ".join(f"{c[0]}x{c[1]}={t:.1f}" for c, t in sorted(res1.items(), key=lambda kv: kv[1])))
            print("      w2  variants: " + ", ".join(f"{c[0]}x{c[1]}={t:.1f}" for c, t in sorted(res3.items(), key=lambda kv: kv[1])))
        rows.append((M, t1_t, res1[b1], t3_t, res3[b3]))

    print("\nM   w13 triton -> gemv    w2 triton -> gemv    per-layer sum")
    for M, a, b, c, d in rows:
        print(f"{M:<3d} {a:7.1f} -> {b:6.1f} us   {c:7.1f} -> {d:6.1f} us   {a + c:7.1f} -> {b + d:6.1f} us")
    print(f"\n{'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
