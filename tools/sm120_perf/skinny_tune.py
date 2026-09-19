#!/usr/bin/env python3
"""Tune vLLM's CuTe-DSL skinny GEMM configs for Qwen3.8-Flash-Next per-rank (TP4) BF16 linear
shapes on sm_120. Cold-L2 timing in a CUDA graph (weight copies rotated), compared with cuBLAS
(F.linear). Prints a plan dict for low_latency_gemm.py.

usage (inside the qwen38 image): python3 skinny_tune.py [shapes "NxK,NxK,..."] [Ms "1,2,4,8,16"]
"""
import itertools
import statistics
import sys
import time

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import SkinnyGemmConfig, shape_dynamic_skinny_gemm
from vllm.models.qwen3_8_flash_next.nvidia.low_latency_gemm import QWEN38NEXT_GEMM_PLANS

dev = torch.device("cuda")
DEFAULT_SHAPES = [(4096, 2560), (2560, 1536), (24, 2560), (3584, 2560), (640, 2560), (320, 2560),
                  (336, 10240), (10240, 320), (2560, 160), (512, 2560), (2560, 2560), (1, 2560), (62080, 2560)]
shapes = [tuple(int(v) for v in s.split("x")) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 and sys.argv[1] != "default" else DEFAULT_SHAPES
Ms = [int(m) for m in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4, 8, 16]


def bench(fn, rotate, reps=6):
    per_graph = len(rotate)
    for i in range(2):
        fn(*rotate[i % per_graph])
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(*rotate[0])
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for i in range(per_graph):
                fn(*rotate[i])
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) * 1000 / per_graph)
    return statistics.median(ts)


def candidates(M, N, K):
    out = []
    for bs, opb, ku, vw in itertools.product([32, 64, 128, 256], [1, 2, 4], [1, 2, 4], [8, 4, 2]):
        if N % opb or K % (bs * vw):
            continue
        if K // (bs * vw) < 1:
            continue
        # keep the grid reasonable: blocks = N/opb; skip configs with < 32 blocks unless N is tiny
        out.append(SkinnyGemmConfig(M, bs, opb, k_unroll=ku, vector_width=vw))
    # static_k variant for the K=10240 HC shape (as in the upstream plan)
    if K == 10240:
        for bs, opb in itertools.product([128, 256], [1, 2]):
            if N % opb == 0:
                out.append(SkinnyGemmConfig(M, bs, opb, static_k=K))
    # prefer fewer configs: drop vw=2 unless nothing else fits
    if any(c.vector_width >= 4 for c in out):
        out = [c for c in out if c.vector_width >= 4]
    return out


plan_out = {}
for (N, K) in shapes:
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    n_copies = min(24, max(2, (160 << 20) // (N * K * 2) + 1))
    w_copies = [w.clone() for _ in range(n_copies)]
    for M in Ms:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ref32 = x.float() @ w.float().t()
        t_cublas = bench(lambda wc: torch.nn.functional.linear(x, wc), [(c,) for c in w_copies])
        best = None
        tried = 0
        t0 = time.time()
        for cfg in candidates(M, N, K):
            try:
                y = shape_dynamic_skinny_gemm(x, w, cfg)
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                continue
            err = (y.float() - ref32).abs().max().item()
            scale = ref32.abs().max().item() + 1e-6
            if err > 0.02 * scale + 0.05:
                print(f"   !! {N}x{K} M={M} {cfg}: maxabs {err:.3e} (scale {scale:.2f}) -> rejected")
                continue
            t = bench(lambda wc: shape_dynamic_skinny_gemm(x, wc, cfg), [(c,) for c in w_copies], reps=4)
            tried += 1
            if best is None or t < best[0]:
                best = (t, cfg)
        upstream = QWEN38NEXT_GEMM_PLANS.get((N, K), {}).get(M)
        t_up = None
        if upstream is not None:
            try:
                t_up = bench(lambda wc: shape_dynamic_skinny_gemm(x, wc, upstream), [(c,) for c in w_copies], reps=4)
            except Exception:  # noqa: BLE001
                t_up = None
        if best is None:
            print(f"{N:6d}x{K:<6d} M={M:2d}: no valid skinny config ({tried} tried); cublas {t_cublas:6.1f} us")
            continue
        t_best, cfg = best
        gbps = N * K * 2 / t_best / 1e3
        tag = "SKINNY" if t_best < t_cublas * 0.95 else "cublas"
        print(f"{N:6d}x{K:<6d} M={M:2d}: cublas {t_cublas:6.1f} us | best skinny {t_best:6.1f} us ({gbps:5.0f} GB/s) {cfg} "
              f"| upstream {f'{t_up:6.1f} us {upstream}' if t_up is not None else '-'} | {tried} cfgs in {time.time()-t0:.0f}s -> {tag}")
        if tag == "SKINNY":
            plan_out.setdefault((N, K), {})[M] = cfg
    del w_copies
    torch.cuda.empty_cache()

print("\n# ---- tuned plan (sm_120) ----")
for (N, K), d in plan_out.items():
    print(f"    ({N}, {K}): {{")
    for M, cfg in sorted(d.items()):
        args = [str(cfg.num_rows), str(cfg.block_size), str(cfg.outputs_per_block)]
        if cfg.k_unroll != 1:
            args.append(f"k_unroll={cfg.k_unroll}")
        if cfg.vector_width != 8:
            args.append(f"vector_width={cfg.vector_width}")
        if cfg.static_k is not None:
            args.append(f"static_k={cfg.static_k}")
        print(f"        {M}: SkinnyGemmConfig({', '.join(args)}),")
    print("    },")
