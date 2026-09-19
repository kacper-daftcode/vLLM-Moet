#!/usr/bin/env python3
"""Aggregate GPU kernel time from a torch.profiler Chrome trace (.json or .json.gz).

usage: trace_agg.py TRACE [top_n] [--steps]   (per-kernel table + category subtotals)
"""
import gzip
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
top_n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 40
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as f:
    data = json.load(f)
events = data["traceEvents"] if isinstance(data, dict) else data

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
kern = [e for e in events if e.get("ph") == "X" and e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}]
if not kern:
    print("no GPU events found; cats present:", sorted({e.get("cat") for e in events if e.get("ph") == "X"})[:20])
    sys.exit(1)
t0 = min(e["ts"] for e in kern)
t1 = max(e["ts"] + e["dur"] for e in kern)
wall = t1 - t0
tot = sum(e["dur"] for e in kern)
by_name = defaultdict(lambda: [0.0, 0])
for e in kern:
    n = e["name"]
    by_name[n][0] += e["dur"]
    by_name[n][1] += 1

CATS = [
    ("nccl / allreduce", r"nccl|AllReduce|ncclDevKernel|all_reduce"),
    ("MoE FP4 grouped GEMM (DeepGEMM)", r"fp8_fp4|m_grouped|grouped_gemm|GroupedGemm|sm120_m_grouped"),
    ("MXFP8 dense GEMM (Cutlass/FlashInfer)", r"[Mm]xfp8|Mxfp8Gemm|cutlass.*[Bb]lock[Ss]cale|Sm120.*Gemm|mxfp8_gemm|xe8m0|Fp8Fp8"),
    ("BF16 GEMM (cuBLAS, emulation wo_a etc.)", r"cutlass_80|gemm.*bf16|nvjet|ampere_bf16|sm90_xmma|cublas|Cijk|gemv|bgemm|bmm"),
    ("sparse MLA attention (FlashInfer SM120)", r"sparse_mla|SparseMla|sparse_attn|mla_decode|mla_prefill|SwaMla|dsv4"),
    ("indexer (paged MQA logits / topk)", r"mqa|indexer|paged_mqa|top_k|topk|fp8_index|logits_metadata|lightning"),
    ("mHC / TileLang / hc GEMM", r"mhc|tilelang|hc_prenorm|sinkhorn|hyper"),
    ("engram lookup", r"engram"),
    ("norm / quant / activation (elementwise)", r"rms_norm|RMSNorm|layernorm|quant|silu|swiglu|act_and_mul|elementwise|vectorized|rotary|rope|fused_add|softplus|sigmoid|exp|mul_|add_|copy_|fill|cat|index|scatter|gather|arange|cumsum|where|clamp|masked|repeat|sort|argsort|unique|nonzero|bincount|histc|reduce|sum"),
    ("sampler / rejection / draft glue (Triton)", r"sampl|reject|resample|gumbel|topp|_topk_topp|dflash|ring_slot|markov|count_expert|residual_mass|global_topk|cumulative_log|dspark|speculat|draft"),
    ("memcpy / memset", r"Memcpy|Memset|memcpy|memset"),
]


def classify(n):
    for c, rx in CATS:
        if re.search(rx, n):
            return c
    return "other"


cat_tot = defaultdict(lambda: [0.0, 0])
for n, (d, c) in by_name.items():
    k = classify(n)
    cat_tot[k][0] += d
    cat_tot[k][1] += c

print(f"window: {wall/1000:.1f} ms wall, GPU busy {tot/1000:.1f} ms ({100*tot/wall:.0f}%), {len(kern)} GPU events")
# step count estimate via a per-step marker: number of ncclAllReduce launches / (2*layers) unknown -> print launches
print("--- category subtotals ---")
for k, (d, c) in sorted(cat_tot.items(), key=lambda x: -x[1][0]):
    print(f"{100*d/tot:5.1f}%  {d/1000:8.2f} ms  {c:6d} launches  avg {d/c:6.1f} us  {k}")
print(f"--- top {top_n} kernels ---")
for n, (d, c) in sorted(by_name.items(), key=lambda x: -x[1][0])[:top_n]:
    print(f"{100*d/tot:5.1f}%  {d/1000:8.2f} ms  {c:6d}x  avg {d/c:6.1f} us  {n[:105]}")
