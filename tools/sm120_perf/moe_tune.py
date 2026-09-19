#!/usr/bin/env python3
"""Focused Triton fused_moe config tuning for Qwen3.8-Flash-Next-FP8 at TP4 on sm_120:
E=512, per-rank N=160 (shard_intermediate 320), K=2560, topk=10, fp8_w8a8, block scales [32,32]
(BLOCK_SIZE_K is capped at 32 by vLLM for this block shape). Decode token counts only; the JSON
also gets default-equivalent entries for large M so the nearest-key lookup stays sane there.

usage (inside the qwen38 image): python3 moe_tune.py OUT.json [Ms]
"""
import itertools
import json
import sys
import time

import torch

sys.path.insert(0, "/vllm-workspace/benchmarks/kernels")
import types  # noqa: E402

if "ray" not in sys.modules:  # benchmark_moe imports ray for its distributed tuner; not needed here
    _ray = types.ModuleType("ray")
    _ray.__path__ = []  # mark as package
    _ray.remote = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    _ray.init = lambda *a, **k: None
    _ray.get = lambda x: x
    _exp = types.ModuleType("ray.experimental")
    _exp.__path__ = []
    _tq = types.ModuleType("ray.experimental.tqdm_ray")
    _tq.tqdm = lambda x=None, **k: x
    _ray.experimental = _exp
    _exp.tqdm_ray = _tq
    sys.modules["ray"] = _ray
    sys.modules["ray.experimental"] = _exp
    sys.modules["ray.experimental.tqdm_ray"] = _tq
from benchmark_moe import benchmark_config  # noqa: E402

from vllm.model_executor.layers.fused_moe.fused_moe import get_default_config  # noqa: E402

torch.set_default_device("cuda")  # benchmark_config allocates with the default device
E, SHARD_N, K, TOPK = 512, 320, 2560, 10
BLOCK = [32, 32]
out_path = sys.argv[1]
Ms = [int(m) for m in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4, 8, 16, 32, 64]


def default_cfg(M):
    c = get_default_config(M, E, SHARD_N // 2, K, TOPK, "fp8_w8a8", BLOCK)
    return {k: c[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages")}


def grid(M):
    bms = [16] if M <= 16 else ([16, 32] if M <= 32 else [16, 32, 64])
    for bm, bn, nw, ns in itertools.product(bms, [32, 64, 160], [2, 4, 8], [2, 3, 4, 5]):
        yield {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1, "num_warps": nw, "num_stages": ns}


def measure(cfg, M, iters=60):
    try:
        return benchmark_config(cfg, M, E, SHARD_N, K, TOPK, torch.bfloat16, True, False, num_iters=iters,
                                block_quant_shape=BLOCK, use_deep_gemm=False)
    except Exception as exc:  # noqa: BLE001
        global _nerr
        _nerr = globals().get("_nerr", 0) + 1
        if _nerr <= 3:
            import traceback; traceback.print_exc()
        return None


results = {}
for M in Ms:
    d = default_cfg(M)
    t_def = measure(d, M)
    best = (t_def, d)
    t0 = time.time()
    n = 0
    for cfg in grid(M):
        t = measure(cfg, M)
        n += 1
        if t is not None and t < best[0]:
            best = (t, cfg)
    print(f"M={M:3d}: default {t_def:7.1f} us {d} -> best {best[0]:7.1f} us {best[1]}  ({n} cfgs, {time.time()-t0:.0f}s)", flush=True)
    results[M] = best[1]

# default-equivalent entries for larger batches (nearest-key lookup in vLLM)
for M in (96, 128, 256, 512, 1024, 2048, 4096, 8192):
    results[M] = default_cfg(M)
with open(out_path, "w") as f:
    json.dump({str(k): v for k, v in sorted(results.items())}, f, indent=2)
print("wrote", out_path)
