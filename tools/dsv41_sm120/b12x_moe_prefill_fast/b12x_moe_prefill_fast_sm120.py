"""JIT loader + host side of b12x_moe_prefill_fast_sm120.cu: the prefill MoE of b12x's `integration.tp_moe.dynamic`
deterministic path (FC1 + SwiGLU + MXFP8 quant, FC2, top-k sum) as pipelined CUDA kernels, bit for bit with spec B
(../b12x_moe_prefill_spec) and with b12x.

    from b12x_moe_prefill_fast_sm120 import moe_prefill_fast
    out, route, a_values, a_scales, inter, inter_sf, row_pair, counts = moe_prefill_fast(hidden, ids, weights, planes)

`planes` as in the spec loader: dict(w13=prepared.w13_rp, w13_sf=prepared.w13_sfb, w2=prepared.w2_rp, w2_sf=prepared.w2_sfb,
alpha=[E], input_global_scale=[E]|[1], down_alpha=[E], global_scale=[E]|[1]). `route_out` (optional, [>= T*topk, K] bf16):
the route rows go there (b12x's workspace.route_output when integrated); `topk_sum=False` skips the sum (b12x's own
TopKSum runs on the route rows instead). The routing (pair counts, padded expert offsets, the stable scatter, the CTA
work lists) is done on the GPU by the front-end kernel: no host synchronization. Tuning knobs (env, default = measured
best): VLLM_MOET_B12X_MOE_PREFILL_FAST_CFG1 / _CFG2 (stages, CTAs per SM: 0 = (3, 2), 1 = (4, 2)), _DYNAMIC (1 = atomic
work counter, 0 = static round-robin), _GRID1 / _GRID2 (CTA caps, 0 = occupancy). Every configuration computes the same bits.
The block-scaled QMMA needs sm_120a; the extension compiles on first import (~60 s) without fast-math / ftz.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
N_INTERMEDIATE = 576


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_B12X_MOE_PREFILL_FAST_ARCH", "12.0a")
    return load(
        name="vllm_moet_b12x_moe_prefill_fast_sm120",
        sources=[str(_HERE / "b12x_moe_prefill_fast_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def moe_prefill_fast(hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, planes: dict, limit: float = 10.0,
                     n: int = N_INTERMEDIATE, route_out: torch.Tensor | None = None, out: torch.Tensor | None = None,
                     topk_sum: bool = True, cfg1: int | None = None, cfg2: int | None = None,
                     dynamic: int | None = None, grid1: int | None = None, grid2: int | None = None):
    cfg1 = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_CFG1", 0) if cfg1 is None else cfg1
    cfg2 = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_CFG2", 0) if cfg2 is None else cfg2
    dynamic = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_DYNAMIC", 1) if dynamic is None else dynamic
    grid1 = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_GRID1", 0) if grid1 is None else grid1
    grid2 = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_GRID2", 0) if grid2 is None else grid2
    ids = ids.to(torch.int32).contiguous()
    return _ext().moe_prefill_fast(hidden.contiguous(), ids, weights.to(torch.float32).contiguous(), planes["w13"],
                                   planes["w13_sf"], planes["w2"], planes["w2_sf"],
                                   planes["alpha"].to(torch.float32).contiguous(),
                                   planes["input_global_scale"].to(torch.float32).contiguous(),
                                   planes["down_alpha"].to(torch.float32).contiguous(),
                                   planes["global_scale"].to(torch.float32).contiguous(), float(limit), int(n),
                                   route_out, out,
                                   bool(topk_sum), int(cfg1), int(cfg2), int(dynamic), int(grid1), int(grid2))


def supported(*, n: int, k: int, num_topk: int, experts: int, ids_dtype) -> bool:
    """The shapes the kernels handle: intermediate 576, hidden a multiple of 128 (>= 640), <= 512 experts, int32 ids."""
    return int(n) == N_INTERMEDIATE and int(k) % 128 == 0 and int(k) >= 640 and 1 <= int(experts) <= 512 and \
        1 <= int(num_topk) <= 64 and ids_dtype == torch.int32
