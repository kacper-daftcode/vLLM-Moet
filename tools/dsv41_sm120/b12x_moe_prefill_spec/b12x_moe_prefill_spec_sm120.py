"""JIT loader + host side of b12x_moe_prefill_spec_sm120.cu: spec B of the prefill MoE (b12x `integration.tp_moe.dynamic`,
deterministic output) as plain CUDA kernels - the executable specification the SASS kernel must match bit for bit.

    from b12x_moe_prefill_spec_sm120 import moe_prefill_spec
    out, route, a_values, a_scales, inter, inter_sf = moe_prefill_spec(hidden, ids, weights, planes, limit=10.0)

`planes` = dict(w13=..., w13_sf=..., w2=..., w2_sf=..., alpha=[E] fp32, input_global_scale=[E]|[1], down_alpha=[E],
global_scale=[E]|[1]) - b12x's prepared n64-repacked planes (prepared.w13_rp / w13_sfb / w2_rp / w2_sfb). The pairs are
sorted by expert and padded to 16 rows per expert here (host side, torch ops); the kernels never see the expert-major
source domain of b12x, only (pair, expert) rows, which is all the per-element arithmetic depends on.
The block-scaled QMMA needs sm_120a; the extension compiles on first import (~40 s) without fast-math / ftz.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_B12X_MOE_PREFILL_SPEC_ARCH", "12.0a")
    return load(
        name="vllm_moet_b12x_moe_prefill_spec_sm120",
        sources=[str(_HERE / "b12x_moe_prefill_spec_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def expert_sorted_rows(ids: torch.Tensor, num_experts: int, tile: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """ids [T, topk] int32 -> (row_pair [rows_padded] int32 with -1 padding, tile_expert [rows_padded / tile] int32):
    the pairs (token * topk + slot) grouped by expert in slot-major order, each expert's group padded to a multiple of
    `tile` rows; pairs with id < 0 (or >= num_experts) are left out."""
    flat = ids.reshape(-1).to(torch.int64)
    pair = torch.arange(flat.numel(), device=ids.device, dtype=torch.int64)
    valid = (flat >= 0) & (flat < num_experts)
    e = flat[valid]
    p = pair[valid]
    order = torch.argsort(e, stable=True)
    e, p = e[order], p[order]
    counts = torch.bincount(e, minlength=num_experts)
    padded = (counts + tile - 1) // tile * tile
    starts = torch.cumsum(padded, 0) - padded
    rows_padded = int(padded.sum().item())
    pos = torch.arange(e.numel(), device=ids.device) - (torch.cumsum(counts, 0) - counts)[e]
    row_pair = torch.full((max(rows_padded, tile),), -1, dtype=torch.int32, device=ids.device)
    row_pair[starts[e] + pos] = p.to(torch.int32)
    # the groups are laid out expert after expert, so the tile sequence is expert e repeated padded[e] / tile times
    tile_expert = torch.repeat_interleave(torch.arange(num_experts, device=ids.device, dtype=torch.int32), padded // tile)
    if tile_expert.numel() < row_pair.numel() // tile:  # no valid pair at all: one all-padding tile
        tile_expert = torch.full((row_pair.numel() // tile,), -1, dtype=torch.int32, device=ids.device)
    return row_pair, tile_expert.contiguous()


def moe_prefill_spec(hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, planes: dict, limit: float = 10.0,
                     n: int = 576):
    E = int(planes["alpha"].numel())
    ids = ids.to(torch.int32).contiguous()
    row_pair, tile_expert = expert_sorted_rows(ids, E)
    return _ext().moe_prefill_spec(hidden.contiguous(), ids, weights.to(torch.float32).contiguous(), row_pair,
                                   tile_expert, planes["w13"], planes["w13_sf"], planes["w2"], planes["w2_sf"],
                                   planes["alpha"].to(torch.float32).contiguous(),
                                   planes["input_global_scale"].to(torch.float32).contiguous(),
                                   planes["down_alpha"].to(torch.float32).contiguous(),
                                   planes["global_scale"].to(torch.float32).contiguous(), float(limit), int(n))
