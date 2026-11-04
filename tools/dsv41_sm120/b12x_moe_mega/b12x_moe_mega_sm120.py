"""JIT loader for b12x_moe_mega_sm120.cu: the MoE sublayer of b12x's compact W4A8 micro path - the router's spread GEMM,
DeepSeek-V4's routing top-6 and the fused FC1 -> activation -> FC2 (b12x_moe_fused) - in one persistent kernel, bit for
bit the separate kernels'.

    from b12x_moe_mega_sm120 import moe_mega, supported, routing_supported
    moe_mega(a_values, a_scales, w13, w13_scales, projections, intermediate, w2, w2_scales, topk_ids, topk_weights,
             alpha1, input_scale, alpha2, down_scale, route_output, num_pairs, num_topk, rows_capacity, k, n,
             swiglu_limit, routing=None)

Without `routing` the call is b12x_moe_fused_sm120.moe_fused (ids / weights are inputs). With `routing` = dict(x, w, logits,
step, bias, scale, is_padding=None, input_ids=None, bias_vl=None, image_lo=0) the kernel first computes the router
logits (x [M][K] bf16 . w [384][K]^T in `step`-wide K slices, spread's reduction order), the routing scores and the
top-6 of every token, writing `logits`, `topk_weights` and `topk_ids` (int32 [M*6]) itself before the MoE phases.
The slice width `step` is cuBLAS's split for the shape (bf16_gemm's calibrated case of kind "spread").

Measured slower than the separate launches in the served graph (pb-deep: critical path +227 us per step, see the kernel's
header and the README) - an opt-in (Dockerfile build-arg VLLM_MOET_MOE_MEGA_PATCH=1, runtime VLLM_MOET_MOE_MEGA=1).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_CTRL: dict[int, torch.Tensor] = {}
_WS: dict[tuple[int, int], torch.Tensor] = {}
TILES = 48


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_B12X_MOE_MEGA_ARCH", "12.0a")
    # no fast-math and no -ftz (as bf16_gemm and b12x_moe_fused): the spread reduction's adds must keep denormals; the
    # routing scores' explicit PTX (sqrt.approx.ftz, mul/add/div.full) is build-independent and step 35 verified that
    # libdevice logf gives the same bits here as in dsv4_topk's -ftz=true build
    return load(
        name="vllm_moet_b12x_moe_mega_sm120",
        sources=[str(_HERE / "b12x_moe_mega_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def _ctrl(device: torch.device) -> torch.Tensor:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    c = _CTRL.get(idx)
    if c is None:
        ext = _ext()
        c = torch.empty(int(ext.ctrl_ints()), dtype=torch.int32, device=torch.device("cuda", idx))
        ext.ctrl_init(c)
        _CTRL[idx] = c
    return c


def workspace(device: torch.device, slices: int) -> torch.Tensor:
    """The routing workspace of a device: tile partials [48][S][16][8] fp32, then the scores w and keys [16][384] each."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (idx, int(slices))
    ws = _WS.get(key)
    if ws is None:
        ws = torch.zeros(TILES * int(slices) * 128 + 2 * 16 * 384, dtype=torch.float32, device=torch.device("cuda", idx))
        _WS[key] = ws
    return ws


def moe_mega(a_values, a_scales, w13, w13_scales, projections, intermediate, w2, w2_scales, topk_ids, topk_weights,
             alpha1, input_scale, alpha2, down_scale, route_output, num_pairs: int, num_topk: int, rows_capacity: int,
             k: int, n: int, swiglu_limit: float, routing: dict | None = None, pdl: bool | None = None) -> None:
    pf = int(os.environ.get("VLLM_MOET_B12X_MOE_FUSED_PF", "4"))
    spin = int(os.environ.get("VLLM_MOET_B12X_MOE_FUSED_SPIN_NS", "512"))
    if pdl is None:  # VLLM_MOET_MOE_MEGA_PDL=1: the PDL launch (routing weights prefetched before griddepcontrol.wait);
        # default plain: in a graph behind a small kernel the PDL launch measured 6 us slower (the CTA -> SM placement
        # no longer follows blockIdx while the predecessor still holds SMs)
        pdl = os.environ.get("VLLM_MOET_MOE_MEGA_PDL", "0") == "1" and routing is not None
    r = routing or {}
    ws = None
    if routing is not None:
        slices = -(-int(k) // int(r["step"]))
        ws = workspace(route_output.device, slices)
    _ext().moe_mega(a_values, a_scales, w13, w13_scales, projections, intermediate, w2, w2_scales, topk_ids,
                    topk_weights, alpha1, input_scale, alpha2, down_scale, route_output, _ctrl(route_output.device),
                    int(num_pairs), int(num_topk), int(rows_capacity), int(k), int(n), float(swiglu_limit), 2, pf,
                    None, 3, spin,
                    r.get("x"), r.get("w"), r.get("logits"), ws, int(r.get("step", 0)), r.get("bias"),
                    float(r.get("scale", 1.0)), r.get("is_padding"), r.get("input_ids"), r.get("bias_vl"),
                    int(r.get("image_lo", 0)), bool(pdl))


def supported(*, n: int, k: int, num_topk: int, max_tokens: int, ids_dtype: torch.dtype, csf_inline: bool,
              experts: int) -> bool:
    """b12x_moe_fused's shape gate (intermediate 576, hidden % 128, <= 8 tokens, 32..64 pairs, <= 384 experts, int32
    ids, no inline CSF scales)."""
    return (
        n == 576
        and k % 128 == 0
        and 1 <= max_tokens <= 8
        and 32 <= max_tokens * num_topk <= 64
        and experts <= 384
        and ids_dtype == torch.int32
        and not csf_inline
    )


def routing_supported(*, m: int, n: int, k: int, step: int, num_topk: int) -> bool:
    """The routing phase: 384 experts, top-6, 1..8 tokens, a spread split (2..32 slices of a multiple of 16)."""
    slices = -(-int(k) // int(step)) if step > 0 else 0
    return n == 384 and num_topk == 6 and 1 <= m <= 8 and step % 16 == 0 and 2 <= slices <= 32
