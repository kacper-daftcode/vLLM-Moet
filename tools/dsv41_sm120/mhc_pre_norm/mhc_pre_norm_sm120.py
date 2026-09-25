"""JIT loader + wrapper for the sm_120 mHC pre epilogue (mhc_pre_norm_sm120.cu).

    from mhc_pre_norm_sm120 import mhc_pre_norm, mhc_pre_norm_applicable

    mhc_pre_norm(mixes, sqrsum, hc_scale, hc_base, residual_cur, post, comb, layer_input, norm_weight,
                 pre_mix_in, next_pre_mix, aux_or_None, rms_numel=..., rms_eps=..., hc_pre_eps=...,
                 hc_sinkhorn_eps=..., hc_post_mult_value=..., sinkhorn_repeat=..., norm_eps=...)

has the calling convention of vLLM's `MHC_PRE_NORM_KERNEL(...)` in
`mhc_fused_post_pre_delayed_tilelang` (shifted mHC: use_pre_mix_in=True, save_pre_mix=True) and
writes the same outputs: post, comb, layer_input (RMSNorm'd), next_pre_mix and, when requested, the
stream mean for draft models. One CTA of H/8 threads per token instead of TileLang's 96.

The first import compiles the extension with torch.utils.cpp_extension.load (~20 s, cached under
$TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
HC_MULT = 4
MIX = HC_MULT * (HC_MULT + 2)


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_MHC_ARCH", "12.0")
    return load(
        name="vllm_moet_mhc_pre_norm_sm120",
        sources=[str(_HERE / "mhc_pre_norm_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=bool(int(os.environ.get("VLLM_MOET_MHC_VERBOSE", "0"))),
    )


def mhc_pre_norm_applicable(mixes: torch.Tensor, residual: torch.Tensor, norm_weight, pre_mix_in) -> bool:
    if norm_weight is None or pre_mix_in is None or residual.dim() != 3:
        return False
    T, hc, H = residual.shape
    return (
        hc == HC_MULT
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
        and H % 8 == 0
        and 1024 <= H <= 5120
        and H % 1024 == 0
        and mixes.dim() == 3
        and mixes.shape[1:] == (T, MIX)
        and mixes.dtype == torch.float32
        and mixes.shape[0] <= 16
        and norm_weight.dtype == torch.bfloat16
        and norm_weight.numel() == H
        and pre_mix_in.dtype == torch.float32
        and pre_mix_in.numel() == T * hc
    )


def mhc_pre_norm(
    mixes: torch.Tensor,
    sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    norm_weight: torch.Tensor,
    pre_mix_in: torch.Tensor,
    pre_mix_out: torch.Tensor,
    aux_out: torch.Tensor | None,
    *,
    rms_numel: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
) -> None:
    _ext().mhc_pre_norm(
        mixes.contiguous(), sqrsum.contiguous(), hc_scale.contiguous(), hc_base.contiguous(), residual,
        norm_weight.contiguous(), pre_mix_in.contiguous(), post_mix, comb_mix, layer_input, pre_mix_out, aux_out,
        float(rms_numel), float(rms_eps), float(hc_pre_eps), float(hc_sinkhorn_eps), float(hc_post_mult_value),
        int(sinkhorn_repeat), float(norm_eps),
    )
