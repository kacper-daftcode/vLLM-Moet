"""JIT loader + wrapper for the sm_120 shifted-mHC critical path (mhc_post_norm_sm120.cu).

    from mhc_post_norm_sm120 import mhc_post_norm, mhc_post_norm_applicable

    residual_out, layer_input, aux = mhc_post_norm(x, residual, post_layer_mix, comb_res_mix, pre_mix,
                                                   norm_weight, norm_eps, capture_aux=False)

computes, in one launch per token CTA, what the next sublayer needs from the boundary: the
post-mapped bf16 residual streams (bit-identical to vLLM's `mhc_fused_tilelang` / `mhc_post_tilelang`),
the collapsed + RMSNorm'd layer input and the draft aux (bit-identical to `mhc_pre_big_fuse_with_norm`).
The projection GEMM and the Sinkhorn coefficients are not computed here: the caller runs them on a side
stream (see patch_vllm_mhc_overlap_sm120.py).

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
# The contraction of the post-mix that matches TileLang's generated code (see the test --modes); the
# other two modes exist only so the test can show which one nvcc picked.
POST_MIX_MODE = 1
# Programmatic dependent launch for the critical-path kernel. In the served graph the kernel follows
# the sublayer's NCCL all-reduce as that node's only child and starts 0.2 us after it. Behind a real
# 4-rank all-reduce (RTX PRO 6000, CUDA graph) it starts at the all-reduce's end with PDL and 0.4-0.5 us
# after it without, equally long; its PDL successors (FlashInfer's MXFP8 quantization, the router) can
# only start during it when it is PDL-launched (it releases them right after its wait): 0.8 us per
# attention boundary, 0.4 per FFN boundary. VLLM_MOET_MHC_PDL=0 turns it off.
PDL_DEFAULT = os.environ.get("VLLM_MOET_MHC_PDL", "1") == "1"


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_MHC_ARCH", "12.0")
    return load(
        name="vllm_moet_mhc_post_norm_sm120",
        sources=[str(_HERE / "mhc_post_norm_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=bool(int(os.environ.get("VLLM_MOET_MHC_VERBOSE", "0"))),
    )


def mhc_post_norm_applicable(x: torch.Tensor, residual: torch.Tensor, pre_mix, norm_weight) -> bool:
    if pre_mix is None or norm_weight is None or residual.dim() != 3 or x.dim() != 2:
        return False
    T, hc, H = residual.shape
    return (
        hc == HC_MULT
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and tuple(x.shape) == (T, H)
        and H % 1024 == 0
        and 1024 <= H <= 5120
        and norm_weight.dtype == torch.bfloat16
        and norm_weight.numel() == H
        and pre_mix.dtype == torch.float32
        and pre_mix.numel() == T * hc
    )


def mhc_post_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    pre_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    *,
    capture_aux: bool = False,
    mode: int = POST_MIX_MODE,
    pdl: bool | None = None,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
    ar: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (residual_out [T, hc, H] bf16, layer_input [T, H] bf16, aux [T or 0, H] bf16).

    ar: the fused P2P all-reduce consumer (step 24) - a dict with data, flags, ctrl, rank, m_max from
    p2p_allreduce_sm120.push_for_fused(x): x is then this rank's partial sum and the kernel adds the three peers'
    partials (fixed source order, fp32, rounded once) before the post-mix; a fourth tensor, the reduced x [T, H], is
    then returned too (the side stream's projection reads it).
    """
    if pdl is None:
        pdl = PDL_DEFAULT
    T, hc, H = residual.shape
    if out is None:
        residual_out = torch.empty_like(residual)
        layer_input = torch.empty(T, H, dtype=torch.bfloat16, device=residual.device)
        aux = torch.empty(T if capture_aux else 0, H, dtype=torch.bfloat16, device=residual.device)
    else:
        residual_out, layer_input, aux_ = out
        aux = aux_ if aux_ is not None else torch.empty(0, H, dtype=torch.bfloat16, device=residual.device)
    if ar is None:
        _ext().mhc_post_norm(
            x, residual, post_layer_mix.reshape(T, hc).contiguous(), comb_res_mix.contiguous(), pre_mix.contiguous(),
            norm_weight.contiguous(), residual_out, layer_input, aux if capture_aux else None, float(norm_eps), int(mode),
            bool(pdl),
        )
    else:
        x_out = torch.empty_like(x)
        _ext().mhc_post_norm(
            x, residual, post_layer_mix.reshape(T, hc).contiguous(), comb_res_mix.contiguous(), pre_mix.contiguous(),
            norm_weight.contiguous(), residual_out, layer_input, aux if capture_aux else None, float(norm_eps), int(mode),
            bool(pdl), ar["data"], ar["flags"], ar["ctrl"], int(ar["rank"]), int(ar["m_max"]), x_out,
        )
        return residual_out, layer_input, aux, x_out
    return residual_out, layer_input, aux


PROJ_SPLITS = 8      # mhc_fused_tilelang's split-k for tokens <= 32 (mhc_fused_post_pre_split_config)
PROJ_TOK_BLOCK = 4   # tokens per CTA (the fn slice is read once per block)


def mhc_proj_applicable(x: torch.Tensor, residual: torch.Tensor, n_splits: int = PROJ_SPLITS) -> bool:
    if residual.dim() != 3 or x.dim() != 2:
        return False
    T, hc, H = residual.shape
    return (
        hc == HC_MULT
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and tuple(x.shape) == (T, H)
        and H % (n_splits * 128) == 0
    )


def mhc_proj(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    *,
    n_splits: int = PROJ_SPLITS,
    tok_block: int = PROJ_TOK_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """mixes [n_splits, T, 24] and sqrsum [n_splits, T] of the fp32 post-mix, as mhc_fused_tilelang writes them
    (yp_out / rp_out) - the inputs of the pre-norm epilogue's coefficient path."""
    T, hc, H = residual.shape
    mixes = torch.empty(n_splits, T, HC_MULT * (HC_MULT + 2), dtype=torch.float32, device=residual.device)
    sqrsum = torch.empty(n_splits, T, dtype=torch.float32, device=residual.device)
    _ext().mhc_proj(
        x, residual, post_layer_mix.reshape(T, hc).contiguous(), comb_res_mix.contiguous(), fn.contiguous(), mixes, sqrsum,
        int(tok_block),
    )
    return mixes, sqrsum
