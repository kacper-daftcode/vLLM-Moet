"""JIT loader for the sm_120 small-M MXFP8 GEMV (see mxfp8_gemv_sm120.cu).

    from mxfp8_gemv_sm120 import mxfp8_gemv
    c = mxfp8_gemv(a_fp8, a_sf_swizzled, w_fp8, w_sf_swizzled)   # [M, N] bf16, M <= 16

The first import compiles the extension with torch.utils.cpp_extension.load
(~40 s, cached under $TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
MAX_GEMV_M = 16


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    # The serving image sets TORCH_CUDA_ARCH_LIST for eight architectures; this
    # kernel is sm_120-only (uses the e4m3 cvt intrinsics), so compile just that.
    os.environ.setdefault("VLLM_MOET_GEMV_ARCH", "12.0")
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ["VLLM_MOET_GEMV_ARCH"]
    cuda_flags = ["-O3", "--use_fast_math", "-std=c++17"]
    if os.environ.get("VLLM_MOET_GEMV_PTXAS", "0") == "1":
        cuda_flags += ["-Xptxas", "-v"]  # register / spill report per instantiation
    return load(
        name="vllm_moet_mxfp8_gemv_sm120",
        sources=[str(_HERE / "mxfp8_gemv_sm120.cu")],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=bool(int(os.environ.get("VLLM_MOET_GEMV_VERBOSE", "0"))),
    )


def mxfp8_gemv(
    a: torch.Tensor, a_sf: torch.Tensor, w: torch.Tensor, w_sf: torch.Tensor
) -> torch.Tensor:
    """C[M, N] = A[M, K] (e4m3 + F8_128x4 ue8m0 scales) x W[N, K]^T (same format)."""
    return _ext().mxfp8_gemv(a.contiguous(), a_sf.contiguous(), w.contiguous(), w_sf.contiguous())


MAX_GROUPED_GEMV_T = 64


def mxfp8_gemv_grouped(
    a: torch.Tensor, a_sf: torch.Tensor, w: torch.Tensor, w_sf: torch.Tensor
) -> torch.Tensor:
    """Grouped o-projection GEMV (DeepSeek-V4 `wo_a`), T <= 64 tokens.

    a    : [G, T, K] e4m3 contiguous (fused_inv_rope_fp8_quant output, group-major)
    a_sf : int32 [G, T, ceil(K/128)] MN-major view (strides (S*T_al, 1, T_al)) — the
           packed ue8m0 scales fused_inv_rope_fp8_quant(tma_aligned_scales=True) writes
    w    : [G*Ng, K] e4m3 (layer.weight), w_sf: [G*Ng, K/32] uint8 (layer.weight_scale)
    returns [T, G, Ng] bf16
    """
    return _ext().mxfp8_gemv_grouped(a, a_sf, w, w_sf)

