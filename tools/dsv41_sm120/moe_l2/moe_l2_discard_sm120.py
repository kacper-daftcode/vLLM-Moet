"""JIT loader for the MoE padding-row L2 discard (see moe_l2_discard_sm120.cu).

    from moe_l2_discard_sm120 import discard_padding_rows
    discard_padding_rows(mm2_out, expert_ids, block_m)   # after DeepGEMM's grouped GEMM wrote mm2_out

The first import compiles the extension with torch.utils.cpp_extension.load (~30 s, cached under
$TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
LINE = 128


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    # the serving image sets TORCH_CUDA_ARCH_LIST for eight architectures; discard.global.L2 needs sm_80+,
    # the image runs on sm_120: build just that, and leave the variable as it was
    saved = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_MOE_L2_ARCH", "12.0")
    try:
        return load(
            name="vllm_moet_moe_l2_discard_sm120",
            sources=[str(_HERE / "moe_l2_discard_sm120.cu")],
            extra_cuda_cflags=["-O3", "-std=c++20"],
            extra_cflags=["-O3", "-std=c++20"],
            verbose=bool(int(os.environ.get("VLLM_MOET_MOE_L2_VERBOSE", "0"))),
        )
    finally:
        if saved is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = saved


def discardable(out: torch.Tensor, m_indices: torch.Tensor) -> bool:
    """Whether out's rows are whole 128-byte lines of its own and m_indices describes them."""
    return (
        out.is_cuda
        and out.dim() == 2
        and out.is_contiguous()
        and (out.size(1) * out.element_size()) % LINE == 0
        and out.data_ptr() % LINE == 0
        and m_indices.is_cuda
        and m_indices.device == out.device
        and m_indices.dtype == torch.int32
        and m_indices.is_contiguous()
        and m_indices.numel() == out.size(0)
    )


def discard_padding_rows(out: torch.Tensor, m_indices: torch.Tensor, block_m: int) -> bool:
    """Drop the padding rows of out's used BLOCK_M tiles from L2 (their content becomes undefined).

    out: the [M_sum, N] output of DeepGEMM's contiguous grouped GEMM, just written on the current stream;
    m_indices: its int32 expert id per row (-1 = padding). Rows outside used tiles (never written) and real rows
    are not touched. Returns False, without launching, when the rows are not whole 128-byte lines.
    """
    if not discardable(out, m_indices) or block_m <= 0:
        return False
    _ext().discard_padding_rows(out, m_indices, int(block_m))
    return True
