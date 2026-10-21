"""JIT loader for dsv4_topk_sm120.cu: DeepSeek-V4 routing top-6, bit-identical to vLLM's Triton dsv4_topk.

    from dsv4_topk_sm120 import dsv4_topk_cuda
    w, ids = dsv4_topk_cuda(gating_output, correction_bias, indices_dtype, routed_scaling_factor,
                            is_padding=None, input_ids=None, bias_vl=None, image_sentinel_lo=0, pdl=True)

Same signature and results as vllm.model_executor.layers.fused_moe.router.dsv4_topk.dsv4_topk for fp32 logits with
256 or 384 experts. VLLM_MOET_DSV4_TOPK_PDL=1 launches with programmatic stream serialization; the default is a plain
launch: in the served decode graph the PDL launch ends the top-k ~3.5 us later (see the README).

The first import compiles the extension with torch.utils.cpp_extension.load (~30 s, cached under
$TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions; the serving image compiles it at build time).
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

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_DSV4_TOPK_ARCH", "12.0")
    # libdevice logf with flush-to-zero (Triton's reflect); no fast-math (logf must stay libdevice's)
    return load(
        name="vllm_moet_dsv4_topk_sm120",
        sources=[str(_HERE / "dsv4_topk_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20", "-ftz=true"],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def dsv4_topk_cuda(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    indices_dtype: torch.dtype,
    routed_scaling_factor: float,
    is_padding: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
    pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = gating_output.shape[0]
    w = gating_output.new_empty((num_tokens, 6), dtype=torch.float32)
    ids = gating_output.new_empty((num_tokens, 6), dtype=indices_dtype)
    if pdl is None:
        pdl = os.environ.get("VLLM_MOET_DSV4_TOPK_PDL", "0") == "1"
    has_vl = bias_vl is not None and image_sentinel_lo > 0
    _ext().dsv4_topk_out(
        gating_output, correction_bias, is_padding, w, ids, float(routed_scaling_factor),
        input_ids if has_vl else None, bias_vl if has_vl else None, int(image_sentinel_lo) if has_vl else 0, bool(pdl),
    )
    return w, ids


def supported(gating_output: torch.Tensor, indices_dtype: torch.dtype) -> bool:
    return (
        gating_output.dtype == torch.float32
        and gating_output.dim() == 2
        and gating_output.shape[1] in (256, 384)
        and gating_output.is_contiguous()
        and indices_dtype in (torch.int32, torch.int64)
        and torch.cuda.get_device_capability(gating_output.device)[0] == 12
    )
