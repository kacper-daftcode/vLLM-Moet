"""JIT loader + vLLM-shaped wrapper for the fused MoE quant+scatter (moe_quant_scatter_sm120.cu).

    from moe_quant_scatter_sm120 import fused_quant_permute, fused_quant_permute_applicable

    aq, aq_scale, expert_ids, inv_perm = fused_quant_permute(
        x_bf16, topk_ids, local_num_experts, M_sum, align_used, aq_out=workspace_rows)

is the drop-in for vLLM's

    aq, aq_scale = per_token_group_quant_fp8(x_bf16, 128)                # fp32 UE8M0 scales
    aq, aq_scale, expert_ids, inv_perm, align = deepgemm_moe_permute(aq, aq_scale, topk_ids, ...)

except that `aq_scale` comes back already in DeepGEMM's packed int32 MN-major layout (what
`transpose_and_pack_fp32_into_ue8m0` would have produced inside the grouped GEMM call), so the
GEMM takes it as is. Same fp8 bytes and scales as vLLM's kernels; the slot each (token, expert)
pair gets inside its expert's block is deterministic here and atomic-ordered in vLLM (the
per-row GEMM results and the gathered output do not depend on it).

The first import compiles the extension with torch.utils.cpp_extension.load (~30 s, cached under
$TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
GROUP_SIZE = 128
FP8_MIN, FP8_MAX = -448.0, 448.0  # torch.float8_e4m3fn
EPS = 1e-10  # per_token_group_quant_fp8 default


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    # sm_120 only (the serving image lists eight architectures in TORCH_CUDA_ARCH_LIST).
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_MOE_QS_ARCH", "12.0")
    # No --use_fast_math: the scale math must match vLLM's per_token_group_quant.cu bit for bit.
    return load(
        name="vllm_moet_moe_quant_scatter_sm120",
        sources=[str(_HERE / "moe_quant_scatter_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=bool(int(os.environ.get("VLLM_MOET_MOE_QS_VERBOSE", "0"))),
    )


MAX_PAIRS = 1024   # M * topk (mirrors kMaxPairs in the .cu)
MAX_EXPERTS = 1024
MAX_K = 8192


def fused_quant_permute_applicable(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
    expert_tokens_meta=None,
) -> bool:
    """The shapes the fused kernel takes: TP without EP (no expert map, no all2all metadata),
    bf16/fp16 rows of a multiple of 128 up to 8192, int32 routing, <= 1024 (token, expert) pairs."""
    return (
        expert_map is None
        and expert_tokens_meta is None
        and x.dim() == 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.stride(1) == 1
        and (x.stride(0) * x.element_size()) % 16 == 0
        and x.data_ptr() % 16 == 0
        and x.size(1) % GROUP_SIZE == 0
        and 0 < x.size(1) <= MAX_K
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and topk_ids.size(0) == x.size(0)
        and 1 <= topk_ids.numel() <= MAX_PAIRS
        and 1 <= local_num_experts <= MAX_EXPERTS
    )


def fused_quant_permute(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    M_sum: int,
    align: int,
    aq_out: torch.Tensor | None = None,
    software_cvt: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (aq [M_sum, K] e4m3, aq_scale int32 [M_sum, ceil(K/128/4)] strides (1, M_sum),
    expert_ids int32 [M_sum] (-1 = padding), inv_perm int32 [M, topk])."""
    M, K = x.shape
    dev = x.device
    if aq_out is None:
        aq_out = torch.empty((M_sum, K), device=dev, dtype=torch.float8_e4m3fn)
    assert aq_out.shape == (M_sum, K), (aq_out.shape, (M_sum, K))
    packed_sf_k = (K // GROUP_SIZE + 3) // 4
    aq_scale = torch.empty_strided((M_sum, packed_sf_k), (1, M_sum), device=dev, dtype=torch.int32)
    expert_ids = torch.empty((M_sum,), device=dev, dtype=torch.int32)
    inv_perm = torch.empty(topk_ids.shape, device=dev, dtype=torch.int32)
    _ext().moe_quant_scatter(
        x, topk_ids, aq_out, aq_scale, expert_ids, inv_perm, local_num_experts, align, EPS, FP8_MIN, FP8_MAX,
        software_cvt,
    )
    return aq_out, aq_scale, expert_ids, inv_perm
