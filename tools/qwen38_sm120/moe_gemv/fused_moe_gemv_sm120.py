"""sm_120 FP8 block-scaled MoE GEMV for vLLM's Triton `fused_moe` path (see the .cu).

    from fused_moe_gemv_sm120 import moe_gemv_applicable, fused_moe_gemv
    if moe_gemv_applicable(A, B, A_scale, B_scale, ...):
        fused_moe_gemv(A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids,
                       expert_ids, num_tokens_post_padded, mul_routed_weight, top_k, config)

The first import compiles the extension with torch.utils.cpp_extension.load (~1 min for
the 32 template instantiations, cached under $TORCH_EXTENSIONS_DIR or
~/.cache/torch_extensions -- the qwen38 launcher bind-mounts /root/.cache).

Environment:
    VLLM_MOET_SM120_MOE_GEMV=0             disable (the patched vLLM falls back to Triton)
    VLLM_MOET_SM120_MOE_GEMV_MAX_PAIRS=320 largest (token, expert) pair count for the GEMV path
    VLLM_MOET_SM120_MOE_GEMV_ALIGNED_MIN_K=512  16-row (moe_align_block_size) launches with a
                                       smaller K stay on Triton (the K = 160 down GEMM)
    VLLM_MOET_SM120_MOE_GEMV_CFG       "K2560:8x5,K2560a:1x4,K160:1x5" -> (ksplit x unroll) per K (a = aligned path)
    VLLM_MOET_SM120_MOE_GEMV_VERBOSE=1 nvcc output
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve().parent
MMA_ROWS = 16
BLOCK_SHAPE = [32, 32]

# (ksplit, unroll) per (K, aligned), tuned on RTX PRO 6000 (2026-09-18) -- see the README.
# naive (one pair per block): K-split over the 8 warps for the long-K gate/up GEMM;
# aligned (16-row blocks, A tile staged in smem): 8 column groups per block.
_DEFAULT_CFG = {(2560, False): (8, 5), (2560, True): (1, 4), (160, False): (1, 5), (160, True): (1, 5)}
_FALLBACK_CFG = {False: (8, 4), True: (1, 4)}


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ.setdefault("VLLM_MOET_GEMV_ARCH", "12.0")
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ["VLLM_MOET_GEMV_ARCH"]
    # no --use_fast_math: the fused silu/quant path mirrors vLLM's act_and_mul and
    # per_token_group_quant kernels (IEEE division, precise expf) to stay bit-identical
    cuda_flags = ["-O3", "-std=c++17"]
    if os.environ.get("VLLM_MOET_SM120_MOE_GEMV_PTXAS", "0") == "1":
        cuda_flags += ["-Xptxas", "-v"]  # register / spill report per instantiation
    return load(
        name="vllm_moet_fused_moe_gemv_sm120",
        sources=[str(_HERE / "fused_moe_gemv_sm120.cu")],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=bool(int(os.environ.get("VLLM_MOET_SM120_MOE_GEMV_VERBOSE", "0"))),
    )


@functools.lru_cache(maxsize=None)
def _cfg_table() -> dict[tuple[int, bool], tuple[int, int]]:
    """Defaults overridden by VLLM_MOET_SM120_MOE_GEMV_CFG="K2560:8x5,K2560a:1x5,K160:1x5"
    (suffix `a` = aligned / 16-row path)."""
    table = dict(_DEFAULT_CFG)
    spec = os.environ.get("VLLM_MOET_SM120_MOE_GEMV_CFG", "")
    for item in filter(None, spec.split(",")):
        k, v = item.split(":")
        k = k.lower().lstrip("k")
        aligned = k.endswith("a")
        ks, un = v.lower().split("x")
        table[(int(k.rstrip("a")), aligned)] = (int(ks), int(un))
    return table


def kernel_cfg(K: int, aligned: bool = False) -> tuple[int, int]:
    return _cfg_table().get((K, aligned), _FALLBACK_CFG[aligned])


def enabled() -> bool:
    return os.environ.get("VLLM_MOET_SM120_MOE_GEMV", "1") != "0"


def max_pairs() -> int:
    """Largest (token, expert) pair count routed to the GEMV (32 tokens x topk 10; measured
    faster than Triton per layer up to there, prefill/larger batches stay on Triton)."""
    return int(os.environ.get("VLLM_MOET_SM120_MOE_GEMV_MAX_PAIRS", "320"))


def aligned_min_k() -> int:
    """Smallest K for which the 16-row (moe_align_block_size) path beats Triton."""
    return int(os.environ.get("VLLM_MOET_SM120_MOE_GEMV_ALIGNED_MIN_K", "512"))


def moe_gemv_applicable(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    B_bias: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    top_k: int,
    config: dict[str, Any],
    use_fp8_w8a8: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None,
) -> bool:
    """True when this launch of vLLM's fused_moe_kernel can go to the GEMV."""
    if not enabled() or not use_fp8_w8a8 or per_channel_quant or B_bias is not None:
        return False
    if block_shape is None or list(block_shape) != BLOCK_SHAPE:
        return False
    if A_scale is None or B_scale is None or A_scale.dim() != 2 or B_scale.dim() != 3:
        return False
    if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn or C.dtype != torch.bfloat16:
        return False
    if A_scale.dtype != torch.float32 or B_scale.dtype != torch.float32:
        return False
    if expert_ids.dtype != torch.int32 or (sorted_token_ids is not None and sorted_token_ids.dtype != torch.int32):
        return False
    K, N = A.size(1), B.size(1)
    if K % 32 or N % 32 or A.stride(1) != 1 or B.stride(2) != 1 or A.stride(0) % 16 or B.stride(1) % 8:
        return False
    if C.dim() != 3 or C.stride(2) != 1:
        return False
    # (token, expert) pairs: A rows x top_k for w13 (rows = tokens), A rows for w2 (top_k = 1)
    if A.size(0) * top_k > max_pairs():
        return False
    if sorted_token_ids is not None:
        if config["BLOCK_SIZE_M"] != MMA_ROWS:
            return False
        # 16-row blocks with a short K (the down GEMM, K = 160): Triton's 2-warp programs
        # keep more weight bytes in flight per SM than this kernel's 256-thread blocks
        # (28 vs 23 us at 16 tokens) -- leave those launches to Triton.
        if K < aligned_min_k():
            return False
    return True


def fused_moe_gemv(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    cfg: tuple[int, int] | None = None,
) -> None:
    """Same contract as vLLM's invoke_fused_moe_triton_kernel for the fp8 [32,32] case."""
    M = A.size(0)
    num_valid_tokens = M * top_k
    block_m_cfg = int(config["BLOCK_SIZE_M"])
    if sorted_token_ids is None:
        # naive assignment: expert_ids = topk_ids.view(-1), block y = pair id
        m_blocks = num_valid_tokens
    else:
        EM = sorted_token_ids.size(0)
        if M < block_m_cfg:
            EM = min(EM, M * top_k * block_m_cfg)
        m_blocks = (EM + MMA_ROWS - 1) // MMA_ROWS
    ksplit, unroll = cfg if cfg is not None else kernel_cfg(A.size(1), sorted_token_ids is not None)
    _ext().fused_moe_gemv(
        A,
        A_scale,
        B,
        B_scale,
        C,
        topk_weights if mul_routed_weight else None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_valid_tokens,
        top_k,
        m_blocks,
        block_m_cfg,
        C.stride(1),
        ksplit,
        unroll,
    )


def fused_act_applicable(
    X: torch.Tensor,
    w2: torch.Tensor,
    C: torch.Tensor,
    w2_scale: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    use_fp8_w8a8: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None,
) -> bool:
    """True when the down GEMM can run with silu(gate)*up + fp8 [32]-group quantization
    fused into the GEMV: naive assignment only (one pair per block), X = the bf16 gate/up
    output viewed as [pairs, 2K]."""
    if not enabled() or fused_act_disabled() or not use_fp8_w8a8 or per_channel_quant or w2_bias is not None:
        return False
    if sorted_token_ids is not None or block_shape is None or list(block_shape) != BLOCK_SHAPE:
        return False
    if w2_scale is None or w2_scale.dim() != 3 or w2_scale.dtype != torch.float32:
        return False
    if X.dtype != torch.bfloat16 or X.dim() != 2 or X.stride(1) != 1 or w2.dtype != torch.float8_e4m3fn:
        return False
    if C.dtype != torch.bfloat16 or C.dim() != 3 or C.stride(2) != 1:
        return False
    K = w2.size(2)
    if X.size(1) != 2 * K or K % 32 or w2.size(1) % 32 or w2.stride(2) != 1 or w2.stride(1) % 8:
        return False
    if expert_ids.dtype != torch.int32 or X.size(0) > max_pairs():
        return False
    return True


def fused_act_disabled() -> bool:
    return os.environ.get("VLLM_MOET_SM120_MOE_GEMV_FUSE_ACT", "1") == "0"


@functools.lru_cache(maxsize=None)
def act_scale_ue8m0() -> bool:
    """vLLM's per_token_group_quant_fp8 rounds activation scales up to powers of two when
    DeepGEMM's E8M0 mode is on (the default on Blackwell); the fused path must match."""
    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import is_deep_gemm_e8m0_used

        return bool(is_deep_gemm_e8m0_used())
    except Exception:  # noqa: BLE001
        return False


def fused_moe_gemv_act(
    X: torch.Tensor,
    w2: torch.Tensor,
    C: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    config: dict[str, Any],
    cfg: tuple[int, int] | None = None,
) -> None:
    """Down GEMM of the naive path with the activation + quantization fused:
    C[pair] = quant32(silu(X[pair, :K]) * X[pair, K:]) @ w2[expert_ids[pair]]^T (* topk_weights[pair])."""
    pairs = X.size(0)
    ksplit, unroll = cfg if cfg is not None else kernel_cfg(w2.size(2), False)
    if ksplit > 2:
        ksplit, unroll = 1, 5
    _ext().fused_moe_gemv_act(
        X,
        w2,
        w2_scale,
        C,
        topk_weights if mul_routed_weight else None,
        expert_ids,
        num_tokens_post_padded,
        pairs,
        pairs,
        int(config["BLOCK_SIZE_M"]),
        C.stride(1),
        ksplit,
        unroll,
        act_scale_ue8m0(),
    )
