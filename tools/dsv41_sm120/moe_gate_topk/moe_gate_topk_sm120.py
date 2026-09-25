"""JIT loader + vLLM-shaped wrapper for the fused MoE router (moe_gate_topk_sm120.cu).

    from moe_gate_topk_sm120 import RoutedTopK, fused_gate_topk, fused_gate_topk_applicable

    routed = fused_gate_topk(x_bf16, gate_weight_bf16, bias, bias_vl, input_ids, sentinel_lo,
                             top_k=6, routed_scaling_factor=1.5)
    routed.weights  # fp32 [T, top_k]     routed.ids  # int32 [T, top_k]

replaces GateLinear's cuBLAS bf16 GEMM (+ split-K reduce) followed by vLLM's `dsv4_topk`
(sqrtsoftplus, correction bias / bias_vl for image tokens, top-k with ties to the smallest id,
renormalization to routed_scaling_factor) for up to 16 tokens. Numerically equivalent (fp32
accumulation of the same bf16 products, in this kernel's order instead of cuBLAS's split-K order);
the selection differs from vLLM's path only where two scores tie within fp32 rounding.

The first import compiles the extension with torch.utils.cpp_extension.load (~20 s, cached under
$TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
MAX_TOKENS = 16
MAX_TOPK = 8
MAX_EXPERTS = 1024


@dataclass
class RoutedTopK:
    """The router's result, carried in place of router_logits through vLLM's MoE runner."""

    weights: torch.Tensor  # fp32 [T, top_k]
    ids: torch.Tensor  # int32 [T, top_k]

    @property
    def shape(self) -> torch.Size:  # a few call sites read router_logits.shape[0]
        return self.weights.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.weights.dtype

    @property
    def device(self) -> torch.device:
        return self.weights.device


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_MOE_GATE_ARCH", "12.0")
    return load(
        name="vllm_moet_moe_gate_topk_sm120",
        sources=[str(_HERE / "moe_gate_topk_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=bool(int(os.environ.get("VLLM_MOET_MOE_GATE_VERBOSE", "0"))),
    )


_counters: dict[torch.device, torch.Tensor] = {}


def _counter(dev: torch.device) -> torch.Tensor:
    """One zero int32 per device: the kernel's arrival counter (it resets it itself)."""
    c = _counters.get(dev)
    if c is None:
        c = torch.zeros(1, device=dev, dtype=torch.int32)
        _counters[dev] = c
    return c


def fused_gate_topk_applicable(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    top_k: int,
    scoring_func: str,
    renormalize: bool,
) -> bool:
    return (
        scoring_func == "sqrtsoftplus"
        and renormalize
        and bias is not None
        and bias.dtype == torch.float32
        and bias.is_contiguous()
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and (x.stride(0) * 2) % 16 == 0
        and x.data_ptr() % 16 == 0
        and 1 <= x.size(0) <= MAX_TOKENS
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and weight.is_contiguous()
        and weight.size(1) == x.size(1)
        and x.size(1) % 512 == 0
        and x.size(1) <= 5120
        and 1 <= weight.size(0) <= MAX_EXPERTS
        and weight.size(0) % 4 == 0
        and bias.numel() == weight.size(0)
        and 1 <= top_k <= min(MAX_TOPK, weight.size(0))
    )


def new_counter(dev: torch.device) -> torch.Tensor:
    """A zero int32 the kernel uses as its arrival counter and resets before it exits. One per
    layer (or per stream) keeps concurrent launches apart."""
    return torch.zeros(1, device=dev, dtype=torch.int32)


def fused_gate_topk(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    bias_vl: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    sentinel_lo: int,
    top_k: int,
    routed_scaling_factor: float,
    counter: torch.Tensor | None = None,
) -> RoutedTopK:
    T, E = x.size(0), weight.size(0)
    dev = x.device
    scores = torch.empty((T, E), device=dev, dtype=torch.float32)
    topk_w = torch.empty((T, top_k), device=dev, dtype=torch.float32)
    topk_ids = torch.empty((T, top_k), device=dev, dtype=torch.int32)
    use_vl = bias_vl is not None and input_ids is not None and sentinel_lo > 0
    if use_vl and input_ids.dtype != torch.int64:
        input_ids = input_ids.to(torch.int64)
    _ext().moe_gate_topk(
        x, weight, bias, bias_vl if use_vl else None, input_ids.contiguous() if use_vl else None,
        int(sentinel_lo) if use_vl else 0, top_k, float(routed_scaling_factor), scores,
        counter if counter is not None else _counter(dev), topk_w, topk_ids,
    )
    return RoutedTopK(topk_w, topk_ids)


def gate_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The gate GEMV alone: fp32 logits [T, E] (what GateLinear's cuBLAS path returns)."""
    T, E = x.size(0), weight.size(0)
    dev = x.device
    logits = torch.empty((T, E), device=dev, dtype=torch.float32)
    dummy_f = torch.empty(0, device=dev, dtype=torch.float32)
    dummy_i = torch.empty(0, device=dev, dtype=torch.int32)
    bias = _zero_bias(dev, E)
    _ext().moe_gate_topk(x, weight, bias, None, None, 0, 1, 1.0, logits, _counter(dev), dummy_f, dummy_i, True)
    return logits


_zero_biases: dict[tuple[torch.device, int], torch.Tensor] = {}


def _zero_bias(dev: torch.device, E: int) -> torch.Tensor:
    b = _zero_biases.get((dev, E))
    if b is None:
        b = torch.zeros(E, device=dev, dtype=torch.float32)
        _zero_biases[(dev, E)] = b
    return b
