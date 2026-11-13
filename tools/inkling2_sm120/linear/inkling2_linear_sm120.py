"""Inkling2's small-batch bf16 linear on sm_120.

vLLM's `linear` sends batches of up to 32 rows to FlashInfer's CuTe DSL low-M kernels (`mm_bf16(backend="cute-dsl")`),
which FlashInfer provides for capabilities 100, 103 and 107 only. `linear` here is the same operation for capability
12 - bf16 `x @ weight.T`, fp32 accumulation, rounded once:

    out = linear(x, weight)            # x [M <= 32, K] bf16, weight [N, K] bf16 -> [M, N] bf16

By default it is cuBLAS (`torch.nn.functional.linear`), which reads these weights at the memory's rate on sm_120.
VLLM_MOET_INKLING2_LINEAR_KERNEL=1 takes the mma.sync kernel of inkling2_linear_sm120.cu instead (one CTA per 32
output columns, K in cp.async stages of 64 elements, the weight rows prefetched into L2): on one RTX PRO 6000 it ties
with cuBLAS at one row and on the 10 MB `wo`, and is 5 - 12 us behind at 8 - 32 rows of the 27 MB `qkvr`, where every
CTA stages the batch's rows beside its weight rows (test_linear_sm120.py --bench) - so it is off. Shapes the kernel
does not take (K not a multiple of 64, an odd N, non-contiguous or unaligned operands) go to cuBLAS either way.
Tuning: VLLM_MOET_INKLING2_LINEAR_PF (K steps of the weight rows prefetched ahead of the ring, default 4).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
MAX_ROWS = 32
_KERNEL = os.environ.get("VLLM_MOET_INKLING2_LINEAR_KERNEL", "0") == "1"
_PF = int(os.environ.get("VLLM_MOET_INKLING2_LINEAR_PF", "4"))


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_INKLING2_LINEAR_ARCH", "12.0a")
    return load(
        name="vllm_moet_inkling2_linear_sm120",
        sources=[str(_HERE / "inkling2_linear_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def kernel_facts(rows: int = 16) -> dict[str, int]:
    smem, regs, resident = _ext().kernel_facts(int(rows > 16))
    return {"shared_memory_bytes": smem, "registers": regs, "ctas_per_sm": resident}


def takes(x: torch.Tensor, weight: torch.Tensor, kernel: bool | None = None) -> bool:
    """Whether the kernel computes this product (else torch.nn.functional.linear does); `kernel` overrides the
    environment's choice (the tests)."""
    return (
        (_KERNEL if kernel is None else kernel)
        and x.dim() == 2
        and weight.dim() == 2
        and 1 <= x.shape[0] <= MAX_ROWS
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.shape[1] % 64 == 0
        and x.shape[1] == weight.shape[1]
        and weight.shape[0] % 2 == 0
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


def linear(x: torch.Tensor, weight: torch.Tensor, kernel: bool | None = None) -> torch.Tensor:
    """bf16 `x @ weight.T` for a batch of at most 32 rows."""
    if not takes(x, weight, kernel):
        return torch.nn.functional.linear(x, weight)
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device)
    _ext().linear_run(x, weight, out, _PF)
    return out
