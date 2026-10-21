"""JIT loader for b12x_moe_fused_sm120.cu: the decode MoE of b12x's compact W4A8 micro path (FC1 -> activation -> FC2)
in one persistent kernel, with the same projections / intermediate / route_output bits as b12x's three kernels.

    from b12x_moe_fused_sm120 import moe_fused, supported
    moe_fused(a_values, a_scales, w13, w13_scales, projections, intermediate, w2, w2_scales, topk_ids, topk_weights,
              alpha1, input_scale, alpha2, down_scale, route_output, num_pairs, num_topk, rows_capacity, k, n,
              swiglu_limit)

Arguments are launch_w4a8_compact_micro's after its rows-quant: the scratch's `a_values` / `a_scales` (the quantized
tokens), the prepared n64-repacked w13 / w2 with their scales, the scratch's `projections` and `intermediate` regions
(the kernel's own staging, left as b12x leaves them), flat int32 ids and fp32 router weights of the pairs, the
per-expert alphas and input / down scales, and route_output [rows_capacity, hidden] bf16 (rows < num_pairs written;
pairs with id -1 zeros), which b12x's TopKSum then reduces as before.

The kernel keeps a small control block per device (work counter, per-expert flags), reset by its last CTA so the
launch replays from a CUDA graph. The first import compiles the extension with torch.utils.cpp_extension.load (~60 s,
cached under $TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions; the serving image compiles it at build time). The
block-scaled QMMA needs sm_120a. Tuning: VLLM_MOET_B12X_MOE_FUSED_PF (K64 slices prefetched into L2 ahead of each FC1
stream, default 4), VLLM_MOET_B12X_MOE_FUSED_SPIN_NS (cap of the wait backoff, default 512).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_CTRL: dict[int, torch.Tensor] = {}


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_B12X_MOE_FUSED_ARCH", "12.0a")
    return load(
        name="vllm_moet_b12x_moe_fused_sm120",
        sources=[str(_HERE / "b12x_moe_fused_sm120.cu")],
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


def moe_fused(a_values, a_scales, w13, w13_scales, projections, intermediate, w2, w2_scales, topk_ids, topk_weights,
              alpha1, input_scale, alpha2, down_scale, route_output, num_pairs: int, num_topk: int, rows_capacity: int,
              k: int, n: int, swiglu_limit: float) -> None:
    pf = int(os.environ.get("VLLM_MOET_B12X_MOE_FUSED_PF", "4"))
    spin = int(os.environ.get("VLLM_MOET_B12X_MOE_FUSED_SPIN_NS", "512"))
    _ext().moe_fused(a_values, a_scales, w13, w13_scales, projections, intermediate, w2, w2_scales, topk_ids,
                     topk_weights, alpha1, input_scale, alpha2, down_scale, route_output, _ctrl(route_output.device),
                     int(num_pairs), int(num_topk), int(rows_capacity), int(k), int(n), float(swiglu_limit), 2, pf,
                     None, 3, spin)


def supported(*, n: int, k: int, num_topk: int, max_tokens: int, ids_dtype: torch.dtype, csf_inline: bool,
              experts: int) -> bool:
    """The shapes the kernel computes like b12x: intermediate 576 (n64 tail), hidden % 128, <= 8 tokens, <= 64 pairs,
    <= 384 experts, int32 ids, no inline CSF scales. Also a performance gate: at least 32 pairs per launch - with few
    distinct experts (the DSpark drafter's MoE: 128 experts, top-3) the fused kernel's FC1 phase has too few streams
    and it measured slower than b12x's kernels (6 / 12 distinct experts: +4 / +3 us), while at 22-36 it is faster."""
    return (
        n == 576
        and k % 128 == 0
        and 1 <= max_tokens <= 8
        and 32 <= max_tokens * num_topk <= 64
        and experts <= 384
        and ids_dtype == torch.int32
        and not csf_inline
    )
