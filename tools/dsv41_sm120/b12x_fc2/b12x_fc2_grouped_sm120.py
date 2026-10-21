"""JIT loader for b12x_fc2_grouped_sm120.cu: FC2 of b12x's compact W4A8 micro MoE path grouped by expert, with the
same route_output bits as b12x's direct phase 2.

    from b12x_fc2_grouped_sm120 import fc2_grouped, supported
    fc2_grouped(intermediate, w2, w2_scales, topk_ids, topk_weights, alpha2, down_scale, route_output,
                num_pairs, rows_capacity)

Arguments are launch_w4a8_compact_micro's: the scratch's `intermediate` region, the prepared n64-repacked w2 and its
scales, flat int32 ids and fp32 router weights of the pairs, the per-expert alpha and down scale, route_output
[rows_capacity, hidden] bf16 (rows < num_pairs written).

The first import compiles the extension with torch.utils.cpp_extension.load (~40 s, cached under
$TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions; the serving image compiles it at build time). The block-scaled
QMMA needs sm_120a.
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

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_B12X_FC2_ARCH", "12.0a")
    return load(
        name="vllm_moet_b12x_fc2_grouped_sm120",
        sources=[str(_HERE / "b12x_fc2_grouped_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def fc2_grouped(intermediate, w2, w2_scales, topk_ids, topk_weights, alpha2, down_scale, route_output,
                num_pairs: int, rows_capacity: int) -> None:
    _ext().fc2_grouped(intermediate, w2, w2_scales, topk_ids, topk_weights, alpha2, down_scale, route_output,
                       int(num_pairs), int(rows_capacity))


def supported(*, n: int, k: int, num_topk: int, max_tokens: int, ids_dtype: torch.dtype, csf_inline: bool) -> bool:
    """The shapes the kernel computes like b12x: intermediate 576 (n64 tail), hidden % 128, <= 8 tokens, int32 ids."""
    return (
        n == 576
        and k % 128 == 0
        and 1 <= max_tokens <= 8
        and max_tokens * num_topk <= 64
        and ids_dtype == torch.int32
        and not csf_inline
    )
