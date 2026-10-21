"""JIT loader for sparse_mla_ldg_sm120.cu: FlashInfer 0.7.2's DeepSeek-V4.1 mixed-cache sparse MLA decode kernel on
sm_120 instantiated with a gather schedule that loads the candidate rows with plain 16-byte loads (one DRAM round trip
per tile instead of 32-64 serialized bulk copies), and its merge kernel with the partial rows and the attention sink
loaded before the LSE reductions. The attention math, the fp4 -> fp8 conversion and the merge arithmetic are
FlashInfer's own code; output, LSE and the split scratch are bit-identical.

    from sparse_mla_ldg_sm120 import decode, supported
    decode(plan_facts, q, cache, indices, mid, mlse, output, lse, scale, lengths, sink, extra, extra_indices,
           extra_lengths, lse_scale, interleaved)

`plan_facts` is the resolved FlashInfer plan's inspect() dict (implementation 'mixed', cpb, scratch_split_stride,
block_threads, shared_bytes); the tensors are those of PreparedCall.execute. The first import compiles the extension
with torch.utils.cpp_extension.load (~2 min, cached under $TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions; the
serving image compiles it at build time) with FlashInfer's own flags for its sm120f module (fast math, O3, the FP8 /
FP4 enables, compute_120f), against the headers the installed FlashInfer ships. VLLM_MOET_SPARSE_MLA_LDG_MERGE=0
keeps FlashInfer's merge kernel (the hoisted-loads copy is the default).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent


def _flashinfer_data() -> Path:
    import flashinfer

    return Path(flashinfer.__file__).resolve().parent / "data"


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    data = _flashinfer_data()
    inc = data / "include"
    cccl = data / "cccl"
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_SPARSE_MLA_LDG_ARCH", "12.0f")
    flags = ["-std=c++20", "-use_fast_math", "-O3", "-DNDEBUG", "-DFLASHINFER_ENABLE_F16", "-DFLASHINFER_ENABLE_BF16",
             "-DFLASHINFER_ENABLE_FP8_E4M3", "-DFLASHINFER_ENABLE_FP8_E5M2", "-DFLASHINFER_ENABLE_FP8_E8M0",
             "-DFLASHINFER_ENABLE_FP4_E2M1", "-gencode=arch=compute_120f,code=sm_120f"]
    return load(
        name="vllm_moet_sparse_mla_ldg_sm120",
        sources=[str(_HERE / "sparse_mla_ldg_sm120.cu")],
        extra_include_paths=[str(cccl / "libcudacxx/include"), str(cccl / "cub"), str(cccl / "thrust"), str(inc),
                             str(inc / "flashinfer/attention/sparse_mla_sm120")],
        extra_cuda_cflags=flags,
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def supported(plan_facts: dict, q: torch.Tensor, extra, extra_indices) -> bool:
    """The served configuration: the mixed-cache (fp8 SWA + fp4 compressed) decode plan, 16 heads x 512 bf16 queries,
    1-16 splits, int32 indices."""
    try:
        return (
            extra is not None
            and extra_indices is not None
            and plan_facts.get("implementation") == "mixed"
            and 1 <= int(plan_facts["scratch_split_stride"]) <= 16
            and q.dtype == torch.bfloat16
            and q.dim() == 3
            and q.shape[1] == 16
            and q.shape[2] == 512
            and extra_indices.dtype == torch.int32
        )
    except (KeyError, TypeError, AttributeError):
        return False


CALLS = 0  # launches so far (the test's subprocess checks the patched execute really routed here)


def decode(plan_facts: dict, q, cache, indices, mid, mlse, output, lse, scale, lengths, sink, extra, extra_indices,
           extra_lengths, lse_scale, interleaved: bool) -> None:
    global CALLS
    CALLS += 1
    hoisted = os.environ.get("VLLM_MOET_SPARSE_MLA_LDG_MERGE", "1") != "0"
    _ext().sparse_mla_decode_ldg(q, cache, indices, mid, mlse, output, lse, int(plan_facts["scratch_split_stride"]),
                                 float(scale), lengths, sink, extra, extra_indices, extra_lengths, bool(interleaved),
                                 float(lse_scale), int(plan_facts["cpb"]), int(plan_facts["block_threads"]),
                                 int(plan_facts["shared_bytes"]), hoisted)
