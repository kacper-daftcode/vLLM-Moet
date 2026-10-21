"""JIT loader for det_topk_sm120.cu: a deterministic exact top-k for the DeepSeek-V4.1 DSA indexer on sm_120.

    from det_topk_sm120 import topk_prefill, topk_decode, topk_reference
    topk_prefill(logits, row_starts, row_ends, out, num_rows, k, relative)   # vLLM top_k_per_row_prefill's rows
    topk_decode(logits, seq_lens, next_n, out, k)                            # vLLM SparseIndexerTopk's rows

Row r takes the k largest values among its columns; equal values go to the lower column; the selected columns are
written in ascending order, then -1 up to k. vLLM's sm_120 kernels (top_k_per_row_prefill, persistent_topk) break ties
and order their output differently from run to run, and FlashInfer's sparse MLA output depends on the index order.

The first import compiles the extension with torch.utils.cpp_extension.load (~30 s, cached under $TORCH_EXTENSIONS_DIR
or ~/.cache/torch_extensions; the serving image compiles it at build time).
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

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_DET_TOPK_ARCH", "12.0")
    return load(
        name="vllm_moet_det_topk_sm120",
        sources=[str(_HERE / "det_topk_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


# Rows of at most SMALL columns are done by one CTA even when the launch has a cluster per row (on 6 rows one CTA is
# faster up to ~8K columns, a 16-CTA cluster from ~16K).
SMALL = int(os.environ.get("VLLM_MOET_DET_TOPK_SMALL", "12288"))


def topk_prefill(logits: torch.Tensor, row_starts: torch.Tensor | None, row_ends: torch.Tensor, out: torch.Tensor,
                 num_rows: int, k: int, relative: bool, cluster: int = -1, small: int | None = None) -> None:
    """cluster: CTAs per row (-1: the largest of 16 / 8 / 4 / 2 whose clusters for all rows are resident at once, else
    one CTA per row)."""
    _ext().det_topk_prefill(logits, row_starts, row_ends, out, int(num_rows), int(k), bool(relative), int(cluster),
                            SMALL if small is None else int(small))


def topk_decode(logits: torch.Tensor, seq_lens: torch.Tensor, next_n: int, out: torch.Tensor, k: int,
                cluster: int = -1, small: int | None = None) -> None:
    lens = seq_lens if seq_lens.is_contiguous() else seq_lens.contiguous()
    _ext().det_topk_decode(logits, lens, int(next_n), out, int(logits.shape[0]), int(k), int(cluster),
                           SMALL if small is None else int(small))


def decode_row_ends(seq_lens: torch.Tensor, next_n: int, num_rows: int) -> torch.Tensor:
    """vLLM SparseIndexerTopk._row_ends."""
    if seq_lens.numel() == num_rows:
        return seq_lens.reshape(-1)
    off = torch.arange(next_n, dtype=torch.int32, device=seq_lens.device)
    ends = (seq_lens.reshape(-1, 1) - next_n + 1 + off).clamp_(min=0).reshape(-1)
    if ends.numel() < num_rows:
        ends = torch.cat([ends, ends.new_zeros(num_rows - ends.numel())])
    return ends[:num_rows]


def topk_reference(logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, k: int, relative: bool) -> torch.Tensor:
    """The same selection in plain torch (slow): stable descending sort, first k, ascending columns, -1 fill."""
    rows, width = logits.shape
    out = torch.full((rows, k), -1, dtype=torch.int32, device=logits.device)
    for r in range(rows):
        s = max(int(starts[r]), 0)
        e = min(int(ends[r]), width)
        if e <= s:
            continue
        v = logits[r, s:e].float().clone()
        v[torch.isnan(v)] = float("inf")  # NaN above +inf ...
        nan_first = torch.isnan(logits[r, s:e])
        order = torch.sort(v, descending=True, stable=True).indices
        # ... and above every +inf: put NaN columns first, keeping column order among them
        order = torch.cat([order[nan_first[order]], order[~nan_first[order]]])
        sel = order[:k].sort().values.to(torch.int32)
        out[r, : sel.numel()] = sel + (0 if relative else s)
    return out


def supported(logits: torch.Tensor) -> bool:
    return (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.dim() == 2
        and torch.cuda.get_device_capability(logits.device)[0] == 12
    )
