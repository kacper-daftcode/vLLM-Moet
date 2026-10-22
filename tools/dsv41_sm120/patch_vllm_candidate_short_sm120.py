#!/usr/bin/env python3
"""Step 31: the DSA candidate-block selection on the deterministic top-k instead of torch.topk's ~20 launches (bit-identical).

DeepSeek-V4.1's two-level indexer (candidate_source_layer_id = 20) publishes the top candidate_topk_blocks = 2048 blocks of
candidate_block_size = 8 positions per decode row; later indexers mask their scores to that set (a set: the consumer is
_candidate_flags_kernel). On sm_120 vLLM has no DeepSelect (SM100 only) and `select_candidate_blocks` runs
_block_scores_kernel -> torch.topk over the static max_model_len-wide score row (the mbtopk radix select: fill, memset,
4 digit passes x 3 kernels, kth counts, two scan_by_key, gather, sort - ~20 launches, ~45 us per decode step on the
critical path, inside the FULL decode graph) -> _store_candidates_kernel.

torch.topk's set is "the k largest values, equal values to the lower index, NaN above +inf, -0.0 == +0.0"; the
selection is then stored as the block index, or -1 where the value is -inf. Step 26's deterministic top-k
(det_topk/det_topk_sm120.cu, topk_prefill) implements exactly that order on [start, end) of a row and writes the
selected columns ascending, -1 past the row's column count; one cluster launch per row count serves every context
length, so it replays from the graph. The patch keeps _block_scores_kernel (it also writes each row's valid block
count), hands the score rows to topk_prefill with ends = valid_blocks (the -inf tail beyond them can only ever be
stored as -1), and a small kernel applies the original's -inf -> -1 rule to the selected blocks (the pinned +inf block,
NaN scores and every finite score stay). No host-side decision: the same kernels run for every context length.

Runtime switch VLLM_MOET_CANDIDATE_DET_TOPK=1 (default off); the det_topk module is taken from VLLM_MOET_DET_TOPK_DIR
(default /opt/vllm-moet/dsv41_sm120/det_topk) independently of step 26's own switch.

Usage: python3 patch_vllm_candidate_short_sm120.py [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
CANDIDATES = VLLM / "model_executor/kernels/attention/dsa/candidate_blocks.py"
MARKER = "[vllm-moet] step 31 candidate blocks on the deterministic top-k"

SITE_OLD = (
    "    # Keep the existing top-k tie behavior.\n"
    "    top = scores.topk(min(topk_blocks, nblocks), dim=-1)\n"
)
SITE_NEW = (
    f"    if _moet_candidates_det(scores, valid_blocks, topk_blocks, nblocks, out):  # {MARKER}\n"
    "        return\n"
    "    # Keep the existing top-k tie behavior.\n"
    "    top = scores.topk(min(topk_blocks, nblocks), dim=-1)\n"
)
DEF_ANCHOR = "def select_candidate_blocks(\n"
CODE = f'''# {MARKER} (tools/dsv41_sm120/patch_vllm_candidate_short_sm120.py): the block
# scores' top-k on step 26's deterministic kernel (the k largest, equal values to the lower column, NaN above +inf -
# torch.topk's set) with ends = the rows' valid block counts, then the original's -inf -> -1 rule on the selected
# blocks. The same launches for every context length (graph-safe). VLLM_MOET_CANDIDATE_DET_TOPK=1 enables it.
_MOET_CANDIDATE_DET = None  # None = not resolved, False = off / unavailable, else the det_topk module


def _moet_candidate_det_topk():
    global _MOET_CANDIDATE_DET
    if _MOET_CANDIDATE_DET is None:
        import os
        import sys

        _MOET_CANDIDATE_DET = False
        if (
            os.environ.get("VLLM_MOET_CANDIDATE_DET_TOPK", "0") == "1"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] == 12
        ):
            d = os.environ.get("VLLM_MOET_DET_TOPK_DIR", "/opt/vllm-moet/dsv41_sm120/det_topk")
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                import det_topk_sm120 as _m

                _m._ext()
                _MOET_CANDIDATE_DET = _m
            except Exception as exc:  # noqa: BLE001
                import logging

                logging.getLogger(__name__).warning(
                    "vllm-moet step 31: det_topk unavailable, candidate blocks stay on torch.topk: %r", exc
                )
    return _MOET_CANDIDATE_DET or None


@triton.jit(do_not_specialize=["k"])
def _moet_candidates_fixup_kernel(
    scores,
    output,
    stride_scores,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    TILE: tl.constexpr,
):
    # The original stores a selected block's index, or -1 where its score is -inf (NaN stays); columns past k are -1.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    in_k = cols < k
    block = tl.load(output + row * out_stride_row + cols * out_stride_col, in_k, other=-1)
    picked = in_k & (block >= 0)
    value = tl.load(scores + row * stride_scores + block, picked, other=0.0)
    tl.store(
        output + row * out_stride_row + cols * out_stride_col,
        tl.where(picked & (value != -float("inf")), block, -1),
        cols < OUT_K,
    )


def _moet_candidates_det(scores, valid_blocks, topk_blocks, nblocks, out) -> bool:
    """scores [rows, nblocks] fp32 from _block_scores_kernel (FULL_WIDTH), valid_blocks [rows] int32 -> out [rows,
    topk_blocks] int32; False = not applicable (the caller runs torch.topk)."""
    m = _moet_candidate_det_topk()
    if m is None or not m.supported(scores) or not scores.is_contiguous():
        return False
    if out.dtype != torch.int32 or out.stride(1) != 1 or out.shape[1] < topk_blocks or out.shape[0] < scores.shape[0]:
        return False
    if valid_blocks.dtype != torch.int32 or not valid_blocks.is_contiguous():
        return False
    rows = scores.shape[0]
    k = min(topk_blocks, nblocks)
    m.topk_prefill(scores, None, valid_blocks, out, rows, k, True)
    _moet_candidates_fixup_kernel[(rows, triton.cdiv(topk_blocks, 1024))](
        scores,
        out,
        scores.stride(0),
        *out.stride(),
        k,
        OUT_K=topk_blocks,
        TILE=1024,
    )
    return True


'''


def patch(path: Path, check: bool) -> int:
    text = path.read_text()
    if MARKER in text:
        print(f"already patched: {path}")
        return 0
    if check:
        print(f"NOT patched: {path}")
        return 1
    if text.count(SITE_OLD) != 1:
        sys.exit(f"top-k site not found exactly once in {path} (found {text.count(SITE_OLD)})")
    if text.count(DEF_ANCHOR) != 1:
        sys.exit(f"'{DEF_ANCHOR.strip()}' not found exactly once in {path}")
    text = text.replace(SITE_OLD, SITE_NEW).replace(DEF_ANCHOR, CODE + DEF_ANCHOR)
    path.write_text(text)
    print(f"patched: {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", type=Path, default=CANDIDATES)
    ap.add_argument("--check", action="store_true", help="exit 1 if the patch is not applied")
    a = ap.parse_args()
    return patch(a.path, a.check)


if __name__ == "__main__":
    sys.exit(main())
