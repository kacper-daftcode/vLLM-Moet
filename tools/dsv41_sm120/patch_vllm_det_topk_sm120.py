#!/usr/bin/env python3
"""The DeepSeek-V4.1 DSA indexer's top-k on sm_120 made deterministic: an exact top-k whose result depends only on the
logits (ties to the lower column, selected columns in ascending order).

On sm_120 vLLM's indexer takes its top-512 with top_k_per_row_prefill (prefill) and persistent_topk (decode). Both return
the right values but break ties and order their output differently from run to run, and persistent_topk's sampled path
is approximate on long rows. FlashInfer's sparse MLA is deterministic for identical inputs, but its output bits depend on
the order of the indices it is given, so the served model's outputs differed from run to run (the same prompt, nonce and
fresh prefill with B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1: different text from the 2nd token on).
det_topk/det_topk_sm120.cu replaces both: bit-exact against a torch reference (stable descending sort, first k, ascending
columns) on 702 cases including tied, -inf-masked, NaN/+-0 rows, run-to-run identical, graph-capturable.

Patched files:
  vllm/model_executor/layers/indexer_topk.py        - SparseIndexerTopk.forward (decode rows from seq_lens / next_n)
  vllm/model_executor/layers/sparse_attn_indexer.py - the prefill call of ops.top_k_per_row_prefill (rows relative to
                                                      their start, as vLLM's kernel returns them)
Runtime switch: VLLM_MOET_DET_TOPK=1 (default 0 = vLLM's kernels). VLLM_MOET_DET_TOPK_DIR overrides the kernel directory.

Usage: python3 patch_vllm_det_topk_sm120.py [--dist DIR] [--kernel-dir DIR] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm")
DEFAULT_KERNEL_DIR = "/opt/vllm-moet/dsv41_sm120/det_topk"
MARKER = "[vllm-moet] deterministic indexer top-k"

HELPER = '''

# {marker} (tools/dsv41_sm120/det_topk): exact, ties to the lower column, ascending columns.
_MOET_DET_TOPK = None


def _moet_det_topk():
    global _MOET_DET_TOPK
    if _MOET_DET_TOPK is None:
        import os
        import sys

        _MOET_DET_TOPK = False
        if os.environ.get("VLLM_MOET_DET_TOPK", "0") == "1" and torch.cuda.get_device_capability()[0] == 12:
            d = os.environ.get("VLLM_MOET_DET_TOPK_DIR", "{kernel_dir}")
            if d not in sys.path:
                sys.path.insert(0, d)
            import det_topk_sm120

            det_topk_sm120._ext()
            _MOET_DET_TOPK = det_topk_sm120
    return _MOET_DET_TOPK or None
'''

DECODE_IMPORT_ANCHOR = "from vllm.v1.worker.workspace import current_workspace_manager\n"
DECODE_CALL_ANCHOR = (
    "        backend = self.resolve_backend(logits, topk_tokens, logits.shape[0])\n"
    "        if backend == \"deep_select\":\n"
)
DECODE_CALL_NEW = (
    "        _moet_fn = _moet_det_topk()  # {marker}\n"
    "        if (\n"
    "            _moet_fn is not None\n"
    "            and _moet_fn.supported(logits)\n"
    "            and seq_lens.dtype == torch.int32\n"
    "            and topk_indices.dtype == torch.int32\n"
    "            and topk_indices.stride(-1) == 1\n"
    "        ):\n"
    "            _moet_fn.topk_decode(logits, seq_lens, next_n, topk_indices, topk_tokens)\n"
    "            return\n"
    "        backend = self.resolve_backend(logits, topk_tokens, logits.shape[0])\n"
    "        if backend == \"deep_select\":\n"
)

PREFILL_IMPORT_ANCHOR = "logger = init_logger(__name__)\n"
PREFILL_CALL_ANCHOR = (
    "                ops.top_k_per_row_prefill(\n"
    "                    logits,\n"
    "                    torch.zeros_like(row_ends) if row_starts is None else row_starts,\n"
    "                    row_ends,\n"
    "                    topk_indices,\n"
    "                    num_rows,\n"
    "                    logits.stride(0),\n"
    "                    logits.stride(1),\n"
    "                    topk_tokens,\n"
    "                )\n"
)
PREFILL_CALL_NEW = (
    "                _moet_fn = _moet_det_topk()  # {marker}\n"
    "                if (\n"
    "                    _moet_fn is not None\n"
    "                    and _moet_fn.supported(logits)\n"
    "                    and row_ends.dtype == torch.int32\n"
    "                    and (row_starts is None or row_starts.dtype == torch.int32)\n"
    "                    and topk_indices.dtype == torch.int32\n"
    "                    and topk_indices.stride(-1) == 1\n"
    "                ):\n"
    "                    _moet_fn.topk_prefill(\n"
    "                        logits,\n"
    "                        None if row_starts is None else row_starts.contiguous(),\n"
    "                        row_ends.contiguous(),\n"
    "                        topk_indices,\n"
    "                        num_rows,\n"
    "                        topk_tokens,\n"
    "                        True,\n"
    "                    )\n"
    "                else:\n"
    "                    ops.top_k_per_row_prefill(\n"
    "                        logits,\n"
    "                        torch.zeros_like(row_ends) if row_starts is None else row_starts,\n"
    "                        row_ends,\n"
    "                        topk_indices,\n"
    "                        num_rows,\n"
    "                        logits.stride(0),\n"
    "                        logits.stride(1),\n"
    "                        topk_tokens,\n"
    "                    )\n"
)

FILES = (
    ("model_executor/layers/indexer_topk.py", DECODE_IMPORT_ANCHOR, DECODE_CALL_ANCHOR, DECODE_CALL_NEW),
    ("model_executor/layers/sparse_attn_indexer.py", PREFILL_IMPORT_ANCHOR, PREFILL_CALL_ANCHOR, PREFILL_CALL_NEW),
)


def patch(src: str, import_anchor: str, call_anchor: str, call_new: str, kernel_dir: str) -> str:
    if MARKER in src:
        return src
    for name, anchor in (("import", import_anchor), ("call", call_anchor)):
        if src.count(anchor) != 1:
            raise SystemExit(f"anchor '{name}' found {src.count(anchor)} times (expected 1) - vLLM changed")
    if "\nimport torch\n" not in src:
        raise SystemExit("'import torch' not found - vLLM changed")
    src = src.replace(import_anchor, import_anchor + HELPER.format(marker=MARKER, kernel_dir=kernel_dir), 1)
    return src.replace(call_anchor, call_new.format(marker=MARKER), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    ap.add_argument("--kernel-dir", default=DEFAULT_KERNEL_DIR)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    for rel, imp, call, new in FILES:
        path = a.dist / rel
        src = path.read_text()
        out = patch(src, imp, call, new, a.kernel_dir)
        if a.check:
            print(f"{rel}: {'patched' if MARKER in src else 'applies'}")
        elif out != src:
            path.write_text(out)
            print(f"patched {path}")
        else:
            print(f"{path} already patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
