#!/usr/bin/env python3
"""Inkling2 step 5: the small-batch bf16 linear on sm_120 (tools/inkling2_sm120/linear).

Where: `linear` in vllm/models/inkling2/ops/linear.py - every projection of the decoder and of the DSpark draft
(qkvr, wo, the router, the draft's MLPs). Batches of up to 32 rows go to FlashInfer's CuTe DSL low-M kernels, which
exist for capabilities 100 / 103 / 107; on capability 12 the call raises and no decode step runs. There the batch
takes inkling2_linear_sm120.linear: the same bf16 `x @ weight.T` with fp32 accumulation - cuBLAS, which is at the
memory's rate for these shapes on sm_120, or with VLLM_MOET_INKLING2_LINEAR_KERNEL=1 the module's mma.sync kernel
(measured: no faster, so off). `k_group_partials` in the
same file (the router's logits and the reference linear: fp32 partials of 1024-wide K groups, summed in order) keeps
its mma.sync kernel for every batch size on capability 12, with the stage count that fits 99 KB of shared memory: the
TMA kernel it takes from 2048 rows needs 197 KB.

Switches: VLLM_MOET_INKLING2_LINEAR=0 leaves vLLM's path (which cannot run on sm_120);
VLLM_MOET_INKLING2_LINEAR_KERNEL=1 computes the small batches with the module's kernel instead of cuBLAS.
VLLM_MOET_INKLING2_DIR is the module directory.

Usage: python3 patch_vllm_inkling2_linear_sm120.py [--root DIR] [--check]   (idempotent; refuses any other file text)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/inkling2")
REL = "ops/linear.py"
MARKER = "[vllm-moet] inkling2 small-batch linear on sm_120"

HELPER_ANCHOR = '''LOW_M_MAX_TOKENS = 32
'''
HELPER = HELPER_ANCHOR + f'''

# {MARKER}: FlashInfer's CuTe DSL low-M kernels are sm_100 kernels; capability 12 takes
# tools/inkling2_sm120/linear. VLLM_MOET_INKLING2_LINEAR=0 turns it off.
_MOET_LINEAR: list = []


def _moet_linear_sm120():
    """The sm_120 small-batch linear module, or None where FlashInfer's kernels apply."""
    if not _MOET_LINEAR:
        import os

        module = None
        capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        if os.environ.get("VLLM_MOET_INKLING2_LINEAR", "1") != "0" and capability[0] == 12:
            import sys

            path = os.path.join(os.environ.get("VLLM_MOET_INKLING2_DIR", "/opt/vllm-moet/inkling2_sm120"), "linear")
            if path not in sys.path:
                sys.path.insert(0, path)
            import inkling2_linear_sm120 as module

            if module._KERNEL:
                module._ext()  # compiled with the image; loaded before any CUDA graph is captured
        _MOET_LINEAR.append(module)
    return _MOET_LINEAR[0]
'''

CALL_ANCHOR = '''        return torch.nn.functional.linear(x, weight)
    from flashinfer import mm_bf16
'''
CALL_NEW = '''        return torch.nn.functional.linear(x, weight)
    moet = _moet_linear_sm120()
    if moet is not None:
        return moet.linear(x, weight)
    from flashinfer import mm_bf16
'''

# `k_group_partials` (the router's and the reference linear's ordered K groups): its TMA kernel stages four 128-wide
# K tiles of both operands (197 KB of shared memory; capability 12 has 99 KB), so sm_120 keeps every batch on the
# mma.sync kernel next to it - the same K groups in the same order - with as many stages as fit.
TMA_ANCHOR = '''    n = weight.shape[0]
    if tokens >= TMA_MIN_TOKENS:
        from triton.tools.tensor_descriptor import TensorDescriptor
'''
TMA_NEW = '''    n = weight.shape[0]
    if tokens >= TMA_MIN_TOKENS and _moet_linear_sm120() is None:
        from triton.tools.tensor_descriptor import TensorDescriptor
'''
STAGES_ANCHOR = '''    num_stages = 4 if not split else 3 if block_n <= 64 else 2
'''
STAGES_NEW = STAGES_ANCHOR + '''    if _moet_linear_sm120() is not None:
        # One stage holds a 128-wide K tile of both operands; sm_120 has 99 KB of shared memory.
        num_stages = max(1, min(num_stages, (101376 - 2048) // ((block_m + block_n) * 128 * 2)))
'''

EDITS = ((HELPER_ANCHOR, HELPER), (CALL_ANCHOR, CALL_NEW), (TMA_ANCHOR, TMA_NEW), (STAGES_ANCHOR, STAGES_NEW))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = a.root / REL
    text = path.read_text()
    if a.check:
        print("patched" if MARKER in text else "not patched")
        return 0 if MARKER in text else 1
    if MARKER in text:
        print(f"already patched {path}")
        return 0
    for anchor, new in EDITS:
        if text.count(anchor) != 1:
            raise SystemExit(f"{REL}: anchor not found exactly once (another vLLM version?):\n{anchor}")
        text = text.replace(anchor, new)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
