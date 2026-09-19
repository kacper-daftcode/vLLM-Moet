#!/usr/bin/env python3
"""Let vLLM's DSA indexer use DeepGEMM's varlen / native multi-row paged MQA
logits on sm_120, exactly as it does on sm_100.

`vllm/v1/attention/backends/mla/indexer.py` gates two DeepGEMM capabilities on
`is_device_capability_family(100)`:

  _supports_varlen_paged_mqa_logits  -> decode rows carry device-side context
        lengths + `indices` (row -> request); required for DSpark
        `enable_adaptive_verification` (the verifier trims drafts on device, so
        CPU and device query lengths disagree) and reads each request's KV once
        for all of its rows.
  _supports_native_decode(next_n)    -> next_n query rows per request in one
        launch (no per-row flattening that re-reads every KV tile).

DeepGEMM 8b1392b9 implements both on arch 12 (csrc/apis/attention.hpp admits
`is_varlen` for arch 10 and 12; sm120 launcher packs next_n in atoms of 2) and
upstream tests/test_attention.py enumerates sm_120 with is_varlen in {False,
True} and next_n 1..6. tools/dsv41_sm120/test_deepgemm_sm120_paged_mqa.py
re-validates both modes against the torch reference on RTX PRO 6000 (2026-09-18:
ref diff ~2.4e-6, bit-exact 64/128-page parity). Without this patch sm_120 falls
back to CPU-uniform flattening (`use_flattening=True supports_varlen=False`) and
the selector rejects adaptive verification ("device-cpu query lens mismatch not
supported").

Idempotent, anchor-based. Usage:
    python3 patch_vllm_indexer_sm120.py [--file PATH] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"
)
MARKER = "# [vllm-moet] sm_120 uses the same DeepGEMM paged-MQA modes as sm_100"

REPLACEMENTS: list[tuple[str, str]] = [
    (
        "def _supports_varlen_paged_mqa_logits() -> bool:\n"
        "    return (\n"
        "        current_platform.is_cuda()\n"
        "        and current_platform.is_device_capability_family(100)\n"
        "        and has_deep_gemm()\n"
        "    )\n",
        "def _supports_varlen_paged_mqa_logits() -> bool:\n"
        f"    {MARKER}\n"
        "    return (\n"
        "        current_platform.is_cuda()\n"
        "        and (\n"
        "            current_platform.is_device_capability_family(100)\n"
        "            or current_platform.is_device_capability_family(120)\n"
        "        )\n"
        "        and has_deep_gemm()\n"
        "    )\n",
    ),
    (
        "    if not (current_platform.is_cuda() and has_deep_gemm()):\n"
        "        return next_n in (1, 2)\n"
        "    if current_platform.is_device_capability_family(100):\n"
        "        return True\n",
        "    if not (current_platform.is_cuda() and has_deep_gemm()):\n"
        "        return next_n in (1, 2)\n"
        "    if current_platform.is_device_capability_family(\n"
        "        100\n"
        "    ) or current_platform.is_device_capability_family(120):\n"
        "        # [vllm-moet] sm120 launcher packs next_n in atoms of 2 (DeepGEMM)\n"
        "        return True\n",
    ),
]


def patch_text(src: str) -> tuple[str, bool]:
    if MARKER in src:
        return src, False
    out = src
    for old, new in REPLACEMENTS:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"anchor found {n} times (expected 1):\n{old}")
        out = out.replace(old, new, 1)
    return out, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    if MARKER in src:
        print(f"{args.file}: already patched")
        return 0
    patched, _ = patch_text(src)
    if args.check:
        print(f"{args.file}: patch applies cleanly (not written)")
        return 0
    compile(patched, str(args.file), "exec")
    args.file.write_text(patched)
    print(f"{args.file}: patched ({len(REPLACEMENTS)} sites)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
