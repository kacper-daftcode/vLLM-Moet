#!/usr/bin/env python3
"""Remove per-step host->device synchronizations from vLLM's PLE short-conv
attention metadata builder (``vllm/v1/attention/backends/short_conv_attn.py``).

Background (4x RTX PRO 6000 host, Qwen3.8-Flash-Next-FP8, TP4, MTP k=3, 2026-09-18):
``ShortConvAttentionMetadataBuilder.build()`` moved small CPU index tensors to
the GPU with pageable ``tensor.to(device)`` calls on every decode step. A
pageable H2D copy with ``non_blocking=False`` makes PyTorch synchronize the
current stream, so the CPU waits for the GPU to drain before the next step's
inputs are ready and async scheduling cannot overlap CPU and GPU work. py-spy
attributed ~27 % of the worker main thread to the copy at line 333 and
``nvidia-smi`` showed 16 % of decode wall time with no kernel running.

The fix replaces the five pageable copies with the same pinned, non-blocking
``async_tensor_h2d`` helper the file already uses for ``spec_sequence_masks``.

Usage (idempotent; safe to re-run):
    python3 patch_short_conv_attn.py [--file PATH] [--check]
Inside the vLLM image the default path is the installed module.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/short_conv_attn.py"
)
# Upstream file this patch was written against (nightly v0.1.dev20073+g8e685d198).
KNOWN_UPSTREAM_SHA256 = "985fd9f320eff615f2bc4b4d5576cd7f301a751380709979a251f92101bda14d"
MARKER = "# [vllm-moet] non-blocking H2D copies (short_conv_attn)"

HELPER = f'''

{MARKER}
def _h2d(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host->device copy that never synchronizes the stream.

    A pageable ``t.to(device)`` synchronizes the current stream and drains the
    GPU pipeline once per decode step; ``async_tensor_h2d`` stages through
    pinned memory and copies with ``non_blocking=True``.
    """
    if device.type == "cpu":
        return t
    return async_tensor_h2d(t, device=device)
'''

# (anchor, replacement) pairs; every anchor must occur exactly once.
REPLACEMENTS: list[tuple[str, str]] = [
    (
        "        spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device)\n"
        "        non_spec_req_idx = non_spec_req_idx_cpu.to(query_start_loc.device)\n",
        "        spec_req_idx = _h2d(spec_req_idx_cpu, query_start_loc.device)\n"
        "        non_spec_req_idx = _h2d(non_spec_req_idx_cpu, query_start_loc.device)\n",
    ),
    (
        "            req_group[decode_req_idx_cpu.to(query_start_loc.device)] = 1\n",
        "            req_group[_h2d(decode_req_idx_cpu, query_start_loc.device)] = 1\n",
    ),
    (
        "        num_accepted_tokens = num_accepted_tokens[\n"
        "            spec_req_idx_cpu.to(num_accepted_tokens.device)\n"
        "        ]\n",
        "        num_accepted_tokens = num_accepted_tokens[\n"
        "            spec_req_idx\n"
        "            if spec_req_idx.device == num_accepted_tokens.device\n"
        "            else _h2d(spec_req_idx_cpu, num_accepted_tokens.device)\n"
        "        ]\n",
    ),
    (
        "                non_spec_req_idx = non_spec_req_idx_cpu.to(num_computed_tokens.device)\n"
        "                num_computed_tokens = num_computed_tokens[non_spec_req_idx]\n",
        "                if non_spec_req_idx.device != num_computed_tokens.device:\n"
        "                    non_spec_req_idx = _h2d(\n"
        "                        non_spec_req_idx_cpu, num_computed_tokens.device\n"
        "                    )\n"
        "                num_computed_tokens = num_computed_tokens[non_spec_req_idx]\n",
    ),
]

HELPER_ANCHOR = "from vllm.v1.kv_cache_interface import MambaSpec\n"


def patch_text(src: str) -> tuple[str, bool]:
    """Return (patched_text, changed)."""
    if MARKER in src:
        return src, False
    if src.count(HELPER_ANCHOR) != 1:
        raise SystemExit(f"helper anchor not found exactly once: {HELPER_ANCHOR!r}")
    out = src.replace(HELPER_ANCHOR, HELPER_ANCHOR + HELPER, 1)
    for old, new in REPLACEMENTS:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"anchor found {n} times (expected 1):\n{old}")
        out = out.replace(old, new, 1)
    leftovers = re.findall(r"_cpu\.to\((?:query_start_loc|num_\w+)\.device\)", out)
    if leftovers:
        raise SystemExit(f"unexpected pageable copies left: {leftovers}")
    return out, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true", help="only report patch state")
    args = ap.parse_args()
    src = args.file.read_text()
    sha = hashlib.sha256(src.encode()).hexdigest()
    if MARKER in src:
        print(f"{args.file}: already patched")
        return 0
    if sha != KNOWN_UPSTREAM_SHA256:
        print(
            f"warning: {args.file} sha256 {sha[:12]} differs from the known upstream "
            f"{KNOWN_UPSTREAM_SHA256[:12]}; anchors will be verified individually",
            file=sys.stderr,
        )
    patched, changed = patch_text(src)
    if args.check:
        print(f"{args.file}: patch applies cleanly (not written)")
        return 0
    compile(patched, str(args.file), "exec")  # syntax check before writing
    args.file.write_text(patched)
    print(f"{args.file}: patched ({len(REPLACEMENTS)} sites + helper)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
