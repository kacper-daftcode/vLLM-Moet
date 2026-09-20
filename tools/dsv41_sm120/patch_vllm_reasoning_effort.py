#!/usr/bin/env python3
"""Align vLLM's DeepSeek-V4.1 reasoning-effort tiers with the checkpoint.

Why: vLLM vendored the V4.1 prompt encoder from a pre-release drop
("ds-code-260903") whose string tiers were low 25 / high 50 / xhigh 75 / max 100.
The released checkpoint (deepseek-ai/DeepSeek-V4.1-Flash, encoding/encoding.py)
and the tech report (Table 2, public API tiers) define low 50 / high 75 / max 100
with "high" as the default. With the stale table every thinking-mode request
without an explicit effort renders "Reasoning Effort: 50" -- the tier DeepSeek
calls "low" -- and "low" renders 25, a budget the API does not expose. The rest of
the prompt grammar is identical (checked against encoding/tests goldens).

Patched files (installed vLLM package):
  vllm/tokenizers/deepseek_v41_encoding.py  REASONING_EFFORT_MAPPINGS -> official
      tiers plus OpenAI-vocabulary aliases (minimal / medium / xhigh) so requests
      carrying those names get an interpolated budget instead of HTTP 400. The
      aliases are NOT DeepSeek tiers; the report says intermediate values
      interpolate (sec. 5.1.4), and its Figure 9 evaluates 25..100.
  vllm/tokenizers/deepseek_v41.py           error message lists the accepted names
      from the table instead of a hard-coded set.

No runtime knob: an integer `reasoning_effort` in chat_template_kwargs bypasses
the table entirely, so callers who want a specific budget can always pin it.

Idempotent, anchor-based. Usage:
    python3 patch_vllm_reasoning_effort.py [--vllm-dir DIR] [--check]
Verification (inside the image, optionally against the checkpoint's encoder):
    python3 test_reasoning_effort_encoding.py [--model-dir /model]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_VLLM_DIR = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "# [vllm-moet] DeepSeek-V4.1 reasoning-effort tiers"

# Official tiers (checkpoint encoding/encoding.py, tech report Table 2).
OFFICIAL = {"low": 50, "high": 75, "max": 100}
# OpenAI-vocabulary aliases accepted by vLLM's chat protocol; interpolated
# between the official anchors (minimal below low; medium between low and high;
# xhigh between high and max).
ALIASES = {"minimal": 25, "medium": 62, "xhigh": 87}

MAPPING_OLD = (
    "REASONING_EFFORT_MAPPINGS: Dict[str, int] = {\n"
    '    "low": 25,\n'
    '    "high": 50,\n'
    '    "xhigh": 75,\n'
    '    "max": 100,\n'
    "}\n"
)
MAPPING_NEW = (
    f"{MARKER}: the released checkpoint\n"
    "# (encoding/encoding.py) and the tech report (Table 2) define low 50 / high 75 /\n"
    "# max 100 with \"high\" as the default; the vendored table (25 / 50 / 75 / 100) came\n"
    "# from a pre-release drop. minimal / medium / xhigh are OpenAI-vocabulary aliases,\n"
    "# not DeepSeek tiers: interpolated budgets so such requests do not fail.\n"
    "REASONING_EFFORT_MAPPINGS: Dict[str, int] = {\n"
    + "".join(f'    "{k}": {v},\n' for k, v in OFFICIAL.items())
    + "".join(f'    "{k}": {v},  # alias\n' for k, v in ALIASES.items())
    + "}\n"
)

MESSAGE_OLD = (
    "                raise ValueError(\n"
    '                    "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, "\n'
    '                    "or an integer within [1, 100] in chat_template_kwargs"\n'
    "                )\n"
)
MESSAGE_NEW = (
    "                raise ValueError(\n"
    "                    \"DeepSeek V4.1 reasoning_effort must be one of \"\n"
    "                    + \", \".join(REASONING_EFFORT_MAPPINGS)\n"
    "                    + \" or an integer within [1, 100] in chat_template_kwargs\"\n"
    "                )\n"
)


def patch_encoding(src: str) -> tuple[str, bool]:
    if MARKER in src:
        return src, False
    if src.count(MAPPING_OLD) != 1:
        raise SystemExit(
            f"deepseek_v41_encoding.py: mapping anchor found {src.count(MAPPING_OLD)} "
            "times (expected 1) -- vLLM changed the table, re-check against the checkpoint"
        )
    return src.replace(MAPPING_OLD, MAPPING_NEW, 1), True


def patch_wrapper(src: str) -> tuple[str, bool]:
    if MESSAGE_NEW in src:
        return src, False
    if src.count(MESSAGE_OLD) != 1:
        raise SystemExit(
            f"deepseek_v41.py: message anchor found {src.count(MESSAGE_OLD)} times (expected 1)"
        )
    return src.replace(MESSAGE_OLD, MESSAGE_NEW, 1), True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-dir", type=Path, default=DEFAULT_VLLM_DIR)
    ap.add_argument("--check", action="store_true", help="verify anchors, write nothing")
    args = ap.parse_args()

    targets = [
        (args.vllm_dir / "tokenizers" / "deepseek_v41_encoding.py", patch_encoding),
        (args.vllm_dir / "tokenizers" / "deepseek_v41.py", patch_wrapper),
    ]
    changed = 0
    for path, fn in targets:
        src = path.read_text()
        out, did = fn(src)
        if not did:
            print(f"{path}: already patched")
            continue
        compile(out, str(path), "exec")
        if args.check:
            print(f"{path}: patch applies cleanly (not written)")
            continue
        path.write_text(out)
        changed += 1
        print(f"{path}: patched")
    if changed or args.check:
        tiers = ", ".join(f"{k}={v}" for k, v in {**OFFICIAL, **ALIASES}.items())
        print(f"reasoning-effort tiers: {tiers}; default high=75")
    return 0


if __name__ == "__main__":
    sys.exit(main())
