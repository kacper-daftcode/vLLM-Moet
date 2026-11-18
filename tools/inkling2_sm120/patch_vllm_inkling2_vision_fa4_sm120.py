#!/usr/bin/env python3
"""Inkling2 step 4: the vision tower's attention on sm_120.

Where: vllm/third_party/tml_fa4/flash_fwd_sm120.py - `FlashAttentionForwardSm120`, the mma.sync forward tml_fa4
selects on capability 12 (the vision tower calls tml_fa4's plain `flash_attn_varlen_func`). The class means to run
the Ampere code paths ("arch = 80 ... no TMA for output"), but the base class's constructor overwrites `arch` with
the device's, so `use_tma_O` comes out true for a kernel that has no TMA atom for the output: the compile fails in
the epilogue (variable-length queries) and the model does not start. vLLM's other copy of the same kernel
(vllm_flash_attn/cute) carries the fix; this is that fix on tml_fa4's copy: the constructor sets `arch` to sm_80
after the base class's.

Usage: python3 patch_vllm_inkling2_vision_fa4_sm120.py [--root DIR] [--check]   (idempotent; refuses any other text)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm/third_party/tml_fa4")
REL = "flash_fwd_sm120.py"
MARKER = "[vllm-moet] sm_120 forward on the Ampere code paths"

IMPORT_ANCHOR = '''import cutlass
import cutlass.utils as utils_basic
'''
IMPORT_NEW = IMPORT_ANCHOR + '''from cutlass.base_dsl.arch import Arch
'''
CLASS_ANCHOR = '''class FlashAttentionForwardSm120(FlashAttentionForwardSm80):
    # Keep arch = 80 to use CpAsync code paths (no TMA for output).
    # The compilation target is determined by the GPU at compile time, not this field.
    arch = 80
'''
CLASS_NEW = CLASS_ANCHOR + f'''
    def __init__(self, *args, **kwargs):
        # {MARKER}: the base constructor sets `arch` to the device's (sm_120), which turns on the TMA output
        # path this kernel has no atom for; the DSL still compiles for the resident GPU.
        super().__init__(*args, **kwargs)
        self.arch = Arch.sm_80
'''

EDITS = ((IMPORT_ANCHOR, IMPORT_NEW), (CLASS_ANCHOR, CLASS_NEW))


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
