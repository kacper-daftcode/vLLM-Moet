#!/usr/bin/env python3
"""The interpreter's heap frozen again whenever a serving process has worked and gone idle (`VLLM_MOET_GC_IDLE_FREEZE=1`,
default off), and a log line for every slow garbage collection (always on; `VLLM_MOET_GC_WARN_MS`, default 50).

Patches vLLM's `utils/gc_utils.py` (vLLM 0.32.1rc1.dev117+gdf3bd094a): `freeze_gc_heap()` - which the API server, the
EngineCore and every worker call once, when their start-up is done - hands over to
`vllm.utils.moet_gc_idle_freeze.after_startup_freeze()`; the module is installed next to it. With the switch off the only
change is the slow-collection log line. The mechanism and the measurements are in the module's docstring.

Idempotent, anchor-based; the old block must match verbatim. Usage:
    python3 patch_vllm_gc_idle_freeze_sm120.py [--dist /usr/local/lib/python3.12/dist-packages/vllm] [--check]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "# [vllm-moet] gc idle freeze"
REL = "utils/gc_utils.py"
HELPER = "moet_gc_idle_freeze.py"

EDITS: list[tuple[str, str, str]] = [
    (
        "freeze_gc_heap: hand over after the start-up freeze",
        "    gc.collect(2)\n"
        "    # Freeze all GC tracked objects\n"
        "    gc.freeze()\n",
        "    gc.collect(2)\n"
        "    # Freeze all GC tracked objects\n"
        "    gc.freeze()\n"
        f"    {MARKER}: the slow-collection log line; with VLLM_MOET_GC_IDLE_FREEZE=1 the heap is frozen again\n"
        "    # whenever this process has worked and gone idle (vllm/utils/moet_gc_idle_freeze.py)\n"
        "    from vllm.utils import moet_gc_idle_freeze as _moet_gc_idle_freeze\n"
        "\n"
        "    _moet_gc_idle_freeze.after_startup_freeze()\n",
    ),
]


def apply(text: str) -> tuple[str, list[str]]:
    done: list[str] = []
    for name, old, new in EDITS:
        if new in text:
            done.append(f"{name}: already applied")
            continue
        if text.count(old) != 1:
            raise SystemExit(f"anchor for '{name}' found {text.count(old)} times (expected 1) - wrong vLLM version?")
        text = text.replace(old, new, 1)
        done.append(f"{name}: applied")
    return text, done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="/usr/local/lib/python3.12/dist-packages/vllm", help="the vllm package directory")
    ap.add_argument("--check", action="store_true", help="only report whether the patch is applied")
    a = ap.parse_args()
    target = Path(a.dist) / REL
    text = target.read_text()
    helper_dst = target.parent / HELPER
    if a.check:
        n = sum(1 for _, _, new in EDITS if new in text)
        same = helper_dst.exists() and helper_dst.read_bytes() == (HERE / HELPER).read_bytes()
        print(f"{target}: {n}/{len(EDITS)} edits present; helper {'present' if same else 'missing or different'}")
        return 0 if n == len(EDITS) and same else 1
    new_text, done = apply(text)
    compile(new_text, str(target), "exec")
    if new_text != text:
        target.write_text(new_text)
    shutil.copyfile(HERE / HELPER, helper_dst)
    for d in done:
        print(d)
    print(f"helper installed: {helper_dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
