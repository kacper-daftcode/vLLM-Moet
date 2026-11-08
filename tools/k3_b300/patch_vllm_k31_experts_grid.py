#!/usr/bin/env python3
"""Kimi-K3.1 specialized decode (`--additional-config '{"decode_backend":"latency_specialized"}'`) on a 148-SM B300.

The persistent routed-experts kernel of the K3.1 specialized decode path
(`models/kimi_k3/nvidia/specialized_decode/cutedsl/moe/experts_runtime.py`,
`PersistentExperts`) launches a grid whose CTAs wait on one another, so the whole
grid must be resident at once (`check_residency`). Its per-batch grid table was
sized on GB300 (152 SMs: "grouped B9-64 on 152", "wide: every SM" = 152), and
the kimi-k31 image refuses to start the path on an HGX B300 whose GPUs report
148 SMs:

    RuntimeError: Kimi K3 MoE experts: 152 CTAs cannot be co-resident (1 per SM x 148 SMs)

The kernel splits its work by `self.grid` (`cta * count // self.grid` slices and a
grid-wide claim counter for the FC2 tail) and asserts only `grid >= capacity`, so
a smaller resident grid is a legal configuration: this patch caps the table and
the default grid at the device's multiprocessor count. GB300 (152 SMs) is unchanged.

Idempotent, anchor-based. Usage:
    python3 patch_vllm_k31_experts_grid.py [--file PATH] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/moe/experts_runtime.py"
)
MARKER = "# [vllm-moet] the persistent grid capped at the device's SM count (GB300: 152, B300: 148)"

OLD = (
    '        grids += f",65-{WIDE_ROWS}:152"  # wide: every SM\n'
    "        self.grids = batch_table(grids)\n"
)
NEW = (
    '        grids += f",65-{WIDE_ROWS}:152"  # wide: every SM\n'
    "        self.grids = batch_table(grids)\n"
    f"        {MARKER}\n"
    "        # check_residency requires the whole grid resident (one CTA per SM for this\n"
    "        # kernel); the work split is proportional to the grid, grid >= capacity holds.\n"
    "        sms = torch.cuda.get_device_properties(self.device).multi_processor_count\n"
    "        self.grid = min(self.grid, sms)\n"
    "        self.grids = {b: min(g, sms) for b, g in self.grids.items()}\n"
)


def patch_text(src: str) -> tuple[str, str]:
    if MARKER in src:
        return src, "already patched"
    n = src.count(OLD)
    if n != 1:
        raise SystemExit(f"anchor found {n} times (expected 1):\n{OLD}")
    return src.replace(OLD, NEW, 1), "K3.1 experts grid capped at the SM count"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    patched, what = patch_text(src)
    if patched is src:
        print(f"{args.file}: {what}")
        return 0
    compile(patched, str(args.file), "exec")
    if args.check:
        print(f"{args.file}: patch applies cleanly ({what}; not written)")
        return 0
    args.file.write_text(patched)
    print(f"{args.file}: patched ({what})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
