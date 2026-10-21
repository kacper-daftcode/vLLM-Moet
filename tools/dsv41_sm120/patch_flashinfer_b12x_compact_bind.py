#!/usr/bin/env python3
"""b12x W4A8 MoE (MXFP4 weights x MXFP8 activations) through vLLM's `--moe-backend b12x` on DeepSeek-V4.1-Flash, TP4.

vLLM's B12xExperts plans with `fused_moe.plan(fused_moe.Caps(...))` (= plan_tp_moe_scratch) and binds the returned
TPMoEScratchPlan directly. At 576 intermediate rows per rank (2304 / 4, 576 % 128 == 64) FlashInfer 0.7.2's b12x runs
the compact N64 micro kernels, which need the launchers `_compact_launches(launch_plan, caps)` compiles - only the
canonical preparation path (`_FusedMoeState.bind`) attaches them, so the first decode-sized call stopped with
"compact W4A8 execution requires its prepared native launchers" (vLLM 49cb2bbf and df3bd094, FlashInfer 0.7.2).

TPMoEScratchPlan.bind now attaches them too: compiled on the plan's first bind (vLLM binds every plan outside CUDA
graph capture first - the profile run, b12x warmup, the capture's warmup forward) and kept per plan. Plans that do
not select the compact path get None, as before. Same kernels and arguments as the canonical path.

Usage: python3 patch_flashinfer_b12x_compact_bind.py [--file PATH] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe/fused_moe/_impl.py"
)
MARKER = "[vllm-moet] b12x compact bind"

CLASS_ANCHOR = "@dataclass(frozen=True)\nclass TPMoEScratchPlan:\n"
HELPER = f'''# {MARKER}: compact W4A8 launchers for plans bound through the tensor-based API.
_MOET_COMPACT_LAUNCHES: dict[int, tuple[object, object]] = {{}}


def _moet_compact_launches(plan):
    hit = _MOET_COMPACT_LAUNCHES.get(id(plan))
    if hit is None or hit[0] is not plan:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("b12x compact W4A8 launchers must be compiled before CUDA graph capture")
        from ._preparation import _compact_launches

        hit = (plan, _compact_launches(plan.launch_plan, plan.caps))
        _MOET_COMPACT_LAUNCHES[id(plan)] = hit
    return hit[1]


'''
BIND_TAIL_OLD = (
    "            route_pack_launches=route_pack_launches,\n"
    "        )\n"
    "        return replace(\n"
    "            binding,\n"
    "            scales_expanded=bool(scales_expanded),\n"
    "            w4a8_csf_inline=self.caps.w4a8_csf_inline,\n"
    "        )\n"
)
BIND_TAIL_NEW = (
    "            route_pack_launches=route_pack_launches,\n"
    "        )\n"
    f"        # {MARKER}: the compact N64 W4A8 micro path runs on these launchers\n"
    "        compact = _moet_compact_launches(self)\n"
    "        return replace(\n"
    "            binding,\n"
    "            scales_expanded=bool(scales_expanded),\n"
    "            w4a8_csf_inline=self.caps.w4a8_csf_inline,\n"
    "            **({} if compact is None else {\"compact_launches\": compact}),\n"
    "        )\n"
)


def patch_text(src: str) -> str:
    if MARKER in src:
        return src
    for name, anchor in (("TPMoEScratchPlan class", CLASS_ANCHOR), ("bind tail", BIND_TAIL_OLD)):
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"_impl.py: {name} anchor found {n} times (expected 1)")
    if "\nimport torch\n" not in src:
        raise SystemExit("_impl.py: torch is not imported at module level (the helper uses it)")
    out = src.replace(CLASS_ANCHOR, HELPER + CLASS_ANCHOR, 1)
    return out.replace(BIND_TAIL_OLD, BIND_TAIL_NEW, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    if MARKER in src:
        print(f"{args.file}: already patched")
        return 0
    patched = patch_text(src)
    compile(patched, str(args.file), "exec")
    if args.check:
        print(f"{args.file}: patch applies cleanly (not written)")
        return 0
    args.file.write_text(patched)
    print(f"{args.file}: patched (TPMoEScratchPlan.bind attaches the compact W4A8 launchers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
