#!/usr/bin/env python3
"""FC2 of b12x's compact W4A8 micro MoE path grouped by expert on sm_120 (DeepSeek-V4.1-Flash, --moe-backend b12x).

b12x's direct-route phase 2 computes each (token, expert) pair of a decode step in its own CTAs, so an expert chosen
by k of the step's tokens has its w2 tile read k times. b12x_fc2/b12x_fc2_grouped_sm120.cu computes all pairs of an
expert in one CTA per output tile, with b12x's MMA sequence and epilogue: route_output keeps its bits. On the routing
recorded from the served model (160 decode-step layers, 22.5 distinct experts of 36 pairs) FC2 takes 22.6 instead of
27.6 us per layer on the RTX PRO 6000.

Patched files (flashinfer/experimental/b12x/moe/...):
  _shared/kernels/w4a8_compact_micro.py  _DirectW4A8CompactLaunch(skip_phase2) leaves phase 2 out of the compiled
      launch (in the compile cache key); launch_w4a8_compact_micro runs the grouped FC2 after it. One function,
      _moet_fc2_grouped_on, decides for the compile and the launch: VLLM_MOET_B12X_FC2_GROUPED (default 1), sm_12x, the
      kernel's shapes (intermediate 576, <= 8 tokens, int32 ids, no inline CSF scales) and the extension importable
      from VLLM_MOET_B12X_FC2_DIR (default /opt/vllm-moet/dsv41_sm120/b12x_fc2).
  fused_moe/_preparation.py              _compact_launches compiles the launchers with the same decision.

Usage: python3 patch_flashinfer_b12x_fc2_grouped_sm120.py [--root DIR] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe")
MARKER = "[vllm-moet] sm_120 b12x FC2 grouped by expert"

HELPERS = '''

# {marker}: the grouped FC2 replaces phase 2 of the compiled launch when this holds,
# at compile time (_compact_launches, the fallback below) and at launch time alike.
@functools.lru_cache(maxsize=None)
def _moet_fc2_module():
    import os
    import sys

    path = os.environ.get("VLLM_MOET_B12X_FC2_DIR", "/opt/vllm-moet/dsv41_sm120/b12x_fc2")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import b12x_fc2_grouped_sm120 as module
    except ImportError:
        return None
    return module


# once per process and shape: the compiled launch and the launch must agree even if the environment changes
@functools.lru_cache(maxsize=None)
def _moet_fc2_grouped_on(*, n, k, num_topk, max_tokens, ids_dtype, csf_inline) -> bool:
    import os

    if os.environ.get("VLLM_MOET_B12X_FC2_GROUPED", "1") != "1":
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        return False
    module = _moet_fc2_module()
    return module is not None and module.supported(
        n=int(n), k=int(k), num_topk=int(num_topk), max_tokens=int(max_tokens), ids_dtype=ids_dtype,
        csf_inline=bool(csf_inline),
    )
'''

MICRO_EDITS = [
    # helpers after the imports (functools for the cache)
    ("helpers", "_ALIGN = 256\n", None),
    ("init signature",
     "        fast_math: bool,\n        csf_inline: bool = False,\n    ):\n        self.capacity = max_tokens\n",
     "        fast_math: bool,\n        csf_inline: bool = False,\n        skip_phase2: bool = False,\n    ):\n"
     "        self.capacity = max_tokens\n"),
    ("init field",
     "        self.csf_inline = bool(csf_inline)\n        self.projection = W4A8CompactMicroProjectionKernel(\n",
     "        self.csf_inline = bool(csf_inline)\n        self.skip_phase2 = bool(skip_phase2)\n"
     "        self.projection = W4A8CompactMicroProjectionKernel(\n"),
    ("compile signature",
     "    fast_math: bool,\n    csf_inline: bool = False,\n):\n    launch = _DirectW4A8CompactLaunch(\n",
     "    fast_math: bool,\n    csf_inline: bool = False,\n    skip_phase2: bool = False,\n):\n"
     "    launch = _DirectW4A8CompactLaunch(\n"),
    ("compile launch",
     "        csf_inline=csf_inline,\n    )\n    ids_type = ",
     "        csf_inline=csf_inline,\n        skip_phase2=skip_phase2,\n    )\n    ids_type = "),
    ("fallback compile",
     "        fast_math,\n        csf_inline,\n    )\n    ids = topk_ids.reshape(-1)\n",
     "        fast_math,\n        csf_inline,\n        skip_phase2=fc2_grouped,\n    )\n    ids = topk_ids.reshape(-1)\n"),
    ("decision",
     "    compiled = _prepared_kernel or _compiled_direct_w4a8_compact(\n",
     "    fc2_grouped = _moet_fc2_grouped_on(\n        n=n, k=k, num_topk=num_topk, max_tokens=cap, ids_dtype=topk_ids.dtype,"
     " csf_inline=csf_inline\n    )\n    compiled = _prepared_kernel or _compiled_direct_w4a8_compact(\n"),
    ("run",
     "        current_cuda_stream(),\n    )\n    return route.narrow(0, 0, num_tokens * int(num_topk))\n",
     "        current_cuda_stream(),\n    )\n    if fc2_grouped:\n"
     "        _moet_fc2_module().fc2_grouped(\n            intermediate, w2, w2_scales, ids, weights, alpha2, down_scale, route,\n"
     "            num_tokens * int(num_topk), cap * int(num_topk),\n        )\n"
     "    return route.narrow(0, 0, num_tokens * int(num_topk))\n"),
]

PHASE2_CALL = re.compile(r"\n(        self\.phase2\(\n(?:            .*\n)+?        \)\n)")

PREP_EDITS = [
    ("import",
     "    from b12x.moe._shared.kernels.w4a8_compact_micro import (\n        _compiled_direct_w4a8_compact,\n    )\n",
     "    from b12x.moe._shared.kernels.w4a8_compact_micro import (\n        _compiled_direct_w4a8_compact,\n"
     "        _moet_fc2_grouped_on,\n    )\n"),
    ("compact launches",
     "            caps.w4a16_fast_math,\n            csf_inline,\n        )\n        for dtype in (torch.int32, torch.int64)\n",
     "            caps.w4a16_fast_math,\n            csf_inline,\n            skip_phase2=_moet_fc2_grouped_on(\n"
     "                n=plan.n, k=plan.k, num_topk=plan.num_topk, max_tokens=m, ids_dtype=dtype, csf_inline=csf_inline\n"
     "            ),\n        )\n        for dtype in (torch.int32, torch.int64)\n"),
]


def _apply(src: str, edits, name: str) -> str:
    for what, old, new in edits:
        if src.count(old) != 1:
            raise SystemExit(f"{name}: anchor '{what}' found {src.count(old)} times (expected 1) - b12x changed")
        if new is not None:
            src = src.replace(old, new, 1)
    return src


def patch_micro(src: str) -> str:
    if MARKER in src:
        return src
    if "import functools" not in src:
        src = src.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nimport functools\n", 1)
    src = _apply(src, MICRO_EDITS, "w4a8_compact_micro.py")
    src = src.replace("_ALIGN = 256\n", "_ALIGN = 256\n" + HELPERS.format(marker=MARKER), 1)
    calls = PHASE2_CALL.findall(src)
    if len(calls) != 1:
        raise SystemExit(f"w4a8_compact_micro.py: phase 2 call found {len(calls)} times (expected 1) - b12x changed")
    body = calls[0]
    guarded = "        if cutlass.const_expr(not self.skip_phase2):\n" + "".join(
        "    " + line + "\n" for line in body.rstrip("\n").split("\n")
    )
    src = src.replace(body, guarded, 1)
    keys = src.count('*(("csf_inline",) if csf_inline else ()),')
    if keys != 2:
        raise SystemExit(f"w4a8_compact_micro.py: {keys} compile cache keys (expected 2) - b12x changed")
    src = re.sub(r'( *)(\*\(\("csf_inline",\) if csf_inline else \(\)\),\n)',
                 lambda m: m.group(1) + m.group(2) + m.group(1) + '*(("moet_fc2_grouped",) if skip_phase2 else ()),\n',
                 src)
    return src


def patch_prep(src: str) -> str:
    if "_moet_fc2_grouped_on" in src:
        return src
    return _apply(src, PREP_EDITS, "_preparation.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    files = [(a.root / "_shared/kernels/w4a8_compact_micro.py", patch_micro),
             (a.root / "fused_moe/_preparation.py", patch_prep)]
    for path, fn in files:
        src = path.read_text()
        out = fn(src)
        if a.check:
            print(f"{path}: {'patched' if out == src else 'applies'}")
            continue
        if out != src:
            compile(out, str(path), "exec")
            path.write_text(out)
            print(f"patched {path}")
        else:
            print(f"{path} already patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
