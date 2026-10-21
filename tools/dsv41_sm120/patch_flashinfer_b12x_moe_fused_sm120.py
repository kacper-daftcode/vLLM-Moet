#!/usr/bin/env python3
"""The decode MoE of b12x's compact W4A8 micro path in one persistent kernel on sm_120 (DeepSeek-V4.1-Flash,
--moe-backend b12x): FC1 (projection) -> activation -> FC2 (down projection) with the same bits as b12x's three
kernels (and as step 22's grouped FC2), behind a runtime switch.

b12x's micro path runs rows-quant, projection, activation and phase 2 (or step 22's grouped FC2) as separate kernels;
each of the weight-streaming ones ramps up and drains on its own, and the activation sits between them. The fused
kernel (b12x_moe_fused/b12x_moe_fused_sm120.cu) streams FC1 and FC2 of all the step's experts from one persistent
grid, starts an expert's FC2 as soon as its activation is done and keeps DRAM busy across the phases. On the routing
recorded from the served model (22.5 distinct experts of 36 pairs; 325 W, memory +5000) it takes 65.9 us instead of
projection 41.9 + activation 1.8 + grouped FC2 23.8 us, and the served MoE path captured in a CUDA graph is 2.5 us per
layer shorter (two kernel nodes fewer): about -100 us per decode step at C1.

Patched file (flashinfer/experimental/b12x/moe/_shared/kernels/w4a8_compact_micro.py): launch_w4a8_compact_micro
runs the fused kernel after its rows-quant and returns, when one function, _moet_moe_fused_on, says so:
VLLM_MOET_B12X_MOE_FUSED=1 (default 0: off), sm_12x, the kernel's shapes (intermediate 576, <= 8 tokens, <= 64
pairs, <= 384 experts, int32 ids, no inline CSF scales, a SwiGLU limit) and the extension importable from
VLLM_MOET_B12X_MOE_FUSED_DIR (default /opt/vllm-moet/dsv41_sm120/b12x_moe_fused). b12x's compiled launch (and step
22's decision) stays in place and runs when the switch is off. Applies on top of step 22's patch or without it.

Usage: python3 patch_flashinfer_b12x_moe_fused_sm120.py [--root DIR] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe")
MARKER = "[vllm-moet] sm_120 b12x fused MoE"

HELPERS = '''

# {marker}: FC1 -> activation -> FC2 in one persistent kernel instead of b12x's projection, activation and
# phase 2 (or the grouped FC2), decided once per process and shape.
@functools.lru_cache(maxsize=None)
def _moet_moe_fused_module():
    import os
    import sys

    path = os.environ.get("VLLM_MOET_B12X_MOE_FUSED_DIR", "/opt/vllm-moet/dsv41_sm120/b12x_moe_fused")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import b12x_moe_fused_sm120 as module
    except ImportError:
        return None
    return module


@functools.lru_cache(maxsize=None)
def _moet_moe_fused_on(*, n, k, num_topk, max_tokens, ids_dtype, csf_inline, experts, has_limit) -> bool:
    import os

    if os.environ.get("VLLM_MOET_B12X_MOE_FUSED", "0") != "1":
        return False
    if not has_limit or not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        return False
    module = _moet_moe_fused_module()
    return module is not None and module.supported(
        n=int(n), k=int(k), num_topk=int(num_topk), max_tokens=int(max_tokens), ids_dtype=ids_dtype,
        csf_inline=bool(csf_inline), experts=int(experts),
    )
'''

CALL_ANCHOR = "        _prepared_quantize(a, values, scale_rows, scale_mma)\n\n"
CALL = '''        _prepared_quantize(a, values, scale_rows, scale_mma)

    if _moet_moe_fused_on(
        n=n, k=k, num_topk=num_topk, max_tokens=cap, ids_dtype=topk_ids.dtype, csf_inline=csf_inline, experts=experts,
        has_limit=swiglu_limit is not None,
    ):
        _moet_moe_fused_module().moe_fused(
            values, scale_rows, w13, w13_scales, projections, intermediate, w2, w2_scales, topk_ids.reshape(-1),
            topk_weights.reshape(-1), alpha1, input_scale, alpha2, down_scale, route, num_tokens * int(num_topk),
            int(num_topk), cap * int(num_topk), k, n, float(swiglu_limit),
        )
        return route.narrow(0, 0, num_tokens * int(num_topk))

'''


def patch_micro(src: str) -> str:
    if MARKER in src:
        return src
    if "import functools" not in src:
        src = src.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nimport functools\n", 1)
    if src.count("_ALIGN = 256\n") != 1:
        raise SystemExit("w4a8_compact_micro.py: anchor '_ALIGN = 256' not found once - b12x changed")
    src = src.replace("_ALIGN = 256\n", "_ALIGN = 256\n" + HELPERS.format(marker=MARKER), 1)
    if src.count(CALL_ANCHOR) != 1:
        raise SystemExit("w4a8_compact_micro.py: the rows-quant call site not found once - b12x changed")
    for name in ("values = ", "scale_rows = ", "projections = ", "intermediate = ", "route = ", "experts = ",
                 "num_tokens = ", "cap, k = "):
        if f"    {name}" not in src:
            raise SystemExit(f"w4a8_compact_micro.py: '{name.strip()}' is not defined before the launch - b12x changed")
    return src.replace(CALL_ANCHOR, CALL, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = a.root / "_shared/kernels/w4a8_compact_micro.py"
    src = path.read_text()
    out = patch_micro(src)
    if a.check:
        print(f"{path}: {'patched' if out == src else 'applies'}")
        return 0
    if out != src:
        compile(out, str(path), "exec")
        path.write_text(out)
        print(f"patched {path}")
    else:
        print(f"{path} already patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
