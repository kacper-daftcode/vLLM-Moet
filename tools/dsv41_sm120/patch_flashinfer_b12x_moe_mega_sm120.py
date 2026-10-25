#!/usr/bin/env python3
"""Step 37: the MoE sublayer's routing (router spread GEMM + DeepSeek-V4 top-6) folded into the fused decode MoE launch
(b12x_moe_mega/b12x_moe_mega_sm120.cu), behind a runtime switch - bit-identical by construction and by test.

With VLLM_MOET_MOE_MEGA=1 the router's spread GEMM is deferred by bf16_gemm (as for step 35) and dsv4_topk_sm120 records the
whole routing - x, the router weights, the logits buffer, the slice width, bias, scale, padding and image arguments, the
ids / weights buffers - instead of launching anything (`_defer_routing`); the ids and weights it returns are unwritten
until the MoE launch. This patch makes step 27's launch site in flashinfer's w4a8_compact_micro.py take that record
(`take_pending_routing(topk_ids)`): when the fused path runs and the routing phase supports the shape (384 experts, top-6,
1..8 tokens, a spread split of 2..32 slices) the megakernel computes the logits, the scores, the top-6 and the MoE in one
persistent launch; otherwise the recorded routing runs as step 35's spread + top-6 launch right there, before anything
reads the ids (`run_routing_now`), and the launch proceeds as before. Between the top-k and this site nothing reads the
ids or weights on the served path (b12x flattens them as views; vLLM's b12x experts normalise int32 / fp32 contiguous
tensors as no-ops) - the routing-replay capture and VLLM_MOET_DSV4_TOPK_CHECK would, and are not for this switch.

Requires steps 27 and 34 (their launch-site text). Usage: python3 patch_flashinfer_b12x_moe_mega_sm120.py [--root DIR] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe")
MARKER = "[vllm-moet] step 37 routing folded into the MoE launch"

HELPERS = '''

# {marker}: the deferred routing (dsv4_topk_sm120._defer_routing, VLLM_MOET_MOE_MEGA=1) taken by the launch and
# computed by the megakernel, or run as the separate launches here.
@functools.lru_cache(maxsize=None)
def _moet_mega_module():
    import os
    import sys

    path = os.environ.get("VLLM_MOET_B12X_MOE_MEGA_DIR", "/opt/vllm-moet/dsv41_sm120/b12x_moe_mega")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import b12x_moe_mega_sm120 as module
    except ImportError:
        return None
    return module


@functools.lru_cache(maxsize=None)
def _moet_topk_module():
    import os
    import sys

    path = os.environ.get("VLLM_MOET_DSV4_TOPK_DIR", "/opt/vllm-moet/dsv41_sm120/dsv4_topk")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import dsv4_topk_sm120 as module  # the router's instance (same name, same sys.modules entry)
    except ImportError:
        return None
    return module


def _moet_mega_take(topk_ids, *, fused, experts, k, num_topk, num_tokens):
    import os

    if os.environ.get("VLLM_MOET_MOE_MEGA", "0") != "1":
        return None
    tk = _moet_topk_module()
    if tk is None:
        return None
    rec = tk.take_pending_routing(topk_ids)
    if rec is None:
        if tk._PENDING_ROUTING:  # a deferred routing this launch does not own: run it before anything reads its ids
            tk.run_pending_routing_late()
        return None
    mg = _moet_mega_module() if fused else None
    if mg is not None and mg.routing_supported(m=int(num_tokens), n=int(experts), k=int(k), step=int(rec["step"]), num_topk=int(num_topk)):
        return rec
    tk.run_routing_now(rec)  # the separate launches, before anything reads the ids
    return None
'''

SITE_OLD = """    if _moet_fused:
        _moet_moe_fused_module().moe_fused(
            values, scale_rows, w13, w13_scales, projections, intermediate, w2, w2_scales, topk_ids.reshape(-1),
            topk_weights.reshape(-1), alpha1, input_scale, alpha2, down_scale, route, num_tokens * int(num_topk),
            int(num_topk), cap * int(num_topk), k, n, float(swiglu_limit),
        )
        return route.narrow(0, 0, num_tokens * int(num_topk))
"""
SITE_NEW = f"""    _moet_routing = _moet_mega_take(  # {MARKER}
        topk_ids, fused=_moet_fused, experts=experts, k=k, num_topk=num_topk, num_tokens=num_tokens,
    )
    if _moet_fused:
        if _moet_routing is not None:
            _moet_mega_module().moe_mega(
                values, scale_rows, w13, w13_scales, projections, intermediate, w2, w2_scales, topk_ids.reshape(-1),
                topk_weights.reshape(-1), alpha1, input_scale, alpha2, down_scale, route, num_tokens * int(num_topk),
                int(num_topk), cap * int(num_topk), k, n, float(swiglu_limit), routing=_moet_routing,
            )
        else:
            _moet_moe_fused_module().moe_fused(
                values, scale_rows, w13, w13_scales, projections, intermediate, w2, w2_scales, topk_ids.reshape(-1),
                topk_weights.reshape(-1), alpha1, input_scale, alpha2, down_scale, route, num_tokens * int(num_topk),
                int(num_topk), cap * int(num_topk), k, n, float(swiglu_limit),
            )
        return route.narrow(0, 0, num_tokens * int(num_topk))
"""
HELPER_ANCHOR = "_ALIGN = 256\n"


def patch_micro(src: str) -> str:
    if MARKER in src:
        return src
    if "import functools" not in src:
        raise SystemExit("w4a8_compact_micro.py: step 27's patch (functools import) not found - apply steps 27 and 34 first")
    if src.count(HELPER_ANCHOR) != 1:
        raise SystemExit("w4a8_compact_micro.py: anchor '_ALIGN = 256' not found once - b12x changed")
    if src.count(SITE_OLD) != 1:
        raise SystemExit("w4a8_compact_micro.py: the step-27/34 launch site not found once - apply steps 27 and 34 first")
    src = src.replace(HELPER_ANCHOR, HELPER_ANCHOR + HELPERS.format(marker=MARKER), 1)
    return src.replace(SITE_OLD, SITE_NEW, 1)


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
