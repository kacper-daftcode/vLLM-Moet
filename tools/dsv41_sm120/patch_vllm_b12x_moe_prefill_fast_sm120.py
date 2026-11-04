#!/usr/bin/env python3
"""The fast prefill MoE kernels (tools/dsv41_sm120/b12x_moe_prefill_fast) behind a runtime switch in b12x's dynamic path
(front "cubit + prefill", Phase 1a; DeepSeek-V4.1-Flash on sm_120, --moe-backend b12x).

Hook point: flashinfer/experimental/b12x/moe/fused_moe/_impl.py, the top of `_launch_dynamic(**kw)` - the one function
through which every prefill MoE call (T > 8 rows, `integration.tp_moe.dynamic`) launches b12x's front-end (routing /
publication / MXFP8 rows quant: MoEDynamicKernelSilu), W4A8MaterializedPhase1Kernel (FC1 + SwiGLU + MXFP8 quant) and
W4A8MaterializedPhase2Kernel (FC2 -> route rows in workspace.route_output). With VLLM_MOET_B12X_MOE_PREFILL_FAST=1
(default 0: off) and a supported call (w4a8_mx, silu with a SwiGLU limit, alpha 1 / beta 0, n64-repacked planes, no
inline CSF / NVFP4 scales, deterministic_output, int32 ids, intermediate 576, hidden a multiple of 128, <= 512 experts),
`_moet_prefill_fast_launch` runs our kernels instead - front-end (rows quant + deterministic routing + work lists), FC1,
FC2 - writing the same token-major route rows into workspace.route_output, and returns before b12x's launch. The top-k
sum: by default (v2 of this patch, 4.11; VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM=1) the kernels also write the output (the
sum is fused into FC2's epilogue - v8 kernels - or done by their own topk_sum kernel with older ones; the same chain as
b12x's W4A16TopKSumKernel: fp32 adds in slot order, one bf16 rounding) and the caller's `_launch_dynamic_topk_sum` returns
at once for that route_output (a module-level marker set by the hook); the shared-expert add that follows in vLLM's
runner is unchanged. With VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM=0 the route rows are the interface as in v1 of the patch:
b12x's TopKSum (0.16 ms, at the DRAM floor) reduces them. So the switch bypasses the dynamic front-end, Phase1, Phase2
and the TopKSum of the layer-chunk. Route rows and output are bit-identical with b12x's
(test_b12x_moe_prefill_fast_sm120.py: synthetic T set + the served dumps, vs b12x and vs spec B).
Extension: VLLM_MOET_B12X_MOE_PREFILL_FAST_DIR (default /opt/vllm-moet/dsv41_sm120/b12x_moe_prefill_fast; JIT on first
use, ~60 s, or prebuilt in the image). Anything unsupported or an import failure -> b12x's own launch, silently.

Usage: python3 patch_vllm_b12x_moe_prefill_fast_sm120.py [--root FLASHINFER_DIR] [--check]   (idempotent; a file carrying
v1 of the patch - the r23 image - is upgraded in place: the v1 hook is removed and v2 applied)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages/flashinfer")
REL = "experimental/b12x/moe/fused_moe/_impl.py"
MARKER = "[vllm-moet] sm_120 fast prefill MoE"
MARKER_V2 = "[vllm-moet] sm_120 fast prefill MoE v2 (fused top-k sum)"
SUM_ANCHOR = ") -> None:\n    if route_output.numel() < m * num_topk * k:\n"  # the body start of _launch_dynamic_topk_sum

DEF_ANCHOR = "def _launch_dynamic(\n    *,\n    workspace: TPDynamicWorkspace,\n"
BODY_ANCHOR = ") -> None:\n    del stream\n    if dynamic_route_mode not in {\"direct\", \"grouped\"}:\n"

HELPERS = '''# {marker} (tools/dsv41_sm120/patch_vllm_b12x_moe_prefill_fast_sm120.py): FC1 + SwiGLU + MXFP8 quant and FC2 of
# the deterministic dynamic path as pipelined CUDA kernels (tools/dsv41_sm120/b12x_moe_prefill_fast), bit for bit with
# the b12x kernels; the route rows land in workspace.route_output and b12x's TopKSum runs on them unchanged.
# VLLM_MOET_B12X_MOE_PREFILL_FAST=1 enables it; decided once per process. {marker_v2}: with
# VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM=1 (default) the kernels also write the top-k sum into the output and
# _launch_dynamic_topk_sum skips that route_output (_MOET_PREFILL_FAST_SUM_DONE).
_MOET_PREFILL_FAST = None
_MOET_PREFILL_FAST_SUM = None
_MOET_PREFILL_FAST_SUM_DONE = None


def _moet_prefill_fast_sum_enabled():
    global _MOET_PREFILL_FAST_SUM
    if _MOET_PREFILL_FAST_SUM is None:
        import os

        _MOET_PREFILL_FAST_SUM = os.environ.get("VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM", "1") == "1"
    return _MOET_PREFILL_FAST_SUM


def _moet_prefill_fast_module():
    global _MOET_PREFILL_FAST
    if _MOET_PREFILL_FAST is None:
        import os
        import sys

        _MOET_PREFILL_FAST = False
        if (
            os.environ.get("VLLM_MOET_B12X_MOE_PREFILL_FAST", "0") == "1"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] == 12
        ):
            path = os.environ.get(
                "VLLM_MOET_B12X_MOE_PREFILL_FAST_DIR",
                "/opt/vllm-moet/dsv41_sm120/b12x_moe_prefill_fast",
            )
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                import b12x_moe_prefill_fast_sm120 as module

                _MOET_PREFILL_FAST = module
                print(
                    "vllm-moet sm_120 fast prefill MoE: active for b12x's dynamic path (FC1 + SwiGLU + FC2 from "
                    + path + "; VLLM_MOET_B12X_MOE_PREFILL_FAST=0 turns it off)",
                    file=sys.stderr,
                    flush=True,
                )
            except ImportError:
                _MOET_PREFILL_FAST = False
    return _MOET_PREFILL_FAST


def _moet_prefill_fast_launch(
    *,
    workspace,
    weights,
    a,
    flat_ids,
    flat_weights,
    scatter_output,
    E,
    m,
    k,
    n,
    num_topk,
    routed_rows,
    topk_ids_dtype,
    activation,
    quant_mode,
    w4a8_prepared,
    w4a8_n64_repacked,
    nvfp4_inline_scales,
    deterministic_output,
    swiglu_limit,
    swiglu_alpha,
    swiglu_beta,
    w4a8_csf_inline,
) -> bool:
    """True when the fast kernels produced the route rows of this call (b12x's launch is then skipped)."""
    module = _moet_prefill_fast_module()
    if not module:
        return False
    if not (
        deterministic_output
        and quant_mode == "w4a8_mx"
        and activation == "silu"
        and w4a8_prepared is not None
        and w4a8_n64_repacked
        and nvfp4_inline_scales is None
        and w4a8_csf_inline is None
        and swiglu_limit is not None
        and float(swiglu_alpha) == 1.0
        and float(swiglu_beta) == 0.0
        and topk_ids_dtype == torch.int32
        and flat_ids.dtype == torch.int32
        and a.dtype == torch.bfloat16
        and a.dim() == 2
        and tuple(a.shape) == (m, k)
        and routed_rows == m * num_topk
        and flat_ids.numel() == routed_rows
        and flat_weights.numel() == routed_rows
        and not w4a8_prepared["w13_sfb"].dtype.is_floating_point
        and workspace.route_output.numel() >= routed_rows * k
        and module.supported(n=n, k=k, num_topk=num_topk, experts=E, ids_dtype=flat_ids.dtype)
    ):
        return False
    planes = dict(
        w13=w4a8_prepared["w13_rp"],
        w13_sf=w4a8_prepared["w13_sfb"],
        w2=w4a8_prepared["w2_rp"],
        w2_sf=w4a8_prepared["w2_sfb"],
        alpha=weights.w1_alpha,
        input_global_scale=workspace.input_gs,
        down_alpha=weights.w2_alpha,
        global_scale=workspace.down_input_scale,
    )
    global _MOET_PREFILL_FAST_SUM_DONE
    fused = (
        _moet_prefill_fast_sum_enabled()
        and scatter_output is not None
        and scatter_output.dtype == torch.bfloat16
        and scatter_output.is_contiguous()
        and tuple(scatter_output.shape) == (m, k)
    )
    module.moe_prefill_fast(
        a,
        flat_ids.view(m, num_topk),
        flat_weights.view(m, num_topk),
        planes,
        float(swiglu_limit),
        n,
        route_out=workspace.route_output,
        out=scatter_output if fused else None,
        topk_sum=fused,
    )
    _MOET_PREFILL_FAST_SUM_DONE = workspace.route_output.data_ptr() if fused else None
    return True


'''

BODY = (
    BODY_ANCHOR[: len(") -> None:\n")]
    + f"    if _moet_prefill_fast_launch(  # {MARKER}\n"
    "        workspace=workspace,\n"
    "        weights=weights,\n"
    "        a=a,\n"
    "        flat_ids=flat_ids,\n"
    "        flat_weights=flat_weights,\n"
    "        scatter_output=scatter_output,\n"
    "        E=E,\n"
    "        m=m,\n"
    "        k=k,\n"
    "        n=n,\n"
    "        num_topk=num_topk,\n"
    "        routed_rows=routed_rows,\n"
    "        topk_ids_dtype=topk_ids_dtype,\n"
    "        activation=activation,\n"
    "        quant_mode=quant_mode,\n"
    "        w4a8_prepared=w4a8_prepared,\n"
    "        w4a8_n64_repacked=w4a8_n64_repacked,\n"
    "        nvfp4_inline_scales=nvfp4_inline_scales,\n"
    "        deterministic_output=deterministic_output,\n"
    "        swiglu_limit=swiglu_limit,\n"
    "        swiglu_alpha=swiglu_alpha,\n"
    "        swiglu_beta=swiglu_beta,\n"
    "        w4a8_csf_inline=w4a8_csf_inline,\n"
    "    ):\n"
    "        return\n"
    + BODY_ANCHOR[len(") -> None:\n"):]
)


SUM_BODY = (
    SUM_ANCHOR[: len(") -> None:\n")]
    + f"    global _MOET_PREFILL_FAST_SUM_DONE  # {MARKER_V2}: the fast kernels wrote the output of this call\n"
    "    if _MOET_PREFILL_FAST_SUM_DONE is not None and _MOET_PREFILL_FAST_SUM_DONE == route_output.data_ptr():\n"
    "        _MOET_PREFILL_FAST_SUM_DONE = None\n"
    "        return\n"
    + SUM_ANCHOR[len(") -> None:\n"):]
)


def unpatch_v1(src: str) -> str:
    """Remove v1 of the hook (the r23 image's _impl.py) so that v2 applies to the pristine anchors."""
    start = src.index(f"# {MARKER} (tools/dsv41_sm120/patch_vllm_b12x_moe_prefill_fast_sm120.py)")
    end = src.index(DEF_ANCHOR)
    if not (0 <= start < end):
        raise SystemExit("_impl.py: v1 hook helpers not where expected")
    src = src[:start] + src[end:]
    b0 = src.index(f"    if _moet_prefill_fast_launch(  # {MARKER}\n")
    b1 = src.index("    ):\n        return\n", b0) + len("    ):\n        return\n")
    return src[:b0] + src[b1:]


def patch(src: str) -> str:
    if MARKER_V2 in src:
        return src
    if MARKER in src:
        src = unpatch_v1(src)
    for what, anchor in (("definition", DEF_ANCHOR), ("body", BODY_ANCHOR), ("top-k sum body", SUM_ANCHOR)):
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"_impl.py: the {what} anchor found {n} times (expected 1) - b12x changed")
    src = src.replace(DEF_ANCHOR, HELPERS.format(marker=MARKER, marker_v2=MARKER_V2) + DEF_ANCHOR, 1)
    src = src.replace(BODY_ANCHOR, BODY, 1)
    return src.replace(SUM_ANCHOR, SUM_BODY, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="flashinfer package dir")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = a.root / REL
    src = path.read_text()
    if MARKER_V2 in src:
        print(f"{path}: already patched (v2)")
        return 0
    out = patch(src)
    compile(out, str(path), "exec")
    if a.check:
        print(f"{path}: patch applies cleanly (not written)")
        return 0
    path.write_text(out)
    print(f"{path}: patched ({MARKER_V2})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
