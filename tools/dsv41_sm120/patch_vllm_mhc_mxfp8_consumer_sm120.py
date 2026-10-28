#!/usr/bin/env python3
"""Let the sm_120 MXFP8 linear kernel take the mHC boundary kernel's MXFP8 copy of its input (step 38).

The decode graph quantizes the sublayer's layer input to MXFP8 twice per layer, right after the boundary kernel
(mhc_post_norm_sm120, step 10) and right before the first GEMV of the sublayer (fused_wqa_wkv; the shared experts'
or dense MLP's w13): FlashInfer's MXFP8QuantizeSwizzledKernel, 3.0-3.5 us per launch in the served graph with PDL,
0.7-1.2 us of it on the critical path per site (the GEMV waits for it, and the GEMV with the quantization in it was
measured slower - tools/dsv41_sm120/sm120_gemv/, variant AQ). With VLLM_MOET_MHC_MXFP8=1 the boundary kernel writes
the e4m3 bytes and the F8_128x4 swizzled scales itself (the lanes that hold the rounded bf16 values; the same bytes
as mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)) and hangs them on the bf16 tensor as `_moet_mxfp8`; the bf16
tensor is unchanged (the router, the indexer's weights_proj, wk and the routed experts read it).

Patched file: vllm/model_executor/kernels/linear/mxfp8/flashinfer.py, FlashInferCutlassMxfp8LinearKernel.apply_weights
(the sm_120 kernel class; after patch_vllm_mxfp8_gemv.py, whose marker anchors this hunk): a plain tensor that carries
`_moet_mxfp8` of its own shape is consumed like a QuantizedActivation; anything else quantizes as before. The GEMV and
the CUTLASS fallback both take the pair (the same data / scale arguments).

Idempotent, anchor-based. Usage:
    python3 patch_vllm_mhc_mxfp8_consumer_sm120.py [--file PATH] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py"
)
MARKER = "# [vllm-moet] sm_120 mHC boundary MXFP8 (step 38)"
GEMV_MARKER = "# [vllm-moet] decode shapes on sm_120: one warp per output column instead\n"

OLD = (
    "        else:\n"
    "            assert isinstance(x, torch.Tensor)\n"
    "            out_dtype, input_shape = x.dtype, x.shape\n"
    "            input_mxfp8, input_scale = mxfp8_e4m3_quantize(\n"
    "                x.view(-1, K), is_sf_swizzled_layout=True\n"
    "            )\n"
    "\n"
    "        if not weight.is_contiguous():\n"
    "            weight = weight.contiguous()\n"
    "\n"
    "        " + GEMV_MARKER
)
NEW = (
    "        else:\n"
    "            assert isinstance(x, torch.Tensor)\n"
    "            out_dtype, input_shape = x.dtype, x.shape\n"
    f"            {MARKER}: the boundary kernel already wrote this\n"
    "            # activation's e4m3 bytes and swizzled scales (mhc_post_norm_sm120.mxfp8_buffers)\n"
    '            _moet_pre = getattr(x, "_moet_mxfp8", None)\n'
    "            if _moet_pre is not None and _moet_pre[0].shape == x.shape:\n"
    "                input_mxfp8, input_scale = _moet_pre\n"
    "            else:\n"
    "                input_mxfp8, input_scale = mxfp8_e4m3_quantize(\n"
    "                    x.view(-1, K), is_sf_swizzled_layout=True\n"
    "                )\n"
    "\n"
    "        if not weight.is_contiguous():\n"
    "            weight = weight.contiguous()\n"
    "\n"
    "        " + GEMV_MARKER
)


def patch_text(src: str) -> tuple[str, bool]:
    if MARKER in src:
        return src, False
    if GEMV_MARKER not in src:
        raise SystemExit("patch_vllm_mxfp8_gemv.py must be applied first (its marker anchors this hunk)")
    if src.count(OLD) != 1:
        raise SystemExit(f"apply_weights anchor found {src.count(OLD)} times (expected 1)")
    return src.replace(OLD, NEW, 1), True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    if MARKER in src:
        print(f"{args.file}: already patched")
        return 0
    patched, _ = patch_text(src)
    if args.check:
        print(f"{args.file}: patch applies cleanly (not written)")
        return 0
    compile(patched, str(args.file), "exec")
    args.file.write_text(patched)
    print(f"{args.file}: patched (mHC boundary MXFP8 consumer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
