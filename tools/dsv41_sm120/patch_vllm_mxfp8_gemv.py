#!/usr/bin/env python3
"""Route decode-shaped MXFP8 dense GEMMs (M <= 16) on sm_120 to the vLLM-Moet GEMV
kernel (tools/dsv41_sm120/sm120_gemv/) instead of FlashInfer's CUTLASS SM120
blockscaled GEMM.

Why: a decode step of DeepSeek-V4.1-Flash (TP4, DSpark k=5 -> M = 6 rows) runs ~230
dense MXFP8 GEMMs; the CUTLASS kernel executes a 128-row MMA tile for those 6 rows
and averages 16 us per call (torch profile on RTX PRO 6000, 2026-09-18: 21 % of the
step). The GEMV streams the fp8 weight once, applies the F8_128x4 swizzled ue8m0
scales per 32-block and accumulates in fp32: 2.2-4.3x faster at M = 6 on every
served shape, bit-for-bit the same inputs (same activation quantization), output
error vs the fp32 reference identical to CUTLASS (bf16 rounding).

Patched file: vllm/model_executor/kernels/linear/mxfp8/flashinfer.py
  FlashInferCutlassMxfp8LinearKernel.apply_weights -> GEMV when M <= 16 and the
  output dtype is bf16; CUTLASS otherwise (prefill, M > 16, other dtypes).
Runtime switch: VLLM_MOET_SM120_GEMV=0 disables the GEMV path.

Idempotent, anchor-based. Usage:
    python3 patch_vllm_mxfp8_gemv.py [--file PATH] [--gemv-dir DIR] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py"
)
DEFAULT_GEMV_DIR = "/opt/vllm-moet/dsv41_sm120/sm120_gemv"
MARKER = "# [vllm-moet] sm_120 small-M MXFP8 GEMV"

HELPER_ANCHOR = "from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig\n"

HELPER_TEMPLATE = '''

{marker}
_SM120_GEMV = None  # None = not resolved yet, False = unavailable, else callable
_SM120_GEMV_MAX_M = 16


def _sm120_gemv_for(num_rows: int, out_dtype: torch.dtype):
    """Return the GEMV callable for this call or None (use CUTLASS)."""
    global _SM120_GEMV
    if num_rows > _SM120_GEMV_MAX_M or out_dtype != torch.bfloat16:
        return None
    if _SM120_GEMV is None:
        import os
        import sys

        _SM120_GEMV = False
        if os.environ.get("VLLM_MOET_SM120_GEMV", "1") == "1" and (
            current_platform.is_cuda() and current_platform.is_device_capability_family(120)
        ):
            gemv_dir = os.environ.get("VLLM_MOET_SM120_GEMV_DIR", "{gemv_dir}")
            if gemv_dir not in sys.path:
                sys.path.insert(0, gemv_dir)
            try:
                from mxfp8_gemv_sm120 import mxfp8_gemv as _gemv

                _gemv(*_sm120_gemv_probe())  # build/load the extension eagerly
                _SM120_GEMV = _gemv
            except Exception as exc:  # noqa: BLE001
                import logging

                logging.getLogger(__name__).warning(
                    "vllm-moet sm_120 MXFP8 GEMV unavailable, using CUTLASS: %r", exc
                )
    return _SM120_GEMV or None


def _sm120_gemv_probe():
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    a_q, a_sf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    w_q, w_sf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=True)
    return a_q, a_sf, w_q, w_sf
'''

APPLY_OLD = (
    "        if not weight.is_contiguous():\n"
    "            weight = weight.contiguous()\n"
    "\n"
    "        output = vllm_flashinfer.mm_mxfp8(\n"
    "            input_mxfp8,\n"
    "            weight.t(),\n"
    "            input_scale,\n"
    "            weight_scale,\n"
    "            out_dtype=out_dtype,\n"
    '            backend="cutlass",\n'
    "        )\n"
)
APPLY_NEW = (
    "        if not weight.is_contiguous():\n"
    "            weight = weight.contiguous()\n"
    "\n"
    "        # [vllm-moet] decode shapes on sm_120: one warp per output column instead\n"
    "        # of a 128-row CUTLASS tile (same MXFP8 inputs, fp32 accumulation).\n"
    "        input_2d = input_mxfp8.view(-1, K)\n"
    "        gemv = _sm120_gemv_for(input_2d.shape[0], out_dtype)\n"
    "        if gemv is not None:\n"
    "            output = gemv(input_2d, input_scale, weight, weight_scale)\n"
    "        else:\n"
    "            output = vllm_flashinfer.mm_mxfp8(\n"
    "                input_mxfp8,\n"
    "                weight.t(),\n"
    "                input_scale,\n"
    "                weight_scale,\n"
    "                out_dtype=out_dtype,\n"
    '                backend="cutlass",\n'
    "            )\n"
)


def patch_text(src: str, gemv_dir: str) -> tuple[str, bool]:
    if MARKER in src:
        return src, False
    if src.count(HELPER_ANCHOR) != 1:
        raise SystemExit("helper anchor not found exactly once")
    if src.count(APPLY_OLD) != 1:
        raise SystemExit(f"apply_weights anchor found {src.count(APPLY_OLD)} times (expected 1)")
    out = src.replace(HELPER_ANCHOR, HELPER_ANCHOR + HELPER_TEMPLATE.format(marker=MARKER, gemv_dir=gemv_dir), 1)
    out = out.replace(APPLY_OLD, APPLY_NEW, 1)
    return out, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--gemv-dir", default=DEFAULT_GEMV_DIR)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    if MARKER in src:
        print(f"{args.file}: already patched")
        return 0
    patched, _ = patch_text(src, args.gemv_dir)
    if args.check:
        print(f"{args.file}: patch applies cleanly (not written)")
        return 0
    compile(patched, str(args.file), "exec")
    args.file.write_text(patched)
    print(f"{args.file}: patched (GEMV dir {args.gemv_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
