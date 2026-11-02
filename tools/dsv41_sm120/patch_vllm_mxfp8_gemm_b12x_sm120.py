#!/usr/bin/env python3
"""The dense MXFP8 GEMMs of prefill-size steps go through FlashInfer's b12x backend on sm_120 (front "cubit + prefill").

vLLM's MXFP8 linear kernel on sm_120 (`model_executor/kernels/linear/mxfp8/flashinfer.py`, the CUTLASS class) calls
`flashinfer.mm_mxfp8(..., backend="cutlass")` for every row count above the GEMV range: the CUTLASS SM120 block-scaled
GEMM (tile 128 x 128 x 128). FlashInfer 0.7.2 also ships a b12x (CuTe-DSL, warp-level MMA) backend for the same inputs
(1D swizzled 128x4 scales, K % 128 == 0). Measured on one RTX PRO 6000 at M = 4096 (Phase 0, 2026-10-29,
/workspace/tmp/par31/L/phase0/out/dense_gemm.json): 6144 -> 6400: 790 -> 564 us, 1280 -> 4096: 108 -> 78,
5120 -> 5120: 527 -> 377, 2048 -> 5120: 226 -> 166 (-26...-29 %), and the outputs are bit-identical (`torch.equal`) -
the same QMMA.SF instruction, fp32 accumulation per 32-wide block, the same epilogue rounding. The dense MXFP8 GEMMs are
10.5 % of a 163K prefill's GPU time, so this is ~-3 % of the chunk for a backend switch.

With VLLM_MOET_MXFP8_GEMM_B12X=1 the CUTLASS call site picks `backend="b12x"` when the row count is at least
VLLM_MOET_MXFP8_GEMM_B12X_MIN_M (default 128: decode-size calls keep CUTLASS / our GEMV) and K % 128 == 0; anything
the b12x backend refuses falls back to CUTLASS once per process (logged), so a shape outside its envelope cannot break
serving. Bits: identical to CUTLASS on every shape of the bit test (tools/dsv41_sm120/mxfp8_gemm_b12x/). Default 0.

RESULT IN SITU: NEGATIVE - not in any image, kept as the documented measurement (session L, 2026-10-29, GPUs 4-7, real
model, served configuration). Bit test 119/119 shapes identical, but end to end the fresh 163K prefill took 8.67-8.73 s
with the switch on against 8.48-8.55 s off (+2.3 %); the 19K profile shows the b12x kernel at 103 ms over 875 launches
against CUTLASS's 93 ms (+11 % in situ, the opposite of the one-GPU harness) and the GPU idle share of the prefill
rising from 1.1 % to 10.4 %: the backend's Python path (AutoTuner.choose_one + plan + CuTe-DSL marshalling per call)
starves the GPU at ~215 GEMMs per 4096-token chunk. Lesson of the front: a harness win is not an in-situ win; measure
every candidate with the GPU-idle share of a real prefill (internal L/phase0/prefill_classes.py).

Usage: python3 patch_vllm_mxfp8_gemm_b12x_sm120.py [--dist DIR] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "[vllm-moet] prefill MXFP8 GEMM via b12x (sm_120)"

IMPORT_ANCHOR = "from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig\n"
HELPER = f'''

# {MARKER}: VLLM_MOET_MXFP8_GEMM_B12X=1 sends prefill-size MXFP8 GEMMs (rows >= VLLM_MOET_MXFP8_GEMM_B12X_MIN_M,
# K % 128 == 0) to FlashInfer's b12x backend (-26...-29 % vs CUTLASS at M = 4096, bit-identical outputs).
_MOET_B12X_GEMM: bool | None = None
_MOET_B12X_GEMM_MIN_M = 128
_MOET_B12X_GEMM_DISABLED_BY_ERROR = False


def _moet_mxfp8_b12x_backend(num_rows: int, k: int) -> str:
    global _MOET_B12X_GEMM, _MOET_B12X_GEMM_MIN_M
    if _MOET_B12X_GEMM is None:
        import os

        _MOET_B12X_GEMM = (
            os.environ.get("VLLM_MOET_MXFP8_GEMM_B12X", "0") == "1"
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        )
        _MOET_B12X_GEMM_MIN_M = int(os.environ.get("VLLM_MOET_MXFP8_GEMM_B12X_MIN_M", "128"))
    if (
        _MOET_B12X_GEMM
        and not _MOET_B12X_GEMM_DISABLED_BY_ERROR
        and num_rows >= _MOET_B12X_GEMM_MIN_M
        and k % 128 == 0
    ):
        return "b12x"
    return "cutlass"


def _moet_mm_mxfp8_prefill(input_mxfp8, weight_t, input_scale, weight_scale, out_dtype):
    """mm_mxfp8 with the b12x backend for prefill-size calls; CUTLASS otherwise, and CUTLASS for good after a refusal."""
    global _MOET_B12X_GEMM_DISABLED_BY_ERROR
    backend = _moet_mxfp8_b12x_backend(input_mxfp8.shape[0], input_mxfp8.shape[1])
    if backend == "b12x":
        try:
            return vllm_flashinfer.mm_mxfp8(
                input_mxfp8, weight_t, input_scale, weight_scale, out_dtype=out_dtype, backend="b12x"
            )
        except Exception as exc:  # noqa: BLE001 - outside the b12x envelope: CUTLASS from now on
            import logging

            _MOET_B12X_GEMM_DISABLED_BY_ERROR = True
            logging.getLogger(__name__).warning(
                "vllm-moet sm_120 MXFP8 GEMM: b12x backend refused %s x %s -> %s (%r); using CUTLASS from now on",
                tuple(input_mxfp8.shape), tuple(weight_t.shape), out_dtype, exc,
            )
    return vllm_flashinfer.mm_mxfp8(
        input_mxfp8, weight_t, input_scale, weight_scale, out_dtype=out_dtype, backend="cutlass"
    )
'''

CALL_OLD = (
    "        if gemv is not None:\n"
    "            output = gemv(input_2d, input_scale, weight, weight_scale)\n"
    "        else:\n"
    "            output = vllm_flashinfer.mm_mxfp8(\n"
    "                input_mxfp8,\n"
    "                weight.t(),\n"
    "                input_scale,\n"
    "                weight_scale,\n"
    "                out_dtype=out_dtype,\n"
    "                backend=\"cutlass\",\n"
    "            )\n"
)
CALL_NEW = (
    "        if gemv is not None:\n"
    "            output = gemv(input_2d, input_scale, weight, weight_scale)\n"
    "        else:\n"
    f"            output = _moet_mm_mxfp8_prefill(  # {MARKER}\n"
    "                input_mxfp8,\n"
    "                weight.t(),\n"
    "                input_scale,\n"
    "                weight_scale,\n"
    "                out_dtype,\n"
    "            )\n"
)


def _once(src: str, anchor: str, what: str, name: str) -> None:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"{name}: {what} anchor found {n} times (expected 1)")


def patch(src: str) -> str:
    if MARKER in src:
        return src
    for what, anchor in (("import", IMPORT_ANCHOR), ("cutlass call", CALL_OLD)):
        _once(src, anchor, what, "mxfp8/flashinfer.py")
    src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + HELPER, 1)
    return src.replace(CALL_OLD, CALL_NEW, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=DIST)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    path = args.dist / "model_executor" / "kernels" / "linear" / "mxfp8" / "flashinfer.py"
    src = path.read_text()
    if MARKER in src:
        print(f"{path}: already patched")
        return 0
    out = patch(src)
    compile(out, str(path), "exec")
    if args.check:
        print(f"{path}: patch applies cleanly (not written)")
        return 0
    path.write_text(out)
    print(f"{path}: patched ({MARKER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
