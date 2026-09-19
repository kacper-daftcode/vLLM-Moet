#!/usr/bin/env python3
"""Keep DeepSeek-V4's grouped o-projection (`wo_a`) in MXFP8 on sm_120 and run it on the
vLLM-Moet tensor-core GEMV (tools/dsv41_sm120/sm120_gemv/mxfp8_gemv_grouped).

Why: DeepGemmMxfp8BmmLinearKernel is gated to sm_100, so on RTX PRO 6000 vLLM picks
EmulationMxfp8LinearKernel for wo_a: BF16 weights (2x bytes) + cuBLAS bmm, 26 + 3 us per
layer at decode = 1.0 ms of a 15.7 ms step (profile 2026-09-18). With this patch the
weight stays MXFP8 and decode batches (<= 64 tokens) use the GEMV on the FP8 activations
fused_inv_rope_fp8_quant already produces for the sm_100 path; larger batches (prefill)
dequantize the weight on the fly and keep the bf16 bmm.

Three edits, all idempotent and anchor-based:
  1. installs sm120_gemv/vllm_sm120_gemv_bmm.py as
     vllm/model_executor/kernels/linear/mxfp8/sm120_gemv_bmm.py
  2. vllm/model_executor/kernels/linear/__init__.py: init_mxfp8_linear_kernel() tries
     Sm120GemvMxfp8BmmLinearKernel before DeepGEMM / emulation for BMM layers
  3. vllm/models/deepseek_v4/nvidia/ops/o_proj.py: deep_gemm_fp8_o_proj() dispatches to
     the kernel when the layer carries it
Runtime switch: VLLM_MOET_SM120_GEMV_BMM=0 restores the emulation path.

Usage: python3 patch_vllm_wo_a_sm120.py [--vllm-dir DIR] [--check]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
HERE = Path(__file__).resolve().parent
MODULE_SRC = HERE / "sm120_gemv" / "vllm_sm120_gemv_bmm.py"
MARKER = "[vllm-moet] sm_120 grouped GEMV"

INIT_OLD = (
    "    if bmm_batch_size is not None:\n"
    "        possible = (\n"
    "            [DeepGemmMxfp8BmmLinearKernel, EmulationMxfp8LinearKernel]\n"
    "            if current_platform.is_cuda()\n"
    "            else []\n"
    "        )\n"
)
INIT_NEW = (
    "    if bmm_batch_size is not None:\n"
    f"        # {MARKER}: MXFP8 wo_a on the vLLM-Moet tensor-core GEMV (sm_120 only).\n"
    "        from .mxfp8.sm120_gemv_bmm import Sm120GemvMxfp8BmmLinearKernel\n"
    "\n"
    "        possible = (\n"
    "            [\n"
    "                Sm120GemvMxfp8BmmLinearKernel,\n"
    "                DeepGemmMxfp8BmmLinearKernel,\n"
    "                EmulationMxfp8LinearKernel,\n"
    "            ]\n"
    "            if current_platform.is_cuda()\n"
    "            else []\n"
    "        )\n"
)

OPROJ_OLD = "    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn\n"
OPROJ_NEW = (
    f"    # {MARKER}: the layer carries the kernel when sm_120 kept wo_a in MXFP8.\n"
    '    sm120_gemv_bmm = getattr(wo_a, "sm120_gemv_bmm", None)\n'
    "    if sm120_gemv_bmm is not None:\n"
    "        return sm120_gemv_bmm.o_proj(\n"
    "            o,\n"
    "            positions,\n"
    "            cos_sin_cache,\n"
    "            wo_a,\n"
    "            wo_b,\n"
    "            n_groups=n_groups,\n"
    "            heads_per_group=heads_per_group,\n"
    "            nope_dim=nope_dim,\n"
    "            rope_dim=rope_dim,\n"
    "            o_lora_rank=o_lora_rank,\n"
    "        )\n"
    "    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn\n"
)


def patch_file(path: Path, old: str, new: str, check: bool) -> bool:
    src = path.read_text()
    if MARKER in src:
        print(f"{path}: already patched")
        return False
    if src.count(old) != 1:
        raise SystemExit(f"{path}: anchor found {src.count(old)} times (expected 1)")
    out = src.replace(old, new, 1)
    compile(out, str(path), "exec")
    if check:
        print(f"{path}: patch applies cleanly (not written)")
        return True
    path.write_text(out)
    print(f"{path}: patched")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-dir", type=Path, default=DEFAULT_VLLM)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    vllm = args.vllm_dir
    module_dst = vllm / "model_executor/kernels/linear/mxfp8/sm120_gemv_bmm.py"
    init_py = vllm / "model_executor/kernels/linear/__init__.py"
    oproj_py = vllm / "models/deepseek_v4/nvidia/ops/o_proj.py"
    for p in (init_py, oproj_py, MODULE_SRC):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    if args.check:
        print(f"{module_dst}: would install from {MODULE_SRC}")
    else:
        shutil.copyfile(MODULE_SRC, module_dst)
        compile(module_dst.read_text(), str(module_dst), "exec")
        print(f"{module_dst}: installed")
    patch_file(init_py, INIT_OLD, INIT_NEW, args.check)
    patch_file(oproj_py, OPROJ_OLD, OPROJ_NEW, args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
