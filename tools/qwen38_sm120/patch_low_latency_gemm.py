#!/usr/bin/env python3
"""Enable vLLM's CuTe-DSL skinny GEMM for Qwen3.8-Flash-Next decode on sm_120 (RTX PRO 6000).

vllm/models/qwen3_8_flash_next/nvidia/low_latency_gemm.py swaps the unquantized linear method of
every BF16 linear whose per-rank (N, K) shape has a measured plan for a low-latency CuTe-DSL
"skinny GEMM" (one block per few output columns, block-wide K split, warp reductions). Upstream
gates it to sm_103 (B300). The kernels use nothing sm_100-specific and compile fine on sm_120;
measured on RTX PRO 6000 at M = 4 tokens (MTP k=3) against cuBLAS's `cutlass_80_wmma` path:
24x2560 10.6 -> 1.7 us, 2560x1536 11.4 -> 6.6, 320x2560 4.7 -> 2.4, 4096x2560 19.6 -> 15.2 us
(1.38 TB/s), so this patch (1) opens the gate for sm_120, (2) adds the sm_120 plan table below
(tuned with tools/sm120_perf/skinny_tune.py: the token counts upstream leaves out - M = 4 for the
GDN QKVZ projection and the LM head, M = 8/16 broadly - and the per-rank shapes upstream has no
plan for at all: HC up-projection 10240x320, shared-expert down 2560x160, router 512x2560, PLE
projections 2560x2560), and (3) logs the inventory of BF16 linear shapes it saw and which got a plan.

The compiled torch graph changes (custom op instead of aten.mm) but vLLM's compile-cache hash
does not include this file, so run with a separate torch_compile_cache directory (the launcher
does this) - otherwise a cached AOT graph bypasses the patch silently.

Usage: python3 patch_low_latency_gemm.py [--file PATH] [--check]  (idempotent, anchor-based)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/low_latency_gemm.py"
)
MARKER = "[vllm-moet] sm_120 skinny GEMM"

GATE_OLD = (
    "def _is_sm103() -> bool:\n"
    "    return current_platform.is_device_capability((10, 3))\n"
)
GATE_NEW = (
    "def _is_sm103() -> bool:\n"
    "    return current_platform.is_device_capability((10, 3))\n"
    "\n"
    "\n"
    f"# {MARKER}: RTX PRO 6000 / RTX 5090 run the same CuTe-DSL kernels; measured plans below.\n"
    "def _is_sm120() -> bool:\n"
    "    import os\n"
    "\n"
    '    if os.environ.get("VLLM_MOET_SM120_LL_GEMM", "1") != "1":\n'
    "        return False\n"
    "    return current_platform.is_device_capability_family(120)\n"
    "\n"
    "\n"
    "QWEN38NEXT_GEMM_PLANS_SM120: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {\n"
    "__SM120_PLAN__"
    "}\n"
    "\n"
    "\n"
    "def _plans() -> dict[tuple[int, int], dict[int, SkinnyGemmConfig]]:\n"
    "    if not _is_sm120():\n"
    "        return QWEN38NEXT_GEMM_PLANS\n"
    "    merged: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {\n"
    "        shape: dict(plan) for shape, plan in QWEN38NEXT_GEMM_PLANS.items()\n"
    "    }\n"
    "    for shape, plan in QWEN38NEXT_GEMM_PLANS_SM120.items():\n"
    "        merged.setdefault(shape, {}).update(plan)\n"
    "    return merged\n"
)

ENABLE_OLD = "    if dtype != torch.bfloat16 or not _is_sm103():\n        return\n"
ENABLE_NEW = (
    "    if dtype != torch.bfloat16 or not (_is_sm103() or _is_sm120()):\n"
    "        return\n"
)

LOOKUP_ENABLE_OLD = "        plan = QWEN38NEXT_GEMM_PLANS.get((weight.shape[0], weight.shape[1]))\n        if plan is None:\n            continue\n"
LOOKUP_ENABLE_NEW = (
    "        plan = _plans().get((weight.shape[0], weight.shape[1]))\n"
    "        _inventory[(weight.shape[0], weight.shape[1], plan is not None)] = (\n"
    "            _inventory.get((weight.shape[0], weight.shape[1], plan is not None), 0) + 1\n"
    "        )\n"
    "        if plan is None:\n"
    "            continue\n"
)
INVENTORY_DECL_OLD = "    warmup_configs: set[SkinnyGemmConfig] = set()\n    for child in module.modules():\n"
INVENTORY_DECL_NEW = (
    "    warmup_configs: set[SkinnyGemmConfig] = set()\n"
    "    _inventory: dict[tuple[int, int, bool], int] = {}\n"
    "    for child in module.modules():\n"
)
WARMUP_OLD = "    if warmup_configs:\n        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)\n"
WARMUP_NEW = (
    "    from vllm.logger import init_logger\n"
    "\n"
    "    init_logger(__name__).info(\n"
    '        "low-latency skinny GEMM (%s): BF16 linear shapes per rank -> %s",\n'
    '        "sm_120" if _is_sm120() else "sm_103",\n'
    '        ", ".join(\n'
    '            f"{n}x{k}:{c}{\'\' if has else \' (no plan)\'}"\n'
    "            for (n, k, has), c in sorted(_inventory.items())\n"
    "        ),\n"
    "    )\n"
    "    if warmup_configs:\n"
    "        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)\n"
)
DISPATCH_OLD = "    plan = QWEN38NEXT_GEMM_PLANS.get((weight.shape[0], weight.shape[1]))\n    config = None if plan is None else plan.get(x.shape[0])\n"
DISPATCH_NEW = "    plan = _plans().get((weight.shape[0], weight.shape[1]))\n    config = None if plan is None else plan.get(x.shape[0])\n"

# Tuned on RTX PRO 6000 Blackwell Server Edition (sm_120), cold L2, tools/sm120_perf/skinny_tune.py
# (2026-09-18). Entries override/extend QWEN38NEXT_GEMM_PLANS; token counts where cuBLAS was
# faster (mostly M = 16 on the small shapes, all M on 10240x320) are left out and fall back to
# torch.nn.functional.linear as upstream does. Comments: cuBLAS -> skinny, us per call.
SM120_PLAN = """\
    # GDN fused QKVZ projection (TP=4): 19.6 -> 15.4 (M=4), 15.9 (8), 17.3 (16)
    (4096, 2560): {
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
        8: SkinnyGemmConfig(8, 64, 2, k_unroll=2),
        16: SkinnyGemmConfig(16, 32, 2, k_unroll=2, vector_width=4),
    },
    # GDN / QSA output projection (TP=4): 11.3 -> 6.9 (M=8), 8.7 (16)
    (2560, 1536): {
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=2),
        16: SkinnyGemmConfig(16, 32, 1, k_unroll=1),
    },
    # QSA fused QKV/gate projection (TP=4): 18.7 -> 14.1 (M=8), 15.6 (16)
    (3584, 2560): {
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=2),
        16: SkinnyGemmConfig(16, 32, 1, k_unroll=1),
    },
    # Shared-expert fused gate/up (TP=4): 3.1/11.1/4.7 -> 1.9/2.1/2.6 (M=1/2/4)
    (320, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 4, k_unroll=4, vector_width=4),
    },
    # HC merged down+inject (replicated): 9.2 -> 4.7/5.0/6.2 (M=1/2/4), 9.4 -> 7.6 (8)
    (336, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 256, 1, static_k=10240),
        4: SkinnyGemmConfig(4, 256, 1, static_k=10240),
        8: SkinnyGemmConfig(8, 256, 2, k_unroll=4),
    },
    # MoE router (replicated): 3.5/4.5/4.6/4.7 -> 2.2/2.4/3.0/4.0
    (512, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=4),
        8: SkinnyGemmConfig(8, 128, 4, k_unroll=4, vector_width=4),
    },
    # PLE key/value projections, MTP fc_* (per rank 2560x2560): 12.3-14.4 -> 10.0-12.6
    (2560, 2560): {
        1: SkinnyGemmConfig(1, 64, 2, k_unroll=4),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=4),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=2),
        16: SkinnyGemmConfig(16, 32, 2, k_unroll=4),
    },
    # shared-expert gate (N=1): 2.6-2.8 -> 1.3-1.9
    (1, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
    },
    # LM head (TP=4): 220 -> 210 (M=4), 221 -> 209.5 (8)
    (62080, 2560): {
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=2),
        8: SkinnyGemmConfig(8, 32, 1, k_unroll=4, vector_width=4),
    },
"""


def apply(src: str) -> str:
    if MARKER in src:
        return src
    for old, new in (
        (GATE_OLD, GATE_NEW.replace("__SM120_PLAN__", SM120_PLAN)),
        (ENABLE_OLD, ENABLE_NEW),
        (INVENTORY_DECL_OLD, INVENTORY_DECL_NEW),
        (LOOKUP_ENABLE_OLD, LOOKUP_ENABLE_NEW),
        (WARMUP_OLD, WARMUP_NEW),
        (DISPATCH_OLD, DISPATCH_NEW),
    ):
        if src.count(old) != 1:
            raise SystemExit(f"anchor found {src.count(old)} times (expected 1):\n{old}")
        src = src.replace(old, new, 1)
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write the patched file here instead of in place")
    args = ap.parse_args()
    src = args.file.read_text()
    if MARKER in src:
        print(f"{args.file}: already patched")
        return 0
    out = apply(src)
    compile(out, str(args.file), "exec")
    if args.check:
        print(f"{args.file}: patch applies cleanly (not written)")
        return 0
    dst = args.out or args.file
    dst.write_text(out)
    print(f"{dst}: patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
