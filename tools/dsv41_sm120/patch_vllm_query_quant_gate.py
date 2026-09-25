#!/usr/bin/env python3
"""Restore vLLM's fused query RMSNorm + MXFP8 quantization for DeepSeek-V4.1 (vllm#57679).

`DeepseekV4Attention._split_qkv_and_norm` has two paths: `fused_q_kv_rmsnorm_quant` (one Triton
kernel: q/kv RMSNorm + the MXFP8 quantization of Q with FlashInfer's F8_128x4 swizzled scales, handed
to `wq_b` and the indexer's `wq_b` as a QuantizedActivation) when `can_fuse_query_quant` allows it,
otherwise `fused_q_kv_rmsnorm` followed by each consumer's own `mxfp8_e4m3_quantize`. The gate reads
`linear.input_quant_key`; vllm#53793 (2026-09-14) renamed the attribute `expose_input_quant_key`
sets to `_input_quant_key` (reader: `get_input_quant_key`) and this call site was not updated, so the
gate has been `False` ever since: every layer runs the BF16 norm and a standalone quantize (in the
served decode graph: `Kernel` [6,2,1] 1.6 us + `MXFP8QuantizeSwizzledKernel` 1.6 us per layer).

This applies the open upstream fix, vllm#57679 (head ad792d17): read the capability through
`get_input_quant_key`, and pass `launch_pdl` to the fused kernel's launch (the kernel already
compiles in the PDL wait/trigger from its constexpr; without the launch attribute the wait is inert).
Bit-exact: the fused kernel reproduces the separate path's FP8 bytes, swizzled scales and normalized
KV (moe_quant_scatter/../test_query_quant_gate.py checks it on the served shapes).

Patched file: vllm/models/deepseek_v41/common/ops/query_quant.py. Idempotent, anchor-based.
Usage: python3 patch_vllm_query_quant_gate.py [--file PATH] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/common/ops/query_quant.py")
MARKER = "# [vllm-moet] vllm#57679"

IMPORT_OLD = "from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation\n"
IMPORT_NEW = (
    f"{MARKER}: read the consumer capability through get_input_quant_key\n"
    "from vllm.model_executor.layers.fusion.quant_activation import (\n"
    "    QuantizedActivation,\n"
    "    get_input_quant_key,\n"
    ")\n"
)
LAUNCH_OLD = (
    "        block = triton.next_power_of_2(max(q_size, kv_size))\n"
    "        _q_kv_norm_quant_kernel[(padded_tokens, 2)](\n"
)
LAUNCH_NEW = (
    "        block = triton.next_power_of_2(max(q_size, kv_size))\n"
    "        launch_pdl = current_platform.is_arch_support_pdl()\n"
    "        _q_kv_norm_quant_kernel[(padded_tokens, 2)](\n"
)
ARGS_OLD = (
    "            block,\n"
    "            current_platform.is_arch_support_pdl(),\n"
    "            num_warps=8 if block >= 2048 else 4,\n"
    "        )\n"
)
ARGS_NEW = (
    "            block,\n"
    "            launch_pdl,\n"
    "            num_warps=8 if block >= 2048 else 4,\n"
    "            launch_pdl=launch_pdl,\n"
    "        )\n"
)
GATE_OLD = '        getattr(linear, "input_quant_key", None) == kMxfp8Dynamic\n'
GATE_NEW = "        get_input_quant_key(linear) == kMxfp8Dynamic\n"


def patch_text(src: str) -> str:
    if MARKER in src:
        return src
    for name, old in (("import", IMPORT_OLD), ("launch", LAUNCH_OLD), ("args", ARGS_OLD), ("gate", GATE_OLD)):
        if src.count(old) != 1:
            raise SystemExit(f"{name} anchor found {src.count(old)} times (expected 1)")
    out = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
    out = out.replace(LAUNCH_OLD, LAUNCH_NEW, 1)
    out = out.replace(ARGS_OLD, ARGS_NEW, 1)
    out = out.replace(GATE_OLD, GATE_NEW, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=None)
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
    (args.out or args.file).write_text(patched)
    print(f"{args.out or args.file}: patched (vllm#57679)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
