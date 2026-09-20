#!/usr/bin/env python3
"""Let vLLM's DeepSeek sparse indexer use the MXFP4 K cache on sm_120.

`vllm/v1/attention/backends/mla/indexer.py::dsa_indexer_uses_fp4` rejects
`--attention-config '{"indexer_kv_dtype":"mxfp4"}'` unless the GPU is sm_10x.
The path itself is arch-generic apart from the logits kernel:

  * Q: fused_indexer_q_rope_quant(use_fp4=True) (CuTe DSL / Triton), K:
    indexer_k_norm_rope_store(use_fp4_cache=True) (Triton, e2m1 via
    cvt.rn.satfinite.e2m1x2.f32) -- both compile and run on sm_120a and match
    DeepSeek's fp4 quantizer bit-for-bit (test_indexer_fp4_sm120.py);
  * prefill gather: ops.cp_gather_indexer_k_quant_cache is byte-width generic
    (64 B values + 4 B scales per key);
  * logits: DeepGEMM 8b1392b9 ships sm120_fp4_paged_mqa_logits.cuh and
    sm120_fp4_mqa_logits.cuh (head_dim 128); its host asserts admit the 128-key
    indexer pages after patch_deepgemm.py (the same change the FP8 cache needed).

What it buys on DeepSeek-V4.1-Flash: 68 instead of 132 B per indexer key, i.e.
the packed KV block shrinks from 230400 to 210240 B (-8.7 %), ~+9.6 % KV tokens
at the same memory, and the indexer scores in the format it was trained with
(the report quantizes indexer Q/K to FP4 with UE8M0 block scales).

Idempotent, anchor-based. Usage:
    python3 patch_vllm_indexer_fp4_sm120.py [--file PATH] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"
)
MARKER = "# [vllm-moet] sm_120 runs the MXFP4 indexer cache on DeepGEMM's sm120_fp4 kernels"

OLD = (
    "    use_fp4 = kv_dtype == \"mxfp4\"\n"
    "    if use_fp4 and not current_platform.is_device_capability_family(100):\n"
    "        raise ValueError(\n"
    "            \"indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs \"\n"
    "            \"(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and \"\n"
    "            \"earlier architectures are not supported.\"\n"
    "        )\n"
    "    return use_fp4\n"
)
NEW = (
    "    use_fp4 = kv_dtype == \"mxfp4\"\n"
    f"    {MARKER}\n"
    "    # (patch_deepgemm.py admits the 128-key pages; Q/K quantizers and the gather\n"
    "    # op are arch-generic). Validated by test_indexer_fp4_sm120.py.\n"
    "    if use_fp4 and not (\n"
    "        current_platform.is_device_capability_family(100)\n"
    "        or current_platform.is_device_capability_family(120)\n"
    "    ):\n"
    "        raise ValueError(\n"
    "            \"indexer_kv_dtype='mxfp4' requires Blackwell GPUs (sm_10x such as \"\n"
    "            \"B200/GB200, or sm_120 with vLLM-Moet's DeepGEMM build); earlier \"\n"
    "            \"architectures are not supported.\"\n"
    "        )\n"
    "    return use_fp4\n"
)


def patch_text(src: str) -> tuple[str, bool]:
    if MARKER in src:
        return src, False
    n = src.count(OLD)
    if n != 1:
        raise SystemExit(f"anchor found {n} times (expected 1):\n{OLD}")
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
    print(f"{args.file}: patched (dsa_indexer_uses_fp4 admits sm_120)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
