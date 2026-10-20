#!/usr/bin/env python3
"""The MoE router's top-6 on sm_120 (DeepSeek-V4.1-Flash): a CUDA kernel bit-identical to vLLM's Triton dsv4_topk,
launched without programmatic dependent launch.

In the served decode graph the router GEMM (step 11's BF16 kernel, PDL) is followed by the Triton top-k (sqrt-softplus
scores + correction bias, noaux_tc top-6), then b12x's MXFP8 quantization of the MoE input and the MoE itself - on the
critical path of every MoE layer. Alone the Triton kernel takes ~2 us; behind the router's early PDL trigger, with the
shared expert running on its side stream, it ends ~8 us after the router (2026-10-20 profiles, rank 0, median over 40
layers x 37 replays). dsv4_topk/dsv4_topk_sm120.cu computes the same bits (3000 randomized cases and 24.3M router rows
of the served model checked against Triton) in one warp per token and is launched plainly: it ends 4.45 us after the
router and the MoE input quantization 1.47 us after it (3.33 before).

Patched file: vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py - the dsv4_topk call of
fused_topk_bias (fp32 logits, 256 or 384 experts, top-6) goes to dsv4_topk_sm120.dsv4_topk_cuda on compute capability
12.x; everything else stays on vLLM's path. Runtime switches: VLLM_MOET_DSV4_TOPK=0 (Triton again),
VLLM_MOET_DSV4_TOPK_PDL=1 (PDL launch), VLLM_MOET_DSV4_TOPK_CHECK=1 (eager calls also run the Triton kernel and log
mismatching rows - a test mode).

Usage: python3 patch_vllm_dsv4_topk_sm120.py [--file PATH] [--kernel-dir DIR] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py"
)
DEFAULT_KERNEL_DIR = "/opt/vllm-moet/dsv41_sm120/dsv4_topk"
MARKER = "[vllm-moet] sm_120 routing top-6"

IMPORT_ANCHOR = (
    "from vllm.model_executor.layers.fused_moe.router.dsv4_topk import (\n"
    "    can_use_dsv4_topk,\n"
    "    dsv4_topk,\n"
    ")\n"
)
HELPER = '''

# {marker}: dsv4_topk_sm120 (tools/dsv41_sm120/dsv4_topk), bit-identical to the Triton dsv4_topk.
_MOET_TOPK = None
_MOET_TOPK_CHECK = False
_MOET_TOPK_STATS = [0, 0, 0]  # checked calls, rows, mismatching rows


def _moet_topk(gating_output):
    global _MOET_TOPK, _MOET_TOPK_CHECK
    if _MOET_TOPK is None:
        import os

        _MOET_TOPK = False
        if os.environ.get("VLLM_MOET_DSV4_TOPK", "1") == "1" and torch.cuda.get_device_capability()[0] == 12:
            d = os.environ.get("VLLM_MOET_DSV4_TOPK_DIR", "{kernel_dir}")
            if d not in sys.path:
                sys.path.insert(0, d)
            import dsv4_topk_sm120

            dsv4_topk_sm120._ext()
            _MOET_TOPK = dsv4_topk_sm120
            _MOET_TOPK_CHECK = os.environ.get("VLLM_MOET_DSV4_TOPK_CHECK", "0") == "1"
    if not _MOET_TOPK or not gating_output.is_contiguous():
        return None
    return _MOET_TOPK.dsv4_topk_cuda


def _moet_topk_check(out, ref):
    """Eager calls only: this kernel's rows against the Triton kernel's on the same inputs, bit for bit."""
    (w, i), (wr, ir) = out, ref
    bad = ((w.view(torch.int32) != wr.view(torch.int32)) | (i != ir)).any(dim=1)
    s = _MOET_TOPK_STATS
    s[0] += 1
    s[1] += w.shape[0]
    s[2] += int(bad.sum())
    if s[0] % 200 == 0 or bad.any():
        from vllm.logger import init_logger

        init_logger("vllm.vllm_moet.dsv4_topk").info(
            "vllm-moet dsv4_topk check: %d calls, %d rows, %d mismatching rows", s[0], s[1], s[2])
'''

CALL_ANCHOR = (
    "            assert e_score_correction_bias is not None\n"
    "            return dsv4_topk(\n"
    "                gating_output,\n"
    "                e_score_correction_bias,\n"
    "                output_indices_dtype,\n"
    "                routed_scaling_factor,\n"
    "                is_padding=padding_mask,\n"
    "                input_ids=input_tokens,\n"
    "                bias_vl=bias_vl,\n"
    "                image_sentinel_lo=image_sentinel_lo,\n"
    "            )\n"
)
CALL_NEW = (
    "            assert e_score_correction_bias is not None\n"
    "            _moet_fn = _moet_topk(gating_output)  # {marker}\n"
    "            if _moet_fn is not None:\n"
    "                out = _moet_fn(\n"
    "                    gating_output, e_score_correction_bias, output_indices_dtype, routed_scaling_factor,\n"
    "                    is_padding=padding_mask, input_ids=input_tokens, bias_vl=bias_vl,\n"
    "                    image_sentinel_lo=image_sentinel_lo,\n"
    "                )\n"
    "                if _MOET_TOPK_CHECK and not torch.cuda.is_current_stream_capturing():\n"
    "                    _moet_topk_check(out, dsv4_topk(\n"
    "                        gating_output, e_score_correction_bias, output_indices_dtype, routed_scaling_factor,\n"
    "                        is_padding=padding_mask, input_ids=input_tokens, bias_vl=bias_vl,\n"
    "                        image_sentinel_lo=image_sentinel_lo,\n"
    "                    ))\n"
    "                return out\n"
    "            return dsv4_topk(\n"
    "                gating_output,\n"
    "                e_score_correction_bias,\n"
    "                output_indices_dtype,\n"
    "                routed_scaling_factor,\n"
    "                is_padding=padding_mask,\n"
    "                input_ids=input_tokens,\n"
    "                bias_vl=bias_vl,\n"
    "                image_sentinel_lo=image_sentinel_lo,\n"
    "            )\n"
)


def patch(src: str, kernel_dir: str) -> str:
    if MARKER in src:
        return src
    for name, anchor in (("import", IMPORT_ANCHOR), ("call", CALL_ANCHOR)):
        if src.count(anchor) != 1:
            raise SystemExit(f"anchor '{name}' found {src.count(anchor)} times (expected 1) - vLLM changed")
    if "\nimport torch\n" not in src:
        raise SystemExit("'import torch' not found - vLLM changed")
    if "\nimport sys\n" not in src:
        src = src.replace("\nimport torch\n", "\nimport sys\n\nimport torch\n", 1)
    src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + HELPER.format(marker=MARKER, kernel_dir=kernel_dir), 1)
    src = src.replace(CALL_ANCHOR, CALL_NEW.format(marker=MARKER), 1)
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--kernel-dir", default=DEFAULT_KERNEL_DIR)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    src = a.file.read_text()
    out = patch(src, a.kernel_dir)
    if a.check:
        print("patched" if MARKER in src else "applies")
        return 0
    if out != src:
        a.file.write_text(out)
        print(f"patched {a.file}")
    else:
        print(f"{a.file} already patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
