#!/usr/bin/env python3
"""DeepSeek-V4.1 Engram: the wkv projection split by output columns over the TP ranks at decode token counts.

`Engram.wkv` (6144 -> 25600, MXFP8, 157 MB) is a ReplicatedLinear: in each of the two Engram layers (1 and 14)
every TP rank streams the whole weight - in the served decode graph `mxfp8_mma_gemv_v3_kernel<1>` [3200,1,1]
takes 123 us per layer on the critical path, at the bandwidth floor for a replica. Its input is already the same
on every rank (`embed()` all-gathers the hash-head rows), so at decode token counts a rank runs the same kernel on
its 6400 rows of the loaded weight (a view; the F8_128x4-swizzled scales of whole 128-row tiles are one byte range
of the layer's) and an all-gather of the [T, 6400] slices rebuilds [T, 25600] in rank order. On 4x RTX PRO 6000
(test_engram_wkv_tp4.py) the Engram forward at 6 tokens goes from 131.8 to 56.4 us; above ~64 tokens the
all-gather costs more than the weight bytes it saves (4096 tokens: +1.8 ms), so prefill keeps the replicated GEMM
and the weights stay replicated.

Bit for bit the replicated result: every output column is computed by one block over the full K whatever N is -
the vLLM-Moet GEMV for M <= 16 (v3 / v1: 8 columns per block, K split over the block's 8 warps the same way for
every N, no split-K at the default VLLM_MOET_GEMV_V3_SPLITK_MAX=1) and FlashInfer's SM120 CUTLASS GEMM above
(persistent scheduler, whole output tiles, no split-K / stream-K in any of its 6 tactics); test_engram_wkv_tp.py
and test_engram_wkv_tp4.py check it. Only that path splits (ModelOptLinearMethod with the default format scheme on
FlashInferCutlassMxfp8LinearKernel); any other linear method, TP = 1 and sequence parallelism (each rank holds
different tokens there) run the replicated layer.

Patched file: vllm/models/deepseek_v41/common/engram.py (Engram.forward calls wkv through _moet_engram_wkv).
Runtime switches: VLLM_MOET_ENGRAM_WKV_TP=0 (replicated at every token count), VLLM_MOET_ENGRAM_WKV_TP_MAX_TOKENS
(64: the largest token count that splits).

Idempotent, anchor-based. Usage:
    python3 patch_vllm_engram_wkv_tp.py [--file PATH] [--out PATH] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/common/engram.py")
MARKER = "# [vllm-moet] Engram wkv split by output columns"

HELPER_ANCHOR = "logger = init_logger(__name__)\n"
HELPER = '''

{marker} at decode token counts (tools/dsv41_sm120).
_MOET_WKV_TP = None  # None = not resolved yet, else (enabled, max_tokens)


def _moet_wkv_tp():
    global _MOET_WKV_TP
    if _MOET_WKV_TP is None:
        import os

        _MOET_WKV_TP = (
            os.environ.get("VLLM_MOET_ENGRAM_WKV_TP", "1") == "1",
            int(os.environ.get("VLLM_MOET_ENGRAM_WKV_TP_MAX_TOKENS", "64")),
        )
    return _MOET_WKV_TP


def _moet_wkv_part(wkv, tp_size):
    """This rank's output rows of the loaded wkv as a layer-like view, or None off the validated path.

    FlashInfer's F8_128x4 layout stores the scales tile by tile (128 rows x all K blocks), so for a rank
    slice of whole tiles the swizzled scales of its rows are one contiguous byte range of the layer's.
    """
    from types import SimpleNamespace

    qm = wkv.quant_method
    weight, scale = wkv.weight, getattr(wkv, "weight_scale", None)
    rows = weight.shape[0]
    if not (
        scale is not None
        and type(qm).__name__ == "ModelOptLinearMethod"
        and type(getattr(qm, "fmt", None)).__name__ == "FormatScheme"
        and type(getattr(qm, "kernel", None)).__name__ == "FlashInferCutlassMxfp8LinearKernel"
        and weight.dtype == torch.float8_e4m3fn
        and weight.is_contiguous()
        and scale.dim() == 1
        and rows % (128 * tp_size) == 0
        and scale.numel() % tp_size == 0
    ):
        return None
    rank = get_tensor_model_parallel_rank()
    cols, chunk = rows // tp_size, scale.numel() // tp_size
    return SimpleNamespace(
        weight=weight[rank * cols : (rank + 1) * cols],
        weight_scale=scale[rank * chunk : (rank + 1) * chunk],
    )


def _moet_engram_wkv(wkv, x, use_sequence_parallel):
    """wkv(x), bit for bit; up to VLLM_MOET_ENGRAM_WKV_TP_MAX_TOKENS tokens each TP rank computes 1/TP of
    the output columns on the same kernel and an all-gather rebuilds the rest.

    Above that the all-gather costs more than the weight bytes it saves, so prefill runs the replicated
    layer; so do TP = 1, sequence parallelism (the ranks hold different tokens) and any linear method
    other than the MXFP8 one the split was validated on.
    """
    enabled, max_tokens = _moet_wkv_tp()
    tp_size = get_tensor_model_parallel_world_size()
    if not enabled or tp_size == 1 or use_sequence_parallel or x.shape[0] > max_tokens:
        return wkv(x)
    scale = getattr(wkv, "weight_scale", None)
    src = (wkv.weight.data_ptr(), 0 if scale is None else scale.data_ptr())
    cached = getattr(wkv, "_moet_wkv_part", None)
    if cached is None or cached[0] != src:
        part = _moet_wkv_part(wkv, tp_size)
        if part is None:
            logger.warning_once(
                "vllm-moet: Engram wkv stays replicated: %s / %s is not the validated MXFP8 path",
                type(wkv.quant_method).__name__,
                type(getattr(wkv.quant_method, "kernel", None)).__name__,
            )
        else:
            logger.info_once(
                "vllm-moet: Engram wkv split over TP=%d up to %d tokens (%d of %d output columns per "
                "rank + all-gather; VLLM_MOET_ENGRAM_WKV_TP=0 keeps it replicated)",
                tp_size,
                max_tokens,
                part.weight.shape[0],
                wkv.weight.shape[0],
            )
        cached = (src, part)
        wkv._moet_wkv_part = cached
    part = cached[1]
    if part is None:
        return wkv(x)
    return tensor_model_parallel_all_gather(wkv.quant_method.apply(part, x, None))

'''

FORWARD_OLD = "        kv = self.wkv(self.embed(hash_ids).flatten(-2))\n"
FORWARD_NEW = (
    "        kv = _moet_engram_wkv(\n"
    "            self.wkv, self.embed(hash_ids).flatten(-2), self.use_sequence_parallel\n"
    "        )\n"
)


def _replace(src: str, old: str, new: str, name: str) -> str:
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"{name}: anchor found {n} times (expected 1)")
    return src.replace(old, new, 1)


def patch_text(src: str) -> str:
    if MARKER in src:
        return src
    for name in ("get_tensor_model_parallel_rank", "get_tensor_model_parallel_world_size",
                 "tensor_model_parallel_all_gather"):
        if f"    {name},\n" not in src:
            raise SystemExit(f"engram.py: {name} is not imported (the helper uses it)")
    if "        self.wkv = ReplicatedLinear(\n" not in src:
        raise SystemExit("engram.py: wkv is not a ReplicatedLinear any more - re-validate the split")
    out = _replace(src, HELPER_ANCHOR, HELPER_ANCHOR + HELPER.replace("{marker}", MARKER), "helper")
    return _replace(out, FORWARD_OLD, FORWARD_NEW, "Engram.forward wkv call")


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
    print(f"{args.out or args.file}: patched (Engram wkv split at decode token counts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
