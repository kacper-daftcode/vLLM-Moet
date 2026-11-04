#!/usr/bin/env python3
"""DeepSeek-V4.1 sparse-MLA prefill on sm_120 through FlashInfer's full-BF16 route (front "cubit + prefill", Phase 0/1).

Today every prefill chunk of the DSV4.1 FlashInfer sparse backend (`deepseek_v41/nvidia/flashinfer_sparse.py::
_forward_prefill`) goes through `trtllm_batch_decode_sparse_mla_dsv4(kv_cache_format="fp8_dsv41_fp4_ca" | "fp8_dsv41",
kv_scale_placement="interleaved")`, whose SM120 route for prefill rows is the single-group FP8 kernel
(`sparse_mla_prefill_kernel<DSV4_1, FP8, 16, 64, Dsv41MixedCachePrefillSchedule>`: Q quantized to FP8, FP8 x FP8 QK with
per-32 scales, FP8 P, FP8 PV). FlashInfer 0.7.2 also exports, for exactly this storage, the full-BF16 prefill
(`sparse_mla_prefill_dsv41_bf16_kernel<16, extra_fp4>`: both caches dequantized exactly to BF16 on chip, BF16 x BF16 QK
with fp32 accumulation, fp32 softmax, BF16 P, fp32 PV accumulation, one BF16 output rounding) - reachable only through
`SparseMLASm120Wrapper(compute_precision="bf16")`, not through the trtllm entry. The static census of the two kernels
(internal: L/census/CENSUS.md) counts ~70 non-MMA instructions per MMA in the FP8 kernel against ~17 in the BF16 one.

With VLLM_MOET_SPARSE_MLA_PREFILL_BF16=1 the prefill chunks of the V4.1-record configuration (`kv_cache_dtype ==
"nvfp4_ds_mla"`: 528-byte fp8 SWA records + 288-byte FP4 compressed records, interleaved scales) run through a cached
wrapper per (has compressed cache) with compute_precision="bf16"; decode rows, the DSV4 record configuration and every
other backend are untouched. The arithmetic changes (this is the Level-A / spec-B decision of the front, taken once,
with the quality battery), so the switch defaults to 0.

Usage: python3 patch_vllm_sparse_mla_prefill_bf16_sm120.py [--dist DIR] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "[vllm-moet] sparse-MLA prefill BF16 route (sm_120)"

IMPORT_ANCHOR = "from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4\n"
HELPER = f'''

# {MARKER}: VLLM_MOET_SPARSE_MLA_PREFILL_BF16=1 routes the V4.1-record prefill chunks through FlashInfer's full-BF16
# SM120 prefill kernel (SparseMLASm120Wrapper(compute_precision="bf16")); the trtllm entry has no precision knob.
def _moet_prefill_bf16_enabled() -> bool:
    import os

    return os.environ.get("VLLM_MOET_SPARSE_MLA_PREFILL_BF16", "0") == "1"


_MOET_PREFILL_BF16_WRAPPERS: dict = {{}}


def _moet_prefill_bf16_wrapper(has_extra: bool, device: "torch.device"):
    key = (bool(has_extra), str(device))
    w = _MOET_PREFILL_BF16_WRAPPERS.get(key)
    if w is None:
        from flashinfer.mla import SparseMLASm120Wrapper

        w = SparseMLASm120Wrapper(
            kv_scale_format="ue8m0_g32",
            kv_cache_format="fp8",
            extra_kv_fp4=bool(has_extra),
            compute_precision="bf16",
            backend="sparse",
            kv_scale_placement="interleaved",
            device=device,
        )
        _MOET_PREFILL_BF16_WRAPPERS[key] = w
    return w
'''

CALL_OLD = (
    "            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(\n"
    "                query=q_chunk,\n"
    "                swa_kv_cache=swa_kv_paged,\n"
    "                workspace_buffer=self._get_workspace(q.device),\n"
    "                sparse_indices=swa_indices_chunk,\n"
    "                compressed_kv_cache=extra_kv_paged,\n"
    "                out=output[query_start:query_end],\n"
    "                bmm1_scale=self.scale,\n"
    "                sinks=self.attn_sink,\n"
    "                kv_layout=\"NHD\",\n"
    "                swa_topk_lens=swa_lens_chunk,\n"
    "                extra_sparse_indices=extra_sparse_indices_chunk,\n"
    "                extra_sparse_topk_lens=extra_sparse_lengths_chunk,\n"
    "                **self._kv_cache_format_kwargs(extra_kv_paged is not None),\n"
    "            )\n"
)
CALL_NEW = (
    f"            if (  # {MARKER}\n"
    "                self.kv_cache_dtype == \"nvfp4_ds_mla\"\n"
    "                and _moet_prefill_bf16_enabled()\n"
    "            ):\n"
    "                _moet_prefill_bf16_wrapper(extra_kv_paged is not None, q.device).run(\n"
    "                    q_chunk,\n"
    "                    swa_kv_paged,\n"
    "                    swa_indices_chunk,\n"
    "                    output[query_start:query_end],\n"
    "                    self.scale,\n"
    "                    topk_length=swa_lens_chunk,\n"
    "                    attn_sink=self.attn_sink,\n"
    "                    extra_kv_cache=extra_kv_paged,\n"
    "                    extra_indices=extra_sparse_indices_chunk,\n"
    "                    extra_topk_length=extra_sparse_lengths_chunk,\n"
    "                )\n"
    "                continue\n"
    + CALL_OLD
)


def _once(src: str, anchor: str, what: str, name: str) -> None:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"{name}: {what} anchor found {n} times (expected 1)")


def patch(src: str) -> str:
    if MARKER in src:
        return src
    for what, anchor in (("import", IMPORT_ANCHOR), ("prefill call", CALL_OLD)):
        _once(src, anchor, what, "deepseek_v41/nvidia/flashinfer_sparse.py")
    src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + HELPER, 1)
    return src.replace(CALL_OLD, CALL_NEW, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=DIST)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    path = args.dist / "models" / "deepseek_v41" / "nvidia" / "flashinfer_sparse.py"
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
