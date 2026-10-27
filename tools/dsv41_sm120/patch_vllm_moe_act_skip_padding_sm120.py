#!/usr/bin/env python3
"""The DeepGEMM MoE activation + FP8 quant skips the expert padding rows on sm_120 (GLM-5.3-Flash line, step G2).

vLLM's DeepGEMM MoE path lays the routed tokens out expert by expert, each expert's rows padded to the GEMM's M alignment
(deepgemm_moe_permute / _fwd_kernel_ep_scatter_1: expert_start_loc = cumsum of the rounded-up counts, live rows first).
Between the two grouped GEMMs, silu_mul_quant_fp8_packed_triton (gate * silu(up) -> FP8 + packed UE8M0 scales) runs over
the whole padded M_sum unless it is given `expert_ends` (the exclusive end of every expert's live rows) - which upstream
only does on sm_90 / sm_100 via the `use_psum_layout` branch; sm_120 gets the dense launch. On GLM-5.3-Flash (288 experts,
top-8) that is most of the work at any batch: profile run1 (26.10, rank 0): `_silu_mul_quant_fp8_packed_kernel`
25.5 us x 43 layers = 1.10 ms of the 25.3 ms C8 step (4.3 %; ~100 live experts x 128 padded rows x 1024 columns streamed
for 256 live rows), 3.6 us x 43 = 0.16 ms at C1.

The patch passes the ends in the standard layout, where they are exactly what the kernel expects (an expert starts at
the previous end rounded up to the alignment - the padded start): expert_ends[e] = expert_start_loc[e] + expert_num_tokens[e],
one int32 add over the experts inside deepgemm_moe_permute (an optional output), handed to _act_mul_quant together with
the alignment the permute used. Per live row the kernel computes the same values as before (the ends only mask rows), so
the bits are unchanged; the padded rows of the FP8 activation stay unwritten, which the second GEMM computes and the
unpermute discards, as it already does for the padding of the first GEMM's output. Rows: sm_120, FP8 block scales with
UE8M0 (the only sm_120 route to the fused kernel), SiLU / swiglu-oai activation, not MXFP8, not the psum layout.

Runtime switch: VLLM_MOET_MOE_ACT_SKIP_PADDING=1 (default 0 = stock dense launch).

Usage: python3 patch_vllm_moe_act_skip_padding_sm120.py [--dist DIR] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "[vllm-moet] act-quant skips expert padding (sm_120)"

# --- deep_gemm_utils.py: deepgemm_moe_permute gets an optional expert_ends output -------------------------------------------
UTILS_SIG_OLD = (
    "    block_size: int | None = None,\n"
    "    use_psum_layout: bool = False,\n"
    "):\n"
    "    assert aq.ndim == 2\n"
)
UTILS_SIG_NEW = (
    "    block_size: int | None = None,\n"
    "    use_psum_layout: bool = False,\n"
    f"    expert_ends_out: torch.Tensor | None = None,  # {MARKER}: int32 [local_num_experts], filled (standard layout only)\n"
    "):\n"
    "    assert aq.ndim == 2\n"
)
UTILS_RET_OLD = (
    "        use_psum_layout=use_psum_layout,\n"
    "    )\n"
    "\n"
    "    return aq_out, aq_scale_out, grouped_layout, inv_perm, align_used\n"
)
UTILS_RET_NEW = (
    "        use_psum_layout=use_psum_layout,\n"
    "    )\n"
    f"    if expert_ends_out is not None and not use_psum_layout:  # {MARKER}\n"
    "        # the exclusive end of each expert's live rows in the padded layout: its padded start + its token count\n"
    "        torch.add(expert_start_loc, expert_num_tokens.to(torch.int32), out=expert_ends_out)\n"
    "\n"
    "    return aq_out, aq_scale_out, grouped_layout, inv_perm, align_used\n"
)

# --- experts/deep_gemm_moe.py: the sm_120 apply path asks for the ends and hands them to the act-quant ----------------------
MOE_IMPORT_ANCHOR = "logger = init_logger(__name__)\n"
MOE_HELPER = f'''

# {MARKER}: VLLM_MOET_MOE_ACT_SKIP_PADDING=1 lets the fused silu-mul-quant between the grouped GEMMs skip the expert
# padding rows on sm_120 (upstream only does so on sm_90 / sm_100 through the psum layout); bits unchanged, rows masked.
_MOET_ACT_SKIP_PADDING: bool | None = None


def _moet_act_skip_padding_enabled() -> bool:
    global _MOET_ACT_SKIP_PADDING
    if _MOET_ACT_SKIP_PADDING is None:
        import os

        _MOET_ACT_SKIP_PADDING = (
            os.environ.get("VLLM_MOET_MOE_ACT_SKIP_PADDING", "0") == "1"
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        )
    return _MOET_ACT_SKIP_PADDING
'''
MOE_CALL_OLD = (
    "            block_size=self.block_shape[1] if self.mxfp8 else None,\n"
    "            use_psum_layout=use_psum_layout,\n"
    "        )\n"
    "        assert a1q.size(0) == M_sum\n"
)
MOE_CALL_NEW = (
    "            block_size=self.block_shape[1] if self.mxfp8 else None,\n"
    "            use_psum_layout=use_psum_layout,\n"
    "            expert_ends_out=_moet_ends,\n"
    "        )\n"
    "        assert a1q.size(0) == M_sum\n"
)
MOE_PRE_OLD = (
    "        a1q, a1q_scale, grouped_layout, inv_perm, align_used = deepgemm_moe_permute(\n"
    "            aq=a1q,\n"
)
MOE_PRE_NEW = (
    f"        # {MARKER}: the live-row ends for the act-quant, standard layout, sm_120 FP8-block UE8M0, gated activations\n"
    "        _moet_ends = (\n"
    "            torch.empty((local_num_experts,), device=a1q.device, dtype=torch.int32)\n"
    "            if (\n"
    "                not use_psum_layout\n"
    "                and not self.mxfp8\n"
    "                and _moet_act_skip_padding_enabled()\n"
    "                and DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0\n"
    "                and activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE)\n"
    "            )\n"
    "            else None\n"
    "        )\n"
    "        a1q, a1q_scale, grouped_layout, inv_perm, align_used = deepgemm_moe_permute(\n"
    "            aq=a1q,\n"
)
MOE_ACT_OLD = (
    "                expert_ends=grouped_layout if use_psum_layout else None,\n"
    "                expert_alignment=align_used if use_psum_layout else 0,\n"
)
MOE_ACT_NEW = (
    f"                expert_ends=grouped_layout if use_psum_layout else _moet_ends,  # {MARKER}\n"
    "                expert_alignment=align_used if (use_psum_layout or _moet_ends is not None) else 0,\n"
)


def _once(src: str, anchor: str, what: str, name: str) -> None:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"{name}: {what} anchor found {n} times (expected 1)")


def patch_utils(src: str) -> str:
    if MARKER in src:
        return src
    for what, anchor in (("permute signature", UTILS_SIG_OLD), ("permute return", UTILS_RET_OLD)):
        _once(src, anchor, what, "deep_gemm_utils.py")
    return src.replace(UTILS_SIG_OLD, UTILS_SIG_NEW, 1).replace(UTILS_RET_OLD, UTILS_RET_NEW, 1)


def patch_moe(src: str) -> str:
    if MARKER in src:
        return src
    for what, anchor in (("platform import", MOE_IMPORT_ANCHOR), ("permute call", MOE_CALL_OLD), ("permute unpack", MOE_PRE_OLD),
                         ("act-quant call", MOE_ACT_OLD)):
        _once(src, anchor, what, "experts/deep_gemm_moe.py")
    src = src.replace(MOE_IMPORT_ANCHOR, MOE_IMPORT_ANCHOR + MOE_HELPER, 1)
    src = src.replace(MOE_PRE_OLD, MOE_PRE_NEW, 1)
    src = src.replace(MOE_CALL_OLD, MOE_CALL_NEW, 1)
    return src.replace(MOE_ACT_OLD, MOE_ACT_NEW, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=DIST)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    files = (
        (args.dist / "model_executor" / "layers" / "fused_moe" / "deep_gemm_utils.py", patch_utils),
        (args.dist / "model_executor" / "layers" / "fused_moe" / "experts" / "deep_gemm_moe.py", patch_moe),
    )
    for path, fn in files:
        src = path.read_text()
        if MARKER in src:
            print(f"{path}: already patched")
            continue
        out = fn(src)
        compile(out, str(path), "exec")
        if args.check:
            print(f"{path}: patch applies cleanly (not written)")
            continue
        path.write_text(out)
        print(f"{path}: patched ({MARKER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
