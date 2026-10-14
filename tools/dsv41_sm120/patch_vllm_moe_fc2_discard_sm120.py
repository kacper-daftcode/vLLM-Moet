#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash MoE on sm_120: FC2's padding rows leave L2 without a write-back (DeepGemmFP4Experts).

DeepGEMM's FP8 x FP4 contiguous grouped GEMM writes every row of each 64-row tile that holds a real row, so at decode
token counts FC2's output is mostly padding (6 tokens x top-6: up to 2268 of 2304 rows, 23.2 MB per MoE layer) that
the gather never reads. Those dirty lines stay in L2 and are written back by whatever streams through it next - the
next layer's FC1, which runs ~15 us slower for it on an RTX PRO 6000 (tools/dsv41_sm120/moe_l2/bench_moe_l2_tp4.py).
Right after FC2, while the workspace is still FC2's output, tools/dsv41_sm120/moe_l2/ invalidates the lines of
exactly those rows (`discard.global.L2`; expert id -1 inside a used tile, rows checked to be whole 128-byte lines).
The gather reads only real rows, so the MoE output is the same bits.

The discard runs on a side stream forked after FC2 and joined after the gather (VLLM_MOET_MOE_FC2_DISCARD_SIDE,
default 1): the gather reads only real rows - disjoint 128-byte lines - so the two run next to each other instead
of the discard in front of the gather; 4x RTX PRO 6000, in a graph: another -2.3 us per MoE layer at 6 tokens and
-3.5 at 48 against the inline launch (bench_moe_l2_tp4.py). The join sits in the same apply() call, so everything
the model stream - or a stream forked from it later - writes into the workspace afterwards (the next FC1, the
indexer's scratch: vLLM's workspace manager hands every consumer views of one buffer) is ordered after the
discard, as with the inline launch. =0 launches it inline on the model stream, as served since 2026-10-13.

Patched file: vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py (DeepGemmFP4Experts.apply: the fork
after the FC2 call, the join after the gather that ends the method; the helpers go in front of the class). The
anchors are untouched by patch_vllm_moe_glue_sm120.py and patch_vllm_moe_quant_scatter_sm120.py, so the patch
applies on the image's already patched file (Dockerfile steps 3-12) as well as on vLLM's own.
Runtime switches (read once, at the first MoE forward): VLLM_MOET_MOE_FC2_DISCARD=0 turns it off;
VLLM_MOET_MOE_FC2_DISCARD_MIN_TOKENS / _MAX_TOKENS (2 / 64): the token counts it runs at - every decode and drafter
graph of this deployment; at 1 token FC1 streams too little to pay for the launch, prefill chunks are left as they are;
VLLM_MOET_MOE_FC2_DISCARD_SIDE=0: inline instead of the side stream.

Idempotent, anchor-based. Usage:
    python3 patch_vllm_moe_fc2_discard_sm120.py [--file PATH] [--l2-dir DIR] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
)
DEFAULT_L2_DIR = "/opt/vllm-moet/dsv41_sm120/moe_l2"
MARKER = "# [vllm-moet] sm_120 MoE FC2 padding rows out of L2"

HELPER_ANCHOR = "class DeepGemmFP4Experts(mk.FusedMoEExpertsModular):\n"
HELPER_TEMPLATE = '''{marker} (tools/dsv41_sm120/moe_l2)
_MOET_L2 = None  # None = not resolved yet, False = off / unavailable, else (module, min_tokens, max_tokens)


def _moet_moe_l2():
    """The padding-row L2 discard and its token range, or None."""
    global _MOET_L2
    if _MOET_L2 is None:
        import os
        import sys

        _MOET_L2 = False
        if os.environ.get("VLLM_MOET_MOE_FC2_DISCARD", "1") == "1" and (
            current_platform.is_cuda() and current_platform.is_device_capability_family(120)
        ):
            l2_dir = os.environ.get("VLLM_MOET_MOE_L2_DIR", "{l2_dir}")
            if l2_dir not in sys.path:
                sys.path.insert(0, l2_dir)
            try:
                import moe_l2_discard_sm120 as _l2

                _l2._ext()  # build / load the extension now, not inside the first forward
                min_tokens = int(os.environ.get("VLLM_MOET_MOE_FC2_DISCARD_MIN_TOKENS", "2"))
                max_tokens = int(os.environ.get("VLLM_MOET_MOE_FC2_DISCARD_MAX_TOKENS", "64"))
                _MOET_L2 = (_l2, min_tokens, max_tokens)
                side = _moet_l2_side_stream()  # resolved here, at the first (eager) MoE forward
                logger.info_once(
                    "vllm-moet sm_120 MoE: FC2's padding rows leave L2 without a write-back at %d..%d tokens, %s "
                    "(VLLM_MOET_MOE_FC2_DISCARD=0 turns it off, VLLM_MOET_MOE_FC2_DISCARD_SIDE=0 launches inline)",
                    min_tokens,
                    max_tokens,
                    "on a side stream joined after the gather" if side is not None else "inline after FC2",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("vllm-moet sm_120 MoE FC2 discard unavailable: %r", exc)
    return _MOET_L2 or None


_MOET_L2_SIDE = None  # None = not resolved yet, False = inline (VLLM_MOET_MOE_FC2_DISCARD_SIDE=0), else the stream


def _moet_l2_side_stream() -> "torch.cuda.Stream | None":
    """The stream the discard forks onto (one per process, every MoE layer forks and joins it), or None."""
    global _MOET_L2_SIDE
    if _MOET_L2_SIDE is None:
        import os

        _MOET_L2_SIDE = (
            torch.cuda.Stream() if os.environ.get("VLLM_MOET_MOE_FC2_DISCARD_SIDE", "1") == "1" else False
        )
    return _MOET_L2_SIDE or None


def _moet_drop_padding_rows(
    out: torch.Tensor, m_indices: torch.Tensor, block_m: int, num_tokens: int
) -> "torch.cuda.Stream | None":
    """After DeepGEMM wrote out on this stream: invalidate its padding rows' L2 lines (never read).

    Returns the side stream the launch was forked onto - the caller joins it once nothing reads out any more
    and before anything writes the workspace again - or None (inline launch, or no launch).
    """
    l2 = _moet_moe_l2()
    if l2 is None or not l2[1] <= num_tokens <= l2[2]:
        return None
    side = _moet_l2_side_stream()
    if side is None or block_m <= 0 or not l2[0].discardable(out, m_indices):
        l2[0].discard_padding_rows(out, m_indices, block_m)  # inline (no launch when not discardable)
        return None
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        l2[0].discard_padding_rows(out, m_indices, block_m)
    return side


'''

FC2_ANCHOR = (
    "            mm2_out = _resize_cache(workspace2, (M_sum, K))\n"
    "            m_grouped_fp8_fp4_gemm_nt_contiguous(\n"
    "                (a2q, a2q_scale),\n"
    "                (w2.view(torch.int8), self.w2_scale),\n"
    "                mm2_out,\n"
    "                expert_ids,\n"
    "                recipe_a=(1, self._ACT_BLOCK_K),\n"
    "                recipe_b=(1, self._WEIGHT_BLOCK_K),\n"
    "            )\n"
)
# The gather that ends DeepGemmFP4Experts.apply (the FP8 class ends with the same lines, so the anchor is the FC2
# call and the gather together; the FP8 class's FC2 is m_grouped_fp8_gemm_nt_contiguous).
GATHER = (
    "\n"
    "        if apply_router_weight_on_input:\n"
    "            topk_weights = torch.ones_like(topk_weights)\n"
    "\n"
    "        deepgemm_unpermute_and_reduce(\n"
    "            a=mm2_out,\n"
    "            topk_ids=topk_ids,\n"
    "            topk_weights=topk_weights,\n"
    "            inv_perm=inv_perm,\n"
    "            expert_map=expert_map,\n"
    "            output=output,\n"
    "        )\n"
)
FC2_GATHER_ANCHOR = FC2_ANCHOR + GATHER
FC2_GATHER_NEW = (
    FC2_ANCHOR
    + f"            {MARKER}: the padding rows FC2 just wrote are never read\n"
    + "            _moet_l2_side = _moet_drop_padding_rows(mm2_out, expert_ids, align_used, topk_ids.size(0))\n"
    + GATHER
    + "        if _moet_l2_side is not None:\n"
    + f"            {MARKER}: joined after the gather, the last reader of mm2_out\n"
    + "            torch.cuda.current_stream().wait_stream(_moet_l2_side)\n"
)


def patch_text(src: str, l2_dir: str) -> str:
    if MARKER in src:
        return src
    for name, anchor in (("helper", HELPER_ANCHOR), ("FC2 call + gather", FC2_GATHER_ANCHOR)):
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"deep_gemm_moe.py: {name} anchor found {n} times (expected 1)")
    if "from vllm.platforms import current_platform\n" not in src:
        raise SystemExit("deep_gemm_moe.py: current_platform is not imported (the helper uses it)")
    helper = HELPER_TEMPLATE.replace("{marker}", MARKER).replace("{l2_dir}", l2_dir)
    out = src.replace(HELPER_ANCHOR, helper + HELPER_ANCHOR, 1)
    return out.replace(FC2_GATHER_ANCHOR, FC2_GATHER_NEW, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--l2-dir", default=DEFAULT_L2_DIR)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    if MARKER in src:
        print(f"{args.file}: already patched")
        return 0
    patched = patch_text(src, args.l2_dir)
    compile(patched, str(args.file), "exec")
    if args.check:
        print(f"{args.file}: patch applies cleanly (not written)")
        return 0
    args.file.write_text(patched)
    print(f"{args.file}: patched (FC2 padding rows out of L2 after FC2, on a side stream joined after the gather)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
