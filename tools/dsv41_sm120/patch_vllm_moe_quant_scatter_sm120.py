#!/usr/bin/env python3
"""Fuse the MoE input quantization with the DeepGEMM grouped-layout permutation at decode token
counts (DeepSeek-V4.1-Flash on sm_120, DeepGemmFP4Experts: FP8 activations x MXFP4 experts).

Per MoE layer (40 per decode step) vLLM runs seven kernels between the router and FC1:
per_token_group_quant_8bit (in prepare()), Fill<int> (m_indices = -1), Fill<float> (scales = 0),
_count_expert_num_tokens, _fwd_kernel_ep_scatter_1, _fwd_kernel_ep_scatter_2, and DeepGEMM's
transpose_and_pack_fp32_into_ue8m0 inside the FC1 call - 9.4 us of a 13.5 ms step per layer on
an RTX PRO 6000 at 6 tokens x top-6 (2026-09-22 profile). tools/dsv41_sm120/moe_quant_scatter/
does the same work in one launch (3.1-3.8 us on an RTX 5090): bit-identical fp8 rows and UE8M0
scales, the scales written directly in DeepGEMM's packed int32 MN-major layout, deterministic
slots.

Patched file: vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
  DeepGemmFP4Experts.expects_unquantized_inputs -> True (prepare() hands the bf16 rows over);
  DeepGemmFP4Experts.apply: <= 1024 (token, expert) pairs, no expert map (TP without EP), UE8M0
  scale format -> fused kernel; otherwise vLLM's per_token_group_quant_fp8 runs here, followed
  by the unchanged deepgemm_moe_permute (prefill and mixed steps: the same kernels as before).
Runtime switch: VLLM_MOET_MOE_QUANT_SCATTER=0 restores the quantization in prepare().

Idempotent, anchor-based (applies before or after patch_vllm_moe_glue_sm120.py). Usage:
    python3 patch_vllm_moe_quant_scatter_sm120.py [--file PATH] [--qs-dir DIR] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
)
DEFAULT_QS_DIR = "/opt/vllm-moet/dsv41_sm120/moe_quant_scatter"
MARKER = "# [vllm-moet] sm_120 MoE quant+scatter"

HELPER_ANCHOR = "logger = init_logger(__name__)\n"
HELPER_TEMPLATE = '''

{marker}
_MOET_QS = None  # None = not resolved yet, False = off / unavailable, else the module


def _moet_quant_scatter():
    """The fused quant+permute module (tools/dsv41_sm120/moe_quant_scatter) or None."""
    global _MOET_QS
    if _MOET_QS is None:
        import os
        import sys

        _MOET_QS = False
        if os.environ.get("VLLM_MOET_MOE_QUANT_SCATTER", "1") == "1" and (
            current_platform.is_cuda() and current_platform.is_device_capability_family(120)
        ):
            qs_dir = os.environ.get("VLLM_MOET_MOE_QUANT_SCATTER_DIR", "{qs_dir}")
            if qs_dir not in sys.path:
                sys.path.insert(0, qs_dir)
            try:
                import moe_quant_scatter_sm120 as _qs

                _qs._ext()  # build / load the extension now, not inside the first forward
                _MOET_QS = _qs
                logger.info_once(
                    "vllm-moet sm_120 MoE quant+scatter: one kernel quantizes and permutes the "
                    "MoE input at decode shapes (VLLM_MOET_MOE_QUANT_SCATTER=0 turns it off)"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "vllm-moet sm_120 MoE quant+scatter unavailable, using vLLM's kernels: %r", exc
                )
    return _MOET_QS or None
'''

PROP_ANCHOR = (
    "    @staticmethod\n"
    "    def _supports_current_device() -> bool:\n"
    "        from vllm.platforms import current_platform\n"
)
PROP_NEW = (
    "    @property\n"
    "    def expects_unquantized_inputs(self) -> bool:\n"
    f"        {MARKER}: the input quantization moves from prepare() into\n"
    "        # apply(), where it is fused with the permutation at decode shapes\n"
    "        return _moet_quant_scatter() is not None\n"
    "\n"
) + PROP_ANCHOR

APPLY_HEAD_OLD = (
    "        assert a1q_scale is not None\n"
    "        assert a2_scale is None\n"
    "        assert self.w1_scale is not None\n"
    "        assert self.w2_scale is not None\n"
    "\n"
    "        a1q = hidden_states\n"
    "        _, N, _ = w1.size()\n"
    "        # K comes from activations (full hidden dim), not from w1 which is\n"
    "        # packed FP4 (E, N, K//2).\n"
    "        K = a1q.size(1)\n"
)
APPLY_HEAD_NEW = (
    f"        {MARKER}: with expects_unquantized_inputs the rows arrive\n"
    "        # unquantized (a1q_scale None). Decode shapes: one kernel quantizes and permutes\n"
    "        # them (packed UE8M0 scales, DeepGEMM takes them as they are); otherwise vLLM's\n"
    "        # quantization runs here instead of in prepare() and the permute is unchanged.\n"
    "        _qs = _moet_quant_scatter() if a1q_scale is None else None\n"
    "        fused_qs = (\n"
    "            _qs is not None\n"
    "            and DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0\n"
    "            and _qs.fused_quant_permute_applicable(\n"
    "                hidden_states, topk_ids, w1.size(0), expert_map, expert_tokens_meta\n"
    "            )\n"
    "        )\n"
    "        if a1q_scale is None and not fused_qs:\n"
    "            hidden_states, a1q_scale = per_token_group_quant_fp8(\n"
    "                hidden_states, self._ACT_BLOCK_K\n"
    "            )\n"
    "        assert fused_qs or a1q_scale is not None\n"
    "        assert a2_scale is None\n"
    "        assert self.w1_scale is not None\n"
    "        assert self.w2_scale is not None\n"
    "\n"
    "        a1q = hidden_states\n"
    "        _, N, _ = w1.size()\n"
    "        # K comes from activations (full hidden dim), not from w1 which is\n"
    "        # packed FP4 (E, N, K//2).\n"
    "        K = a1q.size(1)\n"
)

PERMUTE_OLD = (
    "        M_sum, _ = compute_aligned_M_and_alignment(\n"
    "            M=topk_ids.size(0),\n"
    "            num_topk=topk_ids.size(1),\n"
    "            local_num_experts=local_num_experts,\n"
    "            alignment=get_mk_alignment_for_contiguous_layout()[0],\n"
    "            expert_tokens_meta=expert_tokens_meta,\n"
    "        )\n"
    "\n"
    "        a1q_perm = _resize_cache(\n"
    "            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K)\n"
    "        )\n"
    "        a1q, a1q_scale, expert_ids, inv_perm, align_used = deepgemm_moe_permute(\n"
    "            aq=a1q,\n"
    "            aq_scale=a1q_scale,\n"
    "            topk_ids=topk_ids,\n"
    "            local_num_experts=local_num_experts,\n"
    "            expert_map=expert_map,\n"
    "            expert_tokens_meta=expert_tokens_meta,\n"
    "            aq_out=a1q_perm,\n"
    "        )\n"
    "        assert a1q.size(0) == M_sum\n"
    "\n"
    "        # Cap DG's BLOCK_M heuristic at the workspace's per-expert alignment;\n"
    "        # see DeepGemmExperts.apply for rationale.\n"
)
PERMUTE_NEW = (
    "        M_sum, align_used = compute_aligned_M_and_alignment(\n"
    "            M=topk_ids.size(0),\n"
    "            num_topk=topk_ids.size(1),\n"
    "            local_num_experts=local_num_experts,\n"
    "            alignment=get_mk_alignment_for_contiguous_layout()[0],\n"
    "            expert_tokens_meta=expert_tokens_meta,\n"
    "        )\n"
    "\n"
    "        a1q_perm = _resize_cache(\n"
    "            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K)\n"
    "        )\n"
    "        if fused_qs:\n"
    f"            {MARKER}: quantize + permute in one launch (same\n"
    "            # M_sum / alignment as deepgemm_moe_permute computes)\n"
    "            a1q, a1q_scale, expert_ids, inv_perm = _qs.fused_quant_permute(\n"
    "                a1q, topk_ids, local_num_experts, M_sum, align_used, aq_out=a1q_perm\n"
    "            )\n"
    "        else:\n"
    "            a1q, a1q_scale, expert_ids, inv_perm, align_used = deepgemm_moe_permute(\n"
    "                aq=a1q,\n"
    "                aq_scale=a1q_scale,\n"
    "                topk_ids=topk_ids,\n"
    "                local_num_experts=local_num_experts,\n"
    "                expert_map=expert_map,\n"
    "                expert_tokens_meta=expert_tokens_meta,\n"
    "                aq_out=a1q_perm,\n"
    "            )\n"
    "        assert a1q.size(0) == M_sum\n"
    "\n"
    "        # Cap DG's BLOCK_M heuristic at the workspace's per-expert alignment;\n"
    "        # see DeepGemmExperts.apply for rationale.\n"
)


def patch_text(src: str, qs_dir: str) -> str:
    if MARKER in src:
        return src
    for name, anchor in (("helper", HELPER_ANCHOR), ("property", PROP_ANCHOR),
                         ("apply head", APPLY_HEAD_OLD), ("permute", PERMUTE_OLD)):
        if src.count(anchor) != 1:
            raise SystemExit(f"{name} anchor found {src.count(anchor)} times (expected 1)")
    out = src.replace(HELPER_ANCHOR, HELPER_ANCHOR + HELPER_TEMPLATE.format(marker=MARKER, qs_dir=qs_dir), 1)
    out = out.replace(PROP_ANCHOR, PROP_NEW, 1)
    out = out.replace(APPLY_HEAD_OLD, APPLY_HEAD_NEW, 1)
    out = out.replace(PERMUTE_OLD, PERMUTE_NEW, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--qs-dir", default=DEFAULT_QS_DIR)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    if MARKER in src:
        print(f"{args.file}: already patched")
        return 0
    patched = patch_text(src, args.qs_dir)
    compile(patched, str(args.file), "exec")
    if args.check:
        print(f"{args.file}: patch applies cleanly (not written)")
        return 0
    dst = args.out or args.file
    dst.write_text(patched)
    print(f"{dst}: patched (quant+scatter dir {args.qs_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
