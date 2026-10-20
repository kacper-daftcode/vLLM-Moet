#!/usr/bin/env python3
"""The target model's lm_head at decode token counts as an MXFP8 GEMV on sm_120 (DeepSeek-V4.1-Flash) - step 18 for the target.

Owner's decision 2026-10-20 ("jedziemy całość"): the target's logits may change (MXFP8 weight + activation quantization,
~3.8 % relative error on the logits, top-5 overlap 0.96 on random data - step 18's measurements), gated by GSM8K-200
(McNemar), needle, tools/json/vision and the agreement battery. The BF16 lm_head ([32320, 5120] per TP4 rank, 331 MB)
runs once per decode step OUTSIDE the CUDA graphs and streams in ~214 us (torch profile run30_df3b_b12x_c1, 2.2 % of
the step); the MXFP8 copy (+170 MB per GPU) on step 3's GEMV takes ~104-136 us at 5-8 rows. Only <= 16 BF16 rows go
through the GEMV (C1 = 6, C2 = 12); larger row counts, prefill, soft caps, logit scales keep the BF16 path.

Shares the MXFP8 copy with step 18 (the drafter aliases the same weight tensor) through a process-wide cache keyed by
the weight's data pointer, so the +170 MB is paid once per GPU when both are on.

Patched file: vllm/models/deepseek_v41/nvidia/model.py, DeepseekV41LLMForCausalLM.compute_logits.
VLLM_MOET_TARGET_LM_HEAD_MXFP8=1 enables it (default OFF until the quality gates pass).

Usage: python3 patch_vllm_target_lm_head_mxfp8_sm120.py [--file PATH] [--check]   (idempotent; needs step 3's GEMV)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia/model.py")
MARKER = "[vllm-moet] sm_120 target lm_head MXFP8"

CLASS_ANCHOR = "class DeepseekV41LLMForCausalLM(\n"
HELPER = f'''# {MARKER}: the target's logits for decode-sized row counts on the MXFP8 GEMV (owner-approved output change,
# gated by GSM8K-200 / needle / tools; VLLM_MOET_TARGET_LM_HEAD_MXFP8=1 enables it).
_MOET_LM_HEAD_MXFP8_CACHE: dict = {{}}


def _moet_target_mxfp8_lm_head(model, hidden_states):
    import os

    if os.environ.get("VLLM_MOET_TARGET_LM_HEAD_MXFP8", "0") != "1":
        return None
    if hidden_states.dim() != 2 or hidden_states.shape[0] > 16 or hidden_states.dtype != torch.bfloat16:
        return None
    lp = model.logits_processor
    if (
        lp.logits_as_input
        or lp.soft_cap is not None
        or lp.scale != 1.0
        or lp.head_dtype not in (None, hidden_states.dtype)
    ):
        return None
    weight = model.lm_head.weight
    key = weight.data_ptr()
    state = _MOET_LM_HEAD_MXFP8_CACHE.get(key)
    if state is False:
        return None
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            from vllm.model_executor.kernels.linear.mxfp8.flashinfer import _sm120_gemv_for
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
        except ImportError:
            _MOET_LM_HEAD_MXFP8_CACHE[key] = False
            return None
        gemv = _sm120_gemv_for(hidden_states.shape[0], torch.bfloat16)
        if gemv is None:
            _MOET_LM_HEAD_MXFP8_CACHE[key] = False
            return None
        w_q, w_sf = mxfp8_e4m3_quantize(weight.data, is_sf_swizzled_layout=True)
        state = _MOET_LM_HEAD_MXFP8_CACHE[key] = (gemv, mxfp8_e4m3_quantize, w_q, w_sf)
        logger.info(
            "vllm-moet: target lm_head %s as an MXFP8 GEMV for <= 16 rows "
            "(VLLM_MOET_TARGET_LM_HEAD_MXFP8=0 restores BF16)",
            tuple(weight.shape),
        )
    gemv, quantize, w_q, w_sf = state
    rows = hidden_states.shape[0]
    if rows in (6, 7):  # step 18's measurement: 5 rows 104 us, 6 rows 136 us, 8 rows 111 us -> pad 6-7 to 8
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, 8 - rows))
    a_q, a_sf = quantize(hidden_states, is_sf_swizzled_layout=True)
    logits = gemv(a_q, a_sf, w_q, w_sf)[:rows]
    if model.lm_head.tp_size > 1:
        logits = lp._gather_logits(logits)
    return logits[..., : lp.org_vocab_size]


'''
LOGITS_OLD = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor | None:\n"
    "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
    "        return logits\n"
)
LOGITS_NEW = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor | None:\n"
    f"        # {MARKER}: decode-sized row counts on the GEMV\n"
    "        logits = _moet_target_mxfp8_lm_head(self, hidden_states)\n"
    "        if logits is not None:\n"
    "            return logits\n"
    "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
    "        return logits\n"
)


def patch_text(src: str) -> str:
    if MARKER in src:
        return src
    for name, anchor in (("target class", CLASS_ANCHOR), ("compute_logits", LOGITS_OLD),
                         ("logger", "logger = init_logger(__name__)\n")):
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"model.py: {name} anchor found {n} times (expected 1)")
    out = src.replace(CLASS_ANCHOR, HELPER + CLASS_ANCHOR, 1)
    return out.replace(LOGITS_OLD, LOGITS_NEW, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
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
    args.file.write_text(patched)
    print(f"{args.file}: patched (target lm_head as an MXFP8 GEMV at <= 16 rows, VLLM_MOET_TARGET_LM_HEAD_MXFP8=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
