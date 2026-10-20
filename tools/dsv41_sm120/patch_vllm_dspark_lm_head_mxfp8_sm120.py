#!/usr/bin/env python3
"""The DSpark drafter's lm_head at decode token counts as an MXFP8 GEMV on sm_120 (DeepSeek-V4.1-Flash).

The drafter aliases the target's BF16 lm_head ([32320, 5120] per TP4 rank, 331 MB) and runs it once per decode step
for its draft positions; cuBLAS streams the BF16 weight in ~210 us (torch profile 2026-10-20: 2.2 % of a b12x decode
step). An MXFP8 copy of the weight (e4m3 + ue8m0 per 32, F8_128x4-swizzled scales: +170 MB per GPU) on the vLLM-Moet
GEMV of step 3 takes ~104 us at 5 rows (activations quantized the same way). Only the drafter changes: the target's
logits stay BF16, so outputs are those of the target; the drafts (and with them the acceptance) can differ where the
drafter's top two logits are within the MXFP8 error (~3.8 % relative on the logits).

Patched file: vllm/models/deepseek_v41/nvidia/dspark.py, DSparkDeepseekV4ForCausalLM.compute_logits: <= 16 BF16 rows
go through the GEMV, then the LogitsProcessor's own TP gather and vocab-padding cut; anything else (prefill, a soft cap,
a logit scale, another head dtype, the GEMV unavailable) takes the original path. The copy is made on the first eligible
call outside CUDA graph capture (the memory-profiling forward). VLLM_MOET_DRAFTER_LM_HEAD_MXFP8=0 turns it off.

Usage: python3 patch_vllm_dspark_lm_head_mxfp8_sm120.py [--file PATH] [--check]   (idempotent; needs step 3's patch)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia/dspark.py")
MARKER = "[vllm-moet] sm_120 drafter lm_head MXFP8"

CLASS_ANCHOR = "class DSparkDeepseekV4ForCausalLM(nn.Module):\n"
HELPER = f'''# {MARKER}: the drafter's logits for decode-sized row counts on the MXFP8 GEMV.
def _moet_mxfp8_lm_head(model, hidden_states):
    import os

    if os.environ.get("VLLM_MOET_DRAFTER_LM_HEAD_MXFP8", "1") != "1":
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
    state = getattr(model, "_moet_lm_head_mxfp8", None)
    if state is False:
        return None
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            from vllm.model_executor.kernels.linear.mxfp8.flashinfer import _sm120_gemv_for
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
        except ImportError:
            model._moet_lm_head_mxfp8 = False
            return None
        gemv = _sm120_gemv_for(hidden_states.shape[0], torch.bfloat16)
        if gemv is None:
            model._moet_lm_head_mxfp8 = False
            return None
        w_q, w_sf = mxfp8_e4m3_quantize(model.lm_head.weight.data, is_sf_swizzled_layout=True)
        state = model._moet_lm_head_mxfp8 = (gemv, mxfp8_e4m3_quantize, w_q, w_sf)
        logger.info(
            "vllm-moet: DSpark drafter lm_head %s as an MXFP8 GEMV for <= 16 rows "
            "(VLLM_MOET_DRAFTER_LM_HEAD_MXFP8=0 restores BF16)",
            tuple(model.lm_head.weight.shape),
        )
    gemv, quantize, w_q, w_sf = state
    a_q, a_sf = quantize(hidden_states, is_sf_swizzled_layout=True)
    logits = gemv(a_q, a_sf, w_q, w_sf)
    if model.lm_head.tp_size > 1:
        logits = lp._gather_logits(logits)
    return logits[..., : lp.org_vocab_size]


'''
LOGITS_OLD = (
    "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
    '        """Base logits U_k = lm_head(norm(head_hidden))."""\n'
    "        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))\n"
)
LOGITS_NEW = (
    "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
    '        """Base logits U_k = lm_head(norm(head_hidden))."""\n'
    "        normed = self.model.norm(hidden_states)\n"
    f"        # {MARKER}: decode-sized row counts on the GEMV\n"
    "        logits = _moet_mxfp8_lm_head(self, normed)\n"
    "        if logits is not None:\n"
    "            return logits\n"
    "        return self.logits_processor(self.lm_head, normed)\n"
)


def patch_text(src: str) -> str:
    if MARKER in src:
        return src
    for name, anchor in (("drafter class", CLASS_ANCHOR), ("compute_logits", LOGITS_OLD),
                         ("logger", "logger = init_logger(__name__)\n")):
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"dspark.py: {name} anchor found {n} times (expected 1)")
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
    print(f"{args.file}: patched (drafter lm_head as an MXFP8 GEMV at <= 16 rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
