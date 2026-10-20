#!/usr/bin/env python3
"""DSpark's sequential Markov sampling as one launch per draft step on sm_120 (DeepSeek-V4.1-Flash): vLLM's gfx942
kernels (vllm/models/deepseek_v41/amd/ops/fused_markov.py) on CUDA.

Every decode step the drafter samples its 5 drafts one after the other: per step vLLM's loop gathers the Markov
embedding of the previous token, runs the bias GEMV over the whole vocabulary (markov_w2: 129280 x 256 bf16, 66 MB,
replicated), adds it to the base logits and samples (Gumbel kernel, argmax, gather, copy) - ~8 kernels a step, 158 us of
the drafter graph on the RTX PRO 6000 (2026-10-20 profile, rank 0). vLLM's gfx942 path computes a block of the bias,
adds the base logits, stores the step's logits into the draft-logits cache and takes the Gumbel-max of its block in one
launch; the next step's programs reduce the previous step's block maxima themselves. The kernels are Triton and run
unchanged on sm_120; upstream gates them to gfx942 with --monokernel. The bias is an fp32 sum of 256 products rounded to
bf16 like vLLM's GEMV, added in another order: a bias value can differ in its last bf16 bit and then a draft token can
differ. Only the drafts change - the rejection sampler checks every draft token against the target's distribution.
fused_markov.sample_sequential itself returns to vLLM's loop for what it does not compute the same way (greedy drafts,
a watermarker, a reduced draft vocabulary, the confidence head, the acceptance estimator, > 8 requests, non-bf16
weights or logits).

Patched file: vllm/models/deepseek_v41/nvidia/dspark.py - DSparkDeepseekV4ForCausalLM.sample_sequential, which
DSparkSpeculator._sample_sequential asks before its loop. VLLM_MOET_FUSED_MARKOV=0 keeps vLLM's loop.

Usage: python3 patch_vllm_dspark_fused_markov_sm120.py [--file PATH] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia/dspark.py")
MARKER = "[vllm-moet] sm_120 fused Markov sampling"

ANCHOR = (
    "    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:\n"
    "        return self.model.markov_head.embed(token_ids)\n"
)
METHOD = '''    def sample_sequential(self, speculator, num_reqs: int, head_hidden: torch.Tensor) -> bool:
        """{marker}: DSparkSpeculator._sample_sequential asks this first.
        On sm_120 each Markov step is one launch of vLLM's gfx942 kernels (amd/ops/fused_markov.py); False,
        having done nothing, sends the step to vLLM's loop."""
        import os

        if os.environ.get("VLLM_MOET_FUSED_MARKOV", "1") != "1" or torch.cuda.get_device_capability()[0] != 12:
            return False
        from vllm.models.deepseek_v41.amd.ops import fused_markov

        # upstream gates the kernels to gfx942 + --monokernel; on sm_120 they always apply
        fused_markov.enabled = _moet_fused_markov_on
        return fused_markov.sample_sequential(speculator, num_reqs, head_hidden)

'''
HELPER_ANCHOR = "class DSparkDeepseekV4ForCausalLM(nn.Module):\n"
HELPER = '''# {marker}: the gate fused_markov.sample_sequential checks.
def _moet_fused_markov_on() -> bool:
    return True


'''


def patch(src: str) -> str:
    if MARKER in src:
        return src
    for name, anchor in (("method", ANCHOR), ("class", HELPER_ANCHOR)):
        if src.count(anchor) != 1:
            raise SystemExit(f"anchor '{name}' found {src.count(anchor)} times (expected 1) - vLLM changed")
    if "def sample_sequential(" in src:
        raise SystemExit("dspark.py already defines sample_sequential - vLLM changed")
    src = src.replace(HELPER_ANCHOR, HELPER.format(marker=MARKER) + HELPER_ANCHOR, 1)
    src = src.replace(ANCHOR, METHOD.format(marker=MARKER) + ANCHOR, 1)
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    src = a.file.read_text()
    out = patch(src)
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
