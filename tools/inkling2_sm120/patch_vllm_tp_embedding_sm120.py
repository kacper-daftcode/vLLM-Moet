#!/usr/bin/env python3
"""Inkling2 step 5: the vocabulary-parallel embedding of a TP group on sm_120.

Where: `VocabParallelEmbedding.__init__` in vllm/model_executor/layers/vocab_parallel_embedding.py. With more than one
TP rank vLLM gathers a rank's embedding rows with one kernel of its compiled library
(`torch.ops._C.vocab_parallel_embedding`: range mask, id shift, gather and output mask in one pass). The base image's
library carries no image of that kernel for capability 12; its launch is not checked, so the failure surfaces as
"CUDA error: no kernel image is available for execution on the device" at the next torch operation and the server
ends in its first warm-up step. A single-rank server never reaches the kernel.

What: on a capability-12 device `use_fused_embedding` stays off and the layer takes the path vLLM has for quantized
embeddings - mask, shift, `embedding`, `masked_fill_`, all-reduce: the same rows. VLLM_MOET_FUSED_EMBEDDING=1 leaves
vLLM's rule (for a library that has the image).

Usage: python3 patch_vllm_tp_embedding_sm120.py [--root DIR] [--check]   (idempotent; refuses any other file text)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers")
REL = "vocab_parallel_embedding.py"
MARKER = "[vllm-moet] vocab-parallel embedding on sm_120"

ANCHOR = '''        self.use_fused_embedding = (
            self.tp_size > 1
            and current_platform.is_cuda()
            and isinstance(quant_method, UnquantizedEmbeddingMethod)
        )
'''
NEW = f'''        self.use_fused_embedding = (
            self.tp_size > 1
            and current_platform.is_cuda()
            and isinstance(quant_method, UnquantizedEmbeddingMethod)
            and _moet_fused_embedding_available()
        )
'''
HELPER_ANCHOR = '''# --8<-- [start:vocab_parallel_embedding]
@PluggableLayer.register("vocab_parallel_embedding")
class VocabParallelEmbedding(PluggableLayer):
'''
HELPER = f'''def _moet_fused_embedding_available() -> bool:
    """{MARKER}: the compiled library's fused kernel has no image for capability 12 in this image; there the
    layer takes the unfused path. VLLM_MOET_FUSED_EMBEDDING=1 keeps vLLM's rule."""
    import os

    if os.environ.get("VLLM_MOET_FUSED_EMBEDDING", "0") == "1":
        return True
    capability = current_platform.get_device_capability() if current_platform.is_cuda() else None
    return capability is None or capability.major != 12


''' + HELPER_ANCHOR

EDITS = ((HELPER_ANCHOR, HELPER), (ANCHOR, NEW))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = a.root / REL
    text = path.read_text()
    if a.check:
        print("patched" if MARKER in text else "not patched")
        return 0 if MARKER in text else 1
    if MARKER in text:
        print(f"already patched {path}")
        return 0
    for anchor, new in EDITS:
        if text.count(anchor) != 1:
            raise SystemExit(f"{REL}: anchor not found exactly once (another vLLM version?):\n{anchor}")
        text = text.replace(anchor, new)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
