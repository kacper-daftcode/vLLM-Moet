#!/usr/bin/env python3
"""Inkling2 step 2: the relative attention on sm_120 (tools/inkling2_sm120/attn).

vLLM computes Inkling2's attention with two sm_100 kernels: `_rel_verify_kernel` (Triton Gluon - tcgen05 MMA, tensor
memory, TMA) for decode, verify and the DSpark draft's block attention, and tml_fa4's sheared-bias forward for
prefill. Capability 12 has neither instruction set. The fallback vLLM carries for sm_103 (upstream cute FA4 with a
score_mod relative bias) does not help there: on capability 12 that package's forward is the mma.sync class, which
takes a page table and does not read it.

Where (vllm/models/inkling2/ops/attention.py):
  `_decode_attention`                     the launch of `_rel_verify_kernel` and its combine - the entry of
                                          `rel_decode_attention` and `block_decode_attention`;
  `Inkling2FA4RelAttentionKernel.kernel`  the prefill entry (fp32 output and LSE; under the JIT warm-up's fake
                                          tensors there is nothing to compile).
On a capability-12 device both take inkling2_attn_sm120: the same operands (the paged K/V cache, the sheared relative
logits the prologue writes, gate and sinks), the same outputs. The split plans (`plan_decode_attention`,
`plan_attention`) stay vLLM's.

Switches: VLLM_MOET_INKLING2_ATTN = kernel (default) | ref (the torch definition of the operation - the tests'
oracle; slow) | 0 (vLLM's path, which cannot run on sm_120). VLLM_MOET_INKLING2_DIR is the module directory.

Usage: python3 patch_vllm_inkling2_attn_sm120.py [--root DIR] [--check]   (idempotent; refuses any other file text)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/inkling2")
REL = "ops/attention.py"
MARKER = "[vllm-moet] inkling2 relative attention on sm_120"

HELPER_ANCHOR = '''_FA4_TILE_M = 256
_FA4_SM_FLOPS_PER_US = 10e6
_FA4_MERGE_BYTES_PER_US = 4e6
'''
HELPER = HELPER_ANCHOR + f'''

# {MARKER}: on capability 12 the decode / verify / draft kernel (Gluon, tcgen05) and the prefill kernel (tml_fa4,
# sheared bias) are replaced by tools/inkling2_sm120/attn. VLLM_MOET_INKLING2_ATTN=0 turns it off.
@cache
def _moet_attn_sm120():
    """The sm_120 attention module, or None where vLLM's own kernels apply."""
    import os

    if os.environ.get("VLLM_MOET_INKLING2_ATTN", "kernel") == "0":
        return None
    capability = current_platform.get_device_capability() if current_platform.is_cuda() else None
    if capability is None or capability.major != 12:
        return None
    import sys

    path = os.path.join(os.environ.get("VLLM_MOET_INKLING2_DIR", "/opt/vllm-moet/inkling2_sm120"), "attn")
    if path not in sys.path:
        sys.path.insert(0, path)
    import inkling2_attn_sm120

    if inkling2_attn_sm120.mode() != "ref":
        inkling2_attn_sm120._ext()  # compiled with the image; loaded before any CUDA graph is captured
    return inkling2_attn_sm120
'''

DECODE_ANCHOR = '''    """Launch ``_rel_verify_kernel`` (and the combine for several splits);
    ``rel_gate_sinks`` None is the draft mode, ``window`` 0 a full layer."""
    num_tokens, num_heads, head_dim = q.shape
'''
DECODE_NEW = '''    """Launch ``_rel_verify_kernel`` (and the combine for several splits);
    ``rel_gate_sinks`` None is the draft mode, ``window`` 0 a full layer."""
    moet = _moet_attn_sm120()
    if moet is not None:
        return moet.decode_attention(
            q,
            key_cache,
            value_cache,
            out,
            rel_gate_sinks,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            max_query_len,
            num_actual_tokens,
            softmax_scale,
            window,
            num_splits,
        )
    num_tokens, num_heads, head_dim = q.shape
'''

PREFILL_ANCHOR = '''        """Write ``out`` and the ``(num_heads, num_tokens)`` LSE ``lse``."""
        if not _use_sheared_fa4():
'''
PREFILL_NEW = '''        """Write ``out`` and the ``(num_heads, num_tokens)`` LSE ``lse``."""
        moet = _moet_attn_sm120()
        if moet is not None:
            from torch._subclasses.fake_tensor import is_fake

            if not is_fake(q):
                moet.prefill_attention(
                    q,
                    key_cache,
                    value_cache,
                    block_table=block_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    softmax_scale=softmax_scale,
                    window_size=window_size,
                    sheared_bias=sheared_bias,
                    out=out,
                    lse=lse,
                )
            return
        if not _use_sheared_fa4():
'''

EDITS = ((HELPER_ANCHOR, HELPER), (DECODE_ANCHOR, DECODE_NEW), (PREFILL_ANCHOR, PREFILL_NEW))


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
