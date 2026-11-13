#!/usr/bin/env python3
"""Inkling2 step 8: `reshape_and_cache_flash` on sm_120 - the DSpark draft's K / V write.

Where: `reshape_and_cache_flash` in vllm/_custom_ops.py, the scatter of a step's keys and values into the paged cache
that `FlashAttentionImpl.do_kv_cache_update` calls. The target model writes its cache in its own Triton prologue and
never reaches it; the DSpark draft's layers do (`Inkling2DraftSelfAttention._write_kv`, the block's K / V and the
context K / V of the target features). It is one kernel of vLLM's compiled library, and the base image's library
carries no image of it for capability 12: the launch raises "CUDA error: no kernel image is available for execution
on the device", so a server with the draft ends in the draft's first forward.

What: on a capability-12 device the call goes to vLLM's Triton kernel of the same operation
(`vllm.v1.attention.ops.triton_reshape_and_cache_flash`): the same bytes at the same slots, for both cache layouts,
with no host-side read (it is captured in the draft's CUDA graph). VLLM_MOET_CACHE_FLASH_C=1 keeps the compiled
kernel (for a library that has the image).

Usage: python3 patch_vllm_cache_flash_sm120.py [--root DIR] [--check]   (idempotent; refuses any other file text)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
REL = "_custom_ops.py"
MARKER = "[vllm-moet] reshape_and_cache_flash on sm_120"

HELPER_ANCHOR = '''def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache_flash(
'''
HELPER = f'''_MOET_CACHE_FLASH_TRITON: bool | None = None


def _moet_cache_flash_triton() -> bool:
    """{MARKER}: the compiled library's kernel has no image for capability 12 in this image; there the scatter runs
    on vLLM's Triton kernel of the same operation. VLLM_MOET_CACHE_FLASH_C=1 keeps the compiled kernel."""
    global _MOET_CACHE_FLASH_TRITON
    if _MOET_CACHE_FLASH_TRITON is None:
        import os

        from vllm.platforms import current_platform

        capability = current_platform.get_device_capability() if current_platform.is_cuda() else None
        _MOET_CACHE_FLASH_TRITON = (
            os.environ.get("VLLM_MOET_CACHE_FLASH_C", "0") != "1" and capability is not None and capability.major == 12
        )
    return _MOET_CACHE_FLASH_TRITON


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    if _moet_cache_flash_triton():
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
        )

        triton_reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale
        )
        return
    torch.ops._C_cache_ops.reshape_and_cache_flash(
'''

EDITS = ((HELPER_ANCHOR, HELPER),)


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
            raise SystemExit(f"{REL}: anchor not found exactly once (another vLLM version?):\\n{anchor}")
        text = text.replace(anchor, new)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
