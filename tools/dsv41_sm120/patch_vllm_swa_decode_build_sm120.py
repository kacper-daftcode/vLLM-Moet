#!/usr/bin/env python3
"""Step 32: each SWA metadata builder's decode build in one launch instead of three (bit-identical).

vllm/v1/attention/backends/mla/sparse_swa.py: DeepseekSparseSWAMetadataBuilder.build() runs, for every KV-cache group
of the target model and every decode step outside the CUDA graphs, torch.ge(slot_mapping, 0, out=is_valid_token), a fill
of decode_swa_lens[num_decode_tokens:] = 0 and _compute_swa_indices_and_lens_kernel (one program per decode token) -
DeepSeek-V4.1-Flash has eight such builders: 24 launches of 1-2 us each with launch gaps between them, ~30 us of kernel
time per step. `_moet_swa_decode_build_kernel` does the three in one launch: its decode programs compute the token's
validity from the slot mapping (the same comparison), store it, and run the original kernel's row body unchanged (window
start / end, replay bound, block table lookups, -1 padding; invalid rows cleared as before); tail programs store the
validity of the tokens past the decode rows and the zeros of the lens past them. Every byte the three operations wrote is
written with the same value.

Applies to the causal decode build with image visibility off (the served DeepSeek-V4.1 decode step); the non-causal
(DSpark draft) build, prefill rows and the image variant run the original code. Runtime switch
VLLM_MOET_SWA_DECODE_BUILD=1 (default off).

Usage: python3 patch_vllm_swa_decode_build_sm120.py [--path PATH] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py")
MARKER = "[vllm-moet] step 32 SWA decode build in one launch"

SITE_OLD = (
    "        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]\n"
    "        torch.ge(slot_mapping, 0, out=is_valid_token)\n"
    "        if replay_start is None:\n"
    "            # Graph captures build without it; their dummy batches replay nothing.\n"
    "            replay_start = self.no_replay_start\n"
    "\n"
    "        non_causal = not common_attn_metadata.causal\n"
    "        decode_swa_width = (\n"
    "            self.noncausal_index_width if non_causal else self.window_size\n"
    "        )\n"
    "        decode_swa_indices = self.decode_swa_indices\n"
    "        if num_decode_tokens > 0:\n"
    "            self.decode_swa_lens[num_decode_tokens:] = 0\n"
    "            if non_causal:\n"
)
SITE_NEW = (
    "        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]\n"
    "        if replay_start is None:\n"
    "            # Graph captures build without it; their dummy batches replay nothing.\n"
    "            replay_start = self.no_replay_start\n"
    "\n"
    "        non_causal = not common_attn_metadata.causal\n"
    "        decode_swa_width = (\n"
    "            self.noncausal_index_width if non_causal else self.window_size\n"
    "        )\n"
    "        decode_swa_indices = self.decode_swa_indices\n"
    f"        _moet_fused = _moet_swa_decode_build(  # {MARKER}\n"
    "            self, slot_mapping, is_valid_token, num_decode_tokens, non_causal, query_start_loc, seq_lens,\n"
    "            token_to_req_indices, block_table, replay_start,\n"
    "        )\n"
    "        if not _moet_fused:\n"
    "            torch.ge(slot_mapping, 0, out=is_valid_token)\n"
    "        if num_decode_tokens > 0:\n"
    "            if not _moet_fused:\n"
    "                self.decode_swa_lens[num_decode_tokens:] = 0\n"
    "            if _moet_fused:\n"
    "                pass\n"
    "            elif non_causal:\n"
)
CLASS_ANCHOR = "class ComputeSWAIndicesAndLensKernel(\n"
CODE = f'''# {MARKER} (tools/dsv41_sm120/patch_vllm_swa_decode_build_sm120.py): the validity
# comparison, the lens tail fill and _compute_swa_indices_and_lens_kernel of a decode build in one launch; the row body is
# the original's. VLLM_MOET_SWA_DECODE_BUILD=1 enables it.
_MOET_SWA_BUILD_ON = None


def _moet_swa_decode_build_enabled() -> bool:
    global _MOET_SWA_BUILD_ON
    if _MOET_SWA_BUILD_ON is None:
        import os

        _MOET_SWA_BUILD_ON = os.environ.get("VLLM_MOET_SWA_DECODE_BUILD", "0") == "1"
    return _MOET_SWA_BUILD_ON


@triton.jit(
    do_not_specialize=[
        "swa_indices_stride",
        "block_table_stride",
        "num_decode_tokens",
        "num_slot_tokens",
        "num_lens",
    ]
)
def _moet_swa_decode_build_kernel(
    slot_mapping_ptr,
    is_valid_token_ptr,
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    index_width,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    replay_start_ptr,
    num_decode_tokens,
    num_slot_tokens,
    num_lens,
    TILE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_decode_tokens:
        # Tails: the validity of the tokens past the decode rows (torch.ge over the whole slot mapping) and the zeros
        # of decode_swa_lens[num_decode_tokens:].
        offs = num_decode_tokens + (pid - num_decode_tokens) * TILE + tl.arange(0, TILE)
        slots = tl.load(slot_mapping_ptr + offs, mask=offs < num_slot_tokens, other=-1)
        tl.store(is_valid_token_ptr + offs, slots >= 0, mask=offs < num_slot_tokens)
        tl.store(swa_lens_ptr + offs, 0, mask=offs < num_lens)
        return
    token_idx = pid
    is_valid = tl.load(slot_mapping_ptr + token_idx) >= 0
    tl.store(is_valid_token_ptr + token_idx, is_valid)
    if not is_valid:
        tl.store(swa_lens_ptr + pid, 0)
        # Clear the row so a padded token cannot gather through stale indices.
        for i in range(0, index_width, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(
                swa_indices_ptr + pid * swa_indices_stride + offset,
                -1,
                mask=offset < index_width,
            )
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    left = 0
    right = 0
    left_add = tl.maximum(left - (window_size - 1), 0)
    start_pos = tl.maximum(pos - (window_size - 1) - left_add, 0)
    # SWA bounded replay: no window KV exists below the request's replay start.
    start_pos = tl.maximum(start_pos, tl.load(replay_start_ptr + req_idx))
    end_pos = pos + right + 1

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, index_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + pid * swa_indices_stride + offset,
            slot_ids,
            mask=offset < index_width,
        )


def _moet_swa_decode_build(
    builder, slot_mapping, is_valid_token, num_decode_tokens, non_causal, query_start_loc, seq_lens,
    token_to_req_indices, block_table, replay_start,
) -> bool:
    """The causal decode build's three operations in one launch; False = not applicable (the caller runs them)."""
    if not _moet_swa_decode_build_enabled() or non_causal or num_decode_tokens <= 0:
        return False
    swa_indices = builder.decode_swa_indices
    swa_lens = builder.decode_swa_lens
    index_width = swa_indices.shape[-1]
    if (
        not slot_mapping.is_contiguous()
        or not is_valid_token.is_contiguous()
        or is_valid_token.dtype != torch.bool
        or swa_indices.stride(-1) != 1
        or not swa_lens.is_contiguous()
        or swa_lens.dtype != torch.int32
        or query_start_loc.dtype != torch.int32
        or seq_lens.dtype != torch.int32
        or token_to_req_indices.dtype != torch.int32
        or replay_start.dtype != torch.int32
        or block_table.dtype != torch.int32
        or block_table.stride(-1) != 1
    ):
        return False
    num_slot_tokens = slot_mapping.shape[0]
    num_lens = swa_lens.shape[0]
    if is_valid_token.shape[0] < num_slot_tokens or num_slot_tokens < num_decode_tokens or num_lens < num_decode_tokens:
        return False
    tail = max(num_slot_tokens, num_lens) - num_decode_tokens
    tail_programs = (tail + 1023) // 1024
    _moet_swa_decode_build_kernel[(num_decode_tokens + tail_programs,)](
        slot_mapping,
        is_valid_token,
        swa_indices,
        swa_indices.stride(0),
        swa_lens,
        builder.window_size,
        index_width,
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        builder.block_size,
        replay_start,
        num_decode_tokens,
        num_slot_tokens,
        num_lens,
        TILE=1024,
        TRITON_BLOCK_SIZE=1024,
    )
    return True


'''


def patch(path: Path, check: bool) -> int:
    text = path.read_text()
    if MARKER in text:
        print(f"already patched: {path}")
        return 0
    if check:
        print(f"NOT patched: {path}")
        return 1
    if text.count(SITE_OLD) != 1:
        sys.exit(f"build() site not found exactly once in {path} (found {text.count(SITE_OLD)})")
    if text.count(CLASS_ANCHOR) != 1:
        sys.exit(f"'{CLASS_ANCHOR.strip()}' not found exactly once in {path}")
    text = text.replace(SITE_OLD, SITE_NEW).replace(CLASS_ANCHOR, CODE + CLASS_ANCHOR)
    path.write_text(text)
    print(f"patched: {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", type=Path, default=TARGET)
    ap.add_argument("--check", action="store_true", help="exit 1 if the patch is not applied")
    a = ap.parse_args()
    return patch(a.path, a.check)


if __name__ == "__main__":
    sys.exit(main())
