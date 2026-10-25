#!/usr/bin/env python3
"""Step 36: the DSpark drafter's non-causal SWA metadata update in one launch instead of three (bit-identical).

vllm/v1/attention/backends/mla/sparse_swa.py: DeepseekSparseSWAMetadataBuilder.update_draft_decode_metadata() runs inside
the drafter's CUDA graph, once per SWA builder of the drafter (three in DeepSeek-V4.1-Flash), torch.ge(slot_mapping, 0,
out=is_valid_token), compute_token_to_req_indices (the binary search of _token_request_mapping_kernel) and
_COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL (one program per draft token): nine launches of 0.9-1.6 us per step on the
served C1 step (prod r18 profile: compare 0.89, token map 0.89, indices 1.57 us), each with its launch gap in the graph.
`_moet_swa_draft_build_kernel` does the three in one launch: its token programs compute the token's validity from the
slot mapping (the same comparison), its request index by the same binary search over query_start_loc (scalar, the same
sequence of probes and the same min(lo, num_reqs - 1)), store both, and run the non-causal kernel's row body unchanged
(block-anchored window start, end = seq_len, block table lookups, -1 padding; invalid rows cleared as before); tail
programs store the validity of the slot-mapping entries past the draft tokens. Every byte the three operations wrote is
written with the same value; the causal branch of the update (non-DSpark drafters) runs the original code.

Runtime switch VLLM_MOET_SWA_DRAFT_BUILD=1 (default off). Applies on top of step 32 (which leaves this method alone).

Usage: python3 patch_vllm_swa_draft_build_sm120.py [--path PATH] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py")
MARKER = "[vllm-moet] step 36 SWA draft build in one launch"

SITE_OLD = (
    "        # Recompute the validity mask and token->request map from the device\n"
    "        # buffers. Their padding differs from the dummy batch the graph was\n"
    "        # captured with.\n"
    "        torch.ge(metadata.slot_mapping, 0, out=metadata.is_valid_token)\n"
    "        compute_token_to_req_indices(\n"
    "            metadata.query_start_loc,\n"
    "            metadata.token_to_req_indices,\n"
    "            num_tokens,\n"
    "            num_tokens,\n"
    "        )\n"
    "\n"
    "        # build() points non-causal (DSpark draft) metadata at its own buffer.\n"
    "        noncausal = self.decode_swa_indices_noncausal\n"
    "        if (\n"
    "            noncausal is not None\n"
    "            and metadata.decode_swa_indices.data_ptr() == noncausal.data_ptr()\n"
    "        ):\n"
    "            _COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL(\n"
)
SITE_NEW = (
    "        # Recompute the validity mask and token->request map from the device\n"
    "        # buffers. Their padding differs from the dummy batch the graph was\n"
    "        # captured with.\n"
    "        # build() points non-causal (DSpark draft) metadata at its own buffer.\n"
    "        noncausal = self.decode_swa_indices_noncausal\n"
    "        _moet_noncausal = (\n"
    "            noncausal is not None\n"
    "            and metadata.decode_swa_indices.data_ptr() == noncausal.data_ptr()\n"
    "        )\n"
    f"        _moet_fused = _moet_noncausal and _moet_swa_draft_build(self, metadata, num_tokens)  # {MARKER}\n"
    "        if not _moet_fused:\n"
    "            torch.ge(metadata.slot_mapping, 0, out=metadata.is_valid_token)\n"
    "            compute_token_to_req_indices(\n"
    "                metadata.query_start_loc,\n"
    "                metadata.token_to_req_indices,\n"
    "                num_tokens,\n"
    "                num_tokens,\n"
    "            )\n"
    "        if _moet_fused:\n"
    "            pass\n"
    "        elif _moet_noncausal:\n"
    "            _COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL(\n"
)
CLASS_ANCHOR = "class ComputeSWAIndicesAndLensKernel(\n"
CODE = f'''# {MARKER} (tools/dsv41_sm120/patch_vllm_swa_draft_build_sm120.py): the validity
# comparison, the token -> request binary search and the non-causal indices kernel of the drafter's metadata update in
# one launch; the row body is the original's. VLLM_MOET_SWA_DRAFT_BUILD=1 enables it.
_MOET_SWA_DRAFT_ON = None


def _moet_swa_draft_build_enabled() -> bool:
    global _MOET_SWA_DRAFT_ON
    if _MOET_SWA_DRAFT_ON is None:
        import os

        _MOET_SWA_DRAFT_ON = os.environ.get("VLLM_MOET_SWA_DRAFT_BUILD", "0") == "1"
    return _MOET_SWA_DRAFT_ON


@triton.jit(
    do_not_specialize=[
        "swa_indices_stride",
        "block_table_stride",
        "num_reqs",
        "num_tokens",
        "num_slot_tokens",
    ]
)
def _moet_swa_draft_build_kernel(
    slot_mapping_ptr,
    is_valid_token_ptr,
    token_to_req_indices_ptr,
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    index_width,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    num_reqs,
    num_tokens,
    num_slot_tokens,
    TILE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        # Tails: the validity of the slot-mapping entries past the draft tokens (torch.ge over the whole slot mapping).
        offs = num_tokens + (pid - num_tokens) * TILE + tl.arange(0, TILE)
        slots = tl.load(slot_mapping_ptr + offs, mask=offs < num_slot_tokens, other=-1)
        tl.store(is_valid_token_ptr + offs, slots >= 0, mask=offs < num_slot_tokens)
        return
    token_idx = pid
    is_valid = tl.load(slot_mapping_ptr + token_idx) >= 0
    tl.store(is_valid_token_ptr + token_idx, is_valid)

    # The token -> request map: _token_request's binary search (vllm/v1/attention/ops/metadata.py), one lane.
    lo = tl.full([1], 0, tl.int32)
    hi = tl.full([1], num_reqs, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc_ptr + mid + 1, mid < num_reqs, other=0x7FFFFFFF)
        right = (lo < hi) & (end <= token_idx)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where((lo < hi) & ~right, mid, hi)
    req_idx = tl.sum(tl.minimum(lo, num_reqs - 1), 0)
    tl.store(token_to_req_indices_ptr + token_idx, req_idx)

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

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    # Block-anchored window (shared by every token in the block) + full block.
    start_pos = tl.maximum(prefix_len - window_size, 0)
    end_pos = seq_len

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


def _moet_swa_draft_build(builder, metadata, num_tokens) -> bool:
    """The non-causal draft update's three operations in one launch; False = not applicable (the caller runs them)."""
    if not _moet_swa_draft_build_enabled() or num_tokens <= 0:
        return False
    slot_mapping = metadata.slot_mapping
    is_valid_token = metadata.is_valid_token
    token_to_req_indices = metadata.token_to_req_indices
    swa_indices = metadata.decode_swa_indices
    swa_lens = metadata.decode_swa_lens
    query_start_loc = metadata.query_start_loc
    seq_lens = metadata.seq_lens
    block_table = metadata.block_table
    index_width = builder.noncausal_index_width
    if (
        slot_mapping.dim() != 1
        or not slot_mapping.is_contiguous()
        or is_valid_token.shape != slot_mapping.shape
        or not is_valid_token.is_contiguous()
        or is_valid_token.dtype != torch.bool
        or token_to_req_indices.dim() != 1
        or not token_to_req_indices.is_contiguous()
        or token_to_req_indices.dtype != torch.int32
        or swa_indices.dim() not in (2, 3)
        or (swa_indices.dim() == 3 and swa_indices.shape[1] != 1)  # the builder's [max_tokens, 1, width] buffer
        or swa_indices.stride(-1) != 1
        or swa_indices.dtype != torch.int32
        or swa_indices.shape[-1] < index_width
        or not swa_lens.is_contiguous()
        or swa_lens.dtype != torch.int32
        or query_start_loc.dim() != 1
        or not query_start_loc.is_contiguous()
        or query_start_loc.dtype != torch.int32
        or seq_lens.dtype != torch.int32
        or block_table.dim() != 2
        or block_table.dtype != torch.int32
        or block_table.stride(-1) != 1
    ):
        return False
    num_slot_tokens = slot_mapping.shape[0]
    num_reqs = query_start_loc.shape[0] - 1
    if (
        num_reqs < 1
        or num_slot_tokens < num_tokens
        or token_to_req_indices.shape[0] < num_tokens
        or swa_lens.shape[0] < num_tokens
        or swa_indices.shape[0] < num_tokens
    ):
        return False
    tail_programs = (num_slot_tokens - num_tokens + 1023) // 1024
    _moet_swa_draft_build_kernel[(num_tokens + tail_programs,)](
        slot_mapping,
        is_valid_token,
        token_to_req_indices,
        swa_indices,
        swa_indices.stride(0),
        swa_lens,
        builder.window_size,
        index_width,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        builder.block_size,
        num_reqs,
        num_tokens,
        num_slot_tokens,
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
        sys.exit(f"update_draft_decode_metadata() site not found exactly once in {path} (found {text.count(SITE_OLD)})")
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
