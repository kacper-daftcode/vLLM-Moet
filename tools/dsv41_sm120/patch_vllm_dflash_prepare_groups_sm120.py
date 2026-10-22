#!/usr/bin/env python3
"""Step 30: the DFlash/DSpark drafter's input preparation in one launch for all draft KV-cache groups (bit-identical).

vllm/v1/worker/gpu/spec_decode/dflash/speculator.py runs `_prepare_dflash_inputs_kernel` once per draft KV-cache group
(DeepSeek-V4.1-Flash's DSpark drafter has three: 3 x ~9.4 us per decode step outside the CUDA graphs, one program each,
a chain of dependent loads). The three launches compute the same input_ids / positions / query_start_loc / seq_lens /
sample indices / sampling state and differ only in the group's block table, block size and the two slot-mapping outputs.
`_moet_prepare_dflash_inputs_groups_kernel` is the same kernel with the per-group part (context and query slot
mappings, their padding) instantiated for up to three groups: the shared outputs are written once (the same values the
last of the three launches used to write), the per-group outputs by the same expressions on the group's own table.
Every arithmetic expression, type and clamp is the original's, so every output buffer holds the same bits.

Runtime switch VLLM_MOET_DFLASH_PREP_GROUPS=1 (default off: the original loop). More than three groups, or a CP/DCP
configuration whose block tables differ in anything but table / block size, fall back to the loop.

Usage: python3 patch_vllm_dflash_prepare_groups_sm120.py [--path PATH] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py")
MARKER = "[vllm-moet] step 30 drafter input prep for all KV groups in one launch"

CALL_OLD = (
    "        # Support multiple draft KV cache groups by preparing inputs once for each\n"
    "        for i, gid in enumerate(self.draft_kv_cache_group_ids):\n"
    "            prepare_dflash_inputs(\n"
)
CALL_NEW = (
    "        # Support multiple draft KV cache groups by preparing inputs once for each\n"
    f"        _moet_groups_done = _moet_prepare_dflash_inputs_groups(  # {MARKER}\n"
    "            self, input_batch, num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds, num_reqs\n"
    "        )\n"
    "        for i, gid in enumerate(self.draft_kv_cache_group_ids):\n"
    f"            if _moet_groups_done:  # {MARKER}\n"
    "                break\n"
    "            prepare_dflash_inputs(\n"
)

DEF_ANCHOR = "def prepare_dflash_inputs(\n"

CODE = f'''# {MARKER}
# (tools/dsv41_sm120/patch_vllm_dflash_prepare_groups_sm120.py). The kernel below is _prepare_dflash_inputs_kernel with
# the per-group part - the context / query slot mappings and their padding - instantiated for up to three KV-cache
# groups; the shared outputs are computed by the same expressions once. VLLM_MOET_DFLASH_PREP_GROUPS=1 enables it.
_MOET_PREP_GROUPS_ON = None


def _moet_prep_groups_enabled() -> bool:
    global _MOET_PREP_GROUPS_ON
    if _MOET_PREP_GROUPS_ON is None:
        import os

        _MOET_PREP_GROUPS_ON = os.environ.get("VLLM_MOET_DFLASH_PREP_GROUPS", "0") == "1"
    return _MOET_PREP_GROUPS_ON


@triton.jit
def _moet_dflash_group_slots(
    out_query_slot_mapping_ptr,
    out_context_slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    req_idx,
    ctx_start,
    j,
    is_ctx,
    is_valid_ctx,
    ctx_pos,
    is_query,
    query_pos,
    query_idx,
    cp_rank,
    PAD_SLOT_ID: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
):
    # The context and query slot mappings of one KV-cache group: the original kernel's expressions, unchanged.
    ctx_block_num = ctx_pos // (block_size * CP_SIZE)
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_valid_ctx,
        other=0,
    ).to(tl.int64)
    ctx_resident = is_valid_ctx & (ctx_block_id != 0)
    local_ctx_slot = cp_local_slot(
        ctx_pos, ctx_block_id, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_SLOT_ID
    )
    ctx_slot = tl.where(
        ctx_resident,
        local_ctx_slot,
        PAD_SLOT_ID,
    )
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    q_block_num = query_pos // (block_size * CP_SIZE)
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    q_resident = is_query & (q_block_id != 0)
    local_q_slot = cp_local_slot(
        query_pos,
        q_block_id,
        block_size,
        cp_rank,
        CP_SIZE,
        CP_INTERLEAVE,
        PAD_SLOT_ID,
    )
    q_slot = tl.where(
        q_resident,
        local_q_slot,
        PAD_SLOT_ID,
    )
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)


@triton.jit
def _moet_prepare_dflash_inputs_groups_kernel(
    # Shared outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_is_padding_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_context_positions_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # Per-group outputs and block tables (groups beyond NUM_GROUPS repeat group 0's pointers and are not touched)
    out_query_slot_mapping_ptr0,
    out_context_slot_mapping_ptr0,
    block_table_ptr0,
    block_table_stride0,
    block_size0,
    out_query_slot_mapping_ptr1,
    out_context_slot_mapping_ptr1,
    block_table_ptr1,
    block_table_stride1,
    block_size1,
    out_query_slot_mapping_ptr2,
    out_context_slot_mapping_ptr2,
    block_table_ptr2,
    block_table_stride2,
    block_size2,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Sampling params
    temperature_ptr,
    seeds_ptr,
    # Scalars
    parallel_drafting_token_id,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    cp_rank,
    NUM_GROUPS: tl.constexpr,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    num_valid_ctx = valid_ctx_end - ctx_start

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_valid_ctx = j < num_valid_ctx
    is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)
    query_off = j - num_valid_ctx

    # --- Context positions (shared) ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)

    # --- Query positions / input_ids (shared) ---
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    tl.store(out_is_padding_ptr + query_idx, False, mask=is_query)
    clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
    tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)

    # --- Context / query slot mappings, per KV-cache group ---
    _moet_dflash_group_slots(
        out_query_slot_mapping_ptr0, out_context_slot_mapping_ptr0, block_table_ptr0, block_table_stride0, block_size0,
        req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, is_query, query_pos, query_idx, cp_rank,
        PAD_SLOT_ID, CP_SIZE, CP_INTERLEAVE,
    )
    if NUM_GROUPS > 1:
        _moet_dflash_group_slots(
            out_query_slot_mapping_ptr1, out_context_slot_mapping_ptr1, block_table_ptr1, block_table_stride1, block_size1,
            req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, is_query, query_pos, query_idx, cp_rank,
            PAD_SLOT_ID, CP_SIZE, CP_INTERLEAVE,
        )
    if NUM_GROUPS > 2:
        _moet_dflash_group_slots(
            out_query_slot_mapping_ptr2, out_context_slot_mapping_ptr2, block_table_ptr2, block_table_stride2, block_size2,
            req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, is_query, query_pos, query_idx, cp_rank,
            PAD_SLOT_ID, CP_SIZE, CP_INTERLEAVE,
        )

    # --- Sample indices / positions / idx_mapping (shared) ---
    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    is_sample = is_query & (query_off >= sample_off)
    sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
    sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len),
        )
        tl.store(
            out_temperature_ptr + req_state_idx,
            tl.load(temperature_ptr + req_state_idx),
        )
        tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))
        if req_idx == num_reqs - 1:
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_input_ids_ptr + block, 0, mask=mask)
                tl.store(out_query_positions_ptr + block, 0, mask=mask)
                tl.store(out_is_padding_ptr + block, True, mask=mask)
                tl.store(out_query_slot_mapping_ptr0 + block, PAD_SLOT_ID, mask=mask)
                if NUM_GROUPS > 1:
                    tl.store(out_query_slot_mapping_ptr1 + block, PAD_SLOT_ID, mask=mask)
                if NUM_GROUPS > 2:
                    tl.store(out_query_slot_mapping_ptr2 + block, PAD_SLOT_ID, mask=mask)


def _moet_prepare_dflash_inputs_groups(
    spec, input_batch, num_sampled, num_rejected, last_sampled, next_prefill_tokens, input_temperature, input_seeds,
    num_reqs,
) -> bool:
    """All draft KV-cache groups' inputs in one launch; False = not applicable (the caller runs the per-group loop)."""
    gids = spec.draft_kv_cache_group_ids
    if not _moet_prep_groups_enabled() or not 1 <= len(gids) <= 3:
        return False
    bt = spec.block_tables
    assert num_reqs <= input_batch.num_reqs
    assert num_reqs > 0
    max_target_query_len = int(input_batch.num_scheduled_tokens[:num_reqs].max())
    max_tokens_per_req = max_target_query_len + spec.num_query_per_req
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    groups = []
    for i, gid in enumerate(gids):
        table = bt.input_block_tables[gid]
        groups.append((bt.slot_mappings[gid], spec._context_slot_mappings[i], table, table.stride(0), bt.block_sizes[gid]))
    while len(groups) < 3:
        groups.append(groups[0])
    per_group = [a for g in groups for a in g]
    _moet_prepare_dflash_inputs_groups_kernel[(num_reqs, num_blocks)](
        spec.input_buffers.input_ids,
        spec.input_buffers.positions,
        spec.input_buffers.is_padding,
        spec.input_buffers.query_start_loc,
        spec.input_buffers.seq_lens,
        spec.context_positions,
        spec.sample_indices,
        spec.sample_pos,
        spec.sample_idx_mapping,
        spec.temperature,
        spec.seeds,
        *per_group,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        spec.parallel_drafting_token_id,
        spec.num_query_per_req,
        spec.num_speculative_steps,
        spec.max_num_reqs,
        spec.max_num_tokens,
        spec.max_model_len,
        bt.cp_rank,
        NUM_GROUPS=len(gids),
        SAMPLE_FROM_ANCHOR=spec.sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        CP_SIZE=spec.dcp_size,
        CP_INTERLEAVE=bt.cp_interleave,
        BLOCK_SIZE=BLOCK_SIZE,
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
    if text.count(CALL_OLD) != 1:
        sys.exit(f"call site not found exactly once in {path} (found {text.count(CALL_OLD)})")
    if text.count(DEF_ANCHOR) != 1:
        sys.exit(f"'{DEF_ANCHOR.strip()}' not found exactly once in {path}")
    text = text.replace(CALL_OLD, CALL_NEW).replace(DEF_ANCHOR, CODE + DEF_ANCHOR)
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
