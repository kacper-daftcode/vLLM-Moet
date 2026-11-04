#!/usr/bin/env python3
"""Bit test of step 30: _moet_prepare_dflash_inputs_groups_kernel (one launch, up to 3 KV-cache groups) against the
original _prepare_dflash_inputs_kernel launched once per group, on random inputs (run inside the image with the patched
speculator.py). Every output buffer, including the padded tails, must be identical; also times both on a 1-request step.
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402
import vllm.v1.worker.gpu.spec_decode.dflash.speculator as sp  # noqa: E402

assert hasattr(sp, "_moet_prepare_dflash_inputs_groups_kernel"), "patched speculator.py not in place"
dev = torch.device("cuda")
torch.manual_seed(0)


def run_case(num_reqs, num_groups, steps=7, max_num_reqs=16, max_num_tokens=512, max_model_len=4096, sample_from_anchor=True,
             cp_size=1, cp_interleave=1, cp_rank=0, timing=False):
    # DSpark (sample_from_anchor): num_query_per_req = num_speculative_steps; DFlash: 1 + steps (speculator.py).
    nq = steps if sample_from_anchor else steps + 1
    block_sizes = [64, 128, 32][:num_groups]
    tables = []
    for bs in block_sizes:
        t = torch.randint(1, 200, (max_num_reqs, 64), dtype=torch.int32, device=dev)
        t[torch.rand(t.shape, device=dev) < 0.1] = 0  # null blocks
        tables.append(t)
    # target batch: per request num_ctx in [1, 8], positions increasing per request
    num_ctx = torch.randint(1, 9, (num_reqs,))
    qsl = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    qsl[1:num_reqs + 1] = torch.cumsum(num_ctx, 0).to(torch.int32)
    qsl[num_reqs + 1:] = qsl[num_reqs]
    positions = torch.zeros(max_num_tokens, dtype=torch.int64)
    for r in range(num_reqs):
        base = int(torch.randint(0, 3000, (1,)))
        positions[int(qsl[r]):int(qsl[r + 1])] = torch.arange(base, base + int(num_ctx[r]))
    num_rejected = torch.stack([torch.randint(0, int(n), (1,))[0] for n in num_ctx]).to(torch.int32)
    num_sampled = torch.randint(0, 2, (num_reqs,), dtype=torch.int32)
    idx_mapping = torch.randperm(max_num_reqs)[:num_reqs].to(torch.int32)
    last_sampled = torch.randint(0, 100000, (max_num_reqs,), dtype=torch.int64)
    next_prefill = torch.randint(0, 100000, (max_num_reqs,), dtype=torch.int64)
    temperature = torch.rand(max_num_reqs, dtype=torch.float32)
    seeds = torch.randint(0, 2**62, (max_num_reqs,), dtype=torch.int64)
    ins = [x.to(dev) for x in (positions, qsl, idx_mapping, last_sampled, next_prefill, num_sampled, num_rejected, temperature, seeds)]
    positions, qsl, idx_mapping, last_sampled, next_prefill, num_sampled, num_rejected, temperature, seeds = ins

    def outs():
        o = dict(
            input_ids=torch.full((max_num_tokens,), -7, dtype=torch.int32, device=dev),
            qpos=torch.full((max_num_tokens,), -7, dtype=torch.int64, device=dev),
            is_padding=torch.zeros(max_num_tokens, dtype=torch.bool, device=dev),
            qsl=torch.full((max_num_reqs + 1,), -7, dtype=torch.int32, device=dev),
            seq_lens=torch.full((max_num_reqs,), -7, dtype=torch.int32, device=dev),
            ctx_pos=torch.full((max_num_tokens,), -7, dtype=torch.int64, device=dev),
            sample_indices=torch.full((max_num_reqs * steps,), -7, dtype=torch.int32, device=dev),
            sample_pos=torch.full((max_num_reqs * steps,), -7, dtype=torch.int64, device=dev),
            sample_idx_mapping=torch.full((max_num_reqs * steps,), -7, dtype=torch.int32, device=dev),
            temperature=torch.full((max_num_reqs,), -7.0, dtype=torch.float32, device=dev),
            seeds=torch.full((max_num_reqs,), -7, dtype=torch.int64, device=dev),
        )
        for g in range(num_groups):
            o[f"qslot{g}"] = torch.full((max_num_tokens,), -7, dtype=torch.int64, device=dev)
            o[f"cslot{g}"] = torch.full((max_num_tokens,), -7, dtype=torch.int64, device=dev)
        return o

    max_target_query_len = int(num_ctx.max())
    max_tokens_per_req = max_target_query_len + nq
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    common_scalars = dict(parallel_drafting_token_id=123456, num_query_per_req=nq, num_speculative_steps=steps,
                          max_num_reqs=max_num_reqs, max_num_tokens=max_num_tokens, max_model_len=max_model_len, cp_rank=cp_rank)

    def ref(o):
        for g in range(num_groups):
            sp._prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
                o["input_ids"], o["qpos"], o["is_padding"], o["qsl"], o["seq_lens"], o[f"qslot{g}"], o["ctx_pos"], o[f"cslot{g}"],
                o["sample_indices"], o["sample_pos"], o["sample_idx_mapping"], o["temperature"], o["seeds"],
                positions, qsl, idx_mapping, last_sampled, next_prefill, num_sampled, num_rejected, temperature, seeds,
                tables[g], tables[g].stride(0),
                common_scalars["parallel_drafting_token_id"], block_sizes[g], nq, steps, max_num_reqs, max_num_tokens, max_model_len, cp_rank,
                SAMPLE_FROM_ANCHOR=sample_from_anchor, PAD_SLOT_ID=PAD_SLOT_ID, CP_SIZE=cp_size, CP_INTERLEAVE=cp_interleave,
                BLOCK_SIZE=BLOCK_SIZE)

    def fused(o):
        groups = [(o[f"qslot{g}"], o[f"cslot{g}"], tables[g], tables[g].stride(0), block_sizes[g]) for g in range(num_groups)]
        while len(groups) < 3:
            groups.append(groups[0])
        per_group = [a for gg in groups for a in gg]
        sp._moet_prepare_dflash_inputs_groups_kernel[(num_reqs, num_blocks)](
            o["input_ids"], o["qpos"], o["is_padding"], o["qsl"], o["seq_lens"], o["ctx_pos"],
            o["sample_indices"], o["sample_pos"], o["sample_idx_mapping"], o["temperature"], o["seeds"],
            *per_group,
            positions, qsl, idx_mapping, last_sampled, next_prefill, num_sampled, num_rejected, temperature, seeds,
            common_scalars["parallel_drafting_token_id"], nq, steps, max_num_reqs, max_num_tokens, max_model_len, cp_rank,
            NUM_GROUPS=num_groups, SAMPLE_FROM_ANCHOR=sample_from_anchor, PAD_SLOT_ID=PAD_SLOT_ID, CP_SIZE=cp_size,
            CP_INTERLEAVE=cp_interleave, BLOCK_SIZE=BLOCK_SIZE)

    a, b = outs(), outs()
    ref(a)
    fused(b)
    torch.cuda.synchronize()
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    if timing:
        for name, fn in (("3x original", ref), ("fused", fused)):
            o = outs()
            for _ in range(5):
                fn(o)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                fn(o)
            torch.cuda.synchronize()
            print(f"   {name}: {(time.perf_counter() - t0) / 200 * 1e6:.1f} us per step (eager, incl. launch)")
    return bad


fails = 0
cases = 0
for num_groups in (3, 2, 1):
    for num_reqs in (1, 2, 5, 16):
        for sfa in (True, False):
            for rep in range(4):
                bad = run_case(num_reqs, num_groups, sample_from_anchor=sfa)
                cases += 1
                if bad:
                    fails += 1
                    print(f"MISMATCH groups={num_groups} reqs={num_reqs} sfa={sfa} rep={rep}: {bad}")
print(f"{cases - fails}/{cases} cases identical")
print("timing, 1 request, 3 groups:")
run_case(1, 3, timing=True)
sys.exit(1 if fails else 0)
