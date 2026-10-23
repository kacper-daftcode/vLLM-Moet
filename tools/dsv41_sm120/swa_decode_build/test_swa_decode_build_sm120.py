#!/usr/bin/env python3
"""Bit test of step 32 (tools/dsv41_sm120/patch_vllm_swa_decode_build_sm120.py): _moet_swa_decode_build_kernel against
the three operations of DeepseekSparseSWAMetadataBuilder.build() it replaces - torch.ge(slot_mapping, 0, out=is_valid),
decode_swa_lens[num_decode_tokens:] = 0 and _compute_swa_indices_and_lens_kernel - on random decode batches (1-8
requests x 1 / 8 rows, padded slot mappings and invalid rows, several window / block sizes, replay bounds), inside the
image with sparse_swa.py patched. The validity buffer, the lens buffer and the index rows must be identical byte for
byte; then both are timed for the served step (eight builders, 8 decode tokens).
usage: python3 test_swa_decode_build_sm120.py [--cases N]
"""
import argparse
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
import vllm.v1.attention.backends.mla.sparse_swa as sw  # noqa: E402

assert hasattr(sw, "_moet_swa_decode_build_kernel"), "patched sparse_swa.py not in place"
dev = torch.device("cuda")


def make_case(gen, num_reqs, next_n, window, block_size, max_tokens, pad, max_blocks=64):
    num_decode = num_reqs * next_n
    num_slot = num_decode + pad
    assert num_slot <= max_tokens
    qsl = torch.arange(0, num_decode + 1, next_n, dtype=torch.int32)[: num_reqs + 1]
    qsl = torch.cat([qsl, torch.full((max_tokens + 1 - qsl.numel(),), int(qsl[-1]), dtype=torch.int32)])
    seq_lens = (torch.randint(0, 600, (num_reqs,), generator=gen) + next_n).to(torch.int32)
    seq_lens = torch.cat([seq_lens, torch.zeros(max_tokens - num_reqs, dtype=torch.int32)])
    replay = torch.zeros(max_tokens + 1, dtype=torch.int32)
    if int(torch.randint(0, 2, (1,), generator=gen)):
        replay[:num_reqs] = (torch.rand(num_reqs, generator=gen) * seq_lens[:num_reqs]).to(torch.int32)
    t2r = torch.zeros(max_tokens, dtype=torch.int32)
    for r in range(num_reqs):
        t2r[int(qsl[r]) : int(qsl[r + 1])] = r
    block_table = torch.randint(0, 4000, (max_tokens, max_blocks), generator=gen).to(torch.int32)
    slot_mapping = torch.randint(0, 1 << 20, (num_slot,), generator=gen).to(torch.int64)
    slot_mapping[num_decode:] = -1  # padded tail
    inval = torch.rand(num_decode, generator=gen) < 0.15
    slot_mapping[:num_decode][inval] = -1  # invalid decode rows
    return dict(
        num_decode=num_decode, num_slot=num_slot, window=window, block_size=block_size, max_tokens=max_tokens,
        qsl=qsl.to(dev), seq_lens=seq_lens.to(dev), replay=replay.to(dev), t2r=t2r.to(dev), block_table=block_table.to(dev),
        slot_mapping=slot_mapping.to(dev),
    )


def bufs(c):
    return dict(
        is_valid=torch.ones(c["max_tokens"], dtype=torch.bool, device=dev),
        lens=torch.full((c["max_tokens"],), -7, dtype=torch.int32, device=dev),
        idx=torch.full((c["max_tokens"], c["window"]), -7, dtype=torch.int32, device=dev),
    )


def original(c, b):
    is_valid = b["is_valid"][: c["num_slot"]]
    torch.ge(c["slot_mapping"], 0, out=is_valid)
    b["lens"][c["num_decode"] :] = 0
    sw._compute_swa_indices_and_lens_kernel[(c["num_decode"],)](
        b["idx"], b["idx"].stride(0), b["lens"], c["window"], c["window"], b["lens"], b["lens"],
        c["qsl"], c["seq_lens"], c["t2r"], is_valid, c["block_table"], c["block_table"].stride(0), c["block_size"],
        c["replay"], 0, HAS_IMAGE=False, TRITON_BLOCK_SIZE=1024,
    )


def fused(c, b):
    builder = SimpleNamespace(decode_swa_indices=b["idx"], decode_swa_lens=b["lens"], window_size=c["window"], block_size=c["block_size"])
    ok = sw._moet_swa_decode_build(
        builder, c["slot_mapping"], b["is_valid"][: c["num_slot"]], c["num_decode"], False, c["qsl"], c["seq_lens"],
        c["t2r"], c["block_table"], c["replay"],
    )
    assert ok, "fused build refused"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=96)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(7)
    sw._MOET_SWA_BUILD_ON = True
    fails = 0
    configs = [
        # num_reqs, next_n, window, block_size, max_tokens, pad
        (1, 8, 128, 64, 512, 0),  # the served C1 step
        (1, 8, 128, 64, 512, 8),
        (2, 8, 128, 64, 512, 16),
        (6, 8, 128, 64, 1024, 40),
        (8, 1, 128, 16, 256, 3),
        (3, 8, 64, 64, 512, 0),
        (4, 8, 256, 64, 2048, 24),
        (1, 1, 128, 64, 64, 0),
    ]
    for i in range(a.cases):
        c = make_case(gen, *configs[i % len(configs)])
        b0, b1 = bufs(c), bufs(c)
        original(c, b0)
        fused(c, b1)
        torch.cuda.synchronize()
        bad = [k for k in b0 if not torch.equal(b0[k], b1[k])]
        if bad:
            fails += 1
            print(f"case {i} {configs[i % len(configs)]}: MISMATCH in {bad}")
    print(f"{a.cases - fails}/{a.cases} cases identical (validity, lens and index buffers byte for byte)")
    # timing: eight builders per step at the served shape (8 decode tokens, lens buffer 8192)
    cs = [make_case(gen, 1, 8, 128, 64, 8192, 0) for _ in range(8)]
    bs = [bufs(c) for c in cs]
    for name, fn in (("original (3 ops x 8 builders)", original), ("step 32 (1 launch x 8 builders)", fused)):
        for _ in range(5):
            for c, b in zip(cs, bs):
                fn(c, b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            for c, b in zip(cs, bs):
                fn(c, b)
        torch.cuda.synchronize()
        print(f"   {name}: {(time.perf_counter() - t0) / 100 * 1e6:.1f} us per step (eager, incl. launches)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
