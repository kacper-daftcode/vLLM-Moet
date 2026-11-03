#!/usr/bin/env python3
"""Bit test of step 36 (tools/dsv41_sm120/patch_vllm_swa_draft_build_sm120.py): _moet_swa_draft_build_kernel against the
three operations of DeepseekSparseSWAMetadataBuilder.update_draft_decode_metadata() it replaces on the DSpark path -
torch.ge(slot_mapping, 0, out=is_valid_token), compute_token_to_req_indices and the non-causal indices kernel
(ComputeDSparkNoncausalSWAIndicesKernel.kernel) - on random draft batches (1-8 requests x k draft tokens, padded slot
mappings and query_start_loc buffers, invalid rows, several window / block sizes, short and long prefixes), inside the
image with sparse_swa.py patched. The validity buffer, the token -> request map, the lens and the index rows must be
identical byte for byte; then both are timed for the served step (three builders, 7 draft tokens), eager and in a
CUDA graph (where the update runs in production).
usage: python3 test_swa_draft_build_sm120.py [--cases N]
"""
import argparse
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
import vllm.v1.attention.backends.mla.sparse_swa as sw  # noqa: E402
from vllm.v1.attention.backends.mla.compressor_utils import get_dspark_swa_index_width  # noqa: E402
from vllm.v1.attention.ops.metadata import compute_token_to_req_indices  # noqa: E402

assert hasattr(sw, "_moet_swa_draft_build_kernel"), "patched sparse_swa.py not in place"
dev = torch.device("cuda")
NONCAUSAL = sw.ComputeDSparkNoncausalSWAIndicesKernel.kernel


def make_case(gen, num_reqs, k, window, block_size, max_tokens, pad, qsl_pad, long_prefix, max_blocks=96):
    num_tokens = num_reqs * k
    num_slot = num_tokens + pad
    assert num_slot <= max_tokens
    qsl = torch.arange(0, num_tokens + 1, k, dtype=torch.int32)[: num_reqs + 1]
    if qsl_pad:  # a padded query_start_loc buffer repeats the last boundary (num_reqs counts the padding, as in vLLM)
        qsl = torch.cat([qsl, torch.full((qsl_pad,), int(qsl[-1]), dtype=torch.int32)])
    hi = 6000 if long_prefix else 400
    seq_lens = (torch.randint(0, hi, (num_reqs,), generator=gen) + k).to(torch.int32)
    seq_lens = torch.cat([seq_lens, torch.zeros(max_tokens - num_reqs, dtype=torch.int32)])
    block_table = torch.randint(0, 4000, (max_tokens, max_blocks), generator=gen).to(torch.int32)
    slot_mapping = torch.randint(0, 1 << 20, (num_slot,), generator=gen).to(torch.int64)
    slot_mapping[num_tokens:] = -1  # padded tail
    inval = torch.rand(num_tokens, generator=gen) < 0.15
    slot_mapping[:num_tokens][inval] = -1  # invalid draft rows
    return dict(
        num_tokens=num_tokens, num_slot=num_slot, window=window, block_size=block_size, max_tokens=max_tokens,
        index_width=get_dspark_swa_index_width(window, k), qsl=qsl.to(dev), seq_lens=seq_lens.to(dev),
        block_table=block_table.to(dev), slot_mapping=slot_mapping.to(dev),
    )


def bufs(c):
    return dict(
        is_valid=torch.ones(c["num_slot"], dtype=torch.bool, device=dev),
        t2r=torch.full((c["max_tokens"],), -7, dtype=torch.int32, device=dev),
        lens=torch.full((c["max_tokens"],), -7, dtype=torch.int32, device=dev),
        idx=torch.full((c["max_tokens"], 1, c["index_width"]), -7, dtype=torch.int32, device=dev),
    )


def original(c, b):
    torch.ge(c["slot_mapping"], 0, out=b["is_valid"])
    compute_token_to_req_indices(c["qsl"], b["t2r"], c["num_tokens"], c["num_tokens"])
    NONCAUSAL[(c["num_tokens"],)](
        b["idx"], b["idx"].stride(0), b["lens"], c["window"], c["index_width"], c["qsl"], c["seq_lens"], b["t2r"],
        b["is_valid"], c["block_table"], c["block_table"].stride(0), c["block_size"], 0, TRITON_BLOCK_SIZE=1024,
    )


def fused(c, b):
    builder = SimpleNamespace(noncausal_index_width=c["index_width"], window_size=c["window"], block_size=c["block_size"])
    metadata = SimpleNamespace(
        slot_mapping=c["slot_mapping"], is_valid_token=b["is_valid"], token_to_req_indices=b["t2r"],
        decode_swa_indices=b["idx"], decode_swa_lens=b["lens"], query_start_loc=c["qsl"], seq_lens=c["seq_lens"],
        block_table=c["block_table"],
    )
    assert sw._moet_swa_draft_build(builder, metadata, c["num_tokens"]), "fused build refused"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=96)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(11)
    sw._MOET_SWA_DRAFT_ON = True
    fails = 0
    configs = [
        # num_reqs, k, window, block_size, max_tokens, pad, qsl_pad, long_prefix
        (1, 7, 128, 64, 512, 0, 0, False),  # the served C1 step
        (1, 7, 128, 64, 512, 0, 0, True),
        (1, 7, 128, 64, 512, 9, 3, False),
        (2, 7, 128, 64, 512, 2, 0, True),
        (6, 7, 128, 64, 1024, 22, 5, False),
        (8, 5, 128, 16, 256, 0, 0, True),
        (3, 7, 64, 64, 512, 0, 1, False),
        (4, 7, 256, 64, 2048, 36, 0, True),
        (1, 1, 128, 64, 64, 0, 0, False),
        (5, 3, 128, 64, 1100, 1085, 0, True),  # a long padded slot mapping (two tail tiles)
    ]
    for i in range(a.cases):
        cfg = configs[i % len(configs)]
        c = make_case(gen, *cfg)
        b0, b1 = bufs(c), bufs(c)
        original(c, b0)
        fused(c, b1)
        torch.cuda.synchronize()
        bad = [k for k in b0 if not torch.equal(b0[k], b1[k])]
        if bad:
            fails += 1
            print(f"case {i} {cfg}: MISMATCH in {bad}")
    print(f"{a.cases - fails}/{a.cases} cases identical (validity, token -> request, lens and index buffers byte for byte)")
    # timing: three builders per step at the served shape (one request, 7 draft tokens, slot mapping of 7)
    cs = [make_case(gen, 1, 7, 128, 64, 512, 0, 0, False) for _ in range(3)]
    bs = [bufs(c) for c in cs]
    for name, fn in (("original (3 ops x 3 builders)", original), ("step 36 (1 launch x 3 builders)", fused)):
        for _ in range(5):
            for c, b in zip(cs, bs):
                fn(c, b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            for c, b in zip(cs, bs):
                fn(c, b)
        torch.cuda.synchronize()
        eager = (time.perf_counter() - t0) / 100 * 1e6
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for c, b in zip(cs, bs):
                fn(c, b)
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                for c, b in zip(cs, bs):
                    fn(c, b)
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(500):
            g.replay()
        torch.cuda.synchronize()
        graph = (time.perf_counter() - t0) / 500 * 1e6
        print(f"   {name}: eager {eager:.1f} us per step (incl. launches), CUDA graph replay {graph:.2f} us per step")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
