#!/usr/bin/env python3
"""Bit test of step 31 (tools/dsv41_sm120/patch_vllm_candidate_short_sm120.py): select_candidate_blocks on the
deterministic top-k against the original _block_scores_kernel -> torch.topk -> _store_candidates_kernel path, inside
the image with candidate_blocks.py patched. Random rows - short and long contexts (more valid blocks than the list
holds), values drawn from a few levels so that ties straddle the k-th place, -inf columns and whole -inf blocks inside
the valid range, NaN scores, the pinned newest block, bounds at 0 / 1 / exactly k x block_size, with and without row
starts, row_repeat 1 and 8, k > nblocks - must give the same SET of blocks from both paths and, the criterion that
matters, bit-identical masked logits from apply_candidate_mask. Then the served decode shape (short and long context)
is timed, eagerly and in a CUDA graph.  usage: python3 test_candidate_short_sm120.py [--cases N]
"""
import argparse
import os
import sys
import time

import torch

os.environ.setdefault("VLLM_MOET_CANDIDATE_DET_TOPK", "1")
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
import vllm.model_executor.kernels.attention.dsa.candidate_blocks as cb  # noqa: E402

assert hasattr(cb, "_moet_candidates_det"), "patched candidate_blocks.py not in place"
dev = torch.device("cuda")
MOD = cb._moet_candidate_det_topk()
assert MOD is not None, "det_topk module not resolved (VLLM_MOET_DET_TOPK_DIR?)"


def make_case(rows_b, row_repeat, width, block_size, topk_blocks, with_starts, max_cols, levels, gen):
    rows = rows_b * row_repeat
    ends = torch.randint(0, max_cols + 1, (rows_b,), generator=gen).to(torch.int32)
    ends[0] = max_cols
    if rows_b > 1:
        ends[1] = 0
    if rows_b > 2:
        ends[2] = 1
    starts = None
    if with_starts:
        starts = (ends * torch.rand(rows_b, generator=gen) * 0.5).to(torch.int32)
        starts = torch.minimum(starts, ends)
    if levels:
        # few distinct values: ties everywhere, including at the k-th place
        logits = torch.randint(0, levels, (rows, width), generator=gen).to(torch.float32)
    else:
        logits = torch.rand((rows, width), generator=gen) * 4.0
        logits[torch.rand((rows, width), generator=gen) < 0.3] = 1.0
    logits[torch.rand((rows, width), generator=gen) < 0.02] = -float("inf")
    for r in range(rows):
        e = int(ends[r // row_repeat])
        if r % 3 == 0 and e > 16:
            logits[r, e - 16 : e - 8] = -float("inf")  # a whole valid block of -inf
        if r % 5 == 0 and e > 40:
            logits[r, e - 33] = float("nan")
        if r % 7 == 0 and e > 100:
            logits[r, e - 90] = -0.0
        logits[r, e:] = -float("inf")
    return logits.to(dev), None if starts is None else starts.to(dev), ends.to(dev), rows


def select(logits, starts, ends, topk_blocks, block_size, row_repeat, on):
    out = torch.full((logits.shape[0], topk_blocks), -9, dtype=torch.int32, device=dev)
    cb._MOET_CANDIDATE_DET = MOD if on else False
    cb.select_candidate_blocks(logits, starts, ends, topk_blocks, block_size, out, row_repeat)
    torch.cuda.synchronize()
    return out


def same_sets(a, b):
    for r in range(a.shape[0]):
        sa = set(a[r][a[r] >= 0].tolist())
        sb = set(b[r][b[r] >= 0].tolist())
        if sa != sb:
            return False, r, len(sa), len(sb), sorted(sa ^ sb)[:8]
    return True, None, None, None, None


def masked(logits, starts, ends, cand, block_size, row_repeat):
    x = logits.clone()
    cb.apply_candidate_mask(x, starts, ends, cand, block_size, row_repeat)
    torch.cuda.synchronize()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=72)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(1234)
    fails, cases = 0, 0
    configs = [
        # rows_b, row_repeat, width, block_size, topk_blocks, with_starts, max_cols, levels
        (1, 8, 65536, 8, 2048, False, 2673, 0),  # the served shape, short context
        (1, 8, 65536, 8, 2048, False, 65536, 0),  # long context: 8,192 valid blocks for 2,048 places
        (1, 8, 65536, 8, 2048, False, 65536, 3),  # long context, three value levels: ties at the k-th place
        (2, 8, 65536, 8, 2048, False, 40000, 5),
        (6, 1, 65536, 8, 2048, False, 16384, 0),  # bounds at exactly k x block_size
        (3, 8, 4096, 8, 256, False, 4096, 2),
        (5, 1, 4096, 8, 64, True, 4096, 4),
        (4, 1, 1024, 16, 32, True, 1024, 0),
        (16, 1, 2048, 8, 128, False, 2048, 3),
        (1, 1, 300, 8, 2048, False, 300, 0),  # k > nblocks (the original takes min(k, nblocks))
        (8, 1, 32768, 8, 2048, False, 32768, 1),  # one level: every valid block ties
        (2, 4, 8192, 8, 512, True, 8192, 0),
    ]
    for i in range(a.cases):
        rows_b, row_repeat, width, block_size, topk_blocks, with_starts, max_cols, levels = configs[i % len(configs)]
        logits, starts, ends, rows = make_case(rows_b, row_repeat, width, block_size, topk_blocks, with_starts, max_cols, levels, gen)
        out_ref = select(logits, starts, ends, topk_blocks, block_size, row_repeat, on=False)
        out_new = select(logits, starts, ends, topk_blocks, block_size, row_repeat, on=True)
        cases += 1
        if (out_new == -9).any() or ((out_new < -1) | (out_new >= width)).any():
            fails += 1
            print(f"case {i} {configs[i % len(configs)]}: unwritten or out-of-range slots in the new list")
            continue
        ok, r, na, nb, diff = same_sets(out_ref, out_new)
        if not ok:
            fails += 1
            print(f"case {i} {configs[i % len(configs)]}: row {r} sets differ (ref {na} vs new {nb} blocks; sym diff {diff})")
            continue
        m_ref = masked(logits, starts, ends, out_ref, block_size, row_repeat)
        m_new = masked(logits, starts, ends, out_new, block_size, row_repeat)
        if not torch.equal(m_ref.view(torch.int32), m_new.view(torch.int32)):
            fails += 1
            print(f"case {i} {configs[i % len(configs)]}: masked logits differ")
    print(f"{cases - fails}/{cases} cases identical (block sets and masked logits)")

    # timing at the served shape: 8 rows (1 request x next_n 8), static width 65,536; short and long contexts
    for ctx in (2673, 60000):
        logits = torch.rand((8, 65536), device=dev)
        logits[:, ctx:] = -float("inf")
        ends = torch.full((1,), ctx, dtype=torch.int32, device=dev)
        out = torch.empty((8, 2048), dtype=torch.int32, device=dev)
        for name, on in (("original (torch.topk path)", False), ("step 31 (det top-k)", True)):
            cb._MOET_CANDIDATE_DET = MOD if on else False
            fn = lambda: cb.select_candidate_blocks(logits, None, ends, 2048, 8, out, 8)  # noqa: E731
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                fn()
            torch.cuda.synchronize()
            eager = (time.perf_counter() - t0) / 200 * 1e6
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                fn()
                torch.cuda.synchronize()
                with torch.cuda.graph(g, stream=s):
                    for _ in range(20):
                        fn()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                g.replay()
            torch.cuda.synchronize()
            graph = (time.perf_counter() - t0) / 50 / 20 * 1e6
            print(f"   context {ctx}: {name}: eager {eager:.1f} us, in a CUDA graph {graph:.1f} us per selection")
    cb._MOET_CANDIDATE_DET = None
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
