#!/usr/bin/env python3
"""Tests for det_topk_sm120: bit-exact against the torch reference, run-to-run determinism, CUDA-graph replay, speed.

Run inside the serving image on one sm_120 GPU:  python3 test_det_topk_sm120.py [--quick] [--bench]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_topk_sm120 as dt  # noqa: E402

dev = "cuda"


def make_logits(rows: int, width: int, kind: str, g: torch.Generator) -> torch.Tensor:
    if kind == "normal":
        x = torch.randn(rows, width, generator=g)
    elif kind == "bf16":  # what the indexer produces: bf16 values in fp32 (16 significant bits, ties in the top-k)
        x = torch.randn(rows, width, generator=g).bfloat16().float()
    elif kind == "ties":  # relu(q.k)-like: exact zeros and few distinct positive values
        x = torch.randint(1, 9, (rows, width), generator=g).float() * 0.25
        x[torch.rand(rows, width, generator=g) < 0.7] = 0.0
    elif kind == "zeros":
        x = torch.zeros(rows, width)
    elif kind == "masked":  # candidate-masked rows: -inf outside random blocks
        x = torch.randn(rows, width, generator=g)
        blocks = torch.rand(rows, (width + 7) // 8, generator=g) < 0.3
        x[~blocks.repeat_interleave(8, dim=1)[:, :width]] = float("-inf")
    elif kind == "special":
        x = torch.randn(rows, width, generator=g)
        m = torch.rand(rows, width, generator=g)
        x[m < 0.01] = float("nan")
        x[(m >= 0.01) & (m < 0.02)] = float("inf")
        x[(m >= 0.02) & (m < 0.03)] = float("-inf")
        x[(m >= 0.03) & (m < 0.05)] = -0.0
        x[(m >= 0.05) & (m < 0.07)] = 0.0
    else:
        raise ValueError(kind)
    return x.to(dev)


def check_prefill(rows, width, k, kind, starts_mode, relative, g, padded_width=0, cluster=-1, small=None):
    logits = make_logits(rows, width + padded_width, kind, g)
    if padded_width:  # a strided view like vLLM's sliced logits
        logits = logits[:, :width]
    ends = torch.randint(0, width + 1, (rows,), generator=g, dtype=torch.int32)
    ends[: min(rows, 4)] = torch.tensor([0, k - 1, k, k + 1][: min(rows, 4)], dtype=torch.int32).clamp(max=width)
    if starts_mode == "zero":
        starts = torch.zeros(rows, dtype=torch.int32)
    else:
        starts = (torch.rand(rows, generator=g) * ends.float() * 0.5).to(torch.int32)
    out = torch.full((rows, k + 3), -7, dtype=torch.int32, device=dev)
    dt.topk_prefill(logits, starts.to(dev) if starts_mode != "none" else None, ends.to(dev), out, rows, k, relative,
                    cluster=cluster, small=small)
    ref = dt.topk_reference(logits, starts if starts_mode != "none" else torch.zeros(rows, dtype=torch.int32), ends, k,
                            relative)
    ok = torch.equal(out[:, :k], ref) and bool((out[:, k:] == -7).all())
    return ok


def check_decode(bsz, next_n, width, k, kind, per_row, g, cluster=-1, small=None):
    rows = bsz * next_n
    logits = make_logits(rows, width, kind, g)
    if per_row:
        seq = torch.randint(0, width + 1, (bsz, next_n), generator=g, dtype=torch.int32)
    else:
        seq = torch.randint(0, width + 1, (bsz, 1), generator=g, dtype=torch.int32)
        seq[0, 0] = min(width, k + next_n - 2)
    seq = seq.to(dev)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=dev)
    dt.topk_decode(logits, seq, next_n, out, k, cluster=cluster, small=small)
    ends = dt.decode_row_ends(seq, next_n, rows).cpu()
    ref = dt.topk_reference(logits, torch.zeros(rows, dtype=torch.int32), ends, k, True)
    return torch.equal(out, ref)


def correctness(quick: bool) -> bool:
    g = torch.Generator().manual_seed(1234)
    fails = n = 0
    kinds = ("normal", "bf16", "ties", "zeros", "masked", "special")
    widths = (64, 600, 2048, 9000) if quick else (64, 600, 2048, 9000, 40000)
    for kind in kinds:
        for width in widths:
            for k in (512, 2048):
                for starts_mode, relative in (("zero", True), ("rand", True), ("rand", False), ("none", True)):
                    for rows in (7, 300):
                        n += 1
                        if not check_prefill(rows, width, k, kind, starts_mode, relative, g, padded_width=5 if rows == 7 else 0):
                            fails += 1
                            print(f"  FAIL prefill kind={kind} width={width} k={k} starts={starts_mode} rel={relative} rows={rows}")
                for next_n in (1, 2, 6):
                    for per_row in (False, True):
                        n += 1
                        if not check_decode(8 if next_n == 6 else 3, next_n, width, k, kind, per_row, g):
                            fails += 1
                            print(f"  FAIL decode kind={kind} width={width} k={k} next_n={next_n} per_row={per_row}")
                # every cluster size, with the cluster path forced on rows just above k (small=0)
                for cluster in (1, 2, 4, 8, 16):
                    n += 1
                    if not check_decode(1, 6, width, k, kind, True, g, cluster=cluster, small=0):
                        fails += 1
                        print(f"  FAIL decode cluster={cluster} kind={kind} width={width} k={k}")
                n += 1
                if not check_prefill(5, width, k, kind, "rand", True, g, cluster=8, small=0):
                    fails += 1
                    print(f"  FAIL prefill cluster=8 kind={kind} width={width} k={k}")
    for width in (100000, 300000, 524288):  # long decode rows (ratio-1 index layers at 100K-512K context)
        for cluster in (-1, 2, 4, 8, 16):
            n += 1
            if not check_decode(1, 6, width, 512, "ties", False, g, cluster=cluster):
                fails += 1
                print(f"  FAIL decode long width={width} cluster={cluster}")
    print(f"correctness: {n - fails}/{n} cases bit-exact vs the torch reference")
    return fails == 0


def determinism() -> bool:
    g = torch.Generator().manual_seed(7)
    logits = make_logits(2700, 2048, "ties", g)
    ends = torch.clamp(torch.arange(1, 2701, dtype=torch.int32) * 2048 // 2700, min=1).to(dev)
    outs = []
    for _ in range(30):
        o = torch.empty(2700, 512, dtype=torch.int32, device=dev)
        dt.topk_prefill(logits, None, ends, o, 2700, 512, True)
        outs.append(o)
    d1 = sum(not torch.equal(o, outs[0]) for o in outs[1:])
    d2 = 0
    for width in (32768, 100000):
        logits_d = make_logits(6, width, "ties", g)
        seq = torch.tensor([[width]], dtype=torch.int32, device=dev)
        outs = []
        for _ in range(30):
            o = torch.empty(6, 512, dtype=torch.int32, device=dev)
            dt.topk_decode(logits_d, seq, 6, o, 512)
            outs.append(o)
        d2 += sum(not torch.equal(o, outs[0]) for o in outs[1:])
    print(f"determinism: prefill 2700x2048 differing runs {d1}/29, decode 6x32768 + 6x100000 (cluster) differing runs {d2}/58")
    return d1 == 0 and d2 == 0


def graph_replay() -> bool:
    """One captured graph replayed at several context lengths (short rows -> one CTA, long rows -> the cluster)."""
    g = torch.Generator().manual_seed(9)
    logits = make_logits(6, 120000, "ties", g)
    seq = torch.tensor([[6000]], dtype=torch.int32, device=dev)
    out = torch.empty(6, 512, dtype=torch.int32, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dt.topk_decode(logits, seq, 6, out, 512)
    torch.cuda.current_stream().wait_stream(s)
    ok = True
    for ctx in (6000, 700, 3000, 50000, 120000, 6000):
        seq.fill_(ctx)
        ref = torch.empty_like(out)
        dt.topk_decode(logits, seq, 6, ref, 512)
        for _ in range(5):
            out.fill_(-9)
            graph.replay()
            torch.cuda.synchronize()
            ok &= torch.equal(out, ref)
    print(f"graph replay (one graph, contexts 700..120000): {'identical to eager' if ok else 'MISMATCH'}")
    return ok


def vllm_prefill_semantics() -> None:
    import vllm._custom_ops  # noqa: F401

    logits = torch.arange(1000, dtype=torch.float32, device=dev).flip(0).repeat(2, 1).contiguous()
    starts = torch.tensor([0, 100], dtype=torch.int32, device=dev)
    ends = torch.tensor([600, 700], dtype=torch.int32, device=dev)
    out = torch.full((2, 512), -7, dtype=torch.int32, device=dev)
    torch.ops._C.top_k_per_row_prefill(logits, starts, ends, out, 2, logits.stride(0), logits.stride(1), 512)
    r1 = out[1]
    print(f"vLLM top_k_per_row_prefill, row [100, 700): min {int(r1.min())} max {int(r1.max())} "
          f"-> {'absolute columns' if int(r1.min()) >= 100 else 'relative to the row start'}")
    short = torch.full((1, 512), -7, dtype=torch.int32, device=dev)
    torch.ops._C.top_k_per_row_prefill(logits[:1], starts[:1], torch.tensor([10], dtype=torch.int32, device=dev), short, 1,
                                       logits.stride(0), logits.stride(1), 512)
    print(f"vLLM top_k_per_row_prefill, 10-column row: first 12 = {short[0, :12].tolist()}")


def bench() -> None:
    import vllm._custom_ops  # noqa: F401

    def t(fn, iters=200):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) * 1000 / iters

    g = torch.Generator().manual_seed(3)
    for rows, title in ((6, "C1, k=5"), (48, "C8")):
        print(f"decode, {rows} rows ({title}), us:   vLLM persistent_topk | det_topk (auto) | one CTA/row | cluster forced")
        for width in (1024, 4096, 8192, 12288, 16384, 32768, 100000, 300000, 524288):
            for kind in ("normal", "bf16", "ties"):
                logits = make_logits(rows, width, kind, g)
                lens = torch.tensor([[width - 5 + j for j in range(6)]] * (rows // 6), dtype=torch.int32, device=dev)
                out = torch.empty(rows, 512, dtype=torch.int32, device=dev)
                ws = torch.zeros(1024 * 1024, dtype=torch.uint8, device=dev)
                tp = t(lambda: torch.ops._C.persistent_topk(logits, lens, out, ws, 512, width))
                td = t(lambda: dt.topk_decode(logits, lens, 6, out, 512))
                t1 = t(lambda: dt.topk_decode(logits, lens, 6, out, 512, cluster=1))
                tc = t(lambda: dt.topk_decode(logits, lens, 6, out, 512, small=0))
                print(f"  width {width:6d} {kind:6s}: {tp:8.1f} | {td:8.1f} | {t1:8.1f} | {tc:8.1f}")
    print("prefill, us:                          vLLM top_k_per_row_prefill | det_topk")
    for rows, width in ((2700, 700), (4096, 2048), (4096, 8192), (4096, 32768)):
        for kind in ("normal", "bf16", "ties"):
            logits = make_logits(rows, width, kind, g)
            starts = torch.zeros(rows, dtype=torch.int32, device=dev)
            ends = torch.clamp(torch.arange(1, rows + 1, device=dev, dtype=torch.int32) * width // rows, min=1)
            out = torch.empty(rows, 512, dtype=torch.int32, device=dev)
            tp = t(lambda: torch.ops._C.top_k_per_row_prefill(logits, starts, ends, out, rows, logits.stride(0),
                                                              logits.stride(1), 512), iters=20)
            td = t(lambda: dt.topk_prefill(logits, starts, ends, out, rows, 512, True), iters=20)
            print(f"  {rows}x{width:<6d} {kind:6s}: {tp:8.1f} | {td:8.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    print(torch.cuda.get_device_name(), torch.__version__)
    dt._ext()
    vllm_prefill_semantics()
    ok = correctness(a.quick) & determinism() & graph_replay()
    if a.bench:
        bench()
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)
