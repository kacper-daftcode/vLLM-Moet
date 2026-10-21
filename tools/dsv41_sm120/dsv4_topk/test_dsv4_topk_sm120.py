#!/usr/bin/env python3
"""dsv4_topk_sm120 against vLLM's Triton dsv4_topk, bit for bit, then the time of each in a CUDA graph.

Inputs: 1..64 tokens, 384 and 256 experts, logits drawn from several ranges (incl. > 20, very negative, exact ties,
NaN rows), bias with exact ties and -0.0, optional padding mask, optional image tokens with bias_vl, int32 / int64
indices. Timing: 6 and 48 tokens, after a PDL-launched predecessor (a GEMV-like kernel writing the logits), graph
replays, Triton vs this kernel.
usage (serving image, one sm_120 GPU): test_dsv4_topk_sm120.py [--cases 3000] [--bench]
Exit status 0 only if every case is bit-identical.
"""
import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsv4_topk_sm120 import dsv4_topk_cuda  # noqa: E402
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk  # noqa: E402


def make_case(g, E, T):
    kind = int(torch.randint(0, 6, (1,), generator=g))
    if kind == 0:
        x = torch.randn(T, E, generator=g) * 3
    elif kind == 1:
        x = torch.randn(T, E, generator=g) * 30          # many > 20 (identity branch), many very negative
    elif kind == 2:
        x = torch.randint(-4, 5, (T, E), generator=g).float() * 0.5   # exact ties
    elif kind == 3:
        x = torch.randn(T, E, generator=g) * 0.01
    elif kind == 4:
        x = torch.randn(T, E, generator=g) * torch.exp2(torch.randint(-20, 8, (T, E), generator=g).float())
    else:
        x = torch.randn(T, E, generator=g) * 5
        nan_rows = torch.rand(T, generator=g) < 0.2
        x[nan_rows, torch.randint(0, E, (1,), generator=g)] = float("nan")
    bias = torch.randn(E, generator=g) * float(torch.rand(1, generator=g)) * 0.2
    if kind == 2:
        bias = torch.randint(-2, 3, (E,), generator=g).float() * 0.25
        bias[torch.rand(E, generator=g) < 0.1] = -0.0
    return x, bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=3000)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(1234)
    bad = 0
    for c in range(a.cases):
        E = 384 if c % 4 else 256
        T = int(torch.randint(1, 65, (1,), generator=g))
        x, bias = make_case(g, E, T)
        x, bias = x.to(dev), bias.to(dev)
        kw = {}
        if c % 3 == 1:
            kw["is_padding"] = (torch.rand(T, generator=g) < 0.3).to(dev)
        if c % 5 == 2:
            kw["input_ids"] = torch.randint(95, 110, (T,), generator=g).to(torch.int32 if c % 10 == 2 else torch.int64).to(dev)
            kw["bias_vl"] = (torch.randn(E, generator=g) * 0.3).to(dev)
            kw["image_sentinel_lo"] = 100
        idt = torch.int64 if c % 2 else torch.int32
        scale = float(torch.rand(1, generator=g)) * 3 + 0.5
        w_ref, i_ref = dsv4_topk(x, bias, idt, scale, **kw)
        w, i = dsv4_topk_cuda(x, bias, idt, scale, **kw, pdl=bool(c % 2))
        same = torch.equal(i, i_ref) and torch.equal(w.view(torch.int32), w_ref.view(torch.int32))
        if not same:
            bad += 1
            if bad <= 5:
                d = (w.view(torch.int32) != w_ref.view(torch.int32)) | (i != i_ref)
                r = int(d.any(1).nonzero()[0])
                print(f"MISMATCH case {c} E={E} T={T} kw={list(kw)} row {r}: ids {i[r].tolist()} vs {i_ref[r].tolist()}"
                      f" w {w[r].tolist()} vs {w_ref[r].tolist()}")
    print(f"bit-exact: {a.cases - bad}/{a.cases} cases identical (ids and weight bits)")
    if not a.bench:
        return 0 if bad == 0 else 1

    # timing: a predecessor writes the logits (stands in for the router GEMM), then the top-k, in a graph
    for T in (6, 48):
        x = torch.randn(T, 384, device=dev)
        bias = torch.randn(384, device=dev) * 0.1
        src = torch.randn(T, 384, device=dev)
        pad = torch.zeros(T, dtype=torch.bool, device=dev)
        res = {}
        for name, fn in (("triton", lambda: dsv4_topk(x, bias, torch.int32, 2.5, is_padding=pad)),
                         ("cuda", lambda: dsv4_topk_cuda(x, bias, torch.int32, 2.5, is_padding=pad))):
            def step():
                for _ in range(20):
                    x.copy_(src)
                    fn()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                step()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=s):
                step()
            ts = []
            for _ in range(30):
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record()
                gr.replay()
                en.record()
                torch.cuda.synchronize()
                ts.append(st.elapsed_time(en) * 1000 / 20)
            res[name] = statistics.median(ts)
        # the copy alone
        def cp():
            for _ in range(20):
                x.copy_(src)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            cp()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=s):
            cp()
        ts = []
        for _ in range(30):
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            gr.replay()
            en.record()
            torch.cuda.synchronize()
            ts.append(st.elapsed_time(en) * 1000 / 20)
        base = statistics.median(ts)
        print(f"T={T}: per (copy + top-k) triton {res['triton']:.2f} us, cuda {res['cuda']:.2f} us; copy alone {base:.2f} us"
              f" -> top-k triton {res['triton'] - base:.2f} us, cuda {res['cuda'] - base:.2f} us")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
