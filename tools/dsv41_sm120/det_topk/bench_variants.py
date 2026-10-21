#!/usr/bin/env python3
"""Compile det_topk_sm120.cu with different DET_TOPK_AGG / DET_TOPK_UNROLL and compare speed; outputs must be identical."""
import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_topk_sm120 as dt  # noqa: E402
from test_det_topk_sm120 import make_logits  # noqa: E402

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
src = str(Path(__file__).resolve().parent / "det_topk_sm120.cu")
dev = "cuda"


def build(agg, unroll):
    return load(name=f"det_topk_v_a{agg}_u{unroll}", sources=[src], verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++20", f"-DDET_TOPK_AGG={agg}", f"-DDET_TOPK_UNROLL={unroll}"],
                extra_cflags=["-O3", "-std=c++20"])


def t(fn, iters=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters


variants = [(a, u) for a in (0, 1) for u in (1, 2, 4)]
exts = {v: build(*v) for v in variants}
g = torch.Generator().manual_seed(5)
cases = []
for width in (1024, 8192, 32768, 100000):
    for kind in ("normal", "ties"):
        cases.append(("decode", 6, width, kind, make_logits(6, width, kind, g)))
for rows, width in ((4096, 2048), (4096, 32768)):
    for kind in ("normal", "ties"):
        cases.append(("prefill", rows, width, kind, make_logits(rows, width, kind, g)))

print("case".ljust(28) + "".join(f"a{a}u{u}".rjust(9) for a, u in variants))
for mode, rows, width, kind, logits in cases:
    outs, times = [], []
    for v in variants:
        e = exts[v]
        out = torch.empty(rows, 512, dtype=torch.int32, device=dev)
        if mode == "decode":
            lens = torch.tensor([[width - 5 + j for j in range(6)]], dtype=torch.int32, device=dev)
            fn = lambda: e.det_topk_decode(logits, lens, 6, out, rows, 512)  # noqa: E731
        else:
            ends = torch.clamp(torch.arange(1, rows + 1, device=dev, dtype=torch.int32) * width // rows, min=1)
            fn = lambda: e.det_topk_prefill(logits, None, ends, out, rows, 512, True)  # noqa: E731
        times.append(t(fn, iters=100 if mode == "decode" else 10))
        outs.append(out.clone())
    same = all(torch.equal(o, outs[0]) for o in outs)
    print(f"{mode} {rows}x{width} {kind}".ljust(28) + "".join(f"{x:9.1f}" for x in times) + ("" if same else "  OUTPUTS DIFFER"))
