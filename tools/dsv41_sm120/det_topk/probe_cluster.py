#!/usr/bin/env python3
"""det_topk on sm_120, 6 decode rows (C1, k=5): CTAs per row vs row length, next to vLLM's persistent_topk."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_topk_sm120 as dt  # noqa: E402
from test_det_topk_sm120 import make_logits  # noqa: E402

dev = "cuda"


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


import vllm._custom_ops  # noqa: E402,F401

sizes = (1, 2, 4, 8, 16)
widths = [int(w) for w in sys.argv[1:]] or [4096, 8192, 16384, 24576, 32768, 65536, 131072, 200000, 300000, 400000,
                                            524288]
g = torch.Generator().manual_seed(4)
print("6 rows, us. C=n: n CTAs per row (small=0 forces the cluster path); auto: what topk_decode picks")
print("width   kind      vLLM    auto" + "".join(f"  C={c:<5d}" for c in sizes) + "  1CTA+smem")
for width in widths:
    for kind in ("normal", "bf16", "ties"):
        logits = make_logits(6, width, kind, g)
        lens = torch.tensor([[width - 5 + j for j in range(6)]], dtype=torch.int32, device=dev)
        out = torch.empty(6, 512, dtype=torch.int32, device=dev)
        ws = torch.zeros(1024 * 1024, dtype=torch.uint8, device=dev)
        times = [t(lambda: torch.ops._C.persistent_topk(logits, lens, out, ws, 512, width)),
                 t(lambda: dt.topk_decode(logits, lens, 6, out, 512))]
        times += [t(lambda c=c: dt.topk_decode(logits, lens, 6, out, 512, cluster=c, small=0)) for c in sizes]
        times.append(t(lambda: dt.topk_decode(logits, lens, 6, out, 512, cluster=8, small=10**9)))
        print(f"{width:7d} {kind:6s}" + "".join(f"{x:8.1f}" for x in times))
