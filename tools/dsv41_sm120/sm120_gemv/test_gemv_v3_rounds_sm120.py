#!/usr/bin/env python3
"""GEMV v3 units per round (step 29): the per-shape table (VLLM_MOET_GEMV_V3_R=0, the default) must give the same bits
as R = 1, 2 and 4 on the served decode shapes (the unit -> warp mapping does not depend on R), and its time per shape
is reported against R = 1 (cold L2, CUDA graph of rotated weight copies). The R values are set per child process
(the extension reads the variable once).

  python3 test_gemv_v3_rounds_sm120.py [--iters 50]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SHAPES = ["q_a+kv_a 5120->1792", "q_b+idx 1280->8192", "wo_b 2048->5120", "shared w13 5120->1152", "idx wq_b 1280->4096"]


def child(r_value: str, out_path: str, iters: int) -> None:
    os.environ["VLLM_MOET_GEMV_V3_R"] = r_value
    from mxfp8_gemv_sm120 import mxfp8_gemv
    from test_mxfp8_gemv_sm120 import SHAPES as ALL, bench, quant_rowmajor
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, swizzle_mxfp8_scale

    torch.manual_seed(3)
    dev = torch.device("cuda")
    res = {}
    for name in SHAPES:
        N, K = ALL[name]
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
        w_q, w_sf2d = quant_rowmajor(w)
        w_sf_sw = swizzle_mxfp8_scale(w_sf2d, M=N, K=K)
        outs = []
        for M in (1, 6, 8):
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            a_q, a_sf_sw = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            outs.append(mxfp8_gemv(a_q, a_sf_sw, w_q, w_sf_sw).view(torch.int16).cpu())
        x = torch.randn(6, K, device=dev, dtype=torch.bfloat16)
        a_q, a_sf_sw = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        n_copies = min(64, max(2, (160 << 20) // max(N * K, 1) + 1))
        rot = [(w_q.clone(), w_sf_sw.clone()) for _ in range(n_copies)]
        t = bench(lambda wc, sc: mxfp8_gemv(a_q, a_sf_sw, wc, sc), iters, per_graph=len(rot), rotate=rot)
        res[name] = {"outs": outs, "us": t}
    torch.save(res, out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--child", nargs=2, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        child(a.child[0], a.child[1], a.iters)
        return 0
    results = {}
    with tempfile.TemporaryDirectory() as td:
        for r_value in ("0", "1", "2", "4"):
            out = os.path.join(td, f"r{r_value}.pt")
            p = subprocess.run([sys.executable, __file__, "--iters", str(a.iters), "--child", r_value, out],
                               capture_output=True, text=True)
            if p.returncode != 0:
                print(p.stderr[-3000:])
                return 1
            results[r_value] = torch.load(out)
    ok = True
    print("bits (per-shape table vs R = 1 / 2 / 4, M = 1 / 6 / 8) and time (us, cold L2, graph):")
    for name in SHAPES:
        same = all(torch.equal(results["0"][name]["outs"][i], results[r][name]["outs"][i]) for r in ("1", "2", "4")
                   for i in range(3))
        ok &= same
        t = {r: results[r][name]["us"] for r in ("0", "1", "2", "4")}
        print(f"  [{'ok ' if same else 'BAD'}] {name:<24} table {t['0']:5.2f}  R=1 {t['1']:5.2f}  R=2 {t['2']:5.2f}  R=4 {t['4']:5.2f}")
    print("ALL OK" if ok else "BIT MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
