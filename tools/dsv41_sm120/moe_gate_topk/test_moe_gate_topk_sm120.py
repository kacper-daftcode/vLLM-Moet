#!/usr/bin/env python3
"""The fused MoE router (gate GEMV + sqrtsoftplus + bias + top-k + renorm) against vLLM's path
(GateLinear's cuBLAS bf16 GEMM -> dsv4_topk) on DeepSeek-V4.1-Flash's router (E = 384, K = 5120,
top-6, routed_scaling 1.5, bias + bias_vl for image tokens).

Checks (one sm_120 GPU, inside the image):
  * the top-k logic given identical scores: on rows built so the fp32 logits are exact (weights
    and inputs from small integer grids) the fused kernel and dsv4_topk must agree bit for bit -
    ids, tie-breaking to the smallest id, renormalized weights;
  * on random bf16 rows (the served distribution) vs cuBLAS -> dsv4_topk: ids equal except where
    the two fp32 accumulation orders land on opposite sides of a tie (reported as a rate), weights
    within 1e-5 relative; image tokens (bias_vl) included;
  * CUDA-graph replay (the arrival counter resets itself) and timing vs the three-kernel path.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moe_gate_topk_sm120 import fused_gate_topk, fused_gate_topk_applicable, gate_logits, new_counter  # noqa: E402

from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk  # noqa: E402

E, K, TOPK, SCALING, SENTINEL = 384, 5120, 6, 1.5, 128815


def reference(x, w, bias, bias_vl, input_ids):
    logits = torch.mm(x, w.T, out_dtype=torch.float32)  # GateLinear tier 4 (cuBLAS)
    return dsv4_topk(logits, bias, torch.int32, SCALING, input_ids=input_ids, bias_vl=bias_vl,
                     image_sentinel_lo=SENTINEL if bias_vl is not None else 0)


def bench_graph(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        s.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(reps):
                fn()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / reps)
    return statistics.median(ts), g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--rot", type=int, default=48, help="gate weight copies rotated in the graph (cold L2)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(7)
    fails = 0
    bias = (torch.randn(E, device=dev) * 0.1).float().contiguous()
    bias_vl = (torch.randn(E, device=dev) * 0.1).float().contiguous()

    # 1. exact-logit rows: x, w on small integer grids -> every fp32 partial sum is exact, so both
    #    accumulation orders give the same logits and the top-k logic is compared bit for bit
    w_int = (torch.randint(-2, 3, (E, K), device=dev).to(torch.bfloat16))
    for T in (1, 6, 16):
        x_int = torch.randint(-1, 2, (T, K), device=dev).to(torch.bfloat16)
        x_int[:, 64:] = 0  # 64 non-zero products per logit: exact in fp32
        ids = torch.full((T,), 1000, device=dev, dtype=torch.int64)
        if T >= 6:
            ids[1] = SENTINEL + 2  # an image token
            x_int[3] = x_int[0]  # identical rows -> identical scores -> identical selection
        w_ref, i_ref = reference(x_int, w_int, bias, bias_vl, ids)
        r = fused_gate_topk(x_int, w_int, bias, bias_vl, ids, SENTINEL, TOPK, SCALING)
        torch.cuda.synchronize()
        same_ids = torch.equal(r.ids, i_ref)
        same_w = torch.equal(r.weights, w_ref)
        dw = (r.weights - w_ref).abs().max().item()
        ok = same_ids and dw <= 2e-7
        fails += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] exact logits T={T:2d}: ids {'identical' if same_ids else 'DIFFER'}, "
              f"weights {'bit-identical' if same_w else f'max diff {dw:.2e}'}")
    # exact ties: all-equal logits -> the six smallest ids in order, with the bias deciding
    x0 = torch.zeros(4, K, device=dev, dtype=torch.bfloat16)
    ids0 = torch.full((4,), 1000, device=dev, dtype=torch.int64)
    w_ref, i_ref = reference(x0, w_int, bias, bias_vl, ids0)
    r = fused_gate_topk(x0, w_int, bias, bias_vl, ids0, SENTINEL, TOPK, SCALING)
    torch.cuda.synchronize()
    ok = torch.equal(r.ids, i_ref) and (r.weights - w_ref).abs().max().item() <= 2e-7
    fails += 0 if ok else 1
    print(f"[{'ok ' if ok else 'BAD'}] zero rows (bias decides): ids {r.ids[0].tolist()} vs {i_ref[0].tolist()}")
    zero_bias = torch.zeros(E, device=dev)
    w_ref, i_ref = reference(x0, w_int, zero_bias, None, None)
    r = fused_gate_topk(x0, w_int, zero_bias, None, None, 0, TOPK, SCALING)
    torch.cuda.synchronize()
    ok = torch.equal(r.ids, i_ref) and torch.equal(r.weights, w_ref)
    fails += 0 if ok else 1
    print(f"[{'ok ' if ok else 'BAD'}] all-tie rows: ids {r.ids[0].tolist()} (smallest ids first), weights {r.weights[0, :2].tolist()}")

    # 2. random bf16 rows (served distribution) vs cuBLAS -> dsv4_topk
    w = (torch.randn(E, K, device=dev) * 0.02).to(torch.bfloat16)
    n_tok = n_row_diff = n_set_diff = 0
    max_rel = 0.0
    for seed in range(40):
        torch.manual_seed(100 + seed)
        T = int(torch.randint(1, 17, (1,)).item())
        x = (torch.randn(T, K, device=dev) * float(torch.rand(1).item() * 2 + 0.2)).to(torch.bfloat16)
        ids = torch.randint(0, 100000, (T,), device=dev, dtype=torch.int64)
        ids[torch.rand(T, device=dev) < 0.3] = SENTINEL + 1
        w_ref, i_ref = reference(x, w, bias, bias_vl, ids)
        r = fused_gate_topk(x, w, bias, bias_vl, ids, SENTINEL, TOPK, SCALING)
        torch.cuda.synchronize()
        n_tok += T
        row_same = (r.ids == i_ref).all(dim=1)
        n_row_diff += int((~row_same).sum())
        for t in range(T):
            if set(r.ids[t].tolist()) != set(i_ref[t].tolist()):
                n_set_diff += 1
        if row_same.any():
            rel = ((r.weights[row_same] - w_ref[row_same]).abs() / w_ref[row_same].abs().clamp_min(1e-6)).max().item()
            max_rel = max(max_rel, rel)
    ok = n_set_diff <= max(1, n_tok // 100) and max_rel <= 1e-5
    fails += 0 if ok else 1
    print(f"[{'ok ' if ok else 'BAD'}] random rows: {n_tok} tokens, top-6 set differs for {n_set_diff} "
          f"(order differs for {n_row_diff}), max relative weight diff on equal rows {max_rel:.2e}")

    # 3. applicability + graph replay (counter reset) + timing
    x = (torch.randn(6, K, device=dev) * 0.7).to(torch.bfloat16)
    ids = torch.full((6,), 1000, device=dev, dtype=torch.int64)
    assert fused_gate_topk_applicable(x, w, bias, TOPK, "sqrtsoftplus", True)
    counter = new_counter(dev)
    out = {}

    def fused():
        out["r"] = fused_gate_topk(x, w, bias, bias_vl, ids, SENTINEL, TOPK, SCALING, counter=counter)

    def ref():
        out["ref"] = reference(x, w, bias, bias_vl, ids)

    t_f, g = bench_graph(fused)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    w_ref, i_ref = reference(x, w, bias, bias_vl, ids)
    ok = torch.equal(out["r"].ids, i_ref) and int(counter.item()) == 0
    fails += 0 if ok else 1
    print(f"[{'ok ' if ok else 'BAD'}] graph replay x3: ids {'match' if torch.equal(out['r'].ids, i_ref) else 'DIFFER'}, counter after = {int(counter.item())}")
    if args.bench:
        # cold gate weights: rotate over more copies than the L2 holds (48 x 3.9 MB = 189 MB), as in
        # the served step where 400+ MB of expert weights pass through the L2 between two routers
        ws = [(torch.randn(E, K, device=dev) * 0.02).to(torch.bfloat16) for _ in range(args.rot)]
        for T in (1, 6, 12, 16):
            x = (torch.randn(T, K, device=dev) * 0.7).to(torch.bfloat16)
            ids = torch.full((T,), 1000, device=dev, dtype=torch.int64)
            state = {"i": 0}

            def rot():
                state["i"] = (state["i"] + 1) % len(ws)
                return ws[state["i"]]

            t_f, _ = bench_graph(lambda: fused_gate_topk(x, rot(), bias, bias_vl, ids, SENTINEL, TOPK, SCALING, counter=counter), reps=len(ws))
            t_r, _ = bench_graph(lambda: reference(x, rot(), bias, bias_vl, ids), reps=len(ws))
            t_g, _ = bench_graph(lambda: torch.mm(x, rot().T, out_dtype=torch.float32), reps=len(ws))
            t_fh, _ = bench_graph(lambda: fused_gate_topk(x, w, bias, bias_vl, ids, SENTINEL, TOPK, SCALING, counter=counter))
            t_rh, _ = bench_graph(lambda: reference(x, w, bias, bias_vl, ids))
            # two-kernel variant: this GEMV for the logits, vLLM's dsv4_topk unchanged
            t_l, _ = bench_graph(lambda: gate_logits(x, rot()), reps=len(ws))
            t_lt, _ = bench_graph(lambda: dsv4_topk(gate_logits(x, rot()), bias, torch.int32, SCALING, input_ids=ids,
                                                     bias_vl=bias_vl, image_sentinel_lo=SENTINEL), reps=len(ws))
            lg = gate_logits(x, w)
            lc = torch.mm(x, w.T, out_dtype=torch.float32)
            torch.cuda.synchronize()
            dl = ((lg - lc).abs() / lc.abs().clamp_min(1e-3)).max().item()
            print(f"  T={T:2d}: cold weights: fused {t_f:5.2f} us | cuBLAS GEMM(+splitK) {t_g:5.2f} + dsv4_topk = {t_r:5.2f} us"
                  f"   (L2-hot: fused {t_fh:5.2f}, vLLM {t_rh:5.2f}) | GEMV alone {t_l:5.2f}, GEMV + dsv4_topk {t_lt:5.2f} us"
                  f" (logits vs cuBLAS max rel {dl:.1e})")
    print("ALL OK" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
