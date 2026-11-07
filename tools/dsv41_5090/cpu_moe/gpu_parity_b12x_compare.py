"""Host side 3/3 of the CPU-vs-b12x parity check: the CPU kernel on the GPU's quantized tokens, bit for bit.

    python gpu_parity_b12x_compare.py [--inputs b12x_inputs.npz] [--gpu b12x_gpu.npz] [--report g1b-b12x-parity.json]

For every (token, expert) pair of the routing: the CPU kernel's projections (gate | up), intermediate e4m3 codes
and ue8m0 bytes, and route_output rows (router weight applied) against b12x's; also the host rows quant of the bf16
tokens against b12x's a_values / a_scales, and the scalar reference against the kernel. Gate G1b: zero mismatches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv41_b12x as B  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=os.path.join(B.TABLES_DIR, "b12x_inputs.npz"))
    ap.add_argument("--gpu", default=os.path.join(B.TABLES_DIR, "b12x_gpu.npz"))
    ap.add_argument("--report", default=os.path.join(B.TABLES_DIR, "g1b-b12x-parity.json"))
    ap.add_argument("--cpus", default="8-15")
    args = ap.parse_args()
    Zi = np.load(args.inputs)
    Zg = np.load(args.gpu)
    records = Zi["records"]
    ids, weights = Zi["ids"], Zi["weights"]
    T, TOPK = ids.shape
    E = records.shape[0]
    report = dict(E=int(E), T=int(T), topk=int(TOPK), impl=str(Zg["impl"]), graph_same=int(Zg["graph_same"]),
                  cos_gate_first=float(Zg["cos_gate_first"]), cos_gate_second=float(Zg["cos_gate_second"]))
    for k in ("alpha1", "input_scale", "alpha2", "down_scale"):
        report[k] = np.unique(Zg[f"scal_{k}"]).tolist()
    vs1 = float(np.float32(Zg["scal_alpha1"][0]) * np.float32(Zg["scal_input_scale"][0]))
    vs2 = float(np.float32(Zg["scal_alpha2"][0]) * np.float32(Zg["scal_down_scale"][0]))
    print(f"b12x run: impl {report['impl']}, alpha1 {report['alpha1']} input_scale {report['input_scale']} alpha2 "
          f"{report['alpha2']} down_scale {report['down_scale']} -> vs1 {vs1} vs2 {vs2}; graph replays identical "
          f"{report['graph_same']}/3; gate half of the projections vs the first half of w13 {report['cos_gate_first']:.5f}, "
          f"vs the second {report['cos_gate_second']:.5f}")

    # 1. the host rows quant vs b12x's
    x = torch.from_numpy(Zi["x_bf16"]).view(torch.bfloat16)
    hq, hs = B.rows_quant(x, zero_byte=0)
    dq = int((hq != Zg["a_values"]).sum())
    ds = int((hs != Zg["a_scales"]).sum())
    report["rows_quant"] = dict(code_mismatch=dq, codes=int(hq.size), scale_mismatch=ds, scales=int(hs.size))
    print(f"1. host rows quant vs b12x a_values / a_scales: {dq} of {hq.size} codes differ, {ds} of {hs.size} scale bytes")

    # 2. the kernel on b12x's quantized tokens, every pair
    aq = np.ascontiguousarray(Zg["a_values"])
    as_ = np.ascontiguousarray(Zg["a_scales"])
    pool = B.Pool([int(c) for part in args.cpus.split(",") for c in
                   (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [int(part)])])
    # group pairs by expert (the kernel's job form), output rows = t * topk + k (b12x's pair index)
    by_e: dict[int, list[tuple[int, int]]] = {}
    for t in range(T):
        for k in range(TOPK):
            by_e.setdefault(int(ids[t, k]), []).append((t, k))
    rec_ptrs, m_per, tok, pw, orow = [], [], [], [], []
    recs = [np.ascontiguousarray(records[e]) for e in range(E)]
    for e in sorted(by_e):
        rec_ptrs.append(recs[e].ctypes.data)
        m_per.append(len(by_e[e]))
        for t, k in by_e[e]:
            tok.append(t)
            pw.append(float(weights[t, k]))
            orow.append(t * TOPK + k)
    P = T * TOPK
    out = np.zeros((P, B.N2), dtype=np.uint16)
    t0 = time.perf_counter()
    dbg = pool.experts_forward(rec_ptrs, m_per, tok, pw, orow, aq, as_, out, vs1, vs2, debug=True)
    dt = time.perf_counter() - t0
    # scatter the debug outputs to pair order
    proj = np.zeros((P, B.N13), dtype=np.uint16)
    q = np.zeros((P, B.K2), dtype=np.uint8)
    sb = np.zeros((P, B.B2), dtype=np.uint8)
    for i, r in enumerate(orow):
        proj[r] = dbg["proj"][i]
        q[r] = dbg["q"][i]
        sb[r] = dbg["sbytes"][i]
    gp = Zg["projections"].view(np.uint16)
    gr = Zg["route_output"].view(np.uint16)
    d_proj = int((proj != gp).sum())
    d_q = int((q != Zg["inter_codes"]).sum())
    d_s = int((sb != Zg["inter_scales"]).sum())
    d_out = int((out != gr).sum())
    per_pair_out = [int((out[p] != gr[p]).sum()) for p in range(P)]
    report["kernel_vs_b12x"] = dict(pairs=int(P), proj_mismatch=d_proj, proj_n=int(proj.size), code_mismatch=d_q,
                                    codes=int(q.size), scale_mismatch=d_s, scales=int(sb.size),
                                    route_output_mismatch=d_out, route_output_n=int(out.size),
                                    pairs_with_mismatch=int(sum(1 for v in per_pair_out if v)), cpu_ms=dt * 1e3)
    print(f"2. kernel vs b12x on {P} pairs ({E} experts, m per expert {m_per}): projections {d_proj}/{proj.size}, "
          f"intermediate codes {d_q}/{q.size}, scale bytes {d_s}/{sb.size}, route_output {d_out}/{out.size} "
          f"(pairs with any difference {report['kernel_vs_b12x']['pairs_with_mismatch']}); CPU {dt * 1e3:.1f} ms")
    if d_out:
        bad = [p for p in range(P) if per_pair_out[p]][:5]
        for p in bad:
            cols = np.nonzero(out[p] != gr[p])[0][:8]
            print(f"   pair {p} (token {p // TOPK}, expert {ids[p // TOPK, p % TOPK]}): {per_pair_out[p]} cols, e.g. "
                  f"{[(int(c), hex(out[p, c]), hex(gr[p, c])) for c in cols]}")
    if d_proj:
        rows = np.nonzero((proj != gp).any(axis=1))[0][:3]
        for p in rows:
            cols = np.nonzero(proj[p] != gp[p])[0][:8]
            print(f"   proj pair {p}: {[(int(c), hex(proj[p, c]), hex(gp[p, c])) for c in cols]}")

    # 3. the scalar reference vs the kernel on the same pairs (consistency of the two CPU paths)
    d_ref = 0
    for p in range(P):
        t, k = p // TOPK, p % TOPK
        r = B.reference_pair(recs[int(ids[t, k])], aq[t], as_[t], float(weights[t, k]), vs1, vs2)
        d_ref += int((r["out"] != out[p]).sum()) + int((r["proj"] != proj[p]).sum()) + int((r["q"] != q[p]).sum())
    report["reference_vs_kernel_mismatch"] = d_ref
    print(f"3. scalar reference vs kernel: {d_ref} differing elements")

    # 4. TopKSum reproduction (informational): b12x's output vs the fp32 sequential sum of its own rows
    route_f = torch.from_numpy(gr.astype(np.int32) << 16).view(torch.float32).view(T, TOPK, B.N2)
    out_f = torch.from_numpy(Zg["out"].view(np.uint16).astype(np.int32) << 16).view(torch.float32)
    s = route_f[:, 0].clone()
    for k in range(1, TOPK):
        s = s + route_f[:, k]
    d_sum = int((s.to(torch.bfloat16).float() != out_f).sum())
    report["topksum_sequential_fp32_mismatch"] = d_sum
    print(f"4. b12x output vs sequential fp32 sum of its route_output rows -> bf16: {d_sum} of {out_f.numel()} differ")

    passed = d_proj == 0 and d_q == 0 and d_s == 0 and d_out == 0 and d_ref == 0 and dq == 0 and ds == 0
    report["G1b"] = "PASS" if passed else "FAIL"
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)
    print(f"G1b {'PASS' if passed else 'FAIL'}; report {args.report}")
    pool.close()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
