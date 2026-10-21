"""Exactness of the b12x-line CPU expert kernel (libcpu_moe_b12x.so) against its scalar reference, plus layout sanity.

    python test_cpu_moe_b12x.py [--layer 20] [--experts 0,1,2,3] [--rank 0] [--cpus 8-15] [--seeds 6] [--quick]

  1. synthetic slices x synthetic MXFP8 tokens (realistic, wide-dynamic-range and adversarial groups that reach
     e4m3 subnormals and the FC2 truncation paths): the kernel's projections, intermediate codes + scale bytes and
     route_output rows must equal the scalar reference bit for bit; m = 1..6 per expert, several experts per job,
     permuted output rows, non-unit vs1 / vs2.
  2. real expert slices from the checkpoint with the recorded hidden states (internal/.../parity/inputs.npz) and
     synthetic ones: the same identity, and a cosine check against fp32 math on the dequantized operands.
  3. timing: experts at m = 1 and m = 6 on one CCD (expert bytes per second).
The GPU parity (b12x itself) is gpu_parity_b12x.py; this test does not need a GPU.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv41_b12x as B  # noqa: E402


def parse_cpus(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def realistic_hidden(T: int, gen: torch.Generator, flavor: str) -> torch.Tensor:
    x = torch.randn(T, B.HID, generator=gen)
    if flavor == "realistic":
        heavy = torch.randperm(B.HID, generator=gen)[:48]
        x[:, heavy] *= 12.0
        x[:, heavy[:4]] *= 8.0
    elif flavor == "wide":
        # every 32-group gets a huge outlier and a tail of tiny values: subnormal e4m3 codes everywhere
        xg = x.view(T, -1, 32)
        xg[:, :, 0] *= 400.0
        xg[:, :, 1:8] *= 1e-3
        xg[:, :, 8:12] *= 1e-5
    elif flavor == "tiny":
        x *= 1e-6
        x[:, ::97] *= 3e5
    elif flavor == "big":
        x *= 3e3
    return x.to(torch.bfloat16)


def run_job(pool: B.Pool, records: list[np.ndarray], plan: list[tuple[int, list[int]]], aq, as_, weights,
            vs1: float, vs2: float, gen: torch.Generator):
    """plan = [(expert index into records, [token ids])]; returns kernel outputs + per-pair reference outputs."""
    rec_ptrs, m_per, tok, pw, out_row = [], [], [], [], []
    pairs = []
    for e, toks in plan:
        rec_ptrs.append(records[e].ctypes.data)
        m_per.append(len(toks))
        for t in toks:
            tok.append(t)
            pw.append(float(weights[t, e % weights.shape[1]]))
            pairs.append((e, t))
    P = len(tok)
    perm = torch.randperm(P, generator=gen).tolist()
    out_row = perm
    out = np.full((P + 3, B.N2), 0xFFFF, dtype=np.uint16)  # rows beyond P must stay untouched
    dbg = pool.experts_forward(rec_ptrs, m_per, tok, pw, out_row, aq, as_, out, vs1, vs2, debug=True)
    refs = [B.reference_pair(records[e], aq[t], as_[t], pw[i], vs1, vs2) for i, (e, t) in enumerate(pairs)]
    return out, out_row, dbg, refs, pairs


def check_identity(label: str, out, out_row, dbg, refs) -> bool:
    ok = True
    for i, r in enumerate(refs):
        same_p = np.array_equal(dbg["proj"][i], r["proj"])
        same_q = np.array_equal(dbg["q"][i], r["q"])
        same_s = np.array_equal(dbg["sbytes"][i], r["sbytes"])
        same_o = np.array_equal(out[out_row[i]], r["out"])
        if not (same_p and same_q and same_s and same_o):
            ok = False
            dp = int((dbg["proj"][i] != r["proj"]).sum())
            dq = int((dbg["q"][i] != r["q"]).sum())
            ds = int((dbg["sbytes"][i] != r["sbytes"]).sum())
            do = int((out[out_row[i]] != r["out"]).sum())
            print(f"    MISMATCH {label} pair {i}: proj {dp}/{B.N13}, codes {dq}/{B.K2}, scales {ds}/{B.B2}, out {do}/{B.N2}")
    untouched = int((out[len(refs):] != 0xFFFF).sum())
    if untouched:
        ok = False
        print(f"    {label}: {untouched} bytes written outside the pairs' rows")
    return ok


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--experts", default="0,1,2,3")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--cpus", default="8-15")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-checkpoint", action="store_true")
    args = ap.parse_args()
    cpus = parse_cpus(args.cpus)
    pool = B.Pool(cpus)
    total_ok = True

    # ---- 1. synthetic ------------------------------------------------------------------------------------------
    print("1. synthetic slices x synthetic tokens: kernel vs the scalar reference (bit-exact)")
    gen = torch.Generator().manual_seed(1)
    n_rec = 3
    records = [B.random_record(gen, 116 + 2 * i, 124 + 2 * i) for i in range(n_rec)]
    flavors = ["realistic", "wide", "tiny", "big"]
    t_ref = 0.0
    n_pairs = 0
    for seed in range(args.seeds):
        flavor = flavors[seed % len(flavors)]
        g = torch.Generator().manual_seed(100 + seed)
        T = 6
        x = realistic_hidden(T, g, flavor)
        aq, as_ = B.rows_quant(x, zero_byte=0)
        weights = (torch.rand(T, n_rec, generator=g) + 0.05).numpy()
        if seed % 3 == 0:
            plan = [(0, [0]), (1, [1, 2]), (2, [3, 4, 5]), (0, [5, 0, 1, 2, 3, 4])]  # m = 1, 2, 3, 6
        elif seed % 3 == 1:
            plan = [(2, [0, 1, 2, 3]), (1, [4])]
        else:
            plan = [(0, [t]) for t in range(T)]
        vs1, vs2 = (1.0, 1.0) if seed % 2 == 0 else (1.5, 0.75)
        t0 = time.perf_counter()
        out, out_row, dbg, refs, pairs = run_job(pool, records, plan, aq, as_, weights, vs1, vs2, g)
        t_ref += time.perf_counter() - t0
        n_pairs += len(pairs)
        ok = check_identity(f"seed {seed} ({flavor}, vs {vs1}/{vs2})", out, out_row, dbg, refs)
        # how many pairs took the FC2 truncation paths is not observable here; report the code statistics instead
        sub = int(((dbg["q"] & 0x78) == 0).sum() - (dbg["q"] == 0).sum() - (dbg["q"] == 0x80).sum())
        print(f"  seed {seed:2d} {flavor:9s} pairs {len(pairs):2d} vs1/vs2 {vs1}/{vs2}: "
              f"{'identical' if ok else 'DIFFERENT'}; e4m3 subnormal intermediate codes {sub}")
        total_ok &= ok
        if args.quick and seed >= 1:
            break
    print(f"  ({n_pairs} pairs, reference+kernel {t_ref:.1f} s)")

    # ---- 2. real slices ----------------------------------------------------------------------------------------
    if not args.no_checkpoint:
        print(f"2. checkpoint slices (layer {args.layer}, rank {args.rank}, experts {args.experts})")
        exps = [int(e) for e in args.experts.split(",")]
        t0 = time.perf_counter()
        recs = [B.record_from_checkpoint(args.layer, e, args.rank) for e in exps]
        print(f"  {len(recs)} records built in {time.perf_counter() - t0:.1f} s")
        inputs = os.path.join(B.TABLES_DIR, "inputs.npz")
        xs = []
        if os.path.exists(inputs):
            Z = np.load(inputs)
            xs.append(("recorded hidden states", torch.from_numpy(Z["x_bf16"].astype(np.int16)).view(torch.bfloat16)))
        g = torch.Generator().manual_seed(7)
        xs.append(("synthetic realistic", realistic_hidden(6, g, "realistic")))
        xs.append(("synthetic wide", realistic_hidden(6, g, "wide")))
        for name, x in xs:
            T = x.shape[0]
            aq, as_ = B.rows_quant(x, zero_byte=0)
            weights = (torch.rand(T, len(recs), generator=g) + 0.05).numpy()
            plan = [(e, list(range(T))[: (e % 3) + 1]) for e in range(len(recs))] + [(0, list(range(T)))]
            out, out_row, dbg, refs, pairs = run_job(pool, recs, plan, aq, as_, weights, 1.0, 1.0, g)
            ok = check_identity(name, out, out_row, dbg, refs)
            total_ok &= ok
            # approximate reference on the last plan entry (expert 0, all tokens)
            ap_ = B.approx_forward(recs[0], aq, as_, weights[:, 0])
            i0 = len(pairs) - T
            kp = B.bf16_bits_to_float(dbg["proj"][i0:i0 + T])
            ko = B.bf16_bits_to_float(np.stack([out[out_row[i]] for i in range(i0, i0 + T)]))
            print(f"  {name:24s} T={T}: {'identical' if ok else 'DIFFERENT'}; vs fp32 math: proj cosine "
                  f"{cosine(kp, ap_['proj']):.6f}, route_output cosine {cosine(ko, ap_['out']):.6f}")

    # ---- 3. timing ---------------------------------------------------------------------------------------------
    print(f"3. timing on CPUs {args.cpus} ({len(cpus)} threads)")
    g = torch.Generator().manual_seed(3)
    n_time = 2 if args.quick else 24
    arena = B.RecordArena(n_time)
    for i in range(n_time):
        arena.slot(i)[:] = records[i % n_rec]
    x = realistic_hidden(6, g, "realistic")
    aq, as_ = B.rows_quant(x)
    for m in (1, 2, 6):
        ptrs = [arena.ptr(i) for i in range(n_time)]
        m_per = [m] * n_time
        tok = [t % 6 for e in range(n_time) for t in range(m)]
        pw = [0.5] * (n_time * m)
        out_row = list(range(n_time * m))
        out = np.zeros((n_time * m, B.N2), dtype=np.uint16)
        pool.experts_forward(ptrs, m_per, tok, pw, out_row, aq, as_, out)
        pool.stats(reset=True)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            pool.experts_forward(ptrs, m_per, tok, pw, out_row, aq, as_, out)
            ts.append(time.perf_counter() - t0)
        st = pool.stats(reset=True)
        per = min(ts) / n_time * 1e6
        cyc = st["cyc_prepare"] + st["cyc_fc1"] + st["cyc_act"] + st["cyc_fc2"]
        print(f"  m = {m}: {per:7.1f} us per expert ({B.RECORD_BYTES / per / 1e3:.1f} GB/s of expert bytes; "
              f"{n_time} experts per job, arena {n_time * B.RECORD_BYTES / 2**20:.0f} MiB); worker 0: prepare "
              f"{100 * st['cyc_prepare'] / cyc:.0f} % FC1 {100 * st['cyc_fc1'] / cyc:.0f} % act {100 * st['cyc_act'] / cyc:.0f} % "
              f"FC2 {100 * st['cyc_fc2'] / cyc:.0f} %; FC2 general path {st['fc2_general']} of {st['fc2_blocks']} blocks")
    arena.close()
    pool.close()
    print("PASS" if total_ok else "FAIL")
    return 0 if total_ok else 1


if __name__ == "__main__":
    sys.exit(main())
