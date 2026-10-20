"""Expert residency, misses and decode speed of the CPU-hybrid from routing traces (PLAN-dsv41-5090-cpu 1.3).

    python cache_sim.py --routes routes.npz [--resident 0.25,0.28,0.31] [--alloc uniform,global] [--k 0,3,5]
                        [--gpu-step-ms 14] [--json out.json]
    python cache_sim.py --synthetic 1   # mechanics check on a Zipf-routed fake trace

Model: 384 experts x 40 MoE layers x top-6; a rank's slice of an expert is 4.81 MB in the kernel pack. VRAM holds
`resident` x 15360 slices per rank, placed statically from the frequencies of a training half of the sequences
(`uniform`: the same count per layer; `global`: the most frequent (layer, expert) pairs overall), or managed as
LRU per layer (fetch on miss = background promotion, the optional 2.4 variant) warm-started with the static set.
Every missing (layer, expert) of a decode step is computed on the CPU by the four ranks in parallel at the
measured kernel times (tools/dsv41_5090/cpu_moe/bench: 89.7 us per expert slice at 1 token/expert, +3.2 us per
extra token; four CCDs at once). Step = 1 + k positions (DSpark verify window); tokens per step = 1 (k = 0),
2.6 (k = 3, assumed) or 3.6 (k = 5, the W2 line's measured acceptance). Decode time per step:
  serial  = gpu_step + sum_l misses_l x t_expert + 40 x t_handoff
  overlap = sum_l max(gpu_step / 40, misses_l x t_expert) + 40 x t_handoff      (CPU hidden behind the GPU's own layer)
Statistics over decode positions only (the sequences' assistant / completion spans), test half of the sequences.
Gate G2: predicted single-stream decode >= 40 tok/s (above the W2 quality profile's 26-38), aim >= 59.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys

import numpy as np

E, L, K = 384, 40, 6
PACK_MB = 4.808704
T_EXPERT_US = {1: 89.7, 2: 92.0, 3: 95.8, 4: 99.5, 5: 103.0, 6: 106.4, 7: 115.0, 8: 137.0}
T_HANDOFF_US = 20.0
TOKENS_PER_STEP = {0: 1.0, 3: 2.6, 5: 3.6}
HERE = os.path.dirname(os.path.abspath(__file__))


def load_core():
    so = os.path.join(HERE, "libcache_sim_core.so")
    src = os.path.join(HERE, "cache_sim_core.c")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", so, src])
    lib = ctypes.CDLL(so)
    lib.sim_cache.restype = ctypes.c_int
    lib.sim_cache.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    return lib


def t_expert(m: int) -> float:
    if m in T_EXPERT_US:
        return T_EXPERT_US[m]
    return T_EXPERT_US[8] + 17.0 * (m - 8)


def simulate(lib, ids: np.ndarray, cap: np.ndarray, init: np.ndarray, policy: int, group: int):
    T = ids.shape[0]
    S = (T + group - 1) // group
    acc = np.zeros((S, L), dtype=np.int16)
    miss = np.zeros((S, L), dtype=np.int16)
    ids = np.ascontiguousarray(ids, dtype=np.int16)
    cap = np.ascontiguousarray(cap, dtype=np.int32)
    init = np.ascontiguousarray(init, dtype=np.uint8)
    rc = lib.sim_cache(ids.ctypes.data, T, L, K, E, group, cap.ctypes.data, init.ctypes.data, policy,
                       acc.ctypes.data, miss.ctypes.data)
    assert rc == S, rc
    return acc, miss


def placement(freq: np.ndarray, resident: float, alloc: str) -> tuple[np.ndarray, np.ndarray]:
    """freq [L, E] -> (init [L, E] uint8, cap [L]) with resident x L x E slices in total."""
    total = int(round(resident * L * E))
    init = np.zeros((L, E), dtype=np.uint8)
    if alloc == "uniform":
        per = total // L
        for l in range(L):
            init[l, np.argsort(-freq[l], kind="stable")[:per]] = 1
    else:
        flat = np.argsort(-freq.ravel(), kind="stable")[:total]
        init.ravel()[flat] = 1
    cap = init.sum(axis=1).astype(np.int32)
    return init, cap


def step_time(miss_row: np.ndarray, group: int, gpu_step_ms: float) -> tuple[float, float]:
    te = t_expert(group) * 1e-3  # ms; every position of the step routes to the expert at most once -> <= group tokens
    cpu_l = miss_row.astype(np.float64) * te
    serial = gpu_step_ms + cpu_l.sum() + L * T_HANDOFF_US * 1e-3
    overlap = np.maximum(gpu_step_ms / L, cpu_l).sum() + L * T_HANDOFF_US * 1e-3
    return serial, overlap


def synthetic_routes(n_seq: int = 40, T: int = 2048, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {"kinds": np.array(["code"] * n_seq), "seq_ids": np.array([f"syn/{i}" for i in range(n_seq)]),
           "tokens": np.array([T] * n_seq)}
    # Zipf-ish per-layer popularity + per-sequence affinity (a subset of experts preferred)
    base = 1.0 / np.arange(1, E + 1) ** 0.6
    for i in range(n_seq):
        ids = np.empty((T, L, K), dtype=np.int16)
        for l in range(L):
            perm = rng.permutation(E)
            p = base[perm].copy()
            fav = rng.choice(E, 60, replace=False)
            p[fav] *= 4.0
            p /= p.sum()
            for t in range(T):
                ids[t, l] = rng.choice(E, K, replace=False, p=p)
        out[f"ids_{i}"] = ids
        out[f"decode_{i}"] = np.ones(T, dtype=bool)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default="")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--resident", default="0.25,0.28,0.31")
    ap.add_argument("--alloc", default="uniform,global")
    ap.add_argument("--k", default="0,3,5")
    ap.add_argument("--gpu-step-ms", type=float, default=14.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    lib = load_core()
    if args.synthetic:
        R = synthetic_routes(n_seq=args.synthetic * 20)
    else:
        R = dict(np.load(args.routes, allow_pickle=False))
    kinds = R["kinds"]
    n = len(kinds)
    seqs = [(i, str(kinds[i]), R[f"ids_{i}"], R[f"decode_{i}"].astype(bool)) for i in range(n)]
    # train / test halves by index parity within kind
    train, test = [], []
    per_kind = {}
    for s in seqs:
        c = per_kind.setdefault(s[1], 0)
        (train if c % 2 == 0 else test).append(s)
        per_kind[s[1]] = c + 1
    freq = np.zeros((L, E), dtype=np.int64)
    for _, _, ids, _ in train:
        for l in range(L):
            freq[l] += np.bincount(ids[:, l].ravel().astype(np.int64), minlength=E)
    print(f"{n} sequences ({dict(per_kind)}); train {len(train)} / test {len(test)}; "
          f"{sum(s[2].shape[0] for _, _, s, _ in [(0, 0, x[2], 0) for x in test])} test positions")
    # routing flatness: experts needed for 80/90/95 % of the picks (train)
    cum = np.cumsum(np.sort(freq.ravel())[::-1]) / freq.sum()
    print(f"  (layer, expert) pairs for 80/90/95 % of routed picks: "
          + " / ".join(f"{int(np.searchsorted(cum, q) + 1) / (L * E) * 100:.0f} %" for q in (0.8, 0.9, 0.95)))
    results = []
    for resident in [float(x) for x in args.resident.split(",")]:
        for alloc in args.alloc.split(","):
            init, cap = placement(freq, resident, alloc)
            for policy, pname in ((0, "static"), (1, "lru")):
                for k in [int(x) for x in args.k.split(",")]:
                    group = 1 + k
                    by_kind = {}
                    for i, kind, ids, dmask in test:
                        acc, miss = simulate(lib, ids, cap, init, policy, group)
                        S = acc.shape[0]
                        dsteps = np.array([dmask[s * group:(s + 1) * group].all() for s in range(S)])
                        if dsteps.sum() == 0:
                            continue
                        a, m = acc[dsteps], miss[dsteps]
                        times = np.array([step_time(m[j], group, args.gpu_step_ms) for j in range(m.shape[0])])
                        d = by_kind.setdefault(kind, dict(acc=[], miss=[], serial=[], overlap=[]))
                        d["acc"].append(a.sum(axis=1).mean())
                        d["miss"].append(m.sum(axis=1).mean())
                        d["serial"].append(times[:, 0].mean())
                        d["overlap"].append(times[:, 1].mean())
                    tps = TOKENS_PER_STEP.get(k, 1.0)
                    row = dict(resident=resident, alloc=alloc, policy=pname, k=k, kinds={})
                    line = []
                    for kind, d in by_kind.items():
                        acc_m, miss_m = float(np.mean(d["acc"])), float(np.mean(d["miss"]))
                        ser, ovl = float(np.mean(d["serial"])), float(np.mean(d["overlap"]))
                        row["kinds"][kind] = dict(accesses=acc_m, misses=miss_m, hit=1 - miss_m / max(acc_m, 1e-9),
                                                  ms_serial=ser, ms_overlap=ovl, toks_serial=tps / ser * 1e3,
                                                  toks_overlap=tps / ovl * 1e3)
                        line.append(f"{kind}: hit {100 * (1 - miss_m / max(acc_m, 1e-9)):4.1f}% miss/step {miss_m:6.1f} "
                                    f"-> {tps / ser * 1e3:5.1f}..{tps / ovl * 1e3:5.1f} tok/s")
                    print(f"resident {resident:.2f} {alloc:7s} {pname:6s} k={k}: " + " | ".join(line))
                    results.append(row)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
