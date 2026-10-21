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


GPU_MOE_SHARE = 0.4   # share of a layer's GPU time spent in the expert GEMMs after the router (what the CPU can hide behind)


# the GPU step grows with the DSpark verify window: measured on 4x PRO 6000 TP4 native 9.4 ms (no drafter) ->
# 15.1 ms (k = 5), i.e. x1.6; k = 3 interpolated. --spec-factor overrides.
SPEC_STEP_FACTOR = {0: 1.0, 3: 1.35, 5: 1.6}


def step_time(miss_row: np.ndarray, group: int, gpu_step_ms: float) -> tuple[float, float, float]:
    """(serial, mid, overlap) ms per step. serial: no overlap at all; overlap: the CPU hides behind the whole GPU
    layer (needs the routing one layer ahead); mid: the CPU starts after the layer's router and hides only behind
    the GPU's own expert GEMMs (GPU_MOE_SHARE of the layer), the rest of the layer is serial."""
    te = t_expert(group) * 1e-3  # ms; every position of the step routes to the expert at most once -> <= group tokens
    cpu_l = miss_row.astype(np.float64) * te
    gpu_step_ms = gpu_step_ms * SPEC_STEP_FACTOR.get(group - 1, 1.0 + 0.1 * (group - 1))
    hand = L * T_HANDOFF_US * 1e-3
    per_layer = gpu_step_ms / L
    serial = gpu_step_ms + cpu_l.sum() + hand
    overlap = np.maximum(per_layer, cpu_l).sum() + hand
    mid = (np.maximum(per_layer * GPU_MOE_SHARE, cpu_l) + per_layer * (1 - GPU_MOE_SHARE)).sum() + hand
    return serial, mid, overlap


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
    ap.add_argument("--gpu-step-ms", type=float, default=14.0, help="miss-free GPU decode step at k = 0")
    ap.add_argument("--spec-factor", default="", help="k:factor,... GPU step multipliers for DSpark steps (default 3:1.35,5:1.6)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    lib = load_core()
    if args.spec_factor:
        for kv in args.spec_factor.split(","):
            kk, f = kv.split(":")
            SPEC_STEP_FACTOR[int(kk)] = float(f)
    if args.synthetic:
        R = synthetic_routes(n_seq=args.synthetic * 20)
    else:
        R = dict(np.load(args.routes, allow_pickle=False))
    kinds = R["kinds"]
    n = len(kinds)
    index = R["seq_index"] if "seq_index" in R else np.arange(n)
    seqs = [(int(j), str(kinds[p]), R[f"ids_{j}"], R[f"decode_{j}"].astype(bool)) for p, j in enumerate(index)]
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
          f"{sum(x[2].shape[0] for x in test)} test positions, {sum(int(x[3].sum()) for x in test)} of them decode")
    # routing flatness: experts needed for 80/90/95 % of the picks (train)
    cum = np.cumsum(np.sort(freq.ravel())[::-1]) / freq.sum()
    print(f"  (layer, expert) pairs for 80/90/95 % of routed picks: "
          + " / ".join(f"{int(np.searchsorted(cum, q) + 1) / (L * E) * 100:.0f} %" for q in (0.8, 0.9, 0.95)))
    # per-layer routing perplexity (train) and temporal locality / prefill working sets (test)
    p = freq / np.maximum(freq.sum(axis=1, keepdims=True), 1)
    ppl = np.exp(-(p * np.log(np.maximum(p, 1e-12))).sum(axis=1))
    print(f"  per-layer routing perplexity of {E}: min {ppl.min():.0f}, median {np.median(ppl):.0f}, max {ppl.max():.0f}")
    stats = {}
    for i, kind, ids, dmask in test:
        st = stats.setdefault(kind, dict(prev=[], prev8=[], chunk_ws=[]))
        d = ids[dmask] if dmask.any() else ids
        if d.shape[0] > 9:
            # share of a position's 6 picks that the previous position (and any of the previous 8) also routed
            for lag_name, lag in (("prev", 1), ("prev8", 8)):
                hits = 0
                for t in range(lag, d.shape[0]):
                    prev = d[t - lag:t]  # [lag, L, K]
                    cur = d[t]           # [L, K]
                    for l in range(L):
                        hits += np.isin(cur[l], prev[:, l].ravel()).sum()
                st[lag_name].append(hits / ((d.shape[0] - lag) * L * K))
        for c in range(0, ids.shape[0] - 2047, 2048):
            st["chunk_ws"].append(np.mean([np.unique(ids[c:c + 2048, l]).size for l in range(L)]))
    for kind, st in stats.items():
        line = f"  {kind:10s} decode positions: picks also routed at the previous position {np.mean(st['prev']) * 100:4.1f} %"
        if st["prev8"]:
            line += f", within the previous 8 positions {np.mean(st['prev8']) * 100:4.1f} %"
        if st["chunk_ws"]:
            line += f"; distinct experts per layer per 2048-token chunk {np.mean(st['chunk_ws']):.0f} of {E}"
        print(line)
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
                        d = by_kind.setdefault(kind, dict(acc=[], miss=[], serial=[], mid=[], overlap=[], w=[]))
                        d["acc"].append(a.sum(axis=1).mean())
                        d["miss"].append(m.sum(axis=1).mean())
                        d["serial"].append(times[:, 0].mean())
                        d["mid"].append(times[:, 1].mean())
                        d["overlap"].append(times[:, 2].mean())
                        d["w"].append(m.shape[0])
                    tps = TOKENS_PER_STEP.get(k, 1.0)
                    row = dict(resident=resident, alloc=alloc, policy=pname, k=k, tokens_per_step=tps, kinds={})
                    line = []
                    for kind, d in by_kind.items():
                        w = np.array(d["w"], dtype=np.float64)   # decode steps per sequence -> weight by steps
                        acc_m, miss_m = float(np.average(d["acc"], weights=w)), float(np.average(d["miss"], weights=w))
                        ser, mid, ovl = (float(np.average(d[x], weights=w)) for x in ("serial", "mid", "overlap"))
                        row["kinds"][kind] = dict(accesses=acc_m, misses=miss_m, hit=1 - miss_m / max(acc_m, 1e-9),
                                                  ms_serial=ser, ms_mid=mid, ms_overlap=ovl, toks_serial=tps / ser * 1e3,
                                                  toks_mid=tps / mid * 1e3, toks_overlap=tps / ovl * 1e3,
                                                  decode_steps=int(w.sum()))
                        line.append(f"{kind}: hit {100 * (1 - miss_m / max(acc_m, 1e-9)):4.1f}% miss/step {miss_m:6.1f} "
                                    f"-> {tps / ser * 1e3:5.1f} / {tps / mid * 1e3:5.1f} / {tps / ovl * 1e3:5.1f} tok/s")
                    print(f"resident {resident:.2f} {alloc:7s} {pname:6s} k={k}: " + " | ".join(line))
                    results.append(row)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
