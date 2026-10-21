"""Static expert placement for the hybrid from the native routing traces (phase 2.2).

    python placement.py --routes routes.npz [--resident 0.28] [--alloc uniform|global] [--out placement.json]

The same train half as cache_sim.py (index parity within kind), per-layer routing frequencies, and cache_sim's
placement(): `uniform` keeps the same number of experts per layer (the most frequent ones), `global` the most
frequent (layer, expert) pairs overall. Output: per layer the resident experts in frequency order (slot i = the i-th
one), the per-layer counts, the train-half hit rate of the static set over all positions and over decode positions,
and the test-half hit rate (what the simulator reported as "static").
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sim as CS  # noqa: E402

L, E, K = CS.L, CS.E, CS.K


def split(R: dict):
    kinds = R["kinds"]
    index = R["seq_index"] if "seq_index" in R else np.arange(len(kinds))
    seqs = [(int(j), str(kinds[p]), R[f"ids_{j}"], R[f"decode_{j}"].astype(bool)) for p, j in enumerate(index)]
    train, test, per_kind = [], [], {}
    for s in seqs:
        c = per_kind.setdefault(s[1], 0)
        (train if c % 2 == 0 else test).append(s)
        per_kind[s[1]] = c + 1
    return train, test


def frequencies(seqs) -> np.ndarray:
    freq = np.zeros((L, E), dtype=np.int64)
    for _, _, ids, _ in seqs:
        for l in range(L):
            freq[l] += np.bincount(ids[:, l].ravel().astype(np.int64), minlength=E)
    return freq


def hit_rate(seqs, init: np.ndarray, decode_only: bool) -> float:
    hits = total = 0
    for _, _, ids, dec in seqs:
        sel = ids[dec] if decode_only else ids
        if sel.shape[0] == 0:
            continue
        for l in range(L):
            picks = sel[:, l].ravel().astype(np.int64)
            hits += int(init[l][picks].sum())
            total += picks.size
    return hits / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", required=True)
    ap.add_argument("--resident", type=float, default=0.28)
    ap.add_argument("--alloc", default="uniform")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    R = dict(np.load(args.routes, allow_pickle=False))
    train, test = split(R)
    freq = frequencies(train)
    init, cap = CS.placement(freq, args.resident, args.alloc)
    layers = []
    for l in range(L):
        order = np.argsort(-freq[l], kind="stable")
        resident = [int(e) for e in order if init[l, e]]
        layers.append(dict(layer=l, resident=resident, count=len(resident)))
    out = dict(routes=os.path.abspath(args.routes), resident_fraction=args.resident, alloc=args.alloc,
               train_sequences=len(train), test_sequences=len(test), slices_total=int(init.sum()),
               per_layer_min=int(cap.min()), per_layer_max=int(cap.max()),
               hit_rate_train_all=hit_rate(train, init, False), hit_rate_train_decode=hit_rate(train, init, True),
               hit_rate_test_all=hit_rate(test, init, False), hit_rate_test_decode=hit_rate(test, init, True),
               layers=layers)
    print(f"resident {args.resident} {args.alloc}: {out['slices_total']} slices per rank "
          f"({out['slices_total'] * 4.70016 / 1e3:.1f} GB in b12x's layout), per layer {cap.min()}..{cap.max()}; "
          f"static hit rate train {out['hit_rate_train_all']:.3f} (decode {out['hit_rate_train_decode']:.3f}), "
          f"test {out['hit_rate_test_all']:.3f} (decode {out['hit_rate_test_decode']:.3f})")
    path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.routes)), f"placement-{args.resident}-{args.alloc}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
