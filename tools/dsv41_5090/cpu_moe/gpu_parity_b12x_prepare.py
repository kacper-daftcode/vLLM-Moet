"""Host side 1/3 of the CPU-vs-b12x parity check: the inputs file.

    python gpu_parity_b12x_prepare.py [--layer 20] [--rank 0] [--experts 8] [--out <parity dir>/b12x_inputs.npz]

Picks E real expert slices (checkpoint, TP4 rank slice), T = 6 tokens (the 5 recorded hidden states of inputs.npz +
one synthetic wide-range token), a routing that uses every expert with m = 1..5 pairs per expert, router weights, and
writes everything the GPU run needs in b12x's input format (w13 = [w1 (gate); w3 (up)] rows: measured on the GPU,
b12x's "w31" layout reads the FIRST half of w13 as the gate (the half its activation applies the sigmoid to) - the
order vLLM's w13 tensor has; block scales; w2 [E, 5120, 288] + scales) next to the host-layout records.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv41_b12x as B  # noqa: E402
import dsv41_mxfp4 as D1  # noqa: E402


def canonicalize(packed: np.ndarray) -> np.ndarray:  # vLLM's _canonicalize_fp4_zero_signs_ (-0 -> +0)
    mag = packed & 0x77
    nonzero = (mag | (mag >> 1) | (mag >> 2)) & 0x11
    return (packed & (0x77 | (nonzero << 3))).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--first-expert", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(B.TABLES_DIR, "b12x_inputs.npz"))
    ap.add_argument("--tokens", type=int, default=6, help="T <= 8 (b12x's compact micro path); 6 = the recorded five + one wide")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    E, TOPK = args.experts, 6
    raws = [D1.expert_slice_raw(args.layer, args.first_expert + e, args.rank) for e in range(E)]
    for r in raws:
        for k in ("w1n", "w3n", "w2n"):
            r[k] = canonicalize(r[k])
    records = np.stack([B.build_record(r) for r in raws])
    w13 = np.stack([np.concatenate([r["w1n"], r["w3n"]], axis=0) for r in raws])      # [E, 1152, 2560] gate first
    w13s = np.stack([np.concatenate([r["w1s"], r["w3s"]], axis=0) for r in raws])    # [E, 1152, 160]
    w2 = np.stack([r["w2n"] for r in raws])                                           # [E, 5120, 288]
    w2s = np.stack([r["w2s"] for r in raws])                                          # [E, 5120, 18]
    # tokens
    Z = np.load(os.path.join(B.TABLES_DIR, "inputs.npz"))
    x_rec = torch.from_numpy(Z["x_bf16"].astype(np.int16)).view(torch.bfloat16)
    g = torch.Generator().manual_seed(args.seed)
    x_wide = torch.randn(1, B.HID, generator=g)
    xg = x_wide.view(1, -1, 32)
    xg[:, :, 0] *= 400.0
    xg[:, :, 1:8] *= 1e-3
    xg[:, :, 8:12] *= 1e-5
    if args.tokens == 6:
        x = torch.cat([x_rec, x_wide.to(torch.bfloat16)], dim=0)
    else:
        # synthetic mix: realistic (outlier channels), wide-range groups, tiny, big, and recorded ones
        xs = [x_wide.to(torch.bfloat16)]
        flavors = ["realistic", "tiny", "big", "realistic", "wide", "realistic", "big"]
        for i in range(args.tokens - 1):
            f = flavors[i % len(flavors)]
            xi = torch.randn(1, B.HID, generator=g)
            if f == "realistic":
                heavy = torch.randperm(B.HID, generator=g)[:48]
                xi[:, heavy] *= 12.0
                xi[:, heavy[:4]] *= 8.0
            elif f == "wide":
                xg = xi.view(1, -1, 32)
                xg[:, :, 0] *= 300.0
                xg[:, :, 1:10] *= 1e-4
            elif f == "tiny":
                xi *= 1e-6
                xi[:, ::97] *= 3e5
            elif f == "big":
                xi *= 3e3
            xs.append(xi.to(torch.bfloat16))
        x = torch.cat(xs, dim=0)
    T = x.shape[0]
    # routing: every expert used, m from 1 to ~5; token t's top-6 = a rotation of the expert list
    ids = np.zeros((T, TOPK), dtype=np.int32)
    for t in range(T):
        perm = torch.randperm(E, generator=g).numpy()
        ids[t] = perm[:TOPK] if E >= TOPK else np.array([perm[i % E] for i in range(TOPK)])
    # make sure a few experts have m >= 3 and one has m = 1
    ids[0, :3] = [0, 1, 2]
    ids[1, :3] = [0, 1, 3]
    ids[2, :2] = [0, 1]
    weights = (torch.rand(T, TOPK, generator=g) * 0.9 + 0.1).numpy().astype(np.float32)
    m = np.bincount(ids.reshape(-1), minlength=E)
    np.savez(args.out, layer=args.layer, rank=args.rank, first_expert=args.first_expert, records=records, w13=w13,
             w13s=w13s, w2=w2, w2s=w2s, x_bf16=x.view(torch.int16).numpy(), ids=ids, weights=weights)
    print(f"wrote {args.out}: E={E} T={T} top-{TOPK}; pairs per expert {m.tolist()}; "
          f"{os.path.getsize(args.out) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
