"""Two-level expert residency on the routing traces (the 256 / 128 GB tiers of the hybrid): VRAM (the static placement +
LRU slots filled by promote-on-miss with the admission window) -> a DRAM LRU of N records per layer per rank -> NVMe.
Decode positions only, the sequences concatenated in seq_index order (a long-running server). Reports, per N: the CPU
experts per step (unchanged by N), the NVMe record fetches per step (mean, share of steps with any, p99), the layers per
step with a fetch, and a decode-step estimate; then the prefill picture: the distinct non-static experts of a 4096-token
chunk and how many of them the DRAM LRU of a decode-warm server holds (the rest stream from NVMe).

    python tools/dsv41_5090/routing/nvme_tier_sim.py [--routes ...] [--placement ...] [--spare 70]
        [--window 12] [--cap 4] [--n 344,320,280,240,230,200,160,120,110,100] [--t-expert 0.13] [--t-nvme 0.30,0.45,0.80]
        [--step-ms 7.5]

Step estimate: the CPU time of a layer with k CPU experts of which j come from NVMe = max((k - j) t_expert, t_nvme(j)) +
j t_expert (the DRAM-resident experts are computed while the records are in flight; t_nvme(j) grows with the records in
flight on the array: --t-nvme gives it for 1 / 2-4 / 5+ records of this rank x 4 ranks); the step pays the sum over layers
of the exposed part beyond today's (k t_expert), on top of today's 7.5 ms.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

E, L, K = 384, 40, 6
REC = 4700160


class LRU:
    __slots__ = ("n", "d")

    def __init__(self, n):
        self.n, self.d = n, OrderedDict()

    def hit(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return True
        return False

    def add(self, k):
        self.d[k] = 1
        self.d.move_to_end(k)
        if len(self.d) > self.n:
            self.d.popitem(last=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default=os.path.join(_HERE, "..", "..", "..", "internal", "dsv41-5090-cpu", "routes", "routes.npz"))
    ap.add_argument("--placement", default=os.path.join(_HERE, "..", "..", "..", "internal", "dsv41-5090-cpu", "routes", "placement-0.1042-uniform.json"))
    ap.add_argument("--spare", type=int, default=70)
    ap.add_argument("--window", type=int, default=12, help="promotion admission window (0 = every miss)")
    ap.add_argument("--cap", type=int, default=4, help="promotions per layer per step")
    ap.add_argument("--n", default="344,320,280,240,230,200,160,120,110,100")
    ap.add_argument("--t-expert", type=float, default=0.13, help="ms per CPU expert")
    ap.add_argument("--t-nvme", default="0.30,0.45,0.80", help="ms for a record read with 1 / 2-4 / 5+ records in flight per rank")
    ap.add_argument("--step-ms", type=float, default=7.5)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    R = dict(np.load(args.routes, allow_pickle=False))
    P = json.load(open(args.placement))
    static = [set(int(e) for e in l["resident"]) for l in sorted(P["layers"], key=lambda l: int(l["layer"]))]
    order = np.argsort(R["seq_index"])
    dec = np.concatenate([R[f"ids_{i}"][R[f"decode_{i}"].astype(bool)] for i in order]).astype(np.int32)
    if args.max_steps:
        dec = dec[: args.max_steps]
    steps = dec.shape[0]
    print(f"decode steps {steps} (sequences {len(order)}); VRAM = {len(static[0])} static + {args.spare} LRU slots per layer, "
          f"promotion window {args.window}, cap {args.cap}; DRAM LRU of N records per layer per rank")
    t_nv = [float(x) for x in args.t_nvme.split(",")]

    def t_nvme(j):
        return 0.0 if j == 0 else t_nv[0] if j == 1 else t_nv[1] if j <= 4 else t_nv[2]

    Ns = [int(x) for x in args.n.split(",")]
    # the VRAM policy is independent of N: run it once and record the CPU experts per (step, layer)
    vr = [LRU(args.spare) for _ in range(L)]
    last_miss = [dict() for _ in range(L)]
    cpu_sets = []   # per step: list over layers of the set of CPU experts
    for s in range(steps):
        row = []
        for l in range(L):
            cpu = []
            promo = 0
            for e in set(int(x) for x in dec[s, l]):
                if e in static[l] or vr[l].hit(e):
                    continue
                cpu.append(e)
                lm = last_miss[l].get(e, -10**9)
                admit = args.window == 0 or (s - lm) <= args.window
                last_miss[l][e] = s
                if admit and promo < args.cap:
                    vr[l].add(e)
                    promo += 1
            row.append(cpu)
        cpu_sets.append(row)
    cpu_per_step = np.array([sum(len(c) for c in row) for row in cpu_sets])
    hit = 1.0 - cpu_per_step.sum() / (steps * L * K)
    print(f"VRAM: pair hit rate {hit:.3f}, CPU experts per step {cpu_per_step.mean():.2f} (p99 {np.percentile(cpu_per_step, 99):.0f}), "
          f"layers with a miss per step {np.mean([sum(1 for c in row if c) for row in cpu_sets]):.2f}")
    print(f"{'N/layer':>8} {'GB/rank':>8} {'NVMe/step':>10} {'steps w/ fetch':>14} {'p99/step':>9} {'layers w/ fetch':>15} {'+ms/step':>9} {'est tok/s':>9}")
    dram_snapshots = {}
    for N in Ns:
        dr = [LRU(N) for _ in range(L)]
        nv = np.zeros(steps, np.int32)
        lay = np.zeros(steps, np.int32)
        extra = np.zeros(steps)
        for s in range(steps):
            for l, cpu in enumerate(cpu_sets[s]):
                if not cpu:
                    continue
                j = 0
                for e in cpu:
                    if not dr[l].hit(e):
                        j += 1
                        dr[l].add(e)
                k = len(cpu)
                if j:
                    nv[s] += j
                    lay[s] += 1
                    cost = max((k - j) * args.t_expert, t_nvme(j)) + j * args.t_expert
                    extra[s] += cost - k * args.t_expert
        gb = N * L * REC / 1e9
        add = extra.mean()
        print(f"{N:>8} {gb:>8.1f} {nv.mean():>10.3f} {100 * (nv > 0).mean():>13.1f}% {np.percentile(nv, 99):>9.0f} {lay.mean():>15.2f} "
              f"{add:>9.2f} {1000 / (args.step_ms + add):>9.1f}")
        dram_snapshots[N] = [set(d.d.keys()) for d in dr]
    # prefill: the distinct non-static experts of a 4096-token chunk; the share the decode-warm DRAM LRU holds
    print("\nprefill chunks of 4096 tokens (sequences >= 4096 tokens, all positions): distinct non-static experts per rank per chunk,"
          " and how many the DRAM LRU (as left by the decode stream above) holds")
    chunks = []
    for i in order:
        ids = R[f"ids_{i}"]
        for c in range(0, ids.shape[0] - 4095, 4096):
            ch = ids[c:c + 4096]
            chunks.append([set(int(x) for x in np.unique(ch[:, l])) - static[l] for l in range(L)])
    if not chunks:
        print("  no sequence of >= 4096 tokens in the traces")
        return 0
    tot = np.array([sum(len(c) for c in ch) for ch in chunks])
    print(f"  chunks {len(chunks)}: records per chunk per rank mean {tot.mean():.0f} (min {tot.min()}, max {tot.max()}) of {344 * L} = "
          f"{tot.mean() * REC / 1e9:.1f} GB per rank, {4 * tot.mean() * REC / 1e9:.0f} GB for 4 ranks")
    print(f"{'N/layer':>8} {'in DRAM':>8} {'from NVMe':>10} {'NVMe GB/rank/chunk':>19} {'NVMe s/chunk (4 ranks @ 42 GB/s)':>34} {'est. prefill x slower':>22}")
    for N in Ns:
        snap = dram_snapshots[N]
        in_dram = np.array([sum(len(c & snap[l]) for l, c in enumerate(ch)) for ch in chunks])
        cold = tot - in_dram
        gb = cold.mean() * REC / 1e9
        secs = 4 * gb / 42.0
        # today's chunk: ~1.0 s (4.3-4.8K tok/s); the cold records' NVMe time adds to it when not overlapped
        print(f"{N:>8} {in_dram.mean():>8.0f} {cold.mean():>10.0f} {gb:>19.1f} {secs:>34.2f} {(1.0 + secs) / 1.0:>22.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
