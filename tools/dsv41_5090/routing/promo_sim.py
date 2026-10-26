"""Promote-on-miss policies of the hybrid's placement (static residents + spare slots, cpu_tier/placement_manager.py)
on the routing traces: pair hit rate, CPU experts per step, promotions per step (= H2D DMA of 4.7 MB each per rank;
the promotions' DMA costs the CPU kernel ~15 % in situ, 10-25), and a decode-step estimate.

    python promo_sim.py --routes internal/dsv41-5090-cpu/routes/routes.npz \
        --placement internal/dsv41-5090-cpu/routes/placement-0.28-uniform.json [--spare 13] [--cap 4] [--k 0]
        [--policies 0,1:8,1:16,1:32,2:16] [--caps 4,2,1] [--trade 0,5,10] [--streams 4]

The test half of the sequences (index parity within kind, as cache_sim.py / placement.py), their decode positions,
streamed one request after another (the spare slots carry over, as in the server). --trade N moves N static slots
per layer into spare slots (the static set loses its N least frequent residents). --streams N = N concurrent
sequences (continuous batching): a step carries one position of each, the distinct missing experts per step grow
and some get m > 1 tokens (10-26, for the 4-stream soak).
Decode-step estimate (ms) = sum over layers of max(0.4 gpu_layer, cpu_l) + 0.6 gpu_layer + hook, with
cpu = misses x t_expert + extra tokens x t_extra; t_expert and the GPU layer time are the measured in-situ numbers
(--t-expert 0.185 ms, --t-extra 0.075, --gpu-layer 0.24 ms, --hook 1.2 ms). tok/s = streams / step.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys

import numpy as np

E, L, K = 384, 40, 6
HERE = os.path.dirname(os.path.abspath(__file__))


def load_core():
    so = os.path.join(HERE, "libpromo_sim_core.so")
    src = os.path.join(HERE, "promo_sim_core.c")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", so, src])
    lib = ctypes.CDLL(so)
    lib.promo_sim.restype = ctypes.c_int
    lib.promo_sim.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return lib


def test_stream(routes: str):
    R = dict(np.load(routes, allow_pickle=False))
    kinds = R["kinds"]
    index = R["seq_index"] if "seq_index" in R else np.arange(len(kinds))
    per_kind, test, train_ids = {}, [], []
    for p, j in enumerate(index):
        kind = str(kinds[p])
        c = per_kind.setdefault(kind, 0)
        per_kind[kind] = c + 1
        ids, dmask = R[f"ids_{j}"], R[f"decode_{j}"].astype(bool)
        if c % 2 == 1:
            test.append((kind, ids[dmask]))
        else:
            train_ids.append(ids)
    stream = np.concatenate([x[1] for x in test], axis=0).astype(np.int16)
    bounds = np.cumsum([0] + [x[1].shape[0] for x in test])
    kinds_of = [x[0] for x in test]
    freq = np.zeros((L, E), dtype=np.int64)
    for ids in train_ids:
        for l in range(L):
            freq[l] += np.bincount(ids[:, l].ravel().astype(np.int64), minlength=E)
    return stream, bounds, kinds_of, freq


def interleave(stream: np.ndarray, bounds: np.ndarray, n: int) -> np.ndarray:
    """The decode stream of a server with n concurrent sequences (continuous batching, --max-num-seqs n): n slots, each
    walking one test sequence position by position and taking the next sequence when its own ends; a step = one
    position per busy slot, in slot order. The tail (fewer than n sequences left) is padded with -1 rows (the core
    skips ids < 0), so every step has exactly n rows and `group = n` groups them."""
    seqs = [stream[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    nxt = 0
    cur = []   # per slot: [sequence index, position]
    for _ in range(min(n, len(seqs))):
        cur.append([nxt, 0]); nxt += 1
    steps = []
    pad = np.full(stream.shape[1:], -1, dtype=stream.dtype)
    while cur:
        rows = []
        for slot in cur:
            rows.append(seqs[slot[0]][slot[1]])
            slot[1] += 1
        for slot in cur:
            if slot[1] >= len(seqs[slot[0]]):
                if nxt < len(seqs):
                    slot[0], slot[1] = nxt, 0; nxt += 1
                else:
                    slot[0] = -1
        cur = [s for s in cur if s[0] >= 0]
        while len(rows) < n:
            rows.append(pad)
        steps.extend(rows)
    return np.stack(steps, axis=0)


def init_from_placement(path: str, freq: np.ndarray, trade: int) -> np.ndarray:
    P = json.load(open(path))
    init = np.zeros((L, E), dtype=np.uint8)
    for lay in P["layers"]:
        l = int(lay["layer"])
        res = [int(e) for e in lay["resident"]]
        if trade > 0:
            # drop the trade least frequent residents (training frequencies) in favour of spare slots
            res = sorted(res, key=lambda e: -freq[l, e])[: max(0, len(res) - trade)]
        init[l, res] = 1
    return init


def run(lib, stream, init, spare, cap, policy, window, group):
    T = stream.shape[0]
    S = (T + group - 1) // group
    out = np.zeros((S, 4), dtype=np.int32)
    rc = lib.promo_sim(np.ascontiguousarray(stream).ctypes.data, T, L, K, E, group, np.ascontiguousarray(init).ctypes.data,
                       spare, cap, policy, window, out.ctypes.data)
    assert rc == S, rc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default=os.path.join(HERE, "..", "..", "..", "internal", "dsv41-5090-cpu", "routes", "routes.npz"))
    ap.add_argument("--placement", default=os.path.join(HERE, "..", "..", "..", "internal", "dsv41-5090-cpu", "routes", "placement-0.28-uniform.json"))
    ap.add_argument("--spare", type=int, default=13)
    ap.add_argument("--k", default="0")
    ap.add_argument("--policies", default="0,1:8,1:16,1:32,1:64,2:16,2:32")
    ap.add_argument("--caps", default="4,2,1")
    ap.add_argument("--trade", default="0")
    ap.add_argument("--streams", type=int, default=1, help="concurrent sequences per decode step (continuous batching; k must "
                    "be 0): the soak's 4 streams = 4 tokens per step from 4 sequences")
    ap.add_argument("--t-expert", type=float, default=0.185, help="ms per CPU expert in situ (10-25: 0.16 static, 0.185 promote)")
    ap.add_argument("--t-extra", type=float, default=0.075, help="ms per extra token of an expert (m > 1; bench: 111 us at m = 1, "
                    "183 at 2, 486 at 6 -> ~73 us per token)")
    ap.add_argument("--gpu-layer", type=float, default=0.24, help="ms of GPU per layer at k = 0 (9.6 ms step / 40)")
    ap.add_argument("--hook", type=float, default=1.2, help="ms of host hook per step")
    ap.add_argument("--dma-ms-per-promotion", type=float, default=0.19, help="4.7 MB at ~25 GB/s per link")
    args = ap.parse_args()
    lib = load_core()
    stream, bounds, kinds_of, freq = test_stream(args.routes)
    print(f"test stream: {stream.shape[0]} decode positions from {len(bounds) - 1} sequences "
          f"({', '.join(f'{k}: {kinds_of.count(k)}' for k in sorted(set(kinds_of)))})")
    if args.streams > 1:
        assert args.k == "0", "--streams models concurrent sequences at k = 0"
        stream = interleave(stream, bounds, args.streams)
        print(f"{args.streams} concurrent streams: {stream.shape[0] // args.streams} steps of {args.streams} positions")
    print(f"{'policy':14s} {'cap':>3s} {'spare':>5s} {'static':>6s} | {'hit rate':>8s} {'cpu exp/step':>12s} {'tok/exp':>7s} {'promo/step':>10s} {'DMA MB/step':>11s} | "
          f"{'cpu ms':>6s} {'step ms':>7s} {'tok/s':>6s}")
    for k in [int(x) for x in args.k.split(",")]:
        group = (1 + k) * args.streams
        for trade in [int(x) for x in args.trade.split(",")]:
            init = init_from_placement(args.placement, freq, trade)
            spare = args.spare + trade
            nstatic = int(init[0].sum())
            # the static placement alone (spare 0) as the reference
            configs = [("static", 0, 0, 0, 0)]
            for pol in args.policies.split(","):
                p, _, w = pol.partition(":")
                for cap in [int(c) for c in args.caps.split(",")]:
                    configs.append((f"p{p}" + (f" w{w}" if w else ""), int(p), int(w or 0), cap, spare))
            for name, policy, window, cap, sp in configs:
                out = run(lib, stream, init, sp, cap, policy, window, group)
                pairs, hits, misses, promos = (out[:, i].astype(np.float64) for i in range(4))
                hit_rate = hits.sum() / max(1, pairs.sum())
                # the CPU work of a step: every distinct missing expert costs t_expert (its 4.7 MB slice from DRAM) plus
                # t_extra per further token routed to it in the same step (m > 1: the multi-stream / DSpark case);
                # missed pairs - distinct misses = the extra tokens
                extra = np.maximum(0.0, (pairs - hits) - misses)
                cpu = misses * args.t_expert + extra * args.t_extra
                tok_per_exp = (pairs - hits).sum() / max(1.0, misses.sum())
                # per-step time: the misses are spread over the layers; use the mean per layer (the per-layer
                # distribution is not in the output) -> step = sum_l max(gpu_layer, c_l) ~ 40 * max(gpu_layer, cpu/40)
                # is too optimistic; use the serial-after-GPU-share model of cache_sim: 0.4 of the layer hides the CPU
                c_l = cpu / L
                step = L * (np.maximum(args.gpu_layer * 0.4, c_l) + args.gpu_layer * 0.6) + args.hook
                step_ms = float(np.mean(step))
                print(f"{name:14s} {cap:3d} {sp:5d} {nstatic:6d} | {hit_rate:8.3f} {misses.mean():12.1f} {tok_per_exp:7.2f} {promos.mean():10.1f} "
                      f"{promos.mean() * 4.7:11.0f} | {cpu.mean():6.2f} {step_ms:7.2f} {args.streams * 1000 / step_ms:6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
