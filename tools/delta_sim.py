#!/usr/bin/env python3
"""DeltaSim — offline replay of a captured routing trace through pluggable
FP4-delta promotion/eviction policies.

The trace (.npy of [tick, layer, expert] rows, captured by
VLLM_MOE_W2_DELTA_CAPTURE) is the policy-INDEPENDENT input: which experts the
model routed each manager tick. We replay it through the current manager logic
(baseline) and candidate redesigns, scoring each on the metric the delta tier
actually exists to maximize:

    hit-rate = fraction of routed (layer,expert) activations that were already
               resident in FP4 at the moment the forward read the slot table.

Faithfulness to the real manager (validated in round 0):
  - candidates considered in (layer,expert)-sorted order, <=PROMOTE/tick;
  - eviction only of slots cold >= EVICT_FLOOR ticks and not active this tick;
  - hit-rate measured BEFORE this tick's promotions (the forward saw the
    previous state). Cold-age uses the real captured tick (gaps preserved).
"""
import sys
from collections import Counter, OrderedDict, defaultdict

import numpy as np


def load_trace(path):
    arr = np.load(path)
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    uniq, starts = np.unique(arr[:, 0], return_index=True)
    starts = list(starts) + [len(arr)]
    frames = []
    for k in range(len(uniq)):
        block = arr[starts[k]:starts[k + 1], 1:3]
        block = block[np.lexsort((block[:, 1], block[:, 0]))]
        frames.append((int(uniq[k]), [(int(l), int(e)) for l, e in block]))
    return frames


def _order_candidates(cands, strat, freq):
    if strat == "sorted":            # baseline: (layer,expert) order -> L0 first
        return cands
    if strat == "freq":              # globally hottest first
        return sorted(cands, key=lambda p: -freq[p])
    if strat == "roundrobin":        # one per layer per round -> fair across layers
        by = OrderedDict()
        for p in cands:
            by.setdefault(p[0], []).append(p)
        out = []
        while any(by.values()):
            for l in list(by):
                if by[l]:
                    out.append(by[l].pop(0))
        return out
    raise ValueError(strat)


def simulate(frames, n_slots=170, promote=8, evict_floor=2,
             cand="sorted", evict="lru", promote_threshold=1,
             per_layer_cap=None, warmup_frac=0.15):
    """Replay `frames` through a policy. Returns a metrics dict.

    cand: candidate ordering — "sorted" | "roundrobin" | "freq"
    evict: victim choice — "lru" (coldest) | "lfu" (least-frequent) | "lfu_aged"
    promote_threshold: only promote an expert after it's been seen this many
        times (hysteresis against one-hit-wonders).
    per_layer_cap: optional max slots any single layer may hold (fairness cap).
    """
    mirror = {}                       # (l,e) -> slot
    owner = {}                        # slot -> [l, e, last_seen_tick]
    free = list(range(n_slots))
    freq = defaultdict(int)
    layer_count = Counter()           # cached experts per layer
    promotions = evictions = hits = active = 0
    colds = []
    warmup = int(len(frames) * warmup_frac)

    def victim(seen_set, tick):
        best, best_key = None, None
        for s, (l, e, ls) in owner.items():
            if (l, e) in seen_set or tick - ls < evict_floor:
                continue
            if evict == "lru":
                key = (ls,)
            elif evict == "lfu":
                key = (freq[(l, e)], ls)
            elif evict == "lfu_aged":
                key = (freq[(l, e)] - 0.01 * tick, ls)   # decay favors recent
            else:
                raise ValueError(evict)
            if best_key is None or key < best_key:
                best, best_key = s, key
        return best

    for fi, (tick, frame) in enumerate(frames):
        scoring = fi >= warmup
        seen_set = set(frame)
        # hit-rate as the forward saw it (before this tick's promotions)
        if scoring:
            for p in frame:
                active += 1
                if p in mirror:
                    hits += 1
        cands = []
        for p in frame:
            freq[p] += 1
            if p in mirror:
                owner[mirror[p]][2] = tick
            elif freq[p] >= promote_threshold:
                cands.append(p)
        n = 0
        for p in _order_candidates(cands, cand, freq):
            if n >= promote:
                break
            if per_layer_cap is not None and layer_count[p[0]] >= per_layer_cap:
                continue                      # hard fairness cap per layer
            if free:
                slot = free.pop()
            else:
                slot = victim(seen_set, tick)
                if slot is None:
                    break
                vl, ve, vls = owner[slot]
                del mirror[(vl, ve)]
                layer_count[vl] -= 1
                if scoring:
                    evictions += 1
                    colds.append(tick - vls)
            mirror[p] = slot
            owner[slot] = [p[0], p[1], tick]
            layer_count[p[0]] += 1
            n += 1
            if scoring:
                promotions += 1

    pl = Counter(l for (l, e) in mirror)
    cps = np.percentile(colds, [50, 90, 99]) if colds else [0, 0, 0]
    return dict(
        hit_rate=round(100 * hits / max(active, 1), 2),
        promotions=promotions, evictions=evictions,
        coverage=round(len(mirror) / n_slots, 3),
        layers_covered=len(pl),
        l0_share=round(100 * pl.get(0, 0) / max(len(mirror), 1), 1),
        cold_p50=int(cps[0]), cold_p90=int(cps[1]),
        top_layers=pl.most_common(6), n_frames=len(frames),
    )


def run(path, policies):
    frames = load_trace(path)
    tot = sum(len(f) for _, f in frames)
    print(f"trace: {len(frames)} frames, {tot} activations, "
          f"ticks {frames[0][0]}..{frames[-1][0]}\n")
    cols = ["policy", "hit%", "promo", "evict", "cov", "L#", "L0%", "cP50", "cP90"]
    w = [22, 7, 8, 8, 6, 4, 6, 6, 6]
    print("".join(c.ljust(x) for c, x in zip(cols, w)))
    print("-" * sum(w))
    results = {}
    for name, pol in policies.items():
        m = simulate(frames, **pol)
        results[name] = m
        row = [name, m["hit_rate"], m["promotions"], m["evictions"],
               m["coverage"], m["layers_covered"], m["l0_share"],
               m["cold_p50"], m["cold_p90"]]
        print("".join(str(v).ljust(x) for v, x in zip(row, w)))
    return results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/profiles/route_trace.npy"
    run(path, {"baseline": dict()})
