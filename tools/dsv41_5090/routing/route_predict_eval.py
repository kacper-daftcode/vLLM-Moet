"""One-layer-ahead routing prediction from a hidden state the decode step already has (the lever ds41d uses for its
RAM -> VRAM prefetch; here: could the CPU tier pull layer L+1's missing experts into the CCD's L3 while layer L runs?).

The data are the eager-run MoE dumps of vllm_moet_cpu_tier.py (VLLM_MOET_CPU_TIER_DUMP_DIR=<dir>, DUMP_MAX large,
--enforce-eager): per target-layer MoE call rank{r}_{n:04d}.pt = dict(step, layer, T, x = the MoE input (the router's
input, bf16 as float), ids = the router's expert ids, slots = the layer's expert -> slot table at that step).

For every decode step and every pair of consecutive real layers (L, L+1) the script scores layer L+1's router
(gate.weight bf16 [384, 5120] @ x, sqrt(softplus) + gate.bias, top-6, ties -> the lowest id = dsv4_topk) on

    P1  x_L                               layer L's own MoE input, as is
    P2  x_L / ffn_norm_L * ffn_norm_{L+1}  the same pre-norm vector re-weighted with layer L+1's norm weight
    P0  ids_L                             the trivial baseline: the experts layer L routed to
    PT  ids_{L+1} of the previous step    the temporal baseline (what the admission window already exploits)

against the actual ids of layer L+1 (recomputed with the same code from x_{L+1}; the dumped ids check the
recomputation). Reported per layer pair and overall: top-6 precision (|pred & actual| / 6), the recall of the actual
MISSES (experts not resident in layer L+1's slot table at that step) by the predicted top-6 / top-8 / top-12, and the
predicted-but-not-routed misses (the DRAM reads a prefetch would waste). A prefetch pays only for recalled misses.

    python route_predict_eval.py --dumps /mnt/nvme/cache/ds41/dot-cache/route-probe --model /mnt/nvme/models/DeepSeek-V4.1-Flash
    python route_predict_eval.py --dumps <dir> --model /mnt/nvme/models/mini-real384 --real-layers 0,1,2,3,20,21,24,23
      (the mini-real: 8 real layers of the checkpoint in that order; only consecutive pairs are scored)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

TOPK = 6
E = 384


def load_gates(model_dir: str, layers: list[int]):
    """gate.weight (bf16 -> f32), gate.bias (f32) and ffn_norm.weight (bf16 -> f32) of the given real layers."""
    from safetensors import safe_open
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wmap = json.load(open(idx_path))["weight_map"]
    else:   # the mini-real: one file per layer + model-rest.safetensors, no index -> scan
        wmap = {}
        for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(f, "pt") as sf:
                for k in sf.keys():
                    wmap[k] = os.path.basename(f)
    gates = {}
    by_file: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for L in layers:
        for kind, name in (("w", f"layers.{L}.ffn.gate.weight"), ("b", f"layers.{L}.ffn.gate.bias"),
                           ("n", f"layers.{L}.ffn_norm.weight")):
            key = name if name in wmap else "model." + name
            if key not in wmap:
                raise KeyError(f"{name} not in {model_dir}")
            by_file[wmap[key]].append((L, kind, key))
    for f, items in by_file.items():
        with safe_open(os.path.join(model_dir, f), "pt") as sf:
            for L, kind, key in items:
                gates.setdefault(L, {})[kind] = sf.get_tensor(key).float()
    return gates


def route(x: torch.Tensor, gate: dict, k: int = TOPK) -> torch.Tensor:
    """dsv4_topk's selection: score = sqrt(softplus(logit)) + bias, top-k, ties -> the lowest expert id. x [T, H] f32."""
    logits = x @ gate["w"].T                                  # [T, E] f32 (the GPU gate: bf16 x bf16 -> f32)
    sp = torch.where(logits > 20.0, logits, torch.log1p(torch.exp(logits)))
    score = torch.sqrt(sp) + gate["b"]
    # stable top-k with ties to the lowest id: sort by (-score, id)
    order = torch.argsort(score, dim=1, descending=True, stable=True)
    return order[:, :k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--model", default="/mnt/nvme/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--real-layers", default="", help="mini-real: the real layer of every dumped layer index, in order")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dumps, f"rank{args.rank}_*.pt")))
    if not files:
        print(f"no rank{args.rank}_*.pt in {args.dumps}", file=sys.stderr)
        return 2
    steps: dict[int, dict[int, dict]] = defaultdict(dict)
    nlayers = 0
    for f in files:
        d = torch.load(f, map_location="cpu")
        if int(d["T"]) != 1:
            continue                                           # decode steps of one sequence only
        steps[int(d["step"])][int(d["layer"])] = d
        nlayers = max(nlayers, int(d["layer"]) + 1)
    real = [int(s) for s in args.real_layers.split(",")] if args.real_layers else list(range(nlayers))
    assert len(real) >= nlayers, (len(real), nlayers)
    complete = sorted(s for s, ls in steps.items() if len(ls) == nlayers)
    if args.max_steps:
        complete = complete[: args.max_steps]
    print(f"{len(files)} dumps, {len(steps)} steps with T = 1, {len(complete)} complete steps of {nlayers} layers; "
          f"real layers {real[:nlayers]}")
    gates = load_gates(args.model, sorted(set(real[:nlayers])))

    pairs = [(i, i + 1) for i in range(nlayers - 1) if real[i + 1] == real[i] + 1]
    print(f"{len(pairs)} consecutive layer pairs scored: {[(real[a], real[b]) for a, b in pairs]}")
    # per pair: sums
    agg = {p: defaultdict(float) for p in pairs}
    check_rows = check_ok = 0
    prev_actual: dict[int, torch.Tensor] = {}
    for s in complete:
        ls = steps[s]
        for (a, b) in pairs:
            da, db = ls[a], ls[b]
            gb = gates[real[b]]
            x_a, x_b = da["x"].float(), db["x"].float()
            actual = route(x_b, gb)[0]
            dumped = db["ids"]
            if dumped is not None:
                dumped = dumped.reshape(-1)[:TOPK].long()
                check_rows += 1
                check_ok += int(set(actual.tolist()) == set(dumped.tolist()))
            p1 = route(x_a, gb, 12)[0]
            na, nb = gates[real[a]]["n"], gb["n"]
            x_a2 = x_a / torch.where(na.abs() > 1e-6, na, torch.ones_like(na)) * nb
            p2 = route(x_a2, gb, 12)[0]
            p0 = route(x_a, gates[real[a]])[0]                 # layer a's own routing (the dumped ids_a, recomputed)
            pt = prev_actual.get(b)
            act = set(actual.tolist())
            slots = db["slots"]                                # expert -> slot (-1 = not resident) in layer b now
            missing = {e for e in act if int(slots[e]) < 0}
            A = agg[(a, b)]
            A["steps"] += 1
            A["misses"] += len(missing)
            for name, pred in (("P1", p1), ("P2", p2)):
                for k in (6, 8, 12):
                    pk = set(pred[:k].tolist())
                    if k == 6:
                        A[f"{name}_prec6"] += len(pk & act) / TOPK
                    pm = {e for e in pk if int(slots[e]) < 0}  # predicted experts that are not resident
                    A[f"{name}_miss_recall{k}"] += len(pm & missing)
                    A[f"{name}_miss_waste{k}"] += len(pm - missing)
            A["P0_prec6"] += len(set(p0.tolist()) & act) / TOPK
            A["P0_miss_recall6"] += len({e for e in p0.tolist() if int(slots[e]) < 0} & missing)
            if pt is not None:
                A["PT_steps"] += 1
                A["PT_prec6"] += len(set(pt.tolist()) & act) / TOPK
                A["PT_miss_recall6"] += len({e for e in pt.tolist() if int(slots[e]) < 0} & missing)
                A["PT_misses"] += len(missing)
            prev_actual[b] = actual

    print(f"recomputed routing vs the dumped ids of layer L+1: {check_ok}/{check_rows} rows identical")
    hdr = (f"{'pair':>9} {'steps':>5} {'miss/step':>9} | {'P1 prec6':>8} {'P2 prec6':>8} {'P0 prec6':>8} {'PT prec6':>8} | "
           f"{'P1 rec6':>7} {'rec8':>5} {'rec12':>5} {'waste6':>6} {'w12':>5} | {'P2 rec6':>7} {'rec12':>5} | {'PT rec6':>7}")
    print(hdr)
    tot = defaultdict(float)
    rows = []
    for (a, b) in pairs:
        A = agg[(a, b)]
        n = A["steps"]
        m = A["misses"] or 1e-9
        ptn = A["PT_steps"] or 1e-9
        ptm = A["PT_misses"] or 1e-9
        row = dict(pair=[real[a], real[b]], steps=n, misses_per_step=A["misses"] / n,
                   P1_prec6=A["P1_prec6"] / n, P2_prec6=A["P2_prec6"] / n, P0_prec6=A["P0_prec6"] / n,
                   PT_prec6=A["PT_prec6"] / ptn,
                   P1_miss_recall6=A["P1_miss_recall6"] / m, P1_miss_recall8=A["P1_miss_recall8"] / m,
                   P1_miss_recall12=A["P1_miss_recall12"] / m, P1_waste6=A["P1_miss_waste6"] / n,
                   P1_waste12=A["P1_miss_waste12"] / n,
                   P2_miss_recall6=A["P2_miss_recall6"] / m, P2_miss_recall12=A["P2_miss_recall12"] / m,
                   PT_miss_recall6=A["PT_miss_recall6"] / ptm)
        rows.append(row)
        print(f"{real[a]:>4}->{real[b]:<4} {int(n):>5} {row['misses_per_step']:>9.2f} | {row['P1_prec6']:>8.3f} {row['P2_prec6']:>8.3f} "
              f"{row['P0_prec6']:>8.3f} {row['PT_prec6']:>8.3f} | {row['P1_miss_recall6']:>7.3f} {row['P1_miss_recall8']:>5.3f} "
              f"{row['P1_miss_recall12']:>5.3f} {row['P1_waste6']:>6.2f} {row['P1_waste12']:>5.2f} | {row['P2_miss_recall6']:>7.3f} "
              f"{row['P2_miss_recall12']:>5.3f} | {row['PT_miss_recall6']:>7.3f}")
        for k, v in A.items():
            tot[k] += v
    n = tot["steps"]
    m = tot["misses"] or 1e-9
    summary = dict(pairs=len(pairs), steps_x_pairs=n, misses_per_pair_step=tot["misses"] / n,
                   P1_prec6=tot["P1_prec6"] / n, P2_prec6=tot["P2_prec6"] / n, P0_prec6=tot["P0_prec6"] / n,
                   PT_prec6=tot["PT_prec6"] / (tot["PT_steps"] or 1e-9),
                   P1_miss_recall6=tot["P1_miss_recall6"] / m, P1_miss_recall8=tot["P1_miss_recall8"] / m,
                   P1_miss_recall12=tot["P1_miss_recall12"] / m,
                   P1_waste6_per_pair_step=tot["P1_miss_waste6"] / n, P1_waste12_per_pair_step=tot["P1_miss_waste12"] / n,
                   P2_miss_recall6=tot["P2_miss_recall6"] / m, P2_miss_recall12=tot["P2_miss_recall12"] / m,
                   PT_miss_recall6=tot["PT_miss_recall6"] / (tot["PT_misses"] or 1e-9),
                   routing_check=f"{check_ok}/{check_rows}")
    print("overall:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}))
    if args.out:
        json.dump(dict(summary=summary, pairs=rows, dumps=args.dumps, model=args.model), open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
