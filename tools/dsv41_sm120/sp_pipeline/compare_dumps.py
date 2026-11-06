#!/usr/bin/env python3
"""Compare the per-layer dumps (VLLM_MOET_PREFILL_PIPE_DUMP) of two servers that prefilled the same deterministic
prompt (dump_prefill.py): A = the pipelined run (PIPE=2 at 8192 batched tokens), B = the reference (PIPE=0 at 4096).

A chunk is identified by (rank, pos0, rows): group 1 of A's step s is the chunk that starts at position 4096 + 8192 s,
which is B's step 2 s + 1. Per layer the sha256 of every hc state (x = the MoE partial handed to the next boundary,
residual, post_mix, res_mix, pre_mix) must be identical on every rank.

usage: compare_dumps.py DIR_A DIR_B [--keys x,residual,...] [--kind all|prefill_rows]
  --kind prefill_rows: the mixed-step entries (a group with decode rows ahead of a prefilling request: the hashes
  of the prefill rows alone, keyed by their first position) - the pipelined run vs the same server with
  VLLM_MOET_PREFILL_PIPE_FORCE_SEQUENTIAL=1 under the same number of decode streams.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("a", type=Path)
ap.add_argument("b", type=Path)
ap.add_argument("--keys", default="x,residual,post_mix,res_mix,pre_mix")
ap.add_argument("--kind", default="all", choices=["all", "prefill_rows"])
args = ap.parse_args()
keys = args.keys.split(",")


def load(d: Path):
    out = {}
    for f in sorted(d.glob("rank*.jsonl")):
        rank = int(f.stem[4:])
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("kind", "all") != args.kind:
                continue
            out[(rank, e["pos0"], e["rows"], e["layer"])] = e
    return out


A, B = load(args.a), load(args.b)
print(f"A: {len(A)} layer records, B: {len(B)}")
if not A or not B:
    sys.exit("no records")
common = sorted(set(A) & set(B))
only_a = sorted(set(A) - set(B))
only_b = sorted(set(B) - set(A))
print(f"common chunk-layers: {len(common)}; only in A: {len(only_a)}; only in B: {len(only_b)}")
if only_a[:5]:
    print("  e.g. only in A:", only_a[:5])
if only_b[:5]:
    print("  e.g. only in B:", only_b[:5])
bad = defaultdict(list)
per_group = defaultdict(lambda: [0, 0])  # group -> [identical, different]
for k in common:
    ea, eb = A[k], B[k]
    diff = [key for key in keys if ea.get(key) != eb.get(key)]
    g = ea.get("group", -1)
    if diff:
        bad[k].extend(diff)
        per_group[g][1] += 1
    else:
        per_group[g][0] += 1
for g in sorted(per_group):
    ident, differ = per_group[g]
    print(f"A group {g}: {ident} chunk-layers identical, {differ} different")
if bad:
    print(f"DIFFERENT: {len(bad)} chunk-layers")
    for k in sorted(bad)[:40]:
        print(f"  rank {k[0]} pos0 {k[1]} rows {k[2]} layer {k[3]}: {bad[k]}")
    sys.exit(1)
print(f"ALL IDENTICAL: {len(common)} chunk-layers x {len(keys)} states on {len({k[0] for k in common})} ranks")
