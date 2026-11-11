#!/usr/bin/env python3
"""The `phase layers` lines of the pipelined prefill step's phase-wait instrument (VLLM_MOET_PREFILL_PIPE_PHASE_WAIT=3, see
moet_prefill_pipe.py) as a table: which compute segments outlast the comm phase beside them, by how much, how long they are, and
how long the host took to issue them. CPU only; reads a server log.

A line names, per step and rank,
  slack g<G> p<P>:<excess>/<segment>   group G's comm phase P of the step ended `excess` ms before the compute stream was free; the
                                       compute segment beside it took `segment` ms from the phase's start. That segment is the
                                       OTHER group's:   P even (the phase before the attention of layer P/2)
                                                          G = 1: group 0's attention of layer P/2;  G = 0: group 1's MoE of layer P/2 - 1
                                                        P odd (the phase before the MoE of layer P//2)
                                                          G = 0: group 1's attention of layer P//2; G = 1: group 0's MoE of layer P//2
  host g<G> p<P>:<wall>/<cpu>          group G's thread took `wall` ms (`cpu` ms of its own CPU time) to issue the segment before its
                                       phase P: P odd = its attention of layer P//2, P even = its MoE of layer P/2 - 1.
The last phase of the sequence-parallel section is group 1's, before the "attention" of the first replay layer; the segment beside
it is group 0's KV write of that layer and its replay tail - one long segment per step by construction. With --sp-layers N (the
number of sequence-parallel layers, 20 for DeepSeek-V4.1-Flash) it is listed as `tail`; take it out before any per-layer
arithmetic on the instrument's sums.

usage: phase_layers_table.py <server log> [--tokens 8192] [--min-share 5] [--sp-layers N]
       --tokens     the step size whose lines are read (default 8192)
       --min-share  a segment is listed when it is named in at least this percentage of the lines (default 5)
"""
from __future__ import annotations

import argparse
import collections
import re
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--tokens", default="8192")
ap.add_argument("--min-share", type=float, default=5.0)
ap.add_argument("--sp-layers", type=int, default=0)
a = ap.parse_args()

pat = re.compile(r"phase layers: at a step of (\d+) tokens \(rank (\d+)\): (.*)$")
ent = re.compile(r"p(\d+):([\d.]+)/([\d.na]+)")
slack, host, rest, nlines = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), 0
with open(a.log, errors="replace") as f:
    for line in f:
        m = pat.search(line)
        if not m or m.group(1) != a.tokens:
            continue
        nlines += 1
        for part in m.group(3).split("; "):
            head = part.split()
            kind, g = head[0], int(head[1][1:])
            for p, x, y in ent.findall(part):
                (slack if kind == "slack" else host)[(g, int(p))].append((float(x), float(y) if y != "nan" else float("nan")))
            r = re.search(r"rest: n (\d+), median ([\d.]+)/([\d.]+)", part)
            if r and kind == "host":
                rest[g].append((int(r.group(1)), float(r.group(2)), float(r.group(3))))


def segment_beside(g: int, p: int) -> tuple[int, int, str]:
    layer = p // 2
    if p % 2 == 0:
        return (0, layer, "attention") if g == 1 else (1, layer - 1, "MoE")
    return (1, layer, "attention") if g == 0 else (0, layer, "MoE")


def segment_before(g: int, p: int) -> tuple[int, int, str]:
    return (g, p // 2, "attention") if p % 2 == 1 else (g, p // 2 - 1, "MoE")


print(f"{nlines} `phase layers` lines of {a.tokens}-token steps")
if nlines == 0:
    raise SystemExit(0)
last = 2 * a.sp_layers if a.sp_layers > 0 else -1  # group 1's phase before the first replay layer: the tail runs beside it
need = a.min_share / 100.0 * nlines
host_by = {segment_before(g, p): v for (g, p), v in host.items()}
print("\ncompute segments that outlast the comm phase beside them (excess >= 0.3 ms):")
print("  group layer segment   % of lines   excess ms   segment ms   host issue ms, wall / CPU (% of lines)   [medians]")
for (sg, layer, kind), v in sorted((segment_beside(g, p) if not (g == 1 and p == last) else (0, p // 2, "tail"), v)
                                   for (g, p), v in slack.items()):
    if len(v) < need:
        continue
    h = host_by.get((sg, layer, kind))
    hs = f"{st.median(x[0] for x in h):.2f} / {st.median(x[1] for x in h):.2f} ({100 * len(h) / nlines:.0f} %)" if h else "under 1 ms"
    print(f"  g{sg}  {layer:3d}  {kind:9s}  {100 * len(v) / nlines:5.0f}  {st.median(x[0] for x in v):9.2f}  {st.median(x[1] for x in v):9.2f}     {hs}")
rare = sum(len(v) for v in slack.values() if len(v) < need)
print(f"  ({rare} entries of phases named in under {a.min_share:g} % of the lines are not listed)")
print("\nsegments the host took 1 ms or more to issue:")
for (sg, layer, kind), v in sorted(host_by.items()):
    if len(v) >= need:
        print(f"  host g{sg} layer {layer:2d} {kind:9s}: in {100 * len(v) / nlines:3.0f} % of the lines, wall {st.median(x[0] for x in v):.2f} ms, "
              f"CPU {st.median(x[1] for x in v):.2f} ms")
for g, v in sorted(rest.items()):
    print(f"  host g{g}, the other segments: {st.median(x[0] for x in v):.0f} per step, median wall {st.median(x[1] for x in v):.2f} ms, "
          f"CPU {st.median(x[2] for x in v):.2f} ms")
