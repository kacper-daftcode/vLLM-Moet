#!/usr/bin/env python3
"""phase_balance.py on the level-3 lines of THREE REAL STATES of one image and on synthetic steps - no GPU, no model, no vLLM.

The real lines are fixtures (testdata/phase_balance_state_<x>.lines.gz: the instrument's own text from `prefill pipeline phase` on,
four ranks, the steps of two instrumented requests of about 163 000 tokens each):
  a   four cards whose clocks differ under one power limit (two of them run about a sixth below the other two)
  b   four like cards deep in their power limit
  c   four like cards at the edge of their power limit
In all three the compute stream waits about 0.6 ms behind every phase beside a MoE - the reading "the comm stream sets the pace"
that the wait alone suggests is right in one slot of state c only. The expected numbers below were computed from the same lines by a
separate reading of these states, before this test existed.

What it holds:
  (1) the rank lines are what they were: steps, compute work, comm work, period per rank in each state;
  (2) the glue's end against the MoE segment's end, per rank and MoE slot: from the sums and from the split line's second
      numbers, and the slope over the rank's steps with its 95 % interval and r;
  (3) across the ranks: in state a the two fast ranks wait in the all-gather for the slowest rank's glue, by as much as their
      glue ends earlier; in states b and c no rank waits for another;
  (4) the pace sentence: state a - the slow card paces both MoE slots through its glue, and the words "the comm stream sets the
      pace" do not occur; state b - the glue ends with the MoE segment in both slots; state c - the comm stream sets the pace
      beside the second group's MoE and the glue ends with the segment beside the first group's;
  (5) synthetic steps: a glue that does not move while the MoE segment does -> the comm stream sets the pace; a glue that ends
      with the segment -> it does not say so; three steps only -> not decided, and it does not say so; a compute stream that
      never waits -> the MoE is the long side; one output has one line per rank of each kind and one `pace:` and one `cap:` line.

usage: python3 test_phase_balance.py
"""
from __future__ import annotations

import gzip
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_balance as pb  # noqa: E402

fails = 0


def check(cond: bool, text: str) -> None:
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {text}")
    fails += 0 if cond else 1


def near(got, want, tol: float) -> bool:
    return all(abs(g - w) <= tol for g, w in zip(got, want)) and len(list(got)) == len(list(want))


G1, G0 = "g0 before attention", "g1 before MoE"   # the slots beside the second group's MoE and beside the first group's
EXPECT = {
    "a": {
        "steps": (35, 37, 37, 37), "compute": (9.230, 7.890, 9.045, 8.073), "comm": (10.294, 10.352, 10.298, 10.341),
        "period": (10.816, 10.814, 10.814, 10.821),
        G1: {"sums": (-0.074, -0.100, -0.075, -0.098), "second": (-0.065, -0.098, -0.066, -0.090),
             "slope": ((1.41, 0.06, 0.99), (0.84, 0.05, 0.98), (1.14, 0.12, 0.96), (1.25, 0.19, 0.91))},
        G0: {"sums": (0.110, 0.101, 0.114, 0.101), "second": (0.114, 0.101, 0.115, 0.102),
             "slope": ((1.56, 0.24, 0.92), (0.99, 0.03, 1.00), (1.14, 0.07, 0.98), (1.27, 0.15, 0.95))},
    },
    "b": {
        "steps": (38, 38, 38, 38), "compute": (8.831, 8.635, 8.565, 8.725), "comm": (10.067, 10.080, 10.075, 10.069),
        "period": (10.615, 10.624, 10.613, 10.615),
        G1: {"sums": (-0.062, -0.072, -0.071, -0.059), "second": (-0.063, -0.069, -0.069, -0.064),
             "slope": ((0.82, 0.10, 0.94), (0.99, 0.17, 0.89), (1.05, 0.07, 0.98), (0.81, 0.09, 0.95))},
        G0: {"sums": (0.125, 0.119, 0.119, 0.122), "second": (0.125, 0.119, 0.119, 0.122),
             "slope": ((1.02, 0.04, 0.99), (1.04, 0.02, 1.00), (1.03, 0.02, 1.00), (1.04, 0.01, 1.00))},
    },
    "c": {
        "steps": (37, 37, 38, 38), "compute": (7.399, 7.339, 7.343, 7.386), "comm": (8.557, 8.555, 8.555, 8.552),
        "period": (9.066, 9.065, 9.067, 9.064),
        G1: {"sums": (-0.046, -0.052, -0.053, -0.054), "second": (-0.053, -0.055, -0.053, -0.058),
             "slope": ((-0.07, 0.14, -0.15), (-0.03, 0.17, -0.06), (0.03, 0.14, 0.08), (0.09, 0.15, 0.19))},
        G0: {"sums": (0.106, 0.107, 0.108, 0.105), "second": (0.106, 0.107, 0.109, 0.105),
             "slope": ((0.87, 0.19, 0.85), (0.79, 0.19, 0.83), (0.27, 0.13, 0.57), (0.83, 0.16, 0.86))},
    },
}
COMM = "the comm stream sets the pace"


def run_tool(lines: list, *args: str) -> list:
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out = subprocess.run([sys.executable, "-B", str(HERE / "phase_balance.py"), f.name, *args], capture_output=True, text=True)
    os.unlink(f.name)
    return out.returncode, out.stdout.splitlines()


for state, exp in EXPECT.items():
    print(f"state {state}")
    with gzip.open(HERE / "testdata" / f"phase_balance_state_{state}.lines.gz", "rt", encoding="utf-8") as fh:
        lines = [x.rstrip("\n") for x in fh]
    per, no_tail = pb.harvests(lines, 20, 19, None, True)
    rows = {rank: pb.rank_row(recs) for rank, recs in sorted(per.items())}
    ranks = sorted(rows)
    check(ranks == [0, 1, 2, 3] and no_tail == 0 and tuple(rows[r]["n"] for r in ranks) == exp["steps"],
          f"(1) four ranks, their steps {tuple(rows[r]['n'] for r in ranks)}")
    check(near([rows[r]["compute"] for r in ranks], exp["compute"], 0.0006) and near([rows[r]["comm"] for r in ranks], exp["comm"], 0.0006)
          and near([rows[r]["period"] for r in ranks], exp["period"], 0.0006), "(1) compute work, comm work and period per rank")
    glue = {r: {k: pb.glue_row(per[r], k) for k in pb.MOE_SLOTS} for r in ranks}
    for k, name in ((G1, "beside the second group's MoE"), (G0, "beside the first group's MoE")):
        check(near([glue[r][k]["by_sums"] for r in ranks], exp[k]["sums"], 0.0006) and near([glue[r][k]["by_second"] for r in ranks], exp[k]["second"], 0.0006),
              f"(2) {name}: glue end minus segment end, by the sums {tuple(round(glue[r][k]['by_sums'], 3) for r in ranks)} and by the second numbers")
        got = [glue[r][k]["slope"] for r in ranks]
        check(all(g is not None for g in got) and all(near(g[:3], w, 0.0051) for g, w in zip(got, exp[k]["slope"])),
              f"(2) {name}: the slope of the glue's end on the segment's end, its 95 % half-width and r: "
              + " | ".join("%+.2f +- %.2f (r %+.2f)" % g[:3] for g in got))
    moe = {r: pb.mean(rows[r]["slot"][k][1] for k in pb.MOE_SLOTS) for r in ranks}
    slow = max(moe, key=moe.get)
    w = {k: pb.all_gather_waits(rows, glue, slow, k) for k in pb.MOE_SLOTS}
    waiting = {k: sorted(r for r, _e, _l, yes in w[k] if yes) for k in pb.MOE_SLOTS}
    if state == "a":
        longer = {k: {r: l for r, _e, l, _y in w[k]} for k in pb.MOE_SLOTS}
        earlier = {k: {r: e for r, e, _l, _y in w[k]} for k in pb.MOE_SLOTS}
        check(slow == 0 and waiting == {G1: [1, 3], G0: [1, 3]}, "(3) the slowest rank is rank 0; ranks 1 and 3 wait in the all-gather in both MoE slots, rank 2 does not")
        check(near([longer[G1][1], longer[G1][3], longer[G0][1], longer[G0][3]], (0.407, 0.312, 0.452, 0.346), 0.0006)
              and all(abs(earlier[k][r] - longer[k][r]) < 0.04 for k in pb.MOE_SLOTS for r in (1, 3)),
              "(3) they stay 0.407 / 0.312 and 0.452 / 0.346 ms longer in the all-gather - what their glue ends earlier, within 0.04 ms")
        phases = [rows[r]["slot"][k][0] for r in ranks for k in pb.MOE_SLOTS]
        check(max(phases) - min(phases) < 0.06, f"(3) the phase beside a MoE has one length on the four ranks ({min(phases):.3f} - {max(phases):.3f} ms)")
    else:
        check(waiting == {G1: [], G0: []}, "(3) no rank waits in the all-gather for another")
    rc, out = run_tool(lines)
    pace = [x for x in out if x.startswith("pace: ")]
    check(rc == 0 and sum(x.startswith("rank ") for x in out) == 4 and sum(x.startswith("glue: rank ") for x in out) == 4
          and sum(x.startswith("all-gather: ") for x in out) == 1 and len(pace) == 1 and sum(x.startswith("cap: ") for x in out) == 1,
          "(4) the tool's output: four rank lines, four glue lines, one all-gather line, one pace line, one cap line")
    p = pace[0] if pace else ""
    if state == "a":
        check(p.count("its card paces the slot through its glue") == 2 and p.count("ranks 1, 3 wait") == 2 and COMM not in p,
              "(4) pace: the slow card paces both MoE slots through its glue; the words 'the comm stream sets the pace' do not occur")
    elif state == "b":
        check(p.count("its glue ends with its MoE segment") == 2 and p.count("no other rank waits in the all-gather for it") == 2 and COMM not in p,
              "(4) pace: the glue ends with the MoE segment in both slots, no rank waits for another; not 'the comm stream sets the pace'")
    else:
        first, _, second = p.partition("Beside g0's MoE: ")
        check(first.count(COMM + " of the slot") == 1 and "does not follow its MoE segment" in first and COMM not in second
              and "its glue ends with its MoE segment" in second,
              "(4) pace: beside the second group's MoE the comm stream sets the pace; beside the first group's the glue ends with the segment")


# ---- (5) synthetic steps (level 2: no `phase layers` line): two ranks, the four slots of a step with n phases each
def step(rank: int, moe_segment: float, glue: float, waited: float, n: int = 10) -> list:
    """the two lines of one step: beside an attention everything is fixed; beside a MoE the given segment, glue and wait"""
    rs, ag, lead = 0.9, 0.5, 0.0
    w, s = [], []
    for k in pb.SLOTS:
        if k in pb.MOE_SLOTS:
            seg, g, wt = moe_segment, glue, waited
        else:
            seg, g, wt = 2.0, 0.4, 0.2
        phase = seg + wt
        end = lead + rs + g                      # the glue's end, counted from the phase's start
        after = max(0.0, end - seg)              # its part after the compute stream was free
        before = min(ag, max(0.0, seg - end))    # the all-gather's part before it
        w.append(f"{k}: n {n}, waited {n * wt:.4f} ms in {n}, slack 0.00 ms")
        s.append(f"{k}: n {n}, total {n * phase:.4f}, compute free at {n * seg:.4f} | glue {n * lead:.4f}/0.0000 x{n}, "
                 f"reduce_scatter {n * rs:.4f}/0.0000 x{n}, glue {n * g:.4f}/{n * after:.4f} x{n}, all_gather {n * ag:.4f}/{n * (ag - before):.4f} x{n}")
    head = f"at a step of 8192 tokens (rank {rank}): "
    return ["prefill pipeline phase wait: " + head + "; ".join(w), "prefill pipeline phase split: " + head + "; ".join(s)]


def pace_of(lines: list) -> str:
    rc, out = run_tool(lines, "--no-layers", "--min-phases", "5")
    p = [x for x in out if x.startswith("pace: ")]
    return p[0] if rc == 0 and len(p) == 1 else f"rc {rc}, {len(p)} pace lines"


print("synthetic steps")
segs = [1.50, 1.62, 1.55, 1.71, 1.66, 1.58, 1.75, 1.52]
fixed = [x for i, sg in enumerate(segs) for r in (0, 1) for x in step(r, sg + 0.05 * r, 0.35, 0.6)]
p = pace_of(fixed)
check(p.count(COMM + " of the slot") == 2 and "does not follow its MoE segment" in p,
      "(5) the glue does not move while the MoE segment does: the comm stream sets the pace of both slots")
tied = [x for i, sg in enumerate(segs) for r in (0, 1) for x in step(r, sg + 0.05 * r, sg + 0.05 * r - 0.9 + 0.03, 0.6)]
p = pace_of(tied)
check(COMM not in p and p.count("its glue ends with its MoE segment") == 2 and "slope +1.00 +- 0.00" in p,
      "(5) the glue ends 0.03 ms behind the MoE segment in every step (slope 1): it does not say 'the comm stream sets the pace'")
few = [x for i, sg in enumerate(segs[:3]) for r in (0, 1) for x in step(r, sg + 0.05 * r, 0.35, 0.6)]
p = pace_of(few)
check(COMM not in p and p.count("is not decided") == 2 and "3 steps, 6 needed" in p,
      "(5) three steps only: not decided - and not 'the comm stream sets the pace', although the glue does not move")
nowait = [x for i, sg in enumerate(segs) for r in (0, 1) for x in step(r, sg + 0.05 * r, 0.35, 0.0)]
p = pace_of(nowait)
check(COMM not in p and p.count("no rank's compute stream waits there") == 2, "(5) the compute stream never waits behind a MoE: the MoE is the long side")
rc, out = run_tool(fixed, "--no-layers", "--min-phases", "5")
check(rc == 0 and sum(x.startswith("rank ") for x in out) == 2 and sum(x.startswith("glue: rank ") for x in out) == 2
      and sum(x.startswith("all-gather: ") for x in out) == 1 and sum(x.startswith("cap: ") for x in out) == 1,
      "(5) one line per rank of each kind, one all-gather line, one cap line")
check(pb.regress([1, 2, 3], [1, 2, 3]) is None and pb.regress([1.0] * 8, [float(i) for i in range(8)]) is None
      and pb.regress([float(i) for i in range(8)], [2.0] * 8)[:3] == (0.0, 0.0, 0.0),
      "(5) the slope: none below six points or without a moving segment; 0 for a glue that does not move")

print(f"{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
