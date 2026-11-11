#!/usr/bin/env python3
"""The balance of a pipelined prefill step's two streams, ONE LINE PER RANK, from the phase-wait instrument's log lines
(VLLM_MOET_PREFILL_PIPE_PHASE_WAIT=3, or its runtime key PREFILL_PIPE_PHASE_WAIT, in moet_prefill_pipe.py).

Input: a server log (a file, or `-` = stdin) with the lines `prefill pipeline phase wait:`, `phase split:` and `phase layers:`;
a leading docker timestamp (`docker logs -t`) is used when --since / --until / --runs select a window, and is optional otherwise.
A harvest = the three lines one rank printed at one step's end; it describes the phases of the step before it.

Per harvest, per slot (a comm phase and the compute segment of the OTHER row group beside it; four slots per layer):
   the phase's length, the segment's length, how long the compute stream waited for the phase's end, the phase's slack.
The step's last compute segment - the first group's replay tail - is booked beside the second group's last phase (index
2 x --sp-layers); it is taken out of that slot before anything is divided, and reported on its own.
Per rank (means over its harvests, ms per sequence-parallel layer):
   compute work = the four segments, comm work = the four phases, period = the four (segment + waited);
   the two slots beside a MoE with the phase's reduce-scatter / glue / all-gather, and the two slots beside an attention.
Two readings follow the rank lines:
   pace   which rank has the longest MoE segment, and whether its compute stream still waits behind the phase (then the comm
          stream sets the slot's pace on every rank) or not (then that rank's card does);
   cap    the smallest wait behind a MoE over the ranks and the two slots against what a grid cap of FC2 adds to the MoE
          segment (--cap-cost, ms per cap, as measured in situ on RTX PRO 6000 Server Edition cards): a cap can shorten a
          slot only while the wait there is larger than its cost.

usage: phase_balance.py LOG [--sp-layers 20] [--min-phases 19] [--since ISO] [--until ISO] [--runs RUNS.log --label L[,L...]]
                        [--cap-cost 320:0.05,256:0.11,152:0.25] [--no-layers]
  --runs / --label   only harvests printed inside a request of more than 100 000 tokens that RUNS.log lists under one of the
                     labels (lines `<ISO end time> <label> fresh prefill <tokens> tok: <ms> ms`)
  --no-layers        a log without `phase layers` lines (level 2): the tail stays inside the second group's attention slot
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import statistics as st
import sys

SLOTS = ("g0 before attention", "g1 before attention", "g0 before MoE", "g1 before MoE")
BESIDE = {"g0 before attention": "g1's MoE", "g1 before attention": "g0's attention", "g0 before MoE": "g1's attention",
          "g1 before MoE": "g0's MoE"}
MOE_SLOTS = ("g0 before attention", "g1 before MoE")
ATT_SLOTS = ("g1 before attention", "g0 before MoE")

TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z?\s")
HEAD = re.compile(r"phase (wait|split|layers): at a step of (\d+) tokens \(rank (\d+)\): (.*)$")
WAIT = re.compile(r"(g[01] before (?:attention|MoE)): n (\d+), waited ([\d.]+) ms in (\d+), slack ([\d.]+) ms")
SPLIT = re.compile(r"^(g[01] before (?:attention|MoE)): n (\d+), total ([\d.]+), compute free at ([\d.]+) \| (.*)$")
ELEM = re.compile(r"^(\S+) ([\d.]+)/([\d.]+) x(\d+)$")
RUN = re.compile(r"^(\S+) (\S+) fresh prefill (\d+) tok: (\d+) ms")


def epoch(s: str) -> float:
    s = s.rstrip("Z")
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in s else "%Y-%m-%dT%H:%M:%S"
    return dt.datetime.strptime(s[:26], fmt).replace(tzinfo=dt.timezone.utc).timestamp()


def main_elements(text: str) -> dict:
    """The phase's main chain out of the split line's elements: the first reduce-scatter, the glue behind it, the first all-gather
    behind that glue; each as (ms summed over the step, count)."""
    els = []
    for part in text.split(", "):
        m = ELEM.match(part.strip())
        if m:
            els.append((m.group(1), float(m.group(2)), int(m.group(4))))
    out = {}
    i = next((k for k, e in enumerate(els) if e[0].startswith("reduce_scatter")), None)
    if i is not None:
        out["rs"] = (els[i][1], els[i][2])
        j = next((k for k in range(i + 1, len(els)) if els[k][0] == "glue"), None)
        if j is not None:
            out["glue"] = (els[j][1], els[j][2])
            k = next((q for q in range(j + 1, len(els)) if els[q][0].startswith("all_gather")), None)
            if k is not None:
                out["ag"] = (els[k][1], els[k][2])
    return out


def harvests(lines, sp_layers: int, min_phases: int, windows, with_layers: bool):
    """-> ({rank: [record]}, harvests left out because their `phase layers` line does not name the tail); a record =
    {"slot": {name: (phase, segment, waited, slack)}, "el": {name: {rs, glue, ag}}, "tail": ms | None}, all per phase"""
    tail_re = re.compile(r"slack g1 [^;]*\bp%d:([\d.]+)/([\d.]+)" % (2 * sp_layers))
    cur: dict = {}
    out: dict = {}
    no_tail = 0
    for line in lines:
        h = HEAD.search(line)
        if not h:
            continue
        kind, rank, body = h.group(1), int(h.group(3)), h.group(4)
        t = TS.match(line)
        when = epoch(t.group(1)) if t else None
        c = cur.setdefault(rank, {})
        if kind == "wait":
            c.clear()
            c["t"] = when
            c["w"] = {k: (int(n), float(w), float(s)) for k, n, w, _nw, s in WAIT.findall(body)}
            continue
        if kind == "split" and "w" in c:
            c["s"], c["e"] = {}, {}
            for part in body.split("; "):
                m = SPLIT.match(part.strip())
                if m:
                    c["s"][m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
                    c["e"][m.group(1)] = main_elements(m.group(5))
            if with_layers:
                continue
        elif not (kind == "layers" and "w" in c and "s" in c and with_layers):
            continue
        w, s = c.get("w", {}), c.get("s", {})
        if len(w) != 4 or len(s) != 4 or min(v[0] for v in s.values()) < min_phases:
            cur[rank] = {}
            continue
        if windows is not None and (c["t"] is None or not any(a <= c["t"] <= b for a, b in windows)):
            cur[rank] = {}
            continue
        tail = tail_re.search(body) if kind == "layers" else None
        if with_layers and tail is None:  # the tail cannot be taken out of this harvest: it would sit in an attention slot
            no_tail += 1
            cur[rank] = {}
            continue
        rec = {"slot": {}, "el": {}, "tail": None}
        for k in SLOTS:
            n, total, free = s[k]
            waited, slack = w[k][1], w[k][2]
            if k == "g1 before attention" and tail is not None:
                excess, segment = float(tail.group(1)), float(tail.group(2))
                n, total, free, slack = n - 1, total - (segment - excess), free - segment, slack - excess
                rec["tail"] = segment
            rec["slot"][k] = (total / n, free / n, waited / n, slack / n)
            rec["el"][k] = {name: v[0] / v[1] for name, v in c["e"].get(k, {}).items() if v[1] > 0}
        out.setdefault(rank, []).append(rec)
        cur[rank] = {}
    return out, no_tail


def mean(xs) -> float:
    xs = list(xs)
    return st.mean(xs) if xs else float("nan")


def rank_row(recs: list) -> dict:
    slot = {k: tuple(mean(r["slot"][k][i] for r in recs) for i in range(4)) for k in SLOTS}
    el = {k: {name: mean(r["el"][k][name] for r in recs if name in r["el"][k]) for name in ("rs", "glue", "ag")} for k in SLOTS}
    tails = [r["tail"] for r in recs if r["tail"] is not None]
    return {
        "n": len(recs), "slot": slot, "el": el, "tail": mean(tails) if tails else None,
        "compute": sum(v[1] for v in slot.values()), "comm": sum(v[0] for v in slot.values()),
        "period": sum(v[1] + v[2] for v in slot.values()),
    }


def fmt_slot(row: dict, k: str, elements: bool) -> str:
    phase, segment, waited, slack = row["slot"][k]
    e = row["el"][k]
    inner = f" (rs {e['rs']:.2f}, glue {e['glue']:.2f}, ag {e['ag']:.2f})" if elements and all(x == x for x in e.values()) else ""
    return f"beside {BESIDE[k]}: phase {phase:.3f}{inner}, segment {segment:.3f}, waited {waited:.3f}, slack {slack:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--sp-layers", type=int, default=20)
    ap.add_argument("--min-phases", type=int, default=19)
    ap.add_argument("--since", default="")
    ap.add_argument("--until", default="")
    ap.add_argument("--runs", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--cap-cost", default="320:0.05,256:0.11,152:0.25")
    ap.add_argument("--no-layers", action="store_true")
    a = ap.parse_args()

    windows = None
    if a.runs:
        labels = {x for x in a.label.split(",") if x}
        windows = []
        for line in open(a.runs, encoding="utf-8", errors="replace"):
            m = RUN.match(line)
            if m and int(m.group(3)) > 100000 and (not labels or m.group(2) in labels):
                end = epoch(m.group(1))
                windows.append((end - int(m.group(4)) / 1000.0, end + 0.5))
    if a.since or a.until:
        lo = epoch(a.since) if a.since else float("-inf")
        hi = epoch(a.until) if a.until else float("inf")
        windows = [(max(x, lo), min(y, hi)) for x, y in windows] if windows is not None else [(lo, hi)]

    fh = sys.stdin if a.log == "-" else open(a.log, encoding="utf-8", errors="replace")
    per, no_tail = harvests(fh, a.sp_layers, a.min_phases, windows, not a.no_layers)
    if not per:
        print("no complete harvest (the three lines of one rank at one step's end) in the selected part of the log"
              + (f"; {no_tail} left out: their `phase layers` line does not name the tail (--sp-layers?)" if no_tail else ""))
        return 2
    rows = {rank: rank_row(recs) for rank, recs in sorted(per.items())}
    tail_note = "" if not a.no_layers else " (no `phase layers` lines: the replay tail is INSIDE the slot beside g0's attention)"
    if no_tail:
        tail_note += f" ({no_tail} harvests left out: their `phase layers` line does not name the tail)"
    print(f"per sequence-parallel layer, ms; one line per rank{tail_note}")
    for rank, row in rows.items():
        tail = "" if row["tail"] is None else f" | replay tail {row['tail']:.2f} per step"
        print(
            f"rank {rank}: {row['n']} steps | compute work {row['compute']:.3f}, comm work {row['comm']:.3f}, period {row['period']:.3f}, "
            f"comm - compute {row['comm'] - row['compute']:+.3f} | "
            + " | ".join(fmt_slot(row, k, True) for k in MOE_SLOTS) + " | "
            + " | ".join(fmt_slot(row, k, False) for k in ATT_SLOTS) + tail
        )

    # pace: the rank with the longest MoE segments, and whether it still waits behind the phase
    moe = {rank: mean(row["slot"][k][1] for k in MOE_SLOTS) for rank, row in rows.items()}
    wait = {rank: min(row["slot"][k][2] for k in MOE_SLOTS) for rank, row in rows.items()}
    slow, fast = max(moe, key=moe.get), min(moe, key=moe.get)
    others = [wait[r] for r in rows if r != slow]
    spread = 100.0 * (moe[slow] - moe[fast]) / moe[fast] if moe[fast] > 0 else float("nan")
    if not others:
        print(f"pace: one rank only (MoE segment {moe[slow]:.3f} ms, smallest wait behind a MoE {wait[slow]:.3f} ms)")
    else:
        if wait[slow] >= 0.05:
            verdict = "its compute stream still waits behind the phase: the comm stream sets the pace of the MoE slots on every rank"
        elif min(others) >= 0.05:
            verdict = "its compute stream does not wait while the other ranks' do: this rank's card sets the pace of the MoE slots"
        else:
            verdict = "no rank waits behind a MoE: the MoE is the long side of its slot on every rank"
        print(f"pace: the longest MoE segment is rank {slow}'s ({moe[slow]:.3f} ms, {spread:+.1f} % over rank {fast}'s); its smallest "
              f"wait behind a MoE {wait[slow]:.3f} ms, the other ranks' {min(others):.3f} - {max(others):.3f} ms: {verdict}")

    # cap: the smallest wait behind a MoE against a cap's cost
    w_min = min(wait.values())
    costs = []
    for item in a.cap_cost.split(","):
        if ":" in item:
            cap, cost = item.split(":", 1)
            costs.append((int(cap), float(cost)))
    can = [cap for cap, cost in costs if w_min > cost]
    glue = {k: mean(row["el"][k]["glue"] for row in rows.values()) for k in SLOTS}
    held = mean(glue[k] for k in MOE_SLOTS) - mean(glue[k] for k in ATT_SLOTS)
    print(f"cap: the smallest wait behind a MoE over the ranks and both slots is {w_min:.3f} ms; a cap's cost in MoE segment "
          + ", ".join(f"{cap}: {cost:.2f}" for cap, cost in costs)
          + f" ms -> " + ("a cap can shorten the MoE slots: " + ", ".join(str(c) for c in can) if can else "no cap can shorten a MoE slot (the wait is not larger than any cap's cost)")
          + f"; glue beside a MoE minus glue beside an attention {held:+.3f} ms per phase (the most a cap can give back)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
