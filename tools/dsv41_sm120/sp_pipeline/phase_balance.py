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
Behind the rank lines, for the two slots beside a MoE:
   glue        one line per rank: where the phase's glue ends against the end of the MoE segment beside it - from the sums
               (lead + reduce-scatter + glue - segment) and from the split line's second numbers (the glue's part after the
               compute stream was free minus the all-gather's part before it) - and, over the rank's steps, the slope of the
               glue's end on the segment's end with its 95 % interval. Slope 1: the glue ends when the MoE segment ends, so
               the phase is that segment plus a fixed remainder. Slope 0: the glue does not follow the segment.
   all-gather  one line: against the rank with the longest MoE segment, how much earlier each other rank ends its glue and
               how much longer it then stays in the all-gather - a rank that waits there waits for the slowest rank's glue.
Two readings follow:
   pace   per MoE slot, for the rank with the longest MoE segment. A wait of the compute stream behind the phase does NOT
          by itself say that the comm stream sets the pace: where the glue ends with the MoE segment the MoE's own time
          is on the long side of the slot, and where the other ranks wait in the all-gather for that rank's glue its card
          paces the slot. "The comm stream sets the pace" is printed only when the slope's whole interval lies below 0.5;
          with fewer than 6 steps, or an interval across 0.5, the line says that it is not decided.
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
import math
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


def chain_elements(text: str) -> dict:
    """The same main chain with BOTH numbers of every element (its sum over the step / the part of it that lies after the
    compute stream was free) and the glue in front of the first reduce-scatter: {lead, rs, glue, ag: (sum, after, count)}."""
    els = []
    for part in text.split(", "):
        m = ELEM.match(part.strip())
        if m:
            els.append((m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4))))
    out = {}
    i = next((k for k, e in enumerate(els) if e[0].startswith("reduce_scatter")), None)
    if i is None:
        return out
    lead = [e for e in els[:i] if e[0] == "glue"]
    if lead:
        out["lead"] = lead[-1][1:]
    out["rs"] = els[i][1:]
    j = next((k for k in range(i + 1, len(els)) if els[k][0] == "glue"), None)
    if j is not None:
        out["glue"] = els[j][1:]
        k = next((q for q in range(j + 1, len(els)) if els[q][0].startswith("all_gather")), None)
        if k is not None:
            out["ag"] = els[k][1:]
    return out


def chain_per_phase(ch: dict) -> dict | None:
    """Per phase, from one step's chain: the glue's end counted from the phase's start (lead + reduce-scatter + glue), the
    glue's part after the compute stream was free, the all-gather's part before it; None without the three elements."""
    if not all(name in ch and ch[name][2] > 0 for name in ("rs", "glue", "ag")):
        return None
    lead = ch["lead"][0] / ch["lead"][2] if "lead" in ch and ch["lead"][2] > 0 else 0.0
    return {"glue_end": lead + ch["rs"][0] / ch["rs"][2] + ch["glue"][0] / ch["glue"][2],
            "after": ch["glue"][1] / ch["glue"][2], "before": (ch["ag"][0] - ch["ag"][1]) / ch["ag"][2]}


T95 = {4: 2.776, 5: 2.571, 6: 2.447, 8: 2.306, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}
MIN_STEPS = 6      # fewer steps of a rank give no slope
FOLLOWS = 0.5      # the slope's whole 95 % interval above it: the glue ends with the segment; below it: it does not follow


def regress(xs: list, ys: list):
    """Slope of y on x with its 95 % half-width (Student t, n - 2 degrees of freedom) and r: (slope, half-width, r, n);
    None with fewer than MIN_STEPS points or without variance."""
    n = len(xs)
    if n < MIN_STEPS:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0:
        return None
    if syy == 0:   # y does not move at all while x does: it does not follow x
        return 0.0, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    res = sum((y - my - slope * (x - mx)) ** 2 for x, y in zip(xs, ys))
    df = n - 2
    t = min((abs(k - df), v) for k, v in T95.items())[1] if df < 40 else 1.98 if df < 150 else 1.96
    return slope, t * math.sqrt(res / df / sxx), sxy / math.sqrt(sxx * syy), n


def harvests(lines, sp_layers: int, min_phases: int, windows, with_layers: bool):
    """-> ({rank: [record]}, harvests left out because their `phase layers` line does not name the tail); a record =
    {"slot": {name: (phase, segment, waited, slack)}, "el": {name: {rs, glue, ag}}, "ch": {name: {glue_end, after, before}
    | None}, "tail": ms | None}, all per phase"""
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
            c["s"], c["e"], c["c"] = {}, {}, {}
            for part in body.split("; "):
                m = SPLIT.match(part.strip())
                if m:
                    c["s"][m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
                    c["e"][m.group(1)] = main_elements(m.group(5))
                    c["c"][m.group(1)] = chain_elements(m.group(5))
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
        rec = {"slot": {}, "el": {}, "ch": {}, "tail": None}
        for k in SLOTS:
            rec["ch"][k] = chain_per_phase(c.get("c", {}).get(k, {}))
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


def glue_row(recs: list, k: str) -> dict | None:
    """One rank, one slot: the glue's end against the end of the compute segment beside the phase, over the rank's steps."""
    pts = [(r["slot"][k][1], r["ch"][k]) for r in recs if r["ch"].get(k) is not None]
    if not pts:
        return None
    segs = [s for s, _c in pts]
    ends = [c["glue_end"] for _s, c in pts]
    return {"n": len(pts), "glue_end": mean(ends), "by_sums": mean(ends) - mean(segs),
            "by_second": mean(c["after"] for _s, c in pts) - mean(c["before"] for _s, c in pts), "slope": regress(segs, ends)}


def fmt_slope(g: dict | None) -> str:
    if g is None:
        return "no slope (no step with the phase's reduce-scatter, glue and all-gather)"
    if g["slope"] is None:
        return f"no slope ({g['n']} steps, {MIN_STEPS} needed)"
    slope, half, r, n = g["slope"]
    return f"slope {slope:+.2f} +- {half:.2f} (r {r:+.2f}) over {n} steps"


def follows(g: dict | None) -> str:
    """Does the glue's end follow the segment's end over the rank's steps: yes | no | open (the interval lies across
    FOLLOWS) | few (no slope)."""
    if g is None or g["slope"] is None:
        return "few"
    slope, half = g["slope"][0], g["slope"][1]
    return "yes" if slope - half > FOLLOWS else "no" if slope + half < FOLLOWS else "open"


def all_gather_waits(rows: dict, glue: dict, slow: int, k: str) -> list:
    """[(rank, how much earlier than the slow rank its glue ends, how much longer it stays in the all-gather, waits)] for
    every other rank with the slot's elements; a rank WAITS when its glue ends at least 0.1 ms earlier and its all-gather is
    longer by at least half of that."""
    ref = glue[slow].get(k)
    out = []
    if ref is None:
        return out
    for rank in rows:
        g = glue[rank].get(k)
        if rank == slow or g is None:
            continue
        earlier = ref["glue_end"] - g["glue_end"]
        longer = rows[rank]["el"][k]["ag"] - rows[slow]["el"][k]["ag"]
        out.append((rank, earlier, longer, earlier >= 0.1 and longer >= 0.5 * earlier))
    return out


def pace_slot(rows: dict, glue: dict, slow: int, k: str) -> str:
    """The reading of one MoE slot for the rank with the longest MoE segment (the module's docstring: pace)."""
    phase, segment, waited, _slack = rows[slow]["slot"][k]
    others = [rows[r]["slot"][k][2] for r in rows if r != slow]
    if waited < 0.05:
        if others and min(others) >= 0.05:
            return "its compute stream does not wait there while the other ranks' do: this rank's card sets the pace of the slot"
        return "no rank's compute stream waits there: the MoE is the long side of the slot"
    g = glue[slow].get(k)
    head = f"its compute stream waits {waited:.3f} ms behind the phase"
    how = follows(g)
    if how == "yes":
        waiting = [(r, longer) for r, _earlier, longer, w in all_gather_waits(rows, glue, slow, k) if w]
        tail = (("; rank" + ("s " if len(waiting) > 1 else " ") + ", ".join(str(r) for r, _l in waiting) + " wait"
                 + ("" if len(waiting) > 1 else "s") + " " + ", ".join(f"{l:.2f}" for _r, l in waiting)
                 + " ms in the all-gather for this rank's glue: its card paces the slot through its glue") if waiting
                else "; no other rank waits in the all-gather for it")
        return (f"{head}, but its glue ends with its MoE segment ({fmt_slope(g)}): the phase is that segment plus "
                f"{phase - segment:.2f} ms, the MoE's own time is on the long side of the slot and a shorter MoE shortens it{tail}")
    if how == "no":
        return (f"{head} and its glue does not follow its MoE segment ({fmt_slope(g)}): the comm stream sets the pace of "
                "the slot, a shorter MoE only lengthens the wait")
    if how == "open":
        return f"{head}; whether its glue ends with its MoE segment is not decided ({fmt_slope(g)}, the interval lies across {FOLLOWS})"
    return f"{head}; whether the comm stream or the MoE paces the slot is not decided: {fmt_slope(g)}"


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

    # the glue's end against the MoE segment's end, per rank; who waits in the all-gather, across the ranks
    moe = {rank: mean(row["slot"][k][1] for k in MOE_SLOTS) for rank, row in rows.items()}
    wait = {rank: min(row["slot"][k][2] for k in MOE_SLOTS) for rank, row in rows.items()}
    slow, fast = max(moe, key=moe.get), min(moe, key=moe.get)
    glue = {rank: {k: glue_row(per[rank], k) for k in MOE_SLOTS} for rank in rows}
    for rank in rows:
        parts = []
        for k in MOE_SLOTS:
            g = glue[rank][k]
            parts.append(f"beside {BESIDE[k]} no element line" if g is None else
                         f"beside {BESIDE[k]} the glue ends {g['by_sums']:+.3f} ms from the MoE segment's end (by the second numbers "
                         f"{g['by_second']:+.3f}), {fmt_slope(g)}")
        print(f"glue: rank {rank}: " + " | ".join(parts))
    if len(rows) > 1:
        parts = []
        for k in MOE_SLOTS:
            phases = [row["slot"][k][0] for row in rows.values()]
            w = all_gather_waits(rows, glue, slow, k)
            waiting = [str(r) for r, _e, _l, yes in w if yes]
            parts.append(f"beside {BESIDE[k]} the phase is {min(phases):.3f} - {max(phases):.3f} ms on the ranks; against rank {slow}: "
                         + (", ".join(f"rank {r} ends its glue {e:+.3f} ms earlier and stays {l:+.3f} ms longer in the all-gather" for r, e, l, _y in w)
                            or "no other rank with the phase's elements")
                         + (" -> rank" + ("s " if len(waiting) > 1 else " ") + ", ".join(waiting) + (" wait" if len(waiting) > 1 else " waits")
                            + f" in the all-gather for rank {slow}'s glue" if waiting else " -> no rank waits in the all-gather for another"))
        print("all-gather: " + " | ".join(parts))

    # pace: per MoE slot, the rank with the longest MoE segments (the docstring: a wait alone does not name the pace-setter)
    others = [wait[r] for r in rows if r != slow]
    spread = 100.0 * (moe[slow] - moe[fast]) / moe[fast] if moe[fast] > 0 else float("nan")
    head = (f"pace: one rank only (MoE segment {moe[slow]:.3f} ms, smallest wait behind a MoE {wait[slow]:.3f} ms)" if not others else
            f"pace: the longest MoE segment is rank {slow}'s ({moe[slow]:.3f} ms, {spread:+.1f} % over rank {fast}'s); its smallest "
            f"wait behind a MoE {wait[slow]:.3f} ms, the other ranks' {min(others):.3f} - {max(others):.3f} ms")
    print(head + ". " + ". ".join(f"Beside {BESIDE[k]}: {pace_slot(rows, glue, slow, k)}" for k in MOE_SLOTS))

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
