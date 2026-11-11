#!/usr/bin/env python3
"""The grid caps of the fast prefill MoE as runtime keys (B12X_MOE_PREFILL_FAST_GRID1 / _GRID2 in the control file of
VLLM_MOET_RUNTIME_AB=1) on a FAKE extension - no GPU, no CUDA call, no build: the module's reader, the effective caps, the order
(the poll first, then the caps of the same call), the change lines and the tokens of the stats line.

What it holds:
  - without a control file the caps are the environment's (0 = no cap);
  - a file's value is applied before the caps of THE SAME call are taken; a key the file drops goes back to the environment's
    value; the file's 0 overrides an environment cap; an out-of-range value keeps the current cap with one WARNING;
  - a caller's explicit grid wins over both;
  - the keys travel with the module's other keys in one file (the first claim, the m-tile loops);
  - `switches now` of the stats line carries B12X_MOE_PREFILL_FAST_GRID1=<n> B12X_MOE_PREFILL_FAST_GRID2=<n>, every change is
    logged once as `runtime A/B: B12X_MOE_PREFILL_FAST_GRID2 <a> -> <b>`.
That every grid computes the same bits is the harness's statement (test_b12x_moe_prefill_fast_sm120.py with
VLLM_MOET_B12X_MOE_PREFILL_FAST_GRID2=<n>), not this file's.

usage: python3 test_grid_keys_sm120.py [--module DIR]   (DIR = where b12x_moe_prefill_fast_sm120.py lives; default: next to this file)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--module", type=Path, default=Path(__file__).resolve().parent)
args = ap.parse_args()
sys.path.insert(0, str(args.module))
ctl = os.path.join(tempfile.mkdtemp(), "ab.json")
os.environ.update({"VLLM_MOET_RUNTIME_AB": "1", "VLLM_MOET_RUNTIME_AB_FILE": ctl})
for k in ("VLLM_MOET_B12X_MOE_PREFILL_FAST_GRID1", "VLLM_MOET_B12X_MOE_PREFILL_FAST_GRID2"):
    os.environ.pop(k, None)
import torch  # noqa: E402

import b12x_moe_prefill_fast_sm120 as m  # noqa: E402


class FakeExt:
    def __init__(self):
        self.sw, self.mt, self.fc, self.calls = [9, 1, 1, 1], 0, 0, []

    def get_runtime_switches(self):
        return list(self.sw)

    def set_runtime_switches(self, a, b, c, d):
        self.sw = [9 if a < 0 else a, 1 if b < 0 else b, 1 if c < 0 else c, 1 if d < 0 else d]

    def set_mt_loops(self, v):
        self.mt = 0 if v < 0 else v

    def get_mt_loops(self):
        return self.mt

    def set_first_claim(self, v):
        self.fc = 0 if v < 0 else v

    def get_first_claim(self):
        return self.fc

    def moe_prefill_fast_counters(self, clear):
        return [0] * 8

    def moe_prefill_fast_mt_counters(self, clear):
        return [0] * 4

    def moe_prefill_fast_fc_counters(self, clear):
        return [0] * 4

    def occupancy_facts(self):
        return []

    def prequant_rows_supported(self):
        return True

    def moe_prefill_fast(self, *a):
        self.calls.append(a)
        return "ok"


e = FakeExt()
m._ext = lambda: e
fails = 0


def check(cond: bool, what: str) -> None:
    global fails
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    fails += 0 if cond else 1


planes = {k: torch.zeros(4) for k in ("w13", "w13_sf", "w2", "w2_sf", "alpha", "input_global_scale", "down_alpha", "global_scale")}
x, ids, w = torch.zeros(8, 16, dtype=torch.bfloat16), torch.zeros(8, 6, dtype=torch.int32), torch.zeros(8, 6)
tick = [0]


def write(values: dict) -> None:
    with open(ctl, "w") as f:
        json.dump(values, f)
    tick[0] += 1
    os.utime(ctl, ns=(tick[0], tick[0]))  # a new mtime for every write


def call() -> tuple[int, int]:
    m._RT_STATE["next_poll"] = 0.0  # the 100-ms gate off for the test
    m.moe_prefill_fast(x, ids, w, planes)
    a = e.calls[-1]
    return int(a[19]), int(a[20])  # the entry's grid1_cap, grid2_cap


err = io.StringIO()
with contextlib.redirect_stderr(err):
    check(hasattr(m, "grid_caps") and call() == (0, 0) and m.grid_caps() == {"GRID1": 0, "GRID2": 0},
          "no control file: the caps are (0, 0) = the occupancy grids")
    write({"B12X_MOE_PREFILL_FAST_GRID2": 368})
    check(call() == (0, 368) and m.grid_caps()["GRID2"] == 368, "{GRID2: 368} is applied before the caps of THE SAME call are taken")
    write({"B12X_MOE_PREFILL_FAST_GRID1": 184})
    check(call() == (184, 0), "a file that drops GRID2 and carries GRID1: (184, 0)")
    write({})
    check(call() == (0, 0), "{}: back to the environment's values (0, 0)")
    os.environ["VLLM_MOET_B12X_MOE_PREFILL_FAST_GRID2"] = "352"
    check(call() == (0, 352), "the environment's cap is used when the file has no key")
    write({"B12X_MOE_PREFILL_FAST_GRID2": 0})
    check(call() == (0, 0), "the file's 0 overrides the environment's cap")
    write({"B12X_MOE_PREFILL_FAST_GRID2": 99999})
    check(call() == (0, 0), "an out-of-range value keeps the current cap")
    os.environ.pop("VLLM_MOET_B12X_MOE_PREFILL_FAST_GRID2")
    check(m.moe_prefill_fast(x, ids, w, planes, grid2=77) == "ok" and int(e.calls[-1][20]) == 77, "a caller's explicit grid wins")
    write({"B12X_MOE_PREFILL_FAST_GRID2": 320, "B12X_MOE_PREFILL_FAST_MT_LOOPS": 1, "B12X_MOE_PREFILL_FAST_FIRST_CLAIM": 1})
    check(call() == (0, 320) and e.fc == 1 and e.mt == 1, "the cap travels with the first claim and the m-tile loops in one file")
    write({"B12X_MOE_PREFILL_FAST_GRID2": 320, "B12X_MOE_PREFILL_FAST_MT_LOOPS": 1})
    check(call() == (0, 320) and e.fc == 0 and e.mt == 1, "a file that drops the first claim keeps the cap")
    m._stats_line("on demand, call 64")
text = err.getvalue()
OND = re.compile(r"activation counters \(on demand, call (\d+); [^)]*\): (.*?); switches now ((?:\w+=-?\d+ ?)+)")
CHG = re.compile(r"runtime A/B: (\w+) (-?\d+) -> (-?\d+)")
mo = OND.search(text)
toks = dict(t.split("=") for t in mo.group(3).split()) if mo else {}
check(toks.get("B12X_MOE_PREFILL_FAST_GRID2") == "320" and toks.get("B12X_MOE_PREFILL_FAST_GRID1") == "0" and toks.get("WS1") == "9"
      and toks.get("B12X_MOE_PREFILL_FAST_MT_LOOPS") == "1" and toks.get("B12X_MOE_PREFILL_FAST_FIRST_CLAIM") == "0",
      f"`switches now` carries the caps beside the other tokens: {toks}")
ch = [c for c in CHG.findall(text) if c[0].startswith("B12X_MOE_PREFILL_FAST_GRID")]
want = [("B12X_MOE_PREFILL_FAST_GRID2", "0", "368"), ("B12X_MOE_PREFILL_FAST_GRID2", "368", "0"), ("B12X_MOE_PREFILL_FAST_GRID1", "0", "184"),
        ("B12X_MOE_PREFILL_FAST_GRID1", "184", "0"), ("B12X_MOE_PREFILL_FAST_GRID2", "352", "0"), ("B12X_MOE_PREFILL_FAST_GRID2", "0", "320")]
check(sorted(ch) == sorted(want), f"every change is logged once: {ch}")
check(text.count("out of range") == 1, "one WARNING for the out-of-range value")
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
