#!/usr/bin/env python3
"""The FZ form of the served FC1 kernels of the fast prefill MoE (b12x_moe_prefill_fast_sm120, VLLM_MOET_B12X_MOE_PREFILL_FAST_FC1_FZ and
the runtime key B12X_MOE_PREFILL_FAST_FC1_FZ): the FC1 epilogue with its 32 packed values per thread defined on every path.

The epilogue writes the values for an active m-tile before the consumers' barrier and reads them behind it under the same condition.
The compiler cannot join the two conditions: on the other path the values are undefined, so they stay live around the whole item
loop and hold 32 registers of every consumer thread through the stage loops. The FZ form zero-fills them at the head of the
epilogue; the zeros are never read. The two states are bit-identical alternatives; FC2, the other FC1 modes and the calls below
_WS1_MIN_ROWS rows (FC1 v8) are not affected. The cases:

  --case key      (default; no GPU, no CUDA call, no build - a FAKE extension) the module's side of the key: off without a control
                  file; a file's value is applied before THE SAME call's launch; a key the file drops goes back to the environment's
                  value; an out-of-range value keeps the current one with one WARNING; the key travels with the module's other keys
                  in one file; every change is logged once; `switches now` of the stats line carries
                  B12X_MOE_PREFILL_FAST_FC1_FZ=<n> and the line the counter `fc1 fz <calls> / <rows>`; with a build of the
                  extension that has no such kernels the key is an unknown key (one WARNING), no token, nothing else changes.
  --case runtime  (one GPU, the dumps of a real prefill, the served switch set) the key off: the served kernels, the output is
                  b12x's; the key on: fc1_kernel_wg_fz<4, 1> with one CTA per SM and fewer registers per thread than the served
                  kernel, the output identical at every row count, the counter grows by exactly the calls and rows; with the m-tile
                  loops, with the first claim and with both: the _mt_fz / _fc_fz / _mt_fc_fz kernels, identical; the three keys
                  flipped through their eight states call by call: identical; calls below _WS1_MIN_ROWS rows with the key on:
                  identical, launched as FC1 v8 - the counter does not move; the keys dropped: back to the environment values.

  python3 test_fc1_fz_sm120.py [--case key] [--module DIR]
  python3 test_fc1_fz_sm120.py --case runtime --dumps DIR      (DIR: weights_layer2.pt, call0_layer2.pt, call1_layer2.pt)
Exit 0 only if the case behaves as described.
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

HERE = Path(__file__).resolve().parent
P = "VLLM_MOET_B12X_MOE_PREFILL_FAST"
SERVED = {f"{P}_WS1": "9", f"{P}_WS2": "3", f"{P}_SUM_ITEMS": "1", f"{P}_SUM_ROUTE_HINT": "1", f"{P}_SUM_DISCARD": "1",
          f"{P}_SUM_TOKENS": "128", f"{P}_SUM_MIN_ROWS": "512"}
KEY, KEY_MT, KEY_FC = "B12X_MOE_PREFILL_FAST_FC1_FZ", "B12X_MOE_PREFILL_FAST_MT_LOOPS", "B12X_MOE_PREFILL_FAST_FIRST_CLAIM"
ROWS = (4096, 3000, 1024, 700, 300, 256)  # >= _WS1_MIN_ROWS: every call runs the v10 FC1 kernel
SMALL = (128, 64)  # < _WS1_MIN_ROWS: FC1 v8 - the key cannot act


class FakeExt:
    """The extension's entries the module's runtime side uses; `fz` = whether this build has the FZ kernels."""

    def __init__(self, fz: bool):
        self.sw, self.mt, self.fc, self.fzv, self.calls = [9, 1, 1, 1], 0, 0, 0, []
        self.fz_calls = [0, 0]
        if fz:
            self.set_fc1_fz = self._set_fc1_fz
            self.get_fc1_fz = lambda: self.fzv
            self.moe_prefill_fast_fz_counters = lambda clear: list(self.fz_calls)

    def _set_fc1_fz(self, v):
        assert v in (-1, 0, 1), v
        self.fzv = int(os.environ.get(f"{P}_FC1_FZ", "0")) if v < 0 else v

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
        self.calls.append(self.fzv)  # the form this call's FC1 launch takes
        if self.fzv:
            self.fz_calls[0] += 1
            self.fz_calls[1] += int(a[0].shape[0])
        return "ok"


def key_case(module_dir: Path) -> int:
    sys.path.insert(0, str(module_dir))
    ctl = os.path.join(tempfile.mkdtemp(), "ab.json")
    os.environ.update({"VLLM_MOET_RUNTIME_AB": "1", "VLLM_MOET_RUNTIME_AB_FILE": ctl})
    for k in (f"{P}_FC1_FZ", f"{P}_GRID1", f"{P}_GRID2"):
        os.environ.pop(k, None)
    import torch

    import b12x_moe_prefill_fast_sm120 as m

    fails = [0]

    def check(cond: bool, what: str) -> None:
        print(("PASS " if cond else "FAIL ") + what, flush=True)
        fails[0] += 0 if cond else 1

    planes = {k: torch.zeros(4) for k in ("w13", "w13_sf", "w2", "w2_sf", "alpha", "input_global_scale", "down_alpha", "global_scale")}
    x, ids, w = torch.zeros(8, 16, dtype=torch.bfloat16), torch.zeros(8, 6, dtype=torch.int32), torch.zeros(8, 6)
    tick = [0]

    def write(values: dict) -> None:
        with open(ctl, "w") as f:
            json.dump(values, f)
        tick[0] += 1
        os.utime(ctl, ns=(tick[0], tick[0]))  # a new mtime for every write

    def call(e) -> int:
        m._RT_STATE["next_poll"] = 0.0  # the 100-ms gate off for the test
        m.moe_prefill_fast(x, ids, w, planes)
        return e.calls[-1]

    ond = re.compile(r"activation counters \(on demand, call (\d+); [^)]*\): (.*?); switches now ((?:\w+=-?\d+ ?)+)")
    chg = re.compile(r"runtime A/B: (\w+) (-?\d+) -> (-?\d+)")

    # ---- a build with the FZ kernels
    e = FakeExt(fz=True)
    m._ext = lambda: e
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        check(hasattr(m, "fc1_fz_supported") and m.fc1_fz_supported(), "fc1_fz_supported() with a build that has the kernels")
        check(call(e) == 0 and m.runtime_switches().get("FC1_FZ") == 0, "no control file: the key is off (the environment's value, 0)")
        write({KEY: 1})
        check(call(e) == 1 and m.runtime_switches().get("FC1_FZ") == 1, "{FC1_FZ: 1} is applied before THE SAME call's launch")
        write({KEY_MT: 1})
        check(call(e) == 0 and e.mt == 1, "a file that drops the key and carries the m-tile loops: FC1_FZ back to 0, MT_LOOPS 1")
        write({KEY: 1, KEY_MT: 1, KEY_FC: 1, "B12X_MOE_PREFILL_FAST_GRID2": 256})
        check(call(e) == 1 and e.mt == 1 and e.fc == 1 and m.grid_caps()["GRID2"] == 256,
              "the key travels with the m-tile loops, the first claim and a grid cap in one file")
        m._stats_line("on demand, call 64")
        write({KEY: 7, KEY_MT: 1})
        check(call(e) == 1 and e.fc == 0, "an out-of-range value keeps the current state (1); the first claim, dropped, is off")
        write({KEY: 0})
        check(call(e) == 0, "{FC1_FZ: 0}: off")
        os.environ[f"{P}_FC1_FZ"] = "1"
        write({})
        check(call(e) == 1, "{}: back to the environment's value (here 1)")
        write({KEY: 0})
        check(call(e) == 0, "the file's 0 overrides the environment's 1")
        os.environ.pop(f"{P}_FC1_FZ")
        write({})
        check(call(e) == 0 and m.activation_counters().get("fc1_fz") == (4, 32),
              f"{{}}: off again; the counter saw the 4 calls of 8 rows launched in the FZ form: {m.activation_counters().get('fc1_fz')}")
    text = err.getvalue()
    mo = ond.search(text)
    toks = dict(t.split("=") for t in mo.group(3).split()) if mo else {}
    check(toks.get(KEY) == "1" and toks.get(KEY_MT) == "1" and toks.get(KEY_FC) == "1" and toks.get("B12X_MOE_PREFILL_FAST_GRID2") == "256"
          and toks.get("WS1") == "9", f"`switches now` carries the key's token beside the others: {toks}")
    check(bool(mo) and "fc1 fz 2 / 16" in mo.group(2), f"the stats line carries the counter (2 calls so far): {mo.group(2)[-60:] if mo else None!r}")
    ch = [c for c in chg.findall(text) if c[0] == KEY]
    want = [(KEY, "0", "1"), (KEY, "1", "0")] * 3  # the effective values of the nine calls: 0 1 0 1 1 0 1 0 0
    check(ch == want, f"every change of the key is logged once, in order: {ch}")
    check(text.count("out of range") == 1 and f"{KEY}=7" in text, "one WARNING for the out-of-range value")

    # ---- a build without them: the key is unknown to it
    old = FakeExt(fz=False)
    m._ext = lambda: old
    m._RT_STATE["warned"].clear()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        check(not m.fc1_fz_supported() and "FC1_FZ" not in m.runtime_switches(), "a build without the kernels: fc1_fz_supported() is False")
        write({KEY: 1, KEY_MT: 1})
        check(call(old) == 0 and old.mt == 1, "its control file's key changes nothing; the other keys of the file are applied")
        call(old)
        m._stats_line("on demand, call 128")
    text = err.getvalue()
    mo = ond.search(text)
    check(bool(mo) and KEY not in mo.group(3) and "fc1 fz" not in mo.group(2), "no token and no counter of the key in its stats line")
    check(text.count(f"unknown key {KEY}") == 1, "one WARNING: an unknown key of this module")
    print("ALL PASS" if fails[0] == 0 else f"{fails[0]} FAILED")
    return 1 if fails[0] else 0


def runtime_case(dumps: Path) -> int:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "b12x_moe_prefill_spec"))
    os.environ.update(SERVED)
    os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
    for k in ("MT_LOOPS", "FIRST_CLAIM", "FC1_FZ"):
        os.environ.pop(f"{P}_{k}", None)

    import torch
    import b12x_moe_prefill_fast_sm120 as m
    from test_b12x_moe_prefill_spec_sm120 import _planes, _served
    import b12x.moe.fused_moe._impl as impl

    if not m.fc1_fz_supported():
        print("FAIL: this build of the extension has no FZ form of the served FC1 kernels")
        return 1
    dev = torch.device("cuda")
    served = _served(torch, dev, torch.load(dumps / "weights_layer2.pt", map_location="cpu", weights_only=False))
    spied = {}
    orig = impl._launch_dynamic

    def spy(**kw):
        spied.update(kw)
        return orig(**kw)

    impl._launch_dynamic = spy
    cases, small = [], []  # (tag, x, ids, w, b12x's output)
    planes = None
    for call in ("call1", "call0"):
        c = torch.load(dumps / f"{call}_layer2.pt", map_location="cpu", weights_only=False)
        for rows in ROWS + SMALL:
            x, ids, w = (c[k][:rows].to(dev).contiguous() for k in ("hidden_states", "topk_ids", "topk_weights"))
            ref = torch.empty(x.shape[0], x.shape[1], dtype=torch.bfloat16, device=dev)
            served.run(x, ids, w, ref)
            torch.cuda.synchronize()
            if planes is None:
                planes = _planes(torch, served.prepared, spied)
            (cases if rows in ROWS else small).append((f"{call}[0:{rows}]", x, ids, w, ref))
    impl._launch_dynamic = orig

    def run_all(label: str, which=None) -> tuple[int, int, int]:
        bad = calls = rows = 0
        for tag, x, ids, w, ref in (cases if which is None else which):
            out = m.moe_prefill_fast(x, ids, w, planes, 10.0)[0]
            calls += 1
            rows += int(x.shape[0])
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                bad += 1
                print(f"  {label}: {tag} DIFFERS from b12x")
        torch.cuda.synchronize()
        return bad, calls, rows

    def facts() -> dict:
        return {f"{k}<{s}, {mb}>": (regs, occ, expect, n) for k, s, mb, regs, occ, expect, n in m.occupancy_facts()}

    def fz_counter() -> tuple[int, int]:
        return m.activation_counters()["fc1_fz"]

    ok = True
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        # 1. the key off: the served kernels
        bad, calls, _ = run_all("off")
        f = facts()
        good = (bad == 0 and m.runtime_switches().get("FC1_FZ") == 0 and f.get("fc1_kernel_wg<4, 1>", (0, 0))[1] == 1
                and not any("_fz<" in k for k in f) and fz_counter() == (0, 0))
        print(f"off: {calls} calls, {bad} differ from b12x; kernels launched {sorted(f)}; fz counter {fz_counter()}")
        ok &= good
        exp = [0, 0]
        # 2. the key on, alone and with the module's other kernel-form keys: the four FZ kernels against their own forms
        for label, extra, base, name in (("on", {}, "fc1_kernel_wg<4, 1>", "fc1_kernel_wg_fz<4, 1>"),
                                         ("on + m-tile loops", {KEY_MT: 1}, "fc1_kernel_wg_mt<4, 1>", "fc1_kernel_wg_mt_fz<4, 1>"),
                                         ("on + first claim", {KEY_FC: 1}, "fc1_kernel_wg_fc<4, 1>", "fc1_kernel_wg_fc_fz<4, 1>"),
                                         ("on + m-tile loops + first claim", {KEY_MT: 1, KEY_FC: 1}, "fc1_kernel_wg_mt_fc<4, 1>",
                                          "fc1_kernel_wg_mt_fc_fz<4, 1>")):
            m.apply_runtime_switches(dict(extra))  # the form's own kernel first: its registers for the comparison
            bad0, _, _ = run_all(label + " (key off)")
            c_before = fz_counter()
            ch = m.apply_runtime_switches({KEY: 1, **extra})
            bad, calls, rows = run_all(label)
            f = facts()
            exp[0] += calls
            exp[1] += rows
            k0, k1 = f.get(base), f.get(name)
            good = (ch == [f"{KEY} 0 -> 1"] and bad0 == 0 and bad == 0 and c_before == (exp[0] - calls, exp[1] - rows)
                    and k0 is not None and k1 is not None and k1[1] == 1 and k1[2] == 1 and 0 < k1[0] < k0[0] and fz_counter() == tuple(exp))
            print(f"{label}: change {ch}; {calls} calls, {bad} differ from b12x ({bad0} with the key off); {name} {k1} against {base} {k0} "
                  f"(registers, CTAs per SM, measured with, SMs); fz counter {fz_counter()} (expected {tuple(exp)})")
            ok &= good
            m.apply_runtime_switches({})
        # 3. the three keys through their eight states, call by call
        bad = n = 0
        for i in range(4):
            for j, (tag, x, ids, w, ref) in enumerate(cases):
                s = (i * 3 + j) & 7
                v, t, fc = s & 1, (s >> 1) & 1, (s >> 2) & 1
                m.apply_runtime_switches({KEY: v, KEY_MT: t, KEY_FC: fc})
                out = m.moe_prefill_fast(x, ids, w, planes, 10.0)[0]
                n += 1
                exp[0] += v
                exp[1] += v * int(x.shape[0])
                if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                    bad += 1
                    print(f"  flip: {tag} with FC1_FZ={v} MT_LOOPS={t} FIRST_CLAIM={fc} DIFFERS from b12x")
        torch.cuda.synchronize()
        print(f"flip: {n} calls through the eight states of the three keys, {bad} differ from b12x; fz counter {fz_counter()} (expected {tuple(exp)})")
        ok &= bad == 0 and fz_counter() == tuple(exp)
        # 4. below _WS1_MIN_ROWS rows with the key on: FC1 v8 is launched - the key cannot act, the counter does not move
        m.apply_runtime_switches({KEY: 1})
        a0 = m.activation_counters()
        bad, calls, rows = run_all("small", small)
        a1 = m.activation_counters()
        d_v8 = (a1["fc1_v8_v9"][0] - a0["fc1_v8_v9"][0], a1["fc1_v8_v9"][1] - a0["fc1_v8_v9"][1])
        d_v10 = (a1["fc1_v10"][0] - a0["fc1_v10"][0], a1["fc1_v10"][1] - a0["fc1_v10"][1])
        print(f"small: {calls} calls of {SMALL} rows with the key on, {bad} differ from b12x; launched as fc1 v8-v9 {d_v8}, as fc1 v10 {d_v10}; "
              f"fz counter {a1['fc1_fz']} (unchanged: {a1['fc1_fz'] == tuple(exp)})")
        ok &= bad == 0 and d_v8 == (calls, rows) and d_v10 == (0, 0) and a1["fc1_fz"] == tuple(exp)
        # 5. the keys dropped: the environment values (off)
        ch = m.apply_runtime_switches({})
        bad, calls, _ = run_all("dropped")
        rec = m.watchdog_record()
        print(f"dropped: change {ch}; {calls} calls, {bad} differ from b12x; FC1_FZ now {m.runtime_switches().get('FC1_FZ')}; "
              f"fz counter {fz_counter()} (unchanged: {fz_counter() == tuple(exp)}); watchdog record {rec}")
        ok &= ch == [f"{KEY} 1 -> 0"] and bad == 0 and m.runtime_switches().get("FC1_FZ") == 0 and fz_counter() == tuple(exp) and not any(rec)
    text = err.getvalue()
    sys.stderr.write(text)
    warnings = [line for line in text.splitlines() if "WARNING" in line]
    print(f"WARNING lines {len(warnings)}")
    ok &= not warnings
    print(f"{'PASS' if ok else 'FAIL'}: fc1 fz, case runtime")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["key", "runtime"], default="key")
    ap.add_argument("--module", type=Path, default=HERE, help="case key: where b12x_moe_prefill_fast_sm120.py lives")
    ap.add_argument("--dumps", type=Path, help="case runtime: the dumps of a real prefill")
    a = ap.parse_args()
    if a.case == "key":
        return key_case(a.module)
    if a.dumps is None:
        ap.error("--case runtime needs --dumps DIR")
    return runtime_case(a.dumps)


if __name__ == "__main__":
    sys.exit(main())
