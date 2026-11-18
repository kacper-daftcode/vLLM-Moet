#!/usr/bin/env python3
"""The twins of the served FC1 / FC2 kernels of the fast prefill MoE (b12x_moe_prefill_fast_sm120, VLLM_MOET_B12X_MOE_PREFILL_FAST_FC_TEXT
and the runtime key B12X_MOE_PREFILL_FAST_FC_TEXT): 1 launches the twins that carry a pinned text (another assembler's schedule of the
same instructions, written over the TX = 1 twin by splice_fc_texts.py), 2 the twins with the compiler's own text, 0 the kernels as
before. The key acts only on the kernels whose pinned text the process found in its loaded extension (the loader's mask) and only
on a call that launches exactly such a kernel. The three states are bit-identical alternatives. The cases:

  --case key      (default; no GPU, no CUDA call, no build - a FAKE extension whose file holds the tree's pinned texts) the module's
                  side: with the runtime A/B switch on, the loader finds the texts once and hands the extension their mask; the key
                  is off without a control file; a file's value is applied before THE SAME call's launch; 2 launches the twins with
                  the compiler's text; a key the file drops goes back to the environment's value; an out-of-range value keeps the
                  current one with one WARNING; the key travels with the module's other keys in one file - and with the m-tile
                  loops on, where no pinned text stands, it launches nothing new; every change is logged once; `switches now`
                  carries B12X_MOE_PREFILL_FAST_FC_TEXT=<n> and FC_TEXT_MASK=<m>, the line the counters `fc1 text`, `fc2 text`,
                  `fc1 twin`, `fc2 twin`; an extension file WITHOUT the texts (a build without the splice step): mask 0, one
                  WARNING per text, the key's value is taken and launches nothing new; a build without twins: an unknown key (one
                  WARNING), no token; with the runtime A/B switch off and the environment's switch unset nothing is read at all.
  --case runtime  (one GPU, the dumps of a real prefill, the served switch set, an extension built WITH the splice step) the key
                  off: the served kernels, the output is b12x's; 1: fc1_kernel_wg_text / fc2_kernel_ws_text with the CTAs per SM
                  and the registers of the kernels they are twins of, the output identical at every row count, the `text` counters
                  grow by exactly the calls and rows; 2: the _twin kernels, identical, the `twin` counters; with the m-tile loops,
                  for the kernels in the mask the _mt_text / _mt_twin kernels, for the others no twin (the counters stand); the key
                  and the m-tile loops flipped through their six states call by call: identical; with the first claim or FZ on,
                  and below _WS1_MIN_ROWS rows for FC1: no FC1 twin; the keys dropped: back to the environment values.

  python3 test_fc_text_sm120.py [--case key] [--module DIR]
  python3 test_fc_text_sm120.py --case runtime --dumps DIR      (DIR: weights_layer2.pt, call0_layer2.pt, call1_layer2.pt)
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
KEY, KEY_MT, KEY_FC, KEY_FZ = (f"B12X_MOE_PREFILL_FAST_{k}" for k in ("FC_TEXT", "MT_LOOPS", "FIRST_CLAIM", "FC1_FZ"))
ROWS = (4096, 3000, 1024, 700, 300, 256)  # >= _WS1_MIN_ROWS: every call runs the v10 FC1 kernel
SMALL = (128, 64)  # < _WS1_MIN_ROWS: FC1 v8 - no FC1 twin; FC2 keeps its mode


class FakeExt:
    """The extension's entries the module's runtime side uses; `tx` = whether this build has the twins; `file` = what the loader reads
    as the loaded extension. moe_prefill_fast applies the extension's rule: a twin only of a kernel named in the mask, not with the
    first claim, FC1 not with FZ."""

    def __init__(self, tx: bool, file: str):
        self.sw, self.mt, self.fc, self.fzv, self.mask, self.calls = [9, 1, 1, 1], 0, 0, 0, 0, []
        self.txv = self._env()
        self.cnt, self.mask_sets = [0] * 8, []
        self.__file__ = file
        if tx:
            self.set_fc_text = self._set_fc_text
            self.get_fc_text = lambda: self.txv
            self.set_fc_text_mask = self._set_mask
            self.get_fc_text_mask = lambda: self.mask
            self.moe_prefill_fast_tx_counters = lambda clear: list(self.cnt)

    @staticmethod
    def _env() -> int:
        v = int(os.environ.get(f"{P}_FC_TEXT", "0"))
        return v if v in (1, 2) else 0

    def _set_fc_text(self, v):
        assert v in (-1, 0, 1, 2), v
        self.txv = self._env() if v < 0 else v

    def _set_mask(self, m):
        assert 0 <= m <= 15, m
        self.mask = m
        self.mask_sets.append(m)

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

    def set_fc1_fz(self, v):
        self.fzv = 0 if v < 0 else v

    def get_fc1_fz(self):
        return self.fzv

    def moe_prefill_fast_counters(self, clear):
        return [0] * 8

    def moe_prefill_fast_mt_counters(self, clear):
        return [0] * 4

    def moe_prefill_fast_fc_counters(self, clear):
        return [0] * 4

    def moe_prefill_fast_fz_counters(self, clear):
        return [0] * 2

    def occupancy_facts(self):
        return []

    def prequant_rows_supported(self):
        return True

    def moe_prefill_fast(self, *a):
        rows = int(a[0].shape[0])
        t1 = self.txv if (self.txv and not self.fc and not self.fzv and (self.mask >> (2 if self.mt else 0)) & 1) else 0
        t2 = self.txv if (self.txv and not self.fc and (self.mask >> (3 if self.mt else 1)) & 1) else 0
        for t, at in ((t1, 0), (t2, 2)):
            if t:
                self.cnt[4 * (t - 1) + at] += 1
                self.cnt[4 * (t - 1) + at + 1] += rows
        self.calls.append((t1, t2))  # the twins this call's FC1 and FC2 launches take (0 = the kernel itself)
        return "ok"


def key_case(module_dir: Path) -> int:
    sys.path.insert(0, str(module_dir))
    tmp = Path(tempfile.mkdtemp(prefix="fc_text_key_"))
    ctl = str(tmp / "ab.json")
    os.environ.update({"VLLM_MOET_RUNTIME_AB": "1", "VLLM_MOET_RUNTIME_AB_FILE": ctl})
    for k in (f"{P}_FC_TEXT", f"{P}_FC1_FZ", f"{P}_GRID1", f"{P}_GRID2"):
        os.environ.pop(k, None)
    import torch

    import b12x_moe_prefill_fast_sm120 as m

    fails = [0]

    def check(cond: bool, what: str) -> None:
        print(("PASS " if cond else "FAIL ") + what, flush=True)
        fails[0] += 0 if cond else 1

    man = json.loads((module_dir / "fc_texts" / "fc_texts.json").read_text(encoding="utf-8"))
    want_mask = 0
    spliced = b"a built extension\0" * 64
    for t in man["texts"]:
        spliced += (module_dir / "fc_texts" / t["file"]).read_bytes() + b"\0" * 32
        want_mask |= 1 << man["kernels"][t["kernel"]]["bit"]
    (tmp / "spliced.so").write_bytes(spliced)
    (tmp / "plain.so").write_bytes(b"a built extension\0" * 64)
    check(want_mask & 3 == 3 and want_mask & 12 == 0, f"the tree's manifest pins a text for fc1 and fc2 without the m-tile loops and none with them (mask {want_mask})")

    planes = {k: torch.zeros(4) for k in ("w13", "w13_sf", "w2", "w2_sf", "alpha", "input_global_scale", "down_alpha", "global_scale")}
    x, ids, w = torch.zeros(8, 16, dtype=torch.bfloat16), torch.zeros(8, 6, dtype=torch.int32), torch.zeros(8, 6)
    tick = [0]

    def write(values: dict) -> None:
        with open(ctl, "w") as f:
            json.dump(values, f)
        tick[0] += 1
        os.utime(ctl, ns=(tick[0], tick[0]))  # a new mtime for every write

    def call(e):
        m._RT_STATE["next_poll"] = 0.0  # the 100-ms gate off for the test
        m.moe_prefill_fast(x, ids, w, planes)
        return e.calls[-1]

    def fresh(e, ab: str = "1") -> None:  # a new process as far as the module's runtime side is concerned
        os.environ["VLLM_MOET_RUNTIME_AB"] = ab
        m._ext = lambda: e
        m._RT_STATE.update({"on": None, "next_poll": 0.0, "sig": None, "calls": 0})
        m._RT_STATE["warned"].clear()
        m._FC_TEXT["mask"] = None
        with contextlib.suppress(OSError):
            os.remove(ctl)

    ond = re.compile(r"activation counters \(on demand, call (\d+); [^)]*\): (.*?); switches now ((?:\w+=-?\d+ ?)+)")
    chg = re.compile(r"runtime A/B: (\w+) (-?\d+) -> (-?\d+)")

    # ---- a build with the twins whose file holds the pinned texts
    e = FakeExt(True, str(tmp / "spliced.so"))
    fresh(e)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        check(hasattr(m, "fc_text_supported") and m.fc_text_supported(), "fc_text_supported() with a build that has the twins")
        check(call(e) == (0, 0) and m.runtime_switches().get("FC_TEXT") == 0, "no control file: the key is off (the environment's value, 0)")
        check(e.mask_sets == [want_mask] and m.fc_text_mask() == want_mask,
              f"the loader found the pinned texts in the extension's file once, before the first launch, and handed over their mask: {e.mask_sets}")
        write({KEY: 1})
        check(call(e) == (1, 1) and m.runtime_switches().get("FC_TEXT") == 1, "{FC_TEXT: 1} is applied before THE SAME call's launch: both twins with a pinned text")
        write({KEY: 2})
        check(call(e) == (2, 2), "{FC_TEXT: 2}: both twins with the compiler's text")
        write({KEY_MT: 1})
        check(call(e) == (0, 0) and e.mt == 1, "a file that drops the key and carries the m-tile loops: FC_TEXT back to 0, MT_LOOPS 1")
        write({KEY: 1, KEY_MT: 1, "B12X_MOE_PREFILL_FAST_GRID2": 256})
        check(call(e) == (0, 0) and m.runtime_switches().get("FC_TEXT") == 1 and e.mt == 1 and m.grid_caps()["GRID2"] == 256,
              "the key travels with the m-tile loops and a grid cap in one file - and launches nothing new there: no text of the loop kernels stands")
        m._stats_line("on demand, call 64")
        write({KEY: 1, "B12X_MOE_PREFILL_FAST_GRID2": 256})
        check(call(e) == (1, 1), "the m-tile loops dropped, the key kept: the twins with a pinned text again")
        write({KEY: 7})
        check(call(e) == (1, 1), "an out-of-range value keeps the current state (1)")
        write({KEY: 1, KEY_FC: 1})
        check(call(e) == (0, 0), "with the first claim on the key launches no twin")
        write({KEY: 1, KEY_FZ: 1})
        check(call(e) == (0, 1), "with FZ on FC1 launches no twin, FC2 does")
        write({KEY: 0})
        check(call(e) == (0, 0), "{FC_TEXT: 0}: off")
        os.environ[f"{P}_FC_TEXT"] = "2"
        write({})
        check(call(e) == (2, 2), "{}: back to the environment's value (here 2)")
        write({KEY: 0})
        check(call(e) == (0, 0), "the file's 0 overrides the environment's 2")
        os.environ.pop(f"{P}_FC_TEXT")
        write({})
        c = m.activation_counters()
        check(call(e) == (0, 0) and (c.get("fc1_text"), c.get("fc2_text"), c.get("fc1_twin"), c.get("fc2_twin")) == ((3, 24), (4, 32), (2, 16), (2, 16)),
              f"{{}}: off again; the counters saw the twins' launches (calls, rows): text {c.get('fc1_text')} {c.get('fc2_text')}, twin {c.get('fc1_twin')} {c.get('fc2_twin')}")
    text = err.getvalue()
    mo = ond.search(text)
    toks = dict(t.split("=") for t in mo.group(3).split()) if mo else {}
    check(toks.get(KEY) == "1" and toks.get("FC_TEXT_MASK") == str(want_mask) and toks.get(KEY_MT) == "1" and toks.get("B12X_MOE_PREFILL_FAST_GRID2") == "256"
          and toks.get("WS1") == "9", f"`switches now` carries the key's token and the mask beside the others: {toks}")
    check(bool(mo) and "fc1 text 1 / 8, fc2 text 1 / 8, fc1 twin 1 / 8, fc2 twin 1 / 8" in mo.group(2),
          f"the stats line carries the four counters (one call of each kind so far): {mo.group(2)[-70:] if mo else None!r}")
    ch = [c for c in chg.findall(text) if c[0] == KEY]
    want = [(KEY, "0", "1"), (KEY, "1", "2"), (KEY, "2", "0"), (KEY, "0", "1"), (KEY, "1", "0"), (KEY, "0", "2"), (KEY, "2", "0")]
    check(ch == want, f"every change of the key is logged once, in order: {[c[1] + '>' + c[2] for c in ch]}")
    check(text.count("out of range") == 1 and f"{KEY}=7" in text, "one WARNING for the out-of-range value")
    check(text.count("fc texts: mask") == 1 and all(t["name"] in text for t in man["texts"]), "one line names the texts found and their mask")

    # ---- the same build, an extension file WITHOUT the texts (no splice step): the key's value is taken and launches nothing new
    e2 = FakeExt(True, str(tmp / "plain.so"))
    fresh(e2)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        write({KEY: 1})
        check(call(e2) == (0, 0) and m.runtime_switches().get("FC_TEXT") == 1 and e2.mask_sets == [0],
              "an extension without the splice: mask 0 - {FC_TEXT: 1} is taken and launches no twin")
        write({KEY: 2})
        check(call(e2) == (0, 0), "and {FC_TEXT: 2} launches none either: the twins ride only where a text stands")
    text = err.getvalue()
    check(text.count("does not stand in the loaded extension") == len(man["texts"]), "one WARNING per text that does not stand in the extension")

    # ---- a build without twins: the key is unknown to it
    old = FakeExt(False, str(tmp / "plain.so"))
    fresh(old)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        check(not m.fc_text_supported() and "FC_TEXT" not in m.runtime_switches() and m.fc_text_mask() == 0, "a build without the twins: fc_text_supported() is False, mask 0")
        write({KEY: 1, KEY_MT: 1})
        check(call(old) == (0, 0) and old.mt == 1, "its control file's key changes nothing; the other keys of the file are applied")
        call(old)
        m._stats_line("on demand, call 128")
    text = err.getvalue()
    mo = ond.search(text)
    check(bool(mo) and KEY not in mo.group(3) and "fc1 text" not in mo.group(2), "no token and no counter of the key in its stats line")
    check(text.count(f"unknown key {KEY}") == 1, "one WARNING: an unknown key of this module")

    # ---- the runtime A/B switch off and the environment's switch unset: nothing is read; with the environment's switch: the mask is found
    e3 = FakeExt(True, str(tmp / "spliced.so"))
    fresh(e3, ab="0")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        check(call(e3) == (0, 0) and m._FC_TEXT["mask"] is None and e3.mask_sets == [] and err.getvalue() == "",
              "the runtime A/B switch off, no environment switch: no manifest read, no mask set, no line - the launch as before")
    os.environ[f"{P}_FC_TEXT"] = "1"
    e4 = FakeExt(True, str(tmp / "spliced.so"))
    fresh(e4, ab="0")
    with contextlib.redirect_stderr(io.StringIO()):
        check(call(e4) == (1, 1) and e4.mask_sets == [want_mask], "the environment's switch alone (no runtime A/B): the loader finds the texts, the twins are launched")
    os.environ.pop(f"{P}_FC_TEXT")
    print("ALL PASS" if fails[0] == 0 else f"{fails[0]} FAILED")
    return 1 if fails[0] else 0


def runtime_case(dumps: Path) -> int:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "b12x_moe_prefill_spec"))
    os.environ.update(SERVED)
    os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
    os.environ["VLLM_MOET_RUNTIME_AB"] = "1"  # the loader looks for the texts; no control file is used: the states are applied directly
    os.environ["VLLM_MOET_RUNTIME_AB_FILE"] = os.path.join(tempfile.mkdtemp(prefix="fc_text_rt_"), "absent.json")
    for k in ("MT_LOOPS", "FIRST_CLAIM", "FC1_FZ", "FC_TEXT"):
        os.environ.pop(f"{P}_{k}", None)

    import torch
    import b12x_moe_prefill_fast_sm120 as m
    from test_b12x_moe_prefill_spec_sm120 import _planes, _served
    import b12x.moe.fused_moe._impl as impl

    if not m.fc_text_supported():
        print("FAIL: this build of the extension has no twins of the served FC1 / FC2 kernels")
        return 1
    mask = m.fc_text_mask()
    if mask & 3 != 3:
        print(f"FAIL: the loaded extension does not carry the pinned texts of fc1 and fc2 (mask {mask}) - built without the splice step?")
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

    def tx() -> tuple:
        c = m.activation_counters()
        return c["fc1_text"] + c["fc2_text"] + c["fc1_twin"] + c["fc2_twin"]

    ok = True
    exp = [0] * 8  # [text: fc1 calls, rows, fc2 calls, rows; twin: the same four]
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        # 1. the key off: the served kernels
        bad, calls, _ = run_all("off")
        f = facts()
        good = bad == 0 and m.runtime_switches().get("FC_TEXT") == 0 and not any("_text<" in k or "_twin<" in k for k in f) and tx() == tuple(exp)
        print(f"off: mask {mask}; {calls} calls, {bad} differ from b12x; kernels launched {sorted(f)}; twin counters {tx()}")
        ok &= good
        # 2. the key at 1 and at 2, without and with the m-tile loops: the twins against the kernels they are twins of
        for mt, b1, b2 in ((0, "fc1_kernel_wg<4, 1>", "fc2_kernel_ws<4, 1>"), (1, "fc1_kernel_wg_mt<4, 1>", "fc2_kernel_ws_mt<4, 2>")):
            extra = {KEY_MT: 1} if mt else {}
            in1, in2 = bool(mask >> (2 if mt else 0) & 1), bool(mask >> (3 if mt else 1) & 1)
            m.apply_runtime_switches(dict(extra))  # the kernels themselves first: their registers for the comparison
            bad0, _, _ = run_all(f"loops {mt} (key off)")
            for v, suffix in ((1, "text"), (2, "twin")):
                ch = m.apply_runtime_switches({KEY: v, **extra})
                bad, calls, rows = run_all(f"loops {mt}, key {v}")
                f = facts()
                at = 4 * (v - 1)
                if in1:
                    exp[at] += calls
                    exp[at + 1] += rows
                if in2:
                    exp[at + 2] += calls
                    exp[at + 3] += rows
                n1, n2 = b1.replace("<", f"_{suffix}<"), b2.replace("<", f"_{suffix}<")
                k0, k1, q0, q1 = f.get(b1), f.get(n1), f.get(b2), f.get(n2)
                same1 = (k1 is not None and k0 is not None and k1[:3] == k0[:3]) if in1 else k1 is None
                same2 = (q1 is not None and q0 is not None and q1[:3] == q0[:3]) if in2 else q1 is None
                good = bad0 == 0 and bad == 0 and len(ch) == 1 and ch[0].startswith(KEY + " ") and same1 and same2 and tx() == tuple(exp)
                print(f"loops {mt}, key {v}: change {ch}; {calls} calls, {bad} differ from b12x ({bad0} with the key off); {n1} {k1} against {b1} {k0}; "
                      f"{n2} {q1} against {b2} {q0} (registers, CTAs per SM, expected, SMs; None = not launched: no pinned text for that kernel); "
                      f"twin counters {tx()} (expected {tuple(exp)})")
                ok &= good
            m.apply_runtime_switches({})
        # 3. the key and the m-tile loops through their six states, call by call
        bad = n = 0
        for i in range(4):
            for j, (tag, x, ids, w, ref) in enumerate(cases):
                s = (i * 5 + j) % 6
                v, t = s % 3, s // 3
                m.apply_runtime_switches({KEY: v, KEY_MT: t})
                out = m.moe_prefill_fast(x, ids, w, planes, 10.0)[0]
                n += 1
                if v:
                    for bit, at in ((2 if t else 0, 0), (3 if t else 1, 2)):
                        if mask >> bit & 1:
                            exp[4 * (v - 1) + at] += 1
                            exp[4 * (v - 1) + at + 1] += int(x.shape[0])
                if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                    bad += 1
                    print(f"  flip: {tag} with FC_TEXT={v} MT_LOOPS={t} DIFFERS from b12x")
        torch.cuda.synchronize()
        print(f"flip: {n} calls through the six states of the two keys, {bad} differ from b12x; twin counters {tx()} (expected {tuple(exp)})")
        ok &= bad == 0 and tx() == tuple(exp)
        # 4. where the key cannot act on FC1: the first claim, FZ, and calls below _WS1_MIN_ROWS rows (FC1 v8); FC2 follows its own rule
        for label, extra, which, fc2_rides in (("first claim", {KEY_FC: 1}, None, False), ("FZ", {KEY_FZ: 1}, None, True), ("small", {}, small, True)):
            m.apply_runtime_switches({KEY: 1, **extra})
            bad, calls, rows = run_all(label, which)
            if fc2_rides and mask >> 1 & 1:
                exp[2] += calls
                exp[3] += rows
            print(f"{label}: {calls} calls with the key at 1, {bad} differ from b12x; twin counters {tx()} (expected {tuple(exp)}: no FC1 twin"
                  f"{', FC2 rides its twin' if fc2_rides else ', no FC2 twin'})")
            ok &= bad == 0 and tx() == tuple(exp)
        # 5. the keys dropped: the environment values (off)
        ch = m.apply_runtime_switches({})
        bad, calls, _ = run_all("dropped")
        rec = m.watchdog_record()
        print(f"dropped: change {ch}; {calls} calls, {bad} differ from b12x; FC_TEXT now {m.runtime_switches().get('FC_TEXT')}; "
              f"twin counters {tx()} (unchanged: {tx() == tuple(exp)}); watchdog record {rec}")
        ok &= ch == [f"{KEY} 1 -> 0"] and bad == 0 and m.runtime_switches().get("FC_TEXT") == 0 and tx() == tuple(exp) and not any(rec)
    text = err.getvalue()
    sys.stderr.write(text)
    warnings = [line for line in text.splitlines() if "WARNING" in line and "control file" not in line]
    print(f"WARNING lines {len(warnings)} (the absent control file's one line aside)")
    ok &= not warnings
    print(f"{'PASS' if ok else 'FAIL'}: fc text, case runtime")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["key", "runtime"], default="key")
    ap.add_argument("--module", type=Path, default=HERE, help="case key: where b12x_moe_prefill_fast_sm120.py and fc_texts/ live")
    ap.add_argument("--dumps", type=Path, help="case runtime: the dumps of a real prefill")
    a = ap.parse_args()
    if a.case == "key":
        return key_case(a.module)
    if a.dumps is None:
        ap.error("--case runtime needs --dumps DIR")
    return runtime_case(a.dumps)


if __name__ == "__main__":
    sys.exit(main())
