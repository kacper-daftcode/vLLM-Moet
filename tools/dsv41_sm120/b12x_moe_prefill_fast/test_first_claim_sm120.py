#!/usr/bin/env python3
"""The first claim of the fast prefill MoE (b12x_moe_prefill_fast_sm120, VLLM_MOET_B12X_MOE_PREFILL_FAST_FIRST_CLAIM and the runtime key
B12X_MOE_PREFILL_FAST_FIRST_CLAIM): the served FC1 / FC2 kernels with every CTA claiming its FIRST work item from the work counter.

In the served kernels a CTA's first item is the one of its block index. FC1 then cannot end, and the sum items of FC2's first column
tile cannot pass, before the last CTA of the grid has run that item: a kernel of another stream that holds one of the grid's SMs when
FC2 starts - a collective needs whole SMs - stops FC2 for as long as it stays, and past the sum items' watchdog the launch traps. With
the switch a CTA that has not been dispatched owns nothing. The two states are bit-identical alternatives (which CTA computes an item
changes, as it already does for every item but the first). The cases, on the dumps of a real prefill, in the served switch set:

  --case runtime  (default) the switch off: the served kernels, the output is b12x's; the runtime key on: fc1_kernel_wg_fc<4, 1> with one
                  CTA per SM and fc2_kernel_ws_fc<4, 2> with two (at most 96 registers per thread), no occupancy WARNING, the output
                  identical at every row count, the first-claim counters grow by exactly the calls and rows; with the m-tile loops on
                  as well: the _mt_fc kernels, identical; the two keys flipped through their four states call by call: identical; the
                  keys dropped: back to the environment values.
  --case hold     the switch ON, the call beside a kernel that holds four whole SMs for longer than the sum items' watchdog and is
                  issued while FC1 runs (test_first_claim_hold_sm120.cu; the watchdog is set to 200 ms, the hold to 500 ms): the call
                  ends without an error, its output is b12x's, the watchdog record stays empty. (The launch still ends only when its
                  last CTA has been dispatched - after the hold; its work does not wait.)
  --case trap     the switch OFF, the same hold: the sum items of the first tile wait for the eight CTAs the hold keeps out, the
                  watchdog expires and the launch TRAPS - the expected outcome of this case. The watchdog record must then name the
                  wait: set, what = 100 (the sum items of tile 0), need - seen = 64 arrivals (8 CTAs x 8 consumer warps).
                  THE PROCESS'S CUDA CONTEXT DOES NOT SURVIVE THIS CASE and the driver logs Xid 43; run it alone, and open one fresh
                  CUDA context on the card afterwards (the utilisation reading of a trapped kernel is cleared by it).

  python3 test_first_claim_sm120.py --dumps DIR [--case runtime|hold|trap]      (DIR: weights_layer2.pt, call0_layer2.pt, call1_layer2.pt)
Exit 0 only if the case behaves as described.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b12x_moe_prefill_spec"))

P = "VLLM_MOET_B12X_MOE_PREFILL_FAST"
SERVED = {f"{P}_WS1": "9", f"{P}_WS2": "3", f"{P}_SUM_ITEMS": "1", f"{P}_SUM_ROUTE_HINT": "1", f"{P}_SUM_DISCARD": "1",
          f"{P}_SUM_TOKENS": "128", f"{P}_SUM_MIN_ROWS": "512"}
KEY, KEY_MT = "B12X_MOE_PREFILL_FAST_FIRST_CLAIM", "B12X_MOE_PREFILL_FAST_MT_LOOPS"
ROWS = (4096, 3000, 1024, 700, 300, 256)  # >= _WS1_MIN_ROWS: every call runs the v10 FC1 kernel and the v9 FC2 kernel
WHOLE_SM = 98560  # bytes of shared memory: a CTA that needs an empty SM
WATCHDOG_MS, HOLD_MS, HOLD_AFTER_US = 200, 500, 300


def hold_ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    return load(name="vllm_moet_test_first_claim_hold_sm120", sources=[str(HERE / "test_first_claim_hold_sm120.cu")],
                extra_cuda_cflags=["-O3", "-std=c++20"], extra_cflags=["-O3", "-std=c++20"], verbose=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--case", choices=["runtime", "hold", "trap"], default="runtime")
    a = ap.parse_args()
    os.environ.update(SERVED)
    os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
    os.environ.pop(f"{P}_MT_LOOPS", None)
    os.environ.pop(f"{P}_FIRST_CLAIM", None)
    if a.case != "runtime":
        os.environ[f"{P}_SUM_WATCHDOG_MS"] = str(WATCHDOG_MS)
        os.environ[f"{P}_FIRST_CLAIM"] = "1" if a.case == "hold" else "0"

    import torch
    import b12x_moe_prefill_fast_sm120 as m
    from test_b12x_moe_prefill_spec_sm120 import _planes, _served
    import b12x.moe.fused_moe._impl as impl

    if not m.first_claim_supported():
        print("FAIL: this build of the extension has no first-claim kernels")
        return 1
    dev = torch.device("cuda")
    d = Path(a.dumps)
    served = _served(torch, dev, torch.load(d / "weights_layer2.pt", map_location="cpu", weights_only=False))
    spied = {}
    orig = impl._launch_dynamic

    def spy(**kw):
        spied.update(kw)
        return orig(**kw)

    impl._launch_dynamic = spy
    cases = []  # (tag, x, ids, w, b12x's output)
    planes = None
    for call in ("call1", "call0"):
        c = torch.load(d / f"{call}_layer2.pt", map_location="cpu", weights_only=False)
        for rows in (ROWS if a.case == "runtime" else ROWS[:1]):
            x, ids, w = (c[k][:rows].to(dev).contiguous() for k in ("hidden_states", "topk_ids", "topk_weights"))
            ref = torch.empty(x.shape[0], x.shape[1], dtype=torch.bfloat16, device=dev)
            served.run(x, ids, w, ref)
            torch.cuda.synchronize()
            if planes is None:
                planes = _planes(torch, served.prepared, spied)
            cases.append((f"{call}[0:{rows}]", x, ids, w, ref))
        if a.case != "runtime":
            break
    impl._launch_dynamic = orig

    if a.case != "runtime":
        return hold_case(a.case, torch, m, cases[0], planes)

    def run_all(label: str) -> tuple[int, int, int]:
        bad = calls = rows = 0
        for tag, x, ids, w, ref in cases:
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

    def fc_counters() -> tuple[int, int, int, int]:
        c = m.activation_counters()
        return c["fc1_first_claim"] + c["fc2_first_claim"]

    ok = True
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        # 1. the switch off: the served kernels
        bad0, calls0, _ = run_all("off")
        f0 = facts()
        c0 = fc_counters()
        good = (bad0 == 0 and m.runtime_switches().get("FIRST_CLAIM") == 0 and f0.get("fc1_kernel_wg<4, 1>", (0, 0))[1] == 1
                and f0.get("fc2_kernel_ws<4, 1>", (0, 0))[1] == 2 and not any("_fc<" in k for k in f0) and c0 == (0, 0, 0, 0))
        print(f"off: {calls0} calls, {bad0} differ from b12x; kernels launched {sorted(f0)}; first-claim counters {c0}")
        ok &= good
        # 2. the runtime key on
        ch = m.apply_runtime_switches({KEY: 1})
        bad1, calls1, rows1 = run_all("on")
        f1 = facts()
        c1 = fc_counters()
        fc1, fc2 = f1.get("fc1_kernel_wg_fc<4, 1>"), f1.get("fc2_kernel_ws_fc<4, 2>")
        good = (ch == [f"{KEY} 0 -> 1"] and bad1 == 0 and fc1 is not None and fc1[1] == 1 and fc1[2] == 1 and fc2 is not None and fc2[1] == 2
                and fc2[2] == 2 and fc2[0] <= 96 and c1 == (calls1, rows1, calls1, rows1))
        print(f"on: change {ch}; {calls1} calls, {bad1} differ from b12x; fc1_kernel_wg_fc<4, 1> {fc1}, fc2_kernel_ws_fc<4, 2> {fc2} "
              f"(registers, CTAs per SM, measured with, SMs); first-claim counters {c1} (expected {(calls1, rows1, calls1, rows1)})")
        ok &= good
        # 3. with the m-tile loops as well
        ch = m.apply_runtime_switches({KEY: 1, KEY_MT: 1})
        bad2, calls2, rows2 = run_all("on + m-tile loops")
        f2 = facts()
        c2 = fc_counters()
        g1, g2 = f2.get("fc1_kernel_wg_mt_fc<4, 1>"), f2.get("fc2_kernel_ws_mt_fc<4, 2>")
        exp2 = (calls1 + calls2, rows1 + rows2, calls1 + calls2, rows1 + rows2)
        good = (ch == [f"{KEY_MT} 0 -> 1"] and bad2 == 0 and g1 is not None and g1[1] == 1 and g2 is not None and g2[1] == 2 and g2[0] <= 96
                and c2 == exp2)
        print(f"on + m-tile loops: change {ch}; {calls2} calls, {bad2} differ from b12x; fc1_kernel_wg_mt_fc<4, 1> {g1}, "
              f"fc2_kernel_ws_mt_fc<4, 2> {g2}; first-claim counters {c2} (expected {exp2})")
        ok &= good
        # 4. the two keys through their four states, call by call
        bad3 = n3 = on_calls = on_rows = 0
        for i in range(2):
            for j, (tag, x, ids, w, ref) in enumerate(cases):
                v, t = (i + j) & 1, ((i + j) >> 1) & 1
                m.apply_runtime_switches({KEY: v, KEY_MT: t})
                out = m.moe_prefill_fast(x, ids, w, planes, 10.0)[0]
                n3 += 1
                on_calls += v
                on_rows += v * int(x.shape[0])
                if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                    bad3 += 1
                    print(f"  flip: {tag} with FIRST_CLAIM={v} MT_LOOPS={t} DIFFERS from b12x")
        torch.cuda.synchronize()
        c3 = fc_counters()
        exp3 = (exp2[0] + on_calls, exp2[1] + on_rows, exp2[2] + on_calls, exp2[3] + on_rows)
        print(f"flip: {n3} calls through the four states of the two keys, {bad3} differ from b12x; first-claim counters {c3} (expected {exp3})")
        ok &= bad3 == 0 and c3 == exp3
        # 5. the keys dropped: the environment values (off)
        m.apply_runtime_switches({KEY: 1})
        ch5 = m.apply_runtime_switches({})
        bad5, calls5, _ = run_all("dropped")
        c5 = fc_counters()
        rec = m.watchdog_record()
        print(f"dropped: change {ch5}; {calls5} calls, {bad5} differ from b12x; FIRST_CLAIM now {m.runtime_switches().get('FIRST_CLAIM')}; "
              f"first-claim counters {c5} (unchanged: {c5 == c3}); watchdog record {rec}")
        ok &= ch5 == [f"{KEY} 1 -> 0"] and bad5 == 0 and m.runtime_switches().get("FIRST_CLAIM") == 0 and c5 == c3 and not any(rec)
    text = err.getvalue()
    sys.stderr.write(text)
    warnings = [line for line in text.splitlines() if "WARNING" in line]
    print(f"WARNING lines {len(warnings)}")
    ok &= not warnings
    print(f"{'PASS' if ok else 'FAIL'}: first claim, case runtime")
    return 0 if ok else 1


def hold_case(case: str, torch, m, one, planes) -> int:
    """The served call on one stream, a kernel that holds four whole SMs on another, issued while FC1 runs and held past the watchdog."""
    tag, x, ids, w, ref = one
    h = hold_ext()
    dev = x.device
    compute, other = torch.cuda.Stream(device=dev), torch.cuda.Stream(device=dev)
    rec = torch.zeros(4, 3, dtype=torch.int64, device=dev)
    with torch.cuda.stream(other):
        h.hold(rec, 4, 544, WHOLE_SM, 50_000)
    for _ in range(3):
        with torch.cuda.stream(compute):
            m.moe_prefill_fast(x, ids, w, planes, 10.0)
    torch.cuda.synchronize()
    m.watchdog_record(clear=True)
    fc = m.runtime_switches().get("FIRST_CLAIM")
    print(f"case {case}: {tag}, FIRST_CLAIM={fc}, watchdog {WATCHDOG_MS} ms, a hold of 4 whole-SM CTAs for {HOLD_MS} ms issued {HOLD_AFTER_US} us after the call",
          flush=True)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter_ns()
    error = None
    out = None
    try:
        with torch.cuda.stream(compute):
            e0.record(compute)
            out = m.moe_prefill_fast(x, ids, w, planes, 10.0)[0]
            e1.record(compute)
        while time.perf_counter_ns() < t0 + HOLD_AFTER_US * 1000:
            pass
        with torch.cuda.stream(other):
            h.hold(rec, 4, 544, WHOLE_SM, HOLD_MS * 1_000_000)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - the expected outcome of the trap case
        error = exc
    record = m.watchdog_record()  # a host read: works after a trapped launch
    raw = [int(v) for v in m._ext().moe_prefill_ws_error(False)]  # the extension's own entry: its `set` must not depend on the last store
    if case == "trap":
        good = (error is not None and record[0] == 1 and raw[0] == 1 and record[1] == 100 and record[3] - record[4] == 64
                and record[7] == WATCHDOG_MS)
        print(f"trap: {'the launch failed: ' + type(error).__name__ if error is not None else 'NO error'}; watchdog record "
              f"[set, what, item, need, seen, smid, block, watchdog_ms] = {record}; the extension's entry reports set = {raw[0]}")
        print(f"{'PASS' if good else 'FAIL'}: first claim, case trap (the switch off: the watchdog record names the sum items of tile 0, 64 arrivals missing)",
              flush=True)
        sys.stdout.flush()
        os._exit(0 if good else 1)  # the context is gone: no teardown
    same = out is not None and bool(torch.equal(out.view(torch.int16), ref.view(torch.int16)))
    ms = e0.elapsed_time(e1) if error is None else float("nan")
    good = error is None and same and not any(record)
    print(f"hold: error {error!r}; the call's launches ended after {ms:.1f} ms (the hold: {HOLD_MS} ms); output identical to b12x: {same}; "
          f"watchdog record {record}")
    print(f"{'PASS' if good else 'FAIL'}: first claim, case hold (the switch on: no trap, the output is b12x's)")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
