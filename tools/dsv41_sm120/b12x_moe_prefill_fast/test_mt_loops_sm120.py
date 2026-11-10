#!/usr/bin/env python3
"""The m-tile loops of the fast prefill MoE (b12x_moe_prefill_fast_sm120, VLLM_MOET_B12X_MOE_PREFILL_FAST_MT_LOOPS and the runtime key
B12X_MOE_PREFILL_FAST_MT_LOOPS): the served FC1 / FC2 kernels with the consumers' stage loop specialised on the item's m-tiles.

The served consumers run the QMMAs of an m-tile under a warp-uniform predicate; a QMMA under a false predicate keeps its stall and its
place in the tensor pipe's issue throttle, so an item of one m-tile pays for four. With the switch an item branches once to a loop that
holds only its own m-tiles. The two states are bit-identical alternatives of the same arithmetic; this test checks exactly that, on
the dumps of a real prefill, in the served switch set:

  --case runtime  (default) the switch off in the environment: the served kernels launch (fc1_kernel_wg<4, 1>, fc2_kernel_ws<4, 1>), the
                  output is b12x's; then the runtime key on: fc1_kernel_wg_mt<4, 1> with one CTA per SM and fc2_kernel_ws_mt<4, 2> with
                  two (at most 96 registers per thread), no occupancy WARNING, the output identical to b12x's at every row count
                  (full row chunks and chunks of 1, 2, 3 m-tiles, both dumps), the m-tile-loop counters grow by exactly the calls and
                  rows launched; then the key flipped on every call: identical throughout; then the key dropped: back to the environment
                  value, the counters stand still;
  --case env      VLLM_MOET_B12X_MOE_PREFILL_FAST_MT_LOOPS=1 from the start: the specialised kernels from the first call, identical.

  python3 test_mt_loops_sm120.py --dumps DIR [--case runtime|env]      (DIR: weights_layer2.pt, call0_layer2.pt, call1_layer2.pt)
Exit 0 only if the case behaves as described.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b12x_moe_prefill_spec"))

P = "VLLM_MOET_B12X_MOE_PREFILL_FAST"
SERVED = {f"{P}_WS1": "9", f"{P}_WS2": "3", f"{P}_SUM_ITEMS": "1", f"{P}_SUM_ROUTE_HINT": "1", f"{P}_SUM_DISCARD": "1",
          f"{P}_SUM_TOKENS": "128", f"{P}_SUM_MIN_ROWS": "512"}
KEY = "B12X_MOE_PREFILL_FAST_MT_LOOPS"
ROWS = (4096, 3000, 1024, 700, 300, 256)  # >= _WS1_MIN_ROWS: every call runs the v10 FC1 kernel and the v9 FC2 kernel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--case", choices=["runtime", "env"], default="runtime")
    a = ap.parse_args()
    os.environ.update(SERVED)
    os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
    if a.case == "env":
        os.environ[f"{P}_MT_LOOPS"] = "1"
    else:
        os.environ.pop(f"{P}_MT_LOOPS", None)

    import torch
    import b12x_moe_prefill_fast_sm120 as m
    from test_b12x_moe_prefill_spec_sm120 import _planes, _served
    import b12x.moe.fused_moe._impl as impl

    if not m.mt_loops_supported():
        print("FAIL: this build of the extension has no m-tile loops")
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
        for rows in ROWS:
            x, ids, w = (c[k][:rows].to(dev).contiguous() for k in ("hidden_states", "topk_ids", "topk_weights"))
            ref = torch.empty(x.shape[0], x.shape[1], dtype=torch.bfloat16, device=dev)
            served.run(x, ids, w, ref)
            torch.cuda.synchronize()
            if planes is None:
                planes = _planes(torch, served.prepared, spied)
            cnt = torch.bincount(ids[ids >= 0].flatten().to(torch.int64), minlength=1)
            mt = (cnt + 15) // 16
            part = int(((mt % 4) != 0).sum())  # experts whose last row chunk has 1-3 m-tiles
            cases.append((f"{call}[0:{rows}]", x, ids, w, ref, part))
    sms = torch.cuda.get_device_properties(0).multi_processor_count

    def run_all(label: str) -> tuple[int, int, int]:
        bad = calls = rows = 0
        for tag, x, ids, w, ref, _ in cases:
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

    def mt_counters() -> tuple[int, int, int, int]:
        c = m.activation_counters()
        return c["fc1_mt_loops"] + c["fc2_mt_loops"]

    ok = True
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        if a.case == "env":
            bad, calls, rows = run_all("env")
            f = facts()
            c = mt_counters()
            good = (bad == 0 and m.runtime_switches().get("MT_LOOPS") == 1 and "fc1_kernel_wg<4, 1>" not in f and "fc2_kernel_ws<4, 1>" not in f
                    and f.get("fc1_kernel_wg_mt<4, 1>", (0, 0))[1] == 1 and f.get("fc2_kernel_ws_mt<4, 2>", (999, 0))[1] == 2
                    and f.get("fc2_kernel_ws_mt<4, 2>", (999, 0))[0] <= 96 and c == (calls, rows, calls, rows))
            print(f"env: {calls} calls, {bad} differ from b12x; kernels launched {f}; m-tile-loop counters {c} (expected {(calls, rows, calls, rows)})")
            ok &= good
        else:
            # 1. the switch off: the served kernels
            bad0, calls0, _ = run_all("off")
            f0 = facts()
            c0 = mt_counters()
            good = (bad0 == 0 and m.runtime_switches().get("MT_LOOPS") == 0 and f0.get("fc1_kernel_wg<4, 1>", (0, 0))[1] == 1
                    and f0.get("fc2_kernel_ws<4, 1>", (0, 0))[1] == 2 and not any(k.startswith(("fc1_kernel_wg_mt", "fc2_kernel_ws_mt")) for k in f0)
                    and c0 == (0, 0, 0, 0))
            print(f"off: {calls0} calls, {bad0} differ from b12x; kernels launched {sorted(f0)}; m-tile-loop counters {c0}")
            ok &= good
            # 2. the runtime key on
            ch = m.apply_runtime_switches({KEY: 1})
            bad1, calls1, rows1 = run_all("on")
            f1 = facts()
            c1 = mt_counters()
            fc1, fc2 = f1.get("fc1_kernel_wg_mt<4, 1>"), f1.get("fc2_kernel_ws_mt<4, 2>")
            good = (ch == ["MT_LOOPS 0 -> 1"] and bad1 == 0 and fc1 is not None and fc1[1] == 1 and fc1[2] == 1 and fc2 is not None and fc2[1] == 2
                    and fc2[2] == 2 and fc2[0] <= 96 and c1 == (calls1, rows1, calls1, rows1))
            print(f"on: change {ch}; {calls1} calls, {bad1} differ from b12x; fc1_kernel_wg_mt<4, 1> {fc1}, fc2_kernel_ws_mt<4, 2> {fc2} "
                  f"(registers, CTAs per SM, measured with, SMs; device SMs {sms}); m-tile-loop counters {c1} (expected {(calls1, rows1, calls1, rows1)})")
            ok &= good
            # 3. flipped on every call
            bad2 = n2 = on_calls = on_rows = 0
            for i in range(3):
                for j, (tag, x, ids, w, ref, _) in enumerate(cases):
                    v = (i + j) & 1
                    m.apply_runtime_switches({KEY: v})
                    out = m.moe_prefill_fast(x, ids, w, planes, 10.0)[0]
                    n2 += 1
                    on_calls += v
                    on_rows += v * int(x.shape[0])
                    if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                        bad2 += 1
                        print(f"  flip: {tag} with MT_LOOPS={v} DIFFERS from b12x")
            torch.cuda.synchronize()
            c2 = mt_counters()
            exp2 = (calls1 + on_calls, rows1 + on_rows, calls1 + on_calls, rows1 + on_rows)
            print(f"flip: {n2} calls with the key alternating, {bad2} differ from b12x; m-tile-loop counters {c2} (expected {exp2})")
            ok &= bad2 == 0 and c2 == exp2
            # 4. the key dropped: the environment value (off)
            m.apply_runtime_switches({KEY: 1})
            ch3 = m.apply_runtime_switches({})
            bad3, calls3, _ = run_all("dropped")
            c3 = mt_counters()
            print(f"dropped: change {ch3}; {calls3} calls, {bad3} differ from b12x; MT_LOOPS now {m.runtime_switches().get('MT_LOOPS')}; "
                  f"m-tile-loop counters {c3} (unchanged: {c3 == c2})")
            ok &= ch3 == ["MT_LOOPS 1 -> 0"] and bad3 == 0 and m.runtime_switches().get("MT_LOOPS") == 0 and c3 == c2
    text = err.getvalue()
    sys.stderr.write(text)
    warnings = [line for line in text.splitlines() if "WARNING" in line]
    part = {tag: p for tag, _, _, _, _, p in cases}
    print(f"experts with a partial last row chunk per case: {part}; WARNING lines {len(warnings)}")
    ok &= not warnings and max(part.values()) > 0
    print(f"{'PASS' if ok else 'FAIL'}: m-tile loops, case {a.case}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
