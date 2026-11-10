#!/usr/bin/env python3
"""The occupancy guard of the fast prefill MoE's WS launches (b12x_moe_prefill_fast_sm120): what each WS kernel got from the occupancy
query at its first launch, and the WARNING / refusal when that is fewer CTAs per SM than its mode was measured with.

The grid of a WS launch is `sms x cudaOccupancyMaxActiveBlocksPerMultiprocessor(kernel)`. FC2's v9 kernel (9 warps) runs two CTAs per SM
at exactly the register count that still allows it (96 per thread); with a few registers more the query answers 1, the grid halves
and FC2 is 10-15 % slower - silently. The guard changes no launch; it reports.

One case per process (the knobs are read once):
  --case served   the served set as it is: fc1_kernel_wg<4, 1> one CTA per SM, fc2_kernel_ws<4, 1> two (96 registers, grid 2 x SMs),
                  no WARNING, the output identical to b12x's served run;
  --case warn     VLLM_MOET_B12X_MOE_PREFILL_FAST_OCC_EXPECT_FC2=3 (an expectation the build cannot meet): exactly one WARNING naming
                  the kernel, its registers and both grids, however many calls follow; the output and the grid unchanged;
  --case strict   the same expectation with VLLM_MOET_B12X_MOE_PREFILL_FAST_OCC_STRICT=1: a RuntimeError at the launch, again at the
                  next call.

  python3 test_occupancy_guard_sm120.py --dumps DIR --case served|warn|strict      (DIR: weights_layer2.pt, call1_layer2.pt)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--case", choices=["served", "warn", "strict"], default="served")
    ap.add_argument("--rows", type=int, default=4096)
    a = ap.parse_args()
    os.environ.update(SERVED)
    os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
    if a.case in ("warn", "strict"):
        os.environ[f"{P}_OCC_EXPECT_FC2"] = "3"
    if a.case == "strict":
        os.environ[f"{P}_OCC_STRICT"] = "1"
    else:
        os.environ.pop(f"{P}_OCC_STRICT", None)

    import torch
    import b12x_moe_prefill_fast_sm120 as m
    from test_b12x_moe_prefill_spec_sm120 import _planes, _served
    import b12x.moe.fused_moe._impl as impl

    dev = torch.device("cuda")
    d = Path(a.dumps)
    served = _served(torch, dev, torch.load(d / "weights_layer2.pt", map_location="cpu", weights_only=False))
    spied = {}
    orig = impl._launch_dynamic

    def spy(**kw):
        spied.update(kw)
        return orig(**kw)

    impl._launch_dynamic = spy
    c = torch.load(d / "call1_layer2.pt", map_location="cpu", weights_only=False)
    x, ids, w = (c[k][: a.rows].to(dev).contiguous() for k in ("hidden_states", "topk_ids", "topk_weights"))
    out_b = torch.empty(x.shape[0], x.shape[1], dtype=torch.bfloat16, device=dev)
    served.run(x, ids, w, out_b)
    torch.cuda.synchronize()
    planes = _planes(torch, served.prepared, spied)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    ok = True
    err = io.StringIO()
    raised = []
    outs = []
    with contextlib.redirect_stderr(err):
        for _ in range(70):  # the guard looks at calls 1-4 and at every 64th
            try:
                outs.append(m.moe_prefill_fast(x, ids, w, planes, 10.0)[0])
            except RuntimeError as e:  # noqa: PERF203
                raised.append(str(e).splitlines()[0])
                if len(raised) >= 2:
                    break
        torch.cuda.synchronize()
    text = err.getvalue()
    sys.stderr.write(text)
    warnings = [line for line in text.splitlines() if "WARNING" in line and "CTA(s) per SM" in line]
    facts = {f"{k}<{s}, {mb}>": (regs, occ, expect, n) for k, s, mb, regs, occ, expect, n in m.occupancy_facts()}
    print(f"occupancy facts (kernel: registers, CTAs per SM, measured with, SMs): {facts}; device SMs {sms}")
    fc2 = facts.get("fc2_kernel_ws<4, 1>")
    fc1 = facts.get("fc1_kernel_wg<4, 1>")
    if a.case == "served":
        same = bool(outs) and all(torch.equal(o.view(torch.int16), out_b.view(torch.int16)) for o in outs[:3])
        good = (fc2 is not None and fc2[1] == 2 and fc2[2] == 2 and fc2[0] <= 96 and fc1 is not None and fc1[1] == 1 and fc1[2] == 1
                and not warnings and not raised and same)
        print(f"served: fc2_kernel_ws<4, 1> {fc2}, fc1_kernel_wg<4, 1> {fc1}; WARNING lines {len(warnings)}; raised {len(raised)}; "
              f"output {'IDENTICAL to b12x' if same else 'DIFFERS'}")
        ok &= good
    elif a.case == "warn":
        same = bool(outs) and all(torch.equal(o.view(torch.int16), out_b.view(torch.int16)) for o in outs[:3])
        good = (fc2 is not None and fc2[1] == 2 and fc2[2] == 3 and len(warnings) == 1 and "fc2_kernel_ws<4, 1>" in warnings[0]
                and f"grid {2 * sms}" in warnings[0] and f"grid {3 * sms}" in warnings[0] and not raised and same and len(outs) == 70)
        print(f"warn: fc2_kernel_ws<4, 1> {fc2}; WARNING lines {len(warnings)} in {len(outs)} calls; output "
              f"{'IDENTICAL to b12x' if same else 'DIFFERS'}; line: {warnings[0][:260] if warnings else '-'}")
        ok &= good
    else:
        good = len(raised) == 2 and all("OCC_STRICT=1" in r and "fc2_kernel_ws" in r for r in raised) and not outs
        print(f"strict: RuntimeError at {len(raised)} of 2 calls, {len(outs)} calls returned; first: {raised[0][:260] if raised else '-'}")
        ok &= good
    print(f"{'PASS' if ok else 'FAIL'}: occupancy guard, case {a.case}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
