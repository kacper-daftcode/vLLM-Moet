#!/usr/bin/env python3
"""The bounded waits of the persistent prefill MoE launch (b12x_moe_prefill_fast_sm120, VLLM_MOET_B12X_MOE_PREFILL_FAST_WS >= 20):
the launch beside a kernel that holds whole SMs, a wait that lasts and ends, and a wait that EXPIRES.

Every wait of moe_persistent_kernel is bounded by the launch's watchdog (_WS_WATCHDOG_MS, 2000 ms unless the environment says
otherwise): a wait that has not completed by then writes one record to pinned host memory - which wait, what it needed and what it
saw, its SM and block - and traps, so that a launch that cannot make progress fails loudly instead of spinning. The harness
(test_b12x_moe_prefill_fast_sm120.py) never makes a wait last: its record stays empty and the fail path does not run. Here a wait
is made to last. A CTA's first work item is the one of its block index; an FC2 item waits for the FC1 items of its expert (the two
A-producer warps poll the expert's arrival counter: `what` 1 and 2 of the record); and a CTA that waits there claims nothing
further. A kernel of another stream that holds whole SMs when the launch starts (test_first_claim_hold_sm120.cu: CTAs that each
need an empty SM, as a collective's do) keeps as many CTAs of the launch out: the FC1 items of their block indices are not
computed, and every FC2 item of those experts waits, one CTA per item. With a few SMs held the other CTAs finish the rest and
leave, the CTAs kept out run on the SMs they leave, and the launch ends. With so many SMs held that those experts have more FC2
items than CTAs run, every running CTA comes to wait, none leaves, and the launch stands until the hold ends - or until the
watchdog. The cases, on the dumps of a real prefill, in the served switch set with the persistent mode on:

  --case beside   (default) four whole SMs held for 300 ms: the launch does NOT wait for the hold - it ends without an error
                  well before the hold does, its output is b12x's, the record stays empty.
  --case hold     enough SMs held (the test counts how many from the call's routing) for 300 ms, less than the watchdog: the
                  launch waits for the hold and ends without an error, its output is b12x's, the record stays empty, and the
                  call's launches end no sooner than the hold.
  --case trap     the same hold outlasts the watchdog by 600 ms: the wait expires and the launch TRAPS - the expected outcome of
                  this case. The record must then name the wait: set, what = 1 or 2, need = 8 x the FC1 items of an expert of the
                  call, need - seen = 8 arrivals for each FC1 item of that expert the hold keeps out, the watchdog's milliseconds,
                  a block of the grid on an SM the hold does not occupy; and the failure comes no sooner than the watchdog.
                  The kernel stores the record's last three words (SM, block, watchdog ms) behind two special-register reads,
                  and the threads that lose the record's lock trap at once: when those words have not reached the host (all
                  three read 0) the case says so in one WARNING line and passes; --strict-record makes that a failure.
                  THE PROCESS'S CUDA CONTEXT DOES NOT SURVIVE THIS CASE and the driver logs Xid 43; run it alone, last, and open
                  one fresh CUDA context on the card afterwards (the utilisation reading of a trapped kernel is cleared by it).
  --case criteria no GPU: the trap case's criteria on records written by hand - the record of an expired A-producer wait passes;
                  a launch that did not fail, an empty record, the record of another wait, a `need` no expert has, arrivals that
                  are not whole items, another watchdog, a block outside the grid and an SM of the hold are refused, each for its
                  own reason; a record without its last three words - one written by hand, two as launches of the kernel left
                  them - passes with the warning and is refused under --strict-record; and the count of SMs to hold on a list
                  written by hand.

  python3 test_persistent_watchdog_sm120.py --dumps DIR [--case beside|hold|trap] [--watchdog-ms N] [--strict-record]
  python3 test_persistent_watchdog_sm120.py --case criteria
(DIR: weights_layer2.pt, call1_layer2.pt.) Exit 0 only if the case behaves as described. The persistent launch has no activation
counter of its own: its calls are told by the counters of the kernels it replaces standing still.
"""
from __future__ import annotations

import argparse
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
WS_MODE = 30  # the persistent launch with FC2 items of two column tiles and four ring stages
ROWS = 4096
WHOLE_SM = 98560  # bytes of shared memory: a CTA that needs an empty SM
HOLD_THREADS = 544
BESIDE_CTAS = 4  # the case `beside`: as many whole SMs as a collective holds
STALL_MARGIN = 1.5  # the cases `hold` / `trap`: the experts kept waiting have this many times more FC2 items than CTAs run
HOLD_MS = 300  # the cases `beside` and `hold`
TRAP_EXTRA_MS = 600  # the case `trap`: the hold outlasts the watchdog by this much
HOLD_MAX_MS = 3000  # the helper refuses a longer hold
WATCHDOG_BUILT_MS = 2000
FIELDS = "[set, what, item, need, seen, smid, block, watchdog_ms]"


def hold_ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    return load(name="vllm_moet_test_first_claim_hold_sm120", sources=[str(HERE / "test_first_claim_hold_sm120.cu")],
                extra_cuda_cflags=["-O3", "-std=c++20"], extra_cflags=["-O3", "-std=c++20"], verbose=False)


def kept_out(f1_items: list[int], f2_items: list[int], grid: int, n: int) -> tuple[int, list[int]]:
    """The work list starts with the FC1 blocks of the first experts in expert order, and a CTA's first item is its block index:
    with n CTAs of the grid not dispatched, the FC1 items of n block indices are not computed. Returns the smaller of the two counts
    of FC2 items that then wait - whether the grid's last n blocks are kept out or its first n - and the experts of the last n."""
    owners: list[int] = []
    for e, k in enumerate(f1_items):
        owners += [e] * k
        if len(owners) >= grid:
            break
    if len(owners) < grid:
        return 0, []
    last, first = sorted(set(owners[grid - n:grid])), sorted(set(owners[:n]))
    return min(sum(f2_items[e] for e in last), sum(f2_items[e] for e in first)), last


def stall_ctas(f1_items: list[int], f2_items: list[int], grid: int) -> int:
    """The smallest hold (a multiple of 8 CTAs, at most half the grid) with which the experts kept waiting have STALL_MARGIN times
    more FC2 items than CTAs run: every running CTA then comes to wait. 0: no such hold for this call."""
    for n in range(8, grid // 2 + 1, 8):
        if kept_out(f1_items, f2_items, grid, n)[0] >= STALL_MARGIN * (grid - n):
            return n
    return 0


TAIL_MISSING = "the record's last three words - SM, block, watchdog ms - did not reach the host (all three read 0)"


def trap_findings(failed: bool, record: list[int], raw_set: int, needs: set[int], watchdog_ms: int, grid: int,
                  hold_sms: list[int], strict_record: bool = False) -> tuple[list[str], list[str]]:
    """What is wrong with the outcome of a trap case (no finding: it is as described), and its warnings. `record` is the module's
    reading, `raw_set` the first word of the extension's own entry, `needs` the arrival counts an FC2 item of the call can wait for
    (8 per FC1 item of an expert), `grid` the CTAs of the launch, `hold_sms` the SMs the hold runs on (one CTA of the launch is
    kept out per SM). A record that has its first words and lacks its last three is a warning, a finding under `strict_record`."""
    bad = []
    warn = []
    if not failed:
        bad.append("the launch did not fail")
    if record[0] != 1 or raw_set != 1:
        bad.append(f"the record is not set (the module reads {record[0]}, the extension's entry {raw_set})")
    what, need, seen, smid, block, wd = record[1], record[3], record[4], record[5], record[6], record[7]
    if what not in (1, 2):
        bad.append(f"what = {what}: not the wait of an FC2 item for its expert's FC1 items (1 / 2)")
    if need not in needs:
        bad.append(f"need = {need}: no expert of the call has that many FC1 arrivals")
    missing = need - seen
    most = min(need, 8 * len(hold_sms)) // 8
    if missing < 8 or missing > 8 * most or missing % 8 != 0:
        bad.append(f"need - seen = {missing}: not 8 arrivals for each of 1 .. {most} FC1 items")
    if (smid, block, wd) == (0, 0, 0):
        if strict_record:
            bad.append(TAIL_MISSING)
        elif any(record):
            warn.append(TAIL_MISSING)
        return bad, warn
    if wd != watchdog_ms:
        bad.append(f"watchdog_ms = {wd}: the launch ran with {watchdog_ms}")
    if not 0 <= block < grid:
        bad.append(f"block = {block}: not a block of the grid of {grid}")
    if smid in hold_sms:
        bad.append(f"smid = {smid}: an SM the hold occupies")
    return bad, warn


def criteria_case() -> int:
    kw = dict(needs={80, 120}, watchdog_ms=WATCHDOG_BUILT_MS, grid=188, hold_sms=[3, 9, 100, 187])
    good = [1, 1, 3301, 80, 72, 17, 44, WATCHDOG_BUILT_MS]
    # (what the row is, the launch failed, the record, the extension's first word, the one reason it must be refused for; an empty
    # record is refused for several, the first of them is named)
    table = [
        ("an expired wait of A-producer role 0, one FC1 item kept out", True, good, 1, None),
        ("an expired wait of A-producer role 1, four FC1 items kept out", True, [1, 2, 7, 120, 88, 0, 187, WATCHDOG_BUILT_MS], 1, None),
        ("the same record after a launch that did not fail", False, good, 1, "the launch did not fail"),
        ("an empty record", True, [0] * 8, 0, "the record is not set"),
        ("a record the extension's entry does not report as set", True, good, 0, "the record is not set"),
        ("a consumer's wait for a stage inside an item", True, [1, 11, 5, 80, 72, 17, 44, WATCHDOG_BUILT_MS], 1, "what = 11"),
        ("a sum item's wait of the separate FC2 kernel", True, [1, 100, 5, 80, 72, 17, 44, WATCHDOG_BUILT_MS], 1, "what = 100"),
        ("a need no expert has", True, [1, 1, 3301, 96, 88, 17, 44, WATCHDOG_BUILT_MS], 1, "need = 96"),
        ("nothing missing", True, [1, 1, 3301, 80, 80, 17, 44, WATCHDOG_BUILT_MS], 1, "need - seen = 0"),
        ("arrivals that are not whole items", True, [1, 1, 3301, 80, 75, 17, 44, WATCHDOG_BUILT_MS], 1, "need - seen = 5"),
        ("more items missing than the hold keeps out", True, [1, 1, 3301, 80, 40, 17, 44, WATCHDOG_BUILT_MS], 1, "need - seen = 40"),
        ("another watchdog", True, [1, 1, 3301, 80, 72, 17, 44, 200], 1, "watchdog_ms = 200"),
        ("a block outside the grid", True, [1, 1, 3301, 80, 72, 17, 188, WATCHDOG_BUILT_MS], 1, "block = 188"),
        ("an SM of the hold", True, [1, 1, 3301, 80, 72, 100, 44, WATCHDOG_BUILT_MS], 1, "smid = 100"),
    ]
    ok = True
    for label, failed, record, raw_set, reason in table:
        found, warned = trap_findings(failed, record, raw_set, **kw)
        if reason is None:
            good_row = not found and not warned
        else:
            good_row = bool(found) and found[0].startswith(reason) and (len(found) == 1 or not any(record))
        ok &= good_row
        print(f"{'ok  ' if good_row else 'BAD '} {label}: {record} -> {found if found else 'as described'}")
    # records whose last three words did not arrive - one written by hand, and two as launches of the kernel left them under a
    # hold of SMs 0 .. 39 (seven FC1 items of the expert kept out): each passes with the one warning and is refused for that one
    # reason under --strict-record
    wide = dict(kw, hold_sms=list(range(40)))
    tails = [
        ("the first words only, written by hand", [1, 1, 3301, 80, 72, 0, 0, 0], kw),
        ("the first words only, as a launch left them", [1, 1, 1250, 80, 24, 0, 0, 0], wide),
        ("the first words only, as another launch left them", [1, 1, 1240, 80, 24, 0, 0, 0], wide),
    ]
    for label, record, k in tails:
        found, warned = trap_findings(True, record, 1, **k)
        strict, strict_warned = trap_findings(True, record, 1, strict_record=True, **k)
        good_row = not found and warned == [TAIL_MISSING] and strict == [TAIL_MISSING] and not strict_warned
        ok &= good_row
        print(f"{'ok  ' if good_row else 'BAD '} {label}: {record} -> {found if found else 'as described'} with {len(warned)} warning; "
              f"with --strict-record -> {strict if strict else 'as described'}")
    # the count of SMs to hold, on a list written by hand: 30 experts of 10 FC1 and 40 FC2 items, a grid of 188
    n = stall_ctas([10] * 30, [40] * 30, 188)
    waiting, experts = kept_out([10] * 30, [40] * 30, 188, n)
    good_row = n == 56 and waiting == 240 and experts == [13, 14, 15, 16, 17, 18] and stall_ctas([10] * 30, [1] * 30, 188) == 0
    ok &= good_row
    print(f"{'ok  ' if good_row else 'BAD '} the hold that makes every running CTA wait, 30 experts of 10 FC1 and 40 FC2 items, grid 188: "
          f"{n} CTAs, {waiting} FC2 items of experts {experts} against {188 - n} CTAs that run; with 1 FC2 item per expert: none")
    print(f"{'PASS' if ok else 'FAIL'}: persistent watchdog, case criteria ({len(table) + len(tails)} records, the hold's size)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=os.environ.get("VLLM_MOET_DUMPS", "dumps"),
                    help="directory with the served model's MoE dumps (weights_layer2.pt, call1_layer2.pt); env VLLM_MOET_DUMPS")
    ap.add_argument("--case", choices=["beside", "hold", "trap", "criteria"], default="beside")
    ap.add_argument("--watchdog-ms", type=int, default=0, help="_WS_WATCHDOG_MS of this process (0: what the environment says, else as built)")
    ap.add_argument("--strict-record", action="store_true",
                    help="the case trap: a record whose last three words (SM, block, watchdog ms) did not reach the host is a failure, not a warning")
    a = ap.parse_args()
    if a.case == "criteria":
        return criteria_case()
    os.environ.update(SERVED)
    os.environ[f"{P}_WS"] = str(WS_MODE)
    os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
    for k in ("MT_LOOPS", "FIRST_CLAIM", "FC1_FZ", "WS_STATS", "WS_MIN_ROWS", "WS_LEAD"):  # the launch as it is by default
        os.environ.pop(f"{P}_{k}", None)
    if a.watchdog_ms > 0:
        os.environ[f"{P}_WS_WATCHDOG_MS"] = str(a.watchdog_ms)
    try:
        watchdog_ms = max(1, int(os.environ.get(f"{P}_WS_WATCHDOG_MS", WATCHDOG_BUILT_MS)))
    except ValueError:
        print(f"FAIL: {P}_WS_WATCHDOG_MS is not a number")
        return 2
    hold_ms = watchdog_ms + TRAP_EXTRA_MS if a.case == "trap" else HOLD_MS
    if a.case != "trap" and watchdog_ms < 2 * HOLD_MS:
        print(f"FAIL: the case {a.case} needs a watchdog of at least {2 * HOLD_MS} ms (it is {watchdog_ms})")
        return 2
    if a.case == "trap" and hold_ms > HOLD_MAX_MS:
        print(f"FAIL: the case trap needs a watchdog of at most {HOLD_MAX_MS - TRAP_EXTRA_MS} ms (it is {watchdog_ms})")
        return 2

    import torch
    import b12x_moe_prefill_fast_sm120 as m
    from test_b12x_moe_prefill_spec_sm120 import _planes, _served
    import b12x.moe.fused_moe._impl as impl

    ext = m._ext()
    mode = int(ext.persistent_mode()) if hasattr(ext, "persistent_mode") else 0
    if mode != WS_MODE or not hasattr(ext, "moe_prefill_ws_error"):
        print(f"FAIL: this build of the extension does not run the persistent launch with a watchdog record (persistent_mode {mode})")
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
    c = torch.load(d / "call1_layer2.pt", map_location="cpu", weights_only=False)
    x, ids, w = (c[k][:ROWS].to(dev).contiguous() for k in ("hidden_states", "topk_ids", "topk_weights"))
    rows = int(x.shape[0])
    ref = torch.empty(rows, x.shape[1], dtype=torch.bfloat16, device=dev)
    served.run(x, ids, w, ref)
    torch.cuda.synchronize()
    planes = _planes(torch, served.prepared, spied)
    impl._launch_dynamic = orig

    # the helper's first launch and three calls alone: every kernel has been launched once, the outputs are b12x's
    h = hold_ext()
    compute, other, side = (torch.cuda.Stream(device=dev) for _ in range(3))
    warm = torch.zeros(BESIDE_CTAS, 3, dtype=torch.int64, device=dev)
    with torch.cuda.stream(other):
        h.hold(warm, BESIDE_CTAS, HOLD_THREADS, WHOLE_SM, 50_000)
    torch.cuda.synchronize()
    m.activation_counters(clear=True)
    warm_bad = 0
    res = None
    for _ in range(3):
        with torch.cuda.stream(compute):
            res = m.moe_prefill_fast(x, ids, w, planes, 10.0)
        torch.cuda.synchronize()
        warm_bad += int(not torch.equal(res[0].view(torch.int16), ref.view(torch.int16)))
    cnt = m.activation_counters()
    persistent_calls = (cnt.get("fc1_v8_v9") == (3, 3 * rows) and cnt.get("fc2_topk_sum_kernel") == (3, 3 * rows)
                        and cnt.get("fc1_v10") == (0, 0) and cnt.get("fc2_sum_items") == (0, 0))
    # the call's work list from its routing and the front-end's own counts: per expert its FC1 and FC2 items; what an FC2 item
    # can wait for: 8 arrivals (the consumer warps) per FC1 item of its expert
    valid = ids[ids >= 0].to(torch.int64).flatten()
    per_expert = torch.bincount(valid)
    chunks = ((per_expert + 15) // 16 + 3) // 4
    n_chunks = max(1, int(chunks.sum().item()))
    n_f1, n_items = int(res[7][1].item()), int(res[7][6].item())
    tiles1, tiles2 = n_f1 // n_chunks, (n_items - n_f1) // n_chunks
    f1_items = [int(v) for v in (tiles1 * chunks).tolist()]
    f2_items = [int(v) for v in (tiles2 * chunks).tolist()]
    needs = {8 * n for n in f1_items if n > 0}
    grid = int(torch.cuda.get_device_properties(dev).multi_processor_count)  # one CTA per SM
    hold_ctas = BESIDE_CTAS if a.case == "beside" else stall_ctas(f1_items, f2_items, grid)
    if hold_ctas == 0:
        print(f"FAIL: no hold of at most {grid // 2} SMs makes every running CTA of this call wait ({n_f1} FC1 items, {n_items - n_f1} FC2 items, grid {grid})")
        return 2
    waiting, experts = kept_out(f1_items, f2_items, grid, hold_ctas)
    # for the reader, not a criterion: IF the hold keeps out the grid's last blocks, the first FC2 items to wait are those of the
    # first expert among them
    guess = None
    if experts:
        owners: list[int] = []
        for e, n in enumerate(f1_items):
            owners += [e] * n
        guess = (experts[0], 8 * f1_items[experts[0]], 8 * owners[grid - hold_ctas:grid].count(experts[0]))
    first_record = m.watchdog_record(clear=True)
    print(f"case {a.case}: call1[0:{rows}], persistent_mode {mode}, watchdog {watchdog_ms} ms, a hold of {hold_ctas} whole-SM CTAs for "
          f"{hold_ms} ms issued before the call - by the list order {waiting} FC2 items of {len(experts)} experts wait for the FC1 items "
          f"kept out, {grid - hold_ctas} CTAs run; alone: 3 calls, {warm_bad} differ from b12x, counters {cnt}, record {first_record}; "
          f"{len(needs)} different needs among {sum(1 for n in f1_items if n > 0)} experts, {tiles1} FC1 and {tiles2} FC2 items per chunk, "
          f"grid {grid}", flush=True)

    rec = torch.zeros(hold_ctas, 3, dtype=torch.int64, device=dev)
    torch.cuda.synchronize()
    with torch.cuda.stream(other):
        h.hold(rec, hold_ctas, HOLD_THREADS, WHOLE_SM, hold_ms * 1_000_000)
    # every CTA of the hold writes its SM and its start time first: wait until all of them run
    deadline = time.perf_counter() + 1.0
    started = False
    r = None
    while not started and time.perf_counter() < deadline:
        with torch.cuda.stream(side):
            r = rec.cpu()
        started = bool((r[:, 1] != 0).all())
    hold_sms = sorted(int(v) for v in r[:, 0].tolist()) if started else []
    sms_text = f"{len(hold_sms)} SMs {hold_sms[0]} .. {hold_sms[-1]}" if hold_sms else "no SM"
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    error = None
    out = None
    t0 = time.perf_counter()
    try:
        with torch.cuda.stream(compute):
            e0.record(compute)
            out = m.moe_prefill_fast(x, ids, w, planes, 10.0)[0]
            e1.record(compute)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - the expected outcome of the trap case
        error = exc
    wall_ms = (time.perf_counter() - t0) * 1e3
    record = m.watchdog_record()  # a host read: works after a trapped launch
    raw = [int(v) for v in ext.moe_prefill_ws_error(False)]  # the extension's own entry
    pre = []
    if warm_bad:
        pre.append(f"{warm_bad} of the calls alone differ from b12x")
    if not persistent_calls:
        pre.append("the calls alone were not persistent launches by the counters")
    if any(first_record):
        pre.append("the record was not empty before the hold")
    if not started:
        pre.append("the hold did not start within 1 s")
    elif len(set(hold_sms)) != hold_ctas:
        pre.append(f"the hold's {hold_ctas} CTAs run on {len(set(hold_sms))} SMs")

    if a.case == "trap":
        found, warned = trap_findings(error is not None, record, raw[0], needs, watchdog_ms, grid, hold_sms, a.strict_record)
        bad = pre + found
        if error is not None and wall_ms < watchdog_ms - 5:
            bad.append(f"the launch failed after {wall_ms:.0f} ms, sooner than the watchdog")
        text = "NO error" if error is None else f"{type(error).__name__}: {str(error).splitlines()[0][:100]}"
        print(f"trap: the launch ended after {wall_ms:.0f} ms of host time with {text}; watchdog record {FIELDS} = {record}; the "
              f"extension's entry reports set = {raw[0]}; the hold ran on {sms_text}")
        if guess is not None:
            agrees = record[3] == guess[1] and record[3] - record[4] == guess[2]
            print(f"trap: if the hold keeps out the grid's last {hold_ctas} blocks, the first FC2 items to wait are expert {guess[0]}'s - need "
                  f"{guess[1]}, need - seen {guess[2]}: the record {'agrees' if agrees else 'does NOT agree'} (not a criterion)")
        for b in bad:
            print(f"  not as described: {b}")
        for v in warned:
            print(f"  WARNING: {v} - a failure under --strict-record")
        print(f"{'PASS' if not bad else 'FAIL'}: persistent watchdog, case trap (the wait of an FC2 item for its expert's FC1 items "
              "expired: the launch failed and the record names the wait)", flush=True)
        sys.stdout.flush()
        os._exit(0 if not bad else 1)  # the context is gone: no teardown

    same = out is not None and error is None and bool(torch.equal(out.view(torch.int16), ref.view(torch.int16)))
    ms = e0.elapsed_time(e1) if error is None else float("nan")
    held = ""
    if error is None:
        with torch.cuda.stream(side):
            r = rec.cpu()
        spans = [(int(t1) - int(t0n)) / 1e6 for _, t0n, t1 in r.tolist()]
        held = f"; the hold's CTAs ran {min(spans):.0f} .. {max(spans):.0f} ms"
    bad = list(pre)
    if error is not None:
        bad.append(f"the launch failed: {type(error).__name__}")
    if not same:
        bad.append("the output is not b12x's")
    if any(record) or any(raw):
        bad.append(f"the record is not empty: {record}")
    if error is None and a.case == "hold" and ms < 0.8 * hold_ms:
        bad.append(f"the call's launches ended after {ms:.1f} ms: the launch did not wait for the hold")
    if error is None and a.case == "beside" and ms > 0.5 * hold_ms:
        bad.append(f"the call's launches ended after {ms:.1f} ms: the launch waited for the hold")
    print(f"{a.case}: error {error!r}; the call's launches ended after {ms:.1f} ms (the hold: {hold_ms} ms on {sms_text}{held}); output "
          f"identical to b12x: {same}; watchdog record {record}")
    for b in bad:
        print(f"  not as described: {b}")
    what = ("the launch beside four held SMs does not wait for them: no trap, the output is b12x's, the record stays empty" if a.case == "beside"
            else f"a wait that lasts {hold_ms} ms under a watchdog of {watchdog_ms} ms: no trap, the output is b12x's, the record stays empty")
    print(f"{'PASS' if not bad else 'FAIL'}: persistent watchdog, case {a.case} ({what})", flush=True)
    if error is not None:
        os._exit(1)  # a failed launch: the context is gone, no teardown
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
