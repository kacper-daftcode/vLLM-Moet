#!/usr/bin/env python3
"""Unit test of the pipeline's static SM split (VLLM_MOET_PREFILL_PIPE_GREEN, moet_prefill_pipe.green_sms) without the model
(inside the image: the module imports vllm's forward context and the mHC warm-up pin).

CPU part (no GPU needed; fake streams):
  (1) switch off: no split, no os.stat, no key, no stats entry, the capture scope changes nothing;
  (2) the control-file reader (VLLM_MOET_RUNTIME_AB=1, key PREFILL_PIPE_GREEN): every row of the contract - no file, the key at 0
      / at the start-up value / absent, the file removed after a state was applied, malformed, not an object, out of range, not an
      integer, an unknown key of the namespace, other namespaces, the 100-ms gate, an unchanged (mtime, size), the change lines and
      the on-demand line;
  (3) the fast prefill MoE's grid caps: the CTAs-per-SM rule per mode, the wrapper on a green stream / on any other stream, a
      smaller cap of the caller or the environment, grids given by position, one wrapper however often it is installed;
  (4) the configurations the split refuses.
GPU part (one GPU; skipped with --cpu):
  (5) the split is created once, its streams belong to their green contexts, the SM counts add up;
  (6) the handoff protocol of test_prefill_pipe_unit.py on the split: inside a phase the current stream is the comm partition's,
      outside it the compute partition's, per thread, for vllm's and torch's bookkeeping alike; the attention's aux-stream list
      holds the compute partition's streams during the step and its own afterwards; a tensor the caller's stream produced right
      before the step is read inside it, the step's outputs are read on the caller's stream right after it; the results are
      bit-identical with the unsplit step and the sequential reference;
  (7) the runtime key flips between forwards: state 0 = the caller's stream + the pipeline's comm stream, state N = the split,
      identical outputs, the counters and the change lines;
  (8) graphs captured inside shared_capture_scope - vllm's breakable capture (two segments around an eager break, as the replay
      layers') and a plain torch.cuda.CUDAGraph - are captured on the compute partition's stream and replay with the eager result
      from that stream and from the caller's, launched as a replay tail launches them (stream order + an event join, no host
      synchronisation of the device);
  (12) the replay section of a split step (replay_begin ... replay_end): on the caller's stream with GREEN_TAIL=caller, on the compute
      partition's with GREEN_TAIL=compute, identical outputs, also when only the later group has one;
  with --smid-lib LIB also: (9) where the kernels run - a probe kernel that records %smid, launched in a phase, in a compute
      segment, in the captured graph: SM sets of the two partitions disjoint, their sizes within the partitions' SM counts, the
      captured kernel on the compute partition whichever stream launches the graph. LIB = a shared library with
      `int gc_smid(void* stream, int n_ctas, int threads, int smem_bytes, void* rec, uint64_t spin_ns)` (one 24-byte record per
      CTA: u64 t0, u64 t1, u32 smid, u32 pad) and `int gc_rec_size(void)`.
TP part (torchrun --nproc_per_node=4 ... --tp): (10) the same step with NCCL's reduce-scatter and all-gather of the TP group in
  its comm phases (vllm's sp_reduce_scatter / tensor_model_parallel_all_gather on the comm partition's stream): every rank's
  outputs bit-identical with the unsplit step; (11) a TP all-reduce captured into a CUDA graph on the compute partition's stream
  (what the replay tails' graphs hold) replays with the eager result from that stream and from the caller's.

usage: python3 test_prefill_pipe_green_unit.py [--src DIR] [--cpu] [--comm-sms 32] [--smid-lib LIB]
       torchrun --nproc_per_node=4 test_prefill_pipe_green_unit.py --tp [--src DIR] [--comm-sms 32]
(DIR = where moet_prefill_pipe.py lives, default: next to this file; the option names avoid torchrun's own --module / --n...)
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
import threading
import time
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, default=Path(__file__).resolve().parent, help="where moet_prefill_pipe.py lives")
ap.add_argument("--cpu", action="store_true", help="the CPU part alone")
ap.add_argument("--tp", action="store_true", help="under torchrun: the 4-rank part (NCCL collectives in the comm phases)")
ap.add_argument("--comm-sms", dest="n", type=int, default=32, help="SMs asked for the comm partition")
ap.add_argument("--smid-lib", default="", help="a probe library with gc_smid (see the module docstring): where the kernels run")
args = ap.parse_args()

P = "VLLM_MOET_PREFILL_PIPE"
MOE = "VLLM_MOET_B12X_MOE_PREFILL_FAST"
for k in list(os.environ):  # the test sets every switch it means
    if k.startswith(P) or k.startswith("VLLM_MOET_RUNTIME_"):
        del os.environ[k]

import torch  # noqa: E402

sys.path.insert(0, str(args.src))
import moet_prefill_pipe as pipe  # noqa: E402


RANK = int(os.environ.get("RANK", "0")) if args.tp else 0
failures: list[str] = []


def check(cond, msg):
    if RANK == 0 or not cond:
        print(("PASS " if cond else "FAIL ") + (f"[r{RANK}] " if args.tp else "") + msg, flush=True)
    if not cond:
        failures.append(msg)


def reset_green():
    pipe._GREEN.update(split=None, failed=None, on=True, forwards=0, rows=0, forwards_off=0, rows_off=0, moe_capped=0,
                       graph_scopes=0, warmup=None, ab=None, next_poll=0.0, sig=None, warned=set(), stats_mtime=None)


@contextlib.contextmanager
def stderr_lines():
    buf = io.StringIO()
    old = sys.stderr
    sys.stderr = buf
    try:
        yield buf
    finally:
        sys.stderr = old


class FakeStream:
    def __init__(self, handle: int) -> None:
        self.cuda_stream = handle


class FakeSplit:
    na, nb, device_sms = 32, 156, 188

    def __init__(self) -> None:
        self.comm, self.compute = FakeStream(0xA0), FakeStream(0xB0)
        self.aux = [FakeStream(0xB1), FakeStream(0xB2), FakeStream(0xB3)]
        self.sms = {0xA0: 32, 0xB0: 156, 0xB1: 156, 0xB2: 156, 0xB3: 156}


# ============================================================================================== CPU part
def cpu_part() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="green_unit_"))
    # ---------------------------------------------------------------------------------- (1) switch off
    reset_green()
    real_stat = os.stat
    calls = {"stat": 0}

    def counting_stat(*a, **k):
        calls["stat"] += 1
        return real_stat(*a, **k)

    os.stat = counting_stat
    try:
        os.environ[P] = "2"
        os.environ["VLLM_MOET_RUNTIME_AB"] = "1"
        os.environ["VLLM_MOET_RUNTIME_AB_FILE"] = str(tmp / "ab.json")
        (tmp / "ab.json").write_text('{"PREFILL_PIPE_GREEN": 32}')
        off = (pipe.green_sms() == 0 and pipe.green_active() is None and pipe.green_split() is None
               and pipe._green_capture_enter() is None and pipe._GREEN["split"] is None and pipe._GREEN["failed"] is None)
        scope = pipe.shared_capture_scope(1, 128)
        with scope:
            off = off and scope.green_prev is None
        check(off and calls["stat"] == 0 and "green" not in pipe.stats(),
              f"(1) switch off: no split, no key although the control file names it, {calls['stat']} os.stat calls, no stats entry, "
              "the capture scope changes nothing")
        for bad in ("x", "-8", ""):
            os.environ[P + "_GREEN"] = bad
            off = off and pipe.green_sms() == 0
        check(off, "(1) a value that is not a positive integer = off")
    finally:
        os.stat = real_stat

    # ---------------------------------------------------------------------------------- (2) the control-file reader
    reset_green()
    os.environ[P + "_GREEN"] = "32"
    os.environ["VLLM_MOET_RUNTIME_STATS_FILE"] = str(tmp / "stats")
    ab = tmp / "ab.json"
    ab.unlink()
    fake = FakeSplit()
    pipe._GREEN["split"] = fake
    seq = {"n": 0}

    def write(obj) -> None:  # atomically, as the contract asks; a new size or mtime every time
        seq["n"] += 1
        text = obj if isinstance(obj, str) else json.dumps({**obj, "_seq": seq["n"]})
        t = tmp / "ab.json.tmp"
        t.write_text(text)
        os.utime(t, ns=(time.time_ns(), time.time_ns() + seq["n"] * 1000))
        os.replace(t, ab)

    def poll(expect_on: bool, what: str, expect_line: str | None = None, expect_warning: bool | None = None) -> None:
        pipe._GREEN["next_poll"] = 0.0
        with stderr_lines() as buf:
            got = pipe.green_active()
        text = buf.getvalue()
        ok = (got is fake) == expect_on
        if expect_line is not None:
            ok = ok and expect_line in text
        elif re.search(r"runtime A/B: PREFILL_PIPE_GREEN \d+ -> \d+", text):
            ok = False  # a change line although nothing changed
        if expect_warning is not None:
            ok = ok and (("WARNING" in text) == expect_warning)
        check(ok, f"(2) {what}: {'split' if expect_on else 'unsplit'}" + (f", line `{expect_line}`" if expect_line else "")
              + ("" if ok else f" - got {'split' if got is fake else got}, stderr {text!r}"))

    poll(True, "no control file since start-up = the start-up state, one WARNING", expect_warning=True)
    poll(True, "no control file, second poll: no second WARNING", expect_warning=False)
    write({"PREFILL_PIPE_GREEN": 0})
    poll(False, "key at 0", "[vllm-moet prefill pipeline, device ?] runtime A/B: PREFILL_PIPE_GREEN 32 -> 0"
         if not torch.cuda.is_available() else "runtime A/B: PREFILL_PIPE_GREEN 32 -> 0")
    poll(False, "unchanged (mtime, size): state kept, no line", expect_warning=False)
    write({"PREFILL_PIPE_GREEN": 32})
    pipe._GREEN["next_poll"] = time.monotonic() + 3600
    with stderr_lines() as buf:
        got = pipe.green_active()
    check(got is None and buf.getvalue() == "", "(2) inside the 100-ms gate the file is not looked at (the written flip waits)")
    poll(True, "key at the start-up value", "runtime A/B: PREFILL_PIPE_GREEN 0 -> 32")
    write({"PREFILL_PIPE_GREEN": 0})
    poll(False, "key at 0 again", "runtime A/B: PREFILL_PIPE_GREEN 32 -> 0")
    write({})
    poll(True, "file without the key ({}) = back to the start-up value", "runtime A/B: PREFILL_PIPE_GREEN 0 -> 32")
    write({"PREFILL_PIPE_GREEN": 0})
    poll(False, "key at 0 (before the removal)", "runtime A/B: PREFILL_PIPE_GREEN 32 -> 0")
    ab.unlink()
    poll(False, "file removed after a state was applied: the applied state stays, one WARNING", expect_warning=True)
    write("{")
    poll(False, "malformed JSON: state kept, one WARNING", expect_warning=True)
    write("[1, 2]")
    poll(False, "not an object: state kept (no second WARNING for the same cause)", expect_warning=False)
    write({"PREFILL_PIPE_GREEN": 7})
    poll(False, "value out of range (7): the key keeps its value, one WARNING", expect_warning=True)
    write({"PREFILL_PIPE_GREEN": "32"})
    poll(False, "value not an integer ('32'): kept (the WARNING was given once)", expect_warning=False)
    write({"PREFILL_PIPE_GREEN": True})
    poll(False, "value true: kept", expect_warning=False)
    write({"PREFILL_PIPE_GREEN": 32, "PREFILL_PIPE_NOSUCH": 1})
    poll(True, "an unknown key of the namespace is ignored with one WARNING, the rest applies",
         "runtime A/B: PREFILL_PIPE_GREEN 0 -> 32", expect_warning=True)
    write({"PREFILL_PIPE_GREEN": 0, "B12X_MOE_PREFILL_FAST_WS1": 0, "SPARSE_MLA_PREFILL_SASS_CUBIN_SEL": 1, "WOA_DEQUANT_FUSED": 1})
    poll(False, "other modules' keys next to ours: ours applies, theirs are ignored silently",
         "runtime A/B: PREFILL_PIPE_GREEN 32 -> 0", expect_warning=False)
    write({"B12X_MOE_PREFILL_FAST_WS1": 0})
    poll(True, "only other modules' keys = our start-up value", "runtime A/B: PREFILL_PIPE_GREEN 0 -> 32", expect_warning=False)
    (tmp / "stats").write_text("")
    with stderr_lines() as buf:
        pipe.green_active()
        first = buf.getvalue()
        pipe.green_active()
        second = buf.getvalue()[len(first):]
    check("activation counters (on demand" in first and "switches now PREFILL_PIPE_GREEN=32" in first and second == "",
          "(2) one touch of the stats file = one on-demand line with `switches now PREFILL_PIPE_GREEN=32`")
    # the master switch off: the file is never looked at
    reset_green()
    pipe._GREEN["split"] = fake
    del os.environ["VLLM_MOET_RUNTIME_AB"]
    write({"PREFILL_PIPE_GREEN": 0})
    with stderr_lines() as buf:
        got = pipe.green_active()
    check(got is fake and "runtime A/B" not in buf.getvalue(), "(2) VLLM_MOET_RUNTIME_AB unset: the control file is not read, the split stays on")

    # ---------------------------------------------------------------------------------- (3) the MoE grid caps
    def ctas(rows, **env):
        for k in ("WS1", "WS2", "WS", "WS1_MIN_ROWS", "WS_MIN_ROWS"):
            os.environ.pop(f"{MOE}_{k}", None)
        for k, v in env.items():
            os.environ[f"{MOE}_{k}"] = str(v)
        return pipe.green_moe_ctas(rows)

    os.environ.pop(P + "_GREEN_MOE_CTAS", None)
    table = [
        ((4096, dict(WS1=9, WS2=3)), (1, 2)), ((128, dict(WS1=9, WS2=3)), (2, 2)), ((255, dict(WS1=9, WS2=3)), (2, 2)),
        ((256, dict(WS1=9, WS2=3)), (1, 2)), ((4096, dict(WS1=0, WS2=3)), (2, 2)), ((4096, dict()), (2, 2)),
        ((4096, dict(WS1=8, WS2=3)), (2, 2)), ((4096, dict(WS1=10)), (1, 2)), ((4096, dict(WS1=11)), (1, 2)),
        ((4096, dict(WS1=2)), (1, 2)), ((4096, dict(WS1=1)), (2, 2)), ((4096, dict(WS1=9, WS2=2)), (1, 1)),
        ((4096, dict(WS1=9, WS2=1)), (1, 1)), ((4096, dict(WS1=9, WS2=4)), (1, 1)), ((4096, dict(WS=20, WS2=3)), (1, 2)),
        ((64, dict(WS=20, WS_MIN_ROWS=128)), (2, 2)), ((512, dict(WS1=9, WS1_MIN_ROWS=1024)), (2, 2)),
    ]
    bad = [(a, want, ctas(a[0], **a[1])) for a, want in table if ctas(a[0], **a[1]) != want]
    check(not bad, f"(3) CTAs per SM of the caps (FC1, FC2) for {len(table)} (rows, mode) cases" + (f" - wrong: {bad}" if bad else ""))
    ctas(4096, WS1=9, WS2=3)
    os.environ[P + "_GREEN_MOE_CTAS"] = "2,1"
    check(pipe.green_moe_ctas(4096) == (2, 1), "(3) VLLM_MOET_PREFILL_PIPE_GREEN_MOE_CTAS=2,1 overrides the rule")
    os.environ[P + "_GREEN_MOE_CTAS"] = "two"
    with stderr_lines() as buf:
        got = pipe.green_moe_ctas(4096)
    check(got == (1, 2) and "WARNING" in buf.getvalue(), "(3) a malformed override = the rule + one WARNING")
    os.environ.pop(P + "_GREEN_MOE_CTAS")

    calls: list[tuple] = []
    rt = {"WS1": 9}
    fake_moe = types.ModuleType(pipe._GREEN_MOE_MODULE)

    def moe_prefill_fast(hidden, ids, weights, planes, limit=10.0, n=576, route_out=None, out=None, topk_sum=True, cfg1=None,
                         cfg2=None, dynamic=None, grid1=None, grid2=None, a_values=None, a_scales=None):
        calls.append((grid1, grid2))
        return "out"

    fake_moe.moe_prefill_fast = moe_prefill_fast
    fake_moe.runtime_switches = lambda: dict(rt)
    sys.modules[pipe._GREEN_MOE_MODULE] = fake_moe
    ok = pipe._green_install_moe_caps() and pipe._green_install_moe_caps()
    wrapped = fake_moe.moe_prefill_fast
    check(ok and wrapped is not moe_prefill_fast and wrapped.__wrapped__ is moe_prefill_fast,
          "(3) the wrapper is installed on the module's entry point, once however often it is asked for")
    ids = torch.zeros(4096, 6, dtype=torch.int32)
    ids128 = torch.zeros(128, 6, dtype=torch.int32)

    def call(stream, *a, **k):
        real = torch.cuda.current_stream
        torch.cuda.current_stream = lambda device=None: FakeStream(0) if stream is None else stream  # what torch says is current
        calls.clear()
        try:
            r = fake_moe.moe_prefill_fast(*a, **k)
        finally:
            torch.cuda.current_stream = real
        return calls[0] if r == "out" and len(calls) == 1 else None

    base = pipe._GREEN["moe_capped"]
    got = {
        "compute": call(fake.compute, None, ids, None, {}, 10.0, 576, route_out=None, out=None, topk_sum=True),
        "aux": call(fake.aux[1], None, ids, None, {}),
        "comm": call(fake.comm, None, ids, None, {}),
        "compute 128 rows": call(fake.compute, None, ids128, None, {}),
        "primary": call(FakeStream(0x77), None, ids, None, {}),
        "default stream": call(None, None, ids, None, {}),
        "caller's smaller cap": call(fake.compute, None, ids, None, {}, grid1=100, grid2=1000),
        "by position": call(fake.compute, None, ids, None, {}, 10.0, 576, None, None, True, 0, 0, 1, 1000, 100),
        "by keyword ids": call(fake.compute, hidden=None, ids=ids, weights=None, planes={}),
    }
    rt["WS1"] = 0
    got["WS1 flipped to 0 by its runtime key"] = call(fake.compute, None, ids, None, {})
    rt["WS1"] = 9
    os.environ[f"{MOE}_GRID2"] = "200"
    got["the environment's GRID2=200"] = call(fake.compute, None, ids, None, {})
    os.environ.pop(f"{MOE}_GRID2")
    want = {
        "compute": (156, 312), "aux": (156, 312), "comm": (32, 64), "compute 128 rows": (312, 312), "primary": (None, None),
        "default stream": (None, None), "caller's smaller cap": (100, 312), "by position": (156, 100), "by keyword ids": (156, 312),
        "WS1 flipped to 0 by its runtime key": (312, 312), "the environment's GRID2=200": (156, 200),
    }
    check(got == want and pipe._GREEN["moe_capped"] - base == 9,
          "(3) grid caps (grid1, grid2) the wrapper passes: " + ", ".join(f"{k} {v}" for k, v in got.items())
          + ("" if got == want else f" - WANT {want}") + f"; capped calls counted {pipe._GREEN['moe_capped'] - base}")
    nogrid = types.ModuleType(pipe._GREEN_MOE_MODULE)
    nogrid.moe_prefill_fast = lambda hidden, ids, weights, planes: "out"
    sys.modules[pipe._GREEN_MOE_MODULE] = nogrid
    check(pipe._green_install_moe_caps() is False and not hasattr(nogrid.moe_prefill_fast, "_moet_green_caps"),
          "(3) an entry point without grid1 / grid2 is not wrapped (the split then refuses to exist)")
    del sys.modules[pipe._GREEN_MOE_MODULE]

    # ---------------------------------------------------------------------------------- (4) the refused configurations
    def unsupported(**env):
        saved = {k: os.environ.get(k) for k in (P, P + "_PHASE_GRAPHS", P + "_DEVICE_COMM", P + "_REPLAY", MOE)}
        os.environ.update({P: "2", MOE: "1"})
        for k in (P + "_PHASE_GRAPHS", P + "_DEVICE_COMM", P + "_REPLAY"):
            os.environ.pop(k, None)
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        try:
            return pipe._green_unsupported()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    reasons = [unsupported(**{P: "0"}), unsupported(**{P + "_PHASE_GRAPHS": "1"}), unsupported(**{P + "_DEVICE_COMM": "verify"}),
               unsupported(**{P + "_REPLAY": "stream"}), unsupported(**{MOE: None})]
    check(all(reasons) and unsupported() is None, "(4) refused with a reason: PIPE != 2, PHASE_GRAPHS, DEVICE_COMM, REPLAY=stream, the "
          "fast prefill MoE off; accepted: the served set")
    reset_green()
    os.environ[P + "_PHASE_GRAPHS"] = "1"
    os.environ[MOE] = "1"
    real_api = pipe._green_api
    touched = {"n": 0}

    def no_api():
        touched["n"] += 1
        raise AssertionError("the driver must not be touched")

    pipe._green_api = no_api
    try:
        got = (pipe.green_split(), pipe.green_split(), pipe.green_active())
    finally:
        pipe._green_api = real_api
        os.environ.pop(P + "_PHASE_GRAPHS")
    check(got == (None, None, None) and touched["n"] == 0 and "PHASE_GRAPHS" in str(pipe._GREEN["failed"])
          and pipe.stats()["green"]["on"] is False,
          "(4) a refused configuration: no split, the driver untouched, the reason kept, stats()['green']['on'] False")
    reset_green()


# ============================================================================================== GPU part
def gpu_part() -> None:
    from vllm import forward_context as fc
    import vllm.model_executor.kernels.mhc.warmup as mhc_warmup
    from vllm.utils.torch_utils import current_stream

    dev = torch.device("cuda", torch.cuda.current_device())
    tmp = Path(tempfile.mkdtemp(prefix="green_unit_gpu_"))
    os.environ.update({P: "2", P + "_GREEN": str(args.n), MOE: "1", "VLLM_MOET_RUNTIME_AB": "1",
                       "VLLM_MOET_RUNTIME_AB_FILE": str(tmp / "ab.json"), "VLLM_MOET_RUNTIME_STATS_FILE": str(tmp / "stats")})
    reset_green()
    if pipe._GREEN_MOE_MODULE not in sys.modules and not os.path.isdir(os.environ.get(f"{MOE}_DIR", "/opt/vllm-moet/dsv41_sm120/b12x_moe_prefill_fast")):
        stub = types.ModuleType(pipe._GREEN_MOE_MODULE)  # outside the image: the wrapper needs an entry point to wrap
        stub.moe_prefill_fast = lambda hidden, ids, *a, **k: None
        sys.modules[pipe._GREEN_MOE_MODULE] = stub
    torch.manual_seed(1234 + RANK)

    smid = None
    if args.smid_lib:
        import ctypes as C

        import numpy as np

        lib = C.CDLL(args.smid_lib)
        lib.gc_smid.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_uint64]
        assert lib.gc_rec_size() == 24
        rec_dt = np.dtype({"names": ["t0", "t1", "smid", "pad"], "formats": ["<u8", "<u8", "<u4", "<u4"], "offsets": [0, 8, 16, 20],
                           "itemsize": 24})

        class Smid:
            """A probe launch on the calling thread's current stream: 752 CTAs x 256 threads at one CTA per SM, each spinning 300 us."""

            def launch(self, n_ctas: int = 752, rec=None):
                rec = torch.zeros(n_ctas * 24, dtype=torch.uint8, device=dev) if rec is None else rec
                rc = lib.gc_smid(C.c_void_p(torch.cuda.current_stream().cuda_stream), n_ctas, 256, 60 * 1024,
                                 C.c_void_p(rec.data_ptr()), 300_000)
                assert rc == 0
                return rec

            @staticmethod
            def sms(rec) -> set[int]:
                return set(int(x) for x in rec.cpu().numpy().view(rec_dt)["smid"])

        smid = Smid()

    # ---------------------------------------------------------------------------------- (5) the split
    sp = pipe.green_split(dev)
    check(sp is not None and pipe.green_split() is sp and pipe._GREEN["failed"] is None,
          f"(5) the split is created at the first call and returned by every later one (failed: {pipe._GREEN['failed']})")
    if sp is None:
        return
    handles = {int(s.cuda_stream) for s in [sp.comm, sp.compute, *sp.aux]}
    check(sp.na >= args.n and sp.na % 8 == 0 and sp.na + sp.nb == sp.device_sms and len(handles) == 2 + pipe.GreenSplit.AUX
          and sp.sms[int(sp.comm.cuda_stream)] == sp.na and all(sp.sms[int(s.cuda_stream)] == sp.nb for s in [sp.compute, *sp.aux]),
          f"(5) comm partition {sp.na} SMs (asked {args.n}), compute partition {sp.nb}, device {sp.device_sms}; "
          f"{len(handles)} distinct streams, each a stream of its green context (checked at creation)")
    wrapped = getattr(sys.modules[pipe._GREEN_MOE_MODULE].moe_prefill_fast, "_moet_green_caps", False)
    check(wrapped, "(5) the fast prefill MoE's entry point carries the grid-cap wrapper")

    # ---------------------------------------------------------------------------------- (6) the protocol on the split
    class FakeRunner:
        def __init__(self):
            self.comm_stream = torch.cuda.Stream(device=dev)
            self.replay_stream = torch.cuda.Stream(device=dev)
            self.device = dev
            self.row_buf = torch.zeros(8, 4, dtype=torch.int32, device=dev)
            self.row_switch = pipe.RowBufferSwitch([self.row_buf], 2)
            self.aux_orig = [torch.cuda.Stream(device=dev) for _ in range(3)]
            self.aux_list = list(self.aux_orig)
            self.aux_stream_lists = [self.aux_list]

    runner = FakeRunner()
    pipe._RUNNER = runner
    main_stream = current_stream()
    H, LAYERS, ROWS = 4096, 6, 512  # ~0.4 ms per GEMM: several ms of kernels are still queued when a thread hands over
    W = [torch.randn(H, H, device=dev) * 0.03 for _ in range(LAYERS)]
    base_inputs = [torch.randn(ROWS, H, device=dev) for _ in range(2)]
    tp_ops = None
    if args.tp:
        from vllm.distributed import tensor_model_parallel_all_gather
        from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter

        tp_ops = (sp_reduce_scatter, lambda t: tensor_model_parallel_all_gather(t, 0))

    def make_groups():
        return [pipe.PipeGroup(idx=i, token_offset=i * ROWS, num_tokens=ROWS, forward_context=f"ctx{i}", slot=i + 1) for i in range(2)]

    def h(stream) -> int:
        return int(stream.cuda_stream)

    seen: dict = {}

    def fresh_seen():
        seen.clear()
        seen.update(phase=set(), compute=set(), aux=set(), ctx_ok=True, pin_ok=True, rows_ok=True, order=[], phase_sms=set(),
                    compute_sms=set(), recs=[], tail=set(), after_tail=set())

    lock = threading.Lock()

    def fn_for(inputs, tail=()):
        def fn(g: pipe.PipeGroup):
            x = inputs[g.idx]
            mhc_warmup._MOET_SP_PIN_TOKENS = 1000 + g.idx
            runner.row_buf[: 4].fill_(g.idx + 1)
            for layer in range(LAYERS):
                pipe.phase_begin()
                seen["phase"].add((h(current_stream()), h(torch.cuda.current_stream())))
                if tp_ops is not None:  # the boundary chain's collectives: RS of the full rows, glue on the shard, AG back
                    shard = tp_ops[0](x.to(torch.bfloat16))
                    y = tp_ops[1](torch.tanh(shard.float() @ W[layer]).to(torch.bfloat16)).float()
                else:
                    y = torch.tanh(x @ W[layer])
                if smid is not None and layer == 1:
                    seen["recs"].append(("phase", smid.launch()))
                pipe.record_for_comm(x)
                pipe.phase_end()
                seen["compute"].add((h(current_stream()), h(torch.cuda.current_stream())))
                seen["aux"].add(tuple(h(s) for s in runner.aux_list))
                if fc._forward_context != g.forward_context:
                    seen["ctx_ok"] = False
                if mhc_warmup._MOET_SP_PIN_TOKENS != 1000 + g.idx:
                    seen["pin_ok"] = False
                if int(runner.row_buf[0, 0].item()) != g.idx + 1:
                    seen["rows_ok"] = False
                with lock:
                    seen["order"].append((g.idx, layer))
                if smid is not None and layer == 1:
                    seen["recs"].append(("compute", smid.launch()))
                # an aux-stream branch as the attention's maybe_execute_in_parallel: fork, a GEMM on aux[0], join
                e0, e1 = torch.cuda.Event(), torch.cuda.Event()
                e0.record()
                x = torch.tanh(y @ W[layer].t())
                with torch.cuda.stream(runner.aux_list[0]):
                    e0.wait()
                    side = torch.tanh(y @ W[(layer + 1) % LAYERS])
                    e1.record()
                e1.wait()
                x = x + 0.125 * side
            if g.idx in tail:  # the group's replay section, as the model brackets it: replay_begin ... replay_end, then the collapse
                pipe.replay_begin(x)
                seen["tail"].add((g.idx, h(current_stream()), h(torch.cuda.current_stream())))
                t = x
                for k in range(3):
                    t = torch.tanh(t @ W[k])
                pipe.replay_end(t)
                seen["after_tail"].add((g.idx, h(current_stream())))
                x = x + 0.25 * t
            mhc_warmup._MOET_SP_PIN_TOKENS = None
            return x
        return fn

    def forward(seed: int, tail=()):
        """One 'forward': a head on the caller's stream (slow enough to still run when the groups start), the pipelined step,
        a consumer on the caller's stream right behind it."""
        fresh_seen()
        g = torch.Generator(device=dev).manual_seed(seed)
        noise = torch.randn(ROWS, H, device=dev, generator=g)
        head = noise
        for _ in range(12):  # ~12 GEMMs queued on the caller's stream: the step's first kernel must come behind them
            head = torch.tanh(head @ W[0]) + noise
        inputs = [b + head for b in base_inputs]
        fc._forward_context = "main"
        mhc_warmup._MOET_SP_PIN_TOKENS = None
        outs = pipe.run_groups(make_groups(), fn_for(inputs, tail), pipelined=True)
        merged = pipe.merge_outputs(outs)
        tail = torch.tanh(merged @ W[1]).sum(dim=1)  # issued on the caller's stream, no host sync before it
        torch.cuda.synchronize()
        return merged, tail

    def set_key(value) -> None:
        t = tmp / "ab.json.tmp"
        t.write_text(json.dumps({} if value is None else {"PREFILL_PIPE_GREEN": value, "_t": time.time_ns()}))
        os.replace(t, tmp / "ab.json")
        pipe._GREEN["next_poll"] = 0.0

    green_pair = (h(sp.compute), h(sp.compute))
    comm_pair = (h(sp.comm), h(sp.comm))
    aux_green = tuple(h(s) for s in sp.aux)
    aux_orig = tuple(h(s) for s in runner.aux_orig)

    with stderr_lines() as buf:
        out_on, tail_on = forward(7)
    s_on = dict(seen)
    check(s_on["phase"] == {comm_pair} and s_on["compute"] == {green_pair},
          "(6) inside a phase the current stream is the comm partition's, outside it the compute partition's - vllm's and torch's "
          f"bookkeeping, both threads (phase {len(s_on['phase'])} distinct, compute {len(s_on['compute'])} distinct)")
    check(s_on["aux"] == {aux_green} and tuple(h(s) for s in runner.aux_list) == aux_orig and h(current_stream()) == h(main_stream),
          "(6) the aux-stream list holds the compute partition's streams during the step, its own after it; the caller's stream is current again")
    order = s_on["order"]
    check(s_on["ctx_ok"] and s_on["pin_ok"] and s_on["rows_ok"] and all(
        order.index((0, k)) < order.index((1, k)) for k in range(LAYERS)),
        "(6) per-thread forward context, pin and row storage at every resume; group 0's k-th compute segment before group 1's")
    set_key(0)
    with stderr_lines() as buf:
        out_off, tail_off = forward(7)
    line_off = buf.getvalue()
    s_off = dict(seen)
    main_pair = (h(main_stream), h(main_stream))
    check(s_off["phase"] == {(h(runner.comm_stream), h(runner.comm_stream))} and s_off["compute"] == {main_pair}
          and s_off["aux"] == {aux_orig} and "runtime A/B: PREFILL_PIPE_GREEN" in line_off and f"{args.n} -> 0" in line_off,
          "(7) key at 0: the step runs on the caller's stream and the pipeline's comm stream, the aux list untouched; one change line")
    check(torch.equal(out_on, out_off) and torch.equal(tail_on, tail_off),
          "(6)(7) outputs of the split step and of the unsplit step bit-identical (the step's rows and the caller-stream consumer "
          "behind it)")
    set_key(args.n)
    with stderr_lines() as buf:
        out_on2, tail_on2 = forward(7)
    line_on = buf.getvalue()
    check(seen["phase"] == {comm_pair} and seen["compute"] == {green_pair} and torch.equal(out_on2, out_on) and torch.equal(tail_on2, tail_on)
          and f"0 -> {args.n}" in line_on, "(7) key back at the start-up value: on the split again, identical outputs, one change line")
    set_key(None)
    out3, _ = forward(11)
    set_key(0)
    out4, _ = forward(11)
    set_key(None)
    out5, _ = forward(11)
    check(torch.equal(out3, out4) and torch.equal(out3, out5) and seen["compute"] == {green_pair},
          "(7) {} = the start-up state; another input, three states in a row: identical outputs")
    st = pipe.stats()["green"]
    check(st["forwards"] == 4 and st["forwards_off"] == 2 and st["rows"] == 4 * 2 * ROWS and st["on"] is True
          and st["comm_sms"] == sp.na and st["compute_sms"] == sp.nb,
          f"(7) counters: {st['forwards']} forwards on the split, {st['forwards_off']} off, rows {st['rows']} / {st['rows_off']}")
    (tmp / "stats").write_text("")
    with stderr_lines() as buf:
        forward(11)
    check("activation counters (on demand" in buf.getvalue() and f"switches now PREFILL_PIPE_GREEN={args.n}" in buf.getvalue(),
          "(7) the on-demand line after a touch of the stats file")
    if not args.tp:
        # the sequential oracle of the same step (no phases, the caller's stream): the split's bits are the unpipelined ones
        fresh_seen()
        g = torch.Generator(device=dev).manual_seed(7)
        noise = torch.randn(ROWS, H, device=dev, generator=g)
        head = noise
        for _ in range(12):
            head = torch.tanh(head @ W[0]) + noise
        inputs = [b + head for b in base_inputs]
        seq = pipe.merge_outputs(pipe.run_groups(make_groups(), fn_for(inputs), pipelined=False))
        torch.cuda.synchronize()
        check(torch.equal(seq, out_on), "(6) the split step == the groups run one after the other on the caller's stream (bits)")

    # ---------------------------------------------------------------------------------- (12) the replay tails of a split step
    main_h = h(main_stream)
    res12 = {}
    for name, env, key, groups_with_tail in (
        ("GREEN_TAIL=caller", "caller", None, (0, 1)), ("GREEN_TAIL=compute", "compute", None, (0, 1)), ("key at 0", "caller", 0, (0, 1)),
        ("GREEN_TAIL=caller, only the later group has a tail", "caller", None, (1,)),
        ("GREEN_TAIL=compute, only the later group has a tail", "compute", None, (1,)),
    ):
        os.environ[P + "_GREEN_TAIL"] = env
        set_key(key)
        with stderr_lines():
            out, tl = forward(13, tail=groups_with_tail)
        res12[name] = (out, tl, set(seen["tail"]), set(seen["after_tail"]), set(seen["compute"]))
    os.environ.pop(P + "_GREEN_TAIL")
    set_key(None)
    both, later = (0, 1), (1,)
    want_tail = {
        "GREEN_TAIL=caller": {(i, main_h, main_h) for i in both}, "GREEN_TAIL=compute": {(i, *green_pair) for i in both},
        "key at 0": {(i, main_h, main_h) for i in both},
        "GREEN_TAIL=caller, only the later group has a tail": {(1, main_h, main_h)},
        "GREEN_TAIL=compute, only the later group has a tail": {(1, *green_pair)},
    }
    streams_ok = all(res12[k][2] == want_tail[k] for k in want_tail) and all(
        res12[k][3] == {(i, green_pair[0]) for i in (both if "only" not in k else later)} for k in want_tail if k != "key at 0")
    check(streams_ok, "(12) a split step's replay section: on the caller's stream with GREEN_TAIL=caller, on the compute partition's with "
          "GREEN_TAIL=compute, the compute partition's stream again behind replay_end; a later group with a tail next to an earlier one "
          "without is fine" + ("" if streams_ok else f" - got { {k: (v[2], v[3]) for k, v in res12.items()} }"))
    same12 = (torch.equal(res12["GREEN_TAIL=caller"][0], res12["GREEN_TAIL=compute"][0])
              and torch.equal(res12["GREEN_TAIL=caller"][0], res12["key at 0"][0])
              and torch.equal(res12["GREEN_TAIL=caller"][1], res12["key at 0"][1])
              and torch.equal(res12["GREEN_TAIL=caller, only the later group has a tail"][0],
                              res12["GREEN_TAIL=compute, only the later group has a tail"][0]))
    check(same12, "(12) outputs with the tails on the caller's stream == on the compute partition == the unsplit step (bits)")

    # ---------------------------------------------------------------------------------- (9) where the kernels ran
    if smid is not None:
        set_key(None)
        forward(7)
        sets = {"phase": set(), "compute": set()}
        for kind, rec in seen["recs"]:
            sets[kind] |= Smid.sms(rec)
        check(0 < len(sets["phase"]) <= sp.na and 0 < len(sets["compute"]) <= sp.nb and not (sets["phase"] & sets["compute"]),
              f"(9) probe kernels launched inside the step: phase on {len(sets['phase'])} SMs (partition {sp.na}), compute segment on "
              f"{len(sets['compute'])} (partition {sp.nb}), overlap {len(sets['phase'] & sets['compute'])}")
        set_key(0)
        forward(7)
        off_sets = {"phase": set(), "compute": set()}
        for kind, rec in seen["recs"]:
            off_sets[kind] |= Smid.sms(rec)
        check(len(off_sets["compute"]) > sp.nb or len(off_sets["phase"]) > sp.na,
              f"(9) key at 0, the same probes: phase on {len(off_sets['phase'])} SMs, compute on {len(off_sets['compute'])} - not confined")
        set_key(None)

    # ---------------------------------------------------------------------------------- (8) a graph captured in the scope
    if not args.tp:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        X = torch.randn(1024, 1024, device=dev)
        Y = torch.randn(1024, 1024, device=dev)
        ref = torch.tanh(X @ Y) + 1
        torch.cuda.synchronize()

        def launch_like_a_tail(replay, stream, outs):
            """As a split step's replay tail: on `stream` reset the outputs, launch the graph(s), read them with an eager kernel
            behind the launch, and let the caller's stream wait for that stream by an event (GreenSplit.run's exit join) - stream
            order and events only, no host synchronisation of the device anywhere."""
            with torch.cuda.stream(stream):
                for t in outs:
                    t.zero_()
                replay()
                copies = [t.clone() for t in outs]
                ev = torch.cuda.Event()
                ev.record()
            ev.wait()
            return copies

        before = h(current_stream())
        os.environ[P + "_GREEN_TAIL"] = "compute"  # the mode in which the slots' graphs are captured on the compute partition
        # (a) the way the replay layers' graphs are captured: vllm's breakable capture - two graph segments around an eager break
        t1, z1 = torch.zeros(1024, 1024, device=dev), torch.zeros(1024, 1024, device=dev)
        rec1 = torch.zeros(752 * 24, dtype=torch.uint8, device=dev)
        cap = BreakableCUDAGraphCapture(pool=None)
        with pipe.shared_capture_scope(1, 128):
            inside = (h(current_stream()), h(torch.cuda.current_stream()))
            with cap:
                t1.copy_(torch.tanh(X @ Y))
                if smid is not None:
                    smid.launch(rec=rec1)
                cap.add_eager(lambda: t1.add_(1))  # at replay: an eager kernel on the replaying thread's stream, between the segments
                z1.copy_(t1)
        after = h(current_stream())
        check(inside == green_pair and before == after == h(main_stream) and pipe._GREEN["graph_scopes"] == 1 and cap.num_graphs == 2,
              "(8) inside shared_capture_scope the capturing thread's stream is the compute partition's; restored after it "
              f"({cap.num_graphs} graph segments, {cap.num_eager_breaks} eager break)")
        # (b) a plain torch.cuda.CUDAGraph (no kept template) captured in the scope
        z2 = torch.zeros(1024, 1024, device=dev)
        rec2 = torch.zeros(752 * 24, dtype=torch.uint8, device=dev)
        graph = torch.cuda.CUDAGraph()
        with pipe.shared_capture_scope(1, 128):
            graph.capture_begin()
            z2.copy_(torch.tanh(X @ Y) + 1)
            if smid is not None:
                smid.launch(rec=rec2)
            graph.capture_end()
        results = {}
        for kind, replay, outs in (("breakable capture", cap.replay, [z1, rec1]), ("plain graph", graph.replay, [z2, rec2])):
            for name, stream in (("the compute partition's stream", sp.compute), ("the caller's stream", main_stream)):
                for rep in range(2):
                    zc, rc = launch_like_a_tail(replay, stream, outs)
                    results[f"{kind} from {name} #{rep + 1}"] = (torch.equal(zc, ref), None if smid is None else len(Smid.sms(rc)))
        check(all(ok for ok, _ in results.values()),
              "(8) graphs captured on the compute partition replay with the eager result, launched as a replay tail launches them "
              "(outputs reset, the launch, an eager reader, all on one stream; the caller's stream joined by an event): "
              + ", ".join(f"{k} {'ok' if ok else 'DIFFERS'}" for k, (ok, _) in results.items()))
        if smid is not None:
            check(all(0 < n <= sp.nb for _, n in results.values()),
                  "(9) the captured probe kernel ran on the compute partition whichever stream launched the graph: SMs "
                  + ", ".join(f"{k} {n}" for k, (_, n) in results.items()) + f" (partition {sp.nb})")
        seen_inside = {}
        for name, env in (("GREEN_GRAPHS=0", {P + "_GREEN_GRAPHS": "0"}), ("GREEN_TAIL=caller", {P + "_GREEN_TAIL": "caller"})):
            os.environ.update(env)
            scope = pipe.shared_capture_scope(2, 128)
            with scope:
                seen_inside[name] = (h(current_stream()), scope.green_prev)
            os.environ[P + "_GREEN_TAIL"] = "compute"
            os.environ.pop(P + "_GREEN_GRAPHS", None)
        os.environ.pop(P + "_GREEN_TAIL")
        check(all(v == (h(main_stream), None) for v in seen_inside.values()),
              "(8) GREEN_GRAPHS=0 or GREEN_TAIL=caller: the scope leaves the capturing thread's stream alone (the graphs are today's)")
    else:
        # -------------------------------------------------------------------------- (11) a TP all-reduce captured on the partition
        # (the replay tails' graphs hold the TP all-reduces: NCCL's kernel and its per-communicator event nodes recorded on a
        # stream of the compute partition, replayed from that stream and from the caller's - every rank in step)
        import vllm.compilation.moet_nccl_graph_records as nccl_records
        from vllm.distributed import tensor_model_parallel_all_reduce

        set_key(None)
        os.environ[P + "_GREEN_TAIL"] = "compute"  # the mode in which the slots' graphs are captured on the compute partition
        xa = (torch.randn(128, H, device=dev) * 0.5).to(torch.bfloat16)
        wa = W[2][:, :256].contiguous()
        ref = torch.tanh(tensor_model_parallel_all_reduce(xa.clone()).float() @ wa)
        torch.cuda.synchronize()
        keep = nccl_records.records_enabled()
        graph = torch.cuda.CUDAGraph(keep_graph=keep)
        with pipe.shared_capture_scope(1, 128):
            inside = (h(current_stream()), h(torch.cuda.current_stream()))
            eager = torch.tanh(tensor_model_parallel_all_reduce(xa.clone()).float() @ wa)  # the capture's warm-up pass
            graph.capture_begin()
            za = torch.tanh(tensor_model_parallel_all_reduce(xa.clone()).float() @ wa)
            graph.capture_end()
        if keep:
            nccl_records.finish_capture(graph)
        torch.cuda.synchronize()
        results = {"eager on the compute partition": torch.equal(eager, ref)}
        for name, stream in (("replayed from the compute partition's stream", sp.compute), ("replayed from the caller's stream", main_stream)):
            for rep in range(2):
                with torch.cuda.stream(stream):  # as a tail: reset, launch, an eager reader on the same stream, the caller joined by an event
                    za.zero_()
                    graph.replay()
                    zc = za.clone()
                    ev = torch.cuda.Event()
                    ev.record()
                ev.wait()
                results[f"{name} #{rep + 1}"] = torch.equal(zc, ref)
        check(inside == green_pair and all(results.values()),
              "(11) a TP all-reduce (NCCL) + a GEMM captured inside shared_capture_scope on the compute partition's stream == the eager "
              "result on the caller's stream: " + ", ".join(f"{k} {'ok' if v else 'DIFFERS'}" for k, v in results.items()))
        del graph
        os.environ.pop(P + "_GREEN_TAIL")
    del runner


def main() -> int:
    if args.tp:
        import torch.distributed as dist

        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import init_distributed_environment, initialize_model_parallel

        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
        dev = torch.device("cuda", torch.cuda.current_device())
        with set_current_vllm_config(VllmConfig()), torch.device(dev):
            init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://", local_rank=rank, backend="nccl")
            initialize_model_parallel(tensor_model_parallel_size=world)
            with torch.inference_mode():
                gpu_part()
            bad = torch.tensor([float(len(failures))], device=dev)
            dist.all_reduce(bad)
            if rank == 0:
                print(f"SUMMARY TP ({world} ranks): {'PASS' if bad.item() == 0 else 'FAIL'} - failures on all ranks {int(bad.item())}; "
                      "(10) NCCL reduce-scatter + all-gather in the comm phases on the comm partition's stream, (11) a TP all-reduce captured on "
                      "the compute partition", flush=True)
            return 1 if bad.item() else 0
    cpu_part()
    n_cpu = len(failures)
    print(f"SUMMARY CPU part: {'PASS' if n_cpu == 0 else 'FAIL'} ({n_cpu} failures)", flush=True)
    if not args.cpu:
        if not torch.cuda.is_available():
            print("SUMMARY GPU part: SKIPPED (no CUDA device)", flush=True)
        else:
            torch.cuda.set_device(0)
            with torch.inference_mode():
                gpu_part()
            print(f"SUMMARY GPU part: {'PASS' if len(failures) == n_cpu else 'FAIL'} ({len(failures) - n_cpu} failures)"
                  + ("" if args.smid_lib else "; (9) not run (no --smid-lib)"), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
