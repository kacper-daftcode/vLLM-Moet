#!/usr/bin/env python3
"""The phase-wait instrument of the pipelined prefill step (VLLM_MOET_PREFILL_PIPE_PHASE_WAIT=1 | 2 | 3 in moet_prefill_pipe.py) on
fake streams, fake timing events with a fake GPU clock and a fake host clock - no GPU, no model; it needs vLLM importable (the
image), nothing else.

What it holds:
  (1) off by default: with the variable unset a phase records no timing event and keeps no state;
  (2) the proxy on the communicator (stage 2): a pass-through outside a phase; inside a comm phase one timing event before and one
      after a collective; the real method is called exactly once with the caller's arguments;
  (3) the three lines of a step driven through the module's own phase_begin / phase_end, two groups, three layers:
      `phase wait` (per group and phase kind: n, waited, slack), `phase split` (total, the compute segment beside the phase,
      the elements), `phase layers` (the phases with slack by index with excess / segment; the compute segments the host took
      1 ms or more to issue, wall / thread CPU; the rest as a count and medians);
  (4) the step's end harvests without the host-timing lines as well; a second step does not count the gap between steps as a segment;
  (5) phase_layers_table.py turns the `phase layers` lines into the table of long segments;
  (6) a failing event or clock switches the instrument off with one line and never raises; the plain phase code runs on;
  (7) a phase inside a CUDA graph capture is left alone (an event recorded there would be a node of the graph, not a time);
  (8) the level as a runtime A/B key (PREFILL_PIPE_PHASE_WAIT in the control file of VLLM_MOET_RUNTIME_AB=1): without the master
      switch the file is never looked at; with it the level is latched at a forward's dispatch, at most one stat per 100 ms, one
      change line per change and the on-demand line in the form a driver reads back; going off harvests the last instrumented
      forward first and leaves no state and no proxy; a missing / malformed file, a value out of range and an unknown key keep
      the level with one WARNING each; the SM split's reader passes the key;
  (9) phase_balance.py turns the lines into one line per rank (compute work, comm work, period, the four slots, the tail).

usage: python3 test_prefill_pipe_instrument.py [--module DIR]   (DIR = where moet_prefill_pipe.py lives; default: next to this file)
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--module", type=Path, default=Path(__file__).resolve().parent)
args = ap.parse_args()
HERE = Path(__file__).resolve().parent

os.environ.pop("VLLM_MOET_PREFILL_PIPE_PHASE_WAIT", None)
os.environ.pop("VLLM_MOET_PREFILL_PIPE_HOST_TIMING", None)
spec = importlib.util.spec_from_file_location("moet_prefill_pipe_instrument_test", args.module / "moet_prefill_pipe.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

fails = 0


def check(cond: bool, what: str) -> None:
    global fails
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    fails += 0 if cond else 1


lines: list[str] = []


class _Handler(logging.Handler):
    def emit(self, record):
        lines.append(record.getMessage())


m.logger.addHandler(_Handler())
m.logger.setLevel(logging.INFO)
m._tp_rank = lambda: 2

clock = [0.0]                          # the GPU clock, ms
host = {0: [0.0, 0.0], 1: [0.0, 0.0]}  # per group thread: wall and thread-CPU clocks, s (the other group's turn lies inside
#                                        this group's phase_end, before the stamp is taken)


class Ev:
    made = 0

    def __init__(self, enable_timing=False):
        self.t = None
        self.timing = enable_timing
        Ev.made += 1

    def record(self, stream=None):
        self.t = clock[0]

    def query(self):
        return self.t is not None

    def elapsed_time(self, other):
        return other.t - self.t


class Stream:
    def wait_event(self, ev):
        pass


calls: list[tuple] = []


class Comm:
    disabled = False
    world_size = 4

    def reduce_scatter(self, out, inp, op="SUM", stream=None):
        calls.append(("reduce_scatter", out, inp, op, stream))
        clock[0] += 0.7
        return "rs-ret"

    def all_gather(self, out, inp, stream=None):
        calls.append(("all_gather", out, inp, stream))
        clock[0] += 0.6

    def other(self):
        return 42


real_event, real_set_stream, real_time = m.torch.cuda.Event, m.torch.cuda.set_stream, m.time
m.torch.cuda.Event = Ev
m.torch.cuda.set_stream = lambda s: None
cur = [types.SimpleNamespace(idx=0)]
m.time = types.SimpleNamespace(perf_counter=lambda: host[cur[0].idx][0], thread_time=lambda: host[cur[0].idx][1], time=real_time.time,
                               sleep=real_time.sleep, monotonic=real_time.monotonic)
import vllm.distributed.parallel_state as ps  # noqa: E402

base = Comm()
dc = types.SimpleNamespace(pynccl_comm=base)
real_get_tp_group = ps.get_tp_group
ps.get_tp_group = lambda: types.SimpleNamespace(device_communicator=dc)

free_at: dict[int, float] = {}
step = types.SimpleNamespace(pipelined=True, comm_stream=Stream(), replay_stream=Stream(), compute_stream=Stream(), tail_on_caller=False,
                             check_order=lambda g: None)
step.yield_ = lambda g: clock.__setitem__(0, free_at[g.idx])
groups = [types.SimpleNamespace(idx=i, step=step, phase_open=False, phase_stream=None, phase_kind="", phases_done=0,
                                last_phase_event=None, comm_held=set()) for i in (0, 1)]
cur[0] = groups[0]
m.current_group = lambda: cur[0]

# ---- (1) off by default
n0 = Ev.made
clock[0] = 10.0
free_at[0] = 11.0
m.phase_begin()
m.phase_end()
check(not m.phase_wait() and not m.phase_split() and not m.phase_layers(), "(1) the three stages are off with the variable unset")
check(not m._PW["pending"] and not m._PW["host0"] and not m._PW["host"] and not m._PW["timer"] and dc.pynccl_comm is base,
      "(1) off: a phase keeps no instrument state and installs no proxy")
check(Ev.made - n0 == 2, f"(1) off: a phase creates its two plain events only ({Ev.made - n0})")
groups[0].phases_done = 0

# ---- (2) the proxy (stage 2)
os.environ["VLLM_MOET_PREFILL_PIPE_PHASE_WAIT"] = "3"
check(m.phase_wait() and m.phase_split() and m.phase_layers(), "(2) =3 switches the three stages on")
timer = m.CommTimer(base)
n0 = Ev.made
check(timer.other() == 42 and timer.disabled is False and timer.world_size == 4, "(2) the proxy resolves every other attribute on the communicator")
check(timer.reduce_scatter("o", "i") == "rs-ret" and calls[-1] == ("reduce_scatter", "o", "i", "SUM", None) and Ev.made == n0,
      "(2) outside a phase: the collective is called once with the caller's arguments, no event")
m._phase_split_install()
m._phase_split_install()  # idempotent
proxy = dc.pynccl_comm
check(isinstance(proxy, m.CommTimer) and m._PW["timer"], "(2) the proxy is installed on the TP group's communicator, once")

# ---- (3) a step of three layers through the module's own phase code
# compute segment beside a phase (ms from the phase's start): group 0's attention of layers 0 and 2 is long (beside group 1's
# phases 0 and 4), group 1's attention of layer 0 is long (beside group 0's phase 1); the host needs 6.0 ms wall / 5.5 ms CPU to
# issue group 0's attention of layers 0 and 2 and 0.4 ms for every other segment
F = {(1, 0): 7.5, (1, 4): 7.5, (0, 1): 3.7}
HOSTLONG = {(0, 1), (0, 5)}


def run_step(t_base: float) -> None:
    for g in groups:
        g.phases_done = 0
    seq = 0
    for layer in range(3):
        for par in (0, 1):
            for g in groups:
                p = 2 * layer + par
                cur[0] = g
                w, c = (6.0e-3, 5.5e-3) if (g.idx, p) in HOSTLONG else (0.4e-3, 0.4e-3)
                host[g.idx][0] += w
                host[g.idx][1] += c
                t0 = t_base + 100.0 * seq
                seq += 1
                clock[0] = t0
                m.phase_begin()
                clock[0] = t0 + 0.05
                proxy.reduce_scatter("o", "i", op="SUM")       # 0.7 ms
                clock[0] += 0.4                                 # the glue
                proxy.all_gather("o2", "i2")                    # 0.6 ms
                clock[0] = t0 + 2.0                             # the phase's end
                free_at[g.idx] = t0 + F.get((g.idx, p), 1.6)    # the compute stream is free (the fake yield sets the clock)
                m.phase_end()
                host[g.idx][0] += 0.05e-3


k = len(calls)
run_step(1000.0)
check(len(calls) == k + 24 and calls[-2][:4] == ("reduce_scatter", "o", "i", "SUM") and calls[-1][:3] == ("all_gather", "o2", "i2"),
      "(3) inside a phase every collective is still called once with the caller's arguments")
check(len(m._PW["pending"]) == 12 and len(m._PW["host"]) == 10, f"(3) 12 phases pending, 10 host segments ({len(m._PW['pending'])}, {len(m._PW['host'])})")
check(all(g.phases_done == 6 and not g.phase_open for g in groups), "(3) the phase bookkeeping is the plain one")
n = len(lines)
m._host_step_done(8192, True, 0.0, 1.0)   # (4) host timing is off: the step's end still harvests
new = lines[n:]
for line in new:
    print("    " + line[:330])
wait = [x for x in new if "phase wait:" in x]
split = [x for x in new if "phase split:" in x]
lay = [x for x in new if "phase layers:" in x]
check(len(wait) == 1 and "g1 before attention: n 3, waited 0.40 ms in 1, slack 11.00 ms" in wait[0]
      and "g0 before MoE: n 3, waited 0.80 ms in 2, slack 1.70 ms" in wait[0], "(3) the wait line: n, waited, slack per group and kind")
check(len(split) == 1 and "g0 before MoE: n 3, total 6.00, compute free at 6.90 | glue 0.15/0.00 x3, reduce_scatter 2.10/0.00 x3, "
      "glue 1.20/0.00 x3, all_gather 1.80/0.30 x3, tail 0.75/0.50 x3" in split[0], "(3) the split line: total, the segment beside, the elements")
check(len(lay) == 1 and "slack g0 p1:1.70/3.70" in lay[0] and "slack g1 p0:5.50/7.50 p4:5.50/7.50" in lay[0],
      "(3) the layers line names the phases with slack: index, excess / segment")
check(len(lay) == 1 and "host g0 p1:6.05/5.50 p5:6.05/5.50 (rest: n 3, median 0.45/0.40)" in lay[0]
      and "host g1  (rest: n 5, median 0.45/0.40)" in lay[0], "(3) the layers line names the segments the host took long to issue: wall / CPU")
check(not any("host timing" in x for x in new), "(4) no host-timing line without VLLM_MOET_PREFILL_PIPE_HOST_TIMING")
check(not m._PW["pending"] and not m._PW["host"] and not m._PW["dead"], "(4) the harvest leaves nothing pending")
n = len(lines)
run_step(5000.0)
check(len(m._PW["host"]) == 10, f"(4) a second step: the gap between two steps is not a host segment ({len(m._PW['host'])})")
m._host_step_done(8192, True, 0.0, 1.0)
check(sum("phase layers:" in x for x in lines[n:]) == 1, "(4) one layers line per step")

# ---- (5) the table tool on the lines
with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
    for x in lines:
        f.write("(Worker_TP2 pid=1) INFO " + x + "\n")
out = subprocess.run([sys.executable, str(HERE / "phase_layers_table.py"), f.name], capture_output=True, text=True)
os.unlink(f.name)
rows = [x.split() for x in out.stdout.splitlines() if re.match(r"  g[01] ", x)]
check(out.returncode == 0 and [r[:3] for r in rows] == [["g0", "0", "attention"], ["g0", "2", "attention"], ["g1", "0", "attention"]],
      f"(5) the table: group 0's attention of layers 0 and 2, group 1's of layer 0 ({[r[:3] for r in rows]}; rc {out.returncode})")
check(len(rows) == 3 and rows[0][3:6] == ["100", "5.50", "7.50"] and rows[2][3:6] == ["100", "1.70", "3.70"]
      and rows[0][6:9] == ["6.05", "/", "5.50"], f"(5) the table: share of the lines, excess, segment, host wall / CPU ({rows[0][3:9] if rows else None})")

# ---- (6) failures switch the instrument off and never raise
def boom():
    raise RuntimeError("no clock")


m.time.thread_time = boom
cur[0] = groups[0]
groups[0].phases_done = 3
clock[0] = 9000.0
free_at[0] = 9001.0
m.phase_begin()
m.phase_end()
check(m._PW["dead"] and not m.phase_layers() and not m.phase_split() and not m.phase_wait() and not m._PW["host0"] and not m._PW["host"]
      and "instrument switched off" in lines[-1], "(6) a failing host clock: one line, the instrument is off, nothing raised")
k = len(calls)
m.phase_begin()
check(proxy.all_gather("o4", "i4") is None and len(calls) == k + 1, "(6) off: the proxy is a pass-through")
m.phase_end()
check(groups[0].phases_done == 5 and not groups[0].phase_open, "(6) off: the plain phase code runs on")
m._PW.update({"dead": False, "pending": [], "marks": {}, "start": {}})


class BadEv(Ev):
    def record(self, stream=None):
        if self.timing:
            raise RuntimeError("no record")
        super().record(stream)


m.torch.cuda.Event = BadEv
groups[0].phases_done = 1
m.phase_begin()
k = len(calls)
ok = proxy.reduce_scatter("o3", "i3") == "rs-ret" and len(calls) == k + 1 and calls[-1][:3] == ("reduce_scatter", "o3", "i3")
m.phase_end()
check(ok and m._PW["dead"] and not m._PW["marks"] and not m._PW["pending"] and not groups[0].phase_open,
      "(6) a failing timing event: the collective is still called once, the instrument is off, the phase closes")

# ---- (7) a phase inside a CUDA graph capture is left alone
m._PW.update({"dead": False, "pending": [], "marks": {}, "start": {}, "host0": {}, "host": []})
m.torch.cuda.Event = Ev
real_capturing = m.torch.cuda.is_current_stream_capturing
m.torch.cuda.is_current_stream_capturing = lambda: True
groups[0].phases_done = 1
n0 = Ev.made
m.phase_begin()
m.phase_end()
check(not m._PW["pending"] and not m._PW["start"] and not m._PW["host0"] and Ev.made - n0 == 2 and not m._PW["dead"],
      "(7) a phase inside a graph capture: no timing event, nothing pending, the instrument stays on")
m.torch.cuda.is_current_stream_capturing = real_capturing

# ---- (8) the level as a runtime A/B key: PREFILL_PIPE_PHASE_WAIT in the control file of VLLM_MOET_RUNTIME_AB=1
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402

os.environ.pop("VLLM_MOET_PREFILL_PIPE_PHASE_WAIT", None)
m._phase_split_remove()
check(dc.pynccl_comm is base and not m._PW["timer"], "(8) the proxy comes off: the communicator is the object it was before")
mono = [1000.0]
m.time.monotonic = lambda: mono[0]
m.time.thread_time = lambda: host[cur[0].idx][1]  # (6) left the failing clock in place
real_current_device = m.torch.cuda.current_device
m.torch.cuda.current_device = lambda: 2
real_stat = os.stat
stats: list[str] = []


def counting_stat(path, *a, **k):
    stats.append(str(path))
    return real_stat(path, *a, **k)


os.stat = counting_stat
tmpd = tempfile.mkdtemp()
ctl, sfile = os.path.join(tmpd, "ab.json"), os.path.join(tmpd, "stats")
os.environ["VLLM_MOET_RUNTIME_AB_FILE"] = ctl
os.environ["VLLM_MOET_RUNTIME_STATS_FILE"] = sfile
fwd = [types.SimpleNamespace(num_tokens=4096), types.SimpleNamespace(num_tokens=4096)]
TAG = "[vllm-moet prefill pipeline instrument, device 2] "


def dispatch(dt: float = 1.0) -> list[str]:
    """What run_groups does first for a served forward; returns the lines the reader printed (the contract's stderr lines)."""
    mono[0] += dt
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        m._pw_runtime_hooks(fwd)
    return [x for x in buf.getvalue().splitlines() if x]


writes = [0]


def write_ctl(obj) -> None:
    """A fresh temp file and a rename, as a driver writes; every write gets an mtime of its own (the reader's change test is
    (mtime, size), and two writes of this test can fall into one tick of the file system's clock)."""
    with open(ctl + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(obj if isinstance(obj, str) else json.dumps(obj))
    writes[0] += 1
    os.utime(ctl + ".tmp", ns=(writes[0] * 10**9, writes[0] * 10**9))
    os.replace(ctl + ".tmp", ctl)


def plain_phase(t0: float) -> int:
    """One comm phase of group 0 through the module's phase code; returns the events it created."""
    made = Ev.made
    cur[0] = groups[0]
    groups[0].phases_done = 0
    clock[0] = t0
    free_at[0] = t0 + 1.0
    m.phase_begin()
    m.phase_end()
    return Ev.made - made


os.environ.pop("VLLM_MOET_RUNTIME_AB", None)
m._PWK["ab"] = None
write_ctl({"PREFILL_PIPE_PHASE_WAIT": 3})
n0 = len(stats)
out = dispatch()
check(out == [] and len(stats) == n0 and m._PWK["level"] is None and not m.phase_wait() and plain_phase(20000.0) == 2,
      "(8) without VLLM_MOET_RUNTIME_AB=1 the file is never looked at: no stat, no line, no level, a plain phase")
os.unlink(ctl)
os.environ["VLLM_MOET_RUNTIME_AB"] = "1"
m._PWK["ab"] = None
out = dispatch()
check(len(out) == 1 and out[0].startswith(TAG + "WARNING: runtime A/B: control file") and "missing" in out[0] and m._PWK["level"] == 0,
      "(8) the reader on, no file: one WARNING, the environment's level")
check(dispatch() == [] and plain_phase(20100.0) == 2 and not m._PW["pending"] and dc.pynccl_comm is base,
      "(8) the missing file is said once; at level 0 a phase creates its two plain events only")
write_ctl({"PREFILL_PIPE_PHASE_WAIT": 3, "B12X_MOE_PREFILL_FAST_GRID2": 256, "PREFILL_PIPE_GREEN": 0, "_seq": 1})
out = dispatch()
CHG = re.compile(r"\[([^\[\]]+), device (\d+)\] runtime A/B: (\w+) (-?\d+) -> (-?\d+)")
mo = CHG.search(out[0]) if len(out) == 1 else None
check(mo is not None and mo.groups()[1:] == ("2", "PREFILL_PIPE_PHASE_WAIT", "0", "3") and m.phase_layers(),
      f"(8) the key at 3: one change line in the contract's form, the three stages on; other keys pass silently ({out})")
n0 = stats.count(ctl)
check(dispatch(0.05) == [] and stats.count(ctl) == n0, "(8) at most one stat of the control file per 100 ms")
write_ctl({"PREFILL_PIPE_PHASE_WAIT": 0})
check(m.phase_layers(), "(8) a write alone changes nothing: the level is latched at a forward's dispatch")
write_ctl({"PREFILL_PIPE_PHASE_WAIT": 3, "_seq": 2})
check(dispatch() == [] and m.phase_wait_level() == 3, "(8) a rewrite with the same level: no change line")
k = len(calls)
run_step(30000.0)
check(len(m._PW["pending"]) == 12 and isinstance(dc.pynccl_comm, m.CommTimer) and m._PW["timer"] and len(calls) == k + 24,
      "(8) at 3 the phases are timed and the proxy is on the communicator again")
n = len(lines)
m._host_step_done(8192, True, 0.0, 1.0)
new = lines[n:]
THREE = ("phase wait: at a step", "phase split: at a step", "phase layers: at a step")
check([sum(tag in x for x in new) for tag in THREE] == [1, 1, 1] and not m._PW["pending"], "(8) the step's end harvests the three lines")
run_step(40000.0)  # the last forward with the instrument on: its phases are pending when the key goes away
with open(sfile, "w", encoding="utf-8"):
    pass
out = dispatch()
OND = re.compile(r"\[([^\[\]]+), device (\d+)\] activation counters \(on demand, call (\d+); [^)]*\): (.*?); switches now ((?:\w+=-?\d+ ?)+)")
mo = OND.search(out[0]) if len(out) == 1 else None
check(mo is not None and mo.group(2) == "2" and mo.group(5).strip() == "PREFILL_PIPE_PHASE_WAIT=3"
      and re.search(r"instrumented forwards (\d+) / (\d+), plain forwards (\d+) / (\d+)", mo.group(4)) is not None,
      f"(8) a touch of the stats file: one on-demand line a driver reads back, the level in `switches now` ({out})")
check(dispatch() == [] and len(m._PW["pending"]) == 12, "(8) one touch, one line; a dispatch at the same level harvests nothing")
write_ctl({})
n = len(lines)
out = dispatch()
new = lines[n:]
check(out == [TAG + "runtime A/B: PREFILL_PIPE_PHASE_WAIT 3 -> 0"], f"(8) the key absent: back to the environment's level, one change line ({out})")
check([sum(tag in x for x in new) for tag in THREE] == [1, 1, 1] and not m._PW["pending"],
      "(8) going off harvests the last instrumented forward first: its three lines, nothing pending")
check(sum("the timing proxy is off the communicator" in x for x in new) == 1, "(8) going off says once that the proxy left the communicator")
check(dc.pynccl_comm is base and not m._PW["timer"] and not m._PW["host0"] and not m._PW["host"] and not m._PW["marks"]
      and not m._PW["start"], "(8) off: the proxy is off the communicator and no instrument state is left")
check(plain_phase(50000.0) == 2 and not m._PW["pending"], "(8) off: a phase creates its two plain events only")
write_ctl({"PREFILL_PIPE_PHASE_WAIT": 7, "PREFILL_PIPE_NOPE": 1})
out = dispatch()
check(len(out) == 2 and sum("out of range" in x for x in out) == 1 and sum("unknown key PREFILL_PIPE_NOPE" in x for x in out) == 1
      and m.phase_wait_level() == 0, f"(8) a value out of range and an unknown key of the namespace: one WARNING each, the level kept ({out})")
write_ctl({"PREFILL_PIPE_PHASE_WAIT": True})
check(dispatch() == [] and m.phase_wait_level() == 0, "(8) a boolean is not a level; the WARNING of a key is said once")
write_ctl("not json")
out = dispatch()
check(len(out) == 1 and "malformed" in out[0] and m.phase_wait_level() == 0, "(8) a malformed file: one WARNING, the level kept")
write_ctl({"PREFILL_PIPE_PHASE_WAIT": 1})
out = dispatch()
check(out == [TAG + "runtime A/B: PREFILL_PIPE_PHASE_WAIT 0 -> 1"] and m.phase_wait() and not m.phase_split(), "(8) the key at 1: the first stage only")
check(plain_phase(60000.0) == 3 and len(m._PW["pending"]) == 1 and dc.pynccl_comm is base, "(8) at 1: one more event per phase, no proxy")
os.unlink(ctl)
out = dispatch()
check(len(out) == 1 and "missing" in out[0] and m.phase_wait_level() == 1, "(8) the file removed after a state was applied: the state stays (one WARNING)")
write_ctl({"PREFILL_PIPE_PHASE_WAIT": 2})
out = dispatch()
check(out == [TAG + "runtime A/B: PREFILL_PIPE_PHASE_WAIT 1 -> 2"] and m.phase_split() and not m.phase_layers() and len(m._PW["pending"]) == 1,
      "(8) the level goes up: nothing is harvested or dropped at the flip")
m._phase_wait_fail("the test", RuntimeError("x"))
os.utime(sfile, ns=(1, 1))
out = dispatch()
mo = OND.search(out[0]) if len(out) == 1 else None
check(mo is not None and mo.group(5).strip() == "PREFILL_PIPE_PHASE_WAIT=0", "(8) after the instrument switched itself off the read-back says 0")
write_ctl({})
out = dispatch()
check(len(out) == 1 and "PREFILL_PIPE_PHASE_WAIT 2 -> 0" in out[0] and "switched itself off" in out[0], "(8) the change line says that the instrument was off already")
m._PW["dead"] = False
buf = io.StringIO()
with contextlib.redirect_stderr(buf):
    green_changes = m.green_apply_runtime({"PREFILL_PIPE_PHASE_WAIT": 2})
check(green_changes == [] and "unknown key" not in buf.getvalue(), "(8) the SM split's reader passes the instrument's key without a WARNING")
os.stat = real_stat
m.torch.cuda.current_device = real_current_device
os.unlink(ctl)
os.unlink(sfile)
os.rmdir(tmpd)
for name in ("VLLM_MOET_RUNTIME_AB", "VLLM_MOET_RUNTIME_AB_FILE", "VLLM_MOET_RUNTIME_STATS_FILE"):
    os.environ.pop(name, None)

# ---- (9) phase_balance.py: one line per rank from the lines of this test's steps
with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
    for x in lines:
        f.write("(Worker_TP2 pid=1) INFO " + x + "\n")
out = subprocess.run([sys.executable, str(HERE / "phase_balance.py"), f.name, "--sp-layers", "2", "--min-phases", "3"],
                     capture_output=True, text=True)
os.unlink(f.name)
rank_lines = [x for x in out.stdout.splitlines() if x.startswith("rank ")]
for x in out.stdout.splitlines():
    print("    " + x[:330])
# every step of this test: phases of 2.0 ms; the segment beside a phase 1.6 ms, 3.7 ms beside g0's phase 1 and 7.5 ms beside g1's
# phases 0 and 4 - the last of them is the step's last segment, which the tool takes out as the tail. Per phase and slot then:
# segments 1.6 (beside g1's MoE), (7.5 + 1.6) / 2 = 4.55, (3.7 + 1.6 + 1.6) / 3 = 2.3, 1.6 -> compute work 10.05; four phases
# of 2.0 -> comm work 8.0; waited 0.4, 0.4 / 2, 0.8 / 3, 0.4 -> period 10.05 + 1.2667 = 11.317
check(out.returncode == 0 and len(rank_lines) == 1
      and rank_lines[0].startswith("rank 2: 4 steps | compute work 10.050, comm work 8.000, period 11.317, comm - compute -2.050"),
      f"(9) one line per rank: the steps, compute work, comm work and period per layer with the tail taken out (rc {out.returncode})")
check(len(rank_lines) == 1 and "beside g1's MoE: phase 2.000 (rs 0.70, glue 0.40, ag 0.60), segment 1.600, waited 0.400, slack 0.000" in rank_lines[0]
      and "replay tail 7.50 per step" in rank_lines[0], "(9) a slot: the phase with its reduce-scatter / glue / all-gather, the segment, waited, slack; the tail")
check(sum(x.startswith("pace: ") for x in out.stdout.splitlines()) == 1 and sum(x.startswith("cap: ") for x in out.stdout.splitlines()) == 1,
      "(9) the two readings follow the rank lines")

m.torch.cuda.Event, m.torch.cuda.set_stream, m.time, ps.get_tp_group = real_event, real_set_stream, real_time, real_get_tp_group
print(f"{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
