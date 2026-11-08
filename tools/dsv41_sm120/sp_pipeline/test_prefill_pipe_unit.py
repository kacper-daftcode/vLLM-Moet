#!/usr/bin/env python3
"""Unit test of the pipeline's handoff protocol on one GPU, without the model (inside the image: the module imports
vllm's forward context and the mHC warm-up pin).

Checks: (1) two groups alternate strictly at the phase ends - group 0's k-th compute segment is issued before group 1's
(the KV write-before-read order on the one compute stream); (2) inside a phase the current stream is the comm stream,
outside it the compute stream, per thread; (3) each thread sees its own forward context, mHC pin and row-buffer storage
at every resume, and the runner thread gets its own back; (4) the results are correct and merge in row order (a GEMM
chain per group on real tensors: the compute of one group never reads what the other group's phase wrote); (5) a
group with fewer phases than an earlier one is fine, more phases than an earlier one raises; (6) a failing group
raises on the runner thread instead of hanging.

usage: python3 test_prefill_pipe_unit.py [--module DIR]   (DIR = where moet_prefill_pipe.py lives; default: next to this file)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--module", type=Path, default=Path(__file__).resolve().parent)
args = ap.parse_args()
sys.path.insert(0, str(args.module))
import moet_prefill_pipe as pipe  # noqa: E402

from vllm import forward_context as fc  # noqa: E402
import vllm.model_executor.kernels.mhc.warmup as mhc_warmup  # noqa: E402
from vllm.utils.torch_utils import current_stream  # noqa: E402

dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
torch.manual_seed(0)
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        failures.append(msg)


class FakeRunner:
    def __init__(self):
        self.comm_stream = torch.cuda.Stream(device=dev)
        self.replay_stream = torch.cuda.Stream(device=dev)
        self.device = dev
        self.row_buf = torch.zeros(8, 4, dtype=torch.int32, device=dev)
        self.row_switch = pipe.RowBufferSwitch([self.row_buf], 2)


runner = FakeRunner()
pipe._RUNNER = runner
main_stream = current_stream()
C = main_stream
N = runner.comm_stream


def make_groups(n=2, rows=4):
    return [
        pipe.PipeGroup(idx=i, token_offset=i * rows, num_tokens=rows, forward_context=f"ctx{i}", slot=i + 1)
        for i in range(n)
    ]


# ------------------------------------------------------------------------------------------ (1)-(4): the protocol
H = 256
LAYERS = 6
W = [torch.randn(H, H, device=dev) * 0.05 for _ in range(LAYERS)]
inputs = [torch.randn(4, H, device=dev) for _ in range(2)]
order: list[tuple[str, int, int]] = []  # (what, group, layer) in CPU issue order
seen = {"phase_stream_ok": True, "compute_stream_ok": True, "ctx_ok": True, "pin_ok": True, "rows_ok": True, "replay_ctx_ok": True, "replay_stream_ok": True}
lock = threading.Lock()


def reference(x):
    for layer in range(LAYERS):
        x = torch.tanh(x @ W[layer])  # "phase" op
        x = torch.tanh(x @ W[layer].t())  # "compute" op
    return x


def fn(g: pipe.PipeGroup):
    x = inputs[g.idx]
    mhc_warmup._MOET_SP_PIN_TOKENS = 1000 + g.idx  # the layer's pin (set at the layer start in the real code)
    runner.row_buf[: g.num_tokens].fill_(g.idx + 1)
    for layer in range(LAYERS):
        pipe.phase_begin()
        if current_stream() != N or torch.cuda.current_stream() != N:
            seen["phase_stream_ok"] = False
        y = torch.tanh(x @ W[layer])
        pipe.record_for_comm(x)
        pipe.phase_end()
        # resumed: compute stream, own context / pin / rows
        if current_stream() != C or torch.cuda.current_stream() != C:
            seen["compute_stream_ok"] = False
        if fc._forward_context != g.forward_context:
            seen["ctx_ok"] = False
        if mhc_warmup._MOET_SP_PIN_TOKENS != 1000 + g.idx:
            seen["pin_ok"] = False
        if int(runner.row_buf[0, 0].item()) != g.idx + 1:
            seen["rows_ok"] = False
        with lock:
            order.append(("compute", g.idx, layer))
        x = torch.tanh(y @ W[layer].t())
    mhc_warmup._MOET_SP_PIN_TOKENS = None
    return x


fc._forward_context = "main"
mhc_warmup._MOET_SP_PIN_TOKENS = None
groups = make_groups()
t0 = time.time()
outs = pipe.run_groups(groups, fn, pipelined=True)
torch.cuda.synchronize()
dt = time.time() - t0
merged = pipe.merge_outputs(outs)
ref = torch.cat([reference(inputs[0]), reference(inputs[1])], dim=0)
check(torch.equal(merged, ref), f"(4) pipelined results equal the serial reference, merged in row order ({dt*1000:.0f} ms)")
expect = []
for layer in range(LAYERS):
    expect += [("compute", 0, layer), ("compute", 1, layer)]
check(order == expect, f"(1) strict alternation: {order[:4]} ... == A0 B0 A1 B1 ...")
check(seen["phase_stream_ok"], "(2) inside a phase the current stream is the comm stream (torch and vLLM views)")
check(seen["compute_stream_ok"], "(2) after a phase end the current stream is the compute stream")
check(seen["ctx_ok"], "(3) each thread sees its own forward context at every resume")
check(seen["pin_ok"], "(3) each thread sees its own mHC pin at every resume")
check(seen["rows_ok"], "(3) each thread sees its own row-buffer storage at every resume")
check(fc._forward_context == "main", "(3) the runner thread's forward context is restored")
check(mhc_warmup._MOET_SP_PIN_TOKENS is None, "(3) the runner thread's pin is restored")
check(runner.row_switch.current == 0 and int(runner.row_buf[0, 0].item()) == 1, "(3) the row buffers point at the original storage = group 0's (it wrote 1 there)")
check(current_stream() == main_stream, "(2) the runner thread's current stream is unchanged")
check(pipe.current_group() is None, "no group on the runner thread afterwards")

# sequential mode
order.clear()
outs = pipe.run_groups(make_groups(), fn, pipelined=False)
torch.cuda.synchronize()
check(torch.equal(pipe.merge_outputs(outs), ref), "(4) sequential mode (a group below the SP minimum) gives the same results")
check(order == [("compute", 0, l) for l in range(LAYERS)] + [("compute", 1, l) for l in range(LAYERS)], "sequential mode issues group 0 whole, then group 1")


# ------------------------------------------------------------------------------------------ (5): unequal phase counts
def fn_phases(n_by_group):
    def f(g):
        x = inputs[g.idx]
        for layer in range(n_by_group[g.idx]):
            pipe.phase_begin()
            x = torch.tanh(x @ W[layer % LAYERS])
            pipe.phase_end()
        return x

    return f


try:
    pipe.run_groups(make_groups(), fn_phases([LAYERS, LAYERS - 2]), pipelined=True)
    torch.cuda.synchronize()
    check(True, "(5) group 1 with fewer phases than group 0 runs through")
except Exception as exc:  # noqa: BLE001
    check(False, f"(5) group 1 with fewer phases raised: {exc!r}")
try:
    pipe.run_groups(make_groups(), fn_phases([LAYERS - 2, LAYERS]), pipelined=True)
    torch.cuda.synchronize()
    check(False, "(5) group 1 with more phases than group 0 must raise (dependency order)")
except RuntimeError as exc:
    check("dependency order violated" in str(exc.__cause__), f"(5) group 1 with more phases raises: {str(exc.__cause__)[:90]}")
check(fc._forward_context == "main" and pipe.current_group() is None, "(5) state restored after the error")


# ------------------------------------------------------------------------------------------ (5b): yield points in a tail
def fn_tail(g):
    """Both groups: LAYERS phases, then group 0 a tail of 5 yield points (the replay layers) - must interleave with
    group 1's remaining phases without disturbing them."""
    x = inputs[g.idx]
    for layer in range(LAYERS):
        pipe.phase_begin()
        y = torch.tanh(x @ W[layer])
        pipe.phase_end()
        x = torch.tanh(y @ W[layer].t())
        with lock:
            order.append(("compute", g.idx, layer))
    if g.idx == 0:
        # the replay section: its own stream R, under an overridden forward context (the replay batch's); a handoff
        # inside must bring the override and R back, not the group's context / the compute stream
        pipe.replay_begin(x)
        if current_stream() != runner.replay_stream or torch.cuda.current_stream() != runner.replay_stream:
            seen["replay_stream_ok"] = False
        with fc.override_forward_context("replay0"):
            for i in range(5):
                pipe.yield_point()
                if fc._forward_context != "replay0":
                    seen["replay_ctx_ok"] = False
                if current_stream() != runner.replay_stream:
                    seen["replay_stream_ok"] = False
                x = torch.tanh(x @ W[i])  # on R
                with lock:
                    order.append(("tail", 0, i))
        pipe.replay_end(x)
        if fc._forward_context != g.forward_context or current_stream() != C:
            seen["replay_ctx_ok"] = False
        x = torch.tanh(x @ W[5])  # on C, after the wait for R
    return x


order.clear()
os.environ["VLLM_MOET_PREFILL_PIPE_REPLAY"] = "stream"
try:
    outs = pipe.run_groups(make_groups(), fn_tail, pipelined=True)
    torch.cuda.synchronize()
    ref_tail = ref.clone()
    xt = ref[:4]
    for i in range(5):
        xt = torch.tanh(xt @ W[i])
    ref_tail[:4] = torch.tanh(xt @ W[5])
    check(torch.equal(pipe.merge_outputs(outs), ref_tail), "(5b) the replay section (on R, with handoffs inside) computes the reference")
    tails = [i for i, o in enumerate(order) if o[0] == "tail"]
    check(len(tails) == 5 and tails[0] < len(order) - 1, f"(5b) the tail's yield points interleave with group 1 (first tail at index {tails[0]} of {len(order)})")
    check(seen.get("replay_ctx_ok", True), "(5b) a handoff under an overridden forward context (the replay batch's) brings the override back")
    check(seen.get("replay_stream_ok", True), "(5b) the replay section runs on its own stream R, through the handoffs inside")
except Exception as exc:  # noqa: BLE001
    check(False, f"(5b) yield points raised: {exc!r}")


# serial mode (the default): replay_begin / yield_point / replay_end are no-ops - the tail stays on C, issued whole
os.environ["VLLM_MOET_PREFILL_PIPE_REPLAY"] = "serial"
order.clear()
seen["replay_stream_ok"] = True


def fn_tail_serial(g):
    x = inputs[g.idx]
    for layer in range(LAYERS):
        pipe.phase_begin()
        y = torch.tanh(x @ W[layer])
        pipe.phase_end()
        x = torch.tanh(y @ W[layer].t())
    if g.idx == 0:
        pipe.replay_begin(x)
        if current_stream() != C:
            seen["replay_stream_ok"] = False
        for i in range(5):
            pipe.yield_point()
            x = torch.tanh(x @ W[i])
            with lock:
                order.append(("tail", 0, i))
        pipe.replay_end(x)
        x = torch.tanh(x @ W[5])
    return x


outs = pipe.run_groups(make_groups(), fn_tail_serial, pipelined=True)
torch.cuda.synchronize()
check(torch.equal(pipe.merge_outputs(outs), ref_tail) and seen["replay_stream_ok"], "(5c) REPLAY=serial (default): the replay hooks are no-ops, the tail on C, results unchanged")
tails = [i for i, o in enumerate(order) if o[0] == "tail"]
check(tails == list(range(len(order) - 5, len(order))), "(5c) REPLAY=serial: the tail is issued whole (no handoff inside)")
os.environ.pop("VLLM_MOET_PREFILL_PIPE_REPLAY", None)


# ------------------------------------------------------------------------------------------ (5d): the tail gate
# VLLM_MOET_PREFILL_PIPE_TAIL=barrier|sync: group 0's tail (replay_begin) makes the compute stream wait for group 1's
# last phase (recorded on the comm stream); group 1's gate waits for nothing; serial = no gate. Observable: a slow
# kernel in group 1's last phase on N must have finished before group 0's tail kernel on C runs.
gate_seen = {"a_waited": None, "b_gate_noop": True, "a_sync_drained": None}


def _busy(x, n):
    w = torch.randn(x.shape[1], x.shape[1], device=dev) * 0.01
    for _ in range(n):  # a few ms of GPU work
        x = torch.tanh(x @ w)
    return x


def fn_gate(g):
    x = inputs[g.idx]
    marker = torch.zeros(1, device=dev)  # written on N inside group 1's last phase
    for layer in range(LAYERS):
        pipe.phase_begin()
        y = torch.tanh(x @ W[layer])
        if g.idx == 1 and layer == LAYERS - 1:
            _busy(torch.randn(2048, 2048, device=dev), 40)
            marker.fill_(1.0)  # on N, at the end of the slow phase
            gate_seen["marker"] = marker
        pipe.phase_end()
        x = torch.tanh(y @ W[layer].t())
    if g.idx == 0:
        pipe.replay_begin(x)  # the gate
        if pipe.tail_mode() == "sync":
            gate_seen["a_sync_drained"] = torch.cuda.current_stream().query()  # drained: nothing pending on C
        probe = gate_seen["marker"].clone() if "marker" in gate_seen else None  # on C: 1.0 iff N's phase was done
        if probe is not None:
            gate_seen["a_waited"] = probe
        pipe.replay_end(x)
    else:
        ev_before = torch.cuda.Event()
        ev_before.record(C)
        pipe.replay_begin(x)  # group 1: the last group waits for nothing
        pipe.replay_end(x)
    return x


for mode in ("serial", "barrier", "sync"):
    os.environ["VLLM_MOET_PREFILL_PIPE_TAIL"] = mode
    gate_seen = {"a_waited": None, "b_gate_noop": True, "a_sync_drained": None}
    outs = pipe.run_groups(make_groups(), fn_gate, pipelined=True)
    torch.cuda.synchronize()
    ok_vals = torch.equal(pipe.merge_outputs(outs), ref)
    waited = gate_seen["a_waited"]
    waited_v = None if waited is None else float(waited.item())
    if mode == "serial":
        check(ok_vals and pipe.tail_mode() == "serial", "(5d) TAIL=serial: no gate, results unchanged")
    else:
        check(ok_vals and waited_v == 1.0, f"(5d) TAIL={mode}: group 0's tail on C runs after group 1's last phase on N (marker {waited_v})")
        if mode == "sync":
            check(gate_seen["a_sync_drained"] is True, "(5d) TAIL=sync: the compute stream is drained when group 0's tail is issued")
os.environ.pop("VLLM_MOET_PREFILL_PIPE_TAIL", None)
check(pipe.tail_mode() == "serial", "(5d) TAIL default is serial")


# ------------------------------------------------------------------------------------------ (5e): the mixed-step dump entry
# dump_layer of a group with decode rows ahead of the prefill rows writes the whole-tensor entry and a "prefill_rows"
# entry whose hashes cover rows [from, n) only (and match a direct hash of those rows); without decode rows one entry.
import hashlib
import json
import tempfile

dump_dir = tempfile.mkdtemp()
open(os.path.join(dump_dir, "ENABLE"), "w").close()
pipe._DUMP_DIR = dump_dir
g_mixed = pipe.PipeGroup(idx=0, token_offset=0, num_tokens=8, forward_context=None, slot=1, prefill_rows_from=3)
pos = torch.arange(100, 108, device=dev)
xt = torch.randn(8, 4, device=dev)
shard = torch.randn(8, 2, device=dev)  # tp = 1 here: a full-row tensor too
pipe._tls.group = g_mixed
try:
    pipe.dump_layer(7, pos, x=xt, residual=shard)
finally:
    pipe._tls.group = None
lines = [json.loads(l) for l in open(os.path.join(dump_dir, "rank0.jsonl")).read().splitlines() if l.strip()]
sub = [e for e in lines if e.get("kind") == "prefill_rows"]
whole = [e for e in lines if e.get("kind") != "prefill_rows"]
ok_sub = (
    len(sub) == 1 and len(whole) == 1 and sub[0]["rows"] == 5 and sub[0]["pos0"] == 103 and sub[0]["from"] == 3
    and sub[0]["x"] == hashlib.sha256(xt[3:].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:24]
    and sub[0]["residual"] == hashlib.sha256(shard[3:].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:24]
    and whole[0]["rows"] == 8 and whole[0]["pos0"] == 100
)
check(ok_sub, f"(5e) dump_layer of a mixed group writes the whole entry + a prefill_rows entry (rows {sub[0]['rows'] if sub else None} from {sub[0]['from'] if sub else None})")
g_plain = pipe.PipeGroup(idx=1, token_offset=8, num_tokens=8, forward_context=None, slot=2)
pipe._tls.group = g_plain
try:
    pipe.dump_layer(7, pos, x=xt)
finally:
    pipe._tls.group = None
lines = [json.loads(l) for l in open(os.path.join(dump_dir, "rank0.jsonl")).read().splitlines() if l.strip()]
check(len(lines) == 3 and not [e for e in lines[2:] if e.get("kind") == "prefill_rows"], "(5e) a group without decode rows writes the whole entry only")
pipe._DUMP_DIR = ""
os.environ["VLLM_MOET_PREFILL_PIPE_FORCE_SEQUENTIAL"] = "1"
order.clear()
outs = pipe.run_groups(make_groups(), fn, pipelined=True)
torch.cuda.synchronize()
check(torch.equal(pipe.merge_outputs(outs), ref) and order == [("compute", 0, l) for l in range(LAYERS)] + [("compute", 1, l) for l in range(LAYERS)],
      "(5e) FORCE_SEQUENTIAL=1 runs the groups one after the other with the reference results")
os.environ.pop("VLLM_MOET_PREFILL_PIPE_FORCE_SEQUENTIAL", None)


# ------------------------------------------------------------------------------------------ (6): a failing group
def fn_fail(g):
    pipe.phase_begin()
    pipe.phase_end()
    if g.idx == 1:
        raise ValueError("boom")
    pipe.phase_begin()
    pipe.phase_end()
    return inputs[g.idx]


t0 = time.time()
try:
    pipe.run_groups(make_groups(), fn_fail, pipelined=True)
    check(False, "(6) a failing group must raise")
except RuntimeError as exc:
    check(isinstance(exc.__cause__, ValueError) and time.time() - t0 < 60, f"(6) a failing group raises on the runner thread without hanging ({str(exc)[:60]})")

# merge of (tensor, [aux...]) tuples
a = (torch.ones(2, 3), [torch.zeros(2, 1), torch.zeros(2, 2)])
b = (torch.ones(3, 3) * 2, [torch.ones(3, 1), torch.ones(3, 2)])
m = pipe.merge_outputs([a, b])
check(isinstance(m, tuple) and m[0].shape == (5, 3) and isinstance(m[1], list) and m[1][1].shape == (5, 2) and m[1][0][2].item() == 1, "merge_outputs: (hidden, [aux]) tuples concatenate per element")


# ------------------------------------------------------------------------------------------ (7): the communicator switch (COMM2)
# CommSwitch resolves every attribute on the calling thread's group's communicator: odd pipelined groups -> alt, the even
# group, the runner thread and sequential groups -> base; a call that spans several attribute lookups stays on one side.
class FakeComm:
    def __init__(self, tag):
        self.tag = tag
        self.disabled = False
        self.calls = []

    def all_gather(self, out, inp):
        self.calls.append(("all_gather", self.tag))
        return self.tag


sw = pipe.CommSwitch(FakeComm("base"), FakeComm("alt"))
seen_comm = {}


def fn_comm(g: pipe.PipeGroup):
    seen_comm[g.idx] = (sw.tag, sw.all_gather(None, None), sw.disabled)
    pipe.phase_begin()
    pipe.phase_end()
    return inputs[g.idx]


pipe.run_groups(make_groups(), fn_comm, pipelined=True)
torch.cuda.synchronize()
check(seen_comm.get(0) == ("base", "base", False) and seen_comm.get(1) == ("alt", "alt", False) and sw.tag == "base",
      f"(7) CommSwitch: group 0 -> base, group 1 -> alt, the runner thread -> base ({seen_comm})")
seen_comm.clear()
pipe.run_groups(make_groups(), fn_comm, pipelined=False)
check(seen_comm.get(0) == ("base", "base", False) and seen_comm.get(1) == ("base", "base", False),
      f"(7) CommSwitch: sequential groups stay on base ({seen_comm})")
check(not pipe.comm2_enabled() and pipe.kv_connector_policy() == "allow", "(7) COMM2 off and KV_CONNECTOR=allow by default")
os.environ["VLLM_MOET_PREFILL_PIPE_KV_CONNECTOR"] = "refuse"


class FakeCfg:
    kv_transfer_config = object()


class FakeRunnerCfg:
    vllm_config = FakeCfg()


os.environ["VLLM_MOET_PREFILL_PIPE"] = "2"
saved_runner = pipe._RUNNER
pipe._RUNNER = None
res = pipe.plan_groups(FakeRunnerCfg(), None, None, None, None)
check(res is None and pipe._RUNNER is None and pipe._KV_CONNECTOR_REFUSED["logged"],
      "(7) KV_CONNECTOR=refuse with a connector configured: plan_groups returns None (today's path) without building the runner, logged once")
pipe._RUNNER = saved_runner
os.environ.pop("VLLM_MOET_PREFILL_PIPE_KV_CONNECTOR", None)
os.environ.pop("VLLM_MOET_PREFILL_PIPE", None)

# ------------------------------------------------------------------------------------------ (8): K-7 - thread placement, launch-order bundle
# thread_cpus: distinct CPUs from the top of the allowed set per (rank, group); _place_thread pins the group threads of a
# pipelined step (sched_getaffinity on the thread = its CPU) and leaves the runner thread as it was; ORDER bundles.
cpus_by_rank = [pipe.thread_cpus(2, allowed=list(range(120)), rank=r) for r in range(4)]
flat = [c for d in cpus_by_rank for c in d.values()]
check(len(set(flat)) == 8 and min(flat) == 112 and max(flat) == 119 and cpus_by_rank[0] == {0: 119, 1: 118},
      f"(8) thread_cpus: 4 ranks x 2 groups -> 8 distinct CPUs 112-119 from the top of the allowed set ({cpus_by_rank})")
check(pipe.thread_cpus(2, allowed=[5, 7], rank=3) == {0: 7, 1: 5}, "(8) thread_cpus wraps on a small allowed set")
check(not pipe.affinity_policy() and pipe.order() == "today" and pipe._comm_priority() == 0,
      "(8) AFFINITY unset, ORDER=today, comm priority 0 by default")
main_aff = os.sched_getaffinity(0)
os.environ["VLLM_MOET_PREFILL_PIPE_AFFINITY"] = "pin,bogus"
pipe._PLACEMENT.update({"cpus": None, "warned": set(), "fifo_ok": None})
seen_aff = {}


def fn_aff(g: pipe.PipeGroup):
    seen_aff[g.idx] = (os.sched_getaffinity(0), threading.current_thread().name)
    pipe.phase_begin()
    pipe.phase_end()
    return inputs[g.idx]


pipe.run_groups(make_groups(), fn_aff, pipelined=True)
torch.cuda.synchronize()
want = pipe.thread_cpus(2, allowed=sorted(main_aff), rank=0)
check(pipe.affinity_policy() == frozenset({"pin"}) and seen_aff.get(0, (None,))[0] == {want[0]}
      and seen_aff.get(1, (None,))[0] == {want[1]} and os.sched_getaffinity(0) == main_aff,
      f"(8) AFFINITY=pin: group threads pinned to {want}, the runner thread keeps its {len(main_aff)} CPUs ({seen_aff})")
os.environ["VLLM_MOET_PREFILL_PIPE_AFFINITY"] = "pin,fifo"
pipe._PLACEMENT.update({"cpus": None, "warned": set(), "fifo_ok": None})
pipe.run_groups(make_groups(), fn_aff, pipelined=True)
torch.cuda.synchronize()
check(pipe._PLACEMENT["fifo_ok"] in (True, False) and seen_aff.get(0, (None,))[0] == {want[0]},
      f"(8) AFFINITY=pin,fifo: SCHED_FIFO {'granted' if pipe._PLACEMENT['fifo_ok'] else 'refused (no CAP_SYS_NICE) - logged once, pinning kept'}")
os.environ.pop("VLLM_MOET_PREFILL_PIPE_AFFINITY", None)
pipe._PLACEMENT.update({"cpus": None, "warned": set(), "fifo_ok": None})
os.environ["VLLM_MOET_PREFILL_PIPE_ORDER"] = "comm2"
check(pipe.order() == "comm2" and pipe.comm2_enabled() and pipe._comm_priority() == 0, "(8) ORDER=comm2 -> COMM2 on, priority 0")
os.environ["VLLM_MOET_PREFILL_PIPE_ORDER"] = "comm2prio"
check(pipe.order() == "comm2prio" and pipe.comm2_enabled() and pipe._comm_priority() == -1, "(8) ORDER=comm2prio -> COMM2 on, priority -1")
os.environ["VLLM_MOET_PREFILL_PIPE_ORDER"] = "nonsense"
check(pipe.order() == "today" and not pipe.comm2_enabled(), "(8) an unknown ORDER value is today's")
os.environ.pop("VLLM_MOET_PREFILL_PIPE_ORDER", None)

# ------------------------------------------------------------------------------------------ (9): K-8 - the phase graph
# The step graph protocol on fakes: a GEMM chain per group with phases across the compute / comm streams is captured by
# _phase_capture_step (the two threads under one stream capture), replayed by run_groups on new inputs (static inputs,
# metadata copies, outputs copied out), verified against the eager section (verify mode), and refused for a step whose key
# or metadata differs (eager fallback, counted). Plus the metadata snapshot's rules and record_for_comm's hold-until-join.
from dataclasses import dataclass as _dc  # noqa: E402

check(pipe.phase_graphs() == "0" and pipe.phase_graph_max_ctx() == 262144 and not pipe.phase_graph_capturing(),
      "(9) PHASE_GRAPHS=0 and MAX_CTX=262144 by default, not capturing")
os.environ["VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS"] = "bogus"
check(pipe.phase_graphs() == "0", "(9) an unknown PHASE_GRAPHS value is off")


@_dc
class FakeChunk:
    cu_seqlen_ke: torch.Tensor
    max_seqlen_k: int
    token_start: int


@_dc
class FakeMeta:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    num_prefills: int
    chunks: list


@_dc
class FakeCtx:
    attn_metadata: dict
    slot_mapping: dict
    is_padding: torch.Tensor | None = None
    batch_descriptor: tuple | None = None


persistent_bt = torch.arange(64, device=dev, dtype=torch.int32).view(1, 64)
persistent_sl = torch.zeros(1, dtype=torch.int32, device=dev)
persistent_slots = torch.zeros(2, 8, dtype=torch.int64, device=dev)


def make_ctx(ctx_len: int, fresh_ke: torch.Tensor | None = None, extra: bool = False, prefills: int = 1):
    meta = FakeMeta(
        block_table=persistent_bt, seq_lens=persistent_sl, seq_lens_cpu=torch.tensor([ctx_len]), num_prefills=prefills,
        chunks=[FakeChunk(torch.full((4,), ctx_len, dtype=torch.int32, device=dev) if fresh_ke is None else fresh_ke,
                          ctx_len // 2, 0)],
    )
    attn = {"layers.0": meta}
    if extra:  # a distinct object: the walk dedups shared objects by identity
        attn["layers.1"] = FakeMeta(persistent_bt, persistent_sl, torch.tensor([ctx_len]), prefills, [])
    return FakeCtx(attn_metadata=attn, slot_mapping={"layers.0": persistent_slots[:, :4]}, batch_descriptor=(4,))


cap_ctx = make_ctx(1000)
snap = pipe._MetaSnapshot(cap_ctx)
check(len(snap.tensors) == 4 and "attn_metadata.layers.0.chunks[0].cu_seqlen_ke" in snap.tensors
      and snap.values.get("attn_metadata.layers.0.chunks[0].max_seqlen_k") == 500
      and snap.values.get("attn_metadata.layers.0.seq_lens_cpu") == [1000],
      f"(9) _MetaSnapshot flattens device tensors by path and host values (cpu tensors as lists): {sorted(snap.tensors)}")
plan, why = snap.plan(make_ctx(600))
check(why is None and len(plan) == 1 and plan[0][0].data_ptr() == snap.tensors["attn_metadata.layers.0.chunks[0].cu_seqlen_ke"].data_ptr(),
      f"(9) a smaller context passes (extents <=), persistent buffers are not copied, the fresh tensor is ({why})")
plan, why = snap.plan(make_ctx(1200))
check(plan is None and "extent" in (why or ""), f"(9) a larger context than the capture's is refused: {why}")
plan, why = snap.plan(make_ctx(600, prefills=2))
check(plan is None and "num_prefills" in (why or ""), f"(9) a different host value is refused: {why}")
plan, why = snap.plan(make_ctx(600, extra=True))
check(plan is None and "structure" in (why or ""), f"(9) a different metadata structure is refused: {why}")
plan, why = snap.plan(make_ctx(600, fresh_ke=torch.zeros(3, dtype=torch.int32, device=dev)))
check(why is None and plan[0][0].shape[0] == 3, f"(9) a shorter fresh tensor copies into a prefix of the captured one ({why})")
plan, why = snap.plan(make_ctx(600, fresh_ke=torch.zeros(5, dtype=torch.int32, device=dev)))
check(plan is None and "shape" in (why or ""), f"(9) a longer fresh tensor is refused: {why}")

# the protocol on the GPU: capture, replay, verify, fallback
WG = [torch.randn(H, H, device=dev) * 0.05 for _ in range(3)]
held_seen = {"in_phase": None, "after": None}


def section_fn(g: pipe.PipeGroup, hs, pos, ids, hashes, mask):
    x = hs
    for layer in range(3):
        pipe.phase_begin()
        y = torch.tanh(x @ WG[layer])
        pipe.record_for_comm(x)
        if layer == 0 and g.idx == 0:
            held_seen["in_phase"] = len(g.comm_held)
        pipe.phase_end()
        if layer == 0 and g.idx == 0:
            held_seen["after"] = len(g.comm_held)
        x = torch.tanh(y @ WG[layer].t())
    return pipe.SectionState(x, x * 2, x[:, :4].float(), x[:, :2].float(), x[:, :1].float(), {2: x + 1}, True, ids)


def rest_fn(g: pipe.PipeGroup, pos, ids, state: pipe.SectionState):
    return state.hidden_states + state.residual + state.aux_hidden_by_layer[2]


def section_reference(hs):
    x = hs
    for layer in range(3):
        x = torch.tanh(torch.tanh(x @ WG[layer]) @ WG[layer].t())
    return x + x * 2 + (x + 1)


class FakeReplayLayers:
    _moet_replay_batches = {1: object(), 2: object()}


class FakeModelState:
    decoder_replay_layers = FakeReplayLayers()


runner.model_state = FakeModelState()
runner.phase_graph = None
runner.phase_graph_failed = None
runner.engram_modules = []
runner._warned = set()
runner._warn_once = lambda key, msg, *a: print("  warn-once:", key, msg % a)
positions_buf = torch.arange(8, device=dev, dtype=torch.int64)
hs_steps = [torch.randn(8, H, device=dev) for _ in range(3)]
cur = {"hs": hs_steps[0]}


def inputs_fn(g: pipe.PipeGroup):
    lo, hi = g.token_offset, g.token_offset + g.num_tokens
    return (cur["hs"][lo:hi], positions_buf[lo:hi], None, None, None)


hooks = pipe.PhaseHooks(inputs_fn, section_fn, rest_fn)


def make_graph_groups(ctx_len=1000, rows=4):
    gs = make_groups(rows=rows)
    for g in gs:
        g.forward_context = make_ctx(ctx_len)
    return gs


full_fn = lambda g: rest_fn(g, None, None, section_fn(g, *inputs_fn(g)))  # noqa: E731 - the eager tail
os.environ["VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS"] = "1"
capture_stream = torch.cuda.Stream(device=dev)  # a capture cannot begin on the default stream (PipeRunner.capture_stream)
capture_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(capture_stream):
    pipe._PHASE["request"] = "warmup"
    out_w = pipe.run_groups(make_graph_groups(), full_fn, pipelined=True, phase=hooks)
    pipe._PHASE["request"] = "capture"
    t0 = time.time()
    out_c = pipe.run_groups(make_graph_groups(), full_fn, pipelined=True, phase=hooks)
    pipe._PHASE["request"] = None
torch.cuda.synchronize()
pg = runner.phase_graph
check(pg is not None and pg.graph is not None and len(pg.inputs) == 2 and len(pg.outputs) == 2 and not pipe.phase_graph_capturing(),
      f"(9) capture pass: a step graph with static inputs / outputs for both groups ({(time.time() - t0) * 1000:.0f} ms)")
check(torch.equal(pipe.merge_outputs(out_c), section_reference(hs_steps[0])),
      "(9) the capture pass's output (one replay + the eager rest) equals the eager reference")
check(held_seen == {"in_phase": 1, "after": 0}, f"(9) record_for_comm holds the tensor inside the phase and releases it at the join {held_seen}")
steps0 = dict(pipe.stats())
cur["hs"] = hs_steps[1]
out_g = pipe.run_groups(make_graph_groups(ctx_len=700), full_fn, pipelined=True, phase=hooks)
torch.cuda.synchronize()
st = pipe.stats()
check(torch.equal(pipe.merge_outputs(out_g), section_reference(hs_steps[1])) and pg.num_launches == 1
      and st["phase_graph_steps"] == steps0["phase_graph_steps"] + 1 and st["phase_graph_eager"] == steps0["phase_graph_eager"],
      f"(9) a step on the graph: new inputs copied in, one launch, outputs equal the eager reference ({pg.num_launches} launches)")
os.environ["VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS"] = "verify"
cur["hs"] = hs_steps[2]
out_v = pipe.run_groups(make_graph_groups(ctx_len=900), full_fn, pipelined=True, phase=hooks)
torch.cuda.synchronize()
st = pipe.stats()
check(torch.equal(pipe.merge_outputs(out_v), section_reference(hs_steps[2])) and st["phase_graph_verified"] == 1
      and st["phase_graph_mismatch"] == 0 and pg.num_launches == 2,
      f"(9) verify mode: graph and eager section compared (verified {st['phase_graph_verified']}, mismatch {st['phase_graph_mismatch']})")
# a planted difference is reported and counted, the tails continue from the eager outputs
saved_section = hooks.section


def section_planted(g, hs, pos, ids, hashes, mask):
    s = saved_section(g, hs, pos, ids, hashes, mask)
    s.residual.add_(1.0)  # the eager rerun differs from the graph by construction
    return s


hooks.section = section_planted
out_p = pipe.run_groups(make_graph_groups(ctx_len=900), full_fn, pipelined=True, phase=hooks)
torch.cuda.synchronize()
st = pipe.stats()
hooks.section = saved_section
def section_reference_planted(hs):  # the rest's own evaluation order: hidden + (residual + 1) + aux
    x = hs
    for layer in range(3):
        x = torch.tanh(torch.tanh(x @ WG[layer]) @ WG[layer].t())
    return x + (x * 2 + 1.0) + (x + 1)


check(st["phase_graph_mismatch"] == 1 and pg.num_launches == 3
      and torch.equal(pipe.merge_outputs(out_p), section_reference_planted(hs_steps[2])),
      f"(9) verify mode reports a mismatch (count {st['phase_graph_mismatch']}) and the tails use the eager outputs")
os.environ["VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS"] = "1"
# fallbacks: a context beyond the capture's, a different group shape, a missing replay batch -> eager, counted, loud once
eager0 = pipe.stats()["phase_graph_eager"]
out_e = pipe.run_groups(make_graph_groups(ctx_len=1500), full_fn, pipelined=True, phase=hooks)
gs = make_graph_groups()
gs[1].req_lo = 1
out_e2 = pipe.run_groups(gs, full_fn, pipelined=True, phase=hooks)
FakeReplayLayers._moet_replay_batches = {1: object()}
out_e3 = pipe.run_groups(make_graph_groups(), full_fn, pipelined=True, phase=hooks)
FakeReplayLayers._moet_replay_batches = {1: object(), 2: object()}
torch.cuda.synchronize()
check(pipe.stats()["phase_graph_eager"] == eager0 + 3 and pg.num_launches == 3
      and all(torch.equal(pipe.merge_outputs(o), section_reference(hs_steps[2])) for o in (out_e, out_e2, out_e3)),
      "(9) a step the graph does not serve (context, request slice, replay batch) runs eagerly with today's results")
os.environ["VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS"] = "0"
out_off = pipe.run_groups(make_graph_groups(), full_fn, pipelined=True, phase=hooks)
check(pg.num_launches == 3 and torch.equal(pipe.merge_outputs(out_off), section_reference(hs_steps[2])),
      "(9) PHASE_GRAPHS=0: the graph is never launched, today's path")
os.environ.pop("VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS", None)
pg.release()
runner.phase_graph = None


# ------------------------------------------------------------------------------------------ (10): K-9 - the shared replay tail
# Two groups, each with a replay batch of 2 rows out of 4, and a shared batch of 4 rows. The fake replay layers contain one op
# whose result depends on the row count unless it is issued per group row block (the all-reduce / cuBLAS analogue, through
# pipe.row_blocks) and read the mHC pin the way the layer does (_moet_sp_pin(None) -> pipe.pin_tokens). Checked: the tail runs
# once, on the second thread to arrive, under the shared batch's forward context; the first thread gets its rows of the outputs;
# the results equal today's per-group tails bit for bit (eager and through a fake graph manager); verify mode compares and reports
# a planted difference; the dump hook writes one entry per group; the fallbacks (no plan, a partner that never arrives).
import tempfile  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from vllm.forward_context import get_forward_context  # noqa: E402

R_ROWS, R_H, R_HC = 4, 16, 2
W_T = torch.randn(R_H, R_H, device=dev) * 0.1
tail_seen = {"calls": 0, "thread_groups": [], "ctx": [], "pins": [], "M": []}
tail_split = {"on": True}  # the fake all-reduce honours row_blocks (the shared tail's split) when on


def fake_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Row-count dependent on purpose: x * M for a message of M rows (NCCL's ring chunking stands in for this)."""
    blocks = pipe.row_blocks(x.shape[0]) if tail_split["on"] else [(0, x.shape[0])]
    return torch.cat([x[lo:hi] * float(hi - lo) for lo, hi in blocks], dim=0)


def fake_run_layers(h, pos, ids, pre, post, res, residual):
    tail_seen["calls"] += 1
    g = pipe.current_group()
    tail_seen["thread_groups"].append(None if g is None else g.idx)
    tail_seen["ctx"].append(fc._forward_context)
    tail_seen["M"].append(int(h.shape[0]))
    mhc_warmup._MOET_SP_PIN_TOKENS = pipe.pin_tokens(None)  # the layer's _moet_sp_pin(None) at a replay layer's start
    tail_seen["pins"].append(mhc_warmup._MOET_SP_PIN_TOKENS)
    x = fake_all_reduce(h)
    y = torch.tanh(x @ W_T) + residual[:, 0, :] * pre[:, :1].to(h.dtype) + pos.to(h.dtype)[:, None] * 1e-3
    mhc_warmup._MOET_SP_PIN_TOKENS = pipe.pin_tokens(None)  # the layer's end
    return y, pre * 2.0 + post[:, :, 0], y.mean(dim=1)


class FakeBatch:
    def __init__(self, rows, ctx, run_graph=None):
        self.rows = rows
        self.forward_context = ctx
        self.run_graph = run_graph


class FakeReplay:
    def __init__(self):
        self._moet_replay_batches = {}
        self.run_layers = fake_run_layers


def tail_inputs(g: pipe.PipeGroup, seed: int):
    gen = torch.Generator(device=dev).manual_seed(seed * 10 + g.idx)
    n = g.num_tokens
    return (
        torch.randn(n, R_H, device=dev, generator=gen),
        torch.arange(g.token_offset, g.token_offset + n, device=dev, dtype=torch.int64),
        torch.arange(n, device=dev, dtype=torch.int32),
        torch.randn(n, R_HC, device=dev, generator=gen),
        torch.randn(n, R_HC, 1, device=dev, generator=gen),
        torch.randn(n, R_HC, R_HC, device=dev, generator=gen),
        torch.randn(n, R_HC, R_H, device=dev, generator=gen),
    )


replay = FakeReplay()
rows_g = [torch.tensor([1, 3], device=dev), torch.tensor([0, 2], device=dev)]  # each group's replay rows (local)
shared_ctx = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=4), tag="shared")


def set_batches(shared_run_graph=None):
    replay._moet_replay_batches = {
        1: FakeBatch(rows_g[0], SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=2), tag="g0")),
        2: FakeBatch(rows_g[1], SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=2), tag="g1")),
        pipe.SHARED_SLOT: FakeBatch(torch.tensor([1, 3, 4, 6], device=dev), shared_ctx, shared_run_graph),
    }


def make_shared_groups(plan):
    gs = make_groups(rows=R_ROWS)
    for g in gs:
        g.shared_plan = plan
    return gs


def fn_shared(g: pipe.PipeGroup):
    """A group's tail: two phases, then the replay through the rendezvous (the model's _moet_sp_decoder_replay_rest), then
    a per-group op on the full rows (the collapse-norm analogue)."""
    inp = tail_inputs(g, cur_seed["s"])
    x = inp[0]
    for layer in range(2):
        pipe.phase_begin()
        y = torch.tanh(x @ W_T)
        pipe.phase_end()
        x = torch.tanh(y @ W_T.t())
    inp = (x,) + inp[1:]
    out = pipe.shared_replay(replay, *inp)
    if out is None:
        out = pipe._run_replay_with_batch(replay, replay._moet_replay_batches[g.slot], inp)
    return out[0] * 2.0 + out[1][:, :1].to(out[0].dtype) + out[2][:, None]


def reference_today(seed: int):
    """Today's two tails, one per group (the fake layers at M = 2 each), on the runner thread."""
    outs = []
    for g in make_groups(rows=R_ROWS):
        inp = tail_inputs(g, seed)
        x = inp[0]
        for layer in range(2):
            x = torch.tanh(torch.tanh(x @ W_T) @ W_T.t())
        inp = (x,) + inp[1:]
        with fc.override_forward_context(replay._moet_replay_batches[g.slot].forward_context):
            out = pipe._run_replay_with_batch(replay, replay._moet_replay_batches[g.slot], inp)
        outs.append(out[0] * 2.0 + out[1][:, :1].to(out[0].dtype) + out[2][:, None])
    return torch.cat(outs, dim=0)


cur_seed = {"s": 1}
set_batches()
runner.model_state = SimpleNamespace(_moet_pipe_slot_cudagraphs={})
fc._forward_context = "main"
mhc_warmup._MOET_SP_PIN_TOKENS = None
ref10 = reference_today(1)
tail_seen = {"calls": 0, "thread_groups": [], "ctx": [], "pins": [], "M": []}
plan = pipe.SharedPlan(counts=[2, 2], pin=2, verify=False)
st0 = dict(pipe.stats())
outs = pipe.run_groups(make_shared_groups(plan), fn_shared, pipelined=True)
torch.cuda.synchronize()
check(torch.equal(pipe.merge_outputs(outs), ref10), "(10) the shared tail's per-group outputs equal today's two tails bit for bit (eager; the row-count-dependent op split per group)")
check(tail_seen["calls"] == 1 and tail_seen["M"] == [4], f"(10) the replay layers ran once, on 4 rows ({tail_seen['calls']} calls, M {tail_seen['M']})")
check(tail_seen["thread_groups"] == [1], f"(10) the tail ran on the second thread to arrive (group {tail_seen['thread_groups']}); group 0 parked at the rendezvous")
check(all(getattr(c, "tag", None) == "shared" for c in tail_seen["ctx"]), "(10) the tail ran under the shared batch's forward context")
check(tail_seen["pins"] == [2], f"(10) inside the shared tail the mHC pin is the group's row count ({tail_seen['pins']})")
check(mhc_warmup._MOET_SP_PIN_TOKENS is None and fc._forward_context == "main" and pipe.current_shared() is None,
      "(10) after the step: pin released, the runner's context back, no shared scope")
check(pipe.stats()["shared_tail_steps"] == st0["shared_tail_steps"] + 1, "(10) the step is counted as a shared-tail step")
# the split itself: the all-reduce stand-in splits a tensor of the tail's row count per group inside the scope, nothing outside
st_obj = pipe.SharedTail(plan, make_shared_groups(plan), [None, None], [None, None])
w_ar = pipe._split_all_reduce(lambda t: t * float(t.shape[0]))
x4 = torch.randn(4, 3, device=dev)
with pipe._SharedScope(st_obj):
    inside = w_ar(x4)
    blocks_in = pipe.row_blocks(4)
    other = pipe.row_blocks(3)
outside = w_ar(x4)
check(torch.equal(inside, x4 * 2.0) and torch.equal(outside, x4 * 4.0) and blocks_in == [(0, 2), (2, 4)] and other == [(0, 3)],
      "(10) _split_all_reduce: one call per group row block inside the scope (a 4-row message = 2 x 2 rows), one call outside; other row counts untouched")
# through a fake graph manager: the inputs land in its static buffers per group, the layers run on the padded buffers
bufs = [torch.zeros(4, *t.shape[1:], dtype=t.dtype, device=dev) for t in tail_inputs(make_groups(rows=R_ROWS)[0], 1)]
mgr = SimpleNamespace(input_buffers=bufs, breakable_cg_runner=lambda *b: fake_run_layers(*b))
runner.model_state = SimpleNamespace(_moet_pipe_slot_cudagraphs={pipe.SHARED_SLOT: mgr})
set_batches(shared_run_graph=lambda rows, inputs: (_ for _ in ()).throw(AssertionError("the manager's run must not be used")))
tail_seen = {"calls": 0, "thread_groups": [], "ctx": [], "pins": [], "M": []}
outs_g = pipe.run_groups(make_shared_groups(plan), fn_shared, pipelined=True)
torch.cuda.synchronize()
check(torch.equal(pipe.merge_outputs(outs_g), ref10) and tail_seen["calls"] == 1 and tail_seen["M"] == [4],
      "(10) through the shared slot's graph manager (static input buffers, breakable_cg_runner): the same outputs, one run")
set_batches()
# verify mode: today's tails run afterwards and are compared; a planted row-count dependence is reported, today's outputs win
cur_seed["s"] = 2
ref10b = reference_today(2)
v0 = dict(pipe.stats())
tail_seen = {"calls": 0, "thread_groups": [], "ctx": [], "pins": [], "M": []}
outs_v = pipe.run_groups(make_shared_groups(pipe.SharedPlan(counts=[2, 2], pin=2, verify=True)), fn_shared, pipelined=True)
torch.cuda.synchronize()
sv = pipe.stats()
check(torch.equal(pipe.merge_outputs(outs_v), ref10b) and sv["shared_tail_verified"] == v0["shared_tail_verified"] + 1
      and sv["shared_tail_mismatch"] == v0["shared_tail_mismatch"] and tail_seen["calls"] == 3 and tail_seen["M"] == [4, 2, 2],
      f"(10) verify mode: shared tail + today's two tails, compared IDENTICAL (verified {sv['shared_tail_verified']}, mismatch {sv['shared_tail_mismatch']}, runs {tail_seen['M']})")
tail_split["on"] = False  # the planted difference: the row-count-dependent op is not split any more
outs_m = pipe.run_groups(make_shared_groups(pipe.SharedPlan(counts=[2, 2], pin=2, verify=True)), fn_shared, pipelined=True)
torch.cuda.synchronize()
sm = pipe.stats()
check(torch.equal(pipe.merge_outputs(outs_m), ref10b) and sm["shared_tail_mismatch"] == v0["shared_tail_mismatch"] + 1,
      f"(10) verify mode reports a planted mismatch (count {sm['shared_tail_mismatch']}) and the step continues from today's outputs")
tail_split["on"] = True
# the dump hook inside the shared scope: one entry per group, keyed like today's per-group entries (pos0, rows, layer)
dump_dir = tempfile.mkdtemp(prefix="k9dump")
open(os.path.join(dump_dir, "ENABLE"), "w").close()
saved_dump_dir = pipe._DUMP_DIR
pipe._DUMP_DIR = dump_dir
with pipe._SharedScope(st_obj):
    pipe.dump_layer(21, torch.tensor([101, 103, 204, 206], device=dev), x=torch.randn(4, R_H, device=dev), residual=torch.randn(4, R_HC, R_H, device=dev))
pipe._DUMP_DIR = saved_dump_dir
import json as _json  # noqa: E402

dump_lines = [_json.loads(l) for l in open(os.path.join(dump_dir, "rank0.jsonl")).read().splitlines() if l.strip()]
check(len(dump_lines) == 2 and [(e["group"], e["rows"], e["pos0"], e["pos_last"], e["layer"]) for e in dump_lines] == [(0, 2, 101, 103, 21), (1, 2, 204, 206, 21)]
      and all(e["x_shape"] == [2, R_H] and e.get("shared") for e in dump_lines),
      f"(10) the dump hook splits the shared tail's states into one entry per group ({[(e['group'], e['rows'], e['pos0']) for e in dump_lines]})")
# fallbacks: no plan -> shared_replay returns None and each group runs its own tail; a partner that never arrives -> an error, not a hang
tail_seen = {"calls": 0, "thread_groups": [], "ctx": [], "pins": [], "M": []}
outs_n = pipe.run_groups(make_shared_groups(None), fn_shared, pipelined=True)
torch.cuda.synchronize()
check(torch.equal(pipe.merge_outputs(outs_n), ref10b) and tail_seen["calls"] == 2 and tail_seen["M"] == [2, 2],
      "(10) without a shared plan shared_replay returns None and the groups run today's tails")


def fn_absent(g: pipe.PipeGroup):
    if g.idx == 1:
        return tail_inputs(g, 2)[0]  # group 1 never reaches the rendezvous
    return fn_shared(g)


try:
    pipe.run_groups(make_shared_groups(plan), fn_absent, pipelined=True)
    check(False, "(10) a partner that never arrives at the rendezvous must raise")
except RuntimeError as exc:
    check("shared tail" in str(exc) or "shared tail" in str(exc.__cause__), f"(10) a partner that never arrives raises on the runner thread instead of hanging ({str(exc.__cause__)[:80]})")
check(pipe.slot_graph_config(SimpleNamespace(compilation_config=SimpleNamespace(decoder_replay_cudagraph_capture_sizes=[128, 256])), pipe.SHARED_SLOT, 128).compilation_config.decoder_replay_cudagraph_capture_sizes == [256]
      and pipe.slot_graph_config(SimpleNamespace(compilation_config=SimpleNamespace(decoder_replay_cudagraph_capture_sizes=[128])), pipe.SHARED_SLOT, 128) is None
      and pipe.slot_graph_config("cfg", 1, 128) == "cfg",
      "(10) slot_graph_config: the shared slot keeps the sizes >= 2 x window ([128,256] -> [256]; [128] -> none), the groups' slots the config as is")
os.environ["VLLM_MOET_PREFILL_PIPE_SHARED_TAIL"] = "1"
os.environ["VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS"] = "1"
os.environ["VLLM_MOET_PREFILL_PIPE"] = "2"
slots_all = pipe.graph_slots()
os.environ["VLLM_MOET_PREFILL_PIPE_SHARED_TAIL_GRAPHS"] = "shared"
slots_shared = pipe.graph_slots()
os.environ["VLLM_MOET_PREFILL_PIPE_SHARED_TAIL"] = "0"
slots_off = pipe.graph_slots()
for k in ("VLLM_MOET_PREFILL_PIPE_SHARED_TAIL", "VLLM_MOET_PREFILL_PIPE_SHARED_TAIL_GRAPHS", "VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS", "VLLM_MOET_PREFILL_PIPE"):
    os.environ.pop(k, None)
check(slots_all == [1, 2, 3] and slots_shared == [3] and slots_off == [1, 2], f"(10) graph_slots: {slots_all} / {slots_shared} / {slots_off} (shared tail all / shared / off)")
# the shared slot's graph capture runs inside a scope that splits every captured size into window-sized blocks (the per-group
# split of the all-reduces / GEMMs is recorded into the graph); the groups' slots capture unsplit
mhc_warmup._MOET_SP_PIN_TOKENS = None
with pipe.shared_capture_scope(pipe.SHARED_SLOT, 2):
    cap = (pipe.row_blocks(4), pipe.row_blocks(6), pipe.row_blocks(3), pipe.pin_tokens(None), pipe.pin_tokens(7))
with pipe.shared_capture_scope(1, 2):
    cap_g = (pipe.row_blocks(4), pipe.pin_tokens(None))
check(cap == ([(0, 2), (2, 4)], [(0, 2), (2, 4), (4, 6)], [(0, 3)], 2, 7) and cap_g == ([(0, 4)], None) and pipe.current_shared() is None
      and mhc_warmup._MOET_SP_PIN_TOKENS is None,
      f"(10) shared_capture_scope: the shared slot's capture splits 4 -> 2+2, 6 -> 2+2+2, 3 stays, pin = window; a group slot's capture is unsplit ({cap}, {cap_g})")
# the light precheck (task 2): the runner's host-side decision before the groups' metadata builds, on a fake 8192-row step
import numpy as np  # noqa: E402

fake_ms = SimpleNamespace(decoder_replay_layers=SimpleNamespace(window=128), _keeps_rows_np=np.zeros(8, dtype=bool))
pr = SimpleNamespace(model_state=fake_ms, vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(decoder_replay_cudagraph_capture_sizes=[128, 256]),
                                                                       model_config=SimpleNamespace(hf_config=SimpleNamespace(hc_mult=4, hidden_size=5120))))
pr._group_kept = lambda ib, b: pipe.PipeRunner._group_kept(pr, ib, b)
pr._mhc_splits = lambda n: 16
pr._shared_precheck = lambda ib, b: pipe.PipeRunner._shared_precheck(pr, ib, b)
one_req = SimpleNamespace(num_reqs=1, query_start_loc_np=np.array([0, 8192], dtype=np.int32), idx_mapping_np=np.array([3]), is_prefilling_np=np.array([True]))
mixed = SimpleNamespace(num_reqs=3, query_start_loc_np=np.array([0, 8, 16, 8192], dtype=np.int32), idx_mapping_np=np.array([0, 1, 3]), is_prefilling_np=np.array([False, False, True]))
bounds = [(0, 4096), (4096, 8192)]
os.environ.update({"VLLM_MOET_PREFILL_PIPE_SHARED_TAIL": "1", "VLLM_MOET_PREFILL_PIPE_SHARED_TAIL_LIGHT": "1", "VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS": "1"})
kept1 = pr._group_kept(one_req, bounds)
ok_full = pr._shared_precheck(one_req, bounds)
ok_mixed = pr._shared_precheck(mixed, bounds)  # 16 decode rows + 128 + 128 = 272 > 256: no graph for the shared batch
fake_ms._keeps_rows_np[3] = True  # prompt logprobs: the request keeps every row -> no replay batch
ok_keeps = pr._shared_precheck(one_req, bounds)
fake_ms._keeps_rows_np[3] = False
os.environ["VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS"] = "0"
ok_mixed_eager = pr._shared_precheck(mixed, bounds)
os.environ["VLLM_MOET_PREFILL_PIPE_SHARED_TAIL"] = "verify"
ok_verify = pr._shared_precheck(one_req, bounds)
for k in ("VLLM_MOET_PREFILL_PIPE_SHARED_TAIL", "VLLM_MOET_PREFILL_PIPE_SHARED_TAIL_LIGHT", "VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS"):
    os.environ.pop(k, None)
check(kept1 == [[(4096, 128)], [(4096, 128)]] and ok_full and not ok_mixed and not ok_keeps and ok_mixed_eager and not ok_verify,
      f"(10) the light precheck: a full step yes ({ok_full}), decode rows + 256 > the capture list no ({ok_mixed}; eager yes {ok_mixed_eager}), prompt logprobs no ({ok_keeps}), verify mode no ({ok_verify}); kept {kept1}")
fc._forward_context = "main"
mhc_warmup._MOET_SP_PIN_TOKENS = None

print("ALL PASS" if not failures else f"FAILURES: {len(failures)}")
sys.exit(1 if failures else 0)
