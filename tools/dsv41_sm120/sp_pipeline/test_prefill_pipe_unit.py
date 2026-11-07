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

print("ALL PASS" if not failures else f"FAILURES: {len(failures)}")
sys.exit(1 if failures else 0)
