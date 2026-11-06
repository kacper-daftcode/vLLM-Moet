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
