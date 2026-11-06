# SPDX-License-Identifier: Apache-2.0
"""vllm-moet: the pipelined sequence-parallel prefill of DeepSeek-V4.1 on sm_120 (TP4) — step 47 candidate.

Installed by tools/dsv41_sm120/patch_vllm_prefill_pipe_sm120.py as vllm/v1/worker/moet_prefill_pipe.py; the patched
V2 model runner, the DeepSeek-V4.1 model / model state / replay layers / Engram import it. Everything here is a no-op
unless VLLM_MOET_PREFILL_PIPE=2 and the step qualifies, so the patched files behave as before otherwise.

What it does (K/REPORT.md §4.0 has the dependency graph). With --max-num-batched-tokens 8192 an eager prefill step
carries up to two 4 096-row *groups* of the same request (A = rows [0, 4096), B = the rest: for any prompt length the
groups are exactly today's sequence of 4 096-row chunks). The runner builds attention metadata + a replay batch + a
forward context per group (group-private metadata builders and model-state buffers, "slots" 1 and 2; slot 0 = the
whole step, built as today for the Engram hash). The model's forward runs its head (embedding, Engram hashes + rows)
once and its tail (shard, layers, replay, collapse, norm) once per group on a thread of its own. The two threads
alternate at the sequence-parallel boundaries like vLLM's DBO protocol: a layer's RS -> mHC -> shard GEMMs -> AG chain
("phase") is issued on the comm stream N, the attention / MoE / replay on the compute stream C; at a phase end the
thread records an N event, switches back to C and hands the CPU to the other group; on resume C waits for the event.
So on the GPU, C runs attn(A) attn(B) MoE(A) MoE(B) ... back to back while N runs each group's collectives under the
other group's compute, and A's attention of layer L always precedes B's on C: B's rows read the KV A's rows wrote at
that layer (the only inter-group dependency), by stream order - asserted on the CPU, never silently reordered.

Bits: every kernel of a group runs at the group's row count with the group's rows = today's chunk; only the stream
and the moment differ. PIPE=0 at 8 192 batched tokens is *not* today's numerics (the reference is the served 4 096).

Switches: VLLM_MOET_PREFILL_PIPE (0 = off, 2 = two groups), VLLM_MOET_PREFILL_PIPE_ROWS (4096), VLLM_MOET_PREFILL_PIPE_MIXED
(pipe | today | error: a step whose rows are not all prefill), VLLM_MOET_PREFILL_PIPE_COMM_PRIORITY (0; -1 = high),
VLLM_MOET_PREFILL_PIPE_REPLAY (serial | stream: the replay section on its own stream with handoffs - measured neutral),
VLLM_MOET_PREFILL_PIPE_DUMP=<dir> (+ <dir>/ENABLE present: per-layer sha256 of the hc states, the bits harness).

Measured (test host, 4x RTX PRO 6000 SE, TP4, image r25 + this patch, K/REPORT.md §4.1): fresh prefill 163K 6947 -> 5360 ms
(-22.9 %), 19K 819 -> 671 ms; long_agree / agree ALL IDENTICAL to r25, per-layer dumps 800/800 identical (PIPE=2 at 8192 vs
PIPE=0 at 4096, 4 ranks). What remains: the two replay tails per step (write_kv(20) + the 128-row layers, launch-bound and
skewed across ranks) ~ 80-90 ms of a ~300 ms step.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
import torch

from vllm import forward_context as _fc
from vllm.logger import init_logger

logger = init_logger("vllm.vllm_moet.prefill_pipe")

_TURN_TIMEOUT_S = float(os.environ.get("VLLM_MOET_PREFILL_PIPE_TURN_TIMEOUT_S", "900"))


# ------------------------------------------------------------------------------------------------------- switches
def mode() -> int:
    try:
        return int(os.environ.get("VLLM_MOET_PREFILL_PIPE", "0"))
    except ValueError:
        return 0


def group_rows() -> int:
    try:
        return int(os.environ.get("VLLM_MOET_PREFILL_PIPE_ROWS", "4096"))
    except ValueError:
        return 4096


def mixed_policy() -> str:
    policy = os.environ.get("VLLM_MOET_PREFILL_PIPE_MIXED", "pipe")
    return policy if policy in ("pipe", "today", "error") else "pipe"


def replay_mode() -> str:
    """VLLM_MOET_PREFILL_PIPE_REPLAY: "serial" (default) = a group's replay section (layers 20-39 on 128 rows) is issued
    whole on the compute stream, as the rest of its tail (v1 of K/REPORT.md §4: -22.9 % at 163K); "stream" = the section
    on a stream of its own with a CPU handoff per layer (v3: measured the same within noise - the two groups end their
    sequence-parallel layers together, so one layer is all a tail can overlap)."""
    mode = os.environ.get("VLLM_MOET_PREFILL_PIPE_REPLAY", "serial")
    return mode if mode in ("serial", "stream") else "serial"


def _comm_priority() -> int:
    try:
        return int(os.environ.get("VLLM_MOET_PREFILL_PIPE_COMM_PRIORITY", "0"))
    except ValueError:
        return 0


# ------------------------------------------------------------------------------------------------- per-thread state
_tls = threading.local()
_PREPARE_SLOT = 0  # the slot the runner thread is building metadata for (0 = the whole step)
_PENDING: list["PipeGroup"] | None = None  # the step's groups, published by the runner around the model call
_STATS = {"steps": 0, "pipelined": 0, "sequential": 0, "mixed": 0, "today": 0}


@dataclass
class PipeGroup:
    """One row group of a step: its rows, forward context (metadata of its rows) and runtime bookkeeping."""

    idx: int
    token_offset: int
    num_tokens: int
    forward_context: Any
    slot: int  # = idx + 1: the model-state / replay-batch / builder slot
    step: "PipeStep | None" = None
    phases_done: int = 0
    phase_open: bool = False
    pin: Any = None  # the thread's mHC split-k pin, saved across a handoff
    active_context: Any = None  # the forward context the thread had at its last handoff (the replay batch's inside the replay)
    stream: Any = None  # the thread's current stream at its last handoff (C, or R inside the replay section)
    phase_stream: Any = None  # the stream of the open phase (N for a boundary chain, R for the replay section)
    phase_kind: str = ""  # "comm" | "replay"


def current_group() -> PipeGroup | None:
    return getattr(_tls, "group", None)


def in_pipe() -> bool:
    return current_group() is not None


def token_offset() -> int:
    g = current_group()
    return 0 if g is None else g.token_offset


def slice_rows(t: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """The calling group's rows of a per-row buffer prepared for the whole step (all rows outside a group)."""
    start = token_offset()
    return t[start : start + num_tokens]


def current_slot() -> int:
    g = current_group()
    return 0 if g is None else g.slot


def prepare_slot() -> int:
    return _PREPARE_SLOT


class _PrepareSlot:
    def __init__(self, slot: int) -> None:
        self.slot = slot
        self.prev = 0

    def __enter__(self):
        global _PREPARE_SLOT
        self.prev = _PREPARE_SLOT
        _PREPARE_SLOT = self.slot
        return self

    def __exit__(self, *exc):
        global _PREPARE_SLOT
        _PREPARE_SLOT = self.prev
        return False


def pending_groups() -> list[PipeGroup] | None:
    return _PENDING


class step_context:
    """Runner side: publish the step's groups for the model's dispatch (None = today's path)."""

    def __init__(self, groups: list[PipeGroup] | None) -> None:
        self.groups = groups

    def __enter__(self):
        global _PENDING
        _PENDING = self.groups
        return self

    def __exit__(self, *exc):
        global _PENDING
        _PENDING = None
        return False


# ---------------------------------------------------------------------------------------------- the row buffers
class RowBufferSwitch:
    """One storage per group for buffers the layers index by batch row (the indexer's top-k indices, the candidate
    blocks): the thread that resumes swaps its group's storage in (Tensor.set_), so every holder of the tensor sees its
    group's rows at local row numbers while kernels already enqueued keep the pointers they were launched with.
    Group 0 uses the original storage, restored after the step (the drafter shares the top-k buffer)."""

    def __init__(self, buffers: list[torch.Tensor], num_groups: int) -> None:
        self.buffers = buffers
        self.storages = [
            [buf.view_as(buf)] + [torch.empty_like(buf) for _ in range(num_groups - 1)] for buf in buffers
        ]
        self.current = 0

    def switch(self, group_idx: int) -> None:
        if group_idx == self.current:
            return
        for buf, storages in zip(self.buffers, self.storages):
            buf.set_(storages[group_idx])
        self.current = group_idx


def _mhc_warmup():
    """The mHC warm-up module: its split-k pin (_MOET_SP_PIN_TOKENS, step 41) is a module global the layer sets around a
    boundary - saved and restored per thread across a handoff."""
    import vllm.model_executor.kernels.mhc.warmup as mhc_warmup

    return mhc_warmup


# ---------------------------------------------------------------------------------------------- the handoff
class PipeStep:
    """The threads of one pipelined step and their handoff (vLLM's ubatching protocol, simplified): exactly one
    thread runs at a time; a thread parks at every phase end and wakes the other; a finished thread wakes the other
    for good. Group 0 (the earlier rows) takes the first turn, so its k-th compute segment is issued before group
    1's k-th: the KV write-before-read order on the one compute stream."""

    def __init__(
        self,
        groups: list[PipeGroup],
        compute_stream: torch.cuda.Stream,
        comm_stream: torch.cuda.Stream,
        row_switch: RowBufferSwitch | None,
        device: torch.device,
        pipelined: bool,
        replay_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.groups = groups
        self.compute_stream = compute_stream
        self.comm_stream = comm_stream
        self.replay_stream = replay_stream if replay_stream is not None else comm_stream
        self.row_switch = row_switch
        self.device = device
        self.pipelined = pipelined
        n = len(groups)
        self.wake = [threading.Event() for _ in range(n)]
        self.finished = [False] * n
        self.outputs: dict[int, Any] = {}
        self.errors: dict[int, BaseException] = {}
        for g in groups:
            g.step = self
            g.phases_done = 0
            g.phase_open = False
            g.pin = None
            g.active_context = g.forward_context
            g.stream = compute_stream
            g.phase_stream = None
            g.phase_kind = ""

    # -- per-thread state restored at every resume
    def _resume(self, g: PipeGroup) -> None:
        # the stream the thread had when it handed over: C, or R inside the replay section (vLLM's patched
        # set_stream: current_stream() follows per thread)
        torch.cuda.set_stream(g.stream)
        # the forward context the thread had when it handed over: the group's, or the replay batch's override
        # inside the replay layers (override_forward_context sets the same global)
        _fc._forward_context = g.active_context
        if self.row_switch is not None:
            self.row_switch.switch(g.idx)
        _mhc_warmup()._MOET_SP_PIN_TOKENS = g.pin

    def _save(self, g: PipeGroup) -> None:
        from vllm.utils.torch_utils import current_stream

        g.pin = _mhc_warmup()._MOET_SP_PIN_TOKENS
        g.active_context = _fc._forward_context
        g.stream = current_stream()

    def _wait_turn(self, g: PipeGroup) -> None:
        if not self.wake[g.idx].wait(timeout=_TURN_TIMEOUT_S):
            raise RuntimeError(
                f"prefill pipeline: group {g.idx} waited {_TURN_TIMEOUT_S:.0f} s for its turn "
                f"(the other group is stuck: finished={self.finished}, errors={list(self.errors)})"
            )
        self.wake[g.idx].clear()

    def yield_(self, g: PipeGroup) -> None:
        other = (g.idx + 1) % len(self.groups)
        self._save(g)
        if not self.finished[other]:
            self.wake[other].set()
            self._wait_turn(g)
        self._resume(g)

    def check_order(self, g: PipeGroup) -> None:
        """At group g's k-th phase end: every earlier group must have ended its k-th phase already, so its k-th compute
        segment (the attention that writes the KV g's rows read) is issued before g's at the next turns."""
        for earlier in self.groups[: g.idx]:
            if earlier.phases_done < g.phases_done:
                raise RuntimeError(
                    f"prefill pipeline: dependency order violated - group {g.idx} ended phase {g.phases_done} but "
                    f"group {earlier.idx} has ended only {earlier.phases_done} (the groups' layer structures differ)"
                )

    def _thread_main(self, g: PipeGroup, fn: Callable[[PipeGroup], Any]) -> None:
        try:
            torch.cuda.set_device(self.device)
            with torch.inference_mode():
                self._wait_turn(g)
                _tls.group = g
                self._resume(g)
                self.outputs[g.idx] = fn(g)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the runner thread
            self.errors[g.idx] = exc
        finally:
            _tls.group = None
            self.finished[g.idx] = True
            other = (g.idx + 1) % len(self.groups)
            self.wake[other].set()

    def run(self, fn: Callable[[PipeGroup], Any]) -> list[Any]:
        saved_context = _fc._forward_context
        saved_pin = _mhc_warmup()._MOET_SP_PIN_TOKENS
        try:
            threads = [
                threading.Thread(target=self._thread_main, args=(g, fn), name=f"moet-pipe-{g.idx}")
                for g in self.groups
            ]
            for t in threads:
                t.start()
            self.wake[0].set()
            for t in threads:
                t.join()
        finally:
            _fc._forward_context = saved_context
            if self.row_switch is not None:
                self.row_switch.switch(0)
            _mhc_warmup()._MOET_SP_PIN_TOKENS = saved_pin
            for g in self.groups:
                g.step = None
        if self.errors:
            failed = min(self.errors)
            raise RuntimeError(f"prefill pipeline: group {failed} of {len(self.groups)} failed") from self.errors[
                failed
            ]
        return [self.outputs[i] for i in range(len(self.groups))]

    def run_sequential(self, fn: Callable[[PipeGroup], Any]) -> list[Any]:
        """The groups one after the other on the runner thread (a group below the SP minimum): today's bits, no
        overlap, no phases."""
        saved_context = _fc._forward_context
        outputs = []
        try:
            for g in self.groups:
                _tls.group = g
                g.step = None  # no phases
                _fc._forward_context = g.forward_context
                if self.row_switch is not None:
                    self.row_switch.switch(g.idx)
                outputs.append(fn(g))
        finally:
            _tls.group = None
            _fc._forward_context = saved_context
            if self.row_switch is not None:
                self.row_switch.switch(0)
        return outputs


# ------------------------------------------------------------------------------------------------- the phases
def _phase_begin(kind: str) -> None:
    g = current_group()
    if g is None or g.step is None or not g.step.pipelined:
        return
    if g.phase_open:
        raise RuntimeError(f"prefill pipeline: a {kind} phase was opened inside an open {g.phase_kind} phase")
    step = g.step
    stream = step.comm_stream if kind == "comm" else step.replay_stream
    ev = torch.cuda.Event()
    ev.record(step.compute_stream)
    torch.cuda.set_stream(stream)
    stream.wait_event(ev)
    g.phase_open = True
    g.phase_stream = stream
    g.phase_kind = kind


def phase_begin() -> None:
    """Start of a sequence-parallel boundary chain (RS -> mHC -> shard GEMMs -> AG): issue it on the comm stream
    after everything the compute stream has so far. A no-op outside a pipelined group."""
    _phase_begin("comm")


def phase_end() -> None:
    """End of a phase (the chain after its all-gather, or the replay section): back to the compute stream, hand the
    CPU to the other group, and on resume make the compute stream wait for the phase. A no-op when no phase is open."""
    g = current_group()
    if g is None or g.step is None or not g.step.pipelined or not g.phase_open:
        return
    step = g.step
    ev = torch.cuda.Event()
    ev.record(g.phase_stream)
    torch.cuda.set_stream(step.compute_stream)
    g.phase_open = False
    g.phase_stream = None
    g.phase_kind = ""
    g.phases_done += 1
    step.check_order(g)
    step.yield_(g)
    step.compute_stream.wait_event(ev)


def replay_begin(*inputs: torch.Tensor | None) -> None:
    """Start of a group's replay section (the 128-row layers 20-39 + collapse): a launch-bound chain of small kernels
    and all-reduces that would block the other group's compute if it sat on the compute stream (v2 of §4.2: +1.7 %).
    It runs on a stream of its own, R, after everything the compute stream has so far; the handoffs inside
    (yield_point, once per layer) keep R current. `inputs` are the hc states the section reads that were born on the
    compute stream: kept out of its reuse until R is done with them. A no-op unless VLLM_MOET_PREFILL_PIPE_REPLAY=stream."""
    if replay_mode() != "stream":
        return
    _phase_begin("replay")
    g = current_group()
    if g is None or not g.phase_open:
        return
    for t in inputs:
        if isinstance(t, torch.Tensor):
            t.record_stream(g.phase_stream)


def replay_end(*outputs: Any) -> None:
    """End of the replay section: the compute stream waits for R (after the handoff); the section's outputs - born on
    R, consumed on the compute stream by the collapse / norm / merge - are kept out of R's reuse until then."""
    g = current_group()
    if g is None or g.step is None or not g.phase_open or g.phase_kind != "replay":
        return
    compute = g.step.compute_stream

    def mark(t):
        if isinstance(t, torch.Tensor):
            t.record_stream(compute)
        elif isinstance(t, (list, tuple)):
            for u in t:
                mark(u)

    mark(outputs)
    phase_end()


def yield_point() -> None:
    """A CPU handoff without a stream switch, for the layers of a group that have no sequence-parallel phases (the
    replay layers on 128 rows): their launch-bound issue would otherwise park the other group's thread for the whole
    tail, draining the GPU; handing the CPU over once per layer keeps the other group's compute flowing under it.
    A no-op outside a pipelined group or inside an open phase."""
    g = current_group()
    if g is None or g.step is None or not g.step.pipelined or (g.phase_open and g.phase_kind != "replay"):
        return
    if replay_mode() != "stream":
        return
    g.step.yield_(g)


def phase_is_open() -> bool:
    g = current_group()
    return g is not None and g.phase_open


def record_for_comm(x: torch.Tensor) -> torch.Tensor:
    """A tensor born on the compute stream that the open phase reads on the comm stream: keep its block out of the
    compute stream's reuse until the comm stream is done with it (the one allocator hazard of the two-stream flow;
    the other direction is covered by phase_begin's wait)."""
    g = current_group()
    if g is not None and g.phase_open and g.step is not None:
        x.record_stream(g.step.comm_stream)
    return x


# ------------------------------------------------------------------------------------------------- model side
def merge_outputs(outputs: list[Any]) -> Any:
    """The per-group model outputs in row order: tensors concatenate along the token dim, a tuple's list elements
    (the aux hidden states the drafter reads) element-wise."""
    from vllm.sequence import IntermediateTensors

    first = outputs[0]
    if isinstance(first, IntermediateTensors):
        return IntermediateTensors(
            {key: torch.cat([out.tensors[key] for out in outputs], dim=0) for key in first.tensors}
        )
    if isinstance(first, tuple):
        merged = []
        for parts in zip(*outputs):
            if isinstance(parts[0], (list, tuple)):
                merged.append(type(parts[0])(torch.cat(elems, dim=0) for elems in zip(*parts)))
            else:
                merged.append(torch.cat(parts, dim=0))
        return tuple(merged)
    return torch.cat(outputs, dim=0)


def run_groups(groups: list[PipeGroup], fn: Callable[[PipeGroup], Any], pipelined: bool) -> list[Any]:
    """Run fn(group) for every group: pipelined on two threads (both groups sequence parallel), else one after the
    other. Called on the runner thread inside the model's forward."""
    runner = _RUNNER
    assert runner is not None, "prefill pipeline: groups without a runner"
    from vllm.utils.torch_utils import current_stream

    step = PipeStep(
        groups, current_stream(), runner.comm_stream, runner.row_switch, runner.device, pipelined,
        replay_stream=getattr(runner, "replay_stream", None),
    )
    _STATS["steps"] += 1
    if pipelined:
        _STATS["pipelined"] += 1
        if _STATS["pipelined"] == 1:
            logger.info(
                "vllm-moet sm_120 prefill pipeline: steps of > %d tokens run as %d row groups one layer apart, "
                "the sequence-parallel collectives of one under the compute of the other (VLLM_MOET_PREFILL_PIPE=2)",
                group_rows(),
                len(groups),
            )
        return step.run(fn)
    _STATS["sequential"] += 1
    if _STATS["sequential"] == 1:
        logger.info(
            "vllm-moet sm_120 prefill pipeline: a step whose groups are not all sequence parallel runs them one "
            "after the other (today's bits, no overlap) - sizes %s",
            [g.num_tokens for g in groups],
        )
    return step.run_sequential(fn)


# ------------------------------------------------------------------------------------------------- runner side
_RUNNER: "PipeRunner | None" = None


class PipeRunner:
    """Per model runner: the comm stream, the per-group metadata builders (slots 1..n), the sliced-batch buffers and
    the row-buffer storages."""

    def __init__(self, runner: Any) -> None:
        from vllm.v1.worker.utils import AttentionGroup  # noqa: F401 - the type the copies keep

        self.vllm_config = runner.vllm_config
        self.device = runner.device
        self.model_state = runner.model_state
        self.kv_cache_config = runner.kv_cache_config
        self.max_num_reqs = runner.max_num_reqs
        self.num_groups = 2
        self.comm_stream = torch.cuda.Stream(device=self.device, priority=_comm_priority())
        self.replay_stream = torch.cuda.Stream(device=self.device)
        # The groups' query offsets / seq lens are rebased: not views of the step's buffers.
        self.query_start_loc_bufs = [
            torch.zeros(self.max_num_reqs + 1, dtype=torch.int32, device=self.device) for _ in range(self.num_groups)
        ]
        self.seq_lens_bufs = [
            torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device) for _ in range(self.num_groups)
        ]
        # Slot 0 = the runner's groups (the whole step); slots 1..n = copies with builders of their own (a builder
        # keeps the metadata it built on persistent buffers).
        self.attn_groups_slots = [runner.attn_groups] + [
            self._copy_attn_groups(runner.attn_groups) for _ in range(self.num_groups)
        ]
        self.row_switch: RowBufferSwitch | None = None
        inner = self._find_model_with_row_buffers(runner.model)
        if inner is not None:
            buffers = [
                buf
                for buf in (getattr(inner, "topk_indices_buffer", None), getattr(inner, "candidate_block_buffer", None))
                if isinstance(buf, torch.Tensor)
            ]
            if buffers:
                self.row_switch = RowBufferSwitch(buffers, self.num_groups)
        self._warned: set[str] = set()
        logger.info(
            "vllm-moet sm_120 prefill pipeline ready: %d groups of %d rows, %d row buffers per group, comm stream "
            "priority %d",
            self.num_groups,
            group_rows(),
            0 if self.row_switch is None else len(self.row_switch.buffers),
            _comm_priority(),
        )

    def _copy_attn_groups(self, attn_groups):
        copies_all = []
        for groups in attn_groups:
            copies = []
            for group in groups:
                copy = replace(group, metadata_builders=[])
                copy.create_metadata_builders(self.vllm_config, self.device, group.kernel_block_size)
                copies.append(copy)
            copies_all.append(copies)
        return copies_all

    @staticmethod
    def _find_model_with_row_buffers(model: torch.nn.Module):
        for module in model.modules():
            if hasattr(module, "topk_indices_buffer") and hasattr(module, "candidate_block_buffer") and hasattr(
                module, "layers"
            ):
                return module
        return None

    def _warn_once(self, key: str, msg: str, *args) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg, *args)

    def plan(self, runner: Any, input_batch: Any, batch_desc: Any, block_tables, slot_mappings) -> list[PipeGroup] | None:
        from vllm.config import CUDAGraphMode

        if batch_desc.cg_mode != CUDAGraphMode.NONE:
            return None
        rows = group_rows()
        num_tokens = input_batch.num_tokens
        if num_tokens <= rows:
            _STATS["today"] += 1
            return None
        if input_batch.num_tokens_after_padding != num_tokens:
            self._warn_once("padding", "prefill pipeline: padded eager step (%d != %d) runs today's path",
                            input_batch.num_tokens_after_padding, num_tokens)
            return None
        if runner.dp_size > 1 or not (runner.is_first_pp_rank and runner.is_last_pp_rank) or runner.dcp_size > 1 \
                or getattr(runner, "pcp_manager", None) is not None:
            self._warn_once("parallel", "prefill pipeline: DP / PP / DCP / PCP steps run today's path")
            return None
        num_reqs = input_batch.num_reqs
        if not bool(input_batch.is_prefilling_np[:num_reqs].all()):
            policy = mixed_policy()
            _STATS["mixed"] += 1
            if policy == "error":
                raise RuntimeError(
                    f"prefill pipeline: a step with decode rows ({num_reqs} requests, {num_tokens} tokens) and "
                    "VLLM_MOET_PREFILL_PIPE_MIXED=error"
                )
            if policy == "today":
                self._warn_once("mixed", "prefill pipeline: a step with decode rows runs today's path (MIXED=today)")
                return None
            self._warn_once("mixed", "prefill pipeline: a step with decode rows is pipelined as whole rows "
                            "(the decodes in group 0; MIXED=pipe)")
        bounds = [(0, rows), (rows, num_tokens)]
        if num_tokens > self.num_groups * rows:
            self._warn_once("big", "prefill pipeline: a step of %d tokens > %d x %d rows: the last group takes the rest",
                            num_tokens, self.num_groups, rows)
        return self._build(input_batch, block_tables, slot_mappings, bounds)

    def _build(self, input_batch, block_tables, slot_mappings, bounds) -> list[PipeGroup]:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, create_forward_context
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        from vllm.v1.worker.gpu.ubatch_utils import _slice_input_batch
        from vllm.v1.worker.ubatch_utils import UBatchSlice

        num_reqs = input_batch.num_reqs
        qsl = input_batch.query_start_loc_np[: num_reqs + 1]
        groups: list[PipeGroup] = []
        for i, (lo, hi) in enumerate(bounds):
            req_lo = int(np.searchsorted(qsl, lo, side="right") - 1)
            req_hi = int(np.searchsorted(qsl, hi - 1, side="right") - 1)
            ub_slice = UBatchSlice(slice(req_lo, req_hi + 1), slice(lo, hi))
            ubatch = _slice_input_batch(
                input_batch, ub_slice, self.query_start_loc_bufs[i], self.seq_lens_bufs[i]
            )
            # the group's own longest query (None: prepare_attn derives it from the sliced query lengths)
            ubatch = replace(ubatch, max_query_len=None)
            group_block_tables = tuple(bt[ub_slice.request_slice] for bt in block_tables)
            group_slot_mappings = slot_mappings[:, ub_slice.token_slice]
            slot = i + 1
            with _PrepareSlot(slot):
                attn_metadata = self.model_state.prepare_attn(
                    ubatch,
                    CUDAGraphMode.NONE,
                    group_block_tables,
                    group_slot_mappings,
                    self.attn_groups_slots[slot],
                    self.kv_cache_config,
                )
            forward_ctx = create_forward_context(
                attn_metadata,
                self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=BatchDescriptor(num_tokens=hi - lo),
                slot_mapping=build_slot_mappings_by_layer(group_slot_mappings, self.kv_cache_config),
                is_padding=ubatch.is_padding,
            )
            groups.append(PipeGroup(idx=i, token_offset=lo, num_tokens=hi - lo, forward_context=forward_ctx, slot=slot))
        # the whole step's buffers current again for whatever else the step prepares (the groups keep their own)
        select = getattr(self.model_state, "_moet_pipe_select_slot", None)
        if select is not None:
            select(0)
        return groups


def plan_groups(runner: Any, input_batch: Any, batch_desc: Any, block_tables, slot_mappings) -> list[PipeGroup] | None:
    """Runner side, once per real eager step after the whole step's metadata: the step's groups, or None."""
    global _RUNNER
    if mode() != 2:
        return None
    if _RUNNER is None:
        _RUNNER = PipeRunner(runner)
    return _RUNNER.plan(runner, input_batch, batch_desc, block_tables, slot_mappings)


def stats() -> dict:
    return dict(_STATS)


# ------------------------------------------------------------------------------------------------- bits harness
_DUMP_DIR = os.environ.get("VLLM_MOET_PREFILL_PIPE_DUMP", "")
_DUMP_FORWARD = {"n": 0}


def _dump_active() -> bool:
    return bool(_DUMP_DIR) and os.path.exists(os.path.join(_DUMP_DIR, "ENABLE"))


def dump_forward_begin() -> None:
    """Once per model forward (the head): numbers the forwards of this process."""
    if _DUMP_DIR:
        _DUMP_FORWARD["n"] += 1


def _sha(t: torch.Tensor | None) -> str | None:
    if t is None or not isinstance(t, torch.Tensor):
        return None
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:24]


def dump_layer(layer: int, positions: torch.Tensor, **tensors: torch.Tensor | None) -> None:
    """VLLM_MOET_PREFILL_PIPE_DUMP: after layer `layer` of a group (or of an unpipelined step), one JSON line per rank
    with the sha256 of each hc state - keyed by the rows' first position and count, so the same chunk of the same
    prompt can be matched across a PIPE=2 (8192) and a PIPE=0 (4096) run."""
    if not _dump_active():
        return
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001
        rank = 0
    g = current_group()
    n = int(positions.shape[0])
    entry = {
        "forward": _DUMP_FORWARD["n"],
        "group": -1 if g is None else g.idx,
        "offset": 0 if g is None else g.token_offset,
        "layer": layer,
        "rows": n,
        "pos0": int(positions[0].item()) if n else -1,
        "pos_last": int(positions[-1].item()) if n else -1,
        "t": time.time(),
    }
    for name, t in tensors.items():
        entry[name] = _sha(t)
        if isinstance(t, torch.Tensor):
            entry[name + "_shape"] = list(t.shape)
    with open(os.path.join(_DUMP_DIR, f"rank{rank}.jsonl"), "a") as fh:
        fh.write(json.dumps(entry) + "\n")
