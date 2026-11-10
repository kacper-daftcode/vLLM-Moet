# SPDX-License-Identifier: Apache-2.0
"""vllm-moet: the pipelined sequence-parallel prefill of DeepSeek-V4.1 on sm_120 (TP4) — step 47 candidate.

Installed by tools/dsv41_sm120/patch_vllm_prefill_pipe_sm120.py as vllm/v1/worker/moet_prefill_pipe.py; the patched
V2 model runner, the DeepSeek-V4.1 model / model state / replay layers / Engram import it. Everything here is a no-op
unless VLLM_MOET_PREFILL_PIPE=2 and the step qualifies, so the patched files behave as before otherwise.

What it does. With --max-num-batched-tokens 8192 an eager prefill step
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
VLLM_MOET_PREFILL_PIPE_TAIL (serial | barrier | sync: when a group's replay tail may start relative to the other
group's last sequence-parallel phase, see tail_mode), VLLM_MOET_PREFILL_PIPE_LOG_STEPS (0 | 1 = one log line per
step with decode rows: the batch composition and the groups | 2 = every pipelined step),
VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS (0 | 1: the groups' replay layers as CUDA graphs, one graph set per slot; needs
--compilation-config decoder_replay_cudagraph_capture_sizes >= the 128-row window, see replay_graphs),
VLLM_MOET_PREFILL_PIPE_FORCE_SEQUENTIAL (0 | 1: the groups one after the other, the bits oracle of the pipelined path),
VLLM_MOET_PREFILL_PIPE_DUMP=<dir> (+ <dir>/ENABLE present: per-layer sha256 of the hc states, the bits harness),
VLLM_MOET_PREFILL_PIPE_COMM2 (0 | 1: the odd group's collectives on a second NCCL communicator, see comm2_enabled - K-6),
VLLM_MOET_PREFILL_PIPE_KV_CONNECTOR (allow | refuse: whether the pipeline may activate when a KV transfer connector - the
production KV offload tier - is configured, see kv_connector_policy - K-6, the r26 production window of 2026-11-07),
VLLM_MOET_PREFILL_PIPE_AFFINITY (unset | pin[,fifo]: the group threads' CPU placement and scheduling class, see
affinity_policy - K-7, r26b hardening against host CPU load), VLLM_MOET_PREFILL_PIPE_ORDER (today | comm2 | comm2prio: the
named launch-order variant of K/k6/DESIGN_launch_order.md, a bundle over COMM2 / COMM_PRIORITY, see order - K-7),
VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS (0 | 1 | verify: the whole sequence-parallel section of a full pipelined step - both groups,
shard -> layers 0-19 -> the first replay layer's KV - as one CUDA graph captured at engine start and replayed with one launch per
step, see phase_graphs - K-8), VLLM_MOET_PREFILL_PIPE_PHASE_GRAPH_MAX_CTX (262144: the context length the graph is captured for),
VLLM_MOET_PREFILL_PIPE_SHARED_TAIL (0 | 1 | verify: the replay layers 20-39 + collapse of a full pipelined step run ONCE on the
concatenation of both groups' replay rows (256) instead of once per group, the three row-count-dependent ops split per group so the
bits stay today's, see shared_tail - K-9), VLLM_MOET_PREFILL_PIPE_SHARED_TAIL_GRAPHS (all | shared: which slots get replay graphs
with the shared tail on), VLLM_MOET_PREFILL_PIPE_DEVICE_COMM (0 | 1 | verify: the sequence-parallel reduce-scatters and all-gathers
of a pipelined step as device-initiated PCIe pushes from a small kernel per collective instead of NCCL - moet_device_comm.py, MK-4;
verify = every device collective compared with NCCL's on the same inputs, NCCL's result consumed, counters in stats()),
VLLM_MOET_PREFILL_PIPE_GREEN (0 | N: a static split of the GPU's SMs for the pipelined step - its comm stream on a CUDA green
context of N SMs, its compute stream on the remaining SMs, see green_sms; sub-switches _GREEN_TAIL / _GREEN_GRAPHS /
_GREEN_AUX / _GREEN_WARMUP / _GREEN_WARMUP_CTX / _GREEN_MOE_CTAS; with VLLM_MOET_RUNTIME_AB=1 the control-file key
PREFILL_PIPE_GREEN turns it off and on per forward).

Measured (4x RTX PRO 6000 SE, TP4, image r25 + this patch): fresh prefill 163K 6947 -> 5360 ms
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
    whole on the compute stream, as the rest of its tail (v1: -22.9 % at 163K); "stream" = the section
    on a stream of its own with a CPU handoff per layer (v3: measured the same within noise - the two groups end their
    sequence-parallel layers together, so one layer is all a tail can overlap)."""
    mode = os.environ.get("VLLM_MOET_PREFILL_PIPE_REPLAY", "serial")
    return mode if mode in ("serial", "stream") else "serial"


def order() -> str:
    """VLLM_MOET_PREFILL_PIPE_ORDER: the launch-order variant of the pipelined step (K-6's hardware read of the SP phase,
    K/k6/DESIGN_launch_order.md section 3). "today" (default) = r26: one NCCL communicator, the comm stream at priority 0.
    "comm2" = the odd group's collectives on a second NCCL communicator (= COMM2=1: the two groups' RS -> AG chains are
    no longer one NCCL FIFO, the compute stream's indexer all_gatherv of one group no longer waits for the other's chain;
    8 ms of a 188-ms SP phase). "comm2prio" = "comm2" + the comm stream at priority -1 (the glue + NCCL CTAs take every
    freed SM slot instead of the compute kernels' last wave; K-3 measured the priority alone as a loss without a trace).
    A named bundle: the individual switches COMM2 / COMM_PRIORITY still apply on top (the stricter setting wins).
    Measured (second 4x RTX PRO 6000 host, kv16+FS, "comm2prio"): bits identical on every gate, 163K 5 389 / 5 321 / 5 344 ms = -0.6 % vs
    5 376-5 391 as built, 19K -1 %; the compute stream's gaps in the SP phase 48 -> 25 ms per step (the indexer all_gatherv waits
    are gone) but the compute kernels slow down under a priority -1 comm stream (attention +5 ms, MXFP8 GEMMs +6 per 19K step),
    net -4 % of the SP phase; no protection against host CPU load (+35 %, it never touches the tails). Opt-in."""
    variant = os.environ.get("VLLM_MOET_PREFILL_PIPE_ORDER", "today")
    return variant if variant in ("today", "comm2", "comm2prio") else "today"


def _comm_priority() -> int:
    if order() == "comm2prio":
        return -1
    try:
        return int(os.environ.get("VLLM_MOET_PREFILL_PIPE_COMM_PRIORITY", "0"))
    except ValueError:
        return 0


def affinity_policy() -> frozenset[str]:
    """VLLM_MOET_PREFILL_PIPE_AFFINITY (K-7): how a pipelined step's two group threads are placed on the CPU. Unset / empty
    (default) = as r26: the threads inherit the process's affinity and the CFS class. "pin" = every group thread calls
    sched_setaffinity on one CPU of its own, chosen from the top of the process's allowed set so that the 2 x TP threads
    of a node take distinct CPUs (rank r, group i -> allowed[-(1 + r * 2 + i)]); the kernel's load balancer then moves other
    runnable tasks (a nice'd neighbour) off those CPUs, and the thread is never migrated. "fifo" (with pin) = the thread also
    asks for SCHED_FIFO priority 1 - it then preempts every CFS task on its CPU at once; needs CAP_SYS_NICE in the container
    (refused -> one warning, pinning only). Why: the pipelined step's two replay tails are CPU-launch-bound (K-3: ~1 500
    kernels + 2 500 runtime calls per group, CPU slack ~0), so a group thread that loses its CPU to a neighbour delays the
    tail's 85 all-reduces on every rank (K-6 §8.0: +20…+32 % at 163K under 100 nice -n 10 spinners, r25's path 0 %).
    Measured (second 4x RTX PRO 6000 host, kv16+FS): "pin" is WORSE - 163K quiet 5 519 / 5 480 / 5 487 ms (+2 % vs 5 376-5 391 as built),
    under 100 spinners 7 479 / 7 485 / 7 346 (+36 % vs +26.6 % as built), bits identical: a thread fixed to one CPU shares it
    with whatever lands there, an unpinned one is placed on an idle CPU at every wakeup. Profile under load: issuing one tail
    takes the thread 120-130 ms quiet (for ~50 ms of GPU work; hidden by the 65-77 ms CPU lead built in the SP phase) and
    147-166 ms under load while the lead shrinks to 5-31 ms, so the GPU window becomes the issue span and the other ranks wait
    in the tail's all-reduces for the starved rank. "fifo" alone (SCHED_FIFO 1, CAP_SYS_NICE, no pin) is worse
    still: quiet 5 494 / 5 506 / 5 540, under load 7 880 / 7 724 / 7 411 (+40 %) - the RT group threads preempt the server's own
    CFS threads (the NCCL proxy, the runner, the connector) on their CPUs, not just the spinners. Neither is a default; both stay
    as switches for measurements. The hardening that works is VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS=1 (see replay_graphs)."""
    raw = os.environ.get("VLLM_MOET_PREFILL_PIPE_AFFINITY", "")
    return frozenset(p for p in (s.strip() for s in raw.split(",")) if p in ("pin", "fifo"))


def tail_mode() -> str:
    """VLLM_MOET_PREFILL_PIPE_TAIL: when a group's replay tail (the layer-20 KV writes, the 128-row layers 20-39, the
    collapse) may start. "serial" (default) = as soon as the group's own last phase ended and the other group's turn
    is over (v1). "barrier" = the compute stream first waits for the other groups' last sequence-parallel phase as
    well, so both groups' SP layers are complete on the GPU before either tail starts and the two tails run back to
    back. "sync" = "barrier" plus a host wait for the compute stream to drain before the first tail is issued (the
    tails start with the CPU and the GPU together, in step across ranks as today's single chunk). Neither changes any
    arithmetic. Measured: the tails are launch-bound on the CPU (about 1 500 kernels + 2 500 other runtime calls per
    group), so neither variant can shorten them - see tools/dsv41_sm120/sp_pipeline docs."""
    mode = os.environ.get("VLLM_MOET_PREFILL_PIPE_TAIL", "serial")
    return mode if mode in ("serial", "barrier", "sync") else "serial"


def log_steps() -> int:
    try:
        return int(os.environ.get("VLLM_MOET_PREFILL_PIPE_LOG_STEPS", "0"))
    except ValueError:
        return 0


def host_timing() -> bool:
    """VLLM_MOET_PREFILL_PIPE_HOST_TIMING=1 (K-9 task 2): one log line per pipelined step with the runner thread's host time
    in the step's segments - `between` (the previous step's model call returned -> this step's group planning started: the
    sampler, the scheduler round trip, the whole-step input / metadata preparation), `plan` (the groups' metadata builds, the
    shared replay batch), `head` (planning done -> the model's dispatch reached: the embedding, the Engram hashes and rows),
    `issue` (the dispatch -> both group threads joined: the SP phase's and the tails' launches), and the time of the whole step's
    host path; every 20 pipelined steps a summary (mean / max per segment). Measures what CPU a step costs before its GPU work
    can start - the part a loaded host stretches (K-8: +11 ms per 163K step under 100 spinners)."""
    return os.environ.get("VLLM_MOET_PREFILL_PIPE_HOST_TIMING", "0") == "1"


_HOST = {"last_end": None, "plan_start": None, "plan_end": None, "rows": []}


def _host_mark(name: str) -> None:
    """Runner-thread timestamps of the step's segments (host_timing); a no-op otherwise."""
    if not host_timing():
        return
    _HOST[name] = time.perf_counter()


def _host_step_done(num_tokens: int, pipelined: bool, t_dispatch: float, t_end: float) -> None:
    if not host_timing():
        return
    ps, pe, last = _HOST.get("plan_start"), _HOST.get("plan_end"), _HOST.get("last_end")
    row = {
        "tokens": num_tokens, "pipelined": pipelined,
        "between": None if (last is None or ps is None) else (ps - last) * 1e3,
        "plan": None if (ps is None or pe is None) else (pe - ps) * 1e3,
        "head": None if pe is None else (t_dispatch - pe) * 1e3,
        "issue": (t_end - t_dispatch) * 1e3,
    }
    _HOST["last_end"] = t_end
    _HOST["plan_start"] = _HOST["plan_end"] = None
    rows = _HOST["rows"]
    rows.append(row)
    fmt = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
    logger.info(
        "prefill pipeline host timing: step of %d tokens (%s): between %s ms, plan %s, head %s, issue %s (rank %d)",
        num_tokens, "pipelined" if pipelined else "sequential", fmt(row["between"]), fmt(row["plan"]), fmt(row["head"]),
        fmt(row["issue"]), _tp_rank(),
    )
    if len(rows) % 20 == 0:
        recent = rows[-20:]
        parts = []
        for key in ("between", "plan", "head", "issue"):
            vals = [r[key] for r in recent if r[key] is not None]
            if vals:
                parts.append(f"{key} mean {sum(vals) / len(vals):.1f} / max {max(vals):.1f}")
        logger.info("prefill pipeline host timing: last %d steps (rank %d): %s", len(recent), _tp_rank(), "; ".join(parts))


def force_sequential() -> bool:
    """VLLM_MOET_PREFILL_PIPE_FORCE_SEQUENTIAL=1: the step's groups run one after the other on the runner thread
    (run_sequential: today's bits per group, no overlap) even when both are sequence parallel - the oracle of the
    pipelined path at identical step shapes (the mixed-step bits harness)."""
    return os.environ.get("VLLM_MOET_PREFILL_PIPE_FORCE_SEQUENTIAL", "0") == "1"


def replay_graphs() -> bool:
    """VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS=1: capture vLLM's decoder-replay CUDA graphs (the 128-row layers 20-39,
    DecoderReplayCudaGraphManager, sizes from --compilation-config decoder_replay_cudagraph_capture_sizes) once per
    row-group slot as well, so a pipelined group's replay tail runs as a graph (its replay batch lives in slot-private
    buffers; slot 0's graphs read slot 0's). Off (default): the groups' replay layers run eagerly, whatever slot 0 has.
    Measured (second 4x RTX PRO 6000 host, kv16+FS, sizes [128, 256]): the r26b hardening against host CPU load - a pipelined step's two
    replay tails (~1 400 launches each) take a thread 120-130 ms to issue for ~50 ms of GPU work, so under 100 nice -n 10
    spinners the 163K prefill is +26.6 % as built (the starved rank's tail delays the other ranks' 85 all-reduces) and +3.3 %
    with the graphs (tail GPU window 45 ms, all-reduces 7-9 ms, the CPU 10+ ms ahead on every rank); quiet +0.6 %; bits
    identical on long_agree x2 with prompt logprobs / agree / run-to-run (the per-layer dump hook cannot run inside a graph:
    layers 0-19 400/400 identical, layers 20-39 covered by the end-to-end gates only). Cost: KV 11.31 -> 10.3 GiB per GPU
    at 0.94 (the memory profiler books 2.33 GiB of graphs vs 1.34; really 0.98 vs 0.52). Needs the patcher of the same
    commit (the slot managers are rebuilt at every capture: vLLM's memory-profiling pass leaves a manager created during it
    on a throwaway graph pool)."""
    return os.environ.get("VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS", "0") == "1"


def graph_slots() -> list[int]:
    """The row-group slots that get replay CUDA graphs of their own (model_state.capture_inner_cudagraphs): 1..n with
    VLLM_MOET_PREFILL_PIPE=2 and VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS=1, else none; K-9: + the shared slot with
    VLLM_MOET_PREFILL_PIPE_SHARED_TAIL != 0 (the shared slot alone with SHARED_TAIL_GRAPHS=shared)."""
    if mode() != 2 or not replay_graphs():
        return []
    if shared_tail() == "0":
        return list(range(1, PipeRunner.NUM_GROUPS + 1))
    if shared_tail_graphs() == "shared":
        return [SHARED_SLOT]
    return list(range(1, PipeRunner.NUM_GROUPS + 1)) + [SHARED_SLOT]


def comm2_enabled() -> bool:
    """VLLM_MOET_PREFILL_PIPE_COMM2=1: the odd group's collectives (its boundary reduce-scatters / all-gathers on the comm
    stream, the indexer's all_gatherv on the compute stream, the all-reduces of its replay tail) go through a second NCCL
    communicator over the same TP ranks (PipeRunner.comm2, a PyNcclCommunicator built once on every rank at the first
    pipelined step), the even group's through the TP group's own. Why (K-6, the ws0_3 trace of K-5 read kernel by
    kernel): NCCL serializes every collective of one communicator in issue order, across streams - the compute stream's
    all_gatherv of group B's indexer top-k waits for group A's whole RS2 -> glue -> AG2 chain on the comm stream (6 per
    step, ~1.1 ms each = 8 ms of a 188-ms SP phase), and a late AG of one group delays the other group's next RS. Two
    communicators decouple the groups' chains; the bytes, the algorithm and the summation order of every collective are
    unchanged (same ranks, same topology), so the bits are today's. Off (default): one communicator, as r26.
    Also on with VLLM_MOET_PREFILL_PIPE_ORDER=comm2 | comm2prio (K-7)."""
    return os.environ.get("VLLM_MOET_PREFILL_PIPE_COMM2", "0") == "1" or order() in ("comm2", "comm2prio")


def kv_connector_policy() -> str:
    """VLLM_MOET_PREFILL_PIPE_KV_CONNECTOR: "allow" (default) = the pipeline activates as usual; "refuse" = when the
    server runs a KV transfer connector (vllm_config.kv_transfer_config is set: the production KV offload tier,
    KV_OFFLOAD_GIB > 0) the pipeline stays off - one loud log line at the first eligible step, today's 4 096-row path for
    every step. Background: in the production window of 2026-11-07 r26 (PIPE=2 at 8 192 tokens) measured +31 % at 163K
    with the offload connector, which no test-host arm had exercised; "refuse" is the safe default to ship until the
    pipelined step is validated with the connector on (K/BRIEF_K6.md section 0)."""
    policy = os.environ.get("VLLM_MOET_PREFILL_PIPE_KV_CONNECTOR", "allow")
    return policy if policy in ("allow", "refuse") else "allow"


def phase_graphs() -> str:
    """VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS (K-8): "0" (default) = every pipelined step issues its ~5 200 launches from the two
    group threads, as r26 / r26b. "1" = the whole sequence-parallel section of a full step (both 4 096-row groups: the shard,
    layers 0-19 with their RS -> mHC -> shard GEMMs -> AG chains on the comm stream, the first replay layer's KV write) is
    captured ONCE at engine start (capture_phase_graphs, from the runner's capture_model) as one CUDA graph across the compute,
    comm and aux streams - today's two-thread program recorded as a DAG, the NCCL kernels in today's deterministic issue order -
    and replayed with one cudaGraphLaunch per step; the replay hand-off (the rows gather, the 128-row layers 20-39 as K-7's
    per-slot graphs, the collapse) stays eager after it. The CPU leaves the SP phase: robustness to host load. Steps the graph
    does not serve take today's path, said once per reason (PhaseGraph.mismatch): partial groups, steps with decode rows, a group
    of several requests, a context beyond VLLM_MOET_PREFILL_PIPE_PHASE_GRAPH_MAX_CTX, a metadata structure other than the
    captured one, a missing replay batch. "verify" = "1" plus, every step, the eager two-thread section on the same static
    inputs and a torch.equal of every output tensor (one loud log line per step, counters in stats(); the tails continue
    from the eager outputs) - the bits oracle, since the per-layer dump hook cannot run inside a graph. Design and the
    graph-breaker inventory: K/k8/DESIGN_phase_graph.md.
    Measured (second 4x RTX PRO 6000 host, image r26b, connector kv16+FS, REPLAY_GRAPHS=1 [128,256]): the capture works on the real model (0.4 s,
    2 121 graph nodes, both of vLLM's capture passes); verify 86 steps x 4 ranks = 344 IDENTICAL / 0 MISMATCH, per-layer dumps 400/400
    vs the kv0 and r25 oracles, agree x3 identical. 163K 5 354 / 5 361 / 5 365 ms quiet = -1.0 % vs r26b (5 405-5 413), under 100
    nice -n 10 spinners +4.1 % (r26b +3.3...+4.8 %); with ORDER=comm2 inside the graph 5 345 / 5 338 / 5 327 (-1.3 %) and +1.9 % under
    load. Cost: 2.3 GiB of graph pool (2.64 booked) -> KV 10.3 -> 7.66 GiB at 0.94; with VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 the
    capture runs out of memory (caught: eager, loud). Why only -1 %: the SP phase was GPU-bound already (K-6: its head-of-line
    blocking is CTA dispatch, the persistent MoE kernels' exclusion and NCCL's per-communicator FIFO - GPU-side mechanics the graph
    keeps), so the 2 600 launches per group it removes were hidden behind the GPU; the exposed CPU under load is the runner's
    per-step work between the steps. Opt-in; the foundation for graph-level edits (node priorities on the comm
    branch, edge rewrites, the tails inside the graph)."""
    value = os.environ.get("VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS", "0")
    return value if value in ("0", "1", "verify") else "0"


def phase_graph_max_ctx() -> int:
    """VLLM_MOET_PREFILL_PIPE_PHASE_GRAPH_MAX_CTX: the sequence length (context + the step's rows) the phase graph is captured
    for. The indexer's prefill metadata sizes its logits buffer and K gather by the context (host integers baked into the graph),
    so the capture uses this maximum and a step whose context exceeds it runs eagerly; the kernels bound their work by the real
    extents on the device. Memory: ~2 KB of graph pool per token of context (1 024 rows x fp32 logits / 2)."""
    try:
        return int(os.environ.get("VLLM_MOET_PREFILL_PIPE_PHASE_GRAPH_MAX_CTX", "262144"))
    except ValueError:
        return 262144


# K-9: the slot of the shared replay batch (both groups' replay rows as one batch); slots 1..NUM_GROUPS are the groups'
SHARED_SLOT = 3


def shared_tail() -> str:
    """VLLM_MOET_PREFILL_PIPE_SHARED_TAIL (K-9): "0" (default) = every group of a pipelined step runs its own replay tail (the
    128-row layers 20-39 + the collapse) as r26 / r26b - two tails per step, each streaming every active expert's weights once
    (K-4 6.1: the 128-row fc1 / fc2 sit at the DRAM floor), 2 x 43 all-reduces, ~24 ms of GPU each. "1" = ONE tail per full
    pipelined step: after both groups' layer-20 KV write and rows gather, the second group's thread runs layers 20-39 + the
    collapse once on the concatenation of both groups' replay rows (256 rows = the shared replay batch of SHARED_SLOT, built by
    the runner as a pseudo-batch of the two groups' request slices; its graph is the 256-row one of the capture list), the
    first group's thread waits at the rendezvous (shared_replay) and takes its rows of the outputs; the scatter back to full
    rows, the collapse-norm, the aux states and the mtp copy stay per group. Bits = today's by construction: every kernel of a
    replay layer computes a row independently of the other rows at any M (the inventory in K/k9/DESIGN_shared_tail.md 2) except
    three ops whose arithmetic is a function of the row count - the TP all-reduces (NCCL's ring chunking per byte count), the
    router GEMM and the wo_a bmm (cuBLAS's per-shape kernel / split-K choice) - which the shared tail issues as one call per group
    row block (row_blocks: the same message / the same M as today), and the mHC pre-norm GEMM's split-k count, pinned to the
    group's row count (pin_tokens). "verify" = "1" plus, every shared step, today's two tails on the same inputs and a torch.equal
    of every output tensor per group (one log line per step, counters in stats(); the step continues from today's outputs, the KV
    cache holds today's writes) - the bits oracle on the real model, as K-8's PHASE_GRAPHS=verify. Steps the shared tail does not
    serve run today's tails, said once per reason: a group without a replay batch (prompt logprobs), REPLAY=stream, a shared
    batch without a graph while REPLAY_GRAPHS=1 (decode rows in group 0: 128 + d + 128 > the largest capture size), unequal mHC
    split counts.
    Measured (second 4x RTX PRO 6000 host, image r26b, connector kv16+FS, REPLAY_GRAPHS=1 [128,256], fc2 WS2=3): LEVEL B - verify eager
    96/96 steps IDENTICAL (24 x 4 ranks) with the shared tail's own per-layer dumps 320/320 chunk-layers x 5 states identical to the kv0
    and r25 dumps (layers 20-39, both groups), verify with the graphs 360/360 IDENTICAL (90 x 4 ranks) once the shared slot's capture
    runs inside shared_capture_scope (the first graph arm captured the unsplit 256-row all-reduces / GEMMs: 360/360 MISMATCH - a graph
    replays what it recorded); agree x3 / long x2 identical. Time: 163K 5 008 / 5 014 / 4 999 ms quiet vs r26b 5 371-5 391 (-6.9 %),
    19K 647 / 642 / 639 vs 676-686 (-5.5 %), under 100 nice -n 10 spinners +5.1 % (r26b +4.8 %). Cost: the shared slot's graph manager
    (profiling pass 2.73 vs 2.33 GiB booked, real 1.13 vs 0.98) -> KV 10.3 -> 9.9 GiB per GPU at 0.94 (5.77 M vs 6.01 M tokens)."""
    value = os.environ.get("VLLM_MOET_PREFILL_PIPE_SHARED_TAIL", "0")
    return value if value in ("0", "1", "verify") else "0"


def shared_tail_graphs() -> str:
    """VLLM_MOET_PREFILL_PIPE_SHARED_TAIL_GRAPHS (K-9, with SHARED_TAIL != 0 and REPLAY_GRAPHS=1): "all" (default) = the groups'
    slots keep their replay graphs (a step the shared tail cannot serve - decode rows in group 0 - still runs its tails as graphs)
    and the shared slot gets its own (capture sizes >= 2 x the replay window only: [128,256] -> [256]); "shared" = the shared
    slot alone (one manager less: ~0.23 GiB of graph pool real, ~0.5 GiB booked by the memory profiler = KV; the steps the
    shared tail cannot serve then run eager tails)."""
    value = os.environ.get("VLLM_MOET_PREFILL_PIPE_SHARED_TAIL_GRAPHS", "all")
    return value if value in ("all", "shared") else "all"


def shared_tail_light() -> bool:
    """VLLM_MOET_PREFILL_PIPE_SHARED_TAIL_LIGHT=1 (K-9 task 2, with SHARED_TAIL=1): when the runner can tell before the groups'
    metadata builds that the shared tail will serve the step (two groups whose request slices trim, REPLAY=serial, the summed
    kept rows within the shared slot's capture sizes, equal mHC split counts), each group's own replay batch is built LIGHT -
    its rows (the gather kernel) without the kept batch's two attention-metadata builds and forward context, which only
    today's per-group tail would read. Two metadata builds less per group per step on the runner thread (the per-step host
    path a loaded host stretches). Off (default): the groups' batches are built as today. If the shared plan then fails for a
    reason the precheck could not see, the groups' batches are rebuilt in full and today's tails run (loud once)."""
    return os.environ.get("VLLM_MOET_PREFILL_PIPE_SHARED_TAIL_LIGHT", "0") == "1"


_LIGHT = {"on": False}


def light_replay_batch() -> bool:
    """Model-state side (_prepare_replay_batch, K-9): True while the runner builds a group slot's metadata for a step the shared
    tail will serve - the slot's replay batch is then its rows only (see shared_tail_light)."""
    return _LIGHT["on"]


def slot_graph_config(vllm_config: Any, slot: int, window: int) -> Any:
    """The vllm config a slot's DecoderReplayCudaGraphManager is built with (model_state.capture_inner_cudagraphs): the groups'
    slots take the config as is; the shared slot takes a copy whose decoder_replay_cudagraph_capture_sizes keep only the sizes
    >= 2 x window (a shared batch is never smaller: two groups' kept rows), or None when no configured size qualifies (then the
    shared slot gets no graphs and the shared tail serves only eager steps). An empty (automatic) size list is kept as is."""
    if slot != SHARED_SLOT:
        return vllm_config
    from copy import copy

    compilation = copy(vllm_config.compilation_config)
    sizes = list(getattr(compilation, "decoder_replay_cudagraph_capture_sizes", None) or [])
    if sizes:
        kept = [s for s in sizes if s >= 2 * max(int(window), 1)]
        if not kept:
            return None
        compilation.decoder_replay_cudagraph_capture_sizes = kept
    cfg = copy(vllm_config)
    cfg.compilation_config = compilation
    return cfg


# "capturing" is True on every thread while the step graph is being captured (the patched sites that would otherwise behave
# differently under stream capture consult it); "request" drives the capture passes (None | "warmup" | "capture").
_PHASE: dict[str, Any] = {"capturing": False, "request": None}


def phase_graph_capturing() -> bool:
    """True while the phase graph is being captured: _moet_prefill_sp_step keeps the sequence-parallel path (it declines under
    stream capture otherwise, since vLLM's own graphs never carry it) and Engram's _finish_prefetch skips its wait on an event
    recorded outside the capture (the module waits for it on the compute stream before every launch instead)."""
    return _PHASE["capturing"]


# --------------------------------------------------------------------------------------- the group threads' placement
_PLACEMENT: dict[str, Any] = {"cpus": None, "warned": set(), "fifo_ok": None}


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - outside a distributed process (the unit test)
        return 0


def thread_cpus(num_groups: int, allowed: list[int] | None = None, rank: int | None = None) -> dict[int, int]:
    """The CPU each group thread is pinned to under VLLM_MOET_PREFILL_PIPE_AFFINITY=pin: group i of TP rank r takes
    allowed[-(1 + r * num_groups + i)] of the process's allowed CPUs (sorted), wrapping on a set too small for the node's
    2 x TP threads (then two threads may share a CPU, still away from the lowest-numbered CPUs where the IRQs land)."""
    if allowed is None:
        allowed = sorted(os.sched_getaffinity(0))
    if rank is None:
        rank = _tp_rank()
    n = len(allowed)
    return {i: allowed[-(1 + (rank * num_groups + i) % n)] for i in range(num_groups)}


def _place_thread(idx: int, num_groups: int) -> None:
    """Called on a group thread at its start (one thread per group per step): sched_setaffinity / SCHED_FIFO per
    affinity_policy; every refusal is logged once and the thread goes on as placed by the kernel."""
    policy = affinity_policy()
    if not policy:
        return
    if _PLACEMENT["cpus"] is None:
        _PLACEMENT["cpus"] = thread_cpus(num_groups)
        allowed = sorted(os.sched_getaffinity(0))
        logger.info(
            "vllm-moet sm_120 prefill pipeline: group threads of rank %d - VLLM_MOET_PREFILL_PIPE_AFFINITY=%s%s (%d allowed CPUs, %d-%d)",
            _tp_rank(), ",".join(sorted(policy)),
            f", pinned to CPUs {_PLACEMENT['cpus']}" if "pin" in policy else " (no pinning: the threads migrate)",
            len(allowed), allowed[0], allowed[-1],
        )
    if "pin" in policy:
        try:
            os.sched_setaffinity(0, {_PLACEMENT["cpus"][idx]})
        except OSError as exc:
            if "pin" not in _PLACEMENT["warned"]:
                _PLACEMENT["warned"].add("pin")
                logger.warning("vllm-moet sm_120 prefill pipeline: group thread sched_setaffinity refused (%s) - unpinned", exc)
    if "fifo" in policy:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(1))
            if _PLACEMENT["fifo_ok"] is None:
                _PLACEMENT["fifo_ok"] = True
                logger.info("vllm-moet sm_120 prefill pipeline: group threads run SCHED_FIFO priority 1")
        except (OSError, PermissionError) as exc:
            if _PLACEMENT["fifo_ok"] is None:
                _PLACEMENT["fifo_ok"] = False
                logger.warning(
                    "vllm-moet sm_120 prefill pipeline: group thread SCHED_FIFO refused (%s; CAP_SYS_NICE needed) - CFS, pinning only",
                    exc,
                )


# ------------------------------------------------------------------------------------------------- per-thread state
_tls = threading.local()
_PREPARE_SLOT = 0  # the slot the runner thread is building metadata for (0 = the whole step)
_PENDING: list["PipeGroup"] | None = None  # the step's groups, published by the runner around the model call
_STATS = {
    "steps": 0, "pipelined": 0, "sequential": 0, "mixed": 0, "today": 0,
    # K-8: pipelined steps that ran the phase graph / fell back to eager, verify-mode steps compared / found different
    "phase_graph_steps": 0, "phase_graph_eager": 0, "phase_graph_verified": 0, "phase_graph_mismatch": 0,
    # K-9: pipelined steps whose tails ran as one shared tail / as today's two tails although the switch is on, verify-mode
    # steps compared / found different
    "shared_tail_steps": 0, "shared_tail_today": 0, "shared_tail_verified": 0, "shared_tail_mismatch": 0,
}


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
    last_phase_event: Any = None  # the event recorded at the group's last comm-phase end (the tail gate, tail_mode)
    prefill_rows_from: int = 0  # the group's first row that belongs to a prefilling request (decode rows precede it)
    req_lo: int = 0  # the group's first / last request index in the step's batch (the phase graph's key, K-8)
    req_hi: int = 0
    comm_held: list = field(default_factory=list)  # tensors the open phase reads on the comm stream, released at its join
    shared_plan: "SharedPlan | None" = None  # K-9: the step's shared replay tail, planned by the runner (the same object on every group)


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
        # K-9: the step's shared replay tail (set by run_groups from the groups' plan when the step is pipelined), else None
        self.shared: SharedTail | None = None
        # the static SM split with GREEN_TAIL=caller (_run_groups): the groups' replay sections run as replay phases on the caller's
        # stream (= replay_stream), whatever VLLM_MOET_PREFILL_PIPE_REPLAY says
        self.tail_on_caller = False
        for g in groups:
            g.step = self
            g.phases_done = 0
            g.phase_open = False
            g.pin = None
            g.active_context = g.forward_context
            g.stream = compute_stream
            g.phase_stream = None
            g.phase_kind = ""
            g.last_phase_event = None
            g.comm_held = []

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
            _place_thread(g.idx, len(self.groups))  # K-7: VLLM_MOET_PREFILL_PIPE_AFFINITY, a no-op by default
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
                g.comm_held = []
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
    if g.phase_kind == "comm":
        g.last_phase_event = ev
    # the static SM split's tail phase (GREEN_TAIL=caller) is not one of the sequence-parallel phases the groups are ordered by:
    # a group may have a replay section when an earlier group has none
    counted = not (g.phase_kind == "replay" and step.tail_on_caller)
    g.phase_open = False
    g.phase_stream = None
    g.phase_kind = ""
    if counted:
        g.phases_done += 1
        step.check_order(g)
    step.yield_(g)
    step.compute_stream.wait_event(ev)
    # the join is enqueued: the tensors the chain read on the comm stream may go back to the compute stream's pool (every
    # later compute-stream kernel is stream-ordered after the join) - see record_for_comm
    g.comm_held.clear()


def _tail_gate() -> None:
    """VLLM_MOET_PREFILL_PIPE_TAIL=barrier|sync at the start of a group's replay tail: the compute stream waits for the
    last sequence-parallel phase of every group that comes after this one (its boundary chain on the comm stream),
    so both groups' SP layers are complete before the first tail starts; "sync" also drains the compute stream on the
    host. The last group has nothing to wait for. A no-op in "serial" mode and outside a pipelined group."""
    mode = tail_mode()
    if mode == "serial":
        return
    g = current_group()
    if g is None or g.step is None or not g.step.pipelined:
        return
    step = g.step
    later = [h for h in step.groups if h.idx > g.idx]
    if not later:
        return
    for h in later:
        if h.last_phase_event is not None:
            step.compute_stream.wait_event(h.last_phase_event)
    if mode == "sync":
        step.compute_stream.synchronize()


def replay_begin(*inputs: torch.Tensor | None) -> None:
    """Start of a group's replay section (the 128-row layers 20-39 + collapse): a launch-bound chain of small kernels
    and all-reduces that would block the other group's compute if it sat on the compute stream (v2 of §4.2: +1.7 %).
    It runs on a stream of its own, R, after everything the compute stream has so far; the handoffs inside
    (yield_point, once per layer) keep R current. `inputs` are the hc states the section reads that were born on the
    compute stream: kept out of its reuse until R is done with them. A no-op unless VLLM_MOET_PREFILL_PIPE_REPLAY=stream
    (the tail gate of VLLM_MOET_PREFILL_PIPE_TAIL applies in every replay mode)."""
    _tail_gate()
    if replay_mode() != "stream":
        g = current_group()
        if g is None or g.step is None or not g.step.tail_on_caller:  # (the static SM split's tail on the caller's stream)
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
    the other direction is covered by phase_begin's wait). Done by holding a reference until the phase's join is enqueued
    (phase_end), not by record_stream: the lifetime is the same (the block is freed on the compute stream after the join,
    every later compute-stream kernel is ordered behind it), but under the phase graph's capture (K-8) the caching allocator
    defers a recorded block's free to the capture's end - 80 reduce-scatter inputs x 40 MB = 3.2 GB of graph memory (the
    model's own mhc.py keeps its side-stream inputs the same way, for the same reason)."""
    g = current_group()
    if g is not None and g.phase_open and g.step is not None:
        g.comm_held.append(x)
    return x


# ------------------------------------------------------------------------------- device collectives (MK-4, moet_device_comm.py)
def device_comm_mode() -> str:
    """VLLM_MOET_PREFILL_PIPE_DEVICE_COMM: "0" (default) = NCCL as today; "1" = the SP reduce-scatters / all-gathers of a pipelined
    step's comm phases from the device (a 64-CTA kernel per collective, PCIe P2P pushes, NCCL's summation order from a probed
    table - moet_device_comm.py); "verify" = both, compared every time, NCCL's result consumed."""
    value = os.environ.get("VLLM_MOET_PREFILL_PIPE_DEVICE_COMM", "0")
    return value if value in ("0", "1", "verify") else "0"


_DEVICE_COMM: Any = None  # the module, False when the import failed (said once)


def _device_comm():
    global _DEVICE_COMM
    if _DEVICE_COMM is None:
        try:
            from vllm.v1.worker import moet_device_comm

            _DEVICE_COMM = moet_device_comm
        except Exception as exc:  # noqa: BLE001
            _DEVICE_COMM = False
            logger.warning("vllm-moet sm_120 prefill pipeline: VLLM_MOET_PREFILL_PIPE_DEVICE_COMM=%s but moet_device_comm is not "
                           "importable (%r) - NCCL's collectives stay", device_comm_mode(), exc)
    return _DEVICE_COMM


def device_reduce_scatter(x: torch.Tensor, fallback: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor | None:
    """The model's reduce-scatter hook (_moet_sp_reduce_scatter): the device result inside an open comm phase of a pipelined
    group when VLLM_MOET_PREFILL_PIPE_DEVICE_COMM != 0, else None (the caller runs NCCL's)."""
    if device_comm_mode() == "0":
        return None
    dc = _device_comm()
    if not dc:
        return None
    return dc.reduce_scatter(x, fallback)


def device_all_gather(t: torch.Tensor, fallback: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """The model's all-gather hook (sp_all_gather sites, sp_moe_gather_sm120): the gathered [4c, K] rows from the device inside
    an open device phase, else fallback(t) (NCCL's all-gather)."""
    if device_comm_mode() == "0":
        return fallback(t)
    dc = _device_comm()
    if not dc:
        return fallback(t)
    return dc.all_gather(t, fallback)


def device_all_gather_many(ts: list[torch.Tensor], fallback: Callable[[torch.Tensor], torch.Tensor]) -> list[torch.Tensor]:
    """MK-5 (b): several tensors of one phase as ONE device gather instance (sp_moe_gather_sm120's e4m3 rows + packed small rows);
    element-wise fallback(t) when the device path does not take them (VLLM_MOET_PREFILL_PIPE_DEVICE_COMM_MULTI=0: the per-tensor path)."""
    if device_comm_mode() == "0":
        return [fallback(t) for t in ts]
    dc = _device_comm()
    if not dc:
        return [fallback(t) for t in ts]
    return dc.all_gather_many(ts, fallback)


def device_all_gatherv(local: torch.Tensor, sizes: list[int], out: torch.Tensor, fallback: Callable[[torch.Tensor], torch.Tensor]) -> bool:
    """MK-5 (c): the sparse-attention indexer's variable-size top-k gather (PrefillRowShard.exchange_topk) through the device path on the
    pipeline's comm stream; True = `out` written, False = the caller runs NCCL's all_gatherv (VLLM_MOET_PREFILL_PIPE_DEVICE_COMM_IDX=0,
    outside a pipelined step, or a shape the device path does not take)."""
    if device_comm_mode() == "0":
        return False
    dc = _device_comm()
    if not dc:
        return False
    return dc.all_gatherv(local, sizes, out, fallback)


# ----------------------------------------------------------------------------------- the static SM split (green contexts)
def green_sms() -> int:
    """VLLM_MOET_PREFILL_PIPE_GREEN=N (default 0 = off; N = the SMs asked for the comm side, 32 or 48 of a 188-SM GPU): the
    pipelined step's two streams on two CUDA green contexts - a static split of the device's SMs, created ONCE per process
    (cuDevSmResourceSplitByCount: one group of >= N SMs, rounded up by the driver to its co-scheduling alignment of 8, and the
    remainder; a destroyed green context does not give its ~4 MiB back, so the split is never rebuilt). The comm stream of a
    pipelined step (the boundary chains: NCCL's reduce-scatter / all-gather, the mHC kernels, the shard GEMMs, the quant)
    becomes a stream of the N-SM context, its compute stream (attention, MoE, the GEMMs, the replay tails) a stream of the
    other one, both wrapped as torch.cuda.ExternalStream: every kernel a thread launches while such a stream is current runs
    on that partition's SMs only, on the tensors of the process's primary context - ATen, cuBLAS, CUTLASS, the JIT kernels,
    driver-API launches and NCCL alike; nothing is moved or copied. The step joins the caller's stream at both ends
    (GreenSplit.run).
    Why: on one device the chain's kernels and the compute kernels queue for the same SMs (a 4-CTA NCCL kernel waits for a
    compute kernel's wave, the compute stream loses SMs to the glue); a static split removes the queue at the price of the SM
    share - the compute side's kernels run on 188 - N SMs for good.
    What changes: which SMs run a kernel. The same kernels, in the same order, on the same rows - bits by construction. With
    the switch off nothing below is created, loaded or polled.
    The rules a split imposes, all kept here:
      * every launch of the step goes through its side's green stream - the group threads' current stream (PipeStep) and
        the attention modules' aux streams, swapped for streams of the compute partition during the step
        (VLLM_MOET_PREFILL_PIPE_GREEN_AUX, default 1); a stream of the primary context is NOT confined: it takes whatever
        SMs are free;
      * the fast prefill MoE's grids are sized by the partition, not the device (green_moe_ctas): with the FC2 sum items a
        grid built for the device's SMs waits for CTAs that never become resident on the partition and runs into its own
        2-s watchdog - a trapped launch, the end of the process;
      * the replay tails: by default each group's replay section (the rows gather, the 128-row layers 20-39) is a replay
        phase on the caller's stream (VLLM_MOET_PREFILL_PIPE_GREEN_TAIL=caller, green_tail_on_caller) - the tails and their
        CUDA graphs are today's and nothing of them depends on the split; with GREEN_TAIL=compute the tails stay on the
        compute partition's stream and the row-group slots' replay graphs are captured on it
        (VLLM_MOET_PREFILL_PIPE_GREEN_GRAPHS, default 1; shared_capture_scope) - a captured kernel keeps the context it was
        recorded in, whatever stream launches the graph;
      * every kernel of a step is launched once on its partition before the first request
        (VLLM_MOET_PREFILL_PIPE_GREEN_WARMUP, default 1: one dummy pipelined step at engine start, _green_warmup_step) -
        a kernel's first launch loads its code and waits for the device.
    Not combined (the split stays off, said once): PHASE_GRAPHS != 0, DEVICE_COMM != 0, REPLAY=stream, the fast prefill MoE
    off (the grids of any other MoE path are not capped).
    One thing to know when measuring: after a CUDA graph is launched on a stream of a green context, torch.cuda.synchronize()
    (the device) returns before the graph's kernels have finished - whichever stream the graph was captured on; stream order
    on the launching stream, events recorded behind the launch and that stream's own synchronize() hold. The module joins
    streams by events only, so nothing here depends on it; a timing loop that brackets such launches with device
    synchronisation would read short.
    Runtime A/B (VLLM_MOET_RUNTIME_AB=1 and its control file, as the fast prefill MoE's switches): key PREFILL_PIPE_GREEN =
    0 (the step on the caller's stream and the pipeline's own comm stream, as without the switch) | N (the start-up value:
    on the split); the file is looked at at most once per 100 ms, at the dispatch of a pipelined forward, and the state holds
    for that forward as a whole - a flip never lands inside a step. Only a process started with GREEN=N has the key: both
    states' streams exist from start-up and nothing is created at a flip. What was captured stays as captured: with the
    default tail mode no captured launch depends on the key; with GREEN_TAIL=compute the slots' replay graphs keep the
    compute partition in both states (GREEN_GRAPHS=0 leaves them on the caller's stream in both).
    Measured so far on a model of the layer pair only (4x RTX PRO 6000 SE; two streams + NCCL as here, 20-layer / 200-layer
    runs): 10.1 / 9.96 ms per layer pair -> 8.5-8.7 / 8.28 with N = 32 at 400 W, 10.0 -> 8.5 under a 300 W cap, 10.4 -> 10.1 J
    per pair; the served layer's kernels on 156 SMs: attention +19 %, wo_a +24 %, the CUTLASS MXFP8 GEMMs +7...+10 %, fc1
    +10 %, fc2 +9 % (grids capped), each bit-identical. The served forward is denser than that model: the number that counts
    is the in-situ one."""
    try:
        n = int(os.environ.get("VLLM_MOET_PREFILL_PIPE_GREEN", "0"))
    except ValueError:
        return 0
    return n if n > 0 else 0


def green_graphs() -> bool:
    """VLLM_MOET_PREFILL_PIPE_GREEN_GRAPHS (default 1; with GREEN_TAIL=compute): the slots' replay graphs are captured on the compute
    partition's stream."""
    return os.environ.get("VLLM_MOET_PREFILL_PIPE_GREEN_GRAPHS", "1") != "0"


def green_tail_on_caller() -> bool:
    """VLLM_MOET_PREFILL_PIPE_GREEN_TAIL (caller | compute, default caller): where the replay tails of a split step run.
    "caller" = each group's replay section (the rows gather, the 128-row layers 20-39) as a replay phase on the caller's stream
    - the stream layout of VLLM_MOET_PREFILL_PIPE_REPLAY=stream with the caller's stream as R, without its per-layer handoffs -
    so the tails and their graphs are exactly today's (a stream of the primary context, captured as without the switch;
    nothing of them depends on the runtime key) and only the sequence-parallel section is split. "compute" = the tails on the
    compute partition's stream like the rest of the group's program, their graphs captured on that stream (GREEN_GRAPHS).
    With the shared tail (its rendezvous does not run inside a replay phase) always "compute"."""
    value = os.environ.get("VLLM_MOET_PREFILL_PIPE_GREEN_TAIL", "caller")
    return value != "compute" and shared_tail() == "0"


def green_aux() -> bool:
    """VLLM_MOET_PREFILL_PIPE_GREEN_AUX (default 1): the attention modules' aux streams are streams of the compute partition
    during a split step (0: they stay streams of the primary context - not confined)."""
    return os.environ.get("VLLM_MOET_PREFILL_PIPE_GREEN_AUX", "1") != "0"


def green_warmup() -> bool:
    """VLLM_MOET_PREFILL_PIPE_GREEN_WARMUP (default 1): one dummy pipelined step on the split at engine start."""
    return os.environ.get("VLLM_MOET_PREFILL_PIPE_GREEN_WARMUP", "1") != "0"


def green_warmup_ctx() -> int:
    """VLLM_MOET_PREFILL_PIPE_GREEN_WARMUP_CTX (default 32768): the sequence length of the warm-up step's dummy request (the
    step's rows are its last 2 x group_rows(); the kernels are the same at any context, a short one touches fewer KV blocks)."""
    try:
        return max(int(os.environ.get("VLLM_MOET_PREFILL_PIPE_GREEN_WARMUP_CTX", "32768")), 0)
    except ValueError:
        return 32768


# "split": the process's GreenSplit; "failed": why there is none (said once); "on": the effective state of the runtime A/B key
# (True = the start-up state); the counters of the pipelined forwards that ran on / off the split while the switch is set, of
# the MoE calls whose grids were capped and of the slot captures made on the compute partition; the control-file reader's state
_GREEN: dict[str, Any] = {
    "split": None, "failed": None, "on": True,
    "forwards": 0, "rows": 0, "forwards_off": 0, "rows_off": 0, "moe_capped": 0, "graph_scopes": 0, "warmup": None,
    "ab": None, "next_poll": 0.0, "sig": None, "warned": set(), "stats_mtime": None,
}

_GREEN_MOE_MODULE = "b12x_moe_prefill_fast_sm120"
_GREEN_MOE_ENV = "VLLM_MOET_B12X_MOE_PREFILL_FAST"


def _green_log(msg: str) -> None:
    """The control-file contract's line format: `[<module tag>, device <n>] <text>` on stderr, one line per rank."""
    import sys

    try:
        dev: Any = torch.cuda.current_device()
    except Exception:  # noqa: BLE001
        dev = "?"
    print(f"[vllm-moet prefill pipeline, device {dev}] {msg}", file=sys.stderr, flush=True)


def _green_warn_once(key: str, msg: str) -> None:
    if key not in _GREEN["warned"]:
        _GREEN["warned"].add(key)
        _green_log("WARNING: " + msg)


def _green_api():
    """The CUDA driver's green-context entry points through ctypes (libcuda.so.1; driver API 12.4 or newer). Loaded at the first
    split only. CUdevResource is 144 bytes in the 13.0 headers: type, 92 bytes the driver owns, the union whose `sm` member
    (smCount, minSmPartitionSize, smCoscheduledAlignment) is all that is read here."""
    import ctypes as C

    cu = C.CDLL("libcuda.so.1")

    class SmRes(C.Structure):
        _fields_ = [("smCount", C.c_uint), ("minSmPartitionSize", C.c_uint), ("smCoscheduledAlignment", C.c_uint)]

    class ResU(C.Union):
        _fields_ = [("sm", SmRes), ("_oversize", C.c_ubyte * 48)]

    class DevRes(C.Structure):
        _fields_ = [("type", C.c_int), ("_pad", C.c_ubyte * 92), ("u", ResU)]

    if C.sizeof(DevRes) != 144:
        raise RuntimeError(f"CUdevResource layout: {C.sizeof(DevRes)} bytes, expected 144")
    vp = C.c_void_p
    sigs = {
        "cuDeviceGet": [C.POINTER(C.c_int), C.c_int],
        "cuDeviceGetDevResource": [C.c_int, C.POINTER(DevRes), C.c_int],
        "cuDevSmResourceSplitByCount": [C.POINTER(DevRes), C.POINTER(C.c_uint), C.POINTER(DevRes), C.POINTER(DevRes), C.c_uint,
                                        C.c_uint],
        "cuDevResourceGenerateDesc": [C.POINTER(vp), C.POINTER(DevRes), C.c_uint],
        "cuGreenCtxCreate": [C.POINTER(vp), vp, C.c_int, C.c_uint],
        "cuGreenCtxStreamCreate": [C.POINTER(vp), vp, C.c_uint, C.c_int],
        "cuStreamGetGreenCtx": [vp, C.POINTER(vp)],
        "cuGetErrorName": [C.c_int, C.POINTER(C.c_char_p)],
    }
    for name, args in sigs.items():
        f = getattr(cu, name)  # AttributeError = a driver without green contexts: the split stays off, said once
        f.argtypes, f.restype = args, C.c_int
    return C, cu, DevRes, vp


class GreenSplit:
    """The process's static SM split: partition A (the comm side, >= n SMs) and partition B (the compute side, the remaining
    SMs), one green context each; the streams `comm` (A) and `compute`, `aux[0..2]` (B) as torch.cuda.ExternalStream. `sms` maps
    a raw stream handle to its partition's SM count (what a launch on that stream has to size a persistent grid by). Created
    once (green_split) and never destroyed: the process's end frees it."""

    AUX = 3  # the attention modules' aux streams (one list of three, shared by every layer)

    def __init__(self, n: int, device: torch.device, comm_priority: int = 0) -> None:
        C, cu, DevRes, vp = _green_api()

        def ck(rc: int, what: str) -> None:
            if rc:
                name = C.c_char_p()
                cu.cuGetErrorName(rc, C.byref(name))
                raise RuntimeError(f"CUDA driver error {rc} ({name.value!r}) in {what}")

        index = device.index if device.index is not None else torch.cuda.current_device()
        with torch.cuda.device(index):
            torch.cuda.current_stream()  # the primary context exists and is current on this thread
            dev = C.c_int()
            ck(cu.cuDeviceGet(C.byref(dev), int(index)), "cuDeviceGet")
            res = DevRes()
            ck(cu.cuDeviceGetDevResource(dev.value, C.byref(res), 1), "cuDeviceGetDevResource")  # 1 = the SM resource
            self.device_sms = int(res.u.sm.smCount)
            out, rem, groups = (DevRes * 1)(), DevRes(), C.c_uint(1)
            ck(cu.cuDevSmResourceSplitByCount(out, C.byref(groups), C.byref(res), C.byref(rem), 0, int(n)),
               "cuDevSmResourceSplitByCount")
            self.na = int(out[0].u.sm.smCount)
            self.nb = int(rem.u.sm.smCount) if rem.type == 1 else 0
            if groups.value != 1 or self.na <= 0 or self.nb <= 0:
                raise RuntimeError(f"a split of {n} of {self.device_sms} SMs leaves no second partition ({self.na} + {self.nb})")
            self._handles: list[Any] = []  # the green contexts and raw streams, kept alive with the object

            def context(key: str, resource: Any) -> Any:
                desc, ctx = vp(), vp()
                ck(cu.cuDevResourceGenerateDesc(C.byref(desc), C.byref(resource), 1), f"cuDevResourceGenerateDesc {key}")
                ck(cu.cuGreenCtxCreate(C.byref(ctx), desc, dev.value, 1), f"cuGreenCtxCreate {key}")  # 1 = with a default stream
                self._handles.append(ctx)
                return ctx

            def stream(ctx: Any, priority: int, what: str) -> torch.cuda.Stream:
                raw = vp()
                ck(cu.cuGreenCtxStreamCreate(C.byref(raw), ctx, 1, int(priority)), f"cuGreenCtxStreamCreate {what}")  # 1 = non-blocking
                owner = vp()
                ck(cu.cuStreamGetGreenCtx(raw, C.byref(owner)), f"cuStreamGetGreenCtx {what}")
                if owner.value != ctx.value:
                    raise RuntimeError(f"the {what} stream is not a stream of its green context")
                self._handles.append(raw)
                return torch.cuda.ExternalStream(raw.value, device=index)

            ctx_a, ctx_b = context("A", out[0]), context("B", rem)
            self.comm = stream(ctx_a, comm_priority, "comm")
            self.compute = stream(ctx_b, 0, "compute")
            self.aux = [stream(ctx_b, 0, f"aux{i}") for i in range(self.AUX)]
        self.sms: dict[int, int] = {int(self.comm.cuda_stream): self.na, int(self.compute.cuda_stream): self.nb}
        self.sms.update({int(s.cuda_stream): self.nb for s in self.aux})

    def run(self, runner: Any, step: "PipeStep", fn: Callable[[PipeGroup], Any]) -> list[Any]:
        """One pipelined step on the split (the step was built with `compute` / `comm` as its streams): the compute partition
        starts behind everything the caller's stream holds (the head), the attention's aux streams are the compute partition's
        for the step, and the caller's stream continues behind the step's last compute kernel - the two joins that the one
        compute stream of today's path gets from stream order."""
        from vllm.utils.torch_utils import current_stream

        caller = current_stream()
        head = torch.cuda.Event()
        head.record(caller)
        self.compute.wait_event(head)
        swapped = []
        if green_aux():
            for lst in getattr(runner, "aux_stream_lists", ()):
                swapped.append((lst, list(lst)))
                for i in range(min(len(lst), len(self.aux))):
                    lst[i] = self.aux[i]
        try:
            return step.run(fn)
        finally:
            for lst, orig in swapped:
                lst[:] = orig
            done = torch.cuda.Event()
            done.record(self.compute)
            caller.wait_event(done)


def _green_unsupported() -> str | None:
    """Why the split cannot be combined with this process's configuration (None = it can)."""
    if mode() != 2:
        return "VLLM_MOET_PREFILL_PIPE != 2"
    if phase_graphs() != "0":
        return "VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS != 0 (the phase graph is captured across primary-context streams)"
    if device_comm_mode() != "0":
        return "VLLM_MOET_PREFILL_PIPE_DEVICE_COMM != 0 (the device collectives were not run on a partition in situ)"
    if replay_mode() != "serial":
        return "VLLM_MOET_PREFILL_PIPE_REPLAY=stream"
    if os.environ.get(_GREEN_MOE_ENV, "0") != "1":
        return f"{_GREEN_MOE_ENV} != 1 (only the fast prefill MoE's grids are capped to a partition)"
    return None


def green_split(device: torch.device | None = None) -> GreenSplit | None:
    """The process's split, created at the first call with the switch on (the slots' graph capture at engine start, else the
    first pipelined forward); None when the switch is off, the configuration does not combine, or the creation failed - said
    once, every pipelined step then runs as without the switch."""
    n = green_sms()
    if n == 0:
        return None
    if _GREEN["split"] is not None or _GREEN["failed"] is not None:
        return _GREEN["split"]
    why = _green_unsupported()
    sp = None
    if why is None:
        try:
            if device is None:
                device = torch.device("cuda", torch.cuda.current_device())
            sp = GreenSplit(n, device, _comm_priority())
            if not _green_install_moe_caps():
                sp, why = None, "the fast prefill MoE module is not importable or takes no grid caps: its grids cannot be sized by a partition"
        except Exception as exc:  # noqa: BLE001 - a driver without green contexts, a refused split: today's path, loudly
            sp, why = None, f"{type(exc).__name__}: {str(exc)[:300]}"
    if sp is None:
        _GREEN["failed"] = why
        logger.warning(
            "vllm-moet sm_120 prefill pipeline: VLLM_MOET_PREFILL_PIPE_GREEN=%d requested but the SM split is OFF (%s) - every "
            "pipelined step runs today's path (the caller's stream and the pipeline's comm stream, as without the switch)", n, why,
        )
        return None
    _GREEN["split"] = sp
    logger.info(
        "vllm-moet sm_120 prefill pipeline: static SM split created once for this process (rank %d): comm partition %d SMs "
        "(asked %d), compute partition %d SMs, device %d SMs; comm stream priority %d, %d aux streams on the compute partition; "
        "replay tails %s, aux streams swapped: %s, warm-up step: %s, runtime A/B key PREFILL_PIPE_GREEN: %s "
        "(VLLM_MOET_PREFILL_PIPE_GREEN)",
        _tp_rank(), sp.na, n, sp.nb, sp.device_sms, _comm_priority(), len(sp.aux),
        "on the caller's stream (today's graphs)" if green_tail_on_caller() else
        f"on the compute partition (their graphs captured there: {green_graphs()})", green_aux(), green_warmup(),
        "on" if os.environ.get("VLLM_MOET_RUNTIME_AB", "0") == "1" else "off (VLLM_MOET_RUNTIME_AB != 1)",
    )
    return sp


def _green_current_sms() -> int | None:
    """The SM count of the partition the calling thread's current stream belongs to; None on a stream of the primary context.
    Asks torch itself, not vllm's per-thread bookkeeping: the launch that follows goes to torch's current stream, and a stream
    made current through another door (a `with stream:` of the accelerator API) is not in that bookkeeping."""
    sp = _GREEN["split"]
    if sp is None:
        return None
    return sp.sms.get(int(torch.cuda.current_stream().cuda_stream))


def green_moe_ctas(rows: int, module: Any = None) -> tuple[int, int]:
    """The CTAs per SM the fast prefill MoE's FC1 / FC2 grids are capped to on a partition. The kernels size a grid as
    (occupancy of the kernel) x (SMs of the device) so that every CTA is resident; on a partition of S SMs that holds only for
    occupancy x S. VLLM_MOET_PREFILL_PIPE_GREEN_MOE_CTAS="a,b" gives both numbers; the default follows the as-built modes on
    this GPU: FC1 1 for a one-CTA-per-SM mode at or above its row threshold (the warpgroup modes 9-11, the producer-warp modes
    2-7, the persistent launch), else 2; FC2 2 for the producer-warp mode 3 and for the kernel without producer warps, else 1.
    A number too small only costs CTAs and is always safe; one too large leaves the grid above the partition's capacity, which
    kernels without cross-CTA waits do not mind and FC2 with the sum items answers with its watchdog - hence 1 for every FC2
    mode that was not run on a partition."""
    raw = os.environ.get("VLLM_MOET_PREFILL_PIPE_GREEN_MOE_CTAS", "")
    if raw:
        try:
            a, b = (int(v) for v in raw.split(","))
            if a >= 1 and b >= 1:
                return a, b
        except ValueError:
            pass
        _green_warn_once("moe_ctas", f"VLLM_MOET_PREFILL_PIPE_GREEN_MOE_CTAS={raw!r} is not two integers >= 1 - the default rule applies")

    def env(name: str, default: int) -> int:
        try:
            return int(os.environ.get(f"{_GREEN_MOE_ENV}_{name}", default))
        except ValueError:
            return default

    ws1 = env("WS1", 0)
    if module is not None and hasattr(module, "runtime_switches"):
        try:
            ws1 = int(module.runtime_switches().get("WS1", ws1))  # the fast MoE's own runtime A/B key may have changed it
        except Exception:  # noqa: BLE001 - an extension without the entry: the environment value
            pass
    ws, ws2 = env("WS", 0), env("WS2", 0)
    if ws >= 20 and rows >= env("WS_MIN_ROWS", 0):
        fc1 = 1
    elif (2 <= ws1 <= 7 or 9 <= ws1 <= 11) and rows >= env("WS1_MIN_ROWS", 256):
        fc1 = 1
    else:
        fc1 = 2
    return fc1, 2 if ws2 in (0, 3) else 1


def _green_install_moe_caps() -> bool:
    """Wrap the fast prefill MoE's entry point once - `moe_prefill_fast` of its Python module, which the b12x launcher looks up
    on the module at every call - so that a call issued while a green stream is current passes grid caps of the partition's
    SM count x green_moe_ctas (the module's own grid1 / grid2 arguments; a smaller cap of the caller or of the module's
    environment wins). On any other stream the wrapper passes the call through untouched. False = the module is not there or
    its entry point has no grid1 / grid2 parameters."""
    import sys

    path = os.environ.get(f"{_GREEN_MOE_ENV}_DIR", "/opt/vllm-moet/dsv41_sm120/b12x_moe_prefill_fast")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        module = __import__(_GREEN_MOE_MODULE)
    except ImportError:
        return False
    orig = module.moe_prefill_fast
    if getattr(orig, "_moet_green_caps", False):
        return True
    import inspect

    try:  # where grid1 / grid2 sit when a caller gives them by position (behind hidden, ids): read from the entry point itself
        names = list(inspect.signature(orig).parameters)
        positions = tuple((names.index(name) - 2, name) for name in ("grid1", "grid2"))
    except (TypeError, ValueError):
        return False  # an entry point without grid caps cannot be made safe on a partition

    def env_cap(name: str) -> int:
        try:
            return int(os.environ.get(f"{_GREEN_MOE_ENV}_{name}", 0))
        except ValueError:
            return 0

    def moe_prefill_fast(hidden, ids, *args, **kwargs):
        sms = _green_current_sms()
        if sms is not None:
            fc1, fc2 = green_moe_ctas(int(ids.shape[0]), module)
            args = list(args)
            for (pos, name), cap in zip(positions, (sms * fc1, sms * fc2)):
                cur = args[pos] if pos < len(args) else kwargs.get(name)
                cur = env_cap(name.upper()) if cur is None else int(cur)
                new = cap if cur <= 0 else min(cur, cap)
                if pos < len(args):
                    args[pos] = new
                else:
                    kwargs[name] = new
            _GREEN["moe_capped"] += 1
        return orig(hidden, ids, *args, **kwargs)

    moe_prefill_fast._moet_green_caps = True  # type: ignore[attr-defined]
    moe_prefill_fast.__wrapped__ = orig  # type: ignore[attr-defined]
    moe_prefill_fast.__doc__ = orig.__doc__
    module.moe_prefill_fast = moe_prefill_fast
    return True


def _green_stats_line(reason: str) -> None:
    st = _GREEN
    _green_log(
        f"activation counters ({reason}, forward {st['forwards'] + st['forwards_off']}; forwards / rows): split {st['forwards']} / "
        f"{st['rows']}, unsplit {st['forwards_off']} / {st['rows_off']}; MoE calls with capped grids {st['moe_capped']}; "
        f"switches now PREFILL_PIPE_GREEN={green_sms() if st['on'] else 0}"
    )


def green_apply_runtime(values: dict) -> list[str]:
    """Apply the control file's object: PREFILL_PIPE_GREEN = 0 | the start-up value; absent = the start-up value; out of range
    = the current state kept (one WARNING); an unknown key of this module's namespace = ignored (one WARNING per key); other
    modules' keys ignored silently. Returns the change descriptions (also logged, one line per change)."""
    st = _GREEN
    n = green_sms()
    sp = st["split"]
    want = True
    for key, val in values.items():
        if not isinstance(key, str) or not key.startswith("PREFILL_PIPE_"):
            continue
        if key != "PREFILL_PIPE_GREEN":
            _green_warn_once("key:" + key, f"runtime A/B: unknown key {key} of this module - ignored")
            continue
        v = val if isinstance(val, int) and not isinstance(val, bool) else None
        if v == 0:
            want = False
        elif v is not None and (v == n or (sp is not None and v == sp.na)):
            want = True
        else:
            want = st["on"]
            _green_warn_once("val:" + key, f"runtime A/B: {key}={val!r} out of range (0, or the start-up value {n}: the split "
                             f"is created once) - kept {n if st['on'] else 0}")
    changes = []
    if want != st["on"]:
        changes.append(f"PREFILL_PIPE_GREEN {n if st['on'] else 0} -> {n if want else 0}")
        st["on"] = want
    for c in changes:
        _green_log("runtime A/B: " + c)
    return changes


def _green_ab_poll() -> None:
    """The control-file reader of this module (the contract of the fast prefill MoE's runtime switch, its own copy): at most one
    os.stat per 100 ms, the JSON re-read only when (mtime, size) changed; a missing or malformed file keeps the state."""
    st = _GREEN
    now = time.monotonic()
    if now < st["next_poll"]:
        return
    st["next_poll"] = now + 0.1
    path = os.environ.get("VLLM_MOET_RUNTIME_AB_FILE", "/dev/shm/vllm_moet_ab.json")
    try:
        s = os.stat(path)
    except OSError:
        _green_warn_once("missing", f"runtime A/B: control file {path} missing or unreadable - the current state is kept until it appears")
        return
    sig = (s.st_mtime_ns, s.st_size)
    if sig == st["sig"]:
        return
    st["sig"] = sig
    try:
        with open(path, encoding="utf-8") as fh:
            values = json.load(fh)
        if not isinstance(values, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        _green_warn_once("malformed", f"runtime A/B: control file {path} malformed ({exc}) - the current state is kept")
        return
    st["warned"].discard("missing")
    st["warned"].discard("malformed")
    green_apply_runtime(values)


def green_active() -> GreenSplit | None:
    """The split the pipelined forward being dispatched runs on, or None (the switch off, no split, or the runtime A/B key at
    0). Called once per pipelined forward on the runner thread, before the step's streams are chosen: the control-file poll
    (VLLM_MOET_RUNTIME_AB=1) and the on-demand counters line (`touch` the stats file: one os.stat per forward) sit here."""
    if green_sms() == 0:
        return None
    sp = green_split()
    if sp is None:
        return None
    st = _GREEN
    if st["ab"] is None:
        st["ab"] = os.environ.get("VLLM_MOET_RUNTIME_AB", "0") == "1"
    if st["ab"]:
        _green_ab_poll()
    try:
        mtime = os.stat(os.environ.get("VLLM_MOET_RUNTIME_STATS_FILE", "/dev/shm/vllm_moet_stats")).st_mtime_ns
    except OSError:
        mtime = None
    if mtime is not None and mtime != st["stats_mtime"]:
        st["stats_mtime"] = mtime
        _green_stats_line("on demand")
    return sp if st["on"] else None


def _green_capture_enter() -> Any:
    """shared_capture_scope, for every slot: the calling thread captures the slot's replay graphs on the compute partition's
    stream (GREEN_GRAPHS) - behind everything its own stream holds; the MoE launches recorded there carry the partition's grid
    caps. Returns the stream to go back to (None = nothing was changed)."""
    if green_sms() == 0 or not green_graphs() or green_tail_on_caller():
        return None
    sp = green_split()
    if sp is None:
        return None
    from vllm.utils.torch_utils import current_stream

    prev = current_stream()
    ev = torch.cuda.Event()
    ev.record(prev)
    sp.compute.wait_event(ev)
    torch.cuda.set_stream(sp.compute)
    _GREEN["graph_scopes"] += 1
    return prev


def _green_capture_exit(prev: Any) -> None:
    if prev is None:
        return
    sp = _GREEN["split"]
    ev = torch.cuda.Event()
    ev.record(sp.compute)
    prev.wait_event(ev)
    torch.cuda.set_stream(prev)


def _green_warmup_step(runner: Any, profile_only: bool) -> None:
    """GREEN_WARMUP, from the runner's capture_model (capture_phase_graphs' call, after the slots' replay graphs; the real pass
    only): one dummy pipelined step of 2 x group_rows() rows runs eagerly on the split - the phase graph's dummy step
    (PipeRunner.capture_phase_graph), its warm-up pass alone, at a short context - so that every kernel of a step has been
    launched once on its partition's stream before the first request. A failure is logged with its traceback and changes
    nothing else: the first request then loads the kernels as it goes."""
    global _RUNNER
    if profile_only or force_sequential():
        return
    if kv_connector_policy() == "refuse" and _kv_connector_configured(runner):
        return
    if green_split(runner.device) is None:
        return
    t0 = time.time()
    try:
        _RUNNER = PipeRunner(runner)
        _RUNNER.capture_phase_graph(runner, passes=("warmup",), max_ctx=green_warmup_ctx())
        torch.cuda.synchronize()
        # the step's activation blocks go back to the device: the runner books the free-memory difference around its capture
        # pass as CUDA graph memory, and a dummy step's cached blocks are not graphs
        torch.cuda.empty_cache()
        _GREEN["warmup"] = "done"
        logger.info(
            "vllm-moet sm_120 prefill pipeline: SM split warm-up - one dummy pipelined step of %d rows ran on the split in %.1f s "
            "(every kernel launched once on its partition; MoE calls with capped grids so far %d)",
            group_rows() * PipeRunner.NUM_GROUPS, time.time() - t0, _GREEN["moe_capped"],
        )
    except Exception as exc:  # noqa: BLE001 - never break the engine start: the first request warms the kernels instead
        _GREEN["warmup"] = f"failed: {type(exc).__name__}: {str(exc)[:200]}"
        logger.exception(
            "vllm-moet sm_120 prefill pipeline: SM SPLIT WARM-UP FAILED (%s) - the split stays on, the first pipelined request "
            "launches each kernel for the first time on its partition", _GREEN["warmup"],
        )
    finally:
        _PHASE["request"] = None
        _PHASE["capturing"] = False


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


def run_groups(
    groups: list[PipeGroup], fn: Callable[[PipeGroup], Any], pipelined: bool, phase: "PhaseHooks | None" = None
) -> list[Any]:
    """Run fn(group) for every group: pipelined on two threads (both groups sequence parallel), else one after the
    other. Called on the runner thread inside the model's forward. `phase` (K-8) = the model's split of fn into the
    sequence-parallel section and the rest: with VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS the section of a full step runs as
    the captured graph (or is being captured, during capture_phase_graphs); fn stays the eager path of every other step."""
    runner = _RUNNER
    assert runner is not None, "prefill pipeline: groups without a runner"
    if host_timing() and _PHASE["request"] is None:  # a dummy step at engine start is not a step of the served traffic
        t_dispatch = time.perf_counter()
        try:
            return _run_groups(runner, groups, fn, pipelined, phase)
        finally:
            _host_step_done(sum(g.num_tokens for g in groups), pipelined and not force_sequential(), t_dispatch,
                            time.perf_counter())
    return _run_groups(runner, groups, fn, pipelined, phase)


def _run_groups(
    runner: "PipeRunner", groups: list[PipeGroup], fn: Callable[[PipeGroup], Any], pipelined: bool,
    phase: "PhaseHooks | None",
) -> list[Any]:
    from vllm.utils.torch_utils import current_stream

    if pipelined and force_sequential():
        pipelined = False
        if _STATS["steps"] == 0:
            logger.info("vllm-moet sm_120 prefill pipeline: VLLM_MOET_PREFILL_PIPE_FORCE_SEQUENTIAL=1 - the groups run one "
                        "after the other (the bits oracle of the pipelined path)")
    # the static SM split (VLLM_MOET_PREFILL_PIPE_GREEN): the step's two streams are the partitions' for this forward as a whole
    green = green_active() if pipelined and green_sms() and _PHASE["request"] in (None, "warmup") else None
    if green is None:
        step = PipeStep(
            groups, current_stream(), runner.comm_stream, runner.row_switch, runner.device, pipelined,
            replay_stream=getattr(runner, "replay_stream", None),
        )
    else:
        tail_on_caller = green_tail_on_caller()
        step = PipeStep(
            groups, green.compute, green.comm, runner.row_switch, runner.device, pipelined,
            replay_stream=current_stream() if tail_on_caller else getattr(runner, "replay_stream", None),
        )
        step.tail_on_caller = tail_on_caller
    if pipelined:
        step.shared = SharedTail.for_step(groups)  # K-9: None unless the runner planned a shared replay tail for this step
    if _PHASE["request"] is not None:
        # a capture pass of capture_phase_graphs (the dummy step): warm-up = the whole step eagerly as today, capture = the
        # section under stream capture, the rest eagerly
        if not pipelined or phase is None:
            raise RuntimeError(
                f"prefill pipeline: the phase graph's dummy step is not pipelined (pipelined={pipelined}, hooks={phase is not None})"
            )
        if _PHASE["request"] == "warmup":
            return step.run(fn) if green is None else green.run(runner, step, fn)
        return _phase_capture_step(runner, groups, phase)
    _STATS["steps"] += 1
    if pipelined and phase is not None and phase_graphs() != "0":
        graph = runner.phase_graph
        why = "no graph captured" if graph is None else graph.mismatch(runner, groups)
        if why is None:
            _STATS["pipelined"] += 1
            return _phase_graph_step(runner, groups, phase, graph)
        _STATS["phase_graph_eager"] += 1
        runner._warn_once(
            "phase_graph:" + why.split(":")[0],
            "prefill pipeline: a pipelined step runs eagerly, not as the phase graph (%s)%s", why,
            "" if graph is not None else (
                f" - capture failed: {runner.phase_graph_failed}" if runner.phase_graph_failed else " - nothing was captured"
            ),
        )
    if pipelined:
        _STATS["pipelined"] += 1
        if _STATS["pipelined"] == 1:
            logger.info(
                "vllm-moet sm_120 prefill pipeline: steps of > %d tokens run as %d row groups one layer apart, "
                "the sequence-parallel collectives of one under the compute of the other (VLLM_MOET_PREFILL_PIPE=2)",
                group_rows(),
                len(groups),
            )
        if green is not None:
            _GREEN["forwards"] += 1
            _GREEN["rows"] += sum(g.num_tokens for g in groups)
            if _GREEN["forwards"] == 1:
                logger.info(
                    "vllm-moet sm_120 prefill pipeline: pipelined steps run on the static SM split - the comm stream on %d SMs, "
                    "the compute stream on %d (VLLM_MOET_PREFILL_PIPE_GREEN=%d)", green.na, green.nb, green_sms(),
                )
            outputs = green.run(runner, step, fn)
        else:
            if green_sms():
                _GREEN["forwards_off"] += 1
                _GREEN["rows_off"] += sum(g.num_tokens for g in groups)
            outputs = step.run(fn)
        if device_comm_mode() != "0" and _device_comm():
            _device_comm().step_done()  # the watchdog record (a trapped kernel = a failed step, loudly) + the verify counters
        return outputs
    _STATS["sequential"] += 1
    if _STATS["sequential"] == 1:
        logger.info(
            "vllm-moet sm_120 prefill pipeline: a step whose groups are not all sequence parallel runs them one "
            "after the other (today's bits, no overlap) - sizes %s",
            [g.num_tokens for g in groups],
        )
    return step.run_sequential(fn)


# ------------------------------------------------------------------------------------- the shared replay tail (K-9)
@dataclass
class SharedPlan:
    """The runner's plan of a step's shared replay tail (PipeRunner._plan_shared): the groups' kept row counts (the row blocks
    of the 256-row batch, in group order), the row count the mHC pre-norm GEMM's split-k is pinned to, the verify flag."""

    counts: list[int]
    pin: int
    verify: bool
    capture_window: int = 0  # > 0 only in shared_capture_scope: split any op of >= 2 x window rows into window-sized blocks

    @property
    def total(self) -> int:
        return sum(self.counts)

    @property
    def blocks(self) -> list[tuple[int, int]]:
        out, lo = [], 0
        for c in self.counts:
            out.append((lo, lo + c))
            lo += c
        return out


@dataclass
class SharedTail:
    """The rendezvous state of one pipelined step's shared replay tail (PipeStep.shared): the inputs each group's thread
    brought (the 7 replay inputs, full rows), the outputs the running thread produced for every group (full rows, the group's
    replay rows filled), the row blocks. Thread-local while the tail runs (_tls.shared): row_blocks / pin_tokens / the dump hook
    read it."""

    plan: SharedPlan
    groups: list[PipeGroup]
    inputs: list[tuple | None]
    outputs: list[tuple | None]

    @staticmethod
    def for_step(groups: list[PipeGroup]) -> "SharedTail | None":
        plan = groups[0].shared_plan if groups else None
        if plan is None or any(g.shared_plan is not plan for g in groups):
            return None
        return SharedTail(plan, list(groups), [None] * len(groups), [None] * len(groups))

    @property
    def total(self) -> int:
        return self.plan.total

    @property
    def blocks(self) -> list[tuple[int, int]]:
        return self.plan.blocks

    @property
    def pin(self) -> int:
        return self.plan.pin


def current_shared() -> SharedTail | None:
    return getattr(_tls, "shared", None)


class _SharedScope:
    """The calling thread runs the shared replay tail: _tls.shared set for the splits, the pin and the dump hook."""

    def __init__(self, st: SharedTail) -> None:
        self.st = st
        self.prev = None

    def __enter__(self):
        self.prev = getattr(_tls, "shared", None)
        self.prev_pin = _mhc_warmup()._MOET_SP_PIN_TOKENS
        _tls.shared = self.st
        return self

    def __exit__(self, *exc):
        _tls.shared = self.prev
        _mhc_warmup()._MOET_SP_PIN_TOKENS = self.prev_pin  # the layers' own pin calls inside were redirected (pin_tokens)
        return False


class shared_capture_scope:
    """Model-state side (capture_inner_cudagraphs, K-9): the shared slot's replay graphs are CAPTURED inside a shared scope,
    so that the per-group split of the three row-count-dependent ops (row_blocks: the all-reduces, the router GEMM, the wo_a
    bmm) and the mHC pin (pin_tokens) are recorded into the graph - a graph replays what was captured, and the capture runs
    at engine start, outside any step's rendezvous. Every captured size is split into blocks of `window` rows (the groups'
    kept rows: 256 = 128 + 128). A no-op for the groups' slots. (The first graph arm of K-9 captured the shared slot
    without this: 256-row all-reduces / GEMMs in the graph, eager verify IDENTICAL but graph verify MISMATCH on every step.)"""

    def __init__(self, slot: int, window: int) -> None:
        self.active = slot == SHARED_SLOT and window > 0
        self.plan = SharedPlan(counts=[int(window), int(window)], pin=int(window), verify=False, capture_window=int(window)) \
            if self.active else None
        self.scope = None
        self.green_prev = None  # the static SM split: the stream to go back to after a capture on the compute partition

    def __enter__(self):
        # every slot (the groups' and the shared one): with VLLM_MOET_PREFILL_PIPE_GREEN the graphs are captured on the compute
        # partition's stream (_green_capture_enter; None and no effect with the switch off)
        self.green_prev = _green_capture_enter()
        if self.active:
            st = SharedTail(self.plan, [], [None] * len(self.plan.counts), [None] * len(self.plan.counts))
            self.scope = _SharedScope(st)
            self.scope.__enter__()
        return self

    def __exit__(self, *exc):
        if self.scope is not None:
            self.scope.__exit__(*exc)
        _green_capture_exit(self.green_prev)
        self.green_prev = None
        return False


def row_blocks(num_rows: int) -> list[tuple[int, int]]:
    """The row blocks an op of `num_rows` rows is split into: inside the shared replay tail, for a tensor of the tail's row count,
    one block per group (the same rows / the same M each group's own tail would have had); [(0, num_rows)] otherwise. For the ops
    whose arithmetic depends on the row count - the TP all-reduce (NCCL's chunking), the router GEMM and the wo_a bmm (cuBLAS's
    kernel / split-K choice): the call sites iterate over the blocks."""
    st = current_shared()
    if st is None:
        return [(0, num_rows)]
    w = st.plan.capture_window
    if w > 0:
        # the shared slot's graph capture: every op of >= 2 x window rows (the capture sizes) in window-sized blocks
        if num_rows < 2 * w:
            return [(0, num_rows)]
        return [(i, min(i + w, num_rows)) for i in range(0, num_rows, w)]
    if num_rows != st.total:
        return [(0, num_rows)]
    return st.blocks


def pin_tokens(num_tokens: int | None) -> int | None:
    """The mHC pre-norm GEMM's split-k pin (_moet_sp_pin): inside the shared replay tail a layer that would release the pin
    (None: the replay layers are not sequence parallel) pins it to the group's row count instead, so the 256-row GEMM sums the
    k-slices the 128-row one did. Unchanged otherwise."""
    st = current_shared()
    if st is not None and num_tokens is None:
        return st.pin
    return num_tokens


def _split_all_reduce(orig: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    def all_reduce(input_: torch.Tensor) -> torch.Tensor:
        blocks = row_blocks(input_.shape[0]) if input_.dim() >= 1 else [(0, 0)]
        if len(blocks) <= 1:
            return orig(input_)
        x = input_.contiguous()
        return torch.cat([orig(x[lo:hi]) for lo, hi in blocks], dim=0)

    all_reduce._moet_split = True  # type: ignore[attr-defined]
    return all_reduce


def _install_ar_split() -> bool:
    """K-9: the TP group's device communicator all-reduces one group row block at a time inside the shared replay tail (the
    NCCL message of today's 128-row tail - the ring's chunk map and so an element's reduction order follow the byte count);
    a no-op call everywhere else. Installed on the communicator instance (every path of CudaCommunicator.all_reduce - pynccl,
    the custom / FlashInfer all-reduces - goes through it), once per process."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
    except Exception as exc:  # noqa: BLE001 - outside a distributed process (the unit test installs its own)
        logger.warning("vllm-moet sm_120 prefill pipeline: shared tail - no TP group to split the all-reduces on (%r)", exc)
        return False
    dc = getattr(tp, "device_communicator", None)
    if dc is None or tp.world_size < 2:
        return False
    if getattr(dc.all_reduce, "_moet_split", False):
        return True
    dc.all_reduce = _split_all_reduce(dc.all_reduce)
    logger.info(
        "vllm-moet sm_120 prefill pipeline: shared replay tail - the TP all-reduces inside it run per group row block (rank %d of %d)",
        tp.rank_in_group, tp.world_size,
    )
    return True


def _run_replay_with_batch(replay: Any, batch: Any, inputs: tuple) -> tuple:
    """DecoderReplayLayers._run on an explicit replay batch (today's tail of one group): the layers on the batch's rows, as
    its graph or eagerly, the outputs scattered into full rows (zeros elsewhere)."""
    from vllm.forward_context import override_forward_context

    rows = batch.rows
    with override_forward_context(batch.forward_context):
        if batch.run_graph is not None:
            row_outputs = batch.run_graph(rows, inputs)
        else:
            row_outputs = replay.run_layers(*(None if t is None else t.index_select(0, rows) for t in inputs))
    num_tokens = inputs[0].shape[0]
    return tuple(out.new_zeros((num_tokens, *out.shape[1:])).index_copy_(0, rows, out) for out in row_outputs)


def _scatter_shared_outputs(st: SharedTail, group_rows: list[torch.Tensor], row_outputs: tuple) -> list[tuple]:
    outs: list[tuple] = []
    for (lo, hi), rows, inp in zip(st.blocks, group_rows, st.inputs):
        assert inp is not None
        num_tokens = inp[0].shape[0]
        outs.append(tuple(
            out.new_zeros((num_tokens, *out.shape[1:])).index_copy_(0, rows, out[lo:hi]) for out in row_outputs
        ))
    return outs


def _run_shared_layers(replay: Any, st: SharedTail, shared_batch: Any, group_rows: list[torch.Tensor], manager: Any) -> tuple:
    """The replay layers once on both groups' rows: the 256-row inputs assembled from the groups' full-row tensors by their
    replay rows (into the shared slot's static graph inputs, or a concatenation for the eager layers), run under the shared
    batch's forward context inside the shared scope (the splits, the pin, the dump per group)."""
    from vllm.forward_context import get_forward_context, override_forward_context

    with _SharedScope(st), override_forward_context(shared_batch.forward_context):
        if shared_batch.run_graph is not None:
            if manager is None:
                raise RuntimeError("prefill pipeline: the shared replay batch has a graph but the shared slot has no graph manager")
            for k, buf in enumerate(manager.input_buffers):
                for (lo, hi), rows, inp in zip(st.blocks, group_rows, st.inputs):
                    assert inp is not None
                    t = inp[k]
                    if t is None:
                        raise RuntimeError(f"prefill pipeline: shared tail - replay input {k} is None, the graph needs it")
                    torch.index_select(t, 0, rows, out=buf[lo:hi])
            num_tokens = get_forward_context().batch_descriptor.num_tokens
            outputs = manager.breakable_cg_runner(*(buf[:num_tokens] for buf in manager.input_buffers))
            return tuple(out[: st.total] for out in outputs)
        cat_inputs = []
        for k in range(len(st.inputs[0])):
            parts = [inp[k] for inp in st.inputs]  # type: ignore[index]
            if any(p is None for p in parts):
                cat_inputs.append(None)
                continue
            cat_inputs.append(torch.cat([p.index_select(0, rows) for p, rows in zip(parts, group_rows)], dim=0))
        return replay.run_layers(*cat_inputs)


def _compare_outputs(a: tuple, b: tuple) -> list[str]:
    """The names of the output tensors that differ (torch.equal), each with the size of the difference: the share of
    differing elements, max |a - b|, max |b| (an ulp-level numerics difference vs a wrong-rows bug read differently)."""
    names = ["hidden_states", "pre_mix"] + [f"aux[{i}]" for i in range(max(len(a), len(b)))]
    if len(a) != len(b):
        return [f"structure {len(a)} vs {len(b)} outputs"]
    out = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape or x.dtype != y.dtype:
            out.append(f"{names[i]} shape/dtype {tuple(x.shape)}/{x.dtype} vs {tuple(y.shape)}/{y.dtype}")
            continue
        if torch.equal(x, y):
            continue
        xf, yf = x.float(), y.float()
        ne = int((x != y).sum().item())
        out.append(f"{names[i]} ({ne}/{x.numel()} el = {ne / max(x.numel(), 1):.2%}, max|d| {(xf - yf).abs().max().item():.3g}, "
                   f"max|ref| {yf.abs().max().item():.3g}, rows {int((x != y).reshape(x.shape[0], -1).any(dim=1).sum().item())})")
    return out


def shared_replay(replay: Any, *inputs: torch.Tensor | None) -> tuple | None:
    """Model side (K-9), in _moet_sp_decoder_replay_rest after the rows gather, instead of replay._run: the calling group's
    rows of one replay tail for both groups, or None (no shared tail planned for this step, not a pipelined step: the caller
    runs its own tail as today). The first thread to arrive leaves its inputs and parks until its outputs exist (the other
    thread ran the tail and finished or yielded; a finished partner without outputs = error, never a hang); the second runs
    the layers once for both groups (_run_shared_layers), scatters the outputs per group and returns its own. SHARED_TAIL=verify:
    today's two tails on the same inputs afterwards, torch.equal per output tensor, the step continues from today's outputs."""
    g = current_group()
    if g is None or g.step is None or not g.step.pipelined or g.step.shared is None:
        return None
    step = g.step
    st = step.shared
    st.inputs[g.idx] = inputs
    n = len(step.groups)
    if any(st.inputs[i] is None for i in range(n)):
        # not the last to arrive: hand the CPU over and wait for the tail's outputs
        while st.outputs[g.idx] is None:
            others_done = all(step.finished[h.idx] for h in step.groups if h.idx != g.idx)
            if others_done or step.errors:
                raise RuntimeError(
                    f"prefill pipeline: shared tail - group {g.idx} waited for its outputs but the other group finished without "
                    f"producing them (errors: {list(step.errors)})"
                )
            step.yield_(g)
        return st.outputs[g.idx]
    # the last to arrive runs the tail for every group
    batches = getattr(replay, "_moet_replay_batches", None)
    if batches is None:
        raise RuntimeError("prefill pipeline: shared tail - the replay layers carry no per-slot batches (patcher mismatch)")
    group_batches = [batches.get(h.slot) for h in step.groups]
    shared_batch = batches.get(SHARED_SLOT)
    if shared_batch is None or any(b is None for b in group_batches):
        raise RuntimeError("prefill pipeline: shared tail planned but a replay batch is missing (slots "
                           f"{[h.slot for h in step.groups]} + {SHARED_SLOT}: {[b is not None for b in group_batches]}, "
                           f"{shared_batch is not None})")
    group_rows = [b.rows for b in group_batches]
    counts = [int(r.shape[0]) for r in group_rows]
    if counts != st.plan.counts or int(shared_batch.rows.shape[0]) != st.total:
        raise RuntimeError(f"prefill pipeline: shared tail - the replay batches changed since the plan: {counts} / "
                           f"{int(shared_batch.rows.shape[0])} vs {st.plan.counts}")
    runner = _RUNNER
    manager = None
    if runner is not None:
        manager = getattr(runner.model_state, "_moet_pipe_slot_cudagraphs", {}).get(SHARED_SLOT)
    row_outputs = _run_shared_layers(replay, st, shared_batch, group_rows, manager)
    outputs = _scatter_shared_outputs(st, group_rows, row_outputs)
    _STATS["shared_tail_steps"] += 1
    if _STATS["shared_tail_steps"] == 1:
        logger.info(
            "vllm-moet sm_120 prefill pipeline: the replay layers of a pipelined step run ONCE for both groups (%d + %d rows, %s; "
            "VLLM_MOET_PREFILL_PIPE_SHARED_TAIL=%s)", counts[0], counts[-1],
            "the shared slot's graph" if shared_batch.run_graph is not None else "eager", shared_tail(),
        )
    if st.plan.verify:
        today = [_run_replay_with_batch(replay, b, inp) for b, inp in zip(group_batches, st.inputs)]  # type: ignore[arg-type]
        diffs = {h.idx: _compare_outputs(outputs[h.idx], today[h.idx]) for h in step.groups}
        _STATS["shared_tail_verified"] += 1
        bad = {i: d for i, d in diffs.items() if d}
        if bad:
            _STATS["shared_tail_mismatch"] += 1
            logger.error(
                "vllm-moet sm_120 prefill pipeline: SHARED TAIL VERIFY step %d MISMATCH - shared vs today's tails differ in %s "
                "(the step continues from today's outputs; mismatches so far %d of %d)",
                _STATS["shared_tail_verified"], bad, _STATS["shared_tail_mismatch"], _STATS["shared_tail_verified"],
            )
        else:
            logger.info(
                "vllm-moet sm_120 prefill pipeline: shared tail verify step %d IDENTICAL (%d output tensors per group, torch.equal; "
                "mismatches so far %d)", _STATS["shared_tail_verified"], len(outputs[0]), _STATS["shared_tail_mismatch"],
            )
        outputs = today
    for h in step.groups:
        st.outputs[h.idx] = outputs[h.idx]
    return st.outputs[g.idx]


# ------------------------------------------------------------------------------------------- the phase graph (K-8)
@dataclass
class SectionState:
    """What the sequence-parallel section of a group hands to the rest of its tail: the hc state after the last SP layer
    (hidden_states = the full-row partial sum the next boundary would reduce-scatter, residual / post_mix / res_mix /
    pre_mix = this rank's row shard), the aux hidden states the drafter reads (by layer), the group's input ids as the
    section left them, and whether the first replay layer's KV was written (the replay batch exists). In graph mode the
    tensors live in the graph pool and are copied into private buffers after every launch."""

    hidden_states: torch.Tensor
    residual: torch.Tensor
    post_mix: torch.Tensor
    res_mix: torch.Tensor
    pre_mix: torch.Tensor
    aux_hidden_by_layer: dict[int, torch.Tensor]
    wrote_kv: bool
    input_ids: torch.Tensor | None = None

    def named_tensors(self) -> list[tuple[str, torch.Tensor]]:
        out = [("hidden_states", self.hidden_states), ("residual", self.residual), ("post_mix", self.post_mix),
               ("res_mix", self.res_mix), ("pre_mix", self.pre_mix)]
        out += [(f"aux[{k}]", self.aux_hidden_by_layer[k]) for k in sorted(self.aux_hidden_by_layer)]
        if self.input_ids is not None:
            out.append(("input_ids", self.input_ids))
        return out

    def static_copy(self) -> "SectionState":
        """Private buffers of the same shapes (allocated outside any capture)."""
        return SectionState(
            torch.empty_like(self.hidden_states), torch.empty_like(self.residual), torch.empty_like(self.post_mix),
            torch.empty_like(self.res_mix), torch.empty_like(self.pre_mix),
            {k: torch.empty_like(v) for k, v in self.aux_hidden_by_layer.items()}, self.wrote_kv,
            None if self.input_ids is None else torch.empty_like(self.input_ids),
        )

    def copy_from(self, other: "SectionState") -> None:
        for (name, dst), (name2, src) in zip(self.named_tensors(), other.named_tensors()):
            assert name == name2, (name, name2)
            dst.copy_(src)
        self.wrote_kv = other.wrote_kv


@dataclass
class PhaseHooks:
    """The model's split of a group's tail for the phase graph: inputs(g) = the group's rows of the step's tensors
    (hidden states, positions, input ids, Engram hashes, Engram mask), section(g, *inputs) -> SectionState (the SP
    section: shard, layers 0-19, the first replay layer's KV), rest(g, positions, input_ids, state) -> the group's output
    (the replay hand-off, the 128-row layers, the collapse)."""

    inputs: Callable[[PipeGroup], tuple]
    section: Callable[..., SectionState]
    rest: Callable[..., Any]


# host values of the attention metadata that may be SMALLER at a step than at the capture (the kernels bound their work by
# the device-side extents; the captured value sized the buffers); every other host value must be equal
_META_LE_HINTS = ("seq_len", "seqlen", "total_seq", "query_len", "max_seq", "num_blocks", "context_len")


class _MetaSnapshot:
    """The tensors and host values reachable from a group's forward context (attn_metadata, the per-layer slot mappings,
    is_padding, the batch descriptor), flattened by path. At capture time it is the graph's static metadata; at every step
    plan() compares a fresh context with it: same structure, host values equal (or <= where they are extents), and for each
    device tensor the copy to make (none when it is the same storage = a persistent builder buffer; into the captured
    tensor, or a prefix of it, otherwise - the fresh per-step allocations of the indexer's chunk plan)."""

    ROOTS = ("attn_metadata", "slot_mapping", "is_padding", "batch_descriptor")

    def __init__(self, ctx: Any) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self.values: dict[str, Any] = {}
        self._seen: set[int] = set()
        for root in self.ROOTS:
            self._visit(root, getattr(ctx, root, None), 0)
        self._seen = set()

    @staticmethod
    def _is_plain(obj) -> bool:
        return isinstance(obj, (bool, int, float, str, bytes, type(None)))

    def _visit(self, path: str, obj: Any, depth: int) -> None:
        import dataclasses as _dc

        if self._is_plain(obj):
            if obj is not None:
                self.values[path] = obj
            return
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                self.tensors[path] = obj
            elif obj.numel() <= 65536:
                self.values[path] = obj.tolist()  # host tensors (seq_lens_cpu, query_start_loc_cpu) are host values
            else:
                self.values[path] = ("cpu-tensor", tuple(obj.shape))
            return
        if isinstance(obj, np.ndarray):
            self.values[path] = obj.tolist() if obj.size <= 65536 else ("ndarray", obj.shape)
            return
        if depth > 10 or id(obj) in self._seen:
            return
        self._seen.add(id(obj))
        type_name = type(obj).__name__
        if type_name.endswith(("Config", "Builder", "Backend", "Layer", "Module", "Manager", "Group", "Spec")):
            return  # configuration / machinery an object may point back to: not the step's data
        if isinstance(obj, dict):
            for key in sorted(obj, key=str):
                self._visit(f"{path}.{key}", obj[key], depth + 1)
        elif isinstance(obj, (list, tuple)):
            self.values[path + ".len"] = len(obj)
            for i, v in enumerate(obj):
                self._visit(f"{path}[{i}]", v, depth + 1)
        elif _dc.is_dataclass(obj) and not isinstance(obj, type):
            for f in _dc.fields(obj):
                self._visit(f"{path}.{f.name}", getattr(obj, f.name, None), depth + 1)
        elif isinstance(obj, (torch.nn.Module, torch.cuda.Stream, torch.cuda.Event)) or callable(obj):
            return
        elif hasattr(obj, "__dict__"):
            for key in sorted(vars(obj)):
                if not key.startswith("__"):
                    self._visit(f"{path}.{key}", vars(obj)[key], depth + 1)

    @staticmethod
    def _le(a, b) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a <= b
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return all(_MetaSnapshot._le(x, y) for x, y in zip(a, b))
        return a == b

    def plan(self, ctx: Any) -> tuple[list[tuple[torch.Tensor, torch.Tensor]] | None, str | None]:
        """(the copies to make as (dst captured, src fresh) pairs, None) or (None, why not)."""
        fresh = _MetaSnapshot(ctx)
        if fresh.tensors.keys() != self.tensors.keys():
            missing = sorted(set(self.tensors) ^ set(fresh.tensors))
            return None, f"metadata structure: {len(missing)} path(s) differ, e.g. {missing[:3]}"
        if fresh.values.keys() != self.values.keys():
            missing = sorted(set(self.values) ^ set(fresh.values))
            return None, f"metadata values: {len(missing)} path(s) differ, e.g. {missing[:3]}"
        for path, cap in self.values.items():
            new = fresh.values[path]
            leaf = path.rsplit(".", 1)[-1]
            if any(h in leaf for h in _META_LE_HINTS):
                if not self._le(new, cap):
                    return None, f"metadata extent: {path} = {str(new)[:40]} > captured {str(cap)[:40]}"
            elif new != cap:
                return None, f"metadata value: {path} = {str(new)[:40]} != captured {str(cap)[:40]}"
        copies: list[tuple[torch.Tensor, torch.Tensor]] = []
        for path, cap in self.tensors.items():
            new = fresh.tensors[path]
            if new.dtype != cap.dtype:
                return None, f"metadata dtype: {path} {new.dtype} != {cap.dtype}"
            if new.data_ptr() == cap.data_ptr() and new.shape == cap.shape and new.stride() == cap.stride():
                continue  # the same persistent buffer the builder filled for this step
            if new.shape == cap.shape:
                copies.append((cap, new))
            elif (new.dim() == cap.dim() and new.dim() >= 1 and new.shape[1:] == cap.shape[1:]
                  and new.shape[0] <= cap.shape[0]):
                copies.append((cap[: new.shape[0]], new))
            else:
                return None, f"metadata shape: {path} {tuple(new.shape)} vs captured {tuple(cap.shape)}"
        return copies, None


class PhaseGraph:
    """One captured step graph: the two groups' sequence-parallel sections as today's two-thread program recorded across
    the compute / comm / aux streams. Static inputs (the groups' rows of the step's tensors, private copies), the captured
    forward contexts (their metadata tensors are the graph's), the outputs (graph-pool tensors, copied out after every
    launch), the key the step must match."""

    def __init__(self) -> None:
        self.graph: torch.cuda.CUDAGraph | None = None
        self.key: list[tuple] = []
        self.inputs: list[tuple] = []  # per group: the static input tensors (None where the model passes None)
        self.contexts: list[Any] = []  # per group: the captured forward context (kept alive: the graph reads its tensors)
        self.meta: list[_MetaSnapshot] = []
        self.outputs: list[SectionState] = []  # per group: the section's outputs inside the graph pool
        self.out_static: list[SectionState] = []  # per group: private copies filled after every launch
        self.plans: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        self.captured_ctx: int = 0  # the sequence length the dummy step had
        self.num_launches: int = 0

    @staticmethod
    def group_key(g: PipeGroup) -> tuple:
        return (g.num_tokens, g.token_offset, g.slot, g.req_lo, g.req_hi, g.prefill_rows_from)

    def mismatch(self, runner: "PipeRunner", groups: list[PipeGroup]) -> str | None:
        """None when the graph serves this step, else why not (the first reason)."""
        if self.graph is None:
            return "no graph"
        if len(groups) != len(self.key):
            return f"groups: {len(groups)} != {len(self.key)}"
        for g, key in zip(groups, self.key):
            if self.group_key(g) != key:
                return f"group shape: group {g.idx} (rows, offset, slot, reqs, prefill_from) = {self.group_key(g)} != {key}"
        layers = getattr(runner.model_state, "decoder_replay_layers", None)
        batches = getattr(layers, "_moet_replay_batches", None)
        for g in groups:
            if batches is None or batches.get(g.slot) is None:
                return f"replay batch: group {g.idx} (slot {g.slot}) has none"
        plans = []
        for g, snap in zip(groups, self.meta):
            plan, why = snap.plan(g.forward_context)
            if why is not None:
                return f"group {g.idx}: {why}"
            plans.append(plan)
        self.plans = plans
        return None

    def copy_in(self, fresh_inputs: list[tuple]) -> None:
        for static, fresh in zip(self.inputs, fresh_inputs):
            for dst, src in zip(static, fresh):
                if dst is None or src is None:
                    if (dst is None) != (src is None):
                        raise RuntimeError("prefill pipeline: phase graph input presence differs from the capture")
                    continue
                dst.copy_(src)
        for plan in self.plans:
            for dst, src in plan:
                dst.copy_(src)

    def copy_out(self) -> list[SectionState]:
        for static, out in zip(self.out_static, self.outputs):
            static.copy_from(out)
        return self.out_static

    def release(self) -> None:
        self.graph = None
        self.inputs, self.contexts, self.meta, self.outputs, self.out_static, self.plans = [], [], [], [], [], []


def _phase_graph_static_plan() -> bool:
    """Whether the attention metadata being built belongs to a row group whose step may run as the phase graph: the
    indexer's prefill plan is then made context-independent (indexer_logits_budget, indexer_row_shard), so a real step's
    metadata has the structure of the captured one. Off = today's plan (slot 0, the whole step; the pipeline off)."""
    return mode() == 2 and phase_graphs() != "0" and prepare_slot() != 0


def indexer_logits_budget(budget_bytes: int, compressed_seq_lens_cpu: Any, compress_ratio: int) -> int:
    """The indexer builder's logits budget for the prefill chunk plan. Today's plan splits a request's query rows into
    sub-chunks of max_q = budget / (4 B x context) rows, so the number of chunks - and with it the number of logits
    kernels, the chunk metadata tensors and the graph's structure - changes with the context. For a phase-graph group
    the budget is scaled so that max_q is the one of the captured maximum context (PHASE_GRAPH_MAX_CTX / compress ratio):
    the same chunk boundaries at every context, each chunk's logits buffer smaller at a shorter context (the kernels
    bound the K extent on the device). Bits are untouched: every row's logits and top-k are computed independently of
    how the rows are grouped into launches. Returns budget_bytes unchanged outside a phase-graph group."""
    if not _phase_graph_static_plan() or compress_ratio is None:
        return budget_bytes
    try:
        n_real = int(compressed_seq_lens_cpu.max()) if compressed_seq_lens_cpu.numel() else 0
    except Exception:  # noqa: BLE001 - an unexpected type: today's plan
        return budget_bytes
    if n_real <= 0:
        return budget_bytes
    n_plan = max(phase_graph_max_ctx() // max(int(compress_ratio), 1), 1)
    max_q = max((budget_bytes // 4) // n_plan, 1)
    return 4 * max_q * n_real


def indexer_row_shard(sizes: list[int] | None, num_rows: int, tp_size: int) -> list[int] | None:
    """The indexer's TP row shard for a phase-graph group: equal contiguous shards instead of the work-balanced ones
    (balanced_prefill_row_shard weighs a row by its context, so the shard sizes - and the all_gatherv byte counts captured
    in the graph - would change with the context). Bits are untouched (which rank scores a row does not change the row);
    the balance matters only in a request's first step, where the contexts are short. None (sharding off) stays None."""
    if sizes is None or not _phase_graph_static_plan() or tp_size < 2 or num_rows < tp_size:
        return sizes
    base, rem = divmod(int(num_rows), int(tp_size))
    return [base + (1 if i < rem else 0) for i in range(tp_size)]


def _wait_engram_prefetch(runner: "PipeRunner", stream: torch.cuda.Stream) -> None:
    """The Engram lookups of the head run on a prefetch stream (CPU-offloaded tables) and the layer waits for their event;
    inside the graph that wait is skipped (phase_graph_capturing), so the compute stream waits here, before the launch."""
    for module in runner.engram_modules:
        ev = getattr(module, "_prefetch_done", None)
        if ev is not None:
            stream.wait_event(ev)


def _compare_states(a: SectionState, b: SectionState) -> list[str]:
    """The names of the tensors that differ (torch.equal), or the structural difference."""
    na, nb = a.named_tensors(), b.named_tensors()
    if [n for n, _ in na] != [n for n, _ in nb] or a.wrote_kv != b.wrote_kv:
        return [f"structure {[n for n, _ in na]} / wrote_kv {a.wrote_kv} vs {[n for n, _ in nb]} / {b.wrote_kv}"]
    return [n for (n, x), (_, y) in zip(na, nb) if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y)]


def _phase_capture_step(runner: "PipeRunner", groups: list[PipeGroup], phase: PhaseHooks) -> list[Any]:
    """The capture pass of the dummy step: the groups' sections under stream capture (one graph across the streams, the
    two threads' program as it runs), one replay (the outputs then hold real values), the rest of the step eagerly."""
    import gc

    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import current_stream

    compute = current_stream()
    pg = PhaseGraph()
    pg.key = [PhaseGraph.group_key(g) for g in groups]
    pg.inputs = [
        tuple(None if t is None else t.detach().clone().contiguous() for t in phase.inputs(g)) for g in groups
    ]
    pg.contexts = [g.forward_context for g in groups]
    pg.meta = [_MetaSnapshot(ctx) for ctx in pg.contexts]
    for i, snap in enumerate(pg.meta):
        if not snap.tensors:
            raise RuntimeError(f"prefill pipeline: phase graph capture - group {i}'s forward context has no metadata tensors")

    def section_fn(g: PipeGroup) -> SectionState:
        state = phase.section(g, *pg.inputs[g.idx])
        if not isinstance(state, SectionState):
            raise RuntimeError(f"prefill pipeline: the section returned {type(state).__name__}, not SectionState")
        if not state.wrote_kv:
            raise RuntimeError("prefill pipeline: phase graph capture - the dummy step has no replay batch (the first "
                               "replay layer's KV was not written); the graph would not match any real step")
        bad = [n for n, t in state.named_tensors() if not isinstance(t, torch.Tensor) or not t.is_cuda]
        if bad:
            raise RuntimeError(f"prefill pipeline: phase graph capture - section outputs that are not CUDA tensors: {bad}")
        return state

    if compute == torch.cuda.default_stream(runner.device):
        raise RuntimeError("prefill pipeline: the phase graph must be captured on a non-default stream "
                           "(capture_phase_graph runs the passes on the runner's capture stream)")
    _wait_engram_prefetch(runner, compute)
    torch.cuda.synchronize()
    gc.collect()
    pool = current_platform.get_global_graph_pool()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    _PHASE["capturing"] = True
    t0 = time.time()
    try:
        graph.capture_begin(pool=pool, capture_error_mode="thread_local")
        try:
            step = PipeStep(groups, compute, runner.comm_stream, runner.row_switch, runner.device, True,
                            replay_stream=getattr(runner, "replay_stream", None))
            outputs = step.run(section_fn)
        except BaseException:
            # the capture is invalidated: end it so the streams and the allocator leave capture mode, keep the cause
            try:
                graph.capture_end()
            except Exception:  # noqa: BLE001
                pass
            raise
        graph.capture_end()
    finally:
        _PHASE["capturing"] = False
    try:
        import vllm.compilation.moet_nccl_graph_records as _moet_nccl

        if _moet_nccl.records_enabled():
            _moet_nccl.finish_capture(graph)
        else:
            graph.instantiate()
    except ImportError:
        graph.instantiate()
    pg.graph = graph
    pg.outputs = outputs
    pg.out_static = [state.static_copy() for state in outputs]
    # one replay: the capture recorded the kernels without running them - the outputs hold values afterwards, and a graph
    # that cannot launch fails here, at engine start, not in the first request
    graph.replay()
    torch.cuda.synchronize()
    runner.phase_graph = pg
    runner.phase_graph_failed = None
    nodes = ""
    try:
        import cuda.bindings.runtime as rt

        res = rt.cudaGraphGetNodes(rt.cudaGraph_t(init_value=int(graph.raw_cuda_graph())), 0)
        if int(res[0]) == 0:
            nodes = f", {int(res[2])} graph nodes"
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    logger.info(
        "vllm-moet sm_120 prefill pipeline: phase graph captured in %.1f s - the sequence-parallel section of a %d-row step "
        "(%d groups x %d rows, slots %s) across the compute / comm streams%s; VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS=%s, "
        "MAX_CTX=%d; outputs %s",
        time.time() - t0, sum(g.num_tokens for g in groups), len(groups), groups[0].num_tokens, [g.slot for g in groups],
        nodes, phase_graphs(), phase_graph_max_ctx(),
        [(n, tuple(t.shape), str(t.dtype).replace("torch.", "")) for n, t in outputs[0].named_tensors()],
    )
    rest_states = pg.copy_out()
    tail = PipeStep(groups, compute, runner.comm_stream, runner.row_switch, runner.device, False)
    return tail.run_sequential(lambda g: phase.rest(g, pg.inputs[g.idx][1], pg.inputs[g.idx][2], rest_states[g.idx]))


def _phase_graph_step(runner: "PipeRunner", groups: list[PipeGroup], phase: PhaseHooks, pg: PhaseGraph) -> list[Any]:
    """A real pipelined step on the graph: inputs and metadata in, one launch, outputs out, the tails eagerly; in verify
    mode the eager section on the same inputs and the comparison first."""
    from vllm.utils.torch_utils import current_stream

    compute = current_stream()
    fresh = [phase.inputs(g) for g in groups]
    pg.copy_in(fresh)
    _wait_engram_prefetch(runner, compute)
    pg.graph.replay()
    pg.num_launches += 1
    _STATS["phase_graph_steps"] += 1
    if _STATS["phase_graph_steps"] == 1:
        logger.info(
            "vllm-moet sm_120 prefill pipeline: the sequence-parallel section of the pipelined steps runs as the phase graph "
            "(one launch per step; VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS=%s)", phase_graphs(),
        )
    states = pg.copy_out()
    if phase_graphs() == "verify":
        step = PipeStep(groups, compute, runner.comm_stream, runner.row_switch, runner.device, True,
                        replay_stream=getattr(runner, "replay_stream", None))
        eager = step.run(lambda g: phase.section(g, *pg.inputs[g.idx]))
        diffs = {g.idx: _compare_states(states[g.idx], eager[g.idx]) for g in groups}
        _STATS["phase_graph_verified"] += 1
        bad = {i: d for i, d in diffs.items() if d}
        if bad:
            _STATS["phase_graph_mismatch"] += 1
            logger.error(
                "vllm-moet sm_120 prefill pipeline: PHASE GRAPH VERIFY step %d MISMATCH - graph vs eager differ in %s "
                "(the tails continue from the eager outputs; mismatches so far %d of %d)",
                _STATS["phase_graph_verified"], bad, _STATS["phase_graph_mismatch"], _STATS["phase_graph_verified"],
            )
        else:
            logger.info(
                "vllm-moet sm_120 prefill pipeline: phase graph verify step %d IDENTICAL (%d tensors per group, torch.equal; "
                "mismatches so far %d)", _STATS["phase_graph_verified"], len(states[0].named_tensors()),
                _STATS["phase_graph_mismatch"],
            )
        states = eager
    rest = lambda g: phase.rest(g, fresh[g.idx][1], fresh[g.idx][2], states[g.idx])  # noqa: E731
    shared = SharedTail.for_step(groups)
    if shared is not None:
        # K-9: the shared replay tail needs the two threads (the rendezvous); no phases in the rest, so the threads only
        # alternate there
        tail = PipeStep(groups, compute, runner.comm_stream, runner.row_switch, runner.device, True)
        tail.shared = shared
        return tail.run(rest)
    tail = PipeStep(groups, compute, runner.comm_stream, runner.row_switch, runner.device, False)
    return tail.run_sequential(rest)


def capture_phase_graphs(runner: Any, profile_only: bool = False) -> None:
    """Runner side, from capture_model after model_state.capture_inner_cudagraphs (both of vLLM's passes): build the dummy
    step the graph is captured on - one request of VLLM_MOET_PREFILL_PIPE_PHASE_GRAPH_MAX_CTX tokens with the last
    2 x 4 096 as this step's rows, the production bounds, the per-slot metadata of a real step - run it once eagerly
    (warm-up: cuBLAS workspaces per stream, the JIT kernels, the bf16 replica GEMM's calibration, the indexer workspace at
    the maximum before lock_workspace), then capture its sequence-parallel section. A failure leaves every pipelined step
    eager and says so with the traceback (never a silent fallback). The graph of the memory-profiling pass is dropped
    right away (it lives in that pass's throwaway pool; its size is in the pass's measurement) - K-7's lesson."""
    global _RUNNER
    if mode() == 2 and phase_graphs() == "0" and green_sms() and green_warmup():
        # the static SM split (VLLM_MOET_PREFILL_PIPE_GREEN): no graph - this call is the hook of its warm-up step
        _green_warmup_step(runner, profile_only)
        return
    if mode() != 2 or phase_graphs() == "0":
        return
    if kv_connector_policy() == "refuse" and _kv_connector_configured(runner):
        return
    if _RUNNER is not None and _RUNNER.phase_graph is not None:
        _RUNNER.phase_graph.release()
        _RUNNER.phase_graph = None
    # built anew at every capture pass: the memory-profiling pass runs on a minimal KV cache and its own attention groups /
    # kv_cache_config, which the per-slot builders and the slot mappings of a PipeRunner created then would keep (the real
    # pass sees the real ones; the first real step reuses the runner of the real pass)
    _RUNNER = PipeRunner(runner)
    pr = _RUNNER
    try:
        pr.capture_phase_graph(runner)
    except Exception as exc:  # noqa: BLE001 - the server comes up eagerly, loudly
        pr.phase_graph = None
        pr.phase_graph_failed = f"{type(exc).__name__}: {str(exc)[:300]}"
        logger.exception(
            "vllm-moet sm_120 prefill pipeline: PHASE GRAPH CAPTURE FAILED (%s) - every pipelined step runs eagerly "
            "(VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS=%s requested)", pr.phase_graph_failed, phase_graphs(),
        )
    finally:
        _PHASE["request"] = None
        _PHASE["capturing"] = False
    if profile_only and pr.phase_graph is not None:
        pr.phase_graph.release()
        pr.phase_graph = None
        import gc

        gc.collect()


# ------------------------------------------------------------------------------------------- the second communicator
class CommSwitch:
    """Stands in for the TP group's `device_communicator.pynccl_comm` (VLLM_MOET_PREFILL_PIPE_COMM2=1): every attribute
    and method resolves on the communicator of the calling thread's group - the odd groups' on `alt`, everything else
    (the even group, the runner thread, decode steps, graph capture) on `base`. The choice is per call, so a collective
    that opens a NCCL group (all_gatherv: group_start .. group_end) stays on one communicator. Every rank makes the same
    choice for the same op (the groups' programs are identical across ranks), which is all NCCL asks of a communicator."""

    def __init__(self, base, alt) -> None:
        self._base = base
        self._alt = alt

    def _pick(self):
        g = current_group()
        if g is not None and g.idx % 2 == 1 and g.step is not None and g.step.pipelined:
            return self._alt
        return self._base

    def __getattr__(self, name):
        if name in ("_base", "_alt"):
            raise AttributeError(name)
        return getattr(self._pick(), name)

    def __repr__(self) -> str:
        return f"CommSwitch(base={self._base!r}, alt={self._alt!r})"


def _install_comm2(device: torch.device):
    """Build the second NCCL communicator over the TP ranks (collective: every rank calls this at the same step) and
    install the switch on the TP group's device communicator. Returns the communicator, or None when it is not possible
    (no pynccl, world size 1) - then the single communicator stays and a warning says so."""
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    dc = getattr(tp, "device_communicator", None)
    base = getattr(dc, "pynccl_comm", None)
    if dc is None or base is None or getattr(base, "disabled", True) or tp.world_size < 2:
        logger.warning("vllm-moet sm_120 prefill pipeline: VLLM_MOET_PREFILL_PIPE_COMM2=1 needs the TP group's pynccl "
                       "communicator (found %r) - the groups share one communicator", base)
        return None
    if isinstance(base, CommSwitch):
        return base._alt
    group = tp.cpu_group
    comm2 = PyNcclCommunicator(group=group, device=device)
    if getattr(comm2, "disabled", True):
        logger.warning("vllm-moet sm_120 prefill pipeline: the second NCCL communicator is disabled - the groups share one")
        return None
    dc.pynccl_comm = CommSwitch(base, comm2)
    logger.info(
        "vllm-moet sm_120 prefill pipeline: the odd row group's collectives run on a second NCCL communicator (rank %d of "
        "%d; VLLM_MOET_PREFILL_PIPE_COMM2=1) - the two groups' reduce-scatter / all-gather chains are no longer serialized "
        "in one NCCL queue",
        comm2.rank,
        comm2.world_size,
    )
    return comm2


# ------------------------------------------------------------------------------------------------- runner side
_RUNNER: "PipeRunner | None" = None


class PipeRunner:
    """Per model runner: the comm stream, the per-group metadata builders (slots 1..n), the sliced-batch buffers and
    the row-buffer storages."""

    NUM_GROUPS = 2

    def __init__(self, runner: Any) -> None:
        from vllm.v1.worker.utils import AttentionGroup  # noqa: F401 - the type the copies keep

        self.vllm_config = runner.vllm_config
        self.device = runner.device
        self.model_state = runner.model_state
        self.kv_cache_config = runner.kv_cache_config
        self.max_num_reqs = runner.max_num_reqs
        self.num_groups = self.NUM_GROUPS
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
        # K-8: the phase graph of a full step (capture_phase_graphs), the reason it is missing, the Engram modules whose
        # prefetch event the compute stream waits for before a launch
        self.phase_graph: PhaseGraph | None = None
        self.phase_graph_failed: str | None = None
        self.engram_modules = [m for m in runner.model.modules() if hasattr(m, "_prefetch_done")]
        self.model = runner.model
        # the static SM split (VLLM_MOET_PREFILL_PIPE_GREEN): the attention modules' aux-stream lists (one list shared by every
        # layer in the served model), whose streams are the compute partition's during a split step (GreenSplit.run)
        self.aux_stream_lists: list[list] = []
        if green_sms():
            seen: set[int] = set()
            for module in runner.model.modules():
                lst = getattr(module, "aux_stream_list", None)
                if isinstance(lst, list) and lst and id(lst) not in seen and all(isinstance(s, torch.cuda.Stream) for s in lst):
                    seen.add(id(lst))
                    self.aux_stream_lists.append(lst)
        # the capture passes run on this stream (a capture cannot begin on the default stream; the warm-up runs on it too, so
        # the per-stream cuBLAS workspaces exist before the capture); the real steps launch the graph on the runner's stream
        self.capture_stream = torch.cuda.Stream(device=self.device)
        # K-6: the odd group's collectives on a communicator of their own (VLLM_MOET_PREFILL_PIPE_COMM2=1). Built here, at
        # the first pipelined step, on every rank at once (the groups' first step is the same step on every rank); the
        # one-element warm-up all-reduce inside synchronizes the current stream once.
        self.comm2 = _install_comm2(self.device) if comm2_enabled() else None
        # K-9: the shared replay tail - the pseudo-batch's rebased query offsets / seq lens, the all-reduce split (after comm2:
        # it wraps whatever communicator dispatch is installed)
        self.shared_qsl_buf = torch.zeros(2 * self.max_num_reqs + 1, dtype=torch.int32, device=self.device)
        self.shared_seq_lens_buf = torch.zeros(2 * self.max_num_reqs, dtype=torch.int32, device=self.device)
        self.shared_ar_split = _install_ar_split() if shared_tail() != "0" else False
        if shared_tail() != "0" and replay_mode() != "serial":
            logger.warning("vllm-moet sm_120 prefill pipeline: VLLM_MOET_PREFILL_PIPE_SHARED_TAIL=%s needs REPLAY=serial (found %s) "
                           "- every step runs today's two tails", shared_tail(), replay_mode())
        if order() != "today":
            logger.info(
                "vllm-moet sm_120 prefill pipeline: launch order %s (VLLM_MOET_PREFILL_PIPE_ORDER) = second NCCL communicator "
                "for the odd group %s, comm stream priority %d",
                order(), "on" if self.comm2 is not None else "UNAVAILABLE (one communicator)", _comm_priority(),
            )
        logger.info(
            "vllm-moet sm_120 prefill pipeline ready: %d groups of %d rows, %d row buffers per group, comm stream "
            "priority %d, %s",
            self.num_groups,
            group_rows(),
            0 if self.row_switch is None else len(self.row_switch.buffers),
            _comm_priority(),
            "two NCCL communicators (COMM2=1)" if self.comm2 is not None else "one NCCL communicator",
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
        mixed = not bool(input_batch.is_prefilling_np[:num_reqs].all())
        if mixed:
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
        _host_mark("plan_start")
        groups = self._build(input_batch, block_tables, slot_mappings, bounds)
        _host_mark("plan_end")
        verbosity = log_steps()
        if verbosity >= 2 or (verbosity >= 1 and mixed):
            self._log_step(input_batch, groups, mixed)
        return groups

    @staticmethod
    def _log_step(input_batch, groups: list[PipeGroup], mixed: bool) -> None:
        """VLLM_MOET_PREFILL_PIPE_LOG_STEPS: the batch composition of a pipelined step - the requests' query lengths
        and roles, the decode rows, the groups' row and request ranges (the mixed-step harness reads these)."""
        num_reqs = input_batch.num_reqs
        qsl = input_batch.query_start_loc_np[: num_reqs + 1]
        qlens = [int(x) for x in (qsl[1:] - qsl[:-1])]
        prefilling = [bool(x) for x in input_batch.is_prefilling_np[:num_reqs]]
        decode_rows = sum(q for q, p in zip(qlens, prefilling) if not p)
        parts = []
        for g in groups:
            lo, hi = g.token_offset, g.token_offset + g.num_tokens
            req_lo = int(np.searchsorted(qsl, lo, side="right") - 1)
            req_hi = int(np.searchsorted(qsl, hi - 1, side="right") - 1)
            parts.append(f"g{g.idx}=[{lo},{hi})/reqs {req_lo}-{req_hi}")
        logger.info(
            "prefill pipeline step: %d tokens, %d requests (%d prefill, %d decode; %d decode rows)%s; query lens %s; "
            "roles %s; groups %s",
            input_batch.num_tokens,
            num_reqs,
            sum(prefilling),
            num_reqs - sum(prefilling),
            decode_rows,
            " MIXED" if mixed else "",
            qlens if num_reqs <= 16 else f"{qlens[:16]}...",
            "".join("P" if p else "D" for p in prefilling) if num_reqs <= 64 else f"{sum(prefilling)}P/{num_reqs - sum(prefilling)}D",
            " ".join(parts),
        )

    def _build(self, input_batch, block_tables, slot_mappings, bounds) -> list[PipeGroup]:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, create_forward_context
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        from vllm.v1.worker.gpu.ubatch_utils import _slice_input_batch
        from vllm.v1.worker.ubatch_utils import UBatchSlice

        num_reqs = input_batch.num_reqs
        qsl = input_batch.query_start_loc_np[: num_reqs + 1]
        groups: list[PipeGroup] = []
        ubatches: list[Any] = []
        ub_slices: list[Any] = []
        light = self._shared_precheck(input_batch, bounds)  # K-9: the groups' replay batches as rows only (shared_tail_light)
        for i, (lo, hi) in enumerate(bounds):
            req_lo = int(np.searchsorted(qsl, lo, side="right") - 1)
            req_hi = int(np.searchsorted(qsl, hi - 1, side="right") - 1)
            ub_slice = UBatchSlice(slice(req_lo, req_hi + 1), slice(lo, hi))
            ubatch = _slice_input_batch(
                input_batch, ub_slice, self.query_start_loc_bufs[i], self.seq_lens_bufs[i]
            )
            # the group's own longest query (None: prepare_attn derives it from the sliced query lengths)
            ubatch = replace(ubatch, max_query_len=None)
            ubatches.append(ubatch)
            ub_slices.append(ub_slice)
            group_block_tables = tuple(bt[ub_slice.request_slice] for bt in block_tables)
            group_slot_mappings = slot_mappings[:, ub_slice.token_slice]
            slot = i + 1
            _LIGHT["on"] = light
            try:
                with _PrepareSlot(slot):
                    attn_metadata = self.model_state.prepare_attn(
                        ubatch,
                        CUDAGraphMode.NONE,
                        group_block_tables,
                        group_slot_mappings,
                        self.attn_groups_slots[slot],
                        self.kv_cache_config,
                    )
            finally:
                _LIGHT["on"] = False
            forward_ctx = create_forward_context(
                attn_metadata,
                self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=BatchDescriptor(num_tokens=hi - lo),
                slot_mapping=build_slot_mappings_by_layer(group_slot_mappings, self.kv_cache_config),
                is_padding=ubatch.is_padding,
            )
            ub_reqs = ubatch.num_reqs
            ub_pref = [bool(p) for p in ubatch.is_prefilling_np[:ub_reqs]]
            ub_qsl = ubatch.query_start_loc_np[: ub_reqs + 1]
            prefill_from = next((int(ub_qsl[j]) for j in range(ub_reqs) if ub_pref[j]), hi - lo)
            groups.append(
                PipeGroup(
                    idx=i, token_offset=lo, num_tokens=hi - lo, forward_context=forward_ctx, slot=slot,
                    prefill_rows_from=prefill_from, req_lo=req_lo, req_hi=req_hi,
                )
            )
        if shared_tail() != "0":
            plan = self._plan_shared(input_batch, block_tables, slot_mappings, groups, ubatches, ub_slices)
            for g in groups:
                g.shared_plan = plan
            if light and plan is None:
                # the precheck said the shared tail would serve the step, the plan disagreed: the groups need their full batches
                self._warn_once("light-rebuild", "prefill pipeline: the light replay batches were rebuilt in full (the shared "
                                "plan failed after the precheck passed) - today's tails for this step")
                self._rebuild_group_batches(groups, ubatches, ub_slices, block_tables, slot_mappings)
        # the whole step's buffers current again for whatever else the step prepares (the groups keep their own)
        select = getattr(self.model_state, "_moet_pipe_select_slot", None)
        if select is not None:
            select(0)
        return groups

    def _group_kept(self, input_batch, bounds) -> list[list[tuple[int, int]]] | None:
        """Per group, per request slice in it: (query rows, kept rows) as _prepare_replay_batch computes them (keeps_rows =
        every row, else min(rows, window)); None when the model state has no decoder replay."""
        ms = self.model_state
        layers = getattr(ms, "decoder_replay_layers", None)
        keeps = getattr(ms, "_keeps_rows_np", None)
        if layers is None or keeps is None:
            return None
        window = int(layers.window)
        num_reqs = input_batch.num_reqs
        qsl = input_batch.query_start_loc_np[: num_reqs + 1]
        idx = input_batch.idx_mapping_np[:num_reqs]
        out = []
        for lo, hi in bounds:
            req_lo = int(np.searchsorted(qsl, lo, side="right") - 1)
            req_hi = int(np.searchsorted(qsl, hi - 1, side="right") - 1)
            rows = []
            for r in range(req_lo, req_hi + 1):
                qlen = int(min(qsl[r + 1], hi) - max(qsl[r], lo))
                kept = qlen if bool(keeps[idx[r]]) else min(qlen, window)
                rows.append((qlen, kept))
            out.append(rows)
        return out

    def _shared_precheck(self, input_batch, bounds) -> bool:
        """K-9 (shared_tail_light): will the shared tail serve this step? Decided on the host before the groups' metadata builds,
        with _plan_shared's conditions: SHARED_TAIL=1 (not verify), REPLAY=serial, two groups whose request slices both trim a
        prefilling request (= both get a replay batch), the summed kept rows within the shared slot's capture sizes when the
        replay graphs are on, equal mHC pre-norm split counts."""
        if shared_tail() != "1" or not shared_tail_light() or replay_mode() != "serial" or len(bounds) != 2:
            return False
        kept = self._group_kept(input_batch, bounds)
        if kept is None:
            return False
        num_reqs = input_batch.num_reqs
        qsl = input_batch.query_start_loc_np[: num_reqs + 1]
        prefilling = input_batch.is_prefilling_np[:num_reqs]
        counts = []
        for (lo, hi), rows in zip(bounds, kept):
            req_lo = int(np.searchsorted(qsl, lo, side="right") - 1)
            trims = any(kept_r < qlen and bool(prefilling[req_lo + j]) for j, (qlen, kept_r) in enumerate(rows))
            if not trims:
                return False
            counts.append(sum(k for _, k in rows))
        if replay_graphs():
            window = int(self.model_state.decoder_replay_layers.window)
            cfg = slot_graph_config(self.vllm_config, SHARED_SLOT, window)
            sizes = None if cfg is None else list(cfg.compilation_config.decoder_replay_cudagraph_capture_sizes or [])
            if not sizes or sum(counts) > max(sizes):
                return False
        try:
            if len({self._mhc_splits(c) for c in counts}) != 1:
                return False
        except Exception:  # noqa: BLE001 - no mHC warm-up module: _plan_shared decides
            pass
        return True

    def _rebuild_group_batches(self, groups, ubatches, ub_slices, block_tables, slot_mappings) -> None:
        """The groups' replay batches in full (after a light build the shared plan could not use)."""
        from vllm.config import CUDAGraphMode

        ms = self.model_state
        for g, ubatch, ub_slice in zip(groups, ubatches, ub_slices):
            n = ubatch.num_reqs
            with _PrepareSlot(g.slot):
                ms._moet_pipe_select_slot(g.slot)
                replay_start_np = np.where(
                    ubatch.is_prefilling_np[:n], ms._replay_start_np[ubatch.idx_mapping_np[:n]], 0
                ).astype(np.int32)
                replay_start = ms._replay_start_staging.copy_to_gpu(replay_start_np, out=ms._replay_start[:n])
                ms._prepare_replay_batch(
                    ubatch, CUDAGraphMode.NONE, tuple(bt[ub_slice.request_slice] for bt in block_tables),
                    slot_mappings[:, ub_slice.token_slice], self.attn_groups_slots[g.slot], self.kv_cache_config, replay_start,
                )

    def _shared_pseudo_batch(self, input_batch, groups: list[PipeGroup], ubatches: list[Any]):
        """K-9: the two groups' request slices as one batch of separate requests (what each group's ubatch already is - a
        request straddling the group boundary appears twice, its query truncated to each group's rows): the per-request arrays
        concatenated, the query offsets of the second group rebased by the first group's rows, the step's per-token tensors
        over both groups' rows. Device values are built from the ubatches' device tensors (no host-to-device copy)."""
        a, b = ubatches[0], ubatches[-1]
        na, nb = a.num_reqs, b.num_reqs
        n = na + nb
        lo, hi = groups[0].token_offset, groups[-1].token_offset + groups[-1].num_tokens
        tok_a = groups[0].num_tokens
        qsl = self.shared_qsl_buf[: n + 1]
        qsl[: na + 1].copy_(a.query_start_loc[: na + 1])
        torch.add(b.query_start_loc[1 : nb + 1], tok_a, out=qsl[na + 1 : n + 1])
        qsl_np = np.concatenate(
            [a.query_start_loc_np[: na + 1], b.query_start_loc_np[1 : nb + 1] + tok_a]
        ).astype(np.int32)
        seq_lens = self.shared_seq_lens_buf[:n]
        seq_lens[:na].copy_(a.seq_lens[:na])
        seq_lens[na:n].copy_(b.seq_lens[:nb])

        def cat_np(x, y):
            return None if x is None or y is None else np.concatenate([x[:na], y[:nb]])

        def cat_t(x, y):
            return None if x is None or y is None else torch.cat([x[:na], y[:nb]], dim=0)

        prefilling = cat_np(a.is_prefilling_np, b.is_prefilling_np)
        return replace(
            a,
            req_ids=list(a.req_ids[:na]) + list(b.req_ids[:nb]),
            num_reqs=n,
            num_reqs_after_padding=n,
            idx_mapping=cat_t(a.idx_mapping, b.idx_mapping),
            idx_mapping_np=cat_np(a.idx_mapping_np, b.idx_mapping_np),
            num_scheduled_tokens=np.diff(qsl_np)[:n],
            num_tokens=hi - lo,
            num_tokens_after_padding=hi - lo,
            query_start_loc=qsl,
            query_start_loc_np=qsl_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=cat_t(a.seq_lens_cpu_upper_bound, b.seq_lens_cpu_upper_bound),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=cat_np(a.num_computed_tokens_np, b.num_computed_tokens_np),
            prefill_len_np=cat_np(a.prefill_len_np, b.prefill_len_np),
            num_computed_prefill_tokens_np=cat_np(a.num_computed_prefill_tokens_np, b.num_computed_prefill_tokens_np),
            is_prefilling_np=prefilling,
            has_prefill=bool(prefilling.any()),
            prefill_runs_as_decode_np=cat_np(a.prefill_runs_as_decode_np, b.prefill_runs_as_decode_np),
            input_ids=input_batch.input_ids[lo:hi],
            positions=input_batch.positions[lo:hi],
            is_padding=input_batch.is_padding[lo:hi],
            prompt_lens=cat_t(a.prompt_lens, b.prompt_lens),
            max_seq_len_np=None,
            max_query_len=None,
            fast_prefill=None,
        )

    def _plan_shared(self, input_batch, block_tables, slot_mappings, groups, ubatches, ub_slices) -> SharedPlan | None:
        """K-9: build the shared replay batch (SHARED_SLOT) for a two-group step whose groups both have a replay batch, and
        decide whether the shared tail serves the step; None (today's tails, one warning per reason) otherwise."""
        from vllm.config import CUDAGraphMode

        ms = self.model_state
        layers = getattr(ms, "decoder_replay_layers", None)
        batches = getattr(layers, "_moet_replay_batches", None)
        if layers is None or batches is None or len(groups) != 2:
            return None
        group_batches = [batches.get(g.slot) for g in groups]
        why = None
        if any(b is None for b in group_batches):
            why = "a group has no replay batch (prompt logprobs, or a batch that does not trim)"
        elif replay_mode() != "serial":
            why = "VLLM_MOET_PREFILL_PIPE_REPLAY=stream"
        elif not hasattr(ms, "_prepare_replay_batch") or not hasattr(ms, "_moet_pipe_select_slot"):
            why = "the model state has no per-slot replay batches (patcher mismatch)"
        if why is not None:
            _STATS["shared_tail_today"] += 1
            self._warn_once("shared:" + why[:24], "prefill pipeline: the step runs today's two replay tails, not the shared one (%s)", why)
            return None
        counts = [int(b.rows.shape[0]) for b in group_batches]
        pseudo = self._shared_pseudo_batch(input_batch, groups, ubatches)
        n = pseudo.num_reqs
        shared_block_tables = tuple(
            torch.cat([bt[ub_slices[0].request_slice], bt[ub_slices[-1].request_slice]], dim=0) for bt in block_tables
        )
        lo, hi = groups[0].token_offset, groups[-1].token_offset + groups[-1].num_tokens
        with _PrepareSlot(SHARED_SLOT):
            ms._moet_pipe_select_slot(SHARED_SLOT)
            replay_start_np = np.where(
                pseudo.is_prefilling_np[:n], ms._replay_start_np[pseudo.idx_mapping_np[:n]], 0
            ).astype(np.int32)
            replay_start = ms._replay_start_staging.copy_to_gpu(replay_start_np, out=ms._replay_start[:n])
            ms._prepare_replay_batch(
                pseudo, CUDAGraphMode.NONE, shared_block_tables, slot_mappings[:, lo:hi], self.attn_groups_slots[0],
                self.kv_cache_config, replay_start,
            )
        shared_batch = batches.get(SHARED_SLOT)
        if shared_batch is None:
            why = "the shared batch was not built (nothing to trim)"
        elif int(shared_batch.rows.shape[0]) != sum(counts):
            why = f"the shared batch has {int(shared_batch.rows.shape[0])} rows, the groups {counts}"
        elif replay_graphs() and shared_batch.run_graph is None:
            why = f"no replay graph for a shared batch of {sum(counts)} rows (REPLAY_GRAPHS=1; the capture list is smaller)"
        else:
            try:
                splits = {self._mhc_splits(c) for c in counts}
            except Exception:  # noqa: BLE001 - no mHC warm-up module (the unit test): the pin alone
                splits = {0}
            if len(splits) != 1:
                why = f"the groups' mHC pre-norm split counts differ at {counts} rows ({sorted(splits)})"
        if why is not None:
            _STATS["shared_tail_today"] += 1
            self._warn_once("shared:" + why[:24], "prefill pipeline: the step runs today's two replay tails, not the shared one (%s)", why)
            return None
        return SharedPlan(counts=counts, pin=counts[0], verify=shared_tail() == "verify")

    def _mhc_splits(self, num_tokens: int) -> int:
        """The mHC pre-norm GEMM's split-k count for num_tokens rows (compute_mhc_pre_num_splits, K = hc_mult x hidden)."""
        hf = self.vllm_config.model_config.hf_config
        k = int(getattr(hf, "hc_mult", 4)) * int(hf.hidden_size)
        return int(_mhc_warmup().compute_mhc_pre_num_splits(k, num_tokens))

    def capture_phase_graph(self, runner: Any, passes: tuple[str, ...] = ("warmup", "capture"), max_ctx: int | None = None) -> None:
        """K-8: the dummy step of the phase graph and its two passes (warm-up, capture) through the model's forward - the
        runner's eager path on a dummy batch, so the graph reads the buffers a real step fills (the replay-graph capture's
        principle). One request in the middle of its prompt: context = PHASE_GRAPH_MAX_CTX - the step's rows (the host
        extents of the metadata are then maxima), the step = NUM_GROUPS x group_rows() rows, every row prefilling.
        `passes` / `max_ctx`: the static SM split's warm-up step is this dummy step's warm-up pass alone, at its own (short)
        sequence length (_green_warmup_step); the defaults are the phase graph's."""
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, set_forward_context
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        from vllm.v1.worker.gpu.input_batch import InputBatch, set_dummy_context

        rows = group_rows()
        num_tokens = rows * self.num_groups
        max_ctx = min(phase_graph_max_ctx() if max_ctx is None else max(int(max_ctx), num_tokens), int(runner.max_model_len))
        context_len = max(max_ctx - num_tokens, 0)
        input_batch = InputBatch.make_dummy(1, num_tokens, runner.input_buffers, is_padding=False)
        block_tables, slot_mappings = runner.prepare_dummy_attn(input_batch, False)
        if context_len:
            set_dummy_context(
                input_batch, runner.block_tables, context_len, self.kv_cache_config.num_blocks, runner.max_model_len,
                input_block_tables=block_tables,
            )
        # a prefilling request whose prompt continues after this step (every real full step is one)
        input_batch.is_prefilling_np[:] = True
        input_batch.has_prefill = True
        input_batch.prefill_len_np[:] = input_batch.num_computed_prefill_tokens_np + num_tokens + rows
        if getattr(input_batch, "prefill_runs_as_decode_np", None) is not None:
            input_batch.prefill_runs_as_decode_np[:] = False
        # real slot mappings for the dummy rows (the dummy ones are all PAD: no KV write, every token invalid) so the warm-up
        # and the capture run the KV writes and the validity-dependent kernels as a real step does; the blocks are the
        # dummy block tables' (ids wrapping inside the cache), written at engine start, before any request
        positions64 = input_batch.positions[:num_tokens].to(torch.int64)
        for k, (table, block_size) in enumerate(zip(block_tables, runner.block_tables.block_sizes)):
            if k >= slot_mappings.shape[0]:
                break
            block_idx = torch.div(positions64, block_size, rounding_mode="floor").clamp_(max=table.shape[1] - 1)
            blocks = table[0].to(torch.int64)[block_idx]
            slot_mappings[k, :num_tokens].copy_(blocks * block_size + positions64 % block_size)
        attn_metadata = self.model_state.prepare_attn(
            input_batch, CUDAGraphMode.NONE, block_tables, slot_mappings, runner.attn_groups, self.kv_cache_config
        )
        bounds = [(i * rows, (i + 1) * rows) for i in range(self.num_groups)]
        groups = self._build(input_batch, block_tables, slot_mappings, bounds)
        model_inputs = {
            "input_ids": input_batch.input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": None,
            "intermediate_tensors": None,
            **self.model_state.prepare_inputs(input_batch, runner.req_states),
        }
        if "capture" in passes:
            logger.info(
                "vllm-moet sm_120 prefill pipeline: capturing the phase graph on a dummy step of %d rows (%d groups x %d, context "
                "%d tokens = MAX_CTX %d, %d KV blocks): warm-up pass, then the capture pass",
                num_tokens, self.num_groups, rows, context_len, max_ctx, self.kv_cache_config.num_blocks,
            )
        else:
            logger.info(
                "vllm-moet sm_120 prefill pipeline: SM split warm-up on a dummy step of %d rows (%d groups x %d, context %d tokens, "
                "%d KV blocks)", num_tokens, self.num_groups, rows, context_len, self.kv_cache_config.num_blocks,
            )
        self.capture_stream.wait_stream(torch.cuda.current_stream())
        for request in passes:
            _PHASE["request"] = request
            try:
                with torch.cuda.stream(self.capture_stream), set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
                    slot_mapping=build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config),
                    is_padding=input_batch.is_padding,
                ), step_context(groups):
                    self.model(**model_inputs)
            finally:
                _PHASE["request"] = None
            torch.cuda.synchronize()
        if "capture" not in passes:
            return
        if self.phase_graph is None:
            raise RuntimeError("the capture pass ran but no phase graph was recorded (did the dummy step pipeline?)")
        self.phase_graph.captured_ctx = max_ctx


_KV_CONNECTOR_REFUSED = {"logged": False}


def _kv_connector_configured(runner: Any) -> bool:
    cfg = getattr(runner, "vllm_config", None)
    return getattr(cfg, "kv_transfer_config", None) is not None


def plan_groups(runner: Any, input_batch: Any, batch_desc: Any, block_tables, slot_mappings) -> list[PipeGroup] | None:
    """Runner side, once per real eager step after the whole step's metadata: the step's groups, or None."""
    global _RUNNER
    if mode() != 2:
        return None
    if kv_connector_policy() == "refuse" and _kv_connector_configured(runner):
        # K-6: never a silent slowdown - the pipelined step is not validated with a KV transfer connector (the production
        # KV offload tier); say so once, loudly, and run today's path.
        if not _KV_CONNECTOR_REFUSED["logged"]:
            _KV_CONNECTOR_REFUSED["logged"] = True
            logger.warning(
                "vllm-moet sm_120 prefill pipeline: VLLM_MOET_PREFILL_PIPE=2 requested but a KV transfer connector is "
                "configured (%s) and VLLM_MOET_PREFILL_PIPE_KV_CONNECTOR=refuse - the pipelined prefill stays OFF, every "
                "step runs today's path (the r26 production window of 2026-11-07: +31 %% at 163K with the offload tier). "
                "Set VLLM_MOET_PREFILL_PIPE_KV_CONNECTOR=allow once the pipelined step is validated with the connector.",
                type(getattr(runner.vllm_config, "kv_transfer_config", None)).__name__,
            )
        _STATS["today"] += 1
        return None
    if _RUNNER is None:
        _RUNNER = PipeRunner(runner)
    return _RUNNER.plan(runner, input_batch, batch_desc, block_tables, slot_mappings)


def stats() -> dict:
    d = dict(_STATS)
    if device_comm_mode() != "0" and _DEVICE_COMM:
        d["device_comm"] = _DEVICE_COMM.stats()
    if green_sms():
        sp = _GREEN["split"]
        d["green"] = {
            "asked": green_sms(), "comm_sms": None if sp is None else sp.na, "compute_sms": None if sp is None else sp.nb,
            "failed": _GREEN["failed"], "on": bool(_GREEN["on"]) and sp is not None,
            **{k: _GREEN[k] for k in ("forwards", "rows", "forwards_off", "rows_off", "moe_capped", "graph_scopes", "warmup")},
        }
    return d


# ------------------------------------------------------------------------------------------------- bits harness
_DUMP_DIR = os.environ.get("VLLM_MOET_PREFILL_PIPE_DUMP", "")
_DUMP_FORWARD = {"n": 0}


def _dump_active() -> bool:
    # never inside the phase graph's capture / dummy passes (K-8): the hook reads the tensors back to the host
    return bool(_DUMP_DIR) and _PHASE["request"] is None and not _PHASE["capturing"] and os.path.exists(
        os.path.join(_DUMP_DIR, "ENABLE")
    )


def dump_forward_begin() -> None:
    """Once per model forward (the head): numbers the forwards of this process."""
    if _DUMP_DIR and _PHASE["request"] is None:
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
    st = current_shared()
    if st is not None and n == st.total:
        # K-9: inside the shared replay tail the states hold both groups' replay rows - one entry per group over its row block,
        # the entry today's per-group tail would have written (comparable with the kv0 / r25 dumps of layers 20-39)
        lines = []
        for (lo, hi), h in zip(st.blocks, st.groups):
            e = {
                "forward": _DUMP_FORWARD["n"], "group": h.idx, "offset": h.token_offset, "layer": layer, "rows": hi - lo,
                "pos0": int(positions[lo].item()) if hi > lo else -1,
                "pos_last": int(positions[hi - 1].item()) if hi > lo else -1, "t": time.time(), "shared": True, "kind": "shared",
            }
            for name, t in tensors.items():
                part = t[lo:hi] if isinstance(t, torch.Tensor) and t.shape[0] == n else t
                e[name] = _sha(part)
                if isinstance(part, torch.Tensor):
                    e[name + "_shape"] = list(part.shape)
            lines.append(e)
        with open(os.path.join(_DUMP_DIR, f"rank{rank}.jsonl"), "a") as fh:
            for e in lines:
                fh.write(json.dumps(e) + "\n")
        return
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
    lines = [entry]
    start = 0 if g is None else g.prefill_rows_from
    if g is not None and 0 < start < n:
        # a mixed step: the prefilling request's rows alone (the decode rows ahead of them belong to another request
        # whose position in its generation is not reproducible run to run; the prefill rows' arithmetic does not
        # depend on them at an identical step shape). Full-row tensors: rows [start, n); a row-sharded tensor
        # ([n / tp, ...]): the decode rows sit in rank 0's shard.
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size

            tp = get_tensor_model_parallel_world_size()
        except Exception:  # noqa: BLE001
            tp = 1
        sub = dict(entry)
        sub.update({"kind": "prefill_rows", "rows": n - start, "pos0": int(positions[start].item()), "from": start})
        for name, t in tensors.items():
            if not isinstance(t, torch.Tensor):
                continue
            if t.shape[0] == n:
                sub[name] = _sha(t[start:])
            elif tp > 1 and t.shape[0] * tp == n:
                shard = t.shape[0]
                sub[name] = _sha(t[min(start, shard):]) if rank == 0 else _sha(t)
            else:
                sub[name] = _sha(t)
        lines.append(sub)
    with open(os.path.join(_DUMP_DIR, f"rank{rank}.jsonl"), "a") as fh:
        for e in lines:
            fh.write(json.dumps(e) + "\n")
