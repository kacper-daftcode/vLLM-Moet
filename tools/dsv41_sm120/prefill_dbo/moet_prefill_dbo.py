# SPDX-License-Identifier: Apache-2.0
"""vllm-moet: dual-batch overlap of the TP all-reduces in the DeepSeek-V4.1 prefill (sm_120, TP4, no DP/EP).

Installed by tools/dsv41_sm120/patch_vllm_prefill_dbo_sm120.py as vllm/v1/worker/moet_prefill_dbo.py; the patched
model, model state and runner files import it. Everything here is a no-op outside a microbatch thread, so the
patched code behaves as before whenever ``--enable-dbo`` is off or the step is not split.

What it adds to vLLM's ``ubatching`` protocol:

* ``all_reduce`` / ``all_gather``: the comm region around a TP collective - the collective on the comm stream once
  the compute stream reached it, the compute stream waiting for it, and one CPU handoff to the other microbatch per
  collective. Two orders (``VLLM_MOET_DBO_SCHEDULE``): ``yield_first`` (default) is vLLM's
  ``yield_and_switch_from_compute_to_comm`` / ``switch_to_compute_sync`` order (vllm#57773); ``launch_first``
  launches, then yields, so the collective's host launch never waits for the other microbatch's Python enqueue
  (measured the same on the dummy-weight model: both are paced by the Python enqueue, see the port doc). The output
  is allocated on the compute stream before the launch and the input is recorded on the comm stream, so the caching
  allocator keeps both alive for the stream that actually uses them.
* ``RowBufferSwitch``: a buffer that the layers index by batch row (the indexer's top-k indices, the candidate
  blocks) gets one storage per microbatch, and the thread that resumes swaps its own storage in (``Tensor.set_``
  from a restore hook) - every holder of the tensor sees its microbatch's rows at local row numbers, and the
  kernels the other thread already enqueued keep the pointers they were launched with.
* ``token_offset`` / ``slice_rows``: a microbatch's row range inside the step's batch, for state prepared once for
  the whole batch before the threads start (the Engram n-gram hashes and the table rows they fetch).
"""

from __future__ import annotations

import os
import threading
from typing import Any

import torch

from vllm.v1.worker import ubatching as _ub


def enabled() -> bool:
    """True on a microbatch thread (vLLM's ``dbo_enabled``)."""
    return _ub.dbo_enabled()


def context() -> Any:
    """The calling thread's ``UBatchContext``, or None outside the microbatch threads."""
    idx = _ub._THREAD_ID_TO_CONTEXT.get(threading.get_ident())
    return None if idx is None else _ub._CURRENT_CONTEXTS[idx]


def ubatch_id() -> int:
    return _ub.dbo_current_ubatch_id()


def token_offset() -> int:
    """First batch row of the calling microbatch (0 outside the threads)."""
    ctx = context()
    return 0 if ctx is None else getattr(ctx, "token_offset", 0)


def slice_rows(t: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """The calling microbatch's rows of a tensor prepared for the whole batch."""
    start = token_offset()
    return t[start : start + num_tokens]


def comm_stream_priority() -> int:
    """Priority of the microbatch comm stream (VLLM_MOET_DBO_COMM_PRIORITY, default -1 = high): the ring kernel's
    blocks then get SMs as soon as the other microbatch's GEMM blocks retire (measured +8 points of overlap
    efficiency on 4x RTX PRO 6000, nccl_graph/nccl_overlap_bench.py)."""
    try:
        return int(os.environ.get("VLLM_MOET_DBO_COMM_PRIORITY", "-1"))
    except ValueError:
        return -1


def _pynccl():
    from vllm.distributed.parallel_state import get_tp_group

    comm = getattr(get_tp_group().device_communicator, "pynccl_comm", None)
    if comm is None or comm.disabled:
        return None
    return comm


def schedule() -> str:
    """VLLM_MOET_DBO_SCHEDULE: "yield_first" (default) is vLLM's yield_and_switch_from_compute_to_comm /
    switch_to_compute_sync order (vllm#57773); "launch_first" launches the collective on the comm stream as soon
    as the calling microbatch reaches it, then hands the CPU over, so the other microbatch's Python enqueue never
    gates the collective's launch."""
    return os.environ.get("VLLM_MOET_DBO_SCHEDULE", "yield_first")


def _comm_region(ctx: Any, x: torch.Tensor, launch) -> None:
    """Run ``launch(stream)`` - a collective reading ``x`` - on the comm stream after the compute stream's work so
    far, make the compute stream wait for it, and hand the CPU to the other microbatch once per collective."""
    if schedule() == "yield_first":
        ctx.yield_and_switch_from_compute_to_comm()
        launch(ctx.comm_stream)
        x.record_stream(ctx.comm_stream)
        ctx.switch_to_compute_sync()
        return
    compute_stream = ctx.compute_stream
    comm_stream = ctx.comm_stream
    ready = torch.cuda.Event()
    ready.record(compute_stream)
    comm_stream.wait_event(ready)
    launch(comm_stream)
    x.record_stream(comm_stream)
    done = torch.cuda.Event()
    done.record(comm_stream)
    compute_stream.wait_event(done)
    ctx.yield_()


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """TP all-reduce; on a microbatch thread, inside the comm region (see module docstring)."""
    from vllm.distributed.parallel_state import get_tp_group

    ctx = context()
    comm = _pynccl() if ctx is not None else None
    if ctx is None or comm is None:
        return get_tp_group().all_reduce(x)
    x = x.contiguous()
    out = torch.empty_like(x)
    _comm_region(ctx, x, lambda stream: comm.all_reduce(x, out, stream=stream))
    return out


def all_gather(x: torch.Tensor, dim: int) -> torch.Tensor:
    """TP all-gather along ``dim`` (vLLM's layout: rank-major, then moved to ``dim``); the comm region on a
    microbatch thread."""
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    ctx = context()
    comm = _pynccl() if ctx is not None else None
    if ctx is None or comm is None:
        return group.all_gather(x, dim)
    if dim < 0:
        dim += x.dim()
    x = x.contiguous()
    shape = tuple(x.shape)
    world = group.world_size
    gathered = torch.empty((world, *shape), dtype=x.dtype, device=x.device)
    _comm_region(ctx, x, lambda stream: comm.all_gather(gathered, x, stream=stream))
    if dim == 0:
        return gathered.reshape(world * shape[0], *shape[1:])
    return gathered.movedim(0, dim).reshape(*shape[:dim], world * shape[dim], *shape[dim + 1 :])


class RowBufferSwitch:
    """One storage per microbatch for buffers the layers index by batch row."""

    def __init__(self, buffers: list[torch.Tensor], num_ubatches: int) -> None:
        self.buffers = buffers
        # Index 0 is a view of the original storage (it keeps that storage alive while the tensor points elsewhere).
        self.storages = [
            [buf.view_as(buf)] + [torch.empty_like(buf) for _ in range(num_ubatches - 1)]
            for buf in buffers
        ]
        self.current = 0
        _ub.dbo_register_restore_hook(self.switch)

    def switch(self, ubatch_idx: int) -> None:
        if ubatch_idx == self.current:
            return
        for buf, storages in zip(self.buffers, self.storages):
            buf.set_(storages[ubatch_idx])
        self.current = ubatch_idx

    def reset(self) -> None:
        self.switch(0)


def merge_outputs(outputs: list[Any]) -> Any:
    """Reassemble the per-microbatch model outputs: tensors are concatenated along the token dim, and a tuple's
    list elements (the aux hidden states a drafter reads) element-wise."""
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
                merged.append(
                    type(parts[0])(torch.cat(elems, dim=0) for elems in zip(*parts))
                )
            else:
                merged.append(torch.cat(parts, dim=0))
        return tuple(merged)
    return torch.cat(outputs, dim=0)


def find_hooked_module(model: torch.nn.Module) -> torch.nn.Module | None:
    """The submodule that prepares per-step state for the microbatches (the DeepSeek-V4.1 decoder)."""
    for module in model.modules():
        if hasattr(module, "moet_prepare_ubatched_forward"):
            return module
    return None
