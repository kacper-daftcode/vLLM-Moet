"""vllm-moet sm_120: one-shot push all-reduce over PCIe P2P for the decode-sized TP4 all-reduces of DeepSeek-V4.1.

Replaces NCCL's ring all-reduce ([M, 5120] bf16, M <= VLLM_MOET_P2P_AR_MAX_ROWS) inside the decode graphs with one
phased push kernel (one peer at a time - the platform serialises writes to different peers) and one consumer kernel
that sums the 4 contributions in fixed source order in fp32 (deterministic, identical on every rank). Larger M and
every other shape fall back to the TP group's all-reduce.

Switches: VLLM_MOET_P2P_AR=1 enables it (default off), VLLM_MOET_P2P_AR_MAX_ROWS (16), VLLM_MOET_P2P_AR_DIR (this
directory; the CUDA source p2p_allreduce_v3.cu lives next to this file), TORCH_EXTENSIONS_DIR for the build cache.

The first call is collective (IPC handle exchange over the TP group's CPU group) and must happen at the same point on
every rank - it does, because every rank reaches the first boundary all-reduce of the first forward together, and the
decision whether the fast path applies depends only on the tensor shape, which is identical across TP ranks.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

H = 5120
NSLOTS = 4
M_MAX = 64
_state: dict | None = None
_disabled_reason: str | None = None


def enabled() -> bool:
    return os.environ.get("VLLM_MOET_P2P_AR", "0") == "1"


def oneshot_max_rows() -> int:
    """Rows up to which the one-shot push is used (default 0: measured neutral against NCCL at C1, so NCCL keeps the bits)."""
    return int(os.environ.get("VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS", "0"))


def max_rows() -> int:
    """Rows up to which the two-shot (reduce-scatter + all-gather) push is used above the one-shot range."""
    return int(os.environ.get("VLLM_MOET_P2P_AR_MAX_ROWS", "64"))


def _build():
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    return load(
        name="p2p_allreduce_sm120",
        sources=[os.path.join(here, "p2p_allreduce_sm120.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"],
        verbose=False,
    )


def _init() -> dict | None:
    """Collective over the TP group: allocate the symmetric buffers and exchange IPC handles."""
    global _state, _disabled_reason
    if _state is not None or _disabled_reason is not None:
        return _state
    try:
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
        ws, rank = tp.world_size, tp.rank_in_group
        if ws != 4:
            _disabled_reason = f"tp_size={ws} (needs 4)"
            return None
        ext = _build()
        dev = torch.cuda.current_device()
        data = ext.alloc_shared(NSLOTS * 4 * M_MAX * H * 2)
        flags = ext.alloc_shared(NSLOTS * 4 * 32 * 4)
        handles: list = [None] * ws
        dist.all_gather_object(handles, (ext.ipc_handle(data), ext.ipc_handle(flags)), group=tp.cpu_group)
        peers = [r for r in range(ws) if r != rank]
        peers_data = [ext.ipc_open(handles[r][0]) for r in peers]
        peers_flags = [ext.ipc_open(handles[r][1]) for r in peers]
        ctrl = torch.zeros(8, dtype=torch.int32, device=dev)
        timeout = torch.zeros(1, dtype=torch.int32, device=dev)
        # all-gather buffers (16 KB rows x 64 x 4 sources x 4 slots = 16 MB) with their own flags and counters
        gdata = ext.alloc_shared(NSLOTS * 4 * 64 * 1024 * 16)
        gflags = ext.alloc_shared(NSLOTS * 4 * 32 * 4)
        ghandles: list = [None] * ws
        dist.all_gather_object(ghandles, (ext.ipc_handle(gdata), ext.ipc_handle(gflags)), group=tp.cpu_group)
        gpeers_data = [ext.ipc_open(ghandles[r][0]) for r in peers]
        gpeers_flags = [ext.ipc_open(ghandles[r][1]) for r in peers]
        gctrl = torch.zeros(8, dtype=torch.int32, device=dev)
        # two-shot buffers (own slots and flags: a one-shot all-reduce of another shape in the same step must not alias)
        data2 = ext.alloc_shared(NSLOTS * 4 * M_MAX * H * 2)
        flags2 = ext.alloc_shared(NSLOTS * 4 * 32 * 4)
        h2: list = [None] * ws
        dist.all_gather_object(h2, (ext.ipc_handle(data2), ext.ipc_handle(flags2)), group=tp.cpu_group)
        peers_data2 = [ext.ipc_open(h2[r][0]) for r in peers]
        peers_flags2 = [ext.ipc_open(h2[r][1]) for r in peers]
        ctrl2 = torch.zeros(8, dtype=torch.int32, device=dev)
        _state = dict(ext=ext, rank=rank, data=data, flags=flags, peers_data=peers_data, peers_flags=peers_flags,
                      ctrl=ctrl, timeout=timeout, handles=handles, gdata=gdata, gflags=gflags, gpeers_data=gpeers_data,
                      gpeers_flags=gpeers_flags, gctrl=gctrl, data2=data2, flags2=flags2, peers_data2=peers_data2,
                      peers_flags2=peers_flags2, ctrl2=ctrl2)
        logger.info("vllm-moet sm_120 P2P all-reduce: rank %d of %d, one-shot push for M <= %d rows, two-shot "
                    "(reduce-scatter + all-gather) up to %d rows, NCCL otherwise (VLLM_MOET_P2P_AR=0 turns it off)",
                    rank, ws, oneshot_max_rows(), max_rows())
        return _state
    except Exception as exc:  # noqa: BLE001
        _disabled_reason = repr(exc)
        logger.warning("vllm-moet sm_120 P2P all-reduce unavailable, NCCL stays: %r", exc)
        return None


def mode(x: torch.Tensor) -> str:
    """'oneshot', 'twoshot' or 'nccl' for this tensor (shape-only decision, identical on every TP rank)."""
    if not (x.dim() == 2 and x.shape[1] == H and x.dtype == torch.bfloat16 and x.is_cuda):
        return "nccl"
    m = x.shape[0]
    if 0 < m <= min(oneshot_max_rows(), M_MAX):
        return "oneshot"
    if 0 < m <= min(max_rows(), M_MAX):
        return "twoshot"
    return "nccl"


def applicable(x: torch.Tensor) -> bool:
    return mode(x) != "nccl"


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """TP all-reduce of x: the P2P fast path when applicable, otherwise the TP group's all-reduce."""
    from vllm.distributed.parallel_state import get_tp_group

    st = _init() if enabled() else None
    how = mode(x) if st is not None else "nccl"
    if how == "nccl":
        return get_tp_group().all_reduce(x)
    x = x.contiguous()
    out = torch.empty_like(x)
    ext = st["ext"]
    if how == "oneshot":
        ext.push(x, M_MAX, st["rank"], st["peers_data"], st["peers_flags"], st["ctrl"], 64)
        ext.wait_reduce(x, st["rank"], M_MAX, st["data"], st["flags"], st["ctrl"], out, None, None, None, 0.0, st["timeout"])
    else:
        ext.all_reduce_twoshot(x, out, M_MAX, st["rank"], st["data2"], st["flags2"], st["peers_data2"], st["peers_flags2"],
                               st["ctrl2"], st["timeout"], 64)
    return out


def gather_applicable(x: torch.Tensor) -> bool:
    if not (x.dim() == 2 and x.is_cuda and 0 < x.shape[0] <= 64):
        return False
    row_bytes = x.shape[1] * x.element_size()
    return row_bytes % 16 == 0 and row_bytes <= 16384


def all_gather(x: torch.Tensor) -> torch.Tensor:
    """TP all-gather of x along the last dim (bit-exact): the P2P fast path when applicable, else the TP group's."""
    from vllm.distributed.parallel_state import get_tp_group

    st = _init() if enabled() else None
    if st is None or not gather_applicable(x):
        return get_tp_group().all_gather(x, dim=-1)
    x = x.contiguous()
    out = torch.empty((x.shape[0], 4 * x.shape[1]), dtype=x.dtype, device=x.device)
    ext = st["ext"]
    ext.gather_push(x, st["rank"], st["gpeers_data"], st["gpeers_flags"], st["gctrl"], 64)
    ext.wait_gather(x, st["rank"], st["gdata"], st["gflags"], st["gctrl"], out, st["timeout"])
    return out


def timeouts() -> int:
    """Number of flag-wait timeouts seen so far (debug; synchronises)."""
    return int(_state["timeout"].item()) if _state else 0
