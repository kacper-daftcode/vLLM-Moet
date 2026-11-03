"""vllm-moet sm_120: one-shot push all-reduce over PCIe P2P for the decode-sized TP4 all-reduces of DeepSeek-V4.1.

Replaces NCCL's ring all-reduce ([M, 5120] bf16, M <= VLLM_MOET_P2P_AR_MAX_ROWS) inside the decode graphs with one
phased push kernel (one peer at a time - the platform serialises writes to different peers) and one consumer kernel
that adds the 4 contributions in NCCL's own order with NCCL's bf16 rounding after every add, so the result has NCCL's
bits. That order depends on the element's position (ring chunks, channels, protocol steps); it is measured once at
start-up on the TP group's NCCL communicator for every M (nccl_order_tables) and each M is then checked against NCCL
on random data - an M that does not reproduce NCCL bit for bit stays on NCCL. Larger M and every other shape fall back
to the TP group's all-reduce.

Switches: VLLM_MOET_P2P_AR=1 enables it (default off), VLLM_MOET_P2P_AR_MAX_ROWS (16), VLLM_MOET_P2P_AR_DIR (this
directory; the CUDA source p2p_allreduce_v3.cu lives next to this file), TORCH_EXTENSIONS_DIR for the build cache,
VLLM_MOET_P2P_AR_H (row width the kernels are compiled for: 5120 = DeepSeek-V4.1-Flash, the default and the extension
name the shipped images cache; 4096 = GLM-5.3-Flash, built as p2p_allreduce_sm120_h4096 with -DP2P_AR_H=4096).

The first call is collective (IPC handle exchange over the TP group's CPU group) and must happen at the same point on
every rank - it does, because every rank reaches the first boundary all-reduce of the first forward together, and the
decision whether the fast path applies depends only on the tensor shape, which is identical across TP ranks.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("vllm.moet.p2p_allreduce_sm120")  # under vLLM's logger: its handler shows INFO

H = int(os.environ.get("VLLM_MOET_P2P_AR_H", "5120"))
if H % 128 != 0 or H <= 0:
    raise ValueError(f"VLLM_MOET_P2P_AR_H={H}: the kernels need a multiple of 128 (H/16 threads per row, H/32 per slice)")
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
    # H = 5120 keeps the name and flags the shipped DeepSeek images have cached; other widths get their own extension
    name, flags = "p2p_allreduce_sm120", ["-O3", "-gencode=arch=compute_120a,code=sm_120a"]
    if H != 5120:
        name, flags = f"p2p_allreduce_sm120_h{H}", flags + [f"-DP2P_AR_H={H}"]
    return load(
        name=name,
        sources=[os.path.join(here, "p2p_allreduce_sm120.cu")],
        extra_cuda_cflags=flags,
        verbose=False,
    )


_POPC = (0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4)
_LOW = (0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0)  # lowest set bit of a 4-bit mask
_HIGH = (0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)  # highest set bit


def nccl_order_tables(nccl_all_reduce, rank: int, device, sizes) -> dict[int, torch.Tensor]:
    """NCCL's summation order for every 8 consecutive elements of an [m, H] bf16 all-reduce, per m in sizes: a uint8
    [m * H / 8] tensor, r0 | r1 << 2 | r2 << 4 | r3 << 6 for ((x_r0 + x_r1) + x_r2) + x_r3 with a bf16 rounding per add
    (what the kernels add up). Collective: every rank calls it with the same sizes. An m whose sums are not such
    chains, the same for the 8 elements of a group, is left out.

    Rank r holds 1.0 and the others s: with s = 2^-9 the result is 1 + 2^-7 exactly where 1.0 is added last
    (2^-8 + 2^-9 + 1 rounds up, 1 + 2^-9 and the tie 1 + 2^-8 round to 1), with s = 3 * 2^-10 where it is added third
    or last. Which of the first two comes first does not change the bits.
    """
    popc, low, high = (torch.tensor(t, dtype=torch.int32, device=device) for t in (_POPC, _LOW, _HIGH))
    hit = torch.tensor(1.0 + 2.0 ** -7, dtype=torch.bfloat16, device=device)
    tables = {}
    for m in sizes:
        n = m * H
        last = torch.zeros(n, dtype=torch.int32, device=device)
        late = torch.zeros(n, dtype=torch.int32, device=device)
        for r in range(4):
            for s, mask in ((2.0 ** -9, last), (3 * 2.0 ** -10, late)):
                x = torch.full((n,), 1.0 if rank == r else s, dtype=torch.bfloat16, device=device)
                mask |= (nccl_all_reduce(x) == hit).int() << r
        first = 15 ^ late
        ok = (popc[last] == 1) & (popc[late] == 2) & ((late & last) == last)
        code = low[first] | (high[first] << 2) | (low[late ^ last] << 4) | (low[last] << 6)
        code = code.view(-1, 8)
        if bool(ok.all()) and bool((code == code[:, :1]).all()):
            tables[m] = code[:, 0].to(torch.uint8).contiguous()
    return tables


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
        # all-gather buffers (64 KB rows x 64 x 4 sources x 4 slots = 64 MB: the lm_head's 32,320-wide bf16 logits) with
        # their own flags and counters
        gdata = ext.alloc_shared(NSLOTS * 4 * 64 * 4096 * 16)
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
                      peers_flags2=peers_flags2, ctrl2=ctrl2, orders={})
        _state["orders"] = _measure_orders(tp, rank)
        kept = sorted(_state["orders"])
        span = ("none" if not kept else f"{kept[0]}-{kept[-1]}" if len(kept) == kept[-1] - kept[0] + 1 else str(kept))
        logger.info("vllm-moet sm_120 P2P all-reduce: rank %d of %d, NCCL's summation order reproduced for %d of %d row "
                    "counts (%s); one-shot push for M <= %d rows, two-shot (reduce-scatter + all-gather) up to %d rows, "
                    "NCCL otherwise (VLLM_MOET_P2P_AR=0 turns it off)", rank, ws, len(kept), M_MAX, span,
                    oneshot_max_rows(), max_rows())
        return _state
    except Exception as exc:  # noqa: BLE001
        _state = None
        _disabled_reason = repr(exc)
        logger.warning("vllm-moet sm_120 P2P all-reduce unavailable, NCCL stays: %r", exc)
        return None


def _run(st: dict, x: torch.Tensor, order: torch.Tensor, how: str) -> torch.Tensor:
    out = torch.empty_like(x)
    ext = st["ext"]
    if how == "oneshot":
        ext.push(x, M_MAX, st["rank"], st["peers_data"], st["peers_flags"], st["ctrl"], 64)
        ext.wait_reduce(x, st["rank"], M_MAX, st["data"], st["flags"], st["ctrl"], order, out, None, None, None, 0.0,
                        st["timeout"])
    else:
        ext.all_reduce_twoshot(x, out, M_MAX, st["rank"], st["data2"], st["flags2"], st["peers_data2"], st["peers_flags2"],
                               st["ctrl2"], order, st["timeout"], 64)
    return out


def _measure_orders(tp, rank: int) -> dict[int, torch.Tensor]:
    """NCCL's order for M = 1..M_MAX (nccl_order_tables) on the TP group's NCCL communicator - the one the decode graphs
    capture - then, per M, the one-shot and two-shot results against NCCL's on random data. Kept: the M that every rank
    reproduced bit for bit (the same set on every rank, so all ranks take the same path)."""
    import torch.distributed as dist

    comm = getattr(tp.device_communicator, "pynccl_comm", None)
    if comm is None or comm.disabled or torch.cuda.is_current_stream_capturing():
        logger.warning("vllm-moet sm_120 P2P all-reduce: NCCL's order cannot be measured here (no PyNccl communicator or "
                       "inside a graph capture), NCCL stays")
        return {}
    dev = torch.cuda.current_device()
    tables = nccl_order_tables(comm.all_reduce, rank, dev, range(1, M_MAX + 1))
    good = []
    for m, order in tables.items():
        g = torch.Generator(device=torch.device("cuda", dev)).manual_seed(1009 * m + rank)
        x = torch.randn(m, H, generator=g, device=dev).to(torch.bfloat16)
        ref = comm.all_reduce(x).view(torch.int16)
        if all(torch.equal(_run(_state, x, order, how).view(torch.int16), ref) for how in ("oneshot", "twoshot")):
            good.append(m)
    every: list = [None] * tp.world_size
    dist.all_gather_object(every, good, group=tp.cpu_group)
    keep = set(good).intersection(*map(set, every))
    return {m: tables[m] for m in sorted(keep)}


def mode(x: torch.Tensor) -> str:
    """'oneshot', 'twoshot' or 'nccl' for this tensor (shape-only decision, identical on every TP rank)."""
    if not (x.dim() == 2 and x.shape[1] == H and x.dtype == torch.bfloat16 and x.is_cuda):
        return "nccl"
    m = x.shape[0]
    if _state is None or m not in _state["orders"]:
        return "nccl"
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
    return _run(st, x, st["orders"][x.shape[0]], how)


def fused_max_rows() -> int:
    """Rows up to which the one-shot push feeds the boundary kernel's fused consumer (step 24; default 0 = off)."""
    return int(os.environ.get("VLLM_MOET_P2P_AR_FUSED_MAX_ROWS", "0"))


def push_for_fused(x: torch.Tensor) -> dict | None:
    """Issue the one-shot push of x (this rank's partial) for a consumer fused into the next kernel; returns the
    buffers that consumer needs (data, flags, ctrl, rank, m_max, order) or None when the fused path does not apply."""
    st = _init() if enabled() else None
    if st is None or not (x.dim() == 2 and x.shape[1] == H and x.dtype == torch.bfloat16 and x.is_cuda
                          and 0 < x.shape[0] <= min(fused_max_rows(), M_MAX) and x.shape[0] in st["orders"]):
        return None
    x = x.contiguous()
    st["ext"].push(x, M_MAX, st["rank"], st["peers_data"], st["peers_flags"], st["ctrl"], 64)
    return dict(data=st["data"], flags=st["flags"], ctrl=st["ctrl"], rank=st["rank"], m_max=M_MAX,
                order=st["orders"][x.shape[0]], x=x)


def gather_applicable(x: torch.Tensor) -> bool:
    if not (x.dim() == 2 and x.is_cuda and 0 < x.shape[0] <= 64):
        return False
    row_bytes = x.shape[1] * x.element_size()
    return row_bytes % 16 == 0 and row_bytes <= 4096 * 16


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
