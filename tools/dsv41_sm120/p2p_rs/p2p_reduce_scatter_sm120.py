"""vllm-moet sm_120: TP4 reduce-scatter of a [M, 5120] bf16 partial in the order of NCCL's *all-reduce* (AR track, P).

The sequence-parallel prefill boundary (patch_vllm_prefill_sp_sm120.py) replaces the served all-reduce by a
reduce-scatter and runs the mHC boundary on this rank's row shard. NCCL's reduce-scatter sums the rows of rank r as
((x[r+1] + x[r+2]) + x[r+3]) + x[r] (ring order), its all-reduce finishes an element on the rank of the element's
ring *chunk*, so three quarters of the rows are summed in another cyclic order and 32 % of the elements move by one
bf16 ulp (block B, 29.10). This module pushes the peers' row blocks over PCIe P2P (p2p_reduce_scatter_sm120.cu, one
peer at a time) and adds the four partials of every 8-element group in the order the all-reduce used for that group,
read from a table probed on the TP group's NCCL communicator (nccl_order_tables: 8 all-reduces of [M, 5120] per
distinct M, ~11 ms at 4096, cached) - the arithmetic of tools/dsv41_sm120/p2p_ar that reproduces NCCL's bits for
M <= 64 today, at M <= 4096 rows.

Switches: VLLM_MOET_PREFILL_SP_RS=p2p enables it (default nccl = vLLM's sp_reduce_scatter), VLLM_MOET_P2P_RS_DIR
(this directory, for the patched model), VLLM_MOET_P2P_RS_CTAS (CTAs per destination of the push, default 64: the
PCIe link saturates at 50 GB/s from 64 up, and fewer CTAs leave the SMs to the overlapped mHC side stream),
VLLM_MOET_P2P_RS_UNROLL (16-B chunks in flight per thread, default 4), VLLM_MOET_P2P_RS_SPLIT (1 = one launch per
peer, no spinning CTAs, default; 0 = one launch with CTA-counter phases), VLLM_MOET_P2P_RS_REDUCE_CTAS (persistent
consumer CTAs, default 256; 0 = one per row), VLLM_MOET_P2P_RS_TABLE_CACHE (distinct M kept, default 16, 2.6 MB
each at 4096), VLLM_MOET_P2P_RS_VERIFY (1 = check every new M against NCCL's all-reduce on random data before using
it, default 1), TORCH_EXTENSIONS_DIR (build cache).

Collective calls: _init (IPC handle exchange over the TP group's CPU group) and the probe of every new M happen at
the first reduce-scatter of that M - the same point on every rank, because M is the step's token count.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict

import torch

logger = logging.getLogger("vllm.moet.p2p_reduce_scatter_sm120")

H = 5120
M_MAX = int(os.environ.get("VLLM_MOET_P2P_RS_M_MAX", "4096"))
NSRC = 4
_state: dict | None = None
_disabled_reason: str | None = None


def enabled() -> bool:
    return os.environ.get("VLLM_MOET_PREFILL_SP_RS", "nccl") == "p2p"


def ctas_per_dest() -> int:
    return int(os.environ.get("VLLM_MOET_P2P_RS_CTAS", "64"))


def unroll() -> int:
    return int(os.environ.get("VLLM_MOET_P2P_RS_UNROLL", "4"))


def split_launches() -> bool:
    return os.environ.get("VLLM_MOET_P2P_RS_SPLIT", "1") == "1"


def reduce_ctas() -> int:
    return int(os.environ.get("VLLM_MOET_P2P_RS_REDUCE_CTAS", "256"))


def table_cache_size() -> int:
    return int(os.environ.get("VLLM_MOET_P2P_RS_TABLE_CACHE", "16"))


def verify_new_m() -> bool:
    return os.environ.get("VLLM_MOET_P2P_RS_VERIFY", "1") == "1"


def _build():
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    name = "p2p_reduce_scatter_sm120" if M_MAX == 4096 else f"p2p_reduce_scatter_sm120_m{M_MAX}"
    return load(
        name=name,
        sources=[os.path.join(here, "p2p_reduce_scatter_sm120.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", f"-DP2P_RS_M_MAX={M_MAX}"],
        verbose=False,
    )


_POPC = (0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4)
_LOW = (0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0)  # lowest set bit of a 4-bit mask
_HIGH = (0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)  # highest set bit


def nccl_order_codes(nccl_all_reduce, rank: int, device, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per element of an [m, H] bf16 all-reduce on this communicator: (code, ok). code = r0 | r1 << 2 | r2 << 4 |
    r3 << 6 for ((x_r0 + x_r1) + x_r2) + x_r3 with a bf16 rounding per add; ok = the element's sum is such a chain.

    Rank r holds 1.0 and the others s: with s = 2^-9 the result is 1 + 2^-7 exactly where 1.0 is added last
    (2^-8 + 2^-9 + 1 rounds up, 1 + 2^-9 and the tie 1 + 2^-8 round to 1), with s = 3 * 2^-10 where it is added
    third or last. Which of the first two comes first does not change the bits. Collective (8 all-reduces).
    """
    popc, low, high = (torch.tensor(t, dtype=torch.int32, device=device) for t in (_POPC, _LOW, _HIGH))
    hit = torch.tensor(1.0 + 2.0 ** -7, dtype=torch.bfloat16, device=device)
    n = m * H
    last = torch.zeros(n, dtype=torch.int32, device=device)
    late = torch.zeros(n, dtype=torch.int32, device=device)
    for r in range(NSRC):
        for s, mask in ((2.0 ** -9, last), (3 * 2.0 ** -10, late)):
            x = torch.full((m, H), 1.0 if rank == r else s, dtype=torch.bfloat16, device=device)
            mask |= (nccl_all_reduce(x).view(-1) == hit).int() << r
    first = 15 ^ late
    ok = (popc[last] == 1) & (popc[late] == 2) & ((late & last) == last)
    code = low[first] | (high[first] << 2) | (low[late ^ last] << 4) | (low[last] << 6)
    return code, ok


def nccl_order_tables(nccl_all_reduce, rank: int, device, sizes, diagnostics: dict | None = None) -> dict[int, torch.Tensor]:
    """NCCL's summation order per 8 consecutive elements for every m in sizes: a uint8 [m * H / 8] tensor (the
    kernel's table). An m whose sums are not chains, or whose 8-element groups are not uniform, is left out.
    diagnostics (optional dict) gets per m: the fraction of groups failing each condition, the distinct codes, the
    run lengths of constant order along the buffer and the fraction of rows with one order throughout."""
    tables = {}
    for m in sizes:
        code, ok = nccl_order_codes(nccl_all_reduce, rank, device, m)
        code8 = code.view(-1, 8)
        uniform = (code8 == code8[:, :1]).all(dim=1)
        ok8 = ok.view(-1, 8).all(dim=1)
        good = bool(ok8.all()) and bool(uniform.all())
        if diagnostics is not None:
            diagnostics[m] = _diagnose(code8, ok8, uniform, m)
        if good:
            tables[m] = code8[:, 0].to(torch.uint8).contiguous()
    return tables


def _diagnose(code8: torch.Tensor, ok8: torch.Tensor, uniform: torch.Tensor, m: int) -> dict:
    g = code8.shape[0]
    codes = code8[:, 0]
    change = torch.nonzero(codes[1:] != codes[:-1]).view(-1) + 1
    bounds = torch.cat([torch.zeros(1, dtype=change.dtype, device=change.device), change,
                        torch.tensor([g], dtype=change.dtype, device=change.device)])
    runs = (bounds[1:] - bounds[:-1]).cpu()
    rows = codes.view(m, H // 8)
    row_uniform = (rows == rows[:, :1]).all(dim=1)
    vals, counts = torch.unique(codes, return_counts=True)
    return dict(
        groups=g,
        frac_not_chain=float((~ok8).float().mean()),
        frac_not_uniform=float((~uniform).float().mean()),
        n_codes=int(vals.numel()),
        codes={int(v): int(c) for v, c in zip(vals.cpu(), counts.cpu())},
        n_runs=int(runs.numel()),
        run_groups_min=int(runs.min()), run_groups_median=float(runs.float().median()), run_groups_max=int(runs.max()),
        run_bytes_median=float(runs.float().median()) * 16,
        rows_uniform=int(row_uniform.sum()), rows=m,
        change_points_row_aligned=int((change % (H // 8) == 0).sum()) if change.numel() else 0,
        change_points=int(change.numel()),
    )


def decode_code(code: int) -> tuple[int, int, int, int]:
    """(r0, r1, r2, r3) of a table byte."""
    return tuple((code >> (2 * p)) & 3 for p in range(4))


def _init() -> dict | None:
    """Collective over the TP group: allocate the slots, exchange the IPC handles."""
    global _state, _disabled_reason
    if _state is not None or _disabled_reason is not None:
        return _state
    try:
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
        ws, rank = tp.world_size, tp.rank_in_group
        if ws != NSRC:
            _disabled_reason = f"tp_size={ws} (needs 4)"
            return None
        ext = _build()
        dev = torch.cuda.current_device()
        data = ext.alloc_shared(ext.data_bytes())
        flags = ext.alloc_shared(ext.flags_bytes())
        handles: list = [None] * ws
        dist.all_gather_object(handles, (ext.ipc_handle(data), ext.ipc_handle(flags)), group=tp.cpu_group)
        peers = [r for r in range(ws) if r != rank]
        _state = dict(
            ext=ext, tp=tp, rank=rank, dev=dev, data=data, flags=flags,
            peers_data=[ext.ipc_open(handles[r][0]) for r in peers],
            peers_flags=[ext.ipc_open(handles[r][1]) for r in peers],
            ctrl=torch.zeros(8, dtype=torch.int32, device=dev),
            timeout=torch.zeros(1, dtype=torch.int32, device=dev),
            orders=OrderedDict(), failed=set(), probe_ms={},
        )
        logger.info("vllm-moet sm_120 P2P reduce-scatter (NCCL all-reduce order): rank %d of %d, slots %.1f MB, M <= %d, "
                    "push %d CTAs/peer x unroll %d, %s, consumer %d CTAs (VLLM_MOET_PREFILL_SP_RS=nccl turns it off)", rank, ws,
                    ext.data_bytes() / 1e6, M_MAX, ctas_per_dest(), unroll(),
                    "one launch per peer" if split_launches() else "one launch, CTA-counter phases", reduce_ctas())
        return _state
    except Exception as exc:  # noqa: BLE001
        _state = None
        _disabled_reason = repr(exc)
        logger.warning("vllm-moet sm_120 P2P reduce-scatter unavailable, NCCL's reduce-scatter stays: %r", exc)
        return None


def _run(st: dict, x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    c = (x.shape[0] + NSRC - 1) // NSRC
    out = torch.empty((c, H), dtype=x.dtype, device=x.device)
    st["ext"].reduce_scatter(x, out, st["rank"], st["data"], st["flags"], st["peers_data"], st["peers_flags"],
                             st["ctrl"], order, st["timeout"], ctas_per_dest(), unroll(), split_launches(), reduce_ctas())
    return out


def _table(st: dict, m: int) -> torch.Tensor | None:
    """The order table of m: cached, or probed now on the TP group's NCCL communicator (collective; every rank
    sees the same m). None when NCCL's order at this m is not a uniform chain per 8 elements, or when the kernel
    does not reproduce NCCL's all-reduce rows on random data (VLLM_MOET_P2P_RS_VERIFY) - then NCCL's
    reduce-scatter is used for this m, and the verdict is the same on every rank."""
    orders = st["orders"]
    if m in orders:
        orders.move_to_end(m)
        return orders[m]
    if m in st["failed"]:
        return None
    import torch.distributed as dist

    tp, rank, dev = st["tp"], st["rank"], st["dev"]
    comm = getattr(tp.device_communicator, "pynccl_comm", None)
    if comm is None or comm.disabled or torch.cuda.is_current_stream_capturing():
        return None
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    diag: dict = {}
    tables = nccl_order_tables(comm.all_reduce, rank, dev, [m], diag)
    good = m in tables
    if good and verify_new_m():
        g = torch.Generator(device=torch.device("cuda", dev)).manual_seed(1009 * m + rank)
        x = (0.5 * torch.randn(m, H, generator=g, device=dev)).to(torch.bfloat16)
        ref = comm.all_reduce(x)
        c = (m + NSRC - 1) // NSRC
        n = max(0, min(c, m - rank * c))
        out = _run(st, x, tables[m])
        mine = bool(torch.equal(out[:n].view(torch.int16), ref[rank * c:rank * c + n].view(torch.int16))) and \
            bool((out[n:] == 0).all())
        every: list = [None] * tp.world_size
        dist.all_gather_object(every, mine, group=tp.cpu_group)
        good = all(every)
    t1.record()
    t1.synchronize()
    st["probe_ms"][m] = t0.elapsed_time(t1)
    d = diag.get(m, {})
    if not good:
        st["failed"].add(m)
        logger.warning("vllm-moet sm_120 P2P reduce-scatter: M=%d stays on NCCL's reduce-scatter (order not a uniform "
                       "chain per 8 elements: %.4f / %.4f of the groups; verified: %s)", m, d.get("frac_not_chain", -1),
                       d.get("frac_not_uniform", -1), "no" if m in tables else "n/a")
        return None
    orders[m] = tables[m]
    while len(orders) > table_cache_size():
        orders.popitem(last=False)
    logger.info("vllm-moet sm_120 P2P reduce-scatter: NCCL's all-reduce order probed for M=%d in %.1f ms (%d codes, "
                "%d runs, %d/%d rows uniform)", m, st["probe_ms"][m], d.get("n_codes", 0), d.get("n_runs", 0),
                d.get("rows_uniform", 0), m)
    return orders[m]


def applicable(x: torch.Tensor) -> bool:
    return x.dim() == 2 and x.shape[1] == H and x.dtype == torch.bfloat16 and x.is_cuda and 0 < x.shape[0] <= M_MAX


def reduce_scatter(x: torch.Tensor) -> torch.Tensor | None:
    """[M, H] bf16 partial -> this rank's [ceil(M/4), H] rows of NCCL's all-reduce (bits), zero rows beyond M - the
    shape vLLM's sp_reduce_scatter returns. None when the switch is off or the shape / this M is not served
    (the caller falls back to sp_reduce_scatter)."""
    if not enabled():
        return None
    st = _init()
    if st is None or not applicable(x):
        return None
    order = _table(st, x.shape[0])
    if order is None:
        return None
    return _run(st, x.contiguous(), order)


def timeouts() -> int:
    """Flag-wait timeouts so far (debug; synchronises)."""
    return int(_state["timeout"].item()) if _state else 0
