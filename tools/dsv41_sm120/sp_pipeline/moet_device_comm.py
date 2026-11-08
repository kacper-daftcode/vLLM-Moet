# SPDX-License-Identifier: Apache-2.0
"""vllm-moet sm_120 - MK-4 (program megakernel, stage 3, milestone 1 "the hybrid"): the sequence-parallel collectives of the
pipelined prefill step from the device instead of NCCL.

Installed next to moet_prefill_pipe.py as vllm/v1/worker/moet_device_comm.py (patch_vllm_prefill_device_comm_sm120.py), together
with moet_device_comm_sm120.cu (built with torch.utils.cpp_extension on first use, cached in TORCH_EXTENSIONS_DIR). Everything here
is a no-op unless VLLM_MOET_PREFILL_PIPE_DEVICE_COMM is 1 or verify AND the call comes from an open comm phase of a pipelined group
(moet_prefill_pipe.current_group(): the two-group step of VLLM_MOET_PREFILL_PIPE=2); every other collective stays NCCL's.

What it replaces (per 4 096-row group and layer: MK/REPORT.md 1.1): RS1 / RS2 - the reduce-scatter of the [T, 5120] bf16 partial
(41.9 MB, 1 024 rows per rank) in front of both mHC boundaries - and the all-gathers behind them: the step-45 MoE input as e4m3 +
scales + top-6 (sp_moe_gather_sm120: two gathers), the shard projections [T, 1792] bf16 (attention.py), the full layer input at the
AUX=full layers. NCCL keeps the replay tails' all-reduces, the indexer's all_gatherv, the connector.

How: one small kernel instance per collective on a comm-kernel stream per group (K_g), ordered against the pipeline's comm stream by
events only - the stream reaches the collective's input, K_g starts the kernel, the comm stream waits for the kernel's end. The
kernel (moet_device_comm_sm120.cu): n_push pusher CTAs push this rank's rows over PCIe P2P into the peers' receive slots / gather
arenas, phased per peer (one destination at a time), in `pieces` pieces each closed by a sys fence + a flag line on the receiver;
n_red reducer CTAs reduce every landed piece of the RS in NCCL's own per-element summation order (the order table, below) and
write this rank's rows; for a gather they copy the own shard and wait for the peers' pieces. 32 threads x <= 32 registers x 6.3 KB
smem per CTA: the instance co-resides with every compute kernel of the layer that leaves 1 024 registers per SM (fc1 v10, the CUTLASS
MXFP8 GEMMs, the SASS attention, fc2, cuBLAS, mHC) - fc1 v8 (2 x 256 x 128 = the whole file) needs the 120-register cap (MK-4 item 2).
Every device-side wait is bounded (VLLM_MOET_PREFILL_PIPE_DEVICE_COMM_WATCHDOG_S, default 30): on expiry the kernel writes an error
record to pinned memory and traps - a protocol bug is a failed step, never a spinning kernel (step_done() raises with the record).

Bits (MK/BRIEF_MK4.md item 1): the reduce-scatter's result is NCCL's bit for bit only if the four partials are added in NCCL's
order per element with a bf16 rounding per add. The order is probed on the group's own communicator at the first collective of
each distinct row count M (8 patterned reduce-scatters, ~10 ms: rank r holds 1.0, the others 2^-9 / 3*2^-10 - tools/dsv41_sm120/p2p_ar's
technique applied to the reduce-scatter) into a table of one byte per 4 elements (r0 | r1<<2 | r2<<4 | r3<<6: ((x_r0 + x_r1) + x_r2) + x_r3);
an M whose order is not a chain uniform per 4 elements stays on NCCL (the same verdict on every rank, exchanged over the TP CPU
group). In mode 1 the first VLLM_MOET_PREFILL_PIPE_DEVICE_COMM_VERIFY_FIRST (3) collectives of each kind and M are also run on
NCCL and compared (torch.equal; NCCL's result is what the model consumes until the comparison passed on every rank); mode verify
compares EVERY collective and always consumes NCCL's result (counters in stats(), one log line per step). The all-gathers are copies.

Switches: VLLM_MOET_PREFILL_PIPE_DEVICE_COMM=0|1|verify (0); _PUSH_CTAS (16), _RED_CTAS (128: the reduce is issue-bound at one warp
per SM - 32 reducers took 830 us for 42 MB, MK/mk4/out/unit2.log), _PIECES (8, 1..32: the most pieces per push phase = the landing
granularity the reducers overlap with; each costs a sys fence per pusher), _PIECE_KB (512: a launch uses bytes-per-phase / this pieces,
so a 213-KB gather takes one piece and a 10.5-MB reduce-scatter eight), _SPLIT (1: pushers-only resident, 1-CTA bounded waits, a transient
256-thread reduce; 0 = the resident 144-CTA instance of the first in-situ arms), _PUSH_WARPS (4: warps per pusher CTA in split mode),
_REDUCE_GRID (376), _UNROLL (4: 16-B loads in flight per
pusher lane), _LINK_TOKEN (1: the two groups' instances take turns per push phase - one destination per GPU at a time), _ARENA_MB (48:
the gather arena per group and parity; a gather that does not fit stays on NCCL), _SLOT_ROWS (1024: the receive slot per source =
the largest shard), _STREAM (own | comm: a comm-kernel stream per group, or the pipeline's comm stream itself), _STREAM_PRIORITY (0), _WATCHDOG_S (30), _VERIFY_FIRST (3), _RS=0|1 and _AG=0|1 (1: per-kind master switches).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

import torch

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.vllm_moet.device_comm")
except Exception:  # noqa: BLE001 - unit tests outside vLLM
    import logging

    logger = logging.getLogger("vllm.vllm_moet.device_comm")

NSRC = 4
_POPC = (0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4)
_LOW = (0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0)
_HIGH = (0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)
_ENV = "VLLM_MOET_PREFILL_PIPE_DEVICE_COMM"


# ------------------------------------------------------------------------------------------------------- switches
def mode() -> str:
    v = os.environ.get(_ENV, "0")
    return v if v in ("0", "1", "verify") else "0"


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(f"{_ENV}_{name}", str(default)))
    except ValueError:
        return default


def n_push() -> int:
    """16 single-warp pushers saturate the link and cost the least (MK/mk4/out/unit4.log: RS 896 us at 16, 920 at 32, 978 at 48, 1 031 at 64
    vs NCCL 909 - MK-2/3's law: fewer pushers are faster)."""
    return max(1, min(188, _int("PUSH_CTAS", 16)))


def n_red() -> int:
    return max(1, min(188, _int("RED_CTAS", 128)))


def pieces() -> int:
    """The most pieces per push phase (the flag layout); a launch takes min(this, bytes per phase / piece_min_bytes())."""
    return max(1, min(32, _int("PIECES", 8)))


def piece_min_bytes() -> int:
    return max(16384, _int("PIECE_KB", 512) * 1024)


def pieces_for(bytes_per_phase: int) -> int:
    return max(1, min(pieces(), bytes_per_phase // piece_min_bytes()))


def ag_pieces() -> int:
    return max(1, min(pieces(), _int("AG_PIECES", 1)))


def unroll() -> int:
    return 4 if _int("UNROLL", 4) >= 4 else 2


def link_token() -> int:
    return 1 if _int("LINK_TOKEN", 1) else 0


def arena_mb() -> int:
    return max(1, _int("ARENA_MB", 48))


def slot_rows() -> int:
    return max(4, _int("SLOT_ROWS", 1024))


def stream_priority() -> int:
    return _int("STREAM_PRIORITY", 0)


def split_mode() -> bool:
    """VLLM_MOET_PREFILL_PIPE_DEVICE_COMM_SPLIT=1 (default): only the pushers are resident during a transfer; the landing is awaited by one
    bounded CTA on the comm stream and the reduce is a transient 256-thread kernel behind it (MK/REPORT.md 4.6 item 1). 0 = the first
    form: pushers + resident single-warp reducers in one instance (MK/REPORT.md 4.5: starves NCCL's remaining kernel of empty SMs)."""
    return bool(_int("SPLIT", 1))


def push_warps() -> int:
    """Warps per pusher CTA in split mode (1..4; 4 = 4 096 registers per CTA: co-resident with fc1 at 120 registers, attention, fc2, cuBLAS,
    not with the CUTLASS GEMMs - 16 such CTAs find an SM anyway)."""
    return max(1, min(4, _int("PUSH_WARPS", 4)))


def reduce_grid() -> int:
    return max(1, min(2048, _int("REDUCE_GRID", 376)))


def stream_mode() -> str:
    """VLLM_MOET_PREFILL_PIPE_DEVICE_COMM_STREAM: "own" (default) = one comm-kernel stream per group (the two groups' instances can run
    concurrently; two more streams on the hardware queues), "comm" = the instances run on the pipeline's comm stream itself (NCCL's place
    today: no new streams, no cross-group concurrency, no event hops)."""
    v = os.environ.get(f"{_ENV}_STREAM", "own")
    return v if v in ("own", "comm") else "own"


def watchdog_s() -> float:
    return float(max(1, _int("WATCHDOG_S", 30)))


def verify_first() -> int:
    return max(0, _int("VERIFY_FIRST", 3))


def rs_enabled() -> bool:
    return bool(_int("RS", 1))


def ag_enabled() -> bool:
    return bool(_int("AG", 1))


# ------------------------------------------------------------------------------------------------------- the order probe
def rs_order_codes(rs_fn: Callable[[torch.Tensor], torch.Tensor], rank: int, device, m: int, h: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per element of this rank's [m/4, h] rows of an [m, h] bf16 reduce-scatter on rs_fn's communicator: (code, ok). code =
    r0 | r1 << 2 | r2 << 4 | r3 << 6 for ((x_r0 + x_r1) + x_r2) + x_r3 with a bf16 rounding per add (r0 < r1: the first pair
    is unordered); ok = the element's sum is such a chain. Rank r holds 1.0 and the others s: with s = 2^-9 the result is
    1 + 2^-7 exactly where 1.0 is added last, with s = 3 * 2^-10 where it is added third or last. Collective (8 calls)."""
    popc, low, high = (torch.tensor(t, dtype=torch.int32, device=device) for t in (_POPC, _LOW, _HIGH))
    hit = torch.tensor(1.0 + 2.0 ** -7, dtype=torch.bfloat16, device=device)
    c = m // NSRC
    last = torch.zeros(c * h, dtype=torch.int32, device=device)
    late = torch.zeros(c * h, dtype=torch.int32, device=device)
    for r in range(NSRC):
        for s, mask in ((2.0 ** -9, last), (3 * 2.0 ** -10, late)):
            x = torch.full((m, h), 1.0 if rank == r else s, dtype=torch.bfloat16, device=device)
            out = rs_fn(x)
            if tuple(out.shape) != (c, h):
                raise RuntimeError(f"device comm probe: the reduce-scatter returned {tuple(out.shape)}, expected {(c, h)}")
            mask |= (out.reshape(-1) == hit).int() << r
    first = 15 ^ late
    ok = (popc[last] == 1) & (popc[late] == 2) & ((late & last) == last)
    code = low[first] | (high[first] << 2) | (low[late ^ last] << 4) | (low[last] << 6)
    return code, ok


def order_table(code: torch.Tensor, ok: torch.Tensor) -> torch.Tensor | None:
    """The kernel's table (one byte per 4 consecutive elements, padded to 64 B) or None when some element is not a chain or
    the 4 elements of a group differ in order."""
    if not bool(ok.all()):
        return None
    code4 = code.view(-1, 4)
    if not bool((code4 == code4[:, :1]).all()):
        return None
    t = code4[:, 0].to(torch.uint8)
    pad = (-t.numel()) % 64
    if pad:
        t = torch.cat([t, torch.zeros(pad, dtype=torch.uint8, device=t.device)])
    return t.contiguous()


def describe_codes(code: torch.Tensor) -> str:
    vals, counts = torch.unique(code, return_counts=True)
    parts = []
    for v, k in zip(vals.tolist(), counts.tolist()):
        r = [(v >> (2 * p)) & 3 for p in range(4)]
        parts.append(f"(({r[0]}+{r[1]})+{r[2]})+{r[3]}:{k}")
    return " ".join(parts)


# ------------------------------------------------------------------------------------------------------- state
class _State:
    def __init__(self) -> None:
        self.ext: Any = None
        self.tp: Any = None
        self.rank = -1
        self.world = 0
        self.device: torch.device | None = None
        self.streams: list[torch.cuda.Stream] = []
        self.pseq = [0, 0]  # the next phase sequence per group
        self.cur_pseq = [-1, -1]  # the open device phase per group (-1: the phase's RS went to NCCL)
        self.use: dict[tuple[int, int, int], int] = {}  # (g, par, slot) -> launches so far (the flags' value)
        self.cum_push: dict[tuple[int, int, int], int] = {}  # (g, par, slot) -> cumulative pushers (the arrival / team targets)
        self.cum_red: dict[tuple[int, int, int], int] = {}  # (g, par, slot) -> cumulative pieces x reducers (the reduce-done target)
        self.arena_cursor = [0, 0]
        self.ag_index = [0, 0]
        self.arena_bytes = 0
        self.slot_bytes = 0
        self.tables: dict[tuple[int, int, int], torch.Tensor | None] = {}  # (M, K, comm index) -> table | None
        self.table_info: dict[str, Any] = {}
        self.verified: dict[tuple[str, int], int] = {}  # (kind, M) -> collectives compared so far (mode 1)
        self.trusted: dict[tuple[str, int], bool] = {}  # (kind, M) -> the device result is consumed (False: NCCL only, said once)
        self.v_n = torch.zeros((), dtype=torch.int64)  # replaced by device tensors at init
        self.v_bad = torch.zeros(2, dtype=torch.int64)  # [rs, ag] mismatching collectives
        self.v_pending: list[tuple[str, int, torch.Tensor]] = []  # (kind, M, bad flag tensor) of the first-N comparisons
        self.lock = threading.Lock()


_STATE: _State | None = None
_DISABLED: str | None = None
_STATS: dict[str, Any] = {
    "rs_device": 0, "ag_device": 0, "rs_nccl_fallback": 0, "ag_nccl_fallback": 0, "fallback_reasons": {},
    "verify_rs": 0, "verify_ag": 0, "verify_rs_mismatch": 0, "verify_ag_mismatch": 0, "steps": 0, "probes": {},
    "disabled": None, "watchdog": None,
}


def _fallback(kind: str, why: str) -> None:
    _STATS[f"{kind}_nccl_fallback"] += 1
    _STATS["fallback_reasons"][why] = _STATS["fallback_reasons"].get(why, 0) + 1


def _pipe():
    from vllm.v1.worker import moet_prefill_pipe

    return moet_prefill_pipe


def _capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001 - no CUDA (unit tests)
        return False


class _FakeGroup:
    def __init__(self, idx: int) -> None:
        self.idx = idx


_FORCE_GROUP: int | None = None  # unit tests outside the pipeline: act as this group inside an open phase
_ALLOW_CPU = False  # unit tests on a CPU-only host: skip the .is_cuda checks (a fake extension stands in for the kernel)


def _group_in_phase():
    """The calling pipelined group with an open comm phase, else None."""
    if _FORCE_GROUP is not None:
        return _FakeGroup(_FORCE_GROUP)
    pipe = _pipe()
    g = pipe.current_group()
    if g is None or g.step is None or not g.step.pipelined or not g.phase_open or g.phase_kind != "comm":
        return None
    return g


def _build():
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    return load(
        name="moet_device_comm_sm120",
        sources=[os.path.join(here, "moet_device_comm_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_120a,code=sm_120a", "-maxrregcount=32"],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def _init() -> _State | None:
    """Collective over the TP group (every rank reaches the first device collective of the same phase): build the extension,
    allocate the region, exchange the IPC handles, open the peers. None = unavailable (said once; NCCL stays)."""
    global _STATE, _DISABLED
    if _STATE is not None or _DISABLED is not None:
        return _STATE
    st = _State()
    with st.lock:
        if _STATE is not None or _DISABLED is not None:
            return _STATE
        try:
            import torch.distributed as dist
            from vllm.distributed.parallel_state import get_tp_group

            tp = get_tp_group()
            if tp.world_size != NSRC:
                raise RuntimeError(f"tp_size={tp.world_size} (needs {NSRC})")
            t0 = time.perf_counter()
            ext = _build()
            dev = torch.device("cuda", torch.cuda.current_device())
            h = 5120  # the slot is sized by rows x the widest RS row (bf16 hidden); wider rows fall back
            slot_bytes = slot_rows() * h * 2
            arena_bytes = arena_mb() * 1024 * 1024
            total, _, _, slot_bytes, arena_bytes = ext.init_region(tp.rank_in_group, slot_bytes, arena_bytes, pieces())
            handles: list = [None] * tp.world_size
            dist.all_gather_object(handles, ext.ipc_handle(), group=tp.cpu_group)
            for r in range(tp.world_size):
                ext.open_peer(r, handles[r])
            st.ext, st.tp, st.rank, st.world, st.device = ext, tp, tp.rank_in_group, tp.world_size, dev
            st.slot_bytes, st.arena_bytes = slot_bytes, arena_bytes
            st.streams = [torch.cuda.Stream(device=dev, priority=stream_priority()) for _ in range(2)]
            if split_mode():
                ext.set_push_warps(push_warps())
            st.v_n = torch.zeros((), dtype=torch.int64, device=dev)
            st.v_bad = torch.zeros(2, dtype=torch.int64, device=dev)
            regs, smem_static, smem_dyn, local, per_sm, red_regs = ext.func_info()
            _STATS["kernel"] = dict(regs=regs, smem=smem_static + smem_dyn, local=local, ctas_per_sm=per_sm, reduce_regs=red_regs,
                                    split=split_mode(), push_warps=push_warps() if split_mode() else 1, reduce_grid=reduce_grid(),
                                    n_push=n_push(), n_red=0 if split_mode() else n_red(), pieces=pieces(), unroll=unroll(), link_token=link_token(),
                                    stream=stream_mode(), region_mb=total / 1e6, build_s=round(time.perf_counter() - t0, 1))
            _STATE = st
            logger.info(
                "vllm-moet sm_120 device comm (MK-4 hybrid, %s): rank %d of %d, region %.0f MB (%d receive slots x %.1f MB, 4 gather "
                "arenas x %d MB), kernel %d regs / %d B smem / %d B local / %d CTAs per SM at 32 threads, %s, %d pieces, "
                "unroll %d, link token %d, streams %s, watchdog %.0f s, built in %.1f s (VLLM_MOET_PREFILL_PIPE_DEVICE_COMM=0 turns it off)",
                mode(), st.rank, st.world, total / 1e6, 2 * 2 * NSRC, slot_bytes / 1e6, arena_bytes >> 20, regs, smem_static + smem_dyn, local,
                per_sm, (f"split: {n_push()} pushers x {push_warps()} warps resident, 1-CTA waits + a transient {reduce_grid()}-CTA reduce ({red_regs} regs)"
                         if split_mode() else f"{n_push()} pushers + {n_red()} reducers resident"), pieces(), unroll(), link_token(), stream_mode(),
                watchdog_s(), time.perf_counter() - t0)
            if regs > 32 or local > 0:
                logger.warning("vllm-moet sm_120 device comm: the kernel takes %d registers / %d B local memory - the 1 024-register "
                               "co-residency budget (MK-4 item 2) is not met", regs, local)
            return st
        except Exception as exc:  # noqa: BLE001
            _DISABLED = repr(exc)
            _STATS["disabled"] = _DISABLED
            logger.warning("vllm-moet sm_120 device comm unavailable, NCCL's collectives stay: %r", exc)
            return None


def _agree(st: _State, ok: bool) -> bool:
    import torch.distributed as dist

    every: list = [None] * st.world
    dist.all_gather_object(every, bool(ok), group=st.tp.cpu_group)
    return all(every)


def _table(st: _State, m: int, h: int, gidx: int, rs_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor | None:
    """NCCL's order table for (M, H) on this group's communicator: cached, or probed now (collective: every rank reaches the
    first reduce-scatter of this M in this group together). None = NCCL stays for this M (the same verdict everywhere)."""
    key = (m, h, gidx % 2)
    if key in st.tables:
        return st.tables[key]
    t0 = time.perf_counter()
    code, ok = rs_order_codes(rs_fn, st.rank, st.device, m, h)
    table = order_table(code, ok)
    good = _agree(st, table is not None)
    torch.cuda.synchronize()
    info = dict(M=m, probe_ms=round((time.perf_counter() - t0) * 1e3, 1), chain=float(ok.float().mean()),
                codes=describe_codes(code), good=good)
    _STATS["probes"][f"M{m}_c{gidx % 2}"] = info
    if not good:
        st.tables[key] = None
        logger.warning("vllm-moet sm_120 device comm: M=%d stays on NCCL's reduce-scatter - the summation order is not a chain "
                       "uniform per 4 elements on every rank (this rank: chain on %.4f of the elements; codes %s)", m, info["chain"], info["codes"])
        return None
    st.tables[key] = table
    logger.info("vllm-moet sm_120 device comm: NCCL's reduce-scatter order probed for M=%d on communicator %d in %.1f ms: %s",
                m, gidx % 2, info["probe_ms"], info["codes"])
    return table


def _compare(st: _State, kind: str, m: int, dev: torch.Tensor, ref: torch.Tensor) -> None:
    bad = (dev.reshape(-1).view(torch.uint8) != ref.reshape(-1).view(torch.uint8)).any().to(torch.int64)
    st.v_n += 1
    st.v_bad[0 if kind == "rs" else 1] += bad
    _STATS[f"verify_{kind}"] += 1
    if mode() == "1":
        st.v_pending.append((kind, m, bad))


def _first_n_check(st: _State, kind: str, m: int) -> None:
    """Mode 1: after the first N compared collectives of (kind, M), every rank's verdict decides whether the device result is consumed."""
    key = (kind, m)
    n = st.verified.get(key, 0)
    if n < verify_first():
        return
    mine = [b for k, mm, b in st.v_pending if (k, mm) == key]
    st.v_pending = [(k, mm, b) for k, mm, b in st.v_pending if (k, mm) != key]
    bad = int(torch.stack(mine).sum().item()) if mine else 0
    good = _agree(st, bad == 0)
    st.trusted[key] = good
    if good:
        logger.info("vllm-moet sm_120 device comm: the device %s at M=%d is bit-identical with NCCL's on the first %d collectives on "
                    "every rank - the device result is consumed from now on", kind.upper(), m, n)
    else:
        logger.error("vllm-moet sm_120 device comm: the device %s at M=%d DIFFERS from NCCL's on %d of the first %d collectives on "
                     "this rank (or on another rank) - NCCL alone serves this M from now on", kind.upper(), m, bad, n)


# ------------------------------------------------------------------------------------------------------- the collectives
def reduce_scatter(x: torch.Tensor, fallback: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor | None:
    """The device reduce-scatter of the [M, H] bf16 partial inside an open comm phase of a pipelined group: this rank's [M/4, H]
    rows in NCCL's bits (the order table), or None (the caller runs `fallback` = NCCL). In verify mode / the first-N check the
    NCCL result is computed here too and returned."""
    if mode() == "0" or not rs_enabled():
        return None
    g = _group_in_phase()
    if g is None:
        _fallback("rs", "not in a pipelined comm phase")
        return None
    gidx = g.idx & 1
    if _capturing():
        _fallback("rs", "stream capture")
        return None
    st = _init()
    if st is None:
        _fallback("rs", "unavailable")
        return None
    st.cur_pseq[gidx] = -1
    if x.dim() != 2 or x.dtype != torch.bfloat16 or not (x.is_cuda or _ALLOW_CPU) or x.shape[0] % NSRC or x.shape[1] % 8:
        _fallback("rs", f"shape {tuple(x.shape)} {x.dtype}")
        return None
    m, h = x.shape
    c = m // NSRC
    if c * h * 2 > st.slot_bytes:
        _fallback("rs", f"shard {c}x{h} exceeds the slot")
        return None
    if st.trusted.get(("rs", m)) is False:
        _fallback("rs", f"device RS at M={m} differed from NCCL")
        return None
    table = _table(st, m, h, gidx, fallback)
    if table is None:
        _fallback("rs", f"order at M={m}")
        return None
    x = x.contiguous()
    out = torch.empty((c, h), dtype=x.dtype, device=x.device)
    comm = torch.cuda.current_stream()
    kst = comm if stream_mode() == "comm" else st.streams[gidx]
    pseq = st.pseq[gidx]
    use_key = (gidx, pseq & 1, 0)
    split = split_mode()
    npush, nred, npieces = n_push(), (0 if split else n_red()), pieces_for(c * h * 2)
    st.use[use_key] = st.use.get(use_key, 0) + 1
    st.cum_push[use_key] = st.cum_push.get(use_key, 0) + npush
    st.cum_red[use_key] = st.cum_red.get(use_key, 0) + (reduce_grid() if split else npieces * nred)
    if kst is not comm:
        ev_in = torch.cuda.Event()
        ev_in.record(comm)
        kst.wait_event(ev_in)
    st.ext.rs_launch(kst.cuda_stream, x, out, table, gidx, pseq, st.use[use_key], st.cum_push[use_key], st.cum_red[use_key], npieces,
                     npush, nred, unroll(), link_token(), watchdog_s())
    if split:  # the bounded wait for the peers' pieces + the transient reduce, on the comm stream (concurrent with my pushes on kst)
        st.ext.rs_wait_reduce(comm.cuda_stream, x, out, table, gidx, pseq, st.use[use_key], st.cum_red[use_key], npieces, reduce_grid(), watchdog_s())
    if kst is not comm:
        ev_out = torch.cuda.Event()
        ev_out.record(kst)
        comm.wait_event(ev_out)  # x stays alive until my pushes are done
    st.pseq[gidx] = pseq + 1
    st.cur_pseq[gidx] = pseq
    st.arena_cursor[gidx] = 0
    st.ag_index[gidx] = 0
    _STATS["rs_device"] += 1
    key = ("rs", m)
    if mode() == "verify" or (mode() == "1" and not st.trusted.get(key)):
        ref = fallback(x)
        _compare(st, "rs", m, out, ref)
        if mode() == "1":
            st.verified[key] = st.verified.get(key, 0) + 1
            _first_n_check(st, "rs", m)
        return ref
    return out


def all_gather(t: torch.Tensor, fallback: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """The device all-gather of a [c, K] tensor (any dtype, row bytes a multiple of 16) inside the open device phase: the
    [4c, K] gathered rows (a view of the group's arena, valid until the group's phase after the next), else fallback(t)."""
    if mode() == "0" or not ag_enabled():
        return fallback(t)
    g = _group_in_phase()
    if g is None:
        _fallback("ag", "not in a pipelined comm phase")
        return fallback(t)
    gidx = g.idx & 1
    st = _STATE
    if st is None or st.cur_pseq[gidx] < 0:
        _fallback("ag", "phase not on the device")
        return fallback(t)
    if _capturing():
        _fallback("ag", "stream capture")
        return fallback(t)
    if t.dim() != 2 or not (t.is_cuda or _ALLOW_CPU):
        _fallback("ag", f"shape {tuple(t.shape)}")
        return fallback(t)
    c, k = t.shape
    row_bytes = k * t.element_size()
    if row_bytes % 16 or c <= 0:
        _fallback("ag", f"row bytes {row_bytes}")
        return fallback(t)
    if st.trusted.get(("ag", c)) is False:
        _fallback("ag", f"device AG at c={c} differed from NCCL")
        return fallback(t)
    nbytes = NSRC * c * row_bytes
    off = st.arena_cursor[gidx]
    if off + nbytes > st.arena_bytes or st.ag_index[gidx] >= 4:
        _fallback("ag", "arena full" if off + nbytes > st.arena_bytes else "more than 4 gathers in a phase")
        return fallback(t)
    t = t.contiguous()
    pseq = st.cur_pseq[gidx]
    slot = 1 + st.ag_index[gidx]
    st.ag_index[gidx] += 1
    st.arena_cursor[gidx] = (off + nbytes + 255) & ~255
    use_key = (gidx, pseq & 1, slot)
    split = split_mode()
    npush, nred, npieces = n_push(), (0 if split else n_red()), ag_pieces()  # a gather's receiver only waits for the whole: one piece = 3 fences per pusher
    st.use[use_key] = st.use.get(use_key, 0) + 1
    st.cum_push[use_key] = st.cum_push.get(use_key, 0) + npush
    st.cum_red[use_key] = st.cum_red.get(use_key, 0) + max(1, npieces * nred)
    comm = torch.cuda.current_stream()
    kst = comm if stream_mode() == "comm" else st.streams[gidx]
    view = st.ext.arena_view(gidx, pseq & 1, off, NSRC * c, k, t)
    if split:
        view[st.rank * c:(st.rank + 1) * c].copy_(t)  # my own shard into my arena (the pushers only serve the peers)
    if kst is not comm:
        ev_in = torch.cuda.Event()
        ev_in.record(comm)
        kst.wait_event(ev_in)
    st.ext.ag_launch(kst.cuda_stream, t, off, gidx, pseq, slot, st.use[use_key], st.cum_push[use_key], st.cum_red[use_key], npieces,
                     npush, nred, unroll(), link_token(), watchdog_s())
    if split:
        st.ext.ag_wait(comm.cuda_stream, c, row_bytes, gidx, pseq, slot, st.use[use_key], npieces, watchdog_s())
    if kst is not comm:
        ev_out = torch.cuda.Event()
        ev_out.record(kst)
        comm.wait_event(ev_out)
    _STATS["ag_device"] += 1
    key = ("ag", c)
    if mode() == "verify" or (mode() == "1" and not st.trusted.get(key)):
        ref = fallback(t)
        _compare(st, "ag", c, view, ref)
        if mode() == "1":
            st.verified[key] = st.verified.get(key, 0) + 1
            _first_n_check(st, "ag", c)
        return ref
    return view


def step_done() -> None:
    """From the pipeline after a pipelined step: the watchdog record (a trapped kernel = a failed step, loudly) and the verify
    counters (one line per step in verify mode)."""
    st = _STATE
    if st is None:
        return
    _STATS["steps"] += 1
    rec = st.ext.err_check()
    if rec[0]:
        _STATS["watchdog"] = dict(what=rec[1], expected=rec[2], seen=rec[3], cta=rec[4], rank=rec[5], g=rec[6], pseq=rec[7], waited_ms=rec[8] / 1e6)
        raise RuntimeError(f"vllm-moet sm_120 device comm: a device-side wait expired and the kernel trapped: {_STATS['watchdog']}")
    if mode() == "verify" or st.v_pending:  # mode 1: only while first-N comparisons are open (a host sync per step otherwise)
        try:  # the counters were written on the pipeline's comm stream; order the read after it
            comm_stream = getattr(_pipe()._RUNNER, "comm_stream", None)
            if comm_stream is not None:
                torch.cuda.current_stream().wait_stream(comm_stream)
        except Exception:  # noqa: BLE001 - unit tests without the pipeline
            pass
        n = int(st.v_n.item())
        bad_rs, bad_ag = (int(v) for v in st.v_bad.tolist())
        _STATS["verify_rs_mismatch"], _STATS["verify_ag_mismatch"] = bad_rs, bad_ag
        if mode() == "verify":
            logger.info("vllm-moet sm_120 device comm verify: step %d - %d collectives compared so far (RS %d, AG %d), mismatches RS %d AG %d%s",
                        _STATS["steps"], n, _STATS["verify_rs"], _STATS["verify_ag"], bad_rs, bad_ag,
                        "" if bad_rs or bad_ag else " (all bit-identical with NCCL)")


def stats() -> dict:
    d = dict(_STATS)
    d["mode"] = mode()
    return d
