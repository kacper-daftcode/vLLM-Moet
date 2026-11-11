"""JIT loader + host side of b12x_moe_prefill_fast_sm120.cu: the prefill MoE of b12x's `integration.tp_moe.dynamic`
deterministic path (FC1 + SwiGLU + MXFP8 quant, FC2, top-k sum) as pipelined CUDA kernels, bit for bit with spec B
(../b12x_moe_prefill_spec) and with b12x.

    from b12x_moe_prefill_fast_sm120 import moe_prefill_fast
    out, route, a_values, a_scales, inter, inter_sf, row_pair, counts = moe_prefill_fast(hidden, ids, weights, planes)

`planes` as in the spec loader: dict(w13=prepared.w13_rp, w13_sf=prepared.w13_sfb, w2=prepared.w2_rp, w2_sf=prepared.w2_sfb,
alpha=[E], input_global_scale=[E]|[1], down_alpha=[E], global_scale=[E]|[1]). `route_out` (optional, [>= T*topk, K] bf16):
the route rows go there (b12x's workspace.route_output when integrated); `topk_sum=False` skips the sum (b12x's own
TopKSum runs on the route rows instead). The routing (pair counts, padded expert offsets, the stable scatter, the CTA
work lists) is done on the GPU by the front-end kernel: no host synchronization. Tuning knobs (env, default = measured
best): VLLM_MOET_B12X_MOE_PREFILL_FAST_CFG1 / _CFG2 (stages, CTAs per SM: 0 = (3, 2), 1 = (4, 2)), _DYNAMIC (1 = atomic
work counter, 0 = static round-robin), _GRID1 / _GRID2 (CTA caps, 0 = occupancy). Every configuration computes the same bits.
The block-scaled QMMA needs sm_120a; the extension compiles on first import (~60 s) without fast-math / ftz.

Pre-quantized rows (the sequence-parallel MXFP8 all-gather of the MoE input, ../sp_moe_gather): `moe_prefill_fast(None, ids,
weights, planes, ..., a_values=q, a_scales=s)` with q uint8 [T, K] (e4m3, row-major) and s uint8 [T, K/32] (ue8m0, row-major)
skips the front-end's rows quant stage and feeds FC1 the given rows (the same bytes quant_row writes when they come from
FlashInfer's mxfp8 quantizer, see sp_moe_gather/test_prefill_sp_gather_tp4.py). `prequant_rows_supported()` tells whether the
built extension has that entry; without it the caller must pass bf16 rows. VLLM_MOET_B12X_MOE_PREFILL_FAST_NAME overrides the
extension's build name (a kernel variant next to the image's prebuilt one).

v9 (K-4): the producer-warp kernels (one warp issues the TMA / cp.async copies, 8 consumer warps never stop for a barrier or
address work; bit-identical to v8 by the harness) are selected per kernel by VLLM_MOET_B12X_MOE_PREFILL_FAST_WS1 (fc1) /
_WS2 (fc2): 0 = v8 (default), a mode = the ring (stages, CTAs per SM) - see the .cu; measured best fc1 2 (6 stages, 1 CTA/SM,
-4 % at 4096 rows), fc2 3 (4 stages, 2 CTAs/SM, -16 %); fc1 below _WS1_MIN_ROWS (256) rows stays on v8 (the 128-row replay
layers are at the DRAM floor).

v10 (K-5): fc1 as three warpgroups (`fc1_kernel_wg`: 8 consumer warps + 4 producer warps, one LDS.128 per A fragment from a layout
8-B cp.asyncs fill; at 2 CTAs/SM `setmaxnreg` 32 / 104) = WS1 modes 8 (3 stages, 2 CTAs/SM), 9 (4, 1), 10 (6, 1), 11 (7, 1); measured
best mode 9: fc1 -10.6 % at 4096 rows in the harness (913 vs 1 021 us), bit-identical; in situ (pipelined r26 step, WS1=9 WS2=3)
the 4096-row fc1 launches are -10.5 % (887 -> 794 us) but the 163K / 19K walls move <= 0.3 %.

MK-2 (megakernel stage 1, milestone 1): `VLLM_MOET_B12X_MOE_PREFILL_FAST_WS` >= 20 runs fc1 + SwiGLU/quant + fc2 as ONE persistent launch
(`moe_persistent_kernel`: v10's CTA, 1 CTA/SM x 12 warps, one work list with the fc2 items gated by per-expert fc1_done counters, every wait
watchdog-bounded - an expired wait records to pinned host memory and traps; the separate topk_sum_kernel follows). 20-25 = fc2 items of one
128-column tile with 4 / 6 / 7 / 3 / 5 / 2 ring stages, 30-35 = fc2 items of two tiles (the fc1-shaped stage; the faster form). Knobs:
_WS_LEAD (the work list's lead in experts, default 64; >= the expert count = every fc1 item before the first fc2 item), _WS_MIN_ROWS (default 0),
_WS_WATCHDOG_MS (2000), _WS_STATS=1 (per-item %globaltimer statistics, moe_prefill_ws_stats()), _WS_ROUTE_CS (streaming route-row stores, 1).
Bit-identical to v8/v9/v10 on W's harness (6/6 at 4096 and 128 rows, 3 runs); measured +2.6 % vs WS1=9 + WS2=3 at call1 4096 rows
(MK/REPORT.md 2) - off by default. The persistent launch needs contiguous rows and at least _WS_MIN_ROWS of them; a call without them runs
the separate kernels. In a process with _WS >= 20 no call runs the sum items below (`sum_items_mode()` is 0, the route rows are written
by every call and the separate topk_sum_kernel follows), whatever _SUM_ITEMS says.

K-12 (the route rows' DRAM round trip): `VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_ITEMS=1` (with WS2 >= 1) puts the FC2 work list in column-tile-major
order and runs the top-k sum as work items of `fc2_kernel_ws` (the sum items of a tile follow the FC2 items of tile + _SUM_LAG (2) and wait for
the tile's FC2 items with watchdog-bounded acquire polls; _SUM_TOKENS (256) tokens per item) so that a tile's route rows (6.3 MB at 4096 rows)
are summed while still in L2; `_SUM_DISCARD=1` drops the summed route lines from L2 without a write-back (the route buffer is scratch then:
`route_rows_consumed()`); `_W2_EVICT_FIRST=1` streams W2 with an L2 evict_first hint. The sum is topk_sum_kernel's chain in slot order (level
B); the separate topk_sum_kernel is skipped.

The m-tile loops: `VLLM_MOET_B12X_MOE_PREFILL_FAST_MT_LOOPS=1` (default 0) launches the served FC1 / FC2 modes (WS1 = 9, WS2 = 3) with the
consumers' stage loop specialised on the item's m-tiles. The served consumers run the QMMAs of an m-tile under a warp-uniform predicate,
and a QMMA under a false predicate keeps its stall and its place in the tensor pipe's issue throttle (eight warps per SM issue 614 M
QMMA slots per second when every slot executes and 658 M when none does), so an item pays for four m-tiles whatever it has: at 4096 rows
of a real prefill a row chunk has 2.9 of 4 m-tiles on average, and on FC1's 64-column tile every warp serves two of the four. With the
switch an item branches once (warp-uniform) to a loop that holds only its own m-tiles' loads and QMMAs - the same QMMAs on the same
operands in the same order per accumulator, bit-identical on W's harness. One RTX PRO 6000 Workstation card at 2.61 GHz, 4096 rows,
319 experts with the weights streamed: FC1 736 -> 666 us, FC2 464 -> 449 us, the call's kernels 1 241 -> 1 155 us (-6.9 %); six experts
with the weights in L2: 817 -> 780 us. FC2's specialised kernel is built for two CTAs per SM and takes 92 registers per thread (the
occupancy guard watches it as `fc2_kernel_ws_mt<4, 2>`). `mt_loops_supported()` tells whether the built extension has the kernels.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
N_INTERMEDIATE = 576


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_B12X_MOE_PREFILL_FAST_ARCH", "12.0a")
    return load(
        name=os.environ.get("VLLM_MOET_B12X_MOE_PREFILL_FAST_NAME", "vllm_moet_b12x_moe_prefill_fast_sm120"),
        sources=[str(_HERE / "b12x_moe_prefill_fast_sm120.cu")],
        # MK-4: extra nvcc flags, space- or comma-separated (a docker -e value cannot carry spaces), e.g. -DMOET_FC1_MAXNREG=120
        extra_cuda_cflags=["-O3", "-std=c++20"] + os.environ.get("VLLM_MOET_B12X_MOE_PREFILL_FAST_NVCC_EXTRA", "").replace(",", " ").split(),
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def prequant_rows_supported() -> bool:
    """Whether the built extension takes pre-quantized rows (a_values / a_scales) in place of the bf16 rows."""
    return hasattr(_ext(), "prequant_rows_supported")


def strided_rows_supported() -> bool:
    """Whether the built extension takes a_values / a_scales as row-strided views (unit column stride, row strides and bases a
    multiple of 4 bytes) - e.g. buf[:, :K] and buf[:, K:K + K // 32] of one gathered [T, K + K // 32 + ...] buffer (session
    U's one-all-gather follow-up); older builds get contiguous copies."""
    return hasattr(_ext(), "strided_rows_supported")


def sum_items_mode() -> int:
    """K-12: the fc2 sum-items mode this process runs with (VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_ITEMS with WS2 >= 1; 0 = off / older build)."""
    e = _ext()
    return int(e.sum_items_mode()) if hasattr(e, "sum_items_mode") else 0


def route_rows_consumed() -> bool:
    """K-12: whether the route rows are summed and dropped from L2 inside fc2 (SUM_ITEMS=1 + SUM_DISCARD=1): the returned `route` tensor is
    scratch then (unspecified content); the top-k sum `out` is the interface."""
    e = _ext()
    return bool(e.route_rows_consumed()) if hasattr(e, "route_rows_consumed") else False


# ----------------------------------------------------------------------------------------------------------------------------------
# The runtime A/B switch and the activation counters of this module (both off by default):
#   VLLM_MOET_RUNTIME_AB=1          the runtime A/B switch for this module's level-B paths: a flat JSON object switch -> value in the
#                                   control file VLLM_MOET_RUNTIME_AB_FILE (default /dev/shm/vllm_moet_ab.json; the serving container's
#                                   /dev/shm is shared by its API server, engine and workers, so `docker exec <c> sh -c 'echo ... > ...'`
#                                   reaches every rank). Keys of this module: B12X_MOE_PREFILL_FAST_SUM_ITEMS (0 / 1 - the sum items with
#                                   their _DISCARD / _ROUTE_HINT companions as one unit), B12X_MOE_PREFILL_FAST_WS1 (0..11; 0 = v8,
#                                   9 = v10) and B12X_MOE_PREFILL_FAST_MT_LOOPS (0 / 1 - the m-tile loops of the served FC1 / FC2
#                                   modes; in the log lines this key carries its full name, the two older keys their short tokens
#                                   SUM_ITEMS / WS1). Every rank stats the file at most once per 100 ms (at the entry of moe_prefill_fast, i.e.
#                                   at a MoE-call boundary - this module has no view of the prefill forward; the states are bit-identical
#                                   alternatives, so a flip between two calls of one step changes no bits), re-reads it only when mtime or
#                                   size changed, applies the values through the extension's set_runtime_switches, logs one line per rank
#                                   per change ("runtime A/B: SUM_ITEMS 1 -> 0"). A missing / unreadable / malformed file, a key of this
#                                   module it does not know, or a value out of range = keep the current state (one WARNING, once per
#                                   cause). Keys outside this module's namespace are other modules' business (ignored silently). Default
#                                   0 = today's behaviour exactly (no stat call, no import of json).
#   VLLM_MOET_B12X_MOE_PREFILL_FAST_STATS_EVERY=N   task B: every N calls one line per rank with the activation counters of the switched
#                                   paths (calls / rows through fc1 v10 vs v8/v9, fc2 with the sum items vs fc2 + topk_sum_kernel,
#                                   fc1 / fc2 with the m-tile loops) -
#                                   default 0 = only on demand: `touch /dev/shm/vllm_moet_stats` (VLLM_MOET_RUNTIME_STATS_FILE) -> one
#                                   line per rank at the next call boundary (checked every 64 calls: one os.stat per 64 calls; the file's
#                                   mtime is remembered, so one touch = one line per rank). Cost when not requested: one integer add per
#                                   counter per call in the extension + the 1/64 stat.
_RT_STATE: dict = {"on": None, "next_poll": 0.0, "sig": None, "warned": set(), "calls": 0, "stats_mtime": None, "stats_every": None}


def _rt_log(msg: str) -> None:
    import sys

    try:
        dev = torch.cuda.current_device()
    except Exception:  # noqa: BLE001
        dev = "?"
    print(f"[vllm-moet fast prefill MoE, device {dev}] {msg}", file=sys.stderr, flush=True)


def _rt_warn_once(key: str, msg: str) -> None:
    if key not in _RT_STATE["warned"]:
        _RT_STATE["warned"].add(key)
        _rt_log("WARNING: " + msg)


def occupancy_facts() -> list:
    """What every WS launch family got at its first launch in this process: (kernel, stages, min blocks, registers per thread, CTAs per SM
    from the occupancy query, CTAs per SM its mode was measured with or 0 when that is not on record, SMs). Empty with an older build."""
    e = _ext()
    return [tuple(f) for f in e.occupancy_facts()] if hasattr(e, "occupancy_facts") else []


_OCC_SEEN = [0]


def _occ_guard() -> None:
    """One WARNING per kernel that launches with fewer CTAs per SM than its mode was measured with: the grid is sms x the occupancy the
    build's register count leaves, so a kernel that gained a register (a source change, another ptxas after an image rebuild) can lose
    half its grid without any error - FC2's v9 kernel sits exactly on that count (96 registers per thread for two CTAs per SM)."""
    facts = occupancy_facts()
    for kernel, s, mb, regs, occ, expect, sms in facts[_OCC_SEEN[0]:]:
        if expect > 0 and occ < expect:
            _rt_log(f"WARNING: {kernel}<{s}, {mb}> launches with {occ} CTA(s) per SM (grid {sms * max(occ, 1)}) where its mode was measured with "
                    f"{expect} (grid {sms * expect}): {regs} registers per thread in this build leave no room for more - the kernel is slower than "
                    "measured (FC2: 10-15 %). Check the build (a source change or another ptxas); "
                    "VLLM_MOET_B12X_MOE_PREFILL_FAST_OCC_STRICT=1 refuses such a launch")
    _OCC_SEEN[0] = len(facts)


def mt_loops_supported() -> bool:
    """Whether the built extension has the m-tile loops (VLLM_MOET_B12X_MOE_PREFILL_FAST_MT_LOOPS and its runtime key)."""
    return hasattr(_ext(), "set_mt_loops")


def runtime_switches() -> dict:
    """The effective switch values of this process (env or runtime override): ws1, sum_items, sum_discard, sum_route_hint and, with a
    build that has them, mt_loops."""
    e = _ext()
    if not hasattr(e, "get_runtime_switches"):
        return {}
    ws1, si, sd, sh = (int(v) for v in e.get_runtime_switches())
    sw = {"WS1": ws1, "SUM_ITEMS": si, "SUM_DISCARD": sd, "SUM_ROUTE_HINT": sh}
    if hasattr(e, "get_mt_loops"):
        sw["MT_LOOPS"] = int(e.get_mt_loops())
    return sw


def apply_runtime_switches(values: dict) -> list[str]:
    """Apply {"B12X_MOE_PREFILL_FAST_SUM_ITEMS": 0|1, "B12X_MOE_PREFILL_FAST_WS1": 0..11, "B12X_MOE_PREFILL_FAST_MT_LOOPS": 0|1} (keys of
    other modules ignored); returns the change descriptions. A key the file does not carry goes back to its env value (-1); an
    out-of-range value keeps the current effective value (one WARNING). The sum-items unit sets SUM_ITEMS, SUM_DISCARD and
    SUM_ROUTE_HINT together."""
    e = _ext()
    if not hasattr(e, "set_runtime_switches"):
        _rt_warn_once("ext", "this build of the extension has no set_runtime_switches entry: runtime A/B inactive")
        return []
    cur = runtime_switches()
    ws1_arg, si_arg, mt_arg = -1, -1, -1
    has_mt = hasattr(e, "set_mt_loops")
    for key, val in values.items():
        if not isinstance(key, str) or not key.startswith("B12X_MOE_PREFILL_FAST_"):
            continue
        try:
            v = int(val)
        except (TypeError, ValueError):
            v = -2
        if key == "B12X_MOE_PREFILL_FAST_SUM_ITEMS":
            if v in (0, 1):
                si_arg = v
            else:
                si_arg = cur["SUM_ITEMS"]
                _rt_warn_once("val:" + key, f"runtime A/B: {key}={val!r} out of range (0 / 1) - kept {cur['SUM_ITEMS']}")
        elif key == "B12X_MOE_PREFILL_FAST_WS1":
            if 0 <= v <= 11:
                ws1_arg = v
            else:
                ws1_arg = cur["WS1"]
                _rt_warn_once("val:" + key, f"runtime A/B: {key}={val!r} out of range (0..11) - kept {cur['WS1']}")
        elif key == "B12X_MOE_PREFILL_FAST_MT_LOOPS" and has_mt:
            if v in (0, 1):
                mt_arg = v
            else:
                mt_arg = cur["MT_LOOPS"]
                _rt_warn_once("val:" + key, f"runtime A/B: {key}={val!r} out of range (0 / 1) - kept {cur['MT_LOOPS']}")
        else:
            _rt_warn_once("key:" + key, f"runtime A/B: unknown key {key} of this module - ignored")
    sd_arg = sh_arg = si_arg  # the unit: discard + evict_last hint follow the sum items (-1 = env for all three)
    e.set_runtime_switches(int(ws1_arg), int(si_arg), int(sd_arg), int(sh_arg))
    if has_mt:
        e.set_mt_loops(int(mt_arg))
    now = runtime_switches()
    changes = []
    if now["WS1"] != cur["WS1"]:
        changes.append(f"WS1 {cur['WS1']} -> {now['WS1']}")
    if has_mt and now["MT_LOOPS"] != cur["MT_LOOPS"]:
        changes.append(f"B12X_MOE_PREFILL_FAST_MT_LOOPS {cur['MT_LOOPS']} -> {now['MT_LOOPS']}")  # a newer key: its full name
    if (now["SUM_ITEMS"], now["SUM_DISCARD"], now["SUM_ROUTE_HINT"]) != (cur["SUM_ITEMS"], cur["SUM_DISCARD"], cur["SUM_ROUTE_HINT"]):
        changes.append(f"SUM_ITEMS {cur['SUM_ITEMS']} -> {now['SUM_ITEMS']} (DISCARD {cur['SUM_DISCARD']} -> {now['SUM_DISCARD']}, "
                       f"ROUTE_HINT {cur['SUM_ROUTE_HINT']} -> {now['SUM_ROUTE_HINT']})")
    for c in changes:
        _rt_log("runtime A/B: " + c)
    return changes


def _runtime_ab_poll() -> None:
    import time

    now = time.monotonic()
    if now < _RT_STATE["next_poll"]:
        return
    _RT_STATE["next_poll"] = now + 0.1
    path = os.environ.get("VLLM_MOET_RUNTIME_AB_FILE", "/dev/shm/vllm_moet_ab.json")
    try:
        st = os.stat(path)
    except OSError:
        _rt_warn_once("missing", f"runtime A/B: control file {path} missing or unreadable - the current state is kept until it appears")
        return
    sig = (st.st_mtime_ns, st.st_size)
    if sig == _RT_STATE["sig"]:
        return
    _RT_STATE["sig"] = sig
    import json

    try:
        with open(path, encoding="utf-8") as f:
            values = json.load(f)
        if not isinstance(values, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        _rt_warn_once("malformed", f"runtime A/B: control file {path} malformed ({exc}) - the current state is kept")
        return
    _RT_STATE["warned"].discard("missing")
    _RT_STATE["warned"].discard("malformed")
    apply_runtime_switches(values)


def activation_counters(clear: bool = False) -> dict:
    """Task B: the extension's activation counters of this process (calls / rows per switched path)."""
    e = _ext()
    if not hasattr(e, "moe_prefill_fast_counters"):
        return {}
    c = [int(v) for v in e.moe_prefill_fast_counters(bool(clear))]
    out = {"fc1_v10": (c[0], c[1]), "fc1_v8_v9": (c[2], c[3]), "fc2_sum_items": (c[4], c[5]), "fc2_topk_sum_kernel": (c[6], c[7])}
    if hasattr(e, "moe_prefill_fast_mt_counters"):
        t = [int(v) for v in e.moe_prefill_fast_mt_counters(bool(clear))]
        out["fc1_mt_loops"] = (t[0], t[1])
        out["fc2_mt_loops"] = (t[2], t[3])
    return out


def _stats_line(reason: str) -> None:
    c = activation_counters()
    sw = runtime_switches()
    mt_cnt = mt_tok = ""
    if "MT_LOOPS" in sw and "fc1_mt_loops" in c:  # a build with the m-tile loops: their counters and their token join the line
        mt_cnt = f", fc1 m-tile loops {c['fc1_mt_loops'][0]} / {c['fc1_mt_loops'][1]}, fc2 m-tile loops {c['fc2_mt_loops'][0]} / {c['fc2_mt_loops'][1]}"
        mt_tok = f" B12X_MOE_PREFILL_FAST_MT_LOOPS={sw['MT_LOOPS']}"
    _rt_log(f"activation counters ({reason}; calls / rows): fc1 v10 {c['fc1_v10'][0]} / {c['fc1_v10'][1]}, fc1 v8-v9 {c['fc1_v8_v9'][0]} / {c['fc1_v8_v9'][1]}, "
            f"fc2 sum-items {c['fc2_sum_items'][0]} / {c['fc2_sum_items'][1]}, fc2 + topk_sum_kernel {c['fc2_topk_sum_kernel'][0]} / {c['fc2_topk_sum_kernel'][1]}{mt_cnt}; "
            f"switches now WS1={sw.get('WS1')} SUM_ITEMS={sw.get('SUM_ITEMS')} DISCARD={sw.get('SUM_DISCARD')} ROUTE_HINT={sw.get('SUM_ROUTE_HINT')}{mt_tok}")


def _runtime_hooks() -> None:
    st = _RT_STATE
    if st["on"] is None:
        st["on"] = os.environ.get("VLLM_MOET_RUNTIME_AB", "0") == "1"
        st["stats_every"] = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_STATS_EVERY", 0)
    if st["on"]:
        _runtime_ab_poll()
    st["calls"] += 1
    n = st["calls"]
    if n <= 4 or n % 64 == 0:  # a kernel's first launch is one of the first calls, or follows a switch: seen 64 calls later at the latest
        _occ_guard()
    if st["stats_every"] > 0 and n % st["stats_every"] == 0:
        _stats_line(f"every {st['stats_every']} calls, call {n}")
    if n % 64 == 0:
        try:
            m = os.stat(os.environ.get("VLLM_MOET_RUNTIME_STATS_FILE", "/dev/shm/vllm_moet_stats")).st_mtime_ns
        except OSError:
            m = None
        if m is not None and m != st["stats_mtime"]:
            st["stats_mtime"] = m
            _stats_line(f"on demand, call {n}")


def _rows_arg(x: torch.Tensor) -> torch.Tensor:
    if x.is_contiguous():
        return x
    if strided_rows_supported() and x.dim() == 2 and x.stride(1) == 1 and x.stride(0) % 4 == 0 and x.data_ptr() % 4 == 0:
        return x
    return x.contiguous()


def moe_prefill_fast(hidden: torch.Tensor | None, ids: torch.Tensor, weights: torch.Tensor, planes: dict, limit: float = 10.0,
                     n: int = N_INTERMEDIATE, route_out: torch.Tensor | None = None, out: torch.Tensor | None = None,
                     topk_sum: bool = True, cfg1: int | None = None, cfg2: int | None = None,
                     dynamic: int | None = None, grid1: int | None = None, grid2: int | None = None,
                     a_values: torch.Tensor | None = None, a_scales: torch.Tensor | None = None):
    cfg1 = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_CFG1", 0) if cfg1 is None else cfg1
    cfg2 = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_CFG2", 0) if cfg2 is None else cfg2
    dynamic = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_DYNAMIC", 1) if dynamic is None else dynamic
    grid1 = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_GRID1", 0) if grid1 is None else grid1
    grid2 = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_GRID2", 0) if grid2 is None else grid2
    _runtime_hooks()  # K-13: the runtime A/B poll (VLLM_MOET_RUNTIME_AB=1) and the activation-counter lines (task B); no-ops by default
    ids = ids.to(torch.int32).contiguous()
    args = (ids, weights.to(torch.float32).contiguous(), planes["w13"],
            planes["w13_sf"], planes["w2"], planes["w2_sf"],
            planes["alpha"].to(torch.float32).contiguous(),
            planes["input_global_scale"].to(torch.float32).contiguous(),
            planes["down_alpha"].to(torch.float32).contiguous(),
            planes["global_scale"].to(torch.float32).contiguous(), float(limit), int(n),
            route_out, out,
            bool(topk_sum), int(cfg1), int(cfg2), int(dynamic), int(grid1), int(grid2))
    if a_values is not None:
        if not prequant_rows_supported():
            raise RuntimeError("this build of the fast prefill MoE extension takes bf16 rows only (no a_values / a_scales entry)")
        assert a_scales is not None, "a_scales must come with a_values"
        return _ext().moe_prefill_fast(None if hidden is None else hidden.contiguous(), *args,
                                       _rows_arg(a_values), _rows_arg(a_scales))
    assert hidden is not None, "hidden (bf16 rows) or a_values + a_scales (pre-quantized rows)"
    if prequant_rows_supported():
        return _ext().moe_prefill_fast(hidden.contiguous(), *args, None, None)
    return _ext().moe_prefill_fast(hidden.contiguous(), *args)


def supported(*, n: int, k: int, num_topk: int, experts: int, ids_dtype) -> bool:
    """The shapes the kernels handle: intermediate 576, hidden a multiple of 128 (>= 640), <= 512 experts, int32 ids."""
    return int(n) == N_INTERMEDIATE and int(k) % 128 == 0 and int(k) >= 640 and 1 <= int(experts) <= 512 and \
        1 <= int(num_topk) <= 64 and ids_dtype == torch.int32
