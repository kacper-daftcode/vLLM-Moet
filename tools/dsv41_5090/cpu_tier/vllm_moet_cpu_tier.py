"""Runtime of the CPU expert tier inside vLLM (phase 2.3/2.4): what the patched mxfp4.py / b12x.py /
w4a8_compact_micro.py / model_runner.py call. Installed next to them by patch_vllm_moe_cpu_tier.py together with
cpu_tier.py, placement_manager.py, cpu_tier_cuda.cu, cpu_tier_abi.h and libcpu_tier.so. Everything is a no-op unless
VLLM_MOET_MOE_CPU_TIER=1.

Two modes:
  dummy (no VLLM_MOET_CPU_TIER_STORE): vLLM loads every expert of a target layer onto the GPU (the mini model with
      --load-format dummy); the records are dumped from those tensors into a pinned arena and the resident subset
      is what b12x prepares.
  store (VLLM_MOET_CPU_TIER_STORE=<dir>): the real checkpoint. The layer's expert parameters are created with R rows
      (resident slots + spare slots) and vLLM's weight loader maps a resident expert to its slot and skips the rest
      (the same path it uses for experts that are not local to an EP rank); the records come from build_store.py's
      files, pinned by cpu_tier.ExpertStore (the non-resident ones by default = the 208-GB static-store budget, or all).

Environment:
  VLLM_MOET_MOE_CPU_TIER=1              on
  VLLM_MOET_CPU_TIER_RESIDENT=<int>     resident experts per layer (default: all = a no-op hybrid); or
  VLLM_MOET_CPU_TIER_PLACEMENT=<json>   routing/placement.py's file (per-layer resident lists)
  VLLM_MOET_CPU_TIER_SPARE=<int>        empty slots per layer on top of the residents, filled by promotions (default 0)
  VLLM_MOET_CPU_TIER_TABLES=<dir>       b12x_act_m_limit10.f32, b12x_ue8m0_inv.f32, b12x_layout_*.npy
  VLLM_MOET_CPU_TIER_CPUS=<a-b>         the rank's CPUs (default: 8*rank+1 .. 8*rank+7 = one CCD minus its first core,
                                        left to vLLM's own threads); dispatcher = first pool CPU + 32 (its SMT sibling)
  VLLM_MOET_CPU_TIER_POOL_THREADS=N     how much of the rank's CCD the kernel pool takes (default 6 = cores 8r+2..8r+7;
                                        12 = those cores AND their SMT siblings - the bench: -17 % per expert at m = 1,
                                        -24 % at m = 2; 7 / 14 = 7 cores (x 2) with the dispatcher on 8r+32; 8 = the
                                        whole CCD, main + dispatcher on siblings) - see _cpu_layout (10-28)
  VLLM_MOET_CPU_TIER_STORE=<dir>        build_store.py's files for the real checkpoint (see above)
  VLLM_MOET_CPU_TIER_STORE_MODE         nonresident (default) | all: which records to pin
  VLLM_MOET_CPU_TIER_STORE_EARLY=1      pin the records at the LAST target layer's create_weights (the model constructed,
                                        the Engram tables allocated, the weights not yet loaded; needs a PLACEMENT file)
                                        instead of after loading - no reclaim storm against the loader's page cache (10-26;
                                        not at the first create_weights: the Engram tables' huge-page prefault then stalls
                                        in compaction behind 260 GB of pinned pages)
  VLLM_MOET_CPU_TIER_DUMMY_WEIGHTS=1    (dummy mode) overwrite the dummy expert weights with seeded sane random MXFP4
                                        bytes (scales 118..126), so that a stock run and a hybrid run see the same experts
  VLLM_MOET_CPU_TIER_TIMEOUT_MS         bounded wait (default 500)
  VLLM_MOET_CPU_TIER_MAX_PROMOTIONS     per layer per step (default 4; 0 = static placement only)
  VLLM_MOET_CPU_TIER_LOG=1              KPI lines (every VLLM_MOET_CPU_TIER_LOG_EVERY steps, default 100)
  VLLM_MOET_CPU_TIER_SCRATCH=<int>      slots of the prefill scratch pool (default 0 = off: the missing experts of
                                        non-compact calls are not computed - decode-only serving; 128 = 1.2 GB per rank)
  VLLM_MOET_CPU_TIER_PREFILL_OVERLAP=1  a second scratch set + a side stream: a call's round 0 is staged during the stock
                                        run, round r + 1 during round r's extra run (bit-identical to the serial path;
                                        + SCRATCH x 4.7 MB x 2 of VRAM); DEBUG_OVERLAP=1 re-runs every call serially
                                        and compares bit for bit (mini-real 10-29: 0 mismatches under the deterministic plan)
  VLLM_MOET_CPU_TIER_DEBUG_COMPACT=1    eager runs only (--enforce-eager; needs SCRATCH >= the routed experts of a
                                        call, <= 64): every compact call is recomputed with all its routed experts in
                                        the scratch pool in one b12x run = the all-resident reference of this boot;
                                        the hybrid's route rows (GPU pairs / CPU pairs) and the layer output are
                                        compared bit for bit (R5 within one boot, where TP4 boots are not reproducible)
"""
from __future__ import annotations

import atexit
import os
import re
import sys
import threading
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ENABLED = os.environ.get("VLLM_MOET_MOE_CPU_TIER", "0") == "1"
DEBUG = os.environ.get("VLLM_MOET_CPU_TIER_DEBUG", "0") == "1"
DEBUG_PREFILL = os.environ.get("VLLM_MOET_CPU_TIER_DEBUG_PREFILL", "0") == "1"
DEBUG_COMPACT = os.environ.get("VLLM_MOET_CPU_TIER_DEBUG_COMPACT", "0") == "1"
DUMP_DIR = os.environ.get("VLLM_MOET_CPU_TIER_DUMP_DIR", "")   # eager runs only: per-call MoE inputs/outputs
DUMP_MAX = int(os.environ.get("VLLM_MOET_CPU_TIER_DUMP_MAX", "400"))
STORE = os.environ.get("VLLM_MOET_CPU_TIER_STORE", "")
STORE_MODE = os.environ.get("VLLM_MOET_CPU_TIER_STORE_MODE", "nonresident")
STORE_EARLY = os.environ.get("VLLM_MOET_CPU_TIER_STORE_EARLY", "0") == "1"   # pin the records before the weights load
SPARE = int(os.environ.get("VLLM_MOET_CPU_TIER_SPARE", "0"))
LOG = os.environ.get("VLLM_MOET_CPU_TIER_LOG", "0") == "1"
LOG_EVERY = int(os.environ.get("VLLM_MOET_CPU_TIER_LOG_EVERY", "100"))
RECORD_BYTES = 4700160
HID, N_R, TOPK_MAX = 5120, 576, 8
_PROMOTE_RANKS = ([int(x) for x in os.environ["VLLM_MOET_CPU_TIER_PROMOTE_RANKS"].split(",")]
                  if os.environ.get("VLLM_MOET_CPU_TIER_PROMOTE_RANKS") else None)
_PROMOTE_LAYERS = ([int(x) for x in os.environ["VLLM_MOET_CPU_TIER_PROMOTE_LAYERS"].split("-")]
                   if os.environ.get("VLLM_MOET_CPU_TIER_PROMOTE_LAYERS") else None)
COMPACT_MAX_TOKENS = 8   # b12x's direct compact micro path "owns one through eight live tokens"
_TSC_GHZ = float(os.environ.get("VLLM_MOET_CPU_TIER_TSC_GHZ", "3.9937"))   # the pool's rdtsc rate (rtxdev2: 3993.689 MHz)
SCRATCH_SLOTS = int(os.environ.get("VLLM_MOET_CPU_TIER_SCRATCH", "0"))   # prefill scratch pool (0 = off; 2.5 under test)
# VLLM_MOET_CPU_TIER_PREFILL_OVERLAP=1 (10-29): two scratch sets and a side stream - a non-compact call's round 0 is
# staged + repacked while the stock (resident) b12x run executes, and round r + 1 while round r's extra run executes;
# the extra runs and their `add_` keep their order on the main stream, so the output is bit-identical to the serial
# path (the streaming cost is 0.12-0.13 ms per missing expert of PCIe bytes, note §20.1; this hides the GPU compute
# that today sits between the transfers). Costs a second staging + pool (SCRATCH slots x 2 x 4.7 MB).
# VLLM_MOET_CPU_TIER_DEBUG_OVERLAP=1: after every overlapped call, the same call again on the serial path into a scratch
# output, compared bit for bit (the in-boot A/B of the overlap; eager prefill only, host syncs allowed).
PREFILL_OVERLAP = os.environ.get("VLLM_MOET_CPU_TIER_PREFILL_OVERLAP", "0") == "1"
DEBUG_OVERLAP = os.environ.get("VLLM_MOET_CPU_TIER_DEBUG_OVERLAP", "0") == "1"


def _log(msg: str) -> None:
    print(f"[moet-cpu-tier] rank {_rank()}: {msg}", flush=True)


class _LayerCfg:
    """One hybrid layer, known from create_weights on (store mode) or from process_weights_after_loading (dummy)."""

    def __init__(self, li: int, name: str, model_layer: int, E: int, residents: list[int], spare: int):
        self.li, self.name, self.model_layer, self.E = li, name, model_layer, E
        self.residents = residents
        self.R = len(residents) + spare
        self.slot_of = {e: s for s, e in enumerate(residents)}


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.tier = None
        self.manager = None
        self.pools: list = []
        self.cfgs: dict[int, _LayerCfg] = {}       # id(layer module) -> cfg
        self.layers: list[_LayerCfg] = []          # by hybrid index
        self.skip: set = set()                     # id(layer) of target layers left on the GPU (geometry)
        self.E = 0
        self.remapped: list[torch.Tensor] = []     # per layer, [MAX_PAIRS] int32 (sliced per call)
        self.arena: dict[int, int] = {}            # dummy mode: li -> pinned arena base (E records)
        self.store = None                          # store mode: cpu_tier.ExpertStore
        self.scratch = None                        # phase 2.5: the prefill scratch pool (_Scratch)
        self.scratch2 = None                       # PREFILL_OVERLAP: the second set (rounds alternate between the two)
        self.prefill_stream = None                 # PREFILL_OVERLAP: the side stream the staging + repack run on
        self.overlap_stats = [0, 0, 0]             # calls overlapped, rounds prefetched, DEBUG_OVERLAP mismatches
        self.layer_modules: dict[int, object] = {}  # li -> the RoutedExperts module
        self.store_layers = 0
        self.step = 0
        self.ctx = threading.local()
        self.created = False
        self.seen_apply_tokens: set = set()
        self.seen_compact_tokens: set = set()
        self.noncompact_reported: set = set()
        self.impl_keys: dict = {}
        self.kpi = dict(pairs_all=0, pairs_cpu=0, busy_ns=0, tier_steps=0, promotions=0, last_log_t=time.time())


S = _State()


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def _ranges(cpus: list) -> str:
    """[2,3,4,34,35] -> '2-4,34-35' (the tier-started log line)."""
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(f"{cpus[i]}-{cpus[j]}" if j > i else str(cpus[i]))
        i = j + 1
    return ",".join(out)


def _cpu_layout() -> dict:
    """The rank's CPUs. Default on the 32-core / 4-CCD TR PRO (SMT siblings = core + 32): per rank r its CCD
    8r..8r+7: core 8r = this process's main thread (vLLM's launch loop), 8r+1 = the tier's dispatcher (spins), 8r+2..8r+7
    = the kernel pool (6 threads); the siblings 8r+32 / 8r+33 take the process's other threads; the pool cores' siblings
    stay idle so that the AVX-512 kernel does not share its cores. Overrides: VLLM_MOET_CPU_TIER_CPUS=a-b (pool),
    VLLM_MOET_CPU_TIER_DISPATCHER_CPU, VLLM_MOET_CPU_TIER_MAIN_CPU, VLLM_MOET_CPU_TIER_OTHER_CPUS=a,b,.."""
    r = _rank()
    spec = os.environ.get("VLLM_MOET_CPU_TIER_CPUS")
    # VLLM_MOET_CPU_TIER_POOL_THREADS (6): how much of the rank's CCD the kernel pool takes (10-28; the profile of a decode
    # step shows the GPU waiting 2.6 ms per step for the tier, and the bench runs the kernel 14-17 % faster on 8 threads
    # than on 6): 6 = cores 8r+2..8r+7, main 8r, dispatcher 8r+1 (the default layout above); 7 = cores 8r+1..8r+7, main
    # 8r, the dispatcher on main's SMT sibling 8r+32; 8 = the whole CCD 8r..8r+7, main and dispatcher on the siblings
    # 8r+32 / 8r+33, the process's other threads on 8r+34 / 8r+35 (they then share pool cores when busy)
    # 12 = the 6 pool cores AND their SMT siblings 8r+34..8r+39 (main / dispatcher / other threads as with 6): the
    # DRAM-bound kernel gains memory-level parallelism - bench on real records 10-28: 121 -> 100 us per expert at m = 1,
    # 214 -> 163 at m = 2; 14 = 7 cores x 2 (8r+1..8r+7 + 8r+33..8r+39), the dispatcher and the other threads on 8r+32
    nthreads = int(os.environ.get("VLLM_MOET_CPU_TIER_POOL_THREADS", "6"))
    d_main, d_disp, d_other = 8 * r, 8 * r + 1, [8 * r + 32, 8 * r + 33]
    if spec:
        parts = []
        for part in spec.split(","):
            a, _, b = part.partition("-")
            parts.extend(range(int(a), int(b or a) + 1))
        pool = parts
        disp = int(os.environ.get("VLLM_MOET_CPU_TIER_DISPATCHER_CPU", pool[0] + 32))
    else:
        if nthreads >= 14:
            pool = list(range(8 * r + 1, 8 * r + 8)) + list(range(8 * r + 33, 8 * r + 40))
            d_disp, d_other = 8 * r + 32, [8 * r + 32]
        elif nthreads >= 12:
            pool = list(range(8 * r + 2, 8 * r + 8)) + list(range(8 * r + 34, 8 * r + 40))
        elif nthreads >= 8:
            pool = list(range(8 * r, 8 * r + 8))
            d_main, d_disp, d_other = 8 * r + 32, 8 * r + 33, [8 * r + 34, 8 * r + 35]
        elif nthreads == 7:
            # 10-29 (k0-t7, note §19.5) - DO NOT USE without changing `other`: pool = cores 8r+1..8r+7, main on core 8r,
            # the dispatcher on 8r+32 (core 8r's SMT sibling, next to main) and `other` - the ~100 auxiliary threads of
            # the vLLM worker (NCCL watchdog, Gloo, the executor) - lands on 8r+33, which is the SMT SIBLING OF POOL CORE
            # 8r+1. A foreign busy thread on a pool core's sibling halves that core and the job's barrier waits for it
            # (rule in §9): measured 172 us per expert against 154 with 6 threads, NCCL 2.70 ms/step against 1.18-1.80,
            # the AllReduce p90 52.8 us against 36.7, decode 124.8 / 106.5 against 143 / 120. 6 is the default; 8 is the
            # variant that has a clean layout (pool 8r..8r+7, main + dispatcher on 8r+32/8r+33, other on 8r+34/8r+35)
            # and has not been measured in situ yet.
            pool = list(range(8 * r + 1, 8 * r + 8))
            d_disp, d_other = 8 * r + 32, [8 * r + 33]
        else:
            pool = list(range(8 * r + 2, 8 * r + 8))
        disp = int(os.environ.get("VLLM_MOET_CPU_TIER_DISPATCHER_CPU", d_disp))
    main = os.environ.get("VLLM_MOET_CPU_TIER_MAIN_CPU")
    main = int(main) if main else (d_main if not spec else None)
    other = os.environ.get("VLLM_MOET_CPU_TIER_OTHER_CPUS")
    other = [int(x) for x in other.split(",")] if other else (d_other if not spec else None)
    return dict(pool=pool, dispatcher=disp, main=main, other=other)


def _pin_process(layout: dict) -> str:
    """Pin this process's threads: the calling (main) thread to layout['main'], every other existing thread to
    layout['other'] (the tier's own threads pin themselves afterwards). VLLM_MOET_CPU_TIER_PIN=0 disables."""
    if os.environ.get("VLLM_MOET_CPU_TIER_PIN", "1") != "1" or layout["main"] is None:
        return "no process pinning"
    me = threading.get_native_id()
    n_other = 0
    try:
        for tid in os.listdir(f"/proc/{os.getpid()}/task"):
            tid = int(tid)
            if tid == me:
                continue
            try:
                os.sched_setaffinity(tid, set(layout["other"]))
                n_other += 1
            except OSError:
                pass
        os.sched_setaffinity(0, {layout["main"]})
    except OSError as ex:
        return f"process pinning failed: {ex!r}"
    return f"main thread -> CPU {layout['main']}, {n_other} other threads -> {layout['other']}"


def _num_hidden_layers() -> int:
    limit = int(os.environ.get("VLLM_MOET_CPU_TIER_TARGET_LAYERS", "0"))
    if limit:
        return limit
    try:
        from vllm.config import get_current_vllm_config
        return int(get_current_vllm_config().model_config.hf_config.num_hidden_layers)
    except Exception:
        return 10 ** 9


def _model_layer(layer) -> int | None:
    name = getattr(layer, "layer_name", "") or ""
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def is_target_layer(layer) -> bool:
    """A target-model MoE layer (not the DSpark drafter's: those carry layer indices past num_hidden_layers)."""
    idx = _model_layer(layer)
    return idx is not None and idx < _num_hidden_layers()


_PLACEMENT = None


def _residents_of(model_layer: int, E: int) -> list[int]:
    global _PLACEMENT
    placement = os.environ.get("VLLM_MOET_CPU_TIER_PLACEMENT")
    if placement:
        if _PLACEMENT is None:
            import json
            _PLACEMENT = json.load(open(placement))
        return [e for e in _PLACEMENT["layers"][model_layer]["resident"] if e < E]
    R = int(os.environ.get("VLLM_MOET_CPU_TIER_RESIDENT", str(E)))
    return list(range(min(R, E)))


def _register(layer, E: int) -> _LayerCfg:
    cfg = S.cfgs.get(id(layer))
    if cfg is not None:
        return cfg
    li = len(S.layers)
    name = getattr(layer, "layer_name", f"layer{li}")
    ml = _model_layer(layer)
    ml = li if ml is None else ml
    residents = _residents_of(ml, E)
    shift = int(os.environ.get("VLLM_MOET_CPU_TIER_RESIDENT_SHIFT", "0"))
    shift_ranks = os.environ.get("VLLM_MOET_CPU_TIER_SHIFT_RANKS", "")   # "1,3" = only these ranks drop (shift*rank)
    if shift and (not shift_ranks or _rank() in [int(x) for x in shift_ranks.split(",")]):
        # diagnostics: rank r drops its first shift*r residents (different residency per rank, static)
        residents = residents[shift * _rank():]
    cfg = _LayerCfg(li, name, ml, E, residents, SPARE)
    S.cfgs[id(layer)] = cfg
    S.layers.append(cfg)
    S.E = E
    return cfg


def _slot_mapper(cfg: _LayerCfg):
    def map_global_to_local(expert_id: int) -> int:
        return cfg.slot_of.get(int(expert_id), -1)
    return map_global_to_local


# ---------------------------------------------------------------------------------------------------------------
# called from Mxfp4MoEMethod.create_weights (store mode: the parameters get R rows and the loader maps experts to slots)
# ---------------------------------------------------------------------------------------------------------------
def on_create_weights(layer, num_experts: int, hidden_size: int, intermediate_size_per_partition: int) -> int:
    if not ENABLED or not is_target_layer(layer):
        return num_experts
    if intermediate_size_per_partition != N_R or hidden_size != HID:
        _log(f"layer {getattr(layer, 'layer_name', '?')}: intermediate {intermediate_size_per_partition} / hidden "
             f"{hidden_size} is not the TP4-slice geometry the CPU kernel computes; left on the GPU")
        S.skip.add(id(layer))
        return num_experts
    cfg = _register(layer, num_experts)
    if not STORE:
        return num_experts   # dummy mode: every expert is loaded, the subset is picked at process_weights time
    layer._map_global_expert_id_to_local_expert_id = _slot_mapper(cfg)
    _log(f"layer {cfg.name} -> hybrid layer {cfg.li}: parameters for {cfg.R} slots ({len(cfg.residents)} resident "
         f"experts of {num_experts} + {SPARE} spare); the loader skips the rest")
    # STORE_EARLY: pin the records at the LAST target layer's create_weights - the model is then constructed (vLLM's
    # loader runs after the construction), so the 260 GB of cudaHostAlloc happen before the weights and still AFTER the
    # Engram tables (engram_layer_ids 1 and 14 of V4.1-Flash: exact-size THP mmap + prefault + cudaHostRegister, 2 x 23.6
    # GiB per rank, allocated as those layers are built). Pinning at the FIRST create_weights (10-26 17:53, boot
    # k0-baked-early) fragmented the free memory with 260 GB of unmovable pages before the tables' huge-page prefault,
    # which then sat in direct compaction: 48 min of silence after the store line, the window timed out.
    if STORE_EARLY and S.store is None and os.environ.get("VLLM_MOET_CPU_TIER_PLACEMENT") \
            and not int(os.environ.get("VLLM_MOET_CPU_TIER_RESIDENT_SHIFT", "0")) \
            and cfg.model_layer == _num_hidden_layers() - 1:
        with S.lock:
            _ensure_store(early=True)
    return cfg.R


def _seed_dummy_weights(layer_idx: int, w13: torch.Tensor, w13s: torch.Tensor, w2: torch.Tensor, w2s: torch.Tensor) -> None:
    g = torch.Generator(device=w13.device).manual_seed(1000 + layer_idx)
    for packed in (w13, w2):
        v = torch.randint(0, 256, packed.shape, dtype=torch.int32, device=packed.device, generator=g)
        mag = v & 0x77
        nonzero = (mag | (mag >> 1) | (mag >> 2)) & 0x11
        packed.copy_((v & (0x77 | (nonzero << 3))).to(torch.uint8))
    for sc in (w13s, w2s):
        sc.copy_(torch.randint(118, 127, sc.shape, dtype=torch.int32, device=sc.device, generator=g).to(torch.uint8))


# ---- the records ----------------------------------------------------------------------------------------------
def _ensure_store(early: bool = False) -> None:
    """Pins the records (ExpertStore). Normally at the first process_weights_after_loading (every target layer
    registered by then); with VLLM_MOET_CPU_TIER_STORE_EARLY=1 already at the last target layer's create_weights, i.e.
    BEFORE the weights are loaded (and after the Engram tables, see on_create_weights): the placement file gives every
    layer's residents up front, so the masks are known, and the 260 GB of cudaHostAlloc then happen while the page
    cache is still empty instead of evicting the loader's shard pages under pressure afterwards (10-26: 16 min of
    direct reclaim in one boot)."""
    if S.store is not None:
        return
    from cpu_tier import ExpertStore

    if early:
        n_layers = _num_hidden_layers()
        if n_layers >= 10 ** 9:
            return   # no model config at hand: the normal path pins later
        E = S.E
        mask = np.zeros((n_layers, E), dtype=np.uint8)
        for ml in range(n_layers):
            if STORE_MODE == "all":
                mask[ml, :] = 1
            else:
                mask[ml, :] = 1
                mask[ml, _residents_of(ml, E)] = 0
    else:
        n_layers = max(c.model_layer for c in S.layers) + 1
        mask = np.zeros((n_layers, S.E), dtype=np.uint8)
        for c in S.layers:
            if STORE_MODE == "all":
                mask[c.model_layer, :] = 1
            else:
                mask[c.model_layer, :] = 1
                mask[c.model_layer, c.residents] = 0
    t0 = time.perf_counter()
    S.store = ExpertStore(STORE, _rank(), n_layers, S.E, mask=mask, pinned=True, nthreads=16)
    S.store_layers = n_layers
    _log(f"store {STORE} ({STORE_MODE}{', before the weights' if early else ''}): {S.store.n_records} records = "
         f"{S.store.nbytes / 1e9:.1f} GB pinned, read {S.store.load_seconds:.1f} s, alloc+read {time.perf_counter() - t0:.1f} s")


def _has_record(li: int, e: int) -> bool:
    if S.store is not None:
        return int(S.store.slot_of[S.layers[li].model_layer, e]) >= 0
    return li in S.arena


def _record_ptr(li: int, e: int) -> int:
    if S.store is not None:
        return S.store.ptr(S.layers[li].model_layer, e)
    return S.arena[li] + e * RECORD_BYTES


def _record_view(li: int, e: int) -> torch.Tensor:
    from cpu_tier import cuda_ext
    return cuda_ext().host_view(_record_ptr(li, e), RECORD_BYTES)


def _create_tier_if_needed(device: torch.device):
    """Deferred until the number of hybrid layers is known (after the last target layer registered)."""
    if S.created:
        return
    from cpu_tier import CpuTier
    from placement_manager import PlacementManager, load_layout

    n_layers = len(S.layers)
    tables = os.environ["VLLM_MOET_CPU_TIER_TABLES"]
    layout = _cpu_layout()
    cpus = layout["pool"]
    pin_msg = _pin_process(layout)
    S.tier = CpuTier(n_layers, S.E, cpus, layout["dispatcher"], device, tables,
                     timeout_ms=float(os.environ.get("VLLM_MOET_CPU_TIER_TIMEOUT_MS", "500")))
    n_rec = 0
    for li in range(n_layers):
        for e in range(S.E):
            if _has_record(li, e):
                S.tier.set_record(li, e, _record_ptr(li, e))
                n_rec += 1
    tables_layout = load_layout(tables, device)
    copy_stream = torch.cuda.Stream(device=device)
    S.manager = PlacementManager(S.tier.ext, tables_layout, S.pools, _record_view, copy_stream,
                                 max_promotions_per_layer=int(os.environ.get("VLLM_MOET_CPU_TIER_MAX_PROMOTIONS", "4")),
                                 record_ptr=_record_ptr)
    S.tier.start(first_step=1)
    S.created = True

    def _shutdown():
        try:
            if LOG:
                tot = S.tier.totals()
                _log(f"exit: steps {S.step}, tier steps {tot['steps']} pairs {tot['pairs']} busy {tot['busy_ns'] / 1e9:.1f} s "
                     f"errors {tot['errors']}; pairs all {S.kpi['pairs_all']} cpu {S.kpi['pairs_cpu']} promotions {S.kpi['promotions']}"
                     f"{_debug_compact_summary()}")
                if S.scratch is not None:
                    _log(f"exit: scratch pool {S.scratch.calls} calls, {S.scratch.experts_streamed} experts streamed in "
                         f"{S.scratch.rounds} rounds{S.scratch.timing_summary()}"
                         + (f"; prefill overlap: {S.overlap_stats[0]} calls overlapped, {S.overlap_stats[1]} rounds prefetched"
                            f"{f', DEBUG_OVERLAP mismatching calls {S.overlap_stats[2]}' if DEBUG_OVERLAP else ''}"
                            if PREFILL_OVERLAP else ""))
            S.tier.stop()
            S.tier.close()  # frees the mapped mailboxes before the driver tears the process down
        except Exception:
            pass

    atexit.register(_shutdown)
    _log(f"tier started: {n_layers} layers x {S.E} experts, slots {[c.R for c in S.layers][:3]}..., resident "
         f"{[len(c.residents) for c in S.layers][:3]}..., records {n_rec}, pool CPUs {_ranges(cpus)} ({len(cpus)} threads), "
         f"dispatcher {layout['dispatcher']}, {pin_msg}; promotion window {S.manager.window} steps; layers {[c.name for c in S.layers]}")
    _host_memory_report("after the weights")


def _host_memory_report(when: str) -> None:
    """The RAM item: torch's pinned caching host allocator keeps the loader's staging blocks (Shmem outside our
    store + Engram); VLLM_MOET_CPU_TIER_HOST_EMPTY_CACHE=1 releases them (torch._C._host_emptyCache) and logs the
    effect on MemAvailable."""
    def avail_gb() -> float:
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
        except OSError:
            pass
        return float("nan")
    stats = ""
    try:
        hs = torch.cuda.host_memory_stats()
        res = hs.get("reserved_bytes.all.current", 0)
        alloc = hs.get("allocated_bytes.all.current", 0)
        stats = f"torch pinned host allocator: reserved {res / 2**30:.1f} GiB, allocated {alloc / 2**30:.1f} GiB; "
    except Exception as ex:  # older torch
        stats = f"torch host_memory_stats unavailable ({type(ex).__name__}); "
    msg = f"host memory {when}: {stats}MemAvailable {avail_gb():.1f} GB"
    if os.environ.get("VLLM_MOET_CPU_TIER_HOST_EMPTY_CACHE", "0") == "1":
        try:
            torch.cuda.synchronize()
            torch._C._host_emptyCache()
            msg += f" -> after _host_emptyCache: MemAvailable {avail_gb():.1f} GB"
        except Exception as ex:
            msg += f" (_host_emptyCache failed: {ex!r})"
    _log(msg)


# ---------------------------------------------------------------------------------------------------------------
# called from vLLM's B12xExperts.process_weights_after_loading, BEFORE _prepare_experts
# ---------------------------------------------------------------------------------------------------------------
def on_process_weights(experts_impl, layer, w13: torch.Tensor, w2: torch.Tensor, w13_scale: torch.Tensor,
                       w2_scale: torch.Tensor):
    """Returns None (stock path) or (w13, w2, w13_scale, w2_scale) with R rows: what b12x prepares."""
    if not ENABLED or not is_target_layer(layer) or id(layer) in S.skip:
        return None
    if w13.dim() != 3 or w13.shape[1] != 2 * N_R or w13.shape[2] != HID // 2 or w2.shape[1] != HID:
        _log(f"layer {getattr(layer, 'layer_name', '?')}: geometry {tuple(w13.shape)} / {tuple(w2.shape)} is not the "
             f"TP4-slice geometry the CPU kernel computes; left on the GPU")
        S.skip.add(id(layer))
        return None
    from placement_manager import ResidentPool
    from cpu_tier import cuda_ext

    E = int(getattr(layer, "global_num_experts", 0) or w13.shape[0])
    cfg = _register(layer, E)
    li = cfg.li
    if STORE:
        if int(w13.shape[0]) != cfg.R:
            raise RuntimeError(f"layer {cfg.name}: {w13.shape[0]} expert rows loaded, {cfg.R} slots expected (was "
                               f"Mxfp4MoEMethod.create_weights patched?)")
        _ensure_store()
    elif os.environ.get("VLLM_MOET_CPU_TIER_DUMMY_WEIGHTS", "0") == "1":
        _seed_dummy_weights(li, w13, w13_scale, w2, w2_scale)
    # canonicalize -0 like vLLM does before b12x (so the records equal what prepare_weights sees)
    for packed in (w13, w2):
        mag = packed & 0x77
        nonzero = (mag | (mag >> 1) | (mag >> 2)) & 0x11
        packed.bitwise_and_(0x77 | (nonzero << 3))
    residents = cfg.residents
    if STORE:
        sub = (w13, w2, w13_scale, w2_scale)
    else:
        # dummy mode: the records from the loaded GPU weights into a pinned arena of this layer
        ext = cuda_ext()
        base = int(ext.alloc_pinned(E * RECORD_BYTES))
        for e in range(E):
            rec = torch.cat([w13[e, :N_R].reshape(-1), w13_scale[e, :N_R].reshape(-1), w13[e, N_R:].reshape(-1),
                             w13_scale[e, N_R:].reshape(-1), w2[e].reshape(-1), w2_scale[e].reshape(-1)])
            assert rec.numel() == RECORD_BYTES, rec.numel()
            ext.host_view(base + e * RECORD_BYTES, RECORD_BYTES).copy_(rec.cpu())
        S.arena[li] = base
        idx = torch.tensor(residents + [0] * (cfg.R - len(residents)), dtype=torch.long, device=w13.device)
        sub = tuple(t.index_select(0, idx).contiguous() for t in (w13, w2, w13_scale, w2_scale))
        for t in sub:
            t[len(residents):] = 0   # spare slots: zeros until a promotion fills them
    # the slot table and the pool shell (its tensors are the prepared storages, attached at the first launch)
    slot_table = torch.full((E,), -1, dtype=torch.int32, device=w13.device)
    pool = ResidentPool(li, E, cfg.R, w13.device, pool={}, slot_table=slot_table,
                        record_ok=lambda e, _li=li: _has_record(_li, e))
    for s, e in enumerate(residents):
        slot_table[e] = s
        pool.expert_of_slot[s] = e
        pool.slot_of_expert[e] = s
        pool.last_used[s] = 0
    S.pools.append(pool)
    S.remapped.append(torch.full((TOPK_MAX * 16,), -1, dtype=torch.int32, device=w13.device))
    _log(f"layer {cfg.name} -> hybrid layer {li}: {len(residents)} of {E} experts resident in {cfg.R} slots, records "
         f"{'from the store' if STORE else 'in a pinned arena'}")
    layer._moet_cpu_tier_layer = li
    experts_impl._moet_cpu_tier_layer = li
    S.layer_modules[li] = layer
    return sub


# ---------------------------------------------------------------------------------------------------------------
# called from B12xExperts.apply (before the b12x run) and from the compact launch
# ---------------------------------------------------------------------------------------------------------------
def on_apply(experts_impl, hidden_states: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    """Returns the ids the b12x run gets: the router's on the compact path (publish remaps them), remapped through
    the slot table on any other path (memory-safe with R-row weights; the missing pairs are zero rows there - not
    hybrid yet, phase 2.5)."""
    # the layer context lives from this apply to its compact launch only; an apply that takes another b12x path
    # (prefill-sized calls) must not leave it behind for launches made outside apply (vLLM's b12x warmup units)
    li = getattr(experts_impl, "_moet_cpu_tier_layer", None)
    S.ctx.layer = li
    S.ctx.prefill = None
    S.ctx.last_ids = topk_ids
    if li is None:
        return topk_ids
    if not S.created:
        with S.lock:
            _create_tier_if_needed(hidden_states.device)
    T = int(hidden_states.shape[0])
    if DEBUG:
        prep = getattr(experts_impl, "_prepared_experts", None)
        key = (id(experts_impl), id(prep), int(getattr(prep, "w13_rp", torch.empty(0)).data_ptr()) if prep is not None else 0)
        known = S.impl_keys.setdefault(li, {})
        if key not in known:
            known[key] = f"T {T} step {S.step} capturing {torch.cuda.is_current_stream_capturing()}"
            if len(known) > 1:
                _log(f"WARNING layer {li}: apply from a new (experts_impl, prepared, w13_rp) {key} [{known[key]}]; known {known}")
    if T not in S.seen_apply_tokens:
        S.seen_apply_tokens.add(T)
        _log(f"apply with {T} tokens (compact-path token counts so far: {sorted(S.seen_compact_tokens)})")
    if li == 0:
        S.tier.step_begin()   # captured with the step: the device step counter the mailboxes carry
    if T > COMPACT_MAX_TOKENS:
        # not the compact path (prefill, batched decode): no publish / inject. The stock run gets the ids remapped
        # through the slot table (resident experts; missing pairs -1 = zero rows, which also keeps the kernels inside
        # the R-row weights); on_apply_end adds the missing experts streamed through the scratch pool (phase 2.5).
        S.ctx.layer = None
        S.ctx.prefill = (li, topk_ids)
        S.ctx.prefill_missing = None
        if T not in S.noncompact_reported:
            S.noncompact_reported.add(T)
            _log(f"{T}-token MoE call: b12x's non-compact path; missing experts "
                 f"{'streamed through the scratch pool' if SCRATCH_SLOTS else 'NOT computed (VLLM_MOET_CPU_TIER_SCRATCH=0)'}")
        table = S.pools[li].slot_table
        remapped = torch.where(topk_ids >= 0, table[topk_ids.clamp(min=0).long()], topk_ids.new_full((), -1))
        if PREFILL_OVERLAP and SCRATCH_SLOTS and S.scratch is not None and not torch.cuda.is_current_stream_capturing():
            _prefill_prefetch_round0(li, topk_ids)
        return remapped
    S.ctx.prefill = None
    return topk_ids


# ---------------------------------------------------------------------------------------------------------------
# phase 2.5: the non-compact calls' missing experts, streamed from the pinned store through a scratch pool
# ---------------------------------------------------------------------------------------------------------------
class _Scratch:
    """One scratch pool shared by all layers: SCRATCH_SLOTS experts prepared by b12x (zeros at startup), refilled per
    layer and round by stage_records (pinned -> device staging) + repack_batch (into the prepared storages)."""

    def __init__(self, experts_impl, layer, device: torch.device):
        from b12x.moe.fused_moe import _impl as b12x_impl

        n = SCRATCH_SLOTS
        self.n = n
        w1 = torch.zeros(n, 2 * N_R, HID // 2, dtype=torch.uint8, device=device)
        w2 = torch.zeros(n, HID, N_R // 2, dtype=torch.uint8, device=device)
        w1s = torch.zeros(n, 2 * N_R, HID // 32, dtype=torch.uint8, device=device)
        w2s = torch.zeros(n, HID, N_R // 32, dtype=torch.uint8, device=device)
        qc = experts_impl.quant_config
        saved = (qc._w1.scale, qc._w2.scale)
        try:
            qc._w1.scale, qc._w2.scale = w1s, w2s
            self.prepared = experts_impl._prepare_experts(w1=w1, w2=w2, activation=layer.activation,
                                                          params_dtype=experts_impl.moe_config.in_dtype)
        finally:
            qc._w1.scale, qc._w2.scale = saved
        payload = self.prepared.representation_for("w4a8_mx")
        views = b12x_impl._w4a8_prepared_dict(payload)
        self.pool = dict(w13=views["w13_rp"].view(n, -1), w13s=views["w13_sfb"].view(n, -1),
                         w2=views["w2_rp"].view(n, -1), w2s=views["w2_sfb"].view(n, -1))
        self.staging = torch.empty(n, RECORD_BYTES, dtype=torch.uint8, device=device)
        self.table = torch.full((S.E,), -1, dtype=torch.int32, device=device)
        self.plans: dict = {}
        self.buf = None            # one b12x scratch for every token count (sized for the largest seen)
        self.out = None            # one output buffer [T_max, HID]
        self.impl = experts_impl
        self.activation = layer.activation
        self.rounds = 0
        self.experts_streamed = 0
        self.calls = 0
        self.pending: list = []        # (ev0, ev1, T, missing) of calls whose events may not have completed yet
        self.timed: dict = {}          # T -> [calls, ms, missing experts] of completed calls
        # PREFILL_OVERLAP: `free` is recorded on the main stream after the last extra run that read this set's pool
        # (the side stream waits for it before restaging the set), `ready` on the side stream after the repack (the
        # main stream waits for it before the extra run); an event never recorded is a no-op to wait on
        self.free_event = torch.cuda.Event()
        self.ready_event = torch.cuda.Event()
        self.staged: list = []         # the experts currently staged in this set, in slot order (host-side bookkeeping)

    def timing_summary(self) -> str:
        """Folds the completed events into per-token-count totals; returns a summary (no host sync)."""
        keep = []
        for ev0, ev1, T, n_missing in self.pending:
            if ev1.query():
                acc = self.timed.setdefault(T, [0, 0.0, 0])
                acc[0] += 1
                acc[1] += ev0.elapsed_time(ev1)
                acc[2] += n_missing
            else:
                keep.append((ev0, ev1, T, n_missing))
        self.pending = keep[-64:]
        if not self.timed:
            return ""
        parts = []
        for T in sorted(self.timed):
            calls, ms, n_missing = self.timed[T]
            parts.append(f"T {T}: {calls} calls, {ms / calls:.1f} ms per call, {ms / max(1, n_missing):.2f} ms per missing expert")
        return "; streaming (stage + repack + extra runs): " + " | ".join(parts)

    def plan(self, T: int, K: int, nbytes_fn):
        """b12x's plan for T tokens (cached per (T, K) like vLLM's own) and ONE scratch + output buffer shared by all
        token counts, sized for the largest T seen so far (10-26: a buffer per distinct T leaked ~15 MB per prompt
        length - HumanEval's 104 distinct lengths took the device to OOM at util 0.92)."""
        key = (T, K)
        if key not in self.plans:
            from vllm.model_executor.layers.fused_moe.b12x import _require_b12x_fused_moe
            fused_moe = _require_b12x_fused_moe()
            limit, alpha, beta = self.impl._swiglu_params(self.activation)
            plan = fused_moe.plan(fused_moe.Caps(
                max_tokens=max(T, 1), num_topk=K, device=self.prepared.w1_fp4.device, weight_plan=self.prepared.plan,
                core_token_counts=(max(T, 1),), route_num_experts=0, quant_mode=self.impl._quant_mode,
                apply_router_weight_on_input=False, swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta, frozen=True))
            self.plans[key] = plan
        plan = self.plans[key]
        need = int(nbytes_fn(plan))
        dev = self.prepared.w1_fp4.device
        if self.buf is None or self.buf.numel() < need:
            self.buf = None
            self.buf = torch.empty(need, dtype=torch.uint8, device=dev)
        if self.out is None or self.out.shape[0] < T:
            self.out = None
            self.out = torch.empty(T, HID, dtype=self.impl.moe_config.in_dtype, device=dev)
        return plan, (self.buf, self.out[:T])


def _dump_call(experts_impl, hidden_states, topk_weights, output) -> None:
    """Debug (eager runs: graph replays never reach here): the first DUMP_MAX target-layer MoE calls of this rank -
    hidden states, the router's ids / weights, the layer's MoE output (the rank's partial) - to DUMP_DIR."""
    li = getattr(experts_impl, "_moet_cpu_tier_layer", None)
    if li is None:
        return
    n = S.kpi.get("dumped", 0)
    if n >= DUMP_MAX:
        return
    S.kpi["dumped"] = n + 1
    os.makedirs(DUMP_DIR, exist_ok=True)
    ids = getattr(S.ctx, "last_ids", None)
    torch.cuda.synchronize()
    torch.save(dict(call=n, step=S.step, layer=li, T=int(hidden_states.shape[0]),
                    x=hidden_states.detach().float().cpu(), ids=None if ids is None else ids.detach().cpu(),
                    w=topk_weights.detach().float().cpu(), out=output.detach().float().cpu(),
                    slots=S.pools[li].slot_table.detach().cpu()),
               os.path.join(DUMP_DIR, f"rank{_rank()}_{n:04d}.pt"))


def _stage_and_repack(sc, ptrs: list) -> None:
    """Records ptrs[i] -> scratch slot i: pinned -> device staging, then the batched repack (<= 256 per launch)."""
    ext = S.tier.ext
    ext.stage_records(sc.staging, ptrs)
    lay = S.manager.layout
    for c0 in range(0, len(ptrs), 256):
        slots = list(range(c0, min(len(ptrs), c0 + 256)))
        ext.repack_batch(sc.staging[c0:c0 + len(slots)], lay["w13"], lay["w13s"], lay["w2"], lay["w2s"],
                         sc.pool["w13"], sc.pool["w13s"], sc.pool["w2"], sc.pool["w2s"], slots)


def _missing_experts(li: int, ids: torch.Tensor):
    """The call's missing experts (host list, sorted by torch.unique) and the valid mask - one host sync."""
    P = S.pools[li]
    valid = ids >= 0
    missing_mask = valid & (P.slot_table[ids.clamp(min=0).long()] < 0)
    return valid, torch.unique(ids[missing_mask]).tolist()


def _overlap_sets():
    """The two scratch sets of PREFILL_OVERLAP (the second one created on first use) and the side stream."""
    if S.scratch2 is None:
        with S.lock:
            if S.scratch2 is None:
                li0 = next(iter(S.layer_modules))
                S.scratch2 = _Scratch(S.scratch.impl, S.layer_modules[li0], S.scratch.staging.device)
                S.prefill_stream = torch.cuda.Stream(device=S.scratch.staging.device)
                _log(f"prefill overlap: a second scratch set of {SCRATCH_SLOTS} slots and a side stream "
                     f"(VLLM_MOET_CPU_TIER_PREFILL_OVERLAP=1{', DEBUG_OVERLAP: every call re-run serially and compared' if DEBUG_OVERLAP else ''})")
    return (S.scratch, S.scratch2), S.prefill_stream


def _stage_round_async(sc, li: int, rnd: list) -> None:
    """Stage + repack one round into set `sc` on the side stream: after the set's last reader (free_event, main
    stream), then ready_event for the main stream's extra run."""
    side = S.prefill_stream
    side.wait_event(sc.free_event)
    with torch.cuda.stream(side):
        _stage_and_repack(sc, [_record_ptr(li, e) for e in rnd])
        sc.ready_event.record(side)
    sc.staged = list(rnd)


def _prefill_prefetch_round0(li: int, topk_ids: torch.Tensor) -> None:
    """on_apply of a non-compact call (PREFILL_OVERLAP): the missing set now (one host sync, instead of the one
    on_apply_end would do), round 0 staged + repacked on the side stream while the stock run executes on the main
    stream. on_apply_end picks the list up and skips round 0's staging."""
    valid, missing = _missing_experts(li, topk_ids)
    S.ctx.prefill_missing = (valid, missing)
    if not missing:
        return
    sets, _ = _overlap_sets()
    _stage_round_async(sets[0], li, missing[:sets[0].n])
    S.overlap_stats[1] += 1


def _run_rounds(sc_of_round, li, ids, valid, missing, weights, hidden_states, output, run_plan, plan, buf, tmp,
                prefetched_round0: bool, prefetch_next: bool) -> int:
    """The streamed rounds of one call: for round r, the set sc_of_round(r) is staged (synchronously on the main
    stream, or already / asynchronously on the side stream), its table written, the extra b12x run added to output.
    Returns the number of rounds. The run and the add_ order are the same on every path."""
    n = sc_of_round(0).n
    rounds = [missing[r0:r0 + n] for r0 in range(0, len(missing), n)]
    main = torch.cuda.current_stream()
    for r, rnd in enumerate(rounds):
        sc = sc_of_round(r)
        if r == 0 and prefetched_round0:
            main.wait_event(sc.ready_event)
        elif prefetch_next and r > 0:
            main.wait_event(sc.ready_event)          # staged while round r - 1 ran
        else:
            _stage_and_repack(sc, [_record_ptr(li, e) for e in rnd])
            sc.staged = list(rnd)
        assert sc.staged == list(rnd), f"prefill round {r} of layer {li}: set holds {len(sc.staged)} experts, round wants {len(rnd)}"
        if prefetch_next and r + 1 < len(rounds):
            _stage_round_async(sc_of_round(r + 1), li, rounds[r + 1])
            S.overlap_stats[1] += 1
        sc.table.fill_(-1)
        rnd_t = torch.tensor(rnd, dtype=torch.long, device=ids.device)
        sc.table[rnd_t] = torch.arange(len(rnd), dtype=torch.int32, device=ids.device)
        rids = torch.where(valid, sc.table[ids.clamp(min=0).long()], ids.new_full((), -1)).contiguous()
        run_plan(plan=plan, scratch=buf, hidden_states=hidden_states, prepared=sc.prepared, topk_weights=weights,
                 topk_ids=rids, output=tmp, unit_scale_contract=False)
        output.add_(tmp)
        sc.free_event.record(main)                   # this set's pool may be restaged once the run is done
    return len(rounds)


def on_apply_end(experts_impl, hidden_states: torch.Tensor, topk_weights: torch.Tensor, output: torch.Tensor,
                 run_plan, scratch_nbytes) -> None:
    """After the stock b12x run of a non-compact call: the missing experts of the call, in rounds of SCRATCH_SLOTS:
    records staged from the pinned store, repacked into the scratch pool, one more b12x run on the ids remapped to
    the scratch slots (resident pairs -1), the result added to the output. Eager path (prefill / T > 8): host syncs
    allowed (rule R6)."""
    if DUMP_DIR and S.created:
        _dump_call(experts_impl, hidden_states, topk_weights, output)
    if DEBUG_COMPACT and S.created and getattr(S.ctx, "compact_call", None) is not None \
            and not torch.cuda.is_current_stream_capturing():
        li = S.ctx.compact_call
        S.ctx.compact_call = None
        _debug_compact_reference(experts_impl, li, hidden_states, topk_weights, output, run_plan, scratch_nbytes)
    pf = getattr(S.ctx, "prefill", None)
    if pf is None or not SCRATCH_SLOTS or not S.created:
        return
    S.ctx.prefill = None
    li, ids = pf
    T, K = ids.shape
    with S.lock:
        if S.scratch is None:
            S.scratch = _Scratch(experts_impl, S.layer_modules[li], hidden_states.device)
            _log(f"scratch pool: {SCRATCH_SLOTS} slots ({SCRATCH_SLOTS * RECORD_BYTES / 2**20:.0f} MiB staging + pool)")
    sc = S.scratch
    pre = getattr(S.ctx, "prefill_missing", None)
    S.ctx.prefill_missing = None
    if pre is not None:                 # PREFILL_OVERLAP: on_apply did the host sync and staged round 0 on the side stream
        valid, missing = pre
    else:
        valid, missing = _missing_experts(li, ids)
    sc.calls += 1
    if not missing:
        return
    plan, (buf, tmp) = sc.plan(T, K, scratch_nbytes)
    weights = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.to(torch.float32)
    weights = weights.contiguous()
    # the streaming's cost per call, from CUDA events read later (no host sync here): stage + repack + the extra runs
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    overlapped = pre is not None and S.scratch2 is not None
    if overlapped:
        sets, _ = _overlap_sets()
        before = output.clone() if DEBUG_OVERLAP else None
        n_rounds = _run_rounds(lambda r: sets[r % 2], li, ids, valid, missing, weights, hidden_states, output,
                               run_plan, plan, buf, tmp, prefetched_round0=True, prefetch_next=True)
        S.overlap_stats[0] += 1
        if DEBUG_OVERLAP:
            _debug_overlap_check(li, T, ids, valid, missing, weights, hidden_states, output, before, run_plan, plan, buf, tmp)
    else:
        n_rounds = _run_rounds(lambda r: sc, li, ids, valid, missing, weights, hidden_states, output,
                               run_plan, plan, buf, tmp, prefetched_round0=False, prefetch_next=False)
    sc.rounds += n_rounds
    sc.experts_streamed += len(missing)
    ev1.record()
    sc.pending.append((ev0, ev1, T, len(missing)))
    if sc.calls <= 3 or sc.calls % 200 == 0:
        _log(f"prefill call {sc.calls} (layer {li}, {T} tokens): {len(missing)} missing experts in "
             f"{(len(missing) + sc.n - 1) // sc.n} rounds; totals {sc.experts_streamed} experts / {sc.rounds} rounds"
             f"{sc.timing_summary()}")
    if DEBUG_PREFILL:
        _debug_prefill_reference(sc, li, ids, valid, weights, hidden_states, output, run_plan, plan, buf, T)


def _debug_overlap_check(li, T, ids, valid, missing, weights, hidden_states, output, before, run_plan, plan, buf, tmp) -> None:
    """DEBUG_OVERLAP: the same call once more on the serial path (set 0 only, staged on the main stream) into a
    scratch copy of the pre-streaming output, compared bit for bit with the overlapped result. Host syncs."""
    torch.cuda.synchronize()
    ref = before
    sets, _ = _overlap_sets()
    sc0 = sets[0]
    _run_rounds(lambda r: sc0, li, ids, valid, missing, weights, hidden_states, ref, run_plan, plan, buf, tmp,
                prefetched_round0=False, prefetch_next=False)
    torch.cuda.synchronize()
    nd = int((ref != output).sum().item())
    if nd:
        S.overlap_stats[2] += 1
        d = (ref.float() - output.float()).abs()
        _log(f"DEBUG overlap layer {li} T {T}: {len(missing)} missing experts in {(len(missing) + sc0.n - 1) // sc0.n} rounds: "
             f"{nd} of {output.numel()} elements differ between the overlapped and the serial path, max |d| {d.max().item():.3e}")
    elif S.overlap_stats[0] <= 3 or S.overlap_stats[0] % 200 == 0:
        _log(f"DEBUG overlap layer {li} T {T}: {len(missing)} missing experts in {(len(missing) + sc0.n - 1) // sc0.n} rounds: "
             f"overlapped == serial (0 of {output.numel()} differ); calls overlapped {S.overlap_stats[0]}, rounds prefetched "
             f"{S.overlap_stats[1]}, mismatching calls so far {S.overlap_stats[2]}")


def _debug_prefill_reference(sc, li, ids, valid, weights, hidden_states, output, run_plan, plan, buf, T) -> None:
    """Debug (VLLM_MOET_CPU_TIER_DEBUG_PREFILL=1, needs SCRATCH >= the call's distinct experts): the same call with
    EVERY routed expert in the scratch pool in one b12x run = the all-resident reference for this layer's output;
    run twice to see the kernel's own run-to-run noise; the hybrid output (resident run + streamed rounds) against it."""
    routed = [int(e) for e in torch.unique(ids[valid]).tolist()]
    if len(routed) > sc.n:
        _log(f"DEBUG prefill layer {li}: {len(routed)} routed experts > {sc.n} scratch slots; reference skipped")
        return
    # experts with a record are staged from the pinned store; the static residents have none in the nonresident store
    # (10-28: KeyError (0, 317) killed a boot in the profile run) - their prepared rows are copied from the layer's pool,
    # as _debug_compact_reference does; the promotions in flight are drained first so the slot table is consistent
    from_store = [e for e in routed if _has_record(li, e)]
    from_pool = [e for e in routed if not _has_record(li, e)]
    routed = from_store + from_pool
    P = S.pools[li]
    if from_pool and not P.pool:
        # the layer's prepared storages are attached at its first COMPACT launch (publish); the profile run's 4096-token
        # prefill comes before any (10-28: KeyError 'w13' killed a boot) - the check waits for a later call
        _log(f"DEBUG prefill layer {li}: {len(from_pool)} routed experts without a record and the layer's pool not "
             f"attached yet (no compact launch so far); reference skipped")
        return
    if from_store:
        _stage_and_repack(sc, [_record_ptr(li, e) for e in from_store])
    if from_pool:
        S.manager.copy.synchronize()
        torch.cuda.synchronize()
        table = P.slot_table.cpu().numpy()
        for j, e in enumerate(from_pool, start=len(from_store)):
            s = int(table[e])
            if s < 0:
                _log(f"DEBUG prefill layer {li}: expert {e} has neither a record nor a slot; reference skipped")
                return
            for name in sc.pool:
                sc.pool[name][j].copy_(P.pool[name][s])
    sc.table.fill_(-1)
    sc.table[torch.tensor(routed, dtype=torch.long, device=ids.device)] = torch.arange(len(routed), dtype=torch.int32, device=ids.device)
    rids = torch.where(valid, sc.table[ids.clamp(min=0).long()], ids.new_full((), -1)).contiguous()
    ref = torch.empty_like(output)
    ref2 = torch.empty_like(output)
    run_plan(plan=plan, scratch=buf, hidden_states=hidden_states, prepared=sc.prepared, topk_weights=weights, topk_ids=rids,
             output=ref, unit_scale_contract=False)
    run_plan(plan=plan, scratch=buf, hidden_states=hidden_states, prepared=sc.prepared, topk_weights=weights, topk_ids=rids,
             output=ref2, unit_scale_contract=False)
    torch.cuda.synchronize()
    o, r, r2 = output.float(), ref.float(), ref2.float()
    noise = (r - r2).abs().max().item()
    d = (o - r).abs()
    rel = (d.max() / r.abs().max().clamp(min=1e-20)).item()
    _log(f"DEBUG prefill layer {li} T {T}: {len(routed)} routed experts; hybrid vs all-in-one-run reference: max |d| "
         f"{d.max().item():.3e} (rel {rel:.2e}), elements differing {(d > 0).float().mean().item():.4f}, reference run-to-run "
         f"max |d| {noise:.3e}; |ref| max {r.abs().max().item():.3e}")


def _debug_compact_reference(experts_impl, li, hidden_states, topk_weights, output, run_plan, scratch_nbytes) -> None:
    """Debug (VLLM_MOET_CPU_TIER_DEBUG_COMPACT=1, eager runs): the compact call just made, once more with EVERY routed
    expert of the call in the scratch pool in one stock b12x run = the all-resident reference of this boot for this
    layer and step (TP4 boots are not reproducible among themselves, so the reference has to live in the same boot).
    Experts with a record are staged from the pinned store; the resident ones without a record (the nonresident
    store) are copied from the layer's pool rows. Compared bit for bit: the route rows per pair (the GPU's rows for
    the resident pairs, the host's injected rows for the missing ones) and the layer output after TopKSum."""
    ids = getattr(S.ctx, "last_ids", None)
    hyb = getattr(S.ctx, "hyb_rows", None)
    S.ctx.hyb_rows = None
    if ids is None or hyb is None or not SCRATCH_SLOTS:
        if not S.kpi.get("dc_off_reported"):
            S.kpi["dc_off_reported"] = True
            _log("DEBUG compact: no scratch pool (VLLM_MOET_CPU_TIER_SCRATCH=0) or no rows captured; check skipped")
        return
    T, K = ids.shape
    P = S.pools[li]
    with S.lock:
        if S.scratch is None:
            S.scratch = _Scratch(experts_impl, S.layer_modules[li], hidden_states.device)
            _log(f"scratch pool: {SCRATCH_SLOTS} slots ({SCRATCH_SLOTS * RECORD_BYTES / 2**20:.0f} MiB staging + pool)")
    sc = S.scratch
    S.manager.copy.synchronize()   # promotions in flight: the slot table and the pool rows consistent before reading
    torch.cuda.synchronize()
    ids_h = ids.cpu().numpy().reshape(-1)
    gpu_ids = S.remapped[li][: T * K].cpu().numpy()   # what the compact kernel ran: the slot, or -1 = the host's pair
    table = P.slot_table.cpu().numpy()
    routed = sorted(set(int(e) for e in ids_h if e >= 0))
    if len(routed) > sc.n:
        _log(f"DEBUG compact layer {li}: {len(routed)} routed experts > {sc.n} scratch slots; check skipped")
        return
    from_store = [e for e in routed if _has_record(li, e)]
    from_pool = [e for e in routed if not _has_record(li, e)]
    order = from_store + from_pool
    if from_store:
        _stage_and_repack(sc, [_record_ptr(li, e) for e in from_store])
    for j, e in enumerate(from_pool, start=len(from_store)):
        s = int(table[e])
        if s < 0:
            _log(f"DEBUG compact layer {li}: expert {e} has neither a record nor a slot; check skipped")
            return
        for name in sc.pool:
            sc.pool[name][j].copy_(P.pool[name][s])
    sc.table.fill_(-1)
    sc.table[torch.tensor(order, dtype=torch.long, device=ids.device)] = torch.arange(len(order), dtype=torch.int32, device=ids.device)
    valid = ids >= 0
    rids = torch.where(valid, sc.table[ids.clamp(min=0).long()], ids.new_full((), -1)).contiguous()
    plan, (buf, tmp) = sc.plan(T, K, scratch_nbytes)
    weights = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.to(torch.float32)
    ref = torch.empty_like(output)
    S.ctx.ref_rows = None
    S.ctx.want_ref_rows = True
    try:
        run_plan(plan=plan, scratch=buf, hidden_states=hidden_states, prepared=sc.prepared, topk_weights=weights.contiguous(),
                 topk_ids=rids, output=ref, unit_scale_contract=False)
    finally:
        S.ctx.want_ref_rows = False
    torch.cuda.synchronize()
    ref_rows = getattr(S.ctx, "ref_rows", None)
    S.ctx.ref_rows = None
    n_pairs = T * K
    valid_h = ids_h >= 0
    is_gpu = valid_h & (gpu_ids >= 0)
    is_cpu = valid_h & (gpu_ids < 0)
    out_eq = bool(torch.equal(output.view(torch.int16), ref.view(torch.int16)))
    d_out = (output.float() - ref.float()).abs()
    bad_gpu = bad_cpu = 0
    if ref_rows is None:
        row_msg = "reference rows missing (the reference took a non-compact path)"
    else:
        diff = (hyb[:n_pairs].view(torch.int16) != ref_rows[:n_pairs].view(torch.int16)).any(dim=1).cpu().numpy()
        bad_gpu, bad_cpu = int((diff & is_gpu).sum()), int((diff & is_cpu).sum())
        row_msg = f"rows differing: gpu {bad_gpu} of {int(is_gpu.sum())}, cpu {bad_cpu} of {int(is_cpu.sum())}"
    k = S.kpi
    k["dc_calls"] = k.get("dc_calls", 0) + 1
    k["dc_cpu_pairs"] = k.get("dc_cpu_pairs", 0) + int(is_cpu.sum())
    k["dc_gpu_pairs"] = k.get("dc_gpu_pairs", 0) + int(is_gpu.sum())
    k["dc_bad_out"] = k.get("dc_bad_out", 0) + (0 if out_eq else 1)
    k["dc_bad_rows_gpu"] = k.get("dc_bad_rows_gpu", 0) + bad_gpu
    k["dc_bad_rows_cpu"] = k.get("dc_bad_rows_cpu", 0) + bad_cpu
    k["dc_no_rows"] = k.get("dc_no_rows", 0) + (1 if ref_rows is None else 0)
    seen = k.setdefault("dc_seen", {})
    first = seen.get(li, 0)
    seen[li] = first + 1
    bad = (not out_eq) or bad_gpu or bad_cpu
    if bad or first < 2:
        _log(f"DEBUG compact layer {li} step {S.step} T {T}: {int(valid_h.sum())} pairs, {len(routed)} experts "
             f"({len(from_store)} from the store, {len(from_pool)} from the pool); {row_msg}; output == reference: {out_eq} "
             f"(max |d| {d_out.max().item():.3e}, |ref| max {ref.float().abs().max().item():.3e})"
             f"{'  <-- MISMATCH' if bad else ''}")


def _debug_compact_summary() -> str:
    k = S.kpi
    if not DEBUG_COMPACT or not k.get("dc_calls"):
        return ""
    return (f"; compact check: {k['dc_calls']} calls, pairs gpu {k['dc_gpu_pairs']} cpu {k['dc_cpu_pairs']}; mismatches: "
            f"outputs {k['dc_bad_out']}, rows gpu {k['dc_bad_rows_gpu']} cpu {k['dc_bad_rows_cpu']}, references without rows {k['dc_no_rows']}")


def on_apply_done() -> None:
    S.ctx.layer = None


def current_layer():
    return getattr(S.ctx, "layer", None)


def before_compiled(topk_ids: torch.Tensor, topk_weights: torch.Tensor, values: torch.Tensor, scale_rows: torch.Tensor,
                    num_tokens: int, w13, w13_scales, w2, w2_scales):
    """Inside launch_w4a8_compact_micro after the rows quant: publish; returns the ids for the compact kernel."""
    li = current_layer()
    if li is None or not S.created:
        return topk_ids
    P = S.pools[li]
    ptrs = tuple(int(t.data_ptr()) for t in (w13, w13_scales, w2, w2_scales))
    if not P.pool:  # attach the prepared storages as the pool tensors (first launch of this layer)
        R = P.R
        P.pool = dict(w13=w13.view(R, -1), w13s=w13_scales.view(R, -1), w2=w2.view(R, -1), w2s=w2_scales.view(R, -1))
        P.pool_ptrs = ptrs
    elif ptrs != P.pool_ptrs:
        # not this layer's pool (a launch outside apply that found a stale context): stock path, once reported
        if not getattr(P, "foreign_reported", False):
            P.foreign_reported = True
            _log(f"WARNING layer {li}: a compact launch with other weight storages than the layer's pool; left to b12x")
        S.ctx.layer = None
        return topk_ids
    if DEBUG:
        seen = getattr(P, "ptr_sets", None)
        if seen is None:
            seen = P.ptr_sets = {P.pool_ptrs: "attach"}
        if ptrs not in seen:
            import traceback
            seen[ptrs] = f"tokens {num_tokens} cap {values.shape[0]} capturing {torch.cuda.is_current_stream_capturing()} step {S.step}"
            frames = [f"{os.path.basename(fr.filename)}:{fr.lineno}:{fr.name}" for fr in traceback.extract_stack()[-14:-2]]
            _log(f"WARNING layer {li}: a new weight storage set #{len(seen)} ({seen[ptrs]}): w13 {hex(ptrs[0])} w2 {hex(ptrs[2])} "
                 f"shapes {tuple(w13.shape)} {tuple(w2.shape)}; known: {[(hex(k[0]), v) for k, v in seen.items()]}; stack {frames}")
    T, K = topk_ids.shape
    if num_tokens not in S.seen_compact_tokens:
        S.seen_compact_tokens.add(num_tokens)
        _log(f"compact path with {num_tokens} tokens (cap {values.shape[0]}), layer {li}")
    remapped = S.remapped[li][: T * K].view(T, K)
    S.tier.publish(li, topk_ids.contiguous(), topk_weights.contiguous(), P.slot_table, values, scale_rows, num_tokens, remapped)
    return remapped


def after_compiled(route: torch.Tensor) -> None:
    li = current_layer()
    S.ctx.layer = None  # one compact launch per apply; warmup launches outside apply stay stock
    if li is None or not S.created:
        if DEBUG_COMPACT and getattr(S.ctx, "want_ref_rows", False):
            S.ctx.ref_rows = route.clone()   # the reference run's rows (the stock path: no publish, no inject)
        return
    S.tier.wait_inject(li, route)
    if DEBUG_COMPACT and not torch.cuda.is_current_stream_capturing():
        S.ctx.hyb_rows = route.clone()      # the hybrid's rows: the GPU's for resident pairs, the host's injected
        S.ctx.compact_call = li


# ---------------------------------------------------------------------------------------------------------------
# called around the model runner's execute_model (every target forward: eager, piecewise or a graph replay)
# ---------------------------------------------------------------------------------------------------------------
def runner_before_step() -> None:
    if not S.created or torch.cuda.is_current_stream_capturing():
        return
    S.manager.before_launch()


def runner_after_step() -> None:
    if not S.created or torch.cuda.is_current_stream_capturing():
        return
    t_hook = time.perf_counter()
    S.step += 1
    err = S.tier.error()
    if err:
        tot = S.tier.totals()
        raise RuntimeError(f"[moet-cpu-tier] the CPU tier failed closed (error word {err}, totals {tot})")
    uses = S.tier.uses()
    S.manager.note_uses(uses)
    misses = S.tier.misses()
    t_rings = time.perf_counter()
    # diagnostics: promotions restricted to some ranks / hybrid layers (VLLM_MOET_CPU_TIER_PROMOTE_RANKS=0,1
    # VLLM_MOET_CPU_TIER_PROMOTE_LAYERS=a-b); a static placement elsewhere
    if _PROMOTE_RANKS is not None and _rank() not in _PROMOTE_RANKS:
        misses = misses[:0]
    if _PROMOTE_LAYERS is not None and len(misses):
        misses = misses[(misses[:, 1] >= _PROMOTE_LAYERS[0]) & (misses[:, 1] <= _PROMOTE_LAYERS[1])]
    n = S.manager.after_launch(misses, S.step)
    t_promo = time.perf_counter()
    S.kpi["pairs_all"] += sum(int((ids >= 0).sum()) for _, _, ids in uses)
    S.kpi["pairs_cpu"] += int(misses[:, 3].sum()) if len(misses) else 0
    S.kpi["promotions"] += n
    if DEBUG and n:
        _debug_check_promotions()
    # the hook's parts: the rings (uses / note_uses / misses), the promotion scheduling + its GPU enqueue, the rest
    S.kpi["hook_rings_s"] = S.kpi.get("hook_rings_s", 0.0) + (t_rings - t_hook)
    S.kpi["hook_promo_s"] = S.kpi.get("hook_promo_s", 0.0) + (t_promo - t_rings)
    S.kpi["hook_s"] = S.kpi.get("hook_s", 0.0) + (time.perf_counter() - t_hook)
    if LOG and (S.step % LOG_EVERY == 0 or S.step <= 3):
        tot = S.tier.totals()
        now = time.time()
        d_busy = tot["busy_ns"] - S.kpi["busy_ns"]
        d_steps = max(1, tot["steps"] - S.kpi["tier_steps"])
        hit = 1.0 - S.kpi["pairs_cpu"] / max(1, S.kpi["pairs_all"])
        # the KPI ring since the last line: per (step, layer) request the pairs, distinct experts and the host's
        # compute time (t_done - t_seen)
        detail = ""
        n_exp_interval = 0
        try:
            kp = S.tier.kpis(cap=4096)
            kp = kp[kp[:, 0] > S.kpi.get("last_kpi_step", -1)] if len(kp) else kp
            served = kp[kp[:, 2] > 0] if len(kp) else kp
            if len(served):
                n_exp_interval = int(served[:, 3].sum())
                comp = (served[:, 6] - served[:, 5]) / 1e3
                pe = served[:, 3]
                detail = (f"; per served layer: pairs {served[:, 2].mean():.1f} experts {pe.mean():.1f} compute "
                          f"{np.median(comp):.0f}/{np.percentile(comp, 90):.0f}/{comp.max():.0f} us (med/p90/max), "
                          f"{comp.sum() / max(1, pe.sum()):.0f} us per expert, {comp.sum() / max(1, served[:, 2].sum()):.0f} us per pair, "
                          f"{len(served)} of {len(kp)} requests had misses")
                S.kpi["last_kpi_step"] = int(kp[:, 0].max())
            ps = S.tier.pool_stats(reset=True)
            cyc = [ps["cyc_prepare"], ps["cyc_fc1"], ps["cyc_act"], ps["cyc_fc2"]]
            if sum(cyc) > 0:
                tot_c = sum(cyc)
                detail += (f"; worker-0 cycles: prepare {100 * cyc[0] / tot_c:.0f} % fc1 {100 * cyc[1] / tot_c:.0f} % act "
                           f"{100 * cyc[2] / tot_c:.0f} % fc2 {100 * cyc[3] / tot_c:.0f} % (barrier waits included), fc2 general "
                           f"path {ps['fc2_general']} of {ps['fc2_blocks']} blocks")
                if n_exp_interval:
                    # TSC cycles -> time: the in-kernel time of worker 0 per expert, against the request's compute time per
                    # expert above (the difference = outside the kernel: wake-up, dispatch, the job barrier)
                    detail += f"; worker-0 in-kernel {tot_c / _TSC_GHZ / 1e3 / n_exp_interval:.0f} us per expert"
        except Exception as ex:  # diagnostics only
            detail = f"; kpi detail failed: {ex!r}"
        _log(f"step {S.step}: tier steps {tot['steps']} pairs {tot['pairs']} errors {tot['errors']}; cpu busy "
             f"{d_busy / 1e6 / d_steps:.2f} ms per tier step; pair hit rate {hit:.3f} (all {S.kpi['pairs_all']}, cpu "
             f"{S.kpi['pairs_cpu']}); promotions {S.kpi['promotions']} ({S.manager.stats}); "
             f"{LOG_EVERY / max(1e-6, now - S.kpi['last_log_t']):.1f} steps/s; host hook {S.kpi['hook_s'] * 1e3 / LOG_EVERY:.2f} ms per step "
             f"(rings {S.kpi.get('hook_rings_s', 0.0) * 1e3 / LOG_EVERY:.2f}, promotions {S.kpi.get('hook_promo_s', 0.0) * 1e3 / LOG_EVERY:.2f}){detail}"
             f"{_debug_compact_summary()}")
        S.kpi["busy_ns"], S.kpi["tier_steps"], S.kpi["last_log_t"], S.kpi["hook_s"] = tot["busy_ns"], tot["steps"], now, 0.0
        S.kpi["hook_rings_s"] = S.kpi["hook_promo_s"] = 0.0


_LAYOUT_NP = {}


def _debug_check_promotions() -> None:
    """Debug: after promotions, the touched slots' bytes against a host-side gather of the records (the same
    permutation tables, numpy instead of the repack kernel) and the slot tables against the host mirror."""
    S.manager.copy.synchronize()
    torch.cuda.synchronize()
    if not _LAYOUT_NP:
        tables = os.environ["VLLM_MOET_CPU_TIER_TABLES"]
        for name in ("w13", "w13s", "w2", "w2s"):
            _LAYOUT_NP[name] = np.load(os.path.join(tables, f"b12x_layout_{name}.npy"))
    bad = 0
    checked = 0
    for li, experts in S.manager.last_promoted.items():
        P = S.pools[li]
        for e in experts:
            s_ = P.mapped(e)
            if s_ < 0:
                continue
            rec = _record_view(li, e).numpy()
            w1n, w1s = rec[0:1474560], rec[1474560:1566720]
            w3n, w3s = rec[1566720:3041280], rec[3041280:3133440]
            w2n, w2s = rec[3133440:4608000], rec[4608000:4700160]
            nib13 = np.stack((np.concatenate([w1n, w3n]) & 0xF, np.concatenate([w1n, w3n]) >> 4), axis=-1).reshape(-1)
            nib2 = np.stack((w2n & 0xF, w2n >> 4), axis=-1).reshape(-1)
            exp = {
                "w13": (nib13[_LAYOUT_NP["w13"]][0::2] | (nib13[_LAYOUT_NP["w13"]][1::2] << 4)).astype(np.uint8),
                "w2": (nib2[_LAYOUT_NP["w2"]][0::2] | (nib2[_LAYOUT_NP["w2"]][1::2] << 4)).astype(np.uint8),
                "w13s": np.concatenate([w1s, w3s])[_LAYOUT_NP["w13s"]],
                "w2s": w2s[_LAYOUT_NP["w2s"]],
            }
            for name in exp:
                got = P.pool[name][s_].view(torch.uint8).cpu().numpy()
                checked += 1
                d = int((got != exp[name]).sum())
                if d:
                    bad += 1
                    _log(f"DEBUG layer {li} expert {e} slot {s_} {name}: {d} of {got.size} bytes differ from the host gather")
        tab = P.slot_table.cpu().numpy()
        mirror = np.array([P.mapped(x) for x in range(P.E)], dtype=np.int32)
        if not np.array_equal(tab, mirror):
            bad += 1
            _log(f"DEBUG layer {li}: slot table {tab.tolist()} != mirror {mirror.tolist()}")
    _log(f"DEBUG step {S.step}: promotion check {checked} blocks, {bad} bad")
