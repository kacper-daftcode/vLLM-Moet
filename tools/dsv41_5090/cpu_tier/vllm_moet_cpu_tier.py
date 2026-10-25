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
  VLLM_MOET_CPU_TIER_STORE=<dir>        build_store.py's files for the real checkpoint (see above)
  VLLM_MOET_CPU_TIER_STORE_MODE         nonresident (default) | all: which records to pin
  VLLM_MOET_CPU_TIER_DUMMY_WEIGHTS=1    (dummy mode) overwrite the dummy expert weights with seeded sane random MXFP4
                                        bytes (scales 118..126), so that a stock run and a hybrid run see the same experts
  VLLM_MOET_CPU_TIER_TIMEOUT_MS         bounded wait (default 500)
  VLLM_MOET_CPU_TIER_MAX_PROMOTIONS     per layer per step (default 4; 0 = static placement only)
  VLLM_MOET_CPU_TIER_LOG=1              KPI lines (every VLLM_MOET_CPU_TIER_LOG_EVERY steps, default 100)
  VLLM_MOET_CPU_TIER_SCRATCH=<int>      slots of the prefill scratch pool (default 0 = off: the missing experts of
                                        non-compact calls are not computed - decode-only serving; 128 = 1.2 GB per rank)
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


def _cpu_layout() -> dict:
    """The rank's CPUs. Default on the 32-core / 4-CCD TR PRO (SMT siblings = core + 32): per rank r its CCD
    8r..8r+7: core 8r = this process's main thread (vLLM's launch loop), 8r+1 = the tier's dispatcher (spins), 8r+2..8r+7
    = the kernel pool (6 threads); the siblings 8r+32 / 8r+33 take the process's other threads; the pool cores' siblings
    stay idle so that the AVX-512 kernel does not share its cores. Overrides: VLLM_MOET_CPU_TIER_CPUS=a-b (pool),
    VLLM_MOET_CPU_TIER_DISPATCHER_CPU, VLLM_MOET_CPU_TIER_MAIN_CPU, VLLM_MOET_CPU_TIER_OTHER_CPUS=a,b,.."""
    r = _rank()
    spec = os.environ.get("VLLM_MOET_CPU_TIER_CPUS")
    if spec:
        a, b = spec.split("-")
        pool = list(range(int(a), int(b) + 1))
        disp = int(os.environ.get("VLLM_MOET_CPU_TIER_DISPATCHER_CPU", pool[0] + 32))
    else:
        pool = list(range(8 * r + 2, 8 * r + 8))
        disp = int(os.environ.get("VLLM_MOET_CPU_TIER_DISPATCHER_CPU", 8 * r + 1))
    main = os.environ.get("VLLM_MOET_CPU_TIER_MAIN_CPU")
    main = int(main) if main else (8 * r if not spec else None)
    other = os.environ.get("VLLM_MOET_CPU_TIER_OTHER_CPUS")
    other = [int(x) for x in other.split(",")] if other else ([8 * r + 32, 8 * r + 33] if not spec else None)
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
def _ensure_store() -> None:
    if S.store is not None:
        return
    from cpu_tier import ExpertStore

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
    _log(f"store {STORE} ({STORE_MODE}): {S.store.n_records} records = {S.store.nbytes / 1e9:.1f} GB pinned, read "
         f"{S.store.load_seconds:.1f} s, alloc+read {time.perf_counter() - t0:.1f} s")


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
                         f"{S.scratch.rounds} rounds{S.scratch.timing_summary()}")
            S.tier.stop()
            S.tier.close()  # frees the mapped mailboxes before the driver tears the process down
        except Exception:
            pass

    atexit.register(_shutdown)
    _log(f"tier started: {n_layers} layers x {S.E} experts, slots {[c.R for c in S.layers][:3]}..., resident "
         f"{[len(c.residents) for c in S.layers][:3]}..., records {n_rec}, pool CPUs {cpus[0]}-{cpus[-1]} ({len(cpus)} threads), "
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
        if T not in S.noncompact_reported:
            S.noncompact_reported.add(T)
            _log(f"{T}-token MoE call: b12x's non-compact path; missing experts "
                 f"{'streamed through the scratch pool' if SCRATCH_SLOTS else 'NOT computed (VLLM_MOET_CPU_TIER_SCRATCH=0)'}")
        table = S.pools[li].slot_table
        return torch.where(topk_ids >= 0, table[topk_ids.clamp(min=0).long()], topk_ids.new_full((), -1))
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
        self.bufs: dict = {}
        self.impl = experts_impl
        self.activation = layer.activation
        self.rounds = 0
        self.experts_streamed = 0
        self.calls = 0
        self.pending: list = []        # (ev0, ev1, T, missing) of calls whose events may not have completed yet
        self.timed: dict = {}          # T -> [calls, ms, missing experts] of completed calls

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
            self.bufs[key] = (torch.empty(nbytes_fn(plan), dtype=torch.uint8, device=self.prepared.w1_fp4.device),
                              torch.empty(T, HID, dtype=self.impl.moe_config.in_dtype, device=self.prepared.w1_fp4.device))
        return self.plans[key], self.bufs[key]


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
    P = S.pools[li]
    T, K = ids.shape
    with S.lock:
        if S.scratch is None:
            S.scratch = _Scratch(experts_impl, S.layer_modules[li], hidden_states.device)
            _log(f"scratch pool: {SCRATCH_SLOTS} slots ({SCRATCH_SLOTS * RECORD_BYTES / 2**20:.0f} MiB staging + pool)")
    sc = S.scratch
    valid = ids >= 0
    missing_mask = valid & (P.slot_table[ids.clamp(min=0).long()] < 0)
    missing = torch.unique(ids[missing_mask]).tolist()
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
    for r0 in range(0, len(missing), sc.n):
        rnd = missing[r0:r0 + sc.n]
        ptrs = [_record_ptr(li, e) for e in rnd]
        _stage_and_repack(sc, ptrs)
        sc.table.fill_(-1)
        rnd_t = torch.tensor(rnd, dtype=torch.long, device=ids.device)
        sc.table[rnd_t] = torch.arange(len(rnd), dtype=torch.int32, device=ids.device)
        rids = torch.where(valid, sc.table[ids.clamp(min=0).long()], ids.new_full((), -1)).contiguous()
        run_plan(plan=plan, scratch=buf, hidden_states=hidden_states, prepared=sc.prepared, topk_weights=weights,
                 topk_ids=rids, output=tmp, unit_scale_contract=False)
        output.add_(tmp)
        sc.rounds += 1
        sc.experts_streamed += len(rnd)
    ev1.record()
    sc.pending.append((ev0, ev1, T, len(missing)))
    if sc.calls <= 3 or sc.calls % 200 == 0:
        _log(f"prefill call {sc.calls} (layer {li}, {T} tokens): {len(missing)} missing experts in "
             f"{(len(missing) + sc.n - 1) // sc.n} rounds; totals {sc.experts_streamed} experts / {sc.rounds} rounds"
             f"{sc.timing_summary()}")
    if DEBUG_PREFILL:
        _debug_prefill_reference(sc, li, ids, valid, weights, hidden_states, output, run_plan, plan, buf, T)


def _debug_prefill_reference(sc, li, ids, valid, weights, hidden_states, output, run_plan, plan, buf, T) -> None:
    """Debug (VLLM_MOET_CPU_TIER_DEBUG_PREFILL=1, needs SCRATCH >= the call's distinct experts): the same call with
    EVERY routed expert in the scratch pool in one b12x run = the all-resident reference for this layer's output;
    run twice to see the kernel's own run-to-run noise; the hybrid output (resident run + streamed rounds) against it."""
    routed = torch.unique(ids[valid]).tolist()
    if len(routed) > sc.n:
        _log(f"DEBUG prefill layer {li}: {len(routed)} routed experts > {sc.n} scratch slots; reference skipped")
        return
    _stage_and_repack(sc, [_record_ptr(li, e) for e in routed])
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
