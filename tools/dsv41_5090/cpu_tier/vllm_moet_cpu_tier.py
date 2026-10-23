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
    cfg = _LayerCfg(li, name, ml, E, _residents_of(ml, E), SPARE)
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
                                 max_promotions_per_layer=int(os.environ.get("VLLM_MOET_CPU_TIER_MAX_PROMOTIONS", "4")))
    S.tier.start(first_step=1)
    S.created = True

    def _shutdown():
        try:
            if LOG:
                tot = S.tier.totals()
                _log(f"exit: steps {S.step}, tier steps {tot['steps']} pairs {tot['pairs']} busy {tot['busy_ns'] / 1e9:.1f} s "
                     f"errors {tot['errors']}; pairs all {S.kpi['pairs_all']} cpu {S.kpi['pairs_cpu']} promotions {S.kpi['promotions']}")
            S.tier.stop()
            S.tier.close()  # frees the mapped mailboxes before the driver tears the process down
        except Exception:
            pass

    atexit.register(_shutdown)
    _log(f"tier started: {n_layers} layers x {S.E} experts, slots {[c.R for c in S.layers][:3]}..., resident "
         f"{[len(c.residents) for c in S.layers][:3]}..., records {n_rec}, pool CPUs {cpus[0]}-{cpus[-1]} ({len(cpus)} threads), "
         f"dispatcher {layout['dispatcher']}, {pin_msg}; layers {[c.name for c in S.layers]}")


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
        # not the compact path: no publish / inject; keep the kernels inside the R-row weights
        S.ctx.layer = None
        if T not in S.noncompact_reported:
            S.noncompact_reported.add(T)
            _log(f"WARNING: a {T}-token MoE call takes b12x's non-compact path: missing experts are NOT computed "
                 f"there yet (phase 2.5); ids remapped through the slot table for memory safety")
        table = S.pools[li].slot_table
        return torch.where(topk_ids >= 0, table[topk_ids.clamp(min=0).long()], topk_ids.new_full((), -1))
    return topk_ids


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
        return
    S.tier.wait_inject(li, route)


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
    # diagnostics: promotions restricted to some ranks / hybrid layers (VLLM_MOET_CPU_TIER_PROMOTE_RANKS=0,1
    # VLLM_MOET_CPU_TIER_PROMOTE_LAYERS=a-b); a static placement elsewhere
    if _PROMOTE_RANKS is not None and _rank() not in _PROMOTE_RANKS:
        misses = misses[:0]
    if _PROMOTE_LAYERS is not None and len(misses):
        misses = misses[(misses[:, 1] >= _PROMOTE_LAYERS[0]) & (misses[:, 1] <= _PROMOTE_LAYERS[1])]
    n = S.manager.after_launch(misses, S.step)
    S.kpi["pairs_all"] += sum(int((ids >= 0).sum()) for _, _, ids in uses)
    S.kpi["pairs_cpu"] += int(misses[:, 3].sum()) if len(misses) else 0
    S.kpi["promotions"] += n
    if DEBUG and n:
        _debug_check_promotions()
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
        try:
            kp = S.tier.kpis(cap=4096)
            kp = kp[kp[:, 0] > S.kpi.get("last_kpi_step", -1)] if len(kp) else kp
            served = kp[kp[:, 2] > 0] if len(kp) else kp
            if len(served):
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
        except Exception as ex:  # diagnostics only
            detail = f"; kpi detail failed: {ex!r}"
        _log(f"step {S.step}: tier steps {tot['steps']} pairs {tot['pairs']} errors {tot['errors']}; cpu busy "
             f"{d_busy / 1e6 / d_steps:.2f} ms per tier step; pair hit rate {hit:.3f} (all {S.kpi['pairs_all']}, cpu "
             f"{S.kpi['pairs_cpu']}); promotions {S.kpi['promotions']} ({S.manager.stats}); "
             f"{LOG_EVERY / max(1e-6, now - S.kpi['last_log_t']):.1f} steps/s; host hook {S.kpi['hook_s'] * 1e3 / LOG_EVERY:.2f} ms per step{detail}")
        S.kpi["busy_ns"], S.kpi["tier_steps"], S.kpi["last_log_t"], S.kpi["hook_s"] = tot["busy_ns"], tot["steps"], now, 0.0


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
