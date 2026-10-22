"""Runtime of the CPU expert tier inside vLLM (phase 2.3): what the patched b12x.py / w4a8_compact_micro.py /
gpu_model_runner.py call. Installed next to them by patch_vllm_moe_cpu_tier.py together with cpu_tier.py,
placement_manager.py, cpu_tier_cuda.cu, cpu_tier_abi.h and libcpu_tier.so. Everything is a no-op unless
VLLM_MOET_MOE_CPU_TIER=1.

Environment:
  VLLM_MOET_MOE_CPU_TIER=1              on
  VLLM_MOET_CPU_TIER_RESIDENT=<int>     resident experts per layer (default: all = a no-op hybrid); or
  VLLM_MOET_CPU_TIER_PLACEMENT=<json>   routing/placement.py's file (per-layer resident lists)
  VLLM_MOET_CPU_TIER_TABLES=<dir>       b12x_act_m_limit10.f32, b12x_ue8m0_inv.f32, b12x_layout_*.npy
  VLLM_MOET_CPU_TIER_CPUS=<a-b>         the rank's CPUs (default: 8*rank .. 8*rank+7 = one CCD); dispatcher = first + 32
  VLLM_MOET_CPU_TIER_STORE=<dir>        build_store.py's files for the real checkpoint (not set: the records are
                                        dumped from the loaded GPU weights - the dummy / mini-model mode)
  VLLM_MOET_CPU_TIER_DUMMY_WEIGHTS=1    overwrite the (dummy) expert weights with seeded sane random MXFP4 bytes
                                        before anything reads them (scales 118..126), so that a stock run and a
                                        hybrid run of --load-format dummy see the same experts
  VLLM_MOET_CPU_TIER_TIMEOUT_MS         bounded wait (default 500)
  VLLM_MOET_CPU_TIER_MAX_PROMOTIONS     per layer per step (default 4; 0 = static placement only)
  VLLM_MOET_CPU_TIER_LOG=1              one line per step with the KPIs
"""
from __future__ import annotations

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
RECORD_BYTES = 4700160
HID, N_R, TOPK_MAX = 5120, 576, 8


def _log(msg: str) -> None:
    print(f"[moet-cpu-tier] {msg}", flush=True)


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.tier = None
        self.manager = None
        self.pools: list = []
        self.layer_index: dict[int, int] = {}      # id(layer module) -> hybrid layer index
        self.layer_names: list[str] = []
        self.E = 0
        self.residents: list[list[int]] = []
        self.remapped: list[torch.Tensor] = []     # per layer, [MAX_PAIRS] int32 (sliced per call)
        self.store_ptrs: list[int] = []            # per layer the arena base (dummy mode) or None
        self.store = None
        self.first_layer = None
        self.step = 0
        self.in_forward_first_seen = False
        self.ctx = threading.local()
        self.n_hybrid_layers = int(os.environ.get("VLLM_MOET_CPU_TIER_LAYERS", "0"))  # 0 = count at registration time
        self.pending_layers: list = []
        self.created = False
        self.seen_apply_tokens: set = set()      # diagnostics: token counts seen by apply / by the compact path
        self.seen_compact_tokens: set = set()
        self.apply_counts: dict = {}             # per step: hybrid layer -> applies (a replay would double them)
        self.impl_keys: dict = {}                # debug: per layer the (impl, prepared, storage) identities seen
        self.dup_reports = 0


S = _State()


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def _cpus() -> list[int]:
    spec = os.environ.get("VLLM_MOET_CPU_TIER_CPUS")
    if spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    r = _rank()
    return list(range(8 * r, 8 * r + 8))


def is_target_layer(layer) -> bool:
    """A target-model MoE layer (not the DSpark drafter's: those carry layer indices past num_hidden_layers)."""
    name = getattr(layer, "layer_name", "") or ""
    m = re.search(r"layers\.(\d+)\.", name)
    if m is None:
        return False
    idx = int(m.group(1))
    limit = int(os.environ.get("VLLM_MOET_CPU_TIER_TARGET_LAYERS", "0"))
    if limit == 0:
        try:
            from vllm.config import get_current_vllm_config
            limit = int(get_current_vllm_config().model_config.hf_config.num_hidden_layers)
        except Exception:
            limit = 10 ** 9
    return idx < limit


def _seed_dummy_weights(layer_idx: int, w13: torch.Tensor, w13s: torch.Tensor, w2: torch.Tensor, w2s: torch.Tensor) -> None:
    g = torch.Generator(device=w13.device).manual_seed(1000 + layer_idx)
    for packed in (w13, w2):
        v = torch.randint(0, 256, packed.shape, dtype=torch.int32, device=packed.device, generator=g)
        mag = v & 0x77
        nonzero = (mag | (mag >> 1) | (mag >> 2)) & 0x11
        packed.copy_((v & (0x77 | (nonzero << 3))).to(torch.uint8))
    for sc in (w13s, w2s):
        sc.copy_(torch.randint(118, 127, sc.shape, dtype=torch.int32, device=sc.device, generator=g).to(torch.uint8))


def _create_tier_if_needed(device: torch.device):
    """Deferred until the number of hybrid layers is known (after the last target layer registered)."""
    if S.created:
        return
    from cpu_tier import CpuTier
    from placement_manager import PlacementManager, load_layout

    n_layers = len(S.layer_names)
    tables = os.environ["VLLM_MOET_CPU_TIER_TABLES"]
    cpus = _cpus()
    S.tier = CpuTier(n_layers, S.E, cpus, cpus[0] + 32, device, tables,
                     timeout_ms=float(os.environ.get("VLLM_MOET_CPU_TIER_TIMEOUT_MS", "500")))
    for li in range(n_layers):
        base = S.store_ptrs[li]
        for e in range(S.E):
            S.tier.set_record(li, e, base + e * RECORD_BYTES)
    layout = load_layout(tables, device)
    copy_stream = torch.cuda.Stream(device=device)
    S.manager = PlacementManager(S.tier.ext, layout, S.pools,
                                 lambda li, e: S.tier.ext.host_view(S.store_ptrs[li] + e * RECORD_BYTES, RECORD_BYTES),
                                 copy_stream, max_promotions_per_layer=int(os.environ.get("VLLM_MOET_CPU_TIER_MAX_PROMOTIONS", "4")))
    S.tier.start(first_step=1)
    S.created = True
    import atexit

    def _shutdown():
        try:
            S.tier.stop()
            S.tier.close()  # frees the mapped mailboxes before the driver tears the process down
        except Exception:
            pass

    atexit.register(_shutdown)
    _log(f"rank {_rank()}: tier started: {n_layers} layers x {S.E} experts, resident {[len(r) for r in S.residents]}, "
         f"CPUs {cpus[0]}-{cpus[-1]} (+dispatcher {cpus[0] + 32}), layers {S.layer_names}")


# ---------------------------------------------------------------------------------------------------------------
# called from vLLM's B12xExperts.process_weights_after_loading, BEFORE _prepare_experts
# ---------------------------------------------------------------------------------------------------------------
def on_process_weights(experts_impl, layer, w13: torch.Tensor, w2: torch.Tensor, w13_scale: torch.Tensor,
                       w2_scale: torch.Tensor):
    """Returns None (stock path) or (w13_sub, w2_sub, w13_scale_sub, w2_scale_sub): the resident subset to prepare."""
    if not ENABLED or not is_target_layer(layer):
        return None
    if w13.dim() != 3 or w13.shape[1] != 2 * N_R or w13.shape[2] != HID // 2 or w2.shape[1] != HID:
        _log(f"layer {getattr(layer, 'layer_name', '?')}: geometry {tuple(w13.shape)} / {tuple(w2.shape)} is not the "
             f"TP4-slice geometry the CPU kernel computes; left on the GPU")
        return None
    from placement_manager import ResidentPool
    from cpu_tier import cuda_ext

    li = len(S.layer_names)
    S.layer_names.append(getattr(layer, "layer_name", f"layer{li}"))
    S.layer_index[id(layer)] = li
    E = int(w13.shape[0])
    S.E = E
    if os.environ.get("VLLM_MOET_CPU_TIER_DUMMY_WEIGHTS", "0") == "1":
        _seed_dummy_weights(li, w13, w13_scale, w2, w2_scale)
    # canonicalize -0 like vLLM does before b12x (so the records equal what prepare_weights sees)
    for packed in (w13, w2):
        mag = packed & 0x77
        nonzero = (mag | (mag >> 1) | (mag >> 2)) & 0x11
        packed.bitwise_and_(0x77 | (nonzero << 3))
    # the residents
    placement = os.environ.get("VLLM_MOET_CPU_TIER_PLACEMENT")
    if placement:
        import json
        P = json.load(open(placement))
        m = re.search(r"layers\.(\d+)\.", S.layer_names[-1])
        model_layer = int(m.group(1)) if m else li
        residents = [e for e in P["layers"][model_layer]["resident"] if e < E]
    else:
        R = int(os.environ.get("VLLM_MOET_CPU_TIER_RESIDENT", str(E)))
        residents = list(range(min(R, E)))
    S.residents.append(residents)
    # the records: from the loaded GPU weights (dummy mode) into a pinned arena of this layer
    ext = cuda_ext()
    store = os.environ.get("VLLM_MOET_CPU_TIER_STORE")
    if store:
        raise NotImplementedError("VLLM_MOET_CPU_TIER_STORE (the real checkpoint's store + loader skip) is 2.4")
    base = int(ext.alloc_pinned(E * RECORD_BYTES))
    for e in range(E):
        rec = torch.cat([w13[e, :N_R].reshape(-1), w13_scale[e, :N_R].reshape(-1), w13[e, N_R:].reshape(-1),
                         w13_scale[e, N_R:].reshape(-1), w2[e].reshape(-1), w2_scale[e].reshape(-1)])
        assert rec.numel() == RECORD_BYTES, rec.numel()
        ext.host_view(base + e * RECORD_BYTES, RECORD_BYTES).copy_(rec.cpu())
    S.store_ptrs.append(base)
    # the slot table and the pool shell (its tensors are the prepared storages, attached at the first launch)
    slot_table = torch.full((E,), -1, dtype=torch.int32, device=w13.device)
    pool = ResidentPool(li, E, len(residents), w13.device, pool={}, slot_table=slot_table)
    for s, e in enumerate(residents):
        slot_table[e] = s
        pool.expert_of_slot[s] = e
        pool.slot_of_expert[e] = s
        pool.last_used[s] = 0
    S.pools.append(pool)
    S.remapped.append(torch.full((TOPK_MAX * 16,), -1, dtype=torch.int32, device=w13.device))
    idx = torch.tensor(residents, dtype=torch.long, device=w13.device)
    sub = (w13.index_select(0, idx).contiguous(), w2.index_select(0, idx).contiguous(),
           w13_scale.index_select(0, idx).contiguous(), w2_scale.index_select(0, idx).contiguous())
    _log(f"layer {S.layer_names[-1]} -> hybrid layer {li}: {len(residents)} of {E} experts resident, records in pinned RAM")
    layer._moet_cpu_tier_layer = li
    experts_impl._moet_cpu_tier_layer = li
    return sub


def on_layer_ready(layer, experts_impl) -> None:
    """After process_weights_after_loading: the first forward may follow; create the tier once all target layers
    registered (we cannot know the count up front, so creation is lazy: at the first apply)."""
    return None


# ---------------------------------------------------------------------------------------------------------------
# called from B12xExperts.apply (before the b12x run) and from the compact launch
# ---------------------------------------------------------------------------------------------------------------
def on_apply(experts_impl, hidden_states: torch.Tensor) -> None:
    # the layer context lives from this apply to its compact launch only; an apply that takes another b12x path
    # (prefill-sized calls) must not leave it behind for launches made outside apply (vLLM's b12x warmup units)
    li = getattr(experts_impl, "_moet_cpu_tier_layer", None)
    S.ctx.layer = li
    if li is None:
        return
    if not S.created:
        with S.lock:
            _create_tier_if_needed(hidden_states.device)
    T = int(hidden_states.shape[0])
    if DEBUG:
        prep = getattr(experts_impl, "_prepared_experts", None)
        key = (id(experts_impl), id(prep), int(getattr(prep, "w13_rp", torch.empty(0)).data_ptr()) if prep is not None else 0)
        known = S.impl_keys.setdefault(li, {})
        if key not in known:
            known[key] = f"T {T} step {S.step} capturing {torch.cuda.is_current_stream_capturing()} layer_name {getattr(experts_impl, 'layer_name', '?')}"
            if len(known) > 1:
                _log(f"WARNING layer {li}: apply from a new (experts_impl, prepared, w13_rp) {key} [{known[key]}]; known {known}")
    if T not in S.seen_apply_tokens:
        S.seen_apply_tokens.add(T)
        _log(f"apply with {T} tokens (compact-path token counts so far: {sorted(S.seen_compact_tokens)})")
    if li == 0:
        # a new target forward = a new step: the placement manager's hooks around the previous step, then the
        # step counter (runner-independent: vLLM's V1 and V2 model runners both end up here). Not while a CUDA
        # graph is being captured: event waits there would be captured as dependencies on uncaptured work.
        capturing = torch.cuda.is_current_stream_capturing()
        if S.apply_counts and not capturing:
            dup = {l: c for l, c in S.apply_counts.items() if c > 1}
            if dup and S.dup_reports < 5:
                S.dup_reports += 1
                _log(f"WARNING step {S.step}: hybrid layers applied more than once in one step: {dup} (a decoder "
                     f"replay inside the step publishes a layer twice: not supported yet)")
            _after_step()
        S.apply_counts = {}
        if not capturing:
            _before_step()
        S.tier.step_begin()
    S.apply_counts[li] = S.apply_counts.get(li, 0) + 1


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
# called around the model runner's step
# ---------------------------------------------------------------------------------------------------------------
def runner_before_step() -> None:  # kept for the (inert) runner patch; the work happens in on_apply
    return None


def runner_after_step() -> None:
    return None


def _before_step() -> None:
    if not S.created:
        return
    S.manager.before_launch()


def _after_step() -> None:
    if not S.created:
        return
    S.step += 1
    err = S.tier.error()
    if err:
        tot = S.tier.totals()
        raise RuntimeError(f"[moet-cpu-tier] the CPU tier failed closed (error word {err}, totals {tot})")
    uses = S.tier.uses()
    S.manager.note_uses(uses)
    misses = S.tier.misses()
    n = S.manager.after_launch(misses, S.step)
    if DEBUG and n:
        _debug_check_promotions()
    if os.environ.get("VLLM_MOET_CPU_TIER_LOG", "0") == "1" and (S.step % 50 == 0):
        tot = S.tier.totals()
        _log(f"step {S.step}: tier steps {tot['steps']} pairs {tot['pairs']} busy {tot['busy_ns'] / 1e6:.1f} ms "
             f"errors {tot['errors']}; misses this call {len(misses)} promotions {n} ({S.manager.stats})")


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
            rec = S.tier.ext.host_view(S.store_ptrs[li] + e * RECORD_BYTES, RECORD_BYTES).numpy()
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
