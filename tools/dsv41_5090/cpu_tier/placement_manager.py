"""Resident expert pools and promote-on-miss (phase 2.2 / 2.4), per TP rank.

A layer's resident pool = b12x's prepared tensors for R slots (w13 [R, 737280] int32, w13 scales [R, 46080], w2
[R, 368640], w2 scales [R, 23040]) + a device slot table int32 [E] (-1 = not resident) that the tier's publish op
reads in-graph. A promotion moves one expert from the pinned store into a slot on a copy stream, stream-ordered
against the captured steps without recapture (rule R4):

    after step n is launched:   copy stream waits for step n -> slot[victims] = -1 (all layers)  -> event U_n
                                -> per layer: H2D of the records -> repack (b12x_layout_probe's permutations)
                                   -> slot[e] = s                                               -> event M_n
    before step n+1 launches:   main stream waits U_n            (step n+1 never reads a slot being rewritten)
    before step n+2 launches:   main stream waits M_n            (step n+2 sees the new experts)

Victims: the least recently routed slot (the tier's use log carries every step's routing, hits included), never one
of the step's own misses, never an expert without a record in the tier's store (the CPU would have to compute it
afterwards), never one hotter than the miss (a miss must not evict something more recent than itself: with more
misses per step than spare slots that churned the hot set). Host state in numpy; the GPU side of a layer's
promotions is one extension call (`promote_batch`: copies + repacks + table writes on the current stream).
"""
from __future__ import annotations

import os

import numpy as np
import torch

RECORD_BYTES = 4700160
WORDS = dict(w13=737280, w13s=46080, w2=368640, w2s=23040)
_KEY_USES = 1 << 20   # the LRU key = last_used * 2^20 + min(use_count, 2^20 - 1)


def load_layout(tables_dir: str, device: torch.device) -> dict:
    return {name: torch.from_numpy(np.load(os.path.join(tables_dir, f"b12x_layout_{name}.npy"))).to(device) for name in WORDS}


class ResidentPool:
    def __init__(self, layer: int, n_experts: int, R: int, device: torch.device, pool: dict | None = None,
                 slot_table: torch.Tensor | None = None, record_ok=None):
        self.layer, self.E, self.R, self.device = layer, n_experts, R, device
        # pool=None: own tensors; pool={}: attached later (vLLM's prepared storages, at the layer's first launch)
        self.pool = pool if pool is not None else {name: torch.zeros(R, w, dtype=torch.int32, device=device) for name, w in WORDS.items()}
        self.slot_table = slot_table if slot_table is not None else torch.full((n_experts,), -1, dtype=torch.int32, device=device)
        self.expert_of_slot = np.full(R, -1, dtype=np.int32)        # host mirror of the table
        self.slot_of_expert = np.full(n_experts, -1, dtype=np.int32)
        self.last_used = np.full(R, -1, dtype=np.int64)              # step of the last routing to the slot's expert
        self.use_count = np.zeros(R, dtype=np.int64)                 # routings since the slot was filled (tie-breaker)
        # record_ok(e) -> the CPU tier holds e's record (an expert without one can never be evicted: after the
        # eviction the CPU would have to compute it). None = every expert has a record.
        self.evictable = np.ones(n_experts, dtype=bool) if record_ok is None else np.array([bool(record_ok(e)) for e in range(n_experts)])
        self.staging = torch.empty(RECORD_BYTES, dtype=torch.uint8, device=device)

    def mapped(self, e: int) -> int:
        return int(self.slot_of_expert[int(e)])

    def resident_experts(self) -> set:
        return set(int(e) for e in self.expert_of_slot[self.expert_of_slot >= 0])

    def set_slot(self, s: int, e: int, step: int) -> None:
        """Host mirror: slot s holds e (e < 0 = empty)."""
        old = int(self.expert_of_slot[s])
        if old >= 0:
            self.slot_of_expert[old] = -1
        self.expert_of_slot[s] = e
        if e >= 0:
            self.slot_of_expert[e] = s
            self.last_used[s] = step
            self.use_count[s] = 1 if step > 0 else 0

    def note_use(self, experts: np.ndarray, step: int) -> None:
        ids = np.asarray(experts, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < self.E)]
        s = self.slot_of_expert[ids]
        s = s[s >= 0]
        if len(s):
            self.last_used[s] = np.maximum(self.last_used[s], step)
            np.add.at(self.use_count, s, 1)

    def pick_victim(self, protect: set, candidate_key: tuple) -> int:
        """An empty slot, else the least recently used evictable slot (ties: the least used since it was filled)
        whose expert is not protected AND whose key is older than the candidate's; -1 if none."""
        empty = np.nonzero(self.expert_of_slot < 0)[0]
        if len(empty):
            return int(empty[0])
        e = self.expert_of_slot
        ok = self.evictable[e]
        if protect:
            ok &= ~np.isin(e, np.fromiter(protect, dtype=np.int64, count=len(protect)))
        if not ok.any():
            return -1
        key = self.last_used * _KEY_USES + np.minimum(self.use_count, _KEY_USES - 1)
        key = np.where(ok, key, np.iinfo(np.int64).max)
        s = int(np.argmin(key))
        cand = int(candidate_key[0]) * _KEY_USES + min(int(candidate_key[1]), _KEY_USES - 1)
        return s if key[s] < cand else -1


class PlacementManager:
    def __init__(self, ext, layout: dict, pools: list[ResidentPool], records_of, copy_stream: torch.cuda.Stream,
                 max_promotions_per_layer: int = 4, record_ptr=None):
        """records_of(layer, expert) -> a pinned host uint8 tensor [RECORD_BYTES] (the store); record_ptr(layer,
        expert) -> its host address (for the batched promotion; derived from records_of when None)."""
        self.ext, self.layout, self.pools, self.records_of = ext, layout, pools, records_of
        self.record_ptr = record_ptr or (lambda layer, e: int(records_of(layer, e).data_ptr()))
        self.copy = copy_stream
        self.max_per_layer = max_promotions_per_layer
        self.unmap_events: list[torch.cuda.Event] = []     # U of the last after_launch
        self.map_events_last: list[torch.cuda.Event] = []  # M of the last after_launch
        self.map_events_prev: list[torch.cuda.Event] = []  # M of the one before
        self.last_promoted: dict[int, set] = {}            # layer -> experts promoted by the last after_launch
        self.stats = dict(promotions=0, skipped_no_victim=0, skipped_already=0)
        self.batched = hasattr(ext, "promote_batch")

    def _repack_into(self, P: ResidentPool, layer: int, e: int, s: int) -> None:
        P.staging.copy_(self.records_of(layer, e), non_blocking=True)
        self.ext.repack_record(P.staging, self.layout["w13"], self.layout["w13s"], self.layout["w2"], self.layout["w2s"],
                               P.pool["w13"], P.pool["w13s"], P.pool["w2"], P.pool["w2s"], s)

    def fill_initial(self, layer: int, residents: list[int], stream: torch.cuda.Stream | None = None) -> None:
        """Static placement at startup: residents in slot order (synchronous)."""
        P = self.pools[layer]
        assert len(residents) <= P.R
        st = stream or torch.cuda.current_stream()
        with torch.cuda.stream(st):
            for s, e in enumerate(residents):
                self._repack_into(P, layer, e, s)
                P.slot_table[e] = s
                P.set_slot(s, e, 0)
        st.synchronize()

    def before_launch(self, main_stream: torch.cuda.Stream | None = None) -> None:
        """Order the next captured step after the last promotions' unmaps and the previous ones' maps."""
        ms = main_stream or torch.cuda.current_stream()
        for ev in self.map_events_prev + self.unmap_events:
            ms.wait_event(ev)
        self.map_events_prev = self.map_events_last
        self.map_events_last = []
        self.unmap_events = []

    def note_uses(self, uses) -> None:
        """uses: iterable of (step, layer, ids array) from the tier's use log (hits and misses alike)."""
        for step, layer, ids in uses:
            self.pools[layer].note_use(ids, int(step))

    def plan(self, misses: np.ndarray) -> dict[int, list[tuple[int, int, int, int]]]:
        """Host side: per layer the promotions (expert, slot, victim, miss_step) this call makes, mirrors updated."""
        plans: dict[int, list[tuple[int, int, int, int]]] = {}
        if len(misses) == 0:
            return plans
        layers = misses[:, 1]
        for layer in np.unique(layers):
            rows = misses[layers == layer]
            P = self.pools[int(layer)]
            # distinct experts in miss order with their latest miss step
            experts, first = np.unique(rows[:, 2], return_index=True)
            order = np.argsort(first)
            experts = experts[order]
            miss_step = {int(e): int(rows[rows[:, 2] == e, 0].max()) for e in experts}
            protect = set(int(e) for e in experts)
            out = []
            for e in experts:
                e = int(e)
                if len(out) >= self.max_per_layer:
                    break
                if P.mapped(e) >= 0:
                    self.stats["skipped_already"] += 1
                    continue
                s = P.pick_victim(protect, (miss_step[e], 1))
                if s < 0:
                    self.stats["skipped_no_victim"] += 1
                    continue
                victim = int(P.expert_of_slot[s])
                P.set_slot(s, e, miss_step[e])
                out.append((e, s, victim, miss_step[e]))
                self.stats["promotions"] += 1
            if out:
                plans[int(layer)] = out
        return plans

    def after_launch(self, misses: np.ndarray, step: int, main_stream: torch.cuda.Stream | None = None) -> int:
        """misses: [n, 4] (step, layer, expert, m) of the step just launched; schedules promotions on the copy
        stream behind that step. Returns the number scheduled."""
        self.last_promoted = {}
        plans = self.plan(misses)
        if not plans:
            return 0
        ms = main_stream or torch.cuda.current_stream()
        done = torch.cuda.Event()
        done.record(ms)
        self.copy.wait_event(done)
        n_sched = 0
        with torch.cuda.stream(self.copy):
            # phase 1: every victim of every layer unmapped, then U
            for layer, out in plans.items():
                P = self.pools[layer]
                victims = [v for _, _, v, _ in out if v >= 0]
                if victims:
                    if self.batched:
                        self.ext.table_write(P.slot_table, victims, [-1] * len(victims))
                    else:
                        for v in victims:
                            P.slot_table[v] = -1
            u = torch.cuda.Event()
            u.record(self.copy)
            self.unmap_events.append(u)
            # phase 2: the records into their slots, the table entries after each repack, then M
            for layer, out in plans.items():
                P = self.pools[layer]
                if self.batched:
                    self.ext.promote_batch(P.staging, self.layout["w13"], self.layout["w13s"], self.layout["w2"], self.layout["w2s"],
                                           P.pool["w13"], P.pool["w13s"], P.pool["w2"], P.pool["w2s"], P.slot_table,
                                           [self.record_ptr(layer, e) for e, _, _, _ in out], [e for e, _, _, _ in out],
                                           [s for _, s, _, _ in out])
                else:
                    for e, s, _, _ in out:
                        self._repack_into(P, layer, e, s)
                        P.slot_table[e] = s
                self.last_promoted[layer] = set(e for e, _, _, _ in out)
                n_sched += len(out)
            m = torch.cuda.Event()
            m.record(self.copy)
            self.map_events_last.append(m)
            if os.environ.get("VLLM_MOET_CPU_TIER_PROMOTE_SYNC", "0") == "1":
                # diagnostics: the next step waits for the maps too (no map lands while a step runs)
                self.unmap_events.append(m)
        return n_sched
