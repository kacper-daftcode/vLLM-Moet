"""Resident expert pools and promote-on-miss (phase 2.2), per TP rank.

A layer's resident pool = b12x's prepared tensors for R slots (w13 [R, 737280] int32, w13 scales [R, 46080], w2
[R, 368640], w2 scales [R, 23040]) + a device slot table int32 [E] (-1 = not resident) that the tier's publish op
reads in-graph. A promotion moves one expert from the pinned store into a slot on a copy stream, stream-ordered
against the captured steps without recapture (rule R4):

    after step n is launched:   copy stream waits for step n ->  slot[victim] = -1  -> event U_n
                                -> H2D of the record -> repack (b12x_layout_probe's permutations) -> slot[e] = s -> event M_n
    before step n+1 launches:   main stream waits U_n            (step n+1 never reads a slot being rewritten)
    before step n+2 launches:   main stream waits M_n            (step n+2 sees the new expert)

Victims: least recently routed slot (the tier's use log carries every step's routing, hits included), never one of
the step's own misses.
"""
from __future__ import annotations

import os

import numpy as np
import torch

RECORD_BYTES = 4700160
WORDS = dict(w13=737280, w13s=46080, w2=368640, w2s=23040)


def load_layout(tables_dir: str, device: torch.device) -> dict:
    return {name: torch.from_numpy(np.load(os.path.join(tables_dir, f"b12x_layout_{name}.npy"))).to(device) for name in WORDS}


class ResidentPool:
    def __init__(self, layer: int, n_experts: int, R: int, device: torch.device, pool: dict | None = None,
                 slot_table: torch.Tensor | None = None):
        self.layer, self.E, self.R, self.device = layer, n_experts, R, device
        self.pool = pool or {name: torch.zeros(R, w, dtype=torch.int32, device=device) for name, w in WORDS.items()}
        self.slot_table = slot_table if slot_table is not None else torch.full((n_experts,), -1, dtype=torch.int32, device=device)
        self.expert_of_slot = [-1] * R          # host mirror of the table
        self.slot_of_expert: dict[int, int] = {}
        self.last_used = [-1] * R               # step of the last routing to the slot's expert (use log)
        self.use_count = [0] * R                # routings since the slot was filled (the tie-breaker)
        self.staging = torch.empty(RECORD_BYTES, dtype=torch.uint8, device=device)

    def mapped(self, e: int) -> int:
        return self.slot_of_expert.get(int(e), -1)

    def note_use(self, experts, step: int) -> None:
        for e in experts:
            s = self.slot_of_expert.get(int(e), -1)
            if s >= 0:
                self.last_used[s] = max(self.last_used[s], step)
                self.use_count[s] += 1

    def pick_victim(self, protect: set, candidate_key: tuple) -> int:
        """An empty slot, else the least recently used one (ties: the least used since it was filled) whose expert
        is not protected AND whose key is older than the candidate's (a miss never evicts something hotter than
        itself: with more misses per step than spare slots that would churn the hot set); -1 if none."""
        best, best_key = -1, None
        for s in range(self.R):
            e = self.expert_of_slot[s]
            if e < 0:
                return s
            if e in protect:
                continue
            key = (self.last_used[s], self.use_count[s])
            if best_key is None or key < best_key:
                best, best_key = s, key
        return best if best_key is not None and best_key < candidate_key else -1


class PlacementManager:
    def __init__(self, ext, layout: dict, pools: list[ResidentPool], records_of, copy_stream: torch.cuda.Stream,
                 max_promotions_per_layer: int = 4):
        """records_of(layer, expert) -> a pinned host uint8 tensor [RECORD_BYTES] (the store)."""
        self.ext, self.layout, self.pools, self.records_of = ext, layout, pools, records_of
        self.copy = copy_stream
        self.max_per_layer = max_promotions_per_layer
        self.unmap_events: list[torch.cuda.Event] = []     # U of the last after_launch
        self.map_events_last: list[torch.cuda.Event] = []  # M of the last after_launch
        self.map_events_prev: list[torch.cuda.Event] = []  # M of the one before
        self.last_promoted: dict[int, set] = {}            # layer -> experts promoted by the last after_launch
        self.stats = dict(promotions=0, skipped_no_victim=0, skipped_already=0)

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
                P.expert_of_slot[s] = e
                P.slot_of_expert[e] = s
                P.last_used[s] = 0
                P.use_count[s] = 0
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

    def after_launch(self, misses: np.ndarray, step: int, main_stream: torch.cuda.Stream | None = None) -> int:
        """misses: [n, 4] (step, layer, expert, m) of the step just launched; schedules promotions on the copy
        stream behind that step. Returns the number scheduled."""
        self.last_promoted = {}
        if len(misses) == 0:
            return 0
        ms = main_stream or torch.cuda.current_stream()
        done = torch.cuda.Event()
        done.record(ms)
        self.copy.wait_event(done)
        per_layer: dict[int, list[tuple[int, int]]] = {}
        for row in misses:
            per_layer.setdefault(int(row[1]), []).append((int(row[2]), int(row[0])))
        n_sched = 0
        with torch.cuda.stream(self.copy):
            for layer, ems in per_layer.items():
                P = self.pools[layer]
                protect = {e for e, _ in ems}
                miss_step = {}
                for e, st in ems:
                    miss_step[e] = max(miss_step.get(e, -1), st)
                n_layer = 0
                for e in dict.fromkeys(e for e, _ in ems):  # distinct, in miss order
                    if n_layer >= self.max_per_layer:
                        break
                    if P.mapped(e) >= 0:
                        self.stats["skipped_already"] += 1
                        continue
                    s = P.pick_victim(protect, (miss_step[e], 1))
                    if s < 0:
                        self.stats["skipped_no_victim"] += 1
                        continue
                    victim = P.expert_of_slot[s]
                    if victim >= 0:
                        P.slot_table[victim] = -1
                        del P.slot_of_expert[victim]
                        P.expert_of_slot[s] = -1
                    u = torch.cuda.Event()
                    u.record(self.copy)
                    self.unmap_events.append(u)
                    self._repack_into(P, layer, e, s)
                    P.slot_table[e] = s
                    m = torch.cuda.Event()
                    m.record(self.copy)
                    self.map_events_last.append(m)
                    P.expert_of_slot[s] = e
                    P.slot_of_expert[e] = s
                    P.last_used[s] = miss_step[e]   # the recency the use log would have given it
                    P.use_count[s] = 1
                    self.last_promoted.setdefault(layer, set()).add(e)
                    self.stats["promotions"] += 1
                    n_sched += 1
                    n_layer += 1
        return n_sched
