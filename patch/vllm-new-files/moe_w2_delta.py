# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP4 delta tier for the 1-GPU 2-bit MoE path (quality restoration).

Hot routed experts get their FULL e2m1 nibble planes cached in a small GPU
pool and dispatched to the `moe_w4_mm` kernel; everyone else stays on the
resident 2-bit base (`moe_w2_mm`). Block-32 scale planes are shared by both
tiers (kept on GPU since load).

Pieces:
  - host store: fragment-major FP4 planes per (layer, expert) in PINNED
    memory (built once at load from the checkpoint bytes, D2H);
  - GPU pool: VLLM_MOE_W2_DELTA_GB worth of 12.6 MiB expert slots
    (w13 8.4 MiB + w2 4.2 MiB packed back-to-back per slot);
  - slot table: int32 [layers, 256] on GPU (-1 = base tier), read by the
    desc-build kernel inside CUDA graphs;
  - manager thread: consumes the forward's last-seen expert flags
    (event-synced D2H), promotes seen-but-uncached experts (H2D on a side
    stream, capped per tick), evicts only experts cold for >= 2 ticks.

Consistency model (deliberate): the table update is racy versus graph
replay — the worst case is one step reading the OLD tier for an expert,
which is numerically safe (both tiers are valid weights). Evicting only
cold slots keeps pool rewrites away from in-flight reads.
"""

import os
import threading
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_GB = float(os.getenv("VLLM_MOE_W2_DELTA_GB", "2.0"))
_PROMOTE_PER_TICK = int(os.getenv("VLLM_MOE_W2_DELTA_PROMOTE", "8"))
_TICK_S = float(os.getenv("VLLM_MOE_W2_DELTA_TICK_MS", "5")) / 1e3

# Observability of the precision tiering (default OFF; behaviour-neutral — only
# adds logging). Useful for studying the delta in practice: which experts are
# FP4 right now, and how the working set churns.
#   VLLM_MOE_W2_DELTA_TRACE=0  silent (default)
#                          =1  periodic coverage/churn summary + per-layer
#                              FP4 histogram, every _TRACE_EVERY ticks
#                          =2  + one line per promotion/eviction (verbose)
#   VLLM_MOE_W2_DELTA_TRACE_EVERY=N   ticks between summaries (default 64)
#   VLLM_MOE_W2_DELTA_DUMP=<path>     also write the full precision map
#                                     (which expert is FP4 vs 2-bit) as JSON
#                                     at each summary, atomically (tail-able).
_TRACE = int(os.getenv("VLLM_MOE_W2_DELTA_TRACE", "0"))
_TRACE_EVERY = max(int(os.getenv("VLLM_MOE_W2_DELTA_TRACE_EVERY", "64")), 1)
_DUMP_PATH = os.getenv("VLLM_MOE_W2_DELTA_DUMP", "")

# Routing-trace capture for offline policy study (gated, off by default): record
# each tick's seen (layer,expert) frame and periodically write a .npy of
# [frame, layer, expert] rows. Replay it through candidate promote/evict
# policies in a simulator instead of restarting the 159B model each round.
_CAPTURE = os.getenv("VLLM_MOE_W2_DELTA_CAPTURE", "")
_CAPTURE_TICKS = int(os.getenv("VLLM_MOE_W2_DELTA_CAPTURE_TICKS", "20000"))

# Promotion/eviction policy (chosen via offline trace replay; see tools/delta_sim.py).
# "need" (gate-driven, the right default for a memory-bound decoder): the FP4 pool
# is filled ONLY by the confidence gate's force_promote -- an expert enters FP4
# *because a low-confidence token routed to it* (2-bit was insufficient and forced
# a re-run), never because it is merely hot. This matters because decode is
# HBM-bandwidth-bound and 2-bit is HALF the bytes of FP4: promoting a hot expert to
# FP4 makes the most-read weights SLOWER for no quality reason. Under "need" the
# background manager does NOT promote; it only ages/evicts, keeping the experts with
# the highest (recency-decayed) NEED score and letting everything else stay 2-bit
# (fast). Requires the gate on (VLLM_MOE_W2_GATE=1) to generate the need signal.
# "freq": promote the globally-hottest candidates and evict the least-frequently
# used slot -- maximizes FP4 COVERAGE/hit-rate (good when the pool >= working set so
# the extra FP4 bytes are amortized), but spends FP4 on experts 2-bit handled fine.
# "lru" = old behaviour (promote in order, evict coldest).
_POLICY = os.getenv("VLLM_MOE_W2_DELTA_POLICY", "freq")
_DECAY = float(os.getenv("VLLM_MOE_W2_DELTA_DECAY", "0.5"))
_DECAY_TICKS = max(int(os.getenv("VLLM_MOE_W2_DELTA_DECAY_TICKS", "1000")), 1)

# Token-weighted hit-rate: when observability is on, the forward records per-expert
# routing COUNTS (not a binary flag) so the logged hit-rate reflects the fraction
# of token->expert ROUTINGS served at FP4 — the honest number. A binary-flag
# hit-rate under-counts badly, because the cached hot experts absorb
# disproportionately many tokens (a one-token expert and a 500-token expert count
# the same under a flag). Off by default -> the prod serving path is unchanged.
_COUNT = (_TRACE > 0) or bool(_CAPTURE)

# Per-expert FP4 plane sizes for the SINGLE-GPU (TP1) layout. Under tensor
# parallelism the experts shard, so the real per-rank planes are smaller; the
# plane builder passes the per-rank sizes to get_tier()/DeltaTier and every
# consumer reads the per-instance self.{w13_bytes,w2_bytes,slot_bytes}. These
# module constants stay as the TP1 default / fallback (byte-identical to the
# original single-GPU path).
W13_BYTES = 4096 * 4096 // 2          # 8.0 MiB (TP1)
W2_BYTES = 4096 * 2048 // 2           # 4.0 MiB (TP1)
SLOT_BYTES = W13_BYTES + W2_BYTES     # 12.0 MiB per expert (TP1)


class DeltaTier:
    def __init__(self, n_layers: int, n_experts: int, dev,
                 w13_bytes: int = W13_BYTES, w2_bytes: int = W2_BYTES):
        self.n_layers = n_layers
        self.E = n_experts
        if isinstance(dev, torch.device) and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.dev = dev
        # Per-rank FP4 plane sizes (== the TP1 module constants on a single GPU;
        # halved under TP2, quartered under TP4 as the experts shard). All slot
        # math, host staging, and the desc-kernel pool indexing read these so the
        # tier is correct under tensor parallelism.
        self.w13_bytes = w13_bytes
        self.w2_bytes = w2_bytes
        self.slot_bytes = w13_bytes + w2_bytes
        self.n_slots = max(int(_GB * 2**30) // self.slot_bytes, 8)
        self.pool = torch.empty(self.n_slots, self.slot_bytes, dtype=torch.uint8,
                                device=dev)
        # device table read by the desc kernel; host mirror for the manager
        self.slot_table = torch.full((n_layers, n_experts), -1,
                                     dtype=torch.int32, device=dev)
        self._mirror = torch.full((n_layers, n_experts), -1,
                                  dtype=torch.int32)
        # slot -> (layer, expert, last_seen_tick); -1 layer = free
        self._owner = [(-1, -1, 0)] * self.n_slots
        self._free = list(range(self.n_slots))
        # routing signal written by the forward (graph-replayed scatter): token
        # COUNTS per expert when observability is on (int32, for token-weighted
        # hit-rate), else a cheap binary flag (uint8). Read by the manager only;
        # the desc kernel reads slot_table, never this.
        _seen_dtype = torch.int32 if _COUNT else torch.uint8
        self.seen = torch.zeros(n_layers, n_experts, dtype=_seen_dtype,
                                device=dev)
        self._seen_host = torch.zeros_like(self.seen, device="cpu",
                                           pin_memory=True)
        self._host = {}               # layer -> pinned [E, SLOT_BYTES] u8
        self._stream = torch.cuda.Stream(dev)
        # Guards pool/slot_table/_mirror/_owner/_free/_freq mutations. In steady
        # state only the manager thread mutates them (uncontended). The
        # confidence-gated re-forward (force_promote) mutates from the FORWARD
        # thread, so the two must be serialized. The desc kernel only READS
        # slot_table (never takes the lock), so steady-state decode is unaffected.
        self._lock = threading.Lock()
        self._tick = 0
        self._stop = False
        self._thread = None
        self._last_capture = 0.0    # graph-capture grace (see notify_capture)
        # observability counters: cumulative + per-summary window
        self._n_promoted = 0
        self._n_evicted = 0
        self._win_promoted = 0
        self._win_evicted = 0
        self._last_summary_tick = 0
        self._win_hits = 0.0     # token-weighted FP4-served routings this window
        self._win_active = 0.0   # token-weighted total routings this window
        self._win_hits_d = 0     # distinct FP4-served experts this window
        self._win_active_d = 0   # distinct active experts this window
        self._cap_frames = []
        self._cap_done = False
        # recency-decayed routing frequency per expert (drives the freq policy)
        self._freq = torch.zeros(n_layers, n_experts, dtype=torch.float32)
        # NEED signal (gate-driven policy): how often the confidence gate flagged a
        # step routing to this expert (i.e. 2-bit was insufficient). Recency-decayed
        # like _freq; the eviction key under _POLICY == "need".
        self._need = torch.zeros(n_layers, n_experts, dtype=torch.float32)
        self._last_decay = 0
        logger.info("moe_w2 delta tier: %d slots x %.1f MiB (%.2f GiB pool)",
                    self.n_slots, self.slot_bytes / 2**20,
                    self.n_slots * self.slot_bytes / 2**30)
        if _TRACE:
            logger.info("moe_w2 delta trace ON: level %d, every %d ticks%s",
                        _TRACE, _TRACE_EVERY,
                        f", dump -> {_DUMP_PATH}" if _DUMP_PATH else "")
        if _CAPTURE:
            logger.info("moe_w2 delta CAPTURE ON -> %s (dump every 200 frames)",
                        _CAPTURE)

    # ---- load-time -------------------------------------------------------

    def add_layer_host_planes(self, layer_key: int, w13_plane_gpu, w2_plane_gpu):
        """Stage a layer's fragment-major FP4 planes into pinned host memory.

        Called from the plane builder while the FP4 planes are transiently
        on GPU; w13/w2 are [E, bytes] u8.
        """
        host = torch.empty(self.E, self.slot_bytes, dtype=torch.uint8,
                           pin_memory=True)
        host[:, :self.w13_bytes].copy_(w13_plane_gpu, non_blocking=False)
        host[:, self.w13_bytes:].copy_(w2_plane_gpu, non_blocking=False)
        self._host[layer_key] = host

    def start(self):
        if self._thread is not None:   # idempotent: started once at tier creation
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="moe-w2-delta")
        self._thread.start()

    # ---- manager loop ----------------------------------------------------

    def _loop(self):
        while not self._stop:
            try:
                torch.cuda.set_device(self.dev)
                self._tick_once()
            except Exception as e:  # noqa: BLE001 - never kill serving
                logger.warning("delta tick failed: %s", e)
                time.sleep(1.0)
            time.sleep(_TICK_S)

    def notify_capture(self):
        """Forward calls this while stream capture is active: the manager
        idles through the whole capture phase plus a grace window (captures
        run with thread_local error mode as the primary guard; this avoids
        even benign allocator interleaving)."""
        self._last_capture = time.monotonic()

    def _tick_once(self):
        if time.monotonic() - self._last_capture < 5.0:
            return
        self._tick += 1
        with torch.cuda.stream(self._stream):
            self._seen_host.copy_(self.seen, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self._stream)
        ev.synchronize()
        seen = self._seen_host.nonzero()
        if seen.numel() == 0:
            return
        if _CAPTURE and not self._cap_done:
            self._cap_frames.append((self._tick, seen.to(torch.int16).clone()))
            n = len(self._cap_frames)
            if n % 200 == 0 or n >= _CAPTURE_TICKS:
                self._dump_capture(final=n >= _CAPTURE_TICKS)
        # hit-rate: of this tick's routings, how many hit an FP4 slot. `cnt` is
        # token counts (count-mode) or 1s (binary) per active expert -> the
        # token-weighted ratio is the honest one; the distinct ratio is the old
        # flag-based number, logged alongside for comparison.
        cnt = self._seen_host[seen[:, 0], seen[:, 1]].to(torch.float64)
        cached = self._mirror[seen[:, 0], seen[:, 1]] >= 0
        self._win_hits += float((cnt * cached).sum())
        self._win_active += float(cnt.sum())
        self._win_hits_d += int(cached.sum())
        self._win_active_d += int(seen.shape[0])
        # Mutate shared tier state under the lock (serialized with a concurrent
        # gate-driven force_promote on the forward thread).
        with self._lock:
            # recency-decayed routing frequency (the hotness signal)
            self._freq[seen[:, 0], seen[:, 1]] += 1.0
            # refresh last_seen for cached owners; collect promotion candidates
            cand = []
            seen_set = set()
            for li, ei in seen.tolist():
                seen_set.add((li, ei))
                s = int(self._mirror[li, ei])
                if s >= 0:
                    la, ex, _ = self._owner[s]
                    self._owner[s] = (la, ex, self._tick)
                elif li in self._host:
                    cand.append((li, ei))
            # "need" policy: the background manager does NOT promote — FP4 is filled
            # only by the gate's force_promote (an expert 2-bit handled fine never
            # gets pulled to the slower FP4 path). freq/lru: promote the hottest
            # candidates first so the limited pool tracks genuinely hot experts
            # across ALL layers (vs the layer-sorted order that starved past layer 0).
            if _POLICY != "need":
                if _POLICY == "freq" and len(cand) > 1:
                    ca = torch.tensor(cand)
                    order = torch.argsort(self._freq[ca[:, 0], ca[:, 1]],
                                          descending=True)
                    cand = [cand[i] for i in order.tolist()]
                promoted = 0
                for li, ei in cand:
                    if promoted >= _PROMOTE_PER_TICK:
                        break
                    slot = self._take_slot(seen_set)
                    if slot is None:
                        break
                    self._promote(li, ei, slot)
                    promoted += 1
            if self._tick - self._last_decay >= _DECAY_TICKS:
                self._freq *= _DECAY  # keep the frequency signal recent + bounded
                self._need *= _DECAY  # need decays too -> tracks RECENT 2-bit misses
                self._last_decay = self._tick
        # reset flags for the next window (racy with the forward's scatter
        # of ones — a lost flag only delays promotion by one tick)
        if self._tick % 4 == 0:
            self.seen.zero_()
        if _TRACE and self._tick - self._last_summary_tick >= _TRACE_EVERY:
            self._log_summary()
            self._last_summary_tick = self._tick

    def _take_slot(self, seen_set):
        if self._free:
            return self._free.pop()
        # evict the least-valuable slot not active this window: least-frequently
        # used (freq policy) or coldest last-seen (lru). Both restrict to slots
        # cold >= 2 ticks so in-flight graph reads never hit a rewritten slot.
        best, best_key = None, None
        for s, (li, ei, t) in enumerate(self._owner):
            if (li, ei) in seen_set or self._tick - t < 2:
                continue
            # need: evict the LEAST-needed expert (smallest gate-flag score) so the
            # pool retains the experts 2-bit most struggles with. freq: least-freq.
            # lru: coldest last-seen.
            if _POLICY == "need":
                key = float(self._need[li, ei])
            elif _POLICY == "freq":
                key = float(self._freq[li, ei])
            else:
                key = t
            if best_key is None or key < best_key:
                best, best_key = s, key
        if best is None:
            return None
        li, ei, t = self._owner[best]
        # unmap FIRST (graphs stop dispatching w4 before bytes change)
        self.slot_table[li, ei] = -1
        self._mirror[li, ei] = -1
        self._n_evicted += 1
        self._win_evicted += 1
        if _TRACE >= 2:
            logger.info("[delta] evict   L%-2d E%-3d  slot %-4d (cold %d ticks)",
                        li, ei, best, self._tick - t)
        return best

    def _promote(self, li, ei, slot):
        with torch.cuda.stream(self._stream):
            self.pool[slot].copy_(self._host[li][ei], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self._stream)
        ev.synchronize()           # bytes resident BEFORE mapping
        self.slot_table[li, ei] = slot
        self._mirror[li, ei] = slot
        self._owner[slot] = (li, ei, self._tick)
        self._n_promoted += 1
        self._win_promoted += 1
        if _TRACE >= 2:
            logger.info("[delta] promote L%-2d E%-3d  slot %-4d (tick %d)",
                        li, ei, slot, self._tick)

    # ---- confidence-gated re-forward (directive 2 / Step B) --------------

    def force_promote(self, layers=None, max_promote=None) -> int:
        """Synchronously pull this step's COLD routed experts up to FP4, for a
        confidence-gated re-forward (directive 2 / Step B).

        Reads `seen` (the forward's routed-expert scatter) to find routed
        (layer, expert) pairs still on the 2-bit base (slot_table == -1), copies
        their FP4 planes H2D on the side stream, blocks ONCE on a single event,
        then maps them into `slot_table`. A subsequent CUDA-graph REPLAY then
        recomputes exactly those experts at FP4 "for free". Promotions persist
        (a superset of lazy promotion), so a flagged step also warms the cache.

        Unlike the background `_promote`, this runs on the FORWARD thread, so all
        pool/table mutations are serialized with the manager via `self._lock`.
        Slot writes stay on the default (forward) stream and pool copies on the
        side stream — matching `_promote`/`_take_slot` so in-flight graph reads
        never observe a half-rewritten slot (eviction only touches >=2-tick-cold
        slots). Must NOT be called during graph capture.

        Args:
            layers: optional iterable of layer keys to restrict to (default all).
            max_promote: optional cap on experts promoted this call.
        Returns:
            number of experts newly promoted to FP4.
        """
        if not self._host:
            return 0
        # snapshot the forward's routed-expert scatter. The side stream must
        # WAIT on the forward (main) stream first so the snapshot includes THIS
        # step's mark_seen scatter — cross-stream ordering is not automatic, and
        # a snapshot racing ahead would miss this step's cold experts.
        main = torch.cuda.current_stream(self.dev)
        with torch.cuda.stream(self._stream):
            self._stream.wait_stream(main)
            self._seen_host.copy_(self.seen, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self._stream)
        ev.synchronize()
        seen = self._seen_host.nonzero()
        if seen.numel() == 0:
            return 0
        layer_filter = set(layers) if layers is not None else None
        with self._lock:
            seen_set = set()
            cand = []
            for li, ei in seen.tolist():
                seen_set.add((li, ei))
                if layer_filter is not None and li not in layer_filter:
                    continue
                # NEED signal: this step was gate-flagged (2-bit low-confidence), so
                # every expert active in it gets a need bump -- INCLUDING ones already
                # FP4 (so repeat offenders accumulate need and resist eviction). The
                # true culprits are the experts consistently present across fires;
                # decay washes out the coincidental ones.
                self._need[li, ei] += 1.0
                if li in self._host and int(self._mirror[li, ei]) < 0:
                    cand.append((li, ei))
            if not cand:
                return 0
            # capped promote prioritizes the most-NEEDED experts under the gate-driven
            # policy (repeat offenders first); hottest-first otherwise.
            if len(cand) > 1:
                ca = torch.tensor(cand)
                rank = self._need if _POLICY == "need" else self._freq
                order = torch.argsort(rank[ca[:, 0], ca[:, 1]], descending=True)
                cand = [cand[i] for i in order.tolist()]
            if max_promote is not None:
                cand = cand[:max_promote]
            # take slots (default stream: evictions unmap on the forward stream),
            # issue all copies on the side stream, then a SINGLE sync before
            # mapping — bytes resident before any graph replay can read them.
            plan = []
            for li, ei in cand:
                slot = self._take_slot(seen_set)
                if slot is None:
                    break  # pool full of hot/in-flight slots; promote what we can
                # RESERVE the slot's owner IMMEDIATELY: _take_slot's eviction
                # scans _owner for victims, so without this a later iteration in
                # THIS loop could hand out the same just-taken slot again (two
                # experts -> one slot -> one reads the other's weights -> pool
                # corruption). Owner tick == current tick also cold-protects it.
                # GPU slot_table/_mirror writes stay deferred until after the sync.
                self._owner[slot] = (li, ei, self._tick)
                with torch.cuda.stream(self._stream):
                    self.pool[slot].copy_(self._host[li][ei], non_blocking=True)
                plan.append((li, ei, slot))
            if not plan:
                return 0
            with torch.cuda.stream(self._stream):
                ev = torch.cuda.Event()
                ev.record(self._stream)
            ev.synchronize()
            for li, ei, slot in plan:
                self.slot_table[li, ei] = slot
                self._mirror[li, ei] = slot
                self._freq[li, ei] += 1.0
            self._n_promoted += len(plan)
            self._win_promoted += len(plan)
        if _TRACE >= 2:
            logger.info("[delta] force-promote %d experts (gate)", len(plan))
        return len(plan)

    def mark_need_only(self, layers=None) -> int:
        """MEASUREMENT ONLY: bump _need for THIS step's routed experts (a low-conf,
        gate-fired step) WITHOUT promoting anything. Lets us study whether 2-bit
        difficulty concentrates on a small expert set (=> a small persistent FP4
        pool can cover the 'hard' experts) before committing to a pool policy. No
        slot/pool mutation, no H2D copy, no re-forward -> zero serving perturbation
        beyond the seen snapshot. _freq (all-routing) keeps accruing in _tick_once,
        so _need/_freq gives per-expert over-representation in low-confidence steps."""
        if not self._host:
            return 0
        main = torch.cuda.current_stream(self.dev)
        with torch.cuda.stream(self._stream):
            self._stream.wait_stream(main)
            self._seen_host.copy_(self.seen, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self._stream)
        ev.synchronize()
        seen = self._seen_host.nonzero()
        if seen.numel() == 0:
            return 0
        lf = set(layers) if layers is not None else None
        n = 0
        with self._lock:
            for li, ei in seen.tolist():
                if lf is not None and li not in lf:
                    continue
                self._need[li, ei] += 1.0
                n += 1
        return n

    def stats(self):
        cached = int((self._mirror >= 0).sum())
        return dict(slots=self.n_slots, cached=cached, tick=self._tick,
                    promoted=self._n_promoted, evicted=self._n_evicted)

    # ---- observability ---------------------------------------------------

    def precision_of(self, layer: int, expert: int) -> str:
        """Live tier of one expert: 'fp4' (delta-cached) or '2bit' (base)."""
        return "fp4" if int(self._mirror[layer, expert]) >= 0 else "2bit"

    def precision_map(self) -> dict:
        """{layer: [expert ids currently in FP4]}. Anything not listed is on
        the resident 2-bit base — i.e. the live precision of every expert."""
        out = {}
        cov = self._mirror >= 0
        for li in range(self.n_layers):
            ex = cov[li].nonzero().flatten().tolist()
            if ex:
                out[li] = ex
        return out

    def _log_summary(self):
        cov = self._mirror >= 0
        cached = int(cov.sum())
        # Under pipeline parallelism this rank hosts only ITS layers (local
        # layer_keys); normalize coverage by the layers actually staged here
        # (len(self._host)) rather than the full slot_table (n_layers*E), so the
        # reported %experts is honest per-rank. On TP/1-GPU every layer is hosted
        # on each rank -> len(self._host) == n_layers -> unchanged.
        total = max(len(self._host), 1) * self.E
        hr = 100.0 * self._win_hits / max(self._win_active, 1.0)
        hrd = 100.0 * self._win_hits_d / max(self._win_active_d, 1)
        logger.info(
            "[delta] tick %d: FP4 %d/%d slots, covering %d/%d experts (%.1f%%); "
            "hit-rate %.1f%% tokens / %.1f%% experts; window +%d/-%d, cumulative +%d/-%d",
            self._tick, cached, self.n_slots, cached, total,
            100.0 * cached / max(total, 1), hr, hrd, self._win_promoted,
            self._win_evicted, self._n_promoted, self._n_evicted)
        per_layer = cov.sum(dim=1).tolist()
        hist = " ".join(f"L{li}:{int(c)}" for li, c in enumerate(per_layer) if c)
        if hist:
            logger.info("[delta] FP4 experts per layer: %s", hist)
        # CONCENTRATION study: compare how top-heavy low-confidence routing (_need,
        # from the gate via mark_need_only) is vs overall routing (_freq). If the
        # top few % of experts hold MOST of the _need mass while _freq is spread,
        # 2-bit difficulty concentrates -> a small persistent FP4 set suffices. If
        # _need is as spread as _freq, difficulty is context-driven (no small set).
        nd = self._need.flatten()
        if float(nd.sum()) > 0:
            fr = self._freq.flatten()

            def topmass(v, p):
                vs = torch.sort(v, descending=True).values
                k = max(1, int(vs.numel() * p))
                return 100.0 * float(vs[:k].sum()) / max(float(v.sum()), 1e-9)
            logger.info(
                "[need] low-conf routing top1%%/5%%/10%% = %.0f/%.0f/%.0f  |  "
                "all routing top1%%/5%%/10%% = %.0f/%.0f/%.0f  |  experts need>0: %d/%d",
                topmass(nd, .01), topmass(nd, .05), topmass(nd, .10),
                topmass(fr, .01), topmass(fr, .05), topmass(fr, .10),
                int((nd > 0).sum()), nd.numel())
        self._win_promoted = self._win_evicted = 0
        self._win_hits = self._win_active = 0.0
        self._win_hits_d = self._win_active_d = 0
        if _DUMP_PATH:
            self._dump(_DUMP_PATH)

    def _dump(self, path: str):
        import json
        snap = dict(tick=self._tick, n_slots=self.n_slots,
                    cached=int((self._mirror >= 0).sum()),
                    promoted_total=self._n_promoted,
                    evicted_total=self._n_evicted,
                    fp4_by_layer=self.precision_map())
        try:  # atomic write so a tail/watcher never reads a half file
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(snap, f)
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001 - observability must not kill serving
            logger.warning("delta dump to %s failed: %s", path, e)

    def _dump_capture(self, final=False):
        import numpy as np
        rows = []
        for tk, fr in self._cap_frames:
            a = fr.numpy()
            if a.size == 0:
                continue
            idx = np.full((a.shape[0], 1), tk, dtype=np.int32)
            rows.append(np.hstack([idx, a.astype(np.int32)]))
        arr = np.vstack(rows) if rows else np.zeros((0, 3), np.int32)
        try:
            np.save(_CAPTURE, arr)
            logger.info("delta capture: %d frames, %d activations -> %s%s",
                        len(self._cap_frames), arr.shape[0], _CAPTURE,
                        " (final)" if final else "")
        except Exception as e:  # noqa: BLE001 - capture must not kill serving
            logger.warning("delta capture save failed: %s", e)
        if final:
            self._cap_done = True
            self._cap_frames = []


def mark_seen(seen_row, ids):
    """Record routed experts into a layer's seen row from the forward. Token
    COUNTS when observability is on (token-weighted hit-rate / capture), else a
    cheap binary flag. `ids` = flattened topk_ids (int64). Graph-capture-safe."""
    if _COUNT:
        seen_row.index_add_(0, ids, torch.ones_like(ids, dtype=seen_row.dtype))
    else:
        seen_row.index_fill_(0, ids, 1)


_TIER: DeltaTier | None = None


def enabled() -> bool:
    return _GB > 0 and os.getenv("VLLM_MOE_W2_DELTA", "1") == "1"


def get_tier(n_layers=None, n_experts=256, dev=None,
             w13_bytes=None, w2_bytes=None) -> DeltaTier | None:
    global _TIER
    if not enabled():
        return None
    if _TIER is None:
        if n_layers is None:
            # one slot-table row per built layer_key: the main stack and,
            # when the cutoff includes it, the MTP drafter MoE
            n_layers = int(os.getenv("VLLM_MOE_W2_NUM_LAYERS", "43")) + 1
        # The plane builder passes the per-rank FP4 plane sizes (smaller under
        # TP); fall back to the TP1 module constants when unspecified.
        _TIER = DeltaTier(
            n_layers, n_experts, dev or torch.device("cuda"),
            w13_bytes=W13_BYTES if w13_bytes is None else w13_bytes,
            w2_bytes=W2_BYTES if w2_bytes is None else w2_bytes)
        # Start the background manager as soon as the tier exists. It idles until
        # experts are actually routed (seen empty -> early return) and only
        # promotes layers whose host planes are already staged, so an early start
        # is safe. This fires correctly under PIPELINE PARALLELISM, where
        # layer_keys are LOCAL per rank and never reach NUM_LAYERS-1 -> the old
        # "start on the last layer built" trigger never ran and the tier sat
        # inactive (pool allocated but no promotions).
        _TIER.start()
    return _TIER
