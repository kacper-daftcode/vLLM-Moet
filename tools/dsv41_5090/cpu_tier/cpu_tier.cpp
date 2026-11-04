// cpu_tier.cpp — the host side of the CPU expert tier of one TP rank: a dispatcher thread that watches the per-layer
// request mailboxes the GPU's publish kernel fills (pinned, device-mapped memory), computes the missing (token,
// expert) pairs of a decode step with the b12x-exact kernel (cpu_moe_b12x.cpp, a pool pinned to one CCD) from the
// pinned expert store, writes the bf16 route_output rows into the layer's staging area and bumps the layer's ack,
// which the GPU's wait kernel spins on before injecting the rows ahead of b12x's TopKSum.
//
// Ordering: the GPU publishes the layers of a step in order and the steps in order (the graph replays), so the
// dispatcher walks (step, layer) in that order and polls one mailbox at a time. The step numbers come from a device
// counter bumped in-graph at the start of every step; the dispatcher expects first_step, first_step + 1, ...
// KPIs per (step, layer) go to a ring the engine can read without touching the device.
//
// The store tail (2026-11-04, the 256 / 128 GB tiers): with cpu_tier_store_tail_init() only a DRAM LRU of records
// per layer is pinned; an expert marked cold (cpu_tier_set_record_cold) is read from the store file on a miss - an
// io_uring O_DIRECT read of the whole 4.7 MB record into a spare arena slot, the DRAM-resident experts of the request
// computed while it is in flight, the record computed on arrival (the same kernel, the same bytes: bit-identical rows)
// and inserted into the layer's LRU (the slot swap evicts the layer's least recently served record). Promotions and
// the prefill streaming take their host pointers through cpu_tier_record_acquire (a pin until the copy is done; a cold
// record is read synchronously) / the prefill arena (cpu_tier_prefill_fetch).
#include <fcntl.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cpu_tier_abi.h"
#include "uring.h"   // ../engram_v2/uring.h (build.sh: -I../engram_v2; the patcher copies it next to this file)

extern "C" {
void* moe_b12x_pool_create(int nthreads, const int* cpu_ids);
void moe_b12x_pool_destroy(void* pool);
int moe_b12x_experts_forward(void* pool, int n_experts, const uint8_t* const* records, const int* m_per_expert,
                             const int* token_idx, const float* pair_weight, const int* out_row, int n_tokens,
                             const uint8_t* aq, const uint8_t* as, uint16_t* out, int64_t out_stride, float vs1,
                             float vs2, uint16_t* proj_out, uint8_t* inter_q_out, uint8_t* inter_s_out);
void moe_b12x_pool_stats(void* pool, int64_t* out, int reset);
}

namespace {

constexpr int KPI_RING = 4096;

struct Kpi {
    int64_t step, layer, n_pairs, n_experts, t_publish_gpu_ns, t_seen_ns, t_done_ns, status;
};

constexpr int MISS_RING = 1 << 16;
struct Miss {
    int32_t step, layer, expert, m;  // one entry per distinct (step, layer, expert) served, m = its pairs
};
constexpr int USE_RING = 1 << 14;
struct Use {
    int32_t step, layer, n;          // every (step, layer): the routing's global expert ids (hits and misses)
    int16_t ids[CPU_TIER_MAX_PAIRS];
};

constexpr int64_t REC = 4700160;   // one expert record (build_store.py), 512-B aligned in the store files and the arena

struct Layer {
    CpuTierRequest* req = nullptr;
    CpuTierAck* ack = nullptr;
    uint16_t* staging = nullptr;
    std::vector<const uint8_t*> records;  // [n_experts] host pointer, or nullptr (cold / not in the store)
    // the store tail
    std::vector<uint8_t> cold;            // [n_experts] 1 = in the store file, fetchable
    std::vector<int32_t> slot_of;         // [n_experts] the LRU slot holding the record, -1 = none
    int fd = -1;                          // the layer's store file (O_DIRECT)
    int lru_head = -1, lru_tail = -1;     // the layer's resident slots: head = least recently served, tail = most
    int n_resident = 0;
};

// one arena slot of the store tail (pinned host memory, REC bytes)
struct Slot {
    int32_t expert = -1;
    int32_t layer = -1;
    int32_t prev = -1, next = -1;         // the layer's LRU list
    uint32_t pins = 0;                    // promotions / debug copies holding the pointer: not evictable
};

struct Tail {
    bool on = false;
    std::string dir;
    int rank = 0;
    uint8_t* arena = nullptr;
    int64_t n_slots = 0;
    int per_layer = 0;                    // N: resident records per layer
    unsigned qd = 8;
    std::vector<Slot> slots;
    std::vector<int32_t> free_slots;      // spare slots (read targets)
    std::mutex mu;                        // the LRU / pins: the dispatcher and the engine thread (acquire / release)
    uring::Ring ring;                     // the dispatcher's
    std::vector<uint8_t*> prefill;        // the prefill arena: n_prefill_sets x prefill_set_slots pinned records
    int prefill_sets = 0, prefill_set_slots = 0;
    uint8_t* prefill_arena = nullptr;
    // totals
    std::atomic<int64_t> fetches{0}, fetch_ns{0}, fetch_wait_ns{0}, sync_fetches{0}, prefill_fetches{0}, prefill_ns{0},
        io_errors{0}, evictions{0}, resident_serves{0};
};

struct Tier {
    int n_layers = 0, n_experts = 0;
    float vs1 = 1.0f, vs2 = 1.0f;
    int dispatcher_cpu = -1;
    void* pool = nullptr;
    std::vector<Layer> layers;
    std::thread thread;
    std::atomic<bool> running{false};
    std::atomic<bool> stop{false};
    std::atomic<uint32_t> expected{1};   // the last served sequence number + 1 (informational)
    std::atomic<int> layer_cursor{0};
    std::vector<uint32_t> expected_seq;  // per layer
    // totals
    std::atomic<int64_t> steps_done{0}, layers_done{0}, pairs_done{0}, busy_ns{0}, errors{0};
    // rings
    Kpi ring[KPI_RING];
    std::atomic<int64_t> ring_n{0};
    Miss misses[MISS_RING];
    std::atomic<int64_t> miss_n{0};
    Use uses[USE_RING];
    std::atomic<int64_t> use_n{0};
    Tail tail;
};

static inline int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void pin_to_cpu(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// ---- the store tail's LRU (callers hold T.tail.mu) --------------------------------------------------------------
static void lru_unlink(Tier& T, Layer& L, int s) {
    Slot& S = T.tail.slots[s];
    if (S.prev >= 0) T.tail.slots[S.prev].next = S.next; else L.lru_head = S.next;
    if (S.next >= 0) T.tail.slots[S.next].prev = S.prev; else L.lru_tail = S.prev;
    S.prev = S.next = -1;
}

static void lru_push_tail(Tier& T, Layer& L, int s) {   // most recently served
    Slot& S = T.tail.slots[s];
    S.prev = L.lru_tail; S.next = -1;
    if (L.lru_tail >= 0) T.tail.slots[L.lru_tail].next = s; else L.lru_head = s;
    L.lru_tail = s;
}

// slot s (a spare slot now holding expert e's record) becomes resident in layer l; the layer's least recently served
// unpinned record is evicted when the layer is at capacity (its slot returns to the spares). A record that became
// resident meanwhile (a concurrent fetch) keeps the first copy: s returns to the spares.
// evict the layer's least recently served unpinned records while it holds more than N (their slots return to the
// spares). Pinned slots (a request being computed, a promotion's copy, a prefill round) are skipped, so a layer may hold
// more than N for as long as the pins last; the callers that drop pins trim again.
static void lru_trim(Tier& T, Layer& L) {
    Tail& X = T.tail;
    while (L.n_resident > X.per_layer) {
        int victim = -1;
        for (int v = L.lru_head; v >= 0; v = X.slots[v].next)
            if (X.slots[v].pins == 0) { victim = v; break; }
        if (victim < 0) return;
        Slot& V = X.slots[victim];
        lru_unlink(T, L, victim);
        L.slot_of[V.expert] = -1;
        L.records[V.expert] = nullptr;
        V.expert = -1; V.layer = -1;
        --L.n_resident;
        X.evictions.fetch_add(1, std::memory_order_relaxed);
        X.free_slots.push_back(victim);
    }
}

static void lru_insert(Tier& T, int l, int e, int s) {
    Layer& L = T.layers[l];
    Tail& X = T.tail;
    if (L.slot_of[e] >= 0) { X.free_slots.push_back(s); return; }
    Slot& S = X.slots[s];
    S.expert = e; S.layer = l; S.pins = 1;   // pinned by the inserter until it is done with the record (it unpins)
    L.slot_of[e] = s;
    L.records[e] = X.arena + (int64_t)s * REC;
    lru_push_tail(T, L, s);
    ++L.n_resident;
    lru_trim(T, L);
    S.pins = 0;
}

// the subset {idx} of the grouped request arrays -> one kernel call
static int forward_subset(Tier& T, Layer& L, const CpuTierRequest* req, const int* idx, int n_idx, const uint8_t* const* recs,
                          const int* m_per, const int* off, const int* tok, const float* pw, const int* orow, int n_tokens) {
    const uint8_t* r2[CPU_TIER_MAX_PAIRS];
    int m2[CPU_TIER_MAX_PAIRS], tok2[CPU_TIER_MAX_PAIRS], orow2[CPU_TIER_MAX_PAIRS];
    float pw2[CPU_TIER_MAX_PAIRS];
    int k = 0;
    for (int i = 0; i < n_idx; ++i) {
        const int x = idx[i];
        r2[i] = recs[x]; m2[i] = m_per[x];
        for (int j = 0; j < m_per[x]; ++j) { tok2[k] = tok[off[x] + j]; pw2[k] = pw[off[x] + j]; orow2[k] = orow[off[x] + j]; ++k; }
    }
    return moe_b12x_experts_forward(T.pool, n_idx, r2, m2, tok2, pw2, orow2, n_tokens, &req->a_values[0][0], &req->a_scales[0][0],
                                    L.staging, CPU_TIER_HID, T.vs1, T.vs2, nullptr, nullptr, nullptr);
}

static void serve(Tier& T, int layer, uint32_t step) {
    Layer& L = T.layers[layer];
    CpuTierRequest* req = L.req;
    const int64_t t_seen = now_ns();
    const int n = (int)req->n_pairs;
    const int n_tokens = (int)req->n_tokens;
    int status = 0;
    int n_experts = 0;
    {   // the use log: the whole routing of this (step, layer)
        const int64_t ui = T.use_n.load(std::memory_order_relaxed);
        Use& U = T.uses[ui % USE_RING];
        U.step = (int32_t)step; U.layer = layer;
        U.n = (int)std::min<uint32_t>(req->n_all, CPU_TIER_MAX_PAIRS);
        memcpy(U.ids, (const void*)req->ids, sizeof(int16_t) * U.n);
        T.use_n.store(ui + 1, std::memory_order_release);
    }
    if (n > CPU_TIER_MAX_PAIRS || n_tokens > CPU_TIER_MAX_TOKENS) status = 3;
    if (n > 0 && status == 0) {
        // group the pairs by expert, keeping the request's pair index as the staging row
        int order[CPU_TIER_MAX_PAIRS];
        for (int i = 0; i < n; ++i) order[i] = i;
        // insertion sort by expert (n <= 64), stable
        for (int i = 1; i < n; ++i) {
            int v = order[i], j = i - 1;
            while (j >= 0 && req->pairs[order[j]].expert > req->pairs[v].expert) { order[j + 1] = order[j]; --j; }
            order[j + 1] = v;
        }
        const uint8_t* recs[CPU_TIER_MAX_PAIRS];
        int m_per[CPU_TIER_MAX_PAIRS], off[CPU_TIER_MAX_PAIRS], tok[CPU_TIER_MAX_PAIRS], orow[CPU_TIER_MAX_PAIRS], eid[CPU_TIER_MAX_PAIRS];
        float pw[CPU_TIER_MAX_PAIRS];
        int cold_idx[CPU_TIER_MAX_PAIRS], n_cold = 0;
        int res_idx[CPU_TIER_MAX_PAIRS], n_res = 0;
        int pinned[CPU_TIER_MAX_PAIRS], n_pinned = 0;
        int last_e = -1;
        Tail& X = T.tail;
        if (X.on) X.mu.lock();
        for (int k = 0; k < n; ++k) {
            const CpuTierPair& p = req->pairs[order[k]];
            if (p.expert != last_e) {
                last_e = p.expert;
                const uint8_t* r = p.expert < L.records.size() ? L.records[p.expert] : nullptr;
                if (!r) {
                    if (X.on && p.expert < L.cold.size() && L.cold[p.expert]) {
                        cold_idx[n_cold++] = n_experts;   // fetched below
                    } else { status = 2; break; }       // not in the store
                } else {
                    res_idx[n_res++] = n_experts;
                    if (X.on) {   // keep the record where it is until the kernel has read it
                        const int s = L.slot_of[p.expert];
                        if (s >= 0) { ++X.slots[s].pins; pinned[n_pinned++] = s; lru_unlink(T, L, s); lru_push_tail(T, L, s); }
                    }
                }
                recs[n_experts] = r;
                eid[n_experts] = p.expert;
                m_per[n_experts] = 0;
                off[n_experts] = k;
                ++n_experts;
            }
            m_per[n_experts - 1] += 1;
            tok[k] = p.token;
            pw[k] = p.weight;
            orow[k] = order[k];
        }
        if (X.on) X.mu.unlock();
        if (status == 0 && n_cold == 0) {
            const int rc = moe_b12x_experts_forward(T.pool, n_experts, recs, m_per, tok, pw, orow, n_tokens,
                                                    &req->a_values[0][0], &req->a_scales[0][0], L.staging,
                                                    CPU_TIER_HID, T.vs1, T.vs2, nullptr, nullptr, nullptr);
            if (rc != 0) status = 4;
            if (X.on) X.resident_serves.fetch_add(n_res, std::memory_order_relaxed);
        } else if (status == 0) {
            // the store tail: the cold records' reads first (io_uring, one 4.7 MB read each into a spare slot), the
            // resident experts computed while they are in flight, each cold expert computed as its record lands
            const int64_t t_issue = now_ns();
            int slot_of_cold[CPU_TIER_MAX_PAIRS];
            int issued = 0, landed = 0;
            static const bool dbg = getenv("CPU_TIER_TAIL_DEBUG") != nullptr;
            if (dbg) fprintf(stderr, "[cpu_tier tail] layer %d step %u: %d experts, %d resident, %d cold, free slots %zu, ring space %u in_flight %u\n",
                             layer, step, n_experts, n_res, n_cold, X.free_slots.size(), X.ring.space(), X.ring.in_flight);
            auto issue_some = [&]() {
                std::lock_guard<std::mutex> g(X.mu);
                while (issued < n_cold && !X.free_slots.empty() && X.ring.space() > 0) {
                    const int s = X.free_slots.back();
                    X.free_slots.pop_back();
                    slot_of_cold[issued] = s;
                    X.ring.prep_read(L.fd, X.arena + (int64_t)s * REC, (unsigned)REC, (uint64_t)eid[cold_idx[issued]] * REC, (uint64_t)issued);
                    ++issued;
                }
            };
            issue_some();
            int r = X.ring.submit(0);
            if (dbg) fprintf(stderr, "[cpu_tier tail]   issued %d, submit -> %d, in_flight %u\n", issued, r, X.ring.in_flight);
            if (r < 0) { status = 5; X.io_errors.fetch_add(1); }
            if (status == 0 && n_res > 0) {
                if (forward_subset(T, L, req, res_idx, n_res, recs, m_per, off, tok, pw, orow, n_tokens) != 0) status = 4;
                X.resident_serves.fetch_add(n_res, std::memory_order_relaxed);
            }
            {   // the resident records are read: unpin them now (evictable again - a request with more cold experts than
                // spare slots needs the evictions to make progress)
                std::lock_guard<std::mutex> g(X.mu);
                for (int i = 0; i < n_pinned; ++i) if (X.slots[pinned[i]].pins) --X.slots[pinned[i]].pins;
                n_pinned = 0;
                lru_trim(T, L);
            }
            int batch[CPU_TIER_MAX_PAIRS];
            while (status == 0 && landed < n_cold) {
                if (X.ring.in_flight == 0 && issued < n_cold) { issue_some(); r = X.ring.submit(0); if (r < 0) { status = 5; break; } }
                if (X.ring.in_flight == 0) { _mm_pause(); continue; }   // no spare slot yet (the engine thread holds them): wait for one
                if (!X.ring.cq_ready()) { X.ring.wait(1); }
                int nb = 0;
                uint64_t ud; int res;
                while (X.ring.reap(&ud, &res)) {
                    const int c = (int)ud;
                    const int s = slot_of_cold[c];
                    if (dbg) fprintf(stderr, "[cpu_tier tail]   landed cold #%d (expert %d) res %d slot %d\n", c, eid[cold_idx[c]], res, s);
                    if (res != (int)REC) { status = 5; X.io_errors.fetch_add(1); std::lock_guard<std::mutex> g(X.mu); X.free_slots.push_back(s); continue; }
                    {
                        std::lock_guard<std::mutex> g(X.mu);
                        lru_insert(T, layer, eid[cold_idx[c]], s);
                        const int s2 = L.slot_of[eid[cold_idx[c]]];
                        ++X.slots[s2].pins; pinned[n_pinned++] = s2;
                        recs[cold_idx[c]] = X.arena + (int64_t)s2 * REC;
                    }
                    batch[nb++] = cold_idx[c];
                    ++landed;
                    X.fetches.fetch_add(1, std::memory_order_relaxed);
                }
                if (issued < n_cold) { issue_some(); r = X.ring.submit(0); if (r < 0) { status = 5; break; } }
                if (nb && status == 0) {
                    if (forward_subset(T, L, req, batch, nb, recs, m_per, off, tok, pw, orow, n_tokens) != 0) status = 4;
                }
                if (n_pinned) {   // the landed records are computed: unpin, trim
                    std::lock_guard<std::mutex> g(X.mu);
                    for (int i = 0; i < n_pinned; ++i) if (X.slots[pinned[i]].pins) --X.slots[pinned[i]].pins;
                    n_pinned = 0;
                    lru_trim(T, L);
                }
            }
            if (status != 0) {
                // fail-closed exit: drain the reads still in flight so that no completion of this request leaks into
                // the next one (the slots return to the spares; the step fails on the ack's status anyway)
                while (X.ring.in_flight) {
                    X.ring.wait(1);
                    uint64_t ud; int res;
                    while (X.ring.reap(&ud, &res)) {
                        std::lock_guard<std::mutex> g(X.mu);
                        X.free_slots.push_back(slot_of_cold[(int)ud]);
                    }
                }
            }
            const int64_t t_land = now_ns();
            X.fetch_ns.fetch_add(t_land - t_issue, std::memory_order_relaxed);
            X.fetch_wait_ns.fetch_add(std::max<int64_t>(0, (t_land - t_issue) - (int64_t)n_res * 130000), std::memory_order_relaxed);
        }
        if (X.on && n_pinned) {
            std::lock_guard<std::mutex> g(X.mu);
            for (int i = 0; i < n_pinned; ++i) if (X.slots[pinned[i]].pins) --X.slots[pinned[i]].pins;
            lru_trim(T, L);
        }
        if (status == 0 || status == 4 || status == 5) {
            // the miss log (per distinct expert), for the placement manager
            for (int i = 0; i < n_experts; ++i) {
                const int64_t mi = T.miss_n.fetch_add(1, std::memory_order_relaxed);
                Miss& M = T.misses[mi % MISS_RING];
                M.step = (int32_t)step; M.layer = layer; M.expert = eid[i]; M.m = m_per[i];
            }
        }
    }
    const int64_t t_done = now_ns();
    CpuTierAck* ack = L.ack;
    ack->status = (uint32_t)status;
    ack->t_seen_ns = (uint64_t)t_seen;
    ack->t_done_ns = (uint64_t)t_done;
    ack->n_pairs = (uint32_t)n;
    ack->n_experts = (uint32_t)n_experts;
    __atomic_store_n(&ack->step, step, __ATOMIC_RELEASE);
    // bookkeeping
    const int64_t k = T.ring_n.fetch_add(1, std::memory_order_relaxed);
    Kpi& K = T.ring[k % KPI_RING];
    K.step = step; K.layer = layer; K.n_pairs = n; K.n_experts = n_experts;
    K.t_publish_gpu_ns = (int64_t)req->t_publish_ns; K.t_seen_ns = t_seen; K.t_done_ns = t_done; K.status = status;
    T.layers_done.fetch_add(1, std::memory_order_relaxed);
    T.pairs_done.fetch_add(n, std::memory_order_relaxed);
    T.busy_ns.fetch_add(t_done - t_seen, std::memory_order_relaxed);
    if (status) T.errors.fetch_add(1, std::memory_order_relaxed);
}

// Every layer has its own sequence of requests (the publish kernel numbers them); the dispatcher polls the layers
// round-robin and serves whichever holds its next number - any layer order, a layer published twice per forward
// (decoder replay) included.
static void dispatcher_main(Tier* T) {
    pin_to_cpu(T->dispatcher_cpu);
    int layer = 0;
    while (!T->stop.load(std::memory_order_acquire)) {
        CpuTierRequest* req = T->layers[layer].req;
        const uint32_t expected = T->expected_seq[layer];
        const uint32_t s = __atomic_load_n(&req->step, __ATOMIC_ACQUIRE);
        if (s == expected) {
            serve(*T, layer, expected);
            T->expected_seq[layer] = expected + 1;
            if (layer == 0) T->steps_done.fetch_add(1, std::memory_order_relaxed);
            T->expected.store(expected + 1);
            T->layer_cursor.store(layer);
            // stay on this layer (the next layer in forward order is usually the next to come, but a replay may
            // publish the same layer again): the round-robin below finds whichever is ready
        } else if (s > expected && s - expected < 0x40000000u) {
            // a request we never served (the GPU skipped ahead): count it and resync this layer
            T->errors.fetch_add(1, std::memory_order_relaxed);
            T->expected_seq[layer] = s;
        }
        if (++layer == T->n_layers) layer = 0;
        _mm_pause();
    }
    T->running.store(false);
}

}  // namespace

extern "C" {

struct CpuTierConfig {
    int n_layers;
    int n_experts;
    int n_cpus;
    const int* cpu_ids;
    int dispatcher_cpu;
    float vs1, vs2;
};

void* cpu_tier_create(const CpuTierConfig* cfg) {
    if (!cfg || cfg->n_layers < 1 || cfg->n_experts < 1) return nullptr;
    Tier* T = new Tier();
    T->n_layers = cfg->n_layers;
    T->n_experts = cfg->n_experts;
    T->vs1 = cfg->vs1;
    T->vs2 = cfg->vs2;
    T->dispatcher_cpu = cfg->dispatcher_cpu;
    T->pool = moe_b12x_pool_create(cfg->n_cpus, cfg->cpu_ids);
    if (!T->pool) { delete T; return nullptr; }
    T->layers.resize(cfg->n_layers);
    for (auto& L : T->layers) {
        L.records.assign(cfg->n_experts, nullptr);
        L.cold.assign(cfg->n_experts, 0);
        L.slot_of.assign(cfg->n_experts, -1);
    }
    return T;
}

void cpu_tier_destroy(void* tier) {
    Tier* T = (Tier*)tier;
    if (!T) return;
    if (T->running.load()) {
        T->stop.store(true);
        T->thread.join();
    }
    if (T->tail.on) {
        T->tail.ring.close();
        for (auto& L : T->layers) if (L.fd >= 0) close(L.fd);
    }
    moe_b12x_pool_destroy(T->pool);
    delete T;
}

// ---- the store tail ------------------------------------------------------------------------------------------------
// arena: n_slots pinned records (the first ones may already hold records: cpu_tier_set_record() with pointers into the
// arena registers them as resident; the rest are spares); per_layer = N resident records per layer; prefill_arena:
// prefill_sets x prefill_set_slots pinned records for the prefill streaming's cold records (may be null: no prefill)
// file_layer: the store file index of every tier layer (null = the tier's layer index = the model layer)
int cpu_tier_store_tail_init(void* tier, const char* dir, int rank, uint8_t* arena, int64_t n_slots, int per_layer, int qd,
                             uint8_t* prefill_arena, int prefill_sets, int prefill_set_slots, const int32_t* file_layer) {
    Tier* T = (Tier*)tier;
    if (!T || !arena || n_slots < 1 || per_layer < 1) return -1;
    Tail& X = T->tail;
    X.dir = dir; X.rank = rank; X.arena = arena; X.n_slots = n_slots; X.per_layer = per_layer; X.qd = (unsigned)std::max(1, qd);
    X.slots.assign((size_t)n_slots, Slot());
    X.free_slots.clear();
    for (int64_t s = n_slots - 1; s >= 0; --s) X.free_slots.push_back((int32_t)s);
    for (int l = 0; l < T->n_layers; ++l) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/rank%d/layer%02d.bin", dir, rank, file_layer ? file_layer[l] : l);
        T->layers[l].fd = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (T->layers[l].fd < 0) { fprintf(stderr, "[cpu_tier] store tail: open %s failed: %s\n", path, strerror(errno)); return -2; }
    }
    const int r = X.ring.init(X.qd * 2);
    if (r != 0) { fprintf(stderr, "[cpu_tier] store tail: io_uring_setup failed: %s (docker seccomp? docker/sm120/seccomp-io_uring.json)\n", strerror(-r)); return -3; }
    X.prefill_arena = prefill_arena; X.prefill_sets = prefill_sets; X.prefill_set_slots = prefill_set_slots;
    X.on = true;
    return 0;
}

// expert e of layer l lives in the store file only (read on a miss); the pointer table entry stays null until it is
// fetched. A record already registered with a pointer into the arena becomes the layer's resident slot.
int cpu_tier_set_record_cold(void* tier, int layer, int expert) {
    Tier* T = (Tier*)tier;
    if (!T || !T->tail.on || layer < 0 || layer >= T->n_layers || expert < 0 || expert >= T->n_experts) return -1;
    T->layers[layer].cold[expert] = 1;
    return 0;
}

// after the warm records were registered with cpu_tier_set_record(): adopt every pointer into the arena as a resident
// LRU slot of its layer (load order = LRU order); returns the number adopted
int64_t cpu_tier_store_tail_adopt(void* tier) {
    Tier* T = (Tier*)tier;
    if (!T || !T->tail.on) return -1;
    Tail& X = T->tail;
    std::lock_guard<std::mutex> g(X.mu);
    std::vector<uint8_t> used((size_t)X.n_slots, 0);
    int64_t n = 0;
    for (int l = 0; l < T->n_layers; ++l) {
        Layer& L = T->layers[l];
        for (int e = 0; e < T->n_experts; ++e) {
            const uint8_t* p = L.records[e];
            if (!p) continue;
            const int64_t off = p - X.arena;
            if (off < 0 || off % REC != 0 || off / REC >= X.n_slots) continue;   // not an arena record (dummy mode)
            const int s = (int)(off / REC);
            Slot& S = X.slots[s];
            S.expert = e; S.layer = l; S.pins = 0;
            L.slot_of[e] = s;
            L.cold[e] = 1;   // evictable: it can come back from the file
            lru_push_tail(*T, L, s);
            ++L.n_resident;
            used[s] = 1;
            ++n;
        }
    }
    X.free_slots.clear();
    for (int64_t s = X.n_slots - 1; s >= 0; --s) if (!used[s]) X.free_slots.push_back((int32_t)s);
    return n;
}

// the record's pinned host pointer for a copy (promotion, debug reference): resident -> pinned; cold -> read now
// (pread, O_DIRECT) into a spare slot, inserted and pinned. 0 = not available. Release with cpu_tier_record_release
// once the copy is done.
int64_t cpu_tier_record_acquire(void* tier, int layer, int expert) {
    Tier* T = (Tier*)tier;
    if (!T || layer < 0 || layer >= T->n_layers || expert < 0 || expert >= T->n_experts) return 0;
    Layer& L = T->layers[layer];
    if (!T->tail.on) return (int64_t)(uintptr_t)L.records[expert];
    Tail& X = T->tail;
    int s = -1;
    {
        std::lock_guard<std::mutex> g(X.mu);
        s = L.slot_of[expert];
        if (s >= 0) { ++X.slots[s].pins; return (int64_t)(uintptr_t)(X.arena + (int64_t)s * REC); }
        if (!L.cold[expert] || X.free_slots.empty()) return 0;
        s = X.free_slots.back();
        X.free_slots.pop_back();
    }
    uint8_t* dst = X.arena + (int64_t)s * REC;
    int64_t done = 0;
    const int64_t t0 = now_ns();
    while (done < REC) {
        const ssize_t r = pread(L.fd, dst + done, (size_t)(REC - done), (int64_t)expert * REC + done);
        if (r <= 0) break;
        done += r;
    }
    std::lock_guard<std::mutex> g(X.mu);
    if (done != REC) { X.io_errors.fetch_add(1); X.free_slots.push_back(s); return 0; }
    X.sync_fetches.fetch_add(1);
    X.fetch_ns.fetch_add(now_ns() - t0);
    lru_insert(*T, layer, expert, s);
    const int s2 = L.slot_of[expert];
    ++X.slots[s2].pins;
    return (int64_t)(uintptr_t)(X.arena + (int64_t)s2 * REC);
}

void cpu_tier_record_release(void* tier, int layer, int expert) {
    Tier* T = (Tier*)tier;
    if (!T || !T->tail.on || layer < 0 || layer >= T->n_layers || expert < 0 || expert >= T->n_experts) return;
    Tail& X = T->tail;
    std::lock_guard<std::mutex> g(X.mu);
    const int s = T->layers[layer].slot_of[expert];
    if (s >= 0 && X.slots[s].pins) --X.slots[s].pins;
    lru_trim(*T, T->layers[layer]);
}

// the prefill streaming's host pointers for `n` experts of `layer`: resident records pinned in place (released by
// cpu_tier_prefill_release(set)), cold records read (pread, O_DIRECT, `nthreads` readers) into the prefill arena's set
// `set` (slots 0..n-1 of the set; n <= prefill_set_slots). Blocking. Returns the number read from NVMe, or < 0.
int64_t cpu_tier_prefill_fetch(void* tier, int layer, const int32_t* experts, int n, int set, int64_t* out_ptrs, int nthreads) {
    Tier* T = (Tier*)tier;
    if (!T || !T->tail.on || layer < 0 || layer >= T->n_layers || n < 0) return -1;
    Tail& X = T->tail;
    if (!X.prefill_arena || set < 0 || set >= X.prefill_sets || n > X.prefill_set_slots) return -2;
    Layer& L = T->layers[layer];
    std::vector<int> cold_i;
    {
        std::lock_guard<std::mutex> g(X.mu);
        for (int i = 0; i < n; ++i) {
            const int e = experts[i];
            if (e < 0 || e >= T->n_experts) return -3;
            const int s = L.slot_of[e];
            if (s >= 0) { ++X.slots[s].pins; out_ptrs[i] = (int64_t)(uintptr_t)(X.arena + (int64_t)s * REC); }
            else if (L.cold[e]) { out_ptrs[i] = 0; cold_i.push_back(i); }
            else return -4;
        }
    }
    const int64_t t0 = now_ns();
    std::atomic<int64_t> next{0}, failed{0};
    std::vector<std::thread> th;
    const int nt = std::max(1, std::min(nthreads, 16));
    uint8_t* base = X.prefill_arena + ((int64_t)set * X.prefill_set_slots) * REC;
    for (int t = 0; t < nt; ++t) {
        th.emplace_back([&]() {
            for (;;) {
                const int64_t k = next.fetch_add(1);
                if (k >= (int64_t)cold_i.size()) break;
                const int i = cold_i[(size_t)k];
                uint8_t* dst = base + (int64_t)i * REC;
                int64_t done = 0;
                while (done < REC) {
                    const ssize_t r = pread(L.fd, dst + done, (size_t)(REC - done), (int64_t)experts[i] * REC + done);
                    if (r <= 0) { failed.fetch_add(1); break; }
                    done += r;
                }
                out_ptrs[i] = (int64_t)(uintptr_t)dst;
            }
        });
    }
    for (auto& x : th) x.join();
    X.prefill_fetches.fetch_add((int64_t)cold_i.size());
    X.prefill_ns.fetch_add(now_ns() - t0);
    if (failed.load()) { X.io_errors.fetch_add(failed.load()); return -5; }
    return (int64_t)cold_i.size();
}

// unpin the resident records a prefill round of `layer` used (cpu_tier_prefill_fetch pinned them)
void cpu_tier_prefill_release(void* tier, int layer, const int32_t* experts, int n) {
    Tier* T = (Tier*)tier;
    if (!T || !T->tail.on || layer < 0 || layer >= T->n_layers) return;
    Tail& X = T->tail;
    Layer& L = T->layers[layer];
    std::lock_guard<std::mutex> g(X.mu);
    for (int i = 0; i < n; ++i) {
        const int e = experts[i];
        if (e < 0 || e >= T->n_experts) continue;
        const int s = L.slot_of[e];
        if (s >= 0 && X.slots[s].pins) --X.slots[s].pins;
    }
    lru_trim(*T, L);
}

// is the record of (layer, expert) in DRAM right now (1), cold (0) or absent (-1)
int cpu_tier_record_state(void* tier, int layer, int expert) {
    Tier* T = (Tier*)tier;
    if (!T || layer < 0 || layer >= T->n_layers || expert < 0 || expert >= T->n_experts) return -1;
    Layer& L = T->layers[layer];
    if (L.records[expert]) return 1;
    return (T->tail.on && L.cold[expert]) ? 0 : -1;
}

// [0] fetches (dispatcher, async) [1] fetch ns (sum, issue -> last landing per request) [2] fetch wait ns beyond the
// resident compute [3] sync fetches (acquire) [4] prefill fetches [5] prefill ns [6] io errors [7] evictions
// [8] resident serves [9] resident records now [10] spare slots now
void cpu_tier_tail_stats(void* tier, int64_t* out, int reset) {
    Tier* T = (Tier*)tier;
    if (!T) return;
    Tail& X = T->tail;
    std::atomic<int64_t>* f[] = {&X.fetches, &X.fetch_ns, &X.fetch_wait_ns, &X.sync_fetches, &X.prefill_fetches, &X.prefill_ns, &X.io_errors, &X.evictions, &X.resident_serves};
    for (int i = 0; i < 9; ++i) out[i] = reset ? f[i]->exchange(0) : f[i]->load();
    int64_t res = 0;
    for (auto& L : T->layers) res += L.n_resident;
    out[9] = res;
    std::lock_guard<std::mutex> g(X.mu);
    out[10] = (int64_t)X.free_slots.size();
}

int cpu_tier_set_mailboxes(void* tier, int layer, void* req, void* ack, void* staging) {
    Tier* T = (Tier*)tier;
    if (!T || layer < 0 || layer >= T->n_layers) return -1;
    T->layers[layer].req = (CpuTierRequest*)req;
    T->layers[layer].ack = (CpuTierAck*)ack;
    T->layers[layer].staging = (uint16_t*)staging;
    return 0;
}

int cpu_tier_set_record(void* tier, int layer, int expert, const uint8_t* record) {
    Tier* T = (Tier*)tier;
    if (!T || layer < 0 || layer >= T->n_layers || expert < 0 || expert >= T->n_experts) return -1;
    T->layers[layer].records[expert] = record;
    return 0;
}

int cpu_tier_start(void* tier, uint32_t first_step) {
    Tier* T = (Tier*)tier;
    if (!T || T->running.load()) return -1;
    for (auto& L : T->layers)
        if (!L.req || !L.ack || !L.staging) return -2;
    T->expected_seq.assign(T->n_layers, first_step);
    T->expected.store(first_step);
    T->stop.store(false);
    T->running.store(true);
    T->thread = std::thread(dispatcher_main, T);
    return 0;
}

// start with a per-layer first sequence number (after a fail-closed stop: the device counters + 1)
int cpu_tier_start_seq(void* tier, const uint32_t* first_seq) {
    Tier* T = (Tier*)tier;
    if (!T || T->running.load()) return -1;
    for (auto& L : T->layers)
        if (!L.req || !L.ack || !L.staging) return -2;
    T->expected_seq.assign(first_seq, first_seq + T->n_layers);
    T->expected.store(first_seq[0]);
    T->stop.store(false);
    T->running.store(true);
    T->thread = std::thread(dispatcher_main, T);
    return 0;
}

int cpu_tier_stop(void* tier) {
    Tier* T = (Tier*)tier;
    if (!T || !T->running.load()) return -1;
    T->stop.store(true);
    T->thread.join();
    return 0;
}

// out[0] steps done, [1] layers done, [2] pairs done, [3] busy ns, [4] errors, [5] expected step, [6] layer cursor
void cpu_tier_totals(void* tier, int64_t* out) {
    Tier* T = (Tier*)tier;
    if (!T) return;
    out[0] = T->steps_done.load();
    out[1] = T->layers_done.load();
    out[2] = T->pairs_done.load();
    out[3] = T->busy_ns.load();
    out[4] = T->errors.load();
    out[5] = T->expected.load();
    out[6] = T->layer_cursor.load();
}

// the last min(cap, ring) KPI entries, oldest first, 8 int64 each (step, layer, pairs, experts, t_publish_gpu,
// t_seen, t_done, status); returns the count
int cpu_tier_kpis(void* tier, int64_t* out, int cap) {
    Tier* T = (Tier*)tier;
    if (!T) return 0;
    const int64_t n = T->ring_n.load();
    const int64_t have = n < KPI_RING ? n : KPI_RING;
    const int64_t take = have < cap ? have : cap;
    for (int64_t i = 0; i < take; ++i) {
        const Kpi& K = T->ring[(n - take + i) % KPI_RING];
        int64_t* o = out + i * 8;
        o[0] = K.step; o[1] = K.layer; o[2] = K.n_pairs; o[3] = K.n_experts;
        o[4] = K.t_publish_gpu_ns; o[5] = K.t_seen_ns; o[6] = K.t_done_ns; o[7] = K.status;
    }
    return (int)take;
}

// the miss log entries with index >= *cursor (4 int32 each: step, layer, expert, m); advances *cursor; returns the
// count (entries older than the ring are lost: the count then starts at the oldest kept)
int cpu_tier_misses(void* tier, int64_t* cursor, int32_t* out, int cap) {
    Tier* T = (Tier*)tier;
    if (!T) return 0;
    const int64_t n = T->miss_n.load(std::memory_order_acquire);
    int64_t from = *cursor;
    if (n - from > MISS_RING) from = n - MISS_RING;
    int64_t take = n - from;
    if (take > cap) take = cap;
    for (int64_t i = 0; i < take; ++i) {
        const Miss& M = T->misses[(from + i) % MISS_RING];
        out[i * 4] = M.step; out[i * 4 + 1] = M.layer; out[i * 4 + 2] = M.expert; out[i * 4 + 3] = M.m;
    }
    *cursor = from + take;
    return (int)take;
}

// the use log entries with index >= *cursor: per entry 3 int32 (step, layer, n) + 64 int16 ids, written to `out`
// as int32[3] followed by int16[64] packed into 32 int32 (67 int32 per entry); advances *cursor; returns the count
int cpu_tier_uses(void* tier, int64_t* cursor, int32_t* out, int cap) {
    Tier* T = (Tier*)tier;
    if (!T) return 0;
    const int64_t n = T->use_n.load(std::memory_order_acquire);
    int64_t from = *cursor;
    if (n - from > USE_RING) from = n - USE_RING;
    int64_t take = n - from;
    if (take > cap) take = cap;
    for (int64_t i = 0; i < take; ++i) {
        const Use& U = T->uses[(from + i) % USE_RING];
        int32_t* o = out + i * 67;
        o[0] = U.step; o[1] = U.layer; o[2] = U.n;
        memcpy(o + 3, U.ids, sizeof(int16_t) * CPU_TIER_MAX_PAIRS);
    }
    *cursor = from + take;
    return (int)take;
}

// The expert store loader: for every (layer, expert) with mask[layer * n_experts + expert] != 0, read the record from
// <dir>/rank<rank>/layer<LL>.bin (build_store.py's files) into arena slot k (dense, in (layer, expert) order) and
// write slot_of[layer * n_experts + expert] = k (-1 for the rest). `nthreads` readers with pread (the arena is the
// caller's: pinned or not). Returns the number of records loaded, or a negative errno-ish code.
int64_t cpu_tier_store_load(const char* dir, int rank, int n_layers, int n_experts, const uint8_t* mask, uint8_t* arena,
                            int64_t arena_bytes, int32_t* slot_of, int nthreads) {
    const int64_t REC = 4700160;
    std::vector<std::pair<int, int>> jobs;
    for (int l = 0; l < n_layers; ++l)
        for (int e = 0; e < n_experts; ++e) {
            const int64_t i = (int64_t)l * n_experts + e;
            if (mask[i]) { slot_of[i] = (int32_t)jobs.size(); jobs.push_back({l, e}); }
            else slot_of[i] = -1;
        }
    if ((int64_t)jobs.size() * REC > arena_bytes) return -1;
    std::atomic<int64_t> next{0}, failed{0};
    std::vector<std::thread> th;
    const int nt = std::max(1, std::min(nthreads, 64));
    for (int t = 0; t < nt; ++t) {
        th.emplace_back([&, t]() {
            int fd = -1, fd_layer = -1;
            for (;;) {
                const int64_t k = next.fetch_add(1);
                if (k >= (int64_t)jobs.size()) break;
                const int l = jobs[k].first, e = jobs[k].second;
                if (fd_layer != l) {
                    if (fd >= 0) close(fd);
                    char path[1024];
                    snprintf(path, sizeof(path), "%s/rank%d/layer%02d.bin", dir, rank, l);
                    fd = open(path, O_RDONLY);
                    fd_layer = l;
                    if (fd < 0) { failed.fetch_add(1); continue; }
                }
                uint8_t* dst = arena + k * REC;
                int64_t off = (int64_t)e * REC, done = 0;
                while (done < REC) {
                    const ssize_t r = pread(fd, dst + done, (size_t)(REC - done), off + done);
                    if (r <= 0) { failed.fetch_add(1); break; }
                    done += r;
                }
            }
            if (fd >= 0) close(fd);
        });
    }
    for (auto& x : th) x.join();
    if (failed.load()) return -2;
    return (int64_t)jobs.size();
}

int64_t cpu_tier_request_bytes(void) { return (int64_t)sizeof(CpuTierRequest); }
int64_t cpu_tier_ack_bytes(void) { return (int64_t)sizeof(CpuTierAck); }
int64_t cpu_tier_staging_bytes(void) { return (int64_t)CPU_TIER_MAX_PAIRS * CPU_TIER_HID * 2; }
int64_t cpu_tier_now_ns(void) { return now_ns(); }

// the kernel pool's statistics (6 int64: FC2 blocks, FC2 blocks through the general path, worker 0's TSC cycles in
// prepare / FC1 / activation / FC2 incl. barrier waits); reset != 0 clears them
void cpu_tier_pool_stats(void* tier, int64_t* out, int reset) {
    Tier* T = (Tier*)tier;
    if (!T || !T->pool) return;
    moe_b12x_pool_stats(T->pool, out, reset);
}

}  // extern "C"
