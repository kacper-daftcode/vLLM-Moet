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
#include <fcntl.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "cpu_tier_abi.h"

extern "C" {
void* moe_b12x_pool_create(int nthreads, const int* cpu_ids);
void moe_b12x_pool_destroy(void* pool);
int moe_b12x_experts_forward(void* pool, int n_experts, const uint8_t* const* records, const int* m_per_expert,
                             const int* token_idx, const float* pair_weight, const int* out_row, int n_tokens,
                             const uint8_t* aq, const uint8_t* as, uint16_t* out, int64_t out_stride, float vs1,
                             float vs2, uint16_t* proj_out, uint8_t* inter_q_out, uint8_t* inter_s_out);
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

struct Layer {
    CpuTierRequest* req = nullptr;
    CpuTierAck* ack = nullptr;
    uint16_t* staging = nullptr;
    std::vector<const uint8_t*> records;  // [n_experts]
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
    std::atomic<uint32_t> expected{1};
    std::atomic<int> layer_cursor{0};
    // totals
    std::atomic<int64_t> steps_done{0}, layers_done{0}, pairs_done{0}, busy_ns{0}, errors{0};
    // rings
    Kpi ring[KPI_RING];
    std::atomic<int64_t> ring_n{0};
    Miss misses[MISS_RING];
    std::atomic<int64_t> miss_n{0};
    Use uses[USE_RING];
    std::atomic<int64_t> use_n{0};
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
        int m_per[CPU_TIER_MAX_PAIRS], tok[CPU_TIER_MAX_PAIRS], orow[CPU_TIER_MAX_PAIRS];
        float pw[CPU_TIER_MAX_PAIRS];
        int last_e = -1;
        for (int k = 0; k < n; ++k) {
            const CpuTierPair& p = req->pairs[order[k]];
            if (p.expert != last_e) {
                last_e = p.expert;
                const uint8_t* r = p.expert < L.records.size() ? L.records[p.expert] : nullptr;
                if (!r) { status = 2; break; }  // not in the store
                recs[n_experts] = r;
                m_per[n_experts] = 0;
                ++n_experts;
            }
            m_per[n_experts - 1] += 1;
            tok[k] = p.token;
            pw[k] = p.weight;
            orow[k] = order[k];
        }
        if (status == 0) {
            const int rc = moe_b12x_experts_forward(T.pool, n_experts, recs, m_per, tok, pw, orow, n_tokens,
                                                    &req->a_values[0][0], &req->a_scales[0][0], L.staging,
                                                    CPU_TIER_HID, T.vs1, T.vs2, nullptr, nullptr, nullptr);
            if (rc != 0) status = 4;
            // the miss log (per distinct expert), for the placement manager
            int k = 0;
            for (int i = 0; i < n_experts; ++i) {
                const int64_t mi = T.miss_n.fetch_add(1, std::memory_order_relaxed);
                Miss& M = T.misses[mi % MISS_RING];
                M.step = (int32_t)step; M.layer = layer; M.expert = req->pairs[order[k]].expert; M.m = m_per[i];
                k += m_per[i];
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

static void dispatcher_main(Tier* T) {
    pin_to_cpu(T->dispatcher_cpu);
    uint32_t expected = T->expected.load();
    int layer = 0;
    while (!T->stop.load(std::memory_order_acquire)) {
        CpuTierRequest* req = T->layers[layer].req;
        const uint32_t s = __atomic_load_n(&req->step, __ATOMIC_ACQUIRE);
        if (s == expected) {
            serve(*T, layer, expected);
            if (++layer == T->n_layers) {
                layer = 0;
                ++expected;
                T->expected.store(expected);
                T->steps_done.fetch_add(1, std::memory_order_relaxed);
            }
            T->layer_cursor.store(layer);
        } else if (s > expected && s - expected < 0x40000000u) {
            // the GPU is ahead of us: a step we never saw (should not happen; count it and resync)
            T->errors.fetch_add(1, std::memory_order_relaxed);
            expected = s;
            T->expected.store(expected);
        } else {
            _mm_pause();
        }
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
    for (auto& L : T->layers) L.records.assign(cfg->n_experts, nullptr);
    return T;
}

void cpu_tier_destroy(void* tier) {
    Tier* T = (Tier*)tier;
    if (!T) return;
    if (T->running.load()) {
        T->stop.store(true);
        T->thread.join();
    }
    moe_b12x_pool_destroy(T->pool);
    delete T;
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
    T->expected.store(first_step);
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

}  // extern "C"
