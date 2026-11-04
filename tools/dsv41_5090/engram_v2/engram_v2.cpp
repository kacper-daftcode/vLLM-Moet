// engram_v2.cpp — the host side of Engram v2 ("the Engram tables on NVMe + a DRAM row cache + prefetch") for one TP
// rank: a fetcher thread watches the per-layer request mailboxes the GPU's publish kernel fills (pinned,
// device-mapped), serves every requested row from a DRAM row cache or from the table's bytes on NVMe (the checkpoint's
// own safetensors shard, O_DIRECT, io_uring at queue depth), writes the rows into the layer's pinned staging arrays
// and bumps the layer's ack, which the GPU's wait kernel spins on (bounded) before vLLM's own lookup kernel
// dequantizes the rows from a device copy of the staging. Bit-identity: the rows are the table's bytes, the
// dequantization is vLLM's kernel - only the source of the bytes changes.
//
// Row cache: per layer, set-associative (8 ways x 264-B rows, CLOCK replacement), keyed by the global row id, in
// plain (not pinned) host memory, sized by `cache_bytes` (default 2 GiB per rank). The hash ids have no locality
// (rolling-hash % prime), so the cache is row-granular and the NVMe reads are the smallest aligned windows that
// cover a row's 256 value bytes and its 8 scale bytes (512 B or 1 KB each; the table's byte offsets in the shard
// are not 512-aligned).
//
// Lookahead: engram_v2_prefetch(layer, ids, n) queues rows to be read into the cache in the background (the next
// prefill chunk's rows, hashed on the host); the fetcher issues them in small batches whenever no demand request is
// pending, and demand requests are always submitted as soon as they are seen (their completions are reaped
// first-class from the same ring).
#include <fcntl.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "engram_v2_abi.h"
#include "uring.h"

namespace {

constexpr int WAYS = 8;
constexpr int ROW = ENGRAM_V2_ROW_BYTES;
constexpr unsigned SECTOR = 512;

static inline int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

struct alignas(64) Set {
    int32_t tag[WAYS];     // row id + 1 (0 = empty)
    uint8_t ref[WAYS];     // CLOCK reference bits
    uint8_t hand;
    uint8_t pad[64 - WAYS * 4 - WAYS - 1];
    uint8_t data[WAYS][ROW];
};
static_assert(sizeof(Set) == 64 + WAYS * ROW, "set layout");

class RowCache {
public:
    void init(int64_t bytes) {
        if (bytes <= 0) return;
        int64_t n = bytes / (int64_t)sizeof(Set);
        int64_t p = 1;
        while (p * 2 <= n) p *= 2;
        nsets_ = p;
        shift_ = 64;
        while ((1LL << (64 - shift_)) < nsets_) --shift_;
        const size_t len = (size_t)nsets_ * sizeof(Set);
        void* m = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) { nsets_ = 0; return; }
        madvise(m, len, MADV_HUGEPAGE);   // best effort: the probes are random over the arena
        // pre-fault the whole arena now (one write per 4 KB; a first-touch fault of a 2 MB huge page costs ~150 us and
        // 18 of them on a decode step's inserts blew the hide window in the GPU harness): zeroes the tags too
        for (size_t off = 0; off < len; off += 4096) ((volatile uint8_t*)m)[off] = 0;
        sets_ = (Set*)m;
        len_ = len;
    }
    ~RowCache() { if (sets_) munmap(sets_, len_); }
    int64_t sets() const { return nsets_; }
    int64_t capacity_rows() const { return nsets_ * WAYS; }
    bool probe(int32_t id, uint8_t* out) {
        if (!sets_) return false;
        Set& s = sets_[index(id)];
        const int32_t t = id + 1;
        for (int w = 0; w < WAYS; ++w)
            if (s.tag[w] == t) {
                s.ref[w] = 1;
                memcpy(out, s.data[w], ROW);
                return true;
            }
        return false;
    }
    void insert(int32_t id, const uint8_t* row) {
        if (!sets_) return;
        Set& s = sets_[index(id)];
        const int32_t t = id + 1;
        for (int w = 0; w < WAYS; ++w)
            if (s.tag[w] == t) { memcpy(s.data[w], row, ROW); s.ref[w] = 1; return; }
        for (int w = 0; w < WAYS; ++w)
            if (s.tag[w] == 0) { s.tag[w] = t; s.ref[w] = 1; memcpy(s.data[w], row, ROW); ++filled_; return; }
        for (;;) {   // CLOCK: the first way without a reference bit, clearing the bits passed
            const int w = s.hand;
            s.hand = (uint8_t)((s.hand + 1) % WAYS);
            if (s.ref[w]) { s.ref[w] = 0; continue; }
            s.tag[w] = t; s.ref[w] = 1; memcpy(s.data[w], row, ROW);
            ++evictions_;
            return;
        }
    }
    int64_t filled() const { return filled_; }
    int64_t evictions() const { return evictions_; }

private:
    int64_t index(int32_t id) const {
        if (shift_ >= 64) return 0;
        return (int64_t)(((uint64_t)(uint32_t)id * 0x9E3779B97F4A7C15ULL) >> shift_);
    }
    Set* sets_ = nullptr;
    size_t len_ = 0;
    int64_t nsets_ = 0;
    int shift_ = 64;
    int64_t filled_ = 0, evictions_ = 0;
};

struct LayerCfg {
    std::string path;
    int64_t values_off = 0, scales_off = 0, n_rows = 0, vocab_start = 0, vocab_end = 0;
};

struct Layer {
    LayerCfg cfg;
    int fd = -1;
    EngramV2Req* req = nullptr;
    EngramV2Ack* ack = nullptr;
    uint8_t* st_values = nullptr;
    uint8_t* st_scales = nullptr;
    uint32_t expected = 1;
    RowCache cache;
};

// one distinct row being read (demand or prefetch)
struct Entry {
    int layer = 0;
    int32_t id = -1;
    uint8_t row[ROW];
    uint8_t halves = 0;        // bit 0 values done, bit 1 scales done
    int job = -1;              // demand: the job index; -1 = prefetch
    int first_dst = -1;        // demand: the first staging row; more in `extra`
    std::vector<int> extra;
    bool failed = false;
};

// an in-flight read: which entry and which half, and its bounce buffer
struct Slot {
    uint8_t* buf = nullptr;    // 1 KB aligned
    int entry = -1;
    int half = 0;              // 0 values, 1 scales
    uint32_t window_off = 0;   // byte offset of the row's bytes inside the window
    uint32_t want = 0;         // bytes wanted
    uint32_t len = 0;          // bytes read
};

struct Job {   // one demand request being served
    int layer = 0;
    uint32_t seq = 0;
    int64_t t_seen = 0;
    uint32_t n_rows = 0, n_hits = 0, n_misses = 0, n_reads = 0;
    int remaining = 0;         // entries not yet completed
    bool failed = false;
    uint32_t status = 0;       // the ack's status: 1 io failure, 2 malformed request
    bool active = false;
};

struct Stats {
    std::atomic<int64_t> requests{0}, rows{0}, hits{0}, misses{0}, reads{0}, bytes{0}, io_errors{0}, serve_ns{0},
        serve_max_ns{0}, small_requests{0}, small_serve_ns{0}, small_serve_max_ns{0}, big_requests{0}, big_serve_ns{0},
        big_serve_max_ns{0}, prefetch_rows{0}, prefetch_reads{0}, prefetch_hits{0}, resyncs{0}, zero_rows{0}, dup_rows{0},
        prefetch_queued{0};
    std::atomic<int64_t> small_hist[8] = {};   // small requests by serve time: <=50, 100, 200, 300, 500, 1000, 5000, > us
};

struct Prefetch {
    int layer;
    std::vector<int32_t> ids;
    size_t cursor = 0;
};

struct Fetcher {
    std::vector<Layer> layers;
    int64_t cache_bytes = 0;
    int fetch_cpu = -1;
    unsigned qd = 64;
    bool direct = true;
    int max_rows = 0;
    unsigned prefetch_batch = 16;
    std::thread thread;
    std::atomic<bool> running{false}, stop{false};
    Stats stats;
    // io state (fetcher thread only)
    uring::Ring ring;
    std::vector<Slot> slots;
    std::vector<int> free_slots;
    std::vector<Entry> entries;
    std::vector<int> free_entries;
    std::vector<Job> jobs;           // one per layer at most (a layer's requests are sequential)
    std::unordered_map<int64_t, int> pending;   // (layer << 32 | id) -> entry index, for dedup of in-flight rows
    // prefetch queue (any thread -> fetcher)
    std::mutex pf_mu;
    std::vector<Prefetch> pf_queue;
    int ring_err = 0;
};

static void pin_to_cpu(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static inline int64_t key_of(int layer, int32_t id) { return ((int64_t)layer << 32) | (uint32_t)id; }

static int alloc_entry(Fetcher& F) {
    if (F.free_entries.empty()) { F.entries.emplace_back(); return (int)F.entries.size() - 1; }
    const int e = F.free_entries.back();
    F.free_entries.pop_back();
    Entry& E = F.entries[e];
    E.halves = 0; E.job = -1; E.first_dst = -1; E.extra.clear(); E.failed = false;
    return e;
}

// issue the two reads of entry e (values, scales); returns false when no slot is free (the caller retries later)
static bool issue_entry(Fetcher& F, int e) {
    if (F.free_slots.size() < 2) return false;
    Entry& E = F.entries[e];
    Layer& L = F.layers[E.layer];
    const int64_t offs[2] = {L.cfg.values_off + (int64_t)E.id * ENGRAM_V2_ROW_VALUES, L.cfg.scales_off + (int64_t)E.id * ENGRAM_V2_ROW_SCALES};
    const uint32_t wants[2] = {ENGRAM_V2_ROW_VALUES, ENGRAM_V2_ROW_SCALES};
    for (int h = 0; h < 2; ++h) {
        const int s = F.free_slots.back();
        F.free_slots.pop_back();
        Slot& S = F.slots[s];
        const int64_t start = offs[h] / SECTOR * SECTOR;
        const int64_t end = (offs[h] + wants[h] + SECTOR - 1) / SECTOR * SECTOR;
        S.entry = e; S.half = h; S.window_off = (uint32_t)(offs[h] - start); S.want = wants[h]; S.len = (uint32_t)(end - start);
        F.ring.prep_read(L.fd, S.buf, S.len, (uint64_t)start, (uint64_t)s);
    }
    return true;
}

static void finish_job(Fetcher& F, int j) {
    Job& J = F.jobs[j];
    Layer& L = F.layers[J.layer];
    const int64_t t_done = now_ns();
    EngramV2Ack* ack = L.ack;
    ack->status = J.failed ? J.status : 0u;
    ack->t_seen_ns = (uint64_t)J.t_seen;
    ack->t_done_ns = (uint64_t)t_done;
    ack->n_hits = J.n_hits;
    ack->n_misses = J.n_misses;
    ack->n_reads = J.n_reads;
    ack->n_rows = J.n_rows;
    __atomic_store_n(&ack->step, J.seq, __ATOMIC_RELEASE);
    Stats& S = F.stats;
    const int64_t dt = t_done - J.t_seen;
    S.requests.fetch_add(1); S.rows.fetch_add(J.n_rows); S.hits.fetch_add(J.n_hits); S.misses.fetch_add(J.n_misses);
    S.reads.fetch_add(J.n_reads); S.serve_ns.fetch_add(dt);
    if (dt > S.serve_max_ns.load()) S.serve_max_ns.store(dt);
    if (J.n_rows <= 64) {
        S.small_requests.fetch_add(1); S.small_serve_ns.fetch_add(dt);
        if (dt > S.small_serve_max_ns.load()) S.small_serve_max_ns.store(dt);
        const int64_t us = dt / 1000;
        const int b = us <= 50 ? 0 : us <= 100 ? 1 : us <= 200 ? 2 : us <= 300 ? 3 : us <= 500 ? 4 : us <= 1000 ? 5 : us <= 5000 ? 6 : 7;
        S.small_hist[b].fetch_add(1);
    } else {
        S.big_requests.fetch_add(1); S.big_serve_ns.fetch_add(dt);
        if (dt > S.big_serve_max_ns.load()) S.big_serve_max_ns.store(dt);
    }
    J.active = false;
}

// an entry's two halves are in: deliver the row (staging rows of its job, the cache)
static void complete_entry(Fetcher& F, int e) {
    Entry& E = F.entries[e];
    Layer& L = F.layers[E.layer];
    F.pending.erase(key_of(E.layer, E.id));
    if (!E.failed) L.cache.insert(E.id, E.row);
    if (E.job >= 0) {
        Job& J = F.jobs[E.job];
        if (E.failed) { J.failed = true; J.status = 1; }
        auto put = [&](int dst) {
            memcpy(L.st_values + (size_t)dst * ENGRAM_V2_ROW_VALUES, E.row, ENGRAM_V2_ROW_VALUES);
            memcpy(L.st_scales + (size_t)dst * ENGRAM_V2_ROW_SCALES, E.row + ENGRAM_V2_ROW_VALUES, ENGRAM_V2_ROW_SCALES);
        };
        if (E.first_dst >= 0) put(E.first_dst);
        for (int d : E.extra) put(d);
        if (--J.remaining == 0) finish_job(F, E.job);
    } else {
        F.stats.prefetch_rows.fetch_add(1);
    }
    F.free_entries.push_back(e);
}

static void reap_all(Fetcher& F) {
    uint64_t ud;
    int res;
    while (F.ring.reap(&ud, &res)) {
        Slot& S = F.slots[(size_t)ud];
        Entry& E = F.entries[S.entry];
        if (res < 0 || (uint32_t)res < S.window_off + S.want) {
            E.failed = true;
            F.stats.io_errors.fetch_add(1);
        } else {
            memcpy(E.row + (S.half ? ENGRAM_V2_ROW_VALUES : 0), S.buf + S.window_off, S.want);
            F.stats.bytes.fetch_add(S.len);
        }
        E.halves |= (uint8_t)(1 << S.half);
        if (E.job >= 0) ++F.jobs[E.job].n_reads;
        F.free_slots.push_back((int)ud);
        if (E.halves == 3) complete_entry(F, S.entry);
    }
}

// a request appeared on layer `layer`: cache hits straight into the staging, misses into entries + the ring
static void begin_job(Fetcher& F, int layer, uint32_t seq) {
    Layer& L = F.layers[layer];
    Job& J = F.jobs[layer];
    J.layer = layer; J.seq = seq; J.t_seen = now_ns(); J.n_hits = J.n_misses = J.n_reads = 0; J.failed = false; J.status = 0; J.active = true;
    const EngramV2Req* req = L.req;
    const uint32_t n = req->n_rows;
    J.n_rows = n;
    J.remaining = 1;   // guard: released after the issue loop so a job with only hits acks once
    if (n > (uint32_t)F.max_rows) {
        J.failed = true;
        J.status = 2;
    } else {
        std::vector<int> queued;   // entries of this job to issue
        std::unordered_map<int32_t, int> local;   // id -> entry of this job (dedup within the request)
        local.reserve(n * 2);
        uint8_t row[ROW];
        for (uint32_t i = 0; i < n; ++i) {
            const int32_t id = req->ids[i];
            if (id < L.cfg.vocab_start || id >= L.cfg.vocab_end) {
                memset(L.st_values + (size_t)i * ENGRAM_V2_ROW_VALUES, 0, ENGRAM_V2_ROW_VALUES);
                memset(L.st_scales + (size_t)i * ENGRAM_V2_ROW_SCALES, 0, ENGRAM_V2_ROW_SCALES);
                F.stats.zero_rows.fetch_add(1);
                continue;
            }
            auto it = local.find(id);
            if (it != local.end()) {   // the same row earlier in this request: completes with it
                F.entries[it->second].extra.push_back((int)i);
                F.stats.dup_rows.fetch_add(1);
                continue;
            }
            if (L.cache.probe(id, row)) {
                memcpy(L.st_values + (size_t)i * ENGRAM_V2_ROW_VALUES, row, ENGRAM_V2_ROW_VALUES);
                memcpy(L.st_scales + (size_t)i * ENGRAM_V2_ROW_SCALES, row + ENGRAM_V2_ROW_VALUES, ENGRAM_V2_ROW_SCALES);
                ++J.n_hits;
                continue;
            }
            ++J.n_misses;
            auto pit = F.pending.find(key_of(layer, id));
            if (pit != F.pending.end()) {
                // already in flight for a prefetch (or a previous job of another layer: impossible, ids are per layer):
                // adopt it into this job
                Entry& E = F.entries[pit->second];
                if (E.job < 0) { E.job = layer; E.first_dst = (int)i; ++J.remaining; local[id] = pit->second; }
                else { E.extra.push_back((int)i); }
                continue;
            }
            const int e = alloc_entry(F);
            Entry& E = F.entries[e];
            E.layer = layer; E.id = id; E.job = layer; E.first_dst = (int)i;
            F.pending[key_of(layer, id)] = e;
            local[id] = e;
            ++J.remaining;
            queued.push_back(e);
        }
        // issue (the ring may be smaller than the request: issue as slots free up)
        size_t k = 0;
        while (k < queued.size()) {
            while (k < queued.size() && issue_entry(F, queued[k])) ++k;
            if (F.ring.pending_sqes) {
                const int r = F.ring.submit(0);
                if (r < 0) { F.ring_err = r; J.failed = true; J.status = 1; break; }
            }
            if (k < queued.size()) {
                F.ring.wait(1);
                reap_all(F);
            }
        }
    }
    if (--J.remaining == 0) finish_job(F, layer);
}

// background work: a small batch of prefetch reads when the ring has room
static void prefetch_step(Fetcher& F) {
    const size_t room = std::min<size_t>(F.prefetch_batch, F.free_slots.size() / 2);
    if (room == 0) return;
    std::vector<std::pair<int, int32_t>> batch;
    {
        std::lock_guard<std::mutex> g(F.pf_mu);
        while (!F.pf_queue.empty() && batch.size() < room) {
            Prefetch& P = F.pf_queue.front();
            while (P.cursor < P.ids.size() && batch.size() < room) {
                const int32_t id = P.ids[P.cursor++];
                batch.emplace_back(P.layer, id);
            }
            if (P.cursor >= P.ids.size()) F.pf_queue.erase(F.pf_queue.begin());
        }
    }
    if (batch.empty()) return;
    uint8_t row[ROW];
    bool issued = false;
    for (auto& [layer, id] : batch) {
        Layer& L = F.layers[layer];
        if (id < L.cfg.vocab_start || id >= L.cfg.vocab_end) continue;
        if (L.cache.probe(id, row)) { F.stats.prefetch_hits.fetch_add(1); continue; }
        if (F.pending.count(key_of(layer, id))) continue;
        const int e = alloc_entry(F);
        Entry& E = F.entries[e];
        E.layer = layer; E.id = id; E.job = -1;
        F.pending[key_of(layer, id)] = e;
        if (!issue_entry(F, e)) { F.pending.erase(key_of(layer, id)); F.free_entries.push_back(e); break; }
        F.stats.prefetch_reads.fetch_add(2);
        issued = true;
    }
    if (issued) {
        const int r = F.ring.submit(0);
        if (r < 0) F.ring_err = r;
    }
}

// One iteration of the fetcher: the mailboxes (layer priority), the completions, a prefetch batch when idle. Called in
// a loop by the fetcher's own thread, or - the production mode - by the CPU tier's dispatcher thread through its poll
// hook (engram_v2_poll): the dispatcher already spins on a dedicated core (8r+1), whereas a thread of our own shares
// 8r+33 with the engine core / API server and vLLM's auxiliary threads and lost the ~200 us window to preemption on the
// 11-04 TP4 boot (a quarter of the cold-row requests took 500-1000 us, a few 9 ms). Never blocks except in a big
// request's issue loop (the ring full: prefill-size requests, when the tier has no requests of its own).
static void poll_once(Fetcher& F) {
    {
        bool busy = false;
        for (size_t l = 0; l < F.layers.size(); ++l) {
            Layer& L = F.layers[l];
            if (!L.req) continue;
            // priority by layer: the two Engram layers' requests are published back to back, but layer 1's rows are
            // consumed ~200 us after the hashes and layer 14's ~2.5 ms after - its reads wait until layer 1's job is
            // done (24 + 24 reads at once would push layer 1's completion from ~131 to ~195 us)
            bool earlier_active = false;
            for (size_t k = 0; k < l; ++k) earlier_active |= F.jobs[k].active;
            if (earlier_active) continue;
            const uint32_t s = __atomic_load_n(&L.req->step, __ATOMIC_ACQUIRE);
            if (s == L.expected && !F.jobs[l].active) {
                begin_job(F, (int)l, s);
                L.expected = s + 1;
                busy = true;
            } else if (s > L.expected && s - L.expected < 0x40000000u && !F.jobs[l].active) {
                F.stats.resyncs.fetch_add(1);   // a request we never served (restart after a fail-closed stop)
                L.expected = s;
            }
        }
        if (F.ring.in_flight && F.ring.cq_ready()) { reap_all(F); busy = true; }
        if (!busy) {
            bool any_job = false;
            for (auto& J : F.jobs) any_job |= J.active;
            if (!any_job) prefetch_step(F);
        }
    }
}

// The fetcher's own thread (the standalone mode: tests, the harness, a serve without the CPU tier). Spins like the
// tier's dispatcher: a decode step gives it ~200 us, so neither the mailbox poll nor the reap may sleep. Pinned to
// `fetch_cpu`; the tier's _pin_process re-pins every thread of the worker when the tier starts, so the affinity is
// re-asserted now and then.
static void fetcher_main(Fetcher* Fp) {
    Fetcher& F = *Fp;
    pin_to_cpu(F.fetch_cpu);
    uint32_t iter = 0;
    while (!F.stop.load(std::memory_order_acquire)) {
        if ((++iter & 0xFFFFF) == 0) pin_to_cpu(F.fetch_cpu);
        poll_once(F);
        _mm_pause();
    }
    F.running.store(false);
}

}  // namespace

extern "C" {

struct EngramV2LayerCfg {
    const char* path;
    int64_t values_off, scales_off, n_rows, vocab_start, vocab_end;
};

struct EngramV2Config {
    int n_layers;
    const EngramV2LayerCfg* layers;
    int64_t cache_bytes;      // per rank, split evenly over the layers
    int fetch_cpu;            // -1 = unpinned
    int queue_depth;          // ring entries / in-flight reads
    int direct;               // O_DIRECT (0 = buffered: tmpfs test files)
    int max_rows;             // staging rows per layer
    int prefetch_batch;       // reads per background batch (0 = default 16)
};

void* engram_v2_create(const EngramV2Config* cfg) {
    if (!cfg || cfg->n_layers < 1 || cfg->max_rows < 1) return nullptr;
    Fetcher* F = new Fetcher();
    F->cache_bytes = cfg->cache_bytes;
    F->fetch_cpu = cfg->fetch_cpu;
    F->qd = (unsigned)std::max(8, cfg->queue_depth);
    F->direct = cfg->direct != 0;
    F->max_rows = cfg->max_rows;
    if (cfg->prefetch_batch > 0) F->prefetch_batch = (unsigned)cfg->prefetch_batch;
    F->layers.resize(cfg->n_layers);
    F->jobs.resize(cfg->n_layers);
    for (int l = 0; l < cfg->n_layers; ++l) {
        Layer& L = F->layers[l];
        const EngramV2LayerCfg& c = cfg->layers[l];
        L.cfg.path = c.path; L.cfg.values_off = c.values_off; L.cfg.scales_off = c.scales_off; L.cfg.n_rows = c.n_rows;
        L.cfg.vocab_start = c.vocab_start; L.cfg.vocab_end = c.vocab_end;
        L.fd = open(c.path, O_RDONLY | O_CLOEXEC | (F->direct ? O_DIRECT : 0));
        if (L.fd < 0) { fprintf(stderr, "[engram_v2] open %s failed: %s\n", c.path, strerror(errno)); delete F; return nullptr; }
        struct stat st;
        if (fstat(L.fd, &st) != 0 || c.values_off + c.n_rows * ENGRAM_V2_ROW_VALUES > st.st_size ||
            c.scales_off + c.n_rows * ENGRAM_V2_ROW_SCALES > st.st_size) {
            fprintf(stderr, "[engram_v2] %s: table extent outside the file\n", c.path);
            delete F; return nullptr;
        }
        L.cache.init(cfg->cache_bytes / cfg->n_layers);
    }
    // the ring and the bounce buffers (the fetcher thread owns them; created here so a failure is reported)
    const int r = F->ring.init(F->qd);
    if (r != 0) { fprintf(stderr, "[engram_v2] io_uring_setup failed: %s (docker seccomp? use docker/sm120/seccomp-io_uring.json)\n", strerror(-r)); delete F; return nullptr; }
    const unsigned nslots = F->ring.entries;
    F->slots.resize(nslots);
    uint8_t* arena = (uint8_t*)uring::alloc_aligned((size_t)nslots * 1536, 4096);
    if (!arena) { delete F; return nullptr; }
    for (unsigned i = 0; i < nslots; ++i) { F->slots[i].buf = arena + (size_t)i * 1536; F->free_slots.push_back((int)i); }
    std::reverse(F->free_slots.begin(), F->free_slots.end());
    return F;
}

int engram_v2_set_mailbox(void* h, int layer, void* req, void* ack, uint8_t* staging_values, uint8_t* staging_scales) {
    Fetcher* F = (Fetcher*)h;
    if (!F || layer < 0 || layer >= (int)F->layers.size()) return -1;
    Layer& L = F->layers[layer];
    L.req = (EngramV2Req*)req; L.ack = (EngramV2Ack*)ack; L.st_values = staging_values; L.st_scales = staging_scales;
    return 0;
}

int engram_v2_start(void* h, uint32_t first_seq) {
    Fetcher* F = (Fetcher*)h;
    if (!F || F->running.load()) return -1;
    for (auto& L : F->layers)
        if (!L.req || !L.ack || !L.st_values || !L.st_scales) return -2;
    for (auto& L : F->layers) L.expected = first_seq;
    F->stop.store(false);
    F->running.store(true);
    F->thread = std::thread(fetcher_main, F);
    return 0;
}

int engram_v2_stop(void* h) {
    Fetcher* F = (Fetcher*)h;
    if (!F || !F->running.load()) return -1;
    F->stop.store(true);
    if (F->thread.joinable()) F->thread.join();
    F->running.store(false);
    return 0;
}

// the polled mode: no thread of its own - the owner (the CPU tier's dispatcher, through cpu_tier_set_poll_hook) calls
// engram_v2_poll() from its loop; one caller at a time. Switching from the thread mode: engram_v2_stop() first (the
// mailbox state persists: a request pending at the switch is served by the first poll).
int engram_v2_start_polled(void* h, uint32_t first_seq) {
    Fetcher* F = (Fetcher*)h;
    if (!F || F->running.load()) return -1;
    for (auto& L : F->layers)
        if (!L.req || !L.ack || !L.st_values || !L.st_scales) return -2;
    if (first_seq) for (auto& L : F->layers) L.expected = first_seq;
    F->stop.store(false);
    F->running.store(true);   // "running" = serving (no thread)
    return 0;
}

void engram_v2_poll(void* h) {
    Fetcher* F = (Fetcher*)h;
    if (!F || !F->running.load(std::memory_order_relaxed)) return;
    poll_once(*F);
}

void engram_v2_destroy(void* h) {
    Fetcher* F = (Fetcher*)h;
    if (!F) return;
    if (F->running.load()) { F->stop.store(true); F->thread.join(); }
    F->ring.close();
    for (auto& L : F->layers) if (L.fd >= 0) close(L.fd);
    if (!F->slots.empty()) free(F->slots[0].buf);
    delete F;
}

int64_t engram_v2_request_bytes(int max_rows) { return engram_v2_request_bytes_for(max_rows); }
int64_t engram_v2_ack_bytes(void) { return (int64_t)sizeof(EngramV2Ack); }

// queue rows of `layer` to be read into the row cache in the background (the prefill lookahead); copies the ids
int engram_v2_prefetch(void* h, int layer, const int32_t* ids, int n) {
    Fetcher* F = (Fetcher*)h;
    if (!F || layer < 0 || layer >= (int)F->layers.size() || n < 0) return -1;
    if (n == 0) return 0;
    Prefetch P;
    P.layer = layer;
    P.ids.assign(ids, ids + n);
    std::lock_guard<std::mutex> g(F->pf_mu);
    F->pf_queue.push_back(std::move(P));
    F->stats.prefetch_queued.fetch_add(n);
    return 0;
}

// drop the queued prefetches (a request finished / was preempted)
void engram_v2_prefetch_clear(void* h) {
    Fetcher* F = (Fetcher*)h;
    if (!F) return;
    std::lock_guard<std::mutex> g(F->pf_mu);
    F->pf_queue.clear();
}

// stats: [0] requests [1] rows [2] hits [3] misses [4] reads [5] bytes [6] io_errors [7] serve_ns [8] serve_max_ns
// [9] small_requests [10] small_serve_ns [11] small_serve_max_ns [12] big_requests [13] big_serve_ns [14] big_serve_max_ns
// [15] prefetch_rows [16] prefetch_reads [17] prefetch_hits [18] resyncs [19] zero_rows [20] dup_rows [21] prefetch_queued
// [22..29] small histogram (<=50 100 200 300 500 1000 5000 > us) [30] cache rows filled (sum over layers)
// [31] cache evictions [32] cache capacity rows [33] ring error (errno); reset != 0 clears the counters
void engram_v2_stats(void* h, int64_t* out, int reset) {
    Fetcher* F = (Fetcher*)h;
    if (!F) return;
    Stats& S = F->stats;
    std::atomic<int64_t>* fields[] = {&S.requests, &S.rows, &S.hits, &S.misses, &S.reads, &S.bytes, &S.io_errors, &S.serve_ns,
                                      &S.serve_max_ns, &S.small_requests, &S.small_serve_ns, &S.small_serve_max_ns, &S.big_requests,
                                      &S.big_serve_ns, &S.big_serve_max_ns, &S.prefetch_rows, &S.prefetch_reads, &S.prefetch_hits,
                                      &S.resyncs, &S.zero_rows, &S.dup_rows, &S.prefetch_queued};
    for (int i = 0; i < 22; ++i) out[i] = reset ? fields[i]->exchange(0) : fields[i]->load();
    for (int i = 0; i < 8; ++i) out[22 + i] = reset ? S.small_hist[i].exchange(0) : S.small_hist[i].load();
    int64_t filled = 0, ev = 0, cap = 0;
    for (auto& L : F->layers) { filled += L.cache.filled(); ev += L.cache.evictions(); cap += L.cache.capacity_rows(); }
    out[30] = filled; out[31] = ev; out[32] = cap; out[33] = F->ring_err;
}

int64_t engram_v2_now_ns(void) { return now_ns(); }

// test helper (the fetcher not running): the row's bytes straight from the file through the same windowed reads
// (pread), for comparing against the fetcher's rows
int engram_v2_read_row_direct(void* h, int layer, int32_t id, uint8_t* out264) {
    Fetcher* F = (Fetcher*)h;
    if (!F || layer < 0 || layer >= (int)F->layers.size()) return -1;
    Layer& L = F->layers[layer];
    if (id < 0 || id >= L.cfg.n_rows) return -2;
    alignas(4096) uint8_t buf[2048];
    const int64_t offs[2] = {L.cfg.values_off + (int64_t)id * ENGRAM_V2_ROW_VALUES, L.cfg.scales_off + (int64_t)id * ENGRAM_V2_ROW_SCALES};
    const uint32_t wants[2] = {ENGRAM_V2_ROW_VALUES, ENGRAM_V2_ROW_SCALES};
    for (int hh = 0; hh < 2; ++hh) {
        const int64_t start = offs[hh] / SECTOR * SECTOR;
        const int64_t end = (offs[hh] + wants[hh] + SECTOR - 1) / SECTOR * SECTOR;
        const ssize_t r = pread(L.fd, buf, (size_t)(end - start), start);
        if (r < (ssize_t)(offs[hh] - start + wants[hh])) return -3;
        memcpy(out264 + (hh ? ENGRAM_V2_ROW_VALUES : 0), buf + (offs[hh] - start), wants[hh]);
    }
    return 0;
}

}  // extern "C"
