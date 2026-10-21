// Deterministic exact top-k over rows of fp32 logits, for the DeepSeek-V4.1 DSA indexer on sm_120.
//
// Row r selects the k largest values among its columns [start_r, end_r); among equal values the lower column wins
// (-0.0 equals +0.0; every NaN ranks above +inf, as in torch.sort). The selected columns are written in ascending column
// order, then -1 up to k when the row has fewer than k columns. The result is a function of the input values only: the
// radix select counts with integer atomics (order-independent sums) and the compaction ranks columns with block scans.
//
// The threshold is found by an MSB-first radix select over order-preserving 32-bit keys (11/11/10 bits, shared-memory
// histograms). The passes stop early once the bucket holding the k-th key is taken whole, or when after the second pass
// its keys agree in the last 10 bits too (always so for bf16-valued logits), the ties then going to the lower columns.
// The compaction gives each warp a contiguous segment of columns: one pass counts, a scan over the warps places the
// segments, a second pass writes (lane order is column order). Rows are read as float4 when 16-byte aligned with unit
// stride.
//
// Few rows (decode): a cluster of C CTAs per row, each CTA on a contiguous slice of the row; C is the largest of 16, 8,
// 4, 2 for which the clusters of all rows are resident at once (by row count only, so one CUDA graph serves every
// context length). Every pass, each CTA histograms its slice; after a cluster barrier warp c of every CTA reads CTA c's
// group sums (32 bins each) through distributed shared memory, then, all taking the same group decision, CTA c's bins of
// the chosen group (one warp reading every CTA is several times slower: its remote reads do not overlap). The CTAs also
// keep, from their own histograms, how many of their columns lie above the threshold and on it, and add these into the
// later CTAs' counters before one last barrier; each CTA then writes its slice's columns at their final places. The
// first pass keeps the slice's keys in shared memory when they fit (the later passes and the compaction read them
// there). Rows of at most `small` columns are done by the cluster's first CTA alone (no cluster barrier).
// Many rows (prefill): one CTA of 256 threads per row.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cg = cooperative_groups;

#ifndef DET_TOPK_UNROLL
#define DET_TOPK_UNROLL 1  // float4s per thread per histogram step; deeper unrolling was slower for every row length
#endif

namespace {

constexpr int kVec = 4;
constexpr int kBins = 2048;
constexpr int kGroups = kBins / 32;
constexpr int kClusterThreads = 1024;

__device__ __forceinline__ uint32_t key_of(float x) {
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0xffffffffu;  // NaN
    if ((u & 0x7fffffffu) == 0u) u = 0u;                       // -0.0 == +0.0
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Keys of columns [base, base + 4) of a row; float4 when the row is 16-byte aligned with unit column stride.
__device__ __forceinline__ void load_keys4(const float* __restrict__ row, int64_t stride1, bool vec, int64_t n,
                                           int64_t base, uint32_t (&key)[4], bool (&valid)[4]) {
    if (vec && base + 3 < n) {
        const float4 f = __ldg(reinterpret_cast<const float4*>(row + base));
        key[0] = key_of(f.x);
        key[1] = key_of(f.y);
        key[2] = key_of(f.z);
        key[3] = key_of(f.w);
        valid[0] = valid[1] = valid[2] = valid[3] = true;
        return;
    }
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        valid[v] = base + v < n;
        key[v] = valid[v] ? key_of(row[(base + v) * stride1]) : 0u;
    }
}

// Exclusive scan over the CTA in thread order; *total receives the CTA sum. Ends with a barrier, so it can be called
// again right away.
template <int THREADS>
__device__ __forceinline__ uint32_t block_exclusive_scan(uint32_t v, uint32_t* ws, uint32_t* total) {
    constexpr int kWarps = THREADS / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) ws[warp] = x;
    __syncthreads();
    if (warp == 0) {
        const uint32_t w = lane < kWarps ? ws[lane] : 0u;
        uint32_t s = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        if (lane < kWarps) ws[lane] = s - w;
        if (lane == kWarps - 1) ws[kWarps] = s;
    }
    __syncthreads();
    const uint32_t r = ws[warp] + x - v;
    *total = ws[kWarps];
    __syncthreads();
    return r;
}

__device__ __forceinline__ uint32_t warp_inclusive_scan(uint32_t x) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    return x;
}

struct RowBounds {
    // prefill: explicit [starts[r], ends[r]) (starts may be null = 0); decode: ends from seq_lens like vLLM's
    // SparseIndexerTopk._row_ends (per-row lens when seq_lens has num_rows entries, else len - next_n + 1 + j).
    const int32_t* starts;
    const int32_t* ends;
    const int32_t* seq_lens;
    int64_t num_seq_lens;
    int64_t next_n;
    int64_t num_rows;
    int64_t width;
};

__device__ __forceinline__ void row_bounds(const RowBounds& b, int64_t r, int64_t* start, int64_t* end) {
    int64_t s = 0, e = 0;
    if (b.ends != nullptr) {
        s = b.starts != nullptr ? b.starts[r] : 0;
        e = b.ends[r];
    } else if (b.num_seq_lens == b.num_rows) {
        e = b.seq_lens[r];
    } else {
        const int64_t q = r / b.next_n, j = r % b.next_n;
        e = q < b.num_seq_lens ? b.seq_lens[q] - b.next_n + 1 + j : 0;
    }
    s = s < 0 ? 0 : s;
    e = e > b.width ? b.width : e;
    *start = s;
    *end = e < s ? s : e;
}

// Keys of columns [base, base + 4) from the shared-memory copy of the row (16-byte aligned, filled by the first pass).
__device__ __forceinline__ void smem_keys4(const uint32_t* skeys, int64_t n, int64_t base, uint32_t (&key)[4],
                                           bool (&valid)[4]) {
    if (base + 3 < n) {
        const uint4 v = *reinterpret_cast<const uint4*>(skeys + base);
        key[0] = v.x;
        key[1] = v.y;
        key[2] = v.z;
        key[3] = v.w;
        valid[0] = valid[1] = valid[2] = valid[3] = true;
        return;
    }
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        valid[v] = base + v < n;
        key[v] = valid[v] ? skeys[base + v] : 0u;
    }
}

// Histogram of (key >> shift) & dmask over the columns of [0, n) whose key matches prefix under pmask. Keys come from
// the row in global memory (store_keys: also kept in skeys for the later passes) or, when from_smem, from skeys. With
// lowor, bit d of the 2048-bit mask is set when a key counted in bin d has a nonzero bit under lowmask.
template <int THREADS, int UNROLL>
__device__ __forceinline__ void histogram(const float* __restrict__ row, int64_t stride1, bool vec, int64_t n,
                                          uint32_t prefix, uint32_t pmask, int shift, uint32_t dmask, uint32_t* hist,
                                          uint32_t* skeys = nullptr, bool store_keys = false, bool from_smem = false,
                                          uint32_t* lowor = nullptr, uint32_t lowmask = 0u) {
    for (int64_t q0 = 0; q0 < n; q0 += 4 * THREADS * UNROLL) {
        uint32_t keys[UNROLL][4];
        bool valid[UNROLL][4];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int64_t base = q0 + 4 * (u * THREADS + threadIdx.x);
            if (from_smem)
                smem_keys4(skeys, n, base, keys[u], valid[u]);
            else
                load_keys4(row, stride1, vec, n, base, keys[u], valid[u]);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int64_t base = q0 + 4 * (u * THREADS + threadIdx.x);
            if (store_keys) {
                if (base + 3 < n)
                    *reinterpret_cast<uint4*>(skeys + base) = make_uint4(keys[u][0], keys[u][1], keys[u][2], keys[u][3]);
                else
#pragma unroll
                    for (int v = 0; v < 4; ++v)
                        if (valid[u][v]) skeys[base + v] = keys[u][v];
            }
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                if (valid[u][v] && (keys[u][v] & pmask) == prefix) {
                    const uint32_t d = (keys[u][v] >> shift) & dmask;
                    atomicAdd(&hist[d], 1u);
                    if (lowor != nullptr && (keys[u][v] & lowmask) != 0u) atomicOr(&lowor[d >> 5], 1u << (d & 31));
                }
            }
        }
    }
}

// Writes the selected columns among [0, n) of `row` (column numbers offset by col_base) in ascending order. gt_off /
// eq_off: how many columns of the whole row before this range lie above the threshold / on it. Warp w owns one
// contiguous segment: a first pass counts, a scan over the warps places the segments, a second pass writes (skipping
// the steps without a selected column). ws: 2 * THREADS / 32 words. Ends without a barrier.
template <int THREADS>
__device__ __forceinline__ void compact(const float* __restrict__ row, int64_t stride1, bool vec, int64_t n, int shift,
                                        uint32_t p_hi, uint32_t k_rem, int k, uint32_t gt_off, uint32_t eq_off,
                                        int32_t* __restrict__ orow, int32_t col_base, uint32_t* ws,
                                        const uint32_t* skeys = nullptr) {
    constexpr int kWarps = THREADS / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t seg = ((n + kWarps - 1) / kWarps + 32 * kVec - 1) & ~static_cast<int64_t>(32 * kVec - 1);
    const int64_t w0 = min(n, warp * seg), w1 = min(n, w0 + seg);
    uint32_t cgt = 0, ceq = 0;
    for (int64_t c0 = w0 + kVec * lane; c0 < w1; c0 += 32 * kVec) {
        uint32_t key[kVec];
        bool valid[kVec];
        if (skeys != nullptr)
            smem_keys4(skeys, w1, c0, key, valid);
        else
            load_keys4(row, stride1, vec, w1, c0, key, valid);
#pragma unroll
        for (int v = 0; v < kVec; ++v) {
            const uint32_t hi = key[v] >> shift;
            cgt += (valid[v] && hi > p_hi) ? 1u : 0u;
            ceq += (valid[v] && hi == p_hi) ? 1u : 0u;
        }
    }
    cgt = __reduce_add_sync(0xffffffffu, cgt);
    ceq = __reduce_add_sync(0xffffffffu, ceq);
    if (lane == 0) {
        ws[warp] = cgt;
        ws[kWarps + warp] = ceq;
    }
    __syncthreads();
    if (warp == 0) {
        const uint32_t g = lane < kWarps ? ws[lane] : 0u, e = lane < kWarps ? ws[kWarps + lane] : 0u;
        const uint32_t gx = warp_inclusive_scan(g) - g, ex = warp_inclusive_scan(e) - e;
        if (lane < kWarps) {
            ws[lane] = gx;
            ws[kWarps + lane] = ex;
        }
    }
    __syncthreads();
    uint32_t gt_before = gt_off + ws[warp], eq_before = eq_off + ws[kWarps + warp];
    // Warp-uniform bounds: the warp-wide ballot and scan need every lane.
    for (int64_t b0 = w0; b0 < w1 && gt_before + min(eq_before, k_rem) < static_cast<uint32_t>(k); b0 += 32 * kVec) {
        const int64_t c0 = b0 + kVec * lane;
        uint32_t hi[kVec];
        bool valid[kVec];
        if (skeys != nullptr)
            smem_keys4(skeys, w1, c0, hi, valid);
        else
            load_keys4(row, stride1, vec, w1, c0, hi, valid);
        // Columns on the threshold matter only until k_rem of them are placed (warp-uniform).
        const bool eq_open = eq_before < k_rem;
        uint32_t lg = 0, le = 0;
#pragma unroll
        for (int v = 0; v < kVec; ++v) {
            hi[v] >>= shift;
            lg += (valid[v] && hi[v] > p_hi) ? 1u : 0u;
            le += (eq_open && valid[v] && hi[v] == p_hi) ? 1u : 0u;
        }
        const uint32_t mine = (lg << 16) | le;
        if (__ballot_sync(0xffffffffu, mine != 0) == 0) continue;
        const uint32_t inc = warp_inclusive_scan(mine);
        const uint32_t tot = __shfl_sync(0xffffffffu, inc, 31);
        uint32_t g = gt_before + ((inc - mine) >> 16), e = eq_before + ((inc - mine) & 0xffffu);
#pragma unroll
        for (int v = 0; v < kVec; ++v) {
            if (!valid[v]) break;
            const int32_t col = col_base + static_cast<int32_t>(c0 + v);
            if (hi[v] > p_hi) {
                orow[g + min(e, k_rem)] = col;
                ++g;
            } else if (eq_open && hi[v] == p_hi) {
                if (e < k_rem) orow[g + e] = col;
                ++e;
            }
        }
        gt_before += tot >> 16;
        eq_before += tot & 0xffffu;
    }
}

// The whole row in one CTA (n > k). With skeys (shared memory for cap keys) and n <= cap the first pass keeps the keys
// there and the later passes and the compaction read them instead of the row. lowor: kGroups words.
template <int THREADS>
__device__ void topk_one_cta(const float* __restrict__ row, int64_t stride1, bool vec, int64_t n,
                             int32_t* __restrict__ orow, int32_t col_base, int k, uint32_t* hist, uint32_t* lowor,
                             uint32_t* ws, uint32_t* s_vals, uint32_t* skeys = nullptr, int64_t cap = 0) {
    const bool cache = skeys != nullptr && n <= cap;
    uint32_t prefix = 0, pmask = 0;
    uint32_t k_rem = static_cast<uint32_t>(k);
    bool done = false;
    int shift = 32;
    const int widths[3] = {11, 11, 10};
    for (int p = 0; p < 3 && !done; ++p) {
        const int w = widths[p];
        shift -= w;
        const uint32_t dmask = (1u << w) - 1u;
        const int nbins = 1 << w;
        for (int i = threadIdx.x; i < nbins; i += THREADS) hist[i] = 0;
        if (p == 1)
            for (int i = threadIdx.x; i < kGroups; i += THREADS) lowor[i] = 0;
        __syncthreads();
        histogram<THREADS, DET_TOPK_UNROLL>(row, stride1, vec, n, prefix, pmask, shift, dmask, hist, skeys,
                                            cache && p == 0, cache && p > 0, p == 1 ? lowor : nullptr,
                                            (1u << shift) - 1u);
        __syncthreads();
        // Digit d with (keys of a higher digit) < k_rem <= (keys with digit >= d), bins scanned from the top.
        constexpr int kPer = kBins / THREADS > 0 ? kBins / THREADS : 1;
        const int per = nbins / THREADS > 0 ? nbins / THREADS : 1;
        uint32_t cnt[kPer];
        uint32_t c = 0;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const int d = nbins - 1 - (threadIdx.x * per + j);
            cnt[j] = (j < per && d >= 0) ? hist[d] : 0u;
            c += cnt[j];
        }
        uint32_t tot;
        uint32_t above = block_exclusive_scan<THREADS>(c, ws, &tot);
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            if (j < per && above < k_rem && k_rem <= above + cnt[j]) {
                const int d = nbins - 1 - (threadIdx.x * per + j);
                s_vals[0] = static_cast<uint32_t>(d);
                s_vals[1] = k_rem - above;
                // Done when the bucket is taken whole or (second pass: lowor) its keys agree in the bits below as well,
                // as they do for bf16-valued logits; ties then go to the lower columns in the compaction.
                s_vals[2] = cnt[j] == k_rem - above || (p == 1 && ((lowor[d >> 5] >> (d & 31)) & 1u) == 0u);
            }
            above += cnt[j];
        }
        __syncthreads();
        prefix |= s_vals[0] << shift;
        pmask |= dmask << shift;
        k_rem = s_vals[1];
        done = s_vals[2] != 0;
        __syncthreads();
    }
    compact<THREADS>(row, stride1, vec, n, shift, prefix >> shift, k_rem, k, 0, 0, orow, col_base, ws,
                     cache ? skeys : nullptr);
}

__device__ __forceinline__ void fill_short(int64_t n, int k, int32_t* orow, int32_t col_base, int threads) {
    for (int64_t i = threadIdx.x; i < k; i += threads) orow[i] = i < n ? col_base + static_cast<int32_t>(i) : -1;
}

template <int THREADS>
__global__ void __launch_bounds__(THREADS) det_topk_kernel(const float* logits, int64_t stride0, int64_t stride1,
                                                           RowBounds bounds, int32_t* out, int64_t out_stride0, int k,
                                                           bool relative) {
    __shared__ uint32_t hist[kBins];
    __shared__ uint32_t lowor[kGroups];
    __shared__ uint32_t ws[2 * (THREADS / 32) + 1];
    __shared__ uint32_t s_vals[3];
    const int64_t r = blockIdx.x;
    int64_t start, end;
    row_bounds(bounds, r, &start, &end);
    const int64_t n = end - start;
    const float* row = logits + r * stride0 + start * stride1;
    const bool vec = stride1 == 1 && (reinterpret_cast<uintptr_t>(row) & 15u) == 0;
    int32_t* orow = out + r * out_stride0;
    const int32_t col_base = relative ? 0 : static_cast<int32_t>(start);
    if (n <= k) {
        fill_short(n, k, orow, col_base, THREADS);
        return;
    }
    topk_one_cta<THREADS>(row, stride1, vec, n, orow, col_base, k, hist, lowor, ws, s_vals);
}

template <int C>
__global__ void __launch_bounds__(kClusterThreads) det_topk_cluster_kernel(const float* logits, int64_t stride0,
                                                                           int64_t stride1, RowBounds bounds, int32_t* out,
                                                                           int64_t out_stride0, int k, bool relative,
                                                                           int64_t small, int64_t cap) {
    constexpr int T = kClusterThreads;
    extern __shared__ uint4 dyn_smem[];  // keys of this CTA's columns (cap of them) after the first pass
    uint32_t* skeys = reinterpret_cast<uint32_t*>(dyn_smem);
    __shared__ uint32_t hist[2][kBins];
    __shared__ uint32_t lowor[2][kGroups];
    __shared__ uint32_t coarse[2][kGroups];
    // Per pass (zeroed once, so no reuse barrier): the whole row's group sums, the bins of the chosen group and their
    // lowor flags, and the decision (digit, k_rem, done, this slice's columns above the digit / on it).
    __shared__ uint32_t csum[3][kGroups];
    __shared__ uint32_t bsum[3][32];
    __shared__ uint32_t bor[3];
    __shared__ uint32_t s_dec[3][5];
    __shared__ uint32_t ws[2 * (T / 32) + 1];
    __shared__ uint32_t s_vals[5];
    __shared__ uint32_t s_before[2];  // columns above the threshold / on it in the earlier slices, added in by them
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank());

    const int64_t r = blockIdx.x / C;
    int64_t start, end;
    row_bounds(bounds, r, &start, &end);
    const int64_t n = end - start;
    const float* row = logits + r * stride0 + start * stride1;
    const bool vec = stride1 == 1 && (reinterpret_cast<uintptr_t>(row) & 15u) == 0;
    int32_t* orow = out + r * out_stride0;
    const int32_t col_base = relative ? 0 : static_cast<int32_t>(start);

    // The same branch is taken by the whole cluster (one row): short rows end here without any cluster barrier.
    if (n <= k || n <= small) {
        if (rank == 0) {
            if (n <= k)
                fill_short(n, k, orow, col_base, T);
            else
                topk_one_cta<T>(row, stride1, vec, n, orow, col_base, k, hist[0], lowor[0], ws, s_vals, skeys, cap);
        }
        return;
    }

    // This CTA's slice: [lo, hi) - multiples of 16 columns, so float4 alignment carries over.
    const int64_t slice = ((n + C - 1) / C + 15) & ~static_cast<int64_t>(15);
    const int64_t lo = min(n, static_cast<int64_t>(rank) * slice);
    const int64_t sn = min(n, lo + slice) - lo;
    const float* srow = row + lo * stride1;
    const bool cache = sn <= cap;

    uint32_t prefix = 0, pmask = 0;
    uint32_t k_rem = static_cast<uint32_t>(k);
    bool done = false;
    int shift = 32;
    uint32_t local_gt = 0, local_eq = 0;
    const int widths[3] = {11, 11, 10};
    for (int i = threadIdx.x; i < 2 * kBins; i += T) hist[0][i] = 0;
    for (int i = threadIdx.x; i < 2 * kGroups; i += T) lowor[0][i] = 0;
    for (int i = threadIdx.x; i < 3 * kGroups; i += T) csum[0][i] = 0;
    for (int i = threadIdx.x; i < 3 * 32; i += T) bsum[0][i] = 0;
    if (threadIdx.x < 3) bor[threadIdx.x] = 0;
    if (threadIdx.x < 2) s_before[threadIdx.x] = 0;
    __syncthreads();
    for (int p = 0; p < 3 && !done; ++p) {
        const int b = p & 1;  // double buffer: other CTAs may still read the previous pass's histograms
        const int w = widths[p];
        shift -= w;
        const uint32_t dmask = (1u << w) - 1u;
        const int nbins = 1 << w;
        const int ngroups = nbins >> 5;
        histogram<T, DET_TOPK_UNROLL>(srow, stride1, vec, sn, prefix, pmask, shift, dmask, hist[b], skeys,
                                      cache && p == 0, cache && p > 0, p == 1 ? lowor[b] : nullptr,
                                      (1u << shift) - 1u);
        __syncthreads();
        for (int g = threadIdx.x >> 5; g < ngroups; g += T / 32) {
            const uint32_t s = __reduce_add_sync(0xffffffffu, hist[b][g * 32 + (threadIdx.x & 31)]);
            if ((threadIdx.x & 31) == 0) coarse[b][g] = s;
        }
        cluster.sync();
        // Warp c adds CTA c's group sums into csum (two coalesced reads each; one warp reading all CTAs is several
        // times slower, and remote atomics are slower still when every group is populated). Every CTA has finished
        // reading the previous pass's buffers (they arrived here after reading them): the other warps clear them.
        if (threadIdx.x < 32 * C) {
            const uint32_t* rc = cluster.map_shared_rank(&coarse[b][0], threadIdx.x >> 5);
            for (int g = threadIdx.x & 31; g < ngroups; g += 32) {
                const uint32_t v = rc[g];
                if (v != 0) atomicAdd(&csum[p][g], v);
            }
        } else {
            for (int i = threadIdx.x - 32 * C; i < kBins; i += T - 32 * C) hist[b ^ 1][i] = 0;
            for (int i = threadIdx.x - 32 * C; i < kGroups; i += T - 32 * C) lowor[b ^ 1][i] = 0;
        }
        __syncthreads();
        // Warps below C take the same group decision; warp c then adds CTA c's bins of that group into bsum (and its
        // lowor flags into bor), again because one warp's remote reads do not overlap.
        const int lane = threadIdx.x & 31;
        int group = 0;
        uint32_t above_g = 0, local_above_g = 0;
        if (threadIdx.x < 32 * C) {
            // Groups in descending order: lane l holds positions l * gpl + j (j < gpl), group = ngroups - 1 - position.
            const int gpl = ngroups >> 5;
            uint32_t gsum[2] = {0u, 0u}, lsum[2] = {0u, 0u};
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                if (j < gpl) {
                    const int g = ngroups - 1 - (lane * gpl + j);
                    gsum[j] = csum[p][g];
                    lsum[j] = coarse[b][g];
                }
            }
            const uint32_t gt = gsum[0] + gsum[1], lt = lsum[0] + lsum[1];
            uint32_t ga = warp_inclusive_scan(gt) - gt, la = warp_inclusive_scan(lt) - lt;
            int jw = -1;
            uint32_t ga_w = 0, la_w = 0;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                if (j < gpl && jw < 0 && ga < k_rem && k_rem <= ga + gsum[j]) {
                    jw = j;
                    ga_w = ga;
                    la_w = la;
                }
                ga += gsum[j];
                la += lsum[j];
            }
            const int wl = __ffs(__ballot_sync(0xffffffffu, jw >= 0)) - 1;
            group = ngroups - 1 - (wl * gpl + __shfl_sync(0xffffffffu, jw, wl));
            above_g = __shfl_sync(0xffffffffu, ga_w, wl);
            local_above_g = __shfl_sync(0xffffffffu, la_w, wl);
            // Bins of that group in descending order: lane l -> digit group * 32 + 31 - l.
            const int c = threadIdx.x >> 5;
            const uint32_t v = cluster.map_shared_rank(&hist[b][0], c)[group * 32 + 31 - lane];
            if (v != 0) atomicAdd(&bsum[p][lane], v);
            if (p == 1 && lane == 0) {
                const uint32_t f = cluster.map_shared_rank(&lowor[b][0], c)[group];
                if (f != 0) atomicOr(&bor[p], f);
            }
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            const int d = group * 32 + 31 - lane;
            const uint32_t bs = bsum[p][lane];
            const uint32_t lb = hist[b][d];
            const uint32_t ba = warp_inclusive_scan(bs) - bs, lba = warp_inclusive_scan(lb) - lb;
            if (above_g + ba < k_rem && k_rem <= above_g + ba + bs) {
                const uint32_t need = k_rem - above_g - ba;
                s_dec[p][0] = static_cast<uint32_t>(d);
                s_dec[p][1] = need;
                // As in topk_one_cta: the bucket is taken whole, or its keys agree in the bits below as well.
                s_dec[p][2] = bs == need || (p == 1 && ((bor[p] >> (d & 31)) & 1u) == 0u);
                s_dec[p][3] = local_above_g + lba;
                s_dec[p][4] = lb;
            }
        }
        __syncthreads();
        prefix |= s_dec[p][0] << shift;
        pmask |= dmask << shift;
        k_rem = s_dec[p][1];
        done = s_dec[p][2] != 0;
        local_gt += s_dec[p][3];
        local_eq = s_dec[p][4];
    }

    // This slice's columns above the threshold / on it count for the later slices. After this barrier no CTA touches
    // another's shared memory, so each may finish on its own.
    if (threadIdx.x > rank && threadIdx.x < C) {
        if (local_gt != 0) atomicAdd(cluster.map_shared_rank(&s_before[0], threadIdx.x), local_gt);
        if (local_eq != 0) atomicAdd(cluster.map_shared_rank(&s_before[1], threadIdx.x), local_eq);
    }
    cluster.sync();
    compact<T>(srow, stride1, vec, sn, shift, prefix >> shift, k_rem, k, s_before[0], s_before[1], orow,
               col_base + static_cast<int32_t>(lo), ws, cache ? skeys : nullptr);
}

// Per device and cluster size: the dynamic shared memory for the key copy (the opt-in maximum per block minus the static
// part) and how many such clusters the GPU holds at once.
struct ClusterInfo {
    size_t dyn = 0;
    int max_clusters = 0;
};

template <int C>
ClusterInfo cluster_info(int device) {
    static ClusterInfo cached[64];
    if (device < 64 && cached[device].dyn) return cached[device];
    int optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    cudaFuncAttributes fa;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, det_topk_cluster_kernel<C>));
    ClusterInfo info;
    info.dyn = (static_cast<size_t>(optin) - fa.sharedSizeBytes) & ~static_cast<size_t>(15);
    C10_CUDA_CHECK(cudaFuncSetAttribute(det_topk_cluster_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(info.dyn)));
    if (C > 8)
        C10_CUDA_CHECK(cudaFuncSetAttribute(det_topk_cluster_kernel<C>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(C * 64);
    cfg.blockDim = dim3(kClusterThreads);
    cfg.dynamicSmemBytes = info.dyn;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = C;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    if (cudaOccupancyMaxActiveClusters(&info.max_clusters, det_topk_cluster_kernel<C>, &cfg) != cudaSuccess) {
        (void)cudaGetLastError();
        info.max_clusters = 0;
    }
    if (device < 64) cached[device] = info;
    return info;
}

template <int C>
void launch_cluster(const torch::Tensor& logits, const RowBounds& b, torch::Tensor& out, int64_t k, bool relative,
                    int64_t small, cudaStream_t stream) {
    const size_t dyn = cluster_info<C>(logits.get_device()).dyn;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(b.num_rows * C));
    cfg.blockDim = dim3(kClusterThreads);
    cfg.dynamicSmemBytes = dyn;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = C;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, det_topk_cluster_kernel<C>, logits.data_ptr<float>(), logits.stride(0),
                                      logits.stride(1), b, out.data_ptr<int32_t>(), out.stride(0), static_cast<int>(k),
                                      relative, small, static_cast<int64_t>(dyn / sizeof(uint32_t))));
}

void launch(const torch::Tensor& logits, const RowBounds& b, torch::Tensor& out, int64_t k, bool relative,
            int64_t cluster, int64_t small) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2,
                "logits: fp32 [rows, cols]");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32 && out.dim() == 2 && out.stride(1) == 1,
                "out: int32 [rows, >=k] with unit column stride");
    TORCH_CHECK(out.size(0) >= b.num_rows && out.size(1) >= k, "out too small");
    TORCH_CHECK(k > 0 && k <= 65535, "k out of range");
    if (b.num_rows == 0) return;
    const c10::cuda::CUDAGuard guard(logits.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // cluster < 0: the largest cluster size whose clusters for all rows run at once (by row count only: the row lengths
    // live on the device, and CUDA graphs replay one launch shape for every context length).
    if (cluster < 0) {
        const int dev = logits.get_device();
        cluster = b.num_rows <= cluster_info<16>(dev).max_clusters  ? 16
                  : b.num_rows <= cluster_info<8>(dev).max_clusters ? 8
                  : b.num_rows <= cluster_info<4>(dev).max_clusters ? 4
                  : b.num_rows <= cluster_info<2>(dev).max_clusters ? 2
                                                                    : 1;
    }
    if (cluster == 16) {
        launch_cluster<16>(logits, b, out, k, relative, small, stream);
    } else if (cluster == 8) {
        launch_cluster<8>(logits, b, out, k, relative, small, stream);
    } else if (cluster == 4) {
        launch_cluster<4>(logits, b, out, k, relative, small, stream);
    } else if (cluster == 2) {
        launch_cluster<2>(logits, b, out, k, relative, small, stream);
    } else if (b.num_rows <= 256) {
        det_topk_kernel<1024><<<b.num_rows, 1024, 0, stream>>>(logits.data_ptr<float>(), logits.stride(0),
                                                               logits.stride(1), b, out.data_ptr<int32_t>(),
                                                               out.stride(0), static_cast<int>(k), relative);
    } else {
        det_topk_kernel<256><<<b.num_rows, 256, 0, stream>>>(logits.data_ptr<float>(), logits.stride(0),
                                                             logits.stride(1), b, out.data_ptr<int32_t>(),
                                                             out.stride(0), static_cast<int>(k), relative);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// Prefill form (vLLM top_k_per_row_prefill): rows [row_starts[r], row_ends[r]); row_starts may be None.
// cluster: CTAs per row (-1 = by row count, 1 = one CTA); small: rows of at most this many columns use one CTA.
void det_topk_prefill(const torch::Tensor& logits, const c10::optional<torch::Tensor>& row_starts,
                      const torch::Tensor& row_ends, torch::Tensor out, int64_t num_rows, int64_t k, bool relative,
                      int64_t cluster, int64_t small) {
    TORCH_CHECK(row_ends.scalar_type() == torch::kInt32 && row_ends.is_contiguous() && row_ends.numel() >= num_rows);
    if (row_starts.has_value())
        TORCH_CHECK(row_starts->scalar_type() == torch::kInt32 && row_starts->is_contiguous() &&
                    row_starts->numel() >= num_rows);
    RowBounds b{row_starts.has_value() ? row_starts->data_ptr<int32_t>() : nullptr, row_ends.data_ptr<int32_t>(), nullptr,
                0, 1, num_rows, logits.size(1)};
    launch(logits, b, out, k, relative, cluster, small);
}

// Decode form (vLLM SparseIndexerTopk): row lengths from seq_lens / next_n as in SparseIndexerTopk._row_ends.
void det_topk_decode(const torch::Tensor& logits, const torch::Tensor& seq_lens, int64_t next_n, torch::Tensor out,
                     int64_t num_rows, int64_t k, int64_t cluster, int64_t small) {
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && seq_lens.is_contiguous() && next_n >= 1);
    RowBounds b{nullptr, nullptr, seq_lens.data_ptr<int32_t>(), seq_lens.numel(), next_n, num_rows, logits.size(1)};
    launch(logits, b, out, k, true, cluster, small);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("det_topk_prefill", &det_topk_prefill);
    m.def("det_topk_decode", &det_topk_decode);
}
