// Deterministic exact top-k over rows of fp32 logits, for the DeepSeek-V4.1 DSA indexer on sm_120.
//
// Row r selects the k largest values among its columns [start_r, end_r); among equal values the lower column wins
// (-0.0 equals +0.0; every NaN ranks above +inf, as in torch.sort). The selected columns are written in ascending column order, then -1
// up to k when the row has fewer than k columns. The result is a function of the input values only: the radix select
// counts with integer atomics (order-independent sums) and the compaction ranks columns with block scans.
//
// One CTA per row (1024 threads for <= 256 rows - decode -, 256 above - prefill). The threshold is found by an MSB-first
// radix select over order-preserving 32-bit keys (11/11/10 bits, shared-memory histograms); the passes stop early once
// the bucket holding the k-th key is taken whole. The compaction walks the row in tiles of THREADS * 4 columns; thread
// t holds 4 consecutive columns, so thread order is column order, and one packed block scan per tile ranks both the
// columns above the threshold and those equal to it. Rows are read as float4 when 16-byte aligned with unit stride.
// sm_120, 6 rows (C1 decode): 1K columns 4.2 us (vLLM persistent_topk 5.8), 8K 5.9 (7.5), 32K 16.3 (15.1), 100K 44
// (15.0, its sampled path being approximate there); prefill 4096 rows x 2K columns 24-30 us (top_k_per_row_prefill
// 21-281, the high end on tied rows).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

#ifndef DET_TOPK_AGG
#define DET_TOPK_AGG 0  // 1: warp-aggregated histogram atomics (__match_any_sync) - slower on sm_120 even on tied rows
#endif
#ifndef DET_TOPK_UNROLL
#define DET_TOPK_UNROLL 1  // float4 per thread per step; deeper unrolling only adds idle lanes on short rows
#endif

namespace {

constexpr int kVec = 4;
constexpr int kUnroll = DET_TOPK_UNROLL;

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

template <int THREADS>
__global__ void __launch_bounds__(THREADS) det_topk_kernel(const float* __restrict__ logits, int64_t stride0,
                                                           int64_t stride1, RowBounds bounds, int32_t* __restrict__ out,
                                                           int64_t out_stride0, int k, bool relative) {
    constexpr int kBins = 2048;
    constexpr int kTile = THREADS * kVec;
    __shared__ uint32_t hist[kBins];
    __shared__ uint32_t ws[THREADS / 32 + 1];
    __shared__ uint32_t s_digit, s_krem, s_exact;

    const int64_t r = blockIdx.x;
    int64_t start, end;
    row_bounds(bounds, r, &start, &end);
    const int64_t n = end - start;
    const float* row = logits + r * stride0 + start * stride1;
    const bool vec = stride1 == 1 && (reinterpret_cast<uintptr_t>(row) & 15u) == 0;
    int32_t* orow = out + r * out_stride0;
    const int32_t base_col = relative ? 0 : static_cast<int32_t>(start);

    if (n <= k) {
        for (int64_t i = threadIdx.x; i < k; i += THREADS) orow[i] = i < n ? base_col + static_cast<int32_t>(i) : -1;
        return;
    }

    // ---- radix select: prefix/pmask fix the key bits above `shift`; k_rem keys still to take among those matching
    uint32_t prefix = 0, pmask = 0;
    uint32_t k_rem = static_cast<uint32_t>(k);
    bool exact = false;
    int shift = 32;
    const int widths[3] = {11, 11, 10};
    for (int p = 0; p < 3 && !exact; ++p) {
        const int w = widths[p];
        shift -= w;
        const uint32_t dmask = (1u << w) - 1u;
        const int nbins = 1 << w;
        for (int i = threadIdx.x; i < nbins; i += THREADS) hist[i] = 0;
        __syncthreads();
        for (int64_t q0 = 0; q0 < n; q0 += 4 * THREADS * kUnroll) {
            uint32_t keys[kUnroll][4];
            bool valid[kUnroll][4];
#pragma unroll
            for (int u = 0; u < kUnroll; ++u)
                load_keys4(row, stride1, vec, n, q0 + 4 * (u * THREADS + threadIdx.x), keys[u], valid[u]);
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const bool match = valid[u][v] && (keys[u][v] & pmask) == prefix;
                    const uint32_t digit = (keys[u][v] >> shift) & dmask;
#if DET_TOPK_AGG
                    const uint32_t active = __ballot_sync(0xffffffffu, match);
                    if (match) {
                        const uint32_t same = __match_any_sync(active, digit);
                        if ((threadIdx.x & 31) == __ffs(same) - 1) atomicAdd(&hist[digit], __popc(same));
                    }
#else
                    if (match) atomicAdd(&hist[digit], 1u);
#endif
                }
            }
        }
        __syncthreads();
        // Find digit d: (keys with a higher digit) < k_rem <= (keys with digit >= d), scanning bins from the top.
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
                s_digit = static_cast<uint32_t>(nbins - 1 - (threadIdx.x * per + j));
                s_krem = k_rem - above;
                s_exact = cnt[j] == k_rem - above;
            }
            above += cnt[j];
        }
        __syncthreads();
        prefix |= s_digit << shift;
        pmask |= dmask << shift;
        k_rem = s_krem;
        exact = s_exact != 0;
        __syncthreads();
    }
    const uint32_t p_hi = prefix >> shift;  // key >> shift above p_hi: taken; equal: the first k_rem in column order

    // ---- compaction in column order
    uint32_t gt_off = 0, eq_off = 0;
    for (int64_t t0 = 0; t0 < n && gt_off + min(eq_off, k_rem) < static_cast<uint32_t>(k); t0 += kTile) {
        const int64_t c0 = t0 + static_cast<int64_t>(threadIdx.x) * kVec;
        uint32_t hi[kVec];
        bool valid[kVec];
        load_keys4(row, stride1, vec, n, c0, hi, valid);
        uint32_t cgt = 0, ceq = 0;
#pragma unroll
        for (int v = 0; v < kVec; ++v) {
            hi[v] >>= shift;
            cgt += (valid[v] && hi[v] > p_hi) ? 1u : 0u;
            ceq += (valid[v] && hi[v] == p_hi) ? 1u : 0u;
        }
        uint32_t tot;
        const uint32_t ex = block_exclusive_scan<THREADS>((cgt << 16) | ceq, ws, &tot);
        uint32_t gt_before = gt_off + (ex >> 16), eq_before = eq_off + (ex & 0xffffu);
#pragma unroll
        for (int v = 0; v < kVec; ++v) {
            const int64_t col = c0 + v;
            if (col >= n) break;
            if (hi[v] > p_hi) {
                orow[gt_before + min(eq_before, k_rem)] = base_col + static_cast<int32_t>(col);
                ++gt_before;
            } else if (hi[v] == p_hi) {
                if (eq_before < k_rem) orow[gt_before + eq_before] = base_col + static_cast<int32_t>(col);
                ++eq_before;
            }
        }
        gt_off += tot >> 16;
        eq_off += tot & 0xffffu;
    }
}

void launch(const torch::Tensor& logits, const RowBounds& b, torch::Tensor& out, int64_t k, bool relative) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2, "logits: fp32 [rows, cols]");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32 && out.dim() == 2 && out.stride(1) == 1,
                "out: int32 [rows, >=k] with unit column stride");
    TORCH_CHECK(out.size(0) >= b.num_rows && out.size(1) >= k, "out too small");
    TORCH_CHECK(k > 0 && k <= 65535, "k out of range");
    if (b.num_rows == 0) return;
    const c10::cuda::CUDAGuard guard(logits.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    // Few long rows (decode) get a full CTA each; many rows (prefill) a quarter of one, four per SM.
    if (b.num_rows <= 256) {
        det_topk_kernel<1024><<<b.num_rows, 1024, 0, stream>>>(logits.data_ptr<float>(), logits.stride(0), logits.stride(1),
                                                               b, out.data_ptr<int32_t>(), out.stride(0),
                                                               static_cast<int>(k), relative);
    } else {
        det_topk_kernel<256><<<b.num_rows, 256, 0, stream>>>(logits.data_ptr<float>(), logits.stride(0), logits.stride(1),
                                                             b, out.data_ptr<int32_t>(), out.stride(0),
                                                             static_cast<int>(k), relative);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// Prefill form (vLLM top_k_per_row_prefill): rows [row_starts[r], row_ends[r]); row_starts may be None.
void det_topk_prefill(const torch::Tensor& logits, const c10::optional<torch::Tensor>& row_starts,
                      const torch::Tensor& row_ends, torch::Tensor out, int64_t num_rows, int64_t k, bool relative) {
    TORCH_CHECK(row_ends.scalar_type() == torch::kInt32 && row_ends.is_contiguous() && row_ends.numel() >= num_rows);
    if (row_starts.has_value())
        TORCH_CHECK(row_starts->scalar_type() == torch::kInt32 && row_starts->is_contiguous() &&
                    row_starts->numel() >= num_rows);
    RowBounds b{row_starts.has_value() ? row_starts->data_ptr<int32_t>() : nullptr, row_ends.data_ptr<int32_t>(), nullptr,
                0, 1, num_rows, logits.size(1)};
    launch(logits, b, out, k, relative);
}

// Decode form (vLLM SparseIndexerTopk): row lengths from seq_lens / next_n as in SparseIndexerTopk._row_ends.
void det_topk_decode(const torch::Tensor& logits, const torch::Tensor& seq_lens, int64_t next_n, torch::Tensor out,
                     int64_t num_rows, int64_t k) {
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && seq_lens.is_contiguous() && next_n >= 1);
    RowBounds b{nullptr, nullptr, seq_lens.data_ptr<int32_t>(), seq_lens.numel(), next_n, num_rows, logits.size(1)};
    launch(logits, b, out, k, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("det_topk_prefill", &det_topk_prefill);
    m.def("det_topk_decode", &det_topk_decode);
}
