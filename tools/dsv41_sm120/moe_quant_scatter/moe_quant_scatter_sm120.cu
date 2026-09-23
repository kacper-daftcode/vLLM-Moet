// Fused MoE input quantization + permutation for vLLM's DeepGEMM contiguous grouped layout
// (DeepSeek-V4.1-Flash decode on sm_120; see moe_quant_scatter_sm120.py for the contract).
//
// One launch replaces the seven vLLM runs per MoE layer at decode token counts:
//   per_token_group_quant_8bit (bf16 -> fp8 e4m3, 128-wide groups, UE8M0 scales as fp32)
//   Fill<int> (m_indices = -1), Fill<float> (scales = 0), _count_expert_num_tokens,
//   _fwd_kernel_ep_scatter_1 (expert offsets + m_indices), _fwd_kernel_ep_scatter_2 (row copy),
//   and DeepGEMM's transpose_and_pack_fp32_into_ue8m0 inside the FC1 call.
//
// One CTA per (token, expert) pair, two barriers (three above 64 pairs). The CTA first issues the
// loads of its token's row (they do not depend on the routing), then rebuilds the routing table
// from topk_ids (<= 1024 pairs: a few hundred bytes in shared memory): per-expert counts (thread t
// counts expert t's pairs itself up to 64 pairs, a shared-memory histogram above that), and one
// block reduction yields the rows of the experts before this pair's expert (BLOCK_M-aligned
// regions in expert order), the pair's rank among earlier pairs of the same expert, the expert's
// own count and the total rows in use. The slot is therefore a pure function of topk_ids
// (deterministic; vLLM's scatter assigns slots in atomic order, which the gather undoes either
// way). The CTA quantizes the row exactly like vLLM's kernel (same absmax, same eps, same
// exp2f(ceilf(log2f())) scale, round-to-nearest-even e4m3 with the clamp before the conversion)
// into its slot and writes the UE8M0 exponent byte straight into DeepGEMM's packed int32
// MN-major scale tensor. The rank-0 pair of an expert writes the expert's m_indices region (the
// expert id for the real rows, -1 and zero scales for the padding rows); the tail after the
// last region gets -1 from all CTAs (DeepGEMM skips a block whose first row is -1; the tail's
// scales are never read).
//
// Compiled without --use_fast_math on purpose: vLLM's per_token_group_quant.cu is not, and the
// scale math has to match it bit for bit (log2f/exp2f/IEEE division).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Half.h>

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxPairs = 1024;    // M * topk the fused path accepts (decode shapes: <= 384)
constexpr int kMaxExperts = 1024;  // local experts (V4.1-Flash: 384)
constexpr int kGroup = 128;        // activation quant group (DeepGEMM recipe_a = (1, 128))
constexpr int kVec = 16;           // elements per lane: 32 B of bf16 / fp16, 16 B of fp8
constexpr int kLanesPerGroup = kGroup / kVec;             // 8
constexpr int kGroupsPerWarp = 32 / kLanesPerGroup;        // 4
constexpr int kMaxGroups = kThreads / kLanesPerGroup;      // 64 groups = K <= 8192 in one round
constexpr int kPerThread = kMaxPairs / kThreads;           // 2 pairs / experts per thread
constexpr int kDirectCountPairs = 64;  // up to this many pairs every thread counts its experts itself
static_assert(kMaxExperts / kThreads == kPerThread, "one indexing scheme for pairs and experts");

__device__ __forceinline__ int round_up_i(int v, int a) { return (v + a - 1) / a * a; }

// e4m3 RNE with the value already clamped to [-448, 448]: hardware cvt (sm_89+) or c10's
// software conversion (what vLLM's kernel instantiates). Identical results; the test checks.
template <bool kHwCvt>
__device__ __forceinline__ uint8_t to_e4m3(float q) {
  if constexpr (kHwCvt) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3));
  } else {
    return c10::Float8_e4m3fn(q).x;
  }
}

template <typename T, bool kHwCvt>
__global__ void __launch_bounds__(kThreads) moe_quant_scatter_kernel(
    const T* __restrict__ x, const int64_t x_stride,   // [M, K], row stride in elements
    const int32_t* __restrict__ topk_ids,              // [M, TOPK] contiguous
    const int M, const int TOPK, const int K, const int E, const int align, const int M_sum,
    const int tma_mn,
    uint8_t* __restrict__ out_q,        // [M_sum, K] e4m3 bytes
    int32_t* __restrict__ out_s,        // packed UE8M0: (row, pk) at row + pk * tma_mn
    int32_t* __restrict__ m_indices,    // [M_sum]
    int32_t* __restrict__ inv_perm,     // [M, TOPK]
    const float eps, const float fp8_min, const float fp8_max) {
  __shared__ __align__(16) int32_t s_ids[kMaxPairs + 4];  // +4: the int4 reads may run past P
  __shared__ int32_t s_hist[kMaxExperts];
  __shared__ int32_t s_red[4][kWarps];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int P = M * TOPK;
  const int p = blockIdx.x;
  const int sf_k = K / kGroup;
  const int sf_k_pad = round_up_i(sf_k, 4);

  // 0. issue the row loads now; the routing work below overlaps their latency
  const int g = warp * kGroupsPerWarp + (lane >> 3);   // this lane's 128-group of the row
  const int g_lane = lane & (kLanesPerGroup - 1);
  const bool active = g < sf_k;
  alignas(16) T regs[kVec];
  if (active) {
    const T* row = x + static_cast<int64_t>(p / TOPK) * x_stride + g * kGroup + g_lane * kVec;
    const uint4* src = reinterpret_cast<const uint4*>(row);
    uint4* dst = reinterpret_cast<uint4*>(regs);
    dst[0] = src[0];
    dst[1] = src[1];
  }

  // 1. routing table -> smem
  const bool count_direct = P <= kDirectCountPairs;
  if (!count_direct) {
    for (int e = tid; e < E; e += kThreads) s_hist[e] = 0;
  }
  for (int i = tid; i < P; i += kThreads) s_ids[i] = topk_ids[i];
  __syncthreads();

  // 2. per-expert counts for experts t and t + 512 of thread t: a pass over the table (4 ids per
  //    smem load) for small P - O(P) per thread, no barrier - or a shared-memory histogram
  //    (O(P) per CTA, one barrier) above kDirectCountPairs. From those: rows before my expert,
  //    total rows, my expert's count; and my rank among the earlier pairs of my expert
  const int e_mine = s_ids[p];
  int cnt[kPerThread] = {0, 0};
  if (count_direct) {
    const int n4 = (P + 3) >> 2;
    for (int i = 0; i < n4; ++i) {
      const int4 v = reinterpret_cast<const int4*>(s_ids)[i];
      const int lim = P - 4 * i;  // ids past P are stale smem: mask them
#pragma unroll
      for (int j = 0; j < kPerThread; ++j) {
        const int e = tid + j * kThreads;
        cnt[j] += (v.x == e) + ((lim > 1) & (v.y == e)) + ((lim > 2) & (v.z == e)) + ((lim > 3) & (v.w == e));
      }
    }
  } else {
    for (int i = tid; i < P; i += kThreads) {
      const int e = s_ids[i];
      if (e >= 0 && e < E) atomicAdd(&s_hist[e], 1);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      const int e = tid + j * kThreads;
      cnt[j] = (e < E) ? s_hist[e] : 0;
    }
  }
  int before = 0, total = 0, c_mine = 0, rank = 0;
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    const int e = tid + j * kThreads;
    const int rows = (e < E) ? round_up_i(cnt[j], align) : 0;
    total += rows;
    before += (e < e_mine) ? rows : 0;
    c_mine += (e == e_mine) ? cnt[j] : 0;
    const int i = tid + j * kThreads;  // pair index checked by this thread
    rank += (i < p && i < P && s_ids[i] == e_mine) ? 1 : 0;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    before += __shfl_xor_sync(0xffffffffu, before, off);
    total += __shfl_xor_sync(0xffffffffu, total, off);
    c_mine += __shfl_xor_sync(0xffffffffu, c_mine, off);
    rank += __shfl_xor_sync(0xffffffffu, rank, off);
  }
  if (lane == 0) {
    s_red[0][warp] = before;
    s_red[1][warp] = total;
    s_red[2][warp] = c_mine;
    s_red[3][warp] = rank;
  }
  __syncthreads();
  {
    // lanes 0..15 hold the warp sums, the others 0; a full-warp xor reduction leaves the
    // block sums in every lane
    int v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    if (lane < kWarps) {
      v0 = s_red[0][lane];
      v1 = s_red[1][lane];
      v2 = s_red[2][lane];
      v3 = s_red[3][lane];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v0 += __shfl_xor_sync(0xffffffffu, v0, off);
      v1 += __shfl_xor_sync(0xffffffffu, v1, off);
      v2 += __shfl_xor_sync(0xffffffffu, v2, off);
      v3 += __shfl_xor_sync(0xffffffffu, v3, off);
    }
    before = v0;
    total = v1;
    c_mine = v2;
    rank = v3;
  }

  // 3. the tail after the last region: -1 (spread over all CTAs)
  for (int r = total + p * kThreads + tid; r < M_sum; r += P * kThreads) m_indices[r] = -1;

  if (e_mine < 0 || e_mine >= E) return;  // invalid routing slot: no row (the gather skips it)
  const int dest = before + rank;
  if (tid == 0) inv_perm[p] = dest;

  // 4. the expert's region: written once, by its rank-0 pair
  const int packed_sf_k = sf_k_pad / 4;
  if (rank == 0) {
    const int rows = round_up_i(c_mine, align);
    for (int i = tid; i < rows; i += kThreads) {
      const bool real = i < c_mine;
      m_indices[before + i] = real ? e_mine : -1;
      if (!real) {
        for (int pk = 0; pk < packed_sf_k; ++pk) out_s[before + i + static_cast<int64_t>(pk) * tma_mn] = 0;
      }
    }
  }

  // 5. quantize the row into the slot: absmax over the 128-group (8 lanes)
  float amax = eps;
  if (active) {
#pragma unroll
    for (int i = 0; i < kVec; ++i) amax = fmaxf(amax, fabsf(static_cast<float>(regs[i])));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  // vLLM per_token_group_quant_8bit, SCALE_UE8M0: y_s = absmax / max, power of two rounded up
  float y_s = amax / fp8_max;
  y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));

  if (g < sf_k_pad && g_lane == 0) {
    // the UE8M0 exponent byte, or 0 for the padding bytes of the last int32 (K/128 % 4 != 0)
    uint8_t* srow = reinterpret_cast<uint8_t*>(out_s);
    srow[(static_cast<int64_t>(dest) + static_cast<int64_t>(g >> 2) * tma_mn) * 4 + (g & 3)] =
        active ? static_cast<uint8_t>(__float_as_uint(y_s) >> 23) : uint8_t{0};
  }
  if (active) {
    uint32_t w[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float q = fminf(fmaxf(static_cast<float>(regs[i]) / y_s, fp8_min), fp8_max);
      w[i >> 2] |= static_cast<uint32_t>(to_e4m3<kHwCvt>(q)) << ((i & 3) * 8);
    }
    uint8_t* qrow = out_q + static_cast<int64_t>(dest) * K + g * kGroup + g_lane * kVec;
    *reinterpret_cast<uint4*>(qrow) = make_uint4(w[0], w[1], w[2], w[3]);
  }
}

template <typename T, bool kHwCvt>
void launch(const at::Tensor& x, const at::Tensor& topk_ids, at::Tensor& out_q, at::Tensor& out_s,
            at::Tensor& m_indices, at::Tensor& inv_perm, int E, int align, float eps, float fp8_min,
            float fp8_max, cudaStream_t stream) {
  const int M = x.size(0), K = x.size(1), TOPK = topk_ids.size(1);
  const int M_sum = out_q.size(0);
  const int tma_mn = out_s.stride(1);
  const dim3 grid(M * TOPK), block(kThreads);
  moe_quant_scatter_kernel<T, kHwCvt><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()), x.stride(0), topk_ids.data_ptr<int32_t>(), M, TOPK, K, E,
      align, M_sum, tma_mn, reinterpret_cast<uint8_t*>(out_q.data_ptr()), out_s.data_ptr<int32_t>(),
      m_indices.data_ptr<int32_t>(), inv_perm.data_ptr<int32_t>(), eps, fp8_min, fp8_max);
}

}  // namespace

// out_q [M_sum, K] e4m3 (contiguous), out_s int32 [M_sum, ceil(K/128/4)] with strides (1, M_sum),
// m_indices int32 [M_sum], inv_perm int32 [M, TOPK]; E local experts, align = rows per expert
// block (DeepGEMM BLOCK_M for this call); software_cvt = c10's e4m3 conversion instead of the
// hardware one (test hook).
void moe_quant_scatter(const at::Tensor& x, const at::Tensor& topk_ids, at::Tensor out_q, at::Tensor out_s,
                       at::Tensor m_indices, at::Tensor inv_perm, int64_t E, int64_t align, double eps,
                       double fp8_min, double fp8_max, bool software_cvt) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(1) == 1, "x: 2-D CUDA tensor with unit inner stride");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "x: bf16 or fp16");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.dim() == 2 && topk_ids.is_contiguous() &&
                  topk_ids.scalar_type() == at::kInt && topk_ids.size(0) == x.size(0),
              "topk_ids: contiguous int32 [M, topk]");
  const int64_t M = x.size(0), K = x.size(1), TOPK = topk_ids.size(1), P = M * TOPK;
  TORCH_CHECK(P >= 1 && P <= kMaxPairs, "M * topk must be in [1, ", kMaxPairs, "], got ", P);
  TORCH_CHECK(E >= 1 && E <= kMaxExperts, "local experts must be in [1, ", kMaxExperts, "], got ", E);
  TORCH_CHECK(K % kGroup == 0 && K > 0 && K <= kMaxGroups * kGroup, "K must be a positive multiple of ", kGroup,
              " up to ", kMaxGroups * kGroup);
  TORCH_CHECK((x.stride(0) * x.element_size()) % 16 == 0 &&
                  (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) == 0,
              "x rows must be 16-byte aligned");
  TORCH_CHECK(align >= 1, "align must be positive");
  TORCH_CHECK(out_q.is_cuda() && out_q.dim() == 2 && out_q.is_contiguous() && out_q.size(1) == K &&
                  out_q.element_size() == 1,
              "out_q: contiguous [M_sum, K] 1-byte tensor");
  const int64_t M_sum = out_q.size(0);
  TORCH_CHECK(M_sum >= P && M_sum % 4 == 0, "M_sum must be >= M * topk and a multiple of 4");
  const int64_t sf_k = K / kGroup, packed_sf_k = (sf_k + 3) / 4;
  TORCH_CHECK(out_s.is_cuda() && out_s.scalar_type() == at::kInt && out_s.dim() == 2 && out_s.size(0) == M_sum &&
                  out_s.size(1) == packed_sf_k && out_s.stride(0) == 1 && out_s.stride(1) == M_sum,
              "out_s: int32 [M_sum, ceil(K/128/4)] with strides (1, M_sum)");
  TORCH_CHECK(m_indices.is_cuda() && m_indices.scalar_type() == at::kInt && m_indices.is_contiguous() &&
                  m_indices.numel() == M_sum,
              "m_indices: contiguous int32 [M_sum]");
  TORCH_CHECK(inv_perm.is_cuda() && inv_perm.scalar_type() == at::kInt && inv_perm.is_contiguous() &&
                  inv_perm.size(0) == M && inv_perm.size(1) == TOPK,
              "inv_perm: contiguous int32 [M, topk]");

  const c10::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int e = static_cast<int>(E), a = static_cast<int>(align);
  const float ep = static_cast<float>(eps), lo = static_cast<float>(fp8_min), hi = static_cast<float>(fp8_max);
  if (x.scalar_type() == at::kBFloat16) {
    if (software_cvt)
      launch<c10::BFloat16, false>(x, topk_ids, out_q, out_s, m_indices, inv_perm, e, a, ep, lo, hi, stream);
    else
      launch<c10::BFloat16, true>(x, topk_ids, out_q, out_s, m_indices, inv_perm, e, a, ep, lo, hi, stream);
  } else {
    if (software_cvt)
      launch<c10::Half, false>(x, topk_ids, out_q, out_s, m_indices, inv_perm, e, a, ep, lo, hi, stream);
    else
      launch<c10::Half, true>(x, topk_ids, out_q, out_s, m_indices, inv_perm, e, a, ep, lo, hi, stream);
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_quant_scatter", &moe_quant_scatter,
        "fused per-token-group fp8/UE8M0 quantization + DeepGEMM grouped-layout permutation",
        py::arg("x"), py::arg("topk_ids"), py::arg("out_q"), py::arg("out_s"), py::arg("m_indices"),
        py::arg("inv_perm"), py::arg("E"), py::arg("align"), py::arg("eps"), py::arg("fp8_min"), py::arg("fp8_max"),
        py::arg("software_cvt") = false);
  m.attr("MAX_PAIRS") = kMaxPairs;
  m.attr("MAX_EXPERTS") = kMaxExperts;
  m.attr("MAX_K") = kMaxGroups * kGroup;
}
