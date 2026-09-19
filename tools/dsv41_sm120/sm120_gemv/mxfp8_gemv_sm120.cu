// SPDX-License-Identifier: Apache-2.0
// Small-M MXFP8 x MXFP8 -> BF16 GEMV for sm_120 (decode-shaped dense GEMMs).
//
// C[M, N] = A[M, K] (e4m3, per-32 ue8m0 scales) x B[N, K]^T (e4m3, per-32 ue8m0 scales)
//
// Both scale tensors use FlashInfer's F8_128x4 swizzled layout (the layout vLLM's
// FlashInferCutlassMxfp8LinearKernel already stores for weights and produces for
// activations), so this kernel is a drop-in for mm_mxfp8(backend="cutlass") when
// M <= 16 (the tensor-core path accepts up to 64 rows). The CUTLASS SM120 blockscaled
// GEMM runs a 128-row MMA tile for a 6-row decode batch and averages 16 us for a 5 MB
// weight (profile of DeepSeek-V4.1-Flash on RTX PRO 6000, 2026-09-18); the weight
// stream alone is ~3 us at 1.7 TB/s.
//
// `mxfp8_gemv_grouped` (below) is the same tensor-core kernel with the operand layouts
// of DeepSeek-V4's grouped o-projection (`wo_a`): checkpoint MXFP8 weights + row-major
// ue8m0 scales, activations from fused_inv_rope_fp8_quant with DeepGEMM's packed
// MN-major scales. On sm_120 vLLM otherwise falls back to BF16 weights + cuBLAS bmm.
//
// Mapping: one warp per output column n, 8 warps per block. The block stages A
// (M x CHUNK e4m3) and its scales in shared memory once per K-chunk; every lane then
// issues ALL of its 16-byte weight loads for the chunk up front (up to 10 in flight
// for K = 5120) before converting, so a row costs one HBM latency instead of one per
// 512 elements -- the weights are never L2-resident in the server (a decode step
// streams ~2 GB). Each lane's 16 weights sit in exactly one 32-wide scale block, so
// the block scale product 2^(ea + eb - 254) is applied per lane before the warp
// reduction. Accumulation is fp32 like the CUTLASS kernel.
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

namespace vllm_moet_sm120 {

constexpr int kBlockSize = 32;       // MXFP8 scale block
constexpr int kWarps = 8;            // rows (n) per block
constexpr int kThreads = kWarps * 32;
constexpr int kStepElems = 32 * 16;  // one uint4 per lane per step

// F8_128x4 swizzle: rows in tiles of 128 (4 x 32), scale columns in tiles of 4.
__device__ __forceinline__ int sf_offset(int row, int kc, int num_k_tiles) {
  const int mt = row >> 7, r = row & 127, a = r >> 5, b = r & 31;
  const int kt = kc >> 2, c = kc & 3;
  return ((((mt * num_k_tiles + kt) * 32 + b) * 4 + a) * 4 + c);
}

__device__ __forceinline__ void fp8x16_to_float(const uint4& v, float* out) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(w[i] & 0xffffu), __NV_E4M3);
    const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(w[i] >> 16), __NV_E4M3);
    const float2 flo = __half22float2(__half2(lo));
    const float2 fhi = __half22float2(__half2(hi));
    out[4 * i + 0] = flo.x;
    out[4 * i + 1] = flo.y;
    out[4 * i + 2] = fhi.x;
    out[4 * i + 3] = fhi.y;
  }
}

// 2^(e - 127) for a ue8m0 byte; e == 0 maps to 0 (only produced for all-zero blocks).
__device__ __forceinline__ float ue8m0_to_float(uint32_t e) {
  return e == 0u ? 0.0f : __uint_as_float(e << 23);
}

// blockIdx.x -> group of kWarps output columns, blockIdx.y -> K-chunk (split-K).
// With SPLIT (gridDim.y > 1) every block handles one K chunk and writes its fp32
// partial sums to `Cf[chunk, M, N]`; `finalize_kernel` sums the chunks and converts.
// Without split (gridDim.y == 1) the block loops over all chunks and writes bf16.
template <int MAXM, int CHUNK, bool SPLIT>
__global__ void __launch_bounds__(kThreads)
mxfp8_gemv_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ A_sf,
                  const uint8_t* __restrict__ B, const uint8_t* __restrict__ B_sf,
                  __nv_bfloat16* __restrict__ C, float* __restrict__ Cf, int M, int N, int K,
                  int num_k_tiles) {
  constexpr int STEPS = CHUNK / kStepElems;
  constexpr int SF_PER_ROW = CHUNK / kBlockSize;
  __shared__ __align__(16) uint8_t s_a[MAXM * CHUNK];
  __shared__ float s_asf[MAXM * SF_PER_ROW];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * kWarps + warp;
  const bool row_ok = n < N;

  float acc[MAXM];
#pragma unroll
  for (int m = 0; m < MAXM; ++m) acc[m] = 0.0f;

  const int num_chunks = (K + CHUNK - 1) / CHUNK;
  const int chunk_begin = SPLIT ? blockIdx.y : 0;
  const int chunk_end = SPLIT ? blockIdx.y + 1 : num_chunks;
  for (int chunk = chunk_begin; chunk < chunk_end; ++chunk) {
    const int k0 = chunk * CHUNK;
    const int k_len = min(CHUNK, K - k0);  // multiple of 32 (K % 32 == 0)

    // Issue this row's weight loads for the whole chunk before touching shared memory.
    uint4 wv[STEPS];
    const uint8_t* brow = B + (size_t)n * K + k0;
#pragma unroll
    for (int s = 0; s < STEPS; ++s) {
      const int kk = s * kStepElems + lane * 16;
      wv[s] = (row_ok && kk < k_len) ? *reinterpret_cast<const uint4*>(brow + kk)
                                     : make_uint4(0, 0, 0, 0);
    }

    // Stage A[:M, k0:k0+k_len] (16 B per thread per pass) and its block scales.
    __syncthreads();
    {
      const int vec_per_row = k_len / 16;
      const int total_vec = M * vec_per_row;
      for (int idx = threadIdx.x; idx < total_vec; idx += kThreads) {
        const int m = idx / vec_per_row;
        const int v = idx - m * vec_per_row;
        *reinterpret_cast<uint4*>(s_a + m * CHUNK + v * 16) =
            *reinterpret_cast<const uint4*>(A + (size_t)m * K + k0 + v * 16);
      }
      const int blocks_per_row = k_len / kBlockSize;
      const int total_sf = M * blocks_per_row;
      for (int idx = threadIdx.x; idx < total_sf; idx += kThreads) {
        const int m = idx / blocks_per_row;
        const int kb = idx - m * blocks_per_row;
        s_asf[m * SF_PER_ROW + kb] =
            ue8m0_to_float(A_sf[sf_offset(m, (k0 / kBlockSize) + kb, num_k_tiles)]);
      }
    }
    __syncthreads();
    if (!row_ok) continue;

#pragma unroll
    for (int s = 0; s < STEPS; ++s) {
      const int kk = s * kStepElems + lane * 16;
      if (kk >= k_len) break;
      const int kb = kk / kBlockSize;  // scale block within the chunk (2 lanes per block)
      const float wsf = ue8m0_to_float(B_sf[sf_offset(n, (k0 / kBlockSize) + kb, num_k_tiles)]);
      float w[16];
      fp8x16_to_float(wv[s], w);
#pragma unroll
      for (int m = 0; m < MAXM; ++m) {
        if (m < M) {
          float a[16];
          const uint4 av = *reinterpret_cast<const uint4*>(s_a + m * CHUNK + kk);
          fp8x16_to_float(av, a);
          float part = 0.0f;
#pragma unroll
          for (int j = 0; j < 16; ++j) part = fmaf(a[j], w[j], part);
          acc[m] = fmaf(part, wsf * s_asf[m * SF_PER_ROW + kb], acc[m]);
        }
      }
    }
  }

  if (!row_ok) return;
#pragma unroll
  for (int m = 0; m < MAXM; ++m) {
    float v = acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    acc[m] = v;
  }
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      if (m < M) {
        if (SPLIT) {
          Cf[((size_t)blockIdx.y * M + m) * N + n] = acc[m];
        } else {
          C[(size_t)m * N + n] = __float2bfloat16(acc[m]);
        }
      }
    }
  }
}

__global__ void finalize_kernel(const float* __restrict__ Cf, __nv_bfloat16* __restrict__ C,
                                int total, int num_chunks) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v = 0.0f;
  for (int c = 0; c < num_chunks; ++c) v += Cf[(size_t)c * total + i];
  C[i] = __float2bfloat16(v);
}

// ---------------------------------------------------------------------------
// Tensor-core variant: mma.sync.m16n8k32 (e4m3 x e4m3 -> f32), rows padded to 16.
// A block owns 8 output columns; its 8 warps split the K blocks (kb = warp, warp+8,
// ...). Physical k is permuted inside each 32-wide MX block so that every lane's
// fragment bytes are contiguous (lane tig holds physical k = kb*32 + tig*8 .. +8 as
// {a0/b0 (first 4), a2/b1 (next 4)}); A and B use the same permutation, so the dot
// product is unchanged and every mma stays inside one scale block. The per-block
// result is then scaled by 2^(sa[m]+sb[n]-254) before accumulation.
//
// Generalised (2026-09-18) for the grouped o-projection (`wo_a`, one [T, K] x [Ng, K]^T
// GEMM per head group, blockIdx.y = group) and for M up to 64 (MT m16 tiles share the
// weight fragment). Scale layouts are template parameters:
//   SF_F8_128x4  - FlashInfer's swizzled layout (dense path, both operands)
//   SF_ROWMAJOR  - [rows, K/32] ue8m0 bytes (vLLM's MXFP8 checkpoint weight_scale)
//   SF_MN_PACKED - DeepGEMM/SM100 activation layout written by vLLM's
//                  fused_inv_rope_fp8_quant(tma_aligned_scales=True): int32 words that
//                  pack 4 consecutive K-block ue8m0 bytes, stored [K/128][T_aligned]
constexpr int kMmaCols = 8;
enum : int { SF_F8_128x4 = 0, SF_ROWMAJOR = 1, SF_MN_PACKED = 2 };

struct GemvArgs {
  const uint8_t* A;
  const uint8_t* A_sf;
  const uint8_t* B;
  const uint8_t* B_sf;
  __nv_bfloat16* C;
  int M, N, K, num_k_tiles;
  int ldc;             // C row stride (elements)
  long long a_gstride;     // bytes between groups
  long long a_sf_gstride;  // bytes between groups
  long long b_gstride;     // bytes between groups
  long long b_sf_gstride;  // bytes between groups
  long long c_gstride;     // elements between groups
  int a_sf_t_al;       // SF_MN_PACKED: T_aligned (int32 words per K-block column)
  int b_sf_ld;         // SF_ROWMAJOR: bytes per row
  int lda_s;           // v2 SA: shared-memory row stride of the staged A slice (bytes)
};

template <int LAYOUT>
__device__ __forceinline__ uint32_t load_sf(const uint8_t* __restrict__ sf, int row, int kb,
                                            int num_k_tiles, int t_al, int ld) {
  if constexpr (LAYOUT == SF_F8_128x4) {
    return sf[sf_offset(row, kb, num_k_tiles)];
  } else if constexpr (LAYOUT == SF_ROWMAJOR) {
    return sf[(size_t)row * ld + kb];
  } else {
    return sf[((size_t)(kb >> 2) * t_al + row) * 4 + (kb & 3)];
  }
}

__device__ __forceinline__ void mma_m16n8k32_e4m3(float* d, const uint32_t* a, uint32_t b0,
                                                  uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.0f), "f"(0.0f),
        "f"(0.0f), "f"(0.0f));
}

template <int MT, int A_SF, int B_SF>
__global__ void __launch_bounds__(kThreads) mxfp8_mma_gemv_kernel(const GemvArgs p) {
  __shared__ float s_red[kWarps][32][4 * MT];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;    // groupID: A row (and row + 8), B column
  const int tig = lane & 3;   // thread in group
  const int grp = blockIdx.y;
  const uint8_t* __restrict__ A = p.A + grp * p.a_gstride;
  const uint8_t* __restrict__ A_sf = p.A_sf + grp * p.a_sf_gstride;
  const uint8_t* __restrict__ B = p.B + grp * p.b_gstride;
  const uint8_t* __restrict__ B_sf = p.B_sf + grp * p.b_sf_gstride;
  __nv_bfloat16* __restrict__ C = p.C + grp * p.c_gstride;
  const int M = p.M, N = p.N, K = p.K, num_k_tiles = p.num_k_tiles;
  const int n0 = blockIdx.x * kMmaCols;
  const int KB = K / kBlockSize;

  const int n = n0 + g;
  const bool col_ok = n < N;
  const uint8_t* brow = B + (size_t)n * K + tig * 8;
  const uint8_t* acol = A + tig * 8;
  // Scales needed at the accumulate step: rows g / g+8 (A) and columns tig*2 / tig*2+1
  // (B) of this block. Loaded inline with the weights (1 byte each, L2-resident) rather
  // than staged: a block-wide staging pass of 24 x KB scattered bytes was a serial
  // latency chain in front of the first mma when the scales are cold in the server.
  const int c0 = n0 + tig * 2, c1 = c0 + 1;
  const bool c0_ok = c0 < N, c1_ok = c1 < N;

  float acc[MT][4];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[t][i] = 0.0f;

  constexpr int UNROLL = 4;
  for (int kb0 = warp; kb0 < KB; kb0 += kWarps * UNROLL) {
    uint2 wa0[MT][UNROLL], wa1[MT][UNROLL], wb[UNROLL];
    uint32_t sa0[MT][UNROLL], sa1[MT][UNROLL], sb0[UNROLL], sb1[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int kb = kb0 + u * kWarps;
      const bool ok = kb < KB;
      const size_t off = (size_t)kb * kBlockSize;
      const int kbc = ok ? kb : 0;
      wb[u] = (ok && col_ok) ? *reinterpret_cast<const uint2*>(brow + off) : make_uint2(0, 0);
      sb0[u] = c0_ok ? load_sf<B_SF>(B_sf, c0, kbc, num_k_tiles, 0, p.b_sf_ld) : 0u;
      sb1[u] = c1_ok ? load_sf<B_SF>(B_sf, c1, kbc, num_k_tiles, 0, p.b_sf_ld) : 0u;
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        const int r0 = g + 16 * t, r1 = r0 + 8;
        const bool r0_ok = r0 < M, r1_ok = r1 < M;
        wa0[t][u] = (ok && r0_ok) ? *reinterpret_cast<const uint2*>(acol + (size_t)r0 * K + off)
                                  : make_uint2(0, 0);
        wa1[t][u] = (ok && r1_ok) ? *reinterpret_cast<const uint2*>(acol + (size_t)r1 * K + off)
                                  : make_uint2(0, 0);
        sa0[t][u] = r0_ok ? load_sf<A_SF>(A_sf, r0, kbc, num_k_tiles, p.a_sf_t_al, 0) : 0u;
        sa1[t][u] = r1_ok ? load_sf<A_SF>(A_sf, r1, kbc, num_k_tiles, p.a_sf_t_al, 0) : 0u;
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int kb = kb0 + u * kWarps;
      if (kb >= KB) break;
      const float fb0 = ue8m0_to_float(sb0[u]), fb1 = ue8m0_to_float(sb1[u]);
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        const uint32_t afrag[4] = {wa0[t][u].x, wa1[t][u].x, wa0[t][u].y, wa1[t][u].y};
        float d[4];
        mma_m16n8k32_e4m3(d, afrag, wb[u].x, wb[u].y);
        // d0,d1: row g, cols tig*2 + {0,1}; d2,d3: row g + 8, same cols.
        const float fa0 = ue8m0_to_float(sa0[t][u]), fa1 = ue8m0_to_float(sa1[t][u]);
        acc[t][0] = fmaf(d[0], fa0 * fb0, acc[t][0]);
        acc[t][1] = fmaf(d[1], fa0 * fb1, acc[t][1]);
        acc[t][2] = fmaf(d[2], fa1 * fb0, acc[t][2]);
        acc[t][3] = fmaf(d[3], fa1 * fb1, acc[t][3]);
      }
    }
  }

  // Reduce the 8 K-split warps.
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) s_red[warp][lane][4 * t + i] = acc[t][i];
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      float r[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int w = 0; w < kWarps; ++w)
#pragma unroll
        for (int i = 0; i < 4; ++i) r[i] += s_red[w][lane][4 * t + i];
      const int r0 = g + 16 * t, r1 = r0 + 8;
      if (r0 < M) {
        if (c0_ok) C[(size_t)r0 * p.ldc + c0] = __float2bfloat16(r[0]);
        if (c1_ok) C[(size_t)r0 * p.ldc + c1] = __float2bfloat16(r[1]);
      }
      if (r1 < M) {
        if (c0_ok) C[(size_t)r1 * p.ldc + c0] = __float2bfloat16(r[2]);
        if (c1_ok) C[(size_t)r1 * p.ldc + c1] = __float2bfloat16(r[3]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// v2 (2026-09-19): same math, different work distribution. In the server the v1 kernel
// streamed the dense weights at 0.4-1.1 TB/s: 144-224 blocks for the N = 1152 / 1792
// shapes (under one block per SM), 24 scattered loads per lane per 4 K-blocks (one
// 8-byte weight load + two 1-byte swizzled scale loads + A + two A-scale bytes per
// K-block), and 32-byte weight segments spread over 4 warps. v2:
//   * a warp owns whole 128-wide K tiles (kt = 4 MX blocks): the 4 weight loads of a
//     lane cover 128 contiguous bytes per row, and the 4 scale bytes of a row for one
//     kt sit in one 32-bit word in all three scale layouts -> one word load per row
//     per kt instead of 4 byte loads (F8_128x4: word ((mt*nkt+kt)*32+b)*4+a;
//     row-major: bytes kt*4..+3; MN-packed: exactly one word per (row, kt));
//   * split-K across blocks (gridDim.z = S) when the column count leaves the GPU
//     under-filled: every block writes its fp32 partial [M, 8 cols] to a workspace,
//     bumps a per-column-block counter, and the last block to arrive sums the S
//     partials in fixed order (deterministic) and writes bf16 -- no extra launch.
//     Counters live in a per-stream buffer that the last block resets to 0, so the
//     scheme is CUDA-graph replay safe without a memset node.
// Both layouts of the result are bit-for-bit the same accumulation formula as v1
// (fp32 MMA result x 2^(ea+eb-254) per MX block, fp32 sum); only the summation order
// changes, i.e. the same class of differences as between v1 and CUTLASS.
constexpr int kKtBlocks = 4;  // MX blocks per 128-wide K tile

template <int LAYOUT>
__device__ __forceinline__ uint32_t load_sf_word(const uint8_t* __restrict__ sf, int row, int kt,
                                                 int num_k_tiles, int t_al, int ld) {
  if constexpr (LAYOUT == SF_F8_128x4) {
    const int mt = row >> 7, r = row & 127, a = r >> 5, b = r & 31;
    return reinterpret_cast<const uint32_t*>(sf)[(((mt * num_k_tiles + kt) * 32 + b) * 4 + a)];
  } else if constexpr (LAYOUT == SF_ROWMAJOR) {
    return *reinterpret_cast<const uint32_t*>(sf + (size_t)row * ld + (size_t)kt * 4);
  } else {
    return reinterpret_cast<const uint32_t*>(sf)[(size_t)kt * t_al + row];
  }
}

__device__ __forceinline__ float sf_byte_to_float(uint32_t word, int u) {
  return ue8m0_to_float((word >> (8 * u)) & 0xffu);
}

struct SplitKArgs {
  float* partials;   // [S][groups][M][N] fp32
  int* counters;     // [groups * col_blocks], zero at rest
  int S;
};

// ABL (perf ablations, dense only): 1 = no A loads, 2 = no B loads, 3 = no scale loads,
// 4 = no MMA (sum of raw words), 5 = loads only (no math, no epilogue writes except one)
template <int MT, int A_SF, int B_SF, int UKT, bool SA = false, int ABL = 0>
__global__ void __launch_bounds__(kThreads) mxfp8_mma_gemv_v2_kernel(const GemvArgs p,
                                                                     const SplitKArgs sk) {
  __shared__ float s_red[kWarps][32][4 * MT];
  __shared__ int s_last;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int tig = lane & 3;
  const int grp = blockIdx.y;
  const uint8_t* __restrict__ A = p.A + grp * p.a_gstride;
  const uint8_t* __restrict__ A_sf = p.A_sf + grp * p.a_sf_gstride;
  const uint8_t* __restrict__ B = p.B + grp * p.b_gstride;
  const uint8_t* __restrict__ B_sf = p.B_sf + grp * p.b_sf_gstride;
  __nv_bfloat16* __restrict__ C = p.C + grp * p.c_gstride;
  const int M = p.M, N = p.N, K = p.K, num_k_tiles = p.num_k_tiles;
  const int n0 = blockIdx.x * kMmaCols;
  const int KB = K / kBlockSize;
  const int KT = (KB + kKtBlocks - 1) / kKtBlocks;
  // balanced kt range of this split
  const int S = sk.S, s = blockIdx.z;
  const int kt_begin = (KT * s) / S, kt_end = (KT * (s + 1)) / S;

  const int n = n0 + g;
  const bool col_ok = n < N;
  const uint8_t* brow = B + (size_t)(col_ok ? n : 0) * K + tig * 8;
  const uint8_t* acol = A + tig * 8;
  if constexpr (ABL == 7) acol = B + (size_t)((blockIdx.x * 16) % (N - 16)) * K + tig * 8;
  const int c0 = n0 + tig * 2, c1 = c0 + 1;
  const bool c0_ok = c0 < N, c1_ok = c1 < N;

  float acc[MT][4];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[t][i] = 0.0f;

  // SA: the block's A slice (rows < M, K tiles [kt_begin, kt_end)) and its scale words are
  // staged in dynamic shared memory with 16-byte coalesced loads after the first round of
  // weight loads is in flight; every block otherwise re-reads the same 30 KB of A from
  // L2 with 8-byte row-strided loads, and with 150-600 blocks doing that at once the few
  // L2 slices holding A become the bottleneck (measured 2026-09-19: -2 us per call).
  extern __shared__ __align__(16) uint8_t dyn_smem[];
  const int kt_span = kt_end - kt_begin;
  const int lda_s = SA ? (p.lda_s) : 0;
  const int lds_s = SA ? (kt_span | 1) : 0;
  uint8_t* s_a = dyn_smem;
  uint32_t* s_as = reinterpret_cast<uint32_t*>(dyn_smem + (size_t)(SA ? 16 : 0) * lda_s);
  const int k_begin = kt_begin * kKtBlocks * kBlockSize;

  auto load_b_round = [&](int kt0, uint2 (&wb)[UKT][kKtBlocks], uint32_t (&sb0)[UKT],
                          uint32_t (&sb1)[UKT]) {
#pragma unroll
    for (int v = 0; v < UKT; ++v) {
      const int kt = kt0 + v * kWarps;
      const bool kt_ok = kt < kt_end;
      const int ktc = kt_ok ? kt : kt_begin;
      sb0[v] = (ABL != 3 && ABL != 6 && kt_ok && c0_ok) ? load_sf_word<B_SF>(B_sf, c0, ktc, num_k_tiles, 0, p.b_sf_ld) : 0x7f7f7f7fu;
      sb1[v] = (ABL != 3 && ABL != 6 && kt_ok && c1_ok) ? load_sf_word<B_SF>(B_sf, c1, ktc, num_k_tiles, 0, p.b_sf_ld) : 0x7f7f7f7fu;
#pragma unroll
      for (int u = 0; u < kKtBlocks; ++u) {
        const int kb = ktc * kKtBlocks + u;
        const bool ok = kt_ok && kb < KB;
        if constexpr (ABL == 8) {  // as if the 8 rows' 32-byte chunks were interleaved: one 256-byte segment per warp load
          const uint8_t* bi = B + (size_t)blockIdx.x * 8 * K + (size_t)kb * 256 + g * 32 + tig * 8;
          wb[v][u] = ok ? *reinterpret_cast<const uint2*>(bi) : make_uint2(0, 0);
        } else {
          wb[v][u] = (ABL != 2 && ABL != 6 && ok && col_ok)
                         ? *reinterpret_cast<const uint2*>(brow + (size_t)kb * kBlockSize)
                         : make_uint2(0, 0);
        }
      }
    }
  };

  uint2 wb[UKT][kKtBlocks];
  uint32_t sb0[UKT], sb1[UKT];
  int kt0 = kt_begin + warp;
  load_b_round(kt0, wb, sb0, sb1);

  if constexpr (SA) {
    const int k_len = min(kt_span * kKtBlocks * kBlockSize, K - k_begin);
    const int vec_per_row = k_len / 16;
    for (int idx = threadIdx.x; idx < M * vec_per_row; idx += kThreads) {
      const int r = idx / vec_per_row, v = idx - r * vec_per_row;
      *reinterpret_cast<uint4*>(s_a + (size_t)r * lda_s + v * 16) =
          *reinterpret_cast<const uint4*>(A + (size_t)r * K + k_begin + v * 16);
    }
    for (int idx = threadIdx.x; idx < M * kt_span; idx += kThreads) {
      const int r = idx / kt_span, kt = idx - r * kt_span;
      s_as[r * lds_s + kt] = load_sf_word<A_SF>(A_sf, r, kt_begin + kt, num_k_tiles, p.a_sf_t_al, 0);
    }
    __syncthreads();
  }

  while (true) {
    uint2 wa0[MT][UKT][kKtBlocks], wa1[MT][UKT][kKtBlocks];
    uint32_t sa0[MT][UKT], sa1[MT][UKT];
#pragma unroll
    for (int v = 0; v < UKT; ++v) {
      const int kt = kt0 + v * kWarps;
      const bool kt_ok = kt < kt_end;
      const int ktc = kt_ok ? kt : kt_begin;
#pragma unroll
      for (int u = 0; u < kKtBlocks; ++u) {
        const int kb = ktc * kKtBlocks + u;
        const bool ok = kt_ok && kb < KB;
        const size_t off = (size_t)kb * kBlockSize;
#pragma unroll
        for (int t = 0; t < MT; ++t) {
          const int r0 = g + 16 * t, r1 = r0 + 8;
          if constexpr (SA) {
            const size_t soff = (size_t)(kb - kt_begin * kKtBlocks) * kBlockSize + tig * 8;
            wa0[t][v][u] = (ok && r0 < M) ? *reinterpret_cast<const uint2*>(s_a + (size_t)r0 * lda_s + soff)
                                          : make_uint2(0, 0);
            wa1[t][v][u] = (ok && r1 < M) ? *reinterpret_cast<const uint2*>(s_a + (size_t)r1 * lda_s + soff)
                                          : make_uint2(0, 0);
          } else {
            wa0[t][v][u] = (ABL != 1 && ABL != 6 && ok && r0 < M) ? *reinterpret_cast<const uint2*>(acol + (size_t)r0 * K + off)
                                                      : make_uint2(0, 0);
            wa1[t][v][u] = (ABL != 1 && ABL != 6 && ok && r1 < M) ? *reinterpret_cast<const uint2*>(acol + (size_t)r1 * K + off)
                                                      : make_uint2(0, 0);
          }
        }
      }
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        const int r0 = g + 16 * t, r1 = r0 + 8;
        if constexpr (SA) {
          sa0[t][v] = (kt_ok && r0 < M) ? s_as[r0 * lds_s + (ktc - kt_begin)] : 0x7f7f7f7fu;
          sa1[t][v] = (kt_ok && r1 < M) ? s_as[r1 * lds_s + (ktc - kt_begin)] : 0x7f7f7f7fu;
        } else {
          sa0[t][v] = (ABL != 3 && ABL != 6 && kt_ok && r0 < M) ? load_sf_word<A_SF>(A_sf, r0, ktc, num_k_tiles, p.a_sf_t_al, 0) : 0x7f7f7f7fu;
          sa1[t][v] = (ABL != 3 && ABL != 6 && kt_ok && r1 < M) ? load_sf_word<A_SF>(A_sf, r1, ktc, num_k_tiles, p.a_sf_t_al, 0) : 0x7f7f7f7fu;
        }
      }
    }
#pragma unroll
    for (int v = 0; v < UKT; ++v) {
      const int kt = kt0 + v * kWarps;
      if (kt >= kt_end) break;
#pragma unroll
      for (int u = 0; u < kKtBlocks; ++u) {
        if (kt * kKtBlocks + u >= KB) break;
        const float fb0 = sf_byte_to_float(sb0[v], u), fb1 = sf_byte_to_float(sb1[v], u);
#pragma unroll
        for (int t = 0; t < MT; ++t) {
          const uint32_t afrag[4] = {wa0[t][v][u].x, wa1[t][v][u].x, wa0[t][v][u].y, wa1[t][v][u].y};
          float d[4];
          if constexpr (ABL == 4 || ABL == 5 || ABL == 6) {
            d[0] = __uint_as_float((afrag[0] ^ wb[v][u].x) & 0x3fffffffu);
            d[1] = __uint_as_float((afrag[1] ^ wb[v][u].y) & 0x3fffffffu);
            d[2] = __uint_as_float((afrag[2] ^ wb[v][u].x) & 0x3fffffffu);
            d[3] = __uint_as_float((afrag[3] ^ wb[v][u].y) & 0x3fffffffu);
          } else {
            mma_m16n8k32_e4m3(d, afrag, wb[v][u].x, wb[v][u].y);
          }
          const float fa0 = sf_byte_to_float(sa0[t][v], u), fa1 = sf_byte_to_float(sa1[t][v], u);
          acc[t][0] = fmaf(d[0], fa0 * fb0, acc[t][0]);
          acc[t][1] = fmaf(d[1], fa0 * fb1, acc[t][1]);
          acc[t][2] = fmaf(d[2], fa1 * fb0, acc[t][2]);
          acc[t][3] = fmaf(d[3], fa1 * fb1, acc[t][3]);
        }
      }
    }
    kt0 += kWarps * UKT;
    if (kt0 >= kt_end) break;
    load_b_round(kt0, wb, sb0, sb1);
  }

  // Reduce the 8 K-split warps.
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) s_red[warp][lane][4 * t + i] = acc[t][i];
  __syncthreads();
  if (warp != 0) return;
  float r[MT][4];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float v = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) v += s_red[w][lane][4 * t + i];
      r[t][i] = v;
    }

  if (S > 1) {
    // Publish this split's partial, then let the last-arriving block finish the column block.
    const int col_blocks = gridDim.x;
    float* part = sk.partials + ((size_t)s * gridDim.y + grp) * ((size_t)M * N);
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      const int r0 = g + 16 * t, r1 = r0 + 8;
      if (r0 < M) {
        if (c0_ok) part[(size_t)r0 * N + c0] = r[t][0];
        if (c1_ok) part[(size_t)r0 * N + c1] = r[t][1];
      }
      if (r1 < M) {
        if (c0_ok) part[(size_t)r1 * N + c0] = r[t][2];
        if (c1_ok) part[(size_t)r1 * N + c1] = r[t][3];
      }
    }
    __threadfence();
    __syncwarp();
    if (lane == 0) {
      const int prev = atomicAdd(&sk.counters[grp * col_blocks + blockIdx.x], 1);
      s_last = (prev == S - 1);
    }
    __syncwarp();
    if (!s_last) return;
    __threadfence();
    if (lane == 0) sk.counters[grp * col_blocks + blockIdx.x] = 0;  // re-arm for the next launch
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) r[t][i] = 0.0f;
    for (int q = 0; q < S; ++q) {  // fixed order -> deterministic
      const float* pq = sk.partials + ((size_t)q * gridDim.y + grp) * ((size_t)M * N);
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        const int r0 = g + 16 * t, r1 = r0 + 8;
        if (r0 < M) {
          if (c0_ok) r[t][0] += __ldcg(pq + (size_t)r0 * N + c0);
          if (c1_ok) r[t][1] += __ldcg(pq + (size_t)r0 * N + c1);
        }
        if (r1 < M) {
          if (c0_ok) r[t][2] += __ldcg(pq + (size_t)r1 * N + c0);
          if (c1_ok) r[t][3] += __ldcg(pq + (size_t)r1 * N + c1);
        }
      }
    }
  }

#pragma unroll
  for (int t = 0; t < MT; ++t) {
    const int r0 = g + 16 * t, r1 = r0 + 8;
    if (r0 < M) {
      if (c0_ok) C[(size_t)r0 * p.ldc + c0] = __float2bfloat16(r[t][0]);
      if (c1_ok) C[(size_t)r0 * p.ldc + c1] = __float2bfloat16(r[t][1]);
    }
    if (r1 < M) {
      if (c0_ok) C[(size_t)r1 * p.ldc + c0] = __float2bfloat16(r[t][2]);
      if (c1_ok) C[(size_t)r1 * p.ldc + c1] = __float2bfloat16(r[t][3]);
    }
  }
}

// Per-stream split-K scratch: counters stay zero between launches (the last block of
// every column block resets its counter), partials are overwritten every launch.
struct SplitKScratch {
  torch::Tensor counters;  // int32 [max_col_blocks]
  torch::Tensor partials;  // fp32 [n_floats]
};

// Fixed sizes: the buffers' addresses get baked into captured CUDA graphs, so they are
// never reallocated; a launch that does not fit runs without split-K instead.
constexpr int64_t kSplitKCounters = 4096;
constexpr int64_t kSplitKFloats = 1 << 20;  // 4 MB: S x groups x M x N fp32

static SplitKScratch* splitk_scratch(cudaStream_t stream, int device, int64_t n_counters,
                                     int64_t n_floats) {
  if (n_counters > kSplitKCounters || n_floats > kSplitKFloats) return nullptr;
  static std::mutex mu;
  static std::unordered_map<cudaStream_t, SplitKScratch> pool;
  std::lock_guard<std::mutex> lock(mu);
  auto& sc = pool[stream];
  if (!sc.counters.defined()) {
    const auto opts = torch::TensorOptions().device(torch::kCUDA, device);
    sc.counters = torch::zeros({kSplitKCounters}, opts.dtype(torch::kInt32));
    sc.partials = torch::empty({kSplitKFloats}, opts.dtype(torch::kFloat32));
  }
  return &sc;
}

static int env_int(const char* name, int def) {
  const char* v = std::getenv(name);
  return v == nullptr ? def : std::atoi(v);
}

// VLLM_MOET_GEMV_IMPL: "v1" (default -- the served kernel), "v2" (experiment above: no
// gain on the served shapes, see tools/dsv41_sm120/README.md), "scalar".
static int gemv_impl_version() {
  static const int v = [] {
    const char* s = std::getenv("VLLM_MOET_GEMV_IMPL");
    if (s == nullptr) return 1;
    const std::string str(s);
    if (str == "v2") return 2;
    if (str == "scalar") return 0;
    return 1;
  }();
  return v;
}

// Work distribution (measured 2026-09-19, tools/dsv41_sm120/README.md): with 1-3 blocks per
// SM these GEMVs are bound by the chain of memory round trips per warp, not by bandwidth --
// each "round" (issue the loads of UKT K tiles, wait, MMA) costs a full DRAM latency while
// the whole grid is quiet. So every warp should issue all of its K tiles in ONE round:
// UKT = ceil(KT_per_split / 8), capped at kMaxUkt (register file); K tiles beyond that
// are split across blocks (gridDim.z) and reduced by the last-arriving block.
constexpr int kMaxUkt = 5;      // 5 x (4 weight + 8 A + 4 scale) loads in flight per lane
constexpr int kMaxUktWide = 2;  // M > 16 (MT >= 2): twice the A fragments per K tile

template <typename Kernel>
static void set_dyn_smem(Kernel kernel, size_t smem) {
  static std::mutex mu;
  static std::unordered_map<const void*, size_t> configured;
  std::lock_guard<std::mutex> lock(mu);
  auto& cur = configured[reinterpret_cast<const void*>(kernel)];
  if (smem > cur) {
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) ==
                    cudaSuccess,
                "cannot reserve ", smem, " bytes of dynamic shared memory");
    cur = smem;
  }
}

template <int MT, int A_SF, int B_SF, bool SA>
static void launch_v2_ukt(int ukt, const dim3& grid, const dim3& block, const GemvArgs& p,
                          const SplitKArgs& sk, size_t smem, cudaStream_t stream) {
#define VLLM_MOET_V2_LAUNCH(MTV, UKTV)                                                          \
  {                                                                                            \
    auto* kfn = mxfp8_mma_gemv_v2_kernel<MTV, A_SF, B_SF, UKTV, SA>;                           \
    if (smem > 0) set_dyn_smem(kfn, smem);                                                     \
    kfn<<<grid, block, smem, stream>>>(p, sk);                                                 \
  }
  switch (ukt) {
    case 1: VLLM_MOET_V2_LAUNCH(MT, 1) break;
    case 2: VLLM_MOET_V2_LAUNCH(MT, 2) break;
    default:
      if constexpr (MT == 1) {
        switch (ukt) {
          case 3: VLLM_MOET_V2_LAUNCH(1, 3) break;
          case 4: VLLM_MOET_V2_LAUNCH(1, 4) break;
          default: VLLM_MOET_V2_LAUNCH(1, 5) break;
        }
      } else {
        VLLM_MOET_V2_LAUNCH(MT, 2)
      }
  }
#undef VLLM_MOET_V2_LAUNCH
}

template <int A_SF, int B_SF>
static void launch_mma_v2(const GemvArgs& p, int groups, int device, cudaStream_t stream) {
  const int col_blocks = (p.N + kMmaCols - 1) / kMmaCols;
  const int KT = (p.K / kBlockSize + kKtBlocks - 1) / kKtBlocks;
  const int mt = p.M <= 16 ? 1 : (p.M <= 32 ? 2 : (p.M <= 48 ? 3 : 4));
  const int max_ukt = env_int("VLLM_MOET_GEMV_UKT_MAX", mt == 1 ? kMaxUkt : kMaxUktWide);
  static const int max_split = env_int("VLLM_MOET_GEMV_SPLITK_MAX", 8);
  // smallest split count that lets one round of max_ukt tiles per warp cover the split
  int S = std::min(max_split, std::max(1, (KT + kWarps * max_ukt - 1) / (kWarps * max_ukt)));
  int kt_per_split = (KT + S - 1) / S;
  int ukt = std::min(max_ukt, std::max(1, (kt_per_split + kWarps - 1) / kWarps));
  static const int force_ukt = env_int("VLLM_MOET_GEMV_UKT", 0);
  if (force_ukt > 0) ukt = std::min(force_ukt, mt == 1 ? kMaxUkt : kMaxUktWide);
  static const int force_split = env_int("VLLM_MOET_GEMV_SPLITK", 0);
  if (force_split > 0) S = force_split;
  SplitKArgs sk{nullptr, nullptr, S};
  if (S > 1) {
    SplitKScratch* sc = splitk_scratch(stream, device, (int64_t)groups * col_blocks,
                                       (int64_t)S * groups * p.M * p.N);
    if (sc == nullptr) {
      sk.S = S = 1;
    } else {
      sk.partials = sc->partials.data_ptr<float>();
      sk.counters = sc->counters.data_ptr<int>();
    }
  }
  const dim3 grid(col_blocks, groups, S), block(kThreads);
  static const int abl = env_int("VLLM_MOET_GEMV_ABLATE", 0);
  if (mt == 1 && abl != 0 && A_SF == SF_F8_128x4) {  // perf ablations, UKT = 1
    switch (abl) {
      case 1: mxfp8_mma_gemv_v2_kernel<1, A_SF, B_SF, 1, false, 1><<<grid, block, 0, stream>>>(p, sk); break;
      case 2: mxfp8_mma_gemv_v2_kernel<1, A_SF, B_SF, 1, false, 2><<<grid, block, 0, stream>>>(p, sk); break;
      case 3: mxfp8_mma_gemv_v2_kernel<1, A_SF, B_SF, 1, false, 3><<<grid, block, 0, stream>>>(p, sk); break;
      case 6: mxfp8_mma_gemv_v2_kernel<1, A_SF, B_SF, 1, false, 6><<<grid, block, 0, stream>>>(p, sk); break;
      case 7: mxfp8_mma_gemv_v2_kernel<1, A_SF, B_SF, 1, false, 7><<<grid, block, 0, stream>>>(p, sk); break;
      case 8: mxfp8_mma_gemv_v2_kernel<1, A_SF, B_SF, 1, false, 8><<<grid, block, 0, stream>>>(p, sk); break;
      default: mxfp8_mma_gemv_v2_kernel<1, A_SF, B_SF, 1, false, 4><<<grid, block, 0, stream>>>(p, sk); break;
    }
    return;
  }
  // Stage A in shared memory (M <= 16) when the slice fits in 96 KB (>= 2 blocks per SM).
  static const int sa_enabled = env_int("VLLM_MOET_GEMV_STAGE_A", 1);
  GemvArgs pl = p;
  size_t smem = 0;
  bool sa = false;
  if (mt == 1 && sa_enabled) {
    const int kt_span = (KT + S - 1) / S;
    const int slice = kt_span * kKtBlocks * kBlockSize;              // bytes per row
    const int words = slice / 4;
    const int pad_words = ((8 - (words % 32)) % 32 + 32) % 32;      // rows on distinct banks
    pl.lda_s = slice + pad_words * 4;
    smem = (size_t)16 * pl.lda_s + (size_t)16 * (kt_span | 1) * 4;
    if (smem <= 96 * 1024) {
      sa = true;
    } else {
      smem = 0;
    }
  }
  if (sa) {
    launch_v2_ukt<1, A_SF, B_SF, true>(ukt, grid, block, pl, sk, smem, stream);
    return;
  }
  switch (mt) {
    case 1: launch_v2_ukt<1, A_SF, B_SF, false>(ukt, grid, block, p, sk, 0, stream); break;
    case 2: launch_v2_ukt<2, A_SF, B_SF, false>(ukt, grid, block, p, sk, 0, stream); break;
    case 3: launch_v2_ukt<3, A_SF, B_SF, false>(ukt, grid, block, p, sk, 0, stream); break;
    default: launch_v2_ukt<4, A_SF, B_SF, false>(ukt, grid, block, p, sk, 0, stream); break;
  }
}

template <int A_SF, int B_SF>
static void launch_mma(const GemvArgs& p, int groups, cudaStream_t stream) {
  const dim3 grid((p.N + kMmaCols - 1) / kMmaCols, groups), block(kThreads);
  if (p.M <= 16) {
    mxfp8_mma_gemv_kernel<1, A_SF, B_SF><<<grid, block, 0, stream>>>(p);
  } else if (p.M <= 32) {
    mxfp8_mma_gemv_kernel<2, A_SF, B_SF><<<grid, block, 0, stream>>>(p);
  } else if (p.M <= 48) {
    mxfp8_mma_gemv_kernel<3, A_SF, B_SF><<<grid, block, 0, stream>>>(p);
  } else {
    mxfp8_mma_gemv_kernel<4, A_SF, B_SF><<<grid, block, 0, stream>>>(p);
  }
}

template <int MAXM, int CHUNK>
static void launch(const uint8_t* A, const uint8_t* Asf, const uint8_t* B, const uint8_t* Bsf,
                   __nv_bfloat16* Cp, torch::Tensor& a, int M, int N, int K, int num_k_tiles,
                   cudaStream_t stream) {
  const dim3 block(kThreads);
  const int col_blocks = (N + kWarps - 1) / kWarps;
  const int num_chunks = (K + CHUNK - 1) / CHUNK;
  // Split over K chunks only when the column count alone leaves the GPU under-filled.
  if (num_chunks > 1 && col_blocks < 400) {
    auto cf = torch::empty({num_chunks, M, N}, a.options().dtype(torch::kFloat32));
    float* Cf = cf.data_ptr<float>();
    mxfp8_gemv_kernel<MAXM, CHUNK, true><<<dim3(col_blocks, num_chunks), block, 0, stream>>>(
        A, Asf, B, Bsf, Cp, Cf, M, N, K, num_k_tiles);
    const int total = M * N;
    finalize_kernel<<<(total + 255) / 256, 256, 0, stream>>>(Cf, Cp, total, num_chunks);
  } else {
    mxfp8_gemv_kernel<MAXM, CHUNK, false><<<dim3(col_blocks), block, 0, stream>>>(
        A, Asf, B, Bsf, Cp, nullptr, M, N, K, num_k_tiles);
  }
}

torch::Tensor mxfp8_gemv(torch::Tensor a, torch::Tensor a_sf, torch::Tensor b,
                         torch::Tensor b_sf) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a_sf.is_cuda() && b_sf.is_cuda(), "cuda tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a [M,K], b [N,K]");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && a_sf.is_contiguous() && b_sf.is_contiguous(),
              "contiguous inputs");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kFloat8_e4m3fn,
              "e4m3 operands");
  const int M = a.size(0), K = a.size(1), N = b.size(0);
  TORCH_CHECK(b.size(1) == K, "K mismatch");
  TORCH_CHECK(M >= 1 && M <= 64, "M must be in [1, 64]");
  TORCH_CHECK(K % 32 == 0 && K >= 32, "K must be a positive multiple of 32");
  const int num_k_tiles = (K + 127) / 128;
  TORCH_CHECK(a_sf.numel() >= (int64_t)((M + 127) / 128) * 128 * num_k_tiles * 4, "a_sf too small");
  TORCH_CHECK(b_sf.numel() >= (int64_t)((N + 127) / 128) * 128 * num_k_tiles * 4, "b_sf too small");

  auto c = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const uint8_t* A = reinterpret_cast<const uint8_t*>(a.data_ptr());
  const uint8_t* B = reinterpret_cast<const uint8_t*>(b.data_ptr());
  const uint8_t* Asf = reinterpret_cast<const uint8_t*>(a_sf.data_ptr());
  const uint8_t* Bsf = reinterpret_cast<const uint8_t*>(b_sf.data_ptr());
  __nv_bfloat16* Cp = reinterpret_cast<__nv_bfloat16*>(c.data_ptr());
  const int impl = gemv_impl_version();
  if (impl >= 1) {
    GemvArgs p{};
    p.A = A; p.A_sf = Asf; p.B = B; p.B_sf = Bsf; p.C = Cp;
    p.M = M; p.N = N; p.K = K; p.num_k_tiles = num_k_tiles; p.ldc = N;
    if (impl == 2) {
      launch_mma_v2<SF_F8_128x4, SF_F8_128x4>(p, 1, a.get_device(), stream);
    } else {
      launch_mma<SF_F8_128x4, SF_F8_128x4>(p, 1, stream);
    }
  } else if (M <= 8) {
    launch<8, 5120>(A, Asf, B, Bsf, Cp, a, M, N, K, num_k_tiles, stream);   // 40 KB + 5 KB smem
  } else {
    TORCH_CHECK(M <= 16, "scalar GEMV supports M <= 16");
    launch<16, 2560>(A, Asf, B, Bsf, Cp, a, M, N, K, num_k_tiles, stream);  // 40 KB + 5 KB smem
  }
  return c;
}

// Grouped o-projection GEMV (DeepSeek-V4 `wo_a`): for every head group g
//   Z[t, g, :] = A[g, t, :] (e4m3) x W[g*Ng:(g+1)*Ng, :]^T (e4m3)
// a      : [G, T, K] e4m3, contiguous (fused_inv_rope_fp8_quant's out_buf; the vLLM
//          call site sees it transposed to [T, G, K])
// a_sf   : int32 view of shape [G, T, S] with strides (S*T_al, 1, T_al), S = ceil(K/128):
//          fused_inv_rope_fp8_quant(tma_aligned_scales=True) output transposed back to
//          group-major. Word (g, t, s) packs the ue8m0 scales of K-blocks 4s..4s+3.
// w      : [G*Ng, K] e4m3 (the layer's MXFP8 weight as loaded)
// w_sf   : [G*Ng, K/32] ue8m0 (the layer's weight_scale as loaded, row-major)
// returns Z as [T, G, Ng] bf16.
torch::Tensor mxfp8_gemv_grouped(torch::Tensor a, torch::Tensor a_sf, torch::Tensor w,
                                 torch::Tensor w_sf) {
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && a_sf.is_cuda() && w_sf.is_cuda(), "cuda tensors");
  TORCH_CHECK(a.dim() == 3 && a.is_contiguous(), "a must be a contiguous [G, T, K] tensor");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && w.scalar_type() == torch::kFloat8_e4m3fn,
              "e4m3 operands");
  TORCH_CHECK(a_sf.scalar_type() == torch::kInt32 && a_sf.dim() == 3, "a_sf: int32 [G, T, S] view");
  TORCH_CHECK(w.dim() == 2 && w.is_contiguous(), "w must be a contiguous [G*Ng, K] tensor");
  TORCH_CHECK(w_sf.scalar_type() == torch::kUInt8 && w_sf.dim() == 2 && w_sf.is_contiguous(),
              "w_sf must be a contiguous uint8 [G*Ng, K/32] tensor");
  const int G = a.size(0), T = a.size(1), K = a.size(2);
  TORCH_CHECK(T >= 1 && T <= 64, "T must be in [1, 64]");
  TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128 (packed activation scales)");
  TORCH_CHECK(w.size(1) == K, "K mismatch");
  TORCH_CHECK(w.size(0) % G == 0, "weight rows must split evenly into groups");
  const int Ng = w.size(0) / G;
  TORCH_CHECK(w_sf.size(0) == w.size(0) && w_sf.size(1) == K / 32, "w_sf shape");
  const int S = a_sf.size(2);
  TORCH_CHECK(a_sf.size(0) == G && a_sf.size(1) == T && S * 4 >= K / 32, "a_sf shape");
  TORCH_CHECK(a_sf.stride(1) == 1, "a_sf must be MN-major (token stride 1)");
  const int64_t t_al = a_sf.stride(2);
  TORCH_CHECK(t_al >= T && a_sf.stride(0) == t_al * S, "a_sf strides");

  auto z = torch::empty({T, G, Ng}, a.options().dtype(torch::kBFloat16));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  GemvArgs p{};
  p.A = reinterpret_cast<const uint8_t*>(a.data_ptr());
  p.A_sf = reinterpret_cast<const uint8_t*>(a_sf.data_ptr());
  p.B = reinterpret_cast<const uint8_t*>(w.data_ptr());
  p.B_sf = reinterpret_cast<const uint8_t*>(w_sf.data_ptr());
  p.C = reinterpret_cast<__nv_bfloat16*>(z.data_ptr());
  p.M = T; p.N = Ng; p.K = K; p.num_k_tiles = (K + 127) / 128;
  p.ldc = G * Ng;
  p.a_gstride = (long long)T * K;
  p.a_sf_gstride = (long long)a_sf.stride(0) * 4;
  p.b_gstride = (long long)Ng * K;
  p.b_sf_gstride = (long long)Ng * (K / 32);
  p.c_gstride = Ng;
  p.a_sf_t_al = (int)t_al;
  p.b_sf_ld = K / 32;
  if (gemv_impl_version() == 2) {
    launch_mma_v2<SF_MN_PACKED, SF_ROWMAJOR>(p, G, a.get_device(), stream);
  } else {
    launch_mma<SF_MN_PACKED, SF_ROWMAJOR>(p, G, stream);
  }
  return z;
}

}  // namespace vllm_moet_sm120

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mxfp8_gemv", &vllm_moet_sm120::mxfp8_gemv,
        "MXFP8 x MXFP8 -> BF16 GEMV for M <= 64 (F8_128x4 swizzled scales)");
  m.def("mxfp8_gemv_grouped", &vllm_moet_sm120::mxfp8_gemv_grouped,
        "Grouped MXFP8 GEMV for the DeepSeek-V4 o-projection (T <= 64): "
        "a [G,T,K] e4m3 + packed MN-major ue8m0 scales, w [G*Ng,K] e4m3 + row-major "
        "ue8m0 scales -> [T, G, Ng] bf16");
}
