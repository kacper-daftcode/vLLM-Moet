// SPDX-License-Identifier: Apache-2.0
// Small-M FP8 block-scaled MoE GEMV for sm_120 -- a drop-in for vLLM's Triton
// `fused_moe_kernel` (fp8_w8a8, block_shape [32, 32]) at decode token counts.
//
// For every (token, expert) pair the Triton kernel computes a 16-row MMA tile per
// 64-column slab with BLOCK_SIZE_K capped at 32 by the block scales: 80 K-steps of
// `tl.dot` 16x32x64 + a per-step scale multiply, in ~200 programs of 2 warps for a
// 4-token decode step -- about half of the HBM bandwidth on RTX PRO 6000 (2026-09-18
// profile of Qwen3.8-Flash-Next-FP8 at TP4: ~28 us per expert GEMM per layer vs a
// ~15 us weight-stream floor).
//
// This kernel keeps the tensor-core inner product (mma.sync.m16n8k32 e4m3 -> f32,
// one MMA per 32-wide scale block, fp32 accumulation scaled by a_s[row] * b_s[col
// block] exactly like the Triton kernel) but maps the work like the sm_120 MXFP8
// dense GEMV (tools/dsv41_sm120/sm120_gemv): a block owns one m-block (one pair in
// vLLM's "naive" assignment for M*topk*4 <= E, or one 16-row `moe_align_block_size`
// block) and 8 * (8 / KSPLIT) output columns; its 8 warps split into KSPLIT K-slices
// x (8 / KSPLIT) column groups, every lane issues UNROLL 8-byte weight loads before
// the first MMA, and K-slices are reduced through shared memory. Grid =
// (N / block columns) x (m-blocks), i.e. ~1600 blocks of 256 threads for the 40
// pairs x 320 columns of the gate/up GEMM -- enough loads in flight to stream the
// touched experts at HBM speed (cold L2, 4 tokens: 41 -> 21 us gate/up, 8.7 -> 8.3 us
// down; 1 token: 15 -> 7 / 4.8 -> 3.4 us; outputs equal to Triton's within 1 bf16 ulp).
// In the aligned (16-row) mode the A tile and its scales are staged in shared memory
// once per block; loading them per warp made the 8-column-group variant L2-bound.
//
// Operand layouts are vLLM's (`invoke_fused_moe_triton_kernel` arguments):
//   A       [*, K] e4m3 rows = tokens (w13) or pairs (w2); a_scale [*, K/32] fp32
//   B       [E, N, K] e4m3; b_scale [E, N/32, K/32] fp32
//   C       bf16, row = pair id (sorted_token_ids entry), row stride ldc
//   sorted_token_ids [EM] int32 (nullptr = naive: block y == pair id)
//   expert_ids [m-blocks] int32 (-1 = expert not on this EP rank -> zeros)
//   num_tokens_post_padded [1] int32 (device; blocks past it exit)
//   topk_weights [pairs] fp32, multiplied into the fp32 result when given (w2).
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdlib>
#include <string>

namespace vllm_moet_sm120 {

constexpr int kBlockK = 32;  // block_shape[1]
constexpr int kBlockN = 32;  // block_shape[0]
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kMmaCols = 8;
constexpr int kMmaRows = 16;
constexpr int kInvalidPair = 0x7fffffff;

struct MoeGemvArgs {
  const uint8_t* A;
  const float* A_sf;
  const uint8_t* B;
  const float* B_sf;
  __nv_bfloat16* C;
  const float* topk_w;         // nullptr -> no routed-weight multiply
  const int32_t* sorted_ids;   // nullptr -> naive assignment
  const int32_t* expert_ids;
  const int32_t* npp;          // num_tokens_post_padded
  int N, K, KB;
  int num_valid;               // number of (token, expert) pairs
  int em;                      // sorted_ids length (entries past it are padding)
  int top_k;                   // A row = pair / top_k
  int npp_block;               // Triton BLOCK_SIZE_M: block exits when y * npp_block >= *npp
  int lda_s, lds_s;            // aligned path: shared-memory row strides of the A tile / scales
  const __nv_bfloat16* X;      // FUSE_ACT: bf16 [pairs, 2K] gate/up activations
  long long ldx;               // FUSE_ACT: X row stride (elements)
  int scale_ue8m0;             // FUSE_ACT: power-of-two activation scales (vLLM E8M0 mode)
  long long stride_am, stride_asm, stride_ask;
  long long stride_be, stride_bn, stride_bse, stride_bsn, stride_bsk;
  long long ldc;
};

__device__ __forceinline__ void mma_m16n8k32_e4m3(float* d, const uint32_t* a, uint32_t b0,
                                                  uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.0f), "f"(0.0f),
        "f"(0.0f), "f"(0.0f));
}

// Physical k inside a 32-wide scale block is permuted so that lane `tig` holds
// k = tig*8 .. tig*8+7 as {a0/b0 (first 4 bytes), a2/b1 (next 4)} -- one 8-byte load
// per operand row per block. A and B share the permutation, so the dot product is
// unchanged and every MMA stays inside one scale block.
// FUSE_ACT (naive path only): A is not the quantized intermediate but the bf16 gate/up
// output of the first GEMM, X[pair, 0:K] and X[pair, K:2K]; the block computes
// silu(gate) * up, quantizes it per 32-group (same formulas as vLLM's act_and_mul and
// per_token_group_quant_8bit kernels) into shared memory and runs the GEMM from there
// -- two launches less per layer.
template <int KSPLIT, int UNROLL, bool ALIGNED, bool FUSE_ACT = false>
__global__ void __launch_bounds__(kThreads) fused_moe_gemv_kernel(const MoeGemvArgs p) {
  static_assert(!(ALIGNED && FUSE_ACT), "fused activation is a naive-path feature");
  constexpr int NG = kWarps / KSPLIT;  // column groups (of 8) per block
  constexpr int BLOCK_COLS = NG * kMmaCols;
  __shared__ float s_red[(KSPLIT > 1 ? KSPLIT : 1)][NG][32][4];

  const int mb = blockIdx.y;
  if (mb * p.npp_block >= *p.npp) return;
  const int expert = p.expert_ids[mb];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;   // MMA groupID: A rows g / g+8, B column g
  const int tig = lane & 3;  // thread in group
  const int cg = warp / KSPLIT, ks = warp % KSPLIT;
  const int n0 = blockIdx.x * BLOCK_COLS + cg * kMmaCols;
  const int n = n0 + g;
  const bool col_ok = n < p.N;
  const int c0 = n0 + tig * 2, c1 = c0 + 1;
  const bool c0_ok = c0 < p.N, c1_ok = c1 < p.N;

  int pair0, pair1;
  if constexpr (ALIGNED) {
    const int i0 = mb * kMmaRows + g, i1 = i0 + 8;
    pair0 = i0 < p.em ? p.sorted_ids[i0] : kInvalidPair;
    pair1 = i1 < p.em ? p.sorted_ids[i1] : kInvalidPair;
  } else {
    pair0 = (g == 0) ? mb : kInvalidPair;
    pair1 = kInvalidPair;
  }
  const bool r0_ok = pair0 < p.num_valid, r1_ok = pair1 < p.num_valid;
  __nv_bfloat16* __restrict__ C = p.C;

  if (expert < 0) {  // expert not on this rank (expert parallel): zeros like Triton
    if (ks == 0) {
      const __nv_bfloat16 z = __float2bfloat16(0.0f);
      if (r0_ok) {
        if (c0_ok) C[(size_t)pair0 * p.ldc + c0] = z;
        if (c1_ok) C[(size_t)pair0 * p.ldc + c1] = z;
      }
      if (r1_ok) {
        if (c0_ok) C[(size_t)pair1 * p.ldc + c0] = z;
        if (c1_ok) C[(size_t)pair1 * p.ldc + c1] = z;
      }
    }
    return;
  }

  const int K = p.K, KB = p.KB;
  const uint8_t* __restrict__ bp =
      p.B + (size_t)expert * p.stride_be + (size_t)(col_ok ? n : 0) * p.stride_bn + tig * 8;
  const float* __restrict__ bs =
      p.B_sf + (size_t)expert * p.stride_bse + (size_t)(n0 / kBlockN) * p.stride_bsn;

  // Weight fragments of the current chunk (issued before anything else so the HBM
  // latency overlaps the A staging / A fragment loads).
  uint2 wb[UNROLL];
  float sb[UNROLL];
  auto load_b = [&](int kb0) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int kb = kb0 + u * KSPLIT;
      const bool ok = kb < KB;
      const int kbc = ok ? kb : 0;
      wb[u] = (ok && col_ok) ? *reinterpret_cast<const uint2*>(bp + (size_t)kbc * kBlockK)
                             : make_uint2(0u, 0u);
      sb[u] = ok ? bs[(size_t)kbc * p.stride_bsk] : 0.0f;
    }
  };
  load_b(ks);

  // A operand. Naive path: one valid row (lanes with g == 0) read straight from L2.
  // Aligned path: up to 16 rows shared by all NG column groups -> the tile [16, K] and
  // its scales [16, KB] are staged in (dynamic) shared memory once per block; loading
  // them per warp made the w2 GEMM (K = 160, 8 column groups) L2-bandwidth bound.
  // Row strides in shared memory are padded (host-side, see smem_bytes) so that the 8
  // rows read by a half-warp's 64-bit fragment loads hit distinct banks.
  extern __shared__ __align__(16) uint8_t dyn_smem[];
  uint8_t* s_a = dyn_smem;  // ALIGNED: [16][lda_s]; FUSE_ACT: [K]
  float* s_as = reinterpret_cast<float*>(dyn_smem + (FUSE_ACT ? K : kMmaRows * p.lda_s));
  const uint8_t* __restrict__ a0p = nullptr;
  const uint8_t* __restrict__ a1p = nullptr;
  const float* __restrict__ as0 = nullptr;
  const float* __restrict__ as1 = nullptr;
  if constexpr (ALIGNED) {
    const int vec_per_row = K / 16;
    const int total_vec = kMmaRows * vec_per_row;
    for (int idx = threadIdx.x; idx < total_vec; idx += kThreads) {
      const int r = idx / vec_per_row, v = idx - r * vec_per_row;
      const int i = mb * kMmaRows + r;
      const int pair = i < p.em ? p.sorted_ids[i] : kInvalidPair;
      uint4 val = make_uint4(0u, 0u, 0u, 0u);
      if (pair < p.num_valid) {
        val = *reinterpret_cast<const uint4*>(p.A + (size_t)(pair / p.top_k) * p.stride_am + v * 16);
      }
      *reinterpret_cast<uint4*>(s_a + r * p.lda_s + v * 16) = val;
    }
    for (int idx = threadIdx.x; idx < kMmaRows * KB; idx += kThreads) {
      const int r = idx / KB, kb = idx - r * KB;
      const int i = mb * kMmaRows + r;
      const int pair = i < p.em ? p.sorted_ids[i] : kInvalidPair;
      s_as[r * p.lds_s + kb] =
          pair < p.num_valid
              ? p.A_sf[(size_t)(pair / p.top_k) * p.stride_asm + (size_t)kb * p.stride_ask]
              : 0.0f;
    }
    __syncthreads();
    a0p = s_a + g * p.lda_s + tig * 8;
    a1p = s_a + (g + 8) * p.lda_s + tig * 8;
    as0 = s_as + g * p.lds_s;
    as1 = s_as + (g + 8) * p.lds_s;
  } else if constexpr (FUSE_ACT) {
    // Pair mb: v[e] = bf16(silu(gate[e])) * up[e] (bf16 product, as vLLM's act_and_mul),
    // then per 32-group: s = max(|v|, 1e-10) / 448, rounded up to a power of two when
    // vLLM's quant uses UE8M0 scales (is_deep_gemm_e8m0_used(), the case on this host),
    // q = fp8(clamp(v / s)) -- one warp per group, lane = element. Row 0 of the tile only.
    const __nv_bfloat16* __restrict__ xr = p.X + (size_t)mb * p.ldx;
    for (int kb = warp; kb < KB; kb += kWarps) {
      const int e = kb * kBlockK + lane;
      const float gt = __bfloat162float(xr[e]);
      const __nv_bfloat16 sg = __float2bfloat16(gt / (1.0f + expf(-gt)));
      const float v = __bfloat162float(__hmul(sg, xr[K + e]));
      float amax = fabsf(v);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
      float scale = fmaxf(amax, 1e-10f) / 448.0f;
      if (p.scale_ue8m0) scale = exp2f(ceilf(log2f(fmaxf(fabsf(scale), 1e-10f))));
      const float q = fminf(fmaxf(v / scale, -448.0f), 448.0f);
      s_a[e] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3));
      if (lane == 0) s_as[kb] = scale;
    }
    __syncthreads();
    a0p = s_a + tig * 8;
    as0 = s_as;
  } else {
    const int arow0 = r0_ok ? pair0 / p.top_k : 0;
    a0p = p.A + (size_t)arow0 * p.stride_am + tig * 8;
    as0 = p.A_sf + (size_t)arow0 * p.stride_asm;
  }
  const long long ask = (ALIGNED || FUSE_ACT) ? 1 : p.stride_ask;

  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int kb0 = ks;
  while (true) {
    uint2 wa0[UNROLL], wa1[UNROLL];
    float sa0[UNROLL], sa1[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int kb = kb0 + u * KSPLIT;
      const bool ok = kb < KB;
      const int kbc = ok ? kb : 0;
      const size_t off = (size_t)kbc * kBlockK;
      wa0[u] = (ok && r0_ok) ? *reinterpret_cast<const uint2*>(a0p + off) : make_uint2(0u, 0u);
      sa0[u] = (ok && r0_ok) ? as0[(size_t)kbc * ask] : 0.0f;
      if constexpr (ALIGNED) {
        wa1[u] = (ok && r1_ok) ? *reinterpret_cast<const uint2*>(a1p + off) : make_uint2(0u, 0u);
        sa1[u] = (ok && r1_ok) ? as1[(size_t)kbc * ask] : 0.0f;
      } else {
        wa1[u] = make_uint2(0u, 0u);
        sa1[u] = 0.0f;
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int kb = kb0 + u * KSPLIT;
      if (kb >= KB) break;
      const uint32_t afrag[4] = {wa0[u].x, wa1[u].x, wa0[u].y, wa1[u].y};
      float d[4];
      mma_m16n8k32_e4m3(d, afrag, wb[u].x, wb[u].y);
      // d0,d1: row g, cols tig*2 + {0,1}; d2,d3: row g+8, same cols.
      const float f0 = sa0[u] * sb[u], f1 = sa1[u] * sb[u];
      acc[0] = fmaf(d[0], f0, acc[0]);
      acc[1] = fmaf(d[1], f0, acc[1]);
      acc[2] = fmaf(d[2], f1, acc[2]);
      acc[3] = fmaf(d[3], f1, acc[3]);
    }
    kb0 += KSPLIT * UNROLL;
    if (kb0 >= KB) break;
    load_b(kb0);
  }

  if constexpr (KSPLIT > 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) s_red[ks][cg][lane][i] = acc[i];
    __syncthreads();
    if (ks != 0) return;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float r = 0.0f;
#pragma unroll
      for (int s = 0; s < KSPLIT; ++s) r += s_red[s][cg][lane][i];
      acc[i] = r;
    }
  }

  float w0 = 1.0f, w1 = 1.0f;
  if (p.topk_w != nullptr) {
    w0 = r0_ok ? p.topk_w[pair0] : 0.0f;
    w1 = r1_ok ? p.topk_w[pair1] : 0.0f;
  }
  if (r0_ok) {
    if (c0_ok) C[(size_t)pair0 * p.ldc + c0] = __float2bfloat16(acc[0] * w0);
    if (c1_ok) C[(size_t)pair0 * p.ldc + c1] = __float2bfloat16(acc[1] * w0);
  }
  if (r1_ok) {
    if (c0_ok) C[(size_t)pair1 * p.ldc + c0] = __float2bfloat16(acc[2] * w1);
    if (c1_ok) C[(size_t)pair1 * p.ldc + c1] = __float2bfloat16(acc[3] * w1);
  }
}

// Shared-memory strides of the staged A tile: the row stride in 32-bit words must be
// 8 mod 32 (8 rows x 8 banks of 64-bit fragments per half-warp) and keep 16-byte rows;
// the scale row stride must be odd.
static void smem_layout(int K, int KB, int& lda_s, int& lds_s, size_t& bytes) {
  const int words = K / 4;
  const int pad_words = ((8 - (words % 32)) % 32 + 32) % 32;  // multiple of 4 since K % 32 == 0
  lda_s = K + pad_words * 4;
  lds_s = KB | 1;
  bytes = (size_t)kMmaRows * lda_s + (size_t)kMmaRows * lds_s * sizeof(float);
}

enum Mode : int { MODE_NAIVE = 0, MODE_ALIGNED = 1, MODE_FUSED_ACT = 2 };

template <typename Kernel>
static void ensure_dyn_smem(Kernel kernel, size_t smem, size_t& configured) {
  // static (reduction) + dynamic (A tile) exceed the 48 KB default for K = 2560
  if (smem > configured) {
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem) == cudaSuccess,
                "A tile of ", smem, " bytes does not fit in shared memory");
    configured = smem;
  }
}

template <int KSPLIT, int UNROLL>
static void launch_cfg(MoeGemvArgs p, int m_blocks, int mode, cudaStream_t stream) {
  constexpr int BLOCK_COLS = (kWarps / KSPLIT) * kMmaCols;
  const dim3 grid((p.N + BLOCK_COLS - 1) / BLOCK_COLS, m_blocks), block(kThreads);
  if (mode == MODE_ALIGNED) {
    size_t smem = 0;
    smem_layout(p.K, p.KB, p.lda_s, p.lds_s, smem);
    static size_t configured = 0;  // per instantiation; grows monotonically
    ensure_dyn_smem(fused_moe_gemv_kernel<KSPLIT, UNROLL, true, false>, smem, configured);
    fused_moe_gemv_kernel<KSPLIT, UNROLL, true, false><<<grid, block, smem, stream>>>(p);
  } else if (mode == MODE_FUSED_ACT) {
    if constexpr (KSPLIT <= 2) {
      const size_t smem = (size_t)p.K + (size_t)p.KB * sizeof(float);
      static size_t configured = 0;
      ensure_dyn_smem(fused_moe_gemv_kernel<KSPLIT, UNROLL, false, true>, smem, configured);
      fused_moe_gemv_kernel<KSPLIT, UNROLL, false, true><<<grid, block, smem, stream>>>(p);
    } else {
      TORCH_CHECK(false, "fused activation variant is built for ksplit <= 2 only");
    }
  } else {
    fused_moe_gemv_kernel<KSPLIT, UNROLL, false, false><<<grid, block, 0, stream>>>(p);
  }
}

// ksplit in {1, 2, 4, 8}, unroll in {2, 4, 5, 8}
static void launch(const MoeGemvArgs& p, int m_blocks, int mode, int ksplit, int unroll,
                   cudaStream_t stream) {
#define VLLM_MOET_CASE(KS, UN)                                   \
  if (ksplit == KS && unroll == UN) {                            \
    launch_cfg<KS, UN>(p, m_blocks, mode, stream);               \
    return;                                                      \
  }
  VLLM_MOET_CASE(8, 2) VLLM_MOET_CASE(8, 4) VLLM_MOET_CASE(8, 5) VLLM_MOET_CASE(8, 8)
  VLLM_MOET_CASE(4, 2) VLLM_MOET_CASE(4, 4) VLLM_MOET_CASE(4, 5) VLLM_MOET_CASE(4, 8)
  VLLM_MOET_CASE(2, 2) VLLM_MOET_CASE(2, 4) VLLM_MOET_CASE(2, 5) VLLM_MOET_CASE(2, 8)
  VLLM_MOET_CASE(1, 2) VLLM_MOET_CASE(1, 4) VLLM_MOET_CASE(1, 5) VLLM_MOET_CASE(1, 8)
#undef VLLM_MOET_CASE
  TORCH_CHECK(false, "unsupported (ksplit, unroll) = (", ksplit, ", ", unroll, ")");
}

static void check_common(const torch::Tensor& B, const torch::Tensor& B_scale, const torch::Tensor& C,
                         const c10::optional<torch::Tensor>& topk_weights,
                         const torch::Tensor& expert_ids, const torch::Tensor& num_tokens_post_padded,
                         int64_t num_valid_tokens, int64_t m_blocks, int K) {
  TORCH_CHECK(B.is_cuda() && C.is_cuda(), "cuda tensors");
  TORCH_CHECK(B.scalar_type() == torch::kFloat8_e4m3fn, "e4m3 weights");
  TORCH_CHECK(B.dim() == 3 && B.stride(2) == 1, "B must be [E, N, K] with unit K stride");
  TORCH_CHECK(B_scale.scalar_type() == torch::kFloat32 && B_scale.dim() == 3, "b_scale fp32 [E, N/32, K/32]");
  TORCH_CHECK(C.scalar_type() == torch::kBFloat16, "bf16 output");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32 && expert_ids.is_contiguous(), "expert_ids int32");
  TORCH_CHECK(num_tokens_post_padded.scalar_type() == torch::kInt32, "num_tokens_post_padded int32");
  const int N = B.size(1);
  TORCH_CHECK(B.size(2) == K, "K mismatch");
  TORCH_CHECK(K % kBlockK == 0 && N % kBlockN == 0, "K and N must be multiples of 32");
  TORCH_CHECK(B_scale.size(0) == B.size(0) && B_scale.size(1) == N / kBlockN &&
                  B_scale.size(2) == K / kBlockK,
              "b_scale shape");
  TORCH_CHECK(B.stride(1) % 8 == 0, "8-byte aligned B rows");
  TORCH_CHECK(expert_ids.numel() >= m_blocks, "expert_ids too short for m_blocks");
  if (topk_weights.has_value()) {
    TORCH_CHECK(topk_weights->scalar_type() == torch::kFloat32 && topk_weights->is_contiguous() &&
                    topk_weights->numel() >= num_valid_tokens,
                "topk_weights fp32 contiguous [pairs]");
  }
}

static void fill_b(MoeGemvArgs& p, const torch::Tensor& B, const torch::Tensor& B_scale,
                   torch::Tensor& C, const c10::optional<torch::Tensor>& topk_weights,
                   const torch::Tensor& expert_ids, const torch::Tensor& num_tokens_post_padded,
                   int64_t num_valid_tokens, int64_t top_k, int64_t npp_block, int64_t ldc, int K) {
  p.B = reinterpret_cast<const uint8_t*>(B.data_ptr());
  p.B_sf = B_scale.data_ptr<float>();
  p.C = reinterpret_cast<__nv_bfloat16*>(C.data_ptr());
  p.topk_w = topk_weights.has_value() ? topk_weights->data_ptr<float>() : nullptr;
  p.expert_ids = expert_ids.data_ptr<int32_t>();
  p.npp = num_tokens_post_padded.data_ptr<int32_t>();
  p.N = B.size(1); p.K = K; p.KB = K / kBlockK;
  p.num_valid = (int)num_valid_tokens;
  p.top_k = (int)top_k;
  p.npp_block = (int)npp_block;
  p.stride_be = B.stride(0); p.stride_bn = B.stride(1);
  p.stride_bse = B_scale.stride(0); p.stride_bsn = B_scale.stride(1); p.stride_bsk = B_scale.stride(2);
  p.ldc = ldc;
}

// Naive-path down GEMM with the activation fused in:
//   C[pair, :] = quant32(silu(X[pair, :K]) * X[pair, K:]) x B[expert_ids[pair]]^T * topk_weights[pair]
// X bf16 [pairs, 2K] (the first GEMM's output), one block row per pair (grid.y = pairs).
void fused_moe_gemv_act(torch::Tensor X, torch::Tensor B, torch::Tensor B_scale, torch::Tensor C,
                        c10::optional<torch::Tensor> topk_weights, torch::Tensor expert_ids,
                        torch::Tensor num_tokens_post_padded, int64_t num_valid_tokens,
                        int64_t m_blocks, int64_t npp_block, int64_t ldc, int64_t ksplit,
                        int64_t unroll, bool scale_ue8m0) {
  TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16 && X.dim() == 2 && X.stride(1) == 1,
              "X must be bf16 [pairs, 2K] with unit stride");
  const int K = B.size(2);
  TORCH_CHECK(X.size(1) == 2 * K, "X must have 2K columns (gate | up)");
  TORCH_CHECK(X.size(0) >= num_valid_tokens, "X rows");
  check_common(B, B_scale, C, topk_weights, expert_ids, num_tokens_post_padded, num_valid_tokens,
               m_blocks, K);
  if (m_blocks <= 0) return;
  MoeGemvArgs p{};
  fill_b(p, B, B_scale, C, topk_weights, expert_ids, num_tokens_post_padded, num_valid_tokens, 1,
         npp_block, ldc, K);
  p.X = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
  p.ldx = X.stride(0);
  p.scale_ue8m0 = scale_ue8m0 ? 1 : 0;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(X.get_device()).stream();
  launch(p, (int)m_blocks, MODE_FUSED_ACT, (int)ksplit, (int)unroll, stream);
}

// C[pair, :] = (A[pair / top_k, :] x B[expert_ids[block], :, :]^T) * topk_weights[pair]
// for the pairs of every m-block (see file header). `m_blocks` = grid.y; `npp_block` =
// the Triton BLOCK_SIZE_M the num_tokens_post_padded counter is expressed in.
void fused_moe_gemv(torch::Tensor A, torch::Tensor A_scale, torch::Tensor B, torch::Tensor B_scale,
                    torch::Tensor C, c10::optional<torch::Tensor> topk_weights,
                    c10::optional<torch::Tensor> sorted_token_ids, torch::Tensor expert_ids,
                    torch::Tensor num_tokens_post_padded, int64_t num_valid_tokens, int64_t top_k,
                    int64_t m_blocks, int64_t npp_block, int64_t ldc, int64_t ksplit,
                    int64_t unroll) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kFloat8_e4m3fn, "A must be e4m3");
  TORCH_CHECK(A.dim() == 2 && A.stride(1) == 1, "A must be [*, K] with unit K stride");
  TORCH_CHECK(A_scale.scalar_type() == torch::kFloat32 && A_scale.dim() == 2, "a_scale fp32 [*, K/32]");
  const int K = A.size(1);
  check_common(B, B_scale, C, topk_weights, expert_ids, num_tokens_post_padded, num_valid_tokens,
               m_blocks, K);
  TORCH_CHECK(A_scale.size(0) >= A.size(0) && A_scale.size(1) == K / kBlockK, "a_scale shape");
  TORCH_CHECK(A.stride(0) % 16 == 0, "16-byte aligned A rows");
  const bool aligned = sorted_token_ids.has_value();
  if (aligned) {
    TORCH_CHECK(sorted_token_ids->scalar_type() == torch::kInt32 && sorted_token_ids->is_contiguous(),
                "sorted_token_ids int32 contiguous");
  }
  if (m_blocks <= 0) return;

  MoeGemvArgs p{};
  fill_b(p, B, B_scale, C, topk_weights, expert_ids, num_tokens_post_padded, num_valid_tokens, top_k,
         npp_block, ldc, K);
  p.A = reinterpret_cast<const uint8_t*>(A.data_ptr());
  p.A_sf = A_scale.data_ptr<float>();
  p.sorted_ids = aligned ? sorted_token_ids->data_ptr<int32_t>() : nullptr;
  p.em = aligned ? (int)sorted_token_ids->numel() : 0;
  p.stride_am = A.stride(0);
  p.stride_asm = A_scale.stride(0); p.stride_ask = A_scale.stride(1);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.get_device()).stream();
  launch(p, (int)m_blocks, aligned ? MODE_ALIGNED : MODE_NAIVE, (int)ksplit, (int)unroll, stream);
}

}  // namespace vllm_moet_sm120

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_moe_gemv", &vllm_moet_sm120::fused_moe_gemv,
        "FP8 block-scaled ([32,32]) MoE GEMV for vLLM's fused_moe (sm_120)",
        py::arg("A"), py::arg("A_scale"), py::arg("B"), py::arg("B_scale"), py::arg("C"),
        py::arg("topk_weights"), py::arg("sorted_token_ids"), py::arg("expert_ids"),
        py::arg("num_tokens_post_padded"), py::arg("num_valid_tokens"), py::arg("top_k"),
        py::arg("m_blocks"), py::arg("npp_block"), py::arg("ldc"), py::arg("ksplit"),
        py::arg("unroll"));
  m.def("fused_moe_gemv_act", &vllm_moet_sm120::fused_moe_gemv_act,
        "Down GEMM with silu(gate)*up + per-32-group fp8 quantization fused (naive path)",
        py::arg("X"), py::arg("B"), py::arg("B_scale"), py::arg("C"), py::arg("topk_weights"),
        py::arg("expert_ids"), py::arg("num_tokens_post_padded"), py::arg("num_valid_tokens"),
        py::arg("m_blocks"), py::arg("npp_block"), py::arg("ldc"), py::arg("ksplit"),
        py::arg("unroll"), py::arg("scale_ue8m0"));
}
