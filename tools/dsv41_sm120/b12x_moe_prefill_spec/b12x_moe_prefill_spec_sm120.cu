// Spec B of the prefill MoE (front "cubit + prefill", Phase 1): the arithmetic of b12x's `integration.tp_moe.dynamic`
// path for DeepSeek-V4.1-Flash on sm_120 (TP4 rank: 384 experts, hidden 5120, intermediate 576, top-6, SwiGLU limit 10,
// W4A8: FP4 e2m1 weights + UE8M0/32 scales, MXFP8 e4m3 activations + UE8M0/32 scales, B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1)
// as four plain CUDA kernels with the same instructions, operands, K order and epilogues per output element - a readable,
// executable specification the SASS kernel must match bit for bit, not a fast kernel.
//
//   quant_rows   hidden bf16 [T, K] -> a_values e4m3 [T, K] + a_scales ue8m0 [T, K/32]    (dynamic front-end: _load_bf16x32_to_f32
//                + quantize_block_fp8_mx: amax over the 32 values, scale = pow2_ceil(amax / 448) as UE8M0, payload =
//                cvt.rn.satfinite.e4m3x2(value * 2^(127 - byte)); an all-zero block -> byte 0, payload 0)
//   fc1_act      per routed pair (token, slot) and 128-column tile: gate and up accumulated with the block-scaled QMMA
//                (mma.sync m16n8k32 kind::mxf8f6f4 e4m3 x e2m1, ue8m0 scales, fp32 accumulate IN C) over K in order, k32
//                blocks 0..K/32-1 (W4A8MaterializedPhase1Kernel: two k32 per k64 stage, bid = k32 & 3 of the k128 scale
//                word); alpha[e] * input_global_scale[e] on the fp32 accumulators, SwiGLU with limit (min.NaN / max.NaN),
//                sigmoid = rcp.approx(1 + ex2.approx(gate * -log2e)), gate * sigmoid * up, cvt.rn.satfinite.bf16x2 into
//                a bf16 tile, then per row and 32-column block: bf16 -> fp32, amax by compare-and-select,
//                quantize_block_fp8_mx -> the MXFP8 intermediate (rows of 640 B: 576 columns + a zero 64-column tail,
//                and 5 scale words per row = the k128 slices, the tail word's bytes 2-3 zero)
//   fc2          per pair and 128 output columns: the QMMA chain over the 5 k128 slices x 4 k32 (20 MMAs, the tail
//                slice's k32 2-3 with zero A, zero B and zero scale bytes, as W4A8MaterializedPhase2Kernel), then
//                scale = down_alpha[e] * global_scale[e] * token_weight[pair], cvt.rn.satfinite.bf16x2(scale * acc)
//                into route_rows [T * topk, K] bf16 (token-major pair rows; deterministic_output)
//   topk_sum     out[t, col] = bf16( ((0 + r0) + r1) + ... ) over the pairs of token t in slot order, the routes with
//                id < 0 skipped (W4A16TopKSumKernel, plain path: fp32 adds of the bf16 route values, bf16 at the end)
//
// Operand layouts are b12x's prepared n64-repacked planes exactly as b12x_moe_fused_sm120.cu reads them:
//   w13_rp  u32 [E][2 (up, gate)][5 tiles of 128 cols (the 5th: 64)][K/32][n32 chunk][lane][4 words = n8 tiles nt]
//           (a word holds the lane's 8 consecutive k nibbles of column chunk*32 + nt*8 + lane/4, spread into the two
//           e2m1 byte-container registers of the MMA by spread_e2m1x8)
//   w13_sfb u32 [E][2][tile][K/128][n_cols] (one word per column per k128 slice: the 4 k32 scale bytes)
//   w2_rp   u32 [E][40 tiles of 128 cols][18 k32][4 n32 chunks][lane][4 words]
//   w2_sfb  u8  [E][40 tiles][5 slices][128 cols x 4 B (slices 0-3) | 128 cols x 2 B (tail)]
// The pairs are presented sorted by expert and padded to 16 rows per expert (host side: row_pair[rows_padded] with -1
// padding, tile_expert[rows_padded / 16]); the A fragments of an m16 tile are rows g (lanes with lane & 1 == 0 supply
// the row's scale word) and g + 8.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace b12x_moe_prefill_spec {

constexpr int kThreads = 128;
constexpr int kTileN = 128;
constexpr int kSlices = 5;              // 576 = 4.5 K128 slices
constexpr int kTotalK32 = 18;           // live k32 blocks of the intermediate
constexpr int kInterBytes = kSlices * 128;  // 640 B per row (576 + zero tail)

__device__ __forceinline__ void spread_e2m1x8(uint32_t packed, uint32_t& lo, uint32_t& hi) {
  asm("{\n"
      ".reg .b32 s, t;\n"
      "shl.b32 s, %2, 2;\n"
      "shr.u32 t, %2, 2;\n"
      "prmt.b32 %0, s, t, 0x5140;\n"
      "prmt.b32 %1, s, t, 0x7362;\n"
      "and.b32 %0, %0, 0x3C3C3C3C;\n"
      "and.b32 %1, %1, 0x3C3C3C3C;\n"
      "}\n"
      : "=r"(lo), "=r"(hi)
      : "r"(packed));
}

// the block-scaled MMA with the accumulator in C (phase 1 and phase 2 of b12x's dynamic path chain the accumulator)
template <int KB>
__device__ __forceinline__ void qmma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %13}, {%11}, {%12, %13};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(KB), "n"(0));
}
__device__ __forceinline__ void qmma_kb(int kb, float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  if (kb == 0) qmma<0>(d, a0, a1, a2, a3, b0, b1, sfa, sfb);
  else if (kb == 1) qmma<1>(d, a0, a1, a2, a3, b0, b1, sfa, sfb);
  else if (kb == 2) qmma<2>(d, a0, a1, a2, a3, b0, b1, sfa, sfb);
  else qmma<3>(d, a0, a1, a2, a3, b0, b1, sfa, sfb);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {  // pack_f32x2_to_bfloat2
  uint32_t r;
  asm("cvt.rn.satfinite.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}
__device__ __forceinline__ float bf16_bits_to_f32(uint32_t h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ float fmul(float a, float b) {
  float r;
  asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fadd(float a, float b) {
  float r;
  asm("add.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fmin_nan(float a, float b) {
  float r;
  asm("min.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fmax_nan(float a, float b) {
  float r;
  asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fmax_plain(float a, float b) {
  float r;
  asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fabs_plain(float a) {
  float r;
  asm("abs.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float ex2_approx(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// cute.arch.rcp_approx / cute.math.exp(fastmath=False) in phase 1 lower to the .ftz MUFU forms (no denormal fix-up
// sequence around the 16 activation exps / rcps in the kernel's SASS; the UE8M0 inverse scale keeps the non-ftz ex2 of
// ue8m0_to_output_scale). Both give the same bits here (ex2 of a denormal input and 1 + a denormal are exact anyway).
__device__ __forceinline__ float rcp_approx_ftz(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float ex2_approx_ftz(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// W4A8MaterializedPhase1Kernel._activated_value (silu, swiglu_limit set): fp32 in, fp32 out
__device__ __forceinline__ float activated_value(float gate, float up, float alpha, float limit) {
  gate = fmul(alpha, gate);
  up = fmul(alpha, up);
  gate = fmin_nan(gate, limit);
  up = fmax_nan(fmin_nan(up, limit), -limit);
  const float e = ex2_approx_ftz(fmul(gate, __uint_as_float(0xBFB8AA3Bu)));  // exp(-gate) = 2^(gate * -log2 e)
  const float sigmoid = rcp_approx_ftz(fadd(1.0f, e));
  return fmul(fmul(gate, sigmoid), up);
}

// quantize_block_fp8_mx: 32 fp32 values + their amax -> 8 e4m3x4 words and the UE8M0 scale byte
__device__ __forceinline__ uint32_t quant32(const float (&v)[32], float amax, uint32_t (&payload)[8]) {
  const float sc = fmul(amax, __uint_as_float(0x3B124925u));  // amax * (1 / 448)
  uint32_t bits = __float_as_uint(sc);
  if (bits & 0x007FFFFFu) bits = (bits + 0x00800000u) & 0x7F800000u;  // pow2_ceil_ue8m0
  const uint32_t byte = (bits >> 23) & 0xFFu;
  float inv;  // ue8m0_to_output_scale: 2^(127 - byte), 0 for byte 0
  {
    const float ef = (float)(127 - (int)byte);
    inv = ex2_approx(ef);
    if (byte == 0) inv = 0.f;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    unsigned short lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(fmul(v[4 * i + 0], inv)), "f"(fmul(v[4 * i + 1], inv)));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(fmul(v[4 * i + 2], inv)), "f"(fmul(v[4 * i + 3], inv)));
    payload[i] = (uint32_t)lo | ((uint32_t)hi << 16);
  }
  return byte;
}

// ------------------------------------------------------------------------------------------- 1. input rows quant
// one warp per token row; lane handles the 32-column blocks lane, lane + 32, ... (_load_bf16x32_to_f32: amax by max.f32
// of abs over the values in order)
__global__ void quant_rows_kernel(const uint32_t* __restrict__ hidden, uint32_t* __restrict__ a_values,
                                  uint8_t* __restrict__ a_scales, int T, int K) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= T) return;
  const int blocks = K / 32;
  const uint32_t* src = hidden + (size_t)row * (K / 2);
  for (int b = lane; b < blocks; b += 32) {
    float v[32];
    float amax = 0.f;
#pragma unroll
    for (int vec = 0; vec < 4; ++vec) {
      const uint4 w = __ldg(reinterpret_cast<const uint4*>(src + b * 16 + vec * 4));
      const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        v[vec * 8 + 2 * j] = fmul(bf16_bits_to_f32(ws[j] & 0xFFFFu), 1.0f);  // bfloat2_to_float2_scaled(., 1.0)
        v[vec * 8 + 2 * j + 1] = fmul(bf16_bits_to_f32(ws[j] >> 16), 1.0f);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) amax = fmax_plain(amax, fabs_plain(v[vec * 8 + j]));
    }
    uint32_t payload[8];
    const uint32_t byte = quant32(v, amax, payload);
    uint4* dst = reinterpret_cast<uint4*>(a_values + (size_t)row * (K / 4) + b * 8);
    dst[0] = make_uint4(payload[0], payload[1], payload[2], payload[3]);
    dst[1] = make_uint4(payload[4], payload[5], payload[6], payload[7]);
    a_scales[(size_t)row * blocks + b] = (uint8_t)byte;
  }
}

// ---------------------------------------------------------------------------------------- 2. FC1 + activation
struct Fc1Args {
  const uint32_t* a_values;  // [T][K/4]
  const uint8_t* a_scales;   // [T][K/32]
  const uint32_t* w13;       // n64-repacked
  const uint32_t* w13_sf;
  const int32_t* row_pair;   // [rows_padded] pair index or -1
  const int32_t* tile_expert;  // [rows_padded / 16]
  const float* alpha;        // [E]
  const float* input_global_scale;  // [E] or [1]
  uint32_t* inter;           // [pairs][640 B]
  uint32_t* inter_sf;        // [pairs][5 words]
  int K, N, n_tiles, topk, igs_per_expert;
  float limit;
};

__global__ void __launch_bounds__(kThreads) fc1_act_kernel(Fc1Args a) {
  __shared__ __align__(16) uint32_t s_tile[16 * kTileN / 2];  // bf16 [16 rows][128 cols]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int q = lane >> 2, c = lane & 3;
  const int mtile = blockIdx.x, otile = blockIdx.y;
  const int e = a.tile_expert[mtile];
  const int K = a.K, N = a.N;
  const int n_cols = otile == a.n_tiles - 1 ? (N - otile * kTileN) : kTileN;
  const int n32_chunks = n_cols / 32;
  const int row_lo = mtile * 16 + q, row_hi = row_lo + 8;
  const int pair_lo = a.row_pair[row_lo], pair_hi = a.row_pair[row_hi];
  const int tok_lo = pair_lo >= 0 ? pair_lo / a.topk : 0, tok_hi = pair_hi >= 0 ? pair_hi / a.topk : 0;
  // this lane's scale rows: lanes with lane & 1 supply row g + 8 (b12x: sf_row = q + ((lane & 1) << 3))
  const int sf_tok = (lane & 1) ? tok_hi : tok_lo;
  const int sf_pair = (lane & 1) ? pair_hi : pair_lo;

  float gacc[4][4], uacc[4][4];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) gacc[nt][i] = uacc[nt][i] = 0.f;

  if (e >= 0 && warp < n32_chunks) {
    const size_t wproj = (size_t)N * (K / 32) * 4;  // words per projection
    const size_t wexp = (size_t)e * 2 * wproj + (size_t)otile * kTileN * (K / 32) * 4;
    const uint32_t* w_up = a.w13 + wexp;              // projection 0 = up ("w31": the prepared layout holds up first)
    const uint32_t* w_gate = a.w13 + wexp + wproj;    // projection 1 = gate
    const size_t sproj = (size_t)N * (K / 32) / 4;    // words per projection of scales
    const size_t sexp = (size_t)e * 2 * sproj + (size_t)otile * kTileN * (K / 32) / 4;
    const uint32_t* s_up = a.w13_sf + sexp;
    const uint32_t* s_gate = a.w13_sf + sexp + sproj;
    const int tile_words_per_k32 = n32_chunks * 128;
    const uint32_t* arow_lo = a.a_values + (size_t)tok_lo * (K / 4);
    const uint32_t* arow_hi = a.a_values + (size_t)tok_hi * (K / 4);
    const uint32_t* asc_row = reinterpret_cast<const uint32_t*>(a.a_scales + (size_t)sf_tok * (K / 32));
    const int k32_blocks = K / 32;
    for (int s = 0; s < k32_blocks / 4; ++s) {  // k128 slice: one scale word per row / column
      const uint32_t asc = sf_pair >= 0 ? __ldg(asc_row + s) : 0u;
      uint32_t sfb_up[4], sfb_gate[4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col = warp * 32 + nt * 8 + q;
        sfb_up[nt] = __ldg(s_up + (size_t)s * n_cols + col);
        sfb_gate[nt] = __ldg(s_gate + (size_t)s * n_cols + col);
      }
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        const int k32 = s * 4 + kb;
        // A: 8 consecutive bytes of the k32 block per lane (k 8c..8c+7), rows g and g+8
        const uint2 alo = pair_lo >= 0 ? __ldg(reinterpret_cast<const uint2*>(arow_lo + k32 * 8 + c * 2)) : make_uint2(0u, 0u);
        const uint2 ahi = pair_hi >= 0 ? __ldg(reinterpret_cast<const uint2*>(arow_hi + k32 * 8 + c * 2)) : make_uint2(0u, 0u);
        const uint4 wu = __ldg(reinterpret_cast<const uint4*>(w_up + (size_t)k32 * tile_words_per_k32 + warp * 128 + lane * 4));
        const uint4 wg = __ldg(reinterpret_cast<const uint4*>(w_gate + (size_t)k32 * tile_words_per_k32 + warp * 128 + lane * 4));
        const uint32_t wus[4] = {wu.x, wu.y, wu.z, wu.w}, wgs[4] = {wg.x, wg.y, wg.z, wg.w};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          uint32_t b0, b1;
          spread_e2m1x8(wgs[nt], b0, b1);
          qmma_kb(kb, gacc[nt], alo.x, ahi.x, alo.y, ahi.y, b0, b1, asc, sfb_gate[nt]);
          spread_e2m1x8(wus[nt], b0, b1);
          qmma_kb(kb, uacc[nt], alo.x, ahi.x, alo.y, ahi.y, b0, b1, asc, sfb_up[nt]);
        }
      }
    }
  }

  // epilogue: activation on the fp32 accumulators, bf16 into the tile (columns beyond n_cols: zeros from zero acc)
  const float alpha = e >= 0 ? __ldg(a.alpha + e) * __ldg(a.input_global_scale + (a.igs_per_expert ? e : 0)) : 0.f;
  {
    const int col_base = warp * 32 + (c << 1);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int col = col_base + nt * 8;
      const float v0 = activated_value(gacc[nt][0], uacc[nt][0], alpha, a.limit);
      const float v1 = activated_value(gacc[nt][1], uacc[nt][1], alpha, a.limit);
      const float v2 = activated_value(gacc[nt][2], uacc[nt][2], alpha, a.limit);
      const float v3 = activated_value(gacc[nt][3], uacc[nt][3], alpha, a.limit);
      s_tile[(q * kTileN + col) / 2] = pack_bf16x2(v0, v1);
      s_tile[((q + 8) * kTileN + col) / 2] = pack_bf16x2(v2, v3);
    }
  }
  __syncthreads();
  // per row (threads 0..15): 4 blocks of 32 columns -> MXFP8 payload + scale byte
  if (tid < 16) {
    const int pair = a.row_pair[mtile * 16 + tid];
    if (pair >= 0 && e >= 0) {
      uint32_t scale_word = 0u;
      uint32_t* drow = a.inter + (size_t)pair * (kInterBytes / 4);
#pragma unroll 1
      for (int block = 0; block < 4; ++block) {
        float v[32];
        float block_max = 0.f;
#pragma unroll
        for (int el = 0; el < 32; ++el) {
          const uint32_t w = s_tile[(tid * kTileN + block * 32 + el) / 2];
          const float val = bf16_bits_to_f32((el & 1) ? (w >> 16) : (w & 0xFFFFu));
          v[el] = val;
          const float av = fabs_plain(val);
          if (av > block_max) block_max = av;
        }
        uint32_t payload[8];
        const uint32_t byte = quant32(v, block_max, payload);
        uint32_t* dst = drow + otile * (kTileN / 4) + block * 8;
#pragma unroll
        for (int w = 0; w < 8; ++w) dst[w] = payload[w];
        scale_word |= (byte & 0xFFu) << (block * 8);
      }
      a.inter_sf[(size_t)pair * kSlices + otile] = scale_word;
    }
  }
}

// ---------------------------------------------------------------------------------------------------- 3. FC2
struct Fc2Args {
  const uint32_t* inter;     // [pairs][640 B]
  const uint32_t* inter_sf;  // [pairs][5]
  const uint32_t* w2;        // n64-repacked
  const uint8_t* w2_sf;
  const int32_t* row_pair;
  const int32_t* tile_expert;
  const float* token_weights;  // [pairs]
  const float* down_alpha;     // [E]
  const float* global_scale;   // [E] or [1]
  uint32_t* route;             // [pairs][K] bf16
  int K, out_tiles, gs_per_expert;
};

__global__ void __launch_bounds__(kThreads) fc2_kernel(Fc2Args a) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int q = lane >> 2, c = lane & 3;
  const int mtile = blockIdx.x, tile = blockIdx.y;
  const int e = a.tile_expert[mtile];
  const int row_lo = mtile * 16 + q, row_hi = row_lo + 8;
  const int pair_lo = a.row_pair[row_lo], pair_hi = a.row_pair[row_hi];
  if (e < 0) return;
  const int sf_pair = (lane & 1) ? pair_hi : pair_lo;
  const uint32_t* alo = a.inter + (size_t)(pair_lo >= 0 ? pair_lo : 0) * (kInterBytes / 4);
  const uint32_t* ahi = a.inter + (size_t)(pair_hi >= 0 ? pair_hi : 0) * (kInterBytes / 4);
  const uint32_t* asc_row = a.inter_sf + (size_t)(sf_pair >= 0 ? sf_pair : 0) * kSlices;
  const size_t expert_words = (size_t)a.out_tiles * kTileN * kTotalK32 * 4;
  const uint32_t* wtile = a.w2 + (size_t)e * expert_words + (size_t)tile * (kTileN * kTotalK32 * 4);
  const uint8_t* sftile = a.w2_sf + ((size_t)e * a.out_tiles + tile) * kTileN * kTotalK32;
  float acc[4][4];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[nt][i] = 0.f;

  for (int s = 0; s < kSlices; ++s) {
    const bool tail = s == kSlices - 1;
    const uint32_t asc = sf_pair >= 0 ? asc_row[s] : 0u;
    uint32_t sfb[4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int col = warp * 32 + nt * 8 + q;
      if (tail) sfb[nt] = *reinterpret_cast<const uint16_t*>(sftile + 4 * 512 + col * 2);
      else sfb[nt] = *reinterpret_cast<const uint32_t*>(sftile + s * 512 + col * 4);
    }
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      const int k32 = s * 4 + kb;
      const bool live = !tail || kb < 2;
      uint2 flo = make_uint2(0u, 0u), fhi = make_uint2(0u, 0u);
      uint4 w = make_uint4(0u, 0u, 0u, 0u);
      if (live) {
        if (pair_lo >= 0) flo = *reinterpret_cast<const uint2*>(alo + s * 32 + kb * 8 + c * 2);
        if (pair_hi >= 0) fhi = *reinterpret_cast<const uint2*>(ahi + s * 32 + kb * 8 + c * 2);
        w = __ldg(reinterpret_cast<const uint4*>(wtile + (size_t)k32 * 512 + warp * 128 + lane * 4));
      }
      const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        uint32_t b0, b1;
        spread_e2m1x8(ws[nt], b0, b1);
        qmma_kb(kb, acc[nt], flo.x, fhi.x, flo.y, fhi.y, b0, b1, asc, sfb[nt]);
      }
    }
  }
  const float down_scale = __ldg(a.down_alpha + e) * __ldg(a.global_scale + (a.gs_per_expert ? e : 0));
  const int col_base = tile * kTileN + warp * 32 + (c << 1);
  if (pair_lo >= 0) {
    const float scale = down_scale * __ldg(a.token_weights + pair_lo);
    uint32_t* dst = a.route + (size_t)pair_lo * (a.K / 2);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) dst[(col_base + nt * 8) / 2] = pack_bf16x2(scale * acc[nt][0], scale * acc[nt][1]);
  }
  if (pair_hi >= 0) {
    const float scale = down_scale * __ldg(a.token_weights + pair_hi);
    uint32_t* dst = a.route + (size_t)pair_hi * (a.K / 2);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) dst[(col_base + nt * 8) / 2] = pack_bf16x2(scale * acc[nt][2], scale * acc[nt][3]);
  }
}

// ------------------------------------------------------------------------------------------------ 4. top-k sum
__global__ void topk_sum_kernel(const uint16_t* __restrict__ route, const int32_t* __restrict__ ids,
                                uint16_t* __restrict__ out, int T, int topk, int K) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)T * K) return;
  const int t = (int)(idx / K), col = (int)(idx - (long long)t * K);
  float acc = 0.f;
  for (int r = 0; r < topk; ++r) {
    const int row = t * topk + r;
    if (ids[row] < 0) continue;
    acc = fadd(acc, bf16_bits_to_f32(route[(size_t)row * K + col]));
  }
  unsigned short h;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(acc));
  out[idx] = h;
}

}  // namespace b12x_moe_prefill_spec

// hidden [T, K] bf16, ids [T, topk] int32, weights [T, topk] fp32 (row-major, contiguous)
std::vector<torch::Tensor> moe_prefill_spec(torch::Tensor hidden, torch::Tensor ids, torch::Tensor token_weights,
                                            torch::Tensor row_pair, torch::Tensor tile_expert, torch::Tensor w13,
                                            torch::Tensor w13_sf, torch::Tensor w2, torch::Tensor w2_sf,
                                            torch::Tensor alpha, torch::Tensor input_global_scale,
                                            torch::Tensor down_alpha, torch::Tensor global_scale, double limit,
                                            int64_t n) {
  using namespace b12x_moe_prefill_spec;
  TORCH_CHECK(hidden.scalar_type() == torch::kBFloat16 && hidden.dim() == 2 && hidden.is_contiguous());
  TORCH_CHECK(ids.scalar_type() == torch::kInt32 && token_weights.scalar_type() == torch::kFloat32);
  TORCH_CHECK(row_pair.scalar_type() == torch::kInt32 && tile_expert.scalar_type() == torch::kInt32);
  TORCH_CHECK(n == 576, "intermediate 576 (five K128 slices, n64 tail)");
  const int T = (int)hidden.size(0), K = (int)hidden.size(1), topk = (int)ids.size(1);
  TORCH_CHECK(K % 128 == 0 && ids.size(0) == T && token_weights.sizes() == ids.sizes());
  const int pairs = T * topk, rows_padded = (int)row_pair.numel();
  TORCH_CHECK(rows_padded % 16 == 0 && tile_expert.numel() == rows_padded / 16);
  const int E = (int)alpha.numel();
  TORCH_CHECK(w2.numel() * w2.element_size() == (int64_t)E * K * n / 2, "w2 is not the n64-repacked storage");
  TORCH_CHECK(w13.numel() * w13.element_size() == (int64_t)E * 2 * n * K / 2, "w13 is not the n64-repacked storage");
  auto dev = hidden.device();
  auto o32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  auto o8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto a_values = torch::empty({T, K}, o8);
  auto a_scales = torch::empty({T, K / 32}, o8);
  auto inter = torch::zeros({pairs, kInterBytes}, o8);
  auto inter_sf = torch::zeros({pairs, kSlices}, o32);
  auto route = torch::zeros({pairs, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
  auto out = torch::empty({T, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
  auto stream = c10::cuda::getCurrentCUDAStream();

  quant_rows_kernel<<<(T + 3) / 4, 128, 0, stream>>>(reinterpret_cast<const uint32_t*>(hidden.data_ptr()),
                                                     reinterpret_cast<uint32_t*>(a_values.data_ptr()),
                                                     a_scales.data_ptr<uint8_t>(), T, K);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);

  Fc1Args f1;
  f1.a_values = reinterpret_cast<const uint32_t*>(a_values.data_ptr());
  f1.a_scales = a_scales.data_ptr<uint8_t>();
  f1.w13 = reinterpret_cast<const uint32_t*>(w13.data_ptr());
  f1.w13_sf = reinterpret_cast<const uint32_t*>(w13_sf.data_ptr());
  f1.row_pair = row_pair.data_ptr<int32_t>();
  f1.tile_expert = tile_expert.data_ptr<int32_t>();
  f1.alpha = alpha.data_ptr<float>();
  f1.input_global_scale = input_global_scale.data_ptr<float>();
  f1.inter = reinterpret_cast<uint32_t*>(inter.data_ptr());
  f1.inter_sf = reinterpret_cast<uint32_t*>(inter_sf.data_ptr());
  f1.K = K;
  f1.N = (int)n;
  f1.n_tiles = (int)((n + kTileN - 1) / kTileN);
  f1.topk = topk;
  f1.igs_per_expert = input_global_scale.numel() > 1;
  f1.limit = (float)limit;
  fc1_act_kernel<<<dim3(rows_padded / 16, f1.n_tiles), kThreads, 0, stream>>>(f1);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);

  Fc2Args f2;
  f2.inter = reinterpret_cast<const uint32_t*>(inter.data_ptr());
  f2.inter_sf = reinterpret_cast<const uint32_t*>(inter_sf.data_ptr());
  f2.w2 = reinterpret_cast<const uint32_t*>(w2.data_ptr());
  f2.w2_sf = reinterpret_cast<const uint8_t*>(w2_sf.data_ptr());
  f2.row_pair = row_pair.data_ptr<int32_t>();
  f2.tile_expert = tile_expert.data_ptr<int32_t>();
  f2.token_weights = token_weights.data_ptr<float>();
  f2.down_alpha = down_alpha.data_ptr<float>();
  f2.global_scale = global_scale.data_ptr<float>();
  f2.route = reinterpret_cast<uint32_t*>(route.data_ptr());
  f2.K = K;
  f2.out_tiles = K / kTileN;
  f2.gs_per_expert = global_scale.numel() > 1;
  fc2_kernel<<<dim3(rows_padded / 16, K / kTileN), kThreads, 0, stream>>>(f2);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);

  const long long total = (long long)T * K;
  topk_sum_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(route.data_ptr()), ids.data_ptr<int32_t>(),
      reinterpret_cast<uint16_t*>(out.data_ptr()), T, topk, K);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
  return {out, route, a_values, a_scales, inter, inter_sf};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("moe_prefill_spec", &moe_prefill_spec); }
