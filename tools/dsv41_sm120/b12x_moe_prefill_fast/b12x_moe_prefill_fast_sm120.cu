// Fast prefill MoE for DeepSeek-V4.1-Flash on sm_120 (front "cubit + prefill", Phase 1a): FC1 + SwiGLU + MXFP8 quant and
// FC2 of b12x's `integration.tp_moe.dynamic` deterministic path, bit for bit with spec B
// (../b12x_moe_prefill_spec/b12x_moe_prefill_spec_sm120.cu = b12x's arithmetic as naive kernels), but pipelined:
//
//   frontend   one launch: MXFP8 quant of the token rows (a warp per row; exactly spec B's quant_rows) in all blocks but
//              the last, and in the last block the deterministic routing: per-expert pair counts (shared-memory integer
//              atomics: the sums do not depend on order), padded (to 16 rows) expert offsets, a stable scatter of the
//              pairs into row_pair (pair order within an expert; ranks from __match_any + a per-warp prefix: no global
//              atomics, no ordering race) and the CTA work lists of FC1 / FC2 ((expert, 64-row chunk, column tile),
//              tile-major within an expert so that consecutive items share the expert's weight tile in L2).
//   fc1        persistent CTAs (256 threads = 8 warps), work item = (expert, <= 4 m16 tiles of its rows, 128-column
//              tile of gate+up; the 5th tile: 64 columns); warps = 4 n32 slices x 2 row groups (2 m-tiles each, 64 fp32
//              accumulators per thread). K in K64 stages through a cp.async ring (STAGES deep) holding the k32 blocks
//              of B (b12x's n64-repacked words as laid out in the plane: one LDS.128 per lane per k32 and projection),
//              the A rows (XOR-swizzled 16 B chunks, LDS.64 per row and k32) and the k128 scale words of both; the
//              stage stream is continuous across items (the next item's first stages are in flight during the
//              epilogue). Per output element the QMMA chain is spec B's: accumulate in C over the k32 blocks in order,
//              from +0. Epilogue in registers: alpha * input_global_scale, SwiGLU with limit (min.NaN / max.NaN), ex2 /
//              rcp approx, cvt.rn.satfinite.bf16x2, then per row and 32-column block (= the warp's n32 slice) the amax
//              over the quad (max is commutative: the same value as the sequential compare-and-select),
//              quantize_block_fp8_mx per element, a 4x4 transpose of the e4m3 pairs across the quad, one 8 B store per
//              lane and row, the scale byte.
//   fc2        the same skeleton with K = 640 (18 live k32 blocks + the zero tail: 20 QMMAs per n8 tile as b12x),
//              item = (expert, <= 4 m-tiles, 128 output columns); epilogue: (down_alpha * global_scale) * token weight,
//              cvt.rn.satfinite.bf16x2 into the token-major route rows (a 4x4 word transpose across the quad: 16 B
//              stores).
//   topk_sum   out[t, col] = bf16((((0 + r0) + r1) + ...)) over the token's pairs in slot order (ids < 0 skipped), fp32
//              adds of the bf16 route values - spec B's topk_sum, 8 columns per thread (integrated, b12x's own TopKSum
//              runs instead: the route rows are the interface).
//
// Determinism: no atomics in any reduction (the only atomics are the integer pair counts in shared memory, whose sums
// are order-independent, and the work-item counters, which only decide which CTA computes an item). Three runs give
// identical bits. Plane layouts and operand conventions are the spec's (see its header).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace b12x_moe_prefill_fast {

constexpr int kThreads = 256;
constexpr int kN = 576;             // intermediate size: 4 x 128 + 64 columns
constexpr int kN1Tiles = 5;         // FC1 column tiles (gate+up each)
constexpr int kSlices = 5;          // k128 slices of the intermediate (4.5)
constexpr int kInterBytes = 640;    // MXFP8 intermediate row (576 + zero tail)
constexpr int kF2Stages = 10;       // K64 stages of FC2 (9 live + the zero tail)
constexpr int kMaxE = 512;          // experts per rank (384 served)
constexpr int kFrontThreads = 512;
constexpr int kFrontWarps = kFrontThreads / 32;

// stage layouts (bytes)
constexpr int kF1B = 8192;    // [proj 2][kk 2][n32 chunk 4][lane 32][16 B]
constexpr int kF1A = 4096;    // [64 rows][64 B] (16 B chunks XOR-swizzled by bit 1 of the row)
constexpr int kF1SFA = 256;   // [64 rows] k128 scale word
constexpr int kF1SFB = 1024;  // [proj 2][128 cols] k128 scale word
constexpr int kF1Stage = kF1B + kF1A + kF1SFA + kF1SFB;  // 13568
constexpr int kF2B = 4096;    // [kk 2][n32 chunk 4][lane 32][16 B]
constexpr int kF2A = 4096;
constexpr int kF2SFA = 256;
constexpr int kF2SFB = 512;   // [128 cols] k128 scale word (the tail slice: 128 x u16)
constexpr int kF2Stage = kF2B + kF2A + kF2SFA + kF2SFB;  // 8960

// ------------------------------------------------------------------------------------------------------ helpers
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async16_zfill(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async4_zfill(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// b12x's e2m1x8_to_qmma_e2m1x8: eight FP4 nibbles into E2M1 byte containers (nibble in bits 5:2, bits 7:6 / 1:0 zero).
// The masks are required: the QMMA reads the container padding bits (a variant without the two ands produced different
// intermediate bits on every row in the 2.11 window, `S/out/fix-allcfgs.log`), as the PTX ISA's "padding" wording implies.
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

// the block-scaled MMA with the accumulator in C (the chain of b12x's phase 1 / phase 2)
template <int KB>
__device__ __forceinline__ void qmma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %13}, {%11}, {%12, %13};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(KB), "n"(0));
}

// kk (0 / 1, a constant after unrolling) selects the scale byte KB0 + kk
template <int KB0>
__device__ __forceinline__ void qmma_kk(int kk, float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  if (kk == 0) qmma<KB0>(d, a0, a1, a2, a3, b0, b1, sfa, sfb);
  else qmma<KB0 + 1>(d, a0, a1, a2, a3, b0, b1, sfa, sfb);
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
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ uint32_t cvt_e4m3x2(float lo, float hi) {
  unsigned short r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
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

// quantize_block_fp8_mx's scale: amax -> the UE8M0 byte and the inverse scale 2^(127 - byte) (0 for byte 0)
__device__ __forceinline__ uint32_t mx_scale(float amax, float& inv) {
  const float sc = fmul(amax, __uint_as_float(0x3B124925u));  // amax * (1 / 448)
  uint32_t bits = __float_as_uint(sc);
  if (bits & 0x007FFFFFu) bits = (bits + 0x00800000u) & 0x7F800000u;  // pow2_ceil_ue8m0
  const uint32_t byte = (bits >> 23) & 0xFFu;
  inv = ex2_approx((float)(127 - (int)byte));
  if (byte == 0) inv = 0.f;
  return byte;
}

// 4x4 transpose of 16-bit elements across the 4 lanes of a quad: lane c holds h[nt] (nt = 0..3); afterwards lane c
// holds, as two words, (lane 0's h[c], lane 1's h[c], lane 2's h[c], lane 3's h[c])
__device__ __forceinline__ void quad_transpose16(const uint32_t (&h)[4], int c, uint32_t& o0, uint32_t& o1) {
  const uint32_t w0 = h[0] | (h[1] << 16), w1 = h[2] | (h[3] << 16);
  const uint32_t send = (c & 2) ? w0 : w1;
  const uint32_t recv = __shfl_xor_sync(0xffffffffu, send, 2);
  const uint32_t x0 = (c & 2) ? recv : w0;  // M[c & 1][2 (c >> 1), +1]
  const uint32_t x1 = (c & 2) ? w1 : recv;  // M[(c & 1) + 2][2 (c >> 1), +1]
  const uint32_t send2 = (c & 1) ? prmt(x0, x1, 0x5410u) : prmt(x0, x1, 0x7632u);
  const uint32_t recv2 = __shfl_xor_sync(0xffffffffu, send2, 1);
  if (c & 1) {
    o0 = prmt(x0, recv2, 0x3254u);
    o1 = prmt(x1, recv2, 0x3276u);
  } else {
    o0 = prmt(x0, recv2, 0x5410u);
    o1 = prmt(x1, recv2, 0x7610u);
  }
}
// the same for 32-bit elements: lane c holds w[nt]; afterwards o = (lane 0's w[c], lane 1's w[c], lane 2's, lane 3's)
__device__ __forceinline__ void quad_transpose32(const uint32_t (&w)[4], int c, uint32_t (&o)[4]) {
  const uint32_t s0 = (c & 2) ? w[0] : w[2], s1 = (c & 2) ? w[1] : w[3];
  const uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
  // lanes c < 2: (M[c][0], M[c][1], M[c+2][0], M[c+2][1]); lanes c >= 2: (M[c-2][2], M[c-2][3], M[c][2], M[c][3])
  const uint32_t x0 = (c & 2) ? r0 : w[0], x1 = (c & 2) ? r1 : w[1];
  const uint32_t x2 = (c & 2) ? w[2] : r0, x3 = (c & 2) ? w[3] : r1;
  const uint32_t t0 = (c & 1) ? x0 : x1, t1 = (c & 1) ? x2 : x3;
  const uint32_t u0 = __shfl_xor_sync(0xffffffffu, t0, 1), u1 = __shfl_xor_sync(0xffffffffu, t1, 1);
  if (c & 1) {
    o[0] = u0; o[1] = x1; o[2] = u1; o[3] = x3;
  } else {
    o[0] = x0; o[1] = u0; o[2] = x2; o[3] = u1;
  }
}

// ---------------------------------------------------------------------------------- 1. front-end (quant + routing)
struct FrontArgs {
  const uint32_t* hidden;   // [T][K] bf16
  uint32_t* a_values;       // [T][K] e4m3
  uint8_t* a_scales;        // [T][K/32] ue8m0
  const int32_t* ids;       // [T * topk]
  int32_t* row_pair;        // [row_cap]
  int4* f1_items;           // [f1_cap] (expert, row0, m_tiles, tile)
  int4* f2_items;           // [f2_cap]
  int32_t* counts;          // [8]: rows_padded, n_f1, n_f2, fc1 work counter, fc2 work counter
  int T, K, n_pairs, E, n2_tiles, row_cap, f1_cap, f2_cap;
};

// quantize_block_fp8_mx of 32 fp32 values with their amax (payload = e4m3x4 words, returns the scale byte)
__device__ __forceinline__ uint32_t quant32(const float (&v)[32], float amax, uint32_t (&payload)[8]) {
  float inv;
  const uint32_t byte = mx_scale(amax, inv);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t lo = cvt_e4m3x2(fmul(v[4 * i + 0], inv), fmul(v[4 * i + 1], inv));
    const uint32_t hi = cvt_e4m3x2(fmul(v[4 * i + 2], inv), fmul(v[4 * i + 3], inv));
    payload[i] = lo | (hi << 16);
  }
  return byte;
}

// spec B's quant_rows: one warp per token row, lane = the 32-column blocks lane, lane + 32, ...
__device__ __forceinline__ void quant_row(const FrontArgs& a, int row, int lane) {
  const int blocks = a.K / 32;
  const uint32_t* src = a.hidden + (size_t)row * (a.K / 2);
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
    uint4* dst = reinterpret_cast<uint4*>(a.a_values + (size_t)row * (a.K / 4) + b * 8);
    dst[0] = make_uint4(payload[0], payload[1], payload[2], payload[3]);
    dst[1] = make_uint4(payload[4], payload[5], payload[6], payload[7]);
    a.a_scales[(size_t)row * blocks + b] = (uint8_t)byte;
  }
}

// exclusive scan over the 512 threads of the routing block (one value per thread); total to everyone
__device__ __forceinline__ int block_exclusive_scan(int v, int* s_warp, int& total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) s_warp[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int t = lane < kFrontWarps ? s_warp[lane] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, t, o);
      if (lane >= o) t += y;
    }
    if (lane < kFrontWarps) s_warp[lane] = t;  // inclusive over the warps
  }
  __syncthreads();
  const int before = warp ? s_warp[warp - 1] : 0;
  total = s_warp[kFrontWarps - 1];
  __syncthreads();
  return before + x - v;
}

__global__ void __launch_bounds__(kFrontThreads) frontend_kernel(FrontArgs a) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (blockIdx.x + 1 < gridDim.x) {  // token rows quant
    const int row = blockIdx.x * kFrontWarps + warp;
    if (row < a.T) quant_row(a, row, lane);
    return;
  }
  // ------------------------------------------------------------ deterministic routing (the last block)
  __shared__ int s_cnt[kMaxE];
  __shared__ int s_off[kMaxE];    // padded row offset of the expert
  __shared__ int s_run[kMaxE];    // pairs of the expert scattered so far
  __shared__ int s_f1[kMaxE];     // first FC1 / FC2 item of the expert, its chunk count
  __shared__ int s_f2[kMaxE];
  __shared__ int s_chunks[kMaxE];
  __shared__ int s_w[kFrontWarps][kMaxE];  // per-warp per-expert counts of the chunk -> exclusive prefix over warps
  __shared__ int s_scan[kFrontWarps];
  const int E = a.E, n_pairs = a.n_pairs;
  for (int e = tid; e < kMaxE; e += kFrontThreads) { s_cnt[e] = 0; s_run[e] = 0; }
  __syncthreads();
  const uint32_t lanemask_lt = (1u << lane) - 1u;
  // integer counts: the sums do not depend on the order. The chunk loop has a uniform trip count: every lane of every
  // warp reaches the __match_any_sync (a per-thread trip count left a partial warp in the last chunk whenever
  // n_pairs % 32 != 0 - e.g. T = 300 - and the full-mask match then waited forever for the lanes parked at the loop exit).
  for (int base = 0; base < n_pairs; base += kFrontThreads) {
    const int p = base + tid;
    int e = p < n_pairs ? a.ids[p] : -1;
    if (!(e >= 0 && e < E)) e = -1;
    const uint32_t mask = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && (mask & lanemask_lt) == 0u) atomicAdd(&s_cnt[e], __popc(mask));  // one atomic per (warp, expert)
  }
  __syncthreads();
  int padded = 0, chunks = 0;
  if (tid < E) {
    padded = (s_cnt[tid] + 15) & ~15;
    chunks = (padded / 16 + 3) >> 2;
  }
  int rows_padded, n_f1, n_f2;
  const int off = block_exclusive_scan(padded, s_scan, rows_padded);
  const int f1 = block_exclusive_scan(chunks * kN1Tiles, s_scan, n_f1);
  const int f2 = block_exclusive_scan(chunks * a.n2_tiles, s_scan, n_f2);
  if (tid < E) { s_off[tid] = off; s_f1[tid] = f1; s_f2[tid] = f2; s_chunks[tid] = chunks; }
  if (tid == 0) {
    a.counts[0] = rows_padded; a.counts[1] = min(n_f1, a.f1_cap); a.counts[2] = min(n_f2, a.f2_cap);
    a.counts[3] = 0; a.counts[4] = 0;  // the work counters of the persistent kernels
  }
  for (int r = tid; r < rows_padded && r < a.row_cap; r += kFrontThreads) a.row_pair[r] = -1;
  __syncthreads();
  // the work items, tile-major within the expert (consecutive items share the weight tile): a thread per (expert, tile)
  for (int et = tid; et < E * (kN1Tiles + a.n2_tiles); et += kFrontThreads) {
    const bool fc2 = et >= E * kN1Tiles;
    const int e = fc2 ? (et - E * kN1Tiles) / a.n2_tiles : et / kN1Tiles;
    const int t = fc2 ? (et - E * kN1Tiles) - e * a.n2_tiles : et - e * kN1Tiles;
    const int chunks_e = s_chunks[e], m_tiles = chunks_e == 0 ? 0 : ((s_cnt[e] + 15) >> 4), base = fc2 ? s_f2[e] : s_f1[e];
    for (int ch = 0; ch < chunks_e; ++ch) {
      const int i = base + t * chunks_e + ch;
      const int4 v = make_int4(e, s_off[e] + ch * 64, min(4, m_tiles - ch * 4), t);
      if (fc2) { if (i < a.f2_cap) a.f2_items[i] = v; }
      else if (i < a.f1_cap) a.f1_items[i] = v;
    }
  }
  // stable scatter in chunks of 512 pairs: rank of a pair within its expert = the expert's pairs before it
  for (int base = 0; base < n_pairs; base += kFrontThreads) {
    const int p = base + tid;
    int e = p < n_pairs ? a.ids[p] : -1;
    const bool valid = e >= 0 && e < E;
    if (!valid) e = -1;
    const uint32_t mask = __match_any_sync(0xffffffffu, e);
    const int rank = __popc(mask & lanemask_lt);
    const bool leader = (mask & lanemask_lt) == 0u;
    for (int i = lane; i < E; i += 32) s_w[warp][i] = 0;
    __syncwarp();
    if (valid && leader) s_w[warp][e] = __popc(mask);
    __syncthreads();
    if (tid < E) {  // exclusive prefix over the warps for this expert, carried across chunks
      int run = s_run[tid];
#pragma unroll
      for (int w = 0; w < kFrontWarps; ++w) {
        const int v = s_w[w][tid];
        s_w[w][tid] = run;
        run += v;
      }
      s_run[tid] = run;
    }
    __syncthreads();
    if (valid) {
      const int r = s_off[e] + s_w[warp][e] + rank;
      if (r < a.row_cap) a.row_pair[r] = p;
    }
    __syncthreads();  // the next chunk re-zeroes s_w
  }
}

// ------------------------------------------------------------------------------------- 2. the FC1 / FC2 skeleton
// Both GEMMs run the same persistent, continuously pipelined loop: a CTA takes work items (expert, <= 4 m16 tiles of
// its rows, one column tile); for every item it streams K64 stages through a STAGES-deep cp.async ring; the stage
// stream does not stop at item boundaries (the next item's first stages are in flight during the epilogue).
//
// Stage layout (bytes): B [kk 2][n32 chunk NS][lane 32][16 B] per projection (FC1: 2 projections), then A as
// [m-tile 4][kk 2][quad row 8][c 4][4 words] where the 4 words of a (m-tile, kk, q, c) group are
// (row q: k32 word 2c, row q + 8: word 2c, row q: word 2c + 1, row q + 8: word 2c + 1) = the QMMA A fragment of lane
// (q, c) in register order: one LDS.128 per m-tile and k32, no shuffling. Then the k128 scale words of the 64 rows and
// of the columns ([proj][128] words; FC2's tail slice: 128 x u16).
// A is copied with 4 B cp.async: warp w takes k32 block kk = w / 4 of the rows p, p + 8, p + 1, p + 9 (p = 2 (w % 4)),
// lane = (row of the four, word 0..7 of the k32 block) - the 32 words of one instruction land in 32 distinct banks.
//
// Work-item hand-off: item_0 = blockIdx.x; item_{k+1} = gridDim.x + atomicAdd(counter) (dynamic) or item_k + gridDim.x
// (static). Thread 0 claims it 10 stages before the issue side needs it (ring.items[(k+1) & 1]); the item's metadata
// (four ints, threads 0-3) and its 64 row -> pair entries (threads 0-63) are staged through ring.meta / ring.pairs /
// ring.aoff (slot (k+1) & 1), every global load consumed two stages after it was issued (one register per thread in
// between), every shared hand-off with a barrier in between. The compute side reads the same slots: meta at the start
// of item k + 1, items at the end of item k, pairs in the epilogue (slot k & 1, rewritten only while item k + 1 is
// being issued, long after).
struct ItemRing {
  int items[2];
  int4 meta[2];        // (expert, row0, m_tiles, tile); m_tiles == 0: no item
  int pairs[2][64];    // pair of the 64 rows (-1: padding)
  uint32_t aoff[2][64];  // A row byte offset of the 64 rows (0xFFFFFFFF: padding)
};

__device__ __forceinline__ int claim_next_item(int32_t* counter, int cur, int dynamic_schedule) {
  return dynamic_schedule ? (int)gridDim.x + atomicAdd(counter, 1) : cur + (int)gridDim.x;
}

// the staged hand-off of the next work item (see above): t = the stage being issued, c = claim_at
template <typename AoffFn>
__device__ __forceinline__ void stage_next_item(ItemRing& ring, int t, int c, int nb, int tid, int n_items,
                                                const int4* items, const int32_t* row_pair, int32_t* counter,
                                                int cur_item, int dynamic_schedule, int& nxt_item, int& nxt_meta_i,
                                                int& nxt_pair, AoffFn aoff_of) {
  if (t == c && tid == 0) nxt_item = claim_next_item(counter, cur_item, dynamic_schedule);
  if (t == c + 2 && tid == 0) ring.items[nb] = nxt_item;
  if (t == c + 3 && tid < 4) {
    const int nx = ring.items[nb];
    nxt_meta_i = nx < n_items ? __ldg(reinterpret_cast<const int32_t*>(items) + (size_t)nx * 4 + tid) : 0;
  }
  if (t == c + 5 && tid < 4) reinterpret_cast<int*>(&ring.meta[nb])[tid] = nxt_meta_i;
  if (t == c + 6 && tid < 64) {
    const int4 m = ring.meta[nb];
    nxt_pair = tid < m.z * 16 ? __ldg(row_pair + m.y + tid) : -1;
  }
  if (t == c + 8 && tid < 64) {
    ring.pairs[nb][tid] = nxt_pair;
    ring.aoff[nb][tid] = nxt_pair >= 0 ? aoff_of(nxt_pair) : 0xFFFFFFFFu;
  }
}

struct Fc1Args {
  const uint32_t* a_values;  // [T][K/4]
  const uint8_t* a_scales;   // [T][K/32]
  const uint32_t* w13;       // n64-repacked planes (see the spec header)
  const uint32_t* w13_sf;
  const int32_t* row_pair;
  const int4* items;
  int32_t* counts;           // [1] = item count, [3] = work counter
  const float* alpha;        // [E]
  const float* input_global_scale;  // [E] or [1]
  uint32_t* inter;           // [pairs][640 B]
  uint8_t* inter_sf;         // [pairs][20 B]
  int K, topk, igs_per_expert, dynamic_schedule;
  float limit;
};

template <int STAGES, int MINB>
__global__ void __launch_bounds__(kThreads, MINB) fc1_kernel(Fc1Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ ItemRing ring;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int q = lane >> 2, c = lane & 3;
  const int K = a.K, n_stages = K / 64;
  const int claim_at = n_stages - 10;  // the next item is claimed 10 stages before it is needed (K >= 640)
  const int n_items = a.counts[1];
  const uint32_t wproj_b = (uint32_t)kN * (K / 32) * 16;  // bytes per projection of w13
  const uint32_t sproj_b = (uint32_t)kN * (K / 32);       // bytes per projection of w13_sf
  const uint32_t sbase0 = smem_addr(smem);
  // A-copy role: k32 block kk, rows p / p+8 / p+1 / p+9 of each m-tile, word kw8 (see the header)
  const int kk_c = warp >> 2, sub = (lane >> 3), kw8 = lane & 7;
  const int my_r16 = 2 * (warp & 3) + (sub >> 1) + 8 * (sub & 1);
  const uint32_t a_dst0 = (uint32_t)((kk_c * 8 + 2 * (warp & 3) + (sub >> 1)) * 64 + (kw8 >> 1) * 16 + ((kw8 & 1) * 2 + (sub & 1)) * 4);
  const uint32_t a_src0 = kk_c * 32 + kw8 * 4;
  const bool sfa_role = (tid & 0x87) == 0;  // kk 0, word 0: the row's k128 scale word
  const auto aoff_of = [&](int pair) { return (uint32_t)(pair / a.topk) * (uint32_t)K; };

  // ---- issue side
  int iss_item = blockIdx.x, iss_k = 0, iss_stage = 0, iss_mtiles = 0, iss_slot = 0;
  bool iss_wide = true;
  const uint8_t* iss_b = reinterpret_cast<const uint8_t*>(a.w13);
  const uint8_t* iss_sfb = reinterpret_cast<const uint8_t*>(a.w13_sf);
  int nxt_item = 0, nxt_meta_i = 0, nxt_pair = -1;

  auto start_issue_item = [&](int slot) {
    const int4 it = ring.meta[slot];
    iss_stage = 0;
    iss_slot = slot;
    iss_mtiles = it.z;
    if (it.z > 0) {
      const int e = it.x, tile = it.w;
      iss_wide = tile != kN1Tiles - 1;
      iss_b = reinterpret_cast<const uint8_t*>(a.w13) + (size_t)e * 2 * wproj_b + (size_t)tile * 128 * (K / 32) * 16;
      iss_sfb = reinterpret_cast<const uint8_t*>(a.w13_sf) + (size_t)e * 2 * sproj_b + (size_t)tile * 128 * (K / 32);
    }
  };
  auto issue_stage = [&](int buf) {
    if (iss_mtiles > 0) {
      const uint32_t sb = sbase0 + buf * kF1Stage;
      const int t = iss_stage, nb = (iss_k + 1) & 1;
      uint32_t aoff4[4];  // the rows' A offsets first: their LDS latency overlaps the B address work (ncu: ISETP on them stalled)
#pragma unroll
      for (int j = 0; j < 4; ++j) aoff4[j] = ring.aoff[iss_slot][j * 16 + my_r16];
      stage_next_item(ring, t, claim_at, nb, tid, n_items, a.items, a.row_pair, a.counts + 3, iss_item,
                      a.dynamic_schedule, nxt_item, nxt_meta_i, nxt_pair, aoff_of);
      // B: two projections, the 2 k32 blocks of the stage (128-column tiles: 256 x 16 B per projection)
      if (iss_wide) {
        const uint32_t o = (tid >> 7) * 2048 + (tid & 127) * 16;
        const uint8_t* src = iss_b + (size_t)t * 4096 + o;
        cp_async16(sb + o, src);
        cp_async16(sb + 4096 + o, src + wproj_b);
        if (tid < 64) {
          const uint32_t proj = tid >> 5, j = tid & 31;
          cp_async16(sb + kF1B + kF1A + kF1SFA + proj * 512 + j * 16, iss_sfb + proj * sproj_b + (t >> 1) * 512 + j * 16);
        }
      } else {
        if (tid < 128) {
          const uint32_t kk = tid >> 6, rem = tid & 63;
          const uint8_t* src = iss_b + (size_t)t * 2048 + kk * 1024 + rem * 16;
          cp_async16(sb + kk * 2048 + rem * 16, src);
          cp_async16(sb + 4096 + kk * 2048 + rem * 16, src + wproj_b);
        }
        if (tid < 32) {
          const uint32_t proj = tid >> 4, j = tid & 15;
          cp_async16(sb + kF1B + kF1A + kF1SFA + proj * 512 + j * 16, iss_sfb + proj * sproj_b + (t >> 1) * 256 + j * 16);
        }
      }
      // A: one word per thread and m-tile (+ the row's k128 scale word), zero-filled for padding rows / tiles
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t off = aoff4[j];
        const bool on = j < iss_mtiles && off != 0xFFFFFFFFu;
        const uint32_t o = on ? off : 0u;
        cp_async4_zfill(sb + kF1B + a_dst0 + j * 1024, reinterpret_cast<const uint8_t*>(a.a_values) + o + t * 64 + a_src0, on ? 4 : 0);
        if (sfa_role) cp_async4_zfill(sb + kF1B + kF1A + (j * 16 + my_r16) * 4, a.a_scales + (o >> 5) + (t >> 1) * 4, on ? 4 : 0);
      }
      if (++iss_stage == n_stages) {
        iss_item = ring.items[nb];
        ++iss_k;
        start_issue_item(nb);
      }
    }
    cp_async_commit();
  };

  // ---- the first item: metadata, pairs and offsets straight from global memory
  // the compute side's item index k is not kept: the issue side is exactly STAGES - 1 stages ahead (one issue per
  // computed stage, STAGES - 1 in the prologue, STAGES - 1 < n_stages), so iss_k == k at the start of item k and
  // iss_k == k + 1 after its K-loop (iss_k is bumped when the last stage of an item is issued, also when no item follows).
  int item = blockIdx.x;
  if (tid < 4) reinterpret_cast<int*>(&ring.meta[0])[tid] = item < n_items ? __ldg(reinterpret_cast<const int32_t*>(a.items) + (size_t)item * 4 + tid) : 0;
  __syncthreads();
  if (tid < 64) {
    const int4 m = ring.meta[0];
    const int pr = tid < m.z * 16 ? __ldg(a.row_pair + m.y + tid) : -1;
    ring.pairs[0][tid] = pr;
    ring.aoff[0][tid] = pr >= 0 ? aoff_of(pr) : 0xFFFFFFFFu;
  }
  __syncthreads();
  start_issue_item(0);
  int buf_issue = 0;
#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s > 0) __syncthreads();
    issue_stage(buf_issue);
    buf_issue = buf_issue + 1 == STAGES ? 0 : buf_issue + 1;
  }
  int buf = 0;
  while (item < n_items) {
    const int4 it = ring.meta[iss_k & 1];  // slot k & 1
    const int e = it.x, m_tiles = it.z, tile = it.w;
    const int lg = tile == kN1Tiles - 1 ? 1 : 2, NG = 8 >> lg;  // 4 n32 slices (2 for the 64-column tile)
    const int ns = warp & ((1 << lg) - 1), mg = warp >> lg;
    const int mt0 = mg, mt1 = mg + NG;
    const bool v0 = mt0 < m_tiles, v1 = mt1 < m_tiles;
    const float alpha = fmul(__ldg(a.alpha + e), __ldg(a.input_global_scale + (a.igs_per_expert ? e : 0)));

    float acc[2][2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int r = 0; r < 4; ++r) acc[i][p][nt][r] = 0.f;

    const uint32_t a_off0 = kF1B + (mt0 * 16 + q) * 64 + c * 16;  // + kk * 512
    const uint32_t a_off1 = kF1B + (mt1 * 16 + q) * 64 + c * 16;
    const uint32_t sfa_off0 = kF1B + kF1A + (mt0 * 16 + q + ((lane & 1) << 3)) * 4;
    const uint32_t sfa_off1 = kF1B + kF1A + (mt1 * 16 + q + ((lane & 1) << 3)) * 4;
    const uint32_t b_off = ns * 512 + lane * 16;
    const uint32_t sfb_off = kF1B + kF1A + kF1SFA + (ns * 32 + q) * 4;

    auto compute_stage = [&](const uint8_t* sp, auto kb_tag) {
      constexpr int KB0 = decltype(kb_tag)::value;
      uint32_t sfb[2][4], sfa0 = 0u, sfa1 = 0u;
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) sfb[p][nt] = *reinterpret_cast<const uint32_t*>(sp + sfb_off + p * 512 + nt * 32);
      if (v0) sfa0 = *reinterpret_cast<const uint32_t*>(sp + sfa_off0);
      if (v1) sfa1 = *reinterpret_cast<const uint32_t*>(sp + sfa_off1);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        // both A fragments first, then one projection's B at a time: 8 + 8 live fragment registers instead of 16 + 4
        // (the per-accumulator QMMA chain is unchanged: acc[i][p][nt] still sees the k32 blocks in order)
        uint4 af0 = make_uint4(0u, 0u, 0u, 0u), af1 = make_uint4(0u, 0u, 0u, 0u);
        if (v0) af0 = *reinterpret_cast<const uint4*>(sp + a_off0 + kk * 512);
        if (v1) af1 = *reinterpret_cast<const uint4*>(sp + a_off1 + kk * 512);
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          uint32_t b[4][2];
          const uint4 w = *reinterpret_cast<const uint4*>(sp + p * 4096 + kk * 2048 + b_off);
          spread_e2m1x8(w.x, b[0][0], b[0][1]);
          spread_e2m1x8(w.y, b[1][0], b[1][1]);
          spread_e2m1x8(w.z, b[2][0], b[2][1]);
          spread_e2m1x8(w.w, b[3][0], b[3][1]);
          if (v0) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
              qmma_kk<KB0>(kk, acc[0][p][nt], af0.x, af0.y, af0.z, af0.w, b[nt][0], b[nt][1], sfa0, sfb[p][nt]);
          }
          if (v1) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
              qmma_kk<KB0>(kk, acc[1][p][nt], af1.x, af1.y, af1.z, af1.w, b[nt][0], b[nt][1], sfa1, sfb[p][nt]);
          }
        }
      }
    };

#pragma unroll 1
    for (int t = 0; t < n_stages; t += 2) {
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      issue_stage(buf_issue);
      buf_issue = buf_issue + 1 == STAGES ? 0 : buf_issue + 1;
      compute_stage(smem + buf * kF1Stage, std::integral_constant<int, 0>{});
      buf = buf + 1 == STAGES ? 0 : buf + 1;
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      issue_stage(buf_issue);
      buf_issue = buf_issue + 1 == STAGES ? 0 : buf_issue + 1;
      compute_stage(smem + buf * kF1Stage, std::integral_constant<int, 2>{});
      buf = buf + 1 == STAGES ? 0 : buf + 1;
    }

    // ---- epilogue: activation, bf16, MXFP8 per row and 32-column block (this warp's n32 slice), in registers
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if (!(i == 0 ? v0 : v1)) continue;
      const int mt = i == 0 ? mt0 : mt1;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int pair = ring.pairs[(iss_k + 1) & 1][mt * 16 + q + hf * 8];  // slot k & 1 (iss_k == k + 1 here)
        float v[8];
        uint32_t h[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          // projection 0 of the plane = up, 1 = gate
          const float x0 = activated_value(acc[i][1][nt][2 * hf], acc[i][0][nt][2 * hf], alpha, a.limit);
          const float x1 = activated_value(acc[i][1][nt][2 * hf + 1], acc[i][0][nt][2 * hf + 1], alpha, a.limit);
          const uint32_t pk = pack_bf16x2(x0, x1);
          v[2 * nt] = __uint_as_float(pk << 16);
          v[2 * nt + 1] = __uint_as_float(pk & 0xFFFF0000u);
        }
        float amax = 0.f;
#pragma unroll
        for (int kx = 0; kx < 8; ++kx) amax = fmax_plain(amax, fabs_plain(v[kx]));
        amax = fmax_plain(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmax_plain(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        float inv;
        const uint32_t byte = mx_scale(amax, inv);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) h[nt] = cvt_e4m3x2(fmul(v[2 * nt], inv), fmul(v[2 * nt + 1], inv));
        uint32_t o0, o1;
        quad_transpose16(h, c, o0, o1);
        if (pair >= 0) {
          uint2* dst = reinterpret_cast<uint2*>(reinterpret_cast<uint8_t*>(a.inter) + (size_t)pair * kInterBytes +
                                               tile * 128 + ns * 32 + c * 8);
          *dst = make_uint2(o0, o1);
          uint8_t* sf = a.inter_sf + (size_t)pair * (kSlices * 4) + tile * 4;
          if (c == 0) sf[ns] = (uint8_t)byte;
          if (tile == kN1Tiles - 1 && c == 1) sf[ns + 2] = 0;  // the tail slice's zero scale bytes
        }
      }
    }
    item = ring.items[iss_k & 1];  // slot (k + 1) & 1
  }
  cp_async_wait<0>();
}

// ------------------------------------------------------------------------------------------------------ 3. FC2
struct Fc2Args {
  const uint32_t* inter;     // [pairs][640 B]
  const uint8_t* inter_sf;   // [pairs][20 B]
  const uint32_t* w2;        // n64-repacked
  const uint8_t* w2_sf;
  const int32_t* row_pair;
  const int4* items;
  int32_t* counts;           // [2] = item count, [4] = work counter
  const float* token_weights;  // [pairs]
  const float* down_alpha;     // [E]
  const float* global_scale;   // [E] or [1]
  uint32_t* route;             // [pairs][K] bf16
  int K, out_tiles, gs_per_expert, dynamic_schedule;
};

template <int STAGES, int MINB>
__global__ void __launch_bounds__(kThreads, MINB) fc2_kernel(Fc2Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ ItemRing ring;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int q = lane >> 2, c = lane & 3;
  constexpr int n_stages = kF2Stages;
  constexpr int claim_at = n_stages - 10;
  constexpr uint32_t kTileBytes = 128 * (kSlices * 4 - 2) * 16;  // 36 KB of weights per 128-column tile (18 k32)
  constexpr uint32_t kTileSfBytes = 128 * (kSlices * 4 - 2);     // 2304 B of scales per tile
  const int n_items = a.counts[2];
  const uint32_t sbase0 = smem_addr(smem);
  const int kk_c = warp >> 2, sub = (lane >> 3), kw8 = lane & 7;
  const int my_r16 = 2 * (warp & 3) + (sub >> 1) + 8 * (sub & 1);
  const uint32_t a_dst0 = (uint32_t)((kk_c * 8 + 2 * (warp & 3) + (sub >> 1)) * 64 + (kw8 >> 1) * 16 + ((kw8 & 1) * 2 + (sub & 1)) * 4);
  const uint32_t a_src0 = kk_c * 32 + kw8 * 4;
  const bool sfa_role = (tid & 0x87) == 0;
  const auto aoff_of = [](int pair) { return (uint32_t)pair * (uint32_t)kInterBytes; };

  int iss_item = blockIdx.x, iss_k = 0, iss_stage = 0, iss_mtiles = 0, iss_slot = 0;
  const uint8_t* iss_b = reinterpret_cast<const uint8_t*>(a.w2);
  const uint8_t* iss_sf = a.w2_sf;
  int nxt_item = 0, nxt_meta_i = 0, nxt_pair = -1;

  auto start_issue_item = [&](int slot) {
    const int4 it = ring.meta[slot];
    iss_stage = 0;
    iss_slot = slot;
    iss_mtiles = it.z;
    if (it.z > 0) {
      const int e = it.x, tile = it.w;
      iss_b = reinterpret_cast<const uint8_t*>(a.w2) + ((size_t)e * a.out_tiles + tile) * kTileBytes;
      iss_sf = a.w2_sf + ((size_t)e * a.out_tiles + tile) * kTileSfBytes;
    }
  };
  auto issue_stage = [&](int buf) {
    if (iss_mtiles > 0) {
      const uint32_t sb = sbase0 + buf * kF2Stage;
      const int t = iss_stage, nb = (iss_k + 1) & 1;
      uint32_t aoff4[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) aoff4[j] = ring.aoff[iss_slot][j * 16 + my_r16];
      stage_next_item(ring, t, claim_at, nb, tid, n_items, a.items, a.row_pair, a.counts + 4, iss_item,
                      a.dynamic_schedule, nxt_item, nxt_meta_i, nxt_pair, aoff_of);
      if (t < n_stages - 1) {  // the last stage is the zero tail: nothing to load
        const uint32_t o = (tid >> 7) * 2048 + (tid & 127) * 16;
        cp_async16(sb + o, iss_b + (size_t)t * 4096 + o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t off = aoff4[j];
          const bool on = j < iss_mtiles && off != 0xFFFFFFFFu;
          const uint32_t oo = on ? off : 0u;
          cp_async4_zfill(sb + kF2B + a_dst0 + j * 1024, reinterpret_cast<const uint8_t*>(a.inter) + oo + t * 64 + a_src0, on ? 4 : 0);
          if (sfa_role) cp_async4_zfill(sb + kF2B + kF2A + (j * 16 + my_r16) * 4, a.inter_sf + (oo >> 5) + (t >> 1) * 4, on ? 4 : 0);
        }
        const int s = t >> 1;
        if (tid < (s == kSlices - 1 ? 16 : 32)) cp_async16(sb + kF2B + kF2A + kF2SFA + tid * 16, iss_sf + s * 512 + tid * 16);
      }
      if (++iss_stage == n_stages) {
        iss_item = ring.items[nb];
        ++iss_k;
        start_issue_item(nb);
      }
    }
    cp_async_commit();
  };

  int item = blockIdx.x, k = 0;
  if (tid < 4) reinterpret_cast<int*>(&ring.meta[0])[tid] = item < n_items ? __ldg(reinterpret_cast<const int32_t*>(a.items) + (size_t)item * 4 + tid) : 0;
  __syncthreads();
  if (tid < 64) {
    const int4 m = ring.meta[0];
    const int pr = tid < m.z * 16 ? __ldg(a.row_pair + m.y + tid) : -1;
    ring.pairs[0][tid] = pr;
    ring.aoff[0][tid] = pr >= 0 ? aoff_of(pr) : 0xFFFFFFFFu;
  }
  __syncthreads();
  start_issue_item(0);
  int buf_issue = 0;
#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s > 0) __syncthreads();
    issue_stage(buf_issue);
    buf_issue = buf_issue + 1 == STAGES ? 0 : buf_issue + 1;
  }
  int buf = 0;
  while (item < n_items) {
    const int4 it = ring.meta[k & 1];
    const int e = it.x, m_tiles = it.z, tile = it.w;
    const int ns = warp & 3, mg = warp >> 2;
    const int mt0 = mg, mt1 = mg + 2;
    const bool v0 = mt0 < m_tiles, v1 = mt1 < m_tiles;
    const float down_scale = fmul(__ldg(a.down_alpha + e), __ldg(a.global_scale + (a.gs_per_expert ? e : 0)));
    // the epilogue's pairs and token weights, fetched now (latency hidden behind the item's stages)
    int ep_pair[2][2];
    float ep_tw[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int mt = i == 0 ? mt0 : mt1;
        const int pr = (i == 0 ? v0 : v1) ? ring.pairs[k & 1][mt * 16 + q + hf * 8] : -1;
        ep_pair[i][hf] = pr;
        ep_tw[i][hf] = __ldg(a.token_weights + (pr >= 0 ? pr : 0));
      }

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][nt][r] = 0.f;

    const uint32_t a_off0 = kF2B + (mt0 * 16 + q) * 64 + c * 16;
    const uint32_t a_off1 = kF2B + (mt1 * 16 + q) * 64 + c * 16;
    const uint32_t sfa_off0 = kF2B + kF2A + (mt0 * 16 + q + ((lane & 1) << 3)) * 4;
    const uint32_t sfa_off1 = kF2B + kF2A + (mt1 * 16 + q + ((lane & 1) << 3)) * 4;
    const uint32_t b_off = ns * 512 + lane * 16;
    const int col_q = ns * 32 + q;

    // ZERO: the k32 blocks 18-19 (zero A, zero B, zero scale bytes); TAIL: slice 4 (u16 scale words)
    auto compute_stage = [&](const uint8_t* sp, auto kb_tag, auto zero_tag, auto tail_tag) {
      constexpr int KB0 = decltype(kb_tag)::value;
      constexpr bool ZERO = decltype(zero_tag)::value;
      constexpr bool TAIL = decltype(tail_tag)::value;
      uint32_t sfb[4] = {0u, 0u, 0u, 0u}, sfa0 = 0u, sfa1 = 0u;
      if constexpr (!ZERO) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          if constexpr (TAIL) sfb[nt] = *reinterpret_cast<const uint16_t*>(sp + kF2B + kF2A + kF2SFA + (col_q + nt * 8) * 2);
          else sfb[nt] = *reinterpret_cast<const uint32_t*>(sp + kF2B + kF2A + kF2SFA + (col_q + nt * 8) * 4);
        }
        if (v0) sfa0 = *reinterpret_cast<const uint32_t*>(sp + sfa_off0);
        if (v1) sfa1 = *reinterpret_cast<const uint32_t*>(sp + sfa_off1);
      }
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t b[4][2];
        uint4 af0 = make_uint4(0u, 0u, 0u, 0u), af1 = make_uint4(0u, 0u, 0u, 0u);
        if constexpr (!ZERO) {
          const uint4 w = *reinterpret_cast<const uint4*>(sp + kk * 2048 + b_off);
          spread_e2m1x8(w.x, b[0][0], b[0][1]);
          spread_e2m1x8(w.y, b[1][0], b[1][1]);
          spread_e2m1x8(w.z, b[2][0], b[2][1]);
          spread_e2m1x8(w.w, b[3][0], b[3][1]);
          if (v0) af0 = *reinterpret_cast<const uint4*>(sp + a_off0 + kk * 512);
          if (v1) af1 = *reinterpret_cast<const uint4*>(sp + a_off1 + kk * 512);
        } else {
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) b[nt][0] = b[nt][1] = 0u;
        }
        if (v0) {
#pragma unroll
          for (int nt = 0; nt < 4; ++nt)
            qmma_kk<KB0>(kk, acc[0][nt], af0.x, af0.y, af0.z, af0.w, b[nt][0], b[nt][1], sfa0, sfb[nt]);
        }
        if (v1) {
#pragma unroll
          for (int nt = 0; nt < 4; ++nt)
            qmma_kk<KB0>(kk, acc[1][nt], af1.x, af1.y, af1.z, af1.w, b[nt][0], b[nt][1], sfa1, sfb[nt]);
        }
      }
    };
    using I0 = std::integral_constant<int, 0>;
    using I2 = std::integral_constant<int, 2>;
    using F = std::false_type;
    using Tr = std::true_type;
    auto step = [&](auto kb_tag, auto zero_tag, auto tail_tag) {
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      issue_stage(buf_issue);
      buf_issue = buf_issue + 1 == STAGES ? 0 : buf_issue + 1;
      compute_stage(smem + buf * kF2Stage, kb_tag, zero_tag, tail_tag);
      buf = buf + 1 == STAGES ? 0 : buf + 1;
    };
#pragma unroll 1
    for (int t = 0; t < n_stages - 2; t += 2) {  // slices 0-3: stages 0..7
      step(I0{}, F{}, F{});
      step(I2{}, F{}, F{});
    }
    step(I0{}, F{}, Tr{});   // stage 8: k32 16, 17 (the tail slice's u16 scale words)
    step(I2{}, Tr{}, Tr{});  // stage 9: k32 18, 19 (zeros)

    // ---- epilogue: scale, bf16, token-major route rows (16 B per lane after the quad transpose)
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if (!(i == 0 ? v0 : v1)) continue;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int pair = ep_pair[i][hf];
        const float scale = fmul(down_scale, ep_tw[i][hf]);
        uint32_t w[4], o[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
          w[nt] = pack_bf16x2(fmul(scale, acc[i][nt][2 * hf]), fmul(scale, acc[i][nt][2 * hf + 1]));
        quad_transpose32(w, c, o);
        if (pair >= 0) {
          uint4* dst = reinterpret_cast<uint4*>(a.route + (size_t)pair * (a.K / 2) + (tile * 128 + ns * 32 + c * 8) / 2);
          *dst = make_uint4(o[0], o[1], o[2], o[3]);
        }
      }
    }
    item = ring.items[(k + 1) & 1];
    ++k;
  }
  cp_async_wait<0>();
}

// ------------------------------------------------------------------------------------------------ 4. top-k sum
// spec B's topk_sum: out[t, col] = bf16 of the fp32 sum of the token's route values in slot order (ids < 0 skipped)
__global__ void __launch_bounds__(256) topk_sum_kernel(const uint4* __restrict__ route, const int32_t* __restrict__ ids,
                                                       uint4* __restrict__ out, int T, int topk, int K) {
  const int cols8 = K / 8;
  const long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
  if (idx >= (long long)T * cols8) return;
  const int t = (int)(idx / cols8), c8 = (int)(idx - (long long)t * cols8);
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
  for (int r = 0; r < topk; ++r) {
    const int row = t * topk + r;
    if (__ldg(ids + row) < 0) continue;
    const uint4 v = __ldg(route + (size_t)row * cols8 + c8);
    const uint32_t ws[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc[2 * j] = fadd(acc[2 * j], bf16_bits_to_f32(ws[j] & 0xFFFFu));
      acc[2 * j + 1] = fadd(acc[2 * j + 1], bf16_bits_to_f32(ws[j] >> 16));
    }
  }
  uint32_t o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    unsigned short lo, hi;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(lo) : "f"(acc[2 * j]));
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(hi) : "f"(acc[2 * j + 1]));
    o[j] = (uint32_t)lo | ((uint32_t)hi << 16);
  }
  out[(size_t)t * cols8 + c8] = make_uint4(o[0], o[1], o[2], o[3]);
}

}  // namespace b12x_moe_prefill_fast

// ------------------------------------------------------------------------------------------------------ host
namespace {

template <int S1, int MB1, int S2, int MB2>
struct Launch {
  static int grid1(int sms) {
    using namespace b12x_moe_prefill_fast;
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fc1_kernel<S1, MB1>, kThreads, S1 * kF1Stage);
    return sms * (occ > 0 ? occ : 1);
  }
  static int grid2(int sms) {
    using namespace b12x_moe_prefill_fast;
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fc2_kernel<S2, MB2>, kThreads, S2 * kF2Stage);
    return sms * (occ > 0 ? occ : 1);
  }
  static void prepare() {
    using namespace b12x_moe_prefill_fast;
    static bool done = false;
    if (done) return;
    cudaFuncSetAttribute(fc1_kernel<S1, MB1>, cudaFuncAttributeMaxDynamicSharedMemorySize, S1 * kF1Stage);
    cudaFuncSetAttribute(fc2_kernel<S2, MB2>, cudaFuncAttributeMaxDynamicSharedMemorySize, S2 * kF2Stage);
    done = true;
  }
  static void fc1(const b12x_moe_prefill_fast::Fc1Args& f1, int grid, cudaStream_t stream) {
    using namespace b12x_moe_prefill_fast;
    fc1_kernel<S1, MB1><<<grid, kThreads, S1 * kF1Stage, stream>>>(f1);
  }
  static void fc2(const b12x_moe_prefill_fast::Fc2Args& f2, int grid, cudaStream_t stream) {
    using namespace b12x_moe_prefill_fast;
    fc2_kernel<S2, MB2><<<grid, kThreads, S2 * kF2Stage, stream>>>(f2);
  }
};

}  // namespace

// hidden [T, K] bf16, ids [T, topk] int32, token_weights [T, topk] fp32 (contiguous). Returns
// {out, route, a_values, a_scales, inter, inter_sf, row_pair, counts}. `route_out` (if defined: [>= T*topk, K] bf16, the
// b12x workspace's route_output) receives the route rows; `out` (if defined) the top-k sum (skipped when !topk_sum).
// Tuning: cfg1 / cfg2 select (stages, CTAs per SM) of FC1 / FC2: 0 = (3, 2), 1 = (4, 2); dynamic_schedule (1: atomic work
// counter, 0: static round-robin); grid caps. Every configuration computes the same bits.
std::vector<torch::Tensor> moe_prefill_fast(torch::Tensor hidden, torch::Tensor ids, torch::Tensor token_weights,
                                            torch::Tensor w13, torch::Tensor w13_sf, torch::Tensor w2, torch::Tensor w2_sf,
                                            torch::Tensor alpha, torch::Tensor input_global_scale,
                                            torch::Tensor down_alpha, torch::Tensor global_scale, double limit,
                                            int64_t n, c10::optional<torch::Tensor> route_out_opt,
                                            c10::optional<torch::Tensor> out_opt, bool topk_sum,
                                            int64_t cfg1, int64_t cfg2, int64_t dynamic_schedule,
                                            int64_t grid1_cap, int64_t grid2_cap) {
  using namespace b12x_moe_prefill_fast;
  TORCH_CHECK(hidden.scalar_type() == torch::kBFloat16 && hidden.dim() == 2 && hidden.is_contiguous());
  TORCH_CHECK(ids.scalar_type() == torch::kInt32 && ids.is_contiguous());
  TORCH_CHECK(token_weights.scalar_type() == torch::kFloat32 && token_weights.is_contiguous());
  TORCH_CHECK(n == kN, "intermediate 576 (five K128 slices, n64 tail)");
  const int T = (int)hidden.size(0), K = (int)hidden.size(1), topk = (int)ids.size(1);
  TORCH_CHECK(K % 128 == 0 && K >= 128 && ids.size(0) == T && token_weights.numel() == ids.numel());
  const int E = (int)alpha.numel();
  TORCH_CHECK(E >= 1 && E <= kMaxE);
  TORCH_CHECK(w2.numel() * w2.element_size() == (int64_t)E * K * n / 2, "w2 is not the n64-repacked storage");
  TORCH_CHECK(w13.numel() * w13.element_size() == (int64_t)E * 2 * n * K / 2, "w13 is not the n64-repacked storage");
  TORCH_CHECK(w13_sf.numel() * w13_sf.element_size() == (int64_t)E * 2 * n * K / 32, "w13_sf size");
  TORCH_CHECK(w2_sf.numel() * w2_sf.element_size() == (int64_t)E * (K / 128) * 128 * (kSlices * 4 - 2),
              "w2_sf is not the n64-repacked storage ([E][K/128 tiles][2304 B])");
  TORCH_CHECK(cfg1 >= 0 && cfg1 <= 1 && cfg2 >= 0 && cfg2 <= 1);
  TORCH_CHECK(K >= 640, "K >= 640 (the work-item staging needs 10 K64 stages per item)");
  const int pairs = T * topk;
  auto dev = hidden.device();
  auto o32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  auto o8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto obf = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
  auto a_values = torch::empty({T, K}, o8);
  auto a_scales = torch::empty({T, K / 32}, o8);
  auto inter = torch::empty({pairs, kInterBytes}, o8);
  auto inter_sf = torch::empty({pairs, kSlices * 4}, o8);
  torch::Tensor route, out;
  if (route_out_opt.has_value() && route_out_opt->defined()) {
    route = *route_out_opt;
    TORCH_CHECK(route.scalar_type() == torch::kBFloat16 && route.is_contiguous() && route.numel() >= (int64_t)pairs * K,
                "route_out: bf16, contiguous, >= T * topk * K elements");
  } else {
    route = torch::empty({pairs, K}, obf);
  }
  if (topk_sum) {
    if (out_opt.has_value() && out_opt->defined()) {
      out = *out_opt;
      TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.is_contiguous() && out.numel() == (int64_t)T * K);
    } else {
      out = torch::empty({T, K}, obf);
    }
  } else {
    out = torch::Tensor();
  }
  // routing capacities: every expert may add up to 15 padding rows and one partial chunk
  const int row_cap = pairs + 16 * E;
  const int max_chunks = (pairs + 63) / 64 + E;
  const int n2_tiles = K / 128;
  const int f1_cap = max_chunks * kN1Tiles, f2_cap = max_chunks * n2_tiles;
  auto row_pair = torch::empty({row_cap}, o32);
  auto f1_items = torch::empty({f1_cap, 4}, o32);
  auto f2_items = torch::empty({f2_cap, 4}, o32);
  auto counts = torch::empty({8}, o32);
  auto stream = c10::cuda::getCurrentCUDAStream();

  FrontArgs fa;
  fa.hidden = reinterpret_cast<const uint32_t*>(hidden.data_ptr());
  fa.a_values = reinterpret_cast<uint32_t*>(a_values.data_ptr());
  fa.a_scales = a_scales.data_ptr<uint8_t>();
  fa.ids = ids.data_ptr<int32_t>();
  fa.row_pair = row_pair.data_ptr<int32_t>();
  fa.f1_items = reinterpret_cast<int4*>(f1_items.data_ptr());
  fa.f2_items = reinterpret_cast<int4*>(f2_items.data_ptr());
  fa.counts = counts.data_ptr<int32_t>();
  fa.T = T; fa.K = K; fa.n_pairs = pairs; fa.E = E; fa.n2_tiles = n2_tiles;
  fa.row_cap = row_cap; fa.f1_cap = f1_cap; fa.f2_cap = f2_cap;
  const int quant_blocks = (T + kFrontWarps - 1) / kFrontWarps;
  frontend_kernel<<<quant_blocks + 1, kFrontThreads, 0, stream>>>(fa);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);

  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, hidden.get_device());

  Fc1Args f1;
  f1.a_values = fa.a_values;
  f1.a_scales = fa.a_scales;
  f1.w13 = reinterpret_cast<const uint32_t*>(w13.data_ptr());
  f1.w13_sf = reinterpret_cast<const uint32_t*>(w13_sf.data_ptr());
  f1.row_pair = fa.row_pair;
  f1.items = fa.f1_items;
  f1.counts = fa.counts;
  f1.alpha = alpha.data_ptr<float>();
  f1.input_global_scale = input_global_scale.data_ptr<float>();
  f1.inter = reinterpret_cast<uint32_t*>(inter.data_ptr());
  f1.inter_sf = inter_sf.data_ptr<uint8_t>();
  f1.K = K; f1.topk = topk; f1.igs_per_expert = input_global_scale.numel() > 1;
  f1.dynamic_schedule = (int)dynamic_schedule;
  f1.limit = (float)limit;

  Fc2Args f2;
  f2.inter = f1.inter;
  f2.inter_sf = f1.inter_sf;
  f2.w2 = reinterpret_cast<const uint32_t*>(w2.data_ptr());
  f2.w2_sf = reinterpret_cast<const uint8_t*>(w2_sf.data_ptr());
  f2.row_pair = fa.row_pair;
  f2.items = fa.f2_items;
  f2.counts = fa.counts;
  f2.token_weights = token_weights.data_ptr<float>();
  f2.down_alpha = down_alpha.data_ptr<float>();
  f2.global_scale = global_scale.data_ptr<float>();
  f2.route = reinterpret_cast<uint32_t*>(route.data_ptr());
  f2.K = K; f2.out_tiles = n2_tiles; f2.gs_per_expert = global_scale.numel() > 1;
  f2.dynamic_schedule = (int)dynamic_schedule;

  auto run = [&](auto L) {
    using LL = decltype(L);
    LL::prepare();
    int g1 = LL::grid1(sms), g2 = LL::grid2(sms);
    if (grid1_cap > 0) g1 = std::min<int>(g1, (int)grid1_cap);
    if (grid2_cap > 0) g2 = std::min<int>(g2, (int)grid2_cap);
    g1 = std::min(g1, f1_cap);
    g2 = std::min(g2, f2_cap);
    LL::fc1(f1, g1, stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
    LL::fc2(f2, g2, stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
  };
  const int sel = (int)cfg1 * 2 + (int)cfg2;
  switch (sel) {
    case 0: run(Launch<3, 2, 3, 2>{}); break;
    case 1: run(Launch<3, 2, 4, 2>{}); break;
    case 2: run(Launch<4, 2, 3, 2>{}); break;
    default: run(Launch<4, 2, 4, 2>{}); break;
  }

  if (topk_sum) {
    const long long total = (long long)T * (K / 8);
    topk_sum_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(route.data_ptr()), ids.data_ptr<int32_t>(),
        reinterpret_cast<uint4*>(out.data_ptr()), T, topk, K);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
  }
  return {out, route, a_values, a_scales, inter, inter_sf, row_pair, counts};
}

// the front-end alone (timing): routing_only = 1 launches just the routing block
torch::Tensor moe_prefill_fast_frontend(torch::Tensor hidden, torch::Tensor ids, int64_t E, bool routing_only) {
  using namespace b12x_moe_prefill_fast;
  const int T = (int)hidden.size(0), K = (int)hidden.size(1), topk = (int)ids.size(1), pairs = T * topk;
  auto dev = hidden.device();
  auto o32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  auto o8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto a_values = torch::empty({T, K}, o8);
  auto a_scales = torch::empty({T, K / 32}, o8);
  const int row_cap = pairs + 16 * (int)E, max_chunks = (pairs + 63) / 64 + (int)E, n2_tiles = K / 128;
  const int f1_cap = max_chunks * kN1Tiles, f2_cap = max_chunks * n2_tiles;
  auto row_pair = torch::empty({row_cap}, o32);
  auto f1_items = torch::empty({f1_cap, 4}, o32);
  auto f2_items = torch::empty({f2_cap, 4}, o32);
  auto counts = torch::empty({8}, o32);
  FrontArgs fa;
  fa.hidden = reinterpret_cast<const uint32_t*>(hidden.data_ptr());
  fa.a_values = reinterpret_cast<uint32_t*>(a_values.data_ptr());
  fa.a_scales = a_scales.data_ptr<uint8_t>();
  fa.ids = ids.data_ptr<int32_t>();
  fa.row_pair = row_pair.data_ptr<int32_t>();
  fa.f1_items = reinterpret_cast<int4*>(f1_items.data_ptr());
  fa.f2_items = reinterpret_cast<int4*>(f2_items.data_ptr());
  fa.counts = counts.data_ptr<int32_t>();
  fa.T = T; fa.K = K; fa.n_pairs = pairs; fa.E = (int)E; fa.n2_tiles = n2_tiles;
  fa.row_cap = row_cap; fa.f1_cap = f1_cap; fa.f2_cap = f2_cap;
  const int quant_blocks = routing_only ? 0 : (T + kFrontWarps - 1) / kFrontWarps;
  frontend_kernel<<<quant_blocks + 1, kFrontThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(fa);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
  return counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_prefill_fast", &moe_prefill_fast);
  m.def("moe_prefill_fast_frontend", &moe_prefill_fast_frontend);
}
