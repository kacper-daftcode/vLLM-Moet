// Fast prefill MoE for DeepSeek-V4.1-Flash on sm_120 (front "cubit + prefill", Phase 1a): FC1 + SwiGLU + MXFP8 quant and
// FC2 of b12x's `integration.tp_moe.dynamic` deterministic path, bit for bit with spec B
// (../b12x_moe_prefill_spec/b12x_moe_prefill_spec_sm120.cu = b12x's arithmetic as naive kernels), but pipelined:
//
//   frontend   two launches (v8; v1-v7 routed in one CTA behind the quant blocks): MXFP8 quant of the token rows (a warp
//              per row; exactly spec B's quant_rows) in the first blocks (none with pre-quantized rows), then a block per
//              512-pair chunk with the chunk's per-expert pair counts (shared-memory integer atomics: the sums do not
//              depend on order), of which the last to finish scans: the prefix over the chunks, padded (to 16 rows)
//              expert offsets, the CTA work lists of FC1 / FC2 ((expert, 64-row chunk, column tile), tile-major within an
//              expert so that consecutive items share the expert's weight tile in L2); the scatter kernel (a block per
//              chunk) then places the pairs into row_pair in pair order within an expert (ranks from the chunk prefix +
//              a per-warp prefix + __match_any: no global atomics, no ordering race).
//   fc1        persistent CTAs (256 threads = 8 warps), work item = (expert, <= 4 m16 tiles of its rows, 128-column
//              tile of gate+up; the 5th tile: 64 columns); warps = 8 n16 column slices, every warp over all m-tiles of
//              the item (acc[4][2 proj][2 n8][4] = 64 fp32 per thread; the 64-column tile: 4 slices x 2 m-tile halves).
//              K in K64 stages through a cp.async ring (STAGES deep) holding the k32 blocks of B (b12x's n64-repacked
//              words as laid out in the plane: one LDS.64 per lane per k32 and projection, spread to e2m1 containers
//              once per CTA), the A rows (one LDS.128 per m-tile and k32, see the skeleton header) and the k128 scale
//              words of both; the stage stream is continuous across items (the next item's first stages are in flight
//              during the epilogue). Per output element the QMMA chain is spec B's: accumulate in C over the k32 blocks
//              in order, from +0. Epilogue in registers: alpha * input_global_scale, SwiGLU with limit (min.NaN /
//              max.NaN), ex2 / rcp approx, cvt.rn.satfinite.bf16x2, then per row and 32-column block the amax: over the
//              quad for the warp's 16 columns, then with the other half's partial through shared memory (max is
//              commutative and exact: the same value as the sequential compare-and-select), quantize_block_fp8_mx per
//              element, a transpose of the e4m3 pairs across the quad, two 8 B stores per row and slice, the scale byte.
//   fc2        the same skeleton with K = 640 = 18 live k32 blocks (b12x and spec B also run the 2 zero blocks of the
//              tail: exact +0.0 additions to the chain, skipped since v8 - see the fc2 kernel),
//              item = (expert, <= 4 m-tiles, 128 output columns), acc[4][2][4] = 32 per thread; epilogue:
//              (down_alpha * global_scale) * token weight (staged with the pairs), cvt.rn.satfinite.bf16x2 into the
//              token-major route rows (a word transpose across the quad: two 16 B stores per row and slice).
//   topk_sum   out[t, col] = bf16((((0 + r0) + r1) + ...)) over the token's pairs in slot order (ids < 0 skipped), fp32
//              adds of the bf16 route values - spec B's topk_sum, 8 columns per thread. v8 can also fuse it into fc2's
//              epilogue (VLLM_MOET_B12X_MOE_PREFILL_FAST_FUSED_SUM=1: the CTA that completes the last of a token's rows
//              for a 128-column tile sums the token's rows for that tile; same chain, same order whoever sums) - on
//              call1 break-even with this kernel, so off by default. Integrated (patch v2) the kernels write the output
//              and b12x's TopKSum (0.166 ms) is skipped; with patch v1 the route rows are the interface.
//
// Determinism: no atomics in any reduction (the only atomics are the integer pair counts in shared memory, whose sums
// are order-independent, the work-item counters, which only decide which CTA computes an item, and the fused sum's
// arrival counters, which only decide which CTA sums - in the fixed slot order). Three runs give identical bits. Plane
// layouts and operand conventions are the spec's (see its header).
//
// History (call1 layer of the dumps: 4096 rows x top-6, 319 experts, one RTX PRO 6000 at 400 W, bits always identical):
//   v1 (Q, 29.10) 1.756 ms; v2-v5 (S, 2.11) hang fix, NOMASK removed, fc1 spill removed: 1.746; v6 (S, 2.11) the retile
//   (8 n16 warp slices x all m-tiles): 1.674 = served r23 (fc1 0.87 / fc2 0.56 / topk_sum 0.13 / front-end 0.11).
//   v8 (W, 4.11): the routing front-end in two launches (histogram CTAs + scan in the last one, scatter CTAs; 0.108 ->
//   0.033 ms), fc2 without the zero-tail stage (0.565 -> 0.549), the top-k sum fused into fc2's epilogue, a register
//   diet of the issue side, the pre-quantized rows entry (U). Measured and dropped on the way (W/REPORT.md): a conflict-
//   free B stage layout via 8 B cp.asyncs (+4.5 %), a staggered issue/compute order per warp half (+1-4 %), the fused
//   sum done synchronously in the epilogue (+0.21 ms) or one item later behind the stores (+0.15 ms).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
#include <cstdlib>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace b12x_moe_prefill_fast {

constexpr int kThreads = 256;
constexpr int kN = 576;             // intermediate size: 4 x 128 + 64 columns
constexpr int kN1Tiles = 5;         // FC1 column tiles (gate+up each)
constexpr int kSlices = 5;          // k128 slices of the intermediate (4.5)
constexpr int kInterBytes = 640;    // MXFP8 intermediate row (576 + zero tail)
constexpr int kF2Stages = 9;        // K64 stages of FC2: the 9 live ones (k32 blocks 0-17; the zero tail is not run)
constexpr int kMaxE = 512;          // experts per rank (384 served)
constexpr int kFrontThreads = 512;
constexpr int kFrontWarps = kFrontThreads / 32;

// stage layouts (bytes)
constexpr int kF1B = 8192;    // [proj 2][kk 2][n32 chunk 4][lane 32][16 B] (the plane's layout)
constexpr int kF1A = 4096;    // [m-tile 4][kk 2][quad row 8][c 4][4 words] = the A fragments in register order (see the skeleton header)
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
  const uint32_t* hidden;   // [T][K] bf16 (null with pre-quantized rows)
  uint32_t* a_values;       // [T][K] e4m3
  uint8_t* a_scales;        // [T][K/32] ue8m0
  const int32_t* ids;       // [T * topk]
  int32_t* row_pair;        // [row_cap]
  int4* f1_items;           // [f1_cap] (expert, row0, m_tiles, tile)
  int4* f2_items;           // [f2_cap]
  int32_t* counts;          // [8]: rows_padded, n_f1, n_f2, fc1 work counter, fc2 work counter, histogram arrivals (zeroed by the host)
  int32_t* chunk_cnt;       // [n_chunks][kMaxE]: the 512-pair chunks' per-expert pair counts, after the scan the exclusive prefix over the chunks
  int32_t* exp_off;         // [kMaxE] padded row offset of the expert (the scan CTA -> the scatter kernel)
  int32_t* exp_cnt;         // [kMaxE] pair count of the expert
  uint32_t* out;            // fused top-k sum: the output rows of tokens without a valid pair are zeroed here (nullptr: off)
  int32_t* tok_valid;       // fused top-k sum: [T] bits 0-7 the token's valid pairs (0 <= id < E) = its FC2 arrivals per tile, bits 8-15 the slots with id >= 0
  int T, K, n_pairs, E, n2_tiles, row_cap, f1_cap, f2_cap, n_chunks, quant_blocks, topk;
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

// The routing in two launches (v7; v2-v6 did it all in one CTA that ran 0.081 ms behind the 256 quant blocks' 0.02 ms):
//   frontend_kernel   blocks [0, quant_blocks): MXFP8 quant of the token rows (a warp per row); blocks [quant_blocks, +n_chunks):
//                     the per-expert pair counts of one 512-pair chunk (shared-memory integer atomics -> chunk_cnt[chunk][e]);
//                     the LAST chunk block to finish (arrival counter counts[5], threadfence pattern - the atomic only decides who
//                     does the work) scans: per expert the exclusive prefix over the chunks (in place), the padded (to 16 rows)
//                     expert offsets, the CTA work lists of FC1 / FC2 (tile-major within an expert), the counts.
//   scatter_kernel    a block per chunk: rank of a pair within its expert = pairs of the expert in earlier chunks (the prefix)
//                     + in earlier warps of the chunk (per-warp counts, prefix over the warps) + earlier lanes of the warp
//                     (__match_any + popc) - the same decomposition as the one-CTA scatter, so row_pair is identical; the
//                     padding entries (-1) of the experts are written by the chunk block e % n_chunks.
// No global atomics in anything that decides a value; every pair's row is a pure function of ids.
__global__ void __launch_bounds__(kFrontThreads) frontend_kernel(FrontArgs a) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if ((int)blockIdx.x < a.quant_blocks) {  // token rows quant
    const int row = blockIdx.x * kFrontWarps + warp;
    if (row < a.T) quant_row(a, row, lane);
    return;
  }
  // ------------------------------------------------------------ per-chunk pair counts (a block per 512 pairs)
  __shared__ int s_cnt[kMaxE];
  __shared__ int s_off[kMaxE];    // padded row offset of the expert (scan CTA)
  __shared__ int s_f1[kMaxE];     // first FC1 / FC2 item of the expert, its chunk count (scan CTA)
  __shared__ int s_f2[kMaxE];
  __shared__ int s_chunks[kMaxE];
  __shared__ int s_scan[kFrontWarps];
  __shared__ int s_last;
  const int E = a.E, n_pairs = a.n_pairs, n_chunks = a.n_chunks;
  const int chunk = (int)blockIdx.x - a.quant_blocks;
  for (int e = tid; e < kMaxE; e += kFrontThreads) s_cnt[e] = 0;
  __syncthreads();
  const uint32_t lanemask_lt = (1u << lane) - 1u;
  {  // integer counts: the sums do not depend on the order; every lane of every warp reaches the __match_any_sync
    const int p = chunk * kFrontThreads + tid;
    int e = p < n_pairs ? a.ids[p] : -1;
    if (!(e >= 0 && e < E)) e = -1;
    const uint32_t mask = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && (mask & lanemask_lt) == 0u) atomicAdd(&s_cnt[e], __popc(mask));  // one atomic per (warp, expert)
  }
  __syncthreads();
  if (tid < E) a.chunk_cnt[(size_t)chunk * kMaxE + tid] = s_cnt[tid];
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(a.counts + 5, 1) == n_chunks - 1;
  __syncthreads();
  if (!s_last) return;
  // ------------------------------------------------------------ the scan (the last chunk block): deterministic routing
  __threadfence();
  int cnt = 0;
  if (tid < E) {  // exclusive prefix over the chunks, in place (the scatter kernel reads it), total -> cnt
    int32_t* col = a.chunk_cnt + tid;
    for (int j = 0; j < n_chunks; ++j) {
      const int v = __ldcg(col + (size_t)j * kMaxE);
      col[(size_t)j * kMaxE] = cnt;
      cnt += v;
    }
    s_cnt[tid] = cnt;
  }
  int padded = 0, chunks = 0;
  if (tid < E) {
    padded = (cnt + 15) & ~15;
    chunks = (padded / 16 + 3) >> 2;
  }
  int rows_padded, n_f1, n_f2;
  const int off = block_exclusive_scan(padded, s_scan, rows_padded);
  const int f1 = block_exclusive_scan(chunks * kN1Tiles, s_scan, n_f1);
  const int f2 = block_exclusive_scan(chunks * a.n2_tiles, s_scan, n_f2);
  if (tid < E) { s_off[tid] = off; s_f1[tid] = f1; s_f2[tid] = f2; s_chunks[tid] = chunks; a.exp_off[tid] = off; a.exp_cnt[tid] = cnt; }
  if (tid == 0) {
    a.counts[0] = rows_padded; a.counts[1] = min(n_f1, a.f1_cap); a.counts[2] = min(n_f2, a.f2_cap);
    a.counts[3] = 0; a.counts[4] = 0;  // the work counters of the persistent kernels (also zeroed by the host's memset)
  }
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
}

// the stable scatter: a block per 512-pair chunk (after frontend_kernel: chunk_cnt holds the exclusive prefix over the chunks)
__global__ void __launch_bounds__(kFrontThreads) scatter_kernel(FrontArgs a) {
  __shared__ int s_w[kFrontWarps][kMaxE];  // per-warp per-expert counts of the chunk -> exclusive prefix over the warps
  __shared__ int s_off[kMaxE];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int E = a.E, n_pairs = a.n_pairs, chunk = blockIdx.x;
  const uint32_t lanemask_lt = (1u << lane) - 1u;
  const int p = chunk * kFrontThreads + tid;
  int e = p < n_pairs ? a.ids[p] : -1;
  const bool valid = e >= 0 && e < E;
  if (!valid) e = -1;
  const uint32_t mask = __match_any_sync(0xffffffffu, e);
  const int rank = __popc(mask & lanemask_lt);
  const bool leader = (mask & lanemask_lt) == 0u;
  for (int i = lane; i < E; i += 32) s_w[warp][i] = 0;
  __syncwarp();
  if (valid && leader) s_w[warp][e] = __popc(mask);
  if (tid < E) s_off[tid] = a.exp_off[tid];
  __syncthreads();
  if (tid < E) {  // exclusive prefix over the warps for this expert, on top of the expert's pairs in the earlier chunks
    int run = a.chunk_cnt[(size_t)chunk * kMaxE + tid];
#pragma unroll
    for (int w = 0; w < kFrontWarps; ++w) {
      const int v = s_w[w][tid];
      s_w[w][tid] = run;
      run += v;
    }
    // the expert's padding rows (-1): by the chunk block e % n_chunks
    if (tid % a.n_chunks == chunk) {
      const int c = a.exp_cnt[tid], o = s_off[tid];
      for (int r = o + c; r < o + ((c + 15) & ~15); ++r)
        if (r < a.row_cap) a.row_pair[r] = -1;
    }
  }
  __syncthreads();
  if (valid) {
    const int r = s_off[e] + s_w[warp][e] + rank;
    if (r < a.row_cap) a.row_pair[r] = p;
  }
  // fused top-k sum: the thread of a token's slot 0 counts the token's valid pairs (= its arrivals per output tile in
  // FC2); a token without one gets no arrival - its output row is the sum of nothing = zeros (topk_sum_kernel's acc = +0
  // -> bf16 0), written here (rare: synthetic -1 rows)
  if (a.out != nullptr && p < n_pairs && p % a.topk == 0) {
    int nvalid = 0, mask = 0;  // tok_valid word: bits 0-7 the valid-pair count (arrivals), bits 8-15 the slots with id >= 0 (summed)
    for (int sl = 0; sl < a.topk; ++sl) {
      const int id = a.ids[p + sl];
      nvalid += (id >= 0 && id < E) ? 1 : 0;
      mask |= (id >= 0 ? 1 : 0) << sl;
    }
    a.tok_valid[p / a.topk] = nvalid | (mask << 8);
    if (nvalid == 0) {
      uint4* o = reinterpret_cast<uint4*>(a.out + (size_t)(p / a.topk) * (a.K / 2));
      for (int j = 0; j < a.K / 8; ++j) o[j] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
}

// ------------------------------------------------------------------------------------- 2. the FC1 / FC2 skeleton
// Both GEMMs run the same persistent, continuously pipelined loop: a CTA takes work items (expert, <= 4 m16 tiles of
// its rows, one column tile); for every item it streams K64 stages through a STAGES-deep cp.async ring; the stage
// stream does not stop at item boundaries (the next item's first stages are in flight during the epilogue).
//
// Warp tiling (v6, "retile"): the 8 warps are 8 n16 column slices of the tile and every warp computes ALL m-tiles of the
// item (FC1: acc[m-tile 4][proj 2][n8 2][4] = 64 fp32, FC2: acc[4][2][4] = 32). Every B word is therefore spread by one
// warp (the n32 x 2-row-group tiling of v1-v5 spread each word twice) and 1-, 2- and 3-tile items keep all 8 warps busy
// (v5 idled half of them on 1- and 2-tile items). FC1's 64-column tile (tile 4): n16 slice = warp & 3, m-tiles
// 2 (warp >> 2), +1. The per-element arithmetic and the per-accumulator QMMA chain (k32 blocks in order, acc in C from
// +0) are unchanged; only which warp owns an output element differs.
//
// Stage layout (bytes): B [kk 2][n32 chunk NS][lane 32][16 B] per projection (FC1: 2 projections) = the plane's layout
// (an n16 slice reads 8 B per lane out of the 16 B: LDS.64 at a 16 B stride = 4 wavefronts instead of 2 - measured and
// kept: the conflict-free alternative, the two n16 halves de-interleaved by 8 B cp.asyncs, cost more in LDGSTS than it
// saved in LDS: fc1 0.866 -> 0.907 ms, fc2 0.561 -> 0.595 on call1, W 4.11), then A as [m-tile 4][kk 2][quad row 8][c 4][4 words] where
// the 4 words of a (m-tile, kk, q, c) group are (row q: k32 word 2c, row q + 8: word 2c, row q: word 2c + 1, row q + 8:
// word 2c + 1) = the QMMA A fragment of lane (q, c) in register order: one LDS.128 per m-tile and k32, no shuffling. Then
// the k128 scale words of the 64 rows and of the columns ([proj][128] words; FC2's tail slice: 128 x u16).
// A is copied with 4 B cp.async: warp w takes k32 block kk = w / 4 of the rows p, p + 8, p + 1, p + 9 (p = 2 (w % 4)),
// lane = (row of the four, word 0..7 of the k32 block) - the 32 words of one instruction land in 32 distinct banks.
// The rows' k128 scale words are copied by warps 0-1 (thread = row). Rows of padding and of the tiles >= m_tiles have
// the offset 0xFFFFFFFF in ring.aoff and are zero-filled (cp.async src-size 0).
//
// Work-item hand-off: item_0 = blockIdx.x; item_{k+1} = gridDim.x + atomicAdd(counter) (dynamic) or item_k + gridDim.x
// (static). Thread 0 claims it 10 stages before the issue side needs it (ring.items[(k+1) & 1]); the item's metadata
// (four ints, threads 0-3) and its 64 row -> pair entries (threads 0-63) are staged through ring.meta / ring.pairs /
// ring.aoff (FC2: + ring.tw, the token weights, by cp.async) (slot (k+1) & 1), every global load consumed two stages
// after it was issued (one register per thread in between), every shared hand-off with a barrier in between. The
// compute side reads the same slots: meta at the start of item k + 1, items at the end of item k, pairs / tw in the
// epilogue (slot k & 1, rewritten only while item k + 1 is being issued, long after). The compute side's item index k
// is not kept: the issue side is exactly STAGES - 1 stages ahead (one issue per computed stage, STAGES - 1 in the
// prologue, STAGES - 1 < n_stages), so iss_k == k at the start of item k and iss_k == k + 1 after its K-loop (iss_k is
// bumped when the last stage of an item is issued, also when no item follows).
struct ItemRing {
  int items[2];
  int4 meta[2];          // (expert, row0, m_tiles, tile); m_tiles == 0: no item
  int pairs[2][64];      // pair of the 64 rows (-1: padding)
  uint32_t aoff[2][64];  // A row byte offset of the 64 rows (0xFFFFFFFF: padding)
  float tw[2][64];       // FC2: token weight of the 64 rows (0 for padding)
};

__device__ __forceinline__ int claim_next_item(int32_t* counter, int cur, int dynamic_schedule) {
  return dynamic_schedule ? (int)gridDim.x + atomicAdd(counter, 1) : cur + (int)gridDim.x;
}

// the staged hand-off of the next work item (see above): d = the stage being issued minus claim_at, nb = the next item's
// slot (the current issue item is ring.items[nb ^ 1]); the three global loads share one staging register (`nxt`: their
// live ranges - stages 0-2, 3-5, 6-8 - do not overlap; v7 register diet); on_pair(nb, tid, pair) runs for threads 0-63
// in the step that publishes the pairs (FC2 stages the token weight there)
template <typename AoffFn, typename PairFn>
__device__ __forceinline__ void stage_next_item(ItemRing& ring, int d, int nb, int tid, int n_items, const int4* items,
                                                const int32_t* row_pair, int32_t* counter, int dynamic_schedule, int& nxt,
                                                AoffFn aoff_of, PairFn on_pair) {
  if (d == 0 && tid == 0) nxt = claim_next_item(counter, ring.items[nb ^ 1], dynamic_schedule);
  if (d == 2 && tid == 0) ring.items[nb] = nxt;
  if (d == 3 && tid < 4) {
    const int nx = ring.items[nb];
    nxt = nx < n_items ? __ldg(reinterpret_cast<const int32_t*>(items) + (size_t)nx * 4 + tid) : 0;
  }
  if (d == 5 && tid < 4) reinterpret_cast<int*>(&ring.meta[nb])[tid] = nxt;
  if (d == 6 && tid < 64) {
    const int4 m = ring.meta[nb];
    nxt = tid < m.z * 16 ? __ldg(row_pair + m.y + tid) : -1;
  }
  if (d == 8 && tid < 64) {
    ring.pairs[nb][tid] = nxt;
    ring.aoff[nb][tid] = nxt >= 0 ? aoff_of(nxt) : 0xFFFFFFFFu;
    on_pair(nb, tid, nxt);
  }
}

struct Fc1Args {
  const uint32_t* a_values;  // [T][K/4] words; row r at byte r * lda (lda >= K, a multiple of 4; K when contiguous)
  const uint8_t* a_scales;   // [T][K/32] bytes; row r at byte r * lds (lds >= K/32, a multiple of 4; K/32 when contiguous)
  int lda, lds;              // row strides in bytes of a_values / a_scales
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

// STRIDED: a_values / a_scales are row-strided views (lda / lds; the rows' scale offsets staged in s_aoff_sf); the contiguous
// instantiation (the served path) derives the scale offset from the A offset as before - the same code as v8, bit for bit and
// cycle for cycle (K 5.11: the strided code path costs fc1 +0.5 % on call1, so the served path does not carry it)
template <int STAGES, int MINB, bool STRIDED>
__global__ void __launch_bounds__(kThreads, MINB) fc1_kernel(Fc1Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ ItemRing ring;
  // STRIDED: byte offset of the rows' k32 scale bytes (row * lds; 0xFFFFFFFF: padding), staged like ring.aoff; the A rows and their
  // scales may come as row-strided views of one gathered buffer (the one-all-gather layout). Contiguous: 4 bytes, unused.
  __shared__ uint32_t s_aoff_sf[STRIDED ? 2 : 1][STRIDED ? 64 : 1];
  __shared__ float s_amax[2][8][16];  // epilogue: a warp's partial amax (its 16 columns) of the 16 rows of an m-tile, by m-tile parity
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
  const auto aoff_of = [&](int pair) { return (uint32_t)(pair / a.topk) * (uint32_t)(STRIDED ? a.lda : K); };
  // STRIDED: the row's scale offset is staged next to its A offset (the same step of the hand-off, threads 0-63)
  const auto sf_pair_hook = [&](int nb, int t, int pair) {
    if constexpr (STRIDED) s_aoff_sf[nb][t] = pair >= 0 ? (uint32_t)(pair / a.topk) * (uint32_t)a.lds : 0xFFFFFFFFu;
  };

  // ---- issue side (its item is ring.items[iss_k & 1]; the slot of the item being issued is iss_k & 1)
  int iss_k = 0, iss_stage = 0, iss_mtiles = 0, nxt = 0;
  bool iss_wide = true;
  const uint8_t* iss_b = reinterpret_cast<const uint8_t*>(a.w13);
  const uint8_t* iss_sfb = reinterpret_cast<const uint8_t*>(a.w13_sf);

  auto start_issue_item = [&](int slot) {
    const int4 it = ring.meta[slot];
    iss_stage = 0;
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
      const int t = iss_stage, slot = iss_k & 1, nb = slot ^ 1;
      uint32_t aoff4[4];  // the rows' A offsets first: their LDS latency overlaps the B address work
#pragma unroll
      for (int j = 0; j < 4; ++j) aoff4[j] = ring.aoff[slot][j * 16 + my_r16];
      // warps 0-1: the scale word of row tid (STRIDED: its own offset; contiguous: derived from the A offset, row * K / 32)
      const uint32_t aoff_sf = warp < 2 ? (STRIDED ? s_aoff_sf[slot][tid] : ring.aoff[slot][tid]) : 0xFFFFFFFFu;
      stage_next_item(ring, t - claim_at, nb, tid, n_items, a.items, a.row_pair, a.counts + 3, a.dynamic_schedule, nxt,
                      aoff_of, sf_pair_hook);
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
      // A: one word per thread and m-tile, zero-filled for padding rows and for the tiles >= m_tiles (offset 0xFFFFFFFF)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t off = aoff4[j];
        const bool on = off != 0xFFFFFFFFu;
        const uint32_t o = on ? off : 0u;
        cp_async4_zfill(sb + kF1B + a_dst0 + j * 1024, reinterpret_cast<const uint8_t*>(a.a_values) + o + t * 64 + a_src0, on ? 4 : 0);
      }
      if (warp < 2) {  // the rows' k128 scale words: row tid (its scale bytes at row * lds; k128 block t >> 1 = 4 bytes)
        const bool on = aoff_sf != 0xFFFFFFFFu;
        const uint32_t o = on ? aoff_sf : 0u;
        cp_async4_zfill(sb + kF1B + kF1A + tid * 4, a.a_scales + (STRIDED ? o : (o >> 5)) + (t >> 1) * 4, on ? 4 : 0);
      }
      if (++iss_stage == n_stages) {
        ++iss_k;
        start_issue_item(nb);
      }
    }
    cp_async_commit();
  };

  // ---- the first item: metadata, pairs and offsets straight from global memory
  int item = blockIdx.x;
  if (tid < 4) reinterpret_cast<int*>(&ring.meta[0])[tid] = item < n_items ? __ldg(reinterpret_cast<const int32_t*>(a.items) + (size_t)item * 4 + tid) : 0;
  if (tid == 0) ring.items[0] = item;  // the issue side's current item (claim_next_item's static-schedule base)
  __syncthreads();
  if (tid < 64) {
    const int4 m = ring.meta[0];
    const int pr = tid < m.z * 16 ? __ldg(a.row_pair + m.y + tid) : -1;
    ring.pairs[0][tid] = pr;
    ring.aoff[0][tid] = pr >= 0 ? aoff_of(pr) : 0xFFFFFFFFu;
    sf_pair_hook(0, tid, pr);
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
    const bool wide = tile != kN1Tiles - 1;
    // warp tiling: 128-column tiles: n16 slice = warp, all m-tiles; the 64-column tile: slice = warp & 3, m-tiles 2 (warp >> 2), +1
    const int ns16 = wide ? warp : (warp & 3);
    const int mt_lo = wide ? 0 : 2 * (warp >> 2), mt_hi = wide ? 4 : mt_lo + 2;
    bool act[4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) act[mt] = mt >= mt_lo && mt < mt_hi && mt < m_tiles;  // warp-uniform
    const float alpha = fmul(__ldg(a.alpha + e), __ldg(a.input_global_scale + (a.igs_per_expert ? e : 0)));

    float acc[4][2][2][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int r = 0; r < 4; ++r) acc[mt][p][nt][r] = 0.f;

    const uint32_t a_off = kF1B + q * 64 + c * 16;                        // + mt * 1024 + kk * 512
    const uint32_t sfa_off = kF1B + kF1A + (q + ((lane & 1) << 3)) * 4;    // + mt * 64
    const uint32_t b_off = (ns16 >> 1) * 512 + lane * 16 + (ns16 & 1) * 8; // + p * 4096 + kk * 2048 (8 B of the lane's 16 B)
    const uint32_t sfb_off = kF1B + kF1A + kF1SFA + (ns16 * 16 + q) * 4;   // + p * 512 + nt * 32

    auto compute_stage = [&](const uint8_t* sp, auto kb_tag) {
      constexpr int KB0 = decltype(kb_tag)::value;
      uint32_t sfb[2][2], sfa[4];
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) sfb[p][nt] = *reinterpret_cast<const uint32_t*>(sp + sfb_off + p * 512 + nt * 32);
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) sfa[mt] = *reinterpret_cast<const uint32_t*>(sp + sfa_off + mt * 64);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        // the warp's n16 slice of B (both projections) once, then the A fragment of each m-tile (acc[mt][p][nt] still sees
        // the k32 blocks in order)
        uint32_t b[2][2][2];
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const uint2 w = *reinterpret_cast<const uint2*>(sp + p * 4096 + kk * 2048 + b_off);
          spread_e2m1x8(w.x, b[p][0][0], b[p][0][1]);
          spread_e2m1x8(w.y, b[p][1][0], b[p][1][1]);
        }
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          if (act[mt]) {
            const uint4 af = *reinterpret_cast<const uint4*>(sp + a_off + mt * 1024 + kk * 512);
#pragma unroll
            for (int p = 0; p < 2; ++p)
#pragma unroll
              for (int nt = 0; nt < 2; ++nt)
                qmma_kk<KB0>(kk, acc[mt][p][nt], af.x, af.y, af.z, af.w, b[p][nt][0], b[p][nt][1], sfa[mt], sfb[p][nt]);
          }
        }
      }
    };

    // one K64 stage: wait for its data, barrier (everyone has left the buffer the issue side refills), issue stage
    // t + STAGES - 1, compute stage t. (Letting warps 4-7 compute before they issue - alternating QMMA phases per SM
    // sub-partition - was measured and dropped: fc1 0.907 -> 0.965 ms, fc2 0.595 -> 0.9 ms on call1, W 4.11.)
    auto step = [&](auto kb_tag) {
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      issue_stage(buf_issue);
      compute_stage(smem + buf * kF1Stage, kb_tag);
      buf_issue = buf_issue + 1 == STAGES ? 0 : buf_issue + 1;
      buf = buf + 1 == STAGES ? 0 : buf + 1;
    };
#pragma unroll 1
    for (int t = 0; t < n_stages; t += 2) {
      step(std::integral_constant<int, 0>{});
      step(std::integral_constant<int, 2>{});
    }

    // ---- epilogue: activation, bf16, MXFP8 per row and 32-column block. A warp holds 16 of the block's 32 columns: its
    // partial amax (4 values per lane, quad max) meets the partner warp's (warp ^ 1, the other n16 half of the same rows)
    // through shared memory; max is exact and commutative, so the block amax, the scale byte and the e4m3 bytes are the
    // same as with one warp per block. One barrier per m-tile (the m-tile loop is CTA-uniform: m_tiles is the item's).
    const int pslot = (iss_k + 1) & 1;  // slot k & 1 (iss_k == k + 1 here)
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {  // fully unrolled: acc[mt] stays in registers; `mt < m_tiles` is CTA-uniform
      if (mt >= m_tiles) continue;
      float v[2][4], am[2] = {0.f, 0.f};
      if (act[mt]) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            // projection 0 of the plane = up, 1 = gate
            const float x0 = activated_value(acc[mt][1][nt][2 * hf], acc[mt][0][nt][2 * hf], alpha, a.limit);
            const float x1 = activated_value(acc[mt][1][nt][2 * hf + 1], acc[mt][0][nt][2 * hf + 1], alpha, a.limit);
            const uint32_t pk = pack_bf16x2(x0, x1);
            v[hf][2 * nt] = __uint_as_float(pk << 16);
            v[hf][2 * nt + 1] = __uint_as_float(pk & 0xFFFF0000u);
          }
          float amax = 0.f;
#pragma unroll
          for (int kx = 0; kx < 4; ++kx) amax = fmax_plain(amax, fabs_plain(v[hf][kx]));
          amax = fmax_plain(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
          amax = fmax_plain(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
          am[hf] = amax;
          if (c == 0) s_amax[mt & 1][warp][q + hf * 8] = amax;
        }
      }
      __syncthreads();
      if (act[mt]) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int row = mt * 16 + q + hf * 8;
          const int pair = ring.pairs[pslot][row];
          const float amax = fmax_plain(am[hf], s_amax[mt & 1][warp ^ 1][q + hf * 8]);
          float inv;
          const uint32_t byte = mx_scale(amax, inv);
          uint32_t h[4] = {cvt_e4m3x2(fmul(v[hf][0], inv), fmul(v[hf][1], inv)),
                           cvt_e4m3x2(fmul(v[hf][2], inv), fmul(v[hf][3], inv)), 0u, 0u};
          uint32_t o0, o1;
          quad_transpose16(h, c, o0, o1);  // lanes 0 / 1 now hold the 8 bytes of columns 0-7 / 8-15 of the slice
          if (pair >= 0) {
            if (c < 2) {
              uint2* dst = reinterpret_cast<uint2*>(reinterpret_cast<uint8_t*>(a.inter) + (size_t)pair * kInterBytes +
                                                   tile * 128 + ns16 * 16 + c * 8);
              *dst = make_uint2(o0, o1);
            }
            if ((ns16 & 1) == 0) {  // the block's scale byte (and the tail slice's zero bytes) by the even half
              uint8_t* sf = a.inter_sf + (size_t)pair * (kSlices * 4) + tile * 4;
              if (c == 0) sf[ns16 >> 1] = (uint8_t)byte;
              if (!wide && c == 1) sf[(ns16 >> 1) + 2] = 0;
            }
          }
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
  // the fused top-k sum (v8; out == nullptr: off, the route rows are the interface): out[t][col] = bf16 of the fp32 sum of
  // the token's route values in slot order (ids < 0 skipped) = topk_sum_kernel, done by the CTA whose item completes the
  // last of the token's rows for this 128-column tile (arrival counter per (token, tile), zeroed by the host)
  uint32_t* out;               // [T][K] bf16 or nullptr
  int32_t* arrive;             // [T * out_tiles]
  const int32_t* ids;          // [T * topk]
  const int32_t* tok_valid;    // [T] valid pairs of the token | slot mask << 8 (the scatter kernel; topk <= 8)
  int K, out_tiles, gs_per_expert, dynamic_schedule, topk;
};

template <int STAGES, int MINB>
__global__ void __launch_bounds__(kThreads, MINB) fc2_kernel(Fc2Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ ItemRing ring;
  __shared__ int s_done[2][64];  // fused top-k sum: the lists - token + 1 of the rows whose (token, tile) slice item j completed, slot j & 1
  __shared__ int s_mask[2][64];  // ... the token's slot mask (slots with id >= 0: the ones summed)
  __shared__ int s_ltile[2];     // ... and that item's tile
  __shared__ int s_tile[4];      // the tile of item j, slot j & 3 (read two items later)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int q = lane >> 2, c = lane & 3;
  // The zero tail (k32 blocks 18-19: zero A, zero B, zero scales - b12x and spec B run them as the 10th K64 stage) is
  // not computed (v8): its QMMAs add exact +0.0 to every accumulator, and acc + (+0.0) == acc bit for bit for every acc
  // but -0.0, which the chain never holds (it starts at +0.0 and a round-to-nearest sum of zeros / cancelling terms is
  // +0.0; a nonzero sum would have to underflow below 2^-149 to round to a signed zero - impossible for scales that
  // are not 2^-127). Harness: identical on the dumps and the synthetic set (W, 4.11). 9 stages: claim at stage 0, the
  // pairs published at stage 8 (the hand-off needs 9 stages per item).
  constexpr int n_stages = kF2Stages;
  constexpr int claim_at = n_stages - 9;
  constexpr uint32_t kTileBytes = 128 * (kSlices * 4 - 2) * 16;  // 36 KB of weights per 128-column tile (18 k32)
  constexpr uint32_t kTileSfBytes = 128 * (kSlices * 4 - 2);     // 2304 B of scales per tile
  const int n_items = a.counts[2];
  const uint32_t sbase0 = smem_addr(smem);
  const int kk_c = warp >> 2, sub = (lane >> 3), kw8 = lane & 7;
  const int my_r16 = 2 * (warp & 3) + (sub >> 1) + 8 * (sub & 1);
  const uint32_t a_dst0 = (uint32_t)((kk_c * 8 + 2 * (warp & 3) + (sub >> 1)) * 64 + (kw8 >> 1) * 16 + ((kw8 & 1) * 2 + (sub & 1)) * 4);
  const uint32_t a_src0 = kk_c * 32 + kw8 * 4;
  const auto aoff_of = [](int pair) { return (uint32_t)pair * (uint32_t)kInterBytes; };
  // the token weight of the row goes into ring.tw with the pairs (cp.async, zero for padding rows)
  const auto tw_hook = [&](int nb, int row, int pair) {
    cp_async4_zfill(smem_addr(&ring.tw[nb][row]), a.token_weights + (pair >= 0 ? pair : 0), pair >= 0 ? 4 : 0);
  };

  int iss_k = 0, iss_stage = 0, iss_mtiles = 0, nxt = 0;
  const uint8_t* iss_b = reinterpret_cast<const uint8_t*>(a.w2);
  const uint8_t* iss_sf = a.w2_sf;

  auto start_issue_item = [&](int slot) {
    const int4 it = ring.meta[slot];
    iss_stage = 0;
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
      const int t = iss_stage, slot = iss_k & 1, nb = slot ^ 1;
      uint32_t aoff4[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) aoff4[j] = ring.aoff[slot][j * 16 + my_r16];
      const uint32_t aoff_sf = warp < 2 ? ring.aoff[slot][tid] : 0xFFFFFFFFu;
      stage_next_item(ring, t - claim_at, nb, tid, n_items, a.items, a.row_pair, a.counts + 4, a.dynamic_schedule, nxt,
                      aoff_of, tw_hook);
      {
        const uint32_t o = (tid >> 7) * 2048 + (tid & 127) * 16;
        cp_async16(sb + o, iss_b + (size_t)t * 4096 + o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t off = aoff4[j];
          const bool on = off != 0xFFFFFFFFu;
          const uint32_t oo = on ? off : 0u;
          cp_async4_zfill(sb + kF2B + a_dst0 + j * 1024, reinterpret_cast<const uint8_t*>(a.inter) + oo + t * 64 + a_src0, on ? 4 : 0);
        }
        if (warp < 2) {
          const bool on = aoff_sf != 0xFFFFFFFFu;
          const uint32_t oo = on ? aoff_sf : 0u;
          cp_async4_zfill(sb + kF2B + kF2A + tid * 4, a.inter_sf + (oo >> 5) + (t >> 1) * 4, on ? 4 : 0);
        }
        const int s = t >> 1;
        if (tid < (s == kSlices - 1 ? 16 : 32)) cp_async16(sb + kF2B + kF2A + kF2SFA + tid * 16, iss_sf + s * 512 + tid * 16);
      }
      if (++iss_stage == n_stages) {
        ++iss_k;
        start_issue_item(nb);
      }
    }
    cp_async_commit();
  };

  int item = blockIdx.x;
  if (tid < 4) reinterpret_cast<int*>(&ring.meta[0])[tid] = item < n_items ? __ldg(reinterpret_cast<const int32_t*>(a.items) + (size_t)item * 4 + tid) : 0;
  if (tid == 0) ring.items[0] = item;
  if (tid < 128) (&s_done[0][0])[tid] = 0;  // fused sum: empty lists
  // fused sum, per row thread (lanes 0-7 of every warp, row = warp * 8 + lane): the token / valid-pair count of the row of the latest item (fs_tok1, fs_nv1),
  // of the item before it (fs_tok2, fs_nv2) and the arrival-counter result of that item (fs_old); "not done" initially
  int fs_old = -2, fs_nv1 = 0, fs_tok1 = 0, fs_nv2 = 0, fs_tok2 = 0;
  __syncthreads();
  if (tid < 64) {
    const int4 m = ring.meta[0];
    const int pr = tid < m.z * 16 ? __ldg(a.row_pair + m.y + tid) : -1;
    ring.pairs[0][tid] = pr;
    ring.aoff[0][tid] = pr >= 0 ? aoff_of(pr) : 0xFFFFFFFFu;
    ring.tw[0][tid] = pr >= 0 ? __ldg(a.token_weights + pr) : 0.f;
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
  // fused top-k sum: sum the list in slot `ls` (s_done[ls][row] = token + 1, tile s_ltile[ls]) - a half-warp per row
  auto fused_sum_list = [&](int ls) {
    const int hw = tid >> 4, hl = tid & 15;
    const int ltile = s_ltile[ls];
#pragma unroll 1
    for (int row = hw; row < 64; row += 16) {
      const int tk = s_done[ls][row] - 1;
      if (tk < 0) continue;  // half-warp-uniform
      const int mask = s_mask[ls][row];
      float sum[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) sum[j] = 0.f;
      const int col = ltile * 128 + hl * 8;
      uint4 v[8];  // the (up to 8) slots' 16 B issued together (one L2 round trip; v8g loaded them one behind the other)
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        v[sl] = make_uint4(0u, 0u, 0u, 0u);
        if (sl < a.topk && ((mask >> sl) & 1))
          v[sl] = __ldcg(reinterpret_cast<const uint4*>(a.route + ((size_t)(tk * a.topk + sl) * a.K + col) / 2));
      }
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        if (sl < a.topk && ((mask >> sl) & 1)) {  // skipped slots as topk_sum_kernel skips them
          const uint32_t ws[4] = {v[sl].x, v[sl].y, v[sl].z, v[sl].w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            sum[2 * j] = fadd(sum[2 * j], bf16_bits_to_f32(ws[j] & 0xFFFFu));
            sum[2 * j + 1] = fadd(sum[2 * j + 1], bf16_bits_to_f32(ws[j] >> 16));
          }
        }
      }
      uint32_t o[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        unsigned short lo, hi;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(lo) : "f"(sum[2 * j]));
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(hi) : "f"(sum[2 * j + 1]));
        o[j] = (uint32_t)lo | ((uint32_t)hi << 16);
      }
      *reinterpret_cast<uint4*>(a.out + ((size_t)tk * a.K + col) / 2) = make_uint4(o[0], o[1], o[2], o[3]);
    }
  };

  int buf = 0;
  while (item < n_items) {
    const int4 it = ring.meta[iss_k & 1];  // slot k & 1
    const int e = it.x, m_tiles = it.z, tile = it.w;
    const int ns16 = warp;  // n16 slice of the 128 output columns; every warp takes all m-tiles
    const float down_scale = fmul(__ldg(a.down_alpha + e), __ldg(a.global_scale + (a.gs_per_expert ? e : 0)));

    float acc[4][2][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[mt][nt][r] = 0.f;

    const uint32_t a_off = kF2B + q * 64 + c * 16;                        // + mt * 1024 + kk * 512
    const uint32_t sfa_off = kF2B + kF2A + (q + ((lane & 1) << 3)) * 4;    // + mt * 64
    const uint32_t b_off = (ns16 >> 1) * 512 + lane * 16 + (ns16 & 1) * 8; // + kk * 2048
    const int col_q = ns16 * 16 + q;

    // TAIL: slice 4 (u16 scale words: the n64 tail of the intermediate has k32 blocks 16-17 only)
    auto compute_stage = [&](const uint8_t* sp, auto kb_tag, auto tail_tag) {
      constexpr int KB0 = decltype(kb_tag)::value;
      constexpr bool TAIL = decltype(tail_tag)::value;
      uint32_t sfb[2], sfa[4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        if constexpr (TAIL) sfb[nt] = *reinterpret_cast<const uint16_t*>(sp + kF2B + kF2A + kF2SFA + (col_q + nt * 8) * 2);
        else sfb[nt] = *reinterpret_cast<const uint32_t*>(sp + kF2B + kF2A + kF2SFA + (col_q + nt * 8) * 4);
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) sfa[mt] = *reinterpret_cast<const uint32_t*>(sp + sfa_off + mt * 64);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t b[2][2];
        const uint2 w = *reinterpret_cast<const uint2*>(sp + kk * 2048 + b_off);
        spread_e2m1x8(w.x, b[0][0], b[0][1]);
        spread_e2m1x8(w.y, b[1][0], b[1][1]);
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          if (mt < m_tiles) {
            const uint4 af = *reinterpret_cast<const uint4*>(sp + a_off + mt * 1024 + kk * 512);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
              qmma_kk<KB0>(kk, acc[mt][nt], af.x, af.y, af.z, af.w, b[nt][0], b[nt][1], sfa[mt], sfb[nt]);
          }
        }
      }
    };
    using I0 = std::integral_constant<int, 0>;
    using I2 = std::integral_constant<int, 2>;
    using F = std::false_type;
    using Tr = std::true_type;
    auto step = [&](auto kb_tag, auto tail_tag) {
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      issue_stage(buf_issue);
      compute_stage(smem + buf * kF2Stage, kb_tag, tail_tag);
      buf_issue = buf_issue + 1 == STAGES ? 0 : buf_issue + 1;
      buf = buf + 1 == STAGES ? 0 : buf + 1;
    };
#pragma unroll 1
    for (int t = 0; t < n_stages - 1; t += 2) {  // slices 0-3: stages 0..7
      step(I0{}, F{});
      step(I2{}, F{});
    }
    step(I0{}, Tr{});  // stage 8: k32 16, 17 (the tail slice's u16 scale words); the zero tail (k32 18, 19) is not run

    // ---- epilogue: scale, bf16, token-major route rows (the slice's 16 columns = 32 B per row: lanes 0 / 1 store 16 B)
    const int pslot = (iss_k + 1) & 1;  // slot k & 1 (iss_k == k + 1 here)
    // ---- the fused top-k sum (out != nullptr), pipelined over four items so that nothing here waits on memory and no
    // fence ever sits behind this item's own stores (v8c did fence + arrival right after the stores: +0.21 ms; v8d one
    // item later but still after the stores: +0.15 - the fence drains the 16 KB of route rows just issued). At item k,
    // BEFORE its route-row stores:
    //   (i)   item k - 2: its arrival results (registers, issued one item ago) -> the list s_done[k & 1] (token + 1 per
    //         row whose (token, tile) slice it completed = the last of the token's valid pairs) with the token's slot
    //         mask, its tile -> s_ltile; the token's rows for that tile are prefetched into L2;
    //   (ii)  item k - 1: the arrivals - one thread per row (8 per warp): one fence.acq_rel.gpu per item (release side for item k - 1's
    //         stores, 9+ barriers old; acquire side for the sums of later epilogues w.r.t. the counters read before it -
    //         a __threadfence costs ~1 us even when nothing is in flight: two per item were +0.06 ms), then a relaxed
    //         atomicAdd on the (token, tile) counter, result kept in fs_old: no wait;
    //   (iii) the route-row stores of item k (below);
    //   (iv)  item k - 3: its list (prefetched one item ago) is summed - a half-warp per row, 8 columns per lane, fp32
    //         adds in slot order from +0 (ids < 0 skipped) = topk_sum_kernel's chain, bf16 out;
    //   (v)   item k: its tile -> s_tile[k & 3], its rows' token and valid-pair count -> fs_tok1 / fs_nv1.
    // The atomic only decides who sums; the order is the slot order whoever sums: deterministic, identical to the separate
    // kernel. After the last item the three pending items are flushed (below the loop). No extra barrier: every hand-off
    // through shared memory or registers is separated by the K-loop's barriers of the items in between.
    if (a.out != nullptr) {
      const int k3 = (iss_k - 1) & 3;  // item k & 3
      const int row8 = warp * 8 + lane;  // the row threads: lanes 0-7 of every warp (8 rows per warp - the work spread over all
                                         // warps, since whatever one warp does extra here the whole CTA waits for at the next barrier)
      if (lane < 8) {
        {  // (i) item k - 2 -> list slot k & 1 (the acquire side is the fence of (ii) below: it follows the atomic that
           // produced fs_old and precedes the loads of the sum, two epilogues later)
          const int done = fs_old == (fs_nv2 & 0xFF) - 1;
          s_done[pslot][row8] = done ? fs_tok2 + 1 : 0;
          s_mask[pslot][row8] = fs_nv2 >> 8;
          const int ptile = s_tile[(k3 + 2) & 3];
          if (tid == 0) s_ltile[pslot] = ptile;
          if (done) {
#pragma unroll 1
            for (int sl = 0; sl < a.topk; ++sl) {
              const uint8_t* rowp = reinterpret_cast<const uint8_t*>(a.route) + ((size_t)(fs_tok2 * a.topk + sl) * a.K + ptile * 128) * 2;
              asm volatile("prefetch.global.L2 [%0];" ::"l"(rowp));
              asm volatile("prefetch.global.L2 [%0];" ::"l"(rowp + 128));
            }
          }
        }
        {  // (ii) item k - 1: the arrivals
          fs_old = -2;
          asm volatile("fence.acq_rel.gpu;" ::: "memory");  // one fence per item: release for item k - 1's stores, acquire for the sums
          if ((fs_nv1 & 0xFF) > 0) fs_old = atomicAdd(a.arrive + (size_t)fs_tok1 * a.out_tiles + s_tile[(k3 + 3) & 3], 1);
          fs_tok2 = fs_tok1;
          fs_nv2 = fs_nv1;
        }
      }
      if (tid == 0) s_tile[k3] = tile;  // (v)
      if (lane < 8) {
        const int pair = row8 < m_tiles * 16 ? ring.pairs[pslot][row8] : -1;
        fs_nv1 = 0;
        if (pair >= 0) {
          fs_tok1 = pair / a.topk;
          fs_nv1 = __ldg(a.tok_valid + fs_tok1);  // count >= 1: this pair is one of them
        }
      }
    }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      if (mt >= m_tiles) continue;  // CTA-uniform
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int row = mt * 16 + q + hf * 8;
        const int pair = ring.pairs[pslot][row];
        const float scale = fmul(down_scale, ring.tw[pslot][row]);
        uint32_t w[4], o[4];
        w[0] = pack_bf16x2(fmul(scale, acc[mt][0][2 * hf]), fmul(scale, acc[mt][0][2 * hf + 1]));
        w[1] = pack_bf16x2(fmul(scale, acc[mt][1][2 * hf]), fmul(scale, acc[mt][1][2 * hf + 1]));
        w[2] = w[3] = 0u;
        quad_transpose32(w, c, o);
        if (pair >= 0 && c < 2) {
          uint4* dst = reinterpret_cast<uint4*>(a.route + (size_t)pair * (a.K / 2) + (tile * 128 + ns16 * 16 + c * 8) / 2);
          *dst = make_uint4(o[0], o[1], o[2], o[3]);
        }
      }
    }
    if (a.out != nullptr) fused_sum_list(pslot ^ 1);  // (iv) item k - 3 (slot (k - 3) & 1), after the stores: the accumulators are dead
    item = ring.items[iss_k & 1];  // slot (k + 1) & 1
  }
  cp_async_wait<0>();
  if (a.out != nullptr) {  // fused sum: flush items L - 2 (list ready), L - 1 (arrivals issued) and L (not yet) - L = the last item
    const int Ls = (iss_k - 1) & 1, L3 = (iss_k - 1) & 3;  // iss_k == L + 1
    __syncthreads();     // warps 0-1 may still be writing item L - 2's list (step (i) of the last epilogue); inside the loop
                         // the next item's K-loop barriers separate the two, here nothing else does
    fused_sum_list(Ls);  // item L - 2: slot L & 1 (prefetched at the epilogue of item L)
    __syncthreads();
    const int row8 = warp * 8 + lane;
    if (lane < 8) {  // item L - 1 -> slot (L - 1) & 1 (its previous list, item L - 3, was summed at item L's epilogue)
      __threadfence();
      s_done[Ls ^ 1][row8] = fs_old == (fs_nv2 & 0xFF) - 1 ? fs_tok2 + 1 : 0;
      s_mask[Ls ^ 1][row8] = fs_nv2 >> 8;
      if (tid == 0) s_ltile[Ls ^ 1] = s_tile[(L3 + 3) & 3];
      fs_old = -2;  // item L: its arrivals now (its stores are this CTA's last; the release fence waits for them once)
      if ((fs_nv1 & 0xFF) > 0) {
        __threadfence();
        fs_old = atomicAdd(a.arrive + (size_t)fs_tok1 * a.out_tiles + s_tile[L3], 1);
      }
    }
    __syncthreads();
    fused_sum_list(Ls ^ 1);  // item L - 1
    __syncthreads();
    if (lane < 8) {  // item L -> slot L & 1 (item L - 2's list is summed)
      __threadfence();
      s_done[Ls][row8] = fs_old == (fs_nv1 & 0xFF) - 1 ? fs_tok1 + 1 : 0;
      s_mask[Ls][row8] = fs_nv1 >> 8;
      if (tid == 0) s_ltile[Ls] = s_tile[L3];
    }
    __syncthreads();
    fused_sum_list(Ls);  // item L
  }
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
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fc1_kernel<S1, MB1, false>, kThreads, S1 * kF1Stage);
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
    cudaFuncSetAttribute(fc1_kernel<S1, MB1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, S1 * kF1Stage);
    cudaFuncSetAttribute(fc1_kernel<S1, MB1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, S1 * kF1Stage);
    cudaFuncSetAttribute(fc2_kernel<S2, MB2>, cudaFuncAttributeMaxDynamicSharedMemorySize, S2 * kF2Stage);
    done = true;
  }
  static void fc1(const b12x_moe_prefill_fast::Fc1Args& f1, int grid, cudaStream_t stream) {
    using namespace b12x_moe_prefill_fast;
    if (f1.lda != f1.K || f1.lds != f1.K / 32)  // row-strided views of one gathered buffer (the one-all-gather layout)
      fc1_kernel<S1, MB1, true><<<grid, kThreads, S1 * kF1Stage, stream>>>(f1);
    else
      fc1_kernel<S1, MB1, false><<<grid, kThreads, S1 * kF1Stage, stream>>>(f1);
  }
  static void fc2(const b12x_moe_prefill_fast::Fc2Args& f2, int grid, cudaStream_t stream) {
    using namespace b12x_moe_prefill_fast;
    fc2_kernel<S2, MB2><<<grid, kThreads, S2 * kF2Stage, stream>>>(f2);
  }
};

// the front-end launches shared by moe_prefill_fast and moe_prefill_fast_frontend: a memset of the counters, the quant +
// histogram (+ scan) kernel, the scatter kernel. `routing` = the workspace tensors kept alive by the caller.
struct Routing {
  torch::Tensor row_pair, f1_items, f2_items, counts, chunk_cnt, exp_off, exp_cnt, tok_valid;
};
static bool fused_topk_sum_enabled() {  // VLLM_MOET_B12X_MOE_PREFILL_FAST_FUSED_SUM=1: the sum fused into fc2's epilogue (off by
  // default: on call1 it is break-even with the separate topk_sum_kernel - fc2 +0.125 ms for the kernel's 0.130 - at more
  // moving parts; W/REPORT.md §4)
  static const bool on = [] { const char* v = std::getenv("VLLM_MOET_B12X_MOE_PREFILL_FAST_FUSED_SUM"); return v && v[0] == '1'; }();
  return on;
}
static Routing launch_frontend(b12x_moe_prefill_fast::FrontArgs& fa, bool quant, int T, int K, int topk, int E, torch::Device dev) {
  using namespace b12x_moe_prefill_fast;
  auto o32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  const int pairs = T * topk;
  // routing capacities: every expert may add up to 15 padding rows and one partial chunk
  const int row_cap = pairs + 16 * E;
  const int max_chunks = (pairs + 63) / 64 + E;
  const int n2_tiles = K / 128;
  const int f1_cap = max_chunks * kN1Tiles, f2_cap = max_chunks * n2_tiles;
  const int n_chunks = std::max(1, (pairs + kFrontThreads - 1) / kFrontThreads);
  Routing r;
  r.row_pair = torch::empty({row_cap}, o32);
  r.f1_items = torch::empty({f1_cap, 4}, o32);
  r.f2_items = torch::empty({f2_cap, 4}, o32);
  r.counts = torch::empty({8}, o32);
  r.chunk_cnt = torch::empty({n_chunks, kMaxE}, o32);
  r.exp_off = torch::empty({kMaxE}, o32);
  r.exp_cnt = torch::empty({kMaxE}, o32);
  fa.row_pair = r.row_pair.data_ptr<int32_t>();
  fa.f1_items = reinterpret_cast<int4*>(r.f1_items.data_ptr());
  fa.f2_items = reinterpret_cast<int4*>(r.f2_items.data_ptr());
  fa.counts = r.counts.data_ptr<int32_t>();
  fa.chunk_cnt = r.chunk_cnt.data_ptr<int32_t>();
  fa.exp_off = r.exp_off.data_ptr<int32_t>();
  fa.exp_cnt = r.exp_cnt.data_ptr<int32_t>();
  fa.T = T; fa.K = K; fa.n_pairs = pairs; fa.E = E; fa.n2_tiles = n2_tiles;
  fa.row_cap = row_cap; fa.f1_cap = f1_cap; fa.f2_cap = f2_cap; fa.n_chunks = n_chunks;
  fa.quant_blocks = quant ? (T + kFrontWarps - 1) / kFrontWarps : 0;
  fa.topk = topk;
  fa.out = nullptr;
  fa.tok_valid = nullptr;
  return r;
}
static void run_frontend(const b12x_moe_prefill_fast::FrontArgs& fa, cudaStream_t stream) {
  using namespace b12x_moe_prefill_fast;
  cudaMemsetAsync(fa.counts, 0, 8 * sizeof(int32_t), stream);  // the histogram arrival counter and the work counters
  frontend_kernel<<<fa.quant_blocks + fa.n_chunks, kFrontThreads, 0, stream>>>(fa);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
  scatter_kernel<<<fa.n_chunks, kFrontThreads, 0, stream>>>(fa);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

}  // namespace

// hidden [T, K] bf16 (or None with pre-quantized rows: a_values uint8 [T, K] e4m3 + a_scales uint8 [T, K/32] ue8m0, row-major
// or row-strided views of one buffer (strides a multiple of 4 bytes), exactly the bytes the front-end's quant_row writes;
// the quant stage is then skipped - the MXFP8 all-gather),
// ids [T, topk] int32, token_weights [T, topk] fp32 (contiguous). Returns {out, route, a_values, a_scales, inter, inter_sf,
// row_pair, counts}. `route_out` (if defined: [>= T*topk, K] bf16, the b12x workspace's route_output) receives the route
// rows; `out` (if defined) the top-k sum (skipped when !topk_sum).
// Tuning: cfg1 / cfg2 select (stages, CTAs per SM) of FC1 / FC2: 0 = (3, 2), 1 = (4, 2); dynamic_schedule (1: atomic work
// counter, 0: static round-robin); grid caps. Every configuration computes the same bits.
std::vector<torch::Tensor> moe_prefill_fast(c10::optional<torch::Tensor> hidden_opt, torch::Tensor ids, torch::Tensor token_weights,
                                            torch::Tensor w13, torch::Tensor w13_sf, torch::Tensor w2, torch::Tensor w2_sf,
                                            torch::Tensor alpha, torch::Tensor input_global_scale,
                                            torch::Tensor down_alpha, torch::Tensor global_scale, double limit,
                                            int64_t n, c10::optional<torch::Tensor> route_out_opt,
                                            c10::optional<torch::Tensor> out_opt, bool topk_sum,
                                            int64_t cfg1, int64_t cfg2, int64_t dynamic_schedule,
                                            int64_t grid1_cap, int64_t grid2_cap,
                                            c10::optional<torch::Tensor> a_values_opt, c10::optional<torch::Tensor> a_scales_opt) {
  using namespace b12x_moe_prefill_fast;
  const bool prequant = a_values_opt.has_value() && a_values_opt->defined();
  torch::Tensor hidden, a_values, a_scales;
  int T, K;
  if (prequant) {
    a_values = *a_values_opt;
    TORCH_CHECK(a_scales_opt.has_value() && a_scales_opt->defined(), "a_scales must come with a_values");
    a_scales = *a_scales_opt;
    // row-strided views are accepted (one gathered [T, K + K/32 + ...] buffer: values = buf[:, :K], scales = buf[:, K:K + K/32]):
    // unit column stride, row strides a multiple of 4 bytes (4-byte cp.async), 4-byte aligned bases
    TORCH_CHECK(a_values.scalar_type() == torch::kUInt8 && a_values.dim() == 2 && a_values.stride(1) == 1, "a_values: uint8 [T, K], unit column stride");
    T = (int)a_values.size(0); K = (int)a_values.size(1);
    TORCH_CHECK(a_values.stride(0) >= K && a_values.stride(0) % 4 == 0 && (int64_t)T * a_values.stride(0) < (1LL << 32) &&
                (reinterpret_cast<uintptr_t>(a_values.data_ptr()) & 3) == 0,
                "a_values: row stride >= K, a multiple of 4, 4-byte aligned base, T * stride < 2^32 (the ring holds 32-bit row offsets)");
    TORCH_CHECK(a_scales.scalar_type() == torch::kUInt8 && a_scales.dim() == 2 && a_scales.stride(1) == 1 && a_scales.size(0) == T &&
                a_scales.size(1) == K / 32, "a_scales: uint8 [T, K/32], unit column stride");
    TORCH_CHECK(a_scales.stride(0) >= K / 32 && a_scales.stride(0) % 4 == 0 && (reinterpret_cast<uintptr_t>(a_scales.data_ptr()) & 3) == 0,
                "a_scales: row stride >= K/32, a multiple of 4, 4-byte aligned base");
  } else {
    TORCH_CHECK(hidden_opt.has_value() && hidden_opt->defined(), "hidden (bf16 rows) or a_values + a_scales (pre-quantized rows)");
    hidden = *hidden_opt;
    TORCH_CHECK(hidden.scalar_type() == torch::kBFloat16 && hidden.dim() == 2 && hidden.is_contiguous());
    T = (int)hidden.size(0); K = (int)hidden.size(1);
  }
  TORCH_CHECK(ids.scalar_type() == torch::kInt32 && ids.is_contiguous());
  TORCH_CHECK(token_weights.scalar_type() == torch::kFloat32 && token_weights.is_contiguous());
  TORCH_CHECK(n == kN, "intermediate 576 (five K128 slices, n64 tail)");
  const int topk = (int)ids.size(1);
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
  auto dev = ids.device();
  auto o8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto obf = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
  if (!prequant) {
    a_values = torch::empty({T, K}, o8);
    a_scales = torch::empty({T, K / 32}, o8);
  }
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
  auto stream = c10::cuda::getCurrentCUDAStream();

  const bool fused_sum = topk_sum && topk <= 8 && fused_topk_sum_enabled();  // the slot mask of tok_valid holds 8 slots
  torch::Tensor arrive;
  FrontArgs fa;
  fa.hidden = prequant ? nullptr : reinterpret_cast<const uint32_t*>(hidden.data_ptr());
  fa.a_values = reinterpret_cast<uint32_t*>(a_values.data_ptr());
  fa.a_scales = a_scales.data_ptr<uint8_t>();
  Routing rt = launch_frontend(fa, !prequant, T, K, topk, E, dev);
  fa.ids = ids.data_ptr<int32_t>();
  if (fused_sum) {
    fa.out = reinterpret_cast<uint32_t*>(out.data_ptr());
    rt.tok_valid = torch::empty({T}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
    fa.tok_valid = rt.tok_valid.data_ptr<int32_t>();
    arrive = torch::empty({(int64_t)T * fa.n2_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
    cudaMemsetAsync(arrive.data_ptr(), 0, (size_t)T * fa.n2_tiles * sizeof(int32_t), stream);
  }
  run_frontend(fa, stream);

  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ids.get_device());

  Fc1Args f1;
  f1.a_values = fa.a_values;
  f1.a_scales = fa.a_scales;
  f1.lda = prequant ? (int)a_values.stride(0) : K;        // bytes per row of the e4m3 values
  f1.lds = prequant ? (int)a_scales.stride(0) : K / 32;   // bytes per row of the ue8m0 scales
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
  f2.out = fused_sum ? reinterpret_cast<uint32_t*>(out.data_ptr()) : nullptr;
  f2.arrive = fused_sum ? arrive.data_ptr<int32_t>() : nullptr;
  f2.ids = ids.data_ptr<int32_t>();
  f2.tok_valid = fa.tok_valid;
  f2.K = K; f2.out_tiles = fa.n2_tiles; f2.gs_per_expert = global_scale.numel() > 1;
  f2.dynamic_schedule = (int)dynamic_schedule; f2.topk = topk;

  auto run = [&](auto L) {
    using LL = decltype(L);
    LL::prepare();
    int g1 = LL::grid1(sms), g2 = LL::grid2(sms);
    if (grid1_cap > 0) g1 = std::min<int>(g1, (int)grid1_cap);
    if (grid2_cap > 0) g2 = std::min<int>(g2, (int)grid2_cap);
    g1 = std::min(g1, fa.f1_cap);
    g2 = std::min(g2, fa.f2_cap);
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

  if (topk_sum && !fused_sum) {
    const long long total = (long long)T * (K / 8);
    topk_sum_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(route.data_ptr()), ids.data_ptr<int32_t>(),
        reinterpret_cast<uint4*>(out.data_ptr()), T, topk, K);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
  }
  return {out, route, a_values, a_scales, inter, inter_sf, rt.row_pair, rt.counts};
}

// the front-end alone (timing): routing_only = 1 launches just the routing (histogram / scan / scatter) blocks
torch::Tensor moe_prefill_fast_frontend(torch::Tensor hidden, torch::Tensor ids, int64_t E, bool routing_only) {
  using namespace b12x_moe_prefill_fast;
  const int T = (int)hidden.size(0), K = (int)hidden.size(1), topk = (int)ids.size(1);
  auto dev = hidden.device();
  auto o8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto a_values = torch::empty({T, K}, o8);
  auto a_scales = torch::empty({T, K / 32}, o8);
  FrontArgs fa;
  fa.hidden = reinterpret_cast<const uint32_t*>(hidden.data_ptr());
  fa.a_values = reinterpret_cast<uint32_t*>(a_values.data_ptr());
  fa.a_scales = a_scales.data_ptr<uint8_t>();
  Routing rt = launch_frontend(fa, !routing_only, T, K, topk, (int)E, dev);
  fa.ids = ids.data_ptr<int32_t>();
  run_frontend(fa, c10::cuda::getCurrentCUDAStream());
  return rt.counts;
}

bool prequant_rows_supported() { return true; }  // the loader probes this entry (pre-quantized rows)
// the loader probes this entry too: a_values / a_scales may be row-strided views (unit column stride, 4-byte row strides and
// bases) of one gathered buffer - the one-all-gather layout
bool strided_rows_supported() { return true; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_prefill_fast", &moe_prefill_fast);
  m.def("moe_prefill_fast_frontend", &moe_prefill_fast_frontend);
  m.def("prequant_rows_supported", &prequant_rows_supported);
  m.def("strided_rows_supported", &strided_rows_supported);
}
