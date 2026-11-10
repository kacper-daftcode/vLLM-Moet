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
//              adds of the bf16 route values - spec B's topk_sum, 8 columns per thread; with the v9 FC2 kernel the same
//              chain runs as work items of FC2's list (VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_ITEMS=1, below). The v8
//              variant that summed in FC2's epilogue (..._FUSED_SUM=1) is removed: the sum items replace it, and the
//              extension refuses to run while that variable is still set. Integrated (patch v2) the kernels write the
//              output and b12x's TopKSum (0.166 ms) is skipped; with patch v1 the route rows are the interface.
//
// Determinism: no atomics in any reduction (the only atomics are the integer pair counts in shared memory, whose sums
// are order-independent, and the work-item counters, which only decide which CTA computes an item). Three runs give identical bits. Plane
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
#include <cstring>
#include <string>
#include <tuple>
#include <vector>
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
// MK-2 (section 6): the unified work list's FC2 marker (bit 8 of the tile field; fc1 tiles 0-4, fc2 tiles 0-39) and the persistent kernel's
// consumer warp count (= kWgConsumers, asserted there): fc1_need[e] = kPsConsumerWarps x the expert's FC1 items
constexpr int kPsFc2 = 0x100;
constexpr int kWgConsumersFront = 8;
// K-12 (VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_ITEMS=1, fc2_kernel_ws): the FC2 list in column-tile-major order with the top-k sum as work items
// (bit 9 of the tile field; the tile is bits 0-7): the sum items of tile n follow the FC2 items of tile n + lag and wait for tile_done[n] ==
// 8 x the FC2 items per tile (the scan CTA's count) - the route rows of a tile are summed while they are still in L2 (design: K/k12/DESIGN_route_rows.md)
constexpr int kSumItem = 0x200;
constexpr int kMaxTiles = 64;                           // tile_done counters: K / 128 <= 64
constexpr int kSumConsumerWarps = 8;                    // fc2_kernel_ws's consumer warps = the arrivals per FC2 item
constexpr int kSumMaxTopk = 8;                          // the sum item's unrolled slot loads (topk <= 8; the host falls back above)
// the counts buffer (one memset per call): [0..7] the routing counts / work counters / err lock, then fc1_done[kMaxE] (MK-2), then
// tile_done[kMaxTiles] and the sum items' wait target + spare words (K-12)
constexpr int kCntFc1Done = 8;
constexpr int kCntTileDone = kCntFc1Done + kMaxE;
constexpr int kCntSumNeed = kCntTileDone + kMaxTiles;
constexpr int kCountsWords = kCntSumNeed + 8;

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
// K-12: the same 16-B copy with an L2 eviction-priority hint (W2 streams once per call: evict_first keeps the live route rows resident)
__device__ __forceinline__ void cp_async16_hint(uint32_t dst, const void* src, uint64_t policy) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "l"(policy) : "memory");
}
__device__ __forceinline__ uint64_t l2_policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t l2_policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(p));
  return p;
}
// K-12: a 16-B store with an L2 eviction-priority hint (the route rows of the sum-items mode: evict_last until the sum item discards them)
__device__ __forceinline__ void st_global_hint_v4(uint4* p, const uint4& v, uint64_t policy) {
  asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(policy) : "memory");
}
// K-12: drop a 128-B line from L2 without writing it back (PTX discard, sm_80+): the summed route rows are scratch
__device__ __forceinline__ void discard_l2_line(const void* p) {
  asm volatile("discard.global.L2 [%0], 128;\n" ::"l"(p) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

// gpu-scope release / acquire counters, the global timer and the SM id (MK-2's persistent kernel, K-12's sum items)
__device__ __forceinline__ void red_release_gpu_add(int32_t* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int ld_acquire_gpu_s32(const int32_t* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long globaltimer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned smid() {
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;\n" : "=r"(s));
  return s;
}
// an expired bounded wait (MK-2's discipline): the record (first writer wins the device lock) goes to the pinned host buffer, then the CTA
// traps - the launch fails loudly and the GPU is left clean (never a spinning kernel). err: [set, what, x, need, seen, smid, blockIdx, watchdog_ms]
__device__ __noinline__ void fail_record(uint32_t* err, int32_t* err_lock, unsigned long long watchdog_ns, int what, int x, int need, int seen) {
  if (err_lock != nullptr && atomicCAS(err_lock, 0, 1) == 0 && err != nullptr) {
    err[1] = (uint32_t)what; err[2] = (uint32_t)x; err[3] = (uint32_t)need; err[4] = (uint32_t)seen;
    err[5] = smid(); err[6] = blockIdx.x; err[7] = (uint32_t)(watchdog_ns / 1000000ull);
    __threadfence_system();
    err[0] = 1u;
    __threadfence_system();
  }
  __trap();
}
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
  // MK-2 (WS >= 20): the unified work list of the persistent launch (FC1 items of expert e, then the FC2 items of expert e - lead; nullptr: the
  // two lists above), its capacity, the lead in experts, and fc1_need[e] = the FC2 items' wait target (8 x the expert's FC1 items)
  int4* u_items;
  int32_t* fc1_need;
  int u_cap, lead;
  int u_f2_group;           // 128-column tiles per FC2 item of the unified list (1 = as the two lists, 2 = n256 items: the tile field = the first tile)
  // K-12 (SUM_ITEMS, fc2_kernel_ws): sum_items = 1 -> the FC2 list in column-tile-major order (tile n: every expert's chunks, expert-major)
  // with ceil(T / sum_tokens) sum items (token0, ntok, 1, n | kSumItem) after the FC2 items of tile n + sum_lag (the last lag tiles' after the
  // last FC2 item); counts[kCntSumNeed] = kSumConsumerWarps x the FC2 items per tile (the sum items' wait target)
  int sum_items, sum_tokens, sum_lag;
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
  // K-12: the sum items of the column-tile-major FC2 list (sum_items): C = the FC2 items per tile (= the total 64-row chunks), S = the sum
  // items per tile; the list = per tile n its C FC2 items, with the S sum items of tile n - lag after the FC2 items of tile n
  const int sum_C = a.n2_tiles > 0 ? n_f2 / a.n2_tiles : 0;
  const int sum_S = a.sum_items ? (a.T + a.sum_tokens - 1) / a.sum_tokens : 0;
  if (tid == 0) {
    a.counts[0] = rows_padded; a.counts[1] = min(n_f1, a.f1_cap); a.counts[2] = min(n_f2 + a.n2_tiles * sum_S, a.f2_cap);
    a.counts[3] = 0; a.counts[4] = 0;  // the work counters of the persistent kernels (also zeroed by the host's memset)
    if (a.sum_items) a.counts[kCntSumNeed] = kSumConsumerWarps * sum_C;
  }
  __syncthreads();
  // MK-2: the unified work list of the persistent launch (moe_persistent_kernel): block order fc1(0) .. fc1(L-1), fc1(L) fc2(0), fc1(L+1)
  // fc2(1), ..., fc1(E-1) fc2(E-1-L), fc2(E-L) .. fc2(E-1) (L = the lead in experts; items tile-major within a block as in the two lists);
  // s_u1[e] / s_u2[e] = the start of the expert's FC1 / FC2 block; fc1_need[e] = 8 x the expert's FC1 items (the consumer warps' arrivals)
  __shared__ int s_u1[kMaxE];
  __shared__ int s_u2[kMaxE];
  if (a.u_items != nullptr) {
    const int L = a.lead, n2 = a.n2_tiles / a.u_f2_group;  // FC2 items per chunk of the unified list
    int ua = 0, ub = 0;
    if (tid < E) {
      ua = chunks * kN1Tiles + (tid >= L ? s_chunks[tid - L] * n2 : 0);
      ub = tid >= E - L ? chunks * n2 : 0;  // the FC2 blocks of the last L experts follow the interleaved region
    }
    int tot_a, tot_b;
    const int oa = block_exclusive_scan(ua, s_scan, tot_a);
    const int ob = block_exclusive_scan(ub, s_scan, tot_b);
    if (tid < E) { s_u1[tid] = oa; s_u2[tid] = tot_a + ob; a.fc1_need[tid] = chunks * kN1Tiles * kWgConsumersFront; }
    if (tid == 0) a.counts[6] = min(tot_a + tot_b, a.u_cap);
    __syncthreads();
    if (tid < E && tid + L < E) s_u2[tid] = s_u1[tid + L] + s_chunks[tid + L] * kN1Tiles;
    __syncthreads();
  }
  // the work items, tile-major within the expert (consecutive items share the weight tile): a thread per (expert, tile)
  for (int et = tid; et < E * (kN1Tiles + a.n2_tiles); et += kFrontThreads) {
    const bool fc2 = et >= E * kN1Tiles;
    const int e = fc2 ? (et - E * kN1Tiles) / a.n2_tiles : et / kN1Tiles;
    const int t = fc2 ? (et - E * kN1Tiles) - e * a.n2_tiles : et - e * kN1Tiles;
    const int chunks_e = s_chunks[e], m_tiles = chunks_e == 0 ? 0 : ((s_cnt[e] + 15) >> 4);
    if (a.u_items != nullptr) {
      if (fc2 && (t % a.u_f2_group) != 0) continue;  // n256 items: one item per group of tiles, the tile field = its first tile
      const int base = fc2 ? s_u2[e] : s_u1[e], ti = fc2 ? t / a.u_f2_group : t;
      for (int ch = 0; ch < chunks_e; ++ch) {
        const int i = base + ti * chunks_e + ch;
        if (i < a.u_cap) a.u_items[i] = make_int4(e, s_off[e] + ch * 64, min(4, m_tiles - ch * 4), fc2 ? (t | kPsFc2) : t);
      }
      continue;
    }
    if (fc2 && a.sum_items) {
      // K-12: column-tile-major - tile t's FC2 items start after the FC2 items of tiles < t and the sum items of tiles < t - lag; within the
      // tile the experts in order (the expert's first chunk at its chunk prefix s_f2[e] / n2_tiles), the chunks of (e, t) consecutive
      const int base = t * sum_C + max(0, t - a.sum_lag) * sum_S + s_f2[e] / a.n2_tiles;
      for (int ch = 0; ch < chunks_e; ++ch) {
        const int i = base + ch;
        if (i < a.f2_cap) a.f2_items[i] = make_int4(e, s_off[e] + ch * 64, min(4, m_tiles - ch * 4), t);
      }
      continue;
    }
    const int base = fc2 ? s_f2[e] : s_f1[e];
    for (int ch = 0; ch < chunks_e; ++ch) {
      const int i = base + t * chunks_e + ch;
      const int4 v = make_int4(e, s_off[e] + ch * 64, min(4, m_tiles - ch * 4), t);
      if (fc2) { if (i < a.f2_cap) a.f2_items[i] = v; }
      else if (i < a.f1_cap) a.f1_items[i] = v;
    }
  }
  // K-12: the sum items - tile n's S items after the FC2 items of tile min(n + lag, n2_tiles - 1) and the sum items of tiles < n
  if (a.sum_items) {
    for (int sj = tid; sj < a.n2_tiles * sum_S; sj += kFrontThreads) {
      const int n = sj / sum_S, j = sj - n * sum_S;
      const int i = min(n + a.sum_lag + 1, a.n2_tiles) * sum_C + n * sum_S + j;
      const int token0 = j * a.sum_tokens;
      if (i < a.f2_cap) a.f2_items[i] = make_int4(token0, min(a.sum_tokens, a.T - token0), 1, n | kSumItem);
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
  int rows;                  // T (the host's kernel selection only)
};

// STRIDED: a_values / a_scales are row-strided views (lda / lds; the rows' scale offsets staged in s_aoff_sf); the contiguous
// instantiation (the served path) derives the scale offset from the A offset as before - the same code as v8, bit for bit and
// cycle for cycle (K 5.11: the strided code path costs fc1 +0.5 % on call1, so the served path does not carry it)
// MK-4 (megakernel stage 3, the hybrid): -DMOET_FC1_MAXNREG=<n> caps v8's registers at n instead of launch_bounds' 65536 / (256 x MINB)
// = 128, so that a 32-thread x 32-register comm CTA (1 024 registers) co-resides with the two fc1 CTAs of an SM (2 x 256 x 120 + 1 024
// <= 65 536); the arithmetic is untouched (bits by construction, the harness confirms), the cost is measured on W's harness.
#ifdef MOET_FC1_MAXNREG
#define MOET_FC1_BOUNDS __maxnreg__(MOET_FC1_MAXNREG)
#else
#define MOET_FC1_BOUNDS __launch_bounds__(kThreads, MINB)
#endif
template <int STAGES, int MINB, bool STRIDED>
__global__ void MOET_FC1_BOUNDS fc1_kernel(Fc1Args a) {
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
  // the top-k sum inside FC2 (the sum items of fc2_kernel_ws, below; out == nullptr: off, the route rows are the interface):
  // out[t][col] = bf16 of the fp32 sum of the token's route values in slot order (ids < 0 skipped) = topk_sum_kernel
  uint32_t* out;               // [T][K] bf16 or nullptr
  const int32_t* ids;          // [T * topk]
  int K, out_tiles, gs_per_expert, dynamic_schedule, topk;
  // K-12 (SUM_ITEMS, fc2_kernel_ws only; sum_mode 0 = off): the top-k sum as items of the column-tile-major list. tile_done[n] += 1 per
  // consumer warp per FC2 item of tile n (red.release.gpu after the warp's route stores); a sum item's warps wait for tile_done[n] ==
  // *sum_need (= 8 x the FC2 items per tile, the scan CTA) with an acquire poll bounded by the watchdog (an expired wait: fail_record +
  // trap), then sum the token's route rows of that tile in slot order into `out` (topk_sum_kernel's chain); sum_discard = 1: the summed
  // route lines are dropped from L2 (discard.global.L2; the route buffer is scratch then); w2_evict_first = 1: W2's copies carry an
  // L2::evict_first hint
  int32_t* tile_done;
  const int32_t* sum_need;
  int32_t* err_lock;           // one device word (zeroed per call)
  uint32_t* err;               // pinned host memory [8] (the persistent kernel's record)
  unsigned long long watchdog_ns;
  int sum_mode, sum_discard, w2_evict_first;
  int sum_route_hint;          // 1: the route rows are stored with an L2::evict_last hint (sum_mode only; they are discarded after the sum)
};

// K-12: a sum item of fc2_kernel_ws, run by the 8 consumer warps (256 threads): tokens token0 .. token0 + ntok - 1 of column tile n;
// 16 threads x 16 B (8 bf16 columns) per token, 16 tokens per pass. Per element exactly topk_sum_kernel's chain: acc = +0, then
// add.f32 of the bf16 route value of slot 0, 1, ... (ids < 0 skipped), cvt.rn.bf16. The route rows were published by the FC2 items of
// tile n (release) - lane 0 of every warp acquires tile_done[n] >= need first. After a token's sum (every lane's loads returned: the
// chain consumed them, then __syncwarp) the lanes at the two 128-B line starts discard the six rows' lines of this tile (sum_discard).
__device__ __forceinline__ void fc2_sum_item(const Fc2Args& a, int token0, int ntok, int n, int lane, int tid, int item) {
  if (lane == 0) {
    const int need = __ldg(a.sum_need);
    int seen = ld_acquire_gpu_s32(a.tile_done + n);
    if (seen < need) {
      const unsigned long long t0 = globaltimer_ns();
      while ((seen = ld_acquire_gpu_s32(a.tile_done + n)) < need) {
        if (globaltimer_ns() - t0 > a.watchdog_ns) fail_record(a.err, a.err_lock, a.watchdog_ns, 100 + n, item, need, seen);
        __nanosleep(64);
      }
    }
  }
  __syncwarp();
  const int cols8 = a.K / 8;
  const int t_in = tid >> 4, c16 = tid & 15;
  const uint4* route = reinterpret_cast<const uint4*>(a.route);
  uint4* out = reinterpret_cast<uint4*>(a.out);
  const int topk = a.topk;  // <= kSumMaxTopk (the host)
  // memory-level parallelism: the token's ids and all its slot rows are loaded before the first add (the slot's row address does not depend
  // on its id; a slot with id < 0 reads its (unwritten) row and is not added - topk_sum_kernel's skip); one round trip per pass
  for (int t0 = 0; t0 < ntok; t0 += 16) {
    const int tt = t0 + t_in;
    const bool on = tt < ntok;
    const int t = token0 + (on ? tt : 0);
    const uint4* rows = route + (size_t)t * topk * cols8 + n * 16 + c16;
    int id[kSumMaxTopk];
    uint4 v[kSumMaxTopk];
#pragma unroll
    for (int r = 0; r < kSumMaxTopk; ++r) {
      id[r] = (on && r < topk) ? __ldg(a.ids + t * topk + r) : -1;
      v[r] = (on && r < topk) ? __ldcg(rows + (size_t)r * cols8) : make_uint4(0u, 0u, 0u, 0u);
    }
    uint32_t smask = 0u;
    if (on) {
      float acc[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = 0.f;
#pragma unroll
      for (int r = 0; r < kSumMaxTopk; ++r) {
        if (r < topk && id[r] >= 0) {
          smask |= 1u << r;
          const uint32_t ws[4] = {v[r].x, v[r].y, v[r].z, v[r].w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            acc[2 * j] = fadd(acc[2 * j], bf16_bits_to_f32(ws[j] & 0xFFFFu));
            acc[2 * j + 1] = fadd(acc[2 * j + 1], bf16_bits_to_f32(ws[j] >> 16));
          }
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
      out[(size_t)t * cols8 + n * 16 + c16] = make_uint4(o[0], o[1], o[2], o[3]);
    }
    if (a.sum_discard) {
      __syncwarp();
      if (on && (c16 & 7) == 0) {
#pragma unroll
        for (int r = 0; r < kSumMaxTopk; ++r)
          if ((smask >> r) & 1u) discard_l2_line(rows + (size_t)r * cols8);
      }
    }
  }
}

template <int STAGES, int MINB>
__global__ void __launch_bounds__(kThreads, MINB) fc2_kernel(Fc2Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ ItemRing ring;
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
    item = ring.items[iss_k & 1];  // slot (k + 1) & 1
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

// ------------------------------------------------------------------------- 5. v9: the producer-warp skeleton (K-4, 6.11)
// The same GEMMs with warp specialisation: a CTA of 8 consumer warps (the v6 retile unchanged: warp = n16 slice, all m-tiles;
// MH = 2: 16 warps x 2 m-tiles) + 1 producer warp that issues every copy of the CTA: B and the column scales as TMA bulk
// copies (cp.async.bulk, 2 x 4 KB + 2 x 512 B per fc1 stage: the plane's layout is contiguous per stage), the 64 A rows as
// 16-B cp.asyncs (8 per lane per stage) into a row-major swizzled layout, their k128 scale words as 4-B cp.asyncs. The stage
// ring is handed over with mbarriers (full[s]: 32 x cp.async.mbarrier.arrive.noinc + lane 0's arrive.expect_tx with the TMA
// byte count; empty[s]: one arrive per consumer warp). No __syncthreads and no address work in the consumers' K-loop: a
// consumer warp waits for data that is normally already there, issues every LDS of the stage, then its QMMAs (v8: every
// stage, all 8 warps stop at a barrier and issue the next stage's copies - ~33 % of fc1's time with the tensor pipe idle,
// W/REPORT.md §3). Measured (second 4x RTX PRO 6000 host, call1 4096 rows): fc2 673 -> 567 us (-16 %, 4 stages x 2 CTAs/SM),
// fc1 995 -> 955 (-4 %: 9-warp CTAs fit 2 per SM only at 96 registers = spills, 1 per SM = 2 consumer warps per sub-partition);
// the QMMA peak is ~900 TFLOPS on this card and both kernels reach ~50 % of it. Bits: the QMMA chain and the epilogue are v8's
// byte for byte, the A fragment words are the same four words read from the v9 layout; only who copies and how the hand-off is
// synchronised changes.
constexpr int kWsThreads = kThreads + 32;  // fc2_kernel_ws: 8 consumer warps + the producer warp
constexpr int kWsProducer = 8;             // fc2_kernel_ws: the producer warp

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {  // .release.cta by default
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {  // .acquire.cta by default; spins
  asm volatile(
      "{\n.reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(bar), "r"(parity) : "memory");
}
// the arrive fires when all of this thread's prior cp.asyncs have landed (counts as one arrival: .noinc)
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(bar) : "memory");
}
// one arrival + the number of bulk-copy bytes the phase must see land (TMA: the copies below count them down)
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" ::"r"(bar), "r"(bytes) : "memory");
}
// a bulk global -> shared copy by the TMA unit (16-B aligned, a multiple of 16 B), completing on the mbarrier's tx count
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst), "l"(src),
               "r"(bytes), "r"(bar)
               : "memory");
}
// v9's A stage layout: row-major, 64 B per row (the k64 of the row), the 16-B chunk index XOR-swizzled with bit 1 of the row so
// that the consumers' two LDS.64 per (m-tile, k32) - row q and row q + 8, words 2c and 2c + 1 of the k32 block = the b12x
// k-permuted fragment (a0, a2) / (a1, a3) - are bank-conflict-free (16 lanes x 8 B on 16 distinct bank pairs). Copied with
// 16-B cp.asyncs (8 per producer lane per stage) instead of v8's 4-B ones (32 per lane): the LDGSTS count is what limits one
// producer warp (v9a, 4-B copies: 1.33 us per stage whatever the ring depth - K/k4/out/k4-ws1.log).
__device__ __forceinline__ uint32_t a_swz(int row) { return (uint32_t)(((row >> 1) & 1) << 1); }
__device__ __forceinline__ uint32_t a_row_chunk(int row, int chunk) { return (uint32_t)row * 64 + ((chunk ^ a_swz(row)) << 4); }
template <int NCW>
__device__ __forceinline__ void consumers_sync() {  // the NCW consumer warps only (named barrier 1; the producer never joins)
  asm volatile("bar.sync 1, %0;\n" ::"n"(NCW * 32) : "memory");
}
// cp.async.bulk (TMA) for the B tiles and column scales: 4 instructions per stage instead of 18 cp.async16 per lane - but on this
// driver / card the FIRST cp.async.bulk launch of a process takes 4.03 GiB of device memory (K/k4/mem_probe2.cu: context 0.63 ->
// 4.66 GiB; in situ the KV cache fell 11.31 -> 7.39 GiB per GPU). Off: the producer issues 16-B cp.asyncs.
constexpr bool kProducerTma = false;
constexpr uint32_t kFullCount = 33;  // 32 x cp.async.mbarrier.arrive.noinc + the producer lane 0's release arrive
constexpr uint32_t kEmptyCount = 8;  // one arrive per consumer warp (fc2_kernel_ws: 8 consumer warps)

// the (buffer, phase parity) pair of a round-robin stage ring
struct RingPos {
  int buf = 0;
  uint32_t par = 0;
  template <int STAGES>
  __device__ __forceinline__ void advance() {
    if (++buf == STAGES) { buf = 0; par ^= 1u; }
  }
};

// The producer warp's item hand-off (one warp: registers + shuffles instead of v8's staged shared-memory dance). Lane l owns rows
// l and l + 32 of the 64-row item. At stage d == 0 of the current item lane 0 claims the next one, d == 2 its four ints are
// loaded, d == 4 the 64 pairs, d == 6 everything is written to the ring slot `nb` (the consumers read meta at the start of
// that item and pairs / tw in its epilogue; the slot was last read in the epilogue of the item before the current one, which
// every consumer has left: the producer is at most STAGES stages ahead). on_pairs(nb, pr0, pr1) stages anything else (FC2:
// the token weights, d == 5 load / d == 7 store).
struct ProducerNext {
  int item = 0;        // the claimed next item (valid after d == 2)
  int w = 0;           // lanes 0-3: one int of its metadata (after d == 2)
  int4 meta = make_int4(0, 0, 0, 0);
  int pr0 = -1, pr1 = -1;
};
template <typename AoffFn, typename ExtraFn>
__device__ __forceinline__ void producer_stage_next(ItemRing& ring, ProducerNext& nx, int d, int nb, int lane, int cur_item,
                                                    int n_items, const int4* items, const int32_t* row_pair, int32_t* counter,
                                                    int dynamic_schedule, AoffFn aoff_of, ExtraFn extra) {
  if (d == 0) {
    int c = 0;
    if (lane == 0) c = claim_next_item(counter, cur_item, dynamic_schedule);
    nx.item = __shfl_sync(0xffffffffu, c, 0);
  }
  if (d == 2) nx.w = (lane < 4 && nx.item < n_items) ? __ldg(reinterpret_cast<const int32_t*>(items) + (size_t)nx.item * 4 + lane) : 0;
  if (d == 4) {
    nx.meta = make_int4(__shfl_sync(0xffffffffu, nx.w, 0), __shfl_sync(0xffffffffu, nx.w, 1), __shfl_sync(0xffffffffu, nx.w, 2),
                        __shfl_sync(0xffffffffu, nx.w, 3));
    const int rows = nx.meta.z * 16;
    nx.pr0 = lane < rows ? __ldg(row_pair + nx.meta.y + lane) : -1;
    nx.pr1 = lane + 32 < rows ? __ldg(row_pair + nx.meta.y + lane + 32) : -1;
  }
  if (d == 6) {
    if (lane == 0) ring.meta[nb] = nx.meta;
    ring.pairs[nb][lane] = nx.pr0;
    ring.pairs[nb][lane + 32] = nx.pr1;
    ring.aoff[nb][lane] = nx.pr0 >= 0 ? aoff_of(nx.pr0) : 0xFFFFFFFFu;
    ring.aoff[nb][lane + 32] = nx.pr1 >= 0 ? aoff_of(nx.pr1) : 0xFFFFFFFFu;
  }
  extra(d, nb, nx.pr0, nx.pr1);
}

// the A rows of a stage: 64 rows x 64 B as 256 16-B chunks, 8 cp.async16 per lane; instruction i covers rows 8i..8i+7, a row's
// four chunks by four consecutive lanes (64 contiguous bytes of global memory per row), zero-filled for padding rows. The lane's
// eight row offsets (and the two of its scale rows) are read from the ring once per item (ProducerRows): no LDS in the stage loop.
struct ProducerRows {
  uint32_t a[8];   // rows 8i + (lane >> 2)
  uint32_t sf[2];  // rows lane, lane + 32
  __device__ __forceinline__ void load(const uint32_t* aoff64, int lane) {
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = aoff64[8 * i + (lane >> 2)];
    sf[0] = aoff64[lane];
    sf[1] = aoff64[lane + 32];
  }
};
__device__ __forceinline__ void producer_copy_a(uint32_t sb_a, const uint8_t* a_base, const ProducerRows& rows, int t, int lane) {
  const int chunk = lane & 3, rsub = lane >> 2;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t off = rows.a[i];
    const bool on = off != 0xFFFFFFFFu;
    cp_async16_zfill(sb_a + a_row_chunk(8 * i + rsub, chunk), a_base + (on ? off : 0u) + t * 64 + chunk * 16, on ? 16 : 0);
  }
}

// MH = m-halves: 1 = 8 consumer warps, every warp all 4 m-tiles of the item (v8's retile; 9 warps, 2 CTAs/SM only at 96 registers =
// spills, 1 CTA/SM = 2 consumer warps per sub-partition); 2 = 16 consumer warps, warp = (m-half, n16 slice), 2 m-tiles each (17 warps,
// 1 CTA/SM = 4 consumer warps per sub-partition as v8's two CTAs, ~70 registers, the B fragments read by both m-halves)
// fc1's producer warps: with TMA one warp does everything; without it (kProducerTma = false) the B tiles + column scales (18 x
// cp.async16 per lane per stage) get a second warp, the first keeps the A rows, their scales and the item hand-off - one warp
// issuing all 28 cp.asyncs of a stage is slower than the consumers at 4 m-tiles (K-4 §6.2: 1 121 us vs 955 with TMA)
constexpr int kFc1Producers = kProducerTma ? 1 : 2;

template <int STAGES, int MINB, int MH>
__global__ void __launch_bounds__(8 * MH * 32 + 32 * kFc1Producers, MINB) fc1_kernel_ws(Fc1Args a) {
  constexpr int NCW = 8 * MH;          // consumer warps; the producers are warps NCW (A + ring) [and NCW + 1 (B)]
  constexpr int MT = 4 / MH;           // m-tiles per consumer warp
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ ItemRing ring;
  __shared__ __align__(8) uint64_t bar_full[STAGES], bar_empty[STAGES];
  __shared__ __align__(8) uint64_t bar_meta[2];  // the A producer -> the B producer: ring.meta[slot] of the next item is written
  __shared__ float s_amax[MT][NCW][16];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int K = a.K, n_stages = K / 64;
  const uint32_t sbase0 = smem_addr(smem);
  const int n_items = a.counts[1];
  if (tid < STAGES) {
    mbar_init(smem_addr(&bar_full[tid]), 32 * kFc1Producers + 1);
    mbar_init(smem_addr(&bar_empty[tid]), NCW);
  }
  if (tid < 2) mbar_init(smem_addr(&bar_meta[tid]), 1);
  if (warp == NCW) {  // the first item straight from global memory (before the barrier: the B producer reads meta[0] too)
    int4 m = make_int4(0, 0, 0, 0);
    const int item = blockIdx.x;
    if (item < n_items) {
      const int w = lane < 4 ? __ldg(reinterpret_cast<const int32_t*>(a.items) + (size_t)item * 4 + lane) : 0;
      m = make_int4(__shfl_sync(0xffffffffu, w, 0), __shfl_sync(0xffffffffu, w, 1), __shfl_sync(0xffffffffu, w, 2),
                    __shfl_sync(0xffffffffu, w, 3));
    }
    const int rows = m.z * 16;
    const int p0 = lane < rows ? __ldg(a.row_pair + m.y + lane) : -1;
    const int p1 = lane + 32 < rows ? __ldg(a.row_pair + m.y + lane + 32) : -1;
    if (lane == 0) ring.meta[0] = m;
    ring.pairs[0][lane] = p0;
    ring.pairs[0][lane + 32] = p1;
    ring.aoff[0][lane] = p0 >= 0 ? (uint32_t)(p0 / a.topk) * (uint32_t)K : 0xFFFFFFFFu;
    ring.aoff[0][lane + 32] = p1 >= 0 ? (uint32_t)(p1 / a.topk) * (uint32_t)K : 0xFFFFFFFFu;
  }
  __syncthreads();  // the barriers are initialised and item 0 is in the ring; from here on the roles only meet through the barriers

  if (kFc1Producers == 2 && warp == NCW + 1) {
    // ================================================================================ the B producer warp (no TMA)
    const uint32_t wproj_b = (uint32_t)kN * (K / 32) * 16;
    const uint32_t sproj_b = (uint32_t)kN * (K / 32);
    RingPos pos;
    int round0 = STAGES, k = 0;
    while (true) {
      // the A producer has written meta[k & 1] (its arrive at d == 6 of item k - 1): bar_meta[s] serves the items k = s, s + 2, ... with
      // k >= 1, i.e. its phase for item k is (k - 1) >> 1 (a wrong parity returns at once on a fresh barrier: stale meta, then a
      // producer spinning on `empty` after the consumers have left - the v9j hang, K-4)
      if (k > 0) mbar_wait(smem_addr(&bar_meta[k & 1]), (uint32_t)((k - 1) >> 1) & 1u);
      const int4 it = ring.meta[k & 1];
      if (it.z == 0) {
        if (round0 == 0) mbar_wait(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u);
        cp_async_mbar_arrive_noinc(smem_addr(&bar_full[pos.buf]));
        break;
      }
      const int e = it.x, tile = it.w;
      const bool wide = tile != kN1Tiles - 1;
      const uint8_t* iss_b = reinterpret_cast<const uint8_t*>(a.w13) + (size_t)e * 2 * wproj_b + (size_t)tile * 128 * (K / 32) * 16;
      const uint8_t* iss_sfb = reinterpret_cast<const uint8_t*>(a.w13_sf) + (size_t)e * 2 * sproj_b + (size_t)tile * 128 * (K / 32);
#pragma unroll 1
      for (int t = 0; t < n_stages; ++t) {
        if (round0 > 0) --round0;
        else mbar_wait(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u);
        const uint32_t sb = sbase0 + pos.buf * kF1Stage, full = smem_addr(&bar_full[pos.buf]);
        if (wide) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const uint32_t o = (lane + 32 * i) * 16;
            const uint8_t* src = iss_b + (size_t)t * 4096 + o;
            cp_async16(sb + o, src);
            cp_async16(sb + 4096 + o, src + wproj_b);
          }
#pragma unroll
          for (int proj = 0; proj < 2; ++proj)
            cp_async16(sb + kF1B + kF1A + kF1SFA + proj * 512 + lane * 16, iss_sfb + proj * sproj_b + (t >> 1) * 512 + lane * 16);
        } else {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint32_t j = lane + 32 * i, kk = j >> 6, rem = j & 63;
            const uint8_t* src = iss_b + (size_t)t * 2048 + kk * 1024 + rem * 16;
            cp_async16(sb + kk * 2048 + rem * 16, src);
            cp_async16(sb + 4096 + kk * 2048 + rem * 16, src + wproj_b);
          }
          {
            const uint32_t proj = lane >> 4, j = lane & 15;
            cp_async16(sb + kF1B + kF1A + kF1SFA + proj * 512 + j * 16, iss_sfb + proj * sproj_b + (t >> 1) * 256 + j * 16);
          }
        }
        cp_async_mbar_arrive_noinc(full);
        pos.advance<STAGES>();
      }
      ++k;
    }
    cp_async_wait<0>();
    return;
  }

  if (warp == NCW) {
    // ================================================================================ the (A) producer warp
    const int claim_at = n_stages - 10;
    const uint32_t wproj_b = (uint32_t)kN * (K / 32) * 16;
    const uint32_t sproj_b = (uint32_t)kN * (K / 32);
    const auto aoff_of = [&](int pair) { return (uint32_t)(pair / a.topk) * (uint32_t)K; };
    // the meta of the next item is published to the B producer right after the ring write (d == 6; lane 0 wrote it)
    const auto meta_extra = [&](int d, int nb, int, int) {
      if (kFc1Producers == 2 && d == 6 && lane == 0) mbar_arrive(smem_addr(&bar_meta[nb]));
    };
    const uint8_t* a_base = reinterpret_cast<const uint8_t*>(a.a_values);
    int item = blockIdx.x, k = 0;
    RingPos pos;
    int round0 = STAGES;  // stages issued so far < STAGES: the buffers start empty
    ProducerNext nx;
    while (true) {
      const int4 it = ring.meta[k & 1];
      if (it.z == 0) {  // no item: one more phase of `full` so that the consumers read the terminator, then leave
        if (round0 == 0) mbar_wait(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u);
        __syncwarp();
        if (lane == 0) mbar_arrive_expect_tx(smem_addr(&bar_full[pos.buf]), 0);
        cp_async_mbar_arrive_noinc(smem_addr(&bar_full[pos.buf]));
        break;
      }
      const int e = it.x, tile = it.w, slot = k & 1, nb = slot ^ 1;
      const bool wide = tile != kN1Tiles - 1;
      const uint8_t* iss_b = reinterpret_cast<const uint8_t*>(a.w13) + (size_t)e * 2 * wproj_b + (size_t)tile * 128 * (K / 32) * 16;
      const uint8_t* iss_sfb = reinterpret_cast<const uint8_t*>(a.w13_sf) + (size_t)e * 2 * sproj_b + (size_t)tile * 128 * (K / 32);
      ProducerRows rows;
      rows.load(ring.aoff[slot], lane);
#pragma unroll 1
      for (int t = 0; t < n_stages; ++t) {
        if (round0 > 0) --round0;
        else mbar_wait(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u);  // the consumers have left this buffer (previous round)
        const uint32_t sb = sbase0 + pos.buf * kF1Stage, full = smem_addr(&bar_full[pos.buf]);
        producer_stage_next(ring, nx, t - claim_at, nb, lane, item, n_items, a.items, a.row_pair, a.counts + 3, a.dynamic_schedule,
                            aoff_of, meta_extra);
        __syncwarp();  // the ring writes of this stage (if any) precede lane 0's release arrive below
        // B (both projections) and the column scales by the TMA unit: 128-column tiles 2 x 4 KB + 2 x 512 B contiguous per stage,
        // the 64-column tile 2 x (2 x 1 KB) + 2 x 256 B; lane 0 arrives with the byte count, the copies count it down
        if constexpr (kProducerTma) {
          if (lane == 0) {
            const uint32_t sfb_dst = sb + kF1B + kF1A + kF1SFA;
            if (wide) {
              mbar_arrive_expect_tx(full, 2 * 4096 + 2 * 512);
              const uint8_t* src = iss_b + (size_t)t * 4096;
              bulk_g2s(sb, src, 4096, full);
              bulk_g2s(sb + 4096, src + wproj_b, 4096, full);
              const uint8_t* sf = iss_sfb + (t >> 1) * 512;
              bulk_g2s(sfb_dst, sf, 512, full);
              bulk_g2s(sfb_dst + 512, sf + sproj_b, 512, full);
            } else {
              mbar_arrive_expect_tx(full, 4 * 1024 + 2 * 256);
              const uint8_t* src = iss_b + (size_t)t * 2048;
              bulk_g2s(sb, src, 1024, full);
              bulk_g2s(sb + 2048, src + 1024, 1024, full);
              bulk_g2s(sb + 4096, src + wproj_b, 1024, full);
              bulk_g2s(sb + 4096 + 2048, src + wproj_b + 1024, 1024, full);
              const uint8_t* sf = iss_sfb + (t >> 1) * 256;
              bulk_g2s(sfb_dst, sf, 256, full);
              bulk_g2s(sfb_dst + 512, sf + sproj_b, 256, full);
            }
          }
        } else {  // the B producer warp copies B and the column scales; this warp only arrives (release of the ring writes)
          if (lane == 0) mbar_arrive(full);
          if (kFc1Producers == 1 && wide) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              const uint32_t o = (lane + 32 * i) * 16;
              const uint8_t* src = iss_b + (size_t)t * 4096 + o;
              cp_async16(sb + o, src);
              cp_async16(sb + 4096 + o, src + wproj_b);
            }
#pragma unroll
            for (int proj = 0; proj < 2; ++proj)
              cp_async16(sb + kF1B + kF1A + kF1SFA + proj * 512 + lane * 16, iss_sfb + proj * sproj_b + (t >> 1) * 512 + lane * 16);
          } else if (kFc1Producers == 1) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const uint32_t j = lane + 32 * i, kk = j >> 6, rem = j & 63;
              const uint8_t* src = iss_b + (size_t)t * 2048 + kk * 1024 + rem * 16;
              cp_async16(sb + kk * 2048 + rem * 16, src);
              cp_async16(sb + 4096 + kk * 2048 + rem * 16, src + wproj_b);
            }
            {
              const uint32_t proj = lane >> 4, j = lane & 15;
              cp_async16(sb + kF1B + kF1A + kF1SFA + proj * 512 + j * 16, iss_sfb + proj * sproj_b + (t >> 1) * 256 + j * 16);
            }
          }
        }
        // A: the 64 rows' k64 (zero-filled for padding rows and the tiles >= m_tiles) and their k128 scale words, cp.async
        producer_copy_a(sb + kF1B, a_base, rows, t, lane);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t off = rows.sf[h];
          const bool on = off != 0xFFFFFFFFu;
          cp_async4_zfill(sb + kF1B + kF1A + (lane + 32 * h) * 4, a.a_scales + ((on ? off : 0u) >> 5) + (t >> 1) * 4, on ? 4 : 0);
        }
        cp_async_mbar_arrive_noinc(full);  // this lane's cp.asyncs -> one arrival when they land (32 + lane 0's expect_tx = 33)
        pos.advance<STAGES>();
      }
      item = nx.item;
      ++k;
    }
    cp_async_wait<0>();
    return;
  }

  // ================================================================================ the consumer warps (v8's compute + epilogue)
  // warp tiling: 128-column tiles: n16 slice = warp & 7, m-tiles [mt0, mt0 + MT) with mt0 = (warp >> 3) * MT; the 64-column tile:
  // slice = warp & 3, MH = 1: m-tiles 2 (warp >> 2), +1 (v8); MH = 2: the one m-tile warp >> 2. j indexes the warp's m-tiles.
  const int q = lane >> 2, c = lane & 3;
  RingPos pos;
  int k = 0;
  while (true) {
    mbar_wait(smem_addr(&bar_full[pos.buf]), pos.par);  // stage 0 of item k (or the terminator): meta[k & 1] is published
    const int4 it = ring.meta[k & 1];
    const int e = it.x, m_tiles = it.z, tile = it.w;
    if (m_tiles == 0) break;
    const bool wide = tile != kN1Tiles - 1;
    const int ns16 = wide ? (warp & 7) : (warp & 3);
    const int mt0 = wide ? (warp >> 3) * MT : (MH == 1 ? 2 * (warp >> 2) : (warp >> 2));
    const int n_mt = wide ? MT : (MH == 1 ? 2 : 1);
    bool act[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) act[j] = j < n_mt && mt0 + j < m_tiles;  // warp-uniform
    const float alpha = fmul(__ldg(a.alpha + e), __ldg(a.input_global_scale + (a.igs_per_expert ? e : 0)));

    float acc[MT][2][2][4];
#pragma unroll
    for (int j = 0; j < MT; ++j)
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int r = 0; r < 4; ++r) acc[j][p][nt][r] = 0.f;

    // A fragment of lane (q, c) at (m-tile, k32 block kk) from the row-major swizzled stage layout: words 2c, 2c+1 of the k32
    // block of row q (a0, a2) and of row q + 8 (a1, a3) = the same four words v8 read as one LDS.128 from its fragment layout
    const uint32_t a_lo = kF1B + mt0 * 1024 + a_row_chunk(q, c >> 1) + (c & 1) * 8;  // + j * 1024, chunk ^= kk * 2
    const uint32_t a_hi = kF1B + mt0 * 1024 + a_row_chunk(q + 8, c >> 1) + (c & 1) * 8;
    const uint32_t sfa_off = kF1B + kF1A + (mt0 * 16 + q + ((lane & 1) << 3)) * 4;  // + j * 64
    const uint32_t b_off = (ns16 >> 1) * 512 + lane * 16 + (ns16 & 1) * 8;
    const uint32_t sfb_off = kF1B + kF1A + kF1SFA + (ns16 * 16 + q) * 4;

    // = v8's compute_stage arithmetic (the A words from the v9 layout) as a register pipeline over the stages: the fragments of
    // stage t + 1 (B words, A fragments of all the warp's m-tiles - unconditionally, the inactive ones' bytes are never used -,
    // scales) are read from shared memory before the QMMAs of stage t are issued, so the LDS latency hides behind the QMMAs of the
    // same warp - with 2 consumer warps per sub-partition there is no third warp to hide it behind. Two fragment sets in registers.
    struct Frags {
      const uint8_t* sp;                       // the stage buffer (the scale words are read at QMMA time: broadcast LDS.32, cheap)
      uint2 bw[2][2], alo[MT][2], ahi[MT][2];  // [kk][p], [j][kk]
    };
    auto load_frags = [&](const uint8_t* sp, Frags& f) {
      f.sp = sp;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int p = 0; p < 2; ++p) f.bw[kk][p] = *reinterpret_cast<const uint2*>(sp + p * 4096 + kk * 2048 + b_off);
#pragma unroll
      for (int j = 0; j < MT; ++j) {
        const int jm = j < n_mt ? j : 0;  // the 64-column tile gives a warp fewer m-tiles than MT: mt0 + j would leave the stage's A region
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          f.alo[j][kk] = *reinterpret_cast<const uint2*>(sp + ((a_lo + jm * 1024) ^ (kk * 32)));
          f.ahi[j][kk] = *reinterpret_cast<const uint2*>(sp + ((a_hi + jm * 1024) ^ (kk * 32)));
        }
      }
    };
    auto mma_stage = [&](const Frags& f, auto kb_tag) {
      constexpr int KB0 = decltype(kb_tag)::value;
      uint32_t sfb[2][2], sfa[MT];
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) sfb[p][nt] = *reinterpret_cast<const uint32_t*>(f.sp + sfb_off + p * 512 + nt * 32);
#pragma unroll
      for (int j = 0; j < MT; ++j) sfa[j] = *reinterpret_cast<const uint32_t*>(f.sp + sfa_off + (j < n_mt ? j : 0) * 64);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t b[2][2][2];
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          spread_e2m1x8(f.bw[kk][p].x, b[p][0][0], b[p][0][1]);
          spread_e2m1x8(f.bw[kk][p].y, b[p][1][0], b[p][1][1]);
        }
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          if (act[j]) {
#pragma unroll
            for (int p = 0; p < 2; ++p)
#pragma unroll
              for (int nt = 0; nt < 2; ++nt)
                qmma_kk<KB0>(kk, acc[j][p][nt], f.alo[j][kk].x, f.ahi[j][kk].x, f.alo[j][kk].y, f.ahi[j][kk].y, b[p][nt][0],
                             b[p][nt][1], sfa[j], sfb[p][nt]);
          }
        }
      }
    };
    // the pipeline: F0 holds stage t (loaded), F1 receives stage t + 1 before stage t's QMMAs; a stage's buffer is released to the
    // producer after its QMMAs (the release orders the LDS reads before it; the ring is STAGES deep, one stage of slack is nothing).
    // The last stage of an item has nothing to prefetch (the next item's tile may change the warp's slice); n_stages is even.
    Frags F0, F1;
    mbar_wait(smem_addr(&bar_full[pos.buf]), pos.par);
    load_frags(smem + pos.buf * kF1Stage, F0);
    auto pipe_step = [&](Frags& cur, Frags& nxt, bool has_next, auto kb_tag) {
      RingPos cur_pos = pos;  // cur's buffer is released after its QMMAs (its scale words are read there)
      pos.advance<STAGES>();
      if (has_next) {
        mbar_wait(smem_addr(&bar_full[pos.buf]), pos.par);
        load_frags(smem + pos.buf * kF1Stage, nxt);
      }
      mma_stage(cur, kb_tag);
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_addr(&bar_empty[cur_pos.buf]));
    };
#pragma unroll 1
    for (int t = 0; t < n_stages; t += 2) {
      pipe_step(F0, F1, true, std::integral_constant<int, 0>{});
      pipe_step(F1, F0, t + 2 < n_stages, std::integral_constant<int, 2>{});
    }

    // ---- epilogue = v8's (the partner-warp amax exchange through s_amax; the barrier is the consumers' named barrier, once per
    // j for every warp: CTA-uniform)
    const int pslot = k & 1;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      const int mt = mt0 + j;
      float v[2][4], am[2] = {0.f, 0.f};
      if (act[j]) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            const float x0 = activated_value(acc[j][1][nt][2 * hf], acc[j][0][nt][2 * hf], alpha, a.limit);
            const float x1 = activated_value(acc[j][1][nt][2 * hf + 1], acc[j][0][nt][2 * hf + 1], alpha, a.limit);
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
          if (c == 0) s_amax[j][warp][q + hf * 8] = amax;
        }
      }
      consumers_sync<NCW>();
      if (act[j]) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int row = mt * 16 + q + hf * 8;
          const int pair = ring.pairs[pslot][row];
          const float amax = fmax_plain(am[hf], s_amax[j][warp ^ 1][q + hf * 8]);
          float inv;
          const uint32_t byte = mx_scale(amax, inv);
          uint32_t h[4] = {cvt_e4m3x2(fmul(v[hf][0], inv), fmul(v[hf][1], inv)),
                           cvt_e4m3x2(fmul(v[hf][2], inv), fmul(v[hf][3], inv)), 0u, 0u};
          uint32_t o0, o1;
          quad_transpose16(h, c, o0, o1);
          if (pair >= 0) {
            if (c < 2) {
              uint2* dst = reinterpret_cast<uint2*>(reinterpret_cast<uint8_t*>(a.inter) + (size_t)pair * kInterBytes +
                                                   tile * 128 + ns16 * 16 + c * 8);
              *dst = make_uint2(o0, o1);
            }
            if ((ns16 & 1) == 0) {
              uint8_t* sf = a.inter_sf + (size_t)pair * (kSlices * 4) + tile * 4;
              if (c == 0) sf[ns16 >> 1] = (uint8_t)byte;
              if (!wide && c == 1) sf[(ns16 >> 1) + 2] = 0;
            }
          }
        }
      }
    }
    ++k;
  }
}

// ------------------------------------------------------------------- 5b. v10: fc1 as three warpgroups with setmaxnreg (K-5, 7.11)
// The v9 skeleton's fc1 problem (ncu): the fc1 consumer wants ~110-150 registers, so 9- or
// 10-warp CTAs fit two per SM only at 96 registers (spills) and one CTA per SM leaves 2 consumer warps per SM sub-partition
// (tensor pipe 58 % vs v8's 64 %). Here the CTA is 12 warps = 3 warpgroups: warps 0-7 consumers (v8's retile, all 4 m-tiles
// of an n16 slice), warps 8-11 producers. Launched for 2 CTAs/SM (80 registers per thread, 2 x 384 x 80 = 61 440), the
// producer warpgroup gives registers back (`setmaxnreg.dec 32`) and the consumer warpgroups take them (`setmaxnreg.inc 104`:
// 128 x 32 + 256 x 104 = 30 720 = 384 x 80 exactly). 16 consumer warps per SM = 4 per sub-partition as v8, without v8's
// barrier + issue phase (ncu: barrier + mio_throttle = 24 % of v8's stall cycles) and without the 96-register spills.
// Producers: warp 8 = the A rows of m-tiles 0-1 + the scale words of rows 0-31 + the item hand-off (ProducerNext, published to
// the other producers through bar_meta), warp 9 = A m-tiles 2-3 + scale rows 32-63, warps 10 / 11 = B projection 0 / 1 + its
// column scales (8 + 1 cp.async16 per lane per stage). A is copied with 8-B cp.asyncs into a fragment-order layout the
// consumer reads with one LDS.128 per (m-tile, k32): [mt][kk][q][c][row q words 2c, 2c+1 | row q + 8 words 2c, 2c+1] =
// (a0, a2, a1, a3) - the same four words as v8's LDS.128 / v9's two LDS.64, in a 16-B unit that 8-B copies can fill (v8's word
// interleave needs 4-B copies: 32 per lane per stage, the v9a limit). The LDGSTS writes are bank-conflict-free: a half-warp
// covers (q & 1, c, row half) once each. Bits: the QMMA chain, operands and epilogue are v8's; only the copy issue and the
// stage hand-off differ (harness).
constexpr int kWgThreads = 384;        // 8 consumer warps + 4 producer warps
constexpr int kWgConsumers = 8;
constexpr int kWgProducers = 4;
constexpr int kWgRegsProducer = 32;    // setmaxnreg.dec (multiple of 8, >= 24)
constexpr int kWgRegsConsumer = 104;   // setmaxnreg.inc: (384 x 80 - 128 x 32) / 256

__device__ __forceinline__ void cp_async8_zfill(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
template <int N>
__device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N)); }
template <int N>
__device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N)); }
// mbarrier wait with a suspend-time hint: the thread sleeps until the phase completes (or the hint expires) instead of spinning
// on try_wait - the v9 / v10a producers spin on `empty` most of the time and take issue slots from the consumers (ncu: branch_resolving
// 0.75-1.2 warps per issue cycle, m8 50 % issue-busy). The hint is CUTLASS's ClusterBarrier::wait value (10 ms).
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(bar), "r"(parity), "r"(0x989680u) : "memory");
}

// the v10 A fragment unit of (m-tile, k32 block kk, quad row q, c): 16 B at mt * 1024 + kk * 512 + q * 64 + c * 16
__device__ __forceinline__ uint32_t a10_frag(int mt, int kk, int q, int c) { return (uint32_t)(mt * 1024 + kk * 512 + q * 64 + c * 16); }

// A producer warp w (0 / 1 = m-tiles 2w, 2w + 1): instruction i (0..7) copies the k64 of the four rows {2p, 2p + 1, 2p + 8, 2p + 9}
// (p = i & 3) of m-tile 2w + (i >> 2) as 32 x 8 B: lane = (kk = lane >> 4, j = (lane >> 2) & 3 -> row half hi = j >> 1, q = 2p + (j & 1),
// c = lane & 3). Global: each row's 64 B by 8 lanes (2 sectors); shared: a half-warp writes 16 distinct bank pairs.
struct WgRows {
  uint32_t a[8];  // the lane's row offset per instruction (0xFFFFFFFF: padding / tile >= m_tiles)
  uint32_t sf;    // the scale row 32w + lane
  __device__ __forceinline__ void load(const uint32_t* aoff64, int w, int lane) {
    const int j = (lane >> 2) & 3, hi = j >> 1, qo = j & 1;
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = aoff64[(2 * w + (i >> 2)) * 16 + hi * 8 + 2 * (i & 3) + qo];
    sf = aoff64[32 * w + lane];
  }
};
__device__ __forceinline__ void wg_copy_a(uint32_t sb_a, const uint8_t* a_base, const WgRows& rows, int w, int t, int lane) {
  const int kk = lane >> 4, j = (lane >> 2) & 3, hi = j >> 1, qo = j & 1, c = lane & 3;
  const uint32_t stage_off = (uint32_t)(t * 64 + kk * 32 + c * 8);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t off = rows.a[i];
    const bool on = off != 0xFFFFFFFFu;
    // the source address as base + one 32-bit sum (wrapping): ptxas cannot hoist eight 64-bit row pointers out of the stage loop
    // (at 32 registers it spilled them and reloaded five per stage)
    cp_async8_zfill(sb_a + a10_frag(2 * w + (i >> 2), kk, 2 * (i & 3) + qo, c) + hi * 8, a_base + ((on ? off : 0u) + stage_off), on ? 8 : 0);
  }
}

template <int STAGES, int MINB>
__global__ void __launch_bounds__(kWgThreads, MINB) fc1_kernel_wg(Fc1Args a) {
  constexpr int NCW = kWgConsumers;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ ItemRing ring;
  __shared__ __align__(8) uint64_t bar_full[STAGES], bar_empty[STAGES];
  __shared__ __align__(8) uint64_t bar_meta[2];  // producer 8 -> producers 9-11: ring.meta[slot] / aoff[slot] of the next item are written
  __shared__ float s_amax[4][NCW][16];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int K = a.K, n_stages = K / 64;
  const uint32_t sbase0 = smem_addr(smem);
  const int n_items = a.counts[1];
  if (tid < STAGES) {
    mbar_init(smem_addr(&bar_full[tid]), 32 * kWgProducers + 1);  // every producer lane's cp.async arrive + producer 8 lane 0's release arrive
    mbar_init(smem_addr(&bar_empty[tid]), NCW);
  }
  if (tid < 2) mbar_init(smem_addr(&bar_meta[tid]), 1);
  if (warp == NCW) {  // the first item straight from global memory
    int4 m = make_int4(0, 0, 0, 0);
    const int item = blockIdx.x;
    if (item < n_items) {
      const int w = lane < 4 ? __ldg(reinterpret_cast<const int32_t*>(a.items) + (size_t)item * 4 + lane) : 0;
      m = make_int4(__shfl_sync(0xffffffffu, w, 0), __shfl_sync(0xffffffffu, w, 1), __shfl_sync(0xffffffffu, w, 2),
                    __shfl_sync(0xffffffffu, w, 3));
    }
    const int rows = m.z * 16;
    const int p0 = lane < rows ? __ldg(a.row_pair + m.y + lane) : -1;
    const int p1 = lane + 32 < rows ? __ldg(a.row_pair + m.y + lane + 32) : -1;
    if (lane == 0) ring.meta[0] = m;
    ring.pairs[0][lane] = p0;
    ring.pairs[0][lane + 32] = p1;
    ring.aoff[0][lane] = p0 >= 0 ? (uint32_t)(p0 / a.topk) * (uint32_t)K : 0xFFFFFFFFu;
    ring.aoff[0][lane + 32] = p1 >= 0 ? (uint32_t)(p1 / a.topk) * (uint32_t)K : 0xFFFFFFFFu;
  }
  __syncthreads();  // barriers initialised, item 0 in the ring; from here on the roles only meet through the barriers

  if (warp >= NCW) {
    // ================================================================================ the producer warpgroup (warps 8-11)
    if constexpr (MINB >= 2) setmaxnreg_dec<kWgRegsProducer>();
    const int role = warp - NCW;  // 0 / 1: A m-tiles (2 role, 2 role + 1) + scale rows 32 role..; 2 / 3: B projection role - 2
    const uint32_t wproj_b = (uint32_t)kN * (K / 32) * 16;
    const uint32_t sproj_b = (uint32_t)kN * (K / 32);
    RingPos pos;
    int round0 = STAGES, k = 0;
    if (role == 0) {
      const int claim_at = n_stages - 10;
      const auto aoff_of = [&](int pair) { return (uint32_t)(pair / a.topk) * (uint32_t)K; };
      const auto meta_extra = [&](int d, int nb, int, int) {
        if (d == 6) {
          __syncwarp();  // every lane's ring writes precede lane 0's release arrive
          if (lane == 0) mbar_arrive(smem_addr(&bar_meta[nb]));
        }
      };
      const uint8_t* a_base = reinterpret_cast<const uint8_t*>(a.a_values);
      int item = blockIdx.x;
      ProducerNext nx;
      while (true) {
        const int4 it = ring.meta[k & 1];
        if (it.z == 0) {  // no item: one more phase of `full` so that the consumers read the terminator, then leave
          if (round0 == 0) mbar_wait_sleep(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u);
          __syncwarp();
          if (lane == 0) mbar_arrive(smem_addr(&bar_full[pos.buf]));
          cp_async_mbar_arrive_noinc(smem_addr(&bar_full[pos.buf]));
          break;
        }
        const int slot = k & 1, nb = slot ^ 1;
        WgRows rows;
        rows.load(ring.aoff[slot], 0, lane);
#pragma unroll 1
        for (int t = 0; t < n_stages; ++t) {
          if (round0 > 0) --round0;
          else mbar_wait_sleep(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u);
          const uint32_t sb = sbase0 + pos.buf * kF1Stage, full = smem_addr(&bar_full[pos.buf]);
          producer_stage_next(ring, nx, t - claim_at, nb, lane, item, n_items, a.items, a.row_pair, a.counts + 3, a.dynamic_schedule,
                              aoff_of, meta_extra);
          __syncwarp();  // the ring writes of this stage (if any) precede lane 0's release arrive
          if (lane == 0) mbar_arrive(full);
          wg_copy_a(sb + kF1B, a_base, rows, 0, t, lane);
          {
            const uint32_t off = rows.sf;
            const bool on = off != 0xFFFFFFFFu;
            cp_async4_zfill(sb + kF1B + kF1A + lane * 4, a.a_scales + ((on ? off : 0u) >> 5) + (t >> 1) * 4, on ? 4 : 0);
          }
          cp_async_mbar_arrive_noinc(full);
          pos.advance<STAGES>();
        }
        item = nx.item;
        ++k;
      }
    } else {
      const uint8_t* a_base = reinterpret_cast<const uint8_t*>(a.a_values);
      const int proj = role - 2;
      while (true) {
        // producer 8 has written meta[k & 1] (its arrive at d == 6 of item k - 1): bar_meta[s] serves the items k = s, s + 2, ... with
        // k >= 1, i.e. its phase for item k is ((k - 1) >> 1) & 1 (a fresh barrier returns at once on the other parity: K-4's v9j hang)
        if (k > 0) mbar_wait_sleep(smem_addr(&bar_meta[k & 1]), (uint32_t)((k - 1) >> 1) & 1u);
        const int4 it = ring.meta[k & 1];
        if (it.z == 0) {
          if (round0 == 0) mbar_wait_sleep(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u);
          cp_async_mbar_arrive_noinc(smem_addr(&bar_full[pos.buf]));
          break;
        }
        const int e = it.x, tile = it.w;
        const bool wide = tile != kN1Tiles - 1;
        const uint8_t* iss_b = reinterpret_cast<const uint8_t*>(a.w13) + (size_t)e * 2 * wproj_b + (size_t)tile * 128 * (K / 32) * 16 +
                               (size_t)(proj > 0 ? proj : 0) * wproj_b;
        const uint8_t* iss_sfb = reinterpret_cast<const uint8_t*>(a.w13_sf) + (size_t)e * 2 * sproj_b + (size_t)tile * 128 * (K / 32) +
                                 (size_t)(proj > 0 ? proj : 0) * sproj_b;
        WgRows rows;
        if (role == 1) rows.load(ring.aoff[k & 1], 1, lane);
#pragma unroll 1
        for (int t = 0; t < n_stages; ++t) {
          if (round0 > 0) --round0;
          else mbar_wait_sleep(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u);
          const uint32_t sb = sbase0 + pos.buf * kF1Stage, full = smem_addr(&bar_full[pos.buf]);
          if (role == 1) {
            wg_copy_a(sb + kF1B, a_base, rows, 1, t, lane);
            const uint32_t off = rows.sf;
            const bool on = off != 0xFFFFFFFFu;
            cp_async4_zfill(sb + kF1B + kF1A + (32 + lane) * 4, a.a_scales + ((on ? off : 0u) >> 5) + (t >> 1) * 4, on ? 4 : 0);
          } else if (wide) {  // B projection `proj`: 4 KB contiguous per stage + its 512 B of column scales per K128
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              const uint32_t o = (lane + 32 * i) * 16;
              cp_async16(sb + proj * 4096 + o, iss_b + (size_t)t * 4096 + o);
            }
            cp_async16(sb + kF1B + kF1A + kF1SFA + proj * 512 + lane * 16, iss_sfb + (t >> 1) * 512 + lane * 16);
          } else {  // the 64-column tile: 2 x 1 KB per projection per stage, 256 B of column scales
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const uint32_t j = lane + 32 * i, kk = j >> 6, rem = j & 63;
              cp_async16(sb + proj * 4096 + kk * 2048 + rem * 16, iss_b + (size_t)t * 2048 + kk * 1024 + rem * 16);
            }
            if (lane < 16) cp_async16(sb + kF1B + kF1A + kF1SFA + proj * 512 + lane * 16, iss_sfb + (t >> 1) * 256 + lane * 16);
          }
          cp_async_mbar_arrive_noinc(full);
          pos.advance<STAGES>();
        }
        ++k;
      }
    }
    cp_async_wait<0>();
    return;
  }

  // ================================================================================ the consumer warpgroups (warps 0-7): v8's compute + epilogue
  if constexpr (MINB >= 2) setmaxnreg_inc<kWgRegsConsumer>();
  const int q = lane >> 2, c = lane & 3;
  RingPos pos;
  int k = 0;
  while (true) {
    mbar_wait(smem_addr(&bar_full[pos.buf]), pos.par);  // stage 0 of item k (or the terminator): meta[k & 1] is published
    const int4 it = ring.meta[k & 1];
    const int e = it.x, m_tiles = it.z, tile = it.w;
    if (m_tiles == 0) break;
    const bool wide = tile != kN1Tiles - 1;
    // warp tiling = v8's: 128-column tiles: n16 slice = warp, all m-tiles; the 64-column tile: slice = warp & 3, m-tiles 2 (warp >> 2), +1
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

    const uint32_t a_off = kF1B + a10_frag(0, 0, q, c);                      // + mt * 1024 + kk * 512
    const uint32_t sfa_off = kF1B + kF1A + (q + ((lane & 1) << 3)) * 4;     // + mt * 64
    const uint32_t b_off = (ns16 >> 1) * 512 + lane * 16 + (ns16 & 1) * 8;  // + p * 4096 + kk * 2048
    const uint32_t sfb_off = kF1B + kF1A + kF1SFA + (ns16 * 16 + q) * 4;    // + p * 512 + nt * 32

    // = v8's compute_stage (the A fragment from the v10 unit: (a0, a2, a1, a3) = .x .y .z .w)
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
                qmma_kk<KB0>(kk, acc[mt][p][nt], af.x, af.z, af.y, af.w, b[p][nt][0], b[p][nt][1], sfa[mt], sfb[p][nt]);
          }
        }
      }
    };
    auto step = [&](auto kb_tag) {
      mbar_wait(smem_addr(&bar_full[pos.buf]), pos.par);
      compute_stage(smem + pos.buf * kF1Stage, kb_tag);
      __syncwarp();  // the warp's LDS of this buffer precede lane 0's release arrive
      if (lane == 0) mbar_arrive(smem_addr(&bar_empty[pos.buf]));
      pos.advance<STAGES>();
    };
#pragma unroll 1
    for (int t = 0; t < n_stages; t += 2) {
      step(std::integral_constant<int, 0>{});
      step(std::integral_constant<int, 2>{});
    }

    // ---- epilogue = v8's arithmetic (the partner-warp amax exchange through s_amax), with ONE named barrier of the consumers per
    // item instead of one per m-tile: every m-tile's partial amax is written first (s_amax has a slot per m-tile), then the
    // exchange. The max is exact and commutative - the same block amax, scale byte and e4m3 bytes as v8. The barrier of item k's
    // epilogue also orders the consumers' s_amax reads of item k - 1 before item k's writes (no warp can be an item ahead of another).
    const int pslot = k & 1;
    float v[4][2][4], am[4][2];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      am[mt][0] = am[mt][1] = 0.f;
      if (mt >= m_tiles) continue;  // CTA-uniform
      if (act[mt]) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            const float x0 = activated_value(acc[mt][1][nt][2 * hf], acc[mt][0][nt][2 * hf], alpha, a.limit);
            const float x1 = activated_value(acc[mt][1][nt][2 * hf + 1], acc[mt][0][nt][2 * hf + 1], alpha, a.limit);
            const uint32_t pk = pack_bf16x2(x0, x1);
            v[mt][hf][2 * nt] = __uint_as_float(pk << 16);
            v[mt][hf][2 * nt + 1] = __uint_as_float(pk & 0xFFFF0000u);
          }
          float amax = 0.f;
#pragma unroll
          for (int kx = 0; kx < 4; ++kx) amax = fmax_plain(amax, fabs_plain(v[mt][hf][kx]));
          amax = fmax_plain(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
          amax = fmax_plain(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
          am[mt][hf] = amax;
          if (c == 0) s_amax[mt][warp][q + hf * 8] = amax;
        }
      }
    }
    consumers_sync<NCW>();
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      if (mt >= m_tiles) continue;  // CTA-uniform
      if (act[mt]) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int row = mt * 16 + q + hf * 8;
          const int pair = ring.pairs[pslot][row];
          const float amax = fmax_plain(am[mt][hf], s_amax[mt][warp ^ 1][q + hf * 8]);
          float inv;
          const uint32_t byte = mx_scale(amax, inv);
          uint32_t h[4] = {cvt_e4m3x2(fmul(v[mt][hf][0], inv), fmul(v[mt][hf][1], inv)),
                           cvt_e4m3x2(fmul(v[mt][hf][2], inv), fmul(v[mt][hf][3], inv)), 0u, 0u};
          uint32_t o0, o1;
          quad_transpose16(h, c, o0, o1);
          if (pair >= 0) {
            if (c < 2) {
              uint2* dst = reinterpret_cast<uint2*>(reinterpret_cast<uint8_t*>(a.inter) + (size_t)pair * kInterBytes +
                                                   tile * 128 + ns16 * 16 + c * 8);
              *dst = make_uint2(o0, o1);
            }
            if ((ns16 & 1) == 0) {
              uint8_t* sf = a.inter_sf + (size_t)pair * (kSlices * 4) + tile * 4;
              if (c == 0) sf[ns16 >> 1] = (uint8_t)byte;
              if (!wide && c == 1) sf[(ns16 >> 1) + 2] = 0;
            }
          }
        }
      }
    }
    ++k;
  }
}

// FC2 with the producer-warp skeleton
template <int STAGES, int MINB>
__global__ void __launch_bounds__(kWsThreads, MINB) fc2_kernel_ws(Fc2Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ ItemRing ring;
  __shared__ __align__(8) uint64_t bar_full[STAGES], bar_empty[STAGES];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  constexpr int n_stages = kF2Stages;
  constexpr uint32_t kTileBytes = 128 * (kSlices * 4 - 2) * 16;
  constexpr uint32_t kTileSfBytes = 128 * (kSlices * 4 - 2);
  const uint32_t sbase0 = smem_addr(smem);
  if (tid < STAGES) {
    mbar_init(smem_addr(&bar_full[tid]), kFullCount);
    mbar_init(smem_addr(&bar_empty[tid]), kEmptyCount);
  }
  __syncthreads();

  if (warp == kWsProducer) {
    // ================================================================================ the producer warp
    const int n_items = a.counts[2];
    // 9 stages per item: claim at stage 1, the ring slot written at stage 7 (pairs / offsets / meta) and 8 (token weights) -
    // the slot was last read in the epilogue two items back, which every consumer has left once the producer may issue
    // stage 7 (it waits for the consumers' release of stage 7 - STAGES >= 0 of this item): needs STAGES <= 7
    static_assert(STAGES <= 7, "the ring slot hand-off of fc2_kernel_ws needs STAGES <= 7");
    constexpr int claim_at = 1;
    const auto aoff_of = [](int pair) { return (uint32_t)pair * (uint32_t)kInterBytes; };
    float tw0 = 0.f, tw1 = 0.f;
    const auto tw_extra = [&](int d, int nb, int pr0, int pr1) {
      if (d == 5) {
        tw0 = pr0 >= 0 ? __ldg(a.token_weights + pr0) : 0.f;
        tw1 = pr1 >= 0 ? __ldg(a.token_weights + pr1) : 0.f;
      }
      if (d == 7) {
        ring.tw[nb][lane] = tw0;
        ring.tw[nb][lane + 32] = tw1;
      }
    };
    const uint8_t* a_base = reinterpret_cast<const uint8_t*>(a.inter);
    int item = blockIdx.x, k = 0;
    {
      int4 m = make_int4(0, 0, 0, 0);
      if (item < n_items) {
        const int w = lane < 4 ? __ldg(reinterpret_cast<const int32_t*>(a.items) + (size_t)item * 4 + lane) : 0;
        m = make_int4(__shfl_sync(0xffffffffu, w, 0), __shfl_sync(0xffffffffu, w, 1), __shfl_sync(0xffffffffu, w, 2),
                      __shfl_sync(0xffffffffu, w, 3));
      }
      const int rows = m.z * 16;
      const int p0 = lane < rows ? __ldg(a.row_pair + m.y + lane) : -1;
      const int p1 = lane + 32 < rows ? __ldg(a.row_pair + m.y + lane + 32) : -1;
      if (lane == 0) ring.meta[0] = m;
      ring.pairs[0][lane] = p0;
      ring.pairs[0][lane + 32] = p1;
      ring.aoff[0][lane] = p0 >= 0 ? aoff_of(p0) : 0xFFFFFFFFu;
      ring.aoff[0][lane + 32] = p1 >= 0 ? aoff_of(p1) : 0xFFFFFFFFu;
      ring.tw[0][lane] = p0 >= 0 ? __ldg(a.token_weights + p0) : 0.f;
      ring.tw[0][lane + 32] = p1 >= 0 ? __ldg(a.token_weights + p1) : 0.f;
      __syncwarp();
    }
    RingPos pos;
    int round0 = STAGES;
    ProducerNext nx;
    const bool w2_hint = a.w2_evict_first != 0;
    const uint64_t w2_pol = w2_hint ? l2_policy_evict_first() : 0ull;
    while (true) {
      const int4 it = ring.meta[k & 1];
      if (it.z == 0) {
        if (round0 == 0) mbar_wait(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u);
        __syncwarp();
        if (lane == 0) mbar_arrive_expect_tx(smem_addr(&bar_full[pos.buf]), 0);
        cp_async_mbar_arrive_noinc(smem_addr(&bar_full[pos.buf]));
        break;
      }
      const int e = it.x, tile = it.w, slot = k & 1, nb = slot ^ 1;
      if (tile & kSumItem) {  // K-12: a sum item - the hand-off of the next item runs through the 9 stages, no copies (the consumers sum)
#pragma unroll 1
        for (int t = 0; t < n_stages; ++t) {
          if (round0 > 0) --round0;
          else mbar_wait(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u);
          const uint32_t full = smem_addr(&bar_full[pos.buf]);
          producer_stage_next(ring, nx, t - claim_at, nb, lane, item, n_items, a.items, a.row_pair, a.counts + 4, a.dynamic_schedule,
                              aoff_of, tw_extra);
          __syncwarp();
          if (lane == 0) mbar_arrive(full);
          cp_async_mbar_arrive_noinc(full);
          pos.advance<STAGES>();
        }
        item = nx.item;
        ++k;
        continue;
      }
      const uint8_t* iss_b = reinterpret_cast<const uint8_t*>(a.w2) + ((size_t)e * a.out_tiles + tile) * kTileBytes;
      const uint8_t* iss_sf = a.w2_sf + ((size_t)e * a.out_tiles + tile) * kTileSfBytes;
      ProducerRows rows;
      rows.load(ring.aoff[slot], lane);
#pragma unroll 1
      for (int t = 0; t < n_stages; ++t) {
        if (round0 > 0) --round0;
        else mbar_wait(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u);
        const uint32_t sb = sbase0 + pos.buf * kF2Stage, full = smem_addr(&bar_full[pos.buf]);
        producer_stage_next(ring, nx, t - claim_at, nb, lane, item, n_items, a.items, a.row_pair, a.counts + 4, a.dynamic_schedule,
                            aoff_of, tw_extra);
        __syncwarp();
        {
          const int s = t >> 1;
          if constexpr (kProducerTma) {  // B: 4 KB contiguous per stage + the tile's column scales of the K128 slice (512 B; the tail 256)
            if (lane == 0) {
              const uint32_t sf_bytes = s == kSlices - 1 ? 256 : 512;
              mbar_arrive_expect_tx(full, 4096 + sf_bytes);
              bulk_g2s(sb, iss_b + (size_t)t * 4096, 4096, full);
              bulk_g2s(sb + kF2B + kF2A + kF2SFA, iss_sf + s * 512, sf_bytes, full);
            }
          } else {
            if (lane == 0) mbar_arrive(full);
            if (w2_hint) {  // K-12: W2 streams once per call - evict_first keeps the live route rows (and the A rows) in L2
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                const uint32_t o = (lane + 32 * i) * 16;
                cp_async16_hint(sb + o, iss_b + (size_t)t * 4096 + o, w2_pol);
              }
              if (lane < (s == kSlices - 1 ? 16 : 32)) cp_async16_hint(sb + kF2B + kF2A + kF2SFA + lane * 16, iss_sf + s * 512 + lane * 16, w2_pol);
            } else {
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                const uint32_t o = (lane + 32 * i) * 16;
                cp_async16(sb + o, iss_b + (size_t)t * 4096 + o);
              }
              if (lane < (s == kSlices - 1 ? 16 : 32)) cp_async16(sb + kF2B + kF2A + kF2SFA + lane * 16, iss_sf + s * 512 + lane * 16);
            }
          }
        }
        producer_copy_a(sb + kF2B, a_base, rows, t, lane);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t off = rows.sf[h];
          const bool on = off != 0xFFFFFFFFu;
          cp_async4_zfill(sb + kF2B + kF2A + (lane + 32 * h) * 4, a.inter_sf + ((on ? off : 0u) >> 5) + (t >> 1) * 4, on ? 4 : 0);
        }
        cp_async_mbar_arrive_noinc(full);
        pos.advance<STAGES>();
      }
      item = nx.item;
      ++k;
    }
    cp_async_wait<0>();
    return;
  }

  // ==================================================================================== the 8 consumer warps (v8's compute + epilogue)
  const int q = lane >> 2, c = lane & 3;
  RingPos pos;
  int k = 0;
  const bool route_hint = a.sum_mode != 0 && a.sum_route_hint != 0;
  const uint64_t route_pol = route_hint ? l2_policy_evict_last() : 0ull;
  while (true) {
    mbar_wait(smem_addr(&bar_full[pos.buf]), pos.par);
    const int4 it = ring.meta[k & 1];
    const int e = it.x, m_tiles = it.z, tile = it.w;
    if (m_tiles == 0) break;
    if (tile & kSumItem) {  // K-12: a sum item - release its 9 data-less stages first (the producer then fills the ring for the next item), then sum
#pragma unroll 1
      for (int t = 0; t < n_stages; ++t) {
        mbar_wait(smem_addr(&bar_full[pos.buf]), pos.par);
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_addr(&bar_empty[pos.buf]));
        pos.advance<STAGES>();
      }
      fc2_sum_item(a, it.x, it.y, tile & 0xFF, lane, tid, k);
      ++k;
      continue;
    }
    const int ns16 = warp;
    const float down_scale = fmul(__ldg(a.down_alpha + e), __ldg(a.global_scale + (a.gs_per_expert ? e : 0)));

    float acc[4][2][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[mt][nt][r] = 0.f;

    const uint32_t a_lo = kF2B + a_row_chunk(q, c >> 1) + (c & 1) * 8;  // the v9 A layout (see fc1_kernel_ws)
    const uint32_t a_hi = kF2B + a_row_chunk(q + 8, c >> 1) + (c & 1) * 8;
    const uint32_t sfa_off = kF2B + kF2A + (q + ((lane & 1) << 3)) * 4;
    const uint32_t b_off = (ns16 >> 1) * 512 + lane * 16 + (ns16 & 1) * 8;
    const int col_q = ns16 * 16 + q;

    auto compute_stage = [&](const uint8_t* sp, auto kb_tag, auto tail_tag) {  // = v8's arithmetic; all LDS first (see fc1_kernel_ws)
      constexpr int KB0 = decltype(kb_tag)::value;
      constexpr bool TAIL = decltype(tail_tag)::value;
      uint32_t sfb[2], sfa[4];
      uint2 bw[2], alo[4][2], ahi[4][2];
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) bw[kk] = *reinterpret_cast<const uint2*>(sp + kk * 2048 + b_off);
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          alo[mt][kk] = *reinterpret_cast<const uint2*>(sp + ((a_lo + mt * 1024) ^ (kk * 32)));
          ahi[mt][kk] = *reinterpret_cast<const uint2*>(sp + ((a_hi + mt * 1024) ^ (kk * 32)));
        }
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
        spread_e2m1x8(bw[kk].x, b[0][0], b[0][1]);
        spread_e2m1x8(bw[kk].y, b[1][0], b[1][1]);
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          if (mt < m_tiles) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
              qmma_kk<KB0>(kk, acc[mt][nt], alo[mt][kk].x, ahi[mt][kk].x, alo[mt][kk].y, ahi[mt][kk].y, b[nt][0], b[nt][1], sfa[mt], sfb[nt]);
          }
        }
      }
    };
    using I0 = std::integral_constant<int, 0>;
    using I2 = std::integral_constant<int, 2>;
    using F = std::false_type;
    using Tr = std::true_type;
    auto step = [&](auto kb_tag, auto tail_tag) {
      mbar_wait(smem_addr(&bar_full[pos.buf]), pos.par);
      compute_stage(smem + pos.buf * kF2Stage, kb_tag, tail_tag);
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_addr(&bar_empty[pos.buf]));
      pos.advance<STAGES>();
    };
#pragma unroll 1
    for (int t = 0; t < n_stages - 1; t += 2) {
      step(I0{}, F{});
      step(I2{}, F{});
    }
    step(I0{}, Tr{});

    // ---- epilogue = v8's: scale, bf16, token-major route rows
    const int pslot = k & 1;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      if (mt >= m_tiles) continue;
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
          if (route_hint) st_global_hint_v4(dst, make_uint4(o[0], o[1], o[2], o[3]), route_pol);
          else *dst = make_uint4(o[0], o[1], o[2], o[3]);
        }
      }
    }
    if (a.sum_mode) {  // K-12: publish this warp's route stores of the tile to the sum items (one REDG per warp per item)
      __syncwarp();
      if (lane == 0) red_release_gpu_add(a.tile_done + tile, 1);
    }
    ++k;
  }
}

// -------------------------------------------- 6. MK-2: the fused persistent MoE launch (fc1 -> SwiGLU / MXFP8 quant -> fc2), WS >= 20
// One kernel, 1 CTA/SM, 12 warps = v10's CTA (warps 0-7 consumers, 8-11 producers: 8 = the item hand-off + A m-tiles 0-1 + A scale rows
// 0-31, 9 = A m-tiles 2-3 + scale rows 32-63, 10 / 11 = B), one work list built by the front-end's scan CTA: the FC1 items of expert e
// (tile-major, as today) followed by the FC2 items of expert e - LEAD, then the FC2 items of the last LEAD experts (LEAD = the lead in
// experts, VLLM_MOET_B12X_MOE_PREFILL_FAST_WS_LEAD, default 32: an FC1 item takes ~13 FC2 items' time and an expert has 8x more FC2 items
// than FC1 items, so a one-expert lead would make the FC2 producers wait ~50 us per expert; 32 experts of items ~ 2 600 positions ~ 150 us
// of claim time at 4 096 rows, more than one FC1 item). An FC2 item's A rows (the MXFP8 intermediate) are written by the FC1 items of its
// expert on other CTAs: every consumer warp publishes fc1_done[e] += 1 (red.release.gpu) after its FC1 epilogue stores, and the two A
// producer warps of an FC2 item wait for fc1_done[e] == 8 x the expert's FC1 items (ld.acquire.gpu poll with a %globaltimer watchdog: on
// expiry an error record goes to pinned host memory and the CTA traps - a hang becomes a loud launch failure, never a spinning kernel)
// before their first copy; the B producers do not wait, so the expert's W2 tiles stream into the ring during the wait. The stage ring
// (STAGES x the FC1 stage size; an FC2 stage uses the first 8 960 B of a slot in v9's layout) and the item ring run continuously across
// items and across the FC1 / FC2 boundary. Deadlock-freedom: every CTA is resident (grid = 1 CTA/SM), items are claimed in list order and
// an item waits only on FC1 items that precede it in the list (claimed by CTAs that process their items in order) - the earliest blocked
// item always makes progress. Bits by construction: the FC1 consumer is v10's (compute_stage + epilogue verbatim), the FC2 consumer is
// v9's arithmetic with the A fragment read from the v10 unit (the same four words as v9's two LDS.64), the epilogues are v10's / v9's byte
// for byte; only who launches what when and who copies differ. The separate topk_sum_kernel follows (MK/REPORT.md 1.6, milestone 1).
static_assert(kWgConsumersFront == kWgConsumers, "the front-end's fc1_need uses the persistent kernel's consumer warp count");

__device__ __forceinline__ void st_global_cs_v4(uint4* p, const uint4& v) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

struct PersistArgs {
  // FC1 (contiguous rows: a_values [T][K] e4m3, a_scales [T][K/32] ue8m0)
  const uint32_t* a_values;
  const uint8_t* a_scales;
  const uint32_t* w13;
  const uint32_t* w13_sf;
  const float* alpha;
  const float* input_global_scale;
  uint32_t* inter;
  uint8_t* inter_sf;
  // FC2
  const uint32_t* w2;
  const uint8_t* w2_sf;
  const float* token_weights;
  const float* down_alpha;
  const float* global_scale;
  uint32_t* route;
  // common
  const int32_t* row_pair;
  const int4* items;          // the unified list
  int32_t* counts;            // [6] = its length, [3] = the work counter
  int32_t* fc1_done;          // [kMaxE], zeroed by the host's memset
  const int32_t* fc1_need;    // [kMaxE] (the scan CTA)
  int32_t* err_lock;          // one device word (zeroed per call): the first expired watchdog wins the record
  uint32_t* err;              // pinned host memory [8]: set, item, expert, need, seen, smid, blockIdx, 0
  // optional per-item-type statistics (VLLM_MOET_B12X_MOE_PREFILL_FAST_WS_STATS=1; nullptr: off): [0] FC1 item ns (consumer warp 0, from
  // the `full` of its stage 0 to the end of its epilogue), [1] FC1 items, [2] FC2 item ns, [3] FC2 items, [4] consumer idle ns between items
  // (the wait for the next item's stage 0), [5] the A producer's wait on fc1_done ns, [6] CTA span ns, [7] CTAs; summed over the CTAs
  unsigned long long* stats;
  unsigned long long watchdog_ns;
  int K, topk, igs_per_expert, gs_per_expert, out_tiles, dynamic_schedule;
  int route_cs;               // 1: the route rows are written with st.global.cs (evict-first; VLLM_MOET_B12X_MOE_PREFILL_FAST_WS_ROUTE_CS, default 1)
  float limit;
};

// an expired wait: the record (first writer wins the device lock) goes to the pinned host buffer, then the CTA traps - the launch fails
// loudly and the GPU is left clean (never a spinning kernel). what: 1 / 2 = the FC2 wait on fc1_done (A producer role 0 / 1), 10 + site = an
// mbarrier wait (site: 0 consumer full at the item start, 1 consumer full in the K-loop, 2 producer empty, 3 producer bar_meta, 4 terminator)
// (the primitives - red_release_gpu_add, ld_acquire_gpu_s32, globaltimer_ns, smid, fail_record - are in the helpers section: K-12 shares them)
__device__ __noinline__ void ps_fail(const PersistArgs& a, int what, int x, int need, int seen) {
  fail_record(a.err, a.err_lock, a.watchdog_ns, what, x, need, seen);
}

// one mbarrier try_wait (with the suspend-time hint the thread sleeps until the phase completes or the hint expires)
__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity, bool sleep) {
  uint32_t ok;
  if (sleep)
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  else
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
// the persistent kernel's mbarrier wait: the fast path is one try_wait; a wait that does not complete within the watchdog fails loudly
__device__ __forceinline__ void ps_mbar_wait(uint32_t bar, uint32_t parity, bool sleep, const PersistArgs& a, int site, int k) {
  if (mbar_try(bar, parity, sleep)) return;
  const unsigned long long t0 = globaltimer_ns();
  while (!mbar_try(bar, parity, sleep)) {
    if (globaltimer_ns() - t0 > a.watchdog_ns) ps_fail(a, 10 + site, k, (int)parity, 0);
  }
}

// the FC2 item's wait for its expert's FC1 items (lane 0 polls with acquire semantics, the warp follows): bounded by the watchdog;
// returns the ns waited (lane 0; 0 when the counter was already complete)
__device__ __forceinline__ unsigned long long ps_wait_fc1_done(const PersistArgs& a, int e, int item, int lane, int what) {
  unsigned long long waited = 0;
  if (lane == 0) {
    const int need = __ldg(a.fc1_need + e);
    int seen = ld_acquire_gpu_s32(a.fc1_done + e);
    if (seen < need) {
      const unsigned long long t0 = globaltimer_ns();
      while ((seen = ld_acquire_gpu_s32(a.fc1_done + e)) < need) {
        if (globaltimer_ns() - t0 > a.watchdog_ns) ps_fail(a, what, item, need, seen);
        __nanosleep(32);
      }
      waited = globaltimer_ns() - t0;
    }
  }
  __syncwarp();
  return waited;
}

// the hand-off warp's staging of the next item (= producer_stage_next + the token weights at d == 5 / 7 and the kind-aware A offsets): the
// next item's kind is known from d == 4 on (its meta); the ring slot nb is written at d == 6 (meta, pairs, A offsets; bar_meta released
// for the other producers) and d == 7 (token weights: read by the consumers' FC2 epilogue only)
__device__ __forceinline__ void ps_stage_next(ItemRing& ring, ProducerNext& nx, float& tw0, float& tw1, int d, int nb, int lane, int cur_item,
                                              int n_items, const PersistArgs& a, uint64_t* bar_meta_nb) {
  if (d == 0) {
    int c = 0;
    if (lane == 0) c = claim_next_item(a.counts + 3, cur_item, a.dynamic_schedule);
    nx.item = __shfl_sync(0xffffffffu, c, 0);
  }
  if (d == 2) nx.w = (lane < 4 && nx.item < n_items) ? __ldg(reinterpret_cast<const int32_t*>(a.items) + (size_t)nx.item * 4 + lane) : 0;
  if (d == 4) {
    nx.meta = make_int4(__shfl_sync(0xffffffffu, nx.w, 0), __shfl_sync(0xffffffffu, nx.w, 1), __shfl_sync(0xffffffffu, nx.w, 2),
                        __shfl_sync(0xffffffffu, nx.w, 3));
    const int rows = nx.meta.z * 16;
    nx.pr0 = lane < rows ? __ldg(a.row_pair + nx.meta.y + lane) : -1;
    nx.pr1 = lane + 32 < rows ? __ldg(a.row_pair + nx.meta.y + lane + 32) : -1;
  }
  if (d == 5) {
    tw0 = nx.pr0 >= 0 ? __ldg(a.token_weights + nx.pr0) : 0.f;
    tw1 = nx.pr1 >= 0 ? __ldg(a.token_weights + nx.pr1) : 0.f;
  }
  if (d == 6) {
    const bool fc2 = (nx.meta.w & kPsFc2) != 0;
    const uint32_t row_b = fc2 ? (uint32_t)kInterBytes : (uint32_t)a.K;
    if (lane == 0) ring.meta[nb] = nx.meta;
    ring.pairs[nb][lane] = nx.pr0;
    ring.pairs[nb][lane + 32] = nx.pr1;
    ring.aoff[nb][lane] = nx.pr0 >= 0 ? (uint32_t)(fc2 ? nx.pr0 : nx.pr0 / a.topk) * row_b : 0xFFFFFFFFu;
    ring.aoff[nb][lane + 32] = nx.pr1 >= 0 ? (uint32_t)(fc2 ? nx.pr1 : nx.pr1 / a.topk) * row_b : 0xFFFFFFFFu;
    __syncwarp();  // every lane's ring writes precede lane 0's release arrive
    if (lane == 0) mbar_arrive(smem_addr(bar_meta_nb));
  }
  if (d == 7) {
    ring.tw[nb][lane] = tw0;
    ring.tw[nb][lane + 32] = tw1;
  }
}

// N2 = 128-column tiles per FC2 item: 1 = v9's item (stage = 4 KB W2 + 4 KB A + scales in v9's layout, the consumer with a cross-stage
// fragment prefetch), 2 = n256 items (MK-2 round 3): the stage has fc1's shape and layout exactly - B = the two tiles' 4 KB (roles 2 / 3, one
// tile each, like fc1's two projections), A 4 KB at kF1B, the scale words at fc1's offsets - so a consumer warp issues 32 QMMAs per 8
// A-fragment LDS.128 like fc1 (v9's item: 16 per 8; the A fragments, read once per warp, were the LDS co-limiter at 2 warps per sub-partition),
// half the items, and the route rows go out 512 B per row. Bits: per element the same k32 chain and epilogue; only the warp->column map differs.
template <int STAGES, int N2>
__global__ void __launch_bounds__(kWgThreads, 1) moe_persistent_kernel(PersistArgs a) {
  // an FC2 item has 9 stages and its hand-off writes the ring slot at stages 7 / 8: the consumers must have left the slot's previous
  // reader (the epilogue of the item before the current one), which the `empty` wait of stage 7 guarantees only for STAGES <= 7
  static_assert(STAGES >= 2 && STAGES <= 7, "the FC2 hand-off needs STAGES <= 7");
  static_assert(N2 == 1 || N2 == 2, "FC2 items of one or two 128-column tiles");
  constexpr int NCW = kWgConsumers;
  constexpr uint32_t kF2TileBytes = 128 * (kSlices * 4 - 2) * 16;  // 36 KB of W2 per 128-column tile (18 k32)
  constexpr uint32_t kF2TileSfBytes = 128 * (kSlices * 4 - 2);     // 2304 B of scales per tile
  // the FC2 stage layout inside the kF1Stage slot: N2 == 1 -> v9's (B 4 KB, A, SFA, SFB); N2 == 2 -> fc1's (B 2 x 4 KB, A, SFA, SFB 2 x 512 B)
  constexpr uint32_t F2_A = N2 == 2 ? (uint32_t)kF1B : (uint32_t)kF2B;
  constexpr uint32_t F2_SFA = F2_A + (uint32_t)kF2A;
  constexpr uint32_t F2_SFB = F2_SFA + (uint32_t)kF2SFA;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ ItemRing ring;
  __shared__ __align__(8) uint64_t bar_full[STAGES], bar_empty[STAGES];
  __shared__ __align__(8) uint64_t bar_meta[2];
  __shared__ float s_amax[4][NCW][16];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int K = a.K, n1_stages = K / 64;
  const uint32_t sbase0 = smem_addr(smem);
  const int n_items = a.counts[6];
  if (tid < STAGES) {
    mbar_init(smem_addr(&bar_full[tid]), 32 * kWgProducers + 1);
    mbar_init(smem_addr(&bar_empty[tid]), NCW);
  }
  if (tid < 2) mbar_init(smem_addr(&bar_meta[tid]), 1);
  if (warp == NCW) {  // the first item straight from global memory
    int4 m = make_int4(0, 0, 0, 0);
    const int item = blockIdx.x;
    if (item < n_items) {
      const int w = lane < 4 ? __ldg(reinterpret_cast<const int32_t*>(a.items) + (size_t)item * 4 + lane) : 0;
      m = make_int4(__shfl_sync(0xffffffffu, w, 0), __shfl_sync(0xffffffffu, w, 1), __shfl_sync(0xffffffffu, w, 2),
                    __shfl_sync(0xffffffffu, w, 3));
    }
    const int rows = m.z * 16;
    const bool fc2 = (m.w & kPsFc2) != 0;
    const uint32_t row_b = fc2 ? (uint32_t)kInterBytes : (uint32_t)K;
    const int p0 = lane < rows ? __ldg(a.row_pair + m.y + lane) : -1;
    const int p1 = lane + 32 < rows ? __ldg(a.row_pair + m.y + lane + 32) : -1;
    if (lane == 0) ring.meta[0] = m;
    ring.pairs[0][lane] = p0;
    ring.pairs[0][lane + 32] = p1;
    ring.aoff[0][lane] = p0 >= 0 ? (uint32_t)(fc2 ? p0 : p0 / a.topk) * row_b : 0xFFFFFFFFu;
    ring.aoff[0][lane + 32] = p1 >= 0 ? (uint32_t)(fc2 ? p1 : p1 / a.topk) * row_b : 0xFFFFFFFFu;
    ring.tw[0][lane] = p0 >= 0 ? __ldg(a.token_weights + p0) : 0.f;
    ring.tw[0][lane + 32] = p1 >= 0 ? __ldg(a.token_weights + p1) : 0.f;
  }
  __syncthreads();  // barriers initialised, item 0 in the ring; from here on the roles only meet through the barriers

  if (warp >= NCW) {
    // ================================================================================ the producer warpgroup (warps 8-11)
    const int role = warp - NCW;  // 0 / 1: A m-tiles (2 role, 2 role + 1) + scale rows 32 role..; 2 / 3: B
    const uint32_t wproj_b = (uint32_t)kN * (K / 32) * 16;
    const uint32_t sproj_b = (uint32_t)kN * (K / 32);
    const uint8_t* a1_base = reinterpret_cast<const uint8_t*>(a.a_values);
    const uint8_t* a2_base = reinterpret_cast<const uint8_t*>(a.inter);
    RingPos pos;
    int round0 = STAGES, k = 0;
    if (role == 0) {
      int item = blockIdx.x;
      ProducerNext nx;
      float tw0 = 0.f, tw1 = 0.f;
      unsigned long long wait_ns = 0;
      while (true) {
        const int4 it = ring.meta[k & 1];
        if (it.z == 0) {  // no item: one more phase of `full` so that the consumers read the terminator, then leave
          if (round0 == 0) ps_mbar_wait(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u, true, a, 2, k);
          __syncwarp();
          if (lane == 0) mbar_arrive(smem_addr(&bar_full[pos.buf]));
          cp_async_mbar_arrive_noinc(smem_addr(&bar_full[pos.buf]));
          if (a.stats != nullptr && lane == 0) atomicAdd(a.stats + 5, wait_ns);
          break;
        }
        const bool fc2 = (it.w & kPsFc2) != 0;
        const int e = it.x, slot = k & 1, nb = slot ^ 1;
        const int n_stages = fc2 ? kF2Stages : n1_stages;
        // the next item is claimed 10 stages before the current one ends (FC1: K >= 640 -> >= 10 stages) / at stage 1 of an FC2 item (9 stages:
        // d = 0..7 at stages 1..8 - v9's fc2 hand-off)
        const int claim_at = fc2 ? 1 : n1_stages - 10;
        const uint32_t a_region = fc2 ? F2_A : (uint32_t)kF1B;
        const uint32_t sf_region = fc2 ? F2_SFA : (uint32_t)(kF1B + kF1A);
        const uint8_t* a_base = fc2 ? a2_base : a1_base;
        const uint8_t* sf_base = fc2 ? a.inter_sf : a.a_scales;
        WgRows rows;
        rows.load(ring.aoff[slot], 0, lane);
#pragma unroll 1
        for (int t = 0; t < n_stages; ++t) {
          if (round0 > 0) --round0;
          else ps_mbar_wait(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u, true, a, 2, k);
          if (fc2 && t == 0) wait_ns += ps_wait_fc1_done(a, e, item, lane, 1);  // the expert's FC1 items are complete: the A rows exist
          const uint32_t sb = sbase0 + pos.buf * kF1Stage, full = smem_addr(&bar_full[pos.buf]);
          ps_stage_next(ring, nx, tw0, tw1, t - claim_at, nb, lane, item, n_items, a, &bar_meta[nb]);
          __syncwarp();  // the ring writes of this stage (if any) precede lane 0's release arrive
          if (lane == 0) mbar_arrive(full);
          wg_copy_a(sb + a_region, a_base, rows, 0, t, lane);
          {
            const uint32_t off = rows.sf;
            const bool on = off != 0xFFFFFFFFu;
            cp_async4_zfill(sb + sf_region + lane * 4, sf_base + ((on ? off : 0u) >> 5) + (t >> 1) * 4, on ? 4 : 0);
          }
          cp_async_mbar_arrive_noinc(full);
          pos.advance<STAGES>();
        }
        item = nx.item;
        ++k;
      }
    } else {
      while (true) {
        // producer 8 has written meta[k & 1] (its arrive at d == 6 of item k - 1): bar_meta[s] serves the items k = s, s + 2, ... with
        // k >= 1, i.e. its phase for item k is ((k - 1) >> 1) & 1 (a fresh barrier returns at once on the other parity: K-4's v9j hang)
        if (k > 0) ps_mbar_wait(smem_addr(&bar_meta[k & 1]), (uint32_t)((k - 1) >> 1) & 1u, true, a, 3, k);
        const int4 it = ring.meta[k & 1];
        if (it.z == 0) {
          if (round0 == 0) ps_mbar_wait(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u, true, a, 2, k);
          cp_async_mbar_arrive_noinc(smem_addr(&bar_full[pos.buf]));
          break;
        }
        const bool fc2 = (it.w & kPsFc2) != 0;
        const int e = it.x, tile = it.w & 0xFF;
        const int n_stages = fc2 ? kF2Stages : n1_stages;
        const bool wide = tile != kN1Tiles - 1;  // FC1 only
        const int proj = role - 2;               // FC1: the B projection of roles 2 / 3; FC2: the B half
        // FC2: N2 == 2 -> role 2 / 3 copies the whole tile `tile + proj` (fc1's two-projection pattern); N2 == 1 -> the two halves of one tile
        const int f2_tile = tile + (N2 == 2 && proj > 0 ? proj : 0);
        const uint8_t* iss_b = fc2 ? reinterpret_cast<const uint8_t*>(a.w2) + ((size_t)e * a.out_tiles + f2_tile) * kF2TileBytes
                                   : reinterpret_cast<const uint8_t*>(a.w13) + (size_t)e * 2 * wproj_b + (size_t)tile * 128 * (K / 32) * 16 +
                                         (size_t)(proj > 0 ? proj : 0) * wproj_b;
        const uint8_t* iss_sfb = fc2 ? a.w2_sf + ((size_t)e * a.out_tiles + f2_tile) * kF2TileSfBytes
                                     : reinterpret_cast<const uint8_t*>(a.w13_sf) + (size_t)e * 2 * sproj_b + (size_t)tile * 128 * (K / 32) +
                                           (size_t)(proj > 0 ? proj : 0) * sproj_b;
        const uint32_t a_region = fc2 ? F2_A : (uint32_t)kF1B;
        const uint32_t sf_region = fc2 ? F2_SFA : (uint32_t)(kF1B + kF1A);
        const uint32_t sfb_region = fc2 ? F2_SFB : (uint32_t)(kF1B + kF1A + kF1SFA);
        const uint8_t* a_base = fc2 ? a2_base : a1_base;
        const uint8_t* sf_base = fc2 ? a.inter_sf : a.a_scales;
        WgRows rows;
        if (role == 1) rows.load(ring.aoff[k & 1], 1, lane);
#pragma unroll 1
        for (int t = 0; t < n_stages; ++t) {
          if (round0 > 0) --round0;
          else ps_mbar_wait(smem_addr(&bar_empty[pos.buf]), pos.par ^ 1u, true, a, 2, k);
          const uint32_t sb = sbase0 + pos.buf * kF1Stage, full = smem_addr(&bar_full[pos.buf]);
          if (role == 1) {
            if (fc2 && t == 0) ps_wait_fc1_done(a, e, k, lane, 2);
            wg_copy_a(sb + a_region, a_base, rows, 1, t, lane);
            const uint32_t off = rows.sf;
            const bool on = off != 0xFFFFFFFFu;
            cp_async4_zfill(sb + sf_region + (32 + lane) * 4, sf_base + ((on ? off : 0u) >> 5) + (t >> 1) * 4, on ? 4 : 0);
          } else if (fc2 && N2 == 2) {  // B: the whole 4 KB of W2 tile `tile + proj` of this K64 stage + its column scales (512 B; the tail slice 256)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              const uint32_t o = (lane + 32 * i) * 16;
              cp_async16(sb + proj * 4096 + o, iss_b + (size_t)t * 4096 + o);
            }
            {
              const int s = t >> 1;
              if (lane < (s == kSlices - 1 ? 16 : 32)) cp_async16(sb + sfb_region + proj * 512 + lane * 16, iss_sfb + s * 512 + lane * 16);
            }
          } else if (fc2) {  // B: the 4 KB of W2 of this K64 stage, halves by role (128 x 16 B each); the tile's column scales of the K128 slice by role 2
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const uint32_t o = (lane + 32 * (i + 4 * proj)) * 16;
              cp_async16(sb + o, iss_b + (size_t)t * 4096 + o);
            }
            if (proj == 0) {
              const int s = t >> 1;
              if (lane < (s == kSlices - 1 ? 16 : 32)) cp_async16(sb + sfb_region + lane * 16, iss_sfb + s * 512 + lane * 16);
            }
          } else if (wide) {  // FC1 B projection `proj`: 4 KB contiguous per stage + its 512 B of column scales per K128
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              const uint32_t o = (lane + 32 * i) * 16;
              cp_async16(sb + proj * 4096 + o, iss_b + (size_t)t * 4096 + o);
            }
            cp_async16(sb + sfb_region + proj * 512 + lane * 16, iss_sfb + (t >> 1) * 512 + lane * 16);
          } else {  // the 64-column tile: 2 x 1 KB per projection per stage, 256 B of column scales
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const uint32_t j = lane + 32 * i, kk = j >> 6, rem = j & 63;
              cp_async16(sb + proj * 4096 + kk * 2048 + rem * 16, iss_b + (size_t)t * 2048 + kk * 1024 + rem * 16);
            }
            if (lane < 16) cp_async16(sb + sfb_region + proj * 512 + lane * 16, iss_sfb + (t >> 1) * 256 + lane * 16);
          }
          cp_async_mbar_arrive_noinc(full);
          pos.advance<STAGES>();
        }
        ++k;
      }
    }
    cp_async_wait<0>();
    return;
  }

  // ================================================================================ the consumer warps (warps 0-7)
  const int q = lane >> 2, c = lane & 3;
  RingPos pos;
  int k = 0;
  // statistics (warp 0 lane 0 of the CTA): per-item-type time, idle between items, the CTA's span
  const bool keep_stats = a.stats != nullptr && tid == 0;
  unsigned long long st_t0 = 0, st_prev_end = 0, st_fc1 = 0, st_fc2 = 0, st_idle = 0, st_n1 = 0, st_n2 = 0;
  if (keep_stats) st_prev_end = st_t0 = globaltimer_ns();
  while (true) {
    ps_mbar_wait(smem_addr(&bar_full[pos.buf]), pos.par, false, a, 0, k);  // stage 0 of item k (or the terminator): meta[k & 1] is published
    const int4 it = ring.meta[k & 1];
    const int e = it.x, m_tiles = it.z;
    unsigned long long st_start = 0;
    if (keep_stats) { st_start = globaltimer_ns(); st_idle += st_start - st_prev_end; }
    if (m_tiles == 0) break;
    const bool fc2 = (it.w & kPsFc2) != 0;
    const int tile = it.w & 0xFF;
    const int pslot = k & 1;
    if (!fc2) {
      // ---------------------------------------------------------------- FC1 item = v10's consumer (fc1_kernel_wg) verbatim
      const bool wide = tile != kN1Tiles - 1;
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

      const uint32_t a_off = kF1B + a10_frag(0, 0, q, c);                      // + mt * 1024 + kk * 512
      const uint32_t sfa_off = kF1B + kF1A + (q + ((lane & 1) << 3)) * 4;     // + mt * 64
      const uint32_t b_off = (ns16 >> 1) * 512 + lane * 16 + (ns16 & 1) * 8;  // + p * 4096 + kk * 2048
      const uint32_t sfb_off = kF1B + kF1A + kF1SFA + (ns16 * 16 + q) * 4;    // + p * 512 + nt * 32

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
                  qmma_kk<KB0>(kk, acc[mt][p][nt], af.x, af.z, af.y, af.w, b[p][nt][0], b[p][nt][1], sfa[mt], sfb[p][nt]);
            }
          }
        }
      };
      auto step = [&](auto kb_tag) {
        ps_mbar_wait(smem_addr(&bar_full[pos.buf]), pos.par, false, a, 1, k);
        compute_stage(smem + pos.buf * kF1Stage, kb_tag);
        __syncwarp();  // the warp's LDS of this buffer precede lane 0's release arrive
        if (lane == 0) mbar_arrive(smem_addr(&bar_empty[pos.buf]));
        pos.advance<STAGES>();
      };
#pragma unroll 1
      for (int t = 0; t < n1_stages; t += 2) {
        step(std::integral_constant<int, 0>{});
        step(std::integral_constant<int, 2>{});
      }

      // ---- epilogue = v10's (one named barrier of the consumers per item)
      float v[4][2][4], am[4][2];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        am[mt][0] = am[mt][1] = 0.f;
        if (mt >= m_tiles) continue;  // CTA-uniform
        if (act[mt]) {
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
              const float x0 = activated_value(acc[mt][1][nt][2 * hf], acc[mt][0][nt][2 * hf], alpha, a.limit);
              const float x1 = activated_value(acc[mt][1][nt][2 * hf + 1], acc[mt][0][nt][2 * hf + 1], alpha, a.limit);
              const uint32_t pk = pack_bf16x2(x0, x1);
              v[mt][hf][2 * nt] = __uint_as_float(pk << 16);
              v[mt][hf][2 * nt + 1] = __uint_as_float(pk & 0xFFFF0000u);
            }
            float amax = 0.f;
#pragma unroll
            for (int kx = 0; kx < 4; ++kx) amax = fmax_plain(amax, fabs_plain(v[mt][hf][kx]));
            amax = fmax_plain(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmax_plain(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            am[mt][hf] = amax;
            if (c == 0) s_amax[mt][warp][q + hf * 8] = amax;
          }
        }
      }
      consumers_sync<NCW>();
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        if (mt >= m_tiles) continue;  // CTA-uniform
        if (act[mt]) {
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const int row = mt * 16 + q + hf * 8;
            const int pair = ring.pairs[pslot][row];
            const float amax = fmax_plain(am[mt][hf], s_amax[mt][warp ^ 1][q + hf * 8]);
            float inv;
            const uint32_t byte = mx_scale(amax, inv);
            uint32_t h[4] = {cvt_e4m3x2(fmul(v[mt][hf][0], inv), fmul(v[mt][hf][1], inv)),
                             cvt_e4m3x2(fmul(v[mt][hf][2], inv), fmul(v[mt][hf][3], inv)), 0u, 0u};
            uint32_t o0, o1;
            quad_transpose16(h, c, o0, o1);
            if (pair >= 0) {
              if (c < 2) {
                uint2* dst = reinterpret_cast<uint2*>(reinterpret_cast<uint8_t*>(a.inter) + (size_t)pair * kInterBytes +
                                                     tile * 128 + ns16 * 16 + c * 8);
                *dst = make_uint2(o0, o1);
              }
              if ((ns16 & 1) == 0) {
                uint8_t* sf = a.inter_sf + (size_t)pair * (kSlices * 4) + tile * 4;
                if (c == 0) sf[ns16 >> 1] = (uint8_t)byte;
                if (!wide && c == 1) sf[(ns16 >> 1) + 2] = 0;
              }
            }
          }
        }
      }
      // this warp's rows of the expert's intermediate are written: one arrival of the 8 per FC1 item (the FC2 producers wait for them)
      __syncwarp();
      if (lane == 0) red_release_gpu_add(a.fc1_done + e, 1);
    } else {
      // ---------------------------------------------------------------- FC2 item
      const int ns16 = warp;  // n16 slice of a 128-column tile; every warp takes all m-tiles
      const float down_scale = fmul(__ldg(a.down_alpha + e), __ldg(a.global_scale + (a.gs_per_expert ? e : 0)));
      const uint32_t a_off = F2_A + a10_frag(0, 0, q, c);                        // + mt * 1024 + kk * 512
      const uint32_t sfa_off = F2_SFA + (q + ((lane & 1) << 3)) * 4;             // + mt * 64
      const uint32_t b_off = (ns16 >> 1) * 512 + lane * 16 + (ns16 & 1) * 8;    // (+ p * 4096) + kk * 2048
      const int col_q = ns16 * 16 + q;
      using I0 = std::integral_constant<int, 0>;
      using I2 = std::integral_constant<int, 2>;
      if constexpr (N2 == 2) {
        // ---- n256 item = fc1's compute_stage shape (B of both tiles, the A fragment of each m-tile once, 32 QMMAs per stage), v9's arithmetic
        float acc[4][2][2][4];  // [mt][tile p][nt][4]
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
          for (int pp = 0; pp < 2; ++pp)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
              for (int r = 0; r < 4; ++r) acc[mt][pp][nt][r] = 0.f;
        auto compute_stage = [&](const uint8_t* sp, auto kb_tag, auto tail_tag) {
          constexpr int KB0 = decltype(kb_tag)::value;
          constexpr bool TAIL = decltype(tail_tag)::value;
          uint32_t sfb[2][2], sfa[4];
#pragma unroll
          for (int pp = 0; pp < 2; ++pp)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
              if constexpr (TAIL) sfb[pp][nt] = *reinterpret_cast<const uint16_t*>(sp + F2_SFB + pp * 512 + (col_q + nt * 8) * 2);
              else sfb[pp][nt] = *reinterpret_cast<const uint32_t*>(sp + F2_SFB + pp * 512 + (col_q + nt * 8) * 4);
            }
#pragma unroll
          for (int mt = 0; mt < 4; ++mt) sfa[mt] = *reinterpret_cast<const uint32_t*>(sp + sfa_off + mt * 64);
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            uint32_t b[2][2][2];
#pragma unroll
            for (int pp = 0; pp < 2; ++pp) {
              const uint2 w = *reinterpret_cast<const uint2*>(sp + pp * 4096 + kk * 2048 + b_off);
              spread_e2m1x8(w.x, b[pp][0][0], b[pp][0][1]);
              spread_e2m1x8(w.y, b[pp][1][0], b[pp][1][1]);
            }
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
              if (mt < m_tiles) {
                const uint4 af = *reinterpret_cast<const uint4*>(sp + a_off + mt * 1024 + kk * 512);
#pragma unroll
                for (int pp = 0; pp < 2; ++pp)
#pragma unroll
                  for (int nt = 0; nt < 2; ++nt)
                    qmma_kk<KB0>(kk, acc[mt][pp][nt], af.x, af.z, af.y, af.w, b[pp][nt][0], b[pp][nt][1], sfa[mt], sfb[pp][nt]);
              }
            }
          }
        };
        auto step = [&](auto kb_tag, auto tail_tag) {
          ps_mbar_wait(smem_addr(&bar_full[pos.buf]), pos.par, false, a, 1, k);
          compute_stage(smem + pos.buf * kF1Stage, kb_tag, tail_tag);
          __syncwarp();
          if (lane == 0) mbar_arrive(smem_addr(&bar_empty[pos.buf]));
          pos.advance<STAGES>();
        };
#pragma unroll 1
        for (int t = 0; t < kF2Stages - 1; t += 2) {  // slices 0-3: stages 0..7
          step(I0{}, std::false_type{});
          step(I2{}, std::false_type{});
        }
        step(I0{}, std::true_type{});  // stage 8: k32 16, 17 (the tail slice's u16 scale words); the zero tail is not run

        // ---- epilogue = v9's per tile: scale, bf16, token-major route rows (512 contiguous bytes per row over the two tiles)
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          if (mt >= m_tiles) continue;
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const int row = mt * 16 + q + hf * 8;
            const int pair = ring.pairs[pslot][row];
            const float scale = fmul(down_scale, ring.tw[pslot][row]);
#pragma unroll
            for (int pp = 0; pp < 2; ++pp) {
              uint32_t w[4], o[4];
              w[0] = pack_bf16x2(fmul(scale, acc[mt][pp][0][2 * hf]), fmul(scale, acc[mt][pp][0][2 * hf + 1]));
              w[1] = pack_bf16x2(fmul(scale, acc[mt][pp][1][2 * hf]), fmul(scale, acc[mt][pp][1][2 * hf + 1]));
              w[2] = w[3] = 0u;
              quad_transpose32(w, c, o);
              if (pair >= 0 && c < 2) {  // streaming store (MK-2 round 5): the route rows are read back by topk_sum only
                uint4* dst = reinterpret_cast<uint4*>(a.route + (size_t)pair * (a.K / 2) + ((tile + pp) * 128 + ns16 * 16 + c * 8) / 2);
                if (a.route_cs) st_global_cs_v4(dst, make_uint4(o[0], o[1], o[2], o[3]));
                else *dst = make_uint4(o[0], o[1], o[2], o[3]);
              }
            }
          }
        }
      } else {
      // ---- v9's item (one tile) as a register pipeline over the stages (K-4's v9f idea; it did not fit fc1's 64 accumulators at 168 registers,
      // fc2's 32 + 2 x 42 fit): every LDS of stage t + 1 (B words, the four A fragments of both k32 blocks, the scale words - the tail slice's
      // u16 column scales read as u16) is issued before the QMMAs of stage t, so with 2 consumer warps per sub-partition the LDS latency
      // hides behind the same warp's QMMAs; a stage's buffer is released after its QMMAs (K-4's rule: the release follows the last read of
      // the buffer, here the fragment loads of the previous iteration, by program order).
      float acc[4][2][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int r = 0; r < 4; ++r) acc[mt][nt][r] = 0.f;
      struct Frags2 {
        uint4 af[4][2];
        uint2 bw[2];
        uint32_t sfa[4], sfb[2];
      };
      auto load_frags2 = [&](const uint8_t* sp, Frags2& f, bool tail) {
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) f.bw[kk] = *reinterpret_cast<const uint2*>(sp + kk * 2048 + b_off);
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) f.af[mt][kk] = *reinterpret_cast<const uint4*>(sp + a_off + mt * 1024 + kk * 512);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          if (tail) f.sfb[nt] = *reinterpret_cast<const uint16_t*>(sp + F2_SFB + (col_q + nt * 8) * 2);
          else f.sfb[nt] = *reinterpret_cast<const uint32_t*>(sp + F2_SFB + (col_q + nt * 8) * 4);
        }
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) f.sfa[mt] = *reinterpret_cast<const uint32_t*>(sp + sfa_off + mt * 64);
      };
      auto mma_stage2 = [&](const Frags2& f, auto kb_tag) {
        constexpr int KB0 = decltype(kb_tag)::value;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          uint32_t b[2][2];
          spread_e2m1x8(f.bw[kk].x, b[0][0], b[0][1]);
          spread_e2m1x8(f.bw[kk].y, b[1][0], b[1][1]);
#pragma unroll
          for (int mt = 0; mt < 4; ++mt) {
            if (mt < m_tiles) {
#pragma unroll
              for (int nt = 0; nt < 2; ++nt)
                qmma_kk<KB0>(kk, acc[mt][nt], f.af[mt][kk].x, f.af[mt][kk].z, f.af[mt][kk].y, f.af[mt][kk].w, b[nt][0], b[nt][1], f.sfa[mt], f.sfb[nt]);
            }
          }
        }
      };
      Frags2 F0, F1;
      // stage 0's `full` was waited for at the top of the item (meta): load its fragments
      load_frags2(smem + pos.buf * kF1Stage, F0, false);
      auto pipe_step = [&](Frags2& cur, Frags2& nxt, bool has_next, bool next_tail, auto kb_tag) {
        const int cur_buf = pos.buf;
        pos.advance<STAGES>();
        if (has_next) {
          ps_mbar_wait(smem_addr(&bar_full[pos.buf]), pos.par, false, a, 1, k);
          load_frags2(smem + pos.buf * kF1Stage, nxt, next_tail);
        }
        mma_stage2(cur, kb_tag);
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_addr(&bar_empty[cur_buf]));
      };
#pragma unroll 1
      for (int t = 0; t < kF2Stages - 1; t += 2) {  // slices 0-3: stages 0..7 (stage 7 prefetches stage 8 = the tail slice)
        pipe_step(F0, F1, true, false, I0{});
        pipe_step(F1, F0, true, t + 2 == kF2Stages - 1, I2{});
      }
      pipe_step(F0, F1, false, false, I0{});  // stage 8: k32 16, 17 (the tail slice's u16 scale words); the zero tail (k32 18, 19) is not run

      // ---- epilogue = v9's: scale, bf16, token-major route rows
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        if (mt >= m_tiles) continue;
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
            if (a.route_cs) st_global_cs_v4(dst, make_uint4(o[0], o[1], o[2], o[3]));
            else *dst = make_uint4(o[0], o[1], o[2], o[3]);
          }
        }
      }
      }
    }
    if (keep_stats) {
      const unsigned long long t1 = globaltimer_ns();
      if (fc2) { st_fc2 += t1 - st_start; ++st_n2; } else { st_fc1 += t1 - st_start; ++st_n1; }
      st_prev_end = t1;
    }
    ++k;
  }
  if (keep_stats) {
    atomicAdd(a.stats + 0, st_fc1); atomicAdd(a.stats + 1, st_n1); atomicAdd(a.stats + 2, st_fc2); atomicAdd(a.stats + 3, st_n2);
    atomicAdd(a.stats + 4, st_idle); atomicAdd(a.stats + 6, globaltimer_ns() - st_t0); atomicAdd(a.stats + 7, 1ull);
  }
}

}  // namespace b12x_moe_prefill_fast

// ------------------------------------------------------------------------------------------------------ host
namespace {

// The v9 producer-warp kernels (K-4): VLLM_MOET_B12X_MOE_PREFILL_FAST_WS1=<mode> selects fc1's (contiguous rows only),
// _WS2=<mode> fc2's; 0 = the v8 kernel (the defaults until the in-situ measurement). A mode is the ring
// (stages, CTAs per SM) and, for fc1, the consumer layout: 1 = (3, 2) with 8 consumer warps [2 CTAs x 9 warps: ptxas caps the
// registers at 96 - fc1 spills, fc2 (93 registers) does not], 2 = (6, 1), 3 = (4, 1), 4 = (7, 1) with 8 consumer warps [fc2 at (4, 1)
// still fits 2 CTAs/SM by shared memory], 5 = (6, 1), 6 = (4, 1), 7 = (7, 1) with 16 consumer warps x 2 m-tiles (fc1 only: 17 warps,
// 1 CTA/SM, 4 consumer warps per sub-partition); fc1 modes 8 = (3, 2), 9 = (4, 1), 10 = (6, 1), 11 = (7, 1) are the v10 warpgroup kernel
// fc1_kernel_wg (K-5: 12 warps; at 2 CTAs/SM setmaxnreg = 16 consumer warps per SM at 104 registers, at 1 CTA/SM 148 registers).
// Every configuration computes the same bits (harness).
static int ws_env(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : 0;
}
// The occupancy guard. The grid of a WS launch is sms x cudaOccupancyMaxActiveBlocksPerMultiprocessor(kernel): as many CTAs per SM as
// the build's register count leaves. fc2_kernel_ws<4, 1> (9 warps) is meant to run two per SM, and the SM grants that up to 96
// registers per thread (18 warps -> 5 in one sub-partition -> 65 536 / (5 x 32) = 102, in steps of 8); the kernel uses exactly 96.
// One register more - an edit, or another ptxas after an image rebuild - and the query answers 1: half the grid, FC2 10-15 % slower,
// no error anywhere. So every WS launch family records what it got at its first launch (occupancy_facts()) next to the CTAs per SM
// its mode was measured with, where that is on record (0 = not recorded, no check); the Python module prints one WARNING per kernel
// that starts with fewer, and VLLM_MOET_B12X_MOE_PREFILL_FAST_OCC_STRICT=1 (debug) refuses the launch instead. The grid is what it was.
// _OCC_EXPECT_FC1 / _OCC_EXPECT_FC2 replace the expectation (tests).
struct OccFact { const char* kernel; int s, mb, regs, occ, expect, sms; };
static std::vector<OccFact>& occ_facts() { static std::vector<OccFact> v; return v; }
static int occ_expect(const char* env_name, int recorded) { const char* v = std::getenv(env_name); return v ? std::atoi(v) : recorded; }
template <typename Kernel>
static void occ_check(const char* kernel_name, Kernel kernel, bool& noted, int s, int mb, int occ, int expect, int sms) {
  static const bool strict = ws_env("VLLM_MOET_B12X_MOE_PREFILL_FAST_OCC_STRICT") == 1;
  if (noted && !strict) return;
  cudaFuncAttributes at{};
  int regs = -1;
  if (cudaFuncGetAttributes(&at, kernel) == cudaSuccess) regs = at.numRegs; else cudaGetLastError();
  if (!noted) { noted = true; occ_facts().push_back({kernel_name, s, mb, regs, occ, expect, sms}); }
  TORCH_CHECK(!(strict && expect > 0 && occ < expect), kernel_name, "<", s, ", ", mb, "> would launch with ", occ, " CTA(s) per SM where its mode was "
              "measured with ", expect, " (", regs, " registers per thread in this build): refused under VLLM_MOET_B12X_MOE_PREFILL_FAST_OCC_STRICT=1");
}
// K-13 (R0's r29 task A): runtime A/B overrides of the level-B switches (set_runtime_switches from Python: -1 = the env value read once
// at the first call, as before; the switched paths are bit-identical alternatives of the same compiled kernels, so a flip between two
// calls changes no bits). Plain ints: written by the Python thread between forwards, read by the same thread at the next call.
static int g_rt_ws1 = -1, g_rt_sum_items = -1, g_rt_sum_discard = -1, g_rt_sum_route_hint = -1;
static int ws1_mode() { static const int m = ws_env("VLLM_MOET_B12X_MOE_PREFILL_FAST_WS1"); return g_rt_ws1 >= 0 ? g_rt_ws1 : m; }
static int ws2_mode() { static const int m = ws_env("VLLM_MOET_B12X_MOE_PREFILL_FAST_WS2"); return m; }
// K-13 (task B): activation counters of the switched paths, one add per call (moe_prefill_fast_counters() reads them: calls / rows through
// fc1 v10 (WS1 >= 8) vs fc1 v8/v9 (WS1 < 8), fc2 with the sum items vs fc2 + the separate topk_sum_kernel, the persistent launch)
static unsigned long long g_cnt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
// MK-2: VLLM_MOET_B12X_MOE_PREFILL_FAST_WS >= 20 = the fused persistent launch (moe_persistent_kernel: fc1 + SwiGLU/quant + fc2 as one kernel,
// 1 CTA/SM x 12 warps, the separate topk_sum_kernel after it): 20 = 4 stages, 21 = 6, 22 = 7, 23 = 3, 24 = 5, 25 = 2 (the ring depth);
// _WS_LEAD = the lead in experts of the work list (default 32), _WS_MIN_ROWS = below this many rows the separate kernels run (default 0),
// _WS_WATCHDOG_MS = the bound of every FC2 wait on its expert's FC1 items (default 2000; expiry = an error record + a trapped launch).
// Contiguous rows only (lda == K). Every configuration computes the same bits.
static int ws_mode() { static const int m = ws_env("VLLM_MOET_B12X_MOE_PREFILL_FAST_WS"); return m; }
static int ws_lead() { const char* v = std::getenv("VLLM_MOET_B12X_MOE_PREFILL_FAST_WS_LEAD"); static const int m = v ? std::atoi(v) : 64; return m; }
static int ws_min_rows() { static const int m = ws_env("VLLM_MOET_B12X_MOE_PREFILL_FAST_WS_MIN_ROWS"); return m; }
static int ws_watchdog_ms() { const char* v = std::getenv("VLLM_MOET_B12X_MOE_PREFILL_FAST_WS_WATCHDOG_MS"); static const int m = v ? std::atoi(v) : 2000; return m; }
static bool ws_stats_enabled() { static const bool on = ws_env("VLLM_MOET_B12X_MOE_PREFILL_FAST_WS_STATS") == 1; return on; }
static int ws_route_cs() { const char* v = std::getenv("VLLM_MOET_B12X_MOE_PREFILL_FAST_WS_ROUTE_CS"); static const int m = v ? std::atoi(v) : 1; return m; }
// the per-item-type statistics of the persistent kernel (device memory, 8 x u64, zeroed before every launch when enabled; moe_prefill_ws_stats()
// copies them to the host - a synchronisation, bench only)
static unsigned long long* ps_stats_dev() {
  static unsigned long long* d = [] {
    unsigned long long* p = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&p), 64) != cudaSuccess) { cudaGetLastError(); return static_cast<unsigned long long*>(nullptr); }
    cudaMemset(p, 0, 64);
    return p;
  }();
  return d;
}
// the persistent kernel's error record lives in pinned host memory (readable after a trapped launch; moe_prefill_ws_error() returns it)
static uint32_t* ps_err_host() {
  static uint32_t* h = [] {
    uint32_t* p = nullptr;
    if (cudaHostAlloc(reinterpret_cast<void**>(&p), 256, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) { cudaGetLastError(); return static_cast<uint32_t*>(nullptr); }
    memset(p, 0, 256);
    return p;
  }();
  return h;
}
static uint32_t* ps_err_dev() {
  static uint32_t* d = [] {
    uint32_t* p = nullptr;
    uint32_t* h = ps_err_host();
    if (h == nullptr || cudaHostGetDevicePointer(reinterpret_cast<void**>(&p), h, 0) != cudaSuccess) { cudaGetLastError(); return static_cast<uint32_t*>(nullptr); }
    return p;
  }();
  return d;
}
template <int S, int N2>
struct PsLaunch {  // MK-2: moe_persistent_kernel<S, N2> (12 warps, 1 CTA/SM; the stage slot is the FC1 stage size)
  static constexpr int kT = b12x_moe_prefill_fast::kWgThreads;
  static constexpr int kN2 = N2;
  static int grid(int sms) {
    using namespace b12x_moe_prefill_fast;
    int occ = 0;
    static bool done = false;
    if (!done) { cudaFuncSetAttribute(moe_persistent_kernel<S, N2>, cudaFuncAttributeMaxDynamicSharedMemorySize, S * kF1Stage); done = true; }
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, moe_persistent_kernel<S, N2>, kT, S * kF1Stage);
    return sms * (occ > 0 ? occ : 1);  // every CTA resident (the deadlock-freedom argument of the work list)
  }
  static void run(const b12x_moe_prefill_fast::PersistArgs& p, int grid, cudaStream_t stream) {
    using namespace b12x_moe_prefill_fast;
    moe_persistent_kernel<S, N2><<<grid, kT, S * kF1Stage, stream>>>(p);
  }
};
// 20-25: one 128-column tile per FC2 item (4 / 6 / 7 / 3 / 5 / 2 stages); 30-35: n256 FC2 items (round 3), the same stage counts
static int ws_f2_group(int mode) { return mode >= 30 ? 2 : 1; }
template <typename Fn>
static void with_ws(int mode, Fn fn) {
  switch (mode) {
    case 21: fn(PsLaunch<6, 1>{}); break;
    case 22: fn(PsLaunch<7, 1>{}); break;
    case 23: fn(PsLaunch<3, 1>{}); break;
    case 24: fn(PsLaunch<5, 1>{}); break;
    case 25: fn(PsLaunch<2, 1>{}); break;
    case 30: fn(PsLaunch<4, 2>{}); break;
    case 31: fn(PsLaunch<6, 2>{}); break;
    case 32: fn(PsLaunch<7, 2>{}); break;
    case 33: fn(PsLaunch<3, 2>{}); break;
    case 34: fn(PsLaunch<5, 2>{}); break;
    case 35: fn(PsLaunch<2, 2>{}); break;
    default: fn(PsLaunch<4, 1>{}); break;
  }
}
// below this many rows fc1 stays on v8 (_WS1_MIN_ROWS, default 256): at 128 rows (the replay layers) both kernels are at the DRAM
// floor and the one-CTA/SM v9 fc1 is 4 % slower there (K-4 §6.2); fc2's v9 is neutral at 128 rows and keeps its mode
static int ws1_min_rows() { const char* v = std::getenv("VLLM_MOET_B12X_MOE_PREFILL_FAST_WS1_MIN_ROWS"); static const int m = v ? std::atoi(v) : 256; return m; }
template <int S, int MB, int MH>
struct WsLaunch1 {
  static constexpr int kT = 8 * MH * 32 + 32 * b12x_moe_prefill_fast::kFc1Producers;
  static int grid(int sms) {
    using namespace b12x_moe_prefill_fast;
    int occ = 0;
    static bool done = false;
    if (!done) { cudaFuncSetAttribute(fc1_kernel_ws<S, MB, MH>, cudaFuncAttributeMaxDynamicSharedMemorySize, S * kF1Stage); done = true; }
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fc1_kernel_ws<S, MB, MH>, kT, S * kF1Stage);
    static bool noted = false;
    occ_check("fc1_kernel_ws", fc1_kernel_ws<S, MB, MH>, noted, S, MB, occ, occ_expect("VLLM_MOET_B12X_MOE_PREFILL_FAST_OCC_EXPECT_FC1", 0), sms);
    return sms * (occ > 0 ? occ : 1);
  }
  static void run(const b12x_moe_prefill_fast::Fc1Args& f1, int grid, cudaStream_t stream) {
    using namespace b12x_moe_prefill_fast;
    fc1_kernel_ws<S, MB, MH><<<grid, kT, S * kF1Stage, stream>>>(f1);
  }
};
template <int S, int MB>
struct WgLaunch1 {  // v10: fc1_kernel_wg (12 warps, setmaxnreg at MB = 2)
  static constexpr int kT = b12x_moe_prefill_fast::kWgThreads;
  static int grid(int sms) {
    using namespace b12x_moe_prefill_fast;
    int occ = 0;
    static bool done = false;
    if (!done) { cudaFuncSetAttribute(fc1_kernel_wg<S, MB>, cudaFuncAttributeMaxDynamicSharedMemorySize, S * kF1Stage); done = true; }
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fc1_kernel_wg<S, MB>, kT, S * kF1Stage);
    // on record: <4, 1> one CTA per SM (167 registers, the served mode), <3, 2> two (setmaxnreg)
    static bool noted = false;
    occ_check("fc1_kernel_wg", fc1_kernel_wg<S, MB>, noted, S, MB, occ,
              occ_expect("VLLM_MOET_B12X_MOE_PREFILL_FAST_OCC_EXPECT_FC1", (S == 4 && MB == 1) ? 1 : (S == 3 && MB == 2) ? 2 : 0), sms);
    return sms * (occ > 0 ? occ : 1);
  }
  static void run(const b12x_moe_prefill_fast::Fc1Args& f1, int grid, cudaStream_t stream) {
    using namespace b12x_moe_prefill_fast;
    fc1_kernel_wg<S, MB><<<grid, kT, S * kF1Stage, stream>>>(f1);
  }
};
template <int S, int MB>
struct WsLaunch2 {
  static int grid(int sms) {
    using namespace b12x_moe_prefill_fast;
    int occ = 0;
    static bool done = false;
    if (!done) { cudaFuncSetAttribute(fc2_kernel_ws<S, MB>, cudaFuncAttributeMaxDynamicSharedMemorySize, S * kF2Stage); done = true; }
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fc2_kernel_ws<S, MB>, kWsThreads, S * kF2Stage);
    // on record: <4, 1> (the served mode) and <3, 2> two CTAs per SM - at 96 registers per thread, the most two CTAs of 9 warps get
    static bool noted = false;
    occ_check("fc2_kernel_ws", fc2_kernel_ws<S, MB>, noted, S, MB, occ,
              occ_expect("VLLM_MOET_B12X_MOE_PREFILL_FAST_OCC_EXPECT_FC2", ((S == 4 && MB == 1) || (S == 3 && MB == 2)) ? 2 : 0), sms);
    return sms * (occ > 0 ? occ : 1);
  }
  static void run(const b12x_moe_prefill_fast::Fc2Args& f2, int grid, cudaStream_t stream) {
    using namespace b12x_moe_prefill_fast;
    fc2_kernel_ws<S, MB><<<grid, kWsThreads, S * kF2Stage, stream>>>(f2);
  }
};
template <typename Fn>
static void with_ws1(int mode, Fn fn) {
  switch (mode) {
    case 1: fn(WsLaunch1<3, 2, 1>{}); break;
    case 2: fn(WsLaunch1<6, 1, 1>{}); break;
    case 3: fn(WsLaunch1<4, 1, 1>{}); break;
    case 4: fn(WsLaunch1<7, 1, 1>{}); break;
    case 5: fn(WsLaunch1<6, 1, 2>{}); break;
    case 6: fn(WsLaunch1<4, 1, 2>{}); break;
    case 7: fn(WsLaunch1<7, 1, 2>{}); break;
    case 8: fn(WgLaunch1<3, 2>{}); break;
    case 9: fn(WgLaunch1<4, 1>{}); break;
    case 10: fn(WgLaunch1<6, 1>{}); break;
    default: fn(WgLaunch1<7, 1>{}); break;
  }
}
template <typename Fn>
static void with_ws2(int mode, Fn fn) {
  switch (mode) {
    case 1: fn(WsLaunch2<3, 2>{}); break;
    case 2: fn(WsLaunch2<6, 1>{}); break;
    case 3: fn(WsLaunch2<4, 1>{}); break;
    default: fn(WsLaunch2<7, 1>{}); break;
  }
}

template <int S1, int MB1, int S2, int MB2>
struct Launch {
  static int grid1(int sms) {
    using namespace b12x_moe_prefill_fast;
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fc1_kernel<S1, MB1, false>, kThreads, S1 * kF1Stage);
    return sms * (occ > 0 ? occ : 1);
  }
  static int grid1_ws(int sms) {
    int g = 0;
    with_ws1(ws1_mode(), [&](auto W) { g = decltype(W)::grid(sms); });
    return g;
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
  static bool fc1_is_ws(const b12x_moe_prefill_fast::Fc1Args& f1) {
    return ws1_mode() >= 1 && f1.rows >= ws1_min_rows() && f1.lda == f1.K && f1.lds == f1.K / 32;
  }
  static void fc1(const b12x_moe_prefill_fast::Fc1Args& f1, int grid, cudaStream_t stream) {
    using namespace b12x_moe_prefill_fast;
    if (fc1_is_ws(f1))  // v9: the producer-warp skeleton (contiguous rows)
      with_ws1(ws1_mode(), [&](auto W) { decltype(W)::run(f1, grid, stream); });
    else if (f1.lda != f1.K || f1.lds != f1.K / 32)  // row-strided views of one gathered buffer (the one-all-gather layout)
      fc1_kernel<S1, MB1, true><<<grid, kThreads, S1 * kF1Stage, stream>>>(f1);
    else
      fc1_kernel<S1, MB1, false><<<grid, kThreads, S1 * kF1Stage, stream>>>(f1);
  }
  static bool fc2_is_ws(const b12x_moe_prefill_fast::Fc2Args&) { return ws2_mode() >= 1; }
  static int grid2_ws(int sms) {
    int g = 0;
    with_ws2(ws2_mode(), [&](auto W) { g = decltype(W)::grid(sms); });
    return g;
  }
  static void fc2(const b12x_moe_prefill_fast::Fc2Args& f2, int grid, cudaStream_t stream) {
    using namespace b12x_moe_prefill_fast;
    if (fc2_is_ws(f2))  // v9: the producer-warp skeleton
      with_ws2(ws2_mode(), [&](auto W) { decltype(W)::run(f2, grid, stream); });
    else
      fc2_kernel<S2, MB2><<<grid, kThreads, S2 * kF2Stage, stream>>>(f2);
  }
};

// the front-end launches shared by moe_prefill_fast and moe_prefill_fast_frontend: a memset of the counters, the quant +
// histogram (+ scan) kernel, the scatter kernel. `routing` = the workspace tensors kept alive by the caller.
struct Routing {
  torch::Tensor row_pair, f1_items, f2_items, counts, chunk_cnt, exp_off, exp_cnt, u_items, fc1_need;
};
using b12x_moe_prefill_fast::kCountsWords;  // counts[0..7] + fc1_done[kMaxE] (MK-2) + tile_done[kMaxTiles] + the sum need (K-12), one memset
// K-12: the top-k sum as items of fc2_kernel_ws's column-tile-major list (VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_ITEMS=1; needs WS2 >= 1, not
// with the persistent launch): _SUM_TOKENS = tokens per sum item (256), _SUM_LAG = tiles between an FC2 tile and its sum
// items (2), _SUM_DISCARD=1 = drop the summed route lines from L2 (the route buffer becomes scratch), _SUM_MIN_ROWS = below this many rows
// the separate topk_sum_kernel runs (0), _W2_EVICT_FIRST=1 = W2's copies with an L2 evict_first hint, _SUM_WATCHDOG_MS (2000)
static int sum_items_env() { static const int m = ws_env("VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_ITEMS"); return g_rt_sum_items >= 0 ? g_rt_sum_items : m; }
static int sum_discard_env() { static const int m = ws_env("VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_DISCARD"); return g_rt_sum_discard >= 0 ? g_rt_sum_discard : m; }
static int sum_tokens_env() { const char* v = std::getenv("VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_TOKENS"); static const int m = v ? std::max(16, std::atoi(v)) : 256; return m; }
static int sum_lag_env() { const char* v = std::getenv("VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_LAG"); static const int m = v ? std::max(0, std::atoi(v)) : 2; return m; }
static int sum_min_rows_env() { static const int m = ws_env("VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_MIN_ROWS"); return m; }
static int w2_evict_first_env() { static const int m = ws_env("VLLM_MOET_B12X_MOE_PREFILL_FAST_W2_EVICT_FIRST"); return m; }
static int sum_route_hint_env() { static const int m = ws_env("VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_ROUTE_HINT"); return g_rt_sum_route_hint >= 0 ? g_rt_sum_route_hint : m; }  // 1: route stores evict_last
static int sum_watchdog_ms() { const char* v = std::getenv("VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_WATCHDOG_MS"); static const int m = v ? std::atoi(v) : 2000; return m; }
// the v8 epilogue sum (_FUSED_SUM=1) is gone - the sum items replace it; a process that still asks for it must not run on another path unnoticed
static bool fused_sum_requested() {
  static const bool on = [] { const char* v = std::getenv("VLLM_MOET_B12X_MOE_PREFILL_FAST_FUSED_SUM"); return v && v[0] == '1'; }();
  return on;
}
static Routing launch_frontend(b12x_moe_prefill_fast::FrontArgs& fa, bool quant, int T, int K, int topk, int E, torch::Device dev,
                               bool unified = false, int f2_group = 1, bool sum_items = false) {
  using namespace b12x_moe_prefill_fast;
  auto o32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  const int pairs = T * topk;
  // routing capacities: every expert may add up to 15 padding rows and one partial chunk
  const int row_cap = pairs + 16 * E;
  const int max_chunks = (pairs + 63) / 64 + E;
  const int n2_tiles = K / 128;
  const int sum_tokens = sum_tokens_env();
  const int sum_per_tile = sum_items ? (T + sum_tokens - 1) / sum_tokens : 0;  // K-12: the sum items of the FC2 list
  const int f1_cap = max_chunks * kN1Tiles, f2_cap = max_chunks * n2_tiles + n2_tiles * sum_per_tile;
  const int n_chunks = std::max(1, (pairs + kFrontThreads - 1) / kFrontThreads);
  Routing r;
  r.row_pair = torch::empty({row_cap}, o32);
  r.f1_items = torch::empty({f1_cap, 4}, o32);
  r.f2_items = torch::empty({f2_cap, 4}, o32);
  r.counts = torch::empty({kCountsWords}, o32);
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
  fa.u_items = nullptr;
  fa.fc1_need = nullptr;
  fa.u_cap = 0;
  fa.lead = 0;
  fa.u_f2_group = 1;
  fa.sum_items = sum_items ? 1 : 0;
  fa.sum_tokens = sum_tokens;
  fa.sum_lag = sum_lag_env();
  if (unified) {  // MK-2: the persistent launch's work list (both kinds) + the FC2 items' wait targets
    r.u_items = torch::empty({f1_cap + f2_cap, 4}, o32);
    r.fc1_need = torch::empty({kMaxE}, o32);
    fa.u_items = reinterpret_cast<int4*>(r.u_items.data_ptr());
    fa.fc1_need = r.fc1_need.data_ptr<int32_t>();
    fa.u_cap = f1_cap + f2_cap;
    fa.lead = std::max(1, ws_lead());
    fa.u_f2_group = f2_group;
  }
  return r;
}
static void run_frontend(const b12x_moe_prefill_fast::FrontArgs& fa, cudaStream_t stream) {
  using namespace b12x_moe_prefill_fast;
  cudaMemsetAsync(fa.counts, 0, kCountsWords * sizeof(int32_t), stream);  // the histogram arrival counter, the work counters, fc1_done[]
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

  TORCH_CHECK(!fused_sum_requested(), "VLLM_MOET_B12X_MOE_PREFILL_FAST_FUSED_SUM=1 is set, but the top-k sum in the v8 FC2 epilogue has been "
              "removed: unset it and use VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_ITEMS=1 (with _WS2 >= 1) - the top-k sum as FC2 work items");
  // MK-2: the fused persistent launch (WS >= 20): contiguous rows, at or above _WS_MIN_ROWS rows
  const int lda_b = prequant ? (int)a_values.stride(0) : K, lds_b = prequant ? (int)a_scales.stride(0) : K / 32;
  const bool persistent = ws_mode() >= 20 && lda_b == K && lds_b == K / 32 && T >= ws_min_rows();
  // K-12: the sum items of fc2_kernel_ws (the column-tile-major list): with the v9 fc2 kernel, not with the persistent launch
  const bool sum_items = topk_sum && !persistent && sum_items_env() >= 1 && ws2_mode() >= 1 && T >= sum_min_rows_env() &&
                         K / 128 <= kMaxTiles && topk <= kSumMaxTopk;
  {  // K-13 task B: the activation counters (one add per call per counter pair)
    const bool fc1_v10 = !persistent && ws1_mode() >= 8 && T >= ws1_min_rows() && lda_b == K && lds_b == K / 32;
    g_cnt[fc1_v10 ? 0 : 2] += 1; g_cnt[fc1_v10 ? 1 : 3] += (unsigned long long)T;
    g_cnt[sum_items ? 4 : 6] += 1; g_cnt[sum_items ? 5 : 7] += (unsigned long long)T;
  }
  // n256 FC2 items (modes 30-35) need K % 256 == 0; otherwise the one-tile item kernel of the same ring depth runs
  const int ps_mode = (ws_mode() >= 30 && K % 256 != 0) ? ws_mode() - 10 : ws_mode();
  const int f2_group = ws_f2_group(ps_mode);
  FrontArgs fa;
  fa.hidden = prequant ? nullptr : reinterpret_cast<const uint32_t*>(hidden.data_ptr());
  fa.a_values = reinterpret_cast<uint32_t*>(a_values.data_ptr());
  fa.a_scales = a_scales.data_ptr<uint8_t>();
  Routing rt = launch_frontend(fa, !prequant, T, K, topk, E, dev, persistent, f2_group, sum_items);
  fa.ids = ids.data_ptr<int32_t>();
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
  f1.rows = T;

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
  f2.out = sum_items ? reinterpret_cast<uint32_t*>(out.data_ptr()) : nullptr;
  f2.ids = ids.data_ptr<int32_t>();
  f2.K = K; f2.out_tiles = fa.n2_tiles; f2.gs_per_expert = global_scale.numel() > 1;
  f2.dynamic_schedule = (int)dynamic_schedule; f2.topk = topk;
  // K-12: the sum items (fc2_kernel_ws): the per-tile counters live in the counts buffer (zeroed by run_frontend's memset), the wait target is
  // the scan CTA's word, the watchdog record is the persistent kernel's pinned buffer; discard needs 128-B aligned route rows (K % 64 == 0 holds)
  f2.tile_done = fa.counts + kCntTileDone;
  f2.sum_need = fa.counts + kCntSumNeed;
  f2.err_lock = fa.counts + 7;
  f2.err = ps_err_dev();
  f2.watchdog_ns = (unsigned long long)std::max(1, sum_watchdog_ms()) * 1000000ull;
  f2.sum_mode = sum_items ? 1 : 0;
  f2.sum_discard = (sum_items && sum_discard_env() >= 1 && (reinterpret_cast<uintptr_t>(route.data_ptr()) & 127) == 0) ? 1 : 0;
  f2.w2_evict_first = w2_evict_first_env() >= 1 ? 1 : 0;
  f2.sum_route_hint = (sum_items && sum_route_hint_env() >= 1) ? 1 : 0;

  if (persistent) {  // MK-2: one launch for fc1 + SwiGLU/quant + fc2 (section 6), then the separate top-k sum below
    PersistArgs pa;
    pa.a_values = f1.a_values; pa.a_scales = f1.a_scales; pa.w13 = f1.w13; pa.w13_sf = f1.w13_sf;
    pa.alpha = f1.alpha; pa.input_global_scale = f1.input_global_scale; pa.inter = f1.inter; pa.inter_sf = f1.inter_sf;
    pa.w2 = f2.w2; pa.w2_sf = f2.w2_sf; pa.token_weights = f2.token_weights; pa.down_alpha = f2.down_alpha; pa.global_scale = f2.global_scale;
    pa.route = f2.route;
    pa.row_pair = fa.row_pair; pa.items = fa.u_items; pa.counts = fa.counts;
    pa.fc1_done = fa.counts + 8; pa.fc1_need = fa.fc1_need; pa.err_lock = fa.counts + 7; pa.err = ps_err_dev();
    pa.stats = ws_stats_enabled() ? ps_stats_dev() : nullptr;
    if (pa.stats != nullptr) cudaMemsetAsync(pa.stats, 0, 64, stream);
    pa.watchdog_ns = (unsigned long long)std::max(1, ws_watchdog_ms()) * 1000000ull;
    pa.K = K; pa.topk = topk; pa.igs_per_expert = f1.igs_per_expert; pa.gs_per_expert = f2.gs_per_expert; pa.out_tiles = fa.n2_tiles;
    pa.dynamic_schedule = (int)dynamic_schedule; pa.limit = (float)limit; pa.route_cs = ws_route_cs();
    with_ws(ps_mode, [&](auto W) {
      int g = decltype(W)::grid(sms);
      if (grid1_cap > 0) g = std::min<int>(g, (int)grid1_cap);
      g = std::min(g, fa.u_cap);
      decltype(W)::run(pa, g, stream);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
  } else {
  auto run = [&](auto L) {
    using LL = decltype(L);
    LL::prepare();
    int g1 = LL::fc1_is_ws(f1) ? LL::grid1_ws(sms) : LL::grid1(sms), g2 = LL::fc2_is_ws(f2) ? LL::grid2_ws(sms) : LL::grid2(sms);
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
  }

  if (topk_sum && !sum_items) {
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
// MK-2: the build has the fused persistent launch (VLLM_MOET_B12X_MOE_PREFILL_FAST_WS >= 20); the mode the process runs with
int64_t persistent_mode() { return ws_mode(); }
// K-13 (R0's r29 task A): the runtime A/B overrides. set_runtime_switches(ws1, sum_items, sum_discard, sum_route_hint): -1 = back to the
// env value, otherwise the value used from the next call on (no CUDA state is rebuilt: every mode's kernels live in this .so and set their
// attributes on first use). get_runtime_switches() returns the four effective values; moe_prefill_fast_counters() the eight activation
// counters [fc1 v10 calls, rows, fc1 v8/v9 calls, rows, fc2 sum-items calls, rows, fc2 + topk_sum calls, rows] (task B; `clear` zeroes them).
void set_runtime_switches(int64_t ws1, int64_t sum_items, int64_t sum_discard, int64_t sum_route_hint) {
  TORCH_CHECK(ws1 >= -1 && ws1 <= 11 && sum_items >= -1 && sum_items <= 1 && sum_discard >= -1 && sum_discard <= 1 && sum_route_hint >= -1 && sum_route_hint <= 1,
              "set_runtime_switches: ws1 in [-1, 11], the sum switches in [-1, 1]");
  g_rt_ws1 = (int)ws1; g_rt_sum_items = (int)sum_items; g_rt_sum_discard = (int)sum_discard; g_rt_sum_route_hint = (int)sum_route_hint;
}
std::vector<int64_t> get_runtime_switches() { return {ws1_mode(), sum_items_env(), sum_discard_env(), sum_route_hint_env()}; }
std::vector<int64_t> moe_prefill_fast_counters(bool clear) {
  std::vector<int64_t> r(g_cnt, g_cnt + 8);
  if (clear) for (auto& c : g_cnt) c = 0;
  return r;
}
// K-12: the sum-items mode of this process (VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_ITEMS; active only with WS2 >= 1 and the separate kernels), and
// whether the route rows are consumed in-kernel (summed and discarded from L2: their content after the call is unspecified - the harness
// then compares the output only)
int64_t sum_items_mode() { return (sum_items_env() >= 1 && ws2_mode() >= 1 && ws_mode() < 20) ? sum_items_env() : 0; }
bool route_rows_consumed() { return sum_items_mode() >= 1 && sum_discard_env() >= 1; }
// the persistent kernel's watchdog record (pinned host memory, readable after a trapped launch): [set, item, expert, need, seen, smid, block,
// watchdog_ms]; set == 0: no expired wait since the process started. `clear` resets it.
std::vector<int64_t> moe_prefill_ws_error(bool clear) {
  std::vector<int64_t> r(8, 0);
  uint32_t* h = ps_err_host();
  if (h == nullptr) return r;
  for (int i = 0; i < 8; ++i) r[i] = (int64_t)h[i];
  if (clear) memset(h, 0, 32);
  return r;
}

// the statistics of the LAST persistent launch (VLLM_MOET_B12X_MOE_PREFILL_FAST_WS_STATS=1; synchronises the device): [fc1_ns, fc1_items,
// fc2_ns, fc2_items, idle_ns, fc1_done_wait_ns, cta_span_ns, ctas], sums over the CTAs
std::vector<int64_t> moe_prefill_ws_stats() {
  std::vector<int64_t> r(8, 0);
  unsigned long long* d = ps_stats_dev();
  if (d == nullptr || !ws_stats_enabled()) return r;
  unsigned long long h[8];
  cudaDeviceSynchronize();
  if (cudaMemcpy(h, d, 64, cudaMemcpyDeviceToHost) != cudaSuccess) { cudaGetLastError(); return r; }
  for (int i = 0; i < 8; ++i) r[i] = (int64_t)h[i];
  return r;
}

// what every WS launch family got at its first launch in this process: (kernel, stages, min blocks, registers per thread, CTAs per SM
// from the occupancy query, CTAs per SM its mode was measured with or 0, SMs) - the occupancy guard's record
std::vector<std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>> occupancy_facts() {
  std::vector<std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>> r;
  for (const OccFact& f : occ_facts()) r.emplace_back(std::string(f.kernel), f.s, f.mb, f.regs, f.occ, f.expect, f.sms);
  return r;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("occupancy_facts", &occupancy_facts);
  m.def("moe_prefill_fast", &moe_prefill_fast);
  m.def("moe_prefill_fast_frontend", &moe_prefill_fast_frontend);
  m.def("prequant_rows_supported", &prequant_rows_supported);
  m.def("strided_rows_supported", &strided_rows_supported);
  m.def("persistent_mode", &persistent_mode);
  m.def("sum_items_mode", &sum_items_mode);
  m.def("route_rows_consumed", &route_rows_consumed);
  m.def("set_runtime_switches", &set_runtime_switches);
  m.def("get_runtime_switches", &get_runtime_switches);
  m.def("moe_prefill_fast_counters", &moe_prefill_fast_counters);
  m.def("moe_prefill_ws_error", &moe_prefill_ws_error);
  m.def("moe_prefill_ws_stats", &moe_prefill_ws_stats);
}
