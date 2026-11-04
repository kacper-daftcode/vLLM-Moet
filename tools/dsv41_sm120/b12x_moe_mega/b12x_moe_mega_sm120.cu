// The MoE sublayer of b12x's compact W4A8 micro path as ONE persistent kernel on sm_120 (DeepSeek-V4.1-Flash, TP4 rank:
// 384 experts, hidden 5120, intermediate 576, top-6, SwiGLU limit 10): the router's spread GEMM, DeepSeek-V4's routing
// top-6, then b12x_moe_fused's FC1 -> activation -> FC2 - bit for bit the separate kernels' (spread_kernel<STEPS> of
// bf16_gemm/bf16_replica_gemm_sm120.cu, dsv4_topk_sm120.cu, b12x_moe_fused_sm120.cu).
//
// Work items, pulled from a global counter in order (CTA b starts with item b):
//   [0, n_route)             routing tiles: tile (8 router columns) x slice group (4 slice warps of `step` K each); the
//                            partials go to a workspace, the CTA arriving last at the tile's counter sums them in slice
//                            order (spread's reduction), writes the logits and the per-expert routing scores (w = sqrt-
//                            softplus, order key of w + bias) and counts the tile done;
//   [n_route, n_route + M)   selection: one CTA per token waits for all tiles, selects the top-6 of its row (dsv4_topk's
//                            six redux.max / redux.min selections, butterfly sum, div.full, mul), writes ids / weights
//                            and counts the token routed;
//   the rest                 b12x_moe_fused's FC1 / activation / FC2 items, unchanged: a CTA entering them first waits
//                            until every token is routed and builds its expert tables from the ids it finds in memory.
// Routing items never wait, selections wait only on routing items, FC items on selections and FC1 items: with every
// CTA resident (the host sizes the grid to what fits at once) nothing deadlocks. With routing off the kernel is
// b12x_moe_fused_sm120 (the ids / weights are inputs).
//
// Measured (pb-deep, r18 + steps 30-34 + the deterministic switches, OFF vs ON, greedy outputs identical): 132.3 us per
// launch vs spread 7.0 + top-k 4.0 + fused MoE 117.8 = 128.9 in situ; the main graph 1,313 -> 1,233 kernels but the
// critical path 8,197 -> 8,424 us per step (+2.7 % period). The phases are serial, so the fusion saves the two launch
// gaps (~1 us) and loses the spread's PDL overlap with the preceding boundary kernel (~3 us), step 34's side-stream
// quantization hides behind the router no more (+32 us per step on the path) and the in-kernel hand-offs (selection
// wake-up, expert tables after the routing, the grid polling the routing counter) cost the rest. Kept as an opt-in
// (Dockerfile build-arg VLLM_MOET_MOE_MEGA_PATCH, default 0); routing off = step 27's kernel, bit for bit.
//
// The MoE phases below are b12x_moe_fused_sm120.cu's (step 27) verbatim; their description:
//
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace b12x_moe_mega {

constexpr int kThreads = 128;
constexpr int kTileN = 128;
constexpr int kMaxRows = 8;      // pairs per expert (tokens <= 8)
constexpr int kMaxPairs = 64;
constexpr int kMaxExperts = 64;  // unique experts of a launch (<= pairs)
constexpr int kMaxExpertIds = 384;

// FC1 ring: K64 stages; the rows' K/32 scale bytes (160 B per row) are staged once per tile after the ring
constexpr int kF1Stages = 8;
constexpr int kF1A = kMaxRows * 64;
constexpr int kF1B = 2 * 4 * 32 * 16;  // K32(2) x N32(4) x lane x 16 B = 4 KB
constexpr int kF1Sfb = 128 * 4;
constexpr int kF1Stage = kF1A + kF1B + kF1Sfb;  // 5,120 B
constexpr int kF1RowScales = 256;                // >= K/32 = 160 B per row, 16 B granules
constexpr int kF1Smem = kF1Stages * kF1Stage + kMaxRows * kF1RowScales;  // 43,008 B

// FC2: 64-column items (half of a 128-column w2 tile), two buffers so the next item's weights stream in while the
// current one is computed; the tail slice holds two K32 columns. The A fragments are read straight into registers.
constexpr int kSlices = 5;
constexpr int kTotalK32 = 4 * kSlices - 2;  // 18
constexpr int kF2Cols = 64;
constexpr int kF2BItem = kTotalK32 * kF2Cols * 16;             // 18 K32 x 2 n32 chunks x 512 B = 18,432 B
constexpr int kF2SfbItem = (kSlices - 1) * kF2Cols * 4 + kF2Cols * 2;  // 1,152 B
constexpr int kF2Buf = kF2BItem + kF2SfbItem;                 // 19,584 B
constexpr int kF2Smem = 2 * kF2Buf;                            // 39,168 B

constexpr int kSmem = kF1Smem > kF2Smem ? kF1Smem : kF2Smem;
static_assert(2 * (kSmem + 5 * 1024 + 1024) <= 100 * 1024, "two CTAs per SM within sm_120 100 KB (static tables ~5 KB + 1 KB reserved each)");

// control block (int32)
constexpr int kCtrlNext = 0, kCtrlFinished = 1, kCtrlNReady = 2;
constexpr int kCtrlSliceDone = 4;                                   // [u * 5 + s] FC1 tiles (gate, up) of the slice done
constexpr int kCtrlActDone = kCtrlSliceDone + kMaxExperts * kSlices;  // [u] slices activated
constexpr int kCtrlReady = kCtrlActDone + kMaxExperts;              // [u] expert ready for FC2
constexpr int kMaxRouteTiles = 48;                                  // 384 router columns / 8
// the routing counters on 128-B lines of their own: the tile atomics must not queue behind the polling of the
// scores-done / routing-done words by hundreds of CTAs (ctrl is 256-B aligned: a torch allocation)
constexpr int kCtrlTileCnt = ((kCtrlReady + kMaxExperts + 31) / 32) * 32;  // [tile] slice groups of the routing tile arrived
constexpr int kCtrlScoresDone = kCtrlTileCnt + 64;                  // routing tiles reduced (scores written); own line
constexpr int kCtrlRoutingDone = kCtrlScoresDone + 32;              // tokens whose top-6 is written; own line
constexpr int kCtrlReadyList = kCtrlRoutingDone + 32;               // [k] k-th ready expert among the dynamic ones, -1
constexpr int kCtrlInts = kCtrlReadyList + kMaxExperts;
constexpr int kTraceItems = 16;

struct Args {
  const uint32_t* a_values;    // [cap][K] e4m3 (b12x rows-quant)
  const uint8_t* a_scales;     // [cap][K/32] ue8m0
  const uint32_t* w13;         // prepared n64-repacked [experts][2][N][K/32][16 B]
  const uint32_t* w13_sf;      // [experts][2][N][K/32] bytes
  __nv_bfloat16* projections;  // scratch [cap*topk][2N] bf16: gate | up
  uint32_t* intermediate;      // scratch: rows of 640 B, then scale words
  const uint32_t* w2;          // n64-repacked [experts][hidden/128][18 K32][128 cols][16 B]
  const uint8_t* w2_sf;        // [experts][hidden/128][128 cols][18]
  const int32_t* ids;          // [pairs] int32
  const float* token_weights;  // [pairs]
  const float* alpha1;
  const float* input_scale;
  const float* alpha2;
  const float* down_scale;
  __nv_bfloat16* out;          // route_output [rows_capacity][hidden]
  int* ctrl;
  long long* trace;            // optional [grid][kTraceItems][4]
  float limit;
  int is_per_expert;
  int ds_per_expert;
  int num_pairs;
  int topk;
  int rows_capacity;
  int k;        // hidden (5120)
  int n;        // intermediate (576)
  int n_tiles;  // ceil(n / 128) = 5
  int out_tiles;
  int num_experts;
  int pf_depth;  // FC1: K64 slices prefetched into L2 ahead of the ring (0 = off)
  int phase_mask;  // debug: 1 = FC1 + activation only, 2 = FC2 only (no wait; intermediate as given), 3 = all
  unsigned spin_ns_max;  // cap of the exponential backoff while waiting for an expert
  // routing phase (route_on): spread GEMM logits[M][384] = x[M][K] . wr[384][K]^T in `step`-wide K slices, then top-6
  int route_on;
  const __nv_bfloat16* x;   // [M][ldx] bf16 (the router's input)
  int ldx;
  const __nv_bfloat16* wr;  // [384][K] bf16 router weights
  int M;                    // tokens (<= 8)
  int step;                 // slice width (multiple of 16; cuBLAS's split for this shape, calibrated by bf16_gemm)
  int S;                    // slices = ceil(K / step) (<= 32)
  int groups;               // slice groups per tile = ceil(S / 4)
  float* logits;            // [M][384] fp32 (written as spread does)
  float* ws;                // partials [48][S][16][8] fp32
  float* wk_w;              // [M][384] routing w
  unsigned* wk_k;           // [M][384] order keys
  const float* bias;        // [384] correction bias
  const float* bias_vl;     // [384] or nullptr (image-sentinel tokens)
  const int32_t* tok32;     // input token ids [M] (image test) or nullptr
  const int64_t* tok64;
  long long image_lo;
  const bool* is_padding;   // [M] or nullptr
  float scale;              // routed scaling factor
  float* out_w;             // [M][6] = token_weights
  int32_t* out_ids;         // [M][6] = ids
  int pdl;                  // launched with programmatic stream serialization: the routing CTAs prefetch their router
                            // weight lines into L2 before griddepcontrol.wait; everything else runs after it
};

__device__ __forceinline__ long long globaltimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ void st_shared_u32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(addr), "r"(v));
}
__device__ __forceinline__ void st_shared_zero16(uint32_t addr) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(addr), "r"(0u));
}
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
// the weights are streamed once: an evict_first L2 policy keeps them from displacing the activations and the rows
// (FC1 -0.6 us per layer on the recorded routing, FC2 within noise)
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void cp_async16_ef(uint32_t dst, const void* src, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "l"(pol));
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
  asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// b12x's e2m1x8_to_qmma_e2m1x8: eight FP4 nibbles into E2M1 byte containers (nibble in bits 5:2)
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

template <int KB>
__device__ __forceinline__ void qmma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %13}, {%11}, {%12, %13};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(KB), "n"(0));
}
template <int KB>
__device__ __forceinline__ void qmma0(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                      uint32_t b1, uint32_t sfa, uint32_t sfb) {
  const float z = 0.f;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%14, %14, %14, %14}, {%10}, {%12, %13}, {%11}, {%12, %13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(KB), "n"(0), "f"(z));
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.satfinite.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}
__device__ __forceinline__ uint16_t to_bf16_rn(float x) {
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
  return r;
}

// b12x's compact activation for 4 columns of one row (lane of an 8-lane K32 group): SwiGLU with limit, bf16 boundary,
// MXFP8 quantization with the group's ue8m0 scale. Returns the e4m3x4 word; *scale_byte gets the group's scale.
// `g2`/`u2` are the gate / up bf16x4 of the columns (ignored when !valid).
__device__ __forceinline__ uint32_t act4(uint2 g2, uint2 u2, float limit, bool valid, uint32_t* scale_byte) {
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  if (valid) {
    const uint32_t gw[2] = {g2.x, g2.y}, uw[2] = {u2.x, u2.y};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float g = __uint_as_float((i & 1 ? gw[i >> 1] & 0xffff0000u : gw[i >> 1] << 16));
      float u = __uint_as_float((i & 1 ? uw[i >> 1] & 0xffff0000u : uw[i >> 1] << 16));
      g = g >= limit ? limit : g;
      u = u >= limit ? limit : u;
      u = u <= -limit ? -limit : u;
      float t, e, sg;
      asm("mul.rn.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(g));
      asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(t));
      asm("add.rn.f32 %0, %1, 0f3F800000;" : "=f"(sg) : "f"(e));
      asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(sg) : "f"(sg));
      const float m = __fmul_rn(g, sg);
      const float x = __fmul_rn(u, m);
      unsigned short h;
      asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x));
      v[i] = __uint_as_float((uint32_t)h << 16);
    }
  }
  float amax;
  asm("abs.f32 %0, %1;" : "=f"(amax) : "f"(v[0]));
#pragma unroll
  for (int i = 1; i < 4; ++i) {
    float ab;
    asm("abs.f32 %0, %1;" : "=f"(ab) : "f"(v[i]));
    asm("max.f32 %0, %1, %2;" : "=f"(amax) : "f"(amax), "f"(ab));
  }
#pragma unroll
  for (int sh = 1; sh <= 4; sh <<= 1) {
    const float o = __shfl_xor_sync(0xffffffffu, amax, sh);
    asm("max.f32 %0, %1, %2;" : "=f"(amax) : "f"(amax), "f"(o));
  }
  float sc;
  asm("mul.rn.f32 %0, %1, 0f3B124925;" : "=f"(sc) : "f"(amax));
  uint32_t bits = __float_as_uint(sc);
  if (bits & 0x007FFFFFu) bits = (bits + 0x00800000u) & 0x7F800000u;
  const uint32_t byte = (bits >> 23) & 0xFFu;
  float inv;
  {
    const float ef = (float)(127 - (int)byte);
    asm("ex2.approx.f32 %0, %1;" : "=f"(inv) : "f"(ef));
    if (byte == 0) inv = 0.f;
  }
  if (!valid) { *scale_byte = 0; return 0u; }
  *scale_byte = byte;
  unsigned short lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(__fmul_rn(inv, v[0])), "f"(__fmul_rn(inv, v[1])));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(__fmul_rn(inv, v[2])), "f"(__fmul_rn(inv, v[3])));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

// ------------------------------------------------------------------------------------------- routing: spread GEMM
// (bf16_gemm/bf16_replica_gemm_sm120.cu: one warp per 8 columns x one K slice, mma.m16n8k16 over k ascending, the
// slice partials summed in ascending slice order) and DeepSeek-V4's top-6 (dsv4_topk_sm120.cu) - the same instructions.
__device__ __forceinline__ void mma_16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                          uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint2 ldg64_stream(const __nv_bfloat16* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 ldg64(const __nv_bfloat16* p) {
  uint2 v;
  asm volatile("ld.global.v2.u32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ float ldcg_f32(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];\n" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned ldcg_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ int ldcg_s32(const int* p) {
  int v;
  asm volatile("ld.global.cg.s32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

// spread's slice_chain<STEPS, HI = false> with a runtime step count: the same mma per 16-wide k step, k ascending within
// the slice (the fragments are staged eight steps at a time; the staging does not touch the arithmetic). Rows g (< M);
// rows 8..15 are zero (M <= 8).
constexpr int kRouteChunk = 20;  // the served slice width (320 = 20 x 16): all of a slice's fragments in flight at once, as spread<20>
__device__ __forceinline__ void route_slice(float (&c)[4], const __nv_bfloat16* __restrict__ x, int ldx,
                                            const __nv_bfloat16* __restrict__ w, int M, int K, int n0, int kb,
                                            int nsteps, int g, int t) {
  const bool v0 = g < M;
  const __nv_bfloat16* wp = w + (size_t)(n0 + g) * K + kb + t * 4;
  const __nv_bfloat16* x0 = x + (size_t)g * ldx + kb + t * 4;
  c[0] = c[1] = c[2] = c[3] = 0.f;
#pragma unroll 1
  for (int j0 = 0; j0 < nsteps; j0 += kRouteChunk) {
    uint2 b[kRouteChunk], a0[kRouteChunk];
#pragma unroll
    for (int jj = 0; jj < kRouteChunk; ++jj) {
      const int j = j0 + jj;
      const int idx = j < nsteps ? j : nsteps - 1;  // a shorter chunk re-reads its last step instead of branching
      b[jj] = ldg64_stream(wp + idx * 16);
      a0[jj] = v0 ? ldg64(x0 + idx * 16) : make_uint2(0u, 0u);
    }
#pragma unroll
    for (int jj = 0; jj < kRouteChunk; ++jj)
      if (j0 + jj < nsteps) mma_16816(c, a0[jj].x, 0u, a0[jj].y, 0u, b[jj].x, b[jj].y);
  }
}

__device__ __forceinline__ float tk_ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float tk_sqrt_approx_ftz(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float tk_div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float tk_mul_rn(float a, float b) {
  float y;
  asm("mul.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float tk_add_rn(float a, float b) {
  float y;
  asm("add.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
// order-preserving key of a non-NaN float; +0 and -0 map to the same key
__device__ __forceinline__ unsigned tk_key_of(float f) {
  if (f == 0.0f) f = 0.0f;
  const unsigned b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
// the per-expert score of one logit: w = sqrt-softplus(x), key = order key of w + bias (NaN -> -1e30)
__device__ __forceinline__ void tk_score(float x, float bias, float& w, unsigned& key) {
  const float e = tk_ex2_approx(tk_mul_rn(x, 1.44269502162933349609375f));  // 0x3FB8AA3B
  const float l = logf(tk_add_rn(e, 1.0f));
  w = tk_sqrt_approx_ftz(x > 20.0f ? x : l);
  float sc = tk_add_rn(w, bias);
  if (!(sc == sc)) sc = -1e30f;
  key = tk_key_of(sc);
}
// the bias row of a token: bias_vl for the five image-sentinel tokens, bias otherwise
__device__ __forceinline__ const float* tk_bias_row(const Args& a, int row) {
  if (a.bias_vl != nullptr && a.image_lo > 0) {
    const long long tok = a.tok32 != nullptr ? (long long)a.tok32[row] : (long long)a.tok64[row];
    if (tok >= a.image_lo && tok < a.image_lo + 5) return a.bias_vl;
  }
  return a.bias;
}

// the top-6 of one row over the precomputed scores (384 = 32 lanes x 12), by one warp: dsv4_topk_sm120.cu's six
// (redux.max key, redux.min index) selections, butterfly sum, div.full and mul
__device__ __forceinline__ void topk_row(const Args& a, int row, int lane) {
  constexpr int E = 384, PER = E / 32;
  float wv[PER];
  unsigned kv[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    wv[j] = ldcg_f32(a.wk_w + (size_t)row * E + lane + 32 * j);
    kv[j] = ldcg_u32(a.wk_k + (size_t)row * E + lane + 32 * j);
  }
  float sel = 0.0f;
  int id = 0;
#pragma unroll
  for (int s = 0; s < 6; ++s) {
    unsigned m = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) m = kv[j] > m ? kv[j] : m;
    m = __reduce_max_sync(0xffffffffu, m);
    unsigned c = E;
#pragma unroll
    for (int j = PER - 1; j >= 0; --j)
      if (kv[j] == m) c = lane + 32 * j;
    const unsigned i = __reduce_min_sync(0xffffffffu, c);
    const int o = i & 31, jw = i >> 5;
    float mine = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j)
      if (j == jw) mine = wv[j];
    const float wsel = __shfl_sync(0xffffffffu, mine, o);
    if (lane == o) {
#pragma unroll
      for (int j = 0; j < PER; ++j)
        if (j == jw) kv[j] = 0;
    }
    if (lane == s) {
      sel = wsel;
      id = (int)i;
    }
  }
  const float a0 = __shfl_sync(0xffffffffu, sel, 0), a1 = __shfl_sync(0xffffffffu, sel, 1);
  const float a2 = __shfl_sync(0xffffffffu, sel, 2), a3 = __shfl_sync(0xffffffffu, sel, 3);
  const float a4 = __shfl_sync(0xffffffffu, sel, 4), a5 = __shfl_sync(0xffffffffu, sel, 5);
  const float sum = tk_add_rn(tk_add_rn(tk_add_rn(a0, a4), a2), tk_add_rn(tk_add_rn(a1, a5), a3));
  const float f = tk_div_full(a.scale, sum > 0.0f ? sum : 1.0f);
  float out = tk_mul_rn(f, sel);
  int oid = id;
  if (a.is_padding != nullptr && a.is_padding[row]) {
    out = 0.0f;
    oid = -1;
  }
  if (lane < 6) {
    a.out_w[(size_t)row * 6 + lane] = out;
    a.out_ids[(size_t)row * 6 + lane] = oid;
  }
}

struct Tables {
  int16_t expert[kMaxExperts];
  uint8_t rows[kMaxExperts][kMaxRows];
  uint8_t tok[kMaxExperts][kMaxRows];  // rows / topk
  uint8_t nrows[kMaxExperts];
  uint8_t pad[kMaxPairs];
  int16_t slot_of[kMaxExpertIds];
  int first[kMaxExpertIds];  // first pair of each expert id (prologue scratch)
  int cnt[kMaxExpertIds];    // pairs of each expert id (prologue scratch)
  int16_t ids[kMaxPairs];
  int w0_leaders;
  int w0_pads;
  int n_unique;
  int n_pad;
  int item;
  int next;
  int flag;
  int n_trace;
  int pf_depth;
};

// ---------------------------------------------------------------- FC1: one 128-column projection tile of an expert
__device__ __forceinline__ void fc1_tile(const Args& a, Tables& T, uint8_t* smem, int u, int ptile) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nrows = T.nrows[u];
  const int e = T.expert[u];
  const int K = a.k, N = a.n;
  const int projection = ptile / a.n_tiles, otile = ptile - projection * a.n_tiles;
  const int n_cols = otile == a.n_tiles - 1 ? (N - otile * kTileN) : kTileN;  // 64 in the n64 tail tile
  const size_t wexp = (size_t)e * N * (K / 32) * 8 + (size_t)projection * N * (K / 32) * 4 +
                      (size_t)otile * kTileN * (K / 32) * 4;
  const size_t sexp = (size_t)e * N * (K / 32) / 2 + (size_t)projection * N * (K / 32) / 4 +
                      (size_t)otile * kTileN * (K / 32) / 4;
  const int k64_tiles = K / 64;
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int n32_chunks = n_cols / 32;
  const uint32_t* wtile = a.w13 + wexp;
  const uint32_t* stile = a.w13_sf + sexp;
  const int tile_words_per_k32 = n32_chunks * 128;  // 2 KB (full) or 1 KB (tail) per K32 block
  const uint32_t s_rowsc = sbase + kF1Stages * kF1Stage;  // [row][K/32 bytes]
  const int sc_vecs = K / 32 / 16;                        // 16-byte granules of a row's scales (10)
  const uint64_t pol = policy_evict_first();
  if (tid < nrows * 16) {
    const int row = tid >> 4, v = tid & 15;
    if (v < sc_vecs) cp_async16(s_rowsc + row * kF1RowScales + v * 16, a.a_scales + (size_t)T.tok[u][row] * (K / 32) + v * 16);
  }

  auto issue = [&](int j) {
    const uint32_t st = sbase + (j % kF1Stages) * kF1Stage;
    if (tid < nrows * 4) {
      const int row = tid >> 2, v = tid & 3;
      cp_async16(st + row * 64 + v * 16, a.a_values + (size_t)T.tok[u][row] * (K / 4) + j * 16 + v * 4);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {  // kb
      const int idx = tid + i * kThreads;
      const int l = idx & 31, w = (idx >> 5) & 3, kb = idx >> 7;
      const uint32_t dst = st + kF1A + ((kb * 4 + w) * 32 + l) * 16;
      if (w < n32_chunks) cp_async16_ef(dst, wtile + (size_t)(j * 2 + kb) * tile_words_per_k32 + w * 128 + l * 4, pol);
      else st_shared_zero16(dst);
    }
    if (tid < 32) {  // 16 B = 4 columns' scale words
      const uint32_t dst = st + kF1A + kF1B + tid * 16;
      if (tid < n_cols / 4) cp_async16_ef(dst, stile + (size_t)(j >> 1) * n_cols + tid * 4, pol);
      else st_shared_zero16(dst);
    }
    const int pj = j + T.pf_depth;
    if (T.pf_depth > 0 && pj < k64_tiles) {
      const int lines = tile_words_per_k32 * 2 * 4 / 128;  // 32 (full) or 16 (tail)
      if (tid < lines) prefetch_l2(reinterpret_cast<const uint8_t*>(wtile + (size_t)pj * 2 * tile_words_per_k32) + tid * 128);
      if ((pj & 1) == 0 && tid >= 32 && tid < 32 + n_cols * 4 / 128)
        prefetch_l2(reinterpret_cast<const uint8_t*>(stile + (size_t)(pj >> 1) * n_cols) + (tid - 32) * 128);
    }
  };

#pragma unroll 1
  for (int j = 0; j < kF1Stages - 1; ++j) {
    issue(j);
    cp_async_commit();
  }
  const int q = lane >> 2, c = lane & 3;
  const float vs = __fmul_rn(__ldg(a.alpha1 + e), __ldg(a.input_scale + (a.is_per_expert ? e : 0)));
  float acc[4][4];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[nt][i] = 0.f;

  for (int j = 0; j < k64_tiles; ++j) {
    cp_async_wait<kF1Stages - 2>();
    __syncthreads();
    if (j + kF1Stages - 1 < k64_tiles) issue(j + kF1Stages - 1);
    cp_async_commit();
    const uint32_t st = sbase + (j % kF1Stages) * kF1Stage;
    const uint32_t shift = (j & 1) * 16;
    uint32_t asc;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(asc) : "r"(s_rowsc + q * kF1RowScales + (j >> 1) * 4));
    asc >>= shift;
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      uint32_t a0, a2;
      asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(a0), "=r"(a2) : "r"(st + q * 64 + kb * 32 + c * 8));
      uint32_t w[4];
      asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
                   : "r"(st + kF1A + ((kb * 4 + warp) * 32 + lane) * 16));
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        uint32_t b0, b1;
        spread_e2m1x8(w[nt], b0, b1);
        uint32_t sfb;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sfb) : "r"(st + kF1A + kF1B + ((warp * 4 + nt) * 8 + q) * 4));
        sfb >>= shift;
        float x[4];
        if (kb == 0) qmma0<0>(x, a0, a0, a2, a2, b0, b1, asc, sfb);
        else qmma0<1>(x, a0, a0, a2, a2, b0, b1, asc, sfb);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[nt][i] = __fadd_rn(acc[nt][i], x[i]);
      }
    }
  }
  cp_async_wait<0>();

  if (q < nrows) {
    const int pair = T.rows[u][q];
    int dst = otile * kTileN;
    if (ptile < a.n_tiles) dst += N;  // the prepared layout holds up first; projections hold gate, then up
    uint16_t* row = reinterpret_cast<uint16_t*>(a.projections + (size_t)pair * 2 * N);
    const int col_base = warp * 32 + (c << 1);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int tc = col_base + nt * 8;
      if (otile * kTileN + tc < N) {
        const uint32_t v = (uint32_t)to_bf16_rn(__fmul_rn(vs, acc[nt][0])) |
                           ((uint32_t)to_bf16_rn(__fmul_rn(vs, acc[nt][1])) << 16);
        *reinterpret_cast<uint32_t*>(row + dst + tc) = v;
      }
    }
  }
}

// ---------------------------------------------- activation of one K128 slice of an expert's rows (once per slice)
// by the CTA that finished the second of the slice's two FC1 tiles (gate, up); reads both halves through L2.
__device__ __forceinline__ void activation_slice(const Args& a, Tables& T, int u, int s) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nrows = T.nrows[u];
  const int words_per_row = kSlices * 32;
  const int col = s * 128 + lane * 4;
  const bool valid = col < a.n;
  // a warp handles rows warp and warp + 4: all loads first (one round trip), then the compute and stores
  uint2 g2[2] = {make_uint2(0u, 0u), make_uint2(0u, 0u)}, u2[2] = {make_uint2(0u, 0u), make_uint2(0u, 0u)};
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = warp + i * (kThreads / 32);
    if (valid && r < nrows) {
      const __nv_bfloat16* prow = a.projections + (size_t)T.rows[u][r] * 2 * a.n;
      g2[i] = __ldcg(reinterpret_cast<const uint2*>(prow + col));
      u2[i] = __ldcg(reinterpret_cast<const uint2*>(prow + a.n + col));
    }
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = warp + i * (kThreads / 32);
    if (r >= nrows) break;
    const int pair = T.rows[u][r];
    uint32_t sb;
    const uint32_t w = act4(g2[i], u2[i], a.limit, valid, &sb);
    a.intermediate[(size_t)pair * words_per_row + s * 32 + lane] = w;
    if ((lane & 7) == 0) {
      uint8_t* sw = reinterpret_cast<uint8_t*>(a.intermediate + (size_t)a.rows_capacity * words_per_row +
                                               (size_t)s * a.rows_capacity + pair);
      sw[lane >> 3] = (uint8_t)sb;
    }
  }
}

// ----------------------------------------------------------------- FC2: one 64-column output block of an expert
// part 1 (no dependency on the expert's activation): the w2 half tile and its scales into buffer `buf`, one commit group
__device__ __forceinline__ void fc2_issue_w(const Args& a, uint8_t* smem, int e, int item, int buf) {
  const int tid = threadIdx.x;
  const int tile = item >> 1, half = item & 1;
  const size_t expert_words = (size_t)a.out_tiles * kTileN * kTotalK32 * 4;
  const uint32_t* wsrc = a.w2 + (size_t)e * expert_words + (size_t)tile * (kTileN * kTotalK32 * 4) + half * 256;
  const uint8_t* sfsrc = a.w2_sf + ((size_t)e * a.out_tiles + tile) * kTileN * kTotalK32;
  const uint32_t sb = static_cast<uint32_t>(__cvta_generic_to_shared(smem)) + buf * kF2Buf;
  const uint64_t pol = policy_evict_first();
  // B: per K32 block the half tile's two n32 chunks (1 KB contiguous); 18 blocks = 18 KB
  const int l = tid & 31, n32 = (tid >> 5) & 1, kbo = tid >> 6;  // 128 threads: two K32 blocks per pass
#pragma unroll 1
  for (int kb = kbo; kb < kTotalK32; kb += 2)
    cp_async16_ef(sb + (kb * 2 + n32) * 512 + l * 16, wsrc + (size_t)kb * 512 + n32 * 128 + l * 4, pol);
  // scales: slices 0-3: 64 columns x 4 B (256 B each); tail: 64 columns x 2 B (128 B)
  if (tid < 4 * 16 + 8) {
    const int s = tid >> 4, v = tid & 15;
    cp_async16_ef(sb + kF2BItem + s * 256 + v * 16, sfsrc + (size_t)s * 512 + half * (s == 4 ? 128 : 256) + v * 16, pol);
  }
  cp_async_commit();
}

// part 2a (after the expert is ready, before waiting for the buffer): this thread's A fragments from `intermediate`
// (through L2): 8 bytes of row `pair` per K32 block (the logical vector vec = kb*2 + (c>>1) of each slice-row), the
// row's scale word per slice, and the epilogue's scalars
struct Fc2Regs {
  uint2 af[kTotalK32];
  uint32_t asc[kSlices];
  float scale;
  int pair;
};
__device__ __forceinline__ void fc2_load_a(const Args& a, Tables& T, int u, Fc2Regs& R) {
  const int lane = threadIdx.x & 31;
  const int nrows = T.nrows[u];
  const int e = T.expert[u];
  const int words_per_row = kSlices * 32;
  const int q = lane >> 2, c = lane & 3;
  R.pair = T.rows[u][q < nrows ? q : 0];
  const uint32_t* arow = a.intermediate + (size_t)R.pair * words_per_row;
#pragma unroll
  for (int k32 = 0; k32 < kTotalK32; ++k32) {
    const int s = k32 >> 2, kb = k32 & 3;
    R.af[k32] = __ldcg(reinterpret_cast<const uint2*>(arow + s * 32 + (kb * 2 + (c >> 1)) * 4 + (c & 1) * 2));
  }
#pragma unroll
  for (int s = 0; s < kSlices; ++s)
    R.asc[s] = __ldcg(a.intermediate + (size_t)a.rows_capacity * words_per_row + (size_t)s * a.rows_capacity + R.pair);
  const float down_scale = __ldg(a.alpha2 + e) * __ldg(a.down_scale + (a.ds_per_expert ? e : 0));
  R.scale = down_scale * ldcg_f32(a.token_weights + R.pair);  // written by this kernel's selection items with routing on
}

// part 2b (buffer landed): MMAs and epilogue
__device__ __forceinline__ void fc2_mma(const Args& a, Tables& T, uint8_t* smem, int u, int item, int buf, const Fc2Regs& R) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nrows = T.nrows[u];
  const uint32_t sb = static_cast<uint32_t>(__cvta_generic_to_shared(smem)) + buf * kF2Buf;
  const int q = lane >> 2, c = lane & 3;
  float acc[2][4];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[nt][j] = 0.f;
  // warp w: local columns [16w, 16w+16) = n32 chunk w>>1, n8 tiles 2(w&1), 2(w&1)+1
  const uint32_t bbase = sb + ((warp >> 1) * 32 + lane) * 16 + (warp & 1) * 8;
  const int lc0 = warp * 16 + q;  // local column of nt = 0 (nt adds 8)

#pragma unroll
  for (int s = 0; s < kSlices; ++s) {
    const bool tail = s == kSlices - 1;
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      const int k32 = s * 4 + kb;
      uint32_t w0 = 0u, w1 = 0u;
      if (!tail || kb < 2)  // the tail slice's K32 columns 2-3 have zero B, as in b12x's phase 2
        asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(w0), "=r"(w1) : "r"(bbase + k32 * 1024));
      const uint32_t a0 = (!tail || kb < 2) ? R.af[k32].x : 0u;  // the padding columns are zero, as in b12x
      const uint32_t a2 = (!tail || kb < 2) ? R.af[k32].y : 0u;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        uint32_t b0, b1;
        spread_e2m1x8(nt ? w1 : w0, b0, b1);
        uint32_t sfb;
        if (tail) {
          uint16_t h;
          asm volatile("ld.shared.u16 %0, [%1];" : "=h"(h) : "r"(sb + kF2BItem + 4 * 256 + (lc0 + nt * 8) * 2));
          sfb = h;
        } else {
          asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sfb) : "r"(sb + kF2BItem + s * 256 + (lc0 + nt * 8) * 4));
        }
        if (kb == 0) qmma<0>(acc[nt], a0, a0, a2, a2, b0, b1, R.asc[s], sfb);
        else if (kb == 1) qmma<1>(acc[nt], a0, a0, a2, a2, b0, b1, R.asc[s], sfb);
        else if (kb == 2) qmma<2>(acc[nt], a0, a0, a2, a2, b0, b1, R.asc[s], sfb);
        else qmma<3>(acc[nt], a0, a0, a2, a2, b0, b1, R.asc[s], sfb);
      }
    }
  }

  if (q < nrows) {
    uint32_t* dst = reinterpret_cast<uint32_t*>(a.out + (size_t)R.pair * a.k);
    const int col_base = item * kF2Cols + warp * 16 + (c << 1);
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int col = col_base + nt * 8;
      asm volatile("st.global.u32 [%0], %1;" ::"l"(dst + (col >> 1)),
                   "r"(pack_bf16x2(R.scale * acc[nt][0], R.scale * acc[nt][1])));
    }
  }
}

// --------------------------------------------------------------------------------------------------- the kernel
// the expert table: unique experts in order of their first pair (slot order), each with its pairs (rows); the row
// order within an expert comes from shared-memory atomics (it does not affect any result: rows are independent).
// The ids are read coherently (ld.global.cg): with routing on they were written by this kernel's selection items.
__device__ __forceinline__ void build_tables(const Args& a, Tables& T) {
  const int tid = threadIdx.x;
  if (tid < kMaxPairs) T.ids[tid] = (int16_t)(tid < a.num_pairs ? ldcg_s32(a.ids + tid) : -1);
  for (int i = tid; i < kMaxExpertIds; i += kThreads) {
    T.first[i] = 0x7fff;
    T.cnt[i] = 0;
  }
  __syncthreads();
  const int my_id = tid < kMaxPairs ? T.ids[tid] : -1;
  const bool valid = my_id >= 0 && my_id < a.num_experts;
  int rank = 0;
  if (valid) {
    atomicMin(&T.first[my_id], tid);
    rank = atomicAdd(&T.cnt[my_id], 1);
  }
  __syncthreads();
  const bool leader = valid && T.first[my_id] == tid;
  const int cnt = valid ? T.cnt[my_id] : 0;
  // leaders numbered in pair order (warps 0 and 1 hold the 64 pairs; warp 1 adds warp 0's count); pads likewise
  const unsigned lm = __ballot_sync(0xffffffffu, leader);
  const unsigned pm = __ballot_sync(0xffffffffu, tid < a.num_pairs && !valid);
  if (tid == 0) {
    T.w0_leaders = __popc(lm);
    T.w0_pads = __popc(pm);
  }
  __syncthreads();
  if (tid < kMaxPairs) {
    const int lane = tid & 31;
    const int base_l = tid < 32 ? 0 : T.w0_leaders;
    const int base_p = tid < 32 ? 0 : T.w0_pads;
    const int my_slot = base_l + __popc(lm & ((1u << lane) - 1u));
    const int my_pad = base_p + __popc(pm & ((1u << lane) - 1u));
    if (leader) {
      T.expert[my_slot] = (int16_t)my_id;
      T.nrows[my_slot] = (uint8_t)min(cnt, kMaxRows);
      T.slot_of[my_id] = (int16_t)my_slot;
    }
    if (tid < a.num_pairs && !valid) T.pad[my_pad] = (uint8_t)tid;
    if (tid == kMaxPairs - 1) {
      T.n_unique = base_l + __popc(lm);
      T.n_pad = base_p + __popc(pm);
    }
  }
  __syncthreads();
  if (valid && rank < kMaxRows) {
    T.rows[T.slot_of[my_id]][rank] = (uint8_t)tid;
    T.tok[T.slot_of[my_id]][rank] = (uint8_t)(tid / a.topk);
  }
  __syncthreads();
}

// ------------------------------------------------------------------------------- routing item: tile x slice group
// warp w computes slice group * 4 + w of the tile's 8 columns; the CTA arriving last at the tile's counter sums the
// slices in ascending order (spread's reduction), writes the logits and the routing scores, resets the counter and
// counts the tile done.
__device__ __forceinline__ void route_item(const Args& a, Tables& T, int item) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tile = item / a.groups, group = item - tile * a.groups;
  const int s = group * 4 + warp;
  const int g = lane >> 2, tq = lane & 3;
  const int n0 = tile * 8;
  const int K = a.k;
  if (s < a.S) {
    const int kb = s * a.step;
    float c[4];
    route_slice(c, a.x, a.ldx, a.wr, a.M, K, n0, kb, min(K - kb, a.step) >> 4, g, tq);
    float* p = a.ws + ((size_t)tile * a.S + s) * 128;
    *reinterpret_cast<float2*>(p + g * 8 + tq * 2) = make_float2(c[0], c[1]);
    *reinterpret_cast<float2*>(p + (g + 8) * 8 + tq * 2) = make_float2(c[2], c[3]);
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) T.flag = atomicAdd(a.ctrl + kCtrlTileCnt + tile, 1) == a.groups - 1;
  __syncthreads();
  if (!T.flag) return;
  __threadfence();
  const float* p = a.ws + (size_t)tile * a.S * 128;
  constexpr int E = 384;
  for (int e = tid; e < a.M * 8; e += kThreads) {
    float v[32];
#pragma unroll
    for (int q = 0; q < 32; ++q)
      if (q < a.S) v[q] = ldcg_f32(p + q * 128 + e);
    float acc = 0.f;
#pragma unroll
    for (int q = 0; q < 32; ++q) {
      if (q < a.S) acc = (q == 0) ? v[q] : acc + v[q];
    }
    const int row = e >> 3, col = n0 + (e & 7);
    a.logits[(size_t)row * E + col] = acc;
    float sw;
    unsigned sk;
    tk_score(acc, tk_bias_row(a, row)[col], sw, sk);
    a.wk_w[(size_t)row * E + col] = sw;
    a.wk_k[(size_t)row * E + col] = sk;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    a.ctrl[kCtrlTileCnt + tile] = 0;  // for the next launch (graph replay)
    atomicAdd(a.ctrl + kCtrlScoresDone, 1);
  }
}

// ------------------------------------------------------------------------------------- selection item: one token
__device__ __forceinline__ void select_item(const Args& a, int row) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) {
    unsigned ns = 128;
    while (ld_acquire(a.ctrl + kCtrlScoresDone) < kMaxRouteTiles) {
      __nanosleep(ns);
      ns = min(ns * 2, a.spin_ns_max);
    }
  }
  __syncthreads();  // the scores are visible (thread 0's acquire)
  if (warp == 0) topk_row(a, row, lane);
  __threadfence();
  __syncthreads();
  if (tid == 0) atomicAdd(a.ctrl + kCtrlRoutingDone, 1);
}

// --------------------------------------------------------------------------------------------------- the kernel
__global__ void __launch_bounds__(kThreads, 2) moe_mega_kernel(Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ Tables T;
  __shared__ int s_built;
  const int tid = threadIdx.x;
  long long t_entry = 0;
  if (a.trace != nullptr && tid == 0) t_entry = globaltimer();
  if (tid == 0) {
    T.n_trace = 1;
    s_built = 0;
  }
  const int n_route = a.route_on ? kMaxRouteTiles * a.groups : 0;
  const int n_sel = a.route_on ? a.M : 0;
  const int n_front = n_route + n_sel;
  if (a.pdl) {
    // before the wait only the router weights (constant): this CTA's routing slice group, 8 rows x step columns per
    // warp = step / 64 lines of 128 B per row, into L2 (the predecessor's tail overlaps the DRAM fetch)
    if ((int)blockIdx.x < n_route) {
      const int item = blockIdx.x;
      const int tile = item / a.groups, group = item - tile * a.groups;
      const int s = group * 4 + (tid >> 5);
      if (s < a.S) {
        const int lane = tid & 31;
        const int kb = s * a.step;
        const int cols = min(a.k - kb, a.step);
        const int lines_per_row = (cols * 2 + 127) / 128;
        for (int i = lane; i < 8 * lines_per_row; i += 32) {
          const int row = i / lines_per_row, l = i - row * lines_per_row;
          prefetch_l2(reinterpret_cast<const uint8_t*>(a.wr + (size_t)(tile * 8 + row) * a.k + kb) + l * 128);
        }
      }
    }
    pdl_wait();
    pdl_launch_dependents();
  }
  if (a.trace != nullptr && tid == 0) {
    long long* tr = a.trace + (size_t)blockIdx.x * kTraceItems * 4;
    tr[0] = -1;
    tr[1] = t_entry;
    tr[2] = globaltimer();
    tr[3] = 0;
  }
  // CTA b starts with item b (blockIdx order spreads the FC1 streams one per SM); the counter hands out the rest
  if (tid == 0) {
    T.next = blockIdx.x;
    // L2 prefetch depth of the FC1 streams (K64 slices ahead of the ring): 4 measured best at 22-36 experts
    T.pf_depth = a.pf_depth >= 0 ? a.pf_depth : 4;
  }
  __syncthreads();

  const int per_expert = 2 * a.n_tiles;
  const int fc2_per_expert = a.out_tiles * 2;  // 64-column items
  const bool no_wait = (a.phase_mask & 1) == 0;
  // known once the tables are built (after the routing with route_on):
  int U = 0, n_fc1 = 0, n_items = 0, n_static_experts = 0, n_static = 0;
  int* slice_done = a.ctrl + kCtrlSliceDone;
  int* act_done = a.ctrl + kCtrlActDone;
  int* ready = a.ctrl + kCtrlReady;
  int* ready_list = a.ctrl + kCtrlReadyList;

  // the MoE phases' item counts, computed by every CTA from its tables (the same ids -> the same numbers)
  auto enter_moe = [&]() {
    if (s_built) return;
    if (a.route_on && tid == 0) {
      unsigned ns = 256;
      const unsigned cap = max(a.spin_ns_max, 1024u);  // every CTA of the grid polls this word during the routing
      while (ld_acquire(a.ctrl + kCtrlRoutingDone) < a.M) {
        __nanosleep(ns);
        ns = min(ns * 2, cap);
      }
    }
    __syncthreads();  // the ids / weights are visible (thread 0's acquire)
    build_tables(a, T);
    U = T.n_unique;
    n_fc1 = (a.phase_mask & 1) ? U * per_expert : 0;
    n_items = n_fc1 + ((a.phase_mask & 2) ? U * fc2_per_expert : 0);
    // FC2 items follow the experts' readiness order (phase_mask 2: the slot order, nothing to wait for); a static
    // share for the CTAs left over during FC1 (weights preloaded while waiting) was measured no better
    n_static_experts = no_wait ? U : 0;
    n_static = n_static_experts * fc2_per_expert;
    if (U == 0 && blockIdx.x == 0) {  // nothing routed: zero every pair's output (padding tokens)
      for (int p = 0; p < T.n_pad; ++p) {
        uint32_t* dst = reinterpret_cast<uint32_t*>(a.out + (size_t)T.pad[p] * a.k);
        for (int i = tid; i < a.k / 2; i += kThreads) dst[i] = 0u;
      }
    }
    __syncthreads();
    if (tid == 0) s_built = 1;
    __syncthreads();
  };

  // FC2 pipeline: the item whose weights are in flight / landed in buffer pend_buf, computed when the next item's
  // weights have been issued (or at the end)
  int pend_item = -1, pend_u = 0, pend_buf = 1;
  bool pend_static = false;
  long long* pend_tr = nullptr;

  auto fc2_finish_pending = [&](int wait_groups) {
    if (pend_item < 0) return;
    if (tid == 0) {
      if (pend_static) {
        unsigned ns = 256;
        while (!no_wait && ld_acquire(ready + pend_u) == 0) {
          __nanosleep(ns);
          ns = min(ns * 2, a.spin_ns_max);
        }
      }
      if (pend_tr) pend_tr[2] = globaltimer();
    }
    if (pend_static) __syncthreads();  // the expert's activation is visible (thread 0's acquire)
    Fc2Regs R;
    fc2_load_a(a, T, pend_u, R);
    if (wait_groups == 0) cp_async_wait<0>();
    else cp_async_wait<1>();
    __syncthreads();
    if (pend_tr && tid == 0) pend_tr[2] = globaltimer();  // buffer landed (overwrites the readiness stamp)
    fc2_mma(a, T, smem, pend_u, pend_item, pend_buf, R);
    if (pend_u == 0) {  // zeros for the padding pairs' columns of this item (b12x's phase 2 does the same)
      for (int p = 0; p < T.n_pad; ++p)
        if (tid < kF2Cols / 2)
          reinterpret_cast<uint32_t*>(a.out + (size_t)T.pad[p] * a.k + pend_item * kF2Cols)[tid] = 0u;
    }
    if (pend_tr && tid == 0) pend_tr[3] = globaltimer();
    pend_item = -1;
    pend_tr = nullptr;
  };

  for (;;) {
    // thread 0 alone touches T.next; every thread reads the item from T.item after the barrier (a thread still reading
    // T.next while thread 0 fetched the following item would run a different item: out-of-range expert tables)
    if (tid == 0) T.item = T.next;
    __syncthreads();
    const int gitem = T.item;  // this CTA's item (front items are always valid; MoE items need the tables)
    if (gitem >= n_front) enter_moe();
    if (tid == 0 && gitem < n_front + n_items) T.next = (int)gridDim.x + atomicAdd(a.ctrl + kCtrlNext, 1);  // the following item, fetched ahead
    __syncthreads();
    if (gitem >= n_front + n_items) break;
    long long* tr = nullptr;
    if (a.trace != nullptr && tid == 0) {
      const int slot = T.n_trace++;
      if (slot < kTraceItems) {
        tr = a.trace + ((size_t)blockIdx.x * kTraceItems + slot) * 4;
        tr[0] = gitem;
        tr[1] = globaltimer();
      }
    }
    if (gitem < n_route) {
      route_item(a, T, gitem);
      if (tr && tid == 0) tr[3] = globaltimer();
      continue;
    }
    if (gitem < n_front) {
      select_item(a, gitem - n_route);
      if (tr && tid == 0) tr[3] = globaltimer();
      continue;
    }
    const int item = gitem - n_front;
    if (item < n_fc1) {
      const int u = item / per_expert, t = item - u * per_expert;
      const int otile = t % a.n_tiles;
      fc1_tile(a, T, smem, u, t);
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        T.flag = atomicAdd(slice_done + u * kSlices + otile, 1) == 1;  // the slice's gate and up tiles are both done
        if (tr) tr[2] = globaltimer();
      }
      __syncthreads();
      if (T.flag) {
        __threadfence();
        activation_slice(a, T, u, otile);
        __threadfence();
        __syncthreads();
        if (tid == 0 && atomicAdd(act_done + u, 1) == a.n_tiles - 1) {
          st_release(ready + u, 1);
          if (u >= n_static_experts) {
            const int k = atomicAdd(a.ctrl + kCtrlNReady, 1);
            st_release(ready_list + k, u);
          }
        }
      }
      if (tr && tid == 0) tr[3] = globaltimer();
    } else {
      const int j = item - n_fc1;
      int u, li;
      bool is_static;
      if (j < n_static) {
        u = j / fc2_per_expert;
        li = j - u * fc2_per_expert;
        is_static = true;
      } else {
        const int k = (j - n_static) / fc2_per_expert;
        li = (j - n_static) - k * fc2_per_expert;
        if (tid == 0) {
          int v = k;  // no_wait: the slot order
          unsigned ns = 256;
          while (!no_wait && (v = ld_acquire(ready_list + k)) < 0) {
            __nanosleep(ns);
            ns = min(ns * 2, a.spin_ns_max);
          }
          T.flag = v;
        }
        __syncthreads();
        u = T.flag;
        is_static = false;
      }
      const int buf = pend_buf ^ 1;  // free: the item before the pending one has been computed
      fc2_issue_w(a, smem, T.expert[u], li, buf);
      fc2_finish_pending(1);
      pend_item = li;
      pend_u = u;
      pend_buf = buf;
      pend_static = is_static;
      pend_tr = tr;
    }
  }
  fc2_finish_pending(0);

  // the last CTA out resets the control block for the next launch (graph replay)
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    T.flag = atomicAdd(a.ctrl + kCtrlFinished, 1) == (int)gridDim.x - 1;
  }
  __syncthreads();
  if (T.flag) {
    for (int i = tid; i < kCtrlInts; i += kThreads) a.ctrl[i] = i >= kCtrlReadyList ? -1 : 0;
    __threadfence();
  }
}

__global__ void ctrl_init_kernel(int* ctrl) {
  for (int i = threadIdx.x; i < kCtrlInts; i += blockDim.x) ctrl[i] = i >= kCtrlReadyList ? -1 : 0;
}

}  // namespace b12x_moe_mega

// routing (route_on when `x` is given): x [M][K] bf16 (the router's input, unit inner stride), wr [384][K] bf16, logits
// [M][384] fp32 out, ws fp32 workspace (>= 48 * S * 128 + 2 * 16 * 384 floats), step = the slice width, bias [384]
// fp32, scale, is_padding [M] bool or none, input_ids [M] int32/int64 + bias_vl [384] + image_lo for the image-sentinel
// bias row (or none / 0). ids [M*6] int32 and token_weights [M*6] fp32 are then OUTPUTS of the kernel (num_pairs = M * 6).
void moe_mega(torch::Tensor a_values, torch::Tensor a_scales, torch::Tensor w13, torch::Tensor w13_sf,
              torch::Tensor projections, torch::Tensor intermediate, torch::Tensor w2, torch::Tensor w2_sf,
              torch::Tensor ids, torch::Tensor token_weights, torch::Tensor alpha1, torch::Tensor input_scale,
              torch::Tensor alpha2, torch::Tensor down_scale, torch::Tensor out, torch::Tensor ctrl,
              int64_t num_pairs, int64_t topk, int64_t rows_capacity, int64_t k, int64_t n, double limit,
              int64_t ctas_per_sm, int64_t pf_depth, c10::optional<torch::Tensor> trace, int64_t phase_mask, int64_t spin_ns_max,
              c10::optional<torch::Tensor> x, c10::optional<torch::Tensor> wr, c10::optional<torch::Tensor> logits,
              c10::optional<torch::Tensor> ws, int64_t step, c10::optional<torch::Tensor> bias, double scale,
              c10::optional<torch::Tensor> is_padding, c10::optional<torch::Tensor> input_ids,
              c10::optional<torch::Tensor> bias_vl, int64_t image_lo, bool pdl) {
  using namespace b12x_moe_mega;
  TORCH_CHECK(ids.scalar_type() == torch::kInt32 && token_weights.scalar_type() == torch::kFloat32);
  TORCH_CHECK(alpha1.scalar_type() == torch::kFloat32 && alpha2.scalar_type() == torch::kFloat32);
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.size(1) == k && k % 128 == 0);
  TORCH_CHECK(n == 576, "intermediate 576 (five K128 slices, n64 tail)");
  TORCH_CHECK(0 <= num_pairs && num_pairs <= rows_capacity && rows_capacity <= kMaxPairs);
  TORCH_CHECK(ctrl.scalar_type() == torch::kInt32 && ctrl.numel() >= kCtrlInts);
  TORCH_CHECK(alpha1.numel() <= kMaxExpertIds && alpha1.numel() == alpha2.numel(), "at most 384 experts");
  TORCH_CHECK(w2.numel() * w2.element_size() == alpha2.numel() * k * n / 2, "w2 is not the n64-repacked storage");
  Args a;
  a.a_values = reinterpret_cast<const uint32_t*>(a_values.data_ptr());
  a.a_scales = reinterpret_cast<const uint8_t*>(a_scales.data_ptr());
  a.w13 = reinterpret_cast<const uint32_t*>(w13.data_ptr());
  a.w13_sf = reinterpret_cast<const uint32_t*>(w13_sf.data_ptr());
  a.projections = reinterpret_cast<__nv_bfloat16*>(projections.data_ptr());
  a.intermediate = reinterpret_cast<uint32_t*>(intermediate.data_ptr());
  a.w2 = reinterpret_cast<const uint32_t*>(w2.data_ptr());
  a.w2_sf = reinterpret_cast<const uint8_t*>(w2_sf.data_ptr());
  a.ids = ids.data_ptr<int32_t>();
  a.token_weights = token_weights.data_ptr<float>();
  a.alpha1 = alpha1.data_ptr<float>();
  a.input_scale = input_scale.data_ptr<float>();
  a.alpha2 = alpha2.data_ptr<float>();
  a.down_scale = down_scale.data_ptr<float>();
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.ctrl = ctrl.data_ptr<int32_t>();
  a.trace = trace.has_value() ? reinterpret_cast<long long*>(trace->data_ptr<int64_t>()) : nullptr;
  a.limit = (float)limit;
  a.is_per_expert = input_scale.numel() > 1;
  a.ds_per_expert = down_scale.numel() > 1;
  a.num_pairs = (int)num_pairs;
  a.topk = (int)topk;
  a.rows_capacity = (int)rows_capacity;
  a.k = (int)k;
  a.n = (int)n;
  a.n_tiles = (int)((n + kTileN - 1) / kTileN);
  a.out_tiles = (int)(k / kTileN);
  a.num_experts = (int)alpha1.numel();
  a.pf_depth = (int)pf_depth;
  a.phase_mask = (int)phase_mask;
  a.spin_ns_max = (unsigned)spin_ns_max;
  a.pdl = pdl ? 1 : 0;
  a.route_on = 0;
  a.x = nullptr; a.wr = nullptr; a.logits = nullptr; a.ws = nullptr; a.wk_w = nullptr; a.wk_k = nullptr;
  a.bias = nullptr; a.bias_vl = nullptr; a.tok32 = nullptr; a.tok64 = nullptr; a.is_padding = nullptr;
  a.out_w = nullptr; a.out_ids = nullptr;
  a.ldx = 0; a.M = (int)(num_pairs / (topk > 0 ? topk : 1)); a.step = 0; a.S = 0; a.groups = 0; a.image_lo = 0; a.scale = 0.f;
  if (x.has_value()) {
    TORCH_CHECK(wr.has_value() && logits.has_value() && ws.has_value() && bias.has_value(), "routing needs wr, logits, ws, bias");
    const auto& X = *x;
    const auto& W = *wr;
    TORCH_CHECK(topk == 6 && num_pairs % 6 == 0, "routing: top-6");
    const int M = (int)(num_pairs / 6);
    TORCH_CHECK(1 <= M && M <= kMaxRows, "routing: 1..8 tokens");
    TORCH_CHECK(X.scalar_type() == torch::kBFloat16 && X.dim() == 2 && X.size(0) >= M && X.size(1) == k && X.stride(1) == 1 &&
                X.stride(0) % 8 == 0 && X.data_ptr<at::BFloat16>() != nullptr, "routing: x [M][K] bf16, unit inner stride");
    TORCH_CHECK(W.scalar_type() == torch::kBFloat16 && W.dim() == 2 && W.size(0) == 384 && W.size(1) == k && W.is_contiguous(),
                "routing: router weights [384][K] bf16 contiguous");
    TORCH_CHECK(logits->scalar_type() == torch::kFloat32 && logits->is_contiguous() && logits->numel() >= (int64_t)M * 384);
    TORCH_CHECK(bias->scalar_type() == torch::kFloat32 && bias->numel() == 384 && bias->is_contiguous());
    TORCH_CHECK(step > 0 && step % 16 == 0 && k % 16 == 0, "routing: step a multiple of 16");
    const int S = (int)((k + step - 1) / step);
    TORCH_CHECK(2 <= S && S <= 32, "routing: 2..32 slices (spread)");
    const int groups = (S + 3) / 4;
    TORCH_CHECK(ws->scalar_type() == torch::kFloat32 && ws->is_contiguous() &&
                ws->numel() >= (int64_t)kMaxRouteTiles * S * 128 + 2 * 16 * 384, "routing: workspace too small");
    TORCH_CHECK(ids.numel() >= M * 6 && token_weights.numel() >= M * 6 && ids.is_contiguous() && token_weights.is_contiguous());
    a.route_on = 1;
    a.x = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    a.ldx = (int)X.stride(0);
    a.wr = reinterpret_cast<const __nv_bfloat16*>(W.data_ptr());
    a.M = M;
    a.step = (int)step;
    a.S = S;
    a.groups = groups;
    a.logits = logits->data_ptr<float>();
    a.ws = ws->data_ptr<float>();
    a.wk_w = a.ws + (size_t)kMaxRouteTiles * S * 128;
    a.wk_k = reinterpret_cast<unsigned*>(a.wk_w + (size_t)16 * 384);
    a.bias = bias->data_ptr<float>();
    a.scale = (float)scale;
    a.out_w = token_weights.data_ptr<float>();
    a.out_ids = ids.data_ptr<int32_t>();
    if (is_padding.has_value()) {
      TORCH_CHECK(is_padding->scalar_type() == torch::kBool && is_padding->numel() >= M && is_padding->is_contiguous());
      a.is_padding = is_padding->data_ptr<bool>();
    }
    if (bias_vl.has_value() && image_lo > 0) {
      TORCH_CHECK(input_ids.has_value() && input_ids->numel() >= M && input_ids->is_contiguous(), "routing: input ids for the image test");
      TORCH_CHECK(bias_vl->scalar_type() == torch::kFloat32 && bias_vl->numel() == 384 && bias_vl->is_contiguous());
      a.bias_vl = bias_vl->data_ptr<float>();
      a.image_lo = (long long)image_lo;
      if (input_ids->scalar_type() == torch::kInt32) a.tok32 = input_ids->data_ptr<int32_t>();
      else { TORCH_CHECK(input_ids->scalar_type() == torch::kInt64); a.tok64 = input_ids->data_ptr<int64_t>(); }
    }
  }
  static bool attr = false;
  static int sms = 0;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(moe_mega_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) == cudaSuccess);
    TORCH_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, out.device().index()) == cudaSuccess);
    attr = true;
  }
  if (num_pairs == 0 && !a.route_on) return;
  static int max_resident = 0;
  if (!max_resident) {
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_resident, moe_mega_kernel, kThreads, kSmem) == cudaSuccess);
    TORCH_CHECK(max_resident >= 1, "moe_mega_kernel does not fit on this GPU");
  }
  // every CTA of the grid must be resident at once (FC2 items wait for FC1 items held by other CTAs; selections and
  // MoE items wait for routing items, which CTA b < n_route starts with)
  const int grid = sms * min((int)ctas_per_sm, max_resident);
  if (a.route_on) TORCH_CHECK(grid >= kMaxRouteTiles * a.groups + a.M, "routing: the grid must cover the routing items");
  cudaStream_t st = c10::cuda::getCurrentCUDAStream();
  if (pdl) {  // programmatic stream serialization (graph-capture safe: becomes a programmatic edge)
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kSmem;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, moe_mega_kernel, a) == cudaSuccess, "moe_mega_kernel PDL launch failed");
  } else {
    moe_mega_kernel<<<grid, kThreads, kSmem, st>>>(a);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

// the control block's initial state (also what the kernel leaves behind)
void ctrl_init(torch::Tensor ctrl) {
  TORCH_CHECK(ctrl.scalar_type() == torch::kInt32 && ctrl.numel() >= b12x_moe_mega::kCtrlInts);
  b12x_moe_mega::ctrl_init_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(ctrl.data_ptr<int32_t>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

int64_t ctrl_ints() { return b12x_moe_mega::kCtrlInts; }
int64_t ctrl_ready_list_off() { return b12x_moe_mega::kCtrlReadyList; }
int64_t max_ctas_per_sm() {
  int n = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, b12x_moe_mega::moe_mega_kernel, b12x_moe_mega::kThreads, b12x_moe_mega::kSmem) != cudaSuccess) return 0;
  return n;
}
int64_t trace_items() { return b12x_moe_mega::kTraceItems; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_mega", &moe_mega);
  m.def("route_tiles", []() { return (int64_t)b12x_moe_mega::kMaxRouteTiles; });
  m.def("ctrl_init", &ctrl_init);
  m.def("ctrl_ints", &ctrl_ints);
  m.def("ctrl_ready_list_off", &ctrl_ready_list_off);
  m.def("max_ctas_per_sm", &max_ctas_per_sm);
  m.def("trace_items", &trace_items);
}
