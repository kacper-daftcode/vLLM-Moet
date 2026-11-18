// The batched expert GEMM of the Inkling2 MoE on sm_120 (RTX PRO 6000 / RTX 5090): block-scaled activations x
// block-scaled expert weights -> bf16, for the two formats of the checkpoint and one packed form of the second:
//
//   nvfp4  e2m1 x e2m1, one ue4m3 scale per 16 elements on both sides, an fp32 scale per output row
//          (mma.sync kind::mxf4nvf4, scale_vec::4X, m16n8k64)
//   mxfp8  e4m3 x e4m3, one ue8m0 scale per 32 elements (mma.sync kind::mxf8f6f4, scale_vec::1X, m16n8k32)
//   mxfp6  e4m3 activations x e2m3 weights stored 6 bits per value, ue8m0 scales (the same instruction with an e2m3
//          B operand; the weights are unpacked into byte containers as they are read)
//
// vLLM's Inkling2 MoE calls FlashInfer's TRT-LLM-gen batched GEMM here (models/inkling2/ops/trtllm_bmm.py), which
// exists as sm_100 cubins only. This file is that call for sm_120: the same inputs (expert-sorted activation rows in
// tiles of `tile` rows, each tile one expert; the tiles' expert and row limit; the number of tiles on the device), the
// same output (bf16 [rows, N], out[r] = bf16(row_scale[r] * act[r] @ weight[expert(r)]^T)), row-major operands.
//
// One persistent kernel per (format, shape): every CTA of a grid of SMs x CTAs-per-SM takes work items from a counter
// until none is left (CTA b starts with item b), and the last CTA out resets the counter, so the launch replays from
// a CUDA graph. A work item is a row chunk of one tile and a column block of the output:
//
//   rows8   8 rows x 128 columns   (tile 8: the m16 rows are the 8 rows twice)
//   mt1    16 rows x 128 columns   (tile 16)
//   mt2    32 rows x 128 columns   (tile 32)
//   mt4    64 rows x  64 columns   (tiles 64, 128, 256: tile / 64 row chunks)
//
// The four warps of a CTA split the item: warp w computes m16 tile w % MT of the row chunk and column group w / MT.
// K runs in steps of 64 elements through a ring of cp.async stages (activations from the row-major rows, weights
// and weight scales from the prepared layout below); the rows' scales are staged once per item by plain loads.
// Weights are read as one sequential stream per item, `pf_depth` steps ahead into L2. The ring's copies carry no L2
// eviction policy: in this kernel a cp.async with an L2::cache_hint policy operand ends the launch with an illegal
// instruction on sm_120 (nvcc 13.0), the same copy without the hint runs (test_bmm_sm120.py is the witness).
//
// Prepared weight layout (inkling2_bmm_sm120.py: prepare): [expert][N / 128][K / 64][128][bytes per step], i.e. the
// row-major [N][K] bytes of each 128-row block transposed to step-major, so that a K64 step of a column block is one
// contiguous chunk (32 B per row for nvfp4, 64 B for mxfp8, 48 B for mxfp6); the block scales likewise (4 B per row
// and step for nvfp4, 2 B for mxfp8 / mxfp6).
//
// Numerics: per output element the K64 (nvfp4) or K32 (mxfp8 / mxfp6) block-scaled QMMAs accumulate in the fp32 C
// operand in K order; nvfp4 multiplies the sum by the row's fp32 scale (mul.rn); cvt.rn to bf16.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <vector>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace inkling2_bmm {

constexpr int kThreads = 128;
constexpr int kPrepN = 128;    // rows of a prepared weight block
constexpr int kMaxSteps = 72;  // K <= 4608

enum : int { kNvfp4 = 0, kMxfp8 = 1, kMxfp6 = 2 };

template <int F>
struct Fmt {
  static constexpr int kb = F == kNvfp4 ? 1 : 2;                       // QMMAs per K64 step
  static constexpr int abps = F == kNvfp4 ? 32 : 64;                   // activation bytes per row and step
  static constexpr int wbps = F == kNvfp4 ? 32 : (F == kMxfp8 ? 64 : 48);  // weight bytes per row and step
  static constexpr int sfps = F == kNvfp4 ? 4 : 2;                     // scale bytes per row and step (both sides)
  static constexpr int stages = F == kNvfp4 ? 6 : 4;
};

struct Args {
  const uint8_t* act;        // [rows][steps * abps]
  const uint8_t* act_sf;     // [rows][steps * sfps]
  const float* row_scale;    // [rows], nvfp4 only
  const uint8_t* w;          // prepared [experts][n / 128][steps][128][wbps]
  const uint8_t* w_sf;       // prepared [experts][n / 128][steps][128][sfps]
  __nv_bfloat16* out;        // [rows][n]
  const int32_t* cta_expert; // [tiles] local expert of each tile
  const int32_t* cta_limit;  // [tiles] end row of each tile
  const int32_t* num_ctas;   // [1] tiles in use
  int* ctrl;                 // [2] next item, CTAs finished
  int n;
  int steps;                 // k / 64
  int tile;
  int pf_depth;
};

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
__device__ __forceinline__ uint32_t ld_sh32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr));
  return v;
}
__device__ __forceinline__ uint32_t ld_sh8(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v) : "r"(addr));
  return v;
}
// two fp32 -> two bf16 in one word (round to nearest even, no saturation: as a tensor cast)
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint16_t l, h;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(l) : "f"(lo));
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(hi));
  return (uint32_t)l | ((uint32_t)h << 16);
}

// nvfp4: A 16 x 64 e2m1 (a0 / a2: row q, k 8c.. / 32 + 8c..; a1 / a3: row q + 8), B 64 x 8 e2m1 (column q),
// scale words of four ue4m3 (one per 16 k): A's from the lane's row (even lanes of a quad: row q, odd: q + 8),
// B's from the quad's column.
__device__ __forceinline__ void qmma_nvfp4(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %12}, {%11}, {%12, %12};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(0));
}
// mxfp8: A 16 x 32 e4m3 (a0 / a2: row q, k 4c.. / 16 + 4c..), B 32 x 8 e4m3, one ue8m0 scale in byte 0 of each word
__device__ __forceinline__ void qmma_mxfp8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %12}, {%11}, {%12, %12};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(0));
}
// mxfp6: the same with an e2m3 B operand (six bits per value in the low bits of a byte container)
__device__ __forceinline__ void qmma_mxfp6(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %12}, {%11}, {%12, %12};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(0));
}

// four 6-bit values (24 bits at bit offset 24 * c of a 96-bit group of three words at `addr`) into four byte containers
__device__ __forceinline__ uint32_t unpack6x4(uint32_t addr, int c) {
  const int word = (3 * c) >> 2;                 // 0, 0, 1, 2
  const int shift = 24 * c - 32 * word;          // 0, 24, 16, 8
  const uint32_t lo = ld_sh32(addr + word * 4);
  const uint32_t hi = shift > 8 ? ld_sh32(addr + word * 4 + 4) : 0u;
  const uint32_t v = static_cast<uint32_t>(((static_cast<uint64_t>(hi) << 32) | lo) >> shift) & 0xFFFFFFu;
  return (v & 0x3Fu) | ((v & 0xFC0u) << 2) | ((v & 0x3F000u) << 4) | ((v & 0xFC0000u) << 6);
}

// MT: m16 tiles of an item's row chunk (1, 2, 4); R8: the chunk has 8 rows (MT == 1)
template <int F, int MT, bool R8>
struct Shape {
  static constexpr int rows = R8 ? 8 : 16 * MT;
  static constexpr int cols = MT == 4 ? 64 : 128;      // an item's output columns
  static constexpr int groups = 4 / MT;                // column groups (one per warp of an m16 tile)
  static constexpr int nt = cols / groups / 8;         // n8 tiles per warp (4 or 8)
  static constexpr int a_bytes = rows * Fmt<F>::abps;
  static constexpr int b_bytes = cols * Fmt<F>::wbps;
  static constexpr int s_bytes = cols * Fmt<F>::sfps;
  static constexpr int stage = a_bytes + b_bytes + s_bytes;
  static constexpr int asf_row = (kMaxSteps * Fmt<F>::sfps + 15) / 16 * 16;
  static constexpr int smem = Fmt<F>::stages * stage + rows * asf_row;
};

template <int F, int MT, bool R8>
__global__ void __launch_bounds__(kThreads) bmm_kernel(Args a) {
  using S = Shape<F, MT, R8>;
  using Ft = Fmt<F>;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_item, s_next, s_flag;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int q = lane >> 2, c = lane & 3;
  const int mi = warp % MT, ng = warp / MT;
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t s_asf = sbase + Ft::stages * S::stage;

  const int chunks = a.tile / S::rows;               // row chunks of a tile
  const int nblocks = a.n / S::cols;                 // column blocks
  const int per_tile = chunks * nblocks;
  const int n_items = __ldg(a.num_ctas) * per_tile;
  const int asf_bytes = a.steps * Ft::sfps;          // a row's scale bytes
  const size_t a_row_bytes = (size_t)a.steps * Ft::abps;
  if (tid == 0) s_next = blockIdx.x;
  __syncthreads();

  for (;;) {
    if (tid == 0) {
      s_item = s_next;
      if (s_next < n_items) s_next = (int)gridDim.x + atomicAdd(a.ctrl, 1);
    }
    __syncthreads();
    const int item = s_item;
    if (item >= n_items) break;
    const int t = item / per_tile, r = item - t * per_tile;
    const int nb = r / chunks, ch = r - nb * chunks;
    const int row0 = t * a.tile + ch * S::rows;
    const int limit = __ldg(a.cta_limit + t);
    if (row0 >= limit) continue;
    const int e = __ldg(a.cta_expert + t);
    const int col0 = nb * S::cols;                   // first output column of the item
    const size_t pblock = (size_t)e * (a.n / kPrepN) + col0 / kPrepN;
    const int half = col0 % kPrepN;
    const uint8_t* wsrc = a.w + pblock * a.steps * kPrepN * Ft::wbps + (size_t)half * Ft::wbps;
    const uint8_t* ssrc = a.w_sf + pblock * a.steps * kPrepN * Ft::sfps + (size_t)half * Ft::sfps;
    const uint8_t* asrc = a.act + (size_t)row0 * a_row_bytes;

    // the rows' scales, once per item, by plain 8-byte loads (a row's scales are 72 B at K 2304 / mxfp8, so 16-byte
    // granules do not fit). Visible to the CTA after the ring's first barrier.
    {
      const int per_row = (asf_bytes + 7) / 8;
      for (int g = tid; g < S::rows * per_row; g += kThreads) {
        const int row = g / per_row, part = g - row * per_row;
        const uint64_t v = *reinterpret_cast<const uint64_t*>(a.act_sf + (size_t)(row0 + row) * asf_bytes + part * 8);
        asm volatile("st.shared.u64 [%0], %1;" ::"r"(s_asf + row * S::asf_row + part * 8), "l"(v));
      }
    }

    auto issue = [&](int j) {
      const uint32_t st = sbase + (j % Ft::stages) * S::stage;
      constexpr int n_a = S::a_bytes / 16, n_b = S::b_bytes / 16, n_s = S::s_bytes / 16;
      constexpr int a_parts = Ft::abps / 16;
#pragma unroll
      for (int g = tid; g < n_a + n_b + n_s; g += kThreads) {
        if (g < n_a) {
          const int row = g / a_parts, part = g % a_parts;
          cp_async16(st + g * 16, asrc + (size_t)row * a_row_bytes + (size_t)j * Ft::abps + part * 16);
        } else if (g < n_a + n_b) {
          cp_async16(st + g * 16, wsrc + (size_t)j * kPrepN * Ft::wbps + (size_t)(g - n_a) * 16);
        } else {
          cp_async16(st + g * 16, ssrc + (size_t)j * kPrepN * Ft::sfps + (size_t)(g - n_a - n_b) * 16);
        }
      }
      const int pj = j + a.pf_depth;
      if (a.pf_depth > 0 && pj < a.steps) {
        constexpr int lines = S::b_bytes / 128;
        for (int l = tid; l < lines; l += kThreads) prefetch_l2(wsrc + (size_t)pj * kPrepN * Ft::wbps + (size_t)l * 128);
      }
    };

#pragma unroll 1
    for (int j = 0; j < Ft::stages - 1; ++j) {
      issue(j);
      cp_async_commit();
    }

    float acc[S::nt][4];
#pragma unroll
    for (int nt = 0; nt < S::nt; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[nt][i] = 0.f;

    const int ra = R8 ? q : mi * 16 + q;             // the lane's two rows within the chunk (R8: the same row)
    const int rb = R8 ? q : ra + 8;
    const int col_w = ng * (S::cols / S::groups);    // the warp's first column within the item

#pragma unroll 1
    for (int j = 0; j < a.steps; ++j) {
      cp_async_wait<Ft::stages - 2>();
      __syncthreads();
      if (j + Ft::stages - 1 < a.steps) issue(j + Ft::stages - 1);
      cp_async_commit();
      const uint32_t st = sbase + (j % Ft::stages) * S::stage;
      const uint32_t stb = st + S::a_bytes, sts = stb + S::b_bytes;
#pragma unroll
      for (int kb = 0; kb < Ft::kb; ++kb) {
        const int half_k = kb * (Ft::abps / Ft::kb);           // byte offset of the K32 half within a row's step
        const uint32_t a0 = ld_sh32(st + ra * Ft::abps + half_k + c * 4);
        const uint32_t a1 = ld_sh32(st + rb * Ft::abps + half_k + c * 4);
        const uint32_t a2 = ld_sh32(st + ra * Ft::abps + half_k + 16 + c * 4);
        const uint32_t a3 = ld_sh32(st + rb * Ft::abps + half_k + 16 + c * 4);
        const int rs = (c & 1) ? rb : ra;                      // the row whose scales this lane supplies
        uint32_t sfa;
        if (F == kNvfp4) sfa = ld_sh32(s_asf + rs * S::asf_row + j * 4);
        else sfa = ld_sh8(s_asf + rs * S::asf_row + j * 2 + kb);
#pragma unroll
        for (int nt = 0; nt < S::nt; ++nt) {
          const int col = col_w + nt * 8 + q;
          uint32_t b0, b1, sfb;
          if (F == kNvfp4) {
            b0 = ld_sh32(stb + col * 32 + c * 4);
            b1 = ld_sh32(stb + col * 32 + 16 + c * 4);
            sfb = ld_sh32(sts + col * 4);
            qmma_nvfp4(acc[nt], a0, a1, a2, a3, b0, b1, sfa, sfb);
          } else if (F == kMxfp8) {
            b0 = ld_sh32(stb + col * 64 + kb * 32 + c * 4);
            b1 = ld_sh32(stb + col * 64 + kb * 32 + 16 + c * 4);
            sfb = ld_sh8(sts + col * 2 + kb);
            qmma_mxfp8(acc[nt], a0, a1, a2, a3, b0, b1, sfa, sfb);
          } else {
            b0 = unpack6x4(stb + col * 48 + kb * 24, c);
            b1 = unpack6x4(stb + col * 48 + kb * 24 + 12, c);
            sfb = ld_sh8(sts + col * 2 + kb);
            qmma_mxfp6(acc[nt], a0, a1, a2, a3, b0, b1, sfa, sfb);
          }
        }
      }
    }
    cp_async_wait<0>();

    // epilogue: d0, d1 = (row a, columns 2c, 2c + 1), d2, d3 = (row b, the same columns) of each n8 tile
    const int row_a = row0 + ra, row_b = row0 + rb;
    float sa = 1.f, sb = 1.f;
    if (F == kNvfp4) {
      if (row_a < limit) sa = __ldg(a.row_scale + row_a);
      if (!R8 && row_b < limit) sb = __ldg(a.row_scale + row_b);
    }
#pragma unroll
    for (int nt = 0; nt < S::nt; ++nt) {
      const int col = col0 + col_w + nt * 8 + c * 2;
      if (row_a < limit) {
        const uint32_t v = F == kNvfp4 ? pack_bf16x2(__fmul_rn(sa, acc[nt][0]), __fmul_rn(sa, acc[nt][1]))
                                       : pack_bf16x2(acc[nt][0], acc[nt][1]);
        *reinterpret_cast<uint32_t*>(a.out + (size_t)row_a * a.n + col) = v;
      }
      if (!R8 && row_b < limit) {
        const uint32_t v = F == kNvfp4 ? pack_bf16x2(__fmul_rn(sb, acc[nt][2]), __fmul_rn(sb, acc[nt][3]))
                                       : pack_bf16x2(acc[nt][2], acc[nt][3]);
        *reinterpret_cast<uint32_t*>(a.out + (size_t)row_b * a.n + col) = v;
      }
    }
    __syncthreads();  // the ring is free for the next item
  }

  // the last CTA out resets the counter for the next launch (graph replay)
  if (tid == 0) {
    __threadfence();
    s_flag = atomicAdd(a.ctrl + 1, 1) == (int)gridDim.x - 1;
    if (s_flag) {
      a.ctrl[0] = 0;
      a.ctrl[1] = 0;
      __threadfence();
    }
  }
}

// one QMMA with the registers of every lane given by the host: the operand layouts are established by experiment
// (test_qmma_layout_sm120.py), not taken on trust
__global__ void probe_kernel(int kind, const uint32_t* a, const uint32_t* b, const uint32_t* sfa, const uint32_t* sfb,
                             float* d) {
  const int l = threadIdx.x;
  float x[4] = {0.f, 0.f, 0.f, 0.f};
  const uint32_t a0 = a[l * 4], a1 = a[l * 4 + 1], a2 = a[l * 4 + 2], a3 = a[l * 4 + 3];
  const uint32_t b0 = b[l * 2], b1 = b[l * 2 + 1];
  if (kind == kNvfp4) qmma_nvfp4(x, a0, a1, a2, a3, b0, b1, sfa[l], sfb[l]);
  else if (kind == kMxfp8) qmma_mxfp8(x, a0, a1, a2, a3, b0, b1, sfa[l], sfb[l]);
  else qmma_mxfp6(x, a0, a1, a2, a3, b0, b1, sfa[l], sfb[l]);
#pragma unroll
  for (int i = 0; i < 4; ++i) d[l * 4 + i] = x[i];
}

struct Launch {
  void (*fn)(Args);
  int smem;
  int max_resident;  // CTAs per SM the occupancy query grants (0: not asked yet)
};

template <int F, int MT, bool R8>
Launch make_launch() {
  return Launch{bmm_kernel<F, MT, R8>, Shape<F, MT, R8>::smem, 0};
}

// [format][shape]: shape 0 rows8, 1 mt1, 2 mt2, 3 mt4
Launch g_launch[3][4] = {
    {make_launch<kNvfp4, 1, true>(), make_launch<kNvfp4, 1, false>(), make_launch<kNvfp4, 2, false>(), make_launch<kNvfp4, 4, false>()},
    {make_launch<kMxfp8, 1, true>(), make_launch<kMxfp8, 1, false>(), make_launch<kMxfp8, 2, false>(), make_launch<kMxfp8, 4, false>()},
    {make_launch<kMxfp6, 1, true>(), make_launch<kMxfp6, 1, false>(), make_launch<kMxfp6, 2, false>(), make_launch<kMxfp6, 4, false>()},
};

int shape_of_tile(int64_t tile) { return tile == 8 ? 0 : tile == 16 ? 1 : tile == 32 ? 2 : 3; }

}  // namespace inkling2_bmm

// out[r] = bf16(row_scale[r] * act[r] @ weight[expert(r)]^T) for the rows of the first num_ctas[0] tiles.
// fmt 0 nvfp4, 1 mxfp8, 2 mxfp6 (e2m3 weights, 6 bits per value); weight / weight_scale in the prepared layout.
void bmm_run(int64_t fmt, int64_t tile, torch::Tensor weight, torch::Tensor weight_scale, torch::Tensor act,
             torch::Tensor act_scale, c10::optional<torch::Tensor> row_scale, torch::Tensor out,
             torch::Tensor cta_expert, torch::Tensor cta_limit, torch::Tensor num_ctas, torch::Tensor ctrl, int64_t n,
             int64_t k, int64_t ctas_per_sm, int64_t pf_depth) {
  using namespace inkling2_bmm;
  TORCH_CHECK(fmt >= 0 && fmt <= 2, "fmt 0 nvfp4, 1 mxfp8, 2 mxfp6");
  TORCH_CHECK(tile == 8 || tile == 16 || tile == 32 || tile == 64 || tile == 128 || tile == 256, "tile ", tile);
  TORCH_CHECK(n % kPrepN == 0 && k % 64 == 0 && k / 64 <= kMaxSteps && k / 64 >= 8, "n ", n, " k ", k);
  const int steps = (int)(k / 64);
  const int abps = fmt == kNvfp4 ? 32 : 64, wbps = fmt == kNvfp4 ? 32 : (fmt == kMxfp8 ? 64 : 48), sfps = fmt == kNvfp4 ? 4 : 2;
  const int64_t max_ctas = cta_expert.numel();
  const int64_t rows = max_ctas * tile;
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.size(1) == n && out.size(0) >= rows);
  TORCH_CHECK(act.is_contiguous() && act_scale.is_contiguous() && weight.is_contiguous() && weight_scale.is_contiguous() && out.is_contiguous());
  TORCH_CHECK((int64_t)act.numel() * (int64_t)act.element_size() >= rows * steps * abps, "activation rows");
  TORCH_CHECK((int64_t)act_scale.numel() * (int64_t)act_scale.element_size() >= rows * steps * sfps, "activation scales");
  TORCH_CHECK(weight.numel() * weight.element_size() % (n * (int64_t)steps * wbps) == 0, "weights are not [experts][n][k]");
  const int64_t experts = weight.numel() * weight.element_size() / (n * (int64_t)steps * wbps);
  TORCH_CHECK(weight_scale.numel() * weight_scale.element_size() == experts * n * steps * sfps, "weight scales");
  TORCH_CHECK(cta_expert.scalar_type() == torch::kInt32 && cta_limit.scalar_type() == torch::kInt32 && num_ctas.scalar_type() == torch::kInt32);
  TORCH_CHECK(cta_limit.numel() >= max_ctas && num_ctas.numel() >= 1);
  TORCH_CHECK(ctrl.scalar_type() == torch::kInt32 && ctrl.numel() >= 2);
  TORCH_CHECK(fmt != kNvfp4 || (row_scale.has_value() && row_scale->scalar_type() == torch::kFloat32 && row_scale->numel() >= rows));
  if (max_ctas == 0) return;
  Args a;
  a.act = reinterpret_cast<const uint8_t*>(act.data_ptr());
  a.act_sf = reinterpret_cast<const uint8_t*>(act_scale.data_ptr());
  a.row_scale = row_scale.has_value() ? row_scale->data_ptr<float>() : nullptr;
  a.w = reinterpret_cast<const uint8_t*>(weight.data_ptr());
  a.w_sf = reinterpret_cast<const uint8_t*>(weight_scale.data_ptr());
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.cta_expert = cta_expert.data_ptr<int32_t>();
  a.cta_limit = cta_limit.data_ptr<int32_t>();
  a.num_ctas = num_ctas.data_ptr<int32_t>();
  a.ctrl = ctrl.data_ptr<int32_t>();
  a.n = (int)n;
  a.steps = steps;
  a.tile = (int)tile;
  a.pf_depth = (int)pf_depth;
  Launch& L = g_launch[fmt][shape_of_tile(tile)];
  static int sms = 0;
  if (!sms) TORCH_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, out.device().index()) == cudaSuccess);
  if (!L.max_resident) {
    TORCH_CHECK(cudaFuncSetAttribute(L.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, L.smem) == cudaSuccess);
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&L.max_resident, L.fn, kThreads, L.smem) == cudaSuccess);
    TORCH_CHECK(L.max_resident >= 1, "the kernel does not fit on this GPU");
  }
  // the grid never exceeds the work there can be (every tile in use, every item)
  const int shape = shape_of_tile(tile);
  const int64_t item_rows = shape == 0 ? 8 : shape == 1 ? 16 : shape == 2 ? 32 : 64;
  const int64_t item_cols = shape == 3 ? 64 : 128;
  const int64_t max_items = max_ctas * (tile / item_rows) * (n / item_cols);
  const int64_t want = (int64_t)sms * std::min<int64_t>(ctas_per_sm, L.max_resident);
  const int grid = (int)std::min<int64_t>(want, max_items);
  L.fn<<<grid, kThreads, L.smem, c10::cuda::getCurrentCUDAStream()>>>(a);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "expert GEMM launch: ", cudaGetErrorString(err));
}

// kind 0 nvfp4, 1 mxfp8, 2 mxfp6; a [32][4], b [32][2], sfa [32], sfb [32] (int32 registers of every lane) -> d [32][4]
void qmma_probe(int64_t kind, torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor d) {
  TORCH_CHECK(a.numel() == 128 && b.numel() == 64 && sfa.numel() == 32 && sfb.numel() == 32 && d.numel() == 128);
  TORCH_CHECK(d.scalar_type() == torch::kFloat32 && a.element_size() == 4 && b.element_size() == 4);
  inkling2_bmm::probe_kernel<<<1, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (int)kind, reinterpret_cast<const uint32_t*>(a.data_ptr()), reinterpret_cast<const uint32_t*>(b.data_ptr()),
      reinterpret_cast<const uint32_t*>(sfa.data_ptr()), reinterpret_cast<const uint32_t*>(sfb.data_ptr()),
      d.data_ptr<float>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

// (shared memory bytes, CTAs per SM granted, registers per thread) of the kernel of (fmt, tile)
std::vector<int64_t> kernel_facts(int64_t fmt, int64_t tile) {
  using namespace inkling2_bmm;
  Launch& L = g_launch[fmt][shape_of_tile(tile)];
  int resident = 0;
  cudaFuncSetAttribute(L.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, L.smem);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, L.fn, kThreads, L.smem);
  cudaFuncAttributes attr{};
  cudaFuncGetAttributes(&attr, L.fn);
  return {L.smem, resident, attr.numRegs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bmm_run", &bmm_run);
  m.def("qmma_probe", &qmma_probe);
  m.def("kernel_facts", &kernel_facts);
}
