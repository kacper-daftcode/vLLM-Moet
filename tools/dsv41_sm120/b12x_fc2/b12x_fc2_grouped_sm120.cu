// FC2 (down projection) of b12x's compact W4A8 micro MoE path on sm_120, grouped by expert.
//
// b12x's direct-route phase 2 (W4A8MaterializedPhase2Kernel, direct_routes) runs one CTA per (token, expert) pair
// and 128-column output tile with the pair's row broadcast over the M16 operand, so an expert chosen by several of
// the step's tokens has its w2 tile read once per pair. Here the CTA of a pair's tile exits unless the pair is the
// first of its expert (the leader); the leader's CTA computes every pair of the expert as a row of the M16 operand.
//
// Inputs and output are b12x's: the `intermediate` scratch region (rows of MXFP8 values, 640 bytes each, then one
// scale word per (K128 slice, row) after rows_capacity rows), the n64-repacked w2 with its scales, and route_output
// [pairs, hidden] bf16 with the router weight applied, which b12x's TopKSum then reduces. Per output element the
// MMA instructions, operands, K order and epilogue are b12x's (n64_repacked + n64_tail, deterministic output), so
// route_output has the same bits. All 4.5 K128 slices of a tile are requested at once.
//
// Shapes: intermediate 576 (five K128 slices, the last one half), hidden a multiple of 128, at most 64 pairs and at
// most 8 pairs per expert (each token chooses an expert once: tokens <= 8), int32 expert ids. A pair with an id outside
// [0, experts) (a padding token: -1) gets zeros in route_output, as from b12x's phase 2.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace b12x_fc2 {

constexpr int kThreads = 128;  // four warps, 32 output columns each
constexpr int kTileN = 128;
constexpr int kMaxRows = 8;    // rows 8-15 of the M16 operand repeat rows 0-7 and are not stored
constexpr int kSlices = 5;
constexpr int kTotalK32 = 4 * kSlices - 2;  // the last slice holds two K32 columns

constexpr int kAStage = kMaxRows * 128;
constexpr int kBStage = 4 * 4 * 32 * 16;  // K32 x N32 x lane x 16 bytes
constexpr int kSfbStage = 128 * 4;
constexpr int kOffA = 0;
constexpr int kOffSfa = kOffA + kSlices * kAStage;
constexpr int kOffB = kOffSfa + kSlices * kMaxRows * 4;
constexpr int kOffSfb = kOffB + kSlices * kBStage;
constexpr int kSmem = kOffSfb + kSlices * kSfbStage;

struct Args {
  const uint32_t* intermediate;
  const uint32_t* w2;
  const uint8_t* w2_sf;
  const int32_t* ids;
  const float* token_weights;
  const float* down_alpha;
  const float* global_scale;
  __nv_bfloat16* out;
  int gs_per_expert;
  int num_experts;
  int num_pairs;
  int rows_capacity;
  int out_tiles;
  int hidden;
};

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

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

// b12x's mxfp8_mma_m16n8k32_f32_e2m1 with bid_a = bid_b = KB, tid_a = tid_b = 0
template <int KB>
__device__ __forceinline__ void qmma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %13}, {%11}, {%12, %13};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(KB), "n"(0));
}

// b12x's pack_f32x2_to_bfloat2
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.satfinite.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}

__global__ void __launch_bounds__(kThreads, 2) fc2_grouped_kernel(Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_rows[kMaxRows];
  __shared__ int s_n;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int p = blockIdx.x / a.out_tiles;
  const int tile = blockIdx.x - p * a.out_tiles;
  if (p >= a.num_pairs) return;
  const int e = __ldg(a.ids + p);
  if (e < 0 || e >= a.num_experts) {
    // b12x's phase 2 stores zeros for an invalid route (padding tokens: id -1)
    a.out[(size_t)p * a.hidden + tile * kTileN + tid] = __float2bfloat16_rn(0.f);
    return;
  }

  if (warp == 0) {
    const int v0 = lane < a.num_pairs ? __ldg(a.ids + lane) : -1;
    const int v1 = lane + 32 < a.num_pairs ? __ldg(a.ids + lane + 32) : -1;
    int n = 0;
    bool leader = true;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int i = h * 32 + lane;
      const unsigned eq = __ballot_sync(0xffffffffu, (h ? v1 : v0) == e);
      const unsigned before = __ballot_sync(0xffffffffu, i < p);
      if (eq & before) leader = false;
      const unsigned mine = eq & ~before;
      if ((mine >> lane) & 1u) {
        const int slot = n + __popc(mine & ((1u << lane) - 1u));
        if (slot < kMaxRows) s_rows[slot] = i;
      }
      n += __popc(mine);
    }
    if (lane == 0) s_n = leader ? min(n, kMaxRows) : 0;
  }
  __syncthreads();
  const int nrows = s_n;
  if (nrows == 0) return;

  const int words_per_row = kSlices * 32;
  const size_t expert_words = (size_t)a.out_tiles * kTileN * kTotalK32 * 4;
  const uint32_t* wsrc = a.w2 + (size_t)e * expert_words + (size_t)tile * (kTileN * kTotalK32 * 4);
  const uint8_t* sfsrc = a.w2_sf + ((size_t)e * a.out_tiles + tile) * kTileN * kTotalK32;
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

#pragma unroll
  for (int s = 0; s < kSlices; ++s) {
    const bool tail = s == kSlices - 1;
    // A: the expert's rows, 16-byte vectors XOR-swizzled by row (b12x's grouped staging)
    for (int idx = tid; idx < nrows * 8; idx += kThreads) {
      const int row = idx >> 3, vec = idx & 7, pv = vec ^ (row & 7);
      cp_async16(sbase + kOffA + s * kAStage + row * 128 + pv * 16,
                 a.intermediate + (size_t)s_rows[row] * words_per_row + s * 32 + vec * 4);
    }
    if (tid < nrows)
      cp_async4(sbase + kOffSfa + (s * kMaxRows + tid) * 4,
                a.intermediate + (size_t)a.rows_capacity * words_per_row + s * a.rows_capacity + s_rows[tid]);
    // B: the slice's 8 KB in b12x's shared layout; K32 columns 2-3 of the tail slice are zero
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = tid + i * kThreads;
      const int l = idx & 31, n32 = (idx >> 5) & 3, kb = idx >> 7;
      const uint32_t dst = sbase + kOffB + s * kBStage + ((kb * 4 + n32) * 32 + l) * 16;
      if (!tail || kb < 2) cp_async16(dst, wsrc + s * 2048 + kb * 512 + n32 * 128 + l * 4);
      else asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(dst), "r"(0u));
    }
    if (!tail) {
      cp_async4(sbase + kOffSfb + s * kSfbStage + tid * 4, sfsrc + s * 512 + tid * 4);
    } else {
      // two scale bytes per output column in the tail slice, the other two zero
      const uint8_t* src = sfsrc + s * 512 + tid * 2;
      const uint32_t v = (uint32_t)__ldg(src) | ((uint32_t)__ldg(src + 1) << 8);
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(sbase + kOffSfb + s * kSfbStage + tid * 4), "r"(v));
    }
    cp_async_commit();
  }

  const int q = lane >> 2, c = lane & 3;
  float acc[4][4];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[nt][j] = 0.f;

#pragma unroll
  for (int s = 0; s < kSlices; ++s) {
    if (s == 0) cp_async_wait<kSlices - 1>();
    else if (s == 1) cp_async_wait<kSlices - 2>();
    else if (s == 2) cp_async_wait<kSlices - 3>();
    else if (s == 3) cp_async_wait<kSlices - 4>();
    else cp_async_wait<0>();
    __syncthreads();
    uint32_t asc;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(asc) : "r"(sbase + kOffSfa + (s * kMaxRows + q) * 4));
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      const int u_phys = (kb * 2 + (c >> 1)) ^ q;
      uint32_t a0, a2;
      asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                   : "=r"(a0), "=r"(a2)
                   : "r"(sbase + kOffA + s * kAStage + q * 128 + u_phys * 16 + (c & 1) * 8));
      uint32_t w[4];
      asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
                   : "r"(sbase + kOffB + s * kBStage + ((kb * 4 + warp) * 32 + lane) * 16));
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        uint32_t b0, b1;
        spread_e2m1x8(w[nt], b0, b1);
        uint32_t sfb;
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(sfb)
                     : "r"(sbase + kOffSfb + s * kSfbStage + ((warp * 4 + nt) * 8 + q) * 4));
        if (kb == 0) qmma<0>(acc[nt], a0, a0, a2, a2, b0, b1, asc, sfb);
        else if (kb == 1) qmma<1>(acc[nt], a0, a0, a2, a2, b0, b1, asc, sfb);
        else if (kb == 2) qmma<2>(acc[nt], a0, a0, a2, a2, b0, b1, asc, sfb);
        else qmma<3>(acc[nt], a0, a0, a2, a2, b0, b1, asc, sfb);
      }
    }
  }

  if (q < nrows) {
    const int pair = s_rows[q];
    const float down_scale = __ldg(a.down_alpha + e) * __ldg(a.global_scale + (a.gs_per_expert ? e : 0));
    const float scale = down_scale * __ldg(a.token_weights + pair);
    uint32_t* dst = reinterpret_cast<uint32_t*>(a.out + (size_t)pair * a.hidden);
    const int col_base = tile * kTileN + warp * 32 + (c << 1);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int col = col_base + nt * 8;
      asm volatile("st.global.u32 [%0], %1;" ::"l"(dst + (col >> 1)),
                   "r"(pack_bf16x2(scale * acc[nt][0], scale * acc[nt][1])));
    }
  }
}

}  // namespace b12x_fc2

void fc2_grouped(torch::Tensor intermediate, torch::Tensor w2, torch::Tensor w2_sf, torch::Tensor ids,
                 torch::Tensor token_weights, torch::Tensor down_alpha, torch::Tensor global_scale, torch::Tensor out,
                 int64_t num_pairs, int64_t rows_capacity) {
  using namespace b12x_fc2;
  TORCH_CHECK(ids.scalar_type() == torch::kInt32 && token_weights.scalar_type() == torch::kFloat32);
  TORCH_CHECK(down_alpha.scalar_type() == torch::kFloat32 && global_scale.scalar_type() == torch::kFloat32);
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.size(1) % kTileN == 0);
  TORCH_CHECK(0 <= num_pairs && num_pairs <= rows_capacity && rows_capacity <= 64);
  TORCH_CHECK(w2.numel() * w2.element_size() == down_alpha.numel() * out.size(1) * 576 / 2,
              "w2 is not the n64-repacked [experts, hidden x 576] FP4 storage");
  Args a;
  a.intermediate = reinterpret_cast<const uint32_t*>(intermediate.data_ptr());
  a.w2 = reinterpret_cast<const uint32_t*>(w2.data_ptr());
  a.w2_sf = reinterpret_cast<const uint8_t*>(w2_sf.data_ptr());
  a.ids = ids.data_ptr<int32_t>();
  a.token_weights = token_weights.data_ptr<float>();
  a.down_alpha = down_alpha.data_ptr<float>();
  a.global_scale = global_scale.data_ptr<float>();
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.gs_per_expert = global_scale.numel() > 1;
  a.num_experts = (int)down_alpha.numel();
  a.num_pairs = (int)num_pairs;
  a.rows_capacity = (int)rows_capacity;
  a.hidden = (int)out.size(1);
  a.out_tiles = a.hidden / kTileN;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(fc2_grouped_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) ==
                cudaSuccess);
    attr = true;
  }
  if (num_pairs == 0) return;
  fc2_grouped_kernel<<<(int)num_pairs * a.out_tiles, kThreads, kSmem, c10::cuda::getCurrentCUDAStream()>>>(a);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fc2_grouped", &fc2_grouped); }
