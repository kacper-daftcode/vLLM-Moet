// Inkling2's small-batch bf16 linear on sm_120 (RTX PRO 6000 / RTX 5090): out = bf16(x @ weight^T) for up to 32 rows
// of x, fp32 accumulation, the weight read once at the memory's rate.
//
// vLLM sends every projection of a decode step (qkvr, wo, the DSpark draft's layers) with at most 32 rows to
// FlashInfer's CuTe DSL low-M kernels, which exist for capabilities 100 / 103 / 107. This is that operation on
// mma.sync m16n8k16 (bf16 x bf16 -> fp32):
//
//   x      [M <= 32][K] bf16 row-major (K a multiple of 64)
//   weight [N][K] bf16 row-major as the model holds it (also a row slice of a larger matrix)
//   out    [M][N] bf16
//
// One CTA computes 32 output columns over all of K. K runs in steps of 64 elements through a ring of cp.async
// stages: the step's 128 bytes of each of the 32 weight rows and of each x row, 16-byte chunks XOR-swizzled by the
// row so that ldmatrix reads eight rows without bank conflicts; the weight rows are prefetched into L2 `pf_depth`
// steps ahead. The four warps split the columns (M <= 16: one n8 tile each) or the rows and the columns (M <= 32:
// two m16 tiles x two column groups of two n8 tiles). The grid is the number of column blocks - it depends on the
// shapes only, so a launch replays from a CUDA graph.
//
// Numerics: per output element the K / 16 mma.sync products accumulate in the fp32 C operand in K order; cvt.rn to
// bf16.
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <vector>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace inkling2_linear {

constexpr int kThreads = 128;
constexpr int kCols = 32;        // output columns of a CTA
constexpr int kStepBytes = 128;  // a row's bytes of a K step (64 bf16)
constexpr int kStages = 6;

struct Args {
  const uint8_t* x;      // [m][k] bf16
  const uint8_t* w;      // [n][k] bf16
  uint16_t* out;         // [m][n] bf16
  int m, n, steps;       // steps = k / 64
  int pf_depth;
};

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// Two fp32 -> one register of two bf16 (round to nearest even), `lo` in the low half.
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

// MT: m16 tiles of the batch (1: up to 16 rows, 2: up to 32)
template <int MT>
struct Shape {
  static constexpr int rows = 16 * MT;
  static constexpr int nt = MT;                        // n8 tiles of a warp
  static constexpr int a_bytes = rows * kStepBytes;
  static constexpr int b_bytes = kCols * kStepBytes;
  static constexpr int stage = a_bytes + b_bytes;
  static constexpr int smem = kStages * stage;
};

template <int MT>
__global__ void __launch_bounds__(kThreads) linear_kernel(const Args a) {
  using S = Shape<MT>;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int q8 = lane >> 2, c = lane & 3;       // the thread's place in an MMA fragment
  const int mi = lane & 7, mm = lane >> 3;      // its row and matrix in an ldmatrix.x4
  const int mt = warp % MT, ng = warp / MT;     // the warp's m16 tile and column group
  const int col_w = ng * 8 * S::nt;             // the warp's first column within the block
  const int n0 = blockIdx.x * kCols;
  const size_t row_bytes = (size_t)a.steps * kStepBytes;

  auto issue = [&](int j) {
    const uint32_t st = sbase + (j % kStages) * S::stage;
#pragma unroll
    for (int g = tid; g < (S::rows + kCols) * 8; g += kThreads) {
      const int row = g >> 3, part = g & 7;
      if (row < S::rows) {
        if (row < a.m) cp_async16(st + row * kStepBytes + ((part ^ (row & 7)) << 4), a.x + row * row_bytes + (size_t)j * kStepBytes + part * 16);
      } else {
        const int r = row - S::rows;
        if (n0 + r < a.n) cp_async16(st + S::a_bytes + r * kStepBytes + ((part ^ (r & 7)) << 4), a.w + (n0 + r) * row_bytes + (size_t)j * kStepBytes + part * 16);
      }
    }
    const int pj = j + a.pf_depth;
    if (a.pf_depth > 0 && pj < a.steps && tid < kCols && n0 + tid < a.n) prefetch_l2(a.w + (n0 + tid) * row_bytes + (size_t)pj * kStepBytes);
  };

#pragma unroll 1
  for (int j = 0; j < kStages - 1; ++j) {
    if (j < a.steps) issue(j);
    cp_async_commit();
  }

  float acc[S::nt][4];
#pragma unroll
  for (int t = 0; t < S::nt; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
  const int row_a = mt * 16 + q8, row_b = row_a + 8;
  const bool va = row_a < a.m, vb = row_b < a.m;

#pragma unroll 1
  for (int j = 0; j < a.steps; ++j) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    if (j + kStages - 1 < a.steps) issue(j + kStages - 1);
    cp_async_commit();
    const uint32_t st = sbase + (j % kStages) * S::stage;
    const uint32_t stb = st + S::a_bytes;
    // The step's four k16 steps: x as A fragments (rows mt * 16 .. + 15), the weight rows as B fragments.
    uint32_t xa[4][4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      ldmatrix_x4(xa[s], st + (mt * 16 + 8 * (mm & 1) + mi) * kStepBytes + (((2 * s + (mm >> 1)) ^ mi) << 4));
      if (!va) xa[s][0] = xa[s][2] = 0u;
      if (!vb) xa[s][1] = xa[s][3] = 0u;
    }
    if (MT == 1) {
#pragma unroll
      for (int sp = 0; sp < 2; ++sp) {
        uint32_t b[4];
        ldmatrix_x4(b, stb + (col_w + mi) * kStepBytes + (((4 * sp + mm) ^ mi) << 4));
        mma_bf16(acc[0], xa[2 * sp], b[0], b[1]);
        mma_bf16(acc[0], xa[2 * sp + 1], b[2], b[3]);
      }
    } else {
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        uint32_t b[4];
        ldmatrix_x4(b, stb + (col_w + 8 * (mm >> 1) + mi) * kStepBytes + (((2 * s + (mm & 1)) ^ mi) << 4));
        mma_bf16(acc[0], xa[s], b[0], b[1]);
        mma_bf16(acc[S::nt - 1], xa[s], b[2], b[3]);
      }
    }
  }
  cp_async_wait<0>();

  // c0, c1 = (row a, columns 2c, 2c + 1), c2, c3 = (row b, the same columns) of each n8 tile
#pragma unroll
  for (int t = 0; t < S::nt; ++t) {
    const int col = n0 + col_w + 8 * t + 2 * c;
    if (col >= a.n) continue;
    if (va) *reinterpret_cast<uint32_t*>(a.out + (size_t)row_a * a.n + col) = pack_bf16(acc[t][0], acc[t][1]);
    if (vb) *reinterpret_cast<uint32_t*>(a.out + (size_t)row_b * a.n + col) = pack_bf16(acc[t][2], acc[t][3]);
  }
}

}  // namespace inkling2_linear

namespace {

using inkling2_linear::Args;

struct Launch {
  void (*fn)(Args);
  int smem;
  bool ready;
};
Launch g_launch[2] = {{inkling2_linear::linear_kernel<1>, inkling2_linear::Shape<1>::smem, false},
                      {inkling2_linear::linear_kernel<2>, inkling2_linear::Shape<2>::smem, false}};

// out = bf16(x @ weight^T): x [m <= 32][k], weight [n][k], out [m][n], all bf16 and contiguous; k % 64 == 0, n even.
void linear_run(torch::Tensor x, torch::Tensor weight, torch::Tensor out, int64_t pf_depth) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && out.is_cuda());
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && weight.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && out.dim() == 2 && x.is_contiguous() && weight.is_contiguous() && out.is_contiguous());
  const int64_t m = x.size(0), k = x.size(1), n = weight.size(0);
  TORCH_CHECK(m >= 1 && m <= 32 && k % 64 == 0 && k >= 64 && weight.size(1) == k && n >= 2 && n % 2 == 0, "m ", m, " n ", n, " k ", k);
  TORCH_CHECK(out.size(0) == m && out.size(1) == n);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0, "16-byte aligned operands");
  Args a;
  a.x = reinterpret_cast<const uint8_t*>(x.data_ptr());
  a.w = reinterpret_cast<const uint8_t*>(weight.data_ptr());
  a.out = reinterpret_cast<uint16_t*>(out.data_ptr());
  a.m = (int)m;
  a.n = (int)n;
  a.steps = (int)(k / 64);
  a.pf_depth = (int)pf_depth;
  Launch& L = g_launch[m > 16];
  if (!L.ready) {
    TORCH_CHECK(cudaFuncSetAttribute(L.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, L.smem) == cudaSuccess);
    L.ready = true;
  }
  const int grid = (int)((n + inkling2_linear::kCols - 1) / inkling2_linear::kCols);
  L.fn<<<grid, inkling2_linear::kThreads, L.smem, c10::cuda::getCurrentCUDAStream()>>>(a);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "linear launch: ", cudaGetErrorString(err));
}

// (shared memory bytes, registers per thread, CTAs per SM) of the kernel for m <= 16 (shape 0) or m <= 32 (shape 1).
std::vector<int64_t> kernel_facts(int64_t shape) {
  Launch& L = g_launch[shape];
  cudaFuncAttributes attr;
  TORCH_CHECK(cudaFuncGetAttributes(&attr, L.fn) == cudaSuccess);
  cudaFuncSetAttribute(L.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, L.smem);
  int resident = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, L.fn, inkling2_linear::kThreads, L.smem);
  return {L.smem, attr.numRegs, resident};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_run", &linear_run);
  m.def("kernel_facts", &kernel_facts);
}
