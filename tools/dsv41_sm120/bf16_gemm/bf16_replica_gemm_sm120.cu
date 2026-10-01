// Small-M BF16 GEMM for sm_120 that reproduces cuBLAS's tensor-core results bit for bit (DeepSeek-V4.1 decode).
//
//   out[M, N] = x[M, K] (bf16, unit inner stride) @ w[N, K]^T (bf16, contiguous), fp32 accumulate, bf16 or fp32 out
//
// cuBLAS's small-M kernels at decode token counts (cutlass_80 wmma 16x16 / 32x32, tensorop s16816) compute every
// output element as a chain of mma.m16n8k16 over K inside each split-K slice (gridDim.z slices of one width, the
// last one shorter), and splitKreduce adds the slice partials in ascending slice order in fp32 (for a bf16 output
// the partials are rounded to bf16 first). The mma does not depend on the order of k inside its 16-wide step
// (checked on wide-range data), so each lane loads four consecutive k (8 bytes) per operand row. Given the slice
// width cuBLAS picked for a shape and token count - calibrated against cuBLAS on this GPU at load time, see
// bf16_replica_gemm_sm120.py - three kernels produce the same bits in one launch:
//   chain   one slice (S == 1): one warp per 8 output columns streams the 8 weight rows through shared memory in
//           commit groups and runs the single chain while the later groups are still in flight.
//   split   S >= 2: one CTA per 8 output columns, one warp per slice holding the slice's operand fragments in
//           registers; the partials meet in shared memory and the CTA sums them in ascending slice order.
//   spread  S >= 2 over all SMs: CTAs of up to 4 slice warps, partials to an fp32 workspace in L2; the CTA that
//           arrives last at the tile's counter sums them in ascending order, writes the columns and resets it.
// M <= 16 (one m16 row tile). Built without fast-math: the reduction adds must not flush denormals.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ void mma_16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                          uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// weights are streamed once: no L1 allocation
__device__ __forceinline__ uint2 ldg64_stream(const __nv_bfloat16* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
// activations are read by every CTA (L2-resident)
__device__ __forceinline__ uint2 ldg64(const __nv_bfloat16* p) {
  uint2 v;
  asm volatile("ld.global.v2.u32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ float ldcg(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];\n" : "=f"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// wait until at most n groups are pending (n is a compile-time constant after unrolling)
__device__ __forceinline__ void cp_async_wait_n(int n) {
  switch (n) {
    case 0: cp_async_wait<0>(); break;
    case 1: cp_async_wait<1>(); break;
    case 2: cp_async_wait<2>(); break;
    case 3: cp_async_wait<3>(); break;
    case 4: cp_async_wait<4>(); break;
    case 5: cp_async_wait<5>(); break;
    case 6: cp_async_wait<6>(); break;
    case 7: cp_async_wait<7>(); break;
    case 8: cp_async_wait<8>(); break;
    case 9: cp_async_wait<9>(); break;
    case 10: cp_async_wait<10>(); break;
    case 11: cp_async_wait<11>(); break;
    case 12: cp_async_wait<12>(); break;
    case 13: cp_async_wait<13>(); break;
    case 14: cp_async_wait<14>(); break;
    default: cp_async_wait<15>(); break;
  }
}

template <bool OUT_F32>
__device__ __forceinline__ void store_out(void* out, size_t idx, float v) {
  if constexpr (OUT_F32)
    reinterpret_cast<float*>(out)[idx] = v;
  else
    reinterpret_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(v);
}

template <bool PBF16>
__device__ __forceinline__ float partial_value(float v) {
  if constexpr (PBF16) return __bfloat162float(__float2bfloat16_rn(v));
  return v;
}

// One slice of rows g (and g + 8) x columns n0..n0+7: lane (g, t) holds k {4t..4t+3} of every 16-wide step.
template <int STEPS, bool HI>
__device__ __forceinline__ void slice_chain(float (&c)[4], const __nv_bfloat16* __restrict__ x, int ldx,
                                            const __nv_bfloat16* __restrict__ w, int M, int K, int n0, int kb,
                                            int nsteps, int g, int t) {
  const bool v0 = g < M, v1 = g + 8 < M;
  const __nv_bfloat16* wp = w + (size_t)(n0 + g) * K + kb + t * 4;
  const __nv_bfloat16* x0 = x + (size_t)g * ldx + kb + t * 4;
  const __nv_bfloat16* x1 = x + (size_t)(g + 8) * ldx + kb + t * 4;
  uint2 b[STEPS], a0[STEPS], a1[HI ? STEPS : 1];
#pragma unroll
  for (int j = 0; j < STEPS; ++j) {
    const int jj = j < nsteps ? j : nsteps - 1;  // a shorter last slice re-reads its last step instead of branching
    b[j] = ldg64_stream(wp + jj * 16);
    a0[j] = v0 ? ldg64(x0 + jj * 16) : make_uint2(0u, 0u);
    if constexpr (HI) a1[j] = v1 ? ldg64(x1 + jj * 16) : make_uint2(0u, 0u);
  }
  c[0] = c[1] = c[2] = c[3] = 0.f;
#pragma unroll
  for (int j = 0; j < STEPS; ++j) {
    if (j < nsteps) {
      const uint32_t hx = HI ? a1[j].x : 0u, hy = HI ? a1[j].y : 0u;
      mma_16816(c, a0[j].x, hx, a0[j].y, hy, b[j].x, b[j].y);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// split_kernel: grid N/8, block 32*S. Warp s computes slice s (k in [s*step, min(K, s*step+step))).
// ---------------------------------------------------------------------------------------------------------------
template <int STEPS, bool HI, bool OUT_F32, bool PBF16, int MAXT>
__global__ void __launch_bounds__(MAXT, 1)
split_kernel(const __nv_bfloat16* __restrict__ x, int ldx, const __nv_bfloat16* __restrict__ w, void* __restrict__ out,
             int M, int N, int K, int step) {
  __shared__ float part[32 * 128];  // [S][16][8]
  const int S = blockDim.x >> 5;
  const int lane = threadIdx.x & 31, s = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * 8;
  const int kb = s * step;
  float c[4];
  slice_chain<STEPS, HI>(c, x, ldx, w, M, K, n0, kb, min(K - kb, step) >> 4, g, t);
  float* p = part + s * 128;
  *reinterpret_cast<float2*>(p + g * 8 + t * 2) = make_float2(c[0], c[1]);
  *reinterpret_cast<float2*>(p + (g + 8) * 8 + t * 2) = make_float2(c[2], c[3]);
  __syncthreads();
  for (int e = threadIdx.x; e < M * 8; e += blockDim.x) {
    float acc = 0.f;
    for (int q = 0; q < S; ++q) {
      const float v = partial_value<PBF16>(part[q * 128 + e]);
      acc = (q == 0) ? v : acc + v;
    }
    store_out<OUT_F32>(out, (size_t)(e >> 3) * N + n0 + (e & 7), acc);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// spread_kernel: grid (ceil(S / W), N/8), block 32*W. Warp w of CTA (gy, tile) computes slice gy*W + w; partials go
// to ws[tile][s][16][8]; the CTA that increments cnt[tile] last sums them and resets the counter.
// ---------------------------------------------------------------------------------------------------------------
template <int STEPS, bool HI, bool OUT_F32, bool PBF16>
__global__ void __launch_bounds__(128, 1)
spread_kernel(const __nv_bfloat16* __restrict__ x, int ldx, const __nv_bfloat16* __restrict__ w, void* __restrict__ out,
              float* __restrict__ ws, int* __restrict__ cnt, int M, int N, int K, int step, int S) {
  __shared__ int last;
  const int W = blockDim.x >> 5;
  const int lane = threadIdx.x & 31, wl = threadIdx.x >> 5;
  const int s = blockIdx.x * W + wl;
  const int tile = blockIdx.y;
  const int g = lane >> 2, t = lane & 3;
  const int n0 = tile * 8;
  if (s < S) {
    const int kb = s * step;
    float c[4];
    slice_chain<STEPS, HI>(c, x, ldx, w, M, K, n0, kb, min(K - kb, step) >> 4, g, t);
    float* p = ws + ((size_t)tile * S + s) * 128;
    *reinterpret_cast<float2*>(p + g * 8 + t * 2) = make_float2(c[0], c[1]);
    *reinterpret_cast<float2*>(p + (g + 8) * 8 + t * 2) = make_float2(c[2], c[3]);
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(cnt + tile, 1) == (int)gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const float* p = ws + (size_t)tile * S * 128;
  for (int e = threadIdx.x; e < M * 8; e += blockDim.x) {
    float v[32];
#pragma unroll
    for (int q = 0; q < 32; ++q)
      if (q < S) v[q] = ldcg(p + q * 128 + e);
    float acc = 0.f;
#pragma unroll
    for (int q = 0; q < 32; ++q) {
      if (q < S) {
        const float u = partial_value<PBF16>(v[q]);
        acc = (q == 0) ? u : acc + u;
      }
    }
    store_out<OUT_F32>(out, (size_t)(e >> 3) * N + n0 + (e & 7), acc);
  }
  if (threadIdx.x == 0) cnt[tile] = 0;
}

// ---------------------------------------------------------------------------------------------------------------
// chain_kernel: grid N/8, block 32. G commit groups of KG k each (G * KG == K) bring the 8 weight rows into shared
// memory; the activation fragments of the next group are loaded into registers while the current group computes.
// ---------------------------------------------------------------------------------------------------------------
template <int G, int KG, bool HI, bool OUT_F32>
__global__ void __launch_bounds__(32, 1)
chain_kernel(const __nv_bfloat16* __restrict__ x, int ldx, const __nv_bfloat16* __restrict__ w, void* __restrict__ out,
             int M, int N) {
  constexpr int K = G * KG;
  constexpr int PITCH = K + 16;  // row stride = 8 banks (mod 32): the 8-byte fragment reads are conflict-free
  constexpr int SEGS = KG / 8;   // 16-byte segments per row and group
  constexpr int GS = KG / 16;    // k-steps per group
  extern __shared__ __align__(16) __nv_bfloat16 sw[];  // [8][PITCH]
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * 8;
#pragma unroll
  for (int q = 0; q < G; ++q) {
    for (int i = lane; i < 8 * SEGS; i += 32) {
      const int r = i / SEGS, sg = i - r * SEGS;
      cp_async16(sw + r * PITCH + q * KG + sg * 8, w + (size_t)(n0 + r) * K + q * KG + sg * 8);
    }
    cp_async_commit();
  }
  const bool v0 = g < M, v1 = g + 8 < M;
  const __nv_bfloat16* x0 = x + (size_t)g * ldx + t * 4;
  const __nv_bfloat16* x1 = x + (size_t)(g + 8) * ldx + t * 4;
  uint2 a0[2][GS], a1[2][HI ? GS : 1];
#pragma unroll
  for (int j = 0; j < GS; ++j) {
    a0[0][j] = v0 ? ldg64(x0 + j * 16) : make_uint2(0u, 0u);
    if constexpr (HI) a1[0][j] = v1 ? ldg64(x1 + j * 16) : make_uint2(0u, 0u);
  }
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  const __nv_bfloat16* bw = sw + g * PITCH + t * 4;
#pragma unroll
  for (int q = 0; q < G; ++q) {
    if (q + 1 < G) {
#pragma unroll
      for (int j = 0; j < GS; ++j) {
        const int k = (q + 1) * KG + j * 16;
        a0[(q + 1) & 1][j] = v0 ? ldg64(x0 + k) : make_uint2(0u, 0u);
        if constexpr (HI) a1[(q + 1) & 1][j] = v1 ? ldg64(x1 + k) : make_uint2(0u, 0u);
      }
    }
    cp_async_wait_n(G - 1 - q);  // group q has landed once at most G-1-q younger groups are pending
    __syncwarp();
#pragma unroll
    for (int j = 0; j < GS; ++j) {
      const uint2 bb = *reinterpret_cast<const uint2*>(bw + q * KG + j * 16);
      const uint2 aa = a0[q & 1][j];
      const uint32_t hx = HI ? a1[q & 1][j].x : 0u, hy = HI ? a1[q & 1][j].y : 0u;
      mma_16816(c, aa.x, hx, aa.y, hy, bb.x, bb.y);
    }
  }
  const int col = n0 + t * 2;
  if (v0) { store_out<OUT_F32>(out, (size_t)g * N + col, c[0]); store_out<OUT_F32>(out, (size_t)g * N + col + 1, c[1]); }
  if (v1) { store_out<OUT_F32>(out, (size_t)(g + 8) * N + col, c[2]); store_out<OUT_F32>(out, (size_t)(g + 8) * N + col + 1, c[3]); }
}

// ---------------------------------------------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------------------------------------------
int num_slices(int64_t K, int64_t step) { return (int)((K + step - 1) / step); }

bool slices_ok(int64_t M, int64_t N, int64_t K, int64_t step) {
  if (M < 1 || M > 16 || N < 8 || N % 8 != 0 || K % 16 != 0 || step < 128 || step > 640 || step % 64 != 0) return false;
  const int64_t S = num_slices(K, step);
  return S >= 2 && S <= 32 && (K - (S - 1) * step) % 16 == 0;
}

// Register budgets: a slice warp keeps 4 (rows <= 8) or 6 32-bit registers per 16-wide step. The limits are the
// instantiations ptxas compiles without spills (-Xptxas -v, CUDA 13.0).
bool split_supported(int64_t M, int64_t N, int64_t K, int64_t step) {
  if (!slices_ok(M, N, K, step)) return false;
  const int64_t S = num_slices(K, step), steps = step / 16;
  if (S <= 16) return M <= 8 ? (steps >= 12 && steps <= 16) : steps == 12;  // 512-thread CTAs
  return steps <= 16;                                                       // 1024-thread CTAs
}

bool spread_supported(int64_t M, int64_t N, int64_t K, int64_t step) {
  if (!slices_ok(M, N, K, step)) return false;
  const int64_t steps = step / 16;
  return M <= 8 ? steps <= 40 : (steps >= 16 && steps <= 24);
}

bool chain_supported(int64_t M, int64_t N, int64_t K) {
  return M >= 1 && M <= 16 && N >= 8 && N % 8 == 0 && (K == 5120 || K == 512 || K == 256);
}

void check_operands(const at::Tensor& x, const at::Tensor& w, const at::Tensor& out) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "bf16 operands");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.stride(1) == 1 && w.is_contiguous() && w.size(1) == x.size(1),
              "x [M, K] unit inner stride, w [N, K] contiguous");
  TORCH_CHECK(x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "16-byte aligned rows");
  TORCH_CHECK(out.is_contiguous() && out.dim() == 2 && out.size(0) == x.size(0) && out.size(1) == w.size(0) &&
                  (out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16),
              "out [M, N] contiguous fp32 or bf16");
}

template <int STEPS, bool HI, bool OF, bool PB>
void launch_split(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, int step, int S, cudaStream_t st) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto kern = S <= 16 ? split_kernel<STEPS, HI, OF, PB, 512> : split_kernel<STEPS, HI, OF, PB, 1024>;
  kern<<<N / 8, S * 32, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x.stride(0),
                                 reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), out.data_ptr(), M, N, K, step);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int STEPS, bool HI, bool OF, bool PB>
void launch_spread(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, at::Tensor& ws, at::Tensor& cnt, int step,
                   int S, int W, cudaStream_t st) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  dim3 grid((S + W - 1) / W, N / 8);
  spread_kernel<STEPS, HI, OF, PB><<<grid, W * 32, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(),
      M, N, K, step, S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int G, int KG, bool HI, bool OF>
void launch_chain(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, cudaStream_t st) {
  const int M = x.size(0), N = w.size(0);
  const size_t smem = (size_t)8 * (G * KG + 16) * 2;
  auto kern = chain_kernel<G, KG, HI, OF>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  kern<<<N / 8, 32, smem, st>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x.stride(0),
                                reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), out.data_ptr(), M, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool HI, bool OF, bool PB>
void dispatch_split(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, int step, int S, cudaStream_t st) {
  switch (step / 16) {
    case 8: launch_split<8, HI, OF, PB>(x, w, out, step, S, st); break;
    case 12: launch_split<12, HI, OF, PB>(x, w, out, step, S, st); break;
    case 16: launch_split<16, HI, OF, PB>(x, w, out, step, S, st); break;
    default: TORCH_CHECK(false, "split: unsupported slice width ", step);
  }
}

template <bool HI, bool OF, bool PB>
void dispatch_spread(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, at::Tensor& ws, at::Tensor& cnt, int step,
                     int S, int W, cudaStream_t st) {
  if constexpr (!HI) {
    if (step / 16 == 8) return launch_spread<8, false, OF, PB>(x, w, out, ws, cnt, step, S, W, st);
    if (step / 16 == 12) return launch_spread<12, false, OF, PB>(x, w, out, ws, cnt, step, S, W, st);
  }
  switch (step / 16) {
    case 16: launch_spread<16, HI, OF, PB>(x, w, out, ws, cnt, step, S, W, st); return;
    case 20: launch_spread<20, HI, OF, PB>(x, w, out, ws, cnt, step, S, W, st); return;
    case 24: launch_spread<24, HI, OF, PB>(x, w, out, ws, cnt, step, S, W, st); return;
    default: break;
  }
  if constexpr (!HI) {  // wider slices only for rows <= 8 (registers)
    switch (step / 16) {
      case 28: launch_spread<28, false, OF, PB>(x, w, out, ws, cnt, step, S, W, st); return;
      case 32: launch_spread<32, false, OF, PB>(x, w, out, ws, cnt, step, S, W, st); return;
      case 36: launch_spread<36, false, OF, PB>(x, w, out, ws, cnt, step, S, W, st); return;
      case 40: launch_spread<40, false, OF, PB>(x, w, out, ws, cnt, step, S, W, st); return;
      default: break;
    }
  }
  TORCH_CHECK(false, "spread: unsupported slice width ", step, " for ", x.size(0), " rows");
}

template <bool HI, bool OF>
void dispatch_chain(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, cudaStream_t st) {
  switch (x.size(1)) {
    case 5120: launch_chain<16, 320, HI, OF>(x, w, out, st); break;
    case 512: launch_chain<2, 256, HI, OF>(x, w, out, st); break;
    case 256: launch_chain<1, 256, HI, OF>(x, w, out, st); break;
    default: TORCH_CHECK(false, "chain: unsupported K ", x.size(1));
  }
}

}  // namespace

// One slice (cuBLAS without split-K).
void chain_out(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
  check_operands(x, w, out);
  TORCH_CHECK(chain_supported(x.size(0), w.size(0), x.size(1)), "chain: unsupported M=", x.size(0), " N=", w.size(0), " K=", x.size(1));
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t st = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const bool hi = x.size(0) > 8, of = out.scalar_type() == at::kFloat;
  if (hi) { if (of) dispatch_chain<true, true>(x, w, out, st); else dispatch_chain<true, false>(x, w, out, st); }
  else { if (of) dispatch_chain<false, true>(x, w, out, st); else dispatch_chain<false, false>(x, w, out, st); }
}

// S = ceil(K / step) slices, one CTA per 8 columns; pbf16: round the partials to bf16 before the reduction.
void split_out(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t step, bool pbf16) {
  check_operands(x, w, out);
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(split_supported(M, N, K, step), "split: unsupported M=", M, " N=", N, " K=", K, " step=", step);
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t st = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int S = num_slices(K, step);
  const bool of = out.scalar_type() == at::kFloat;
  if (M > 8) {
    if (of) dispatch_split<true, true, false>(x, w, out, step, S, st);
    else if (pbf16) dispatch_split<true, false, true>(x, w, out, step, S, st);
    else dispatch_split<true, false, false>(x, w, out, step, S, st);
  } else {
    if (of) dispatch_split<false, true, false>(x, w, out, step, S, st);
    else if (pbf16) dispatch_split<false, false, true>(x, w, out, step, S, st);
    else dispatch_split<false, false, false>(x, w, out, step, S, st);
  }
}

// The same over all SMs: ws fp32 >= N/8 * S * 128 (any contents), cnt int32 >= N/8 (zero before the first call; the
// kernel leaves it zero again); W slice warps per CTA (1..4).
void spread_out(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t step, bool pbf16, torch::Tensor ws,
                torch::Tensor cnt, int64_t W) {
  check_operands(x, w, out);
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(spread_supported(M, N, K, step), "spread: unsupported M=", M, " N=", N, " K=", K, " step=", step);
  const int S = num_slices(K, step);
  TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == at::kFloat && ws.is_contiguous() && ws.numel() >= (int64_t)(N / 8) * S * 128,
              "ws: fp32 [N/8 * S * 128]");
  TORCH_CHECK(cnt.is_cuda() && cnt.scalar_type() == at::kInt && cnt.is_contiguous() && cnt.numel() >= N / 8, "cnt: int32 [N/8]");
  TORCH_CHECK(W >= 1 && W <= 4, "W in 1..4");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t st = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const bool of = out.scalar_type() == at::kFloat;
  if (M > 8) {
    if (of) dispatch_spread<true, true, false>(x, w, out, ws, cnt, step, S, W, st);
    else if (pbf16) dispatch_spread<true, false, true>(x, w, out, ws, cnt, step, S, W, st);
    else dispatch_spread<true, false, false>(x, w, out, ws, cnt, step, S, W, st);
  } else {
    if (of) dispatch_spread<false, true, false>(x, w, out, ws, cnt, step, S, W, st);
    else if (pbf16) dispatch_spread<false, false, true>(x, w, out, ws, cnt, step, S, W, st);
    else dispatch_spread<false, false, false>(x, w, out, ws, cnt, step, S, W, st);
  }
}

// The CUDA-graph capture sequence the current stream is part of (0 when it is not capturing).
int64_t capture_id() {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  unsigned long long id = 0;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(c10::cuda::getCurrentCUDAStream().stream(), &status, &id));
  return status == cudaStreamCaptureStatusActive ? (int64_t)id : 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("capture_id", &capture_id);
  m.def("chain_out", &chain_out, "one-slice chain", py::arg("x"), py::arg("w"), py::arg("out"));
  m.def("split_out", &split_out, "split-K slices, one CTA per 8 columns", py::arg("x"), py::arg("w"), py::arg("out"),
        py::arg("step"), py::arg("pbf16"));
  m.def("spread_out", &spread_out, "split-K slices over all SMs, last CTA reduces", py::arg("x"), py::arg("w"),
        py::arg("out"), py::arg("step"), py::arg("pbf16"), py::arg("ws"), py::arg("cnt"), py::arg("W"));
  m.def("chain_supported", &chain_supported);
  m.def("split_supported", &split_supported);
  m.def("spread_supported", &spread_supported);
}
