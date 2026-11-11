// Test helper of test_first_claim_sm120.py (not part of the served path): one kernel whose CTAs hold an SM for a given time.
// A CTA asks for `smem` bytes of dynamic shared memory - 98 560 B is what a kernel of an NCCL collective asks for: no other CTA fits
// beside it, it needs an EMPTY SM - records the SM it runs on (%smid) and when it started and ended (%globaltimer), and spins on the
// timer in between. The spin is bounded (the host refuses more than 3 s); the kernel has no memory access besides its record.
//   out [grid][3] int64: smid, start ns, end ns
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ unsigned long long hold_timer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned hold_smid() {
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;\n" : "=r"(s));
  return s;
}

__global__ void hold_kernel(long long* out, unsigned long long hold_ns) {
  extern __shared__ unsigned char smem[];
  const unsigned long long t0 = hold_timer();
  if (threadIdx.x == 0) {
    smem[0] = 1;
    out[3 * blockIdx.x + 0] = (long long)hold_smid();
    out[3 * blockIdx.x + 1] = (long long)t0;
  }
  unsigned long long t = t0;
  while (t - t0 < hold_ns) t = hold_timer();
  if (threadIdx.x == 0) out[3 * blockIdx.x + 2] = (long long)t;
}

int64_t g_attr = 0;

// launches on the current stream and returns at once
void hold(torch::Tensor out, int64_t grid, int64_t threads, int64_t smem, int64_t hold_ns) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt64 && out.is_contiguous() && out.numel() >= 3 * grid, "out: int64 [grid, 3] on the device");
  TORCH_CHECK(grid >= 1 && grid <= 4096 && threads >= 32 && threads <= 1024 && threads % 32 == 0, "grid 1..4096, threads a multiple of 32 up to 1024");
  TORCH_CHECK(smem >= 16 && smem <= 100 * 1024, "shared memory 16 B .. 100 KB");
  TORCH_CHECK(hold_ns >= 0 && hold_ns <= 3000000000LL, "the hold is bounded at 3 s");
  if (smem > g_attr) {
    TORCH_CHECK(cudaFuncSetAttribute(hold_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess,
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) refused ", smem, " B");
    g_attr = smem;
  }
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  hold_kernel<<<(unsigned)grid, (unsigned)threads, (size_t)smem, stream>>>(reinterpret_cast<long long*>(out.data_ptr<int64_t>()), (unsigned long long)hold_ns);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "hold_kernel launch");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("hold", &hold); }
