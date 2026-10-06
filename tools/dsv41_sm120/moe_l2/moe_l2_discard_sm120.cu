// DeepGEMM grouped-GEMM padding rows dropped from L2 without a write-back (sm_80+ `discard.global.L2`).
//
// In the contiguous grouped layout every expert owns a block of rows aligned to BLOCK_M, its real rows a prefix of
// the block, the rest marked m_indices == -1. DeepGEMM's sm_120 FP8 x FP4 kernel writes every row of a BLOCK_M tile
// that holds a real row - padding rows included - and none of a tile without one. At decode token counts that is
// most of the output (6 tokens x top-6: 2268 of FC2's 2304 rows, 23.2 MB), nobody reads those rows, and their dirty
// L2 lines are written back by whatever streams through L2 next (the following FC1, ~11 us per MoE layer on an RTX
// PRO 6000). Discarding them right after the GEMM invalidates the lines instead.
//
// Only lines of padding rows inside used tiles are touched; the host side checks that rows are whole 128-byte lines
// (row_bytes % 128 == 0, base 128-byte aligned), so no discarded line holds a byte of a real row or of another tensor.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace {

__global__ void discard_padding_rows_kernel(char* __restrict__ out, const int* __restrict__ m_indices, int rows,
                                            int row_lines, int64_t row_bytes, int block_m) {
  const int64_t total = static_cast<int64_t>(rows) * row_lines;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
    const int r = static_cast<int>(i / row_lines);
    const int l = static_cast<int>(i - static_cast<int64_t>(r) * row_lines);
    if (__ldg(m_indices + r) < 0 && __ldg(m_indices + (r / block_m) * block_m) >= 0) {
      const char* p = out + static_cast<int64_t>(r) * row_bytes + static_cast<int64_t>(l) * 128;
      asm volatile("discard.global.L2 [%0], 128;" ::"l"(p) : "memory");
    }
  }
}

}  // namespace

void discard_padding_rows(torch::Tensor out, torch::Tensor m_indices, int64_t block_m) {
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.is_contiguous(), "out: contiguous 2-D CUDA tensor");
  TORCH_CHECK(m_indices.is_cuda() && m_indices.scalar_type() == at::kInt && m_indices.is_contiguous() &&
                  m_indices.numel() == out.size(0) && m_indices.device() == out.device(),
              "m_indices: contiguous int32, one per row of out");
  const int64_t row_bytes = out.size(1) * out.element_size();
  TORCH_CHECK(row_bytes % 128 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 128 == 0,
              "rows must be whole 128-byte lines");
  TORCH_CHECK(block_m > 0, "block_m > 0");
  const int rows = static_cast<int>(out.size(0));
  if (rows == 0) return;
  const c10::cuda::OptionalCUDAGuard guard(out.device());
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, out.get_device());
  const int row_lines = static_cast<int>(row_bytes / 128);
  const int64_t total = static_cast<int64_t>(rows) * row_lines;
  const int threads = 256;
  const int64_t max_blocks = 4LL * std::max(sms, 1);
  const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, max_blocks));
  discard_padding_rows_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream(out.get_device()).stream()>>>(
      static_cast<char*>(out.data_ptr()), m_indices.data_ptr<int>(), rows, row_lines, row_bytes,
      static_cast<int>(block_m));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("discard_padding_rows", &discard_padding_rows,
        "discard.global.L2 over the padding rows (m_indices == -1) of used BLOCK_M tiles of a grouped-GEMM output");
}
