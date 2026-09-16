// SPDX-License-Identifier: Apache-2.0
// vLLM-Moet: DeepSeek-V4.1 sparse-MLA instantiations for SM120 (RTX PRO 6000 / RTX 5090).
//
// FlashInfer 0.6.18 instantiates its SM120 DSV4 sparse-MLA kernels for the
// DeepSeek-V4-Flash geometry only: SWA page 64, compressed ("extra") pages 64
// or 2, SWA rows 128..2048. vLLM serves DeepSeek-V4.1 (deepseek_v4_1) with
//   * SWA cache page **32** tokens (DeepseekV4SWACache block_size=32),
//   * compressed cache **128** states/page on ratio-1 layers and 64 on ratio-2
//     layers (128-token KV block / compress_ratio),
//   * vision-padded prefill SWA rows of 128 + 1024 = **1152** candidates
//     (prefill_index_width = window_size + max_image_tokens),
//   * DSpark non-causal decode rows of **192** (window 128 + k, aligned to 64).
// The stock orchestrator either rejected these shapes ("Unsupported sparse-MLA
// prefill configuration") or, on the single-cache path, silently ran the
// PBS=64 kernel on 32-token pages. This TU adds exactly the missing template
// instantiations; the kernels themselves are unchanged (PAGE_BLOCK_SIZE and
// PAGE_BLOCK_SIZE_EXTRA only enter the page-address arithmetic
// idx / PBS, idx % PBS and the scale-footer offset PBS * 576).
//
// The launchers below mirror the anonymous-namespace launchers of
// sparse_mla_sm120_prefill.cu (they are not linkable from another TU).
//
// Instantiated for every DeepSeek-V4.1 sharding: 8/16/32/64 heads
// (TP8/TP4/TP2/TP1 of the 64 query heads).

#include <cuda_runtime.h>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

#include <flashinfer/attention/sparse_mla_sm120/arch/common.cuh>
#include <flashinfer/attention/sparse_mla_sm120/common/smem_layout.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>
#include <flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh>

namespace flashinfer::sparse_mla_sm120 {

namespace dsv41 {

constexpr int kMaxCachedCudaDevices = 32;
constexpr int kSwaPageBlockSize = 32;      // DeepseekV4SWACache block_size
constexpr int kVisionPaddedSwaTopk = 1152;  // 128 window + 1024 image tokens

template <typename Kernel>
void configure_dynamic_smem_per_device(Kernel kernel, size_t smem_bytes,
                                       bool (&configured)[kMaxCachedCudaDevices]) {
  if (smem_bytes <= 48 * 1024) return;
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  const bool cacheable_device = device >= 0 && device < kMaxCachedCudaDevices;
  if (cacheable_device && configured[device]) return;
  const cudaError_t rc = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(smem_bytes));
  if (rc == cudaSuccess) {
    if (cacheable_device) configured[device] = true;
    return;
  }
  CUDA_CHECK(rc);
}

// Single-cache MG launcher (SWA-only layers: compress_ratio == 0).
template <ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE, int MG_N_HG_T>
void launch_prefill_mg(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                       const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                       int num_tokens, size_t stride_kv_block, const int* topk_length_ptr,
                       cudaStream_t stream) {
  constexpr ModelType MT = ModelType::DSV4;
  constexpr size_t smem_bytes = SmemLayoutMG<MT, CM>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0 || (MG_N_HG_T == 1 && NUM_HEADS < HPB),
                "NUM_HEADS must fill the MG head tile, except a padded NH=8 tile");
  constexpr int REPLICATE_H = (NUM_HEADS + MG_HEADS_PER_CTA_LOCAL - 1) / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, MG_N_HG_T>;
  static bool configured[kMaxCachedCudaDevices] = {};
  configure_dynamic_smem_per_device(kernel, smem_bytes, configured);

  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         stride_kv_block,
                         /*stride_kv_block_extra=*/(size_t)0,
                         /*topk_extra=*/0,
                         attn_sink,
                         topk_length_ptr,
                         /*topk_length_extra=*/(const int*)nullptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,       (void*)&KV_cache,  (void*)&indices, (void*)&output,
                  (void*)&out_lse, (void*)&attn_sink, (void*)&cold};
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

// Dual-cache MG launcher (compressed layers: SWA cache + compressed cache).
template <ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE, int PAGE_BLOCK_SIZE_EXTRA,
          int MG_N_HG_T>
void launch_prefill_mg_dual(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                            const uint8_t* KV_cache_extra, const int32_t* indices_extra,
                            const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                            int num_tokens, int topk_extra, size_t stride_kv_block,
                            size_t stride_kv_block_extra, const int* topk_length_ptr,
                            const int* topk_length_extra_ptr, cudaStream_t stream) {
  constexpr ModelType MT = ModelType::DSV4;
  constexpr size_t smem_bytes = SmemLayoutMG<MT, CM>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0 || (MG_N_HG_T == 1 && NUM_HEADS < HPB),
                "NUM_HEADS must fill the MG head tile, except a padded NH=8 tile");
  constexpr int REPLICATE_H = (NUM_HEADS + MG_HEADS_PER_CTA_LOCAL - 1) / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_dual_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE,
                                                  PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T>;
  static bool configured[kMaxCachedCudaDevices] = {};
  configure_dynamic_smem_per_device(kernel, smem_bytes, configured);

  PrefillColdParams cold{sm_scale,   num_tokens, stride_kv_block, stride_kv_block_extra,
                         topk_extra, attn_sink,  topk_length_ptr, topk_length_extra_ptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,
                  (void*)&KV_cache,
                  (void*)&indices,
                  (void*)&KV_cache_extra,
                  (void*)&indices_extra,
                  (void*)&output,
                  (void*)&out_lse,
                  (void*)&attn_sink,
                  (void*)&cold};
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

// Dual-cache full-tile launcher (fixed-length decode rows: topk 128, no length arrays).
template <int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE, int PAGE_BLOCK_SIZE_EXTRA, int MG_N_HG_T>
void launch_prefill_mg_dual_fulltile(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                                     const uint8_t* KV_cache_extra, const int32_t* indices_extra,
                                     const float* attn_sink, bf16* output, float* out_lse,
                                     float sm_scale, int num_tokens, int topk_extra,
                                     size_t stride_kv_block, size_t stride_kv_block_extra,
                                     cudaStream_t stream) {
  constexpr ModelType MT = ModelType::DSV4;
  constexpr size_t smem_bytes = SmemLayoutMG<MT, ComputeMode::BF16>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0 || (MG_N_HG_T == 1 && NUM_HEADS < HPB),
                "NUM_HEADS must fill the MG head tile, except a padded NH=8 tile");
  constexpr int REPLICATE_H = (NUM_HEADS + MG_HEADS_PER_CTA_LOCAL - 1) / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_dual_fulltile_kernel<MT, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE,
                                                           PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T>;
  static bool configured[kMaxCachedCudaDevices] = {};
  configure_dynamic_smem_per_device(kernel, smem_bytes, configured);

  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         stride_kv_block,
                         stride_kv_block_extra,
                         topk_extra,
                         attn_sink,
                         /*topk_length=*/(const int*)nullptr,
                         /*topk_length_extra=*/(const int*)nullptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,
                  (void*)&KV_cache,
                  (void*)&indices,
                  (void*)&KV_cache_extra,
                  (void*)&indices_extra,
                  (void*)&output,
                  (void*)&out_lse,
                  (void*)&attn_sink,
                  (void*)&cold};
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

// ---------------------------------------------------------------- dispatch
// Head-count table: NH -> MG_N_HG (8/16 share the padded 16-head tile, 32/64
// use the 32-head tile). Same rule as the stock dispatcher.
#define DSV41_FOR_EACH_NH(X) \
  X(8, 1)                    \
  X(16, 1)                   \
  X(32, 2)                   \
  X(64, 2)

struct PrefillArgs {
  const bf16* Q;
  const uint8_t* KV;
  const int32_t* indices;
  const uint8_t* KV_extra;
  const int32_t* idx_extra;
  const float* attn_sink;
  bf16* output;
  float* out_lse;
  float sm_scale;
  int num_tokens;
  int topk_extra;
  size_t stride_kv_block;
  size_t stride_kv_block_extra;
  const int* topk_length;
  const int* extra_topk_length;
  cudaStream_t stream;
};

template <ComputeMode CM, int TOPK, int NH, int NHG>
inline void run_single(const PrefillArgs& a) {
  launch_prefill_mg<CM, NH, TOPK, kSwaPageBlockSize, NHG>(
      a.Q, a.KV, a.indices, a.attn_sink, a.output, a.out_lse, a.sm_scale, a.num_tokens,
      a.stride_kv_block, a.topk_length, a.stream);
}

template <ComputeMode CM, int TOPK, int PBSX, int NH, int NHG>
inline void run_dual(const PrefillArgs& a) {
  launch_prefill_mg_dual<CM, NH, TOPK, kSwaPageBlockSize, PBSX, NHG>(
      a.Q, a.KV, a.indices, a.KV_extra, a.idx_extra, a.attn_sink, a.output, a.out_lse, a.sm_scale,
      a.num_tokens, a.topk_extra, a.stride_kv_block, a.stride_kv_block_extra, a.topk_length,
      a.extra_topk_length, a.stream);
}

template <int PBSX, int NH, int NHG>
inline void run_dual_fulltile(const PrefillArgs& a) {
  launch_prefill_mg_dual_fulltile<NH, 128, kSwaPageBlockSize, PBSX, NHG>(
      a.Q, a.KV, a.indices, a.KV_extra, a.idx_extra, a.attn_sink, a.output, a.out_lse, a.sm_scale,
      a.num_tokens, a.topk_extra, a.stride_kv_block, a.stride_kv_block_extra, a.stream);
}

// Small SWA rows keep BF16 QK (skips the FP8 Q-quantize prologue, as the stock
// 128/192/256 single-cache table does); the 1152-wide vision-padded rows use
// the FP8 compute mode like the stock >= 512 table.
template <int TOPK, int NH, int NHG>
inline void run_single_by_topk(const PrefillArgs& a) {
  if constexpr (TOPK >= 512) {
    run_single<ComputeMode::FP8, TOPK, NH, NHG>(a);
  } else {
    run_single<ComputeMode::BF16, TOPK, NH, NHG>(a);
  }
}
template <int TOPK, int PBSX, int NH, int NHG>
inline void run_dual_by_topk(const PrefillArgs& a) {
  if constexpr (TOPK >= 512) {
    run_dual<ComputeMode::FP8, TOPK, PBSX, NH, NHG>(a);
  } else {
    run_dual<ComputeMode::BF16, TOPK, PBSX, NH, NHG>(a);
  }
}

template <int TOPK>
inline bool dispatch_single_topk(int num_heads, const PrefillArgs& a) {
#define DSV41_SINGLE_CASE(NH, NHG)          \
  if (num_heads == NH) {                    \
    run_single_by_topk<TOPK, NH, NHG>(a);   \
    return true;                            \
  }
  DSV41_FOR_EACH_NH(DSV41_SINGLE_CASE)
#undef DSV41_SINGLE_CASE
  return false;
}

template <int TOPK, int PBSX>
inline bool dispatch_dual_topk(int num_heads, const PrefillArgs& a) {
#define DSV41_DUAL_CASE(NH, NHG)                \
  if (num_heads == NH) {                        \
    run_dual_by_topk<TOPK, PBSX, NH, NHG>(a);   \
    return true;                                \
  }
  DSV41_FOR_EACH_NH(DSV41_DUAL_CASE)
#undef DSV41_DUAL_CASE
  return false;
}

template <int PBSX>
inline bool dispatch_dual_fulltile(int num_heads, const PrefillArgs& a) {
#define DSV41_FULLTILE_CASE(NH, NHG)          \
  if (num_heads == NH) {                      \
    run_dual_fulltile<PBSX, NH, NHG>(a);      \
    return true;                              \
  }
  DSV41_FOR_EACH_NH(DSV41_FULLTILE_CASE)
#undef DSV41_FULLTILE_CASE
  return false;
}

template <int PBSX>
inline bool dispatch_dual(int num_heads, int topk, const PrefillArgs& a) {
  if (topk == 128 && a.topk_length == nullptr && a.extra_topk_length == nullptr &&
      a.topk_extra % BI == 0) {
    return dispatch_dual_fulltile<PBSX>(num_heads, a);
  }
  switch (topk) {
    case 128:
      return dispatch_dual_topk<128, PBSX>(num_heads, a);
    case 192:
      return dispatch_dual_topk<192, PBSX>(num_heads, a);
    case kVisionPaddedSwaTopk:
      return dispatch_dual_topk<kVisionPaddedSwaTopk, PBSX>(num_heads, a);
    default:
      return false;
  }
}

}  // namespace dsv41

// Consulted by sparse_mla_prefill_dispatch (sparse_mla_sm120.cu hook) for
// ModelType::DSV4 whenever the SWA page is not the stock 64. Returns false if
// no DSV4.1 instantiation matches, so the caller can report the envelope.
bool sparse_mla_prefill_dispatch_dsv41(int num_heads, int topk, int page_block_size,
                                       int topk_extra, int extra_page_block_size, const bf16* Q,
                                       const uint8_t* KV_cache, const int32_t* indices,
                                       const uint8_t* extra_KV_cache,
                                       const int32_t* extra_indices, bf16* output,
                                       float* out_lse, float sm_scale, int num_tokens,
                                       size_t stride_kv_block, size_t stride_kv_block_extra,
                                       const float* attn_sink, const int* topk_length,
                                       const int* extra_topk_length, cudaStream_t stream) {
  using namespace dsv41;
  if (page_block_size != kSwaPageBlockSize) return false;
  const PrefillArgs a{Q,          KV_cache,   indices,         extra_KV_cache,
                      extra_indices, attn_sink, output,       out_lse,
                      sm_scale,   num_tokens, topk_extra,      stride_kv_block,
                      stride_kv_block_extra, topk_length, extra_topk_length, stream};
  if (extra_KV_cache == nullptr) {
    switch (topk) {
      case 128:
        return dispatch_single_topk<128>(num_heads, a);
      case 192:
        return dispatch_single_topk<192>(num_heads, a);
      case kVisionPaddedSwaTopk:
        return dispatch_single_topk<kVisionPaddedSwaTopk>(num_heads, a);
      default:
        return false;
    }
  }
  // Compressed cache pages: 128 states (ratio-1 at a 128-token block) or 64 (ratio-2).
  if (extra_page_block_size == 128) return dispatch_dual<128>(num_heads, topk, a);
  if (extra_page_block_size == 64) return dispatch_dual<64>(num_heads, topk, a);
  return false;
}

}  // namespace flashinfer::sparse_mla_sm120
