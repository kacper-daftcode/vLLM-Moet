// DeepSeek-V4.1 sparse MLA decode on sm_120 (FlashInfer 0.7.2's sparse_mla_sm120 mixed-cache kernel) with the
// candidate rows gathered by plain loads instead of one cp.async.bulk per row, and the split merge with its loads
// issued up front. The attention kernel is FlashInfer's own `sparse_mla_decode_dsv4_kernel` template, compiled from
// the installed headers with FlashInfer's flags and instantiated with the gather schedule below in place of
// Dsv41MixedCacheDecodeSchedule: the math, the fp4 -> fp8 conversion and every barrier of the consumers are
// FlashInfer's; only the producer's data movement differs (the same bytes land in the same shared-memory slots, the
// same mbarrier completes). The merge kernel is FlashInfer's `sparse_mla_decode_dsv4_merge_kernel` expression by
// expression, with the partial rows and the attention sink loaded before the LSE reductions instead of inside them.
//
// Measured (6 tokens, 128 SWA fp8 + 512 compressed fp4 candidates, 10 splits, cold L2, CUDA graph): decode 10.5 ->
// 9.1 us, merge 2.6-2.9 -> 2.2 us; output, LSE and the split scratch bit-identical on 40 random shapes.
//
// Why the gather is faster: the mixed schedule's gather warp issues 32 (fp8) or 64 (fp4) bulk copies of 288-528 B
// per tile; the TMA unit works them off one by one and the tile's bytes land ~2.5 us after the indices arrive. 128
// gather threads loading the same rows as 16-B words (16 lanes per fp4 row, 32 per fp8 row) land them in one DRAM
// round trip (~1 us cold), then store them to the slot and arrive on the slot's mbarrier without transaction bytes.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <cstdlib>
#include <string>

#include "kernels/fp8_decode/decode.cuh"
#include "kernels/dsv41_fp8/decode_schedule.cuh"
#include "execution/attention_params.cuh"
#include "execution/launch_validation.cuh"
#include "execution/merge.cuh"
#include "common/lse.cuh"

namespace flashinfer::sparse_mla_sm120 {

// The mixed-cache decode schedule (kernels/dsv41_fp8/decode_schedule.cuh) with the gather group loading the rows.
struct LdgMixedCacheDecodeSchedule : kernels::dsv41_fp8::Dsv41MixedCacheDecodeResources {
  using Plan = kernels::dsv41_fp8::Dsv41MixedCacheDecodeResources;
  static constexpr bool RAW_PIPELINE = true;
  static constexpr int LDG_GATHER_REGS = 96;  // 128 x (240 + 112 + 96) = 57,344 <= 65,536
  static_assert(GROUP_THREADS * (MATH_REGS + CONVERT_REGS + LDG_GATHER_REGS) <= 65536);
  __device__ static void gather_registers_ldg() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(LDG_GATHER_REGS));
  }

  // One tile's rows by plain loads. FAST: power-of-two page size and a 32-bit page stride (shift / mask / one
  // IMAD.WIDE per address); otherwise the generic division path. Row addresses (FooterScaleLayout::data_offset /
  // scale_offset): page = kv + page_idx * stride, data = page + local * A, scales = page + local * B + C with
  // (A, B, C) = interleaved ? (BPT, BPT, DATA) : (DATA, SCALE, pbs * DATA) of the section's record. Rows past the list
  // end (idx < 0) get null pointers: their loads are predicated off and zeros are stored (the original copies a zero
  // row; same bytes).
  template <bool FAST, typename Smem, typename Section>
  __device__ __forceinline__ static void gather_tile(Smem sm, char* scratch, const Section& sec, int buf, int tid,
                                                     int idx_lane) {
    const int lane = tid & 31, warp = tid >> 5;
    const int pbs_shift = __ffs(sec.pbs) - 1;
    const int rec_data = sec.extra ? Raw::DATA_BYTES : Dsv41Fp8Layout::DATA_BYTES;
    const int rec_scale = sec.extra ? Raw::SCALE_BYTES : Dsv41Fp8Layout::SCALE_BYTES;
    const int rec_bpt = rec_data + rec_scale;
    const uint32_t mul_a = sec.interleaved ? rec_bpt : rec_data;
    const uint32_t mul_b = sec.interleaved ? rec_bpt : rec_scale;
    const long long add_c = sec.interleaved ? (long long)rec_data : (long long)sec.pbs * rec_data;
    const uint32_t stride32 = (uint32_t)sec.stride;
    // lane l computes row l's pointers once; the rows a lane loads come by shuffling those pointers
    const uint8_t* data_l = nullptr;
    const uint8_t* scales_l = nullptr;
    if (idx_lane >= 0) {
      const uint8_t* page;
      uint32_t local;
      if constexpr (FAST) {
        const uint32_t page_idx = (uint32_t)idx_lane >> pbs_shift;
        local = (uint32_t)idx_lane & (uint32_t)(sec.pbs - 1);
        page = sec.kv + (uint64_t)page_idx * stride32;
      } else {
        const int page_idx = idx_lane / sec.pbs;
        local = idx_lane - page_idx * sec.pbs;
        page = sec.kv + (size_t)page_idx * sec.stride;
      }
      data_l = page + (uint64_t)local * mul_a;
      scales_l = page + (uint64_t)local * mul_b + add_c;
    }
    auto shfl_ptr = [](const uint8_t* p, int src) -> const uint8_t* {
      const uint64_t v = reinterpret_cast<uint64_t>(p);
      const uint32_t lo = __shfl_sync(0xffffffffu, (uint32_t)v, src);
      const uint32_t hi = __shfl_sync(0xffffffffu, (uint32_t)(v >> 32), src);
      return reinterpret_cast<const uint8_t*>(((uint64_t)hi << 32) | lo);
    };
    if (sec.extra) {
      // V41_FP4 rows: 256 B data as 16 lanes x 16 B (2 rows per warp instruction, 8 rows per round, 4 rounds),
      // 32 B scales as 2 lanes x 16 B (threads 0..63)
      auto* raw = Plan::raw(scratch, buf);
      uint4 d[4];
      const uint8_t* dp[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) dp[k] = shfl_ptr(data_l, 8 * k + 2 * warp + (lane >> 4));
      const uint8_t* scp = nullptr;
      const int srow = tid >> 1;
      if (warp < 2) scp = shfl_ptr(scales_l, srow & 31);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        d[k] = make_uint4(0u, 0u, 0u, 0u);
        if (dp[k] != nullptr) d[k] = *reinterpret_cast<const uint4*>(dp[k] + (lane & 15) * 16);
      }
      uint4 sc = make_uint4(0u, 0u, 0u, 0u);
      if (scp != nullptr) sc = *reinterpret_cast<const uint4*>(scp + (tid & 1) * 16);
#pragma unroll
      for (int k = 0; k < 4; ++k)
        *reinterpret_cast<uint4*>(raw + (8 * k + 2 * warp + (lane >> 4)) * Raw::DATA_BYTES + (lane & 15) * 16) = d[k];
      if (warp < 2) *reinterpret_cast<uint4*>(raw + RAW_SCALE_OFFSET + srow * Raw::SCALE_BYTES + (tid & 1) * 16) = sc;
    } else {
      // DSV4_1 FP8 rows: 512 B data as 32 lanes x 16 B (one row per warp instruction, 4 rows per round, 8 rounds),
      // the 16 B scale word of row r by thread r
      uint4 d[8];
      const uint8_t* dp[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) dp[k] = shfl_ptr(data_l, 4 * k + warp);
      const uint8_t* scp = nullptr;
      if (warp == 0) scp = scales_l;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        d[k] = make_uint4(0u, 0u, 0u, 0u);
        if (dp[k] != nullptr) d[k] = *reinterpret_cast<const uint4*>(dp[k] + lane * 16);
      }
      uint4 sc = make_uint4(0u, 0u, 0u, 0u);
      if (scp != nullptr) sc = __ldg(reinterpret_cast<const uint4*>(scp));
#pragma unroll
      for (int k = 0; k < 8; ++k)
        *reinterpret_cast<uint4*>(sm.kv_fp8(buf) + (4 * k + warp) * KV::KV_SMEM_STRIDE + lane * 16) = d[k];
      if (warp == 0) *reinterpret_cast<uint4*>(sm.kv_sc(buf) + lane * KV::SCALE_BYTES_PER_TOKEN) = sc;
    }
  }

  // Dsv41MixedCacheDecodeSchedule::produce with the gather branch replaced; the convert branch is the original's.
  template <typename Smem, typename Section>
  __device__ __forceinline__ static void produce(Smem sm, char* scratch, Section section, int count) {
    using pipeline::BulkReady;
    using pipeline::TwoSlotCursor;
    auto* ready = Plan::ready(scratch);
    int tid = Plan::role_tid();
    if (Plan::is_gather()) {
      gather_registers_ldg();
      const int lane = tid & 31;
      for (int t = 0; t < count; ++t) {
        int buf = TwoSlotCursor::slot(t);
        auto sec = section(t);
        if (TwoSlotCursor::reuses(t)) RawFree::acquire(buf);
        if (!sec.extra && TwoSlotCursor::reuses(t)) KvFree::acquire(buf);
        // lane l of every warp holds row l's index (one 128 B line); rows are dealt to lanes by shuffles
        const int idx_lane = sec.index(lane);
        const bool fast = ((sec.pbs & (sec.pbs - 1)) == 0) && ((sec.stride >> 32) == 0);  // uniform
        if (fast)
          gather_tile<true>(sm, scratch, sec, buf, tid, idx_lane);
        else
          gather_tile<false>(sm, scratch, sec, buf, tid, idx_lane);
        __threadfence_block();
        GatherSync::wait();
        auto* barrier = sec.extra ? ready + buf : sm.mbar_full(buf);
        if (tid == 0) pipeline::CountedHandoff<1>::publish(barrier);  // one arrival, no transaction bytes
      }
    } else {
      Plan::convert_registers();
      pipeline::SelectiveSlotPhases phases;
      for (int t = 0; t < count; ++t) {
        int buf = TwoSlotCursor::slot(t);
        auto sec = section(t);
        if (sec.extra) {
          BulkReady::wait(ready + buf, phases.current(buf));
          phases.advance(buf);
          if (TwoSlotCursor::reuses(t)) KvFree::acquire(buf);
          kernels::dsv41_fp8::convert_raw<Plan>(Plan::raw(scratch, buf), sm.kv_fp8(buf), sm.kv_sc(buf), tid,
                                                [](int) { return true; });
        } else {
          BulkReady::wait(sm.mbar_full(buf), TwoSlotCursor::phase(t));
        }
        RawFree::release(buf);
        KvReady::publish(buf);
      }
    }
  }
};

// FlashInfer's merge kernel (execution/merge.cuh) with its memory traffic re-scheduled: all splits' partial rows and
// the attention sink are loaded up front (one round trip, before the LSE reductions) instead of one dependent load
// per split inside the accumulation loop and a sink load inside the reduction. The arithmetic is the merge kernel's,
// expression by expression and in the same split order (skipping the same empty splits): identical output bits.
template <int NUM_HEADS, int D_V_VAL, int BLOCK_THREADS, int DIMS_PER_THREAD, int MAX_SPLITS>
static __global__ void __launch_bounds__(BLOCK_THREADS, 8)
    sparse_mla_decode_dsv4_merge_hoisted_kernel(const bf16* __restrict__ mid_out, const float* __restrict__ mid_lse,
                                                bf16* __restrict__ output, float* __restrict__ out_lse,
                                                const float* __restrict__ attn_sink, int num_tokens, int num_splits,
                                                int num_heads, int mid_heads, size_t stride_out_lse, float lse_scale) {
  static_assert(BLOCK_THREADS % 32 == 0, "BLOCK_THREADS must be multiple of 32");
  static_assert(DIMS_PER_THREAD % 8 == 0, "DIMS_PER_THREAD must be multiple of 8 (uint4)");
  static_assert(BLOCK_THREADS * DIMS_PER_THREAD == D_V_VAL, "block must cover the full D_V row");
  constexpr int VECS_PER_THREAD = DIMS_PER_THREAD / 8;
  const int q_heads = (NUM_HEADS == 0) ? num_heads : NUM_HEADS;
  const int m_heads = (NUM_HEADS == 0) ? mid_heads : NUM_HEADS;

  const int t_idx = blockIdx.x;
  const int h = blockIdx.y;
  if (t_idx >= num_tokens || h >= q_heads) return;
  const int tid = threadIdx.x;

  __shared__ float sm_lse[MAX_SPLITS];
  __shared__ float sm_gmax;
  __shared__ float sm_inv_gsum;
  __shared__ float sm_glse;

  const float* lse_ptr = mid_lse + (size_t)t_idx * m_heads * num_splits + (size_t)h * num_splits;
  const bf16* mid_base = mid_out + ((size_t)t_idx * m_heads + h) * (size_t)num_splits * D_V_VAL;
  bf16* out_ptr = output + ((size_t)t_idx * q_heads + h) * D_V_VAL;
  const int dim_base = tid * DIMS_PER_THREAD;

  // the attention sink (a dependent global load the original issues inside the reduction), issued up front
  const float sink_raw = (attn_sink != nullptr) ? __ldg(attn_sink + h) : 0.f;
  // every split's partial row for this thread's dims, issued before anything waits on the LSEs
  uint4 rows[MAX_SPLITS][VECS_PER_THREAD];
#pragma unroll
  for (int sp = 0; sp < MAX_SPLITS; sp++) {
    if (sp < num_splits) {
      const bf16* row_base = mid_base + (size_t)sp * D_V_VAL + dim_base;
#pragma unroll
      for (int v = 0; v < VECS_PER_THREAD; v++) rows[sp][v] = *reinterpret_cast<const uint4*>(row_base + v * 8);
    }
  }

  for (int sp = tid; sp < num_splits; sp += BLOCK_THREADS) {
    sm_lse[sp] = lse_ptr[sp];
  }
  __syncthreads();

  if (tid < 32) {
    float local_max = -1e30f;
    for (int sp = tid; sp < num_splits; sp += 32) {
      local_max = fmaxf(local_max, sm_lse[sp]);
    }
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
      local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, s));
    }
    float gmax = (local_max > -1e29f) ? local_max : 0.f;
    float sink_log2 = 0.f;
    if (attn_sink != nullptr) {
      sink_log2 = sink_raw * LOG2E;
      gmax = local_max > -1e29f ? fmaxf(gmax, sink_log2) : sink_log2;
    }

    float local_sum = 0.f;
    for (int sp = tid; sp < num_splits; sp += 32) {
      float lse_sp = sm_lse[sp];
      if (lse_sp > -1e29f) local_sum += exp2f(lse_sp - gmax);
    }
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
      local_sum += __shfl_xor_sync(0xffffffff, local_sum, s);
    }
    if (tid == 0) {
      float total_sum = local_sum;
      if (attn_sink != nullptr) total_sum += exp2f(sink_log2 - gmax);
      sm_gmax = gmax;
      sm_inv_gsum = (total_sum > 0.f) ? (1.f / total_sum) : 0.f;
      sm_glse = (total_sum > 0.f) ? (log2f(total_sum) + gmax) : -INFINITY;
    }
  }
  __syncthreads();
  const float global_max = sm_gmax;
  const float inv_global_sum = sm_inv_gsum;

  float acc[DIMS_PER_THREAD];
#pragma unroll
  for (int d = 0; d < DIMS_PER_THREAD; d++) acc[d] = 0.f;

#pragma unroll
  for (int sp = 0; sp < MAX_SPLITS; sp++) {
    if (sp >= num_splits) break;
    float lse_sp = sm_lse[sp];
    if (lse_sp <= -1e29f) continue;
    const float weight = exp2f(lse_sp - global_max);
#pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const uint4 packed = rows[sp][v];
      const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
      for (int p = 0; p < 4; p++) {
        const float2 f = __bfloat1622float2(pairs[p]);
        acc[v * 8 + p * 2 + 0] += weight * f.x;
        acc[v * 8 + p * 2 + 1] += weight * f.y;
      }
    }
  }

#pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; v++) {
    uint4 packed;
    __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
    for (int p = 0; p < 4; p++) {
      pairs[p] = __floats2bfloat162_rn(acc[v * 8 + p * 2 + 0] * inv_global_sum,
                                       acc[v * 8 + p * 2 + 1] * inv_global_sum);
    }
    *reinterpret_cast<uint4*>(out_ptr + dim_base + v * 8) = packed;
  }
  if (out_lse != nullptr && tid == 0) {
    out_lse[(size_t)t_idx * stride_out_lse + h] = scale_output_lse(sm_glse, lse_scale);
  }
}

}  // namespace flashinfer::sparse_mla_sm120

// ------------------------------------------------------------------------------------------------- host side
namespace {

using namespace flashinfer::sparse_mla_sm120;

constexpr ModelType kModel = ModelType::DSV4_1;
constexpr int kHeads = 16;  // the TP4 rank of DeepSeek-V4.1-Flash: the NUM_HEADS=16 instantiation FlashInfer serves
constexpr int kMaxSplits = 16;

struct PagedLayout {
  int page_block_size;
  size_t stride_block;
};

// attention_binding.cu's parse_paged_kv_layout for the 2-4D uint8 caches vLLM passes ([blocks, page, 1, bytes]).
PagedLayout parse_layout(const torch::Tensor& kv, int bpt, const char* name) {
  TORCH_CHECK(kv.scalar_type() == torch::kUInt8 && kv.dim() >= 2 && kv.dim() <= 4, name, ": uint8, 2-4D");
  TORCH_CHECK(kv.stride(-1) == 1 && kv.stride(0) % 16 == 0, name, ": contiguous rows, 16 B block stride");
  if (kv.dim() == 2) {
    TORCH_CHECK(kv.size(1) % bpt == 0, name, ": block width / bytes_per_token");
    return {int(kv.size(1) / bpt), size_t(kv.stride(0))};
  }
  int axis = 1;
  if (kv.dim() == 4) {
    TORCH_CHECK(kv.size(1) == 1 || kv.size(2) == 1, name, ": singleton KV-head axis");
    axis = kv.size(1) == 1 ? 2 : 1;
  }
  TORCH_CHECK(kv.size(-1) >= bpt && kv.stride(axis) == bpt, name, ": tightly packed ", bpt, "-byte rows");
  TORCH_CHECK(kv.stride(0) >= kv.size(axis) * kv.stride(axis), name, ": block stride");
  return {int(kv.size(axis)), size_t(kv.stride(0))};
}

void launch(const execution::AttentionParams& p, int head_blocks, int block_threads, int smem_bytes, bool hoisted_merge,
            cudaStream_t stream) {
  auto kernel = sparse_mla_decode_dsv4_kernel<kModel, kHeads, LdgMixedCacheDecodeSchedule>;
  static bool attr_set = false;
  if (!attr_set) {
    TORCH_CHECK(execution::validate_mixed_smem(kernel, smem_bytes) == cudaSuccess, "mixed schedule shared memory");
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) == cudaSuccess);
    attr_set = true;
  }
  kernel<<<dim3(p.num_tokens, head_blocks, p.allocated_splits), dim3(block_threads), smem_bytes, stream>>>(
      p.q, p.kv, p.indices, p.mid_out, p.mid_lse, p.topk_length, p.extra_kv, p.extra_indices, p.extra_topk_length,
      p.extra_topk, p.extra_page_size, p.extra_page_stride_bytes, p.num_tokens, kHeads, p.topk, p.allocated_splits,
      p.chunks_per_block, p.sm_scale, p.page_stride_bytes, p.indices_stride_elems, p.extra_indices_stride_elems,
      Dsv41Pages{p.page_size, p.interleaved_scales});
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "sparse MLA decode launch: ", cudaGetErrorString(e));
  constexpr int MERGE_THREADS = 64;
  if (hoisted_merge) {
    auto merge = sparse_mla_decode_dsv4_merge_hoisted_kernel<kHeads, KVCacheTraits<kModel>::D_V, MERGE_THREADS,
                                                             KVCacheTraits<kModel>::D_V / MERGE_THREADS, kMaxSplits>;
    merge<<<dim3(p.num_tokens, kHeads), dim3(MERGE_THREADS), 0, stream>>>(
        p.mid_out, p.mid_lse, p.output, p.out_lse, p.attn_sink, p.num_tokens, p.allocated_splits, kHeads,
        head_blocks * HPB, p.out_lse_stride_elems, p.lse_scale);
  } else {
    auto merge = sparse_mla_decode_dsv4_merge_kernel<kHeads, KVCacheTraits<kModel>::D_V, MERGE_THREADS,
                                                     KVCacheTraits<kModel>::D_V / MERGE_THREADS>;
    merge<<<dim3(p.num_tokens, kHeads), dim3(MERGE_THREADS), size_t(p.allocated_splits) * sizeof(float), stream>>>(
        p.mid_out, p.mid_lse, p.output, p.out_lse, p.attn_sink, p.num_tokens, p.allocated_splits, kHeads,
        head_blocks * HPB, p.out_lse_stride_elems, p.lse_scale);
  }
  e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "sparse MLA merge launch: ", cudaGetErrorString(e));
}

}  // namespace

// The decode-dsv4 binding's argument derivation (attention_binding.cu) for DSV4_1 / 16 heads / the mixed cache
// (extra_fp4), then the decode kernel with the load-based gather and the merge. `plan_*`: the resolved plan's cpb,
// scratch_split_stride (chunk_capacity = the launched splits), block_threads and shared_bytes (plan.inspect()).
void sparse_mla_decode_ldg(torch::Tensor q, torch::Tensor kv_cache, torch::Tensor indices, torch::Tensor mid_out,
                           torch::Tensor mid_lse, torch::Tensor output, torch::Tensor out_lse, int64_t num_splits,
                           double sm_scale, c10::optional<torch::Tensor> topk_length,
                           c10::optional<torch::Tensor> attn_sink, torch::Tensor extra_kv_cache,
                           torch::Tensor extra_indices, c10::optional<torch::Tensor> extra_topk_length,
                           bool interleaved, double lse_scale, int64_t plan_cpb, int64_t plan_block_threads,
                           int64_t plan_shared_bytes, bool hoisted_merge) {
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.dim() == 3 && q.size(1) == kHeads && q.size(2) == 512 &&
              q.is_contiguous(), "q [tokens, 16, 512] bf16 expected");
  const int num_tokens = (int)q.size(0);
  TORCH_CHECK(indices.scalar_type() == torch::kInt32 && indices.dim() == 2 && indices.size(0) == num_tokens &&
              indices.stride(1) == 1);
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16 && output.sizes() == q.sizes() && output.is_contiguous());
  TORCH_CHECK(out_lse.scalar_type() == torch::kFloat32 && out_lse.dim() == 2 && out_lse.stride(1) == 1);
  TORCH_CHECK(mid_out.scalar_type() == torch::kBFloat16 && mid_out.is_contiguous() &&
              mid_lse.scalar_type() == torch::kFloat32 && mid_lse.is_contiguous());
  TORCH_CHECK(num_splits >= 1 && num_splits <= kMaxSplits, "1-16 splits expected, got ", num_splits);
  TORCH_CHECK(mid_out.numel() >= (int64_t)num_tokens * kHeads * num_splits * 512 &&
              mid_lse.numel() >= (int64_t)num_tokens * kHeads * num_splits);
  constexpr int kMixedThreads = DecodeTileCfg<kModel, true>::block_threads<LdgMixedCacheDecodeSchedule>();
  TORCH_CHECK(plan_block_threads == kMixedThreads, "mixed schedule block threads");
  const int bpt = bytes_per_token(kModel);  // 528
  const PagedLayout kv = parse_layout(kv_cache, bpt, "kv_cache");
  TORCH_CHECK(extra_indices.scalar_type() == torch::kInt32 && extra_indices.dim() == 2 &&
              extra_indices.size(0) == num_tokens && extra_indices.stride(1) == 1);
  const PagedLayout ex = parse_layout(extra_kv_cache, Dsv41Fp4Layout::BYTES_PER_TOKEN, "extra_kv_cache");
  if (topk_length.has_value()) TORCH_CHECK(topk_length->scalar_type() == torch::kInt32 && topk_length->numel() >= num_tokens);
  if (extra_topk_length.has_value())
    TORCH_CHECK(extra_topk_length->scalar_type() == torch::kInt32 && extra_topk_length->numel() >= num_tokens);
  if (attn_sink.has_value()) TORCH_CHECK(attn_sink->scalar_type() == torch::kFloat32 && attn_sink->numel() >= kHeads);

  execution::AttentionParams p{};
  p.num_heads = kHeads;
  p.topk = (int)indices.size(1);
  p.q = reinterpret_cast<const bf16*>(q.data_ptr());
  p.kv = static_cast<const uint8_t*>(kv_cache.data_ptr());
  p.indices = indices.data_ptr<int32_t>();
  p.mid_out = reinterpret_cast<bf16*>(mid_out.data_ptr());
  p.mid_lse = mid_lse.data_ptr<float>();
  p.topk_length = topk_length.has_value() ? topk_length->data_ptr<int32_t>() : nullptr;
  p.output = reinterpret_cast<bf16*>(output.data_ptr());
  p.out_lse = out_lse.data_ptr<float>();
  p.attn_sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
  p.extra_kv = static_cast<const uint8_t*>(extra_kv_cache.data_ptr());
  p.extra_indices = extra_indices.data_ptr<int32_t>();
  p.extra_topk_length = extra_topk_length.has_value() ? extra_topk_length->data_ptr<int32_t>() : nullptr;
  p.extra_topk = (int)extra_indices.size(1);
  p.extra_page_size = ex.page_block_size;
  p.extra_page_stride_bytes = ex.stride_block;
  p.num_tokens = num_tokens;
  p.allocated_splits = (int)num_splits;
  p.chunks_per_block = (int)plan_cpb;
  p.sm_scale = (float)sm_scale;
  p.page_stride_bytes = kv.stride_block;
  p.indices_stride_elems = (size_t)indices.stride(0);
  p.extra_indices_stride_elems = (size_t)extra_indices.stride(0);
  p.out_lse_stride_elems = (size_t)out_lse.stride(0);
  p.page_size = kv.page_block_size;
  p.extra_fp4 = true;
  p.lse_scale = (float)lse_scale;
  p.interleaved_scales = interleaved;
  launch(p, /*head_blocks=*/1, (int)plan_block_threads, (int)plan_shared_bytes, hoisted_merge,
         c10::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("sparse_mla_decode_ldg", &sparse_mla_decode_ldg); }
