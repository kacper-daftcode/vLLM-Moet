// Fused MoE router for DeepSeek-V4 / V4.1 at decode token counts (sm_120): the gate GEMM
// (bf16 x bf16 -> fp32 logits), sqrtsoftplus, the correction bias (bias_vl for image tokens),
// the top-k selection and the renormalization in one launch.
//
// STATUS (2026-09-24): measured, NOT applied. Bit-exact routing against vLLM's dsv4_topk (330 random
// tokens, exact ties), but with cold gate weights on an RTX 5090 it is not faster than the three
// kernels it replaces (8.6 vs 8.1 us at 6 tokens; the GEMV phase alone 5.4 vs cuBLAS + split-K 6.1):
// handing the top-k to the last CTA (threadfence + atomic) serializes ~2 us that the separate
// kernels overlap. Kept as the measurement harness (test_moe_gate_topk_sm120.py --bench).
//
// vLLM runs three kernels per MoE layer for this: cuBLAS's bf16 GEMM with a split-K reduce
// (torch.mm(x, W.T, out_dtype=float32): M = 6, N = 384, K = 5120) and the Triton
// `_dsv4_topk_kernel` (fused_moe/router/dsv4_topk.py) - 4.9 + 2.7 + 3.4 us plus two launch gaps
// in the served decode graph.
//
// Phase 1: a CTA owns kExperts (4) expert rows; its warps split K (two 256-element steps per
// warp, so K = 512 x warps: 10 warps for K = 5120), every lane issues all of its loads - the 4 x 2
// weight vectors once, then the x vectors of 4 tokens at a time (T <= kMaxTokens) - before it
// consumes them, so a warp pays one memory round trip per 4 tokens instead of one per K-step, and
// x (from L2) is read once per 4 experts instead of once per expert. The partial sums cross the
// warps through shared memory; one warp per expert finishes the logits, applies sqrtsoftplus and
// writes the scores to a [T, E] scratch. Phase 2: the last CTA to finish (threadfence + atomic
// counter, reset for the next launch so CUDA graphs replay it as is) selects the top-k per token
// exactly like the Triton kernel: score + bias (bias_vl for tokens whose id is in
// [sentinel_lo, sentinel_lo + 5)), NaN -> -1e30, k rounds of argmax with ties resolved to the
// smallest expert id, renormalized to routed_scaling_factor / sum (sum <= 0 -> no division). The
// score uses the same exp2.approx / logf / sqrt.approx that Triton lowers tl.exp / tl.log / tl.sqrt to.
//
// Numerics: the logit is an fp32 sum of exact bf16 x bf16 products in this kernel's order; cuBLAS
// sums them in its split-K order. Both are fp32-accurate; they are not bit-identical, so routing
// can differ from vLLM's kernels where two scores tie within fp32 rounding (measured by the test).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {

constexpr int kExperts = 4;            // expert rows per CTA (share the x loads)
constexpr int kStepsPerWarp = 2;       // 256-element K-steps per warp
constexpr int kMaxWarps = 10;          // K <= 10 * 512 = 5120 (registers: 64 acc + 64 load regs per lane)
constexpr int kMaxTokens = 16;
constexpr int kTokenPass = 8;          // tokens whose x vectors a lane holds at once (16 uint4)
constexpr int kMaxTopK = 8;
constexpr int kVec = 8;                // bf16 elements per 16-byte load
constexpr int kStepElems = 32 * kVec;  // 256

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float sqrt_approx(float x) {
  float y;
  asm("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// Triton: tl.sqrt(tl.where(l > 20, l, tl.log(1 + tl.exp(l)))) - tl.exp on fp32 lowers to
// ex2.approx(x * log2e), tl.log to libdevice __nv_logf (= CUDA logf), tl.sqrt to sqrt.approx.
__device__ __forceinline__ float sqrtsoftplus(float l) {
  const float e = ex2_approx(l * 1.4426950408889634f);
  const float sp = logf(1.0f + e);
  return sqrt_approx(l > 20.0f ? l : sp);
}

// N values per lane (N = 32 * 2^m) -> lane l holds the warp sums of values l * 2^m + i, i < 2^m.
// Round r (lane distance 16 >> r): the lanes with that lane bit clear keep the lower half of
// their values and send the upper half to the partner; the others do the opposite; both add.
template <int N>
__device__ __forceinline__ void butterfly_reduce(float (&red)[N], const int lane) {
#pragma unroll
  for (int r = 0; r < 5; ++r) {
    const int half = N >> (r + 1);
    const int off = 16 >> r;
    const bool upper = (lane & off) != 0;
#pragma unroll
    for (int i = 0; i < half; ++i) {
      const float mine = upper ? red[i + half] : red[i];
      const float send = upper ? red[i] : red[i + half];
      red[i] = mine + __shfl_xor_sync(0xffffffffu, send, off);
    }
  }
}

__device__ __forceinline__ float dot8(const uint4& a, const uint4& b, float acc) {
  const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
  const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
  for (int j = 0; j < kVec / 2; ++j) {
    const float2 fa = __bfloat1622float2(a2[j]);
    const float2 fb = __bfloat1622float2(b2[j]);
    acc = fmaf(fa.x, fb.x, acc);
    acc = fmaf(fa.y, fb.y, acc);
  }
  return acc;
}

template <int T_MAX, int kPerLane, bool kLogitsOnly>  // kPerLane = ceil(E / 32): 12 for DeepSeek-V4.1's 384 experts
__global__ void __launch_bounds__(kMaxWarps * 32) moe_gate_topk_kernel(
    const __nv_bfloat16* __restrict__ x, const int64_t x_stride,   // [T, K]
    const __nv_bfloat16* __restrict__ w,                           // [E, K] contiguous
    const float* __restrict__ bias,                                // [E]
    const float* __restrict__ bias_vl,                             // [E] or nullptr
    const int64_t* __restrict__ input_ids,                         // [T] or nullptr
    const int64_t sentinel_lo, const int T, const int K, const int E, const int top_k,
    const float routed_scaling,
    float* __restrict__ scores,                                    // [T, E] scratch
    int* __restrict__ counter,                                     // 1 int, zero at rest
    float* __restrict__ topk_w,                                    // [T, top_k]
    int32_t* __restrict__ topk_ids) {                              // [T, top_k]
  __shared__ float s_part[kMaxWarps][kExperts][T_MAX];
  __shared__ int s_last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n_warps = blockDim.x >> 5;  // = K / 512

  // ---- phase 1: this CTA's kExperts rows, this warp's two K-steps ----
  const int e0 = blockIdx.x * kExperts;
  const int k0 = warp * kStepsPerWarp * kStepElems + lane * kVec;
  uint4 wv[kExperts][kStepsPerWarp];
#pragma unroll
  for (int g = 0; g < kExperts; ++g) {
#pragma unroll
    for (int s = 0; s < kStepsPerWarp; ++s) {
      wv[g][s] = *reinterpret_cast<const uint4*>(w + static_cast<int64_t>(e0 + g) * K + k0 + s * kStepElems);
    }
  }
  float acc[kExperts][T_MAX];
#pragma unroll
  for (int g = 0; g < kExperts; ++g) {
#pragma unroll
    for (int t = 0; t < T_MAX; ++t) acc[g][t] = 0.f;
  }
#pragma unroll
  for (int t0 = 0; t0 < T_MAX; t0 += kTokenPass) {
    if (t0 < T) {
      uint4 xv[kTokenPass][kStepsPerWarp];
#pragma unroll
      for (int i = 0; i < kTokenPass; ++i) {
        const int t = t0 + i < T ? t0 + i : t0;  // clamp: the extra loads hit a valid row and are unused
#pragma unroll
        for (int s = 0; s < kStepsPerWarp; ++s) {
          xv[i][s] = *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(t) * x_stride + k0 + s * kStepElems);
        }
      }
#pragma unroll
      for (int i = 0; i < kTokenPass; ++i) {
        if (t0 + i < T) {
#pragma unroll
          for (int g = 0; g < kExperts; ++g) {
            float a = acc[g][t0 + i];
#pragma unroll
            for (int s = 0; s < kStepsPerWarp; ++s) a = dot8(xv[i][s], wv[g][s], a);
            acc[g][t0 + i] = a;
          }
        }
      }
    }
  }
  // Warp reduction of the kExperts x T partials as one transposed butterfly (values ordered
  // v = t * kExperts + g so the real ones are a prefix): each round halves both the values a lane
  // holds and the lane distance, so N values cost N - N/32 shuffles instead of 5 N, and lane l ends
  // up with the complete sums of values l * (N / 32) + i. N = 32 covers T <= 8, N = 64 up to 16.
  if (T <= 8) {
    float red[32];
#pragma unroll
    for (int v = 0; v < 32; ++v) red[v] = (v / kExperts < T_MAX) ? acc[v % kExperts][v / kExperts] : 0.f;
    butterfly_reduce<32>(red, lane);
    if (lane / kExperts < T) s_part[warp][lane % kExperts][lane / kExperts] = red[0];
  } else {
    float red[64];
#pragma unroll
    for (int v = 0; v < 64; ++v) red[v] = acc[v % kExperts][v / kExperts];
    butterfly_reduce<64>(red, lane);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = 2 * lane + i;
      if (v / kExperts < T) s_part[warp][v % kExperts][v / kExperts] = red[i];
    }
  }
  __syncthreads();
  if (warp < kExperts && lane < T) {
    float v = 0.f;
    for (int wi = 0; wi < n_warps; ++wi) v += s_part[wi][warp][lane];
    scores[static_cast<int64_t>(lane) * E + e0 + warp] = kLogitsOnly ? v : sqrtsoftplus(v);
  }
  if constexpr (kLogitsOnly) return;

  // ---- hand-off: the last CTA to arrive does the top-k ----
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(counter, 1) == static_cast<int>(gridDim.x) - 1) ? 1 : 0;
  __syncthreads();
  if (!s_last) return;
  __threadfence();

  // ---- phase 2: top-k per token, one warp per token ----
  for (int t = warp; t < T; t += n_warps) {
    bool image = false;
    if (bias_vl != nullptr && input_ids != nullptr) {
      const int64_t id = __ldg(input_ids + t);
      image = (id >= sentinel_lo) && (id < sentinel_lo + 5);
    }
    const float* brow = image ? bias_vl : bias;
    float score[kPerLane], cur[kPerLane];
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      const int idx = lane + 32 * j;
      if (idx < E) {
        const float s = __ldcg(scores + static_cast<int64_t>(t) * E + idx);  // L2, written by other CTAs
        score[j] = s;
        float c = s + __ldg(brow + idx);
        c = (c == c) ? c : -1e30f;  // NaN -> -1e30 like the Triton kernel
        cur[j] = c;
      } else {
        score[j] = 0.f;
        cur[j] = -INFINITY;
      }
    }
    float sel_w[kMaxTopK];
    int sel_id[kMaxTopK];
    float wsum = 0.f;
    for (int slot = 0; slot < top_k; ++slot) {
      // max value over all experts
      float m = -INFINITY;
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) m = fmaxf(m, cur[j]);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
      // smallest expert id holding the max
      int cand = E;
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        const int idx = lane + 32 * j;
        if (idx < E && cur[j] == m) cand = min(cand, idx);
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) cand = min(cand, __shfl_xor_sync(0xffffffffu, cand, off));
      // its score (the pre-bias weight) and removal
      float sw = 0.f;
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        const int idx = lane + 32 * j;
        if (idx == cand) {
          sw = score[j];
          cur[j] = -INFINITY;
        }
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) sw += __shfl_xor_sync(0xffffffffu, sw, off);
      sel_w[slot] = sw;
      sel_id[slot] = cand;
      wsum += sw;
    }
    const float scale = routed_scaling / (wsum > 0.f ? wsum : 1.0f);
    if (lane < top_k) {
      topk_w[static_cast<int64_t>(t) * top_k + lane] = sel_w[lane] * scale;
      topk_ids[static_cast<int64_t>(t) * top_k + lane] = sel_id[lane];
    }
  }
  __syncthreads();
  if (tid == 0) *counter = 0;  // ready for the next launch / graph replay
}

}  // namespace

// x [T, K] bf16 (unit inner stride), w [E, K] bf16 contiguous, bias / bias_vl fp32 [E], input_ids
// int64 [T] (both optional: pass None), scores fp32 [T, E] scratch, counter int32 [1] (zero),
// topk_w fp32 [T, top_k], topk_ids int32 [T, top_k].
void moe_gate_topk(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bias,
                   const c10::optional<at::Tensor>& bias_vl, const c10::optional<at::Tensor>& input_ids,
                   int64_t sentinel_lo, int64_t top_k, double routed_scaling, at::Tensor scores,
                   at::Tensor counter, at::Tensor topk_w, at::Tensor topk_ids, bool logits_only) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "x: bf16 [T, K]");
  TORCH_CHECK(w.is_cuda() && w.dim() == 2 && w.scalar_type() == at::kBFloat16 && w.is_contiguous(), "w: bf16 [E, K] contiguous");
  const int64_t T = x.size(0), K = x.size(1), E = w.size(0);
  TORCH_CHECK(w.size(1) == K, "K mismatch");
  TORCH_CHECK(1 <= T && T <= kMaxTokens, "T must be in [1, ", kMaxTokens, "]");
  TORCH_CHECK(K % (kStepsPerWarp * kStepElems) == 0 && K / (kStepsPerWarp * kStepElems) <= kMaxWarps,
              "K must be a multiple of 512 up to ", kMaxWarps * kStepsPerWarp * kStepElems);
  TORCH_CHECK((x.stride(0) * 2) % 16 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x rows 16-byte aligned");
  TORCH_CHECK(1 <= E && E <= 1024 && E % kExperts == 0, "E must be a multiple of ", kExperts, " up to 1024");
  TORCH_CHECK(1 <= top_k && top_k <= kMaxTopK && top_k <= E, "top_k must be in [1, ", kMaxTopK, "]");
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kFloat && bias.is_contiguous() && bias.numel() == E, "bias: fp32 [E]");
  const bool has_vl = bias_vl.has_value() && input_ids.has_value();
  if (has_vl) {
    TORCH_CHECK(bias_vl->is_cuda() && bias_vl->scalar_type() == at::kFloat && bias_vl->is_contiguous() && bias_vl->numel() == E,
                "bias_vl: fp32 [E]");
    TORCH_CHECK(input_ids->is_cuda() && input_ids->scalar_type() == at::kLong && input_ids->is_contiguous() &&
                    input_ids->numel() >= T,
                "input_ids: int64 [T]");
  }
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat && scores.is_contiguous() && scores.numel() >= T * E,
              "scores: fp32 [T, E]");
  TORCH_CHECK(counter.is_cuda() && counter.scalar_type() == at::kInt && counter.numel() == 1, "counter: int32 [1]");
  if (!logits_only) {
    TORCH_CHECK(topk_w.is_cuda() && topk_w.scalar_type() == at::kFloat && topk_w.is_contiguous() && topk_w.numel() == T * top_k,
                "topk_w: fp32 [T, top_k]");
    TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous() &&
                    topk_ids.numel() == T * top_k,
                "topk_ids: int32 [T, top_k]");
  }

  const c10::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int n_warps = static_cast<int>(K / (kStepsPerWarp * kStepElems));
  const dim3 grid(E / kExperts), block(n_warps * 32);
  auto kernel = logits_only ? moe_gate_topk_kernel<kMaxTokens, 12, true>
                : (E <= 12 * 32) ? moe_gate_topk_kernel<kMaxTokens, 12, false> : moe_gate_topk_kernel<kMaxTokens, 32, false>;
  kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), bias.data_ptr<float>(),
      has_vl ? bias_vl->data_ptr<float>() : nullptr, has_vl ? input_ids->data_ptr<int64_t>() : nullptr,
      sentinel_lo, static_cast<int>(T), static_cast<int>(K), static_cast<int>(E), static_cast<int>(top_k),
      static_cast<float>(routed_scaling), scores.data_ptr<float>(), counter.data_ptr<int>(), topk_w.data_ptr<float>(),
      topk_ids.data_ptr<int32_t>());
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_gate_topk", &moe_gate_topk, "fused MoE gate GEMV + sqrtsoftplus + bias + top-k + renormalization",
        py::arg("x"), py::arg("w"), py::arg("bias"), py::arg("bias_vl"), py::arg("input_ids"), py::arg("sentinel_lo"),
        py::arg("top_k"), py::arg("routed_scaling"), py::arg("scores"), py::arg("counter"), py::arg("topk_w"),
        py::arg("topk_ids"), py::arg("logits_only") = false);
  m.attr("MAX_TOKENS") = kMaxTokens;
  m.attr("MAX_TOPK") = kMaxTopK;
}
