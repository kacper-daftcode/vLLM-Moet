// DeepSeek-V4 MoE routing top-6 (sqrt-softplus scores + correction bias, noaux_tc) for sm_120, bit-identical to
// vLLM's Triton _dsv4_topk_kernel (model_executor/layers/fused_moe/router/dsv4_topk.py).
//
// One warp per token holds 12 experts per lane (384 = 32 x 12; 256 = 32 x 8) and each of the six selections is two
// warp reductions in hardware: redux.sync.max over an order-preserving integer key of the biased score (+0 and -0
// share a key: Triton compares with ==), then redux.sync.min over the lowest expert index holding that key (Triton's
// tie-break). The arithmetic is Triton's instruction for instruction:
//   e = ex2.approx(x * log2e); l = __nv_logf(1 + e) (libdevice, FTZ); w = sqrt.approx.ftz(x > 20 ? x : l)
//   score = w + bias (NaN -> -1e30); weight_sum = ((w0 + w4) + w2) + ((w1 + w5) + w3) (Triton's xor 4/2/1 butterfly
//   over the 8 slots, 6 and 7 zero); weight = div.full(scale, sum > 0 ? sum : 1) * w.
// Build without fast-math, with -ftz=true (Triton's libdevice reflect).
//
// The PDL launch (pdl=true) reads the bias rows and the image-token test before griddepcontrol.wait and releases its
// dependents before the stores. The default is the plain launch: in the served decode graph a PDL top-k behind the
// router's early trigger ends ~3.5 us later than a plain one, and the MoE input quantization behind it ~1.8 us later.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <type_traits>

namespace {

__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float sqrt_approx_ftz(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float mul_rn(float a, float b) {
  float y;
  asm("mul.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float add_rn(float a, float b) {
  float y;
  asm("add.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

// order-preserving key of a non-NaN float; +0 and -0 map to the same key
__device__ __forceinline__ unsigned key_of(float f) {
  if (f == 0.0f) f = 0.0f;
  const unsigned b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

template <int E, bool HAS_PAD, bool HAS_VL, typename IdT, bool PDL, typename TokT>
__global__ void __launch_bounds__(256) dsv4_topk_kernel(const float* __restrict__ logits,
                                                        const float* __restrict__ bias,
                                                        const bool* __restrict__ is_padding, float* __restrict__ out_w,
                                                        IdT* __restrict__ out_ids, float scale,
                                                        const TokT* __restrict__ input_ids,
                                                        const float* __restrict__ bias_vl, int64_t image_lo,
                                                        int num_tokens) {
  constexpr int PER = E / 32;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  if (row >= num_tokens) {
    if constexpr (PDL) griddep_wait();
    return;
  }
  // expert of slot j on this lane: lane + 32 * j (lane-interleaved, coalesced loads)
  float b[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) b[j] = bias[lane + 32 * j];
  if constexpr (HAS_VL) {
    const int64_t tok = (int64_t)input_ids[row];
    if (tok >= image_lo && tok < image_lo + 5) {
#pragma unroll
      for (int j = 0; j < PER; ++j) b[j] = bias_vl[lane + 32 * j];
    }
  }
  if constexpr (PDL) griddep_wait();
  float w[PER];
  unsigned k[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const float x = logits[(size_t)row * E + lane + 32 * j];
    const float e = ex2_approx(mul_rn(x, 1.44269502162933349609375f));  // 0x3FB8AA3B
    const float l = logf(add_rn(e, 1.0f));
    w[j] = sqrt_approx_ftz(x > 20.0f ? x : l);
    float s = add_rn(w[j], b[j]);
    if (!(s == s)) s = -1e30f;
    k[j] = key_of(s);
  }
  float sel_w = 0.0f;  // lane s < 6 keeps slot s
  int sel_id = 0;
#pragma unroll
  for (int s = 0; s < 6; ++s) {
    unsigned m = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) m = k[j] > m ? k[j] : m;
    m = __reduce_max_sync(0xffffffffu, m);
    unsigned cand = E;
#pragma unroll
    for (int j = PER - 1; j >= 0; --j)
      if (k[j] == m) cand = lane + 32 * j;
    const unsigned id = __reduce_min_sync(0xffffffffu, cand);
    const int owner = id & 31, jj = id >> 5;
    float mine = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j)
      if (j == jj) mine = w[j];
    const float wsel = __shfl_sync(0xffffffffu, mine, owner);
    if (lane == owner) {
#pragma unroll
      for (int j = 0; j < PER; ++j)
        if (j == jj) k[j] = 0;  // -inf: never selected again
    }
    if (lane == s) {
      sel_w = wsel;
      sel_id = (int)id;
    }
  }
  // Triton's butterfly over the 8 slots (xor 4, 2, 1): ((w0 + w4) + w2) + ((w1 + w5) + w3)
  const float w0 = __shfl_sync(0xffffffffu, sel_w, 0), w1 = __shfl_sync(0xffffffffu, sel_w, 1);
  const float w2 = __shfl_sync(0xffffffffu, sel_w, 2), w3 = __shfl_sync(0xffffffffu, sel_w, 3);
  const float w4 = __shfl_sync(0xffffffffu, sel_w, 4), w5 = __shfl_sync(0xffffffffu, sel_w, 5);
  const float sum = add_rn(add_rn(add_rn(w0, w4), w2), add_rn(add_rn(w1, w5), w3));
  const float f = div_full(scale, sum > 0.0f ? sum : 1.0f);
  float outw = mul_rn(f, sel_w);
  IdT outid = (IdT)sel_id;
  if constexpr (HAS_PAD) {
    if (is_padding[row]) {
      outw = 0.0f;
      outid = (IdT)-1;
    }
  }
  if constexpr (PDL) griddep_launch_dependents();
  if (lane < 6) {
    out_w[(size_t)row * 6 + lane] = outw;
    out_ids[(size_t)row * 6 + lane] = outid;
  }
}

template <int E, bool HP, bool HV, typename IdT, bool PDL, typename TokT>
void launch_one(int blocks, int threads, cudaStream_t st, const float* lg, const float* bias, const bool* pad,
                float* ow, IdT* oi, float scale, const TokT* ids, const float* bvl, int64_t lo, int T) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks);
  cfg.blockDim = dim3(threads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = PDL ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dsv4_topk_kernel<E, HP, HV, IdT, PDL, TokT>, lg, bias, pad, ow, oi, scale, ids, bvl,
                                    lo, T));
}

template <int E, typename IdT, typename TokT>
void dispatch(bool hp, bool hv, bool pdl, int blocks, int threads, cudaStream_t st, const float* lg, const float* bias,
              const bool* pad, float* ow, IdT* oi, float scale, const TokT* ids, const float* bvl, int64_t lo,
              int T) {
#define L(HP, HV, P) launch_one<E, HP, HV, IdT, P, TokT>(blocks, threads, st, lg, bias, pad, ow, oi, scale, ids, bvl, lo, T)
  if (pdl) {
    if (hp) { if (hv) L(true, true, true); else L(true, false, true); }
    else { if (hv) L(false, true, true); else L(false, false, true); }
  } else {
    if (hp) { if (hv) L(true, true, false); else L(true, false, false); }
    else { if (hv) L(false, true, false); else L(false, false, false); }
  }
#undef L
}

}  // namespace

// out_w [T, 6] fp32, out_ids [T, 6] int32/int64 (preallocated). is_padding / input_ids / bias_vl optional (None).
void dsv4_topk_out(torch::Tensor logits, torch::Tensor bias, c10::optional<torch::Tensor> is_padding,
                   torch::Tensor out_w, torch::Tensor out_ids, double scale, c10::optional<torch::Tensor> input_ids,
                   c10::optional<torch::Tensor> bias_vl, int64_t image_lo, bool pdl) {
  const int T = (int)logits.size(0), E = (int)logits.size(1);
  TORCH_CHECK(E == 384 || E == 256, "dsv4_topk_sm120: 256 or 384 experts");
  if (T == 0) return;
  const bool hp = is_padding.has_value(), hv = bias_vl.has_value() && image_lo > 0;
  const int warps = T < 8 ? T : 8;
  const int threads = 32 * warps, blocks = (T + warps - 1) / warps;
  auto st = at::cuda::getCurrentCUDAStream();
  const bool* pad = hp ? is_padding->data_ptr<bool>() : nullptr;
  const float* bvl = hv ? bias_vl->data_ptr<float>() : nullptr;
  auto lg = logits.data_ptr<float>();
  auto bs = bias.data_ptr<float>();
  auto ow = out_w.data_ptr<float>();
  const float sc = (float)scale;
  auto run = [&](auto* ids) {
    using TokT = std::remove_const_t<std::remove_pointer_t<decltype(ids)>>;
    if (out_ids.scalar_type() == torch::kInt32) {
      auto oi = out_ids.data_ptr<int32_t>();
      if (E == 384) dispatch<384, int32_t, TokT>(hp, hv, pdl, blocks, threads, st, lg, bs, pad, ow, oi, sc, ids, bvl, image_lo, T);
      else dispatch<256, int32_t, TokT>(hp, hv, pdl, blocks, threads, st, lg, bs, pad, ow, oi, sc, ids, bvl, image_lo, T);
    } else {
      TORCH_CHECK(out_ids.scalar_type() == torch::kInt64);
      auto oi = out_ids.data_ptr<int64_t>();
      if (E == 384) dispatch<384, int64_t, TokT>(hp, hv, pdl, blocks, threads, st, lg, bs, pad, ow, oi, sc, ids, bvl, image_lo, T);
      else dispatch<256, int64_t, TokT>(hp, hv, pdl, blocks, threads, st, lg, bs, pad, ow, oi, sc, ids, bvl, image_lo, T);
    }
  };
  if (hv && input_ids->scalar_type() == torch::kInt32) run((const int32_t*)input_ids->data_ptr<int32_t>());
  else if (hv) run((const int64_t*)input_ids->data_ptr<int64_t>());
  else run((const int32_t*)nullptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("dsv4_topk_out", &dsv4_topk_out, "DeepSeek-V4 routing top-6"); }
