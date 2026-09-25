// DeepSeek-V4.1 mHC "pre" epilogue for decode on sm_120: the second of the two TileLang kernels
// vLLM runs per sublayer (`mhc_pre_big_fuse_with_norm_tilelang`, shifted mHC with the carried
// pre-mix, RMSNorm fused), as one CUDA kernel with a CTA of 32 + H/8 threads per token instead of
// 96 - and bit-identical to it.
//
// STATUS (2026-09-24): bit-identical on every output (post_mix, comb_mix, next_pre_mix, layer_input,
// aux; 1..64 tokens), NOT applied: on an RTX 5090 it takes 4.5 us against TileLang's 3.5. The
// critical path is warp 0's 20 Sinkhorn iterations (~93 ns each: two IEEE divisions and four
// dependent shuffles per iteration, the same chain TileLang has); the phase timing (warp 0 alone
// 4.0 us, the collapse threads alone 2.5 us) puts ~1 us of unexplained overhead in warp 0's path
// before the Sinkhorn. The natural continuation is the fused sm_120 mHC (post + fn GEMM + this).
//
// Per token i (grid = tokens):
//   mixes[j]   = sum_s gemm_out_mul[s, i, j] (s ascending) * rsqrt(sum_s sqrsum[s, i] / rms_numel + rms_eps)
//   next_pre_mix[j] = sigmoid(mixes[j] * scale[0] + base[j]) + hc_pre_eps            j < hc
//   post_mix[j]     = sigmoid(mixes[hc + j] * scale[1] + base[hc + j]) * hc_post_mult
//   comb_mix        = Sinkhorn(mixes[2 hc + jk] * scale[2] + base[2 hc + jk])         (hc x hc)
//   collapsed[h]    = bf16( sum_c pre_mix_in[c] * residual[i, c, h] )                 (fma chain from 0)
//   layer_input[h]  = bf16( (float(collapsed[h]) * rsqrt(sum_h collapsed[h]^2 / H + norm_eps)) * w[h] )
//   aux[h]          = bf16( (sum_c residual[i, c, h]) / hc )                          (optional)
//
// Warp 0 owns the coefficient path (its 24 + 1 split sums, the sigmoids, the 4 x 4 Sinkhorn with
// TileLang's xor-butterflies: rows 2,1 - columns 8,4); warps 1.. own eight positions per lane
// each: the four stream vectors and the norm weight are loaded at once, the collapse is the same
// fma chain, the rounded bf16 values go to shared memory, and 64 of those threads re-run
// TileLang's sum of squares in its exact order (16 positions per thread accumulated over the
// 1024-wide blocks with fma, the 0,8,1,9,... summation, then the 64-wide butterfly 32..1) so
// the norm's rsqrt is the same bit pattern. Named barriers keep warp 0 out of the collapse syncs.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kHC = 4;
constexpr int kMix = kHC * (kHC + 2);  // 24
constexpr int kVec = 8;                // bf16 per collapse thread
constexpr int kTLBlock = 1024;         // TileLang's hidden_block for H % 1024 == 0
constexpr int kTLThreads = 64;         // TileLang's collapse threads (its 96 minus warp 0)
constexpr int kMaxH = 5120;            // DeepSeek-V4.1-Flash; the launch bound keeps ~97 registers per thread
constexpr int kMaxThreads = 32 + kMaxH / kVec;
constexpr int kMaxSplits = 16;         // vLLM buckets the split count at <= 16

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(0.0f - x)); }

__device__ __forceinline__ void collapse_barrier(int nthreads) {
  asm volatile("bar.sync 1, %0;" ::"r"(nthreads) : "memory");
}

__global__ void __launch_bounds__(kMaxThreads) mhc_pre_norm_kernel(
    const float* __restrict__ mixes,       // [S, T, kMix]
    const float* __restrict__ sqrsum,      // [S, T]
    const float* __restrict__ hc_scale,    // [3]
    const float* __restrict__ hc_base,     // [kMix]
    const __nv_bfloat16* __restrict__ residual,  // [T, kHC, H]
    const __nv_bfloat16* __restrict__ norm_w,    // [H]
    const float* __restrict__ pre_mix_in,  // [T, kHC]
    float* __restrict__ post_mix,          // [T, kHC]
    float* __restrict__ comb_mix,          // [T, kHC * kHC]
    __nv_bfloat16* __restrict__ layer_input,     // [T, H]
    float* __restrict__ next_pre_mix,      // [T, kHC]
    __nv_bfloat16* __restrict__ aux,       // [T, H] or nullptr
    const int S, const int H, const float rms_numel, const float rms_eps, const float hc_pre_eps,
    const float hc_sinkhorn_eps, const float hc_post_mult, const int sinkhorn_repeat, const float norm_eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* s_round = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [H]
  float* s_red = reinterpret_cast<float*>(smem_raw + H * sizeof(__nv_bfloat16));  // [64] + rsqrt
  const int i = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n_collapse = blockDim.x - 32;
  const unsigned full = 0xffffffffu;

  if (warp == 0) {
    // ---- coefficient path: lanes 0..23 the mixes, lane 24 the hc RMS sum of squares ----
    // all split loads in flight at once (TileLang unrolls its compile-time split loop), summed in
    // ascending split order
    float vals[kMaxSplits];
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s) {
      vals[s] = 0.f;
      if (s < S && lane <= kMix) {
        vals[s] = (lane < kMix) ? mixes[(static_cast<int64_t>(s) * gridDim.x + i) * kMix + lane]
                                : sqrsum[static_cast<int64_t>(s) * gridDim.x + i];
      }
    }
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s) {
      if (s < S) acc = acc + vals[s];
    }
    const float rms = rsqrtf(__shfl_sync(full, acc, kMix) / rms_numel + rms_eps);
    const float m = acc * rms;  // lanes < kMix
    if (lane < kHC) {
      next_pre_mix[i * kHC + lane] = sigmoidf_(m * hc_scale[0] + hc_base[lane]) + hc_pre_eps;
    }
    const float m_post = __shfl_sync(full, m, lane < kHC ? kHC + lane : 0);
    if (lane < kHC) {
      post_mix[i * kHC + lane] = sigmoidf_(m_post * hc_scale[1] + hc_base[kHC + lane]) * hc_post_mult;
    }
    // Sinkhorn: lane l holds cm[(l & 15) / 4][(l & 15) % 4] (lanes 16..31 mirror 0..15)
    const int e = lane & 15;
    float cm = __shfl_sync(full, m, 2 * kHC + e) * hc_scale[2] + hc_base[2 * kHC + e];
    float row_max = fmaxf(-INFINITY, cm);
    row_max = fmaxf(row_max, __shfl_xor_sync(full, row_max, 2));
    row_max = fmaxf(row_max, __shfl_xor_sync(full, row_max, 1));
    cm = expf(cm - row_max);
    auto row_sum = [&](float v) {
      float s = 0.0f + v;
      s = s + __shfl_xor_sync(full, s, 2);
      s = s + __shfl_xor_sync(full, s, 1);
      return s;
    };
    auto col_sum = [&](float v) {
      float s = 0.0f + v;
      s = s + __shfl_xor_sync(full, s, 8);
      s = s + __shfl_xor_sync(full, s, 4);
      return s;
    };
    cm = cm / row_sum(cm) + hc_sinkhorn_eps;
    cm = cm / (col_sum(cm) + hc_sinkhorn_eps);
    for (int r = 0; r < sinkhorn_repeat - 1; ++r) {
      cm = cm / (row_sum(cm) + hc_sinkhorn_eps);
      cm = cm / (col_sum(cm) + hc_sinkhorn_eps);
    }
    if (lane < kHC * kHC) comb_mix[i * kHC * kHC + lane] = cm;
    return;
  }

  // ---- collapse path: thread ct handles positions [8 ct, 8 ct + 8) ----
  const int ct = tid - 32;
  const int h0 = ct * kVec;
  uint4 xv[kHC];
  const __nv_bfloat16* rrow = residual + static_cast<int64_t>(i) * kHC * H;
#pragma unroll
  for (int c = 0; c < kHC; ++c) xv[c] = *reinterpret_cast<const uint4*>(rrow + c * H + h0);
  const uint4 wv = *reinterpret_cast<const uint4*>(norm_w + h0);
  float pre[kHC];
#pragma unroll
  for (int c = 0; c < kHC; ++c) pre[c] = __ldg(pre_mix_in + i * kHC + c);

  uint4 rv;  // the eight bf16-rounded collapsed values
  __nv_bfloat162* r2 = reinterpret_cast<__nv_bfloat162*>(&rv);
  float auxv[kVec];
  {
    const __nv_bfloat162* x2[kHC];
#pragma unroll
    for (int c = 0; c < kHC; ++c) x2[c] = reinterpret_cast<const __nv_bfloat162*>(&xv[c]);
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p) {
      float o0 = 0.f, o1 = 0.f, a0 = 0.f, a1 = 0.f;
#pragma unroll
      for (int c = 0; c < kHC; ++c) {
        const float2 f = __bfloat1622float2(x2[c][p]);
        o0 = fmaf(pre[c], f.x, o0);
        o1 = fmaf(pre[c], f.y, o1);
        a0 = a0 + f.x;
        a1 = a1 + f.y;
      }
      r2[p] = __floats2bfloat162_rn(o0, o1);
      auxv[2 * p] = a0;
      auxv[2 * p + 1] = a1;
    }
  }
  *reinterpret_cast<uint4*>(s_round + h0) = rv;
  collapse_barrier(n_collapse);

  // ---- TileLang's sum of squares, in its order, by 64 of the collapse threads ----
  if (ct < kTLThreads) {
    const int n_blocks = H / kTLBlock;
    float part[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) part[j] = 0.f;
    for (int b = 0; b < n_blocks; ++b) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int g = j >> 2, q = j & 3;
        const int p = b * kTLBlock + (g >> 1) * 512 + ct * 8 + (g & 1) * 4 + q;
        const float v = __bfloat162float(s_round[p]);
        part[j] = fmaf(v, v, part[j]);
      }
    }
    float sumsq = 0.f;
#pragma unroll
    for (int rv_ = 0; rv_ < 16; ++rv_) sumsq = sumsq + part[((rv_ & 1) * 8) + (rv_ >> 1)];
    // AllReduce<SumOp, 64>: the xor-32 step through shared memory, then xor 16..1 in the warp
    s_red[ct] = sumsq;
    collapse_barrier(n_collapse);  // (all collapse threads take part; only 64 use it)
    sumsq = sumsq + s_red[ct ^ 32];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sumsq = sumsq + __shfl_xor_sync(full, sumsq, off);
    if (ct == 0) s_red[kTLThreads] = rsqrtf(sumsq / static_cast<float>(H) + norm_eps);
  } else {
    collapse_barrier(n_collapse);
  }
  collapse_barrier(n_collapse);
  const float rsqrt_norm = s_red[kTLThreads];

  // ---- (collapsed * rsqrt) * weight, store; aux = stream mean ----
  {
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
    uint4 ov;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p) {
      const float2 wf = __bfloat1622float2(w2[p]);
      const float2 rf = __bfloat1622float2(r2[p]);
      o2[p] = __floats2bfloat162_rn((rf.x * rsqrt_norm) * wf.x, (rf.y * rsqrt_norm) * wf.y);
    }
    *reinterpret_cast<uint4*>(layer_input + static_cast<int64_t>(i) * H + h0) = ov;
    if (aux != nullptr) {
      uint4 av;
      __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(&av);
#pragma unroll
      for (int p = 0; p < kVec / 2; ++p) {
        a2[p] = __floats2bfloat162_rn(auxv[2 * p] / static_cast<float>(kHC), auxv[2 * p + 1] / static_cast<float>(kHC));
      }
      *reinterpret_cast<uint4*>(aux + static_cast<int64_t>(i) * H + h0) = av;
    }
  }
}

}  // namespace

void mhc_pre_norm(const at::Tensor& mixes, const at::Tensor& sqrsum, const at::Tensor& hc_scale, const at::Tensor& hc_base,
                  const at::Tensor& residual, const at::Tensor& norm_w, const at::Tensor& pre_mix_in, at::Tensor post_mix,
                  at::Tensor comb_mix, at::Tensor layer_input, at::Tensor next_pre_mix, const c10::optional<at::Tensor>& aux,
                  double rms_numel, double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps, double hc_post_mult,
                  int64_t sinkhorn_repeat, double norm_eps) {
  TORCH_CHECK(residual.is_cuda() && residual.dim() == 3 && residual.scalar_type() == at::kBFloat16 && residual.is_contiguous(),
              "residual: bf16 [T, hc, H] contiguous");
  const int64_t T = residual.size(0), hc = residual.size(1), H = residual.size(2);
  TORCH_CHECK(hc == kHC, "hc_mult must be ", kHC);
  TORCH_CHECK(H % kTLBlock == 0 && H <= kMaxH, "H must be a multiple of 1024 up to ", kMaxH);
  TORCH_CHECK(mixes.is_cuda() && mixes.scalar_type() == at::kFloat && mixes.is_contiguous() && mixes.dim() == 3 &&
                  mixes.size(1) == T && mixes.size(2) == kMix && mixes.size(0) <= kMaxSplits,
              "mixes: fp32 [S, T, 24]");
  const int64_t S = mixes.size(0);
  TORCH_CHECK(sqrsum.is_cuda() && sqrsum.scalar_type() == at::kFloat && sqrsum.is_contiguous() && sqrsum.dim() == 2 &&
                  sqrsum.size(0) == S && sqrsum.size(1) == T,
              "sqrsum: fp32 [S, T]");
  TORCH_CHECK(hc_scale.is_cuda() && hc_scale.scalar_type() == at::kFloat && hc_scale.is_contiguous() && hc_scale.numel() == 3,
              "hc_scale: fp32 [3]");
  TORCH_CHECK(hc_base.is_cuda() && hc_base.scalar_type() == at::kFloat && hc_base.is_contiguous() && hc_base.numel() == kMix,
              "hc_base: fp32 [24]");
  TORCH_CHECK(norm_w.is_cuda() && norm_w.scalar_type() == at::kBFloat16 && norm_w.is_contiguous() && norm_w.numel() == H,
              "norm_weight: bf16 [H]");
  TORCH_CHECK(pre_mix_in.is_cuda() && pre_mix_in.scalar_type() == at::kFloat && pre_mix_in.is_contiguous() &&
                  pre_mix_in.numel() == T * kHC,
              "pre_mix_in: fp32 [T, hc]");
  TORCH_CHECK(post_mix.is_contiguous() && post_mix.numel() == T * kHC && post_mix.scalar_type() == at::kFloat, "post_mix");
  TORCH_CHECK(comb_mix.is_contiguous() && comb_mix.numel() == T * kHC * kHC && comb_mix.scalar_type() == at::kFloat, "comb_mix");
  TORCH_CHECK(layer_input.is_contiguous() && layer_input.numel() == T * H && layer_input.scalar_type() == at::kBFloat16,
              "layer_input");
  TORCH_CHECK(next_pre_mix.is_contiguous() && next_pre_mix.numel() == T * kHC && next_pre_mix.scalar_type() == at::kFloat,
              "next_pre_mix");
  if (aux.has_value()) {
    TORCH_CHECK(aux->is_contiguous() && aux->numel() == T * H && aux->scalar_type() == at::kBFloat16, "aux: bf16 [T, H]");
  }
  if (T == 0) return;
  const c10::cuda::OptionalCUDAGuard guard(residual.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(residual.get_device()).stream();
  const int threads = 32 + static_cast<int>(H) / kVec;
  const size_t smem = static_cast<size_t>(H) * sizeof(__nv_bfloat16) + (kTLThreads + 4) * sizeof(float);
  mhc_pre_norm_kernel<<<dim3(T), dim3(threads), smem, stream>>>(
      mixes.data_ptr<float>(), sqrsum.data_ptr<float>(), hc_scale.data_ptr<float>(), hc_base.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr()),
      pre_mix_in.data_ptr<float>(), post_mix.data_ptr<float>(), comb_mix.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr()), next_pre_mix.data_ptr<float>(),
      aux.has_value() ? reinterpret_cast<__nv_bfloat16*>(aux->data_ptr()) : nullptr, static_cast<int>(S),
      static_cast<int>(H), static_cast<float>(rms_numel), static_cast<float>(rms_eps), static_cast<float>(hc_pre_eps),
      static_cast<float>(hc_sinkhorn_eps), static_cast<float>(hc_post_mult), static_cast<int>(sinkhorn_repeat),
      static_cast<float>(norm_eps));
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mhc_pre_norm", &mhc_pre_norm, "DeepSeek-V4.1 shifted mHC pre epilogue: mixes, Sinkhorn, collapse, RMSNorm",
        py::arg("mixes"), py::arg("sqrsum"), py::arg("hc_scale"), py::arg("hc_base"), py::arg("residual"), py::arg("norm_w"),
        py::arg("pre_mix_in"), py::arg("post_mix"), py::arg("comb_mix"), py::arg("layer_input"), py::arg("next_pre_mix"),
        py::arg("aux"), py::arg("rms_numel"), py::arg("rms_eps"), py::arg("hc_pre_eps"), py::arg("hc_sinkhorn_eps"),
        py::arg("hc_post_mult"), py::arg("sinkhorn_repeat"), py::arg("norm_eps"));
}
