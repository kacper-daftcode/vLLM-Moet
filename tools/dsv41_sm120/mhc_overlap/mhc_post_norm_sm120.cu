// DeepSeek-V4.1 shifted mHC on sm_120: the critical path of one sublayer boundary in one launch.
//
// Between two sublayers vLLM's decode step runs two TileLang kernels: `mhc_fused_tilelang` (the
// post-mapping of the four residual streams fused with the fn projection GEMM, grid
// [tokens, 12, 8] x 128) and `mhc_pre_big_fuse_with_norm` (split sums, sigmoids, Sinkhorn, the
// collapse of the streams with the carried pre-mix, RMSNorm; grid [tokens] x 96). Only part of
// that is on the critical path: the next sublayer needs the collapsed, normalized input, and
// that depends on the post-mapped streams and the pre-mix the previous sublayer produced - not
// on this sublayer's projection, whose coefficients (post mix, residual mix, next pre-mix) are
// first read after the next sublayer. This kernel computes exactly the critical part - post-mix,
// the bf16 streams, the collapse, the RMSNorm, the draft aux - with one CTA of H/8 threads per
// token, so the projection and the Sinkhorn can run on a side stream behind the sublayer
// (upstream's `mhc_pre_delayed_overlap` design, which is gated to SM100 + DeepGEMM).
//
// Per token i:
//   new_r[j, h]     = post_layer_mix[j] * x[h] + sum_k comb_res_mix[k, j] * residual[k, h]   (fp32, TileLang's order)
//   residual_out    = bf16(new_r)
//   collapsed[h]    = bf16( sum_c pre_mix_in[c] * float(residual_out[c, h]) )               (fma chain from 0)
//   layer_input[h]  = bf16( (float(collapsed[h]) * rsqrt(sum_h collapsed[h]^2 / H + eps)) * w[h] )
//   aux[h]          = bf16( (sum_c float(residual_out[c, h])) / hc )                        (optional)
//
// Bit-identical to the TileLang pair: the post-mix reproduces the contraction nvcc chose for
// mhc_fused_tilelang's / mhc_post_tilelang's `pm * x + cm[0] * r0 + ...` (the first product
// rounded, pm * x fused into it, then one fma per remaining stream - mode 1 below; the test shows
// the other two candidates differ in the last bit), the collapse and the sum of squares reproduce
// mhc_pre_big_fuse_with_norm (16 positions per thread over 64 threads, 0,8,1,9,... summation, the
// 64-wide butterfly), see mhc_pre_norm/mhc_pre_norm_sm120.cu.
//
// Launched with programmatic stream serialization (PDL, `pdl=`, default on): the CTAs come up
// while the kernel that produces x drains; only the x load waits for it, and every store comes
// after that wait. In the served graph the kernel is the only child of the sublayer's NCCL
// all-reduce (the side stream joins onto the all-reduce node, see the patcher) and starts 0.2 us
// after it, like the TileLang kernel it replaced; without PDL, or with a second parent, 2.2 us.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kHC = 4;
constexpr int kVec = 8;         // bf16 per thread
constexpr int kTLBlock = 1024;  // TileLang's hidden_block for H % 1024 == 0
constexpr int kTLThreads = 64;  // TileLang's collapse threads (96 minus warp 0)
constexpr int kMaxH = 5120;
constexpr int kMaxThreads = kMaxH / kVec;

template <int kMode>
__device__ __forceinline__ float post_mix_value(const float pm, const float xf, const float (&cm)[kHC * kHC], const int j,
                                                const float (&rf)[kHC]) {
  // cm[k * kHC + j] multiplies stream k into stream j (TileLang: cm[k, j] * residual_in[k])
  float v;
  if (kMode == 0) {
    v = pm * xf;
#pragma unroll
    for (int k = 0; k < kHC; ++k) v = fmaf(cm[k * kHC + j], rf[k], v);
  } else if (kMode == 1) {
    v = fmaf(pm, xf, cm[j] * rf[0]);
#pragma unroll
    for (int k = 1; k < kHC; ++k) v = fmaf(cm[k * kHC + j], rf[k], v);
  } else {
    v = __fmul_rn(pm, xf);
#pragma unroll
    for (int k = 0; k < kHC; ++k) v = __fadd_rn(v, __fmul_rn(cm[k * kHC + j], rf[k]));
  }
  return v;
}

template <int kMode, bool kPDL>
__global__ void __launch_bounds__(kMaxThreads) mhc_post_norm_kernel(
    const __nv_bfloat16* __restrict__ x,            // [T, H]
    const __nv_bfloat16* __restrict__ residual_in,  // [T, kHC, H]
    const float* __restrict__ post_layer_mix,       // [T, kHC]
    const float* __restrict__ comb_res_mix,         // [T, kHC, kHC]
    const float* __restrict__ pre_mix_in,           // [T, kHC]
    const __nv_bfloat16* __restrict__ norm_w,       // [H]
    __nv_bfloat16* __restrict__ residual_out,       // [T, kHC, H]
    __nv_bfloat16* __restrict__ layer_input,        // [T, H]
    __nv_bfloat16* __restrict__ aux,                // [T, H] or nullptr
    const int H, const float norm_eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* s_round = reinterpret_cast<__nv_bfloat16*>(smem_raw);                 // [H]
  float* s_red = reinterpret_cast<float*>(smem_raw + H * sizeof(__nv_bfloat16));       // [64] + rsqrt
  const int i = blockIdx.x;
  const int ct = threadIdx.x;
  const int h0 = ct * kVec;
  const unsigned full = 0xffffffffu;

  // All loads in flight first. With PDL the kernel may start while the kernel before it on the
  // stream (the one producing x) is still draining: everything but x comes from earlier work
  // (the previous boundary's streams, the coefficients joined from the side stream, the weight),
  // so only the x load waits for the grid dependency.
  const __nv_bfloat16* rrow = residual_in + static_cast<int64_t>(i) * kHC * H;
  uint4 rv[kHC];
#pragma unroll
  for (int c = 0; c < kHC; ++c) rv[c] = *reinterpret_cast<const uint4*>(rrow + c * H + h0);
  const uint4 wv = *reinterpret_cast<const uint4*>(norm_w + h0);
  float pm[kHC], pre[kHC], cm[kHC * kHC];
#pragma unroll
  for (int c = 0; c < kHC; ++c) {
    pm[c] = __ldg(post_layer_mix + i * kHC + c);
    pre[c] = __ldg(pre_mix_in + i * kHC + c);
  }
#pragma unroll
  for (int e = 0; e < kHC * kHC; ++e) cm[e] = __ldg(comb_res_mix + i * kHC * kHC + e);
  if (kPDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // a PDL-launched successor may start its own prologue now; its wait covers this grid's completion
    asm volatile("griddepcontrol.launch_dependents;");
  }
  uint4 xv = *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(i) * H + h0);

  // post-mix -> bf16 streams; collapse + aux from the rounded streams
  uint4 ov[kHC];
  uint4 cv;  // the eight bf16-rounded collapsed values
  float auxv[kVec];
  {
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
    const __nv_bfloat162* r2[kHC];
#pragma unroll
    for (int c = 0; c < kHC; ++c) r2[c] = reinterpret_cast<const __nv_bfloat162*>(&rv[c]);
    __nv_bfloat162* o2[kHC];
#pragma unroll
    for (int c = 0; c < kHC; ++c) o2[c] = reinterpret_cast<__nv_bfloat162*>(&ov[c]);
    __nv_bfloat162* c2 = reinterpret_cast<__nv_bfloat162*>(&cv);
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p) {
      const float2 xf = __bfloat1622float2(x2[p]);
      float rfx[kHC], rfy[kHC];
#pragma unroll
      for (int c = 0; c < kHC; ++c) {
        const float2 f = __bfloat1622float2(r2[c][p]);
        rfx[c] = f.x;
        rfy[c] = f.y;
      }
      float col_x = 0.f, col_y = 0.f, aux_x = 0.f, aux_y = 0.f;
#pragma unroll
      for (int j = 0; j < kHC; ++j) {
        const float nx = post_mix_value<kMode>(pm[j], xf.x, cm, j, rfx);
        const float ny = post_mix_value<kMode>(pm[j], xf.y, cm, j, rfy);
        const __nv_bfloat162 nb = __floats2bfloat162_rn(nx, ny);
        o2[j][p] = nb;
        const float2 nf = __bfloat1622float2(nb);
        col_x = fmaf(pre[j], nf.x, col_x);
        col_y = fmaf(pre[j], nf.y, col_y);
        aux_x = aux_x + nf.x;
        aux_y = aux_y + nf.y;
      }
      c2[p] = __floats2bfloat162_rn(col_x, col_y);
      auxv[2 * p] = aux_x;
      auxv[2 * p + 1] = aux_y;
    }
  }
  __nv_bfloat16* orow = residual_out + static_cast<int64_t>(i) * kHC * H;
#pragma unroll
  for (int c = 0; c < kHC; ++c) *reinterpret_cast<uint4*>(orow + c * H + h0) = ov[c];
  *reinterpret_cast<uint4*>(s_round + h0) = cv;
  if (aux != nullptr) {
    uint4 av;
    __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(&av);
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p) {
      a2[p] = __floats2bfloat162_rn(auxv[2 * p] / static_cast<float>(kHC), auxv[2 * p + 1] / static_cast<float>(kHC));
    }
    *reinterpret_cast<uint4*>(aux + static_cast<int64_t>(i) * H + h0) = av;
  }
  __syncthreads();

  // TileLang's sum of squares, in its order, by 64 threads
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
    for (int r = 0; r < 16; ++r) sumsq = sumsq + part[((r & 1) * 8) + (r >> 1)];
    s_red[ct] = sumsq;
  }
  __syncthreads();
  if (ct < kTLThreads) {
    // AllReduce<SumOp, 64>: the xor-32 step through shared memory, then xor 16..1 in the warp
    float sumsq = s_red[ct] + s_red[ct ^ 32];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sumsq = sumsq + __shfl_xor_sync(full, sumsq, off);
    if (ct == 0) s_red[kTLThreads] = rsqrtf(sumsq / static_cast<float>(H) + norm_eps);
  }
  __syncthreads();
  const float rsqrt_norm = s_red[kTLThreads];

  // (collapsed * rsqrt) * weight
  {
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
    const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(&cv);
    uint4 lv;
    __nv_bfloat162* l2 = reinterpret_cast<__nv_bfloat162*>(&lv);
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p) {
      const float2 wf = __bfloat1622float2(w2[p]);
      const float2 cf = __bfloat1622float2(c2[p]);
      l2[p] = __floats2bfloat162_rn((cf.x * rsqrt_norm) * wf.x, (cf.y * rsqrt_norm) * wf.y);
    }
    *reinterpret_cast<uint4*>(layer_input + static_cast<int64_t>(i) * H + h0) = lv;
  }
}

// ---------------------------------------------------------------------------------------------
// The projection for the side stream: mixes[s, i, n] = sum over split s of fn[n, j, h] * new_r[j, h]
// and sqrsum[s, i] = sum of new_r^2, with new_r the fp32 post-mix recomputed from the same inputs
// as the kernel above - exactly what mhc_fused_tilelang computes ([T, 12, 8] x 128 CTAs: two
// outputs per CTA, the post-mix recomputed twelve times per token, the fn slice read once per
// token) but with one CTA per (split, block of kTok tokens) holding all 24 outputs (two groups of
// 128 threads, 12 outputs each) and reading each fn value once per token block: 8 x ceil(T / 4)
// CTAs instead of 96 T. It runs behind the sublayer, so what matters is how few SM slots it holds
// for how long, not how fast it is alone (6 us at 1-16 tokens against the TileLang kernel's 3-7).
// Bit-identical to the TileLang kernel: the same position-to-thread mapping (thread t of a split
// takes positions split_start + it * 128 + t), the same fma chains per token and output (post-mix
// mode 1, acc = fma(w, new_r, acc) over (it, j), sqr = fma(new_r, new_r, sqr)), TileLang's warp
// butterfly 16..1 and the cross-warp sum 0..3 in order.

constexpr int kMix = kHC * (kHC + 2);   // 24
constexpr int kSplitThreads = 128;      // TileLang's n_thr
constexpr int kProjGroups = 2;          // 128-thread groups per CTA
constexpr int kProjPer = kMix / kProjGroups;  // outputs per group
constexpr int kProjThreads = kSplitThreads * kProjGroups;

template <int kTok>
__global__ void __launch_bounds__(kProjThreads, 1) mhc_proj_kernel(
    const __nv_bfloat16* __restrict__ x,            // [T, H]
    const __nv_bfloat16* __restrict__ residual_in,  // [T, kHC, H]
    const float* __restrict__ post_layer_mix,       // [T, kHC]
    const float* __restrict__ comb_res_mix,         // [T, kHC, kHC]
    const float* __restrict__ fn,                   // [kMix, kHC, H]
    float* __restrict__ mixes,                      // [S, T, kMix]
    float* __restrict__ sqrsum,                     // [S, T]
    const int H, const int S, const int T) {
  __shared__ float s_part[kProjGroups][4][kTok][kProjPer + 1];
  const int s = blockIdx.x, i0 = blockIdx.y * kTok;
  const int tid = threadIdx.x;
  const int g = tid / kSplitThreads, t = tid % kSplitThreads;
  const int warp_in_group = t >> 5, lane = t & 31;
  const int h_per_split = H / S;
  const int h_iters = h_per_split / kSplitThreads;
  const int n0 = g * kProjPer;
  const int n_tok = min(kTok, T - i0);
  const unsigned full = 0xffffffffu;

  float acc[kTok][kProjPer];
#pragma unroll
  for (int tok = 0; tok < kTok; ++tok) {
#pragma unroll
    for (int n = 0; n < kProjPer; ++n) acc[tok][n] = 0.f;
  }
  float sqr[kTok];
#pragma unroll
  for (int tok = 0; tok < kTok; ++tok) sqr[tok] = 0.f;

  for (int it = 0; it < h_iters; ++it) {
    const int pos = s * h_per_split + it * kSplitThreads + t;
    // the post-mix of every token of the block at this position (mode 1, as the kernel above)
    float new_r[kTok][kHC];
#pragma unroll
    for (int tok = 0; tok < kTok; ++tok) {
      const int i = i0 + tok;
      const bool valid = tok < n_tok;
      const int ic = valid ? i : i0;  // clamp the tail to a valid row; its results are discarded
      const float xf = __bfloat162float(x[static_cast<int64_t>(ic) * H + pos]);
      float rf[kHC];
#pragma unroll
      for (int k = 0; k < kHC; ++k) rf[k] = __bfloat162float(residual_in[(static_cast<int64_t>(ic) * kHC + k) * H + pos]);
      float pm[kHC], cm[kHC * kHC];
#pragma unroll
      for (int c = 0; c < kHC; ++c) pm[c] = __ldg(post_layer_mix + ic * kHC + c);
#pragma unroll
      for (int e = 0; e < kHC * kHC; ++e) cm[e] = __ldg(comb_res_mix + ic * kHC * kHC + e);
#pragma unroll
      for (int j = 0; j < kHC; ++j) new_r[tok][j] = post_mix_value<1>(pm[j], xf, cm, j, rf);
      if (g == 0) {
#pragma unroll
        for (int j = 0; j < kHC; ++j) sqr[tok] = fmaf(new_r[tok][j], new_r[tok][j], sqr[tok]);
      }
    }
    // each fn value once for the whole token block
    const float* wcol = fn + static_cast<int64_t>(n0) * kHC * H + pos;
#pragma unroll
    for (int n = 0; n < kProjPer; ++n) {
#pragma unroll
      for (int j = 0; j < kHC; ++j) {
        const float w = __ldg(wcol + (static_cast<int64_t>(n) * kHC + j) * H);
#pragma unroll
        for (int tok = 0; tok < kTok; ++tok) acc[tok][n] = fmaf(w, new_r[tok][j], acc[tok][n]);
      }
    }
  }
  // tl::warp_reduce_sum: butterfly 16, 8, 4, 2, 1
#pragma unroll
  for (int tok = 0; tok < kTok; ++tok) {
#pragma unroll
    for (int n = 0; n < kProjPer; ++n) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc[tok][n] = acc[tok][n] + __shfl_xor_sync(full, acc[tok][n], off);
    }
    if (g == 0) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) sqr[tok] = sqr[tok] + __shfl_xor_sync(full, sqr[tok], off);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int tok = 0; tok < kTok; ++tok) {
#pragma unroll
      for (int n = 0; n < kProjPer; ++n) s_part[g][warp_in_group][tok][n] = acc[tok][n];
      if (g == 0) s_part[0][warp_in_group][tok][kProjPer] = sqr[tok];
    }
  }
  __syncthreads();
  // cross-warp sum in warp order, one output per lane; lanes 12..15 of group 0 take the sqrsum
  if (warp_in_group == 0) {
    for (int tok = 0; tok < n_tok; ++tok) {
      const int i = i0 + tok;
      if (lane < kProjPer) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) v = v + s_part[g][w][tok][lane];
        mixes[(static_cast<int64_t>(s) * T + i) * kMix + n0 + lane] = v;
      } else if (g == 0 && lane == kProjPer) {
        float v2 = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) v2 = v2 + s_part[0][w][tok][kProjPer];
        sqrsum[static_cast<int64_t>(s) * T + i] = v2;
      }
    }
  }
}

}  // namespace

void mhc_proj(const at::Tensor& x, const at::Tensor& residual_in, const at::Tensor& post_layer_mix,
              const at::Tensor& comb_res_mix, const at::Tensor& fn, at::Tensor mixes, at::Tensor sqrsum, int64_t tok_block) {
  TORCH_CHECK(residual_in.is_cuda() && residual_in.dim() == 3 && residual_in.scalar_type() == at::kBFloat16 &&
                  residual_in.is_contiguous(),
              "residual: bf16 [T, hc, H] contiguous");
  const int64_t T = residual_in.size(0), hc = residual_in.size(1), H = residual_in.size(2);
  TORCH_CHECK(hc == kHC, "hc_mult must be ", kHC);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(0) == T &&
                  x.size(1) == H,
              "x: bf16 [T, H]");
  TORCH_CHECK(post_layer_mix.is_cuda() && post_layer_mix.scalar_type() == at::kFloat && post_layer_mix.is_contiguous() &&
                  post_layer_mix.numel() == T * kHC,
              "post_layer_mix: fp32 [T, hc]");
  TORCH_CHECK(comb_res_mix.is_cuda() && comb_res_mix.scalar_type() == at::kFloat && comb_res_mix.is_contiguous() &&
                  comb_res_mix.numel() == T * kHC * kHC,
              "comb_res_mix: fp32 [T, hc, hc]");
  TORCH_CHECK(fn.is_cuda() && fn.scalar_type() == at::kFloat && fn.is_contiguous() && fn.numel() == kMix * kHC * H,
              "fn: fp32 [24, hc * H]");
  TORCH_CHECK(mixes.is_contiguous() && mixes.dim() == 3 && mixes.size(1) == T && mixes.size(2) == kMix &&
                  mixes.scalar_type() == at::kFloat,
              "mixes: fp32 [S, T, 24]");
  const int64_t S = mixes.size(0);
  TORCH_CHECK(sqrsum.is_contiguous() && sqrsum.numel() == S * T && sqrsum.scalar_type() == at::kFloat, "sqrsum: fp32 [S, T]");
  TORCH_CHECK(S >= 1 && H % (S * kSplitThreads) == 0, "H must split into S x 128-thread slices");
  TORCH_CHECK(tok_block == 1 || tok_block == 2 || tok_block == 4, "tok_block must be 1, 2 or 4");
  if (T == 0) return;
  const c10::cuda::OptionalCUDAGuard guard(residual_in.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(residual_in.get_device()).stream();
  auto launch = [&](auto kernel, int tok) {
    kernel<<<dim3(S, (T + tok - 1) / tok), dim3(kProjThreads), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(residual_in.data_ptr()),
        post_layer_mix.data_ptr<float>(), comb_res_mix.data_ptr<float>(), fn.data_ptr<float>(), mixes.data_ptr<float>(),
        sqrsum.data_ptr<float>(), static_cast<int>(H), static_cast<int>(S), static_cast<int>(T));
  };
  if (tok_block == 1) launch(mhc_proj_kernel<1>, 1);
  else if (tok_block == 2) launch(mhc_proj_kernel<2>, 2);
  else launch(mhc_proj_kernel<4>, 4);
  C10_CUDA_CHECK(cudaGetLastError());
}

void mhc_post_norm(const at::Tensor& x, const at::Tensor& residual_in, const at::Tensor& post_layer_mix,
                   const at::Tensor& comb_res_mix, const at::Tensor& pre_mix_in, const at::Tensor& norm_w,
                   at::Tensor residual_out, at::Tensor layer_input, const c10::optional<at::Tensor>& aux, double norm_eps,
                   int64_t mode, bool pdl) {
  TORCH_CHECK(residual_in.is_cuda() && residual_in.dim() == 3 && residual_in.scalar_type() == at::kBFloat16 &&
                  residual_in.is_contiguous(),
              "residual: bf16 [T, hc, H] contiguous");
  const int64_t T = residual_in.size(0), hc = residual_in.size(1), H = residual_in.size(2);
  TORCH_CHECK(hc == kHC, "hc_mult must be ", kHC);
  TORCH_CHECK(H % kTLBlock == 0 && H <= kMaxH, "H must be a multiple of 1024 up to ", kMaxH);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(0) == T &&
                  x.size(1) == H,
              "x: bf16 [T, H]");
  TORCH_CHECK(post_layer_mix.is_cuda() && post_layer_mix.scalar_type() == at::kFloat && post_layer_mix.is_contiguous() &&
                  post_layer_mix.numel() == T * kHC,
              "post_layer_mix: fp32 [T, hc]");
  TORCH_CHECK(comb_res_mix.is_cuda() && comb_res_mix.scalar_type() == at::kFloat && comb_res_mix.is_contiguous() &&
                  comb_res_mix.numel() == T * kHC * kHC,
              "comb_res_mix: fp32 [T, hc, hc]");
  TORCH_CHECK(pre_mix_in.is_cuda() && pre_mix_in.scalar_type() == at::kFloat && pre_mix_in.is_contiguous() &&
                  pre_mix_in.numel() == T * kHC,
              "pre_mix_in: fp32 [T, hc]");
  TORCH_CHECK(norm_w.is_cuda() && norm_w.scalar_type() == at::kBFloat16 && norm_w.is_contiguous() && norm_w.numel() == H,
              "norm_weight: bf16 [H]");
  TORCH_CHECK(residual_out.is_contiguous() && residual_out.sizes() == residual_in.sizes() &&
                  residual_out.scalar_type() == at::kBFloat16,
              "residual_out: bf16 [T, hc, H]");
  TORCH_CHECK(layer_input.is_contiguous() && layer_input.numel() == T * H && layer_input.scalar_type() == at::kBFloat16,
              "layer_input: bf16 [T, H]");
  if (aux.has_value()) {
    TORCH_CHECK(aux->is_contiguous() && aux->numel() == T * H && aux->scalar_type() == at::kBFloat16, "aux: bf16 [T, H]");
  }
  if (T == 0) return;
  const c10::cuda::OptionalCUDAGuard guard(residual_in.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(residual_in.get_device()).stream();
  const int threads = static_cast<int>(H) / kVec;
  const size_t smem = static_cast<size_t>(H) * sizeof(__nv_bfloat16) + (kTLThreads + 4) * sizeof(float);
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(T);
  config.blockDim = dim3(threads);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attr;
  config.numAttrs = pdl ? 1 : 0;
  auto launch = [&](auto kernel) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config, kernel, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(residual_in.data_ptr()), post_layer_mix.data_ptr<float>(),
        comb_res_mix.data_ptr<float>(), pre_mix_in.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()), reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr()),
        aux.has_value() ? reinterpret_cast<__nv_bfloat16*>(aux->data_ptr()) : nullptr, static_cast<int>(H),
        static_cast<float>(norm_eps)));
  };
  if (pdl) {
    if (mode == 0) launch(mhc_post_norm_kernel<0, true>);
    else if (mode == 1) launch(mhc_post_norm_kernel<1, true>);
    else launch(mhc_post_norm_kernel<2, true>);
  } else {
    if (mode == 0) launch(mhc_post_norm_kernel<0, false>);
    else if (mode == 1) launch(mhc_post_norm_kernel<1, false>);
    else launch(mhc_post_norm_kernel<2, false>);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mhc_post_norm", &mhc_post_norm,
        "DeepSeek-V4.1 shifted mHC critical path: post-mix, bf16 streams, collapse with the carried pre-mix, RMSNorm, aux",
        py::arg("x"), py::arg("residual_in"), py::arg("post_layer_mix"), py::arg("comb_res_mix"), py::arg("pre_mix_in"),
        py::arg("norm_w"), py::arg("residual_out"), py::arg("layer_input"), py::arg("aux"), py::arg("norm_eps"),
        py::arg("mode") = 1, py::arg("pdl") = true);
  m.def("mhc_proj", &mhc_proj,
        "DeepSeek-V4.1 mHC fn projection of the fp32 post-mix, split partials (mhc_fused_tilelang's outputs, one CTA per token x split)",
        py::arg("x"), py::arg("residual_in"), py::arg("post_layer_mix"), py::arg("comb_res_mix"), py::arg("fn"), py::arg("mixes"),
        py::arg("sqrsum"), py::arg("tok_block") = 4);
}
