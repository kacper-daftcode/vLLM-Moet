// Inkling2's relative attention on sm_120 (RTX PRO 6000 / RTX 5090): decode, verify, the DSpark draft's block
// attention and prefill in one kernel.
//
// vLLM computes this attention with `_rel_verify_kernel` (Triton Gluon: tcgen05 MMA into tensor memory, TMA) and with
// tml_fa4's sheared-bias forward - sm_100 kernels both. This file is the same operation on what capability 12 has:
// mma.sync m16n8k16 (bf16 x bf16 -> fp32), ldmatrix, cp.async.
//
//   logits[t, h, k] = scale * (q[t, h] . K[k, h / G]) + bias[t, h, k]        over the keys visible to token t
//   out[t, h]       = softmax(logits[t, h]) @ V[:, h / G]                    lse[t, h] = log(sum(exp(logits[t, h])))
//
// A CTA holds 64 query rows - up to 64 / G tokens of one request times the G heads of one KV head - in four warps of
// 16 rows, and walks the request's keys in steps of 64:
//
//   K and V   the paged cache keeps a key's K and V next to each other; a step's 64 rows of K | V (32 KB) are copied
//             by cp.async into a ring of two stages (page-table lookup per row; 16-byte chunks XOR-swizzled by the
//             row so that ldmatrix reads eight rows without bank conflicts). Rows past the last key are zeros.
//   S = Q K^T Q stays in registers as A fragments for the whole request (read straight from global memory);
//             K comes from the stage by ldmatrix. 64 mma.sync per warp and step.
//   softmax   z = S * scale * log2(e) + bias * log2(e), masked by position (causal; the window in a window layer);
//             the running row maximum and sum in the log2 domain, rescaling O when the maximum moves; P = ex2(z - m)
//             rounded to bf16 as the A fragments of the next product (the C fragment of S has that layout).
//   O += P V  V from the stage by ldmatrix.trans. 64 mma.sync per warp and step.
//
// `bias` is the sheared table the attention prologue writes: for a token, column c holds key c + base, base =
// ceil((past + tile_end) / 128) * 128 - (extent + 256) for the token's 128-token query tile; keys left of the table
// have no relative logit.
//
// Modes (the two entries of vLLM this kernel stands behind):
//   decode   grid (splits, KV heads, requests): split s of a request takes the 128-key blocks s, s + splits, ...
//            (of the window in a window layer) - vLLM's split plan. One split writes the gated bf16 output itself
//            (out * sigmoid(gate) * sigmoid(lse - sink); the draft writes out); more splits write fp32 partials and
//            their log2 LSE, which `combine_kernel` merges and gates. Every query of a draft request sits at the
//            request's last position and has no bias.
//   prefill  grid (ceil(tokens / (64 / G)), KV heads): a CTA takes 64 / G consecutive tokens of the batch and
//            walks the keys once per request it meets among them. fp32 output, natural-log LSE [heads, tokens].
//
// Nothing here reads a length on the host: the grid depends on the batch shape only and every CTA finds its request,
// positions and key range in the device tensors, so a launch is replayed from a CUDA graph.
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <limits>
#include <vector>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace inkling2_attn {

constexpr int kThreads = 128;
constexpr int kD = 128;                    // head dimension
constexpr int kRows = 64;                  // query rows of a CTA
constexpr int kKeys = 64;                  // keys of a step
constexpr int kBlock = 128;                // vLLM's key block (splits) and the sheared table's tile
constexpr int kRowBytes = kD * 2;          // a key's K (or V) in shared memory
constexpr int kTileBytes = kKeys * kRowBytes;
constexpr int kStages = 2;
constexpr int kSmemBytes = kStages * 2 * kTileBytes;   // 64 KB

enum : int { kPart = 0, kFuseGate = 1, kFusePlain = 2, kPrefill = 3 };

struct Args {
  const uint16_t* q;             // [T][heads][128] bf16, tokens q_stride elements apart
  const uint16_t* kv;            // the paged cache: per slot and KV head K (128) | V (128)
  const uint16_t* rel;           // sheared relative logits [T][heads][rel_pad]
  const int32_t* block_table;    // [requests][pages]
  const int32_t* seqlens;        // [requests] keys of the request, this step's tokens included
  const int32_t* cu_q;           // [requests + 1] first token of each request
  float* o_out;                  // partials [splits][T][heads][128], or the prefill output [T][heads][128]
  float* lse_out;                // partials [splits][T][heads] (log2), or the prefill LSE [heads][T] (natural log)
  const uint16_t* gate;          // [T][heads * 128]
  const uint16_t* sink;          // [T][heads]
  uint16_t* y;                   // [T][heads * 128] bf16
  int64_t q_stride, rel_stride, bt_stride, gate_stride, sink_stride, lse_stride, slot_stride;
  int32_t page, page_rows, head_rows, head_cols;
  int32_t rel_pad;               // extent + 256; 0: no relative logits
  int32_t num_tokens, num_reqs, num_splits, heads, group, window, tok;
  int32_t mode, draft;
  float scale_log2;              // softmax scale * log2(e)
};

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
#define NEG_INF (-std::numeric_limits<float>::infinity())

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ float ex2_ftz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float lg2_ftz(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
// vLLM's gate: 1 / (1 + 2^(-x log2 e)) on the non-ftz approximations.
__device__ __forceinline__ float sigmoid(float x) {
  float e, y;
  asm("ex2.approx.f32 %0, %1;\n" : "=f"(e) : "f"(-x * kLog2e));
  asm("rcp.approx.f32 %0, %1;\n" : "=f"(y) : "f"(e + 1.0f));
  return y;
}
// Two fp32 -> one register of two bf16 (round to nearest even), `lo` in the low half.
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ float bf16_to_float(uint32_t bits) { return __uint_as_float(bits << 16); }
__device__ __forceinline__ uint32_t load32(const uint16_t* p) { return *reinterpret_cast<const uint32_t*>(p); }

__global__ void __launch_bounds__(kThreads, 1) attn_kernel(const Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int q8 = lane >> 2, c = lane & 3;       // the thread's place in an MMA fragment: row q8 (and q8 + 8), column pair c
  const int mi = lane & 7, mm = lane >> 3;      // its row and matrix in an ldmatrix.x4
  const int kv_head = blockIdx.y;
  const int G = a.group;

  // The tokens of this CTA: one request's (decode), or `tok` consecutive tokens of the batch (prefill).
  int t, t_end, req, split = 0;
  if (a.mode == kPrefill) {
    t = blockIdx.x * a.tok;
    t_end = min(t + a.tok, a.num_tokens);
    int lo = 0, hi = a.num_reqs;                // the last request that starts at or before token t
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (a.cu_q[mid] <= t) lo = mid; else hi = mid;
    }
    req = lo;
  } else {
    split = blockIdx.x;
    req = blockIdx.z;
    t = a.cu_q[req];
    t_end = t + min(a.cu_q[req + 1] - t, a.tok);
  }

  while (t < t_end) {
    int q_start = a.cu_q[req], q_next = a.cu_q[req + 1];
    if (a.mode == kPrefill) {
      while (q_next <= t && req + 1 < a.num_reqs) {
        ++req;
        q_start = q_next;
        q_next = a.cu_q[req + 1];
      }
      if (q_next <= t) break;                   // tokens past the last request
    }
    const int q_len = q_next - q_start;
    const int n_tok = min(t_end, q_next) - t;
    const int seqlen = a.seqlens[req];
    const int past = seqlen - q_len;
    const int q_idx0 = t - q_start;

    // The thread's two rows: row = token * G + head of the KV head's group.
    bool valid[2];
    int token[2], head[2], kmax[2], kmin[2], rel_lo[2];
    const uint16_t* relp[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = warp * 16 + q8 + 8 * r;
      const int tk = row / G;
      valid[r] = tk < n_tok;
      token[r] = t + tk;
      head[r] = kv_head * G + (row - tk * G);
      const int pos = a.draft ? seqlen - 1 : past + q_idx0 + tk;
      kmax[r] = valid[r] ? pos : -1;
      kmin[r] = a.window ? pos - a.window + 1 : std::numeric_limits<int>::min();
      rel_lo[r] = 0;
      relp[r] = nullptr;
      if (a.rel_pad) {
        const int qi = q_idx0 + tk;
        const int tile_end = min((qi / kBlock + 1) * kBlock, q_len);
        rel_lo[r] = (past + tile_end + kBlock - 1) / kBlock * kBlock - a.rel_pad;
        relp[r] = a.rel + (int64_t)token[r] * a.rel_stride + (int64_t)head[r] * a.rel_pad - rel_lo[r];
      }
    }

    // Q as A fragments: k step s covers dimensions 16 s .. 16 s + 15.
    uint32_t qa[8][4];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint16_t* qp = a.q + (int64_t)token[r] * a.q_stride + head[r] * kD + 2 * c;
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        qa[s][r] = valid[r] ? load32(qp + 16 * s) : 0u;
        qa[s][2 + r] = valid[r] ? load32(qp + 16 * s + 8) : 0u;
      }
    }

    // The keys of this CTA: 128-key blocks n_first, n_first + splits, ..., each in two steps of 64.
    const int pos0 = a.draft ? seqlen - 1 : past + q_idx0;
    const int key_lo = a.window ? max(pos0 - (a.window - 1), 0) : 0;
    const int key_hi = a.draft ? seqlen : past + q_idx0 + n_tok;
    const int n_first = split + key_lo / kBlock;
    const int iters = max(((key_hi + kBlock - 1) / kBlock - n_first + a.num_splits - 1) / a.num_splits, 0);
    int ib = 0, nb = 2 * iters;
    if (iters > 0) {
      if (n_first * kBlock + kKeys <= key_lo) ib = 1;
      if ((n_first + (iters - 1) * a.num_splits) * kBlock + kKeys >= key_hi) nb -= 1;
    }
    auto step_key = [&](int i) { return (n_first + (i >> 1) * a.num_splits) * kBlock + (i & 1) * kKeys; };

    // Two threads copy one key row of a step: K (thread 2 r) and V (thread 2 r + 1), sixteen 16-byte chunks each.
    auto issue = [&](int i, int stage) {
      const int kr = tid >> 1, half = tid & 1;
      const int key = step_key(i) + kr;
      const int off = stage * 2 * kTileBytes + half * kTileBytes + kr * kRowBytes;
      if (key < key_hi) {
        const int blk = a.block_table[req * a.bt_stride + key / a.page];
        const uint16_t* src = a.kv + ((int64_t)blk * a.page_rows + kv_head * a.head_rows + key % a.page) * a.slot_stride +
                              kv_head * a.head_cols + half * kD;
#pragma unroll
        for (int ch = 0; ch < 16; ++ch) cp_async16(smem_base + off + ((ch ^ (kr & 7)) << 4), src + ch * 8);
      } else {
        uint64_t* dst = reinterpret_cast<uint64_t*>(smem + off);
#pragma unroll
        for (int w = 0; w < kRowBytes / 8; ++w) dst[w] = 0;
      }
    };

    float o[16][4];
#pragma unroll
    for (int u = 0; u < 16; ++u) o[u][0] = o[u][1] = o[u][2] = o[u][3] = 0.0f;
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.0f, 0.0f};

    if (ib < nb) issue(ib, 0);
    cp_async_commit();
    if (ib + 1 < nb) issue(ib + 1, 1);
    cp_async_commit();

    for (int i = ib; i < nb; ++i) {
      const int stage = (i - ib) & 1;
      const int key0 = step_key(i);
      const uint32_t k_tile = smem_base + stage * 2 * kTileBytes;
      const uint32_t v_tile = k_tile + kTileBytes;

      // The step's relative logits for this thread's elements, read while the copies land.
      uint32_t bias[8][2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const bool has = a.rel_pad && valid[r] && key0 >= rel_lo[r];
#pragma unroll
        for (int j = 0; j < 8; ++j) bias[j][r] = has ? load32(relp[r] + key0 + 8 * j + 2 * c) : 0u;
      }

      cp_async_wait<1>();
      __syncthreads();

      // S = Q K^T: n tile j is keys 8 j .. 8 j + 7.
      float s[8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
          uint32_t b[4];
          ldmatrix_x4(b, k_tile + (16 * jp + 8 * (mm >> 1) + mi) * kRowBytes + (((2 * ks + (mm & 1)) ^ mi) << 4));
          mma_bf16(s[2 * jp], qa[ks], b[0], b[1]);
          mma_bf16(s[2 * jp + 1], qa[ks], b[2], b[3]);
        }
      }

      // Logits in the log2 domain, masked by position; the rows' maxima over their four threads.
      float zmax[2] = {NEG_INF, NEG_INF};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int k = key0 + 8 * j + 2 * c + e;
            const float b = bf16_to_float(e ? bias[j][r] >> 16 : bias[j][r] & 0xffffu) * kLog2e;
            float z = fmaf(s[j][2 * r + e], a.scale_log2, b);
            z = (k <= kmax[r] && k >= kmin[r]) ? z : NEG_INF;
            s[j][2 * r + e] = z;
            zmax[r] = fmaxf(zmax[r], z);
          }
        }
      }
      float alpha[2], msafe[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        zmax[r] = fmaxf(zmax[r], __shfl_xor_sync(0xffffffffu, zmax[r], 1));
        zmax[r] = fmaxf(zmax[r], __shfl_xor_sync(0xffffffffu, zmax[r], 2));
        const float mn = fmaxf(m[r], zmax[r]);
        msafe[r] = mn == NEG_INF ? 0.0f : mn;
        alpha[r] = mn == m[r] ? 1.0f : ex2_ftz(m[r] - msafe[r]);
        m[r] = mn;
      }
      float psum[2] = {0.0f, 0.0f};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float p = ex2_ftz(s[j][2 * r + e] - msafe[r]);
            s[j][2 * r + e] = p;
            psum[r] += p;
          }
        }
      }
      l[0] = l[0] * alpha[0] + psum[0];
      l[1] = l[1] * alpha[1] + psum[1];
      if (alpha[0] != 1.0f || alpha[1] != 1.0f) {
#pragma unroll
        for (int u = 0; u < 16; ++u) {
          o[u][0] *= alpha[0];
          o[u][1] *= alpha[0];
          o[u][2] *= alpha[1];
          o[u][3] *= alpha[1];
        }
      }

      // O += P V: P's A fragment for keys 16 t .. 16 t + 15 is S's C fragments of n tiles 2 t and 2 t + 1.
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) {
        uint32_t pa[4];
        pa[0] = pack_bf16(s[2 * kt][0], s[2 * kt][1]);
        pa[1] = pack_bf16(s[2 * kt][2], s[2 * kt][3]);
        pa[2] = pack_bf16(s[2 * kt + 1][0], s[2 * kt + 1][1]);
        pa[3] = pack_bf16(s[2 * kt + 1][2], s[2 * kt + 1][3]);
#pragma unroll
        for (int up = 0; up < 8; ++up) {
          uint32_t b[4];
          ldmatrix_x4_trans(b, v_tile + (16 * kt + 8 * (mm & 1) + mi) * kRowBytes + (((2 * up + (mm >> 1)) ^ mi) << 4));
          mma_bf16(o[2 * up], pa, b[0], b[1]);
          mma_bf16(o[2 * up + 1], pa, b[2], b[3]);
        }
      }

      __syncthreads();
      if (i + 2 < nb) issue(i + 2, stage);
      cp_async_commit();
    }
    cp_async_wait<0>();
    __syncthreads();

    // Epilogue: the rows' sums over their four threads, then the mode's output.
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
      l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
      if (!valid[r]) continue;
      const bool empty = !(l[r] > 0.0f);
      const float lse2 = empty ? NEG_INF : m[r] + lg2_ftz(l[r]);
      const float inv = empty ? 0.0f : 1.0f / l[r];
      if (a.mode == kPart || a.mode == kPrefill) {
        const int64_t row = a.mode == kPart ? ((int64_t)split * a.num_tokens + token[r]) * a.heads + head[r]
                                            : (int64_t)token[r] * a.heads + head[r];
        float2* op = reinterpret_cast<float2*>(a.o_out + row * kD + 2 * c);
#pragma unroll
        for (int u = 0; u < 16; ++u) op[4 * u] = make_float2(o[u][2 * r] * inv, o[u][2 * r + 1] * inv);
        if (c == 0) {
          if (a.mode == kPart) a.lse_out[row] = lse2;
          else a.lse_out[(int64_t)head[r] * a.lse_stride + token[r]] = lse2 * kLn2;
        }
      } else {
        uint32_t* yp = reinterpret_cast<uint32_t*>(a.y + ((int64_t)token[r] * a.heads + head[r]) * kD + 2 * c);
        if (a.mode == kFuseGate) {
          const float sink = bf16_to_float(a.sink[(int64_t)token[r] * a.sink_stride + head[r]]);
          const float scale = inv * sigmoid(lse2 * kLn2 - sink);
          const uint16_t* gp = a.gate + (int64_t)token[r] * a.gate_stride + head[r] * kD + 2 * c;
#pragma unroll
          for (int u = 0; u < 16; ++u) {
            const uint32_t g = load32(gp + 8 * u);
            yp[4 * u] = pack_bf16(o[u][2 * r] * scale * sigmoid(bf16_to_float(g & 0xffffu)),
                                  o[u][2 * r + 1] * scale * sigmoid(bf16_to_float(g >> 16)));
          }
        } else {
#pragma unroll
          for (int u = 0; u < 16; ++u) yp[4 * u] = pack_bf16(o[u][2 * r] * inv, o[u][2 * r + 1] * inv);
        }
      }
    }
    t += n_tok;
  }
}

// The merge of a token and head's split partials (weights 2^(lse_s - max)), then the gate and the sink. One CTA per
// (head, token), one thread per dimension. Tokens from `num_actual` on are zeros.
__global__ void __launch_bounds__(kD) combine_kernel(const float* o_part, const float* lse_part, const uint16_t* gate,
                                                     const uint16_t* sink, uint16_t* y, int64_t gate_stride,
                                                     int64_t sink_stride, int num_tokens, int num_actual, int heads,
                                                     int num_splits, int gated) {
  const int head = blockIdx.x, token = blockIdx.y, d = threadIdx.x;
  uint16_t* yp = y + ((int64_t)token * heads + head) * kD + d;
  if (token >= num_actual) {
    *yp = 0;
    return;
  }
  float lse_max = NEG_INF;
  for (int s = 0; s < num_splits; ++s) lse_max = fmaxf(lse_max, lse_part[((int64_t)s * num_tokens + token) * heads + head]);
  if (lse_max == NEG_INF) lse_max = 0.0f;
  float w_sum = 0.0f, acc = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const int64_t row = ((int64_t)s * num_tokens + token) * heads + head;
    const float w = ex2_ftz(lse_part[row] - lse_max);
    w_sum += w;
    if (w > 0.0f) acc += o_part[row * kD + d] * w;
  }
  float out = acc / (w_sum == 0.0f ? 1.0f : w_sum);
  if (gated) {
    const float lse = (lg2_ftz(w_sum) + lse_max) * kLn2;
    out *= sigmoid(bf16_to_float(gate[(int64_t)token * gate_stride + head * kD + d]));
    out *= sigmoid(lse - bf16_to_float(sink[(int64_t)token * sink_stride + head]));
  }
  *yp = static_cast<uint16_t>(pack_bf16(out, 0.0f) & 0xffffu);
}

}  // namespace inkling2_attn

namespace {

using inkling2_attn::Args;

const uint16_t* u16(const torch::Tensor& t) { return reinterpret_cast<const uint16_t*>(t.data_ptr()); }

// tensors: q, key cache, relative logits, block table, seqlens, cu_seqlens_q, o_out, lse_out, gate, sink, y
// p: mode, draft, q_stride, slot_stride, page, page_rows, head_rows, head_cols, rel_stride, rel_pad, bt_stride,
//    gate_stride, sink_stride, lse_stride, num_tokens, num_reqs, num_splits, heads, kv_heads, window
void attn_run(std::vector<torch::Tensor> t, std::vector<int64_t> p, double softmax_scale) {
  TORCH_CHECK(t.size() == 11 && p.size() == 20, "attn_run: 11 tensors and 20 integers");
  for (const auto& x : t) TORCH_CHECK(x.is_cuda(), "attn_run: device tensors");
  Args a{};
  a.q = u16(t[0]);
  a.kv = u16(t[1]);
  a.rel = u16(t[2]);
  a.block_table = reinterpret_cast<const int32_t*>(t[3].data_ptr());
  a.seqlens = reinterpret_cast<const int32_t*>(t[4].data_ptr());
  a.cu_q = reinterpret_cast<const int32_t*>(t[5].data_ptr());
  a.o_out = reinterpret_cast<float*>(t[6].data_ptr());
  a.lse_out = reinterpret_cast<float*>(t[7].data_ptr());
  a.gate = u16(t[8]);
  a.sink = u16(t[9]);
  a.y = reinterpret_cast<uint16_t*>(t[10].data_ptr());
  a.mode = (int)p[0];
  a.draft = (int)p[1];
  a.q_stride = p[2];
  a.slot_stride = p[3];
  a.page = (int)p[4];
  a.page_rows = (int)p[5];
  a.head_rows = (int)p[6];
  a.head_cols = (int)p[7];
  a.rel_stride = p[8];
  a.rel_pad = (int)p[9];
  a.bt_stride = p[10];
  a.gate_stride = p[11];
  a.sink_stride = p[12];
  a.lse_stride = p[13];
  a.num_tokens = (int)p[14];
  a.num_reqs = (int)p[15];
  a.num_splits = (int)p[16];
  a.heads = (int)p[17];
  const int kv_heads = (int)p[18];
  a.window = (int)p[19];
  TORCH_CHECK(a.mode >= 0 && a.mode <= 3 && kv_heads > 0 && a.heads % kv_heads == 0, "mode / heads");
  a.group = a.heads / kv_heads;
  TORCH_CHECK(a.group <= inkling2_attn::kRows, "GQA group ", a.group);
  a.tok = inkling2_attn::kRows / a.group;
  a.scale_log2 = (float)(softmax_scale * 1.4426950408889634);
  TORCH_CHECK(t[3].scalar_type() == torch::kInt32 && t[4].scalar_type() == torch::kInt32 && t[5].scalar_type() == torch::kInt32);
  TORCH_CHECK(a.page > 0 && a.num_splits >= 1 && a.num_reqs >= 0 && a.num_tokens >= 0 && a.window >= 0);
  TORCH_CHECK(t[5].numel() >= a.num_reqs + 1 && t[4].numel() >= a.num_reqs, "request tensors");
  if (a.num_tokens == 0 || a.num_reqs == 0) return;
  static bool configured = false;
  if (!configured) {
    TORCH_CHECK(cudaFuncSetAttribute(inkling2_attn::attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     inkling2_attn::kSmemBytes) == cudaSuccess);
    configured = true;
  }
  dim3 grid;
  if (a.mode == inkling2_attn::kPrefill) grid = dim3((a.num_tokens + a.tok - 1) / a.tok, kv_heads, 1);
  else grid = dim3(a.num_splits, kv_heads, a.num_reqs);
  inkling2_attn::attn_kernel<<<grid, inkling2_attn::kThreads, inkling2_attn::kSmemBytes, c10::cuda::getCurrentCUDAStream()>>>(a);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "attention launch: ", cudaGetErrorString(err));
}

// p: num_tokens, num_actual, heads, num_splits, gated, gate_stride, sink_stride
void combine_run(torch::Tensor o_part, torch::Tensor lse_part, torch::Tensor gate, torch::Tensor sink, torch::Tensor y,
                 std::vector<int64_t> p) {
  TORCH_CHECK(p.size() == 7, "combine_run: 7 integers");
  const int num_tokens = (int)p[0], heads = (int)p[2];
  if (num_tokens == 0) return;
  inkling2_attn::combine_kernel<<<dim3(heads, num_tokens, 1), inkling2_attn::kD, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const float*>(o_part.data_ptr()), reinterpret_cast<const float*>(lse_part.data_ptr()), u16(gate),
      u16(sink), reinterpret_cast<uint16_t*>(y.data_ptr()), p[5], p[6], num_tokens, (int)p[1], heads, (int)p[3], (int)p[4]);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "attention combine launch: ", cudaGetErrorString(err));
}

// (shared memory bytes, registers per thread, CTAs per SM) of the attention kernel on the current device.
std::vector<int64_t> kernel_facts() {
  cudaFuncAttributes attr;
  TORCH_CHECK(cudaFuncGetAttributes(&attr, inkling2_attn::attn_kernel) == cudaSuccess);
  cudaFuncSetAttribute(inkling2_attn::attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, inkling2_attn::kSmemBytes);
  int resident = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, inkling2_attn::attn_kernel, inkling2_attn::kThreads,
                                                inkling2_attn::kSmemBytes);
  return {inkling2_attn::kSmemBytes, attr.numRegs, resident};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_run", &attn_run);
  m.def("combine_run", &combine_run);
  m.def("kernel_facts", &kernel_facts);
}
