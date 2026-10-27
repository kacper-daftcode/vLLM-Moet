// B2 v3 (sm120-perf-32, session A): one-shot push all-reduce over PCIe P2P for decode-sized [M, 5120] bf16, TP4.
//
// Platform rule (measured, p2p_bw_all.py): one GPU must write to ONE peer at a time - a kernel writing to several peers
// concurrently is ~16x slower. push_phased therefore streams to peer 0, then peer 1, then peer 2, inside one launch:
// CTAs are partitioned by destination and a destination's CTAs wait (spin on a local counter) until the previous
// destination's CTAs have issued their stores. Per CTA a gpu-scope fence orders the stores before the counter; the last
// CTA of each phase does one system fence + st.release.sys of the peer's flag (one PCIe round trip per phase).
//
// Sequence numbers: a per-rank device counter of completed all-reduces (bumped by the consumer's last CTA). Every rank
// runs the same all-reduces in the same order, so counters agree; slot = seq % NSLOTS. No per-graph bookkeeping: works
// across target graphs, drafter graphs and eager calls alike.
//
// Layout per rank: data [NSLOTS][4 src][M_MAX][H] bf16, flags [NSLOTS][4 src][32] u32 (128 B per flag).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <vector>

// hidden size of the message rows: 5120 (DeepSeek-V4.1-Flash) unless the build passes -DP2P_AR_H=<width> (4096 for
// GLM-5.3-Flash; p2p_allreduce_sm120.py: VLLM_MOET_P2P_AR_H). A plain -DH=... on the command line would clobber the
// template parameter `H` of torch's headers, hence the indirection; H itself is defined after the includes.
#ifdef P2P_AR_H
#define H P2P_AR_H
#else
#define H 5120
#endif
#define NSRC 4
#define NPEER 3
#define NSLOTS 4
#define FLAG_STRIDE 32
#define M_MAX_G 64

struct Peers3 {
  __nv_bfloat16* data[NPEER];
  unsigned* flags[NPEER];
};

__device__ __forceinline__ void st_release_sys_u32(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_sys_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_acquire_gpu_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_volatile_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// NCCL's summation order, measured at start-up (p2p_allreduce_sm120.py: nccl_order_tables): one byte per 8
// consecutive elements of the [M, H] message, r0 | r1 << 2 | r2 << 4 | r3 << 6 for ((x_r0 + x_r1) + x_r2) + x_r3 with
// one bf16 rounding per add, as NCCL's reduce kernels add (__hadd2). src(r) points at rank r's 8 elements.
template <typename Src>
__device__ __forceinline__ uint4 nccl_order_sum8(unsigned order, Src src) {
  uint4 v[NSRC];
#pragma unroll
  for (int p = 0; p < NSRC; ++p) v[p] = __ldcg(src((order >> (2 * p)) & 3u));
  uint4 acc;
  const __nv_bfloat162* a[NSRC];
#pragma unroll
  for (int p = 0; p < NSRC; ++p) a[p] = reinterpret_cast<const __nv_bfloat162*>(&v[p]);
  __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(&acc);
#pragma unroll
  for (int k = 0; k < 4; ++k) o[k] = __hadd2(__hadd2(__hadd2(a[0][k], a[1][k]), a[2][k]), a[3][k]);
  return acc;
}

// ctrl: [0] seq counter (completed all-reduces), [1..3] phase counters, [4] consumer done counter
__global__ void push_phased_kernel(const __nv_bfloat16* __restrict__ x, int M, int M_MAX, int rank, Peers3 peers,
                                   unsigned* __restrict__ ctrl, int ctas_per_dest) {
  const unsigned seq = ld_volatile_u32(ctrl) + 1u;
  const int slot = seq % NSLOTS;
  const int p = blockIdx.x / ctas_per_dest;
  const int b = blockIdx.x - p * ctas_per_dest;
  unsigned* phase = ctrl + 1;
  if (p > 0) {
    if (threadIdx.x == 0) {
      while (ld_acquire_gpu_u32(phase + p - 1) < (unsigned)ctas_per_dest) {}
    }
    __syncthreads();
  }
  const int chunks = M * (H / 8);
  const uint4* src = reinterpret_cast<const uint4*>(x);
  uint4* dst = reinterpret_cast<uint4*>(peers.data[p]) + (size_t)(slot * NSRC + rank) * M_MAX * (H / 8);
  for (int i = b * blockDim.x + threadIdx.x; i < chunks; i += ctas_per_dest * blockDim.x) dst[i] = src[i];
  __threadfence();  // gpu scope: this thread's P2P stores are ordered before the phase counter
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned done = atomicAdd(phase + p, 1u);
    if (done == (unsigned)ctas_per_dest - 1) {  // last CTA of this destination
      __threadfence_system();
      st_release_sys_u32(peers.flags[p] + (size_t)(slot * NSRC + rank) * FLAG_STRIDE, seq);
      if (p == NPEER - 1) {  // every CTA of every phase has passed its spin by now: reset for the next all-reduce
        phase[0] = 0u; phase[1] = 0u; phase[2] = 0u;
      }
    }
  }
}

// grid: M CTAs (one row each), block: H/16 = 320 threads, 16 elements per thread.
// Sums x_local (own contribution, read from the producer's output) and the 3 peers' slots in NCCL's order.
template <bool NORM>
__global__ void wait_reduce_kernel(const __nv_bfloat16* __restrict__ x_local, int rank, int M_MAX,
                                   const __nv_bfloat16* __restrict__ rbuf, const unsigned* __restrict__ flags,
                                   unsigned* __restrict__ ctrl, const uint8_t* __restrict__ order,
                                   const __nv_bfloat16* __restrict__ residual,
                                   const __nv_bfloat16* __restrict__ weight, float eps, __nv_bfloat16* __restrict__ out,
                                   __nv_bfloat16* __restrict__ residual_out, unsigned* __restrict__ timeout_flag) {
  __shared__ float red[32];
  const unsigned seq = ld_volatile_u32(ctrl) + 1u;
  const int slot = seq % NSLOTS;
  const int m = blockIdx.x;
  if (threadIdx.x < NSRC && threadIdx.x != rank) {
    const unsigned* f = flags + (size_t)(slot * NSRC + threadIdx.x) * FLAG_STRIDE;
    unsigned spins = 0;
    while ((int)(ld_acquire_sys_u32(f) - seq) < 0) {
      if (++spins > (1u << 27)) { atomicOr(timeout_flag, 1u << threadIdx.x); break; }
    }
  }
  __syncthreads();
  __threadfence();
  const int c = threadIdx.x * 16;
  const size_t e0 = (size_t)m * H + c;
  uint4 sum[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    sum[h] = nccl_order_sum8(order[(e0 >> 3) + h], [&](unsigned s) {
      return reinterpret_cast<const uint4*>((s == (unsigned)rank)
                                                ? x_local + e0 + 8 * h
                                                : rbuf + ((size_t)(slot * NSRC + s) * M_MAX + m) * H + c + 8 * h);
    });
  }
  const __nv_bfloat16* acc = reinterpret_cast<const __nv_bfloat16*>(sum);
  if (!NORM) {
    reinterpret_cast<uint4*>(out + e0)[0] = sum[0];
    reinterpret_cast<uint4*>(out + e0)[1] = sum[1];
  } else {
    // all-reduce result (NCCL's bits), + residual in fp32, residual stored as bf16, RMSNorm over it
    float z[16]; float ss = 0.f;
    const __nv_bfloat16* r = residual + (size_t)m * H + c;
    const uint4 a = reinterpret_cast<const uint4*>(r)[0], bq = reinterpret_cast<const uint4*>(r)[1];
    const __nv_bfloat16* pa = reinterpret_cast<const __nv_bfloat16*>(&a);
    const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&bq);
    __nv_bfloat16 ro[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const float v = __bfloat162float(acc[k]) + __bfloat162float(k < 8 ? pa[k] : pb[k - 8]);
      ro[k] = __float2bfloat16(v);
      z[k] = __bfloat162float(ro[k]);
      ss += z[k] * z[k];
    }
    reinterpret_cast<uint4*>(residual_out + (size_t)m * H + c)[0] = reinterpret_cast<uint4*>(ro)[0];
    reinterpret_cast<uint4*>(residual_out + (size_t)m * H + c)[1] = reinterpret_cast<uint4*>(ro)[1];
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
      float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
#pragma unroll
      for (int o2 = 16; o2 > 0; o2 >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o2);
      if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float rstd = rsqrtf(red[0] / (float)H + eps);
    __nv_bfloat16 o[16];
    const __nv_bfloat16* w = weight + c;
#pragma unroll
    for (int k = 0; k < 16; ++k) o[k] = __float2bfloat16(z[k] * rstd * __bfloat162float(w[k]));
    reinterpret_cast<uint4*>(out + (size_t)m * H + c)[0] = reinterpret_cast<uint4*>(o)[0];
    reinterpret_cast<uint4*>(out + (size_t)m * H + c)[1] = reinterpret_cast<uint4*>(o)[1];
  }
  // the last CTA to finish bumps the completed-all-reduce counter (every CTA has read it by then)
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned d = atomicAdd(ctrl + 4, 1u);
    if (d == gridDim.x - 1) { ctrl[4] = 0u; atomicAdd(ctrl, 1u); }
  }
}


// ---- all-gather (bit-exact): the same phased push, generic row width (16-B chunks), consumer concatenates the 4 sources ----
#define CH_MAX 4096  // 64 KB rows: the lm_head logits [rows, 32320] bf16 = 4040 chunks per rank

__global__ void push_phased_gather_kernel(const uint4* __restrict__ x, int M, int row_chunks, int rank, Peers3 peers,
                                          unsigned* __restrict__ ctrl, int ctas_per_dest) {
  const unsigned seq = ld_volatile_u32(ctrl) + 1u;
  const int slot = seq % NSLOTS;
  const int p = blockIdx.x / ctas_per_dest;
  const int b = blockIdx.x - p * ctas_per_dest;
  unsigned* phase = ctrl + 1;
  if (p > 0) {
    if (threadIdx.x == 0) {
      while (ld_acquire_gpu_u32(phase + p - 1) < (unsigned)ctas_per_dest) {}
    }
    __syncthreads();
  }
  const int chunks = M * row_chunks;
  uint4* dst = reinterpret_cast<uint4*>(peers.data[p]) + (size_t)(slot * NSRC + rank) * M_MAX_G * CH_MAX;
  for (int i = b * blockDim.x + threadIdx.x; i < chunks; i += ctas_per_dest * blockDim.x) {
    const int m = i / row_chunks, c = i - m * row_chunks;
    dst[(size_t)m * CH_MAX + c] = x[i];
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned done = atomicAdd(phase + p, 1u);
    if (done == (unsigned)ctas_per_dest - 1) {
      __threadfence_system();
      st_release_sys_u32(peers.flags[p] + (size_t)(slot * NSRC + rank) * FLAG_STRIDE, seq);
      if (p == NPEER - 1) { phase[0] = 0u; phase[1] = 0u; phase[2] = 0u; }
    }
  }
}

// out[m][s*row_chunks + c] = (s == rank) ? x[m][c] : slot[s][m][c];  grid = M CTAs x 256 threads
__global__ void wait_gather_kernel(const uint4* __restrict__ x_local, int rank, int M, int row_chunks,
                                   const uint4* __restrict__ rbuf, const unsigned* __restrict__ flags,
                                   unsigned* __restrict__ ctrl, uint4* __restrict__ out,
                                   unsigned* __restrict__ timeout_flag) {
  const unsigned seq = ld_volatile_u32(ctrl) + 1u;
  const int slot = seq % NSLOTS;
  const int m = blockIdx.x;
  if (threadIdx.x < NSRC && threadIdx.x != rank) {
    const unsigned* f = flags + (size_t)(slot * NSRC + threadIdx.x) * FLAG_STRIDE;
    unsigned spins = 0;
    while ((int)(ld_acquire_sys_u32(f) - seq) < 0) {
      if (++spins > (1u << 27)) { atomicOr(timeout_flag, 1u << threadIdx.x); break; }
    }
  }
  __syncthreads();
  __threadfence();
#pragma unroll
  for (int s = 0; s < NSRC; ++s) {
    const uint4* src = (s == rank) ? x_local + (size_t)m * row_chunks
                                   : rbuf + ((size_t)(slot * NSRC + s) * M_MAX_G + m) * CH_MAX;
    uint4* o = out + (size_t)m * (NSRC * row_chunks) + (size_t)s * row_chunks;
    for (int c = threadIdx.x; c < row_chunks; c += blockDim.x) o[c] = (s == rank) ? src[c] : __ldcg(src + c);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned d = atomicAdd(ctrl + 4, 1u);
    if (d == gridDim.x - 1) { ctrl[4] = 0u; atomicAdd(ctrl, 1u); }
  }
}


// ---- two-shot all-reduce (reduce-scatter + all-gather) for the larger decode batches (C8: [48, 5120]) -----------------
// Phase p of a push goes to rank (rank + 1 + p) % 4, so every GPU writes to one peer at a time (platform rule) and every
// peer receives from one source at a time. Slice j = columns [j*1280, (j+1)*1280) = 160 x 16 B per row. Buffers: a second
// set (rbuf2 / flags2 / ctrl2) so a one-shot all-reduce of another shape in the same step cannot alias a slot.
//   RS: rank r pushes x[:, slice d] to each peer d (flag seq_rs = 2n+1); rank j sums slice j over the 4 sources in fixed
//       order (own rows from x) and writes it into out[:, slice j].
//   AG: rank j pushes out[:, slice j] to every peer (flag seq_ag = 2n+2); each rank copies the 3 received slices into out.
// On rank p the RS data of source r lands in [src=r][slice p] and the AG data of source j in [src=j][slice j]: disjoint.
#define SLICE_CH (H / 8 / NSRC)  // 160 chunks of 16 B

__device__ __forceinline__ int peer_index(int rank, int dest) { return dest - (dest > rank ? 1 : 0); }

// mode 0: RS (source = x[:, slice dest], dest slice region), mode 1: AG (source = out[:, slice rank], region slice rank)
template <int MODE>
__global__ void push_twoshot_kernel(const uint4* __restrict__ src_rows, int M, int M_MAX, int rank, Peers3 peers,
                                    unsigned* __restrict__ ctrl, int ctas_per_dest) {
  const unsigned n = ld_volatile_u32(ctrl);
  const unsigned seq = 2u * n + (MODE == 0 ? 1u : 2u);
  const int slot = n % NSLOTS;
  const int p = blockIdx.x / ctas_per_dest;
  const int b = blockIdx.x - p * ctas_per_dest;
  const int d = (rank + 1 + p) & 3;
  const int pidx = peer_index(rank, d);
  unsigned* phase = ctrl + 1;
  if (p > 0) {
    if (threadIdx.x == 0) {
      while (ld_acquire_gpu_u32(phase + p - 1) < (unsigned)ctas_per_dest) {}
    }
    __syncthreads();
  }
  const int slice = (MODE == 0) ? d : rank;
  const int chunks = M * SLICE_CH;
  uint4* dst = reinterpret_cast<uint4*>(peers.data[pidx]) + (size_t)(slot * NSRC + rank) * M_MAX * (H / 8) + slice * SLICE_CH;
  for (int i = b * blockDim.x + threadIdx.x; i < chunks; i += ctas_per_dest * blockDim.x) {
    const int m = i / SLICE_CH, c = i - m * SLICE_CH;
    dst[(size_t)m * (H / 8) + c] = src_rows[(size_t)m * (H / 8) + slice * SLICE_CH + c];
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned done = atomicAdd(phase + p, 1u);
    if (done == (unsigned)ctas_per_dest - 1) {
      __threadfence_system();
      st_release_sys_u32(peers.flags[pidx] + (size_t)(slot * NSRC + rank) * FLAG_STRIDE, seq);
      if (p == NPEER - 1) { phase[0] = 0u; phase[1] = 0u; phase[2] = 0u; }
    }
  }
}

// grid: M CTAs, block: SLICE_CH threads (one 16-B chunk = 8 bf16 each). out[:, slice rank] = sum over sources in NCCL's
// order.
__global__ void reduce_slice_kernel(const __nv_bfloat16* __restrict__ x_local, int rank, int M_MAX,
                                    const __nv_bfloat16* __restrict__ rbuf, const unsigned* __restrict__ flags,
                                    const unsigned* __restrict__ ctrl, const uint8_t* __restrict__ order,
                                    __nv_bfloat16* __restrict__ out, unsigned* __restrict__ timeout_flag) {
  const unsigned n = ld_volatile_u32(ctrl);
  const unsigned seq = 2u * n + 1u;
  const int slot = n % NSLOTS;
  const int m = blockIdx.x;
  if (threadIdx.x < NSRC && threadIdx.x != rank) {
    const unsigned* f = flags + (size_t)(slot * NSRC + threadIdx.x) * FLAG_STRIDE;
    unsigned spins = 0;
    while ((int)(ld_acquire_sys_u32(f) - seq) < 0) {
      if (++spins > (1u << 27)) { atomicOr(timeout_flag, 1u << threadIdx.x); break; }
    }
  }
  __syncthreads();
  __threadfence();
  const int c = (rank * SLICE_CH + threadIdx.x) * 8;  // element column
  const size_t e0 = (size_t)m * H + c;
  *reinterpret_cast<uint4*>(out + e0) = nccl_order_sum8(order[e0 >> 3], [&](unsigned s) {
    return reinterpret_cast<const uint4*>((s == (unsigned)rank) ? x_local + e0
                                                                : rbuf + ((size_t)(slot * NSRC + s) * M_MAX + m) * H + c);
  });
}

// grid: M CTAs, block: SLICE_CH threads. out[:, slice s] = received slice s for s != rank; the last CTA bumps ctrl[0].
__global__ void assemble_kernel(int rank, int M_MAX, const __nv_bfloat16* __restrict__ rbuf,
                                const unsigned* __restrict__ flags, unsigned* __restrict__ ctrl,
                                __nv_bfloat16* __restrict__ out, unsigned* __restrict__ timeout_flag) {
  const unsigned n = ld_volatile_u32(ctrl);
  const unsigned seq = 2u * n + 2u;
  const int slot = n % NSLOTS;
  const int m = blockIdx.x;
  if (threadIdx.x < NSRC && threadIdx.x != rank) {
    const unsigned* f = flags + (size_t)(slot * NSRC + threadIdx.x) * FLAG_STRIDE;
    unsigned spins = 0;
    while ((int)(ld_acquire_sys_u32(f) - seq) < 0) {
      if (++spins > (1u << 27)) { atomicOr(timeout_flag, 1u << threadIdx.x); break; }
    }
  }
  __syncthreads();
  __threadfence();
#pragma unroll
  for (int s = 0; s < NSRC; ++s) {
    if (s == rank) continue;
    const int c = (s * SLICE_CH + threadIdx.x) * 8;
    const uint4 v = __ldcg(reinterpret_cast<const uint4*>(rbuf + ((size_t)(slot * NSRC + s) * M_MAX + m) * H + c));
    *reinterpret_cast<uint4*>(out + (size_t)m * H + c) = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned d = atomicAdd(ctrl + 4, 1u);
    if (d == gridDim.x - 1) { ctrl[4] = 0u; atomicAdd(ctrl, 1u); }
  }
}

// ---- host ---------------------------------------------------------------------------------------------------------
static void check(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }

torch::Tensor alloc_shared(int64_t nbytes) {
  void* p = nullptr;
  check(cudaMalloc(&p, nbytes), "cudaMalloc");
  check(cudaMemset(p, 0, nbytes), "cudaMemset");
  auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  return torch::from_blob(p, {nbytes}, [](void* q) { cudaFree(q); }, opts);
}

std::vector<uint8_t> ipc_handle(torch::Tensor t) {
  cudaIpcMemHandle_t h;
  check(cudaIpcGetMemHandle(&h, t.data_ptr()), "cudaIpcGetMemHandle");
  return std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&h), reinterpret_cast<uint8_t*>(&h) + sizeof(h));
}

int64_t ipc_open(std::vector<uint8_t> bytes) {
  cudaIpcMemHandle_t h;
  TORCH_CHECK(bytes.size() == sizeof(h), "bad handle size");
  memcpy(&h, bytes.data(), sizeof(h));
  void* p = nullptr;
  check(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess), "cudaIpcOpenMemHandle");
  return reinterpret_cast<int64_t>(p);
}

// peers_data / peers_flags: the 3 REMOTE ranks in increasing rank order (own rank skipped)
void push(torch::Tensor x, int64_t M_MAX, int64_t rank, std::vector<int64_t> peers_data, std::vector<int64_t> peers_flags,
          torch::Tensor ctrl, int64_t max_ctas_per_dest) {
  TORCH_CHECK(x.is_contiguous() && x.dtype() == torch::kBFloat16 && x.size(1) == H && x.size(0) <= M_MAX);
  TORCH_CHECK(peers_data.size() == NPEER && peers_flags.size() == NPEER);
  Peers3 peers;
  for (int p = 0; p < NPEER; ++p) {
    peers.data[p] = reinterpret_cast<__nv_bfloat16*>(peers_data[p]);
    peers.flags[p] = reinterpret_cast<unsigned*>(peers_flags[p]);
  }
  const int M = x.size(0);
  const int chunks = M * (H / 8);
  const int block = 256;
  int ctas = (chunks + block - 1) / block;
  if (ctas > max_ctas_per_dest) ctas = max_ctas_per_dest;
  auto stream = c10::cuda::getCurrentCUDAStream();
  push_phased_kernel<<<ctas * NPEER, block, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), M, M_MAX,
                                                         (int)rank, peers, reinterpret_cast<unsigned*>(ctrl.data_ptr()), ctas);
}

// order: NCCL's summation order for this M (uint8, M * H / 8 bytes; see nccl_order_sum8)
static const uint8_t* order_ptr(const torch::Tensor& order, int64_t M) {
  TORCH_CHECK(order.is_cuda() && order.scalar_type() == torch::kUInt8 && order.is_contiguous() &&
                  order.numel() >= M * H / 8,
              "order: uint8 [M * H / 8] on the GPU");
  return order.data_ptr<uint8_t>();
}

void wait_reduce(torch::Tensor x_local, int64_t rank, int64_t M_MAX, torch::Tensor rbuf, torch::Tensor flags,
                 torch::Tensor ctrl, torch::Tensor order, torch::Tensor out, c10::optional<torch::Tensor> residual,
                 c10::optional<torch::Tensor> weight, c10::optional<torch::Tensor> residual_out, double eps,
                 torch::Tensor timeout_flag) {
  const int M = x_local.size(0);
  const uint8_t* ord = order_ptr(order, M);
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (residual.has_value()) {
    wait_reduce_kernel<true><<<M, H / 16, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_local.data_ptr()), (int)rank, (int)M_MAX,
        reinterpret_cast<const __nv_bfloat16*>(rbuf.data_ptr()), reinterpret_cast<const unsigned*>(flags.data_ptr()),
        reinterpret_cast<unsigned*>(ctrl.data_ptr()), ord, reinterpret_cast<const __nv_bfloat16*>(residual->data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight->data_ptr()), (float)eps,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), reinterpret_cast<__nv_bfloat16*>(residual_out->data_ptr()),
        reinterpret_cast<unsigned*>(timeout_flag.data_ptr()));
  } else {
    wait_reduce_kernel<false><<<M, H / 16, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_local.data_ptr()), (int)rank, (int)M_MAX,
        reinterpret_cast<const __nv_bfloat16*>(rbuf.data_ptr()), reinterpret_cast<const unsigned*>(flags.data_ptr()),
        reinterpret_cast<unsigned*>(ctrl.data_ptr()), ord, nullptr, nullptr, 0.f,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nullptr, reinterpret_cast<unsigned*>(timeout_flag.data_ptr()));
  }
}

// all-gather along the last dim: x [M, W] contiguous (W*itemsize % 16 == 0), out [M, 4*W]; peers in increasing rank order
void gather_push(torch::Tensor x, int64_t rank, std::vector<int64_t> peers_data, std::vector<int64_t> peers_flags,
                 torch::Tensor ctrl, int64_t max_ctas_per_dest) {
  TORCH_CHECK(x.is_contiguous() && x.dim() == 2);
  const int64_t row_bytes = x.size(1) * x.element_size();
  TORCH_CHECK(row_bytes % 16 == 0 && row_bytes / 16 <= CH_MAX && x.size(0) <= M_MAX_G);
  Peers3 peers;
  for (int p = 0; p < NPEER; ++p) {
    peers.data[p] = reinterpret_cast<__nv_bfloat16*>(peers_data[p]);
    peers.flags[p] = reinterpret_cast<unsigned*>(peers_flags[p]);
  }
  const int M = x.size(0), row_chunks = row_bytes / 16;
  const int block = 256;
  int ctas = (M * row_chunks + block - 1) / block;
  if (ctas > max_ctas_per_dest) ctas = max_ctas_per_dest;
  auto stream = c10::cuda::getCurrentCUDAStream();
  push_phased_gather_kernel<<<ctas * NPEER, block, 0, stream>>>(reinterpret_cast<const uint4*>(x.data_ptr()), M, row_chunks,
                                                                (int)rank, peers, reinterpret_cast<unsigned*>(ctrl.data_ptr()), ctas);
}

void wait_gather(torch::Tensor x_local, int64_t rank, torch::Tensor rbuf, torch::Tensor flags, torch::Tensor ctrl,
                 torch::Tensor out, torch::Tensor timeout_flag) {
  const int M = x_local.size(0), row_chunks = x_local.size(1) * x_local.element_size() / 16;
  auto stream = c10::cuda::getCurrentCUDAStream();
  wait_gather_kernel<<<M, 256, 0, stream>>>(reinterpret_cast<const uint4*>(x_local.data_ptr()), (int)rank, M, row_chunks,
                                            reinterpret_cast<const uint4*>(rbuf.data_ptr()),
                                            reinterpret_cast<const unsigned*>(flags.data_ptr()),
                                            reinterpret_cast<unsigned*>(ctrl.data_ptr()), reinterpret_cast<uint4*>(out.data_ptr()),
                                            reinterpret_cast<unsigned*>(timeout_flag.data_ptr()));
}

// two-shot all-reduce: 4 launches (push RS, reduce slice, push AG, assemble) on the second buffer set.
void all_reduce_twoshot(torch::Tensor x, torch::Tensor out, int64_t M_MAX, int64_t rank, torch::Tensor rbuf2,
                        torch::Tensor flags2, std::vector<int64_t> peers_data2, std::vector<int64_t> peers_flags2,
                        torch::Tensor ctrl2, torch::Tensor order, torch::Tensor timeout_flag, int64_t max_ctas_per_dest) {
  TORCH_CHECK(x.is_contiguous() && x.dtype() == torch::kBFloat16 && x.size(1) == H && x.size(0) <= M_MAX);
  TORCH_CHECK(out.is_contiguous() && out.sizes() == x.sizes());
  const uint8_t* ord = order_ptr(order, x.size(0));
  Peers3 peers;
  for (int p = 0; p < NPEER; ++p) {
    peers.data[p] = reinterpret_cast<__nv_bfloat16*>(peers_data2[p]);
    peers.flags[p] = reinterpret_cast<unsigned*>(peers_flags2[p]);
  }
  const int M = x.size(0);
  const int block = 256;
  int ctas = (M * SLICE_CH + block - 1) / block;
  if (ctas > max_ctas_per_dest) ctas = max_ctas_per_dest;
  auto stream = c10::cuda::getCurrentCUDAStream();
  unsigned* ctrl = reinterpret_cast<unsigned*>(ctrl2.data_ptr());
  const unsigned* fl = reinterpret_cast<const unsigned*>(flags2.data_ptr());
  const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(rbuf2.data_ptr());
  __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  unsigned* to = reinterpret_cast<unsigned*>(timeout_flag.data_ptr());
  push_twoshot_kernel<0><<<ctas * NPEER, block, 0, stream>>>(reinterpret_cast<const uint4*>(x.data_ptr()), M, (int)M_MAX,
                                                             (int)rank, peers, ctrl, ctas);
  reduce_slice_kernel<<<M, SLICE_CH, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)rank, (int)M_MAX,
                                                  rb, fl, ctrl, ord, o, to);
  push_twoshot_kernel<1><<<ctas * NPEER, block, 0, stream>>>(reinterpret_cast<const uint4*>(out.data_ptr()), M, (int)M_MAX,
                                                             (int)rank, peers, ctrl, ctas);
  assemble_kernel<<<M, SLICE_CH, 0, stream>>>((int)rank, (int)M_MAX, rb, fl, ctrl, o, to);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("alloc_shared", &alloc_shared);
  m.def("ipc_handle", &ipc_handle);
  m.def("ipc_open", &ipc_open);
  m.def("push", &push);
  m.def("wait_reduce", &wait_reduce);
  m.def("all_reduce_twoshot", &all_reduce_twoshot);
  m.def("gather_push", &gather_push);
  m.def("wait_gather", &wait_gather);
}
