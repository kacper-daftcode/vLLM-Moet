// vllm-moet sm_120, AR track (session P, 29.10.2026): reduce-scatter of a [M, 5120] bf16 TP4 partial over PCIe P2P
// whose rows carry the bits of NCCL's *all-reduce* of the same partials - the sequence-parallel prefill boundary
// (patch_vllm_prefill_sp_sm120.py) can then run on this rank's row shard without changing a single bit of the served
// prefill (NCCL's own reduce-scatter sums three quarters of the rows in another cyclic order than its all-reduce:
// 32 % of the elements differ by one bf16 ulp, M/REPORT.md §0/§5).
//
// Data movement (one launch, push_rs_kernel): rank r pushes the row block of peer d = (r + 1 + p) & 3 in phase p
// = 0, 1, 2 into d's IPC slot - every GPU writes to ONE peer at a time and every GPU receives from one source at a
// time (the platform rule of p2p_ar: a kernel writing to several peers concurrently is ~16x slower); the three
// phases are serialised inside the launch by per-phase CTA counters. Consumer (reduce_rows_kernel): for every
// 8-element group of this rank's rows the four partials (own rows of x + 3 slots) are added in the order NCCL's
// all-reduce used for that group - one byte per group from the order table probed on the TP communicator
// (p2p_reduce_scatter_sm120.py: nccl_order_tables), bf16 rounding after every add (__hadd2, what NCCL's bf16 sum
// does). Rows >= M (the pad of a shard count that does not divide M, as sp_reduce_scatter pads) come out as zeros.
//
// Layout per rank: data [NSLOTS][NPEER src-index][C_MAX][H] bf16 (the slot of source s on rank d is indexed by
// s - (s > d)), flags [NSLOTS][NSRC][FLAG_STRIDE] u32 (one 128-B line per flag, indexed by the source rank), ctrl
// [0] completed reduce-scatters (seq), [1..3] phase counters, [4] consumer done counter. Sequence numbers: every
// rank runs the same reduce-scatters in the same order, slot = seq % NSLOTS; two slots suffice because a rank's
// push of seq n+2 is issued after its consumer of n+1, which needs every peer's push of n+1, issued after that
// peer's consumer of n (stream order) - the slot of n is free by then.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <vector>

#ifdef P2P_RS_H
#define H P2P_RS_H
#else
#define H 5120
#endif
#ifndef P2P_RS_M_MAX
#define P2P_RS_M_MAX 4096
#endif
#define NSRC 4
#define NPEER 3
#define NSLOTS 2
#define FLAG_STRIDE 32
#define M_MAX P2P_RS_M_MAX
#define C_MAX (M_MAX / NSRC)
#define ROW_CH (H / 8)  // 16-B chunks per row

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

// index of rank `src` among the peers of rank `dst` (own rank skipped)
__device__ __host__ __forceinline__ int peer_index(int dst, int src) { return src - (src > dst ? 1 : 0); }

// NCCL's summation order for 8 consecutive elements: r0 | r1 << 2 | r2 << 4 | r3 << 6 for ((x_r0 + x_r1) + x_r2) +
// x_r3 with one bf16 rounding per add (what NCCL's bf16 reduce does: __hadd2). src(r) points at rank r's 8 elements.
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

// Phase p pushes to destination (rank + 1 + p) & 3. Two launch modes (the host's `split`):
//   single launch (only_phase < 0): grid NPEER * ctas_per_dest, phase p = blockIdx.x / ctas_per_dest; the CTAs of phase p wait
//     (one thread spinning) until phase p - 1's CTAs have issued their stores; serialize = 0 lets the three destinations run
//     concurrently (the bandwidth probe's control arm; the platform rule says this is slow);
//   one launch per phase (only_phase = p): grid ctas_per_dest, no spinning CTAs at all - stream order serialises the phases;
//     in the served model the waiting CTAs of the single launch would hold SM slots the overlapped mHC side stream needs.
// Block: 256 threads, UNROLL 16-B chunks in flight per thread.
template <int UNROLL>
__global__ void push_rs_kernel(const uint4* __restrict__ x, int M, int rank, Peers3 peers, unsigned* __restrict__ ctrl,
                               int ctas_per_dest, int serialize, int only_phase) {
  const unsigned seq = ld_volatile_u32(ctrl) + 1u;
  const int slot = seq % NSLOTS;
  const int p = only_phase >= 0 ? only_phase : blockIdx.x / ctas_per_dest;
  const int b = only_phase >= 0 ? blockIdx.x : blockIdx.x - p * ctas_per_dest;
  const int d = (rank + 1 + p) & 3;
  unsigned* phase = ctrl + 1;
  if (only_phase < 0 && serialize && p > 0) {
    if (threadIdx.x == 0) {
      while (ld_acquire_gpu_u32(phase + p - 1) < (unsigned)ctas_per_dest) {}
    }
    __syncthreads();
  }
  const int c = (M + NSRC - 1) / NSRC;           // rows per shard
  const int row0 = d * c;
  const int n = min(c, M - row0) > 0 ? min(c, M - row0) : 0;  // valid rows of d's shard
  const int chunks = n * ROW_CH;
  const uint4* src = x + (size_t)row0 * ROW_CH;
  uint4* dst = reinterpret_cast<uint4*>(peers.data[peer_index(rank, d)]) +
               (size_t)(slot * NPEER + peer_index(d, rank)) * C_MAX * ROW_CH;
  const int stride = ctas_per_dest * blockDim.x;
  int i = b * blockDim.x + threadIdx.x;
  for (; i + (UNROLL - 1) * stride < chunks; i += UNROLL * stride) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) v[u] = src[i + u * stride];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) dst[i + u * stride] = v[u];
  }
  for (; i < chunks; i += stride) dst[i] = src[i];
  __threadfence();  // gpu scope: this thread's P2P stores are ordered before the phase counter
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned done = atomicAdd(phase + p, 1u);
    if (done == (unsigned)ctas_per_dest - 1) {  // last CTA of this destination
      __threadfence_system();
      st_release_sys_u32(peers.flags[peer_index(rank, d)] + (size_t)(slot * NSRC + rank) * FLAG_STRIDE, seq);
      if (only_phase >= 0) {
        phase[p] = 0u;  // this phase's launch is complete: its counter is free for the next push
      } else if (serialize && p == NPEER - 1) {
        // single launch, serialised: every CTA of every phase has passed its spin by now -> reset for the next push
        // (the concurrent probe arm has no such point; its caller zeroes ctrl[1..3] between launches)
        phase[0] = 0u; phase[1] = 0u; phase[2] = 0u;
      }
    }
  }
}

// grid: min(c, reduce_ctas) persistent CTAs looping over the rows m of this rank's shard (c = ceil(M/4) rows; rows >= M
// are the zero pad), block: H/16 = 320 threads (16 elements = 2 groups per thread). out[m] = sum over the 4 partials of
// row rank*c + m in NCCL's all-reduce order. Few CTAs wait on the flags, so the wait holds few SM slots.
__global__ void reduce_rows_kernel(const __nv_bfloat16* __restrict__ x, int M, int c, int rank,
                                   const __nv_bfloat16* __restrict__ rbuf, const unsigned* __restrict__ flags,
                                   unsigned* __restrict__ ctrl, const uint8_t* __restrict__ order,
                                   __nv_bfloat16* __restrict__ out, unsigned* __restrict__ timeout_flag) {
  const unsigned seq = ld_volatile_u32(ctrl) + 1u;
  const int slot = seq % NSLOTS;
  if (threadIdx.x < NSRC && threadIdx.x != rank) {
    const unsigned* f = flags + (size_t)(slot * NSRC + threadIdx.x) * FLAG_STRIDE;
    unsigned spins = 0;
    while ((int)(ld_acquire_sys_u32(f) - seq) < 0) {
      if (++spins > (1u << 28)) { atomicOr(timeout_flag, 1u << threadIdx.x); break; }
    }
  }
  __syncthreads();
  __threadfence();
  const int col = threadIdx.x * 16;
  for (int m = blockIdx.x; m < c; m += gridDim.x) {
    const int g = rank * c + m;  // global row
    uint4* o = reinterpret_cast<uint4*>(out + (size_t)m * H + col);
    if (g >= M) {
      o[0] = make_uint4(0u, 0u, 0u, 0u);
      o[1] = make_uint4(0u, 0u, 0u, 0u);
    } else {
      const size_t e0 = (size_t)g * H + col;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        o[h] = nccl_order_sum8(order[(e0 >> 3) + h], [&](unsigned s) {
          return reinterpret_cast<const uint4*>(
              (s == (unsigned)rank) ? x + e0 + 8 * h
                                    : rbuf + ((size_t)(slot * NPEER + peer_index(rank, (int)s)) * C_MAX + m) * H + col + 8 * h);
        });
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned d = atomicAdd(ctrl + 4, 1u);
    if (d == gridDim.x - 1) {  // last CTA: next sequence number (every CTA has read the current one by now)
      ctrl[4] = 0u;
      __threadfence();
      atomicAdd(ctrl, 1u);
    }
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

int64_t data_bytes() { return (int64_t)NSLOTS * NPEER * C_MAX * H * 2; }
int64_t flags_bytes() { return (int64_t)NSLOTS * NSRC * FLAG_STRIDE * 4; }
int64_t m_max() { return M_MAX; }
int64_t hidden() { return H; }

static Peers3 make_peers(const std::vector<int64_t>& peers_data, const std::vector<int64_t>& peers_flags) {
  TORCH_CHECK(peers_data.size() == NPEER && peers_flags.size() == NPEER, "3 peers (increasing rank order, own rank skipped)");
  Peers3 peers;
  for (int p = 0; p < NPEER; ++p) {
    peers.data[p] = reinterpret_cast<__nv_bfloat16*>(peers_data[p]);
    peers.flags[p] = reinterpret_cast<unsigned*>(peers_flags[p]);
  }
  return peers;
}

template <int UNROLL>
static void launch_push(const uint4* xp, int M, int rank, const Peers3& peers, unsigned* ct, int ctas, int ser, int only_phase,
                        cudaStream_t stream) {
  const dim3 grid(only_phase >= 0 ? ctas : NPEER * ctas);
  push_rs_kernel<UNROLL><<<grid, 256, 0, stream>>>(xp, M, rank, peers, ct, ctas, ser, only_phase);
}

// the push alone (the bandwidth probe uses it too): x [M, H] bf16 contiguous, M <= M_MAX. split = one launch per peer
// (no spinning CTAs; serialize is implied), else one launch for the three phases (serialize: CTA counters, or concurrent).
void push(torch::Tensor x, int64_t rank, std::vector<int64_t> peers_data, std::vector<int64_t> peers_flags,
          torch::Tensor ctrl, int64_t ctas_per_dest, int64_t unroll, bool serialize, bool split) {
  TORCH_CHECK(x.is_contiguous() && x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.size(1) == H && x.size(0) <= M_MAX,
              "x: contiguous bf16 [M <= ", M_MAX, ", ", H, "]");
  TORCH_CHECK(ctas_per_dest >= 1 && ctas_per_dest <= 1024);
  Peers3 peers = make_peers(peers_data, peers_flags);
  auto stream = c10::cuda::getCurrentCUDAStream();
  const uint4* xp = reinterpret_cast<const uint4*>(x.data_ptr());
  unsigned* ct = reinterpret_cast<unsigned*>(ctrl.data_ptr());
  const int M = x.size(0), ctas = (int)ctas_per_dest, ser = serialize ? 1 : 0;
  const int n_launch = split ? NPEER : 1;
  for (int l = 0; l < n_launch; ++l) {
    const int only = split ? l : -1;
    switch (unroll) {
      case 1: launch_push<1>(xp, M, (int)rank, peers, ct, ctas, ser, only, stream); break;
      case 2: launch_push<2>(xp, M, (int)rank, peers, ct, ctas, ser, only, stream); break;
      case 8: launch_push<8>(xp, M, (int)rank, peers, ct, ctas, ser, only, stream); break;
      default: launch_push<4>(xp, M, (int)rank, peers, ct, ctas, ser, only, stream); break;
    }
  }
}

// the consumer: out [ceil(M/4), H] bf16; order uint8 [>= M * H / 8] (the full table of this M); reduce_ctas persistent
// CTAs (<= 0: one per row)
void reduce(torch::Tensor x, int64_t rank, torch::Tensor rbuf, torch::Tensor flags, torch::Tensor ctrl,
            torch::Tensor order, torch::Tensor out, torch::Tensor timeout_flag, int64_t reduce_ctas) {
  const int M = x.size(0), c = (M + NSRC - 1) / NSRC;
  TORCH_CHECK(out.is_contiguous() && out.dtype() == torch::kBFloat16 && out.size(0) == c && out.size(1) == H);
  TORCH_CHECK(order.is_cuda() && order.scalar_type() == torch::kUInt8 && order.is_contiguous() && order.numel() >= (int64_t)M * H / 8,
              "order: uint8 [M * H / 8] on the GPU");
  auto stream = c10::cuda::getCurrentCUDAStream();
  const int grid = (reduce_ctas <= 0 || reduce_ctas > c) ? c : (int)reduce_ctas;
  reduce_rows_kernel<<<grid, H / 16, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), M, c, (int)rank,
      reinterpret_cast<const __nv_bfloat16*>(rbuf.data_ptr()), reinterpret_cast<const unsigned*>(flags.data_ptr()),
      reinterpret_cast<unsigned*>(ctrl.data_ptr()), order.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), reinterpret_cast<unsigned*>(timeout_flag.data_ptr()));
}

// reduce-scatter = push (1 or 3 launches) + reduce
void reduce_scatter(torch::Tensor x, torch::Tensor out, int64_t rank, torch::Tensor rbuf, torch::Tensor flags,
                    std::vector<int64_t> peers_data, std::vector<int64_t> peers_flags, torch::Tensor ctrl,
                    torch::Tensor order, torch::Tensor timeout_flag, int64_t ctas_per_dest, int64_t unroll, bool split,
                    int64_t reduce_ctas) {
  push(x, rank, peers_data, peers_flags, ctrl, ctas_per_dest, unroll, true, split);
  reduce(x, rank, rbuf, flags, ctrl, order, out, timeout_flag, reduce_ctas);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("alloc_shared", &alloc_shared);
  m.def("ipc_handle", &ipc_handle);
  m.def("ipc_open", &ipc_open);
  m.def("data_bytes", &data_bytes);
  m.def("flags_bytes", &flags_bytes);
  m.def("m_max", &m_max);
  m.def("hidden", &hidden);
  m.def("push", &push);
  m.def("reduce", &reduce);
  m.def("reduce_scatter", &reduce_scatter);
}
