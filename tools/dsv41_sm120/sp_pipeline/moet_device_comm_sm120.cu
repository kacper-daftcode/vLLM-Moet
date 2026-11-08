// vllm-moet sm_120 - MK-4 (program megakernel, stage 3, milestone 1 "the hybrid"): the sequence-parallel collectives of the
// pipelined prefill step (reduce-scatter of the [T, 5120] bf16 partials, all-gathers of the MXFP8 MoE input / the shard
// projections) as device-initiated PCIe P2P pushes from ONE small kernel instance per collective, instead of NCCL.
//
// Why a kernel per collective and not NCCL (K-6 / MK-3): NCCL's ring kernel needs an empty SM (98.5 KB smem, 52 K registers) and
// one NCCL FIFO serializes both groups' chains; the comm CTAs here are 32 threads x <= 32 registers x <= 6.4 KB smem, so they
// co-reside with every compute kernel of the layer that leaves 1 024 registers per SM free (fc1 v10, the CUTLASS MXFP8 GEMMs,
// the SASS attention, fc2, cuBLAS, mHC) and start pushing the moment the stream reaches them. MK-3 §3.3: the comm role is
// 32-48 CTAs of 1-2 warps (one pusher per SM slows every compute tile 38-60 % through the L2/XBAR), dependency checks once
// per task group, every device-side wait bounded by %globaltimer (an error record in pinned memory + __trap, never a spinning
// kernel - MK-1's GPU-3 incident).
//
// One instance = one collective of one row group g (0/1) at phase sequence pseq (monotonic per group, host-side):
//   RS : grid = n_push + n_red CTAs x 32 threads. Pushers (CTAs [0, n_push)) push the partial's rows of peer p into the peer's
//        receive slot slot[g][pseq&1][me] in 3 PHASES (p = me+1+ph: one destination at a time - MK-2: interleaved destinations are
//        25x slower), each phase in `pieces` pieces: after a piece every pusher fences (sys) its stores and counts on a local
//        arrival counter; the last arriver writes the receiver's flag line rs_flag[g][par][me][piece] = pseq + 1 (st.release.sys).
//        Reducers (CTAs [n_push, n_push+n_red)) wait per piece for the three sources' flags, then reduce the piece: for every
//        4 elements (8 B) the table byte says in which order the four partials are added - bf16(bf16(bf16(x_r0 + x_r1) + x_r2)
//        + x_r3) = NCCL's ring chain for that element (probed, rs_order_probe.py / the module's table) - operands staged by
//        cp.async in a 3-stage smem ring (a 32-register budget leaves no room for 4 x 16-B loads in flight in registers).
//        The last reducer writes rs_done[g] = pseq + 1 (sys-visible: the peers' slot-reuse check). The kernel's exit is the
//        completion signal for the host (an event on the comm-kernel stream) - no stream memops, no wait kernels.
//   AG : pushers push my [c, row_bytes] shard into the peers' arena region (the same offset on every rank; rank-major rows),
//        reducers copy the shard into my own arena and wait for the peers' flags, so the kernel exits when the gathered buffer is
//        complete.
// Buffer reuse: two parities per group. A pusher pushes RS pieces of phase P into slot parity P&1 of peer q only after q's
// rs_done[g] >= P - 1 (q reduced phase P-2); AG pieces into q's arena parity P&1 only after q's consumed[g] >= P - 1 (q's compute
// consumed phase P-2's gather: the RS kernel of phase P-1 writes consumed[g] = P-1 at its start, because the stream order puts it
// after the compute segment that read the previous gather). A per-GPU link token (optional) serializes the push phases of the two
// groups' concurrent instances (one destination per GPU at a time).
//
// build (torch.utils.cpp_extension.load): -O3 -std=c++20 -gencode arch=compute_120a,code=sm_120a -maxrregcount=32
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <string.h>
#include <vector>

namespace {

constexpr int kWorld = 4;
constexpr int kGroups = 2;
constexpr int kPar = 2;
constexpr int kPhases = kWorld - 1;
constexpr int kMaxPieces = 32;
constexpr int kSlots = 5;                              // per phase: slot 0 = the RS, slots 1..4 = up to four all-gathers
constexpr int kThreads = 32;
constexpr int kLineBytes = 128;
constexpr int kStages = 3;
constexpr int kUnitRowBytes = kThreads * 16;          // 512 B of one operand per unit-row
constexpr int kTableRowBytes = kUnitRowBytes / 8;      // 1 byte per 4 elements (8 B): 64 B per unit-row
constexpr int kStageBytes = kWorld * kUnitRowBytes + kTableRowBytes;  // 2 112
constexpr int kSmemBytes = kStages * kStageBytes;      // 6 336

enum { K_RS = 0, K_AG = 1 };
enum { E_NONE = 0, E_FLAG_RS = 1, E_FLAG_AG = 2, E_TEAM = 3, E_REUSE_RS = 4, E_REUSE_AG = 5, E_TOKEN = 6, E_GO = 7 };

struct ErrRec { uint32_t set, what, expected, seen, cta, rank, g, pseq; uint64_t waited_ns; };

// region layout (bytes), all lines 128-B aligned; offsets computed on the host (Layout) and passed by value
struct Layout {
  size_t flag;        // [g][par][slot][src][piece] lines (value = the use count of (g, par, slot) that landed)
  size_t ctrl;        // [g] x 4 lines: 0 rs_done (= pseq + 1 of the reduced phase), 1 consumed (= pseq: phases < pseq consumed), 2-3 spare
  size_t cnt;         // [g][par][slot] x (arrive[ph][piece] + team[ph] + go[ph] + red_done) u32 words, padded to lines
  size_t token;       // 1 line
  size_t slots;       // [g][par][src] x slot_bytes
  size_t arena;       // [g][par] x arena_bytes
  size_t slot_bytes, arena_bytes, total;
  int pieces;
};
constexpr int kCntArrive = 0;                                   // [ph][piece]
constexpr int kCntTeam = kCntArrive + kPhases * kMaxPieces;     // [ph]
constexpr int kCntGo = kCntTeam + kPhases;                      // [ph]
constexpr int kCntRed = kCntGo + kPhases;                       // 1
constexpr int kCntWords = ((kCntRed + 1 + 31) / 32) * 32;       // padded to lines

__host__ __device__ inline size_t off_flag(const Layout& L, int g, int par, int slot, int src, int piece) { return L.flag + (((((size_t)g * kPar + par) * kSlots + slot) * kWorld + src) * kMaxPieces + piece) * kLineBytes; }
__host__ __device__ inline size_t off_ctrl(const Layout& L, int g, int which) { return L.ctrl + ((size_t)g * 4 + which) * kLineBytes; }
__host__ __device__ inline size_t off_cnt(const Layout& L, int g, int par, int slot, int word) { return L.cnt + ((((size_t)g * kPar + par) * kSlots + slot) * kCntWords + word) * 4; }
__host__ __device__ inline size_t off_slot(const Layout& L, int g, int par, int src) { return L.slots + (((size_t)g * kPar + par) * kWorld + src) * L.slot_bytes; }
__host__ __device__ inline size_t off_arena(const Layout& L, int g, int par) { return L.arena + ((size_t)g * kPar + par) * L.arena_bytes; }

Layout make_layout(size_t slot_bytes, size_t arena_bytes, int pieces) {
  Layout L; memset(&L, 0, sizeof(L));
  size_t o = 0;
  L.flag = o; o += (size_t)kGroups * kPar * kSlots * kWorld * kMaxPieces * kLineBytes;
  L.ctrl = o; o += (size_t)kGroups * 4 * kLineBytes;
  L.cnt = o; o += (size_t)kGroups * kPar * kSlots * kCntWords * 4;
  L.token = o; o += kLineBytes;
  o = (o + 4095) & ~(size_t)4095;
  L.slots = o; o += (size_t)kGroups * kPar * kWorld * slot_bytes;
  L.arena = o; o += (size_t)kGroups * kPar * arena_bytes;
  L.slot_bytes = slot_bytes; L.arena_bytes = arena_bytes; L.total = o; L.pieces = pieces;
  return L;
}

struct Args {
  int kind, rank, g;
  uint32_t pseq;         // the group's phase sequence (monotonic): slot / arena parity, the reuse checks
  int slot;              // 0 = RS, 1..4 = the k-th all-gather of the phase
  uint32_t use;          // how many times (g, par, slot) has been launched including this one (>= 1): the flags' value
  uint32_t arrive_target;// the host's cumulative pusher count of (g, par, slot) including this launch: the team barriers' target (the per-piece
                         // arrival words reset themselves: a launch may use fewer pieces than the previous one of this slot)
  uint32_t red_target;   // the host's cumulative pieces x reducers of (g, par, slot) including this launch: the reduce-done target
  int n_push, n_red, pieces, unroll;
  int link_token;
  uint64_t wd_ns;
  // RS
  const uint8_t* x;      // [4c, row_bytes] partial (contiguous)
  uint8_t* out;          // [c, row_bytes]
  const uint8_t* table;  // [c * row_bytes / 8] u8 order codes (per 4 bf16)
  // AG
  const uint8_t* src;    // [c, row_bytes]
  size_t arena_off;      // where the gathered [4c, row_bytes] starts inside arena[g][par]
  // common
  int c; size_t row_bytes;
  uint8_t* my;
  uint8_t* peer[kWorld];
  Layout L;
  ErrRec* err;
};

// ------------------------------------------------------------------------------------------------------- device helpers
__device__ __forceinline__ uint64_t gtimer() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) { asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t* p) { uint32_t v; asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t* p) { uint32_t v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void fence_acq_rel_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void nanosleep_(uint32_t ns) { asm volatile("nanosleep.u32 %0;" ::"r"(ns)); }
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __noinline__ void fail(const Args& a, uint32_t what, uint32_t expected, uint32_t seen, uint64_t waited) {
  // the record lives in pinned host memory (plain stores: no atomics over PCIe); the first writer wins in practice, any writer is loud
  if (a.err->set == 0u) {
    a.err->what = what; a.err->expected = expected; a.err->seen = seen; a.err->cta = blockIdx.x; a.err->rank = a.rank;
    a.err->g = a.g; a.err->pseq = a.pseq; a.err->waited_ns = waited;
    __threadfence_system();
    a.err->set = 1u;
    __threadfence_system();
  }
  __trap();
}
// bounded waits (lane 0 calls; the warp syncs after). Relaxed polls + nanosleep, then one acquire fence (MK-3 §3.3.2).
__device__ __forceinline__ void wait_ge_sys(const Args& a, const uint32_t* p, uint32_t v, uint32_t what) {
  if (ld_relaxed_sys(p) >= v) return;
  const uint64_t t0 = gtimer();
  for (;;) {
    nanosleep_(1000);
    if (ld_relaxed_sys(p) >= v) return;
    const uint64_t dt = gtimer() - t0;
    if (dt > a.wd_ns) fail(a, what, v, ld_relaxed_sys(p), dt);
  }
}
__device__ __forceinline__ void wait_ge_gpu(const Args& a, const uint32_t* p, uint32_t v, uint32_t what) {
  if (ld_relaxed_gpu(p) >= v) return;
  const uint64_t t0 = gtimer();
  for (;;) {
    nanosleep_(500);
    if (ld_relaxed_gpu(p) >= v) return;
    const uint64_t dt = gtimer() - t0;
    if (dt > a.wd_ns) fail(a, what, v, ld_relaxed_gpu(p), dt);
  }
}
__device__ __forceinline__ uint32_t* word(uint8_t* base, size_t off) { return reinterpret_cast<uint32_t*>(base + off); }

// ------------------------------------------------------------------------------------------------------- the push stripe
// units [u0, u1) of 16 B from src to dst, interleaved over the n_push pushers: unit = base + (u * n_push + k) * 32 + lane
template <int U>
__device__ __forceinline__ void push_range(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t u0, size_t u1, int k, int n_push) {
  // lanes = the CTA's threads (1..4 warps per pusher CTA: more bytes in flight under the compute's DRAM load)
  const int lane = threadIdx.x, nthr = blockDim.x;
  const size_t stripe = (size_t)n_push * nthr * U;
  size_t base = u0;
  for (; base + stripe <= u1; base += stripe) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = src[base + ((size_t)u * n_push + k) * nthr + lane];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const size_t i = base + ((size_t)u * n_push + k) * nthr + lane;
      asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(dst + i), "r"(v[u].x), "r"(v[u].y), "r"(v[u].z), "r"(v[u].w) : "memory");
    }
  }
  // tail: rows of nthr units, round-robin over the pushers
  for (size_t row = base + (size_t)k * nthr; row < u1; row += (size_t)n_push * nthr) {
    const size_t i = row + lane;
    if (i < u1) { const uint4 v = src[i]; asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(dst + i), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
  }
}

__device__ __forceinline__ void piece_range(size_t total_u, int piece, int pieces, size_t& u0, size_t& u1) {
  // piece boundaries on unit-row (512 B) multiples so that reducers' unit-rows never straddle pieces
  const size_t rows = (total_u + kThreads - 1) / kThreads;
  const size_t per = (rows + pieces - 1) / pieces;
  u0 = min((size_t)piece * per * kThreads, total_u);
  u1 = min(((size_t)piece + 1) * per * kThreads, total_u);
}

// ------------------------------------------------------------------------------------------------------- pusher role
__device__ void pusher(const Args& a) {
  const int k = blockIdx.x, lane = threadIdx.x;
  const int par = a.pseq & 1;
  const uint32_t use = a.use;
  uint8_t* my = a.my;
  const size_t total_u = (size_t)a.c * a.row_bytes / 16;
  uint32_t* token = word(my, a.L.token);
  for (int ph = 0; ph < kPhases; ++ph) {
    const int p = (a.rank + 1 + ph) % kWorld;
    uint8_t* peer = a.peer[p];
    // reuse check on the destination (one lane): the peer consumed the previous use of this parity
    if (lane == 0 && a.pseq >= 2) {
      if (a.kind == K_RS) wait_ge_sys(a, word(peer, off_ctrl(a.L, a.g, 0)), a.pseq - 1, E_REUSE_RS);
      else wait_ge_sys(a, word(peer, off_ctrl(a.L, a.g, 1)), a.pseq - 1, E_REUSE_AG);
    }
    if (a.kind == K_RS && k == 0 && lane == 0 && a.n_red == 0) st_release_sys(word(my, off_ctrl(a.L, a.g, 1)), a.pseq);  // split mode: consumed[g] = pseq
    // the team enters the phase together (one destination per team at a time); CTA 0 takes the per-GPU link token first
    uint32_t* go = word(my, off_cnt(a.L, a.g, par, a.slot, kCntGo + ph));
    if (lane == 0) {
      if (k == 0) {
        if (a.link_token) {
          const uint64_t t0 = gtimer();
          while (atomicCAS(token, 0u, 1u) != 0u) {
            nanosleep_(500);
            const uint64_t dt = gtimer() - t0;
            if (dt > a.wd_ns) fail(a, E_TOKEN, 0, 1, dt);
          }
        }
        st_release_gpu(go, use);
      } else {
        wait_ge_gpu(a, go, use, E_GO);
      }
    }
    __syncthreads();
    const uint4* src; uint4* dst;
    if (a.kind == K_RS) {
      src = reinterpret_cast<const uint4*>(a.x + (size_t)p * a.c * a.row_bytes);
      dst = reinterpret_cast<uint4*>(peer + off_slot(a.L, a.g, par, a.rank));
    } else {
      src = reinterpret_cast<const uint4*>(a.src);
      dst = reinterpret_cast<uint4*>(peer + off_arena(a.L, a.g, par) + a.arena_off + (size_t)a.rank * a.c * a.row_bytes);
    }
    for (int piece = 0; piece < a.pieces; ++piece) {
      size_t u0, u1; piece_range(total_u, piece, a.pieces, u0, u1);
      if (a.unroll >= 4) push_range<4>(src, dst, u0, u1, k, a.n_push);
      else push_range<2>(src, dst, u0, u1, k, a.n_push);
      __syncthreads();
      if (lane == 0) {
        __threadfence_system();  // my stores to the peer are performed at system scope before I count
        // per-piece arrival: the last of the n_push pushers resets the word for the next use (the next instance of this (g, par, slot)
        // runs after this one on the same stream, so nobody touches it in between) and publishes the piece to the receiver
        uint32_t* arrive = word(my, off_cnt(a.L, a.g, par, a.slot, kCntArrive + ph * kMaxPieces + piece));
        const uint32_t old = atomicAdd(arrive, 1u);
        if (old + 1 == (uint32_t)a.n_push) {
          atomicExch(arrive, 0u);
          uint32_t* flag = word(peer, off_flag(a.L, a.g, par, a.slot, a.rank, piece));
          __threadfence_system();
          st_release_sys(flag, use);
        }
      }
    }
    // the team leaves the phase together
    uint32_t* team = word(my, off_cnt(a.L, a.g, par, a.slot, kCntTeam + ph));
    if (lane == 0) {
      atomicAdd(team, 1u);
      wait_ge_gpu(a, team, a.arrive_target, E_TEAM);
      if (k == 0 && a.link_token) { __threadfence(); atomicExch(token, 0u); }
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------------- reducer role (RS)
// 4 bf16 (8 B) + 4 bf16 with one RN-even rounding per element = NCCL's FuncSum<__nv_bfloat16> on sm_80+ (__hadd2); for two bf16 operands
// this equals the fp32 add followed by cvt.rn.bf16 (no double rounding is possible), i.e. torch's bf16 add and the probe's emulation
__device__ __forceinline__ uint2 hadd4(uint2 a, uint2 b) {
  __nv_bfloat162 a0 = *reinterpret_cast<__nv_bfloat162*>(&a.x), a1 = *reinterpret_cast<__nv_bfloat162*>(&a.y);
  __nv_bfloat162 b0 = *reinterpret_cast<__nv_bfloat162*>(&b.x), b1 = *reinterpret_cast<__nv_bfloat162*>(&b.y);
  __nv_bfloat162 r0 = __hadd2(a0, b0), r1 = __hadd2(a1, b1);
  uint2 r; r.x = *reinterpret_cast<uint32_t*>(&r0); r.y = *reinterpret_cast<uint32_t*>(&r1); return r;
}

__device__ void reducer_rs(const Args& a, uint8_t* smem) {
  const int j = blockIdx.x - a.n_push, lane = threadIdx.x;
  const int par = a.pseq & 1;
  const uint32_t use = a.use;
  uint8_t* my = a.my;
  if (blockIdx.x == a.n_push && lane == 0) st_release_sys(word(my, off_ctrl(a.L, a.g, 1)), a.pseq);  // consumed[g] = pseq: the previous phase's gather is consumed (stream order)
  const size_t total_u = (size_t)a.c * a.row_bytes / 16;
  const uint8_t* op[kWorld];
  for (int r = 0; r < kWorld; ++r) op[r] = (r == a.rank) ? a.x + (size_t)a.rank * a.c * a.row_bytes : my + off_slot(a.L, a.g, par, r);
  uint32_t* red_done = word(my, off_cnt(a.L, a.g, par, a.slot, kCntRed));
  for (int piece = 0; piece < a.pieces; ++piece) {
    size_t u0, u1; piece_range(total_u, piece, a.pieces, u0, u1);
    if (lane == 0) {
      for (int r = 0; r < kWorld; ++r) if (r != a.rank) wait_ge_sys(a, word(my, off_flag(a.L, a.g, par, a.slot, r, piece)), use, E_FLAG_RS);
      fence_acq_rel_sys();
    }
    __syncwarp();
    // unit-rows of this piece handled by reducer j: row = u0/32 + j, + n_red, ...
    const size_t row0 = u0 / kThreads, row1 = (u1 + kThreads - 1) / kThreads;
    size_t rows_mine = 0;
    for (size_t row = row0 + j; row < row1; row += a.n_red) ++rows_mine;
    auto issue = [&](size_t idx, int stage) {
      const size_t row = row0 + j + idx * a.n_red;
      const size_t unit = row * kThreads + lane;
      uint8_t* st = smem + stage * kStageBytes;
      if (unit < u1) {
#pragma unroll
        for (int r = 0; r < kWorld; ++r) cp_async16(st + r * kUnitRowBytes + lane * 16, op[r] + unit * 16);
        if (lane < kTableRowBytes / 16) cp_async16(st + kWorld * kUnitRowBytes + lane * 16, a.table + row * kTableRowBytes + lane * 16);
      }
      cp_async_commit();
    };
    if (rows_mine > 0) issue(0, 0);
    if (rows_mine > 1) issue(1, 1);
    for (size_t idx = 0; idx < rows_mine; ++idx) {
      if (idx + 2 < rows_mine) { issue(idx + 2, (int)((idx + 2) % kStages)); cp_async_wait<2>(); }
      else if (idx + 1 < rows_mine) { cp_async_wait<1>(); }
      else { cp_async_wait<0>(); }
      __syncwarp();
      const int stage = (int)(idx % kStages);
      const uint8_t* st = smem + stage * kStageBytes;
      const size_t row = row0 + j + idx * a.n_red;
      const size_t unit = row * kThreads + lane;
      if (unit < u1) {
        const uint8_t* tb = st + kWorld * kUnitRowBytes + lane * 2;
        uint2 o2[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t code = tb[h];
          uint2 acc = *reinterpret_cast<const uint2*>(st + (code & 3u) * kUnitRowBytes + lane * 16 + h * 8);
          acc = hadd4(acc, *reinterpret_cast<const uint2*>(st + ((code >> 2) & 3u) * kUnitRowBytes + lane * 16 + h * 8));
          acc = hadd4(acc, *reinterpret_cast<const uint2*>(st + ((code >> 4) & 3u) * kUnitRowBytes + lane * 16 + h * 8));
          acc = hadd4(acc, *reinterpret_cast<const uint2*>(st + ((code >> 6) & 3u) * kUnitRowBytes + lane * 16 + h * 8));
          o2[h] = acc;
        }
        uint4 o; o.x = o2[0].x; o.y = o2[0].y; o.z = o2[1].x; o.w = o2[1].y;
        *reinterpret_cast<uint4*>(a.out + unit * 16) = o;
      }
      __syncwarp();
    }
    __syncwarp();
    if (lane == 0) {
      const uint32_t old = atomicAdd(red_done, 1u);
      if (old + 1 == a.red_target) { __threadfence(); st_release_sys(word(my, off_ctrl(a.L, a.g, 0)), a.pseq + 1); }
    }
  }
}

// ------------------------------------------------------------------------------------------------------- reducer role (AG)
__device__ void reducer_ag(const Args& a) {
  const int j = blockIdx.x - a.n_push, lane = threadIdx.x;
  const int par = a.pseq & 1;
  uint8_t* my = a.my;
  // my own shard into my arena (interleaved over the reducers), 4 x 16 B in flight per lane
  {
    const uint4* src = reinterpret_cast<const uint4*>(a.src);
    uint4* dst = reinterpret_cast<uint4*>(my + off_arena(a.L, a.g, par) + a.arena_off + (size_t)a.rank * a.c * a.row_bytes);
    const size_t total_u = (size_t)a.c * a.row_bytes / 16;
    push_range<4>(src, dst, 0, total_u, j, a.n_red);
  }
  __syncwarp();
  if (lane == 0) {
    for (int piece = 0; piece < a.pieces; ++piece)
      for (int r = 0; r < kWorld; ++r) if (r != a.rank) wait_ge_sys(a, word(my, off_flag(a.L, a.g, par, a.slot, r, piece)), a.use, E_FLAG_AG);
    fence_acq_rel_sys();
  }
  __syncwarp();
}

__global__ void __maxnreg__(32) comm_kernel(Args a) {
  extern __shared__ __align__(16) uint8_t smem[];
  if ((int)blockIdx.x < a.n_push) pusher(a);
  else if (a.kind == K_RS) reducer_rs(a, smem);
  else reducer_ag(a);
}

// ---- split mode (MK-4 §4.6 item 1): only the pushers stay resident during the transfer; the landing is awaited by one bounded CTA on
// the pipeline's comm stream and the reduce runs as a transient full-occupancy kernel behind it (no co-residency constraint: it exits in
// 40-80 us), so NCCL's remaining kernels and the CUTLASS GEMMs see <= n_push occupied SMs instead of 144.
__global__ void __maxnreg__(32) wait_kernel(Args a) {
  if (threadIdx.x == 0) {
    const int par = a.pseq & 1;
    for (int piece = 0; piece < a.pieces; ++piece)
      for (int r = 0; r < kWorld; ++r) if (r != a.rank) wait_ge_sys(a, word(a.my, off_flag(a.L, a.g, par, a.slot, r, piece)), a.use, a.kind == K_RS ? E_FLAG_RS : E_FLAG_AG);
    fence_acq_rel_sys();
  }
}
constexpr int kRedThreads = 256;
__global__ void __launch_bounds__(kRedThreads) reduce_kernel(Args a) {
  const int par = a.pseq & 1;
  const uint8_t* op[kWorld];
  for (int r = 0; r < kWorld; ++r) op[r] = (r == a.rank) ? a.x + (size_t)a.rank * a.c * a.row_bytes : a.my + off_slot(a.L, a.g, par, r);
  const size_t total_u = (size_t)a.c * a.row_bytes / 16;
  for (size_t u = (size_t)blockIdx.x * kRedThreads + threadIdx.x; u < total_u; u += (size_t)gridDim.x * kRedThreads) {
    uint4 v[kWorld];
#pragma unroll
    for (int r = 0; r < kWorld; ++r) v[r] = *reinterpret_cast<const uint4*>(op[r] + u * 16);
    const uint16_t codes = *reinterpret_cast<const uint16_t*>(a.table + u * 2);  // 2 bytes: the two 4-element halves of this 16-B unit
    uint2 o2[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t code = (codes >> (8 * h)) & 0xFFu;
      const uint2 h0 = h ? make_uint2(v[0].z, v[0].w) : make_uint2(v[0].x, v[0].y);
      const uint2 h1 = h ? make_uint2(v[1].z, v[1].w) : make_uint2(v[1].x, v[1].y);
      const uint2 h2 = h ? make_uint2(v[2].z, v[2].w) : make_uint2(v[2].x, v[2].y);
      const uint2 h3 = h ? make_uint2(v[3].z, v[3].w) : make_uint2(v[3].x, v[3].y);
      auto pick = [&](uint32_t r) -> uint2 { return r == 0 ? h0 : (r == 1 ? h1 : (r == 2 ? h2 : h3)); };  // selects, no local memory
      uint2 acc = pick(code & 3u);
      acc = hadd4(acc, pick((code >> 2) & 3u));
      acc = hadd4(acc, pick((code >> 4) & 3u));
      acc = hadd4(acc, pick((code >> 6) & 3u));
      o2[h] = acc;
    }
    uint4 o; o.x = o2[0].x; o.y = o2[0].y; o.z = o2[1].x; o.w = o2[1].y;
    *reinterpret_cast<uint4*>(a.out + u * 16) = o;
  }
  // the last CTA publishes rs_done (the peers' slot-reuse check); the kernel's exit is the host's completion signal
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    uint32_t* done = word(a.my, off_cnt(a.L, a.g, par, a.slot, kCntRed));
    const uint32_t old = atomicAdd(done, 1u);
    if (old + 1 == a.red_target) { __threadfence(); st_release_sys(word(a.my, off_ctrl(a.L, a.g, 0)), a.pseq + 1); }
  }
}

// ------------------------------------------------------------------------------------------------------- host side
struct State {
  Layout L; uint8_t* my = nullptr; uint8_t* peer[kWorld] = {nullptr, nullptr, nullptr, nullptr};
  ErrRec* err_host = nullptr; ErrRec* err_dev = nullptr; int rank = -1; bool attr_set = false;
};
State g_state;

#define CK(x) do { cudaError_t e_ = (x); TORCH_CHECK(e_ == cudaSuccess, "[moet_device_comm] CUDA error ", #x, ": ", cudaGetErrorString(e_)); } while (0)

std::vector<int64_t> init_region(int64_t rank, int64_t slot_bytes, int64_t arena_bytes, int64_t pieces) {
  TORCH_CHECK(pieces >= 1 && pieces <= kMaxPieces, "pieces 1..32");  // the flag layout's maximum; every launch chooses <= this
  TORCH_CHECK(g_state.my == nullptr, "already initialised");
  g_state.L = make_layout((size_t)slot_bytes, (size_t)arena_bytes, (int)pieces);
  g_state.rank = (int)rank;
  CK(cudaMalloc(&g_state.my, g_state.L.total));
  CK(cudaMemset(g_state.my, 0, g_state.L.total));
  CK(cudaHostAlloc(&g_state.err_host, sizeof(ErrRec), cudaHostAllocMapped | cudaHostAllocPortable));
  memset(g_state.err_host, 0, sizeof(ErrRec));
  CK(cudaHostGetDevicePointer((void**)&g_state.err_dev, g_state.err_host, 0));
  for (int r = 0; r < kWorld; ++r) g_state.peer[r] = nullptr;
  g_state.peer[rank] = g_state.my;
  CK(cudaFuncSetAttribute(comm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  g_state.attr_set = true;
  CK(cudaDeviceSynchronize());
  return {(int64_t)g_state.L.total, (int64_t)g_state.L.slots, (int64_t)g_state.L.arena, (int64_t)g_state.L.slot_bytes, (int64_t)g_state.L.arena_bytes};
}
py::bytes ipc_handle() {
  TORCH_CHECK(g_state.my != nullptr, "not initialised");
  cudaIpcMemHandle_t h; CK(cudaIpcGetMemHandle(&h, g_state.my));
  return py::bytes(reinterpret_cast<const char*>(&h), sizeof(h));
}
void open_peer(int64_t r, py::bytes handle) {
  TORCH_CHECK(g_state.my != nullptr, "not initialised");
  if (r == g_state.rank) return;
  std::string s = handle; TORCH_CHECK(s.size() == sizeof(cudaIpcMemHandle_t), "bad handle size");
  cudaIpcMemHandle_t h; memcpy(&h, s.data(), sizeof(h));
  void* p = nullptr; CK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
  g_state.peer[r] = reinterpret_cast<uint8_t*>(p);
}
int64_t arena_ptr(int64_t g, int64_t par) {
  TORCH_CHECK(g_state.my != nullptr, "not initialised");
  return (int64_t)(g_state.my + off_arena(g_state.L, (int)g, (int)par));
}
// a tensor aliasing [rows, cols] of `like`'s dtype at byte offset `off` of arena[g][par] (no ownership: the region lives for the process)
torch::Tensor arena_view(int64_t g, int64_t par, int64_t off, int64_t rows, int64_t cols, torch::Tensor like) {
  TORCH_CHECK(g_state.my != nullptr, "not initialised");
  TORCH_CHECK(g >= 0 && g < kGroups && par >= 0 && par < kPar, "g / par");
  TORCH_CHECK(off >= 0 && (size_t)off + (size_t)rows * cols * like.element_size() <= g_state.L.arena_bytes, "the view exceeds the arena");
  return torch::from_blob(g_state.my + off_arena(g_state.L, (int)g, (int)par) + off, {rows, cols}, like.options());
}
std::vector<int64_t> func_info() {
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, comm_kernel));
  int per_sm = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, comm_kernel, kThreads, kSmemBytes));
  cudaFuncAttributes fr; CK(cudaFuncGetAttributes(&fr, reduce_kernel));
  return {(int64_t)fa.numRegs, (int64_t)fa.sharedSizeBytes, (int64_t)kSmemBytes, (int64_t)fa.localSizeBytes, (int64_t)per_sm, (int64_t)fr.numRegs};
}

static Args base_args(int kind, int64_t g, int64_t pseq, int64_t slot, int64_t use, int64_t arrive_target, int64_t red_target, int64_t pieces, int64_t n_push, int64_t n_red, int64_t unroll, int64_t link_token, double wd_s, int64_t c, int64_t row_bytes) {
  TORCH_CHECK(g_state.my != nullptr, "not initialised");
  for (int r = 0; r < kWorld; ++r) TORCH_CHECK(g_state.peer[r] != nullptr, "peer ", r, " not opened");
  TORCH_CHECK(n_push >= 1 && n_red >= 0 && n_push + n_red <= 1024, "bad CTA counts");
  TORCH_CHECK(row_bytes % 16 == 0, "row bytes must be a multiple of 16");
  TORCH_CHECK((size_t)c * row_bytes <= g_state.L.slot_bytes || kind == K_AG, "the shard exceeds the slot");
  TORCH_CHECK(slot >= 0 && slot < kSlots && (kind == K_RS) == (slot == 0), "slot 0 = RS, 1..4 = AG");
  TORCH_CHECK(use >= 1 && arrive_target >= n_push && red_target >= 1, "use / targets");
  TORCH_CHECK(pieces >= 1 && pieces <= g_state.L.pieces, "pieces 1..", g_state.L.pieces);
  Args a; memset(&a, 0, sizeof(a));
  a.kind = kind; a.rank = g_state.rank; a.g = (int)g; a.pseq = (uint32_t)pseq; a.slot = (int)slot; a.use = (uint32_t)use;
  a.arrive_target = (uint32_t)arrive_target; a.red_target = (uint32_t)red_target; a.n_push = (int)n_push; a.n_red = (int)n_red;
  a.pieces = (int)pieces; a.unroll = (int)unroll; a.link_token = (int)link_token; a.wd_ns = (uint64_t)(wd_s * 1e9);
  a.c = (int)c; a.row_bytes = (size_t)row_bytes; a.my = g_state.my;
  for (int r = 0; r < kWorld; ++r) a.peer[r] = g_state.peer[r];
  a.L = g_state.L; a.err = g_state.err_dev;
  return a;
}

static int g_push_threads = kThreads;
void set_push_warps(int64_t warps) { TORCH_CHECK(warps >= 1 && warps <= 4, "1..4 warps per pusher CTA"); g_push_threads = (int)warps * 32; }

void rs_launch(int64_t stream, torch::Tensor x, torch::Tensor out, torch::Tensor table, int64_t g, int64_t pseq, int64_t use, int64_t arrive_target,
               int64_t red_target, int64_t pieces, int64_t n_push, int64_t n_red, int64_t unroll, int64_t link_token, double wd_s) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.dtype() == torch::kBFloat16, "x: contiguous 2-D bf16 CUDA");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2 && out.dtype() == torch::kBFloat16, "out: contiguous 2-D bf16 CUDA");
  TORCH_CHECK(x.size(0) == kWorld * out.size(0) && x.size(1) == out.size(1), "x [4c, H], out [c, H]");
  const int64_t c = out.size(0), row_bytes = out.size(1) * 2;
  TORCH_CHECK(table.is_cuda() && table.dtype() == torch::kUInt8 && table.is_contiguous() && table.numel() >= c * row_bytes / 8, "table: u8 [>= c*H/4]");
  Args a = base_args(K_RS, g, pseq, 0, use, arrive_target, red_target, pieces, n_push, n_red, unroll, link_token, wd_s, c, row_bytes);
  a.x = reinterpret_cast<const uint8_t*>(x.data_ptr()); a.out = reinterpret_cast<uint8_t*>(out.data_ptr()); a.table = reinterpret_cast<const uint8_t*>(table.data_ptr());
  TORCH_CHECK(n_red == 0 || g_push_threads == kThreads, "multi-warp pushers only in split mode (n_red = 0)");
  comm_kernel<<<(unsigned)(n_push + n_red), (unsigned)(n_red == 0 ? g_push_threads : kThreads), kSmemBytes, (cudaStream_t)stream>>>(a);
  CK(cudaGetLastError());
}
// split mode: the bounded wait for the three sources' pieces (1 CTA) and the transient reduce (grid CTAs x 256), both on the comm stream
void rs_wait_reduce(int64_t stream, torch::Tensor x, torch::Tensor out, torch::Tensor table, int64_t g, int64_t pseq, int64_t use, int64_t red_target,
                    int64_t pieces, int64_t grid, double wd_s) {
  const int64_t c = out.size(0), row_bytes = out.size(1) * 2;
  Args a = base_args(K_RS, g, pseq, 0, use, 1, red_target, pieces, 1, 0, 4, 0, wd_s, c, row_bytes);
  a.x = reinterpret_cast<const uint8_t*>(x.data_ptr()); a.out = reinterpret_cast<uint8_t*>(out.data_ptr()); a.table = reinterpret_cast<const uint8_t*>(table.data_ptr());
  wait_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(a);
  CK(cudaGetLastError());
  reduce_kernel<<<(unsigned)grid, kRedThreads, 0, (cudaStream_t)stream>>>(a);
  CK(cudaGetLastError());
}
void ag_wait(int64_t stream, int64_t c, int64_t row_bytes, int64_t g, int64_t pseq, int64_t slot, int64_t use, int64_t pieces, double wd_s) {
  Args a = base_args(K_AG, g, pseq, slot, use, 1, 1, pieces, 1, 0, 4, 0, wd_s, c, row_bytes);
  wait_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(a);
  CK(cudaGetLastError());
}
void ag_launch(int64_t stream, torch::Tensor src, int64_t arena_off, int64_t g, int64_t pseq, int64_t slot, int64_t use, int64_t arrive_target,
               int64_t red_target, int64_t pieces, int64_t n_push, int64_t n_red, int64_t unroll, int64_t link_token, double wd_s) {
  TORCH_CHECK(src.is_cuda() && src.is_contiguous() && src.dim() == 2, "src: contiguous 2-D CUDA");
  const int64_t c = src.size(0), row_bytes = src.size(1) * src.element_size();
  TORCH_CHECK((size_t)arena_off + (size_t)kWorld * c * row_bytes <= g_state.L.arena_bytes, "the gather exceeds the arena");
  Args a = base_args(K_AG, g, pseq, slot, use, arrive_target, red_target, pieces, n_push, n_red, unroll, link_token, wd_s, c, row_bytes);
  a.src = reinterpret_cast<const uint8_t*>(src.data_ptr()); a.arena_off = (size_t)arena_off;
  TORCH_CHECK(n_red == 0 || g_push_threads == kThreads, "multi-warp pushers only in split mode (n_red = 0)");
  comm_kernel<<<(unsigned)(n_push + n_red), (unsigned)(n_red == 0 ? g_push_threads : kThreads), kSmemBytes, (cudaStream_t)stream>>>(a);
  CK(cudaGetLastError());
}
std::vector<int64_t> err_check() {
  if (g_state.err_host == nullptr) return {0, 0, 0, 0, 0, 0, 0, 0, 0};
  const ErrRec& e = *g_state.err_host;
  return {(int64_t)e.set, (int64_t)e.what, (int64_t)e.expected, (int64_t)e.seen, (int64_t)e.cta, (int64_t)e.rank, (int64_t)e.g, (int64_t)e.pseq, (int64_t)e.waited_ns};
}
// debug / tests: read a control word of my region
int64_t read_word(int64_t off) {
  TORCH_CHECK(g_state.my != nullptr, "not initialised");
  uint32_t v = 0; CK(cudaMemcpy(&v, g_state.my + off, 4, cudaMemcpyDeviceToHost)); return (int64_t)v;
}
std::vector<int64_t> ctrl_offsets(int64_t g) { return {(int64_t)off_ctrl(g_state.L, (int)g, 0), (int64_t)off_ctrl(g_state.L, (int)g, 1)}; }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_region", &init_region, "allocate my IPC region (rank, slot_bytes, arena_bytes, pieces) -> [total, slots_off, arena_off, slot_bytes, arena_bytes]");
  m.def("ipc_handle", &ipc_handle);
  m.def("open_peer", &open_peer);
  m.def("arena_ptr", &arena_ptr);
  m.def("arena_view", &arena_view, "(g, par, off, rows, cols, like) -> a tensor aliasing the arena");
  m.def("func_info", &func_info, "[regs, static smem, dynamic smem, local bytes, max CTAs per SM]");
  m.def("rs_launch", &rs_launch);
  m.def("ag_launch", &ag_launch);
  m.def("rs_wait_reduce", &rs_wait_reduce, "split mode: the bounded wait + the transient reduce on the comm stream");
  m.def("ag_wait", &ag_wait, "split mode: the bounded wait for the gathered pieces on the comm stream");
  m.def("set_push_warps", &set_push_warps, "1..4 warps per pusher CTA (split mode)");
  m.def("err_check", &err_check);
  m.def("read_word", &read_word);
  m.def("ctrl_offsets", &ctrl_offsets);
  m.attr("WORLD") = kWorld;
  m.attr("SMEM_BYTES") = kSmemBytes;
}
