"""Generates sparse_mla_spec_kernel.inc: FlashInfer's `sparse_mla_decode_dsv4_kernel` template copied verbatim from the
installed `kernels/fp8_decode/decode.cuh`, renamed `sparse_mla_decode_dsv4_spec_kernel`, with two insertions:

  1. after the kernel has its block ids and before it loads the candidate lengths: the gather threads load the index
     word of their first tile speculatively (the main-section word at this split's first chunk, and the extra-section
     word assuming full lengths - which section a split starts in depends on the lengths), so that load overlaps the
     lengths load instead of following it;
  2. the schedule's produce() call receives those words (and the starts they were loaded for); the schedule uses a
     word only when the section and start the lengths give match - the same memory word index() would read.

  3. the chunk loop's softmax / P-quantization / XV section (from the cross-warp reduction to the rope stage, pinned by
     the hash of the original text) with 3 barriers per tile instead of 4 + N_V_CHUNKS: the same expressions in the
     same order, evaluated by every thread that needs a value instead of through shared-memory round trips (16 threads
     reducing, a barrier, everyone loading), and the FP8 P x V-scale operand of all scale groups quantized before one
     barrier instead of one group per step with a barrier each; each step's accumulator is its own, so the sums are
     unchanged. Values the original passed through shared memory pass through an empty asm, so the compiler cannot
     combine the arithmetic across the boundary either.

Everything else in the kernel (the QK stage, the masking, the local reductions, the writeback) is the header's text:
`check()` regenerates and verifies that undoing the insertions and the replacement gives the header's function back
byte for byte. Runs at extension build time; a header the anchors or the hash do not fit fails loudly instead of
compiling something else.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

KERNEL_HEAD = "template <ModelType MT, int NUM_HEADS, typename GatherSchedule = Fp8DecodeGatherSchedule>\n__global__ void __launch_bounds__("
KERNEL_NAME = "    sparse_mla_decode_dsv4_kernel(\n"
NEW_NAME = "    sparse_mla_decode_dsv4_spec_kernel(\n"
SPEC_ANCHOR = "  const int valid_h = RUNTIME_H ? min(num_heads - h_start, HPB) : VALID_HPB;\n"
SPEC_BLOCK = '''  // [vllm-moet] speculative index loads for the gather group's first tile, issued before the lengths are known:
  // the main-section word at this split's first chunk, and the extra-section word assuming full lengths (which
  // section a split starts in depends on topk_len). produce() uses a word only if the section it was loaded from and
  // the start match what the lengths give; it is the same memory word index() reads, so the bits cannot differ.
  int spec_main = -1, spec_extra = -1, spec_main_start = -1, spec_extra_start = -1;
  if (threadIdx.x >= GatherSchedule::GATHER_BASE) {
    constexpr int SUB = kernels::dsv41_fp8::Dsv41MixedCacheDecodeResources::SUBTILES;
    constexpr int LW = Cfg::CAND_WINDOW * SUB;
    const int lane_s = threadIdx.x & 31;
    const int chunk0 = split_idx * chunks_per_block * SUB;
    const int main_full = (topk + LW - 1) / LW * SUB;
    const int main_start = chunk0 * Cfg::BI;
    if (main_start + lane_s < topk) {
      spec_main_start = main_start;
      spec_main = indices[(size_t)t_idx * indices_stride_elems + main_start + lane_s];
    }
    const int extra_start = (chunk0 - main_full) * Cfg::BI;
    if (extra_KV_cache != nullptr && chunk0 >= main_full && extra_start + lane_s < extra_topk) {
      spec_extra_start = extra_start;
      spec_extra = extra_indices[(size_t)t_idx * extra_indices_stride_elems + extra_start + lane_s];
    }
  }
'''
CALL_ANCHOR = "      GatherSchedule::produce(sm, smem_raw + RAW_OFFSET, section, chunk_hi - chunk_lo);\n"
CALL = ("      GatherSchedule::produce(sm, smem_raw + RAW_OFFSET, section, chunk_hi - chunk_lo, spec_main, spec_main_start,\n"
        "                              spec_extra, spec_extra_start);\n")

# 3. The softmax / P-quantization / XV section of the chunk loop, from the cross-warp reduction to the rope stage,
#    with its 4 + N_V_CHUNKS barriers per tile replaced by 3: the same expressions in the same order, evaluated by
#    every thread that needs the value instead of through shared memory round trips, and the FP8 operand of every
#    scale group quantized before one barrier instead of one group per step. The original block is pinned by hash.
SMEM_ANCHOR = "  const int32_t* idx_base = indices + (size_t)t_idx * indices_stride_elems;\n"
SMEM_BLOCK = '''  // [vllm-moet] buffers of the restructured XV phase, after the schedule's regions: the FP8 P x V-scale operand of
  // every scale group (the original keeps two and refills them per step) and the per-warp |P x V-scale| maxima.
  static_assert(GatherSchedule::RAW_PIPELINE, "the restructured XV phase is laid out for the mixed-cache schedule");
  constexpr size_t MOET_W_OFFSET =
      (GatherSchedule::template Layout<RAW_OFFSET>::TOTAL_BYTES + 127) / 128 * 128;
  constexpr size_t MOET_W_BYTES = (size_t)(KV::D_NOPE / KV::QUANT_TILE) * HPB * (Cfg::BI + 16);
  constexpr size_t MOET_AMAX_OFFSET = MOET_W_OFFSET + MOET_W_BYTES;
  uint8_t* moet_w_all = reinterpret_cast<uint8_t*>(smem_raw) + MOET_W_OFFSET;
  float* moet_amax = reinterpret_cast<float*>(smem_raw + MOET_AMAX_OFFSET);  // [group][head][warp]
'''
MATH_START = "    // Cross-warp reduce.\n"
MATH_END = "    // ── Stage 3 RoPE bf16 ─────────────────────────────────────\n"
MATH_ORIG_SHA256 = "a87b88fa25309c386908b4cb7da93afa71b3157a757b010f4b5b3e7e0d17ab53"
MATH_BLOCK = '''    // Cross-warp reduce. [vllm-moet] one barrier instead of two: every thread reduces its two heads' per-warp
    // partials itself, with the head threads' expressions in their order, instead of 16 threads reducing into
    // shared memory and a second barrier publishing the result. The reduced values pass through an empty asm so
    // the compiler treats them as the original's shared-memory loads (no algebra across the boundary).
    if (tid == 0) {
      sm.warp_max()[warp_id * HPB + gid] = local_max[0];
      sm.warp_max()[warp_id * HPB + gid + 8] = local_max[1];
      sm.warp_sum()[warp_id * HPB + gid] = local_sum[0];
      sm.warp_sum()[warp_id * HPB + gid + 8] = local_sum[1];
    }
    bar_sync_t<Cfg::MATH_BARRIER, Cfg::MATH_THREADS>();
    float block_local_max0, block_local_max1, block_local_sum0, block_local_sum1;
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int h = gid + half * 8;
      float wmax[Cfg::N_WARPS], wsum[Cfg::N_WARPS];
#pragma unroll
      for (int w = 0; w < Cfg::N_WARPS; w++) {
        wmax[w] = sm.warp_max()[w * HPB + h];
        wsum[w] = sm.warp_sum()[w * HPB + h];
      }
      float bmax = -1e30f;
#pragma unroll
      for (int w = 0; w < Cfg::N_WARPS; w++) bmax = fmaxf(bmax, wmax[w]);
      float bsum = 0.f;
#pragma unroll
      for (int w = 0; w < Cfg::N_WARPS; w++) bsum += wsum[w] * exp2f(wmax[w] - bmax);
      asm volatile("" : "+f"(bmax));
      asm volatile("" : "+f"(bsum));
      if (half == 0) {
        block_local_max0 = bmax;
        block_local_sum0 = bsum;
      } else {
        block_local_max1 = bmax;
        block_local_sum1 = bsum;
      }
    }

    // Online softmax update.
    float new_gmax0 = fmaxf(global_max[0], block_local_max0);
    float new_gmax1 = fmaxf(global_max[1], block_local_max1);
    const float alpha0 = (global_max[0] > -1e29f) ? exp2f(global_max[0] - new_gmax0) : 0.f;
    const float alpha1 = (global_max[1] > -1e29f) ? exp2f(global_max[1] - new_gmax1) : 0.f;
    // Two distinct rescale factors:
    //   block_rescale = exp(block_local_max - new_gmax) — rescales the
    //     block-wide sum (already weighted by per-warp local_max during
    //     cross-warp reduction) into the new global frame. Used for the
    //     global_sum update.
    //   warp_rescale  = exp(local_max[w] - new_gmax) — rescales THIS
    //     warp's p (computed in the warp's own local_max frame) into the
    //     new global frame, independently of the block-wide sum.
    const float block_rescale0 = exp2f(block_local_max0 - new_gmax0);
    const float block_rescale1 = exp2f(block_local_max1 - new_gmax1);
    const float warp_rescale0 = exp2f(local_max[0] - new_gmax0);
    const float warp_rescale1 = exp2f(local_max[1] - new_gmax1);

    if (chunk_idx > chunk_lo) {
#pragma unroll
      for (int vs = 0; vs < ACC_V_STEPS; vs++) {
#pragma unroll
        for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
          acc_nope[vs][nt][0] *= alpha0;
          acc_nope[vs][nt][1] *= alpha0;
          acc_nope[vs][nt][2] *= alpha1;
          acc_nope[vs][nt][3] *= alpha1;
        }
      }
      if constexpr (V_ROPE) {
#pragma unroll
        for (int nt = 0; nt < ROPE_N_TILES; nt++) {
          acc_rope[nt][0] *= alpha0;
          acc_rope[nt][1] *= alpha0;
          acc_rope[nt][2] *= alpha1;
          acc_rope[nt][3] *= alpha1;
        }
      }
      global_sum[0] = global_sum[0] * alpha0 + block_local_sum0 * block_rescale0;
      global_sum[1] = global_sum[1] * alpha1 + block_local_sum1 * block_rescale1;
    } else {
      global_sum[0] = block_local_sum0 * block_rescale0;
      global_sum[1] = block_local_sum1 * block_rescale1;
    }
    global_max[0] = new_gmax0;
    global_max[1] = new_gmax1;

    // Stage 2.75: rescale each warp's probabilities into the global max frame.
    // w_pre is shared: the XV-nope FP8 quantization below reads it too, so it
    // stays unconditional. Only the bf16 sm_p_full staging is rope-only.
    float w_pre[Cfg::QK_N_TILES][4];
#pragma unroll
    for (int nt = 0; nt < Cfg::QK_N_TILES; nt++) {
      w_pre[nt][0] = p[nt][0] * warp_rescale0;
      w_pre[nt][1] = p[nt][1] * warp_rescale0;
      w_pre[nt][2] = p[nt][2] * warp_rescale1;
      w_pre[nt][3] = p[nt][3] * warp_rescale1;
    }
    if constexpr (V_ROPE) {
      const int cand_col_base = warp_id * Cfg::ENTRIES_PER_WARP;
#pragma unroll
      for (int nt = 0; nt < Cfg::QK_N_TILES; nt++) {
        const int c0 = nt * 8 + tid * 2;
        const int c1 = c0 + 1;
        sm_p_full[gid][cand_col_base + c0] = __float2bfloat16(w_pre[nt][0]);
        sm_p_full[gid][cand_col_base + c1] = __float2bfloat16(w_pre[nt][1]);
        sm_p_full[gid + 8][cand_col_base + c0] = __float2bfloat16(w_pre[nt][2]);
        sm_p_full[gid + 8][cand_col_base + c1] = __float2bfloat16(w_pre[nt][3]);
      }
    }

    // ── Stage 3 NoPE FP8 ──────────────────────────────────────
    // [vllm-moet] per-(scale group, head) maximum of |P x V-scale| over the tile's candidates: the lanes sharing a
    // head reduce by shuffles and the warps' partials go through shared memory (the original: shared-memory atomicMax
    // of the float bits by every lane for every group, after a barrier for the zero-init). fmaxf over the same
    // non-negative values gives the same maximum in any order.
    static_assert(Cfg::QK_N_TILES == 1, "the per-head reductions below hold one candidate pair per lane");
    static_assert(SCALE_BYTES_PER_TOKEN == N_V_CHUNKS && N_V_CHUNKS == 16,
                  "a candidate's scale bytes are read as one 16-byte word");
    // a candidate's N_V_CHUNKS scale bytes as one 16-byte word (the original reads them one byte at a time; the
    // byte -> float conversion is the original's on the same byte)
    const int cand_e0 = warp_id * Cfg::ENTRIES_PER_WARP + tid * 2;
    const int cand_e1 = cand_e0 + 1;
    const uint4 scw0 = *reinterpret_cast<const uint4*>(sm_kv_sc + (size_t)cand_e0 * SCALE_BYTES_PER_TOKEN);
    const uint4 scw1 = *reinterpret_cast<const uint4*>(sm_kv_sc + (size_t)cand_e1 * SCALE_BYTES_PER_TOKEN);
    auto scale_byte = [](const uint4& w, int vc) -> uint8_t {
      const uint32_t word = vc < 4 ? w.x : (vc < 8 ? w.y : (vc < 12 ? w.z : w.w));
      return static_cast<uint8_t>(word >> (8 * (vc & 3)));
    };
    {
      // the phases kept apart so the 16 groups' chains overlap (the compiler serialized them when interleaved):
      // products, then the two shuffle rounds, then the stores
      float m0[N_V_CHUNKS], m1[N_V_CHUNKS];
#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; vc++) {
        const float vsc0 = fp32_from_exponent_byte(scale_byte(scw0, vc));
        const float vsc1 = fp32_from_exponent_byte(scale_byte(scw1, vc));
        m0[vc] = fmaxf(fabsf(w_pre[0][0] * vsc0), fabsf(w_pre[0][1] * vsc1));
        m1[vc] = fmaxf(fabsf(w_pre[0][2] * vsc0), fabsf(w_pre[0][3] * vsc1));
      }
#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; vc++) m0[vc] = fmaxf(m0[vc], __shfl_xor_sync(0xffffffff, m0[vc], 1));
#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; vc++) m1[vc] = fmaxf(m1[vc], __shfl_xor_sync(0xffffffff, m1[vc], 1));
#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; vc++) m0[vc] = fmaxf(m0[vc], __shfl_xor_sync(0xffffffff, m0[vc], 2));
#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; vc++) m1[vc] = fmaxf(m1[vc], __shfl_xor_sync(0xffffffff, m1[vc], 2));
      if (tid == 0) {
#pragma unroll
        for (int vc = 0; vc < N_V_CHUNKS; vc++) {
          moet_amax[(vc * HPB + gid) * Cfg::N_WARPS + warp_id] = m0[vc];
          moet_amax[(vc * HPB + gid + 8) * Cfg::N_WARPS + warp_id] = m1[vc];
        }
      }
    }
    bar_sync_t<Cfg::MATH_BARRIER, Cfg::MATH_THREADS>();
    // the head scales, two per thread as in the original's normalization pass (its loop and expression, on the
    // maximum of the warps' partials instead of the atomics' result), into the original's shared-memory array
    for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += Cfg::MATH_THREADS) {
      const float4 pm = *reinterpret_cast<const float4*>(&moet_amax[(size_t)i * Cfg::N_WARPS]);
      sm.w_head_sc()[i] = fmaxf(fmaxf(fmaxf(fmaxf(pm.x, pm.y), pm.z), pm.w), 1e-10f) / FP8_MAX;
    }
    bar_sync_t<Cfg::MATH_BARRIER, Cfg::MATH_THREADS>();

    // [vllm-moet] the FP8 P x V-scale operand of every scale group, then one barrier (the original: one group per
    // step, a barrier between the group's quantization and its MMAs, N_V_CHUNKS steps). Same expressions per element.
    {
      // the reciprocals of this thread's two heads' scales first (independent loads and MUFU.RCP, overlapped),
      // then every group's four conversions (the original's expressions)
      float si0[N_V_CHUNKS], si1[N_V_CHUNKS];
#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; vc++) {
        si0[vc] = 1.f / sm.w_head_sc()[vc * HPB + gid];
        si1[vc] = 1.f / sm.w_head_sc()[vc * HPB + gid + 8];
      }
#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; vc++) {
        uint8_t* sm_w_fp8 = moet_w_all + (size_t)vc * HPB * W_FP8_STRIDE;
        const float vsc0 = fp32_from_exponent_byte(scale_byte(scw0, vc));
        const float vsc1 = fp32_from_exponent_byte(scale_byte(scw1, vc));
        __nv_fp8_e4m3 f00(fmaxf(FP8_MIN, fminf(FP8_MAX, w_pre[0][0] * vsc0 * si0[vc])));
        __nv_fp8_e4m3 f01(fmaxf(FP8_MIN, fminf(FP8_MAX, w_pre[0][1] * vsc1 * si0[vc])));
        __nv_fp8_e4m3 f10(fmaxf(FP8_MIN, fminf(FP8_MAX, w_pre[0][2] * vsc0 * si1[vc])));
        __nv_fp8_e4m3 f11(fmaxf(FP8_MIN, fminf(FP8_MAX, w_pre[0][3] * vsc1 * si1[vc])));
        sm_w_fp8[(size_t)gid * W_FP8_STRIDE + cand_e0] = f00.__x;
        sm_w_fp8[(size_t)gid * W_FP8_STRIDE + cand_e1] = f01.__x;
        sm_w_fp8[(size_t)(gid + 8) * W_FP8_STRIDE + cand_e0] = f10.__x;
        sm_w_fp8[(size_t)(gid + 8) * W_FP8_STRIDE + cand_e1] = f11.__x;
      }
    }
    bar_sync_t<Cfg::MATH_BARRIER, Cfg::MATH_THREADS>();

    // Phase 4 FP8 MMA, every step without a barrier: the original's warp / chunk / dim mapping, its MMA helper and
    // its epilogue per step accumulator (each step owns its accumulator, so the step order does not enter the sums).
#pragma unroll
    for (int vs = 0; vs < ACC_V_STEPS; vs++) {
      const int vc0 = vs * XV_FOLD;
      const int vc = vc0 + warp_id / XV_WARPS;
      uint8_t* sm_w_fp8 = moet_w_all + (size_t)vc * HPB * W_FP8_STRIDE;
      const float sc0 = sm.w_head_sc()[vc * HPB + gid];
      const float sc1 = sm.w_head_sc()[vc * HPB + gid + 8];
#pragma unroll
      for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
        const int dim = vc * V_CHUNK + (warp_id % XV_WARPS) * (NT_PER_WARP_XV * 8) + nt * 8;
        float xv[4] = {0.f, 0.f, 0.f, 0.f};
        pv_fp8_d2_16x8<KV_SMEM_STRIDE, W_FP8_STRIDE, XV_KSTEPS>(xv, sm_w_fp8, sm_kv_fp8, dim, lane);
        acc_nope[vs][nt][0] += xv[0] * sc0;
        acc_nope[vs][nt][1] += xv[1] * sc0;
        acc_nope[vs][nt][2] += xv[2] * sc1;
        acc_nope[vs][nt][3] += xv[3] * sc1;
      }
    }

'''


def header_path() -> Path:
    import flashinfer

    return (Path(flashinfer.__file__).resolve().parent / "data/include/flashinfer/attention/sparse_mla_sm120"
            / "kernels/fp8_decode/decode.cuh")


def kernel_text(header: str) -> str:
    """The kernel template's text, from its `template <...>` line to the function's closing brace."""
    start = header.index(KERNEL_HEAD)
    name_at = header.index(KERNEL_NAME, start)
    # the function body ends at the first line that is exactly "}" after the opening of the body
    body_open = header.index(") {\n", name_at)
    end = header.index("\n}\n", body_open) + len("\n}\n")
    return header[start:end]


def math_block(text: str) -> str:
    a = text.index(MATH_START)
    b = text.index(MATH_END, a)
    return text[a:b]


def generate(header: str) -> str:
    text = kernel_text(header)
    for anchor, what in ((KERNEL_NAME, "kernel name"), (SPEC_ANCHOR, "valid_h line"), (CALL_ANCHOR, "produce call"),
                         (SMEM_ANCHOR, "idx_base line"), (MATH_START, "cross-warp reduce"), (MATH_END, "rope stage")):
        if text.count(anchor) != 1:
            raise RuntimeError(f"decode.cuh: anchor '{what}' not found once in sparse_mla_decode_dsv4_kernel - FlashInfer changed")
    if hashlib.sha256(math_block(text).encode()).hexdigest() != MATH_ORIG_SHA256:
        raise RuntimeError("decode.cuh: the softmax / XV block of sparse_mla_decode_dsv4_kernel is not the one the "
                           "restructured block was derived from - FlashInfer changed")
    # the kernel parameters / locals the insertion reads (the schedule's and Cfg's members are checked by the compiler)
    for sym in ("Cfg::BI", "chunks_per_block", "indices_stride_elems", "extra_indices_stride_elems", "extra_KV_cache",
                "extra_topk", "split_idx", "t_idx", "using Cfg = "):
        if sym not in text:
            raise RuntimeError(f"decode.cuh: '{sym}' not in the kernel - FlashInfer changed")
    orig_math = math_block(text)
    out = text.replace(KERNEL_NAME, NEW_NAME, 1)
    out = out.replace(SPEC_ANCHOR, SPEC_ANCHOR + SPEC_BLOCK, 1)
    out = out.replace(CALL_ANCHOR, CALL, 1)
    out = out.replace(SMEM_ANCHOR, SMEM_BLOCK + SMEM_ANCHOR, 1)
    out = out.replace(orig_math, MATH_BLOCK, 1)
    return ("// Generated by gen_spec_kernel.py from the installed FlashInfer decode.cuh: its sparse_mla_decode_dsv4_kernel\n"
            "// with the [vllm-moet] insertions and the restructured softmax / XV block (see gen_spec_kernel.py). Do not edit.\n"
            + out)


def stripped(generated: str, header: str) -> str:
    """The generated text with the insertions and the block replacement undone (what check() compares with the
    header's kernel)."""
    out = generated.split("Do not edit.\n", 1)[1]
    out = out.replace(SPEC_ANCHOR + SPEC_BLOCK, SPEC_ANCHOR, 1).replace(CALL, CALL_ANCHOR, 1)
    out = out.replace(SMEM_BLOCK + SMEM_ANCHOR, SMEM_ANCHOR, 1)
    out = out.replace(MATH_BLOCK, math_block(kernel_text(header)), 1)
    return out.replace(NEW_NAME, KERNEL_NAME, 1)


def write(out_dir: Path) -> Path:
    header = header_path().read_text()
    gen = generate(header)
    if stripped(gen, header) != kernel_text(header):
        raise RuntimeError("gen_spec_kernel: the generated kernel does not reduce to the header's (generator bug)")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sparse_mla_spec_kernel.inc"
    if not path.exists() or path.read_text() != gen:
        path.write_text(gen)
    return path


def check(path: Path) -> None:
    """The generated file on disk is the installed header's kernel plus exactly the two insertions."""
    header = header_path().read_text()
    gen = path.read_text()
    if gen != generate(header) or stripped(gen, header) != kernel_text(header):
        raise RuntimeError(f"{path}: not the installed FlashInfer kernel plus the recorded insertions / replacement")


if __name__ == "__main__":
    p = write(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)
    check(p)
    print(f"generated {p} ({len(p.read_text().splitlines())} lines) from {header_path()}")
