#!/usr/bin/env python3
"""Generate sass/sparse_mla_prefill_v41.sass — DeepSeek-V4.1 sparse-MLA prefill attention on sm_120, bit-identical with
spec B = FlashInfer 0.7.2 `sparse_mla_prefill_dsv41_bf16_kernel<16, true>` (front "cubit + prefill", Phase 1b, session V).

One CTA = one token x 16 heads, 8 warps, 1 CTA/SM (smem 87 040 B > 100 KB / 2). Per 64-candidate chunk of the SWA cache
(528-byte fp8 records: 512 e4m3 + 16 UE8M0 scales per 32) and then of the compressed cache (288-byte FP4 records: 256 B of
e2m1 nibbles + 32 E4M3 scales per 16), both with the scales after each row's data (interleaved placement):
  gather  : the raw records of chunk c+1 are loaded into 40 staging registers (thread = 16 B of 8 rows: one LDG.128 (fp8)
            or LDG.64 (fp4) + one scale byte per row, all 16 loads of the chunk in flight at once) right after the barrier
            that publishes chunk c's tile, i.e. BEFORE chunk c's QK/softmax/PV; at the top of the next iteration they are
            dequantized to BF16 in smem exactly as spec B (e4m3/e2m1 -> f16 -> f32 -> bf16, x bf16 scale, HMUL2.BF16_V2;
            every product is exact). Spec B: 8 dependent gather round trips per chunk, nothing in flight during compute;
  QK      : HMMA.16816.F32.BF16, warp = 8 candidates, k = 0..512 step 16 in order, A = q (LDSM.x4), B = 2 x LDS;
  softmax : qk * (sm_scale*log2e) [FMUL.FTZ], -1e30 for invalid candidates, max via SHFL 1,2 + smem across warps 0..7,
            alpha = EX2(m_old - m_new), p = EX2(qk - m) [MUFU], P -> bf16 (F2FP.BF16.F32.PACK_AB) to smem, row sums
            lane-pair -> SHFL 1,2 -> warps 0..7: sum = FFMA(alpha, sum, r0) + r1 + ... + r7, acc *= alpha;
  PV      : HMMA, A = P (LDSM.x4), B = V (LDSM.MT88.2 from the same bf16 tile), warp = 64 columns of D_V, k order 0..64;
  epilogue: norm = RCP(sum) or 0, lse = LG2(sum) + m or -inf, optional sink (peak/weight/total as spec B), bf16 output,
            lse written by warp 0 lane%4 == 0 and scaled by lse_scale unless -1e30.
The arithmetic instructions, their modifiers (FTZ, MUFU approximations) and the reduction orders are those of the frozen
SASS of spec B (V/census/specB_16_true.frozen.sass). No control codes in the output: the assembler schedules (assemble with
CUBIT_MMA_WAIT=1 CUBIT_MMA_WB_DRAIN=13: the ptxas discipline for the cooperative HMMA chain); frozen prefixes may be added
per instruction once the bits are proven. No predicated write ever targets a register with a result still in flight (SEL /
FSEL after unpredicated math) and no MUFU destination is reused as an address: both tripped cubit's SCHED2 stall model.

Registers stay within R0-R125 (regcount 128): cubit's scheduler keys predicates as "register >= 128" and does not track GPRs
R128+ (V/cubit/BUG_sched_gpr128.md) — the spec kernel itself uses 108.

smem (dynamic, 87 040 B): q bf16 [16][520] @0, kv bf16 [64][520] @0x4100, P bf16 [16][72] @0x14500, reduce f32 [2][8][16]
@0x14e00, valid i32 [2][64] @0x15200 (double-buffered, permuted: row r at (r & 7) * 8 + (r >> 3) so a thread's 8 rows are
one 32-byte run).

Launch: grid = num_tokens, block = 256, dynamic smem 87 040 B. Parameters (in order; all pointers 64-bit, scalars u32,
page sizes as log2): q, kv, indices, topk_length (0 -> topk), extra_kv (0 -> no compressed cache), extra_indices,
extra_topk_length, output, out_lse (0 -> not written), attn_sink (0 -> none), num_tokens, topk, extra_topk,
page_shift, extra_page_shift, page_stride (bytes), extra_page_stride, indices_stride (elems), extra_indices_stride,
out_lse_stride, sm_scale (f32 bits), lse_scale (f32 bits).
"""
import os
import sys

OUT = []
STOP_AFTER = os.environ.get("SMLA_STOP_AFTER", "")   # debug: insert EXIT after a phase (prologue|dequant|issue|qk|softmax|pv)


def L(s=""):
    OUT.append(s)


def stop(phase):
    if STOP_AFTER == phase:
        L("    BAR.SYNC.DEFER_BLOCKING 0x0 ;")
        L("    EXIT ;")


# ---------------------------------------------------------------- geometry
HEADS, D, CAND, WARPS = 16, 512, 64, 8
QSTRIDE = (D + 8) * 2          # 1040 B per q / kv row (spec B: QK_STRIDE_ELEMS = 520)
PSTRIDE = (CAND + 8) * 2       # 144 B per P row
SM_Q, SM_KV = 0x0, HEADS * QSTRIDE                      # 0x4100
SM_P = SM_KV + CAND * QSTRIDE                            # 0x14500
SM_RED = SM_P + HEADS * PSTRIDE                          # 0x14e00  [2][8][16] f32
SM_VALID = SM_RED + 2 * WARPS * HEADS * 4                # 0x15200  [2][64] i32 (permuted, double-buffered)
SMEM = SM_VALID + 2 * CAND * 4                           # 0x15400 = 87 040
assert SM_KV == 0x4100 and SM_P == 0x14500 and SM_RED == 0x14e00 and SM_VALID == 0x15200 and SMEM == 87040

# ---------------------------------------------------------------- parameters (c[0x0][0x380 + ...])
PARAMS = [("q", 8), ("kv", 8), ("indices", 8), ("topk_length", 8), ("extra_kv", 8), ("extra_indices", 8),
          ("extra_topk_length", 8), ("output", 8), ("out_lse", 8), ("attn_sink", 8),
          ("num_tokens", 4), ("topk", 4), ("extra_topk", 4), ("page_shift", 4), ("extra_page_shift", 4),
          ("page_stride", 4), ("extra_page_stride", 4), ("indices_stride", 4), ("extra_indices_stride", 4),
          ("out_lse_stride", 4), ("sm_scale", 4), ("lse_scale", 4)]
POFF = {}
_o = 0x380
for _n, _sz in PARAMS:
    POFF[_n] = _o
    _o += _sz


def c(name, off=0):
    return f"c[0x0][{POFF[name] + off:#x}]"


# ---------------------------------------------------------------- registers (R0-R125; R126/R127 = regcount shadow)
# R0-R11 scratch (phase temps); dequant: outputs R4-R11, scale R28, temps R29-R31, R72-R75
R_TID, R_WARP, R_LANE, R_GID, R_T4 = 12, 13, 14, 15, 16
R_TOK, R_LEN, R_ELEN, R_MAINC, R_CHUNKS, R_C, R_SCALE2 = 17, 18, 19, 20, 21, 22, 23
R_M0, R_M1, R_S0, R_S1 = 24, 25, 26, 27
R_QK = 28                      # R28-R31 (QK accumulator; gather/issue temps outside QK)
R_ACC = 32                     # acc[v][j] = R32 + 4v + j, v = 0..7 -> R32-R63


def acc(v, j):
    return R_ACC + 4 * v + j


R_A0, R_A1, R_B0, R_B1 = 64, 68, 72, 74                  # fragments R64-R75 (issue phase: idx R64-R71, temps R72-R75)
R_QA, R_KB, R_PST, R_PA, R_VB, R_RED0, R_REDR, R_VALR, R_GSTS, R_GIDX = 76, 77, 78, 79, 80, 81, 82, 83, 84, 85
# staging of the next chunk: 8 tasks x (data quad + scale byte) = 40 registers R86-R125
R_STGQ = [88 + 4 * k for k in range(8)]                  # quads R88..R119
R_STGS = [86, 87, 120, 121, 122, 123, 124, 125]          # scale bytes
NREGS = 126
# chunk parameters live in R28-R31 / R72-R75 between the dequant of chunk c and the QK of chunk c (issue of c+1)
P_CACHE, P_SHIFT, P_MASK, P_REC, P_PSTR, P_OFFD, P_OFFS = 28, 30, 31, 72, 73, 74, 75   # P_CACHE = R28:R29
# epilogue-only (the staging window is free then)
R_OTOK, R_LSETOK, R_SINK, R_LSESC = 64, 66, 68, 70      # fragment window (free in the epilogue; never a load target)

# ---------------------------------------------------------------- header
L("// sparse_mla_prefill_v41 — DeepSeek-V4.1 sparse-MLA prefill attention, bit-identical with spec B")
L("// (FlashInfer 0.7.2 sparse_mla_prefill_dsv41_bf16_kernel<16,true>). GENERATED by gen/gen_sparse_mla_prefill_v41.py —")
L("// edit the generator, not this file. 1 token x 16 heads per CTA, 8 warps, dynamic smem 87040 B (q bf16 16x520, kv bf16")
L("// 64x520, P bf16 16x72, reduce 2x8x16 f32, valid[2][64] permuted). The raw records of chunk c+1 are in flight during")
L("// chunk c's QK/softmax/PV (40 staging registers).")
L(".entry sparse_mla_prefill_v41")
L(f"    .reg R0-R{NREGS - 1}")
for n, sz in PARAMS:
    L(f"    .param u{sz * 8} {n}")
L()

# ---------------------------------------------------------------- prologue
L("    // ---- prologue: ids, token guard, lengths ----")
L(f"    S2R R{R_TID}, SR_TID.X ;")
L(f"    S2R R{R_TOK}, SR_CTAID.X ;")
L("    LDCU.64 UR4, c[0x0][0x358] ;")
L(f"    LDC R0, {c('num_tokens')} ;")
L(f"    ISETP.GE.U32.AND P0, PT, R{R_TOK}, R0, PT ;")
L("    @P0 EXIT ;")
L(f"    SHF.R.U32.HI R{R_WARP}, RZ, 0x5, R{R_TID} ;")
L(f"    LOP3.LUT R{R_LANE}, R{R_TID}, 0x1f, RZ, 0xc0, !PT ;")
L(f"    SHF.R.U32.HI R{R_GID}, RZ, 0x2, R{R_LANE} ;")
L(f"    LOP3.LUT R{R_T4}, R{R_LANE}, 0x3, RZ, 0xc0, !PT ;")
# len = topk_length ? max(0, min(topk, topk_length[t])) : topk
L(f"    LDC R{R_LEN}, {c('topk')} ;")
L(f"    LDC.64 R0, {c('topk_length')} ;")
L("    ISETP.NE.U32.AND P0, PT, R0, RZ, PT ;")
L("    ISETP.NE.U32.OR P0, PT, R1, RZ, P0 ;")
L(f"    IMAD.WIDE.U32 R0, PT, R{R_TOK}, 0x4, R0 ;")
L("    @P0 LDG.E R2, desc[UR4][R0.64] ;")
L(f"    @P0 VIMNMX.S32 R2, R2, R{R_LEN}, PT ;")      # min(topk_length, topk)
L(f"    @P0 VIMNMX.S32 R{R_LEN}, R2, RZ, !PT ;")     # max(., 0)
# elen = extra_kv ? (extra_topk_length ? clamp : extra_topk) : 0
L(f"    LDC R{R_ELEN}, {c('extra_topk')} ;")
L(f"    LDC.64 R0, {c('extra_topk_length')} ;")
L("    ISETP.NE.U32.AND P0, PT, R0, RZ, PT ;")
L("    ISETP.NE.U32.OR P0, PT, R1, RZ, P0 ;")
L(f"    IMAD.WIDE.U32 R0, PT, R{R_TOK}, 0x4, R0 ;")
L("    @P0 LDG.E R2, desc[UR4][R0.64] ;")
L(f"    @P0 VIMNMX.S32 R2, R2, R{R_ELEN}, PT ;")
L(f"    @P0 VIMNMX.S32 R{R_ELEN}, R2, RZ, !PT ;")
L(f"    LDC.64 R0, {c('extra_kv')} ;")
L("    ISETP.NE.U32.AND P1, PT, R0, RZ, PT ;")
L("    ISETP.NE.U32.OR P1, PT, R1, RZ, P1 ;")
L(f"    @!P1 MOV R{R_ELEN}, RZ ;")
L(f"    IADD3 R{R_MAINC}, PT, PT, R{R_LEN}, 0x3f, RZ ;")
L(f"    SHF.R.U32.HI R{R_MAINC}, RZ, 0x6, R{R_MAINC} ;")
L(f"    IADD3 R{R_CHUNKS}, PT, PT, R{R_ELEN}, 0x3f, RZ ;")
L(f"    SHF.R.U32.HI R{R_CHUNKS}, RZ, 0x6, R{R_CHUNKS} ;")
L(f"    IADD3 R{R_CHUNKS}, PT, PT, R{R_CHUNKS}, R{R_MAINC}, RZ ;")
L(f"    LDC R0, {c('sm_scale')} ;")
L(f"    FMUL.FTZ R{R_SCALE2}, R0, 1.4426950216293334961 ;")                     # sm_scale * log2e (spec B: UFMUL.FTZ)
L()
L("    // ---- q -> smem (16 x 512 bf16, row stride 1040 B): thread = 4 vectors of 16 B ----")
L(f"    LDC.64 R{R_STGQ[0]}, {c('q')} ;")                                        # q base in the staging window (free here)
L(f"    IMAD.WIDE.U32 R{R_STGQ[0]}, PT, R{R_TOK}, 0x4000, R{R_STGQ[0]} ;")       # q + t * 16 heads * 512 * 2
for k in range(4):
    L(f"    IADD3 R0, PT, PT, R{R_TID}, {256 * k:#x}, RZ ;")                      # i
    L(f"    IMAD.WIDE.U32 R2, PT, R0, 0x10, R{R_STGQ[0]} ;")                     # q_tok + i*16
    L(f"    LDG.E.128 R{4 + 4 * (k % 2)}, desc[UR4][R2.64] ;")
    L("    SHF.R.U32.HI R1, RZ, 0x6, R0 ;")                                      # h
    L("    LOP3.LUT R0, R0, 0x3f, RZ, 0xc0, !PT ;")                              # i & 63
    L(f"    IMAD R1, R1, {QSTRIDE:#x}, RZ ;")
    L("    IMAD R1, R0, 0x10, R1 ;")                                             # h*1040 + vec*16
    L(f"    STS.128 [R1+{SM_Q:#x}], R{4 + 4 * (k % 2)} ;")
L()
L("    // ---- per-thread smem addresses ----")
L(f"    LOP3.LUT R0, R{R_LANE}, 0x7, RZ, 0xc0, !PT ;")
L(f"    LOP3.LUT R1, R{R_LANE}, 0x8, RZ, 0xc0, !PT ;")
L("    IADD3 R0, PT, PT, R0, R1, RZ ;")                                          # ldmatrix row
L(f"    SHF.R.U32.HI R1, RZ, 0x4, R{R_LANE} ;")                                  # lane >> 4
L(f"    IMAD R{R_QA}, R0, {QSTRIDE:#x}, RZ ;")
L(f"    IMAD R{R_QA}, R1, 0x10, R{R_QA} ;")
L(f"    IMAD R{R_PA}, R0, {PSTRIDE:#x}, RZ ;")
L(f"    IMAD R{R_PA}, R1, 0x10, R{R_PA} ;")
L(f"    IADD3 R{R_PA}, PT, PT, R{R_PA}, {SM_P:#x}, RZ ;")
L(f"    IMAD R0, R{R_WARP}, 0x8, R{R_GID} ;")
L(f"    IMAD R{R_KB}, R0, {QSTRIDE:#x}, RZ ;")
L(f"    IMAD R{R_KB}, R{R_T4}, 0x4, R{R_KB} ;")
L(f"    IADD3 R{R_KB}, PT, PT, R{R_KB}, {SM_KV:#x}, RZ ;")
L(f"    IMAD R0, R{R_WARP}, 0x8, RZ ;")
L(f"    IMAD R0, R{R_T4}, 0x2, R0 ;")
L(f"    IMAD R{R_PST}, R{R_GID}, {PSTRIDE:#x}, RZ ;")
L(f"    IMAD R{R_PST}, R0, 0x2, R{R_PST} ;")
L(f"    IADD3 R{R_PST}, PT, PT, R{R_PST}, {SM_P:#x}, RZ ;")
L(f"    LOP3.LUT R0, R{R_LANE}, 0xf, RZ, 0xc0, !PT ;")
L(f"    IMAD R{R_VB}, R0, {QSTRIDE:#x}, RZ ;")
L(f"    IMAD R{R_VB}, R{R_WARP}, 0x10, R{R_VB} ;")
L(f"    IADD3 R{R_VB}, PT, PT, R{R_VB}, {SM_KV:#x}, RZ ;")
L(f"    IMAD R{R_RED0}, R{R_WARP}, 0x40, RZ ;")
L(f"    IMAD R{R_RED0}, R{R_GID}, 0x4, R{R_RED0} ;")
L(f"    IADD3 R{R_RED0}, PT, PT, R{R_RED0}, {SM_RED:#x}, RZ ;")
L(f"    IMAD R{R_REDR}, R{R_GID}, 0x4, RZ ;")
L(f"    IADD3 R{R_REDR}, PT, PT, R{R_REDR}, {SM_RED:#x}, RZ ;")
# mask read: candidates 8w + 2t4 (+1) sit at permuted positions 16*t4 + w (+8) -> byte (16*t4 + w)*4 (+0x20)
L(f"    IMAD R0, R{R_T4}, 0x10, R{R_WARP} ;")
L(f"    IMAD R{R_VALR}, R0, 0x4, RZ ;")
L(f"    IADD3 R{R_VALR}, PT, PT, R{R_VALR}, {SM_VALID:#x}, RZ ;")
# gather: row group g = tid >> 5 (rows g + 8k), pair j = tid & 31 (elements 16j..16j+15 -> bf16 byte offset 32j)
L(f"    SHF.R.U32.HI R0, RZ, 0x5, R{R_TID} ;")                                   # g
L(f"    LOP3.LUT R1, R{R_TID}, 0x1f, RZ, 0xc0, !PT ;")                           # j
L(f"    IMAD R{R_GSTS}, R0, {QSTRIDE:#x}, RZ ;")
L(f"    IMAD R{R_GSTS}, R1, 0x20, R{R_GSTS} ;")
L(f"    IADD3 R{R_GSTS}, PT, PT, R{R_GSTS}, {SM_KV:#x}, RZ ;")                   # KV + g*1040 + j*32 (+ k*8*1040)
L(f"    IMAD R{R_GIDX}, R0, 0x20, RZ ;")
L(f"    IADD3 R{R_GIDX}, PT, PT, R{R_GIDX}, {SM_VALID:#x}, RZ ;")                # VALID + g*32 (+ buffer*0x100)
L()
L("    // ---- state ----")
L(f"    MOV R{R_M0}, 0xf149f2ca ;")                                              # -1e30f
L(f"    MOV R{R_M1}, 0xf149f2ca ;")
L(f"    MOV R{R_S0}, RZ ;")
L(f"    MOV R{R_S1}, RZ ;")
for v in range(8):
    L(f"    CS2R R{acc(v, 0)}, SRZ ;")
    L(f"    CS2R R{acc(v, 2)}, SRZ ;")
L(f"    MOV R{R_C}, RZ ;")
stop("prologue")
L(f"    ISETP.EQ.AND P0, PT, R{R_CHUNKS}, RZ, PT ;")
L("    @P0 BRA epilogue ;")
L()


def emit_params_and_valid(cidx_reg: int):
    """Chunk parameters of chunk `cidx_reg` into R28-R31 / R72-R75 (kept across the barrier), valid[] of that chunk
    (tid < 64) into the buffer (cidx & 1), and P6 = compressed chunk. Scratch: R0-R11."""
    L(f"    ISETP.GE.U32.AND P6, PT, R{cidx_reg}, R{R_MAINC}, PT ;")              # P6 = compressed (fp4) chunk
    L(f"    IADD3 R0, PT, PT, R{cidx_reg}, -R{R_MAINC}, RZ ;")
    L(f"    SEL R0, R0, R{cidx_reg}, P6 ;")
    L("    SHF.L.U32 R9, R0, 0x6, RZ ;")                                         # start = chunk_in_cache * 64
    L(f"    SEL R8, R{R_ELEN}, R{R_LEN}, P6 ;")                                   # n
    L(f"    LDC.64 R0, {c('indices')} ;")
    L(f"    LDC R2, {c('indices_stride')} ;")
    L(f"    LDC.64 R4, {c('extra_indices')} ;")
    L(f"    LDC R6, {c('extra_indices_stride')} ;")
    L("    SEL R0, R4, R0, P6 ;")
    L("    SEL R1, R5, R1, P6 ;")
    L("    SEL R2, R6, R2, P6 ;")
    L(f"    IMAD R2, R{R_TOK}, R2, RZ ;")
    L("    IMAD.WIDE.U32 R6, PT, R2, 0x4, R0 ;")                                  # R6:R7 = index row of this token
    L(f"    LDC.64 R0, {c('kv')} ;")
    L(f"    LDC.64 R4, {c('extra_kv')} ;")
    L(f"    SEL R{P_CACHE}, R4, R0, P6 ;")
    L(f"    SEL R{P_CACHE + 1}, R5, R1, P6 ;")
    L(f"    LDC R0, {c('page_shift')} ;")
    L(f"    LDC R1, {c('extra_page_shift')} ;")
    L(f"    SEL R{P_SHIFT}, R1, R0, P6 ;")
    L(f"    LDC R0, {c('page_stride')} ;")
    L(f"    LDC R1, {c('extra_page_stride')} ;")
    L(f"    SEL R{P_PSTR}, R1, R0, P6 ;")
    L("    MOV R0, 0x210 ;")
    L("    MOV R1, 0x120 ;")
    L(f"    SEL R{P_REC}, R1, R0, P6 ;")                                          # record bytes 528 / 288
    L("    MOV R0, 0x1 ;")
    L(f"    SHF.L.U32 R{P_MASK}, R0, R{P_SHIFT}, RZ ;")
    L(f"    IADD3 R{P_MASK}, PT, PT, R{P_MASK}, -0x1, RZ ;")                      # page_size - 1
    L(f"    LOP3.LUT R0, R{R_TID}, 0x1f, RZ, 0xc0, !PT ;")                       # j
    L("    SHF.L.U32 R1, R0, 0x4, RZ ;")                                         # fp8 data offset 16j
    L("    LOP3.LUT R2, R1, 0xffffffe0, RZ, 0xc0, !PT ;")
    L("    SHF.R.U32.HI R2, RZ, 0x1, R2 ;")                                      # fp4: 16-B aligned 16*(j>>1) (half picked by j&1)
    L(f"    SEL R{P_OFFD}, R2, R1, P6 ;")
    L("    SHF.R.U32.HI R1, RZ, 0x1, R0 ;")
    L("    IADD3 R1, PT, PT, R1, 0x200, RZ ;")                                   # fp8 scale: 512 + (j >> 1)
    L("    IADD3 R2, PT, PT, R0, 0x100, RZ ;")                                   # fp4 scale: 256 + j
    L(f"    SEL R{P_OFFS}, R2, R1, P6 ;")
    # valid[] of this chunk, buffer (cidx & 1): row tid -> permuted slot (tid & 7) * 8 + (tid >> 3)
    L(f"    ISETP.GT.U32.AND P0, PT, R{R_TID}, 0x3f, PT ;")
    L("    @P0 BRA valid_skip_%s ;" % cidx_reg)
    L(f"    IADD3 R0, PT, PT, R9, R{R_TID}, RZ ;")                                # pos
    L("    ISETP.LT.U32.AND P1, PT, R0, R8, PT ;")
    L("    IMAD.WIDE.U32 R2, PT, R0, 0x4, R6 ;")
    L("    MOV R4, 0xffffffff ;")
    L("    @P1 LDG.E R4, desc[UR4][R2.64] ;")                                     # idx or -1
    L(f"    LOP3.LUT R5, R{R_TID}, 0x7, RZ, 0xc0, !PT ;")
    L(f"    SHF.R.U32.HI R10, RZ, 0x3, R{R_TID} ;")
    L("    IMAD R5, R5, 0x8, R10 ;")                                             # permuted slot
    L(f"    LOP3.LUT R10, R{cidx_reg}, 0x1, RZ, 0xc0, !PT ;")
    L("    IMAD R5, R10, 0x40, R5 ;")                                            # + buffer * 64 slots
    L(f"    IMAD R5, R5, 0x4, RZ ;")
    L(f"    STS [R5+{SM_VALID:#x}], R4 ;")
    L("valid_skip_%s:" % cidx_reg)


def emit_issue(cidx_reg: int):
    """Issue the loads of chunk `cidx_reg` into the staging registers (parameters in R28-R31/R72-R75, P6 set).
    idx of the thread's 8 rows: 2 x LDS.128 from the chunk's valid buffer. Scratch: R0-R11, R64-R71."""
    L(f"    LOP3.LUT R0, R{cidx_reg}, 0x1, RZ, 0xc0, !PT ;")
    L(f"    IMAD R0, R0, 0x100, R{R_GIDX} ;")                                     # VALID + buffer*0x100 + g*32
    L("    LDS.128 R64, [R0] ;")
    L("    LDS.128 R68, [R0+0x10] ;")
    # one load shape for both record types: 16 B of data (fp8: the 16 e4m3 bytes; fp4: the 8 nibble bytes + 8 bytes
    # of the same record that are ignored — bytes 8j..8j+15 <= 263 < 288) + the scale byte, one task per row;
    # the record type is resolved in the dequant (if/else, cubit BUG-421 fix), not here.
    # The address pairs rotate over three sets: a load latches its address late (read barrier), so recomputing the
    # same pair for the next task would wait for the previous load's latch every time (ncu: the issue block was 1/3
    # of all stall samples, LDG.E.U8 + IMAD.WIDE at the top). With three sets the pair of task k is rewritten by task
    # k+3, ~100 cycles later. Page is computed in place of the (dead after this) index, slot in the task's own scale
    # staging register (zeroed before its load).
    ADDR_SETS = ((6, 10), (2, 8), (4, 0))      # (data pair, scale pair); R0/R1 free after the LDS address above
    if os.environ.get("SMLA_DBG_ISSUE_SERIAL"):   # debug: one address set (the v1 shape)
        ADDR_SETS = ((6, 10),) * 3
    for k in range(8):
        q, s = R_STGQ[k], R_STGS[k]
        ix = 64 + k
        d, sc = ADDR_SETS[k % 3]
        L(f"    ISETP.GE.AND P0, PT, R{ix}, RZ, PT ;")                            # valid index
        L(f"    LOP3.LUT R{s}, R{ix}, R{P_MASK}, RZ, 0xc0, !PT ;")                # slot (temp: scale staging reg)
        L(f"    SHF.R.U32.HI R{ix}, RZ, R{P_SHIFT}, R{ix} ;")                     # page (in place)
        L(f"    IMAD.WIDE.U32 R{d}, PT, R{ix}, R{P_PSTR}, R{P_CACHE} ;")          # cache + page*stride
        L(f"    IMAD.WIDE.U32 R{d}, PT, R{s}, R{P_REC}, R{d} ;")                  # + slot*record
        L(f"    IMAD.WIDE.U32 R{sc}, PT, R{P_OFFS}, 0x1, R{d} ;")                 # scale address
        L(f"    IMAD.WIDE.U32 R{d}, PT, R{P_OFFD}, 0x1, R{d} ;")                  # data address
        L(f"    CS2R R{q}, SRZ ;")
        L(f"    CS2R R{q + 2}, SRZ ;")
        L(f"    MOV R{s}, RZ ;")
        L(f"    @P0 LDG.E.128 R{q}, desc[UR4][R{d}.64] ;")
        L(f"    @P0 LDG.E.U8 R{s}, desc[UR4][R{sc}.64] ;")


def emit_dequant(tag: str):
    """Dequantize the staged records of a chunk into the bf16 tile (P6 = fp4 chunk, as set by that chunk's parameters).
    Outputs R4-R11 (two STS.128 per task), scale R28, temps R29-R31, R72-R75.

    The type branch is an if/else diamond: the staging loads' consumers sit in both arms and both arms end with
    stores whose data registers are overwritten right after the join. The barrier allocator must follow the
    control flow here (cubit BUG-421 fix, V/cubit/BUG421_fwd_branch_join_barriers.md); no manual drain."""
    L(f"    @P6 BRA dequant_fp4_{tag} ;")
    for fp4 in (False, True):
        L(f"dequant_fp4_{tag}:" if fp4 else f"dequant_fp8_{tag}:")
        if fp4:
            L(f"    LOP3.LUT R29, R{R_TID}, 0x1, RZ, 0xc0, !PT ;")
            L("    ISETP.NE.AND P4, PT, R29, RZ, PT ;")                          # P4 = j odd -> high 8 bytes
        for k in range(8):
            q, s = R_STGQ[k], R_STGS[k]
            S, T0, T1 = 28, 29, 30
            L(f"    IMAD R{S}, R{s}, 0x101, RZ ;")
            if fp4:
                L(f"    F2FP.F16.E4M3.UNPACK_B R{S}, R{S} ;")                    # E4M3 scale -> f16x2
                L(f"    HADD2.F32 R{T0}, -RZ, R{S}.H0_H0 ;")
                L(f"    HADD2.F32 R{S}, -RZ, R{S}.H1_H1 ;")
                L(f"    F2FP.BF16.F32.PACK_AB R{S}, R{S}, R{T0} ;")              # bf16x2 scale (both halves)
                # the thread's 8 nibble bytes are the low (j even) or high (j odd) half of the 16-B load
                L(f"    SEL R74, R{q + 2}, R{q}, P4 ;")
                L(f"    SEL R75, R{q + 3}, R{q + 1}, P4 ;")
                # 8 byte pairs of nibbles from the two raw words -> unpack into R4..R11
                for w in range(2):
                    src = 74 + w
                    L(f"    SHF.R.U32.HI R{T0}, RZ, 0x8, R{src} ;")
                    L(f"    SHF.R.U32.HI R{T1}, RZ, 0x10, R{src} ;")
                    L(f"    SHF.R.U32.HI R31, RZ, 0x18, R{src} ;")
                    L(f"    F2FP.F16.E2M1.UNPACK_B R{4 + 4 * w}, R{src} ;")
                    L(f"    F2FP.F16.E2M1.UNPACK_B R{5 + 4 * w}, R{T0} ;")
                    L(f"    F2FP.F16.E2M1.UNPACK_B R{6 + 4 * w}, R{T1} ;")
                    L(f"    F2FP.F16.E2M1.UNPACK_B R{7 + 4 * w}, R31 ;")
            else:
                L(f"    F2FP.BF16.E8.UNPACK_B R{S}, R{S} ;")                      # UE8M0 scale -> bf16x2
                for w in range(4):
                    L(f"    F2FP.F16.E4M3.UNPACK_B R{4 + 2 * w}, R{q + w} ;")
                    L(f"    F2FP.F16.E4M3.UNPACK_B R{5 + 2 * w}, R{q + w}.H1 ;")
            # f16x2 -> f32 -> bf16x2, x scale: 8 pairs, interleaved two at a time (independent chains)
            for p0 in range(4, 12, 2):
                p1 = p0 + 1
                L(f"    HADD2.F32 R{T0}, -RZ, R{p0}.H0_H0 ;")
                L(f"    HADD2.F32 R{T1}, -RZ, R{p0}.H1_H1 ;")
                L(f"    HADD2.F32 R72, -RZ, R{p1}.H0_H0 ;")
                L(f"    HADD2.F32 R73, -RZ, R{p1}.H1_H1 ;")
                L(f"    F2FP.BF16.F32.PACK_AB R{p0}, R{T1}, R{T0} ;")
                L(f"    F2FP.BF16.F32.PACK_AB R{p1}, R73, R72 ;")
                L(f"    HMUL2.BF16_V2 R{p0}, R{p0}, R{S} ;")
                L(f"    HMUL2.BF16_V2 R{p1}, R{p1}, R{S} ;")
            L(f"    STS.128 [R{R_GSTS}+{8 * QSTRIDE * k:#x}], R4 ;")
            L(f"    STS.128 [R{R_GSTS}+{8 * QSTRIDE * k + 16:#x}], R8 ;")
        if not fp4:
            L(f"    BRA dequant_done_{tag} ;")
    L(f"dequant_done_{tag}:")


# ---------------------------------------------------------------- chunk 0: parameters, valid, issue, dequant
# The loads of a chunk and their dequant sit in the same straight-line body (prologue: chunk 0; loop body: chunk c+1
# issued before QK(c) and dequantized after PV(c)): the staging registers are live across QK/softmax/PV only, which
# keeps the loop-carried state to the accumulators, m, sum and the chunk counter.
L("    // ---- chunk 0: parameters + valid[] -> barrier -> issue -> dequant into the tile ----")
emit_params_and_valid(R_C)
L("    BAR.SYNC.DEFER_BLOCKING 0x0 ;")
emit_issue(R_C)
emit_dequant("c0")
stop("dequant")
if os.environ.get("SMLA_DEBUG_TILE"):   # debug: copy the whole chunk-0 tile (64 x 1024 B) to out_lse + t*65536 and exit
    L("    BAR.SYNC.DEFER_BLOCKING 0x0 ;")
    L(f"    LDC.64 R2, {c('out_lse')} ;")
    L(f"    IMAD.WIDE.U32 R2, PT, R{R_TOK}, 0x14000, R2 ;")                       # 80 KB per token: KV tile 64 KB + Q tile 16 KB
    for rr in range(5):                                                            # rows tid>>4 + 16*rr (rr = 4: q rows)
        L(f"    SHF.R.U32.HI R0, RZ, 0x4, R{R_TID} ;")
        L(f"    IADD3 R0, PT, PT, R0, {16 * (rr % 4):#x}, RZ ;")                    # row
        L(f"    LOP3.LUT R1, R{R_TID}, 0xf, RZ, 0xc0, !PT ;")                     # 64-B segment
        L(f"    IMAD R10, R0, {QSTRIDE:#x}, RZ ;")
        L("    IMAD R10, R1, 0x40, R10 ;")
        L(f"    IADD3 R10, PT, PT, R10, {SM_KV if rr < 4 else SM_Q:#x}, RZ ;")    # smem source
        L("    IMAD R11, R0, 0x400, RZ ;")
        L("    IMAD R11, R1, 0x40, R11 ;")
        if rr == 4:
            L("    IADD3 R11, PT, PT, R11, 0x10000, RZ ;")
        L("    IMAD.WIDE.U32 R6, PT, R11, 0x1, R2 ;")                             # gmem dest
        for i in range(4):
            L(f"    LDS.128 R{64 if i % 2 == 0 else 68}, [R10+{16 * i:#x}] ;")
            L(f"    STG.E.128 desc[UR4][R6.64+{16 * i:#x}], R{64 if i % 2 == 0 else 68} ;")
    L("    EXIT ;")
L()

# ---------------------------------------------------------------- chunk loop
L("chunk_loop:")
L("    // ---- chunk c+1: parameters + valid[] (other buffer); barrier publishes tile(c) and valid(c+1) ----")
L(f"    IADD3 R11, PT, PT, R{R_C}, 0x1, RZ ;")                                   # c + 1 (the helpers use R0-R10 as scratch)
L(f"    ISETP.LT.U32.AND P5, PT, R11, R{R_CHUNKS}, PT ;")                        # P5 = there is a next chunk
L("    @!P5 BRA next_params_done ;")
emit_params_and_valid(11)
L("next_params_done:")
L("    BAR.SYNC.DEFER_BLOCKING 0x0 ;")
L("    @!P5 BRA next_issue_done ;")
emit_issue(11)
L("next_issue_done:")
stop("issue")
L()
L("    // ---- QK: 32 k-steps, one accumulator chain per warp; fragments loaded two steps ahead (spec B's order) ----")
NKS = D // 16


def qk_loads(ks: int):
    A = R_A0 if ks % 2 == 0 else R_A1
    B = R_B0 if ks % 2 == 0 else R_B1
    L(f"    LDSM.16.M88.4 R{A}, [R{R_QA}+{32 * ks:#x}] ;")
    L(f"    LDS R{B}, [R{R_KB}+{32 * ks:#x}] ;")
    L(f"    LDS R{B + 1}, [R{R_KB}+{32 * ks + 16:#x}] ;")


QK_PIPE = not os.environ.get("SMLA_DBG_QK_SERIAL")      # debug: loads right before their HMMA
if QK_PIPE:
    qk_loads(0)
    qk_loads(1)
for ks in range(NKS):
    A = R_A0 if ks % 2 == 0 else R_A1
    B = R_B0 if ks % 2 == 0 else R_B1
    C = "RZ" if ks == 0 else f"R{R_QK}"
    if not QK_PIPE:
        qk_loads(ks)
    L(f"    HMMA.16816.F32.BF16 R{R_QK}, R{A}, R{B}, {C} ;")
    if QK_PIPE and ks + 2 < NKS:
        qk_loads(ks + 2)                      # refills the buffer this HMMA just read (its read barrier covers the WAR)
stop("qk")
if os.environ.get("SMLA_DEBUG_QK"):   # debug: store the raw qk accumulators of chunk 0 (16 B per thread) at out_lse + t*8192 + tid*16
    L(f"    ISETP.NE.AND P3, PT, R{R_C}, RZ, PT ;")
    L("    @P3 BRA dbgqk_skip ;")
    L(f"    LDC.64 R2, {c('out_lse')} ;")
    L(f"    IMAD.WIDE.U32 R2, PT, R{R_TOK}, 0x2000, R2 ;")
    L(f"    IMAD.WIDE.U32 R2, PT, R{R_TID}, 0x10, R2 ;")
    L(f"    STG.E.128 desc[UR4][R2.64], R{R_QK} ;")
    L("dbgqk_skip:")
L()
L("    // ---- softmax (spec B instruction chain); mask from this chunk's valid buffer ----")
L(f"    LOP3.LUT R0, R{R_C}, 0x1, RZ, 0xc0, !PT ;")
L(f"    IMAD R0, R0, 0x100, R{R_VALR} ;")
L("    LDS R2, [R0] ;")                                                         # valid[8w + 2t4]
L("    LDS R3, [R0+0x20] ;")                                                    # valid[8w + 2t4 + 1]
L("    ISETP.GE.AND P0, PT, R2, RZ, PT ;")
L("    ISETP.GE.AND P1, PT, R3, RZ, PT ;")
for j in range(4):
    L(f"    FMUL.FTZ R{R_QK + j}, R{R_QK + j}, R{R_SCALE2} ;")
for j in range(4):
    L(f"    FSEL R{R_QK + j}, R{R_QK + j}, -1.00000001504746621988e+30, P{j & 1} ;")
L(f"    FMNMX.FTZ R2, R{R_QK}, R{R_QK + 1}, !PT ;")                             # mx[0]
L(f"    FMNMX.FTZ R3, R{R_QK + 2}, R{R_QK + 3}, !PT ;")                         # mx[1]
for s in (1, 2):
    L(f"    SHFL.BFLY PT, R4, R2, {s:#x}, 0x1f ;")
    L(f"    SHFL.BFLY PT, R5, R3, {s:#x}, 0x1f ;")
    L("    FMNMX.FTZ R2, R2, R4, !PT ;")
    L("    FMNMX.FTZ R3, R3, R5, !PT ;")
L(f"    ISETP.NE.AND P2, PT, R{R_T4}, RZ, PT ;")
L(f"    @!P2 STS [R{R_RED0}], R2 ;")
L(f"    @!P2 STS [R{R_RED0}+0x20], R3 ;")
L("    BAR.SYNC.DEFER_BLOCKING 0x0 ;")
for h in range(2):
    for w in range(8):
        L(f"    LDS R{4 + w}, [R{R_REDR}+{w * 0x40 + h * 0x20:#x}] ;")
    mnew = 2 + h
    L(f"    FMNMX.FTZ R{mnew}, R{R_M0 + h}, R4, !PT ;")
    for w in range(1, 8):
        L(f"    FMNMX.FTZ R{mnew}, R{mnew}, R{4 + w}, !PT ;")
for h in range(2):   # alpha0 -> R6, alpha1 -> R7 (m_old still in R_M0/R_M1)
    L(f"    FADD.FTZ R{6 + h}, -R{2 + h}, R{R_M0 + h} ;")
    L(f"    MUFU.EX2 R{6 + h}, R{6 + h} ;")
    L(f"    MOV R{R_M0 + h}, R{2 + h} ;")
for j in range(4):   # p = valid ? exp2(qk - m) : 0 — computed unpredicated, selected afterwards
    pj = 8 + j
    L(f"    FADD.FTZ R{pj}, -R{R_M0 + (j >> 1)}, R{R_QK + j} ;")
    L(f"    MUFU.EX2 R{pj}, R{pj} ;")
    L(f"    FSEL R{pj}, R{pj}, RZ, P{j & 1} ;")
    L(f"    F2FP.BF16.F32.PACK_AB R{R_A0}, RZ, R{pj} ;")
    L(f"    STS.U16 [R{R_PST}+{(j & 1) * 2 + (j >> 1) * 8 * PSTRIDE:#x}], R{R_A0} ;")
if os.environ.get("SMLA_DEBUG_QK"):   # debug: p values of chunk 0 (16 B per thread) at out_lse + t*8192 + 4096 + tid*16
    L(f"    ISETP.NE.AND P3, PT, R{R_C}, RZ, PT ;")
    L("    @P3 BRA dbgp_skip ;")
    L(f"    LDC.64 R2, {c('out_lse')} ;")
    L(f"    IMAD.WIDE.U32 R2, PT, R{R_TOK}, 0x2000, R2 ;")
    L(f"    IMAD.WIDE.U32 R2, PT, R{R_TID}, 0x10, R2 ;")
    L("    STG.E.128 desc[UR4][R2.64+0x1000], R8 ;")
    L("dbgp_skip:")
L("    FADD.FTZ R2, R8, R9 ;")
L("    FADD.FTZ R3, R10, R11 ;")
for s in (1, 2):
    L(f"    SHFL.BFLY PT, R4, R2, {s:#x}, 0x1f ;")
    L(f"    SHFL.BFLY PT, R5, R3, {s:#x}, 0x1f ;")
    L("    FADD.FTZ R2, R2, R4 ;")
    L("    FADD.FTZ R3, R3, R5 ;")
L(f"    @!P2 STS [R{R_RED0}+0x200], R2 ;")
L(f"    @!P2 STS [R{R_RED0}+0x220], R3 ;")
L("    BAR.SYNC.DEFER_BLOCKING 0x0 ;")
for h in range(2):
    for w in range(8):  # temps: the A fragment window (free here)
        L(f"    LDS R{R_A0 + w}, [R{R_REDR}+{0x200 + w * 0x40 + h * 0x20:#x}] ;")
    L(f"    FFMA.FTZ R{R_S0 + h}, R{6 + h}, R{R_S0 + h}, R{R_A0} ;")
    for w in range(1, 8):
        L(f"    FADD.FTZ R{R_S0 + h}, R{R_S0 + h}, R{R_A0 + w} ;")
for v in range(8):
    for j in range(4):
        L(f"    FMUL.FTZ R{acc(v, j)}, R{6 + (j >> 1)}, R{acc(v, j)} ;")
stop("softmax")
L()
L("    // ---- PV: 4 k-steps x 8 column tiles per warp; P fragment one k-step ahead, V fragments two tiles ahead ----")


def pv_b_load(ks: int, v: int):
    B = R_B0 if v % 2 == 0 else R_B1
    L(f"    LDSM.16.MT88.2 R{B}, [R{R_VB}+{ks * 16 * QSTRIDE + v * 0x80:#x}] ;")


PV_PIPE = not os.environ.get("SMLA_DBG_PV_SERIAL")      # debug: loads right before their HMMA
if PV_PIPE:
    L(f"    LDSM.16.M88.4 R{R_A0}, [R{R_PA}] ;")
for ks in range(CAND // 16):
    A = R_A0 if ks % 2 == 0 else R_A1
    if PV_PIPE:
        if ks + 1 < CAND // 16:
            L(f"    LDSM.16.M88.4 R{R_A1 if ks % 2 == 0 else R_A0}, [R{R_PA}+{32 * (ks + 1):#x}] ;")
        pv_b_load(ks, 0)
        pv_b_load(ks, 1)
    else:
        L(f"    LDSM.16.M88.4 R{A}, [R{R_PA}+{32 * ks:#x}] ;")
    for v in range(8):
        B = R_B0 if v % 2 == 0 else R_B1
        if not PV_PIPE:
            pv_b_load(ks, v)
        L(f"    HMMA.16816.F32.BF16 R{acc(v, 0)}, R{A}, R{B}, R{acc(v, 0)} ;")
        if PV_PIPE and v + 2 < 8:
            pv_b_load(ks, v + 2)
L("    BAR.SYNC.DEFER_BLOCKING 0x0 ;")
stop("pv")
L("    // ---- dequantize the staged chunk c+1 into the tile (its loads were issued before QK(c)) ----")
L("    @!P5 BRA epilogue ;")
emit_dequant("next")
L(f"    IADD3 R{R_C}, PT, PT, R{R_C}, 0x1, RZ ;")
L("    BRA chunk_loop ;")
L()

# ---------------------------------------------------------------- epilogue
L("epilogue:")
L("    // ---- epilogue: norm / lse per head (gid, gid + 8), sink, bf16 output ----")
L(f"    LDC.64 R{R_SINK}, {c('attn_sink')} ;")
L(f"    ISETP.NE.U32.AND P5, PT, R{R_SINK}, RZ, PT ;")
L(f"    ISETP.NE.U32.OR P5, PT, R{R_SINK + 1}, RZ, P5 ;")                       # P5 = sink present
L(f"    LDC R{R_LSESC}, {c('lse_scale')} ;")
L(f"    LDC.64 R{R_LSETOK}, {c('out_lse')} ;")
L(f"    ISETP.NE.U32.AND P4, PT, R{R_LSETOK}, RZ, PT ;")
L(f"    ISETP.NE.U32.OR P4, PT, R{R_LSETOK + 1}, RZ, P4 ;")
L(f"    ISETP.EQ.AND P4, PT, R{R_WARP}, RZ, P4 ;")
L(f"    ISETP.EQ.AND P4, PT, R{R_T4}, RZ, P4 ;")                                # P4 = this thread writes lse
L(f"    LDC R0, {c('out_lse_stride')} ;")
L(f"    IMAD R0, R{R_TOK}, R0, R{R_GID} ;")                                      # t*stride + gid
L(f"    IMAD.WIDE.U32 R{R_LSETOK}, PT, R0, 0x4, R{R_LSETOK} ;")
L(f"    LDC.64 R{R_OTOK}, {c('output')} ;")
L(f"    IMAD.WIDE.U32 R{R_OTOK}, PT, R{R_TOK}, 0x4000, R{R_OTOK} ;")
L(f"    IMAD R0, R{R_WARP}, 0x8, RZ ;")
L(f"    IMAD R0, R{R_T4}, 0x2, R0 ;")
L(f"    IMAD R0, R{R_GID}, {D:#x}, R0 ;")
L(f"    IMAD.WIDE.U32 R{R_OTOK}, PT, R0, 0x2, R{R_OTOK} ;")
for h in range(2):
    S, M = R_S0 + h, R_M0 + h
    L(f"    FSETP.GT.FTZ.AND P1, PT, R{S}, RZ, PT ;")                            # sum > 0
    L(f"    MUFU.RCP R2, R{S} ;")
    L("    FSEL R2, R2, RZ, P1 ;")                                              # norm
    L(f"    MUFU.LG2 R4, R{S} ;")
    L(f"    FADD.FTZ R4, R4, R{M} ;")                                            # log2(sum) + m
    L("    MOV R3, 0xff800000 ;")                                               # -inf
    L("    FSEL R3, R4, R3, P1 ;")                                              # lse
    L(f"    @!P5 BRA sink_done_{h} ;")
    L(f"    IMAD.WIDE.U32 R10, PT, R{R_GID}, 0x4, R{R_SINK} ;")                 # address pair R10:R11 (not a MUFU target)
    L(f"    LDG.E R6, desc[UR4][R10.64+{h * 0x20:#x}] ;")                        # sink[gid + 8h]
    L("    FMUL.FTZ R6, R6, 1.4426950216293334961 ;")                            # sink * log2e
    L(f"    FMNMX.FTZ R7, R6, R{M}, !PT ;")                                      # peak
    L(f"    FADD.FTZ R8, -R7, R{M} ;")                                           # m - peak
    L("    FADD.FTZ R9, R6, -R7 ;")                                             # sink - peak
    L("    MUFU.EX2 R8, R8 ;")                                                  # weight
    L("    MUFU.EX2 R9, R9 ;")
    L(f"    FFMA.FTZ R9, R8, R{S}, R9 ;")                                        # total = weight*sum + exp2(sink-peak)
    L("    MUFU.LG2 R10, R9 ;")
    L("    MUFU.RCP R11, R9 ;")
    L("    FADD.FTZ R10, R7, R10 ;")                                            # peak + log2(total)
    L("    FMUL.FTZ R11, R8, R11 ;")                                            # weight * rcp(total)
    L("    FSEL R3, R10, R6, P1 ;")                                             # lse = sum > 0 ? peak + log2(total) : sink
    L("    FSEL R2, R11, R2, P1 ;")                                             # norm = sum > 0 ? weight/total : 0
    L(f"sink_done_{h}:")
    for v in range(8):
        L(f"    FMUL.FTZ R4, R2, R{acc(v, 2 * h)} ;")
        L(f"    FMUL.FTZ R5, R2, R{acc(v, 2 * h + 1)} ;")
        L("    F2FP.BF16.F32.PACK_AB R4, R5, R4 ;")
        L(f"    STG.E desc[UR4][R{R_OTOK}.64+{h * 0x2000 + v * 0x80:#x}], R4 ;")
    L("    FSETP.NEU.FTZ.AND P0, PT, R3, -1.00000001504746621988e+30, PT ;")
    L(f"    FMUL.FTZ R4, R3, R{R_LSESC} ;")
    L("    FSEL R4, R4, -1.00000001504746621988e+30, P0 ;")
    L(f"    @P4 STG.E desc[UR4][R{R_LSETOK}.64+{h * 0x20:#x}], R4 ;")
L("    EXIT ;")
L(".endentry")

out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sass",
                                                                "sparse_mla_prefill_v41.sass")
with open(out_path, "w") as f:
    f.write("\n".join(OUT) + "\n")
print(f"wrote {out_path}: {sum(1 for l in OUT if l.strip() and not l.strip().startswith(('//', '.')) and not l.strip().endswith(':'))} instructions")
