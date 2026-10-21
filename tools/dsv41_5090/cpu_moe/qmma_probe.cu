// qmma_probe.cu — what the sm_120 block-scaled MMA (the instruction b12x's W4A8 compact path and our fused kernel
// use for FC1 and FC2) computes, bit for bit, so that the CPU expert kernel can reproduce it:
//
//   mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0
//
// Two questions, answered on random operands against an exact (128-bit fixed-point) host reference:
//   zero-C  : D = sum_k A[k] B[k] * 2^(sa+sb-254)        — FC1 (b12x adds the block's D to an fp32 accumulator with
//             fadd.rn afterwards): is the 32-product block sum rounded to nearest (even), or truncated, or something
//             with fewer internal bits?
//   accum-C : D = C + sum_k A[k] B[k]                    — FC2 (b12x accumulates the 18 K32 blocks in C): the same
//             question with a wide-exponent C in the sum.
// It also writes the two tables the CPU kernel needs to match b12x's activation exactly:
//   act_m[65536]   fp32: m(g) = mul.rn(g, rcp.approx.ftz(add.rn(1, ex2.approx.ftz(mul.rn(g, -log2 e))))) for every
//                  bf16 bit pattern g (the limit clamp applied first, as act4 does); the activation is then
//                  bf16(mul.rn(u, m(g))) on the CPU with plain IEEE multiplies.
//   ue8m0_inv[256] fp32: ex2.approx.f32((float)(127 - byte)), 0 for byte 0 (b12x's ue8m0_to_output_scale).
//
// build: /usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o qmma_probe qmma_probe.cu
// run:   CUDA_VISIBLE_DEVICES=1 flock -w 3600 /tmp/exl3diff-gpu1.lock ./qmma_probe [--cases 200000] [--tables DIR]
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

// ------------------------------------------------------------------------------------------------ device side
// One warp per case. A: rows 0..7 (rows 8..15 duplicate them), 32 k of e4m3; B: 8 columns of 32 e2m1 (4-bit codes in
// bits 5:2 of 8-bit containers, as b12x's spread of its packed nibbles); scales: one byte per A row and per B column
// (all four bytes of the register equal, the selector picks byte 0). C: 8x8 fp32 (zero for the zero-C mode).
struct Case {
    uint8_t a[8][32];      // e4m3 codes
    uint8_t b[8][32];      // e2m1 codes 0..15, [col][k]
    uint8_t sa[8];         // ue8m0 bytes per row
    uint8_t sb[8];         // per column
    float c[8][8];
};

__device__ __forceinline__ uint32_t pack4(uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3) {
    return (uint32_t)x0 | ((uint32_t)x1 << 8) | ((uint32_t)x2 << 16) | ((uint32_t)x3 << 24);
}

template <bool ZERO_C>
__global__ void qmma_kernel(const Case* cases, float* out, int n) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= n) return;
    const Case& K = cases[warp];
    const int g = lane >> 2, t = lane & 3;
    // A fragments (m16n8k32, 8-bit containers): a0 = row g, k 4t..4t+3; a1 = row g+8 (dup); a2 = row g, k 16+4t..; a3 dup
    const uint32_t a0 = pack4(K.a[g][4 * t], K.a[g][4 * t + 1], K.a[g][4 * t + 2], K.a[g][4 * t + 3]);
    const uint32_t a2 = pack4(K.a[g][16 + 4 * t], K.a[g][16 + 4 * t + 1], K.a[g][16 + 4 * t + 2], K.a[g][16 + 4 * t + 3]);
    // B fragments: b0 = col g, k 4t..4t+3; b1 = col g, k 16+4t..; the e2m1 code in bits 5:2 of each byte
    const uint32_t b0 = pack4(K.b[g][4 * t] << 2, K.b[g][4 * t + 1] << 2, K.b[g][4 * t + 2] << 2, K.b[g][4 * t + 3] << 2);
    const uint32_t b1 = pack4(K.b[g][16 + 4 * t] << 2, K.b[g][16 + 4 * t + 1] << 2, K.b[g][16 + 4 * t + 2] << 2, K.b[g][16 + 4 * t + 3] << 2);
    const uint32_t sfa = 0x01010101u * K.sa[g];
    const uint32_t sfb = 0x01010101u * K.sb[g];
    float d[4];
    if (ZERO_C) {
        const float z = 0.f;
        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%14, %14, %14, %14}, {%10}, {%12, %13}, {%11}, {%12, %13};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a0), "r"(a0), "r"(a2), "r"(a2), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(0), "n"(0), "f"(z));
    } else {
        d[0] = K.c[g][2 * t]; d[1] = K.c[g][2 * t + 1]; d[2] = d[0]; d[3] = d[1];
        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %13}, {%11}, {%12, %13};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a0), "r"(a2), "r"(a2), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(0), "n"(0));
    }
    // D fragments: d0 = (g, 2t), d1 = (g, 2t+1); d2/d3 are rows g+8 (duplicates, checked equal)
    float* o = out + (size_t)warp * 64;
    o[g * 8 + 2 * t] = d[0];
    o[g * 8 + 2 * t + 1] = d[1];
    if (d[2] != d[0] || d[3] != d[1]) o[g * 8 + 2 * t] = NAN;  // rows 8..15 must agree with 0..7
}

// the activation's m(g) with b12x's / the fused kernel's exact instruction sequence (act4)
__global__ void act_table_kernel(float* tab, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 65536) return;
    float g = __uint_as_float((uint32_t)i << 16);
    g = g >= limit ? limit : g;
    float t, e, sg;
    asm("mul.rn.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(g));
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(t));
    asm("add.rn.f32 %0, %1, 0f3F800000;" : "=f"(sg) : "f"(e));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(sg) : "f"(sg));
    tab[i] = __fmul_rn(g, sg);
}

__global__ void inv_table_kernel(float* tab) {
    const int byte = threadIdx.x;
    if (byte >= 256) return;
    float inv;
    const float ef = (float)(127 - byte);
    asm("ex2.approx.f32 %0, %1;" : "=f"(inv) : "f"(ef));
    tab[byte] = byte == 0 ? 0.f : inv;
}

// ------------------------------------------------------------------------------------------------ host side
// exact values: e4m3 code -> integer multiple of 2^-9 (normal (8+M) * 2^(E-10) = (8+M) * 2^(E-1) units of 2^-9, hmm:
// value = (8+M)/8 * 2^(E-7) = (8+M) * 2^(E-10); in units of 2^-9 that is (8+M) * 2^(E-1) for E >= 1, M for E == 0.
static int64_t e4m3_units(uint8_t c) {  // units of 2^-9; NaN codes excluded by the generator
    int s = c >> 7, E = (c >> 3) & 15, M = c & 7;
    int64_t v = E == 0 ? M : (int64_t)(8 + M) << (E - 1);
    return s ? -v : v;
}
static int64_t e2m1_units(uint8_t c) {  // units of 2^-1
    int s = c >> 3, E = (c >> 1) & 3, M = c & 1;
    int64_t v = E == 0 ? M : (int64_t)(2 + M) << (E - 1);
    return s ? -v : v;
}
static float e4m3_value(uint8_t c) { return ldexpf((float)e4m3_units(c), -9); }
static float e2m1_value(uint8_t c) { return ldexpf((float)e2m1_units(c), -1); }

typedef __int128 i128;

// exact x = sum of products in units of 2^-10 times 2^(sa+sb-254), plus C — as an i128 in units of 2^-E_UNIT
constexpr int E_UNIT = 90;  // 2^-90 resolution: fp32 C down to 2^-66 exactly; sums up to 2^(127-90) = 2^37 fine

static bool f32_to_units(float x, i128& out) {
    if (x == 0.f) { out = 0; return true; }
    int e;
    float m = frexpf(x, &e);                           // x = m * 2^e, |m| in [0.5, 1)
    int64_t mi = (int64_t)ldexpf(m, 24);               // 24-bit integer
    int shift = e - 24 + E_UNIT;
    if (shift < 0) return false;
    if (shift > 100) return false;
    out = (i128)mi << shift;
    return true;
}

// round an exact i128 (units 2^-E_UNIT) to fp32: RNE or RZ
static float units_to_f32(i128 v, bool rne) {
    if (v == 0) return 0.f;
    bool neg = v < 0;
    unsigned __int128 u = neg ? (unsigned __int128)(-v) : (unsigned __int128)v;
    int top = 127;
    while (top > 0 && !((u >> top) & 1)) --top;        // msb index
    int shift = top - 23;                              // keep 24 bits
    uint64_t mant;
    bool inexact = false;
    if (shift <= 0) {
        mant = (uint64_t)(u << (-shift));
    } else {
        mant = (uint64_t)(u >> shift);
        unsigned __int128 rem = u & (((unsigned __int128)1 << shift) - 1);
        unsigned __int128 half = (unsigned __int128)1 << (shift - 1);
        inexact = rem != 0;
        if (rne && (rem > half || (rem == half && (mant & 1)))) {
            ++mant;
            if (mant == (1ull << 24)) { mant >>= 1; ++shift; }
        }
    }
    (void)inexact;
    int e2 = shift - E_UNIT;                           // value = mant * 2^e2
    float f = ldexpf((float)mant, e2);
    return neg ? -f : f;
}

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

struct Stats {
    long n = 0, exact_repr = 0, eq_rne = 0, eq_rz = 0, eq_neither = 0, nan_dup = 0;
    long inexact = 0, inexact_eq_rne = 0, inexact_eq_rz = 0, inexact_eq_rne_only = 0, inexact_eq_rz_only = 0;
    long ties = 0, ties_eq_rne = 0, ties_away = 0;
    long shown = 0;
};

static void print_units(i128 v) {  // binary of |v| with the point at 2^0
    bool neg = v < 0;
    unsigned __int128 u = neg ? -v : v;
    int top = 127;
    while (top > 0 && !((u >> top) & 1)) --top;
    int low = 0;
    while (low < top && !((u >> low) & 1)) ++low;
    printf("%s2^%d x 1.", neg ? "-" : "", top - E_UNIT);
    for (int i = top - 1; i >= low; --i) putchar(((u >> i) & 1) ? '1' : '0');
    printf(" (%d significant bits)", top - low + 1);
}

// ---- accumulation models for the accumulate-in-C form -------------------------------------------------------------
// Every addend is an exact integer in units of 2^-E_UNIT. A model: pick the alignment exponent e_ref, keep w bits
// below and including it (truncating each addend toward zero), add, round the result to fp32 (RNE or RZ).
//   e_ref: 0 = max over C and every product; 1 = max over C and the exact product sum; 2 = exponent of the exact result
//   variant: 0 = truncate the products and C individually; 1 = truncate the exact product sum and C; 2 = truncate the
//            exact result (C + sum) only; 3 = round the product sum to fp32 (RNE) first, then truncate it and C
static int msb_index(i128 v) {
    unsigned __int128 u = v < 0 ? -v : v;
    if (u == 0) return -1;
    int top = 127;
    while (top > 0 && !((u >> top) & 1)) --top;
    return top;
}
static i128 trunc_to(i128 v, int top_ref, int w) {  // keep bits top_ref..top_ref-w+1 (toward zero)
    if (w >= 120) return v;
    int shift = top_ref - w + 1;
    if (shift <= 0) return v;
    bool neg = v < 0;
    unsigned __int128 u = neg ? -v : v;
    u = (u >> shift) << shift;
    return neg ? -(i128)u : (i128)u;
}
struct Model { int e_ref, w, variant, rne; long hits = 0; };
static const int kWs[] = {24, 25, 26, 27, 28, 29, 30, 32, 120};
// e_ref 3 / 4: the products' exponent taken from the operands' exponent FIELDS (the way an aligner that does not
// normalize the 6-bit product would see it): e4m3 normal E-7 (subnormal: -7, value 0.M), e2m1 E>0: E-1 (E=0: -1,
// hmm: 0.5 = 1.0 * 2^-1 and 0 = 0), plus the two ue8m0 scales; 3 = +1 (the product of two significands in [1,2)
// can reach [2,4)), 4 = +0
static std::vector<Model> make_models() {
    std::vector<Model> ms;
    for (int e = 0; e < 5; ++e)
        for (int w : kWs)
            for (int v = 0; v < 4; ++v)
                for (int r = 0; r < 2; ++r) {
                    if (e >= 3 && (v != 0 || r != 0)) continue;  // field-based refs: only the leading variant
                    ms.push_back(Model{e, w, v, r});
                }
    return ms;
}
struct ProdInfo { int field_top; };  // bit index (i128 units) of 2^(ea+eb+scale)
static float model_predict(const Model& M, const i128* p, const ProdInfo* pi, int np, i128 c) {
    i128 sum = 0;
    for (int k = 0; k < np; ++k) sum += p[k];
    i128 exact = sum + c;
    int top_ref;
    if (M.e_ref == 0) {
        top_ref = msb_index(c);
        for (int k = 0; k < np; ++k) top_ref = std::max(top_ref, msb_index(p[k]));
    } else if (M.e_ref == 1) {
        top_ref = std::max(msb_index(c), msb_index(sum));
    } else if (M.e_ref == 2) {
        top_ref = msb_index(exact);
    } else {
        top_ref = msb_index(c);
        for (int k = 0; k < np; ++k)
            if (p[k] != 0) top_ref = std::max(top_ref, pi[k].field_top + (M.e_ref == 3 ? 1 : 0));
    }
    i128 s;
    if (M.variant == 0) {
        s = trunc_to(c, top_ref, M.w);
        for (int k = 0; k < np; ++k) s += trunc_to(p[k], top_ref, M.w);
    } else if (M.variant == 1) {
        s = trunc_to(c, top_ref, M.w) + trunc_to(sum, top_ref, M.w);
    } else if (M.variant == 2) {
        s = trunc_to(exact, top_ref, M.w);
    } else {
        float fs = units_to_f32(sum, true);
        i128 su;
        if (!f32_to_units(fs, su)) su = sum;
        s = trunc_to(c, top_ref, M.w) + trunc_to(su, top_ref, M.w);
    }
    return units_to_f32(s, M.rne != 0);
}

static void classify(const char* mode, const std::vector<Case>& cases, const std::vector<float>& got, Stats& S, int show) {
    const bool accum = strcmp(mode, "accum-C") == 0;
    std::vector<Model> models = accum ? make_models() : std::vector<Model>();
    long model_outputs = 0;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const Case& K = cases[ci];
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                float hw = got[ci * 64 + r * 8 + c];
                S.n++;
                if (std::isnan(hw)) { S.nan_dup++; continue; }
                int scale_e = (int)K.sa[r] + (int)K.sb[c] - 254;
                int sh = E_UNIT - 10 + scale_e;                 // product units 2^-10 -> units of 2^-E_UNIT
                i128 prods[32];
                ProdInfo pinfo[32];
                i128 sum = 0;
                for (int k = 0; k < 32; ++k) {
                    i128 pk = (i128)e4m3_units(K.a[r][k]) * e2m1_units(K.b[c][k]);
                    prods[k] = sh >= 0 ? (pk << sh) : (pk >> (-sh));
                    sum += prods[k];
                    const int Ea = (K.a[r][k] >> 3) & 15, Eb = (K.b[c][k] >> 1) & 3;
                    const int ea = (Ea == 0 ? 1 : Ea) - 7, eb = (Eb == 0 ? 1 : Eb) - 1;  // field exponents
                    pinfo[k].field_top = ea + eb + scale_e + E_UNIT;
                }
                i128 x = sum;
                i128 cu = 0;
                if (accum) {
                    if (!f32_to_units(K.c[r][c], cu)) { continue; }
                    x += cu;
                }
                float rne = units_to_f32(x, true), rz = units_to_f32(x, false);
                bool exact = f2u(rne) == f2u(rz);
                bool ern = f2u(hw) == f2u(rne), erz = f2u(hw) == f2u(rz);
                if (accum && !exact) {
                    model_outputs++;
                    for (Model& M : models)
                        if (f2u(model_predict(M, prods, pinfo, 32, cu)) == f2u(hw)) M.hits++;
                    // the leading candidate's misses, a few of them, for the eye
                    static int shown_miss = 0;
                    const Model lead{3, 26, 0, 0};
                    if (shown_miss < 8 && f2u(model_predict(lead, prods, pinfo, 32, cu)) != f2u(hw)) {
                        shown_miss++;
                        printf("  [lead-miss e_ref=3 w=26 case %zu r%d c%d] hw 0x%08x lead 0x%08x RZ 0x%08x; exact ", ci, r, c, f2u(hw),
                               f2u(model_predict(lead, prods, pinfo, 32, cu)), f2u(rz));
                        print_units(x);
                        printf("\n    sum "); print_units(sum); printf("\n    C   "); print_units(cu);
                        int top_val = -1, top_field = -1;
                        for (int k = 0; k < 32; ++k) if (prods[k] != 0) { top_val = std::max(top_val, msb_index(prods[k])); top_field = std::max(top_field, pinfo[k].field_top); }
                        printf("\n    max product msb 2^%d, max field exponent 2^%d, C msb 2^%d\n", top_val - E_UNIT, top_field - E_UNIT, msb_index(cu) - E_UNIT);
                    }
                }
                if (exact) S.exact_repr++; else S.inexact++;
                if (ern) S.eq_rne++;
                if (erz) S.eq_rz++;
                if (!ern && !erz) S.eq_neither++;
                if (!exact) {
                    if (ern) S.inexact_eq_rne++;
                    if (erz) S.inexact_eq_rz++;
                    if (ern && !erz) S.inexact_eq_rne_only++;
                    if (erz && !ern) S.inexact_eq_rz_only++;
                    // tie detection: the discarded part is exactly half an ulp
                    {
                        bool neg = x < 0;
                        unsigned __int128 u = neg ? -x : x;
                        int top = 127;
                        while (top > 0 && !((u >> top) & 1)) --top;
                        int shift = top - 23;
                        if (shift > 0) {
                            unsigned __int128 rem = u & (((unsigned __int128)1 << shift) - 1);
                            if (rem == ((unsigned __int128)1 << (shift - 1))) {
                                S.ties++;
                                if (ern) S.ties_eq_rne++;
                                float away = units_to_f32(x, false);
                                // away-from-zero = RZ bumped by one ulp
                                uint32_t ub = f2u(away) + 1;
                                float fa; memcpy(&fa, &ub, 4);
                                if (f2u(hw) == f2u(fa)) S.ties_away++;
                            }
                        }
                    }
                }
                if ((!ern && !erz) || (!exact && S.shown < show && !ern)) {
                    if (S.shown < show) {
                        S.shown++;
                        printf("  [%s case %zu r%d c%d] hw %.9e (0x%08x)  RNE %.9e (0x%08x)  RZ %.9e (0x%08x)  C %.9e\n    exact ",
                               mode, ci, r, c, hw, f2u(hw), rne, f2u(rne), rz, f2u(rz), K.c[r][c]);
                        print_units(x);
                        if (accum) { printf("\n    sum   "); print_units(sum); printf("\n    C     "); print_units(cu); }
                        printf("\n");
                    }
                }
            }
        }
    }
    if (accum && model_outputs) {
        std::sort(models.begin(), models.end(), [](const Model& a, const Model& b) { return a.hits > b.hits; });
        printf("  accumulation models on %ld inexact outputs (hits):\n", model_outputs);
        for (size_t i = 0; i < models.size() && i < 12; ++i) {
            const Model& M = models[i];
            printf("    e_ref=%d w=%-3d variant=%d final=%s : %ld (%.4f)\n", M.e_ref, M.w, M.variant, M.rne ? "RNE" : "RZ ",
                   M.hits, (double)M.hits / model_outputs);
        }
    }
}

static void report(const char* mode, const Stats& S) {
    printf("%s: %ld outputs; exactly representable %ld; inexact %ld — hw == RNE %ld, hw == RZ %ld, neither %ld; "
           "among inexact: RNE-only %ld, RZ-only %ld; ties %ld (hw == RNE %ld, == away %ld); row-dup NaN %ld\n",
           mode, S.n, S.exact_repr, S.inexact, S.eq_rne, S.eq_rz, S.eq_neither, S.inexact_eq_rne_only,
           S.inexact_eq_rz_only, S.ties, S.ties_eq_rne, S.ties_away, S.nan_dup);
}

// generators: 0 = random codes over the full e4m3 range; 1 = narrow exponents (sums always exact); 2 = big codes
// with 1/8 subnormals, random signs; 3 = same-sign big codes (448 * 6 class) with 1/8 tiny products: block sums of
// 25-28 significant bits, i.e. the zero-C rounding is exercised; 4 = like 3 with random signs on the big ones
static void gen_cases(std::vector<Case>& cases, int n, std::mt19937_64& rng, int flavor, bool with_c) {
    std::uniform_int_distribution<int> u8(0, 255), u16(0, 15), ue(0, 2);
    cases.resize(n);
    for (int i = 0; i < n; ++i) {
        Case& K = cases[i];
        memset(&K, 0, sizeof(K));
        for (int r = 0; r < 8; ++r) {
            for (int k = 0; k < 32; ++k) {
                uint8_t c;
                if (flavor == 0) {                          // full range e4m3, no NaN
                    do { c = (uint8_t)u8(rng); } while ((c & 0x7F) == 0x7F);
                } else if (flavor == 1) {                   // narrow: codes with exponent 6..12 (values 2^-1..2^6)
                    int E = 6 + (int)(rng() % 7), M = (int)(rng() % 8), s = (int)(rng() & 1);
                    c = (uint8_t)((s << 7) | (E << 3) | M);
                } else if (flavor == 2) {                   // mostly big with a few tiny ones (no NaN codes)
                    if (rng() % 8 == 0) { c = (uint8_t)(rng() % 8); }  // subnormal 0..7 * 2^-9
                    else { int E = 10 + (int)(rng() % 6), M = (int)(rng() % 8), s = (int)(rng() & 1); if (E == 15 && M == 7) M = 6; c = (uint8_t)((s << 7) | (E << 3) | M); }
                } else {                                    // 3 / 4: big same-sign (or random-sign) + tiny
                    int s = flavor == 3 ? 0 : (int)(rng() & 1);
                    if (rng() % 8 == 0) { c = (uint8_t)(1 + rng() % 7); }  // subnormal 1..7 * 2^-9, positive
                    else { int E = 13 + (int)(rng() % 3), M = (int)(rng() % 8); if (E == 15 && M == 7) M = 6; c = (uint8_t)((s << 7) | (E << 3) | M); }
                }
                K.a[r][k] = c;
            }
            K.sa[r] = (uint8_t)(flavor == 1 ? 127 : 125 + ue(rng));
            K.sb[r] = (uint8_t)(flavor == 1 ? 127 : 125 + ue(rng));
            for (int k = 0; k < 32; ++k) {
                if (flavor >= 3) {
                    // big e4m3 -> e2m1 in {4, 6}; tiny e4m3 -> e2m1 in {0.5, 1, 1.5}, all positive
                    bool tiny = ((K.a[r][k] >> 3) & 15) == 0;
                    K.b[r][k] = tiny ? (uint8_t)(1 + rng() % 3) : (uint8_t)(6 + rng() % 2);
                } else {
                    K.b[r][k] = (uint8_t)u16(rng);
                }
            }
        }
        if (with_c) {
            for (int r = 0; r < 8; ++r)
                for (int c = 0; c < 8; ++c) {
                    // C with an exponent spread around the block sums (2^-20 .. 2^30), random mantissa, random sign
                    int e = -20 + (int)(rng() % 51);
                    uint32_t mant = (uint32_t)(rng() & 0x7FFFFF);
                    uint32_t bits = ((uint32_t)(e + 127) << 23) | mant | ((uint32_t)(rng() & 1) << 31);
                    memcpy(&K.c[r][c], &bits, 4);
                }
        }
    }
}

int main(int argc, char** argv) {
    int n_cases = 200000;
    std::string tables_dir;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--cases") && i + 1 < argc) n_cases = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tables") && i + 1 < argc) tables_dir = argv[++i];
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s sm_%d%d\n", prop.name, prop.major, prop.minor);

    std::mt19937_64 rng(1234567);
    Case* d_cases = nullptr;
    float* d_out = nullptr;
    CK(cudaMalloc(&d_cases, sizeof(Case) * (size_t)n_cases));
    CK(cudaMalloc(&d_out, sizeof(float) * 64 * (size_t)n_cases));
    std::vector<Case> cases;
    std::vector<float> got((size_t)64 * n_cases);
    const int threads = 256, warps_per_block = threads / 32;
    const int blocks = (n_cases + warps_per_block - 1) / warps_per_block;

    const char* flavors[5] = {"full-range codes", "narrow exponents (sums exact)", "big codes + 1/8 subnormals",
                              "same-sign big (448*6 class) + 1/8 tiny products", "random-sign big + 1/8 tiny products"};
    for (int flavor = 0; flavor < 5; ++flavor) {
        // zero-C
        gen_cases(cases, n_cases, rng, flavor, false);
        CK(cudaMemcpy(d_cases, cases.data(), sizeof(Case) * (size_t)n_cases, cudaMemcpyHostToDevice));
        qmma_kernel<true><<<blocks, threads>>>(d_cases, d_out, n_cases);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), d_out, sizeof(float) * 64 * (size_t)n_cases, cudaMemcpyDeviceToHost));
        Stats S0;
        printf("== zero-C, %s ==\n", flavors[flavor]);
        classify("zero-C", cases, got, S0, 6);
        report("zero-C", S0);
        // accum-C
        gen_cases(cases, n_cases, rng, flavor, true);
        CK(cudaMemcpy(d_cases, cases.data(), sizeof(Case) * (size_t)n_cases, cudaMemcpyHostToDevice));
        qmma_kernel<false><<<blocks, threads>>>(d_cases, d_out, n_cases);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), d_out, sizeof(float) * 64 * (size_t)n_cases, cudaMemcpyDeviceToHost));
        Stats S1;
        printf("== accum-C, %s ==\n", flavors[flavor]);
        classify("accum-C", cases, got, S1, 6);
        report("accum-C", S1);
    }

    // value sanity: a single product check (code tables right?)
    {
        gen_cases(cases, 1, rng, 1, false);
        Case& K = cases[0];
        memset(&K, 0, sizeof(K));
        for (int r = 0; r < 8; ++r) { K.sa[r] = 127; K.sb[r] = 127; }
        K.a[0][0] = 0x40;  // 2.0
        K.b[0][0] = 0x7;   // 6.0
        K.a[1][5] = 0x01;  // 2^-9
        K.b[1][5] = 0x1;   // 0.5
        K.a[2][0] = 0x7E; K.b[2][0] = 0x7;  // 448 * 6 = 2688
        K.a[2][1] = 0xFE; K.b[2][1] = 0x1;  // -448 * 0.5 = -224
        CK(cudaMemcpy(d_cases, cases.data(), sizeof(Case), cudaMemcpyHostToDevice));
        qmma_kernel<true><<<1, 32>>>(d_cases, d_out, 1);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), d_out, sizeof(float) * 64, cudaMemcpyDeviceToHost));
        printf("sanity: (0,0) 2*6 = %g (expect 12), (1,1) 2^-9*0.5 = %g (expect %g), (2,2) 2688-224 = %g (expect 2464), "
               "(2,0) 448*6 = %g (expect 2688); e4m3(0x40)=%g e2m1(7)=%g\n",
               got[0], got[9], ldexp(1.0, -10), got[18], got[16], e4m3_value(0x40), e2m1_value(7));
    }

    if (!tables_dir.empty()) {
        float* d_tab = nullptr;
        CK(cudaMalloc(&d_tab, sizeof(float) * 65536));
        act_table_kernel<<<256, 256>>>(d_tab, 10.0f);
        CK(cudaDeviceSynchronize());
        std::vector<float> act(65536);
        CK(cudaMemcpy(act.data(), d_tab, sizeof(float) * 65536, cudaMemcpyDeviceToHost));
        inv_table_kernel<<<1, 256>>>(d_tab);
        CK(cudaDeviceSynchronize());
        std::vector<float> inv(256);
        CK(cudaMemcpy(inv.data(), d_tab, sizeof(float) * 256, cudaMemcpyDeviceToHost));
        std::string p1 = tables_dir + "/b12x_act_m_limit10.f32", p2 = tables_dir + "/b12x_ue8m0_inv.f32";
        FILE* f = fopen(p1.c_str(), "wb"); fwrite(act.data(), 4, 65536, f); fclose(f);
        f = fopen(p2.c_str(), "wb"); fwrite(inv.data(), 4, 256, f); fclose(f);
        // how approximate: compare with exact 2^(127-byte) and with the fp64 sigmoid
        int inv_inexact = 0;
        for (int b = 1; b < 256; ++b) if (inv[b] != ldexpf(1.f, 127 - b)) inv_inexact++;
        double max_rel = 0; int n_rel = 0; uint32_t worst = 0;
        for (uint32_t i = 0; i < 65536; ++i) {
            float g; uint32_t u = i << 16; memcpy(&g, &u, 4);
            if (!std::isfinite(g)) continue;
            double gg = g > 10.0 ? 10.0 : g;
            double ref = gg / (1.0 + exp(-gg));
            if (ref == 0.0) continue;
            double rel = fabs((double)act[i] - ref) / fabs(ref);
            if (std::isfinite(rel)) { n_rel++; if (rel > max_rel) { max_rel = rel; worst = i; } }
        }
        printf("tables written: %s (65536 fp32), %s (256 fp32); ue8m0_inv entries != exact 2^(127-b): %d; "
               "act m(g) max rel err vs fp64 sigmoid %.3e at bf16 0x%04x\n", p1.c_str(), p2.c_str(), inv_inexact, max_rel, worst);
        CK(cudaFree(d_tab));
    }
    CK(cudaFree(d_cases));
    CK(cudaFree(d_out));
    return 0;
}
