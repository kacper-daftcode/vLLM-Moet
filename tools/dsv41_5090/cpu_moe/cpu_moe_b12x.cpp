// cpu_moe_b12x.cpp — the CPU twin of b12x's compact W4A8 micro MoE path (FlashInfer 0.7.2, `--moe-backend b12x`) for
// DeepSeek-V4.1-Flash TP4 expert slices: MXFP4 weights (e2m1 + ue8m0 per 32) x MXFP8 activations (e4m3 + ue8m0 per
// 32), bit for bit the GPU's projections, intermediate and route_output rows. AVX-512 BW/VNNI/BF16 (Zen 5).
//
// What the GPU computes (b12x's kernels; verified by tools/dsv41_sm120/b12x_moe_fused and by qmma_probe.cu):
//   FC1   per K32 block: the block-scaled QMMA with a zero C = the exact 32-product sum TRUNCATED to fp32 (RZ);
//         the blocks are added to an fp32 accumulator with fadd.rn in K order; gate | up = bf16_rn(vs * acc),
//         vs = alpha1[e] * input_scale (1 * 1 for the MXFP4 checkpoint).
//   act   SwiGLU with limit 10 through b12x's approximate ex2/rcp, bf16 boundary, amax over 32 columns,
//         sc = amax * fl(1/448) rounded UP to a power of two (ue8m0 byte), inv = ex2.approx(127 - byte),
//         e4m3 = cvt.rn.satfinite(inv * y). On the CPU: m(g) comes from a table generated on the GPU for every bf16
//         g (qmma_probe --tables), y = bf16_rn(u * m(g)) with IEEE multiplies, inv from a 256-entry table.
//   FC2   accumulate-in-C QMMA over the 18 K32 blocks: the addends (C and the 32 products) are aligned to
//         e_ref = max(exp(C), max_k(field exponent of product k)) [field exponents: e4m3 max(E,1)-7, e2m1 max(E,1)-1,
//         plus the two ue8m0 scales], each truncated toward zero to 26 bits below e_ref, summed exactly, and the sum
//         truncated to fp32; route_output = bf16_rn((alpha2 * down_scale * w_pair) * acc).
// Products are exact integers here: e4m3 codes in units of 2^-9 (|.| <= 245,760), e2m1 codes in units of 2^-1
// (|.| <= 12); a block's 32 products sum to < 2^27, so the block sum is an exact int32 and the per-block truncation
// rules are applied to integers. The 16-bit VNNI dot (vpdpwssd) takes the activation units split into 9-bit
// halves (x = hi * 512 + lo), the weights as int16 units.
//
// Expert slice record = the checkpoint's own layout for one TP rank (4,700,160 B; see dsv41_b12x.py):
//   w1 nibbles [576][2560] (low nibble = even k), w1 scales [576][160], w3 nibbles, w3 scales,
//   w2 nibbles [5120][288], w2 scales [5120][18].
// Threading as in cpu_moe_mxfp4.cpp: a pool pinned to one CCD, every expert computed cooperatively (rows striped over
// the workers, barriers between FC1, the activation and FC2).

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr int K1 = 5120;
constexpr int INTER_R = 576;
constexpr int N13 = 2 * INTER_R;
constexpr int K2 = INTER_R;
constexpr int N2 = 5120;
constexpr int B1 = K1 / 32;   // 160 blocks per FC1 row
constexpr int B2 = K2 / 32;   // 18 blocks per FC2 row
constexpr int64_t OFF_W1N = 0;
constexpr int64_t OFF_W1S = OFF_W1N + (int64_t)INTER_R * (K1 / 2);
constexpr int64_t OFF_W3N = OFF_W1S + (int64_t)INTER_R * B1;
constexpr int64_t OFF_W3S = OFF_W3N + (int64_t)INTER_R * (K1 / 2);
constexpr int64_t OFF_W2N = OFF_W3S + (int64_t)INTER_R * B1;
constexpr int64_t OFF_W2S = OFF_W2N + (int64_t)N2 * (K2 / 2);
constexpr int64_t RECORD_BYTES = OFF_W2S + (int64_t)N2 * B2;  // 4,700,160
static_assert(RECORD_BYTES == 4700160, "record layout");
constexpr float CLAMP = 10.0f;
constexpr float FP8_MAX = 448.0f;
constexpr int MAX_M = 64;
constexpr int MAX_THREADS = 64;
constexpr int WINDOW = 26;  // bits kept below e_ref by the accumulate-in-C MMA

// ------------------------------------------------------------------------------------------------------------
// scalar numerics
// ------------------------------------------------------------------------------------------------------------
static inline uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float bits_f32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline float bf16_to_f32(uint16_t b) { return bits_f32((uint32_t)b << 16); }

static inline uint16_t bf16_rne(float f) {
    uint32_t u = f32_bits(f);
    if ((u & 0x7F800000u) == 0x7F800000u) return (uint16_t)((u >> 16) | (((u & 0x7FFFFFu) != 0) ? 0x40u : 0u));
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

// cvt.rn.satfinite.e4m3x2.f32 for finite x
static inline uint8_t f32_to_e4m3_rne(float x) {
    uint32_t u = f32_bits(x);
    uint8_t sign = (u >> 31) ? 0x80 : 0;
    float a = fabsf(x);
    if (a == 0.0f) return sign;
    if (a >= FP8_MAX) {
        // values in [448, 464) round to 448; everything above saturates to 448 (satfinite)
        return sign | 0x7E;
    }
    if (a < 0.015625f) {  // below 2^-6: subnormal, value = m * 2^-9
        float m = nearbyintf(ldexpf(a, 9));
        if (m >= 8.0f) return sign | 0x08;
        return sign | (uint8_t)m;
    }
    int E;
    (void)frexpf(a, &E);
    int e = E - 1;
    float mant = nearbyintf((ldexpf(a, -e) - 1.0f) * 8.0f);
    if (mant >= 8.0f) { mant = 0.0f; e += 1; }
    if (e > 8 || (e == 8 && mant > 6.0f)) return sign | 0x7E;
    return sign | (uint8_t)((e + 7) << 3) | (uint8_t)mant;
}

// b12x's pow2_ceil_ue8m0 on sc = amax * fl(1/448): the ue8m0 byte
static inline uint32_t pow2_ceil_byte(float sc) {
    uint32_t bits = f32_bits(sc);
    if (bits & 0x007FFFFFu) bits = (bits + 0x00800000u) & 0x7F800000u;
    return (bits >> 23) & 0xFFu;
}

static inline int f32_exp(float a) {  // floor(log2 |a|) for normal a; INT_MIN for zero (denormals: as normal-ish)
    uint32_t u = f32_bits(a) & 0x7FFFFFFFu;
    if (u == 0) return INT_MIN;
    int e = (int)(u >> 23);
    if (e == 0) {  // denormal: position of the leading mantissa bit
        int lz = __builtin_clz(u) - 9;
        return -127 - lz;
    }
    return e - 127;
}

// drop the bits of a (toward zero) with weight < 2^e_cut
static inline float trunc_below(float a, int e_cut) {
    if (a == 0.0f) return a;
    int ea = f32_exp(a);
    int s = e_cut - (ea - 23);  // number of low mantissa bits to clear
    if (s <= 0) return a;
    if (s >= 24) return 0.0f;
    uint32_t u = f32_bits(a);
    u = (u >> s) << s;
    return bits_f32(u);
}

// RZ of an integer to fp32
static inline float rz_i64_to_f32(int64_t v) {
    bool neg = v < 0;
    uint64_t u = neg ? (uint64_t)(-v) : (uint64_t)v;
    if (u == 0) return 0.0f;
    int top = 63 - __builtin_clzll(u);
    int shift = top - 23;
    float f;
    if (shift <= 0) f = (float)u;
    else f = ldexpf((float)(u >> shift), shift);
    return neg ? -f : f;
}

static inline float ldexp_exact(float f, int e) {  // f * 2^e for normal results
    if (f == 0.0f) return f;
    uint32_t u = f32_bits(f);
    return bits_f32(u + ((uint32_t)e << 23));
}

static const float INV_FP8_MAX = bits_f32(0x3B124925u);  // fl(1/448), the constant b12x's kernels multiply by

// ------------------------------------------------------------------------------------------------------------
// tables
// ------------------------------------------------------------------------------------------------------------
alignas(64) static int32_t g_e4m3_units[256];     // value / 2^-9 (0x80 -> 0)
alignas(64) static int32_t g_e4m3_fexp[256];      // field exponent max(E,1)-7; zero codes -> -1000
alignas(64) static uint16_t g_e4m3_bf16[256];     // bf16 bits of the value (-0 -> +0)
alignas(64) static int16_t g_e2m1_units16[32];    // vpermw table: code -> value / 2^-1 (codes 8.. negative)
alignas(64) static int32_t g_e2m1_fexp[16];       // field exponent max(E,1)-1; zero codes -> -1000
alignas(64) static float g_act_m[65536];          // m(g) for every bf16 g (limit clamp baked in), from the GPU
alignas(64) static float g_inv[256];              // ex2.approx(127 - byte), 0 for byte 0, from the GPU
static std::atomic<int> g_tables_state{0};        // 0 none, 1 static, 2 static + gpu tables

static void init_static_tables() {
    for (int c = 0; c < 256; ++c) {
        int s = c >> 7, E = (c >> 3) & 15, M = c & 7;
        int64_t units = E == 0 ? M : (int64_t)(8 + M) << (E - 1);
        g_e4m3_units[c] = (int32_t)(s ? -units : units);
        g_e4m3_fexp[c] = (E == 0 && M == 0) ? -1000 : ((E == 0 ? 1 : E) - 7);
        float v = ldexpf((float)units, -9);
        if (s) v = -v;
        uint16_t b = bf16_rne(v);
        if (E == 0 && M == 0) b = 0;
        g_e4m3_bf16[c] = b;
    }
    for (int c = 0; c < 16; ++c) {
        int s = c >> 3, E = (c >> 1) & 3, M = c & 1;
        int units = E == 0 ? M : (2 + M) << (E - 1);
        g_e2m1_units16[c] = (int16_t)(s ? -units : units);
        g_e2m1_units16[16 + c] = g_e2m1_units16[c];
        g_e2m1_fexp[c] = (E == 0 && M == 0) ? -1000 : ((E == 0 ? 1 : E) - 1);
    }
}

// ------------------------------------------------------------------------------------------------------------
// per-token prepared activations
// ------------------------------------------------------------------------------------------------------------
// FC1 operand: x units split into 9-bit halves as int16 lanes in natural k order, plus the per-block scale byte
struct XPrep {
    int16_t hi[K1];
    int16_t lo[K1];
    uint8_t sa[B1];
};
// FC2 operand (after the activation): the 576 intermediate codes as int32 units (FC2 needs the individual products
// for the truncation), their field exponents (for the exact alignment max when acc is small), per block the scale
// byte and the max field exponent of the nonzero codes (the alignment bound)
struct YPrep {
    int32_t x32[K2];
    int16_t fexp[K2];   // field exponent per k (-1000 for zero codes)
    uint8_t sa[B2];
    int8_t ea_max[B2];
};

static inline void split_units(int32_t u, int16_t& hi, int16_t& lo) {
    // u in [-245760, 245760]: u = hi * 512 + lo with lo in [0, 511] (hi carries the sign)
    int32_t h = u >> 9;  // arithmetic: floor
    int32_t l = u - (h << 9);
    hi = (int16_t)h;
    lo = (int16_t)l;
}

static void prepare_x(const uint8_t* aq, const uint8_t* as, XPrep& X) {
    for (int k = 0; k < K1; ++k) split_units(g_e4m3_units[aq[k]], X.hi[k], X.lo[k]);
    memcpy(X.sa, as, B1);
}

// ------------------------------------------------------------------------------------------------------------
// the block kernels
// ------------------------------------------------------------------------------------------------------------
struct Consts {
    __m512i dup_idx;      // word i of the 16 decoded bytes -> lanes 2i, 2i+1
    __m512i lut_units;    // vpermw: nibble -> int16 units
    __m512i lut_fexp;     // vpermw: nibble -> int16 field exponent (-1000 for the zero codes)
    __m512i nib_mask;     // 0x000F per word
};

static Consts make_consts() {
    Consts C;
    alignas(64) int16_t idx[32], fx[32];
    for (int i = 0; i < 32; ++i) {
        idx[i] = (int16_t)(i >> 1);
        fx[i] = (int16_t)g_e2m1_fexp[i & 15];
    }
    C.dup_idx = _mm512_load_si512(idx);
    C.lut_units = _mm512_load_si512(g_e2m1_units16);
    C.lut_fexp = _mm512_load_si512(fx);
    C.nib_mask = _mm512_set1_epi16(0x000F);
    return C;
}

// 16 B of nibbles (one K32 block, low nibble = even k) -> 32 int16 lanes of e2m1 units in k order
static inline __m512i decode_block_units(const uint8_t* p, const Consts& C) {
    __m256i w16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)p));           // byte i -> word i
    __m512i d = _mm512_permutexvar_epi16(C.dup_idx, _mm512_castsi256_si512(w16));      // word i -> lanes 2i, 2i+1
    d = _mm512_mask_srli_epi16(d, 0xAAAAAAAAu, d, 4);                                  // odd lanes: high nibble
    d = _mm512_and_si512(d, C.nib_mask);
    return _mm512_permutexvar_epi16(d, C.lut_units);
}

// the same block's codes (0..15) as int16 lanes (for the field exponents)
static inline __m512i decode_block_codes(const uint8_t* p, const Consts& C) {
    __m256i w16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)p));
    __m512i d = _mm512_permutexvar_epi16(C.dup_idx, _mm512_castsi256_si512(w16));
    d = _mm512_mask_srli_epi16(d, 0xAAAAAAAAu, d, 4);
    return _mm512_and_si512(d, C.nib_mask);
}

// exact block dot, 16 int32 lanes of pair sums (units of 2^-10): sum of lanes = sum_k w_k * x_k
static inline __m512i dot_lanes(__m512i w16, const int16_t* xhi, const int16_t* xlo) {
    __m512i h = _mm512_dpwssd_epi32(_mm512_setzero_si512(), w16, _mm512_loadu_si512((const void*)xhi));
    __m512i l = _mm512_dpwssd_epi32(_mm512_setzero_si512(), w16, _mm512_loadu_si512((const void*)xlo));
    return _mm512_add_epi32(_mm512_slli_epi32(h, 9), l);
}

// 16 lane-sum vectors -> one vector whose lane b is the total of v[b] (a transpose-add network, 45 uops)
static inline __m512i transpose_sum16(const __m512i* v) {
    __m512i t[8], u[4];
    for (int i = 0; i < 8; ++i)
        t[i] = _mm512_add_epi32(_mm512_unpacklo_epi32(v[2 * i], v[2 * i + 1]), _mm512_unpackhi_epi32(v[2 * i], v[2 * i + 1]));
    for (int i = 0; i < 4; ++i)
        u[i] = _mm512_add_epi32(_mm512_unpacklo_epi64(t[2 * i], t[2 * i + 1]), _mm512_unpackhi_epi64(t[2 * i], t[2 * i + 1]));
    __m512i s01 = _mm512_add_epi32(_mm512_shuffle_i32x4(u[0], u[1], 0x44), _mm512_shuffle_i32x4(u[0], u[1], 0xEE));
    __m512i s23 = _mm512_add_epi32(_mm512_shuffle_i32x4(u[2], u[3], 0x44), _mm512_shuffle_i32x4(u[2], u[3], 0xEE));
    return _mm512_add_epi32(_mm512_shuffle_i32x4(s01, s23, 0x88), _mm512_shuffle_i32x4(s01, s23, 0xDD));
}

// zero-C QMMA for 16 blocks at once: RZ of the exact block sums, scaled by 2^(sa + sb - 254 - 10) (exact exponent
// adds; the products of this model never leave the fp32 normal range)
static inline __m512 block_values_rz16(__m512i dots, const uint8_t* sa, const uint8_t* sb) {
    __m512 f = _mm512_cvt_roundepi32_ps(dots, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    __m512i e = _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)sa)),
                                 _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)sb)));
    e = _mm512_slli_epi32(_mm512_sub_epi32(e, _mm512_set1_epi32(264)), 23);
    __m512i fb = _mm512_castps_si512(f);
    __mmask16 nz = _mm512_test_epi32_mask(fb, fb);
    return _mm512_castsi512_ps(_mm512_mask_add_epi32(fb, nz, fb, e));
}

static inline double pow2_d(int e) {  // 2^e as a double (normal range)
    uint64_t u = (uint64_t)(1023 + e) << 52;
    double d;
    memcpy(&d, &u, 8);
    return d;
}

// ------------------------------------------------------------------------------------------------------------
// the pool
// ------------------------------------------------------------------------------------------------------------
struct Job {
    int n_experts = 0;
    const uint8_t* const* records = nullptr;
    const int* m_per_expert = nullptr;
    const int* tok_off = nullptr;       // prefix sums (n_experts + 1)
    const int* token_idx = nullptr;     // concatenated token lists (pair order)
    const float* pair_weight = nullptr; // router weight per pair
    const int* out_row = nullptr;       // output row per pair
    int n_tokens = 0;
    const uint8_t* aq = nullptr;        // [n_tokens][K1] e4m3
    const uint8_t* as = nullptr;        // [n_tokens][B1] ue8m0
    uint16_t* out = nullptr;            // bf16 rows
    int64_t out_stride = 0;
    float vs1 = 1.0f;                   // alpha1 * input_scale
    float vs2 = 1.0f;                   // alpha2 * down_scale
    uint16_t* proj_out = nullptr;       // optional [pairs][N13] bf16 projections (tests)
    uint8_t* inter_q_out = nullptr;     // optional [pairs][K2] e4m3 intermediate codes (tests)
    uint8_t* inter_s_out = nullptr;     // optional [pairs][B2] scale bytes (tests)
};

struct Pool {
    int nthreads = 0;
    std::vector<int> cpus;
    std::vector<std::thread> threads;
    std::atomic<uint64_t> gen{0};
    std::atomic<int> done{0};
    std::atomic<bool> stop{false};
    std::atomic<int> bar_count{0};
    std::atomic<int> bar_gen{0};
    std::atomic<int> error{0};
    Job job;
    std::vector<int> tok_off_storage;
    XPrep* X = nullptr;                 // [x_cap]
    int x_cap = 0;
    uint16_t* proj[2] = {nullptr, nullptr};   // [MAX_M][N13] bf16, double-buffered per expert
    YPrep* Y[2] = {nullptr, nullptr};         // [MAX_M]
    Consts C;
    // statistics (per worker, merged on request): FC2 blocks, FC2 blocks through the general path; tid 0's cycle
    // counts per phase (prepare, FC1, activation, FC2; barrier waits included)
    struct alignas(64) Stats { int64_t fc2_blocks = 0, fc2_general = 0, cyc[4] = {0, 0, 0, 0}; } stats[MAX_THREADS];
};

static inline int64_t rdtsc_now() { return (int64_t)__rdtsc(); }

static void* big_alloc(int64_t nbytes) {
    void* p = mmap(nullptr, (size_t)nbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    madvise(p, (size_t)nbytes, MADV_HUGEPAGE);
    memset(p, 0, (size_t)nbytes);
    return p;
}
static void big_free(void* p, int64_t nbytes) { if (p) munmap(p, (size_t)nbytes); }

static inline void barrier(Pool& P) {
    int g = P.bar_gen.load(std::memory_order_acquire);
    if (P.bar_count.fetch_add(1, std::memory_order_acq_rel) + 1 == P.nthreads) {
        P.bar_count.store(0, std::memory_order_relaxed);
        P.bar_gen.store(g + 1, std::memory_order_release);
    } else {
        while (P.bar_gen.load(std::memory_order_acquire) == g) _mm_pause();
    }
}

static inline void split_rows(int total, int nthreads, int tid, int& r0, int& r1) {
    int per = (total + nthreads - 1) / nthreads;
    r0 = std::min(total, tid * per);
    r1 = std::min(total, r0 + per);
}

// ---- FC1: rows [r0, r1) of the 1152 (gate rows 0..575 = w1, up rows 576..1151 = w3) for m tokens ---------------
static void fc1_rows(const uint8_t* rec, int r0, int r1, const XPrep* const* X, int m, uint16_t* proj, float vs1,
                     const Consts& C) {
    for (int r = r0; r < r1; ++r) {
        const bool up = r >= INTER_R;
        const int n = up ? r - INTER_R : r;
        const uint8_t* nib = rec + (up ? OFF_W3N : OFF_W1N) + (int64_t)n * (K1 / 2);
        const uint8_t* sc = rec + (up ? OFF_W3S : OFF_W1S) + (int64_t)n * B1;
        float acc[MAX_M];
        for (int t = 0; t < m; ++t) acc[t] = 0.0f;
        for (int g = 0; g < B1 / 16; ++g) {  // 16 blocks = 512 k per group
            __m512i w16[16];
            for (int i = 0; i < 16; ++i) w16[i] = decode_block_units(nib + (g * 16 + i) * 16, C);
            for (int t = 0; t < m; ++t) {
                __m512i s[16];
                const int16_t* xh = X[t]->hi + g * 512;
                const int16_t* xl = X[t]->lo + g * 512;
                for (int i = 0; i < 16; ++i) s[i] = dot_lanes(w16[i], xh + i * 32, xl + i * 32);
                const __m512 v = block_values_rz16(transpose_sum16(s), X[t]->sa + g * 16, sc + g * 16);
                alignas(64) float fb[16];
                _mm512_store_ps(fb, v);
                float a = acc[t];
                for (int i = 0; i < 16; ++i) a = a + fb[i];  // fadd.rn, K order
                acc[t] = a;
            }
        }
        for (int t = 0; t < m; ++t) proj[(int64_t)t * N13 + r] = bf16_rne(vs1 * acc[t]);
    }
}

// ---- activation: one 32-group of one token's projections -> codes, scale, prepared FC2 operand ----------------
static void act_group(const uint16_t* prow, int grp, YPrep& Y, uint8_t* q_out) {
    const int j0 = grp * 32;
    float y[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const uint16_t gb = prow[j0 + i];
        float u = bf16_to_f32(prow[INTER_R + j0 + i]);
        u = u >= CLAMP ? CLAMP : u;
        u = u <= -CLAMP ? -CLAMP : u;
        const float m = g_act_m[gb];
        const float x = u * m;             // mul.rn
        const float v = bf16_to_f32(bf16_rne(x));
        y[i] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    const float sc = amax * INV_FP8_MAX;    // mul.rn
    const uint32_t byte = pow2_ceil_byte(sc);
    const float inv = g_inv[byte];
    Y.sa[grp] = (uint8_t)byte;
    int ea_max = -1000;
    for (int i = 0; i < 32; ++i) {
        const uint8_t code = f32_to_e4m3_rne(inv * y[i]);
        if (q_out) q_out[j0 + i] = code;
        Y.x32[j0 + i] = g_e4m3_units[code];
        const int fe = g_e4m3_fexp[code];
        Y.fexp[j0 + i] = (int16_t)fe;
        ea_max = std::max(ea_max, fe);
    }
    Y.ea_max[grp] = (int8_t)std::max(ea_max, -100);
}

// ---- FC2: the accumulate-in-C QMMA chain on the CPU ------------------------------------------------------------
// Per block the alignment exponent is e_ref = max(exp(acc), M) with M the max field exponent of the block's nonzero
// products (M <= sc + 10). Products are integers in units of 2^(sc - 10); the window keeps bits >= 2^(e_ref - 25),
// i.e. the products are truncated toward zero to multiples of 2^q units with q = e_ref - sc - 15 (q <= 0: nothing),
// and acc to the same bound (only possible when M > exp(acc) + 2, i.e. acc below a quarter of the largest product:
// rare, so that branch recomputes M exactly; the products' truncation is unconditional and branch-free, q = 0 being
// the identity). The truncated addends are summed exactly in fp64 and truncated to fp32.
struct Fc2Block {
    __m512i wlo, whi;  // the block's 32 e2m1 units as int32 lanes
};

static inline Fc2Block fc2_decode(const uint8_t* nib16, const Consts& C) {
    const __m512i w16 = decode_block_units(nib16, C);
    Fc2Block B;
    B.wlo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(w16));
    B.whi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(w16, 1));
    return B;
}

// the exact alignment max M of one block (the rarer path: acc below a quarter of the largest possible product):
// field exponent of weight + activation per k (zero codes carry -1000 and drop out of the max), max over the 32
static inline int fc2_block_max_fexp(const uint8_t* nib16, const YPrep& Y, int b, int sc, const Consts& C) {
    const __m512i codes = decode_block_codes(nib16, C);
    __m512i v = _mm512_add_epi16(_mm512_permutexvar_epi16(codes, C.lut_fexp),
                                 _mm512_loadu_si512((const void*)(Y.fexp + b * 32)));
    v = _mm512_max_epi16(v, _mm512_shuffle_i64x2(v, v, 0x4E));
    v = _mm512_max_epi16(v, _mm512_shuffle_i64x2(v, v, 0xB1));
    v = _mm512_max_epi16(v, _mm512_shuffle_epi32(v, (_MM_PERM_ENUM)0x4E));
    v = _mm512_max_epi16(v, _mm512_shuffle_epi32(v, (_MM_PERM_ENUM)0xB1));
    v = _mm512_max_epi16(v, _mm512_srli_epi32(v, 16));
    const int M = (int)(int16_t)(_mm_cvtsi128_si32(_mm512_castsi512_si128(v)) & 0xFFFF);
    return M < -500 ? INT_MIN : M + sc;
}

// four rows' lane vectors -> their four totals in one xmm (13 uops)
static inline __m128i reduce4_epi32(__m512i v0, __m512i v1, __m512i v2, __m512i v3) {
    const __m512i t01 = _mm512_add_epi32(_mm512_unpacklo_epi32(v0, v1), _mm512_unpackhi_epi32(v0, v1));
    const __m512i t23 = _mm512_add_epi32(_mm512_unpacklo_epi32(v2, v3), _mm512_unpackhi_epi32(v2, v3));
    const __m512i u = _mm512_add_epi32(_mm512_unpacklo_epi64(t01, t23), _mm512_unpackhi_epi64(t01, t23));
    const __m256i h = _mm256_add_epi32(_mm512_castsi512_si256(u), _mm512_extracti64x4_epi64(u, 1));
    return _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
}

// one block of the chains of four rows (acc4 = their accumulators) for one token
static inline __m128 fc2_step4(__m128 acc4, const Fc2Block* W, const uint8_t* const* nib16, const YPrep& Y, int b,
                               const int32_t* sb4, const Consts& C, int64_t& n_rare) {
    alignas(16) float accs[4], acc_t[4];
    alignas(16) int32_t sc4[4];
    int q[4];
    _mm_store_ps(accs, acc4);
    const int sa = Y.sa[b];
    const int ea = Y.ea_max[b];
    for (int r = 0; r < 4; ++r) {
        const int sc = sa + sb4[r] - 254;
        sc4[r] = sc;
        q[r] = 0;
        acc_t[r] = accs[r];
        if (accs[r] != 0.0f) {
            const int e_acc = f32_exp(accs[r]);
            const int bound = ea + 2 + sc;
            if (e_acc >= bound) {
                q[r] = std::max(e_acc - sc - 15, 0);
            } else if (bound > e_acc + 2) {
                ++n_rare;
                const int M = fc2_block_max_fexp(nib16[r], Y, b, sc, C);
                if (M > e_acc + 2) acc_t[r] = trunc_below(accs[r], M - (WINDOW - 1));
            }
        }
    }
    const __m512i xlo = _mm512_loadu_si512((const void*)(Y.x32 + b * 32));
    const __m512i xhi = _mm512_loadu_si512((const void*)(Y.x32 + b * 32 + 16));
    __m512i t[4];
    for (int r = 0; r < 4; ++r) {
        const __m512i pa = _mm512_mullo_epi32(W[r].wlo, xlo);
        const __m512i pb = _mm512_mullo_epi32(W[r].whi, xhi);
        const __m512i cnt = _mm512_set1_epi32(std::min(q[r], 31));
        const __mmask16 na = _mm512_cmplt_epi32_mask(pa, _mm512_setzero_si512());
        const __mmask16 nb = _mm512_cmplt_epi32_mask(pb, _mm512_setzero_si512());
        __m512i ta = _mm512_sllv_epi32(_mm512_srlv_epi32(_mm512_abs_epi32(pa), cnt), cnt);
        __m512i tb = _mm512_sllv_epi32(_mm512_srlv_epi32(_mm512_abs_epi32(pb), cnt), cnt);
        ta = _mm512_mask_sub_epi32(ta, na, _mm512_setzero_si512(), ta);
        tb = _mm512_mask_sub_epi32(tb, nb, _mm512_setzero_si512(), tb);
        t[r] = _mm512_add_epi32(ta, tb);
    }
    const __m128i dots = reduce4_epi32(t[0], t[1], t[2], t[3]);
    // 2^(sc - 10) per row as doubles: exponent field 1023 + sc - 10
    const __m256i eb = _mm256_slli_epi64(_mm256_add_epi64(_mm256_cvtepi32_epi64(_mm_load_si128((const __m128i*)sc4)),
                                                          _mm256_set1_epi64x(1013)), 52);
    const __m256d total = _mm256_fmadd_pd(_mm256_cvtepi32_pd(dots), _mm256_castsi256_pd(eb), _mm256_cvtps_pd(_mm_load_ps(acc_t)));
    return _mm256_castps256_ps128(_mm512_cvt_roundpd_ps(_mm512_castpd256_pd512(total), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}

// rows in groups of eight: two independent four-row chains per token, the blocks of all eight rows decoded once
static void fc2_rows(const uint8_t* rec, int r0, int r1, const YPrep* const* Y, int m, const float* scales,
                     uint16_t* const* out_rows, const Consts& C, Pool::Stats& st) {
    constexpr int RG = 8;
    st.fc2_blocks += (int64_t)(r1 - r0) * m * B2;
    for (int n0 = r0; n0 < r1; n0 += RG) {
        const int R = std::min(RG, r1 - n0);
        const uint8_t* nib[RG];
        alignas(16) int32_t sb[B2][RG];
        for (int r = 0; r < RG; ++r) {
            const int n = n0 + std::min(r, R - 1);  // tail rows duplicate the last row (their results are dropped)
            nib[r] = rec + OFF_W2N + (int64_t)n * (K2 / 2);
            const uint8_t* s = rec + OFF_W2S + (int64_t)n * B2;
            for (int b = 0; b < B2; ++b) sb[b][r] = s[b];
        }
        Fc2Block W[B2][RG];
        for (int b = 0; b < B2; ++b)
            for (int r = 0; r < RG; ++r) W[b][r] = fc2_decode(nib[r] + b * 16, C);
        for (int t = 0; t < m; ++t) {
            const YPrep& Yt = *Y[t];
            __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
            for (int b = 0; b < B2; ++b) {
                const uint8_t* nb0[4] = {nib[0] + b * 16, nib[1] + b * 16, nib[2] + b * 16, nib[3] + b * 16};
                const uint8_t* nb1[4] = {nib[4] + b * 16, nib[5] + b * 16, nib[6] + b * 16, nib[7] + b * 16};
                a0 = fc2_step4(a0, W[b], nb0, Yt, b, sb[b], C, st.fc2_general);
                a1 = fc2_step4(a1, W[b] + 4, nb1, Yt, b, sb[b] + 4, C, st.fc2_general);
            }
            alignas(32) float accs[RG];
            _mm_store_ps(accs, a0);
            _mm_store_ps(accs + 4, a1);
            for (int r = 0; r < R; ++r) out_rows[t][n0 + r] = bf16_rne(scales[t] * accs[r]);
        }
    }
}

static void run_job(Pool& P, int tid) {
    const Job& J = P.job;
    const int n = P.nthreads;
    Pool::Stats& st = P.stats[tid];
    int64_t t0 = rdtsc_now();
    // phase 0: prepared activations, tokens striped
    for (int t = tid; t < J.n_tokens; t += n) prepare_x(J.aq + (int64_t)t * K1, J.as + (int64_t)t * B1, P.X[t]);
    barrier(P);
    int64_t t1 = rdtsc_now();
    st.cyc[0] += t1 - t0;
    const XPrep* xptr[MAX_M];
    const YPrep* yptr[MAX_M];
    float scales[MAX_M];
    uint16_t* orow[MAX_M];
    for (int e = 0; e < J.n_experts; ++e) {
        const uint8_t* rec = J.records[e];
        const int m = J.m_per_expert[e];
        const int p0 = J.tok_off[e];
        for (int i = 0; i < m; ++i) xptr[i] = &P.X[J.token_idx[p0 + i]];
        uint16_t* proj = P.proj[e & 1];
        YPrep* Yb = P.Y[e & 1];
        int r0, r1;
        split_rows(N13, n, tid, r0, r1);
        t0 = rdtsc_now();
        fc1_rows(rec, r0, r1, xptr, m, proj, J.vs1, P.C);
        barrier(P);
        t1 = rdtsc_now();
        st.cyc[1] += t1 - t0;
        for (int item = tid; item < m * (K2 / 32); item += n) {
            const int i = item / (K2 / 32), grp = item % (K2 / 32);
            act_group(proj + (int64_t)i * N13, grp, Yb[i],
                      J.inter_q_out ? J.inter_q_out + (int64_t)(p0 + i) * K2 : nullptr);
        }
        barrier(P);
        t0 = rdtsc_now();
        st.cyc[2] += t0 - t1;
        if (J.proj_out && tid == 0)
            memcpy(J.proj_out + (int64_t)p0 * N13, proj, (size_t)m * N13 * sizeof(uint16_t));
        if (J.inter_s_out && tid == 0)
            for (int i = 0; i < m; ++i) memcpy(J.inter_s_out + (int64_t)(p0 + i) * B2, Yb[i].sa, B2);
        for (int i = 0; i < m; ++i) {
            yptr[i] = &Yb[i];
            scales[i] = J.vs2 * J.pair_weight[p0 + i];  // mul.rn, as the GPU's down_scale * token_weight
            orow[i] = J.out + (int64_t)J.out_row[p0 + i] * J.out_stride;
        }
        split_rows(N2, n, tid, r0, r1);
        fc2_rows(rec, r0, r1, yptr, m, scales, orow, P.C, st);
        st.cyc[3] += rdtsc_now() - t0;
        // the next expert's FC1 writes the other proj buffer; its activation the other Y buffer: no barrier needed
        // here beyond the FC1 barrier that follows (all workers pass it before anyone activates)
    }
}

static void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void worker_main(Pool* P, int tid) {
    pin_to_cpu(P->cpus[tid]);
    uint64_t seen = 0;
    while (!P->stop.load(std::memory_order_acquire)) {
        uint64_t g;
        while ((g = P->gen.load(std::memory_order_acquire)) == seen) {
            if (P->stop.load(std::memory_order_acquire)) return;
            _mm_pause();
        }
        seen = g;
        run_job(*P, tid);
        P->done.fetch_add(1, std::memory_order_acq_rel);
    }
}

// ------------------------------------------------------------------------------------------------------------
// scalar reference (independent code path: exact integers, the probe's model literally)
// ------------------------------------------------------------------------------------------------------------
static inline int nib_code(const uint8_t* nib, int k) { return (nib[k >> 1] >> ((k & 1) * 4)) & 15; }
static inline int e2m1_units(int c) { int s = c >> 3, E = (c >> 1) & 3, M = c & 1; int u = E == 0 ? M : (2 + M) << (E - 1); return s ? -u : u; }

static void ref_fc1(const uint8_t* rec, const uint8_t* aq, const uint8_t* as, float vs1, uint16_t* proj) {
    for (int r = 0; r < N13; ++r) {
        const bool up = r >= INTER_R;
        const int nrow = up ? r - INTER_R : r;
        const uint8_t* nib = rec + (up ? OFF_W3N : OFF_W1N) + (int64_t)nrow * (K1 / 2);
        const uint8_t* sc = rec + (up ? OFF_W3S : OFF_W1S) + (int64_t)nrow * B1;
        float acc = 0.0f;
        for (int b = 0; b < B1; ++b) {
            int64_t dot = 0;
            for (int k = b * 32; k < b * 32 + 32; ++k) dot += (int64_t)g_e4m3_units[aq[k]] * e2m1_units(nib_code(nib, k));
            const int e = (int)as[b] + (int)sc[b] - 254 - 10;
            const float v = ldexpf(rz_i64_to_f32(dot), e);
            acc = acc + v;
        }
        proj[r] = bf16_rne(vs1 * acc);
    }
}

static void ref_act(const uint16_t* proj, uint8_t* q, uint8_t* sbytes) {
    for (int grp = 0; grp < B2; ++grp) {
        float y[32], amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const int j = grp * 32 + i;
            float u = bf16_to_f32(proj[INTER_R + j]);
            u = u >= CLAMP ? CLAMP : u;
            u = u <= -CLAMP ? -CLAMP : u;
            y[i] = bf16_to_f32(bf16_rne(u * g_act_m[proj[j]]));
            amax = fmaxf(amax, fabsf(y[i]));
        }
        const uint32_t byte = pow2_ceil_byte(amax * INV_FP8_MAX);
        sbytes[grp] = (uint8_t)byte;
        for (int i = 0; i < 32; ++i) q[grp * 32 + i] = f32_to_e4m3_rne(g_inv[byte] * y[i]);
    }
}

// one accumulate-in-C block with the probe's model (e_ref=4, w=26, every addend truncated, RZ)
static float ref_fc2_block(float acc, const uint8_t* nib, int b, const uint8_t* q, int sa, int sb) {
    const int sc = sa + sb - 254;
    int64_t prods[32];
    int top_ref = f32_exp(acc);
    for (int i = 0; i < 32; ++i) {
        const int k = b * 32 + i;
        const int cw = nib_code(nib, k);
        prods[i] = (int64_t)g_e4m3_units[q[k]] * e2m1_units(cw);
        if (prods[i] != 0) top_ref = std::max(top_ref, g_e4m3_fexp[q[k]] + g_e2m1_fexp[cw] + sc);
    }
    if (top_ref == INT_MIN) return acc;  // everything zero
    const int e_cut = top_ref - (WINDOW - 1);  // keep bits with weight >= 2^e_cut
    // products: integers in units of 2^(sc - 10); truncate each toward zero to multiples of 2^e_cut
    const int q_shift = e_cut - (sc - 10);
    __int128 total = 0;  // in units of 2^e_unit
    const int e_unit = std::min(e_cut, sc - 10);
    for (int i = 0; i < 32; ++i) {
        int64_t p = prods[i];
        if (q_shift >= 1) {
            if (q_shift >= 62) p = 0;
            else {
                int64_t a = p < 0 ? -p : p;
                a = (a >> q_shift) << q_shift;
                p = p < 0 ? -a : a;
            }
        }
        total += (__int128)p << ((sc - 10) - e_unit);
    }
    // acc truncated to the window, then in units of 2^e_unit (exact: acc is a multiple of 2^(e_acc - 23) >= 2^e_cut after truncation)
    const float acc_t = trunc_below(acc, e_cut);
    if (acc_t != 0.0f) {
        const int ea = f32_exp(acc_t);
        uint32_t u = f32_bits(acc_t);
        int64_t mant = (ea >= -126) ? (int64_t)((u & 0x7FFFFFu) | 0x800000u) : (int64_t)(u & 0x7FFFFFu);
        const int e_lsb = (ea >= -126) ? ea - 23 : -149;
        // the bits below e_cut are zero after the truncation, so a right shift down to e_unit is exact
        __int128 av = e_lsb >= e_unit ? ((__int128)mant << (e_lsb - e_unit)) : ((__int128)mant >> (e_unit - e_lsb));
        total += (u >> 31) ? -av : av;
    }
    // RZ to fp32: |total| < 2^60 in practice; convert through the exact integer path
    bool neg = total < 0;
    unsigned __int128 mag = neg ? (unsigned __int128)(-total) : (unsigned __int128)total;
    if (mag == 0) return 0.0f;
    int top = 127;
    while (!((mag >> top) & 1)) --top;
    int shift = top - 23;
    uint64_t m24 = shift > 0 ? (uint64_t)(mag >> shift) : (uint64_t)(mag << (-shift));
    float f = ldexpf((float)m24, shift + e_unit);
    return neg ? -f : f;
}

static void ref_fc2(const uint8_t* rec, const uint8_t* q, const uint8_t* sbytes, float scale, uint16_t* out) {
    for (int nrow = 0; nrow < N2; ++nrow) {
        const uint8_t* nib = rec + OFF_W2N + (int64_t)nrow * (K2 / 2);
        const uint8_t* sc = rec + OFF_W2S + (int64_t)nrow * B2;
        float acc = 0.0f;
        for (int b = 0; b < B2; ++b) acc = ref_fc2_block(acc, nib, b, q, sbytes[b], sc[b]);
        out[nrow] = bf16_rne(scale * acc);
    }
}

}  // namespace

// ------------------------------------------------------------------------------------------------------------
// C API
// ------------------------------------------------------------------------------------------------------------
extern "C" {

int64_t moe_b12x_record_bytes(void) { return RECORD_BYTES; }
void* moe_b12x_alloc(int64_t nbytes) { return big_alloc(nbytes); }
void moe_b12x_free(void* p, int64_t nbytes) { big_free(p, nbytes); }

// act_m: 65536 fp32 (m(g) per bf16 pattern), inv: 256 fp32 — the tables qmma_probe --tables writes
int moe_b12x_init_tables(const float* act_m, const float* inv) {
    init_static_tables();
    memcpy(g_act_m, act_m, sizeof(g_act_m));
    memcpy(g_inv, inv, sizeof(g_inv));
    g_tables_state.store(2, std::memory_order_release);
    return 0;
}

void* moe_b12x_pool_create(int nthreads, const int* cpu_ids) {
    if (g_tables_state.load(std::memory_order_acquire) < 2) return nullptr;
    if (nthreads < 1 || nthreads > MAX_THREADS) return nullptr;
    Pool* P = new Pool();
    P->nthreads = nthreads;
    P->cpus.assign(cpu_ids, cpu_ids + nthreads);
    P->C = make_consts();
    P->x_cap = 256;
    P->X = (XPrep*)big_alloc((int64_t)P->x_cap * sizeof(XPrep));
    for (int b = 0; b < 2; ++b) {
        P->proj[b] = (uint16_t*)big_alloc((int64_t)MAX_M * N13 * sizeof(uint16_t));
        P->Y[b] = (YPrep*)big_alloc((int64_t)MAX_M * sizeof(YPrep));
    }
    for (int t = 0; t < nthreads; ++t) P->threads.emplace_back(worker_main, P, t);
    return P;
}

void moe_b12x_pool_destroy(void* pool) {
    Pool* P = (Pool*)pool;
    if (!P) return;
    P->stop.store(true, std::memory_order_release);
    for (auto& th : P->threads) th.join();
    big_free(P->X, (int64_t)P->x_cap * sizeof(XPrep));
    for (int b = 0; b < 2; ++b) {
        big_free(P->proj[b], (int64_t)MAX_M * N13 * sizeof(uint16_t));
        big_free(P->Y[b], (int64_t)MAX_M * sizeof(YPrep));
    }
    delete P;
}

// n_experts experts (records[e] = the slice record), m_per_expert[e] tokens each; token_idx / pair_weight / out_row
// concatenated in pair order; n_tokens prepared activations aq [n_tokens][5120] e4m3, as [n_tokens][160] ue8m0
// (b12x's rows quant); out: bf16 rows (out_row[p] * out_stride) = route_output rows. Optional test outputs:
// proj_out [pairs][1152] bf16, inter_q_out [pairs][576] e4m3, inter_s_out [pairs][18] bytes. Blocking; 0 on success.
int moe_b12x_experts_forward(void* pool, int n_experts, const uint8_t* const* records, const int* m_per_expert,
                             const int* token_idx, const float* pair_weight, const int* out_row, int n_tokens,
                             const uint8_t* aq, const uint8_t* as, uint16_t* out, int64_t out_stride, float vs1,
                             float vs2, uint16_t* proj_out, uint8_t* inter_q_out, uint8_t* inter_s_out) {
    Pool* P = (Pool*)pool;
    if (!P || n_experts < 0) return -1;
    for (int e = 0; e < n_experts; ++e)
        if (m_per_expert[e] < 1 || m_per_expert[e] > MAX_M) return -2;
    if (n_tokens > P->x_cap) {
        big_free(P->X, (int64_t)P->x_cap * sizeof(XPrep));
        P->x_cap = n_tokens;
        P->X = (XPrep*)big_alloc((int64_t)P->x_cap * sizeof(XPrep));
        if (!P->X) return -3;
    }
    P->tok_off_storage.assign(n_experts + 1, 0);
    for (int e = 0; e < n_experts; ++e) P->tok_off_storage[e + 1] = P->tok_off_storage[e] + m_per_expert[e];
    Job& J = P->job;
    J = Job();
    J.n_experts = n_experts;
    J.records = records;
    J.m_per_expert = m_per_expert;
    J.tok_off = P->tok_off_storage.data();
    J.token_idx = token_idx;
    J.pair_weight = pair_weight;
    J.out_row = out_row;
    J.n_tokens = n_tokens;
    J.aq = aq;
    J.as = as;
    J.out = out;
    J.out_stride = out_stride;
    J.vs1 = vs1;
    J.vs2 = vs2;
    J.proj_out = proj_out;
    J.inter_q_out = inter_q_out;
    J.inter_s_out = inter_s_out;
    P->done.store(0, std::memory_order_release);
    P->gen.fetch_add(1, std::memory_order_acq_rel);
    while (P->done.load(std::memory_order_acquire) < P->nthreads) _mm_pause();
    return 0;
}

// statistics since the last reset: out[0] FC2 blocks, out[1] FC2 blocks through the general path, out[2..5] worker
// 0's TSC cycles in prepare / FC1 / activation / FC2 (barrier waits included); reset != 0 clears them
void moe_b12x_pool_stats(void* pool, int64_t* out, int reset) {
    Pool* P = (Pool*)pool;
    if (!P) return;
    int64_t blocks = 0, general = 0;
    for (int t = 0; t < P->nthreads; ++t) { blocks += P->stats[t].fc2_blocks; general += P->stats[t].fc2_general; }
    out[0] = blocks;
    out[1] = general;
    for (int i = 0; i < 4; ++i) out[2 + i] = P->stats[0].cyc[i];
    if (reset)
        for (int t = 0; t < P->nthreads; ++t) P->stats[t] = Pool::Stats();
}

// the scalar reference for one (token, expert) pair: proj [1152] bf16, q [576], sbytes [18], out [5120] bf16
int moe_b12x_reference_pair(const uint8_t* record, const uint8_t* aq, const uint8_t* as, float weight, float vs1,
                            float vs2, uint16_t* proj, uint8_t* q, uint8_t* sbytes, uint16_t* out) {
    if (g_tables_state.load(std::memory_order_acquire) < 2) return -1;
    ref_fc1(record, aq, as, vs1, proj);
    ref_act(proj, q, sbytes);
    ref_fc2(record, q, sbytes, vs2 * weight, out);
    return 0;
}

// b12x's MXFP8 rows quant (quantize_mxfp8_rows_cute, linear order): bf16 [n][5120] -> codes + scale bytes.
// zero_byte: the scale byte written for an all-zero group (0 on the <= 8-row specialization, 127 above it)
void moe_b12x_rows_quant(const uint16_t* x, int n, uint8_t* q, uint8_t* s, int zero_byte) {
    init_static_tables();
    for (int t = 0; t < n; ++t) {
        for (int b = 0; b < B1; ++b) {
            float amax = 0.0f;
            float v[32];
            for (int i = 0; i < 32; ++i) {
                v[i] = bf16_to_f32(x[(int64_t)t * K1 + b * 32 + i]);
                amax = fmaxf(amax, fabsf(v[i]));
            }
            uint32_t byte = pow2_ceil_byte(amax * INV_FP8_MAX);
            if (amax == 0.0f) byte = (uint32_t)zero_byte;
            const float inv = g_inv[byte];
            s[(int64_t)t * B1 + b] = (uint8_t)byte;
            for (int i = 0; i < 32; ++i) q[(int64_t)t * K1 + b * 32 + i] = f32_to_e4m3_rne(v[i] * inv);
        }
    }
}

}  // extern "C"
