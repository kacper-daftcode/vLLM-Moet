// cpu_moe_mxfp4.cpp — DeepSeek-V4.1-Flash MXFP4 expert slices on the CPU (AVX-512 BW/VBMI/BF16, Zen 5).
//
// One TP4 rank's slice of a routed expert: gate/up [576, 5120] and down [5120, 576] in the checkpoint's
// MXFP4 record (e2m1 nibbles, one UE8M0 scale per 32 k), computed for m tokens exactly like vLLM's
// DeepGemmFP4Experts on the GPU up to fp32 summation order:
//   FC1 (fp8 x fp4, fp32 acc) -> bf16 -> clamp(10) SwiGLU in fp32 -> bf16 -> fp8 per 128 (UE8M0) -> FC2 -> bf16.
// The nibbles are turned into bf16 by two byte LUTs (vpermb) and an unpack; the power-of-two scales of both
// the weights (per 32) and the activations (per 128) are folded into the bf16 exponents exactly, so the
// inner loop is pure vdpbf16ps with fp32 accumulation. The kernel is DRAM-bound: one expert slice is
// 4.81 MB in the pack layout (4.70 MB of record bytes); a Zen 5 CCD streams ~50-55 GB/s.
//
// Pack layout (dsv41_mxfp4.py builds it): row-major, row = [scales | nibble chunks].
//   W13: 1152 rows (576 gate, 576 up) x 2752 B: 192 B scales (160 used, natural block order) + 40 x 64 B.
//   W2:  5120 rows x 320 B: 32 B scales (18 used) + 4 x 64 B + one 32-B half chunk.
//   A 64-B chunk = 128 consecutive k. Low nibbles hold k 0..63, high nibbles k 64..127, in the byte order
//   that makes vpunpcklbw / vpunpckhbw of the LUT'd low/high bytes produce the 32-k scale blocks in order
//   (byte b: lane = b/16, pos = b%16; pos < 8 -> k = lane*8 + pos, else k = 32 + lane*8 + pos - 8).
//
// Threading: a pool of workers pinned to one CCD; every expert is computed cooperatively (rows striped
// over the workers, one barrier between FC1 and FC2; every worker computes the 576-wide activation of all
// m tokens itself so that FC2 needs no second barrier). The job interface is a list of experts with their
// token lists; the outputs are bf16 rows in (expert, token) order — the rows the GPU's gather consumes.
//
// MIT-licensed design notes: the handoff design of the later phases follows exllamav3's cpu/moe_handoff.cu.

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
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
constexpr int CH = 128;
constexpr int W13_SCALE_BYTES = 192;
constexpr int W13_ROW_BYTES = W13_SCALE_BYTES + K1 / 2;  // 2752
constexpr int W2_SCALE_BYTES = 32;
constexpr int W2_ROW_BYTES = W2_SCALE_BYTES + K2 / 2;    // 320
constexpr int64_t W13_BYTES = (int64_t)N13 * W13_ROW_BYTES;
constexpr int64_t W2_BYTES = (int64_t)N2 * W2_ROW_BYTES;
constexpr int64_t PACK_BYTES = W13_BYTES + W2_BYTES;
constexpr int ACT_GROUP = 128;
constexpr int X_GROUPS = K1 / ACT_GROUP;                       // 40
constexpr int Y_GROUPS = (K2 + ACT_GROUP - 1) / ACT_GROUP;     // 5 (the last one is 64 wide)
constexpr float CLAMP = 10.0f;
constexpr float FP8_MAX = 448.0f;
constexpr int MAX_M = 64;        // tokens per expert the pool's buffers hold
constexpr int MAX_THREADS = 64;
constexpr int MAX_M_KERNEL = 8;  // tokens per kernel pass (register budget); larger m loops in passes

// ------------------------------------------------------------------------------------------------------------
// scalar numerics
// ------------------------------------------------------------------------------------------------------------
static inline uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float bits_f32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline float bf16_to_f32(uint16_t b) { return bits_f32((uint32_t)b << 16); }

static inline uint16_t bf16_rne(float f) {
    uint32_t u = f32_bits(f);
    if ((u & 0x7F800000u) == 0x7F800000u) {  // inf / nan: keep, quiet the nan
        return (uint16_t)((u >> 16) | (((u & 0x7FFFFFu) != 0) ? 0x40u : 0u));
    }
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

// fp32 (already clamped to +-448) -> e4m3fn code, round to nearest even, like cvt.rn.satfinite.e4m3x2.f32
static inline uint8_t f32_to_e4m3_rne(float x) {
    uint32_t u = f32_bits(x);
    uint8_t sign = (u >> 31) ? 0x80 : 0;
    float a = fabsf(x);
    if (a == 0.0f || std::isnan(x)) return sign;  // nan cannot occur after the clamp
    if (a >= FP8_MAX) return sign | 0x7E;         // 448 = 1.75 * 2^8
    if (a < 0.015625f) {                          // below 2^-6: subnormal, value = m * 2^-9
        float m = nearbyintf(ldexpf(a, 9));
        if (m >= 8.0f) return sign | 0x08;
        return sign | (uint8_t)m;
    }
    int E;
    (void)frexpf(a, &E);            // a = f * 2^E, f in [0.5, 1)
    int e = E - 1;                  // floor(log2 a) in [-6, 8]
    float mant = nearbyintf((ldexpf(a, -e) - 1.0f) * 8.0f);
    if (mant >= 8.0f) { mant = 0.0f; e += 1; }
    if (e > 8 || (e == 8 && mant > 6.0f)) return sign | 0x7E;
    return sign | (uint8_t)((e + 7) << 3) | (uint8_t)mant;
}

// exact ceil(log2(max(amax / 448, 1e-10))) with the fp32 division the GPU kernel does
static inline int ue8m0_exponent(float amax) {
    float raw = std::max(amax / FP8_MAX, 1e-10f);
    int e;
    float m = frexpf(raw, &e);      // raw = m * 2^e, m in [0.5, 1): log2(raw) in (e-1, e]
    if (m == 0.5f) e -= 1;          // exact power of two
    return e;
}

alignas(64) static uint16_t g_e4m3_to_bf16[256];
static std::atomic<bool> g_tables_ready{false};

static void init_tables() {
    for (int c = 0; c < 256; ++c) {
        int s = c >> 7, E = (c >> 3) & 0xF, m = c & 7;
        float v;
        if (E == 15 && m == 7) v = NAN;
        else if (E == 0) v = ldexpf((float)m, -9);
        else v = ldexpf(1.0f + (float)m / 8.0f, E - 7);
        if (s) v = -v;
        uint16_t b = bf16_rne(v);
        if (E == 0 && m == 0) b = 0;  // -0 -> +0: the exponent add is masked on (bits != 0)
        g_e4m3_to_bf16[c] = b;
    }
    g_tables_ready.store(true, std::memory_order_release);
}

static inline void ensure_tables() {
    if (!g_tables_ready.load(std::memory_order_acquire)) init_tables();
}

// e4m3 codes with UE8M0 exponents (byte = e + 127, one per 128) -> bf16 with the scale folded in
static void prepare_x_row(const uint8_t* xq, const uint8_t* xs, uint16_t* xb16, int K, int group) {
    for (int k = 0; k < K; ++k) {
        uint16_t b = g_e4m3_to_bf16[xq[k]];
        if (b != 0) b = (uint16_t)(b + (uint16_t)((((int)xs[k / group]) - 127) << 7));
        xb16[k] = b;
    }
}

// FC1 fp32 outputs (gate g, up u) of one token -> the FC2 input of one 128-wide quantization group:
// bf16 of the fp8-requantized clamp-SwiGLU activation with its UE8M0 scale folded in. Optionally emits the
// fp8 codes and the scale byte (tests).
static void act_quant_group(const float* g, const float* u, int grp, uint16_t* yb16, uint8_t* yq_out,
                            uint8_t* ys_out) {
    const int j0 = grp * ACT_GROUP, j1 = std::min(K2, j0 + ACT_GROUP);
    float y[ACT_GROUP];
    float amax = 0.0f;
    for (int j = j0; j < j1; ++j) {
        float gg = bf16_to_f32(bf16_rne(g[j]));
        float uu = bf16_to_f32(bf16_rne(u[j]));
        gg = std::min(gg, CLAMP);
        uu = std::min(std::max(uu, -CLAMP), CLAMP);
        float glu = gg / (1.0f + expf(-gg));
        float v = bf16_to_f32(bf16_rne(uu * glu));
        y[j - j0] = v;
        amax = std::max(amax, fabsf(v));
    }
    int e = ue8m0_exponent(amax);
    float inv = ldexpf(1.0f, -e);
    uint16_t add = (uint16_t)(e << 7);
    if (ys_out) ys_out[grp] = (uint8_t)std::min(std::max(e + 127, 0), 255);
    for (int j = j0; j < j1; ++j) {
        float q = std::min(std::max(y[j - j0] * inv, -FP8_MAX), FP8_MAX);
        uint8_t code = f32_to_e4m3_rne(q);
        if (yq_out) yq_out[j] = code;
        uint16_t b = g_e4m3_to_bf16[code];
        if (b != 0) b = (uint16_t)(b + add);
        yb16[j] = b;
    }
}

static void act_quant_row(const float* g, const float* u, uint16_t* yb16, uint8_t* yq_out, uint8_t* ys_out) {
    for (int grp = 0; grp < Y_GROUPS; ++grp) act_quant_group(g, u, grp, yb16, yq_out, ys_out);
}

// ------------------------------------------------------------------------------------------------------------
// the vector kernel
// ------------------------------------------------------------------------------------------------------------
struct Tables {
    __m512i tbl_lo, tbl_hi;
};

static Tables make_tables() {
    // bf16 of e2m1 codes 0..15: 0, .5, 1, 1.5, 2, 3, 4, 6, -0(->+0), -.5, ... ; low and high bytes
    static const uint8_t lo[16] = {0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
                                   0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
    static const uint8_t hi[16] = {0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
                                   0x00, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};
    alignas(64) uint8_t tl[64], th[64];
    for (int i = 0; i < 64; ++i) { tl[i] = lo[i & 15]; th[i] = hi[i & 15]; }
    Tables t;
    t.tbl_lo = _mm512_load_si512(tl);
    t.tbl_hi = _mm512_load_si512(th);
    return t;
}

static inline __m512i scale_add(uint8_t sbyte) {
    return _mm512_set1_epi16((short)((((int)sbyte) - 127) << 7));
}

static inline __m512i apply_scale(__m512i w, __m512i add) {
    return _mm512_mask_add_epi16(w, _mm512_test_epi16_mask(w, w), w, add);
}

// 64 B of nibbles (128 k) + 4 scale bytes -> 4 bf16 vectors of 32 k each (blocks A B C D in k order)
static inline void decode_chunk(const uint8_t* p, const uint8_t* sc, const Tables& T,
                                __m512i& wA, __m512i& wB, __m512i& wC, __m512i& wD) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    __m512i v = _mm512_loadu_si512((const void*)p);
    __m512i lo = _mm512_and_si512(v, m4);
    __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), m4);
    __m512i loL = _mm512_permutexvar_epi8(lo, T.tbl_lo);
    __m512i loH = _mm512_permutexvar_epi8(lo, T.tbl_hi);
    __m512i hiL = _mm512_permutexvar_epi8(hi, T.tbl_lo);
    __m512i hiH = _mm512_permutexvar_epi8(hi, T.tbl_hi);
    wA = apply_scale(_mm512_unpacklo_epi8(loL, loH), scale_add(sc[0]));
    wB = apply_scale(_mm512_unpackhi_epi8(loL, loH), scale_add(sc[1]));
    wC = apply_scale(_mm512_unpacklo_epi8(hiL, hiH), scale_add(sc[2]));
    wD = apply_scale(_mm512_unpackhi_epi8(hiL, hiH), scale_add(sc[3]));
}

// 32 B of nibbles (64 k) + 2 scale bytes -> 4 vectors with 16 valid k each in lanes 0..15 (A: k 0..15,
// B: 16..31 share scale 0; C: 32..47, D: 48..63 share scale 1); lanes 16..31 are zero.
static inline void decode_half_chunk(const uint8_t* p, const uint8_t* sc, const Tables& T,
                                     __m512i& wA, __m512i& wB, __m512i& wC, __m512i& wD) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    __m512i v = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)p));
    __m512i lo = _mm512_and_si512(v, m4);
    __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), m4);
    __m512i loL = _mm512_permutexvar_epi8(lo, T.tbl_lo);
    __m512i loH = _mm512_permutexvar_epi8(lo, T.tbl_hi);
    __m512i hiL = _mm512_permutexvar_epi8(hi, T.tbl_lo);
    __m512i hiH = _mm512_permutexvar_epi8(hi, T.tbl_hi);
    __m512i s0 = scale_add(sc[0]), s1 = scale_add(sc[1]);
    wA = apply_scale(_mm512_unpacklo_epi8(loL, loH), s0);
    wB = apply_scale(_mm512_unpackhi_epi8(loL, loH), s0);
    wC = apply_scale(_mm512_unpacklo_epi8(hiL, hiH), s1);
    wD = apply_scale(_mm512_unpackhi_epi8(hiL, hiH), s1);
}

static inline __m512i load_x16(const uint16_t* p) {  // 16 bf16 into lanes 0..15, upper lanes zero
    return _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)p));
}

// nrows rows of a pack section, K columns, M tokens. out_f32[t*out_stride + col0 + r] or bf16 out.
template <int M, bool BF16_OUT>
static void fc_rows(const uint8_t* rows, int nrows, int row_bytes, int scale_bytes, int K,
                    const uint16_t* const* x, float* out_f32, uint16_t* out_b16, int64_t out_stride,
                    int col0, const Tables& T) {
    constexpr int NACC = (M == 1) ? 4 : ((M == 2) ? 2 : 1);
    const int nfull = K / CH;
    const bool half = (K % CH) != 0;
    for (int r = 0; r < nrows; ++r) {
        const uint8_t* row = rows + (int64_t)r * row_bytes;
        const uint8_t* sc = row;
        const uint8_t* nib = row + scale_bytes;
        __m512 acc[M][NACC];
        for (int t = 0; t < M; ++t)
            for (int a = 0; a < NACC; ++a) acc[t][a] = _mm512_setzero_ps();
        for (int c = 0; c < nfull; ++c) {
            __m512i wA, wB, wC, wD;
            decode_chunk(nib + (int64_t)c * 64, sc + c * 4, T, wA, wB, wC, wD);
            const int k0 = c * CH;
            for (int t = 0; t < M; ++t) {
                const uint16_t* xp = x[t] + k0;
                acc[t][0 % NACC] = _mm512_dpbf16_ps(acc[t][0 % NACC], (__m512bh)wA,
                                                    (__m512bh)_mm512_loadu_si512((const void*)(xp)));
                acc[t][1 % NACC] = _mm512_dpbf16_ps(acc[t][1 % NACC], (__m512bh)wB,
                                                    (__m512bh)_mm512_loadu_si512((const void*)(xp + 32)));
                acc[t][2 % NACC] = _mm512_dpbf16_ps(acc[t][2 % NACC], (__m512bh)wC,
                                                    (__m512bh)_mm512_loadu_si512((const void*)(xp + 64)));
                acc[t][3 % NACC] = _mm512_dpbf16_ps(acc[t][3 % NACC], (__m512bh)wD,
                                                    (__m512bh)_mm512_loadu_si512((const void*)(xp + 96)));
            }
        }
        if (half) {
            __m512i wA, wB, wC, wD;
            decode_half_chunk(nib + (int64_t)nfull * 64, sc + nfull * 4, T, wA, wB, wC, wD);
            const int k0 = nfull * CH;
            for (int t = 0; t < M; ++t) {
                const uint16_t* xp = x[t] + k0;
                acc[t][0 % NACC] = _mm512_dpbf16_ps(acc[t][0 % NACC], (__m512bh)wA, (__m512bh)load_x16(xp));
                acc[t][1 % NACC] = _mm512_dpbf16_ps(acc[t][1 % NACC], (__m512bh)wB, (__m512bh)load_x16(xp + 16));
                acc[t][2 % NACC] = _mm512_dpbf16_ps(acc[t][2 % NACC], (__m512bh)wC, (__m512bh)load_x16(xp + 32));
                acc[t][3 % NACC] = _mm512_dpbf16_ps(acc[t][3 % NACC], (__m512bh)wD, (__m512bh)load_x16(xp + 48));
            }
        }
        for (int t = 0; t < M; ++t) {
            __m512 s = acc[t][0];
            for (int a = 1; a < NACC; ++a) s = _mm512_add_ps(s, acc[t][a]);
            float v = _mm512_reduce_add_ps(s);
            if (BF16_OUT) out_b16[t * out_stride + col0 + r] = bf16_rne(v);
            else out_f32[t * out_stride + col0 + r] = v;
        }
    }
}

template <bool BF16_OUT>
static void fc_rows_dispatch(int m, const uint8_t* rows, int nrows, int row_bytes, int scale_bytes, int K,
                             const uint16_t* const* x, float* out_f32, uint16_t* out_b16, int64_t out_stride,
                             int col0, const Tables& T) {
    switch (m) {
        case 1: fc_rows<1, BF16_OUT>(rows, nrows, row_bytes, scale_bytes, K, x, out_f32, out_b16, out_stride, col0, T); break;
        case 2: fc_rows<2, BF16_OUT>(rows, nrows, row_bytes, scale_bytes, K, x, out_f32, out_b16, out_stride, col0, T); break;
        case 3: fc_rows<3, BF16_OUT>(rows, nrows, row_bytes, scale_bytes, K, x, out_f32, out_b16, out_stride, col0, T); break;
        case 4: fc_rows<4, BF16_OUT>(rows, nrows, row_bytes, scale_bytes, K, x, out_f32, out_b16, out_stride, col0, T); break;
        case 5: fc_rows<5, BF16_OUT>(rows, nrows, row_bytes, scale_bytes, K, x, out_f32, out_b16, out_stride, col0, T); break;
        case 6: fc_rows<6, BF16_OUT>(rows, nrows, row_bytes, scale_bytes, K, x, out_f32, out_b16, out_stride, col0, T); break;
        case 7: fc_rows<7, BF16_OUT>(rows, nrows, row_bytes, scale_bytes, K, x, out_f32, out_b16, out_stride, col0, T); break;
        default: fc_rows<8, BF16_OUT>(rows, nrows, row_bytes, scale_bytes, K, x, out_f32, out_b16, out_stride, col0, T); break;
    }
}

// ------------------------------------------------------------------------------------------------------------
// the pool
// ------------------------------------------------------------------------------------------------------------
struct Job {
    int n_experts = 0;
    const uint8_t* const* packs = nullptr;
    const int* m_per_expert = nullptr;
    const int* tok_off = nullptr;     // prefix sums of m_per_expert (n_experts + 1)
    const int* token_idx = nullptr;   // concatenated token lists
    int n_tokens = 0;
    const uint8_t* xq = nullptr;      // [n_tokens][K1]
    const uint8_t* xs = nullptr;      // [n_tokens][X_GROUPS]
    uint16_t* out = nullptr;          // [tok_off[n_experts]][K1] bf16
    bool read_only = false;           // benchmark: stream the packs, compute nothing
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
    Job job;
    std::vector<int> tok_off_storage;
    float* gu[2] = {nullptr, nullptr};          // [MAX_M][N13] each
    uint16_t* xb16 = nullptr;                    // [xb16_cap][K1]
    int64_t xb16_cap = 0;
    uint16_t* ybuf[2] = {nullptr, nullptr};      // [MAX_M][K2] each, double-buffered like gu
    uint64_t read_sink[MAX_THREADS] = {};
    Tables T;
};

static void* big_alloc(int64_t nbytes) {
    void* p = mmap(nullptr, (size_t)nbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    madvise(p, (size_t)nbytes, MADV_HUGEPAGE);
    memset(p, 0, (size_t)nbytes);
    return p;
}

static void big_free(void* p, int64_t nbytes) {
    if (p) munmap(p, (size_t)nbytes);
}

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

// read-only pass over the experts' packs with the FC1/FC2 row striping (the DRAM ceiling of this placement)
static void run_read_job(Pool& P, int tid) {
    const Job& J = P.job;
    const int n = P.nthreads;
    __m512i acc = _mm512_setzero_si512();
    for (int e = 0; e < J.n_experts; ++e) {
        const uint8_t* pack = J.packs[e];
        int r0, r1;
        split_rows(N13, n, tid, r0, r1);
        const uint8_t* p = pack + (int64_t)r0 * W13_ROW_BYTES;
        const uint8_t* end = pack + (int64_t)r1 * W13_ROW_BYTES;
        for (; p + 64 <= end; p += 64) acc = _mm512_xor_si512(acc, _mm512_loadu_si512((const void*)p));
        split_rows(N2, n, tid, r0, r1);
        p = pack + W13_BYTES + (int64_t)r0 * W2_ROW_BYTES;
        end = pack + W13_BYTES + (int64_t)r1 * W2_ROW_BYTES;
        for (; p + 64 <= end; p += 64) acc = _mm512_xor_si512(acc, _mm512_loadu_si512((const void*)p));
    }
    alignas(64) uint64_t lanes[8];
    _mm512_store_si512((void*)lanes, acc);
    uint64_t x = 0;
    for (int i = 0; i < 8; ++i) x ^= lanes[i];
    P.read_sink[tid] = x;
}

static void run_job(Pool& P, int tid) {
    const Job& J = P.job;
    if (J.read_only) { run_read_job(P, tid); return; }
    const int n = P.nthreads;
    // phase 0: prepared activations, tokens striped over the workers
    for (int t = tid; t < J.n_tokens; t += n)
        prepare_x_row(J.xq + (int64_t)t * K1, J.xs + (int64_t)t * X_GROUPS, P.xb16 + (int64_t)t * K1, K1, ACT_GROUP);
    barrier(P);
    const uint16_t* xptr[MAX_M];
    const uint16_t* yptr[MAX_M];
    for (int e = 0; e < J.n_experts; ++e) {
        const uint8_t* pack = J.packs[e];
        const int m = J.m_per_expert[e];
        const int* toks = J.token_idx + J.tok_off[e];
        float* gu = P.gu[e & 1];
        for (int i = 0; i < m; ++i) xptr[i] = P.xb16 + (int64_t)toks[i] * K1;
        // FC1: gate | up rows striped
        int r0, r1;
        split_rows(N13, n, tid, r0, r1);
        for (int t0 = 0; t0 < m; t0 += MAX_M_KERNEL) {
            int mm = std::min(MAX_M_KERNEL, m - t0);
            fc_rows_dispatch<false>(mm, pack + (int64_t)r0 * W13_ROW_BYTES, r1 - r0, W13_ROW_BYTES, W13_SCALE_BYTES,
                                    K1, xptr + t0, gu + (int64_t)t0 * N13, nullptr, N13, r0, P.T);
        }
        barrier(P);
        // activation + requant: (token, 128-group) items striped over the workers into the shared y buffer
        uint16_t* yb = P.ybuf[e & 1];
        for (int item = tid; item < m * Y_GROUPS; item += n) {
            int i = item / Y_GROUPS, grp = item % Y_GROUPS;
            act_quant_group(gu + (int64_t)i * N13, gu + (int64_t)i * N13 + INTER_R, grp, yb + (int64_t)i * K2,
                            nullptr, nullptr);
        }
        for (int i = 0; i < m; ++i) yptr[i] = yb + (int64_t)i * K2;
        barrier(P);
        // FC2: down rows striped, bf16 rows in (expert, token) order
        split_rows(N2, n, tid, r0, r1);
        uint16_t* out_rows = J.out + (int64_t)J.tok_off[e] * K1;
        for (int t0 = 0; t0 < m; t0 += MAX_M_KERNEL) {
            int mm = std::min(MAX_M_KERNEL, m - t0);
            fc_rows_dispatch<true>(mm, pack + W13_BYTES + (int64_t)r0 * W2_ROW_BYTES, r1 - r0, W2_ROW_BYTES,
                                   W2_SCALE_BYTES, K2, yptr + t0, nullptr, out_rows + (int64_t)t0 * K1, K1, r0, P.T);
        }
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

}  // namespace

// ------------------------------------------------------------------------------------------------------------
// C API
// ------------------------------------------------------------------------------------------------------------
extern "C" {

int64_t moe_mxfp4_pack_bytes(void) { return PACK_BYTES; }

void* moe_mxfp4_alloc(int64_t nbytes) { return big_alloc(nbytes); }
void moe_mxfp4_free(void* p, int64_t nbytes) { big_free(p, nbytes); }

void* moe_mxfp4_pool_create(int nthreads, const int* cpu_ids) {
    ensure_tables();
    if (nthreads < 1 || nthreads > MAX_THREADS) return nullptr;
    Pool* P = new Pool();
    P->nthreads = nthreads;
    P->cpus.assign(cpu_ids, cpu_ids + nthreads);
    P->T = make_tables();
    for (int b = 0; b < 2; ++b) P->gu[b] = (float*)big_alloc((int64_t)MAX_M * N13 * sizeof(float));
    for (int b = 0; b < 2; ++b) P->ybuf[b] = (uint16_t*)big_alloc((int64_t)MAX_M * K2 * sizeof(uint16_t));
    P->xb16_cap = 256;
    P->xb16 = (uint16_t*)big_alloc(P->xb16_cap * K1 * sizeof(uint16_t));
    for (int t = 0; t < nthreads; ++t) P->threads.emplace_back(worker_main, P, t);
    return P;
}

void moe_mxfp4_pool_destroy(void* pool) {
    Pool* P = (Pool*)pool;
    if (!P) return;
    P->stop.store(true, std::memory_order_release);
    for (auto& th : P->threads) th.join();
    for (int b = 0; b < 2; ++b) big_free(P->gu[b], (int64_t)MAX_M * N13 * sizeof(float));
    for (int b = 0; b < 2; ++b) big_free(P->ybuf[b], (int64_t)MAX_M * K2 * sizeof(uint16_t));
    big_free(P->xb16, P->xb16_cap * K1 * sizeof(uint16_t));
    delete P;
}

// n_experts experts, packs[e], m_per_expert[e] tokens each (token_idx concatenated), n_tokens prepared
// activations xq [n_tokens][5120] e4m3 codes, xs [n_tokens][40] UE8M0 bytes (e + 127);
// out: bf16 [sum m][5120] rows in (expert, token) order. Blocking; returns 0 on success.
int moe_mxfp4_experts_forward(void* pool, int n_experts, const uint8_t* const* packs, const int* m_per_expert,
                              const int* token_idx, int n_tokens, const uint8_t* xq, const uint8_t* xs,
                              uint16_t* out) {
    Pool* P = (Pool*)pool;
    if (!P || n_experts < 0) return -1;
    for (int e = 0; e < n_experts; ++e)
        if (m_per_expert[e] < 1 || m_per_expert[e] > MAX_M) return -2;
    if (n_tokens > P->xb16_cap) {
        big_free(P->xb16, P->xb16_cap * K1 * sizeof(uint16_t));
        P->xb16_cap = n_tokens;
        P->xb16 = (uint16_t*)big_alloc(P->xb16_cap * K1 * sizeof(uint16_t));
        if (!P->xb16) return -3;
    }
    P->tok_off_storage.assign(n_experts + 1, 0);
    for (int e = 0; e < n_experts; ++e) P->tok_off_storage[e + 1] = P->tok_off_storage[e] + m_per_expert[e];
    Job& J = P->job;
    J.n_experts = n_experts;
    J.packs = packs;
    J.m_per_expert = m_per_expert;
    J.tok_off = P->tok_off_storage.data();
    J.token_idx = token_idx;
    J.n_tokens = n_tokens;
    J.xq = xq;
    J.xs = xs;
    J.out = out;
    J.read_only = false;
    P->done.store(0, std::memory_order_release);
    P->gen.fetch_add(1, std::memory_order_acq_rel);
    while (P->done.load(std::memory_order_acquire) < P->nthreads) _mm_pause();
    return 0;
}

// benchmark helper: stream n_experts packs through the workers (same striping as the kernel), no compute
uint64_t moe_mxfp4_read_bench(void* pool, int n_experts, const uint8_t* const* packs) {
    Pool* P = (Pool*)pool;
    if (!P) return 0;
    Job& J = P->job;
    J = Job();
    J.n_experts = n_experts;
    J.packs = packs;
    J.read_only = true;
    P->done.store(0, std::memory_order_release);
    P->gen.fetch_add(1, std::memory_order_acq_rel);
    while (P->done.load(std::memory_order_acquire) < P->nthreads) _mm_pause();
    uint64_t x = 0;
    for (int t = 0; t < P->nthreads; ++t) x ^= P->read_sink[t];
    return x;
}

// test helpers
void moe_mxfp4_prepare_x(const uint8_t* xq, const uint8_t* xs, uint16_t* xb16, int n_tokens) {
    ensure_tables();
    for (int t = 0; t < n_tokens; ++t)
        prepare_x_row(xq + (int64_t)t * K1, xs + (int64_t)t * X_GROUPS, xb16 + (int64_t)t * K1, K1, ACT_GROUP);
}

void moe_mxfp4_act_quant(const float* g, const float* u, uint16_t* yb16, uint8_t* yq, uint8_t* ys, int m) {
    ensure_tables();
    for (int i = 0; i < m; ++i)
        act_quant_row(g + (int64_t)i * K2, u + (int64_t)i * K2, yb16 + (int64_t)i * K2,
                      yq ? yq + (int64_t)i * K2 : nullptr, ys ? ys + (int64_t)i * Y_GROUPS : nullptr);
}

}  // extern "C"
