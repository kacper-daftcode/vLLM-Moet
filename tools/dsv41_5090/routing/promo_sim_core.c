/* Promote-on-miss simulation of the hybrid's implemented placement (cpu_tier/placement_manager.py) on routing traces.
   Per MoE layer: `nstatic[l]` static residents (never evicted: their records are not in the nonresident store) and
   `spare` slots filled by promotions. A decode step (group = 1 position; 1 + k for a DSpark verify step) routes the
   union of its positions' top-k sets; a missing expert is computed on the CPU and becomes a promotion candidate.
   Candidates (distinct, in miss order, at most `cap` per layer per step) are admitted by `policy`:
     0  every miss (the implemented policy)
     1  a miss whose previous miss in this layer was <= window steps ago (a second miss within the window)
     2  a miss whose misses within the last `window` steps number >= 2 OR whose previous miss was <= window/4 ago
        (approximate "hot enough"; cheap variant of 1 with a shorter grace)
   The victim is the least recently used spare slot (key = last routed step, ties: uses since filled) that was not
   routed in this step (the implemented "no hotter victim" rule with the candidate's key (step, 1)); an empty slot
   first. Outputs per step: pairs, pairs hit, distinct missing experts, promotions (int32 [S][4]).
   build: gcc -O2 -shared -fPIC -o libpromo_sim_core.so promo_sim_core.c */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int promo_sim(const int16_t* ids, int T, int L, int K, int E, int group, const uint8_t* init, int spare, int cap,
              int policy, int window, int32_t* out /* [S][4] */) {
    int S = (T + group - 1) / group;
    int W = group * K;
    int* acc = (int*)malloc(sizeof(int) * W);
    int* seen = (int*)malloc(sizeof(int) * E);
    int* slot_of = (int*)malloc(sizeof(int) * E);       /* -1 = not resident, -2 = static resident, >= 0 spare slot */
    int* prev_miss = (int*)malloc(sizeof(int) * E);
    int* miss_cnt = (int*)malloc(sizeof(int) * E);      /* misses within the window (approximate: decays by reset) */
    int* miss_cnt_step = (int*)malloc(sizeof(int) * E);
    int* sp_expert = (int*)malloc(sizeof(int) * (spare > 0 ? spare : 1));
    int64_t* sp_last = (int64_t*)malloc(sizeof(int64_t) * (spare > 0 ? spare : 1));
    int64_t* sp_uses = (int64_t*)malloc(sizeof(int64_t) * (spare > 0 ? spare : 1));
    if (!acc || !seen || !slot_of || !prev_miss || !miss_cnt || !miss_cnt_step || !sp_expert || !sp_last || !sp_uses) return -1;
    memset(out, 0, sizeof(int32_t) * (size_t)S * 4);
    for (int l = 0; l < L; ++l) {
        for (int e = 0; e < E; ++e) {
            seen[e] = -1; slot_of[e] = init[(size_t)l * E + e] ? -2 : -1; prev_miss[e] = -1000000; miss_cnt[e] = 0; miss_cnt_step[e] = -1000000;
        }
        for (int s = 0; s < spare; ++s) { sp_expert[s] = -1; sp_last[s] = -1; sp_uses[s] = 0; }
        for (int s = 0; s < S; ++s) {
            int n = 0, pairs = 0;
            int t0 = s * group, t1 = t0 + group < T ? t0 + group : T;
            for (int t = t0; t < t1; ++t) {
                const int16_t* p = ids + ((size_t)t * L + l) * K;
                for (int k = 0; k < K; ++k) {
                    int e = p[k];
                    if (e < 0 || e >= E) continue;
                    ++pairs;
                    if (seen[e] != s) { seen[e] = s; acc[n++] = e; }
                }
            }
            int hit_pairs = 0, misses = 0, promos = 0;
            /* hits: refresh the spare slots' keys */
            for (int t = t0; t < t1; ++t) {
                const int16_t* p = ids + ((size_t)t * L + l) * K;
                for (int k = 0; k < K; ++k) {
                    int e = p[k];
                    if (e < 0 || e >= E) continue;
                    int so = slot_of[e];
                    if (so == -2) { ++hit_pairs; }
                    else if (so >= 0) { ++hit_pairs; sp_last[so] = s; sp_uses[so] += 1; }
                }
            }
            /* misses -> candidates (distinct, in miss order) */
            for (int i = 0; i < n; ++i) {
                int e = acc[i];
                if (slot_of[e] != -1) continue;
                ++misses;
                int admit = 0;
                if (policy == 0) admit = 1;
                else if (policy == 1) admit = (s - prev_miss[e]) <= window;
                else {
                    if (s - miss_cnt_step[e] > window) { miss_cnt[e] = 0; }
                    admit = (miss_cnt[e] >= 1) || ((s - prev_miss[e]) <= window / 4);
                }
                /* bookkeeping of the miss history */
                if (s - miss_cnt_step[e] > window) miss_cnt[e] = 0;
                miss_cnt[e] += 1; miss_cnt_step[e] = s;
                prev_miss[e] = s;
                if (!admit || promos >= cap || spare <= 0) continue;
                /* victim: an empty spare slot, else the LRU spare slot not routed in this step */
                int v = -1;
                for (int x = 0; x < spare; ++x) if (sp_expert[x] < 0) { v = x; break; }
                if (v < 0) {
                    int64_t best = 0; int found = 0;
                    for (int x = 0; x < spare; ++x) {
                        if (sp_last[x] >= s) continue;   /* routed in this step: protected / not older than the candidate */
                        int64_t key = sp_last[x] * 64 + (sp_uses[x] < 63 ? sp_uses[x] : 63);
                        if (!found || key < best) { best = key; v = x; found = 1; }
                    }
                }
                if (v < 0) continue;
                if (sp_expert[v] >= 0) slot_of[sp_expert[v]] = -1;
                sp_expert[v] = e; sp_last[v] = s; sp_uses[v] = 1; slot_of[e] = v;
                ++promos;
            }
            int32_t* o = out + (size_t)s * 4;
            o[0] += pairs; o[1] += hit_pairs; o[2] += misses; o[3] += promos;
        }
    }
    free(acc); free(seen); free(slot_of); free(prev_miss); free(miss_cnt); free(miss_cnt_step); free(sp_expert); free(sp_last); free(sp_uses);
    return S;
}
