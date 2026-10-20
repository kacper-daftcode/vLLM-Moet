/* Expert-cache simulation core for DeepSeek-V4.1-Flash routing traces (PLAN-dsv41-5090-cpu 1.3).
   ids: int16 [T][L][K] routed experts per position and MoE layer. Steps = groups of `group` consecutive
   positions (1 = plain decode, 1 + k = a DSpark verify step); a step's accesses in a layer are the union of
   its positions' top-k sets. One cache per layer with `cap[l]` slots; policy 0 = static (resident set fixed =
   init), 1 = LRU (fetch on miss, insert, evict the least recently used). Writes per step and layer the number
   of distinct accesses and of misses (int16 [S][L]).
   build: gcc -O2 -shared -fPIC -o libcache_sim_core.so cache_sim_core.c */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int sim_cache(const int16_t* ids, int T, int L, int K, int E, int group, const int* cap, const uint8_t* init,
              int policy, int16_t* out_acc, int16_t* out_miss) {
    int S = (T + group - 1) / group;
    int W = group * K;
    int* acc = (int*)malloc(sizeof(int) * W);
    int* seen = (int*)malloc(sizeof(int) * E);
    uint8_t* res = (uint8_t*)malloc(E);
    int* key = (int*)malloc(sizeof(int) * E);
    if (!acc || !seen || !res || !key) return -1;
    for (int l = 0; l < L; ++l) {
        int c = cap[l];
        for (int e = 0; e < E; ++e) { seen[e] = -1; res[e] = init[(size_t)l * E + e]; key[e] = -1; }
        int nres = 0;
        for (int e = 0; e < E; ++e) nres += res[e];
        for (int s = 0; s < S; ++s) {
            int n = 0;
            int t0 = s * group, t1 = t0 + group < T ? t0 + group : T;
            for (int t = t0; t < t1; ++t) {
                const int16_t* p = ids + ((size_t)t * L + l) * K;
                for (int k = 0; k < K; ++k) {
                    int e = p[k];
                    if (e < 0 || e >= E) continue;
                    if (seen[e] != s) { seen[e] = s; acc[n++] = e; }
                }
            }
            int m = 0;
            for (int i = 0; i < n; ++i) {
                int e = acc[i];
                if (res[e]) { key[e] = s; continue; }
                ++m;
                if (policy == 1) {
                    if (nres < c) { res[e] = 1; key[e] = s; ++nres; }
                    else {
                        int v = -1;
                        for (int x = 0; x < E; ++x) if (res[x] && (v < 0 || key[x] < key[v])) v = x;
                        if (v >= 0) { res[v] = 0; }
                        res[e] = 1; key[e] = s;
                    }
                }
            }
            out_acc[(size_t)s * L + l] = (int16_t)n;
            out_miss[(size_t)s * L + l] = (int16_t)m;
        }
    }
    free(acc); free(seen); free(res); free(key);
    return S;
}
