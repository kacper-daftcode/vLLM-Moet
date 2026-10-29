// Read bandwidth of one CCD: T threads pinned to the given CPUs, each streaming its own BUF_MB buffer with
// 64-byte AVX-512 loads (sum into a register, result kept alive). Prints GB/s per thread and aggregate.
//   g++ -O2 -march=znver5 -pthread ccd_bw.cpp -o /tmp/ccd_bw && /tmp/ccd_bw 10,11,12,13,14,15 [BUF_MB=512] [REPS=4]
// 10-29, TR PRO 9975WX (rtxdev2), CCD 1: 1 thread 47.2 GB/s; 3 threads 55.1; 6 threads 57.0; 8 cores 56.9; 12 SMT
// threads 55.2 -> a CCD's DRAM read ceiling is ~57 GB/s (= 82 us per 4,700,160-byte expert record), and one core alone
// reaches 47; a 55 GB/s stream on CCD 3 did not slow a kernel bench on CCD 1 (the limit is per CCD, not the socket).
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

static std::atomic<int> go{0};
struct Res { double gbps; long long sink; };

static void worker(int cpu, size_t bytes, int reps, Res* r) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set); pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    char* buf = (char*)aligned_alloc(64, bytes); memset(buf, 1, bytes);
    while (!go.load()) {}
    __m512i acc = _mm512_setzero_si512();
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < reps; ++k)
        for (size_t o = 0; o < bytes; o += 256) {
            acc = _mm512_add_epi64(acc, _mm512_load_si512((const void*)(buf + o)));
            acc = _mm512_add_epi64(acc, _mm512_load_si512((const void*)(buf + o + 64)));
            acc = _mm512_add_epi64(acc, _mm512_load_si512((const void*)(buf + o + 128)));
            acc = _mm512_add_epi64(acc, _mm512_load_si512((const void*)(buf + o + 192)));
        }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    long long tmp[8]; _mm512_storeu_si512(tmp, acc);
    r->gbps = (double)bytes * reps / s / 1e9; r->sink = tmp[0] + tmp[3];
    free(buf);
}

int main(int argc, char** argv) {
    std::vector<int> cpus; std::string s = argc > 1 ? argv[1] : "10,11,12,13,14,15";
    for (size_t p = 0; p < s.size();) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); cpus.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; }
    size_t mb = argc > 2 ? atoi(argv[2]) : 512; int reps = argc > 3 ? atoi(argv[3]) : 4;
    std::vector<Res> res(cpus.size()); std::vector<std::thread> th;
    for (size_t i = 0; i < cpus.size(); ++i) th.emplace_back(worker, cpus[i], mb << 20, reps, &res[i]);
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); go.store(1);
    for (auto& t : th) t.join();
    double tot = 0; printf("threads %zu x %zu MB x %d:", cpus.size(), mb, reps);
    for (size_t i = 0; i < cpus.size(); ++i) { printf(" %.1f", res[i].gbps); tot += res[i].gbps; }
    printf("  -> aggregate %.1f GB/s (sink %lld)\n", tot, res[0].sink & 1);
    return 0;
}
