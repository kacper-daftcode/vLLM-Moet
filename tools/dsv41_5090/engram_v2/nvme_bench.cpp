// nvme_bench.cpp — random O_DIRECT reads of a file (the Engram shard or the expert store) through io_uring: the
// latency / IOPS figures behind the Engram v2 and store-tail designs.
//
//   g++ -O2 -std=c++17 -pthread nvme_bench.cpp -o /tmp/nvme_bench
//   /tmp/nvme_bench FILE --size 512 --qd 1 --threads 1 --secs 3            # steady random reads: latency percentiles, IOPS
//   /tmp/nvme_bench FILE --size 512 --burst 24 --gap-us 7000 --secs 5      # the decode pattern: 24 reads issued at once every
//                                                                          #   7 ms, the time until the LAST completes
//   /tmp/nvme_bench FILE --size 4700160 --qd 4 --threads 1 --secs 3        # expert records
// Every thread has its own ring; offsets are uniform over the file, aligned to --size (or 512 for sizes >= 4096).
#include <fcntl.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "uring.h"

static inline int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

struct Args {
    std::string file;
    unsigned size = 512, qd = 1, threads = 1, burst = 0;
    double secs = 3.0;
    int gap_us = 0;
    int cpu0 = -1;
};

static void pct(std::vector<double>& v, const char* what) {
    if (v.empty()) { printf("%s: no samples\n", what); return; }
    std::sort(v.begin(), v.end());
    auto q = [&](double p) { return v[std::min(v.size() - 1, (size_t)(p * v.size()))]; };
    printf("%s: n %zu  median %.1f us  p90 %.1f  p99 %.1f  p99.9 %.1f  max %.1f\n", what, v.size(), q(0.5), q(0.9), q(0.99),
           q(0.999), v.back());
}

int main(int argc, char** argv) {
    Args a;
    if (argc < 2) { fprintf(stderr, "usage: nvme_bench FILE [--size B] [--qd N] [--threads N] [--secs S] [--burst K --gap-us G] [--cpu0 C]\n"); return 2; }
    a.file = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (k == "--size") a.size = std::stoul(next());
        else if (k == "--qd") a.qd = std::stoul(next());
        else if (k == "--threads") a.threads = std::stoul(next());
        else if (k == "--secs") a.secs = std::stod(next());
        else if (k == "--burst") a.burst = std::stoul(next());
        else if (k == "--gap-us") a.gap_us = std::stoi(next());
        else if (k == "--cpu0") a.cpu0 = std::stoi(next());
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    struct stat st;
    if (stat(a.file.c_str(), &st) != 0) { perror("stat"); return 1; }
    const uint64_t align = a.size >= 4096 ? 4096 : 512;
    const uint64_t nblocks = (uint64_t)st.st_size / a.size;
    std::atomic<int64_t> total_ios{0};
    std::vector<std::vector<double>> lat(a.threads);
    std::vector<std::thread> th;
    const int64_t t_end = now_ns() + (int64_t)(a.secs * 1e9);
    for (unsigned t = 0; t < a.threads; ++t) {
        th.emplace_back([&, t]() {
            if (a.cpu0 >= 0) {
                cpu_set_t set; CPU_ZERO(&set); CPU_SET(a.cpu0 + (int)t, &set);
                pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
            }
            int fd = open(a.file.c_str(), O_RDONLY | O_DIRECT);
            if (fd < 0) { perror("open O_DIRECT"); return; }
            uring::Ring ring;
            const unsigned depth = std::max(a.qd, std::max(a.burst, 1u));
            if (ring.init(std::max(8u, depth * 2)) != 0) { fprintf(stderr, "io_uring_setup failed (seccomp?)\n"); return; }
            std::mt19937_64 rng(1234 + t);
            std::vector<void*> bufs(depth);
            for (auto& b : bufs) b = uring::alloc_aligned(a.size, align);
            std::vector<int64_t> t0(depth);
            if (a.burst) {
                // the decode pattern: K reads at once, the time until the last completion, then a gap
                while (now_ns() < t_end) {
                    const int64_t ts = now_ns();
                    for (unsigned i = 0; i < a.burst; ++i)
                        ring.prep_read(fd, bufs[i], a.size, (rng() % nblocks) * a.size, i);
                    ring.submit(0);
                    unsigned done = 0;
                    while (done < a.burst) {
                        ring.wait(1);
                        uint64_t ud; int res;
                        while (ring.reap(&ud, &res)) {
                            if (res != (int)a.size) fprintf(stderr, "short read %d\n", res);
                            ++done;
                        }
                    }
                    lat[t].push_back((now_ns() - ts) / 1e3);
                    total_ios.fetch_add(a.burst);
                    if (a.gap_us > 0) {
                        const int64_t until = now_ns() + (int64_t)a.gap_us * 1000;
                        while (now_ns() < until) _mm_pause();
                    }
                }
            } else {
                // steady state at queue depth qd: refill after every completion
                for (unsigned i = 0; i < a.qd; ++i) {
                    t0[i] = now_ns();
                    ring.prep_read(fd, bufs[i], a.size, (rng() % nblocks) * a.size, i);
                }
                ring.submit(0);
                while (now_ns() < t_end) {
                    ring.wait(1);
                    uint64_t ud; int res;
                    while (ring.reap(&ud, &res)) {
                        if (res != (int)a.size) fprintf(stderr, "short read %d\n", res);
                        lat[t].push_back((now_ns() - t0[ud]) / 1e3);
                        total_ios.fetch_add(1);
                        t0[ud] = now_ns();
                        ring.prep_read(fd, bufs[ud], a.size, (rng() % nblocks) * a.size, ud);
                    }
                    ring.submit(0);
                }
            }
            ring.close();
            close(fd);
        });
    }
    const int64_t t_start = now_ns();
    for (auto& x : th) x.join();
    const double secs = (now_ns() - t_start) / 1e9;
    std::vector<double> all;
    for (auto& v : lat) all.insert(all.end(), v.begin(), v.end());
    const int64_t ios = total_ios.load();
    printf("%s: size %u B, %s, threads %u, %.1f s: %lld IOs = %.0f IOPS, %.2f GB/s\n", a.file.c_str(), a.size,
           a.burst ? ("burst " + std::to_string(a.burst) + " gap " + std::to_string(a.gap_us) + " us").c_str()
                   : ("qd " + std::to_string(a.qd)).c_str(),
           a.threads, secs, (long long)ios, ios / secs, ios * (double)a.size / secs / 1e9);
    pct(all, a.burst ? "burst completion (all K done)" : "per-read latency");
    return 0;
}
