// handoff_proto.cu — phase 2.0 of HANDOFF-moet-hybrid: the GPU<->CPU handoff of the hybrid inside ONE captured CUDA
// graph ("graph or death"), measured standalone (no vLLM, no torch).
//
// Per decode step the graph runs L layers; per layer:
//   k_work     — a placeholder for the GPU's own layer work (configurable µs)
//   k_publish  — writes the layer's missing pairs (ids) + the step id into a mailbox in MAPPED PINNED host memory
//                (system-scope release store), i.e. the routing becomes visible to the host without a sync
//   wait       — the graph blocks until the host has produced the rows: (a) a spin kernel reading the mailbox's ack
//                with system-scope acquire loads and a bounded timeout (fail-closed), or (b) a cuStreamWaitValue32
//                node captured into the graph (reset-to-0 protocol)
//   inject     — the host's bf16 rows enter the GPU: (z) a zero-copy kernel reading the pinned staging rows over
//                PCIe, or (m) a cudaMemcpyAsync H2D node
// One host thread plays the CPU tier: polls req, "computes" for --host-us, writes n rows into pinned staging, acks.
// Timestamps (%globaltimer) around publish and wait give the round trip per layer with no host work.
//
// build: nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -o handoff_proto handoff_proto.cu -lcuda
// run:   CUDA_VISIBLE_DEVICES=1 ./handoff_proto --steps 2000 --rows 8 --wait spin|memop --inject zerocopy|memcpy
//        [--host-us 0] [--work-us 300] [--timeout-ms 2000] [--kill-host-at STEP]
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/atomic>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CKD(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = nullptr; cuGetErrorString(r_, &s_); fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, s_ ? s_ : "?"); exit(1); } } while (0)

constexpr int L = 40;
constexpr int ROW = 5120;        // bf16 per row (hidden size)
constexpr int MAXROWS = 64;

struct alignas(64) Mailbox {
    alignas(64) unsigned req;    // GPU -> host: step id published (spin protocol) ; written 0 then step (memop protocol)
    alignas(64) unsigned ack;    // host -> GPU: step id done (spin) ; 1 = done (memop)
    alignas(64) unsigned n;
    int ids[MAXROWS];
    int rowslot[MAXROWS];
};

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void k_work(float* buf, int iters) {
    float v = buf[blockIdx.x * blockDim.x + threadIdx.x];
    for (int i = 0; i < iters; ++i) v = v * 0.999f + 0.001f;
    buf[blockIdx.x * blockDim.x + threadIdx.x] = v;
}

__global__ void k_bump(unsigned* p) { if (threadIdx.x == 0) *p = *p + 1; }

// publish: ids + n into the mailbox, then (optionally reset ack) and release-store the step id as the request
__global__ void k_publish(Mailbox* mb, const int* ids, int n, const unsigned* step, int memop, unsigned long long* ts) {
    __shared__ unsigned st;
    if (threadIdx.x == 0) st = *step;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) { mb->ids[i] = ids[i]; mb->rowslot[i] = i; }
    if (threadIdx.x == 0) mb->n = n;
    __syncthreads();
    if (threadIdx.x == 0) {
        if (memop) { cuda::atomic_ref<unsigned, cuda::thread_scope_system> a(mb->ack); a.store(0u, cuda::memory_order_relaxed); }
        __threadfence_system();
        cuda::atomic_ref<unsigned, cuda::thread_scope_system> r(mb->req);
        r.store(st, cuda::memory_order_release);
        *ts = gtimer();
    }
}

// wait: spin on the ack with system-scope acquire loads; bounded (fail-closed: error flag, no hang)
__global__ void k_wait(Mailbox* mb, const unsigned* step, unsigned long long timeout_ns, int* err, unsigned long long* ts) {
    if (threadIdx.x == 0) {
        unsigned st = *step;
        cuda::atomic_ref<int, cuda::thread_scope_system> e(*err);
        cuda::atomic_ref<unsigned, cuda::thread_scope_system> a(mb->ack);
        unsigned long long t0 = gtimer();
        while (e.load(cuda::memory_order_relaxed) == 0 && a.load(cuda::memory_order_acquire) < st) {
            if (gtimer() - t0 > timeout_ns) { e.store(1, cuda::memory_order_release); break; }
            __nanosleep(100);
        }
        *ts = gtimer();
    }
}

__global__ void k_stamp(unsigned long long* ts) {
    if (threadIdx.x == 0) *ts = gtimer();
}

// zero-copy: read n rows of bf16 from mapped host memory, write to the device buffer (16-byte vectors)
__global__ void k_inject_zerocopy(const uint4* __restrict__ host_rows, uint4* __restrict__ dst, int n) {
    int per_row = ROW * 2 / 16;  // 640 uint4 per row
    int total = n * per_row;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x)
        dst[i] = __ldcg(host_rows + i);
}

__global__ void k_check(const __nv_bfloat16* d, int n, const unsigned* step, int layer, int* bad) {
    int r = blockIdx.x;
    if (r < n && threadIdx.x == 0) {
        float want = (float)((*step % 200) + layer);
        if (__bfloat162float(d[(size_t)r * ROW]) != want) atomicAdd(bad, 1);
    }
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void busy_us(double us) {
    if (us <= 0) return;
    double t0 = now_s();
    while ((now_s() - t0) * 1e6 < us) _mm_pause();
}

struct Args {
    int steps = 2000, rows = 8, host_us = 0, work_us = 0, timeout_ms = 2000, kill_host_at = -1;
    std::string wait = "spin", inject = "zerocopy";
};

int main(int argc, char** argv) {
    Args A;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (k == "--steps") A.steps = atoi(next().c_str());
        else if (k == "--rows") A.rows = atoi(next().c_str());
        else if (k == "--host-us") A.host_us = atoi(next().c_str());
        else if (k == "--work-us") A.work_us = atoi(next().c_str());
        else if (k == "--timeout-ms") A.timeout_ms = atoi(next().c_str());
        else if (k == "--kill-host-at") A.kill_host_at = atoi(next().c_str());
        else if (k == "--wait") A.wait = next();
        else if (k == "--inject") A.inject = next();
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    if (A.rows > MAXROWS) A.rows = MAXROWS;
    const bool memop = (A.wait == "memop");
    const bool zerocopy = (A.inject == "zerocopy");

    CK(cudaSetDevice(0));
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int can_map = 0, memops = 0;
    CK(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, 0));
    CKD(cuInit(0));
    CUdevice cdev;
    CKD(cuDeviceGet(&cdev, 0));
    CKD(cuDeviceGetAttribute(&memops, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1, cdev));
    printf("%s: mapped host memory %d, stream mem ops %d; L=%d rows=%d steps=%d wait=%s inject=%s host_us=%d work_us=%d\n",
           prop.name, can_map, memops, L, A.rows, A.steps, A.wait.c_str(), A.inject.c_str(), A.host_us, A.work_us);

    // mapped pinned memory: mailboxes and staging rows
    Mailbox* mb_h = nullptr;
    CK(cudaHostAlloc((void**)&mb_h, sizeof(Mailbox) * L, cudaHostAllocMapped));
    memset(mb_h, 0, sizeof(Mailbox) * L);
    Mailbox* mb_d = nullptr;
    CK(cudaHostGetDevicePointer((void**)&mb_d, mb_h, 0));
    const size_t stage_bytes = (size_t)MAXROWS * ROW * 2;
    __nv_bfloat16* stage_h[L];
    __nv_bfloat16* stage_d[L];
    for (int l = 0; l < L; ++l) {
        CK(cudaHostAlloc((void**)&stage_h[l], stage_bytes, cudaHostAllocMapped));
        memset(stage_h[l], 0, stage_bytes);
        CK(cudaHostGetDevicePointer((void**)&stage_d[l], stage_h[l], 0));
    }
    // device buffers
    int* ids_dev;
    CK(cudaMalloc(&ids_dev, sizeof(int) * MAXROWS));
    {
        std::vector<int> ids(MAXROWS);
        for (int i = 0; i < MAXROWS; ++i) ids[i] = (i << 16) | (i * 37 % 384);
        CK(cudaMemcpy(ids_dev, ids.data(), sizeof(int) * MAXROWS, cudaMemcpyHostToDevice));
    }
    __nv_bfloat16* dst[L];
    for (int l = 0; l < L; ++l) CK(cudaMalloc(&dst[l], stage_bytes));
    float* work;
    CK(cudaMalloc(&work, sizeof(float) * 256 * 1024));
    CK(cudaMemset(work, 0, sizeof(float) * 256 * 1024));
    int *err_h = nullptr, *err_d = nullptr, *bad_d;
    CK(cudaHostAlloc((void**)&err_h, 64, cudaHostAllocMapped));
    *err_h = 0;
    CK(cudaHostGetDevicePointer((void**)&err_d, err_h, 0));
    CK(cudaMalloc(&bad_d, sizeof(int)));
    CK(cudaMemset(bad_d, 0, sizeof(int)));
    unsigned long long* ts_d;  // [L][4]: publish, wait-done, inject-done, work-start
    CK(cudaMalloc(&ts_d, sizeof(unsigned long long) * L * 4));
    unsigned* step_d;          // the step id lives in device memory, bumped in-graph so the graph is replayable as is
    CK(cudaMalloc(&step_d, sizeof(unsigned)));
    CK(cudaMemset(step_d, 0, sizeof(unsigned)));

    // calibrate k_work iterations for --work-us
    int work_iters = 0;
    if (A.work_us > 0) {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        work_iters = 1000;
        for (int tries = 0; tries < 8; ++tries) {
            CK(cudaEventRecord(e0));
            for (int i = 0; i < 20; ++i) k_work<<<256, 1024>>>(work, work_iters);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            double us = ms * 1000.0 / 20;
            if (us > A.work_us * 0.95 && us < A.work_us * 1.05) break;
            work_iters = (int)(work_iters * (A.work_us / us));
            if (work_iters < 1) work_iters = 1;
        }
    }

    // ---- host thread: the CPU tier -----------------------------------------------------------------------------
    std::atomic<bool> stop{false};
    std::atomic<long> host_layers{0};
    std::thread host([&] {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(16, &set);   // a core on CCD 2, away from the main thread
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        for (unsigned step = 1; !stop.load(std::memory_order_relaxed); ++step) {
            if (A.kill_host_at >= 0 && (int)step >= A.kill_host_at) { while (!stop.load()) usleep(1000); return; }
            for (int l = 0; l < L; ++l) {
                Mailbox* mb = &mb_h[l];
                while (__atomic_load_n(&mb->req, __ATOMIC_ACQUIRE) < step) {
                    if (stop.load(std::memory_order_relaxed)) return;
                    _mm_pause();
                }
                unsigned n = mb->n;
                busy_us(A.host_us);
                // "results": row r's first element = bf16(step % 251 + l), rest copied pattern
                __nv_bfloat16 v = __float2bfloat16((float)((step % 200) + l));
                for (unsigned r = 0; r < n; ++r) stage_h[l][(size_t)r * ROW] = v;
                __atomic_store_n(&mb->ack, memop ? 1u : step, __ATOMIC_RELEASE);
                host_layers.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    // ---- the graph: one step --------------------------------------------------------------------------------------
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CUstream cs = (CUstream)s;
    // the step counter is read by the kernels from device memory; we bump it with a tiny kernel at the start
    cudaGraph_t graph;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    k_bump<<<1, 32, 0, s>>>(step_d);
    for (int l = 0; l < L; ++l) {
        k_stamp<<<1, 32, 0, s>>>(ts_d + l * 4 + 3);
        if (work_iters > 0) k_work<<<256, 1024, 0, s>>>(work, work_iters);
        k_publish<<<1, 128, 0, s>>>(mb_d + l, ids_dev, A.rows, step_d, memop ? 1 : 0, ts_d + l * 4 + 0);
        if (memop) {
            CUdeviceptr ack_ptr = (CUdeviceptr)(&mb_d[l].ack);
            CKD(cuStreamWaitValue32(cs, ack_ptr, 1u, CU_STREAM_WAIT_VALUE_GEQ));
            k_stamp<<<1, 32, 0, s>>>(ts_d + l * 4 + 1);
        } else {
            k_wait<<<1, 32, 0, s>>>(mb_d + l, step_d, (unsigned long long)A.timeout_ms * 1000000ull, err_d, ts_d + l * 4 + 1);
        }
        if (zerocopy) {
            k_inject_zerocopy<<<16, 256, 0, s>>>((const uint4*)stage_d[l], (uint4*)dst[l], A.rows);
        } else {
            CK(cudaMemcpyAsync(dst[l], stage_h[l], (size_t)A.rows * ROW * 2, cudaMemcpyHostToDevice, s));
        }
        k_stamp<<<1, 32, 0, s>>>(ts_d + l * 4 + 2);
        k_check<<<A.rows > 0 ? A.rows : 1, 32, 0, s>>>(dst[l], A.rows, step_d, l, bad_d);
    }
    CK(cudaStreamEndCapture(s, &graph));
    cudaGraphExec_t exec;
    CK(cudaGraphInstantiate(&exec, graph, 0));
    size_t n_nodes = 0;
    CK(cudaGraphGetNodes(graph, nullptr, &n_nodes));
    printf("graph captured: %zu nodes (%d layers)\n", n_nodes, L);

    // warm-up replays
    for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(exec, s));
    CK(cudaStreamSynchronize(s));
    if (__atomic_load_n(err_h, __ATOMIC_ACQUIRE)) { printf("timeout during warm-up (host thread dead?) - aborting\n"); stop = true; host.join(); return 1; }

    // measured replays
    std::vector<double> ts_pub_wait, ts_wait_inj, ts_work;
    double t0 = now_s();
    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    CK(cudaEventRecord(ev0, s));
    std::vector<unsigned long long> ts_h(L * 4);
    double sum_rt = 0, sum_inj = 0, max_rt = 0; long n_rt = 0;
    int launched = 0;
    double t_err = -1;
    for (int i = 0; i < A.steps; ++i) {
        if (__atomic_load_n(err_h, __ATOMIC_ACQUIRE)) { t_err = now_s() - t0; break; }   // fail closed: stop issuing steps
        CK(cudaGraphLaunch(exec, s));
        ++launched;
        if (i % 97 == 0) {   // sample the timestamps (a sync per sample, outside the steady-state measurement window)
            CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(ts_h.data(), ts_d, sizeof(unsigned long long) * L * 4, cudaMemcpyDeviceToHost));
            for (int l = 0; l < L; ++l) {
                double rt = (double)(ts_h[l * 4 + 1] - ts_h[l * 4 + 0]) * 1e-3;
                double inj = (double)(ts_h[l * 4 + 2] - ts_h[l * 4 + 1]) * 1e-3;
                sum_rt += rt; sum_inj += inj; n_rt++; if (rt > max_rt) max_rt = rt;
            }
        }
    }
    CK(cudaEventRecord(ev1, s));
    CK(cudaStreamSynchronize(s));
    double wall = now_s() - t0;
    float gpu_ms; CK(cudaEventElapsedTime(&gpu_ms, ev0, ev1));
    int bad_h = 0;
    int err_v = __atomic_load_n(err_h, __ATOMIC_ACQUIRE);
    CK(cudaMemcpy(&bad_h, bad_d, sizeof(int), cudaMemcpyDeviceToHost));
    stop = true;
    host.join();

    double per_step_us = gpu_ms * 1000.0 / (launched > 0 ? launched : 1);
    if (t_err >= 0) printf("error flag seen by the host after %.3f s (%d of %d steps launched); the in-flight graph drained in %.3f s\n", t_err, launched, A.steps, wall - t_err);
    printf("steps %d: %.1f us per step on the GPU (%.3f s wall), %.1f us per layer; sampled GPU-side round trip "
           "publish->wait-done mean %.1f us (max %.1f), wait-done->inject-done mean %.1f us; host layers served %ld; "
           "timeout flag %d; bad rows %d (%s)\n",
           launched, per_step_us, wall, per_step_us / L, sum_rt / n_rt, max_rt, sum_inj / n_rt, host_layers.load(),
           err_v, bad_h, (bad_h == 0 && err_v == 0) ? "OK" : (A.kill_host_at >= 0 ? "expected: host killed" : "FAIL"));
    return 0;
}
