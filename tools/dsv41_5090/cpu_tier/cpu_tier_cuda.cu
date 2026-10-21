// cpu_tier_cuda.cu — the GPU side of the CPU expert tier as a torch extension: the step counter bumped in-graph, the
// publish kernel (split the routing against the resident slot table, hand the missing pairs and the quantized tokens
// to the host through a mapped mailbox, system-scope release of the step id), and the wait + inject kernel (spin on
// the host's ack with system-scope acquire loads and a bounded timeout, then copy the host's bf16 rows into
// route_output; on a timeout arm the error word so that the engine fails the step closed and later waits short-
// circuit). Everything is capturable: no host syncs, no allocations, the stream's order only.
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>

#include "cpu_tier_abi.h"

namespace {

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const volatile uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release_sys(volatile uint32_t* p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__global__ void step_begin_kernel(int* step_dev) {
    if (threadIdx.x == 0) step_dev[0] += 1;
}

// one CTA: remap, compact the misses (thread 0, pair order), copy the tokens, release the step
__global__ void publish_kernel(CpuTierRequest* req, const int* step_dev, const int* ids, const float* wts,
                               const int* slot, int* remapped, int n_pairs, int topk, const uint8_t* a_values,
                               const uint8_t* a_scales, int n_tokens) {
    __shared__ int s_n;
    const int tid = threadIdx.x;
    for (int p = tid; p < n_pairs; p += blockDim.x) {
        const int e = ids[p];
        remapped[p] = e >= 0 ? slot[e] : -1;
        req->ids[p] = (int16_t)e;  // the use log for the placement manager (hits and misses)
    }
    if (tid == 0) {
        int n = 0;
        for (int p = 0; p < n_pairs; ++p) {
            const int e = ids[p];
            if (e >= 0 && slot[e] < 0 && n < CPU_TIER_MAX_PAIRS) {
                CpuTierPair pr;
                pr.token = (uint16_t)(p / topk);
                pr.expert = (uint16_t)e;
                pr.row = (uint16_t)p;
                pr.pad = 0;
                pr.weight = wts[p];
                req->pairs[n++] = pr;
            }
        }
        s_n = n;
    }
    // the tokens' quantized activations (16-B vector stores into host memory)
    {
        const uint4* src = reinterpret_cast<const uint4*>(a_values);
        uint4* dst = reinterpret_cast<uint4*>(&req->a_values[0][0]);
        const int nv = n_tokens * (CPU_TIER_HID / 16);
        for (int i = tid; i < nv; i += blockDim.x) dst[i] = src[i];
        const uint4* ssrc = reinterpret_cast<const uint4*>(a_scales);
        uint4* sdst = reinterpret_cast<uint4*>(&req->a_scales[0][0]);
        const int ns = n_tokens * (CPU_TIER_SCALES / 16);
        for (int i = tid; i < ns; i += blockDim.x) sdst[i] = ssrc[i];
    }
    __syncthreads();
    __threadfence_system();
    if (tid == 0) {
        req->n_pairs = (uint32_t)s_n;
        req->n_tokens = (uint32_t)n_tokens;
        req->n_all = (uint32_t)n_pairs;
        req->flags = 0;
        req->t_publish_ns = gtimer();
        __threadfence_system();
        st_release_sys(&req->step, (uint32_t)step_dev[0]);
    }
}

// every CTA spins on the ack (thread 0), then copies its share of the rows from the staging area
__global__ void wait_inject_kernel(const CpuTierAck* ack, const CpuTierRequest* req, const uint16_t* staging,
                                   const int* step_dev, uint16_t* route, int route_stride, volatile uint32_t* err,
                                   long long timeout_ns) {
    __shared__ int s_n;
    const int tid = threadIdx.x;
    if (tid == 0) {
        const uint32_t step = (uint32_t)step_dev[0];
        int n = 0;
        if (ld_acquire_sys(err) == 0) {
            const unsigned long long t0 = gtimer();
            bool ok = false;
            for (;;) {
                const uint32_t a = ld_acquire_sys(&ack->step);
                if (a == step) { ok = true; break; }
                if ((long long)(gtimer() - t0) > timeout_ns) break;
            }
            if (!ok) {
                st_release_sys(err, CPU_TIER_ERR_TIMEOUT);
            } else if (ack->status != 0) {
                st_release_sys(err, 16u + ack->status);
            } else {
                n = (int)req->n_pairs;
            }
        }
        s_n = n;
    }
    __syncthreads();
    __threadfence_system();
    const int n = s_n;
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        const int row = req->pairs[i].row;
        const uint4* src = reinterpret_cast<const uint4*>(staging + (size_t)i * CPU_TIER_HID);
        uint4* dst = reinterpret_cast<uint4*>(route + (size_t)row * route_stride);
        for (int j = tid; j < CPU_TIER_HID / 8; j += blockDim.x) dst[j] = src[j];
    }
}

// ---- promotion: a checkpoint-layout record -> b12x's prepared layout at pool slot s (the permutations measured by
// b12x_layout_probe.py). Record: w1n [576][2560] | w1s [576][160] | w3n | w3s | w2n [5120][288] | w2s [5120][18].
// The w13 input b12x saw is [w1; w3] nibbles (gate rows then up rows), its scales [w1s; w3s] bytes.
constexpr int64_t REC_W1N = 0, REC_W1S = 1474560, REC_W3N = 1566720, REC_W3S = 3041280, REC_W2N = 3133440, REC_W2S = 4608000;
constexpr int64_t W13_NIB = 2 * 2949120, W13S_B = 2 * 92160, W2_NIB = 2949120, W2S_B = 92160;

__device__ __forceinline__ uint8_t rec_nibble_w13(const uint8_t* rec, int64_t i) {  // input nibble i of [w1n; w3n]
    const int64_t byte = i >> 1;
    const uint8_t b = byte < 1474560 ? rec[REC_W1N + byte] : rec[REC_W3N + byte - 1474560];
    return (i & 1) ? (b >> 4) : (b & 0xF);
}
__device__ __forceinline__ uint8_t rec_nibble_w2(const uint8_t* rec, int64_t i) {
    const uint8_t b = rec[REC_W2N + (i >> 1)];
    return (i & 1) ? (b >> 4) : (b & 0xF);
}
__device__ __forceinline__ uint8_t rec_scale_w13(const uint8_t* rec, int64_t i) {
    return i < 92160 ? rec[REC_W1S + i] : rec[REC_W3S + i - 92160];
}

__global__ void repack_kernel(const uint8_t* rec, const int32_t* idx_w13, const int32_t* idx_w13s, const int32_t* idx_w2,
                              const int32_t* idx_w2s, uint8_t* out_w13, uint8_t* out_w13s, uint8_t* out_w2, uint8_t* out_w2s) {
    const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    // output bytes: w13 2,949,120; w2 1,474,560; w13s 184,320; w2s 92,160 — one grid, ranges in that order
    if (j < W13_NIB / 2) {
        out_w13[j] = rec_nibble_w13(rec, idx_w13[2 * j]) | (rec_nibble_w13(rec, idx_w13[2 * j + 1]) << 4);
    } else if (j < W13_NIB / 2 + W2_NIB / 2) {
        const int64_t k = j - W13_NIB / 2;
        out_w2[k] = rec_nibble_w2(rec, idx_w2[2 * k]) | (rec_nibble_w2(rec, idx_w2[2 * k + 1]) << 4);
    } else if (j < W13_NIB / 2 + W2_NIB / 2 + W13S_B) {
        const int64_t k = j - W13_NIB / 2 - W2_NIB / 2;
        out_w13s[k] = rec_scale_w13(rec, idx_w13s[k]);
    } else if (j < W13_NIB / 2 + W2_NIB / 2 + W13S_B + W2S_B) {
        const int64_t k = j - W13_NIB / 2 - W2_NIB / 2 - W13S_B;
        out_w2s[k] = rec[REC_W2S + idx_w2s[k]];
    }
}

}  // namespace

// record (device uint8 [4,700,160]) -> slot `slot` of the pool tensors ([R, words] int32 each, b12x's prepared layout)
void repack_record(torch::Tensor record, torch::Tensor idx_w13, torch::Tensor idx_w13s, torch::Tensor idx_w2,
                   torch::Tensor idx_w2s, torch::Tensor pool_w13, torch::Tensor pool_w13s, torch::Tensor pool_w2,
                   torch::Tensor pool_w2s, int64_t slot) {
    TORCH_CHECK(record.scalar_type() == torch::kUInt8 && record.numel() == 4700160 && record.is_cuda());
    TORCH_CHECK(idx_w13.numel() == W13_NIB && idx_w2.numel() == W2_NIB && idx_w13s.numel() == W13S_B && idx_w2s.numel() == W2S_B);
    TORCH_CHECK(pool_w13.size(1) * (int64_t)pool_w13.element_size() == W13_NIB / 2 && pool_w2.size(1) * (int64_t)pool_w2.element_size() == W2_NIB / 2);
    TORCH_CHECK(pool_w13s.size(1) * (int64_t)pool_w13s.element_size() == W13S_B && pool_w2s.size(1) * (int64_t)pool_w2s.element_size() == W2S_B);
    TORCH_CHECK(slot >= 0 && slot < pool_w13.size(0));
    const int64_t total = W13_NIB / 2 + W2_NIB / 2 + W13S_B + W2S_B;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    repack_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        record.data_ptr<uint8_t>(), idx_w13.data_ptr<int32_t>(), idx_w13s.data_ptr<int32_t>(), idx_w2.data_ptr<int32_t>(),
        idx_w2s.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(pool_w13.data_ptr()) + slot * (W13_NIB / 2),
        reinterpret_cast<uint8_t*>(pool_w13s.data_ptr()) + slot * W13S_B, reinterpret_cast<uint8_t*>(pool_w2.data_ptr()) + slot * (W2_NIB / 2),
        reinterpret_cast<uint8_t*>(pool_w2s.data_ptr()) + slot * W2S_B);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

// pinned (page-locked, portable) host memory for the expert store: H2D promotions DMA from it; not zeroed
int64_t alloc_pinned(int64_t nbytes) {
    void* p = nullptr;
    TORCH_CHECK(cudaHostAlloc(&p, (size_t)nbytes, cudaHostAllocPortable) == cudaSuccess, "cudaHostAlloc (pinned) failed");
    return (int64_t)(uintptr_t)p;
}

// a CPU uint8 tensor view of host memory (the store's records as copy sources)
torch::Tensor host_view(int64_t ptr, int64_t nbytes) {
    return torch::from_blob((void*)(uintptr_t)ptr, {nbytes}, torch::TensorOptions().dtype(torch::kUInt8));
}

int64_t alloc_mapped(int64_t nbytes) {
    void* p = nullptr;
    TORCH_CHECK(cudaHostAlloc(&p, (size_t)nbytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess, "cudaHostAlloc failed");
    memset(p, 0, (size_t)nbytes);
    return (int64_t)(uintptr_t)p;
}

void free_mapped(int64_t p) { cudaFreeHost((void*)(uintptr_t)p); }

int64_t device_ptr(int64_t host_ptr) {
    void* d = nullptr;
    TORCH_CHECK(cudaHostGetDevicePointer(&d, (void*)(uintptr_t)host_ptr, 0) == cudaSuccess, "cudaHostGetDevicePointer failed");
    return (int64_t)(uintptr_t)d;
}

void step_begin(torch::Tensor step_dev) {
    step_begin_kernel<<<1, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(step_dev.data_ptr<int>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void publish(int64_t req_dev_ptr, torch::Tensor step_dev, torch::Tensor topk_ids, torch::Tensor topk_weights,
             torch::Tensor slot_table, torch::Tensor remapped, torch::Tensor a_values, torch::Tensor a_scales,
             int64_t n_tokens) {
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 && topk_weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(slot_table.scalar_type() == torch::kInt32 && remapped.scalar_type() == torch::kInt32);
    TORCH_CHECK(topk_ids.is_contiguous() && topk_weights.is_contiguous() && remapped.is_contiguous());
    TORCH_CHECK(a_values.scalar_type() == torch::kUInt8 && a_scales.scalar_type() == torch::kUInt8);
    TORCH_CHECK(n_tokens >= 0 && n_tokens <= CPU_TIER_MAX_TOKENS, "at most CPU_TIER_MAX_TOKENS tokens per step on the CPU tier");
    const int n_pairs = (int)topk_ids.numel();
    TORCH_CHECK(n_pairs <= CPU_TIER_MAX_PAIRS && topk_ids.dim() == 2);
    publish_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (CpuTierRequest*)(uintptr_t)req_dev_ptr, step_dev.data_ptr<int>(), topk_ids.data_ptr<int>(),
        topk_weights.data_ptr<float>(), slot_table.data_ptr<int>(), remapped.data_ptr<int>(), n_pairs,
        (int)topk_ids.size(1), a_values.data_ptr<uint8_t>(), a_scales.data_ptr<uint8_t>(), (int)n_tokens);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void wait_inject(int64_t ack_dev_ptr, int64_t req_dev_ptr, int64_t staging_dev_ptr, torch::Tensor step_dev,
                 torch::Tensor route_output, int64_t err_dev_ptr, int64_t timeout_ns) {
    TORCH_CHECK(route_output.scalar_type() == torch::kBFloat16 && route_output.dim() == 2 &&
                route_output.size(1) == CPU_TIER_HID && route_output.stride(1) == 1);
    wait_inject_kernel<<<8, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const CpuTierAck*)(uintptr_t)ack_dev_ptr, (const CpuTierRequest*)(uintptr_t)req_dev_ptr,
        (const uint16_t*)(uintptr_t)staging_dev_ptr, step_dev.data_ptr<int>(),
        reinterpret_cast<uint16_t*>(route_output.data_ptr()), (int)route_output.stride(0),
        (volatile uint32_t*)(uintptr_t)err_dev_ptr, timeout_ns);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

int64_t request_bytes() { return (int64_t)sizeof(CpuTierRequest); }
int64_t ack_bytes() { return (int64_t)sizeof(CpuTierAck); }
int64_t staging_bytes() { return (int64_t)CPU_TIER_MAX_PAIRS * CPU_TIER_HID * 2; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alloc_mapped", &alloc_mapped);
    m.def("alloc_pinned", &alloc_pinned);
    m.def("host_view", &host_view);
    m.def("free_mapped", &free_mapped);
    m.def("device_ptr", &device_ptr);
    m.def("step_begin", &step_begin);
    m.def("publish", &publish);
    m.def("wait_inject", &wait_inject);
    m.def("repack_record", &repack_record);
    m.def("request_bytes", &request_bytes);
    m.def("ack_bytes", &ack_bytes);
    m.def("staging_bytes", &staging_bytes);
}
