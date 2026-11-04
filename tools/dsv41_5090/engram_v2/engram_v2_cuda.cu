// engram_v2_cuda.cu — the GPU side of Engram v2 as a torch extension: the publish kernel (this rank's hash-column
// row ids of one Engram layer into the layer's mapped mailbox, system-scope release of the sequence number) and
// the wait + copy kernel (spin on the host's ack with system-scope acquire loads and a bounded timeout, then copy
// the host's staging rows - fp8 values and ue8m0 scales - into device "landing" arrays that vLLM's own
// `_engram_lookup_kernel` dequantizes). Capturable: no host syncs, no allocations, fixed addresses.
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>

#include "engram_v2_abi.h"

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

// ids: [n_tokens, n_cols] int32 (strided), this rank's columns col_start .. col_start + n_local - 1
__global__ void publish_kernel(EngramV2Req* req, int* seq_dev, const int* ids, long stride_t, long stride_h, int col_start,
                               int n_local, int n_tokens) {
    const int n = n_tokens * n_local;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int t = i / n_local, h = i - t * n_local;
        req->ids[i] = ids[(long)t * stride_t + (long)(col_start + h) * stride_h];
    }
    __syncthreads();
    __threadfence_system();
    if (threadIdx.x == 0) {
        req->n_rows = (uint32_t)n;
        req->n_tokens = (uint32_t)n_tokens;
        req->local_heads = (uint32_t)n_local;
        req->flags = 0;
        req->t_publish_ns = gtimer();
        const int seq = seq_dev[0] + 1;
        seq_dev[0] = seq;
        __threadfence_system();
        st_release_sys(&req->step, (uint32_t)seq);
    }
}

// every CTA: thread 0 spins on the ack, then the CTA copies its share of the rows (values 256 B = 16 x uint4,
// scales 8 B = 1 x uint2 per row) from the host staging (zero-copy reads) into the device landing arrays
__global__ void wait_copy_kernel(const EngramV2Ack* ack, const EngramV2Req* req, const int* seq_dev, const uint8_t* st_values,
                                 const uint8_t* st_scales, uint8_t* land_values, uint8_t* land_scales, volatile uint32_t* err,
                                 long long timeout_ns) {
    __shared__ int s_n;
    const int tid = threadIdx.x;
    if (tid == 0) {
        const uint32_t step = (uint32_t)seq_dev[0];
        int n = 0;
        if (ld_acquire_sys(err) == 0) {
            const unsigned long long t0 = gtimer();
            bool ok = false;
            for (;;) {
                const uint32_t a = ld_acquire_sys(&ack->step);
                if (a == step) { ok = true; break; }
                if ((long long)(gtimer() - t0) > timeout_ns) break;
            }
            if (!ok) st_release_sys(err, ENGRAM_V2_ERR_TIMEOUT);
            else if (ack->status != 0) st_release_sys(err, ENGRAM_V2_ERR_STATUS + ack->status);
            else n = (int)req->n_rows;
        }
        s_n = n;
    }
    __syncthreads();
    __threadfence_system();
    const int n = s_n;
    // values: 16 uint4 per row; one thread per uint4
    const int total_v = n * 16;
    for (int i = blockIdx.x * blockDim.x + tid; i < total_v; i += gridDim.x * blockDim.x)
        reinterpret_cast<uint4*>(land_values)[i] = reinterpret_cast<const uint4*>(st_values)[i];
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        reinterpret_cast<uint2*>(land_scales)[i] = reinterpret_cast<const uint2*>(st_scales)[i];
}

}  // namespace

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

void publish(int64_t req_dev_ptr, torch::Tensor seq_dev, torch::Tensor ids, int64_t col_start, int64_t n_local, int64_t n_tokens,
             int64_t max_rows) {
    TORCH_CHECK(ids.scalar_type() == torch::kInt32 && ids.dim() == 2 && ids.is_cuda());
    TORCH_CHECK(seq_dev.scalar_type() == torch::kInt32 && seq_dev.is_cuda());
    TORCH_CHECK(n_tokens >= 0 && n_tokens <= ids.size(0) && col_start >= 0 && col_start + n_local <= ids.size(1));
    TORCH_CHECK(n_tokens * n_local <= max_rows, "engram v2: more rows than the mailbox holds");
    publish_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (EngramV2Req*)(uintptr_t)req_dev_ptr, seq_dev.data_ptr<int>(), ids.data_ptr<int>(), (long)ids.stride(0), (long)ids.stride(1),
        (int)col_start, (int)n_local, (int)n_tokens);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void wait_copy(int64_t ack_dev_ptr, int64_t req_dev_ptr, torch::Tensor seq_dev, int64_t st_values_dev, int64_t st_scales_dev,
               torch::Tensor land_values, torch::Tensor land_scales, int64_t err_dev_ptr, int64_t timeout_ns, int64_t n_rows) {
    TORCH_CHECK(land_values.scalar_type() == torch::kUInt8 && land_values.is_cuda() && land_values.is_contiguous());
    TORCH_CHECK(land_scales.scalar_type() == torch::kUInt8 && land_scales.is_cuda() && land_scales.is_contiguous());
    TORCH_CHECK(n_rows * ENGRAM_V2_ROW_VALUES <= land_values.numel() && n_rows * ENGRAM_V2_ROW_SCALES <= land_scales.numel());
    const int blocks = (int)std::min<int64_t>(64, std::max<int64_t>(1, (n_rows * 16 + 255) / 256));
    wait_copy_kernel<<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const EngramV2Ack*)(uintptr_t)ack_dev_ptr, (const EngramV2Req*)(uintptr_t)req_dev_ptr, seq_dev.data_ptr<int>(),
        (const uint8_t*)(uintptr_t)st_values_dev, (const uint8_t*)(uintptr_t)st_scales_dev, land_values.data_ptr<uint8_t>(),
        land_scales.data_ptr<uint8_t>(), (volatile uint32_t*)(uintptr_t)err_dev_ptr, (long long)timeout_ns);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

int64_t request_bytes(int64_t max_rows) { return engram_v2_request_bytes_for((int)max_rows); }
int64_t ack_bytes() { return (int64_t)sizeof(EngramV2Ack); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alloc_mapped", &alloc_mapped);
    m.def("free_mapped", &free_mapped);
    m.def("device_ptr", &device_ptr);
    m.def("publish", &publish);
    m.def("wait_copy", &wait_copy);
    m.def("request_bytes", &request_bytes);
    m.def("ack_bytes", &ack_bytes);
}
