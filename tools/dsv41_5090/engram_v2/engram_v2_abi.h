// engram_v2_abi.h — the mailbox shared by the GPU (publish / wait kernels of engram_v2_cuda.cu) and the host row
// fetcher (engram_v2.cpp) of one Engram layer of one TP rank. Everything lives in pinned, device-mapped host memory:
// the GPU writes the request's row ids and bumps `step` last (system-scope release); the host fetches the rows
// (DRAM row cache, else NVMe) into the layer's staging arrays and bumps the ack's `step` (release); the GPU's wait
// kernel reads it with system-scope acquire loads, then copies the staging rows to device memory for vLLM's own
// lookup kernel.
#pragma once
#include <stdint.h>

#define ENGRAM_V2_ROW_VALUES 256   // fp8 e4m3 bytes per row
#define ENGRAM_V2_ROW_SCALES 8     // ue8m0 bytes per row (one per 32 values)
#define ENGRAM_V2_ROW_BYTES 264

struct EngramV2Req {
    volatile uint32_t step;   // sequence number of this request (per layer), written last
    uint32_t n_rows;          // tokens x local heads
    uint32_t n_tokens;
    uint32_t local_heads;
    uint64_t t_publish_ns;    // GPU globaltimer at publish (informational)
    uint32_t flags;
    uint32_t reserved;
    int32_t ids[1];           // n_rows global row ids of this layer's table (-1 = no row: zeros); sized by the allocator
};

struct EngramV2Ack {
    volatile uint32_t step;
    uint32_t status;          // 0 ok; 1 = an NVMe read failed / short; 2 = malformed request
    uint64_t t_seen_ns;       // host CLOCK_MONOTONIC when the request was seen
    uint64_t t_done_ns;       // host CLOCK_MONOTONIC when the staging rows were complete
    uint32_t n_hits;          // rows served from the DRAM row cache
    uint32_t n_misses;        // distinct rows read from NVMe
    uint32_t n_reads;         // NVMe reads issued for this request
    uint32_t n_rows;
};

// staging per layer: uint8_t values[max_rows][256] + uint8_t scales[max_rows][8], row i = the request's i-th id

#define ENGRAM_V2_ERR_TIMEOUT 1u   // the error word (one per rank, mapped): the wait kernel timed out
#define ENGRAM_V2_ERR_STATUS 16u   // 16 + the host's status

static inline int64_t engram_v2_request_bytes_for(int max_rows) {
    return (int64_t)(sizeof(EngramV2Req) - sizeof(int32_t)) + (int64_t)max_rows * (int64_t)sizeof(int32_t);
}
