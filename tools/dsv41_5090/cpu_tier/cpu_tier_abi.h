// cpu_tier_abi.h — the mailbox layout shared by the GPU kernels (publish / wait+inject) and the host dispatcher.
// Everything lives in pinned, device-mapped host memory; the GPU writes a request and bumps `step` last (system-scope
// release), the host answers by writing the rows into the staging area and bumping the ack's `step` (release); the
// GPU's wait kernel reads it with system-scope acquire loads.
#pragma once
#include <stdint.h>

#define CPU_TIER_HID 5120
#define CPU_TIER_SCALES 160      // ue8m0 bytes per token (5120 / 32)
#define CPU_TIER_MAX_PAIRS 96    // pairs per layer per step the mailbox can hold (16 tokens x top-6; vLLM pads decode batches to its capture sizes)
#define CPU_TIER_MAX_TOKENS 16

struct CpuTierPair {
    uint16_t token;   // index into the request's a_values / a_scales
    uint16_t expert;  // global expert id of the layer
    uint16_t row;     // route_output row (= token * topk + k) the result goes to
    uint16_t pad;
    float weight;     // router weight of the pair
};

struct CpuTierRequest {
    // header (written last: n_pairs / n_tokens before the fence, step after it)
    volatile uint32_t step;
    uint32_t n_pairs;
    uint32_t n_tokens;
    uint32_t flags;
    uint64_t t_publish_ns;  // GPU globaltimer at publish (for the KPIs)
    uint32_t n_all;         // all pairs of the step (hits and misses): their global expert ids below, -1 = padding
    uint32_t reserved;
    int16_t ids[CPU_TIER_MAX_PAIRS];
    CpuTierPair pairs[CPU_TIER_MAX_PAIRS];
    uint8_t a_values[CPU_TIER_MAX_TOKENS][CPU_TIER_HID];
    uint8_t a_scales[CPU_TIER_MAX_TOKENS][CPU_TIER_SCALES];
};

struct CpuTierAck {
    volatile uint32_t step;
    uint32_t status;        // 0 ok, else the host's error code for this step/layer
    uint64_t t_seen_ns;     // host CLOCK_MONOTONIC when the request was seen
    uint64_t t_done_ns;     // host CLOCK_MONOTONIC when the rows were complete
    uint32_t n_pairs;
    uint32_t n_experts;
};

// staging rows: uint16_t [CPU_TIER_MAX_PAIRS][CPU_TIER_HID] bf16 per layer, row i = the request's i-th pair

// the error word (one per rank, mapped): nonzero once a wait timed out; the host engine polls it, the GPU's later
// waits short-circuit on it so that an in-flight graph drains
#define CPU_TIER_ERR_TIMEOUT 1u
