# Canonical mapped-host W2

## Scope and tradeoff

This opt-in Runner V2 path places explicitly selected, complete MXFP4-derived
W2 layers in CUDA-mapped pinned host memory. The normal MoET cubins read their
stable UVA pointers over PCIe. It is targeted placement, not a generic CUDA
allocator, automatic `cudaMalloc` overflow, base-cache replay, or staging
cache.

The validated DeepSeek-V4-Flash shape maps target layers 40–42 and leaves
DSpark layers 43–45 GPU-resident. It sacrifices decode throughput to recover
substantially more GPU space for KV/context capacity on one 96 GiB card.

## Allocation and lifecycle

Before any complete output tensor is allocated, the MXFP4 builder calculates
the sizes of `planes13`, `sc13`, `planes2`, and `sc2`. One
`cudaHostAllocMapped | cudaHostAllocPortable` allocation owns the layer; the
four canonical CPU tensors are non-overlapping views of it. Quantization writes
directly into those views, and no redundant complete GPU W2 copy is retained.

The active CUDA device supplies its PCI address. Sysfs supplies the local NUMA
node and CPUs. Allocation temporarily binds CPU affinity and memory policy to
that node, restores both afterward, and verifies every page with
`move_pages`. Startup also verifies the mapped device pointer, UVA equality,
tensor bounds, and the production cubin dispatch. The allocation survives
normal full/piecewise CUDA graph capture. Construction failure rolls back the
partial layer; worker shutdown synchronizes before `cudaFreeHost`; `atexit`
provides a final cleanup guard.

Mapped selection is accepted only for the MXFP4 Runner V2 resident/delta path.
V1, base-cache replay, unsupported builders, pipeline parallelism, expert
parallelism/EPLB, and invalid topology guards fail closed. Mapping is disabled
by default, preserving existing V1 and TP behavior. Tensor-parallel mapped
placement has not been hardware-validated.

## Configuration

| Option | Default | Contract |
|---|---|---|
| `VLLM_MOE_W2_MAPPED_LAYERS` | empty | Comma-separated non-negative W2 layer keys. The validated selection is `40,41,42`. |
| `VLLM_MOE_W2_MAPPED_NUMA_NODE` | auto | Optional node guard; topology still comes from the active GPU. A mismatch fails. |
| `VLLM_MOE_W2_MAPPED_PCI` | auto | Optional PCI-BDF guard such as `0000:31:00.0`; it does not select a GPU. |
| `VLLM_MOE_W2_MAPPED_AUDIT_PATH` | unset | Optional atomically replaced JSON lifecycle/accounting audit. |

The validated recipe is
[`deepseek-v4-flash/pro6000x1-mapped-w2-dspark3`](../bench/recipes/deepseek-v4-flash/pro6000x1-mapped-w2-dspark3.yaml).
It uses Runner V2, DSpark-3, normal CUDA graphs, FP8 MLA KV, a fixed 512-slot
/ 6 GiB FP4 correction tier, `gpu_memory_utilization=0.988`, and
`VLLM_MOE_W2_DRAFT=1` with `VLLM_MOE_W2_NUM_LAYERS=46`. That pair selects all
three DSpark layers for GPU-resident W2 in this recipe. Kacper's existing
partial DSpark W2 selection remains supported; it is independent of whether a
selected main-model layer uses mapped host backing.

For DeepSeek-V4-Flash TP1, one mapped layer is 1,811,939,328 bytes (1.6875
GiB); three layers are 5,435,817,984 bytes (5.0625 GiB). These bytes consume
host RAM and pinned CUDA resources. Both mapped W2 and the MoET FP4 correction
tier are outside vLLM's `gpu_memory_utilization` accounting, so this recipe's
memory settings must not be generalized to another model or GPU.

## Evidence and limits

The implementation was validated natively on one RTX PRO 6000 with
DeepSeek-V4-Flash-0731:

| Area | Result |
|---|---|
| Allocation | Layers 40–42 mapped; 5,435,817,984 bytes on NUMA node 0; zero redundant complete GPU W2 bytes |
| Startup | Runner V2, DSpark-3, all normal CUDA graph captures, 512 FP4 slots / 6 GiB, 1,057 MiB physical VRAM free after capture |
| Bounded decode | Sealed 55.02 tok/s; clean reproduction 53.83 tok/s over exactly 1,024 generated tokens |
| Frozen quality | Baseline 92.66/100 reasoning and 30/30 tools at `reasoning_effort=high` |
| Production settings | 96.86/100 reasoning and 30/30 tools at `reasoning_effort=max`, `top_p=0.95` |
| Context accounting | 393,216-token configured admission; 625,757-token runtime-reported KV capacity |

The two context figures are allocation/configuration evidence, not exercised
context. The validation does not establish concurrency, soak, container,
other-GPU, other-checkpoint, tensor-parallel, arbitrary-workload, or production
deployment behavior. The frozen-suite methodology and remaining C5 gaps are
preserved in the [quality report](https://github.com/jpezzulli/vLLM-Moet/blob/agent/mapped-host-w2-public/docs/mapped-w2-quality-validation.md).

Focused unit coverage exercises parsing and compatibility guards, exact
allocation geometry, canonical views, duplicate prevention, UVA/topology
failures, accounting, construction rollback, synchronization, shutdown, and
mapped-layer configuration guards:

```bash
python -m pytest -q \
  tests/model_executor/layers/test_moe_w2_mapped_host.py \
  tests/model_executor/layers/test_moe_w2_persistence.py
```
