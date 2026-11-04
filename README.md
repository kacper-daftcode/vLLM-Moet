# vLLM-Moet

Frontier Mixture-of-Experts models on RTX PRO 6000 and RTX 5090 (sm_120), where the official vLLM
images do not run them or the official checkpoints do not fit. The changes are kernel-side patches
on official vLLM images, bit-identical to the paths they replace unless a doc says otherwise, each
behind a runtime switch.

| line | what | status | doc |
|---|---|---|---|
| **DeepSeek-V4.1-Flash on 4× RTX PRO 6000** | the official checkpoint on vLLM nightly: the sm_120 gaps closed, the decode step and the prefill rebuilt around small tensor-core kernels | **active, in production** | [docs/dsv41-sm120.md](docs/dsv41-sm120.md) |
| **DeepSeek-V4.1-Flash on 4× RTX 5090 (MoET hybrid)** | 110 of the 384 experts per layer resident on the cards, every other routed expert computed on the CPU from the checkpoint's own bytes, bit-identical to the all-resident path; 145 / 125 tok/s on one stream, 4.6K tok/s prefill, 512K context, 512 GB of host RAM | serving on one reference box | [docs/moet-hybrid.md](docs/moet-hybrid.md), [the measurements and the field](docs/benchmarks/moet-hybrid-4x5090-2026-11-04.md) |
| **2-bit experts on vLLM v0.24.0** | GLM-5.2 (753B), DeepSeek-V4-Flash (159B) and Kimi-K2.7-Code (1T) on one to four cards that cannot hold their checkpoints: a 2-bit codebook on hand-written SASS, an FP4 delta cache | frozen (builds, fixes accepted, no development) | [docs/v024-2bit.md](docs/v024-2bit.md) |
| **Qwen3.8-Flash-Next-FP8 on 4× RTX PRO 6000** | the official nightly with the PLE table on the GPUs, the skinny GEMM and an FP8 MoE GEMV | as built | [docs/qwen38-sm120.md](docs/qwen38-sm120.md) |

## Results, as served

DeepSeek-V4.1-Flash, 4× RTX PRO 6000 at 400 W, image `vllm-moet-sm120:dsv41-nightly-df3bd094-r23`
(TP4, 512K context, DSpark k = 7, vision, thinking, tool calling, KV offload to host RAM and disk):

| | vLLM-Moet | official vLLM image on the same cards |
|---|---|---|
| decode, 1 stream | prose ~293 tok/s, code ~802 tok/s (129 steps/s); Fraser bench 688 tok/s median, 746 best over 40 reps | 60 steps/s with only the fixes it needs to start |
| decode, 8 streams | prose ~721, code ~2,258 tok/s aggregate | — |
| fresh prefill | 163K tokens in 7.2 s (22.7K tok/s), 19K in 0.86 s; BRRR aggregate +11…+28 % over the recipe's stock vLLM at 2K–65K | 163K in ~15.5 s |
| KV cache | 9.62M tokens on the GPUs (18× concurrency at 512K) + 64 GiB host RAM + a disk tier | 1.49M (official KV formats) |
| quality | GSM8K-200 194/200, needle PASS at 29K / 106K / 400K, tool calls / JSON / vision pass; identical requests return identical logits at any batch size and prompt length | the same checkpoint's quality |

DeepSeek-V4.1-Flash, 4× RTX 5090 at 400 W + 512 GB DDR5, the MoET hybrid (TP4, 512K context, thinking, tool calling;
110 of the 384 experts per layer on the cards, the rest computed on the CPU, bit-identical to the all-resident path;
2026-11-03/04, [the record](docs/benchmarks/moet-hybrid-4x5090-2026-11-04.md)):

| | 4× RTX 5090 + CPU tier | the field, as published by the authors |
|---|---|---|
| decode, 1 stream | **145 / 125 tok/s** on short prompts, 84 on mixed prompts over 30 min, 70–87 behind 2K–512K of context; **94 tok/s on Fraser Price's BRRR code bench** (2K / 16K / 64K alike) | native 4× RTX PRO 6000 (this line, Fraser's recipe): 700–770 tok/s with a 7-token draft; CPU-expert hybrids (SGLang + KTransformers on 2× 5070 Ti + 2× EPYC / 512 GB; 0xSero's EXL3 tier on 1× 3090 + 503 GiB): 19–50 tok/s; disk-streaming ports on 1× 5090 + 128 GB: 5 (native precision) to 28 (2-bit experts) |
| decode, several streams | 2 streams 107 tok/s, 4 streams 123 aggregate (10-min soaks); BRRR 155 / 166 at 4 / 8 streams; 60 min × 4 streams 2,197 requests, 0 errors | native 4× PRO 6000: 1,960 at 8 streams |
| prefill | **4.6–4.7K tok/s from 8K to 512K**; TTFT 1.8 s at 8K, 14 s at 64K, 28 s at 128K, ~113 s at 512K | native: 10–18K tok/s; the CPU-expert hybrids: 0.7–1.5K |
| context, quality | 524,288 tokens served; needle 3/3 at 64K / 256K / ~512K; **HumanEval 160/164 vs the native 160/164, MATH-500 L5 55/60 vs 55/60** (paired, McNemar p = 1.0) | the native checkpoint's quality; the EXL3 / IQ2 ports quantize the experts |
| footprint, power | ~29 GB per card; ~457 GiB of host RAM pinned (the experts' store 241 GiB + the Engram tables 189 GiB); 1.16 kW → 13.8 J per token (8.3 at 4 streams). **Less RAM** (branch `moet-hybrid-next`, measured): the Engram tables on NVMe → 278–307 GiB at the same speed (a 384 GiB host); + the cold experts on NVMe → 236 GiB at 138 / 120 tok/s and 1.4–1.5× the TTFT (256 GiB), 112 GiB at 112 / 83 tok/s and 3.3× (128 GiB); **fewer cards** (the same branch): 2× RTX 5090 46 / 47 tok/s, 1× RTX 5090 40 / 34, each also with the RAM tiers (2× 5090 + 256 GiB: 40 / 43; 1× 5090 + 128 GiB: 38 / 26) — bit-identical at every cell | the CPU-expert hybrids run in 256–512 GB of DDR4; the disk-streaming ports in 128 GB |

Measurement records: [docs/benchmarks/](docs/benchmarks/). The hybrid's numbers are in
[docs/moet-hybrid.md](docs/moet-hybrid.md), the 2-bit line's tables in [docs/v024-2bit.md](docs/v024-2bit.md).

## Quick start: DeepSeek-V4.1-Flash

Four RTX PRO 6000 (an RTX 5090 does not fit the model at TP4), Docker with the NVIDIA container
toolkit, PCIe peer-to-peer, ~330 GB of host shared memory (less: `KV_OFFLOAD_GIB=0`).

```bash
git clone https://github.com/kacper-daftcode/vLLM-Moet && cd vLLM-Moet
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir /srv/models/DeepSeek-V4.1-Flash-1017
export IMAGE=vllm-moet-sm120:dsv41-nightly-df3bd094-r23 \
  MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017 CACHE_DIR=/srv/cache/ds41
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly -t $IMAGE .     # ~11 min

GPU_MEM_UTIL=0.92 docker/sm120/run-dsv41.sh      # first start (~22 min: FlashInfer autotunes into CACHE_DIR)
docker stop -t 60 ds41-flash
GPU_MEM_UTIL=0.94 KV_OFFLOAD_GIB=64 KV_OFFLOAD_FS_DIR=/srv/kvcache KV_OFFLOAD_PROMPT_ONLY=0 \
  INDEXER_TP_SHARD_MIN_ROWS=2048 SERVED=1 docker/sm120/run-dsv41.sh            # the served configuration
```

The OpenAI-compatible API listens on `127.0.0.1:8001`. `SERVED=1` turns on the switches production
runs with; every knob and switch is listed in [docs/dsv41-sm120.md](docs/dsv41-sm120.md).

## Kernels and tools

- [`blackwell-isa`](https://github.com/kacper-daftcode/blackwell-isa) (a machine-readable SM120 SASS
  ISA database, [searchable reference](https://kacper-daftcode.github.io/blackwell-isa/SM120_ISA_REFERENCE.html))
  and [`cubit`](https://github.com/kacper-daftcode/cubit) (an SM120 assembler / disassembler built on
  it) produce the 2-bit line's SASS kernels in [`kernels/`](kernels/MANIFEST.md); every kernel the
  DeepSeek image ships round-trips through cubit bit for bit (`tools/dsv41_sm120/cubit_gate.py`).
- The DeepSeek-V4.1 kernels are CUDA C++ compiled into the image, each with an op-level test:
  [`tools/dsv41_sm120/`](tools/dsv41_sm120/README.md). Qwen3.8's MoE GEMV: `tools/qwen38_sm120/moe_gemv/`.
- Host-side benchmark and profiling scripts: [`tools/sm120_perf/`](tools/sm120_perf/README.md).
- SM120 kernel-design notes from the hardware counters: [docs/sm120-alu-subunits.md](docs/sm120-alu-subunits.md).

## Repository layout

| path | what |
|---|---|
| `Dockerfile.sm120-dsv41-nightly`, `tools/dsv41_sm120/`, `docker/sm120/run-dsv41.sh` | DeepSeek-V4.1-Flash: the image, its patchers, kernels and tests, the launcher (+ the cron scripts in `docker/sm120/`) |
| `Dockerfile.sm120-qwen38`, `tools/qwen38_sm120/`, `docker/sm120/run-qwen38.sh` | Qwen3.8-Flash-Next-FP8 |
| `patch/`, `Dockerfile.sm120-v024`, `Dockerfile.recipes`, `docker/serve_recipe.py`, `bench/`, `kernels/`, `tools/*.py` | the 2-bit line: the generated v0.24.0 patch (never edited by hand), its images, the recipes and the benchmark runner, SASS sources and cubins, its tests |
| `tools/sm120_perf/` | benchmark and profiling scripts |
| `docs/` | one doc per line, the benchmark records, the kernel-design notes |

## Contributing and license

[CONTRIBUTING.md](CONTRIBUTING.md) says where a change goes and how it is validated;
[CHANGELOG.md](CHANGELOG.md) lists the served images. Apache License 2.0 ([LICENSE](LICENSE));
third-party components in [NOTICE](NOTICE).
