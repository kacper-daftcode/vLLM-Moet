# vLLM-Moet

Frontier Mixture-of-Experts models on RTX PRO 6000 and RTX 5090 (sm_120), where the official vLLM
images do not run them or the official checkpoints do not fit. The changes are kernel-side patches
on official vLLM images, bit-identical to the paths they replace unless a doc says otherwise, each
behind a runtime switch.

| line | what | status | doc |
|---|---|---|---|
| **DeepSeek-V4.1-Flash on 4× RTX PRO 6000** | the official checkpoint on vLLM nightly: the sm_120 gaps closed, the decode step and the prefill rebuilt around small tensor-core kernels | **active, serving** | [docs/dsv41-sm120.md](docs/dsv41-sm120.md) |
| **DeepSeek-V4.1-Flash on 1–4× RTX 5090 or 1–2× RTX PRO 6000 (the MoET hybrid)** | a subset of every layer's 384 experts on the cards, every other routed expert computed on the CPU in-step from the checkpoint's own bytes — bit-identical to the all-resident path; the Engram tables and the cold experts optionally on NVMe for 128–384 GB hosts | serving; every configuration below measured | [docs/moet-hybrid.md](docs/moet-hybrid.md), [the record and the field](docs/benchmarks/moet-hybrid-4x5090-2026-11-04.md) |
| **2-bit experts on vLLM v0.24.0** | GLM-5.2 (753B), DeepSeek-V4-Flash (159B) and Kimi-K2.7-Code (1T) on one to four cards that cannot hold their checkpoints: a 2-bit codebook on hand-written SASS, an FP4 delta cache | frozen (builds, fixes accepted, no development) | [docs/v024-2bit.md](docs/v024-2bit.md) |
| **Qwen3.8-Flash-Next-FP8 on 4× RTX PRO 6000** | the official nightly with the PLE table on the GPUs, the skinny GEMM and an FP8 MoE GEMV | as built | [docs/qwen38-sm120.md](docs/qwen38-sm120.md) |
| **Kimi-K3 + DSpark on 8× B300 (HGX, sm_103)** | the official v0.32.0 image with what DSpark needs on a hybrid KDA / MLA model: RecoverSSM admitted next to KV offload (the pool back from 19.3M to 21.6M tokens), the RedHat sliding-window draft under DCP8 with prefix caching | superseded by K3.1 on the same host (2026-11-04); rollback kept | [docs/k3-b300-dspark-kv.md](docs/k3-b300-dspark-kv.md) |
| **Kimi-K3.1-Pro + DSpark on 8× B300 (HGX, sm_103)** | the day-0 `kimi-k31` image with its specialized decode path unlocked on a 148-SM B300 (the persistent experts grid was sized for GB300's 152 SMs): single-stream 252 vs 143 tok/s, 8K C=32 61.5 vs 48.4 tok/s per session at the same acceptance and GSM8K | **active, in production** | [docs/benchmarks/k31-b300-2026-11-04.md](docs/benchmarks/k31-b300-2026-11-04.md) |
| **Ripple-65B Preview (NVFP4) on 4× RTX PRO 6000** | Ricursive's 65B dense model with its DFlash2 drafter on vLLM main: the TP all-reduces on the PCIe P2P push, the draft length by batch size, the drafter at the schedule's k, two microbatches at 16+ requests; code ~630 tok/s single stream, ~3050 aggregate at 8 requests, ~5100 at 32 on the NVFP4 checkpoint, ~740 / ~3350 / ~5800 with an optional derived checkpoint (attention projections in MXFP8) | serving | [docs/ripple-sm120.md](docs/ripple-sm120.md) |

## Results, as served

### 4× RTX PRO 6000: DeepSeek-V4.1-Flash native

DeepSeek-V4.1-Flash, 4× RTX PRO 6000 at 400 W, image `vllm-moet-sm120:dsv41-nightly-df3bd094-r25`
(TP4, 512K context, DSpark k = 7, vision, thinking, tool calling, KV offload to host RAM and disk):

| | vLLM-Moet | official vLLM image on the same cards |
|---|---|---|
| decode, 1 stream | prose ~293 tok/s, code ~802 tok/s (129 steps/s); Fraser bench 688 tok/s median, 746 best over 40 reps | 60 steps/s with only the fixes it needs to start |
| decode, 8 streams | prose ~721, code ~2,258 tok/s aggregate | — |
| fresh prefill | 163K tokens in 6.6 s (24.5K tok/s), 19K in 0.79 s (`-r25`: the sparse-MLA prefill attention on the cubit-assembled SASS kernel, bit-identical to `-r24`'s 6.8 s and `-r23`'s 7.1 s); BRRR aggregate +11…+28 % over the recipe's stock vLLM at 2K–65K | 163K in ~15.5 s |
| KV cache | 9.62M tokens on the GPUs (18× concurrency at 512K) + 64 GiB host RAM + a disk tier | 1.49M (official KV formats) |
| quality | GSM8K-200 194/200, needle PASS at 29K / 106K / 400K, tool calls / JSON / vision pass; identical requests return identical logits at any batch size and prompt length | the same checkpoint's quality |

### 4× RTX PRO 6000: Ripple-65B Preview

Ripple-65B Preview with the DFlash2 drafter, 4× RTX PRO 6000 at 400 W, image `vllm-moet-sm120:ripple-nightly-a2bcb565-r6`
(TP4, 512K context, draft length 15 / 11 / 7 by batch size, two microbatches at 16+ requests); the first number is the
derived checkpoint with the attention projections in MXFP8, the second the NVFP4 checkpoint ([the doc](docs/ripple-sm120.md)):

| | vLLM-Moet | stock vLLM (`03bd3c68`) on the same cards, static k = 7 |
|---|---|---|
| decode, 1 stream | code ~740 / 630 tok/s at 71 / 64 steps/s; prose ~195 / 170 | code 416 (~600 with k = 15, a stock option), prose 169 at 63 steps/s |
| decode, 8 streams | code ~3,350 / 3,050 tok/s aggregate; prose ~1,050 / 980 | code 2,270, prose 915 |
| decode, 32 streams | code ~5,800 / 5,100; prose ~2,360 / 2,080 | — |
| fresh prefill | 19K tokens in 2.2 s, 163K in 28.5 s, 391K in 106 s | 2.4 s, 31 s, 112 s |
| KV cache | 3.27M / 3.14M tokens on the GPUs (bf16) | 3.17M |
| quality | GSM8K-200 194/200, needle 6/6 at 28K / 105K / 399K, tool calls / JSON pass; on the NVFP4 checkpoint the steps keep stock vLLM's numerics (step 1 reproduces NCCL's bits, greedy parity identical) | GSM8K-200 193/200 |

### 1–4× RTX 5090, 1–2× RTX PRO 6000: DeepSeek-V4.1-Flash as the MoET hybrid

The cards hold the dense weights, the KV cache and 16–200 of every layer's 384 experts (the card's VRAM decides); every other
routed expert of a decode step is **computed on the CPU** from the checkpoint's MXFP4 bytes inside the captured CUDA graph, so the
output is **bit-identical to the all-resident GPU path** (0 mismatches over ~19K MoE calls per rank, checked in every cell below)
and the paired quality probes equal the native checkpoint's (**HumanEval 160/164 vs 160/164, MATH-500 L5 55/60 vs 55/60**). Images
`…-r23-hyb` / `…-r23-hyb-engram` (Dockerfile steps 43–44 on the image above); cards at 400 W; thinking and tool calling on. Cell =
host RAM used · decode on short prompts ('Hello' / 'def fib(n):', tok/s, one stream) · TTFT at 2K / 14K / 30K of prompt
([every window](docs/benchmarks/moet-hybrid-4x5090-2026-11-04.md), 2026-11-03/05):

| GPUs (context) | 512 GB host: everything in RAM | 384 GB: `ENGRAM_MODE=nvme` (+ the full LRU) | 256 GB: + the cold experts on NVMe | 128 GB |
|---|---|---|---|---|
| **4× RTX 5090** (512K) | 457 GiB · **145 / 125** · 0.9 / 3.3 / 6.5 s | 307 GiB · **147 / 126** · 0.9 / 3.4 / 6.6 s | 236 GiB · **138 / 120** · 1.3 / 4.9 / 9.2 s | 112 GiB · **112 / 82** · 2.8 / 11 / ~20 s |
| **2× RTX 5090** (64K) | 450 GiB · **46 / 47** · 1.9 / 7.3 s | 269 GiB · 45 / 47 · 2.0 / 7.4 s | 190 GiB · **40 / 43** · 2.2 / 8.6 s | 103 GiB · **36 / 35** · 3.4 / 13.4 s |
| **1× RTX 5090** (32K) | ~470 GiB · **40 / 34** · 3.8 / 15 s | 285 GiB · 43 / 36 · 3.9 s | 184 GiB · **40 / 32** · 4.1 s | 98 GiB · **38 / 26** · 4.8 s |
| **2× RTX PRO 6000** (256K) ¹ | — | 272 GiB · **97–102 / 39–77** · 1.6 / 4.3 / 8.5 s | — | — |
| **1× RTX PRO 6000** (128K) ¹ | — | 304 GiB · **64 / 49** · 3.6 / 8.1 / 15.3 s | — | — |

The 4× RTX 5090 + 512 GB flagship (TR PRO 9975WX, 498 GiB DDR5, NVMe RAID0): **84 tok/s flat over 30 min on mixed prompts**
(HumanEval + MATH + questions; 86.5 with the full LRU), 70–87 behind 2K–512K of context, **94 tok/s on Fraser Price's BRRR code bench**
(155–166 aggregate at 4–8 streams); **prefill 4.6–4.7K tok/s from 8K to 512K** (TTFT 28 s at 128K, ~113 s at 512K; PCIe-bound);
**524,288 tokens served**, needle 3/3 to ~512K; 2 / 4 streams 107 / 123 tok/s aggregate (CPU-bound from two — one stream is the
serving default), 60 min × 4 streams 2,197 requests, 0 errors; ~29 GB per card; 1.16 kW → **13.8 J per token**; boot 5.5–7 min.
Less RAM costs TTFT first (the cold experts stream from NVMe: 1.4× at 256 GB, 3× at 128 GB) and decode second; the 2× / 1× cards
are the CPU tier's wait for the misses of a small slot budget. ¹ The 96 GB cards were measured on a 354 GB test VM whose single
virtio disk serves the Engram rows (mixed-prompt soaks 55 / 35 tok/s there); at TP2 the step is GPU-bound at ~10 ms — a PRO 6000 host
with an NVMe or 512 GB of RAM puts 2× PRO 6000 at ~100 tok/s on short prompts and ~80 on mixed ones. Among the published ways to run
this model below a 384 GB-VRAM machine (CPU-expert hybrids 19–50 tok/s, disk-streaming 1× 5090 ports 5–28, DGX Station GB300 with
host offload 90) the hybrid is the only one at the released precision above 100 tok/s — the table with sources is on the record page.

Measurement records for every line: [docs/benchmarks/](docs/benchmarks/).

## Quick start: DeepSeek-V4.1-Flash

Four RTX PRO 6000 (an RTX 5090 does not fit the model at TP4), Docker with the NVIDIA container
toolkit, PCIe peer-to-peer, ~330 GB of host shared memory (less: `KV_OFFLOAD_GIB=0`).

```bash
git clone https://github.com/kacper-daftcode/vLLM-Moet && cd vLLM-Moet
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir /srv/models/DeepSeek-V4.1-Flash-1017
export IMAGE=vllm-moet-sm120:dsv41-nightly-df3bd094-r25 \
  MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017 CACHE_DIR=/srv/cache/ds41
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly -t $IMAGE .     # ~11 min

GPU_MEM_UTIL=0.92 docker/sm120/run-dsv41.sh      # first start (~22 min: FlashInfer autotunes into CACHE_DIR)
docker stop -t 60 ds41-flash
GPU_MEM_UTIL=0.94 KV_OFFLOAD_GIB=64 KV_OFFLOAD_FS_DIR=/srv/kvcache KV_OFFLOAD_PROMPT_ONLY=0 \
  INDEXER_TP_SHARD_MIN_ROWS=2048 SERVED=1 docker/sm120/run-dsv41.sh            # the served configuration
```

The OpenAI-compatible API listens on `127.0.0.1:8001`. `SERVED=1` turns on the switches production
runs with; every knob and switch is listed in [docs/dsv41-sm120.md](docs/dsv41-sm120.md).

### The MoET hybrid on RTX 5090 / RTX PRO 6000

The same image with the CPU tier built in, the expert store written once from the checkpoint, the kernel tables probed once per GPU
model; the launcher picks the slot budget and the context from `TP`, the RAM tiers from the knobs (port 8041):

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly --build-arg VLLM_MOET_CPU_TIER_PATCH=1 \
  --build-arg VLLM_MOET_ENGRAM_V2_PATCH=1 --build-arg VLLM_MOET_CPU_TIER_ARCH=znver4 \
  -t vllm-moet-sm120:dsv41-nightly-df3bd094-r23-hyb-engram .                      # -march of the host: znver4 | znver5 | sapphirerapids
export MODEL_DIR STORE=/srv/packs/dsv41-hybrid CACHE_DIR=/srv/cache/ds41-hyb CPU_TIER_TABLES=/srv/cache/ds41-hyb/cpu-tier-tables
DSV41_MODEL_DIR=$MODEL_DIR python tools/dsv41_5090/cpu_moe/build_store.py --out $STORE --check 8       # the expert store: 269 GB, ~1 min
mkdir -p $CPU_TIER_TABLES && nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o tools/dsv41_5090/cpu_moe/qmma_probe \
  tools/dsv41_5090/cpu_moe/qmma_probe.cu && tools/dsv41_5090/cpu_moe/qmma_probe --tables $CPU_TIER_TABLES && \
  python tools/dsv41_5090/cpu_tier/b12x_layout_probe.py --out $CPU_TIER_TABLES                         # the kernel tables, once per GPU model
docker/sm120/run-dsv41-5090.sh                                                       # 4x 5090 + 512 GB
ENGRAM_MODE=nvme STORE_MODE=all RESIDENT=0 SPARE=110 docker/sm120/run-dsv41-5090.sh  # 4x 5090 + 384 GB: Engram on NVMe + the full LRU
ENGRAM_MODE=nvme STORE_DRAM_RECORDS=auto docker/sm120/run-dsv41-5090.sh             # 4x 5090 + 256 / 128 GB: the cold experts on NVMe
TP=2 ENGRAM_MODE=nvme STORE_DRAM_RECORDS=auto docker/sm120/run-dsv41-5090.sh        # 2 cards (TP=1: one); 96 GB cards: SPARE=160 / 80 + the run book's knobs
```

The host needs AVX-512 (BF16 / VNNI / VBMI), four CCDs or four groups of six free cores (`VRANK_CPUS` for another geometry) and
NVMe for the store and, in the NVMe modes, the Engram shards. Knobs, the per-host recommendations and the 96 GB-card lines:
[docs/moet-hybrid.md](docs/moet-hybrid.md).

## Kernels and tools

- [`blackwell-isa`](https://github.com/kacper-daftcode/blackwell-isa) (a machine-readable SM120 SASS
  ISA database, [searchable reference](https://kacper-daftcode.github.io/blackwell-isa/SM120_ISA_REFERENCE.html))
  and [`cubit`](https://github.com/kacper-daftcode/cubit) (an SM120 assembler / disassembler built on
  it) produce the 2-bit line's SASS kernels in [`kernels/`](kernels/MANIFEST.md); every kernel the
  DeepSeek image ships round-trips through cubit bit for bit (`tools/dsv41_sm120/cubit_gate.py`).
- The DeepSeek-V4.1 kernels are CUDA C++ compiled into the image, each with an op-level test:
  [`tools/dsv41_sm120/`](tools/dsv41_sm120/README.md). The hybrid's CPU tier, its kernel and tools:
  [`tools/dsv41_5090/`](tools/dsv41_5090/README.md). Qwen3.8's MoE GEMV: `tools/qwen38_sm120/moe_gemv/`.
- Host-side benchmark and profiling scripts: [`tools/sm120_perf/`](tools/sm120_perf/README.md).
- SM120 kernel-design notes from the hardware counters: [docs/sm120-alu-subunits.md](docs/sm120-alu-subunits.md).

## Repository layout

| path | what |
|---|---|
| `Dockerfile.sm120-dsv41-nightly`, `tools/dsv41_sm120/`, `docker/sm120/run-dsv41.sh` | DeepSeek-V4.1-Flash: the image, its patchers, kernels and tests, the launcher (+ the cron scripts in `docker/sm120/`) |
| `tools/dsv41_5090/`, `docker/sm120/run-dsv41-5090.sh`, `docker/sm120/host/`, `docker/sm120/seccomp-io_uring.json` | the MoET hybrid (Dockerfile steps 43–44): the CPU expert tier and its kernel, Engram v2, the routing tools, the measurement harness; its launcher; the host units (GPU power cap, kernel-thread affinity); the seccomp profile io_uring needs |
| `Dockerfile.sm120-qwen38`, `tools/qwen38_sm120/`, `docker/sm120/run-qwen38.sh` | Qwen3.8-Flash-Next-FP8 |
| `Dockerfile.b300-k3`, `tools/k3_b300/`, `docker/b300/run-k3.sh` | Kimi-K3 + DSpark on 8× B300: the image (official v0.32.0 + anchored patchers), the probes and the KV budget model, the launcher |
| `Dockerfile.b300-k31`, `tools/k3_b300/patch_vllm_k31_experts_grid.py` | Kimi-K3.1-Pro on 8× B300: the day-0 image + the specialized decode's experts grid capped at the device's SM count; the same launcher serves both families |
| `Dockerfile.sm120-ripple`, `tools/ripple_sm120/`, `docker/sm120/run-ripple.sh` | Ripple-65B Preview (the all-reduce module is `tools/dsv41_sm120/p2p_ar/`, shared with DeepSeek) |
| `patch/`, `Dockerfile.sm120-v024`, `Dockerfile.recipes`, `docker/serve_recipe.py`, `bench/`, `kernels/`, `tools/*.py` | the 2-bit line: the generated v0.24.0 patch (never edited by hand), its images, the recipes and the benchmark runner, SASS sources and cubins, its tests |
| `tools/sm120_perf/` | benchmark and profiling scripts |
| `docs/` | one doc per line, the benchmark records, the kernel-design notes |

## Contributing and license

[CONTRIBUTING.md](CONTRIBUTING.md) says where a change goes and how it is validated;
[CHANGELOG.md](CHANGELOG.md) lists the served images. Apache License 2.0 ([LICENSE](LICENSE));
third-party components in [NOTICE](NOTICE).
