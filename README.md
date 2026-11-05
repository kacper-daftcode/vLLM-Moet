# vLLM-Moet

Frontier Mixture-of-Experts models on RTX PRO 6000 and RTX 5090 (sm_120), where the official vLLM
images do not run them or the official checkpoints do not fit. The changes are kernel-side patches
on official vLLM images, bit-identical to the paths they replace unless a doc says otherwise, each
behind a runtime switch.

| line | what | status | doc |
|---|---|---|---|
| **DeepSeek-V4.1-Flash on 4× RTX PRO 6000** | the official checkpoint on vLLM nightly: the sm_120 gaps closed, the decode step and the prefill rebuilt around small tensor-core kernels | **active, in production** | [docs/dsv41-sm120.md](docs/dsv41-sm120.md) |
| **DeepSeek-V4.1-Flash on 1–4× RTX 5090 (MoET hybrid)** | a subset of the 384 experts per layer resident on the cards, every other routed expert computed on the CPU from the checkpoint's own bytes, bit-identical to the all-resident path; 4× 5090 + 512 GB: 145 / 125 tok/s on one stream, 4.6K tok/s prefill, 512K context; the same code on 2× / 1× 5090 and on 128–384 GB hosts (the Engram tables and the cold experts on NVMe) | serving; every configuration below measured on the reference box | [docs/moet-hybrid.md](docs/moet-hybrid.md), [the measurements and the field](docs/benchmarks/moet-hybrid-4x5090-2026-11-04.md) |
| **2-bit experts on vLLM v0.24.0** | GLM-5.2 (753B), DeepSeek-V4-Flash (159B) and Kimi-K2.7-Code (1T) on one to four cards that cannot hold their checkpoints: a 2-bit codebook on hand-written SASS, an FP4 delta cache | frozen (builds, fixes accepted, no development) | [docs/v024-2bit.md](docs/v024-2bit.md) |
| **Qwen3.8-Flash-Next-FP8 on 4× RTX PRO 6000** | the official nightly with the PLE table on the GPUs, the skinny GEMM and an FP8 MoE GEMV | as built | [docs/qwen38-sm120.md](docs/qwen38-sm120.md) |
| **Ripple-65B Preview (NVFP4) on 4× RTX PRO 6000** | Ricursive's 65B dense model with its DFlash2 drafter on vLLM main: the TP all-reduces on the PCIe P2P push, the draft length by batch size; code ~600 tok/s single stream, ~3000 aggregate at 8 requests | serving (first image) | [docs/ripple-sm120.md](docs/ripple-sm120.md) |

## Results, as served

DeepSeek-V4.1-Flash, 4× RTX PRO 6000 at 400 W, image `vllm-moet-sm120:dsv41-nightly-df3bd094-r25` (served since 2026-11-05 21:36Z; `-r24` before it)
(TP4, 512K context, DSpark k = 7, vision, thinking, tool calling, KV offload to host RAM and disk):

| | vLLM-Moet | official vLLM image on the same cards |
|---|---|---|
| decode, 1 stream | prose ~293 tok/s, code ~802 tok/s (129 steps/s); Fraser bench 688 tok/s median, 746 best over 40 reps | 60 steps/s with only the fixes it needs to start |
| decode, 8 streams | prose ~721, code ~2,258 tok/s aggregate | — |
| fresh prefill | 163K tokens in 6.6 s (24.5K tok/s), 19K in 0.79 s (`-r25`: the sparse-MLA prefill attention on the cubit-assembled SASS kernel, bit-identical to `-r24`'s 6.8 s and `-r23`'s 7.1 s); BRRR aggregate +11…+28 % over the recipe's stock vLLM at 2K–65K | 163K in ~15.5 s |
| KV cache | 9.62M tokens on the GPUs (18× concurrency at 512K) + 64 GiB host RAM + a disk tier | 1.49M (official KV formats) |
| quality | GSM8K-200 194/200, needle PASS at 29K / 106K / 400K, tool calls / JSON / vision pass; identical requests return identical logits at any batch size and prompt length | the same checkpoint's quality |

DeepSeek-V4.1-Flash, the MoET hybrid on GeForce RTX 5090 cards at 400 W (TP4 on four cards; thinking, tool calling; a
subset of the 384 experts per layer on the cards, the rest computed on the CPU in-step, bit-identical to the all-resident
path; one reference box — TR PRO 9975WX, 498 GiB DDR5, NVMe RAID0 — 2026-11-03/04, [the record and the field](docs/benchmarks/moet-hybrid-4x5090-2026-11-04.md)).
The flagship, 4× RTX 5090 + 512 GB, image `vllm-moet-sm120:dsv41-nightly-df3bd094-r23-hyb`, the launcher's defaults:

| | 4× RTX 5090 + 512 GB RAM |
|---|---|
| decode, 1 stream | **145 / 125 tok/s** on short prompts ('Hello' / 'def fib(n):', greedy; boot-to-boot band ±3.5 %); **84 tok/s flat over 30 min** on mixed prompts (HumanEval + MATH + questions); 70–87 behind 2K–512K of context; **94 tok/s on Fraser Price's BRRR code bench** at 2K / 16K / 64K alike (native 4× RTX PRO 6000 with a 7-token draft: 700–770) |
| decode, several streams | 2 streams 107 tok/s, 4 streams 123 aggregate (10-min soaks, 0 errors); BRRR 155 / 166 aggregate at 4 / 8 streams; 60 min × 4 streams: 2,197 requests, 0 errors. CPU-bound from two streams — `--max-num-seqs 1` is the serving default |
| prefill | **4.6–4.7K tok/s from 8K to 512K** (2.4K on a single 2K chunk); TTFT 0.9 s at 2K, 1.8 s at 8K, 14 s at 64K, 28 s at 128K, ~113 s at 512K — PCIe-bound (the missing experts of each 4096-token chunk stream from pinned RAM) |
| context | **524,288 tokens served** (`--max-model-len`); needle-in-a-haystack 3/3 at 32K / 64K / 256K / ~512K; a cached 512K prefix answers in 1.5 s |
| quality, paired against the native FP4 reference | **HumanEval 160/164 vs 160/164** (0 discordant, McNemar p = 1.0); **MATH-500 level 5: 55/60 vs 55/60** (1 / 1 discordant, p = 1.0) — same prompts, greedy, thinking off |
| identity | bit-exact against the all-resident b12x path within one boot: 19,040–19,080 compact MoE calls per rank, ~25K CPU-computed rows, **0 differ** — re-run on every image and at TP2 / TP1; the CPU kernel bit-exact against b12x's GPU kernels on four recorded sets |
| the step | 7.5 ms: b12x MoE 3.1, the GPU waiting for the CPU tier 2.2 (~10 of 40 layers miss an expert per step, ~130 µs each), mHC / norms 1.25, attention 1.1, NCCL 0.8–1.0, dense GEMVs 0.7 |
| footprint, power | ~29.3 GB per card (110 expert slots); ~457 GiB of host RAM pinned (the expert store 241 GiB + the Engram tables 189 GiB); boot 5.5–7 min; 1.16 kW under one stream → **13.8 J per token** (8.3 at 4 streams), 366 W idle |

Every configuration measured (the same code and image family; `TP=2` / `TP=1` run one expert as its 576-row slices on fewer
cards, `ENGRAM_MODE=nvme` keeps the Engram tables on NVMe, `STORE_NVME=1 STORE_DRAM_RECORDS=N|auto` keeps only N expert records
per layer in DRAM and reads the cold ones from NVMe; cell = host RAM used · decode 'Hello' / 'def fib(n):' tok/s · TTFT at 2K / 14K
of prompt; the within-boot identity check passes in every cell; the per-window numbers in
[docs/benchmarks/moet-hybrid-4x5090-2026-11-04.md](docs/benchmarks/moet-hybrid-4x5090-2026-11-04.md)):

| GPUs (context) | 512 GB host (the defaults: everything pinned) | 384 GB host (`ENGRAM_MODE=nvme`) | 256 GB host (+ the store's tail on NVMe) | 128 GB host |
|---|---|---|---|---|
| **4× RTX 5090** (512K) | 457 GiB · **146 / 126** · 0.9 / 3.3 s (30K: 6.5 s) | 278 GiB · 145 / 126 · 0.9 / 3.3 s; **+ the full LRU** (`STORE_MODE=all RESIDENT=0 SPARE=110`): 307 GiB · **147 / 126** · 0.9 / 3.5 s, **mixed prompts 86.5 tok/s flat over 30 min** (hit rate 0.88 vs 0.87; the 512 GB defaults' soak 84.4) | N = 280: **236 GiB · 138 / 120 · 1.3 / 4.9 s** (30K: 9.2 s, 1.4–1.5× the TTFT); N = 230: 200 GiB · 132 / 109 · 1.7 / 6.3 s | N = 110 (`ENGRAM_CACHE_GIB=1`): **112 GiB · 112 / 82 · 2.8 / 11.1 s**; N = 120: 120 GiB · 114 / 81 · 2.7 / 10.4 s (3.0–3.6× the TTFT) |
| **2× RTX 5090** (64K) | 450 GiB · **46 / 47** · 1.9 / 7.3 s | 269 GiB · 45 / 47 · 2.0 / 7.4 s | N = 230: 190 GiB · **40 / 43** · 2.2 / 8.6 s | N = 110: 103 GiB · **36 / 35** · 3.4 / 13.4 s |
| **1× RTX 5090** (32K) | ~470 GiB · **40 / 34** · 3.8 / 15 s | 285 GiB · 43 / 36 · 3.9 s | N = 230: 184 GiB · **40 / 32** · 4.1 s | N = 110: 98 GiB · **38 / 26** · 4.8 s |
| **2× RTX PRO 6000** Server Edition (200 slots, 256K; a 354 GB test VM with one virtio disk, 2026-11-05) | — | 272 GiB · **97–102 / 39–77** · 1.6 / 4.3 s (30K: 8.5 s); hit rate 0.96; mixed prompts 55 tok/s with the disk's Engram tail; the step is GPU-bound at ~10 ms | — | — |
| **1× RTX PRO 6000** Server Edition (89 slots, 128K; the same VM) | — | 304 GiB · **64 / 49** · 3.6 / 8.1 s; hit rate 0.86; mixed prompts 35 tok/s (disk-bound) | — | — |

Reading the matrix: on four cards the NVMe tail costs 5–14 % of decode at N = 230–280 and 22–35 % at N = 110–120 (the prefill
1.4× / 2× / 3× slower); on one or two cards the step is already the CPU tier's wait for the misses of a small slot budget
(46 / 16 slots), so the tail costs little more. The footprint is ≈ 35 GiB + 0.72 GiB × N with Engram on NVMe (`STORE_DRAM_RECORDS=auto`
fills the host with a 20 GiB headroom). The 96 GB cards run the same code bit-identically (identity 0 mismatches at both TPs) with
the hit rates the traces predicted; on the test host they are held back by its disk (the Engram rows' tail, 0.3–3 ms per step and
0.1–0.5 s per new prompt) and, at TP2, by the card's GPU side (~8 ms per step + 1.9 ms of CPU-tier wait) — a PRO 6000 host with an
NVMe or 512 GB of RAM would put 2× PRO 6000 at ~100 tok/s short / ~80 mixed ([the record page](docs/benchmarks/moet-hybrid-4x5090-2026-11-04.md) "2× and 1× RTX PRO 6000").

Where the hybrid sits among the published ways to run this model below the datacenter (CPU-expert hybrids on DDR4 hosts
19–50 tok/s, disk-streaming single-5090 ports 5–28, DGX Station GB300 with host offload 90, native 4× RTX PRO 6000 700–770):
the only entry below 384 GB of VRAM that keeps the released precision and decodes above 100 tok/s — the table with sources is
on [the record page](docs/benchmarks/moet-hybrid-4x5090-2026-11-04.md).

Measurement records: [docs/benchmarks/](docs/benchmarks/). The hybrid's design and knobs are in
[docs/moet-hybrid.md](docs/moet-hybrid.md), the 2-bit line's tables in [docs/v024-2bit.md](docs/v024-2bit.md).

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

### The MoET hybrid on RTX 5090

The same image with the CPU tier built in (Dockerfile steps 43–44), the expert store written once from the checkpoint, the
kernel tables probed once per GPU model, then the launcher's defaults pick the configuration from `TP` and the RAM knobs:

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly --build-arg VLLM_MOET_CPU_TIER_PATCH=1 \
  --build-arg VLLM_MOET_ENGRAM_V2_PATCH=1 --build-arg VLLM_MOET_CPU_TIER_ARCH=znver4 \
  -t vllm-moet-sm120:dsv41-nightly-df3bd094-r23-hyb-engram .                      # znver4 | znver5 | sapphirerapids
DSV41_MODEL_DIR=$MODEL_DIR python tools/dsv41_5090/cpu_moe/build_store.py --out /srv/packs/dsv41-hybrid --check 8   # 269 GB, ~1 min
export MODEL_DIR STORE=/srv/packs/dsv41-hybrid CACHE_DIR=/srv/cache/ds41-hyb CPU_TIER_TABLES=/srv/cache/ds41-hyb/cpu-tier-tables
mkdir -p $CPU_TIER_TABLES && nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o tools/dsv41_5090/cpu_moe/qmma_probe \
  tools/dsv41_5090/cpu_moe/qmma_probe.cu && tools/dsv41_5090/cpu_moe/qmma_probe --tables $CPU_TIER_TABLES   # once per GPU model
python tools/dsv41_5090/cpu_tier/b12x_layout_probe.py --out $CPU_TIER_TABLES                                 # (in the image: docs)
docker/sm120/run-dsv41-5090.sh                                      # 4x 5090 + 512 GB: the flagship (port 8041)
ENGRAM_MODE=nvme STORE_MODE=all RESIDENT=0 SPARE=110 docker/sm120/run-dsv41-5090.sh   # 4x 5090 + 384 GB (the full LRU)
ENGRAM_MODE=nvme STORE_DRAM_RECORDS=auto docker/sm120/run-dsv41-5090.sh            # 4x 5090 + 256 / 128 GB (N from the host)
TP=2 ENGRAM_MODE=nvme STORE_DRAM_RECORDS=auto docker/sm120/run-dsv41-5090.sh       # 2x 5090; TP=1 for one card
```

The host needs AVX-512 (BF16 / VNNI / VBMI) and four CCDs or four groups of six free cores (`VRANK_CPUS` for another
geometry), NVMe for the store and, in the NVMe modes, for the Engram shards. The kernel tables, the knobs and the per-host
recommendations: [docs/moet-hybrid.md](docs/moet-hybrid.md) ("Running it", "Hosts with less RAM").

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
| `tools/dsv41_5090/`, `docker/sm120/run-dsv41-5090.sh`, `docker/sm120/host/`, `docker/sm120/seccomp-io_uring.json` | the MoET hybrid (Dockerfile steps 43–44): the CPU expert tier and its kernel, Engram v2, the routing tools, the measurement harness ([index](tools/dsv41_5090/README.md)); its launcher; the host units (GPU power cap, kernel-thread affinity); the seccomp profile io_uring needs |
| `Dockerfile.sm120-qwen38`, `tools/qwen38_sm120/`, `docker/sm120/run-qwen38.sh` | Qwen3.8-Flash-Next-FP8 |
| `Dockerfile.sm120-ripple`, `tools/ripple_sm120/`, `docker/sm120/run-ripple.sh` | Ripple-65B Preview (the all-reduce module is `tools/dsv41_sm120/p2p_ar/`, shared with DeepSeek) |
| `patch/`, `Dockerfile.sm120-v024`, `Dockerfile.recipes`, `docker/serve_recipe.py`, `bench/`, `kernels/`, `tools/*.py` | the 2-bit line: the generated v0.24.0 patch (never edited by hand), its images, the recipes and the benchmark runner, SASS sources and cubins, its tests |
| `tools/sm120_perf/` | benchmark and profiling scripts |
| `docs/` | one doc per line, the benchmark records, the kernel-design notes |

## Contributing and license

[CONTRIBUTING.md](CONTRIBUTING.md) says where a change goes and how it is validated;
[CHANGELOG.md](CHANGELOG.md) lists the served images. Apache License 2.0 ([LICENSE](LICENSE));
third-party components in [NOTICE](NOTICE).
