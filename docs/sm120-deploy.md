# Serving DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next-FP8 on 4× RTX PRO 6000 (sm_120)

Two self-contained serving images built from the official vLLM images for these models, with
the vLLM-Moet sm_120 fixes and decode kernels baked in. Everything a new host needs is in this
repository plus the model checkpoints; nothing is bind-mounted from a host-specific directory.

| model | image | Dockerfile | launcher | single-stream decode (TP4, greedy) |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash (official MXFP4/MXFP8 checkpoint, vision on) | `vllm-moet-sm120:dsv41-nightly-20260923` (served since 2026-09-23; `dsv41-0909` = rollback) | `Dockerfile.sm120-dsv41-nightly` (`Dockerfile.sm120-dsv41` for the 0909 image) | `docker/sm120/run-dsv41.sh` | 74 steps/s, prose 163 / code 385 tok/s (DSpark k=5), 3.45M-token FP4 KV at 512K context + KV offload (64 GiB host RAM, disk tier; since 2026-09-23) (0909 image: 67 steps/s, 152 / 360 tok/s, 3.06M tokens) |
| Qwen3.8-Flash-Next-FP8 (official checkpoint) | `vllm-moet-sm120:qwen38-20073` | `Dockerfile.sm120-qwen38` | `docker/sm120/run-qwen38.sh` | 98.5 steps/s, prose 243 / code 346 tok/s (MTP k=3), 2.28M-token KV at 256K context |

What the images change relative to the official ones, and the measurements behind each change:
`docs/dsv41-sm120-port.md`, `tools/dsv41_sm120/README.md`, `tools/qwen38_sm120/README.md`,
`tools/sm120_perf/README.md` (the profiling method).

## Host requirements

- 4× RTX PRO 6000 Blackwell (96 GB, sm_120) — RTX 5090 (32 GB) does not fit either model at TP4.
  The images were validated on a KVM guest with the GPUs on PCIe (no NVLink); a bare-metal host
  with the same cards behaves the same or better.
- NVIDIA driver with CUDA 13 support (the images ship CUDA 13.0 / torch 2.13), Docker with the
  NVIDIA container toolkit. `--ipc host` and `--shm-size 32g` are required (NCCL, PLE worker).
- Disk: the checkpoints (~180 GB Qwen3.8-Flash-Next-FP8, ~510 GB DeepSeek-V4.1-Flash), ~30 GB per
  image, a few GB of per-image caches (`CACHE_DIR`).
- Host RAM: ≥ 64 GB is comfortable. Qwen3.8's 51 GB n-gram table lives on the GPUs in this
  configuration (`VLLM_PLE_CPU_OFFLOAD=0`, +12 GB/GPU), so no pinned host memory is needed for it.
- NCCL: if `nvidia-smi topo -m` shows `PHB`/`SYS` for every pair (typical for passthrough VMs),
  NCCL refuses P2P by default and falls back to SHM (18–30 µs per 4-rank allreduce). P2P works in
  those VMs (`cudaDeviceCanAccessPeer` is true); the launchers set `NCCL_P2P_LEVEL=SYS`, which
  brings the ~90–116 allreduces per decode step to ~11–13 µs each. vLLM's own one-shot custom
  allreduce is pull-based and slower than NCCL on PCIe — leave it off (the launchers do not
  enable it).

## What you need (nothing is patched by hand)

Everything below the model weights is produced by the two Dockerfiles from public inputs:

| input | where it comes from | pin |
|---|---|---|
| base image, DeepSeek (served) | Docker Hub `vllm/vllm-openai:nightly@sha256:42090442…` (vLLM main 0961bbae, torch 2.13+cu130) + FlashInfer nightly wheels `0.7.0.dev20260922` (`flashinfer.ai/whl/nightly`) + `vllm-project/DeepGEMM` e1f418c2 | digest / version pins in `Dockerfile.sm120-dsv41-nightly` |
| base image, DeepSeek (0909, rollback) | Docker Hub `vllm/vllm-openai:deepseekv41-flash-0909` (vLLM 0.30, FlashInfer 0.6.18, vendored DeepGEMM, nvcc 13.0, torch 2.13) | the vLLM recipe's NVIDIA tag |
| base image, Qwen | Docker Hub `vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` (= the `qwen38-flash-next` nightly, v0.1.dev20073+g8e685d198) | digest |
| DeepGEMM source (rebuilt inside the DeepSeek image with three host-side asserts relaxed) | `github.com/deepseek-ai/DeepGEMM` | commit `8b1392b978f5a03c828dd1711090d7fb50958b8a` |
| FlashInfer sparse-MLA sm_120 instantiations for the V4.1 geometry, dispatch hook | this repo, `tools/dsv41_sm120/` (installed into the image's FlashInfer by `patch_flashinfer.py`, precompiled) | — |
| vLLM patches (anchored, refuse to apply to other versions) and the two CUDA kernels | this repo, `tools/dsv41_sm120/`, `tools/qwen38_sm120/` | — |
| weights | Hugging Face `deepseek-ai/DeepSeek-V4.1-Flash` (MIT, ~510 GB), `Qwen/Qwen3.8-Flash-Next-FP8` (Apache-2.0, ~180 GB); no gating | the launchers read the directory you point them at |

```bash
pip install -U huggingface_hub
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir /srv/models/DeepSeek-V4.1-Flash
hf download Qwen/Qwen3.8-Flash-Next-FP8 --local-dir /srv/models/Qwen3.8-Flash-Next-FP8
```

The build needs network access (base image pull, DeepGEMM clone); serving does not. Both images
were rebuilt from scratch (`docker build --no-cache`) from the committed tree on 2026-09-19: the
patched vLLM/FlashInfer files, the DeepGEMM `_C` and the compiled kernel extensions came out
identical to the images serving on the reference host. Without network, `docker load` the saved
image tarballs instead (see Build).

## Build

```bash
git clone <this repo> && cd vllm-moet
# DeepSeek, served image: vLLM main line -- vllm/vllm-openai:nightly (pinned digest) + FlashInfer nightly wheels (the DSv4.1
# dual-cache sparse MLA reads the FP4 KV record itself; no TU/hook, no scratch/pool) + DeepGEMM _C at vLLM main's pin with
# the SM120 per-group BLOCK_M restored (docs/dsv41-sm120-port.md, "The vLLM main line as a candidate image")
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly -t vllm-moet-sm120:dsv41-nightly-20260923 .   # ~12 min
# DeepSeek, rollback image: the recipe's 0909 image + the same fixes on its own FlashInfer/DeepGEMM pins
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41  -t vllm-moet-sm120:dsv41-0909  .   # ~5 min after the base pull (DeepGEMM _C rebuild + FlashInfer JIT precompile)
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-qwen38 -t vllm-moet-sm120:qwen38-20073 .   # ~4 min (MoE GEMV extension compile)
```

The launcher picks the KV plumbing from the image label `com.vllm-moet.kv-mode` (`KV_MODE=auto`): the 0909 image
keeps `--kv-cache-dtype fp8` + `VLLM_MOET_KV_RECORD`, the nightly image maps `KV_RECORD=nvfp4` to
`--kv-cache-dtype nvfp4_ds_mla` (FlashInfer's own reader of the record) and `fp8_ds_mla` to `--kv-cache-dtype fp8_ds_mla`.

`KV_OFFLOAD_GIB=N` (vLLM-main image only) adds `--kv-offloading-size N`: evicted KV of the compressed
cache and the indexer is kept in N GiB of pinned host RAM (one `/dev/shm` region; the container runs
with `--ipc host`) and restored on a prefix hit instead of being recomputed — a 179K-token context
comes back in 0.5 s instead of a 17 s prefill, 3.6 KB of host RAM per token at TP4, decode and
prefill unchanged. `KV_OFFLOAD_FS_DIR=DIR` adds a disk tier behind it: every offloaded block is also
written to DIR as a file named by its content hash, read back on a hit that misses RAM (a 209K-token
context in 2.5 s from a virtio disk) and still valid after a restart (174K tokens in 3.2 s instead of
16.7 s). vLLM never deletes the files — run `docker/sm120/kvcache-ttl.sh DIR` hourly from cron (files
unread for 72 h, then the least recently read above 200 GB; `TTL_HOURS` / `MAX_GB`).
`KV_OFFLOAD_PROMPT_ONLY=0` offloads generated tokens too, so the next turn of an agent conversation
also hits the previous answer. The served deployment runs all three (64 GiB, a disk directory, 0);
`docs/dsv41-sm120-port.md`, "KV outside HBM".

All bases are pinned (`vllm/vllm-openai:nightly@sha256:42090442…` + FlashInfer `0.7.0.dev20260922`,
`vllm/vllm-openai:deepseekv41-flash-0909`, `vllm/vllm-openai@sha256:fc120ece…` = the `qwen38-flash-next`
nightly, v0.1.dev20073); the patchers are anchored on those exact files and refuse to apply to anything
else. To move a host without rebuilding, `docker save` / `docker load` the tags (~30–40 GB each).

Run the in-image tests once per build (one GPU, ~2 min each):

```bash
docker run --rm --gpus '"device=0"' --ipc host --entrypoint bash vllm-moet-sm120:qwen38-20073 -c \
  'python3 /opt/vllm-moet/qwen38_sm120/moe_gemv/test_fused_moe_integration.py'
docker run --rm --gpus '"device=0"' --ipc host --entrypoint bash vllm-moet-sm120:dsv41-nightly-20260923 -c \
  'python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_mxfp8_gemv_sm120.py --ms 1,6,16 --shapes decode &&
   python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_wo_a_integration.py &&
   python3 /opt/vllm-moet/dsv41_sm120/test_deepgemm_sm120_paged_mqa.py --packed-stride &&
   python3 /opt/vllm-moet/dsv41_sm120/test_indexer_fp4_sm120.py'
```

## Serve

```bash
# DeepSeek-V4.1-Flash, GPUs 0-3, port 8001 (first start ~25 min: FlashInfer autotune + DeepGEMM JIT fill CACHE_DIR; later ~17 min)
MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash CACHE_DIR=/srv/cache/ds41 GPU_MEM_UTIL=0.92 docker/sm120/run-dsv41.sh
# second start onwards: GPU_MEM_UTIL=0.94 (3.06M KV tokens with the defaults KV_RECORD=nvfp4 + INDEXER_KV_DTYPE=mxfp4;
# 2.29M with KV_RECORD=fp8_ds_mla, 1.49M with INDEXER_KV_DTYPE=fp8 as well); 0.95 OOMs during the autotune sweep on a cold cache

# Qwen3.8-Flash-Next-FP8, GPUs 4-7, port 8000 (first start ~10 min: torch.compile + FlashInfer JIT; later ~4 min)
MODEL_DIR=/srv/models/Qwen3.8-Flash-Next-FP8 CACHE_DIR=/srv/cache/qwen38 GPUS=4,5,6,7 docker/sm120/run-qwen38.sh
```

The launchers print the log lines that confirm the patched paths are active. Every knob (context
length, batch limits, speculative tokens, profiler, bind address, extra `vllm serve` arguments) is
an environment variable documented in the launcher header; the defaults are the validated
configuration. `BIND=` publishes the port on all interfaces (default: loopback only — put a proxy
with authentication in front).

Runtime kill switches (env through `EXTRA_DOCKER_ARGS="-e …"`, no rebuild): `VLLM_MOET_SM120_GEMV=0`
/ `VLLM_MOET_SM120_GEMV_BMM=0` (DeepSeek dense / grouped GEMV → CUTLASS / BF16 emulation),
`VLLM_MOET_GEMV_IMPL=v1` (dense GEMV: the kernel served before 2026-09-20 instead of v3),
`VLLM_MOET_SM120_MOE_GEMV=0` (Qwen MoE GEMV → Triton), `VLLM_MOET_SM120_MOE_GEMV_FUSE_ACT=0`,
`VLLM_MOET_SM120_LL_GEMM=0` (Qwen skinny GEMM → cuBLAS; changes the compiled graph, so use a fresh
`CACHE_DIR`), `VLLM_PLE_CPU_OFFLOAD=1` (Qwen PLE table in a CPU worker; then `KV_CACHE_MEMORY=`).

## Validate a new host

```bash
curl -s 127.0.0.1:8001/health; curl -s 127.0.0.1:8000/health
python3 tools/sm120_perf/decode_bench.py http://127.0.0.1:8001 deepseek-ai/DeepSeek-V4.1-Flash '{"thinking": false}'
python3 tools/sm120_perf/decode_bench.py http://127.0.0.1:8000 Qwen3.8-Flash-Next '{"enable_thinking": false}'
python3 tools/sm120_perf/needle_any.py http://127.0.0.1:8000 Qwen3.8-Flash-Next '{"chat_template_kwargs": {"enable_thinking": false}}' 30000,100000
python3 tools/sm120_perf/needle_any.py http://127.0.0.1:8001 deepseek-ai/DeepSeek-V4.1-Flash '{"chat_template_kwargs": {"thinking": false}}' 30000,100000
```

Expected on 4× RTX PRO 6000 with P2P: DeepSeek 66–68 steps/s (prose ~2.2, code ~5.1 tok/step),
Qwen 97–99 steps/s (prose ~2.5, code ~3.5 tok/step), needle PASS at both lengths. A first request
after start is slower (warm-up). If steps/s are ~10 % low and the log shows NCCL on `SHM`, P2P is
not available on the host — check IOMMU / ACS settings. The DeepSeek log should say `Using MXFP4
indexer cache for Lightning Indexer` (the default since 2026-09-20; `INDEXER_KV_DTYPE=fp8` gives the
previous 132 B/key cache) and `GPU KV cache size: 3,05x,xxx tokens` at `GPU_MEM_UTIL=0.94` with the
default `KV_RECORD=nvfp4` (2,28x,xxx with `KV_RECORD=fp8_ds_mla`). Expect prefill ~3 % and decode ~1 %
below the fp8-record figures with the FP4 record (`docs/dsv41-sm120-port.md`).

The DeepSeek image renders the checkpoint's reasoning-effort tiers (`low` 50 / `high` 75 / `max`
100, default `high`; see `docs/dsv41-sm120-port.md`). To confirm on a host without a GPU free:

```bash
docker run --rm --entrypoint python3 -v /srv/models/DeepSeek-V4.1-Flash:/model:ro vllm-moet-sm120:dsv41-nightly-20260923 \
  /opt/vllm-moet/dsv41_sm120/test_reasoning_effort_encoding.py --model-dir /model
```

## Memory headroom

DeepSeek at `GPU_MEM_UTIL=0.94` peaks at ~96.0 GB/GPU under C8 with 7.7K-token prompts (1.2 GB from
the wall). Where the card goes (vLLM's own start-up accounting, per GPU of 94.97 GiB usable):
84.4 GiB weights + non-torch, 1.8 GiB peak activation at 4096 batched tokens, 0.4 GiB CUDA graphs,
**3.1 GiB KV = 1.47M tokens**; another ~4 GiB appear during serving (FlashInfer/DeepGEMM workspaces,
NCCL, allocator) — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` changes none of it (84.41 GiB
consumed, same KV), so this is real allocation, not fragmentation. The 4.2M-token pools that the
EP4 SGLang recipe reports on the same cards come from its 72.6 GiB weight footprint, not from a
knob on this side.

Those numbers are the FP8 indexer cache and the fp8 compressed record. With the **MXFP4 indexer
cache** (the default since 2026-09-20, `docs/dsv41-sm120-port.md`) the same `0.94 / 4096` start reports
**4.39 GiB KV = 2.29M tokens (4.4× at 512K)**; with the **FP4 compressed record** on top
(`KV_RECORD=nvfp4`, the default since 2026-09-21) the block shrinks 210 240 → 115 200 B and the same
memory holds **3.87 GiB KV = 3.06M tokens (5.8× at 512K)** (the packed path reserves a 306 MB prefill
pool and a 19 MB decode scratch in the profile run, so they are accounted for; peak under the stress
battery 95 897 MiB — 1.99 GB of margin). The MXFP4-indexer-only figures: the indexer page shrinks (+9.6 %) and vLLM's profile run counts
1.25 GiB less non-torch memory, which it hands to the KV cache. Under load the cards then level
off at **97 001 of 97 887 MiB (886 MiB from the wall)** — GSM8K C4, 8 concurrent fresh 126K
prefills, a fresh 139K prefill, the 367K needle and vision all passed there, but the margin is the
0.95-profile's, not the old default's; `GPU_MEM_UTIL=0.93` with MXFP4 gives ~3.4 GiB KV (~1.8M
tokens) with the old ~1.9 GB margin.

Capacity profile, measured 2026-09-19 with the FP8 indexer: `GPU_MEM_UTIL=0.95
MAX_NUM_BATCHED_TOKENS=2048` gives **4.19 GiB KV = 2.21M tokens (+50 %, 4.2× concurrency at
512K)** at the same decode speed (67 steps/s, 147 / 347 tok/s), **−7 % prefill** (10.1–11.0k vs
11.0–11.9k tok/s) and a thinner margin: 716 MiB free at the peak of 8×131K-token streams + a 367K
needle + vision (all passed). Use it only with the FlashInfer autotune cache already populated
(the cold sweep OOMs at 0.95), and keep `0.94 / 4096` where prefill or margin matter more (not
re-measured with the MXFP4 indexer). Qwen with `--kv-cache-memory 32GiB` uses ~88 GB/GPU; vLLM's own
start-up estimate would give only 5.6 GiB of KV because the GPU-PLE profile run over-reports its
peak activation (a compile-time transient), hence the explicit KV size. Image/video prompts on
Qwen (it is a VL model) were not stress-tested for memory; drop `KV_CACHE_MEMORY` to 28 GiB if
multimodal traffic OOMs.
