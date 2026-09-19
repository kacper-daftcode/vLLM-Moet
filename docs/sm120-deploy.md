# Serving DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next-FP8 on 4× RTX PRO 6000 (sm_120)

Two self-contained serving images built from the official vLLM images for these models, with
the vLLM-Moet sm_120 fixes and decode kernels baked in. Everything a new host needs is in this
repository plus the model checkpoints; nothing is bind-mounted from a host-specific directory.

| model | image | Dockerfile | launcher | single-stream decode (TP4, greedy) |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash (official MXFP4/MXFP8 checkpoint, vision on) | `vllm-moet-sm120:dsv41-0909` | `Dockerfile.sm120-dsv41` | `docker/sm120/run-dsv41.sh` | 67 steps/s, prose 148 / code 346 tok/s (DSpark k=5), 1.49M-token fp8 KV at 512K context |
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
| base image, DeepSeek | Docker Hub `vllm/vllm-openai:deepseekv41-flash-0909` (vLLM 0.30, FlashInfer 0.6.18, vendored DeepGEMM, nvcc 13.0, torch 2.13) | the vLLM recipe's NVIDIA tag |
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

The build needs network access (base image pull, DeepGEMM clone); serving does not. The images
were rebuilt from scratch (`docker build --no-cache`) from the committed tree on 2026-09-19 to
confirm this. Without network, `docker load` the saved image tarballs instead (see Build).

## Build

```bash
git clone <this repo> && cd vllm-moet
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41  -t vllm-moet-sm120:dsv41-0909  .   # ~20 min (DeepGEMM _C rebuild + JIT precompile)
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-qwen38 -t vllm-moet-sm120:qwen38-20073 .   # ~3 min
```

Both bases are pinned (`vllm/vllm-openai:deepseekv41-flash-0909`, `vllm/vllm-openai@sha256:fc120ece…`
= the `qwen38-flash-next` nightly, v0.1.dev20073); the patchers are anchored on those exact files
and refuse to apply to anything else. To move a host without rebuilding, `docker save` / `docker load`
the two tags (~30 GB each).

Run the in-image tests once per build (one GPU, ~2 min each):

```bash
docker run --rm --gpus '"device=0"' --ipc host --entrypoint bash vllm-moet-sm120:qwen38-20073 -c \
  'python3 /opt/vllm-moet/qwen38_sm120/moe_gemv/test_fused_moe_integration.py'
docker run --rm --gpus '"device=0"' --ipc host --entrypoint bash vllm-moet-sm120:dsv41-0909 -c \
  'python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_mxfp8_gemv_sm120.py --ms 1,6,16 --shapes decode &&
   python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_wo_a_integration.py'
```

## Serve

```bash
# DeepSeek-V4.1-Flash, GPUs 0-3, port 8001 (first start ~25 min: FlashInfer autotune + DeepGEMM JIT fill CACHE_DIR; later ~17 min)
MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash CACHE_DIR=/srv/cache/ds41 GPU_MEM_UTIL=0.92 docker/sm120/run-dsv41.sh
# second start onwards: GPU_MEM_UTIL=0.94 (1.49M KV tokens); 0.95 OOMs during the autotune sweep on a cold cache

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

Expected on 4× RTX PRO 6000 with P2P: DeepSeek 66–67 steps/s (prose ~2.2, code ~5.2 tok/step),
Qwen 97–99 steps/s (prose ~2.5, code ~3.5 tok/step), needle PASS at both lengths. A first request
after start is slower (warm-up). If steps/s are ~10 % low and the log shows NCCL on `SHM`, P2P is
not available on the host — check IOMMU / ACS settings.

## Memory headroom

DeepSeek at `GPU_MEM_UTIL=0.94` peaks at ~96.0 GB/GPU under C8 with 7.7K-token prompts (1.2 GB from
the wall) — do not raise it. Qwen with `--kv-cache-memory 32GiB` uses ~88 GB/GPU; vLLM's own
start-up estimate would give only 5.6 GiB of KV because the GPU-PLE profile run over-reports its
peak activation (a compile-time transient), hence the explicit KV size. Image/video prompts on
Qwen (it is a VL model) were not stress-tested for memory; drop `KV_CACHE_MEMORY` to 28 GiB if
multimodal traffic OOMs.
