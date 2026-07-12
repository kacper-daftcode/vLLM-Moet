# The v0.25.0 upgrade candidate

This repository now carries a side-by-side W2 overlay for official vLLM
`v0.25.0`. It is an upgrade candidate, not yet the production default. The
proven v0.24 image, patch, recipes, and benchmark receipts remain intact as the
rollback boundary until the v0.25 candidate passes the SM120 hardware canary.

## Exact source identity

- Official tag: `v0.25.0`
- Official tag commit: `702f4814fe54fabff350d43cb753ae3e47c0c276`
- Linux/amd64 base image manifest: `sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97`
- W2 overlay: `patch/vllm-moet-v0.25.0.patch`
- Overlay SHA-256: `9ebd246059592ce2966f63854785f4c98f7c75f4f00d940a351902207a8e0072`
- Overlay scope: 60 files, 12,976 insertions, 133 deletions

Apply it directly to an official checkout with:

```bash
git clone --branch v0.25.0 https://github.com/vllm-project/vllm && cd vllm
git apply --check /path/to/vLLM-Moet/patch/vllm-moet-v0.25.0.patch
git apply /path/to/vLLM-Moet/patch/vllm-moet-v0.25.0.patch
```

Or build the pinned serving image:

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-v025 \
  -t vllm-moet-sm120:v025-w2candidate .
```

## What v0.25 absorbed

The port was produced by applying the frozen v0.24 overlay to `v0.24.0`, then
rebasing that exact tree onto `v0.25.0` and resolving conflicts against the new
Model Runner V2 paths. Ten old overlay files disappeared because v0.25 now owns
their behavior, including the core DSpark/DFlash model registrations, DeepSeek
V4 DSpark implementation, Gumbel sampling, and SM120 cooperative-top-k guard.
The v0.25 overlay therefore drops those redundant hunks instead of shadowing
upstream.

Exact paths removed from the overlay:

```text
vllm/model_executor/models/qwen3_dflash.py
vllm/model_executor/models/registry.py
vllm/models/deepseek_v4/__init__.py
vllm/models/deepseek_v4/nvidia/dspark.py
vllm/models/deepseek_v4/nvidia/model.py
vllm/transformers_utils/configs/speculators/algos.py
vllm/v1/worker/gpu/sample/gumbel.py
vllm/v1/worker/gpu/spec_decode/__init__.py
vllm/v1/worker/gpu/spec_decode/dspark/__init__.py
vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py
```

The retained delta is the project-specific W2 stack: 2-bit planes, FP4
recovery and confidence gate, tiered/NVMe expert stores, persistent pack-cache
safety, SM120 cubins, NVFP4 KV, pipeline-aware replay, and the optional
hardware-aware DSpark confidence scheduler.

## Compatibility decisions

- **Model Runner V2 stays enabled.** The port preserves v0.25's new default and
  composes W2 padded-slot, prefill, replay, Mamba-preprocess, and graph metadata
  with it. It does not restore a V1/PagedAttention escape hatch.
- **DeepGEMM uses the release copy.** v0.25 already vendors exact commit
  `a6b593d2826719dcf4892609af7b84ee23aaf32a`, the same SM120-capable commit the
  v0.24 recipe built separately. The v0.25 Dockerfile removes that duplicate
  wheel build.
- **FlashInfer remains 0.6.14 temporarily.** Official v0.25 pins 0.6.13, while
  the W2 NVFP4 sparse-MLA source patch and JIT kwargs were hardware-validated on
  0.6.14. The candidate preserves the proven pair and makes that deviation
  explicit. Qualifying 0.6.13 is a separate canary, not an assumption.
- **SM120 raw FP8 scales remain.** The v0.25 release still uses the SM100 packed
  scale recipe in the DeepSeek V4 output projection. Consumer Blackwell needs
  the raw row-major scale layout carried by this overlay.
- **The DSpark extensions remain optional.** v0.25 supplies the core DSpark
  engine; the overlay adds per-request confidence widths, profiled cost tables,
  online calibration, hysteresis, and live dynamic-SD re-derivation.

## Verification completed before image build

The source port passed:

- `git diff --check` against the exact v0.25 tag;
- Python compilation across every changed Python file;
- 20 passed / 1 skipped focused W2 memory, padded-route, and step-pin tests;
- 6 passed CPU DSpark scheduling and live-re-derivation regressions;
- clean patch application and a committed 60-file lost-line manifest.

These are source gates only. They do **not** establish CUDA kernel, model-load,
quality, context, or throughput parity.

## Promotion gates

The v0.25 candidate must stay side-by-side with the live v0.24 image. Promotion
requires, in order:

1. build the pinned image on an SM120 host and run the baked import/compile
   checks;
2. run CUDA op tests for W2/W4 cubins, raw-scale output projection, NVFP4 cache
   write/read, and CUDA-graph capture;
3. cold-start a disposable DS4 canary without replacing the live router lane;
4. prove the 128K serve configuration, exact retrieval, frozen-rule quality,
   memory/cgroup safety, and no pack corruption on the real endpoint;
5. compare decode, prefill, MTP acceptance, replay rate, and memory receipts to
   the frozen v0.24 baseline;
6. only then move the router lane, retaining the v0.24 image and packs for an
   immediate rollback.

No existing seed or benchmark receipt is relabeled as v0.25 evidence. The
upgrade reuses the test definitions, but the candidate must earn its own
runtime receipts because the execution engine and dependency base changed.
