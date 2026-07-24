# The v0.25.0 upgrade candidate

This repository now carries a side-by-side W2 overlay for official vLLM
`v0.25.0`. It is an upgrade candidate, not yet the production default. The
proven v0.24 image, patch, recipes, and benchmark receipts remain intact as the
rollback boundary until the v0.25 candidate passes the SM120 hardware canary.

## Exact source identity

- Official tag: `v0.25.0`
- Official tag commit: `702f4814fe54fabff350d43cb753ae3e47c0c276`
- Production fork: `https://github.com/OmarB97/vllm`
- Production branch: `moet-v0.25.0`
- Production source commit: `6023898a814230ea839107ad82ca0141b71062b6`
- Linux/amd64 base image manifest: `sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97`
- W2 overlay: `patch/vllm-moet-v0.25.0.patch`
- Overlay SHA-256: `ea1e8462008e8d3530e8938483a4f8974258196acc6a0bbcc4124bc4a719ed5d`
- Overlay scope: 61 files, 13,342 insertions, 134 deletions

The production fork branch and the exact source SHA recorded in
`patch/SOURCE-v025.txt` are the rollout authority. Pull requests to another
fork or to official upstream are welcome follow-up contributions, but their
review or merge state never gates this production overlay.

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
- **Tier managers stop before interpreter teardown.** The v0.25 stable-libtorch
  extension can abort if a daemon manager still owns Torch tensors during
  Python shutdown. Each tier now has an explicit stop/join boundary and the
  module registers a deduplicated `atexit` shutdown for serving workers.
- **The v0.24 starvation fix is present in the v0.25 tree.** The source change
  from `96bc1a406d57557a1d1d4f6f8ed3e7b8272ea51f` was adapted to Model Runner
  V2: synchronous recovery gets a third eviction pass that may drop the
  drained step's seen window while retaining step pins and split-FP4 coupling;
  both routing windows close before tier managers wake; and a final fetch with
  no following replay does not pin the pool. The scheduling guard remains
  method-agnostic before draft extraction, covering both n-gram and native MTP
  with `num_speculative_tokens=1`.

## Verification completed before image build

The source port passed:

- `git diff --check` against the exact v0.25 tag;
- Python compilation across every changed Python file;
- 27 passed / 1 skipped focused W2 memory, padded-route, step-pin, and manager
  shutdown tests;
- 6 passed CPU DSpark scheduling and live-re-derivation regressions;
- clean patch application and a committed 61-file lost-line manifest.

These are source gates only. They do **not** establish CUDA kernel, model-load,
quality, context, or throughput parity.

## Bounded SM120 image receipt (2026-07-12)

This receipt belongs to the earlier `25ac6fea...` overlay, not the current
`ea1e8462...` source candidate. It remains bounded evidence for that image. The
structured-output scheduler delta was separately built and canaried on taro.
The complete regenerated overlay then built successfully as
`vllm-moet-sm120:v025-w2candidate-ea1e8462`, image ID `sha256:3acf5a707966`;
its label matches the full overlay hash and its installed scheduler guard
imports successfully. That exact complete image has not been served or
deployed.

The digest-pinned recipe built on taro as
`vllm-moet-sm120:v025-w2candidate-25ac6fea`, local image ID
`sha256:1b3dc4a340a6`. On its RTX 5090 (SM120), the exact image passed:

- stable-libtorch native extension import with zero allocated GPU bytes;
- the baked 22 passed / 1 skipped W2 suite and 6 passed DSpark suite;
- bounded W2/W4 decode (`max_rel` 0.01358 / cosine 0.999911), full-FP4 delta
  (`0.01611` / `0.999906`), and split-FP4 delta (`0.01333` / `0.999922`);
- split three-tier mixed dispatch, base-miss zeroing, coupled eviction, and
  clean interpreter shutdown;
- byte-identical pinned, pack, reboot, tiered arena, eviction, overflow,
  scan-resistance, and preheat store paths;
- baked NVFP4 packed-cache writes and FlashInfer sparse-MLA JIT-cache load.

The first v0.25 candidate exposed the manager teardown abort after its
three-tier assertions passed. The same test exited clean on frozen v0.24; the
explicit stop/join fix then exited clean on corrected v0.25 and is covered by
two CPU regressions. The superseded image tag was removed. Throughout these
bounded checks, taro's live llama-swap Qwen process stayed at 23,114 MiB and
was not restarted or rerouted.

This receipt still does **not** establish a DS4 checkpoint load, 128K context,
quality, or performance result on v0.25.

## DS4 W2 speculative-decoding canary (2026-07-12)

The later `f2989ad1...` candidate image was exercised against the real
DeepSeek-V4-Flash W2 checkpoint on taro with a 100 GiB cgroup, 96 GiB
`memory.high`, FP8 KV cache, 131,072-token model length, and one active
sequence. Results are decode throughput after the first request:

| Mode | Warm median | Change from no-spec |
| --- | ---: | ---: |
| No speculation | 23.56 tok/s | baseline |
| n-gram, 3 tokens | 39.13 tok/s | +66.1% |
| n-gram, 4 tokens | 41.05 tok/s | +74.3% |
| native MTP, 1 token | 32.46 tok/s | +37.8% |

The n-gram 3-token run also completed the exact 120K retrieval canary in
250.3 seconds versus 260.8 seconds without speculation. Warm acceptance was
98.2% for n-gram 3 and 97.8% for n-gram 4. None of the runs recorded cgroup
high, max, OOM, or OOM-kill events.

Native MTP exposed a vLLM 0.25 structured-output failure: consecutive forced
tool calls could commit duplicate or otherwise grammar-invalid target/bonus
blocks, causing xgrammar FSM rejection and HTTP 500. Copying the grammar mask,
fencing its CUDA stream, and validating/rolling back a committed block did not
fix the defect; the rollback variant also corrupted valid arguments. The
correctness containment in this overlay discards drafts for structured-output
requests before scheduling while preserving MTP for other requests.

The contained image completed 20 consecutive forced tool calls with exact
`report_result(ok=true, label="SPEC_OK")` arguments, 20 HTTP 200 responses,
zero grammar rejections, and zero 500s. Ordinary MTP generation remained at a
32.86 tok/s warm median (+39.5% from no-spec). The tool response retained
`finish_reason="stop"`, which is the existing vLLM behavior for named tool
choice; the tool call and arguments were present. After the canary, the
disposable container was removed and taro's 8080, 8081, and 9090 production
health checks all returned 200.

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
