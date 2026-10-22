# Serving DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next-FP8 on 4× RTX PRO 6000 (sm_120)

Two self-contained serving images built from the official vLLM images for these models, with
the vLLM-Moet sm_120 fixes and decode kernels baked in. Everything a new host needs is in this
repository plus the model checkpoints; nothing is bind-mounted from a host-specific directory.

| model | image | Dockerfile | launcher | single-stream decode (TP4, greedy) |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash (official MXFP4/MXFP8 checkpoint, vision on) | `vllm-moet-sm120:dsv41-nightly-df3bd094-r13` (served since 2026-10-22: vLLM nightly df3bd094 + steps 17–28, fused decode MoE and sparse-MLA gather on, DSpark k=7; before it `-r11` (2026-10-21), then `dsv41-nightly-20260923-engram` (served since 2026-10-02 18:23Z with `VLLM_MOET_BF16_GEMM_SITES=indexer,wk,router`; `-bf16` = the same without the Engram wkv split, served 2026-10-01..02; `-mhc` = also without the small-M BF16 GEMMs, served 2026-09-25..10-01; `-moeqs` = also without the mHC boundary off the critical path and vllm#57679, `-ced` = also without the fused MoE quant+scatter, `-kvdedup` = also without the CED prefill, `dsv41-nightly-20260923` = also one offloaded KV copy per rank, `dsv41-0909` = rollback) | `Dockerfile.sm120-dsv41-nightly` (`Dockerfile.sm120-dsv41` for the 0909 image) | `docker/sm120/run-dsv41.sh` | **641 tok/s median, 686 best-of-10, 118 steps/s on the Fraser bench (DSpark k=7, `-r13`)**; earlier: 80.7–83.4 / 82.8–83.2 steps/s (prose / code), 169–195 / 420–433 tok/s (DSpark k=5; `-bf16`: 80.2–81.8 / 81.9–82.1), fresh prefill 18k tok/s (CED), 3.40M-token FP4 KV at 512K context + KV offload (64 GiB host RAM = 76.7M tokens, disk tier; since 2026-09-23) (0909 image: 67 steps/s, 152 / 360 tok/s, 3.06M tokens) |
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
| base image, DeepSeek (served 2026-09-23 .. 10-13) | Docker Hub `vllm/vllm-openai:nightly@sha256:42090442…` (vLLM main 0961bbae, torch 2.13+cu130) + FlashInfer nightly wheels `0.7.0.dev20260922` (`flashinfer.ai/whl/nightly`) + `vllm-project/DeepGEMM` e1f418c2 | digest / version pins in `Dockerfile.sm120-dsv41-nightly` at f7cc6d9 |
| base image, DeepSeek (served 2026-10-13 .. 10-16) | Docker Hub `vllm/vllm-openai:nightly-21d93d0d…@sha256:bcee860c…` (vLLM main 21d93d0d = 0.30.1rc1.dev709, the same torch 2.13+cu130 build) + FlashInfer `0.7.1rc2` (python, cubin, jit-cache shim and the `sm120f` AOT provider from `flashinfer.ai/whl`, sha256-pinned) + `vllm-project/DeepGEMM` e1f418c2 | digest / version / hash pins in `Dockerfile.sm120-dsv41-nightly` at 59a0136 |
| base image, DeepSeek (served 2026-10-16, windows 10-11) | Docker Hub `vllm/vllm-openai:nightly-3f1bcfd82b…@sha256:cb9bc64f…` (vLLM main 3f1bcfd82b = 0.32.0rc1.dev190, torch 2.14+cu130, FlashInfer 0.7.1 with all its AOT providers) + `vllm-project/DeepGEMM` 6ce1c5ec; the last nightly before vllm#61371, whose interleaved V4.1 KV records FlashInfer's SM120 reader does not read (on a later base step 5 patches the reader: `tools/dsv41_sm120/flashinfer_interleaved/`) | digest / commit pins in `Dockerfile.sm120-dsv41-nightly` |
| base image, DeepSeek (served since 2026-10-16, window 12) | Docker Hub `vllm/vllm-openai:nightly-49cb2bbf…@sha256:a8d84c55…` (vLLM main 49cb2bbf = 0.32.0rc1.dev281, torch 2.14+cu130, FlashInfer 0.7.2 with all its AOT providers; its SM120 DSv4.1 reader patched for the interleaved V4.1 records vLLM writes since vllm#61371, `tools/dsv41_sm120/flashinfer_interleaved/`) + `vllm-project/DeepGEMM` a7a92773 | digest / commit pins in `Dockerfile.sm120-dsv41-nightly` |
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
# the SM120 per-group BLOCK_M restored (docs/dsv41-sm120-port.md, "The vLLM main line as a candidate image") + one host copy
# of the TP-replicated KV in the offload tiers ("KV outside HBM") + the CED prefill ("Decoder SWA bounded replay";
# --build-arg VLLM_PR_58132=0 leaves it out) + the MoE input quantization fused with the DeepGEMM permutation at decode
# shapes ("MoE glue in one launch") + vllm#57679 (the fused query norm + quantization; --build-arg VLLM_PR_57679=0 leaves
# it out) + the mHC sublayer boundary off the decode critical path ("The mHC boundary off the critical path";
# --build-arg VLLM_MOET_MHC_OVERLAP=0 leaves it out) + the small-M BF16 GEMMs with cuBLAS's bits from one launch
# ("The small-M BF16 GEMMs with cuBLAS's bits"; --build-arg VLLM_MOET_BF16_GEMM=0 leaves it out = the -mhc tag).
# Images built before the commit that served it default to VLLM_MOET_BF16_GEMM_SITES=indexer,wk,compressor:
# pass -e VLLM_MOET_BF16_GEMM_SITES=indexer,wk,router to those
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly --build-arg VLLM_MOET_ENGRAM_WKV_TP=0 \
  -t vllm-moet-sm120:dsv41-nightly-20260923-bf16 .   # ~12 min
# DeepSeek, served 2026-10-02 .. 10-13: the same + Engram's wkv split over the TP ranks at decode token counts
# ("The Engram wkv split at decode"; Dockerfile step 12); the line above is the -bf16 tag served before it
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly -t vllm-moet-sm120:dsv41-nightly-20260923-engram .
# (the two lines above are the tags as built from their commits - -engram from f7cc6d9)
# DeepSeek, built on the same base from c066be9, never served (the fallback of window 8): + PDL in our decode kernels,
# NCCL's per-collective graph event records reduced to one, FC2's padding rows out of L2 ("The idle GPU between the
# decode kernels"; Dockerfile steps 13, 14)
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly -t vllm-moet-sm120:dsv41-nightly-20260923-pdl .
# DeepSeek, served 2026-10-13 .. 10-16 (window 8; commit e4de0d9): the same patches on vLLM nightly 21d93d0d + FlashInfer
# 0.7.1rc2 with its sm120f AOT modules ("The rebase onto vLLM nightly 21d93d0d"); steps 7 and 9 are vLLM's own there.
# Built from this file at 59a0136 or before (its base, FlashInfer and DeepGEMM pins); -side (6aafbb6) ran window 9.
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly -t vllm-moet-sm120:dsv41-nightly-21d93d0d-pdl .   # ~15 min
# DeepSeek, served 2026-10-16 (windows 10-11; commit 4e73df6): the same patches on vLLM nightly 3f1bcfd82b (torch 2.14,
# FlashInfer 0.7.1 from the base, DeepGEMM 6ce1c5ec), the DSpark drafter's mHC side stream (step 10;
# --build-arg VLLM_MOET_DSPARK_MHC_STREAM=0 leaves it out), the opt-in side-stream FC2 discard (step 14), the indexer
# row-sharding threshold (step 16); the prefill DBO (step 15) does not fit this base ("The rebase onto vLLM nightly 3f1bcfd82b")
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly -t vllm-moet-sm120:dsv41-nightly-3f1bcfd8-r2 .   # ~10 min
# DeepSeek, served image since 2026-10-16 21:35Z (window 12; commit 206c353): the same patches on vLLM nightly 49cb2bbf
# (FlashInfer 0.7.2 taught the interleaved V4.1 records in step 5, DeepGEMM a7a92773 on the stable ABI); the image label
# com.vllm-moet.v41-kv-records=interleaved sends the launcher's disk tier to $KV_OFFLOAD_FS_DIR/v41-interleaved
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly -t vllm-moet-sm120:dsv41-nightly-49cb2bbf-r1 .   # ~10 min
# DeepSeek, served image since 2026-10-22 (-r13): the same line on vLLM nightly df3bd094 (FlashInfer 0.7.2 reading the
# interleaved V4.1 records in its DSv4.1 dual cache, DeepGEMM 727f8470), plus Dockerfile steps 17-28: b12x compact MoE
# at TP4, the DSpark drafter's and the target's lm_head on the MXFP8 GEMV, the router's top-6 in one CUDA kernel, DSpark's
# Markov sampling in one launch, b12x FC2 grouped by expert, the two-shot P2P all-reduce with its consumer fused into the
# mHC boundary (in NCCL's summation order since -r12), the deterministic indexer top-k, the fused decode MoE (step 27)
# and the sparse-MLA candidate rows by plain loads (step 28). As served: VLLM_MOET_B12X_MOE_FUSED=1,
# VLLM_MOET_SPARSE_MLA_LDG=1, VLLM_MOET_TARGET_LM_HEAD_MXFP8=1, VLLM_MOET_P2P_AR=1 (+ fused consumer to 16 rows),
# DSpark k=7. Fraser bench on 4x RTX PRO 6000 (325 W, memory +5000): median 641 tok/s, best-of-10 686, 118 steps/s.
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly -t vllm-moet-sm120:dsv41-nightly-df3bd094-r13 .   # ~10 min
# DeepSeek, rollback image: the recipe's 0909 image + the same fixes on its own FlashInfer/DeepGEMM pins
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41  -t vllm-moet-sm120:dsv41-0909  .   # ~5 min after the base pull (DeepGEMM _C rebuild + FlashInfer JIT precompile)
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-qwen38 -t vllm-moet-sm120:qwen38-20073 .   # ~4 min (MoE GEMV extension compile)
```

The launcher picks the KV plumbing from the image label `com.vllm-moet.kv-mode` (`KV_MODE=auto`): the 0909 image
keeps `--kv-cache-dtype fp8` + `VLLM_MOET_KV_RECORD`, the nightly image maps `KV_RECORD=nvfp4` to
`--kv-cache-dtype nvfp4_ds_mla` (FlashInfer's own reader of the record) and `fp8_ds_mla` to `--kv-cache-dtype fp8_ds_mla`.

`KV_OFFLOAD_GIB=N` (vLLM-main image only) adds `--kv-offloading-size N`: evicted KV of the compressed
cache and the indexer is kept in N GiB of pinned host RAM (one `/dev/shm` region; the container runs
with `--ipc host`) and restored on a prefix hit instead of being recomputed — a 221K-token context
comes back in 0.6 s instead of a 22 s prefill, decode and prefill unchanged. The image keeps one
host copy of that KV for all TP ranks (it is the same bytes on every rank): 896 B of host RAM per
token, so 64 GiB hold 76.7M tokens (the plain vLLM-main build stores one copy per rank, 3.6 KB per
token at TP4). `KV_OFFLOAD_FS_DIR=DIR` adds a disk tier behind it: every offloaded block is also
written to DIR as a file named by its content hash (112 KiB per 128 tokens), read back on a hit that
misses RAM (a 221K-token context in 1.6 s from a virtio disk) and still valid after a restart (174K
tokens in 1.5 s instead of a 16.5 s prefill). vLLM never deletes the files — run `docker/sm120/kvcache-ttl.sh DIR`
hourly from cron (files unread for 72 h, then the least recently read above 200 GB; `TTL_HOURS` /
`MAX_GB`). With `INDEXER_KV_DTYPE=fp8` the launcher uses `DIR/indexer-fp8`: vLLM's directory name
does not cover the indexer format, and a file of the other format would load as garbage. The
host-RAM region is not removed when the server stops (vLLM's default shutdown kills the engine
before its workers clean up), so the launcher deletes regions no process maps before it starts.
`KV_OFFLOAD_PROMPT_ONLY=0` offloads generated tokens too, so the next turn of an agent conversation
also hits the previous answer. The served deployment runs all three (64 GiB, a disk directory, 0) and
keeps disk files for 7 days (`TTL_HOURS=168` in the cron line: with 72 h a weekend without requests emptied the tier);
`docs/dsv41-sm120-port.md`, "KV outside HBM".

The containers run with `--restart no` (maintenance windows stop and replace them), so a crash stays down until
someone looks: on 2026-10-14 the host's OOM killer took the production worker while a second DeepSeek server was
starting on the other GPUs (each holds ~330 GB of host shared memory — the loader's `psm_*` segments plus the offload
region; one DeepSeek-V4.1 server per host), and the service was down for 14 hours. Since 2026-10-15
`docker/sm120/ds41-watchdog.sh` runs from cron every minute: a container that is not running is logged, its log and
`docker inspect` saved as `<name>-crash-<UTC>.*`, renamed `<name>-crash-<UTC>` and started again (`ds41-flash` with the
wrapper, `qwen38` with `docker start`); a running one that fails `GET /health` three minutes in a row after its
35-minute startup grace is stopped and replaced the same way; at most one restart per 30 minutes and three per six
hours per container, then it only logs "NEEDS A HUMAN". `touch /mnt/k3x/ds41flash/MAINTENANCE` before stopping anything
by hand (`w8_stop.sh ds41-flash` does), `rm` it after the window; while it exists the watchdog does nothing but remind.
Log: `/mnt/k3x/logs/watchdog.log` (a healthy minute writes nothing).

Images built on 21d93d0d after 2026-10-15 carry an experiment switch (not the 3f1bcfd82b images: step 15 does not fit
that base yet), `PREFILL_DBO=1` (`--enable-dbo`): a prefill step of ≥ 1024
tokens runs as two microbatches whose TP all-reduces overlap each other's compute. Leave it off: on this host the
step is paced by the Python that launches each microbatch (measured 2.4× slower on a 1/5-scale dummy-weight model;
`docs/dsv41-sm120-port.md`, "Dual-batch overlap of the prefill all-reduces").

The 3f1bcfd82b images take `INDEXER_TP_SHARD_MIN_ROWS=N` (step 16, the vllm#61647 backport): from N prefill rows in
a step the sparse indexer scores each rank's share of the rows and the ranks exchange the top-k (vllm#54951). Unset,
vLLM's own 65,536 never fires with 4096-token chunks, and a full chunk holds 4,092 rows (DSpark keeps 4 draft slots
per request out of the budget), so N has to stay below that. Production runs 2048 since window 11 (2026-10-16): fresh
prefill of 391K tokens −4 %, 163K −0.5 %, the outputs identical, decode and KV unchanged; `INDEXER_TP_SHARD_MIN_ROWS=
./run.sh` turns it off.

The image also runs the report's CED prefill: the layers after the last KV-source layer run on each
prompt's last 128 tokens only (vllm#58132, not merged yet, applied as a patch). Fresh prefill of long
prompts is ~1.7× faster (19K / 163K / 391K tokens: 1.63 / 1.68 / 1.75×), decode unchanged, GPU KV
−1.2 %, and the quality gate passed (`docs/dsv41-sm120-port.md`, "Decoder SWA bounded replay"); it
changes what a prefill computes, so it has a switch: `EXTRA_DOCKER_ARGS="-e VLLM_MOET_DECODER_REPLAY=0"`
(same image) or the `-kvdedup` tag (`--build-arg VLLM_PR_58132=0`). On the 21d93d0d image the replay is vLLM's own
(vllm#58132 merged, `--swa-bounded-replay` default on); `EXTRA_ARGS="--no-swa-bounded-replay"` turns it off and also
takes the SWA KV back into prefix caching (the upstream flag covers both sides of the replay; a decoder-only switch
like `VLLM_MOET_DECODER_REPLAY=0` does not exist there).

All bases are pinned (`vllm/vllm-openai:nightly-3f1bcfd82b…@sha256:cb9bc64f…` with its own FlashInfer 0.7.1, before it
`vllm/vllm-openai:nightly-21d93d0d…@sha256:bcee860c…` + FlashInfer `0.7.1rc2` and
`vllm/vllm-openai:nightly@sha256:42090442…` + FlashInfer `0.7.0.dev20260922`, `vllm/vllm-openai:deepseekv41-flash-0909`,
`vllm/vllm-openai@sha256:fc120ece…` = the `qwen38-flash-next` nightly, v0.1.dev20073); the patchers are anchored on
those exact files and refuse to apply to anything else. To move a host without rebuilding, `docker save` /
`docker load` the tags (~30–40 GB each).

FlashInfer packaging: since 0.7.0 `flashinfer-jit-cache` is a 4 kB shim that requires per-architecture provider
wheels (`flashinfer-jit-cache-sm120f` carries the sm_120 / sm_121 modules: `sparse_mla_sm120`,
`mxfp8_gemm_cutlass_sm120`, `sampling`, … 562 in 0.7.1rc2). The 0961bbae images (`dsv41-nightly-20260923*`)
installed the shim with `--no-deps`, i.e. without any provider: FlashInfer JIT-compiled its sm_120 modules at the
first start into the launcher's `CACHE_DIR/dot-cache` (that is the ~25 min first start; losing that directory means
compiling again). The 21d93d0d image installs the sm120f provider and removes the base image's other providers, so
FlashInfer loads its modules AOT (checked on an RTX PRO 6000 with `FLASHINFER_DISABLE_JIT=1`); the 3f1bcfd82b base
ships FlashInfer 0.7.1 with every provider already (vLLM installs them since vllm#58765). Still compiled at
runtime: FlashInfer's autotuner delay kernel `trtllm_utils` (not in FlashInfer's sm_120 AOT list; 4 s, only while
autotuning, i.e. the first start of a FlashInfer version), the CuTe-DSL MXFP8 activation quantizer (cutlass-dsl,
in-process), and the Triton / TileLang / DeepGEMM kernels.

Run the in-image tests once per build (one GPU, ~2 min each):

```bash
docker run --rm --gpus '"device=0"' --ipc host --entrypoint bash vllm-moet-sm120:qwen38-20073 -c \
  'python3 /opt/vllm-moet/qwen38_sm120/moe_gemv/test_fused_moe_integration.py'
docker run --rm --gpus '"device=0"' --ipc host --entrypoint bash vllm-moet-sm120:dsv41-nightly-20260923-bf16 -c \
  'python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_mxfp8_gemv_sm120.py --ms 1,6,16 --shapes decode &&
   python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_wo_a_integration.py &&
   python3 /opt/vllm-moet/dsv41_sm120/test_deepgemm_sm120_paged_mqa.py --packed-stride &&
   python3 /opt/vllm-moet/dsv41_sm120/test_indexer_fp4_sm120.py &&
   python3 /opt/vllm-moet/dsv41_sm120/test_offload_replicated_mla.py &&
   python3 /opt/vllm-moet/dsv41_sm120/moe_quant_scatter/test_moe_quant_scatter_sm120.py &&
   python3 /opt/vllm-moet/dsv41_sm120/moe_quant_scatter/test_moe_quant_scatter_integration.py &&
   python3 /opt/vllm-moet/dsv41_sm120/test_query_quant_gate.py &&
   python3 /opt/vllm-moet/dsv41_sm120/mhc_overlap/test_mhc_post_norm_sm120.py &&
   python3 /opt/vllm-moet/dsv41_sm120/mhc_overlap/test_mhc_overlap_integration.py &&
   python3 /opt/vllm-moet/dsv41_sm120/bf16_gemm/test_bf16_replica_gemm_sm120.py &&
   python3 /opt/vllm-moet/dsv41_sm120/bf16_gemm/test_bf16_gemm_integration.py &&
   cd /opt/vllm-moet/dsv41_sm120/decoder_replay && python3 -m pytest tests/models -q --noconftest'
# the -engram image in addition (one GPU, then four):
docker run --rm --gpus '"device=0"' --ipc host --entrypoint bash vllm-moet-sm120:dsv41-nightly-20260923-engram -c \
  'python3 /opt/vllm-moet/dsv41_sm120/test_engram_wkv_tp.py'
docker run --rm --gpus '"device=0,1,2,3"' --ipc host -e NCCL_P2P_LEVEL=SYS --entrypoint bash \
  vllm-moet-sm120:dsv41-nightly-20260923-engram -c 'torchrun --nproc_per_node=4 /opt/vllm-moet/dsv41_sm120/test_engram_wkv_tp4.py'
# the -pdl candidate in addition: tools/dsv41_sm120/README.md, "On the -pdl tag" (one GPU, then four)
# the 21d93d0d image: the same list without the decoder_replay pytest (vLLM's own code there) plus
#   test_gemv_pdl_sm120.py, moe_l2/test_moe_l2_discard_sm120.py, nccl_graph/test_nccl_graph_records_tp4.py,
#   test_reasoning_effort_encoding.py --model-dir /model, nvfp4_kv/test_flashinfer_dsv41_fp4_extra.py --quick --no-perf
#   (FlashInfer's AOT sparse MLA; FLASHINFER_DISABLE_JIT=1 proves nothing compiles), then on four GPUs
#   torchrun --nproc_per_node=4 nccl_graph/test_nccl_graph_records_tp4.py, nccl_graph/test_nccl_graph_mixing_tp4.py,
#   test_engram_wkv_tp4.py --checkpoint /model (each with /model mounted for the checkpoint tests).
# images built after 2026-10-14 (drafter mHC stream; side-stream FC2 discard opt-in) in addition, one GPU:
#   mhc_overlap/test_dspark_mhc_stream.py --bench (moe_l2/test_moe_l2_discard_sm120.py covers both discard modes).
#   Next to a serving job: --max-gpu-mem-gib 2.2 for the Engram tests, --chain-experts 64 / --experts 64 for MoE
#   quant+scatter, --experts 256 for the FC2 discard test, and no test_deepgemm_sm120_paged_mqa.py (~9 GB).
```

## Serve

```bash
# DeepSeek-V4.1-Flash, GPUs 0-3, port 8001 (first start ~25 min: FlashInfer autotune + DeepGEMM JIT fill CACHE_DIR; later ~7 min)
MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017 CACHE_DIR=/srv/cache/ds41 GPU_MEM_UTIL=0.92 docker/sm120/run-dsv41.sh
# second start onwards: GPU_MEM_UTIL=0.94 (9.6M GPU KV tokens on the df3bd094 image with the defaults
# KV_RECORD=nvfp4 + INDEXER_KV_DTYPE=mxfp4 at 512K context, DSpark on); 0.95 OOMs during the autotune sweep on a cold cache

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
`VLLM_MOET_DECODER_REPLAY=0` (0961bbae images: the CED prefill off, layers 21-39 on every prompt token again; on the
21d93d0d image `EXTRA_ARGS="--no-swa-bounded-replay"` instead, which turns off both sides of the replay),
`VLLM_MOET_MOE_QUANT_SCATTER=0` (MoE input quantization back in prepare(), vLLM's permute kernels),
`VLLM_MOET_MHC_OVERLAP=0` (the mHC boundary back to the TileLang pair on the model stream, all-reduce
back in the linear layers), `VLLM_MOET_MHC_FUSE_ALLREDUCE=0` / `VLLM_MOET_MHC_PDL=0` /
`VLLM_MOET_MHC_PROJ=fused|tf32` (the boundary's sub-options, see the port doc),
`VLLM_MOET_BF16_GEMM=0` (`-bf16` image: the small-M BF16 GEMMs back on cuBLAS; `VLLM_MOET_BF16_GEMM_SITES`
picks the sites, served `indexer,wk,router`, `compressor` opt-in),
`VLLM_MOET_ENGRAM_WKV_TP=0` (`-engram` image: Engram's wkv replicated at every token count again;
`VLLM_MOET_ENGRAM_WKV_TP_MAX_TOKENS`, default 64, is the largest token count that splits),
`VLLM_MOET_GEMV_PDL=0` / `VLLM_MOET_GEMV_EVICT_FIRST=0` (`-pdl` image: the MXFP8 GEMVs launched and loading as before),
`VLLM_MOET_BF16_GEMM_PDL=0` / `VLLM_MOET_MOE_QUANT_SCATTER_PDL=0` (the router / MoE quant+scatter without PDL; the BF16
switch also takes a site list, default `router`), `VLLM_MOET_BF16_CHAIN=v2` (opt-in: the indexer's one-slice chain with
the activations in the shared-memory ring), `VLLM_MOET_NCCL_GRAPH_RECORDS=0` (graphs captured with all of NCCL's event
records, as before), `VLLM_MOET_MOE_FC2_DISCARD=0` (FC2's padding rows stay in L2; `_MIN_TOKENS` / `_MAX_TOKENS`, 2 / 64;
`VLLM_MOET_MOE_FC2_DISCARD_SIDE=1`, images built after 2026-10-14: the discard on a side stream joined after the gather
instead of inline — opt-in, slower in the served graph of window 9),
`VLLM_MOET_SM120_MOE_GEMV=0` (Qwen MoE GEMV → Triton), `VLLM_MOET_SM120_MOE_GEMV_FUSE_ACT=0`,
`VLLM_MOET_SM120_LL_GEMM=0` (Qwen skinny GEMM → cuBLAS; changes the compiled graph, so use a fresh
`CACHE_DIR`), `VLLM_PLE_CPU_OFFLOAD=1` (Qwen PLE table in a CPU worker; then `KV_CACHE_MEMORY=`),
`VLLM_MOET_B12X_MOE_FUSED=0` (step 27 back to b12x's three kernels; bit-identical either way, served on),
`VLLM_MOET_SPARSE_MLA_LDG=0` (step 28 back to FlashInfer's cp.async.bulk gather; bit-identical either way, served on),
`VLLM_MOET_DET_TOPK=1` (step 26: the deterministic indexer top-k; served off),
`VLLM_MOET_TARGET_LM_HEAD_MXFP8=0` (step 21: the target lm_head back to the BF16 GEMM; changes logits, served on),
`VLLM_MOET_P2P_AR=0` / `VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=0` (steps 23/24: the TP all-reduces back to NCCL; the P2P
sum follows NCCL's order, so the bits are NCCL's either way).

## Validate a new host

```bash
curl -s 127.0.0.1:8001/health; curl -s 127.0.0.1:8000/health
python3 tools/sm120_perf/decode_bench.py http://127.0.0.1:8001 deepseek-ai/DeepSeek-V4.1-Flash '{"thinking": false}'
python3 tools/sm120_perf/decode_bench.py http://127.0.0.1:8000 Qwen3.8-Flash-Next '{"enable_thinking": false}'
python3 tools/sm120_perf/needle_any.py http://127.0.0.1:8000 Qwen3.8-Flash-Next '{"chat_template_kwargs": {"enable_thinking": false}}' 30000,100000
python3 tools/sm120_perf/needle_any.py http://127.0.0.1:8001 deepseek-ai/DeepSeek-V4.1-Flash '{"chat_template_kwargs": {"thinking": false}}' 30000,100000
python3 tools/sm120_perf/fraser_bench.py --base http://127.0.0.1:8001 --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --prompt-file tools/sm120_perf/fraser_prompt.txt --reps 10
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
docker run --rm --entrypoint python3 -v /srv/models/DeepSeek-V4.1-Flash:/model:ro vllm-moet-sm120:dsv41-nightly-20260923-mhc \
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
