# Changelog

Served images and what changed between them. Image tags are `vllm-moet-sm120:<line>-<base>-rNN`; the
commit an image was built from is in its release tag.

## DeepSeek-V4.1-Flash on 4× RTX PRO 6000

### `dsv41-nightly-df3bd094-r23` — 2026-11-03 (`v2026.11.03-r23`)
- Step 42: the prefill MoE (FC1 + SwiGLU + MXFP8 quant, FC2 of b12x's dynamic path) on our pipelined
  kernels, bit-identical with b12x; fresh prefill 163K 7.8 → 7.2 s (22.7K tok/s).
- Fraser bench unchanged (688 tok/s median, 746 best, 129.0 steps/s over 40 reps); BRRR prefill
  aggregate +11…+28 % over the recipe's stock vLLM.

### `dsv41-nightly-df3bd094-r22` — 2026-11-02
- Step 40: the sparse-MLA prefill on FlashInfer's full-BF16 SM120 kernel (24× closer to fp32 on the real
  records; GSM8K-200 194/200, needle 6/6).
- Step 41: sequence-parallel mHC boundaries for eager prefill steps ≥ 2048 tokens (NCCL reduce-scatter /
  all-gather); fresh prefill −9.4 % at 163K, −10.5 % at 19K.
- Step 26 (deterministic indexer top-k) served on: the same prompt reproduces token for token at any length.

### `dsv41-nightly-df3bd094-r21-mxfp8` — 2026-10-29
- Steps 38–39: the mHC boundary kernel writes the MXFP8 layer input itself; the attention boundary's
  side-stream deferral. Fraser 679 / 740 tok/s at 128.4 steps/s; fresh prefill 163K 9.0 s.

### `dsv41-nightly-df3bd094-r21` — 2026-10-26
- The b12x MoE at TP4 with the fused decode MoE, the P2P all-reduce in NCCL's order, the sparse-MLA gather
  on plain loads, the drafter's launches fused, the mHC boundary as a CTA cluster (steps 17–37); DSpark k = 7;
  the cards at 400 W. 128 steps/s single stream.

### Earlier (2026-09-16 … 10-16)
- The bring-up on the recipe's `deepseekv41-flash-0909` image: the sm_120 gap inventory closed kernel-side
  (page-32 sparse-MLA kernels, DeepGEMM's 128-row pages), then the move to vLLM nightly bases; the MXFP8
  GEMV, the FP4 KV record and the MXFP4 indexer cache on sm_120, single-copy KV offload with a disk tier,
  the mHC boundary off the critical path, the small-M BF16 GEMMs with cuBLAS's bits, Engram's `wkv` split.

## MoET hybrid on 4× RTX 5090

### `dsv41-nightly-df3bd094-r23-hyb` — Dockerfile step 43 (`--build-arg VLLM_MOET_CPU_TIER_PATCH=1`)
- 110 of 384 experts per layer resident (40 static + 70 LRU), the rest computed on the CPU from the pinned expert
  store, bit-identical to the all-resident b12x path; decode ~148 / ~129 tok/s on short prompts, 512K context,
  needle 3/3 at 32K / 64K / 512K, HumanEval 159/164 vs 160/164 native, MATH-500 L5 55/60 vs 55/60.

## Qwen3.8-Flash-Next-FP8 on 4× RTX PRO 6000

### `qwen38-20073` — 2026-09-19
- PLE table on the GPUs, the CuTe-DSL skinny GEMM on sm_120, the FP8 MoE GEMV: 65 → 98.5 steps/s.

## 2-bit experts on vLLM v0.24.0 (frozen)

- 2026-11-03: the DS4 MTP drafter's experts keep the checkpoint's MXFP4 method under an NVFP4 main stack
  (fork `abeba41cb6`; issues #20, #21). `check_patch_files.py` terminates its `git diff` revision lists (#14).
- 2026-09-08: last regeneration before the freeze (fork `2eb26753`: the fp32 confidence-gate signal).
- 2026-08-28: release `v2026.08.28-x4-prefill` — the tables in `docs/v024-2bit.md`.
