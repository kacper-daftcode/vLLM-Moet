# Changelog

Released images and what changed between them. Image tags are `vllm-moet-sm120:<line>-<base>-rNN`; the
commit an image was built from stands in its entry's heading where it is on record — as a commit, or as
the release tag that points at it.

## DeepSeek-V4.1-Flash on 4× RTX PRO 6000

### In the tree after `-r32` (in no released image)
- The m-tile loops of step 42 (`VLLM_MOET_B12X_MOE_PREFILL_FAST_MT_LOOPS` and its runtime key), off by default.
- `sparse_mla_prefill_v43c.cubin` — `v43b` with every dequant store waited on every path — in `kernels/cubins-sm120/`
  (no Dockerfile step copies it), and the store path check `tools/dsv41_sm120/sparse_mla_sass/store_war_paths.py`.

### `dsv41-nightly-df3bd094-r32` — 2026-11-11 (commit `3e3d519`)
`-r28`'s kernels and switches bit for bit, plus the following — every switch off by default in the image; `SERVED=1`
sets the three marked *served*:
- Step 42b, the runtime keys (`VLLM_MOET_RUNTIME_AB`, *served* with no control file): every rank reads a control file
  and switches between bit-identical alternatives without a restart — step 42's sum items and FC1 kernel, step 3b's
  dequant, step 46b's cubin — and prints activation counters per rank on request.
- Step 3b: the prefill dequant of `wo_a` in one kernel (`VLLM_MOET_WOA_DEQUANT_FUSED`).
- Step 46b: a second cubin of the attention kernel loaded next to the shipped one and selected per launch by a
  runtime key (`VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN_B`, *served*: `v43b` loaded, never selected). The image ships
  three more cubins under `/opt/vllm-moet/kernels/cubins-sm120/` (the build asserts each sha256, as for the default
  cubin): the second schedule `sparse_mla_prefill_v41s1.cubin`, `sparse_mla_prefill_v43.cubin` and
  `sparse_mla_prefill_v43b.cubin`. `v43` and `v43b` must not be selected (stores whose data registers are rewritten
  without a wait; `docs/dsv41-sm120-internals.md`, step 46b).
- Step 42's occupancy guard: one warning when a kernel of the family starts with fewer CTAs per SM than its mode was
  measured with (`VLLM_MOET_B12X_MOE_PREFILL_FAST_OCC_STRICT=1` refuses the launch).
- Step 49, the idle freeze (`VLLM_MOET_GC_IDLE_FREEZE`, *served*), and a log line for every garbage collection of
  50 ms or more.
- `SERVED=1` also sets `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:256`: the loader's pinned tables take 189.0
  instead of 264.3 GiB of host memory.
- Measured at the change of the served image: fresh prefill 163K 5.44 s, 19K 0.66 s and decode 128.0 steps/s on runs
  with no other request in flight; bits identical to `-r28`; in the serving server `-r28`'s step-42 kernels take
  2.24 % ± 0.20 % less time at 163K than the set before them — [record](docs/benchmarks/dsv41-sm120-r32-2026-11-11.md).
- `-r29`, `-r30` and `-r31` were builds on the way and were not served: `-r29` brought the runtime keys of step 42,
  `-r30` steps 3b and 46b with `sparse_mla_prefill_v41s1.cubin`, `-r31` the occupancy guard, step 49 and
  `sparse_mla_prefill_v43.cubin`; `-r32` adds `sparse_mla_prefill_v43b.cubin`.

### `dsv41-nightly-df3bd094-r28` — 2026-11-10 (commit `7e2b74f`)
- Step 42, two more of its stages on new kernels (`SERVED=1` sets the switches), bits identical: FC1 on the warpgroup
  kernel v10 (`_WS1=9`: −12 % of FC1's kernel time per step), and the top-k sum as work items of the FC2 kernel, the
  routed rows summed while still in L2 and dropped from it instead of written to DRAM and read back (`_SUM_ITEMS=1
  _SUM_DISCARD=1 _SUM_ROUTE_HINT=1 _SUM_TOKENS=128 _SUM_MIN_ROWS=512`: fc2 + sum 777 → 585 µs per call, the step's DRAM
  −12…−17 %, −2 % at 163K and −1.5…−3 % energy per 8 192-token step at 400 W, paired).
- Step 48 (the KV offload connector's copy path off the runner's main thread, `VLLM_MOET_KV_SWAP_ASYNC`) is in the
  image, off by default.

### `dsv41-nightly-df3bd094-r26b` — 2026-11-08
- Step 47: the pipelined sequence-parallel prefill (`PREFILL_PIPE=2`: two 4 096-row groups of an 8 192-token step one
  layer apart, the collectives of one group under the other's compute) with the groups' 128-row replay layers as CUDA
  graphs per slot, and FC2 of step 42 on the producer-warp kernel (`_WS2=3`); fresh prefill 163K 6.6 → 5.6–5.8 s
  (28–29K tok/s) at 400 W (5.4 s on the second host), 19K 0.79 → 0.76 s, bit-identical to `-r25`; +1.7 % under 100
  nice'd CPU hogs; decode unchanged (129 steps/s); GPU KV 9.62M → 6.12M tokens (the 8 192-token step and the graphs
  cost 2 GiB per GPU). `-r26` — the same without the replay graphs — is superseded: its eager replay tails are
  launch-bound and a host whose cores other processes load pays +31 % at 163K.

### `dsv41-nightly-df3bd094-r25` — 2026-11-05
- Step 46: the sparse-MLA prefill attention chunks on the cubit-assembled SASS kernel; fresh prefill 163K 6.8 → 6.6 s,
  bit-identical to `-r24`.

### `dsv41-nightly-df3bd094-r24` — 2026-11-05
- Step 45 on by default: the sequence-parallel prefill's MoE input all-gathered as MXFP8 (0.52× the bytes); fresh
  prefill 163K 7.1 → 6.8 s, bits identical.

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

## MoET hybrid on 1–4× RTX 5090

### `dsv41-nightly-df3bd094-r23-hyb-engram` — 2026-11-04, Dockerfile steps 43 + 44 (`--build-arg VLLM_MOET_CPU_TIER_PATCH=1 --build-arg VLLM_MOET_ENGRAM_V2_PATCH=1`)
- Step 44, Engram v2: the Engram tables stay on NVMe (io_uring + a DRAM row cache, vLLM's own lookup kernel on a device
  landing buffer, bit-identical rows) — `ENGRAM_MODE=nvme`, −189 GiB of host RAM at the same speed (278 GiB used).
- The expert store's NVMe tail (`STORE_NVME=1 STORE_DRAM_RECORDS=N|auto`): only N records per layer per rank pinned, the
  cold ones read on a miss; 236 GiB at 138 / 120 tok/s (256 GB hosts), 112 GiB at 112 / 82 (128 GB hosts); `auto` sizes N
  from the host. The full LRU on 384 GB hosts (`STORE_MODE=all RESIDENT=0 SPARE=110`): 307 GiB, 147 / 125, +3.6 % on mixed prompts.
- TP2 / TP1 through virtual ranks (step 43's runtime, in both tags): 2× RTX 5090 46 / 47 tok/s, 1× RTX 5090 40 / 34, each
  also with the RAM tiers; identity 0 mismatches at every TP. The default `docker run` at TP4 is unchanged.

### `dsv41-nightly-df3bd094-r23-hyb` — 2026-11-04, Dockerfile step 43 (`--build-arg VLLM_MOET_CPU_TIER_PATCH=1`)
- The CPU expert tier on the `-r23` image: 110 of 384 experts per layer resident per rank (40 static + 70 LRU), the rest
  computed on the CPU from the pinned expert store inside the captured decode graph, bit-identical to the all-resident
  b12x path (19,080 compact calls per rank, 0 mismatches); decode 145 / 125 tok/s on short prompts, 84 on mixed prompts
  over 30 min, 94 on the BRRR code bench; prefill 4.6–4.7K tok/s from 8K to 512K; 512K context (needle 3/3 to ~512K);
  HumanEval 160/164 vs 160/164 native, MATH-500 L5 55/60 vs 55/60; 457 GiB of host RAM; 13.8 J per token.

## Ripple-65B Preview on 4× RTX PRO 6000

### `ripple-nightly-a2bcb565-r6` — 2026-11-06
- Step 3: two-microbatch decode for TP without DP (`DECODE_UBATCH=2`, off by default): +4 % at 16 requests, +5 % at 32.
- The derived checkpoint with the attention projections in MXFP8 (`tools/ripple_sm120/make_mxfp8_attn_checkpoint.py`,
  an option: a numerics change): +15 / +9 / +8 % at 1 / 8 / 32 requests over NVFP4.

### `ripple-nightly-a2bcb565-r4` — 2026-11-06
- Step 2: the DFlash2 drafter drafts the schedule's k (1 + k rows per request, graphs per width); +6.5 % at 32 requests.

### `ripple-nightly-a2bcb565-r3` — 2026-11-05
- Steps 0–1: vLLM `03bd3c68` with the Ripple architecture on nightly `a2bcb565`; the TP all-reduces of ≤ 128 rows on the
  PCIe P2P push (+14 % at 8 requests); the dynamic draft length 15 / 11 / 7 by batch size. Code 600 / 3050 / 4740 tok/s at
  1 / 8 / 32 requests against stock vLLM's 416 / 2270 at static k 7.

## Qwen3.8-Flash-Next-FP8 on 4× RTX PRO 6000

### `qwen38-20073` — 2026-09-19
- PLE table on the GPUs, the CuTe-DSL skinny GEMM on sm_120, the FP8 MoE GEMV: 65 → 98.5 steps/s.

## 2-bit experts on vLLM v0.24.0 (frozen)

- 2026-11-03: the DS4 MTP drafter's experts keep the checkpoint's MXFP4 method under an NVFP4 main stack
  (fork `abeba41cb6`; issues #20, #21). `check_patch_files.py` terminates its `git diff` revision lists (#14).
- 2026-09-08: last regeneration before the freeze (fork `2eb26753`: the fp32 confidence-gate signal).
- 2026-08-28: release `v2026.08.28-x4-prefill` — the tables in `docs/v024-2bit.md`.
