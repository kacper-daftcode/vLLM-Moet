# DeepSeek-V4.1-Flash on sm_120: what the image changes and why

Companion to [dsv41-sm120.md](dsv41-sm120.md) (build, serve, switches). The numbers are from the served
configuration (4× RTX PRO 6000 at 400 W, TP4, DSpark k = 7) unless stated.

## What was missing on sm_120

The official image fails at engine start on sm_120. An instrumented dry run of the stock image (every
sparse-MLA / paged-MQA call logged, the rejected shapes zero-filled so the forward reaches all 40 layers,
DSpark, prefill and vision) gave the inventory:

| gap | stock behaviour | fix |
|---|---|---|
| FlashInfer's SM120 DSv4 decode kernels exist for 64-token pages only | V4.1's SWA cache uses 32-token pages: `no decode kernel for this shape` | decode instantiations for page 32, top-k ∈ {128, 192, 256, 1152} |
| its SM120 DSv4 prefill kernels: page 64 only, SWA rows ≤ 2048, dual-cache extra page 64 | ratio-1 layers (128 states per page) and the 1152-row vision-padded prefill rows rejected | a new translation unit for the V4.1 geometry (`tools/dsv41_sm120/sparse_mla_sm120_dsv41.cu`) — the existing kernel templates instantiated, no kernel body changed |
| the dispatcher never looked at the page size | SWA-only and ratio-2 layers silently ran the page-64 kernel over 32-token pages | an orchestrator hook: wrong page size → the V4.1 table or a loud failure |
| DeepGEMM's SM120 paged MQA logits (the indexer) assert 32/64-row pages | ratio-1 indexer layers assert | the host asserts admit 128 (the device kernels are templated on the page size; bit-exact to 64-row pages) |
| vLLM gates the MXFP4 indexer cache and the FP4 KV record on sm_10x | 132 B/key and 584 B/state FP8 caches: 1.5M GPU KV tokens | both formats — the checkpoint's own — admitted on sm_120 through FlashInfer's DSv4.1 dual cache: 9.6M tokens |
| DSpark `enable_adaptive_verification: true` (the recipe's NVIDIA default) | unsupported on this backend | served `false` (the recipe's AMD setting) |

FlashInfer `main` ships the V4.1 instantiations since [flashinfer#5197](https://github.com/flashinfer-ai/flashinfer/pull/5197);
the vLLM nightly base writes the V4.1 records interleaved ([vllm#61371](https://github.com/vllm-project/vllm/pull/61371)),
which FlashInfer's reader is taught at build time (`tools/dsv41_sm120/flashinfer_interleaved/`). DeepGEMM's SM120
per-group BLOCK_M policy ([vllm-project/DeepGEMM](https://github.com/vllm-project/DeepGEMM) 727f8470) is restored by
`patch_deepgemm.py`: without it every MoE decode batch is padded to 128 rows per expert and the grouped GEMM takes 2×.

## Where the time goes

**Decode step** (single stream, ~7.7 ms, 129 steps/s). Four cards on PCIe without NVLink: ~90 TP collectives per
step, each ~11–13 µs with P2P (18–30 µs over SHM), and every launch gap counts. The official image with only the
fixes it needs to start ran 60 steps/s on the same cards; the served image runs 129 (part of the gain is newer vLLM
bases). The changes behind it:

- **Dense GEMMs on a tensor-core MXFP8 GEMV** (step 3; `tools/dsv41_sm120/sm120_gemv/`): CUTLASS's 128-row tile on a
  6-row batch and `wo_a` as BF16 through cuBLAS bmm were 1.8–3.4× / 2.9× slower. The GEMV family is ALU-issue-bound at
  M = 1 ([sm120-alu-subunits.md](sm120-alu-subunits.md)); programmatic dependent launch and evict-first loads on top.
- **The decode MoE** (steps 17, 19, 22, 27, 33): FlashInfer's b12x W4A8 compact path bound at TP4; the router's top-6 in
  one CUDA kernel bit-identical to Triton's; FC2 grouped by expert; then FC1 → activation → FC2 in one persistent kernel
  with the shared-expert add fused into the top-k sum. All bit-identical to b12x's own kernels.
- **The mHC boundary** (steps 10, 38, 39): the Mega-mHC sublayer boundary (TileLang pair on the model stream, SM100's
  side-stream version is gated) moved off the critical path as a cluster of five CTAs per token, writing the MXFP8 copy
  of the layer input the next linear kernel consumes.
- **Collectives** (steps 13, 23): the decode step's all-reduces as a two-shot push over PCIe P2P with the consumer fused
  into the boundary kernel ([48, 5120] in 23.5 vs 69.8 µs from a graph), summed in NCCL's order so the bits are NCCL's;
  NCCL's per-collective graph event records reduced to one.
- **The drafter** (steps 18, 20, 30, 32, 36): DSpark's `lm_head` on the MXFP8 GEMV, Markov sampling in one launch per
  draft step, the drafter's input preparation and the SWA metadata builds in one launch each.
- **Sparse-MLA decode** (step 28): FlashInfer's gather warp issued one `cp.async.bulk` per candidate row and the TMA
  unit worked them off one by one (~2.5 µs until a tile's bytes land); plain 16-byte loads, a shared fp4 → fp8
  conversion and 4 barriers per tile instead of 20: 10.5 → 7.5 µs per layer, bits identical on 175 mixed-cache shapes.

**Prefill** (163K tokens in 7.2 s, rank 0; the remaining profile): NCCL reduce-scatter / all-gather **36 %** (the MoE
input's all-gather alone 15 %), the MoE **19 %** (0.94 ms per layer-chunk against a 0.83 ms DRAM floor), the
sparse-MLA prefill kernel **13 %** (BF16, one CTA per SM, latency-bound), CUTLASS MXFP8 **9 %**, cuBLAS bf16 4 %,
GPU idle 2 %. The steps that took it from ~15.5 s (the official image) through ~8.5–9 s (newer vLLM bases: the SWA
bounded replay, the indexer's row sharding) to 7.2 s:

- **Step 40** — the V4.1-record prefill rows on sm_120 went through FlashInfer's single-group FP8 kernel (Q, QK, P and
  PV in FP8; H100 / B200 compute the same attention in BF16). The full-BF16 SM120 kernel FlashInfer also ships is 24×
  closer to the fp32 reference on the real records at the same speed. A change of the prefill's arithmetic: GSM8K-200
  194/200 vs 193/200 (one flip, McNemar p = 1), needle 6/6, decode untouched.
- **Step 41** — two `[4096, 5120]` bf16 all-reduces per layer and chunk on the PCIe ring were 29 % of the prefill.
  Sequence-parallel boundaries from 2048 tokens per step: reduce-scatter, the boundary on this rank's row shard
  (bit-identical per row), all-gather of the attention projections (35 % of the bytes) and of the MoE input. −9.4 % at
  163K, −10.5 % at 19K. The bits above the gate follow NCCL's reduce-scatter order; `tools/dsv41_sm120/p2p_rs/`
  reproduces the all-reduce's order exactly at +1 %.
- **Step 42** — b12x's `dynamic` W4A8 kernels (FC1 + SwiGLU + MXFP8 quant, FC2) took 2.6 ms per 4096-token layer-chunk,
  29 % of the prefill, against a 0.83 ms floor set by the expert weight bytes (173 / 102 registers, one wave per SM).
  The same arithmetic — QMMA in b12x's k order, accumulate in C, b12x's activation and quantization ops — in persistent
  CTAs, 8 warps on 8 n16 column slices × every m16 tile of the expert chunk, a cp.async ring, the epilogue in registers:
  1.67 ms per layer, bit-identical with b12x on dumps of a real prefill and end to end (0 of 30,137 prompt positions
  differ, greedy identical), −9.2 % fresh prefill.

## How changes are validated

Three levels, in this order, because a harness win is not an end-to-end win (the b12x dense GEMM backend: −28 % in the
harness, +2.3 % end to end — kept as a negative result in `tools/dsv41_sm120/mxfp8_gemm_b12x/`):

1. **Op level**: every kernel has a test against the path it replaces (`tools/dsv41_sm120/*/test_*.py`, one GPU inside
   the image; cold L2 for the timings). Bit-identical means `torch.equal`.
2. **In situ**: a profile of the served step or of a fresh prefill with the GPU idle share (`tools/sm120_perf/`
   `profile_capture.py`, `trace_*.py`).
3. **End to end**: fresh prefill 19K / 163K, the Fraser bench (median of 40 reps), the BRRR bench; bits by `torch.equal`
   on dumps of a real prefill plus greedy tokens and prompt logprobs of long prompts — always with
   `VLLM_MOET_DET_TOPK=1` in both arms (without it long prompts are not reproducible between runs). A change of the
   prefill's or decode's arithmetic additionally passes GSM8K-200, needle retrieval and greedy agreement.

Every kernel the image ships round-trips through cubit bit for bit (`tools/dsv41_sm120/cubit_gate.py`); the SASS
sources in `kernels/sass/` are canonical for it.

## Open items

- The MoE input's all-gather in MXFP8 instead of bf16 (b12x quantizes the rows anyway) ≈ −3.5 % prefill; then the
  replay rows' `all_gatherv` and the sequence-parallel threshold.
- The sparse-MLA prefill kernel in SASS (2–4 CTAs per SM, raw records staged, dequant at `ldmatrix`): −5…6 % prefill.
- The prefill MoE from 0.94 to its 0.83 ms floor ≈ −2 %.
- The deterministic top-k on long rows at eight streams (one CTA per row today; −3…−4 % aggregate decode at 65K × 8).
- Each request on the disk tier pays ~20 ms more before its first token (cause open).
