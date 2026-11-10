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
bounded replay, the indexer's row sharding) to 7.2 s, and on to 5.6–5.8 s:

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
- **Steps 45–47** — from 7.2 s to 5.6–5.8 s on the reference host (5.4 s on the second host): the MoE input all-gathered as
  e4m3 + UE8M0 scales + the shard's top-6 instead of bf16 (0.52× the bytes; step 45), the prefill attention chunks on the
  cubit-assembled SASS kernel (0.66–0.72× FlashInfer's time; step 46), and the pipelined step (step 47): at 8 192 batched
  tokens a step is two 4 096-row groups one layer apart, the reduce-scatters / all-gathers of one group on a comm stream
  under the other group's compute, the 128-row replay layers 20–39 as CUDA graphs captured per row-group slot (the eager
  tails are ~1 400 launches each and launch-bound under host CPU load). The profile of one 4 096-row layer-chunk of the
  pipelined step (rank 0): 49 kernels on two streams, 3.55 ms on the compute stream and 3.36 ms on the comm stream for
  7.95 ms of GPU span; the two groups interleave to a period of 9.66 ms per layer against ~7.2–7.3 ms of compute per layer
  pair (compute stream busy 75 %, comm stream 72 %); the host issues 304 runtime calls per layer per group thread = 6.4 ms
  of CPU for 7.95 ms of GPU. Per layer-chunk: the two reduce-scatters 1 850 µs and the all-gathers 1 036 µs (NCCL, 91 %
  of the comm stream), attention 938, fc1 935, fc2 488, CUTLASS MXFP8 GEMMs 466 (our kernels are 71 % of the compute
  stream). The 26 % between the period and the compute is the cost of running the collectives as NCCL kernels next to the
  compute — CTA dispatch and NCCL's per-communicator FIFO — not bandwidth: a synthetic model of the layer pair with the
  collectives issued from a persistent kernel of 32–48 CTAs recovers 57–68 % of it and costs the compute kernels ≤ 1.1 %.

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

- The pipelined step's collectives from the device instead of NCCL (`tools/dsv41_sm120/sp_pipeline/moet_device_comm*`,
  `VLLM_MOET_PREFILL_PIPE_DEVICE_COMM`, off; not yet an image step): one 32-thread kernel instance per collective pushes
  this rank's rows over PCIe P2P into the peers' buffers, phased per peer, and 128 reducer CTAs sum the landed pieces in
  NCCL's per-element order from a table probed on the communicator — NCCL's bits by construction (4-rank unit test 20/20
  `torch.equal`, a 2-group × 40-phase interleave 0 mismatches), reduce-scatter 896–944 µs against NCCL's 909 alone; every
  device-side wait is `%globaltimer`-bounded with a pinned error record. The in-situ gain against the 9.66-ms period is
  the open measurement; the expected whole is −11…−13 % at 163K.
- A static SM split for the pipelined step (`VLLM_MOET_PREFILL_PIPE_GREEN=N`, off): the comm stream on a CUDA green
  context of N SMs, the compute stream on the other 188 − N, so that the boundary chain and the compute kernels stop
  queueing for the same SMs. On a model of the layer pair (two streams + NCCL) 10.1 → 8.5–8.7 ms per pair at N = 32 and
  10.0 → 8.5 under a 300 W cap, bits identical; the served kernels pay the SM share on 156 SMs — attention +19 %, `wo_a`
  +24 %, the MXFP8 GEMMs +7…+10 %, fc1 +10 %, fc2 +9 % with their grids capped to the partition (a grid sized for the
  whole device runs into FC2's watchdog there). In the served forward, which is denser than the model, the overlap does
  not win that back: with the split switched per forward inside one server life (N = 32, 16 + 16 alternating units per
  power cap) a 163K-token prefill is 0.7 % slower at a 400 W and at a 300 W cap alike — 0.3 % in the units that repeat
  their state, the rest being the cost of the first request after a change of the key — and the energy per
  8 192-token step is not lower (+1.6 % and +0.5 %), bits identical. The switch stays off.
- The reduce-scatter in fewer bytes (fp8 with a per-row scale, 1 850 → ≈ 950 µs per layer-chunk): a change of the
  prefill's arithmetic, quality-gated.
- The prefill MoE's distance to its DRAM floor: FC1 and FC2 in one persistent launch
  (`VLLM_MOET_B12X_MOE_PREFILL_FAST_WS` ≥ 20) is bit-identical and +2.6 % — FC2 is bound by the expert weights and the
  routed rows it reads (W2 plus ~252 MB of rows per layer-chunk), not by the launches; the lever is bytes.
- The deterministic top-k on long rows at eight streams (one CTA per row today; −3…−4 % aggregate decode at 65K × 8).
