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
bounded replay, the indexer's row sharding) to 7.2 s and on to 5.6–5.8 s; the served configuration is at 5.44 s with
step 42's later kernels (−2 %) and the host's CPU governor `performance` (−5.49 %):

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

## Steps in detail

The mechanism of the steps whose row in the [switch table](dsv41-sm120.md#switches) is only a summary, in
step order. What was measured, on what and how:
[benchmarks/dsv41-sm120-steps-2026-11-10.md](benchmarks/dsv41-sm120-steps-2026-11-10.md).

### Step 3b — the prefill dequant in one kernel

In the prefill branch of step 3's `wo_a` (more than 64 tokens) the bf16 weight that cuBLAS needs is
rebuilt from MXFP8 on every layer call: vLLM's `dequant_mxfp8_to_bf16`, six elementwise kernels, ≈ 50 µs
of GPU time and ≈ 160 MB of traffic per call, 60 calls per 8 192-token step.
With `VLLM_MOET_WOA_DEQUANT_FUSED=1` one Triton kernel does the same arithmetic in the same order
(e4m3 → fp32, `exp2` of the scale, one multiply, one rounding to bf16): ≈ 6 µs and 25 MB per call.
`VLLM_MOET_WOA_DEQUANT_CACHE=1` instead keeps the bf16 weight per layer after the first call (no kernel
at all; +16.8 MB per layer = 0.63 GiB per rank).

The fused kernel differs from the six on one input: scale code 0 next to a non-zero value byte (the
descale 2^-127 is an fp32 subnormal that it flushes). At a layer's first call its scales are scanned
once for that input, and such a layer stays on the six kernels (logged and counted), so the switch is
bit-identical on any weights; the served checkpoint carries scale codes 114-118 only.

The runtime key `WOA_DEQUANT_FUSED` (0 / 1) of step 42b's control file overrides the environment value
from the next prefill `o_proj` call on: polled by the module itself at most once per 100 ms, a missing /
malformed file or a value other than 0 / 1 keeps the state (one warning), a key the file drops goes back
to the environment's value, one line per rank per change (`runtime A/B: WOA_DEQUANT_FUSED 0 -> 1`). Per
`touch /dev/shm/vllm_moet_stats` or every `VLLM_MOET_WOA_DEQUANT_STATS_EVERY` calls each rank prints one
line with the calls / weight rows through each path, the layers the guard kept on the six kernels and
`switches now …`.

In situ the switch does not pay, so the default stays off.

### Step 42 — the prefill MoE kernels

The step's default kernels (v8) are described under [Where the time goes](#where-the-time-goes). The
sub-switches replace them stage by stage, with the same bits:

- `_WS2=3` runs FC2 on the producer-warp kernel v9: one warp issues TMA / cp.async, eight consume.
- `_WS1=9` runs FC1 on the warpgroup kernel v10.
- `_SUM_ITEMS=1` runs the top-k sum as work items of the FC2 kernel on a column-tile-major list: a
  tile's routed rows are summed while still in L2 and dropped with `discard.global.L2`
  (`_SUM_DISCARD=1`), so the 252 MB of routed rows per 4 096-row call are no longer written to and read
  back from DRAM. `_SUM_MIN_ROWS=512` keeps the 128/256-row replay layers on the separate sum kernel.

**The occupancy guard.** The grid of these kernels is the SM count times the CTAs per SM that the
occupancy query grants, and FC2's v9 kernel runs two CTAs per SM at exactly the register count that
still allows it (96 per thread): a build in which it needs more — a source change, or another ptxas
after an image rebuild — gets one, half the grid, and FC2 is 10-15 % slower without any error. Each rank
therefore logs one `WARNING: fc2_kernel_ws<4, 1> launches with 1 CTA(s) per SM (grid 188) where its mode
was measured with 2 (grid 376): … registers per thread …` when a kernel of this family starts with
fewer CTAs per SM than its mode was measured with. `VLLM_MOET_B12X_MOE_PREFILL_FAST_OCC_STRICT=1`
refuses such a launch instead (a debug switch); the check changes no launch.

**The m-tile loops.** `_MT_LOOPS=1` (off by default) launches the served FC1 / FC2 kernels with the
consumers' stage loop specialised on the item's m-tiles. The served loops run the QMMAs of an m-tile
under a warp-uniform predicate, and a QMMA under a false predicate keeps its stall and its place in the
tensor pipe's issue throttle (eight warps per SM issue 614 M QMMA slots per second when every slot
executes and 658 M when none does), so an item pays for four m-tiles whatever it has: at 4 096 rows of a
real prefill a row chunk has 2.9 of 4 on average, and on FC1's 64-column tile every warp serves two of
the four. With the switch an item branches once to a loop that holds only its own m-tiles' loads and
QMMAs: the same QMMAs on the same operands in the same order per accumulator, identical on the harness
in every mode. FC2's specialised kernel is built for two CTAs per SM (92 registers per thread; the
warning above covers it as `fc2_kernel_ws_mt<4, 2>`). On one card the loops make the MoE call's kernels
6.9 % shorter. In the served forward they are bit-identical and worth 0.19–0.32 % of a 163K prefill at a
400 W limit, 1.1–1.3 % at 300 W, and 0.84 % at 400 W when the two row groups run one after the other:
the pipelined step is paced by the communication phase that runs beside the MoE, not by the MoE — the
compute stream waits about 0.6 ms after every MoE call for that phase to end, 24 ms of a 239-ms step,
and a shorter MoE lengthens the wait. The arms:
[record](benchmarks/dsv41-sm120-steps-2026-11-10.md#step-42--the-prefill-moe-kernels).

### Step 42b — the runtime keys

The runtime switch between the bit-identical kernel alternatives of step 42, for paired A/B
measurements in one server lifetime.
With `VLLM_MOET_RUNTIME_AB=1` every rank reads the control file — one flat JSON object switch → value,
written from outside with `docker exec <container> sh -c 'printf … > /dev/shm/vllm_moet_ab.json'` — at
most once per 100 ms at a MoE-call boundary and applies `B12X_MOE_PREFILL_FAST_SUM_ITEMS` (0/1, the
sum-items unit), `B12X_MOE_PREFILL_FAST_WS1` (0..11) and `B12X_MOE_PREFILL_FAST_MT_LOOPS` (0/1, the
m-tile loops of step 42) without a restart. A missing / malformed file or an unknown value keeps the
state (one warning), a key the file drops goes back to the environment's value, each change is logged
once per rank.

The file is the whole state of every step that has a key, each key read by its own module:
`WOA_DEQUANT_FUSED` (step 3b) and `SPARSE_MLA_PREFILL_SASS_CUBIN_SEL` (step 46b) live in the same file.
The two states of every key are bit-identical alternatives.

The launcher runs the container with the host's IPC namespace, so the control file is a file of the
host's `/dev/shm` and outlives the server: a file left behind is applied by the next server at its
first prefill, and removing a file keeps the state it last set until a restart. Write `{}` before
removing it. With the switch on the launcher refuses to start while a control file exists in the host's
`/dev/shm` (`RUNTIME_AB_KEEP_FILE=1` starts anyway) — also when the watchdog restarts a server that died
with its file in place. `SERVED=1` turns the switch on with no control file: every key is in its
environment's state (one warning per rank and module that the file is missing).

The activation counters (calls / rows through FC1 v10 vs v8/v9, FC2 with the sum items vs with the
separate sum kernel, and FC1 / FC2 with the m-tile loops) are printed per rank every
`VLLM_MOET_B12X_MOE_PREFILL_FAST_STATS_EVERY` MoE calls or on `touch /dev/shm/vllm_moet_stats`.

### Step 45 — the MoE input gathered as MXFP8

The sequence-parallel prefill's MoE input is all-gathered as e4m3 + UE8M0 scales + the shard's top-6
instead of bf16: 0.52× the bytes. The router GEMM runs with M = T on a stand-in buffer, and the
quantizer equals b12x's.

### Step 46 — the prefill attention in SASS

The step-40 prefill attention chunks (> 64 rows) run on the cubit-assembled SASS kernel
`kernels/sass/sparse_mla_prefill_v41.sass` (generator `kernels/gen/gen_sparse_mla_prefill_v41.py`, cubin
`kernels/cubins-sm120/`): FlashInfer's full-BF16 kernel's arithmetic hand-scheduled (occupancy 16.7 % →
register prefetch of the next 64-candidate chunk), 0.66–0.72× its time on the harness. Chunks of ≤ 64
rows stay on FlashInfer, and any launch problem disables the route after one warning.

**The cubins.** One entry symbol and one launcher for all of them; `kernels/MANIFEST.md` has the status
of each file, [the record](benchmarks/sparse-mla-prefill-cubins-2026-11-11.md) its build, gates and
times.

- `v41`, the default: a stall behind every producer, the MMA chain between writeback drains.
- `v41s1`: the same instruction words under the dependency-aware schedule — a stall is charged where a
  consumer needs it instead of behind every producer (cubit's pair-latency table for sm_120: ptxas's
  practice on 1.8 M instructions, measured on RTX PRO 6000 silicon, enforced on every control-flow
  path), and the HMMA chain runs at ptxas's issue distances instead of between writeback drains.
- `v42`: the dequant of the staged records by sm_120's scaled BF16 unpack,
  `F2FP.BF16.<E4M3|E2M1>.UNPACK_B Rd, Rs[.sel], Rscale.B0` — two codes of `Rs` become two bf16 halves,
  each times 2^(byte 0 of `Rscale` − 127). nvdisasm names the instruction for SM120 and it executes
  there; ptxas does not emit it for this target (0 of the 1.8 M instructions of the sm_120 census). An
  fp8 chunk is an e4m3 pair × UE8M0 block scale in one instruction, the scale byte straight from its
  staging register (`v41`: unpack to f16, 2 × widen to f32, pack to bf16, `HMUL2` by the unpacked
  scale). An FP4 chunk is an e2m1 pair → bf16 with the unit scale code, the four bytes of a word
  through the source's byte selector instead of three shifts, the E4M3 block scale by one more scaled
  unpack, then the `HMUL2.BF16_V2` of `v41`. A task's store data rotates over two register sets, because
  a store latches its data when the memory pipe serves it. Everything else is the text of `v41`.
- `v43`: `v42` with two load waits taken out — the same values, the same arithmetic, the same order;
  loads are issued earlier. `v41` / `v42` load the candidate index of chunk c+1 at the top of the loop
  and store it into `valid[]` at once: warps 0–1 wait for a load of the token's own index row and the
  other six wait for them at the barrier. `v43` issues the load for chunk c+2 behind the gather loads of
  chunk c+1 (after the barrier, branch-free, the `valid[]` threads selected by the load's predicate),
  reads it once in front of the dequant and stores it at the next loop top. And the prologue's two
  length loads and four `q` loads are issued together and stored afterwards, where `v41` / `v42` ran six
  load latencies in a row per token. It is assembled with a rule that gives long loads a scoreboard of
  their own: scoreboards are counters, and under cubit's sharing policy other uses ended up on the
  scoreboard of the gather prefetch (in `v41s1` / `v42` the fifth `LDSM` of QK waits for all sixteen
  gather loads); with the prologue change the gather moved to a scoreboard the first QK instruction
  waits on.
- `v41s1b`, `v43b`: `v41s1` and `v43` with one stall raised, so that the last `HMMA` of PV stands at
  least 14 issue cycles in front of the two conditional branches behind it. The result of a warp-wide
  MMA is to land before an instruction that can change the warp's set of threads, and ptxas never
  issues a conditional branch closer. Both branches are uniform over the CTA (their predicates depend
  on the token's chunk counter only), so the cubins without the distance are bit-identical in every
  gate; these two are the builds that do not rely on that.
- `v43c`: `v43b` with every store of the dequant waited on every path
  ([below](#step-46b--two-cubins-of-the-attention-kernel)).

### Step 46b — two cubins of the attention kernel

The runtime A/B of step 46 between two cubins of the attention kernel — two schedules of the same
instruction words, e.g. `sparse_mla_prefill_v41.cubin` and `sparse_mla_prefill_v41s1.cubin`. With
`VLLM_MOET_RUNTIME_AB=1` and `VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN_B` naming the second cubin, it is
loaded next to the first when a device's launcher is created, and the key
`SPARSE_MLA_PREFILL_SASS_CUBIN_SEL` of the control file (0 = the cubin of `…_SASS_CUBIN`, the start-up
state; 1 = the cubin of `…_SASS_CUBIN_B`) selects the cubin per launch — read by the route itself at the
entry of a launch, at most one `stat` per 100 ms. The key absent = the start-up cubin; a missing /
malformed file, an unknown key of the `SPARSE_MLA_PREFILL_SASS_` namespace or a value other than the
integer 0 / 1 keeps the state (one warning).

Each change is one line per rank (`runtime A/B: SPARSE_MLA_PREFILL_SASS_CUBIN_SEL 0 -> 1`) and, per
`touch /dev/shm/vllm_moet_stats`, one line with the launches / rows per cubin and `switches now …`. At a
device's first launch outside a CUDA-graph capture both cubins run once on the same chunk and their
outputs are compared bit for bit (unequal = the second cubin is refused, one warning); a launch recorded
into a CUDA graph keeps the cubin it was captured with. Without either variable nothing of this runs (no
`stat`, one cubin per device).

Cubins that `…_SASS_CUBIN_B` may name ship in the image under `/opt/vllm-moet/kernels/cubins-sm120/`
(the build asserts each sha256, as for the default cubin): `sparse_mla_prefill_v41s1.cubin`,
`sparse_mla_prefill_v43.cubin` and `sparse_mla_prefill_v43b.cubin` —
[what each is](#step-46--the-prefill-attention-in-sass). All give bit-identical outputs
(`kernels/MANIFEST.md` has their status).

**Which cubin may be selected.** `v42`, `v43` and `v43b` must not be: in the iteration of a token's
last chunk each rewrites the data registers of a shared-memory store before anything has waited on
that store (4 / 3 / 3 places; the shipped cubin and `v41s1` have none). A store reads its registers
when the memory pipe serves it, not when it is issued. In these three the rewriting instruction is a
shared-memory load, which the pipe serves behind the store, so no wrong bit was ever produced — a
property measured on two card models, not a rule of the architecture, and a cubin that rests on it is
never selected. `v43c` is `v43b` with those stores waited (one control word, the same time); it is in
`kernels/cubins-sm120/` and in no image. `SERVED=1` names `v43b` as the second cubin: it is loaded and
compared with the shipped one at each rank's first eager launch, and its key stays 0. The places, the
probe and `v43c`'s checks: [record](benchmarks/sparse-mla-prefill-v43c-2026-11-10.md).

### Step 47 — the pipelined prefill

`VLLM_MOET_PREFILL_PIPE=2` with `--max-num-batched-tokens 8192` runs a step as two 4 096-row groups one
layer apart, the SP reduce-scatters / all-gathers of one group on a comm stream under the other's
compute (NCCL was 38 % of the prefill on one stream). `VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS=1` with
`--compilation-config {"decoder_replay_cudagraph_capture_sizes":[128,256]}` runs the 128-row replay
layers 20–39 of every step as vLLM's decoder-replay CUDA graphs, captured per row-group slot as well
(vLLM captures none under `--max-cudagraph-capture-size 64`). The launcher's `PREFILL_PIPE=2` implies
both.

Without the replay graphs the pipelined step's eager replay tails — ~1 400 launches each — are
launch-bound, and a host whose cores other processes load pays for it at 163K; the KV offload tier is
not what costs there.

The step costs a longer engine step for a concurrent decode, and −1.0 GiB KV per GPU for the graphs on
top of −10 % at 8 192 batched tokens.

The bits are identical: every kernel runs at the row count and on the rows it has without the step.
Per-layer dumps cover the layers outside the graphs; layers 20–39 run inside the graphs and are covered
by the end-to-end gates.

The comm stream runs at the compute stream's priority. A higher one (`VLLM_MOET_PREFILL_PIPE_COMM_PRIORITY=-1`,
or the launch order `VLLM_MOET_PREFILL_PIPE_ORDER=comm2prio`) is refused with a warning while FC2's sum
items of step 42 can be on, and the step runs at priority 0. Every CTA of the prefill MoE's FC1 and FC2
kernels takes its first work item by its block index, and a sum item of FC2 waits, bounded by its
watchdog, for every FC2 item of its column tile: FC1 cannot end, and FC2 makes no progress, while one CTA
of the grid has not been dispatched. A kernel that holds whole SMs when one of the two starts — an NCCL
collective is four CTAs that each need an empty SM — therefore delays the MoE call by its own length.
At equal priority that is all: the work distributor gives freed SMs to the earlier grid's pending CTAs.
Above the compute stream's priority a collective issued later takes them; dispatched in part on the SMs
FC2's last CTAs wait for, it cannot complete itself, and the two wait for each other until the watchdog
traps the launch. When a group of a pipelined step fails, the pipeline logs the MoE's watchdog record if
it is not empty.

### Step 48 — the KV offload copy path

With the host KV tier, each of the four TP workers registers the shared offload region with
`cudaHostRegister` — the whole range in one `pin_user_pages`, all four at the same moment. On a
KVM-guest host that fails for three of them at every start (`host_register failed … the offload region
stays pageable` in the log). A pageable rank's stores are host-synchronous, and when rank 0 — the only
writer of the TP-replicated MLA KV — is one of them, it blocks 211–219 ms per 8 192-token step under
load.

`VLLM_MOET_KV_SWAP_ASYNC=1` takes the KV offload connector's copy path off the runner's main thread: the
four workers' host registration of the shared CPU region runs under a cross-process lock in rank order,
in 2 GiB chunks with retries, and a still-pageable rank's copies are staged through pinned memory on the
connector's own threads (a transfer stream). The connector's save-before-evict protocol is unchanged: a
store completes only after the copy into the region. With the switch on, start-up waits for the
serialized registration (≈ 3.5 min for the last rank at 17 GB).

The image default is off. The reference host registers the offload region on 4/4 ranks, so the step is
a safety net there. Code and its CPU unit test: `tools/dsv41_sm120/kv_offload/`
(`test_kv_swap_async.py`).

### Step 49 — the idle freeze

vLLM freezes each serving process's heap once, when its start-up is done (1.25 M objects in a TP
worker), so that Python's cyclic collector never walks it again; what becomes long-lived afterwards is
not frozen. In a serving process that is:

- in each worker, ≈ 800 objects for every step token count it has not seen before — b12x's fused-MoE
  plans, 20 plan sets of ≈ 40 objects (`TPMoEScratchPlan` with its 25 `_TensorAllocSpec`, `Caps`,
  `TPMoEPlan`, …); a fresh prompt brings one new count (its last step);
- in the EngineCore, one KV-offload index entry (`ChunkStatus`) per stored chunk;
- in the API server, nothing: its old generation does not grow.

Every full collection walks all of it on the thread that happened to allocate, with the interpreter
lock held. As started, a worker runs no full collection at all at first — CPython's rule still
remembers the pre-freeze heap as the old generation's size, and ≈ 312 K promotions must pass — and then
runs them inside requests, in all four ranks within a quarter of a second.

With `VLLM_MOET_GC_IDLE_FREEZE=1` the heap is frozen again whenever a serving process has worked and
gone idle:

- one more `gc.collect()` right after the start-up freeze: it resets the old-generation size that
  CPython's full-collection rule remembers, so that the first automatic full collection does not come
  late and meet everything accumulated until then;
- `gc.collect()` + `gc.freeze()` whenever the process has worked and then been idle for
  `VLLM_MOET_GC_IDLE_FREEZE_IDLE_S` seconds (default 1.0; idle = all its threads together below 2 % of
  one core): a full collection then walks only what the requests since the last idle second left. The
  workers freeze 2 s after the last request, because their main threads spin for the first second.

Independent of the switch, every collection of `VLLM_MOET_GC_WARN_MS` milliseconds or more (default 50)
is logged with its generation, duration and thread (at most one line per 5 s; 0 turns it off). With the
switch on, each serving process logs `gc idle freeze: on` once when its start-up is done and any idle
freeze of 20 ms or more.

Costs: one thread per serving process that wakes four times a second; an idle freeze holds the
interpreter lock for the milliseconds of its collection; an object that is alive at an idle moment and
later dies inside a reference cycle is no longer reclaimed by the collector (as with vLLM's own start-up
freeze). In a worker and in the EngineCore no request is in flight at such a moment, but the API server
is idle while the engine works on a long prompt, so its freezes fall inside such requests (≈ 1 ms, off
the request's path).

## Host memory

The loader pins its tables in host memory on every rank, and PyTorch's pinned host allocator rounds
each request up to a power of two: the tables hold 264.3 GiB of shared memory.
`PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:256`, which `SERVED=1` sets, stops the rounding
above 256 MiB: 189.0 GiB, with the same lookups. The KV offload region (64 GiB as served) comes on
top ([record](benchmarks/dsv41-sm120-r32-2026-11-11.md#host-memory)).

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
