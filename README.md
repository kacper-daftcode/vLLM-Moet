# vLLM-Moet: frontier MoE models on consumer/workstation Blackwell (SM120)

vLLM-Moet serves frontier Mixture-of-Experts models on RTX PRO 6000 and RTX 5090 (sm_120), where the
official vLLM images do not run them or the official checkpoints do not fit. Two lines of work:

- **Official checkpoints on official vLLM images**: the sm_120 gaps closed and the decode step rebuilt
  around small tensor-core kernels. DeepSeek-V4.1-Flash, served this way on 4× RTX PRO 6000 (the
  active line), gets 1.5× the decode and 2.5× the GPU KV tokens of the official image with only the
  fixes it needs to start; Qwen3.8-Flash-Next-FP8 decodes 1.5× faster.
- **2-bit experts on vLLM v0.24.0**: GLM-5.2 (753B), DeepSeek-V4-Flash (159B) and Kimi-K2.7-Code (1T)
  on one to four cards that cannot hold their checkpoints.

## DeepSeek-V4.1-Flash on 4× RTX PRO 6000

The official checkpoint ([`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017):
552B plus a 196B Engram table, MXFP8 dense, MXFP4 experts, 510 GB) on four RTX PRO 6000 (96 GB, PCIe):
TP4, 512K context, DSpark speculative decoding (k = 7), vision, thinking, tool calling, KV offload to
host RAM and disk. Served image: `vllm-moet-sm120:dsv41-nightly-df3bd094-r23` (the cards at 400 W; served
since 2026-11-03 20:20Z), i.e. vLLM nightly df3bd094 (FlashInfer 0.7.2, DeepGEMM 727f8470) plus Dockerfile
steps 17–42 in [`tools/dsv41_sm120/`](tools/dsv41_sm120/) (b12x MoE at TP4, the fused decode MoE, the
deterministic indexer top-k, P2P all-reduce in NCCL's summation order, the sparse-MLA gather with its XV
loads on `ldmatrix`, the mHC boundary kernel as a CTA cluster, the drafter's input prep, the SWA builders'
decode build and the drafter's SWA update in one launch each, the DSA candidate blocks on the
deterministic top-k, the shared-expert add fused into the MoE top-k sum, the boundary kernel's MXFP8
copy of the layer input — most behind runtime switches; as served: the fused MoE, the sparse-MLA gather
and steps 26, 30–33, 36, 38–42 on). The prefill front of the last week: `-r22` (steps 40–41, served
2026-11-02) moved the sparse-MLA prefill onto FlashInfer's full-BF16 kernel and made the eager prefill's
mHC boundaries sequence parallel — a quality-gated change of the prefill's arithmetic (GSM8K-200 194/200,
needle 6/6, decode untouched); `-r23` (step 42, served 2026-11-03) runs the prefill MoE — FC1 + SwiGLU +
MXFP8 quant and FC2 of b12x's dynamic path — on our own pipelined CUDA kernels, **bit-identical with b12x**
(`torch.equal` on dumps of a real prefill, greedy tokens and prompt logprobs of long prompts identical end
to end). Together: **a fresh 163K-token prefill takes 7.2 s (22.7K tok/s) on production against 9.0 s on
`-r21-mxfp8` a week earlier (−20 %) and ~15.5 s on the recipe's image**; 19K tokens 0.86 s against 1.07 s
([`tools/dsv41_sm120/README.md`](tools/dsv41_sm120/README.md), steps 40–42). Since `-r22` the
deterministic indexer top-k (step 26) is served on as well: the same prompt reproduces token for token at
every length (+0.5–1 % prefill time).

| | measured on the served image (`-r23` = `-r21-mxfp8` + steps 40–42 + the deterministic top-k of step 26, DSpark k = 7, 400 W) |
|---|---|
| decode, 1 stream | prose **~293 tok/s median (waves 277–301)**, code **~802 tok/s median (waves 786–813)** (128–130 steps/s on short prompts, 2026-10-28 on `-r21-mxfp8`; decode is untouched by steps 40–42: Fraser ×40 on `-r23` 129.0 steps/s, 688 tok/s median, 746 best) |
| decode, 8 streams | prose ~721 tok/s median (716–759), code ~2,258 tok/s median (2,227–2,289) aggregate, 8 × 1,024 tokens (2026-10-28) |
| fresh prefill | **163K tokens in 7.17–7.21 s (22.6–22.8K tok/s), 19K in 0.86 s** (production, 2026-11-03 20:20Z, right after the switch; `-r22` the same evening 7.82 s / 0.94 s, `-r21-mxfp8` on 2026-10-29 9.0 s / 1.07 s, 391K in 22.6 s — not re-measured) |
| KV cache | **9.62M tokens** on the GPUs (18× concurrency at 512K), a 64 GiB host offload region, plus a disk tier |
| quality | GSM8K-200 (no-think, 4 streams) **194/200** on `-r22` vs 193/200 on `-r21-mxfp8` (one flip, exact McNemar p = 1); needle PASS at 29K / 106K / 400K with the same answers; tool calls / JSON / vision pass; `-r23` is bit-identical with `-r22` (long prompts: 0 of 30,137 prompt positions differ, greedy identical); identical requests return identical logits at any batch size and, with step 26 on, at any prompt length (2026-10-30 / 11-02 / 11-03) |

### What we changed against official vLLM

| | official vLLM image on sm_120 | vLLM-Moet image |
|---|---|---|
| indexer pages | DeepGEMM's SM120 paged MQA logits admit 32/64-row pages, V4.1's ratio-1 layers use 128: the indexer asserts | the image's DeepGEMM build admits 128-row pages (bit-exact to 64-row pages), with the per-group BLOCK_M restored on sm_120 |
| KV formats | fp8 compressed record (584 B/state) and FP8 indexer cache (132 B/key); the checkpoint's FP4 record and the MXFP4 indexer are SM100-only | FP4 record (288 B) read by FlashInfer's DSv4.1 dual cache (its reader taught the interleaved V4.1 records) and the MXFP4 indexer (68 B/key) enabled on sm_120: **9.63M GPU KV tokens** at 512K context |
| KV offload | one host copy per TP rank, 3.6 KB per token | one copy, 896 B per token |
| dense decode GEMMs | CUTLASS's 128-row tile on a 6-row batch; `wo_a` as BF16 weights through cuBLAS bmm | tensor-core MXFP8 GEMV: 1.8–3.4× faster, `wo_a` 2.9×; the target's `lm_head` on the same GEMV (runtime switch, served on) |
| MoE at decode | DeepGEMM grouped MoE: every routed expert padded to 128 rows (grouped GEMM ~2× slower); the b12x W4A8 backend does not bind at TP4 | b12x W4A8 compact bound at TP4; FC2 grouped by expert (bit-identical, 27.6 → 22.6 µs/layer on recorded routing); the whole decode MoE — FC1 → activation → FC2 — in one persistent kernel (bit-identical, runtime switch, served on); the router's top-6 in one plain-launched CUDA kernel, bit-identical to Triton's (−136 µs on the served step's critical path) |
| DSpark drafter | the drafter's `lm_head` as a BF16 GEMM; Markov sampling in several launches per draft step | the drafter's `lm_head` on the MXFP8 GEMV; Markov sampling in one launch per draft step (−60 µs of drafter graph per decode step) |
| sparse MLA decode | FlashInfer's gather warp issues one `cp.async.bulk` per candidate row and the TMA unit works them off one by one (~2.5 µs until a tile's bytes land) | the same FlashInfer kernel with candidate rows gathered by plain 16-byte loads and the merge's loads hoisted (bit-identical; runtime switch, served on): decode 10.5 → 8.2 µs, merge 2.6 → 2.1 µs per layer at the served shape |
| TP collectives | an NCCL all-reduce per sublayer boundary; in graphs each NCCL kernel starts ~2.8 µs after its predecessor | the decode step's all-reduces as a two-shot push over PCIe P2P, the C1 consumer fused into the mHC boundary kernel ([48, 5120] in 23.5 vs 69.8 µs from a graph), summed in NCCL's own order so the bits are NCCL's; NCCL's graph event records reduced to one |
| indexer top-k (optional) | vLLM's sm_120 top-k kernels break FP4-logit ties differently from run to run, and sparse MLA's bits follow the selection order | an exact deterministic top-k (ties to the lower column, ascending output; runtime switch, served off) — the same prompt then reproduces token-for-token |
| mHC boundary | on the model stream (the side-stream version is SM100-only) | off the critical path: +3–5 % steps/s |
| launch gaps | the indexer's `weights_proj` on 2 cuBLAS CTAs (the main stream waits ~26 µs per layer) | one launch with cuBLAS's bits; 0.4 µs; programmatic dependent launch in our kernels |
| prefill | the indexer's TP row sharding starts at 65,536 rows, which a 4096-token chunk never reaches; Engram's `wkv` all-gathers on every chunk (+1.8 ms per layer) | sharding from 2048 rows (vllm#61647, −4 % at 391K); `wkv` split over the ranks at decode only |
| prefill attention (`-r22`) | the V4.1-record prefill rows on sm_120 go through FlashInfer's single-group FP8 kernel (Q, QK, P and PV in FP8; H100 / B200 compute the same attention in BF16) | the full-BF16 SM120 kernel FlashInfer also ships for this storage (both caches dequantized exactly, BF16 QK with fp32 accumulation, fp32 softmax and PV): 24× closer to the fp32 reference on the real records at the same speed (±8 %); runtime switch |
| prefill collectives (`-r22`) | two `[4096, 5120]` bf16 all-reduces per layer and chunk on the PCIe ring, 29 % of a 163K prefill's GPU time | sequence-parallel mHC boundaries from 2048 tokens per step: reduce-scatter, the boundary on this rank's row shard (bit-identical per row), all-gather of the attention projections (35 % of the bytes) and of the MoE input; fresh prefill −9.4 % at 163K, −10.5 % at 19K; the bits above the gate follow NCCL's reduce-scatter order (`p2p_rs/` reproduces the all-reduce's order exactly, +1 %); runtime switch |
| prefill MoE (`-r23`) | b12x's `dynamic` W4A8 kernels (FC1 + SwiGLU + MXFP8 quant, FC2): 2.6 ms per full layer-chunk of 4096 tokens — 29 % of a 163K prefill — against a 0.83 ms floor set by the expert weight bytes (173 / 102 registers, one wave per SM, 17 % occupancy; DeepGEMM does the same shapes in 1.0 ms) | the same arithmetic (QMMA in b12x's k order, accumulate in C, b12x's activation and quantization ops) in our pipelined kernels: persistent CTAs, 8 warps on 8 n16 column slices × every m16 tile of the expert chunk, a cp.async ring, the epilogue in registers, padding to 16 rows (the median expert has 34 rows) — **bit-identical with b12x** on dumps of a real prefill and end to end; 1.67 ms per layer on the dump, fresh prefill −9.2 % at 163K in situ (−8.1 % on production the night of the switch); the extension's cubin round-trips strictly through [cubit](https://github.com/kacper-daftcode/cubit); runtime switch |
| `reasoning_effort` | `minimal` / `medium` answer HTTP 400 | accepted (25 / 62) |

Two earlier items are upstream in the current base (vLLM nightly df3bd094) and are no longer differences:
the SWA bounded replay and the fused query-norm + quantization gate; the image labels them `upstream`.

Net on the same four cards, against the recipe's official image with only the fixes it needed to start
(2026-09-17): single-stream decode **60 → 88–90 steps/s (1.5×)**, GPU KV **1.49M → 3.69M tokens
(2.5×)**, quality unchanged. Most of the decode gain and 2× of the KV come from the changes above; newer
vLLM bases added the rest and the first part of the prefill speed-up (163K tokens: ~15.5 → 8.5 s); the prefill front
of steps 40–42 took it to 7.2 s. The changes
are bit-identical to the paths they replace, except the dense GEMVs (within one bf16 rounding of the
fp32 reference) and the two KV formats, which passed a quality gate (GSM8K-200, needle retrieval,
greedy agreement). Most have a runtime switch (`VLLM_MOET_*=0`, [docs/sm120-deploy.md](docs/sm120-deploy.md)).

### Build and serve

Needs 4× RTX PRO 6000 (an RTX 5090 does not fit the model at TP4), Docker with the NVIDIA container
toolkit, GPU peer-to-peer over PCIe, and ~330 GB of host shared memory per server (the loader's segments
plus the 64 GiB offload region), so one server per host; on a host with less RAM (a 354 GB VM with four
PRO 6000 Server Edition cards OOM-killed a worker while it populated the offload region) start with
`KV_OFFLOAD_GIB=0`, which costs nothing below the GPU KV capacity.

```bash
git clone https://github.com/kacper-daftcode/vLLM-Moet && cd vLLM-Moet
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir /srv/models/DeepSeek-V4.1-Flash-1017
export IMAGE=vllm-moet-sm120:dsv41-nightly-df3bd094-r23 \
  MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017 CACHE_DIR=/srv/cache/ds41
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly -t $IMAGE .   # ~11 min

# first start, ~22 min: FlashInfer autotunes into CACHE_DIR (on an empty cache the sweep OOMs at 0.95)
GPU_MEM_UTIL=0.92 docker/sm120/run-dsv41.sh
# later starts, as served: 0.94, KV offload to RAM + disk, indexer sharding from 2048 rows, the fused
# decode MoE and the sparse-MLA gather on, DSpark k = 7 on the b12x MoE backend (target and drafter);
# the last four switches are the prefill front (BF16 sparse-MLA prefill, sequence-parallel boundaries,
# the prefill MoE on our kernels) and the deterministic indexer top-k
docker stop -t 60 ds41-flash
KV_OFFLOAD_GIB=64 KV_OFFLOAD_FS_DIR=/srv/kvcache KV_OFFLOAD_PROMPT_ONLY=0 INDEXER_TP_SHARD_MIN_ROWS=2048 \
  EXTRA_DOCKER_ARGS="-e VLLM_MOET_B12X_MOE_FUSED=1 -e VLLM_MOET_SPARSE_MLA_LDG=1 -e VLLM_MOET_TARGET_LM_HEAD_MXFP8=1 -e VLLM_MOET_P2P_AR=1 -e VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=8 -e VLLM_MOET_DFLASH_PREP_GROUPS=1 -e VLLM_MOET_CANDIDATE_DET_TOPK=1 -e VLLM_MOET_SWA_DECODE_BUILD=1 -e VLLM_MOET_MOE_TOPKSUM_ADD=1 -e VLLM_MOET_SWA_DRAFT_BUILD=1 -e B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1 -e VLLM_MOET_MHC_MXFP8=1 -e VLLM_MOET_MHC_SIDE_DEFER=1 -e VLLM_MOET_SPARSE_MLA_PREFILL_BF16=1 -e VLLM_MOET_PREFILL_SP=1 -e VLLM_MOET_B12X_MOE_PREFILL_FAST=1 -e VLLM_MOET_DET_TOPK=1" \
  EXTRA_ARGS='--moe-backend b12x --speculative-config {"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false,"moe_backend":"b12x"}' \
  docker/sm120/run-dsv41.sh
```

The OpenAI-compatible API listens on `127.0.0.1:8001` (`BIND=` publishes it; put an authenticating
proxy in front). Every knob is an environment variable documented in the header of
[`docker/sm120/run-dsv41.sh`](docker/sm120/run-dsv41.sh). vLLM never deletes disk-tier files, so run
[`docker/sm120/kvcache-ttl.sh`](docker/sm120/kvcache-ttl.sh) from cron;
[`docker/sm120/ds41-watchdog.sh`](docker/sm120/ds41-watchdog.sh) restarts a crashed server. Validate with
`tools/sm120_perf/spec_matrix.py` (88–90 steps/s single stream) and `tools/sm120_perf/needle_any.py`.

Known limits: with the disk tier, each request on the 49cb2bbf base pays ~20 ms more before its first
token (cause open); TP8 with a 1M window on eight cards was validated at the bring-up, not on this image.

### Fraser bench: single-stream code decode

One fixed 2,678-token code-generation prompt, temperature 0, 2,048 output tokens, one stream; each rep
starts from a fresh prefill (random cache-bust nonce). The bench reports completion tokens over the
first-to-last-token interval, the engine's steps/s and DSpark's accepted length from its spec-decode
counters. The median is the number we engineer against: per-run acceptance — hence tok/s —
moves by several percent between runs on identical hardware (the serving path is not bit-deterministic
by default), so the best mostly samples that spread. We quote both, because community speed tests
quote the best.

| served configuration (4× RTX PRO 6000, 400 W, GDDR7 +5000) | median tok/s | best | steps/s | accepted length |
|---|---|---|---|---|
| `-r21-mxfp8`, fused MoE + sparse-MLA gather + steps 30–33, 36, 38–39 on, `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1`, `P2P_AR_FUSED_MAX_ROWS=8`, DSpark k = 7 (40 reps, 2026-10-29) | **679** | **740** | **128.4** | 5.29 |
| `-r22` = the row above + steps 40–41 + the deterministic top-k (10 reps on production before the `-r23` switch, 2026-11-03) | 679 | 745 | 129.9 | 5.23 |
| `-r23` = `-r22` + step 42 (10 reps on production right after the switch, 2026-11-03) | 677 | 708 | 129.3 | 5.25 |
| `-r23`, the same server an hour later (40 reps, 2026-11-03 21:10Z; TTFT median 160 ms) | **688** | **746** | **129.0** | 5.37 |

k = 7 steps are individually slower (more rows verified per step) but accept ~19 % more tokens each;
in mixed traffic with lower draft acceptance the wider draft does not pay for itself, which is what
`enable_adaptive_verification` exists for. The prefill front (steps 40–42) leaves this bench where it was:
the decode graphs, the replay rows and the drafter keep the old code; the best-of-N column is the maximum of the
per-run acceptance (0.68–0.69 behind the 745 bests, 0.64 behind the 708 of the ten-rep snapshot), which is why we
engineer against the median and quote 40 reps. Reproduce (the switches are production's):

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly -t vllm-moet-sm120:dsv41-nightly-df3bd094-r23 .
MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017 GPU_MEM_UTIL=0.94 KV_OFFLOAD_GIB=64 \
  KV_OFFLOAD_FS_DIR=/srv/kvcache KV_OFFLOAD_PROMPT_ONLY=0 INDEXER_TP_SHARD_MIN_ROWS=2048 \
  EXTRA_DOCKER_ARGS="-e VLLM_MOET_B12X_MOE_FUSED=1 -e VLLM_MOET_SPARSE_MLA_LDG=1 -e VLLM_MOET_TARGET_LM_HEAD_MXFP8=1 -e VLLM_MOET_P2P_AR=1 -e VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=8 -e VLLM_MOET_P2P_AG=0 -e VLLM_MOET_DFLASH_PREP_GROUPS=1 -e VLLM_MOET_CANDIDATE_DET_TOPK=1 -e VLLM_MOET_SWA_DECODE_BUILD=1 -e VLLM_MOET_MOE_TOPKSUM_ADD=1 -e VLLM_MOET_SWA_DRAFT_BUILD=1 -e B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1 -e VLLM_MOET_MHC_MXFP8=1 -e VLLM_MOET_MHC_SIDE_DEFER=1 -e VLLM_MOET_SPARSE_MLA_PREFILL_BF16=1 -e VLLM_MOET_PREFILL_SP=1 -e VLLM_MOET_B12X_MOE_PREFILL_FAST=1 -e VLLM_MOET_DET_TOPK=1" \
  EXTRA_ARGS='--enable-prompt-tokens-details --moe-backend b12x --speculative-config {"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false,"moe_backend":"b12x"}' \
  docker/sm120/run-dsv41.sh
python3 tools/sm120_perf/fraser_bench.py --base http://127.0.0.1:8001 \
  --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --prompt-file tools/sm120_perf/fraser_prompt.txt --reps 10
```

#### Fraser Price's BRRR bench on the served configuration

[`fraserprice/DeepSeek-4.1-Flash-1017-BRRR`](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR) ships
`src/brrr/bench.py`, the script behind his published table: a calibrated code prompt of 2K / 16K / 65K tokens at 1, 4, 8
(and 16) concurrent streams, five rounds per cell, **temperature 1.0, thinking off**, generation to natural EOS (≤ 2,048
tokens), the GPUs cooled to 70 °C before every round; PP from a `max_tokens=1` pass, TG from a prefix-cached second pass
timed while every stream decodes. Run unmodified against the served `:8001` (the server needs
`--enable-prompt-tokens-details`; production serves `--max-num-seqs 8`, so the 16-stream column is not run) on 2026-10-28
(`-r21-mxfp8`, [record](docs/benchmarks/brrr-dsv41-2026-10-28.md)) and on 2026-11-03, ten minutes after the switch to `-r23`
([record](docs/benchmarks/brrr-dsv41-2026-11-03.md)). The `-r23` run:

| prompt | conc | TTFT p50 | PP p50 tok/s | PP total | TG p50 tok/s (per stream) | TG total | accepted / step |
|---|---|---|---|---|---|---|---|
| 2,147 | 1 | 156 ms | 13,747 | 13,854 | 776 | 772 | 5.98 (71 %) |
| 2,148 | 4 | 350 ms | 6,149 | 17,510 | 345 | 1,377 | 5.98 (71 %) |
| 2,148 | 8 | 568 ms | 3,778 | 19,343 | 233 | 1,875 | 5.96 (71 %) |
| 16,483 | 1 | 842 ms | 19,574 | 18,881 | 758 | 757 | 5.99 (71 %) |
| 16,483 | 4 | 1,946 ms | 8,849 | 22,264 | 338 | 1,347 | 5.99 (71 %) |
| 16,483 | 8 | 3,418 ms | 4,869 | 22,908 | 233 | 1,857 | 5.98 (71 %) |
| 65,636 | 1 | 2,997 ms | 21,902 | 21,883 | 756 | 750 | 5.99 (71 %) |
| 65,635 | 4 | 7,337 ms | 9,269 | 23,003 | 324 | 1,308 | 5.96 (71 %) |
| 65,636 | 8 | 12,915 ms | 5,142 | 23,213 | 225 | 1,794 | 6.01 (72 %) |

(`accepted / step` is the script's `1 + accepted / drafts` with the draft-token acceptance rate; sampling at temperature 1.0
with block verification accepts more of DSpark's seven drafts than greedy decoding does.)

**Prefill against stock vLLM.** Fraser's published table is the same script on his build — stock vLLM on 4× RTX PRO 6000
Max-Q (325 W), FP8 KV; the prefill was the one column where it led a week ago. PP total (aggregate prefill tok/s) and the
single-stream TTFT:

| prompt × streams | stock vLLM (BRRR recipe, 325 W) | vLLM-Moet `-r21-mxfp8` (2026-10-28) | **vLLM-Moet `-r23` (2026-11-03)** | `-r23` vs stock |
|---|---|---|---|---|
| 2K × 1 / 4 / 8 | 12.5K (172 ms) / 14.6K / 15.1K | 9.1K (221 ms) / 13.1K / 14.2K | **13.9K (156 ms) / 17.5K / 19.3K** | **+11 % / +20 % / +28 %** |
| 16K × 1 / 4 / 8 | 16.9K (972 ms) / 18.7K / 19.0K | 16.0K (1,034 ms) / 18.0K / 17.9K | **18.9K (842 ms) / 22.3K / 22.9K** | **+12 % / +19 % / +21 %** |
| 65K × 1 / 4 / 8 | 17.9K (3.68 s) / 18.2K / 18.1K | 17.5K (3.74 s) / 16.0K / 16.7K | **21.9K (3.00 s) / 23.0K / 23.2K** | **+22 % / +26 % / +28 %** |

Decode per stream is 7–12 % above the recipe's as before (776 / 758 / 756 vs 703 / 684 / 675 tok/s at 2K / 16K / 65K; the 75 W
and the memory clock are on our side of that comparison), eight-stream aggregates 1,794–1,875 vs 1,686–1,847. The fresh
163K prefill of `prefill_time.py`: 9.0 s → 7.66 s (`-r22`) → **7.2 s (`-r23`, 22.7K tok/s)**.

The port with every gap and measurement: **[docs/dsv41-sm120-port.md](docs/dsv41-sm120-port.md)**.
Build, serve, validate and every image tag: **[docs/sm120-deploy.md](docs/sm120-deploy.md)**. Op-level
tests: [tools/dsv41_sm120/README.md](tools/dsv41_sm120/README.md). Head-to-head with an SGLang recipe on
the same host (2026-09-19): [docs/dsv41-sm120-vs-0xsero.md](docs/dsv41-sm120-vs-0xsero.md).

## GLM-5.3-Flash on 4× RTX PRO 6000

Served next to DeepSeek on the host's other four cards (`:8002`): the official FP8 checkpoint
([`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash): 45 layers — 34 KDA linear
attention + 11 DeepSeek sparse attention with the kpool indexer — 288 routed experts top-8, MTP) on the
unmodified `vllm/vllm-openai:nightly-49cb2bbf` image, TP4, 512K context, MTP k = 3, FP8 KV, a 64 GiB
host offload region plus a disk tier (launcher `glm53flash/run.sh` on the host). No vLLM-Moet kernels
yet — this is the baseline the sm_120 work on this model starts from.

| | measured on the served configuration (400 W), 2026-10-26 |
|---|---|
| decode, 1 stream | prose 175–190 tok/s, code 244–258 tok/s (71–75 steps/s; MTP accepts 1.4 / 2.6 of 3 drafts per step) |
| decode, 8 streams | prose 545–557 tok/s, code 710–726 tok/s aggregate |
| fresh prefill | 20K tokens in 2.0 s (10K tok/s), 169K in 17.8 s (9.4K), 406K in 47.6 s (8.5K) |
| KV cache | 613K tokens on the GPUs at 0.92 (1.17× concurrency at 512K), 64 GiB host offload, disk tier |
| quality | GSM8K-200 (reasoning effort low) 192/200, needle PASS at 29K / 106K / 400K, tool calls / JSON / vision pass |

## Qwen3.8-Flash-Next-FP8 on 4× RTX PRO 6000

The official `vllm/vllm-openai` nightly ran this 177B-A6B model on sm_120 at 65 decode steps/s as first
deployed. [`Dockerfile.sm120-qwen38`](Dockerfile.sm120-qwen38) keeps the model's numerics and changes
where the time goes: the 51 GB n-gram PLE table stays on the GPUs instead of the CPU offload worker
(whose handoff stalled every decode graph by ~1.2 ms), vLLM's CuTe-DSL skinny GEMM (gated to B300
upstream) runs the ~480 BF16 decode GEMMs on sm_120 with plans tuned for it, and the routed experts
get a tuned Triton config plus an FP8 MoE GEMV with silu·up and the quantization fused in,
bit-identical to Triton. TP4, MTP k = 3: **98.5 steps/s, prose 243 / code 346 tok/s single stream,
1.5× the first deployment's 65 steps/s and 137–160 / 228 tok/s** (2026-09-19). Launcher
`docker/sm120/run-qwen38.sh`; write-up: [tools/qwen38_sm120/README.md](tools/qwen38_sm120/README.md).

## 2-bit experts on vLLM v0.24.0

A generated patch on official vLLM v0.24.0 ([`patch/`](patch/), image `Dockerfile.sm120-v024`) serves
checkpoints the hardware cannot hold: GLM-5.2 (753B) on two RTX PRO 6000 at 28–32 tok/s (128K window)
and on four at 105 tok/s, DeepSeek-V4-Flash (159B) on one RTX 5090 at ~31 tok/s (32K) and on one
RTX PRO 6000 at 161 tok/s, Kimi-K2.7-Code (1T) on four RTX PRO 6000 at 51 tok/s (256K).

- **2-bit experts at FP4 quality.** The routed experts go to a sign-symmetric 2-bit codebook on
  hand-written SASS kernels (naive 2-bit fails through sign asymmetry, not error size); a delta cache
  and a confidence gate bring FP4 back where it matters.
- **Experts beyond VRAM.** The 2-bit base can live in host RAM or NVMe pack files with the GPU as an
  expert cache; a miss is fetched and the step's graph replayed, bit-identical to a resident forward.
- **Against official v0.24.0.** The release cannot serve these models on SM120 as shipped: the patch
  moves the DeepGEMM and FlashInfer pins, gates the cluster-launch top-k off, fixes the o_proj scale
  layout and captures CUDA graphs in thread-local error mode, and adds MTP under pipeline parallelism
  and an NVFP4 KV cache. With the `VLLM_MOE_W2*` knobs off, the SM120 fixes are the only change.

Hooks, knobs and measurements: **[docs/v024-port.md](docs/v024-port.md)**; quality method:
[docs/quality.md](docs/quality.md); kernels: [kernels/MANIFEST.md](kernels/MANIFEST.md). The recipes
image serves each benchmarked configuration ([bench/README.md](bench/README.md)):

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-v024 -t vllm-moet-sm120:v024 .
DOCKER_BUILDKIT=1 docker build -f Dockerfile.recipes -t vllm-moet-recipes:v024 .
docker run --rm --gpus all --network host --ipc host --shm-size 64g -v /srv/models:/models \
  -e HF_TOKEN=... vllm-moet-recipes:v024 glm-5.2-nvfp4/pro6000x4-tp4-mtp   # --list: all recipes
```

### Benchmark results

<!-- bench:table:begin (generated by bench/runner/render.py - do not edit) -->

Release **`v2026.08.28-x4-prefill`** — one row per supported recipe (`bench/recipes/`), measured by `bench/runner/bench.py`; full report: [`docs/benchmarks/v2026.08.28-x4-prefill.md`](docs/benchmarks/v2026.08.28-x4-prefill.md). Single-stream decode and prefill are medians; batch is aggregate tok/s at the noted concurrency.

| model | hardware | config | ctx | decode tok/s | batch | prefill 8K | needle | notes |
|---|---|---|---:|---:|---:|---:|---|---|
| deepseek-v4-flash | 1x RTX 5090 (32 GB) | host-resident 2-bit base, GPU as expert cache | 8K | **38** | — | — | — | acc 2.83 † |
| deepseek-v4-flash | 4x RTX 5090 TP4 | consumer-card throughput | 16K | **214.4** | 1 560 @32 | 6 101 | — | acc 2.6 † |
| deepseek-v4-flash | 2x RTX PRO 6000 TP2 | throughput | 24K | **209.6** | 380 @3 | 5 791 | — | acc 2.6 † |
| glm-5.2-exl3 | 4x RTX PRO 6000 TP4 | EXL3 trellis base (2,2,1) resident via EP, MTP k=2, decode-wave M8+M16 dual-set | 8K | **71.9** | 196.7 @32 | 2 106 | PASS ≤6K tok | acc 2.62 |
| glm-5.2-nvfp4 | 2x RTX PRO 6000 TP2 | host-resident base, 44 GiB/rank expert cache | 32K | **33** | — | — | PASS ≤27K tok | acc 3 † |
| glm-5.2-nvfp4 | 4x RTX PRO 6000 TP4 | 2-bit base + MTP k=2, 128K window | 128K | **105** | — | 2 500 | PASS ≤276K tok | † |
| glm-5.2-nvfp4 | 4x RTX PRO 6000 TP4 | + FP4 delta (auto) + confidence gate tau=0.60 | 128K | **84** | — | — | — | † |
| kimi-k2.7-code-nvfp4 | 2x RTX PRO 6000 TP2 | host-resident base, 52 GiB/rank cache (~39% coverage) | 16K | **14.4** | — | — | PASS ≤8K tok | † |
| kimi-k2.7-code-nvfp4 | 4x RTX PRO 6000 TP4 | GPU-resident 2-bit + FP4 delta, 256K window | 256K | **51** | 222 @8 | 2 448 | PASS ≤248K tok | † |
| kimi-k2.7-code-nvfp4 | 4x RTX PRO 6000 TP4 | + Eagle3 drafter (k=3, drafter TP4) | 256K | **57** (±44%) | — | — | — | acc 3.5 † |

† imported from pre-harness measurements (README/docs history) — re-measured on the next release.

<!-- bench:table:end -->

### Quality vs native

The row below is retained as explicitly imported historical evidence, not as a
fresh release gate. A correctness audit found that the W2 path had omitted the
checkpoint's SwiGLU clamp and that the attached THINK artifact did not encode
the request/baseline metadata claimed by the compact result. It must be
re-measured through the schema-v2 harness before being certified again.

<!-- bench:quality:begin (generated by bench/runner/render.py - do not edit) -->

Quality release **`v2026.07.30-quality`** — dataset evals vs the committed **native baselines** (`bench/baselines/`, same tool/checkpoint/hardware, stock serving path). Cells: accuracy (Δpp vs native, completion-token inflation vs native). Full report: [`docs/benchmarks/v2026.07.30-quality.md`](docs/benchmarks/v2026.07.30-quality.md); process: [`bench/README.md`](bench/README.md).

| model | hardware | config | GSM8K | GPQA | GPQA think | needle | notes |
|---|---|---|---|---|---|---|---|
| deepseek-v4-flash | 2x RTX PRO 6000 TP2 | max-quality candidate (revalidation required) | **95.5%** (-1.5pp, tok +1.3%) | **73.2%** (-1.0pp, tok +10.6%) | — | — | † |

† imported from the measurement campaign logs — re-measured by the harness on the next quality release.

<!-- bench:quality:end -->

## Kernels and tools

- [`blackwell-isa`](https://github.com/kacper-daftcode/blackwell-isa) (a machine-readable SM120 SASS ISA
  database, [searchable reference](https://kacper-daftcode.github.io/blackwell-isa/SM120_ISA_REFERENCE.html))
  and [`cubit`](https://github.com/kacper-daftcode/cubit) (an SM120 assembler / disassembler built on it)
  produce the 2-bit line's SASS kernels in `kernels/`: stock CUDA compiles the block-scaled MMA, but not
  the scheduling and register shaping these kernels need.
- The DeepSeek-V4.1 decode kernels are CUDA C++ compiled into the image, each with op-level tests, in
  `tools/dsv41_sm120/` (`sm120_gemv/`, `moe_quant_scatter/`, `mhc_overlap/`, `bf16_gemm/`, `moe_l2/`),
  Qwen3.8's MoE GEMV in `tools/qwen38_sm120/moe_gemv/`; the profiling and benchmark scripts behind the
  numbers are in [`tools/sm120_perf/`](tools/sm120_perf/).

## Repository layout

| path | what |
|---|---|
| `Dockerfile.sm120-dsv41-nightly`, `tools/dsv41_sm120/`, `docker/sm120/run-dsv41.sh` | DeepSeek-V4.1-Flash: the served image, its patchers, kernels and tests, the launcher (`docker/sm120/` also holds its cron scripts) |
| `Dockerfile.sm120-dsv41` | the same on the recipe's `deepseekv41-flash-0909` image (served until 2026-09-23) |
| `Dockerfile.sm120-qwen38`, `tools/qwen38_sm120/`, `docker/sm120/run-qwen38.sh` | Qwen3.8-Flash-Next-FP8 |
| `tools/sm120_perf/` | profiling and benchmark scripts |
| `patch/`, `Dockerfile.sm120-v024`, `Dockerfile.recipes`, `docker/serve_recipe.py` | the 2-bit line: the generated v0.24.0 patch (never edited by hand, see [AGENTS.md](AGENTS.md)), its image, the recipes image |
| `bench/` | recipes, the runner and committed results; the tables above are rendered from it and checked in CI |
| `kernels/`, `tools/*.py` | SASS sources, prebuilt SM120 cubins and generators; the 2-bit line's tests |
| `docs/` | port notes, the deploy guide, the quality method, benchmark reports |

## License

Apache License 2.0 ([LICENSE](LICENSE)); third-party components in [NOTICE](NOTICE).
