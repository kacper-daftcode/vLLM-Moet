# Ripple-65B Preview on 4× RTX PRO 6000 (sm_120)

Ricursive's [`Ripple-65B-Preview`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) — a 65B dense decoder
(80 layers, d_model 5120, 80 query / 8 KV heads; four 1024-token sliding-window RoPE layers per NoPE full-attention
layer; PeriNorm, gated attention, value residuals, clamped SwiGLU; 2M positions) — in its
[NVFP4 checkpoint](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4) (ModelOpt: MLPs in NVFP4 g16 with
static input scales, attention / embeddings / lm_head in BF16, 57 GB) with the
[DFlash2 drafter](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2) (4B BF16, blocks of 16, taps on
target layers 1 / 20 / 39 / 58 / 77), TP4, 512K context. vLLM knows the architecture since
[vllm#65745](https://github.com/vllm-project/vllm/pull/65745) (`RicursiveForCausalLM`, `--reasoning-parser ripple`,
`--tool-call-parser ripple`, `DFlash2RicursiveDraftModel`); [`Dockerfile.sm120-ripple`](../Dockerfile.sm120-ripple)
puts that vLLM on the official nightly image and adds the sm_120 steps. Patchers and probes:
[`tools/ripple_sm120/README.md`](../tools/ripple_sm120/README.md).

| step | change | switch | effect (this host) |
|---|---|---|---|
| 0 | vLLM at `03bd3c68` (the per-commit wheel on nightly `a2bcb565`, same pins) | — | the model serves; stock vLLM is the baseline below |
| 1 | the TP group's bf16 `[M, 5120]` all-reduces of `M ≤ 128` rows on the PCIe P2P push with NCCL's summation order (the DeepSeek line's `p2p_ar` module behind a generic hook in `GroupCoordinator._all_reduce_out_place`; vLLM has only PyNCCL for 4 PCIe GPUs) | `VLLM_MOET_P2P_AR=1`, `VLLM_MOET_P2P_AR_M_MAX` / `_MAX_ROWS` (128 served; 64 default) | +2 % steps/s at 1 request, **+14 % at 8** (k = 7; 160 + 10 all-reduces per step); at 256 rows (32 requests × k 7) the push is 3 % slower than NCCL, so the cut-off is 128; outputs identical to NCCL |
| — | vLLM's dynamic draft length by batch size (`num_speculative_tokens_per_batch_size`) | `SPEC_SCHEDULE` | 15 / 11 / 7 draft tokens for 1–2 / 3–8 / 9–32 requests: the verification of 16 tokens is free at 1 request (steps/s do not depend on k), compute-bound at 32 |
| 2 | the DFlash2 drafter drafts the schedule's k instead of always 15: 1 + k query rows per request through its 5 layers (its grouped convolutions read each row's offset in its request's block from the speculator instead of `row % 16`), k sample rows, (1 + k) × requests context rows, CUDA graphs per query width as the target's (`patch_vllm_dflash_dynamic_k_sm120.py`) | — (no schedule: k = 15 every step, as before) | **+6.5 % at 32 requests** (code 4738 → 5045 tok/s, 22.4 → 24.0 steps/s = the static-k7 level with k 15 kept at 1 request; prose 1950 → 2080), the same at 1–8 (the drafter's 96 vs 128 rows at 8 requests were ~0.4 ms of 22.7); outputs are the target's argmax either way: parity identical, GSM8K-200 194 vs 193, needle 6/6 |

**Baseline** (stock vLLM `03bd3c68`, k = 7, 512 output tokens, `glm_spec_matrix.py`): code 416 tok/s single stream
at 63 steps/s (6.6 tokens per step), 2270 tok/s aggregate at 8 requests (283 per request, 43 steps/s); prose 169 / 915
(the drafter accepts 1.7 of 7 on prose); fresh prefill 7.9K tok/s at 19K tokens, 5.3K at 163K, 3.5K at 391K; GPU KV
cache 3.17M tokens (bf16); GSM8K-200 193/200 at 8 requests; needle 6/6 at 28K / 105K / 399K tokens; start 4 min.

**As served** (`-r4`: steps 1 (up to 128 rows) and 2, dynamic draft length 15 / 11 / 7, 32 requests, 8192-token chunks,
bf16 KV, no KV offload): see the table in [Measurements](#measurements) — code **~600 tok/s single stream, ~3000 tok/s
aggregate at 8 requests, ~5000 at 32**; prose 170 / 980 / 2080; prefill and quality as the baseline.

On the same four cards GLM-5.3-Flash (the slot's previous model, TP4, MTP k = 3, FP8 KV) did code 244–258 / prose
175–190 tok/s single stream at 71–75 steps/s, ~720 / ~550 aggregate at 8 requests, prefill 8.3K / 7.9K / 7.1K tok/s,
GSM8K-200 192/200.

## Build and serve

```bash
hf download ricursiveAI/Ripple-65B-Preview-NVFP4 --local-dir /srv/models/Ripple-65B-Preview-NVFP4     # 57 GB
hf download ricursiveAI/Ripple-65B-Preview-DFlash2 --local-dir /srv/models/Ripple-65B-Preview-DFlash2  # 8 GB
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-ripple -t vllm-moet-sm120:ripple-nightly-a2bcb565-r4 .   # ~3 min
MODEL_DIR=/srv/models/Ripple-65B-Preview-NVFP4 DRAFT_DIR=/srv/models/Ripple-65B-Preview-DFlash2 \
  GPUS=4,5,6,7 PORT=8002 SPEC_TOKENS=15 SPEC_SCHEDULE='[[1,2,15],[3,8,11],[9,32,7]]' \
  EXTRA_DOCKER_ARGS="-e VLLM_MOET_P2P_AR=1 -e VLLM_MOET_P2P_AR_M_MAX=128 -e VLLM_MOET_P2P_AR_MAX_ROWS=128" \
  docker/sm120/run-ripple.sh
```

The first start compiles for ~4 minutes (torch.compile into `CACHE_DIR`), later starts take ~3.5; the API listens on
`127.0.0.1:8002` (`BIND=` for all interfaces). Every knob is an environment variable documented in the header of
[`docker/sm120/run-ripple.sh`](../docker/sm120/run-ripple.sh). Check a built image on one GPU with
`tools/ripple_sm120/image_check.sh IMAGE GPU MODEL_DIR DRAFT_DIR`; probe a server with `tools/ripple_sm120/ripple_smoke.py`.

Clients: OpenAI chat completions with `reasoning_content` separated from `content` (Ripple decides per turn whether
to reason — `reasoning_content` is often empty on short questions), Hermes-style tool calls with `tool_choice: auto`,
JSON schema outputs. In multi-turn agent traffic send the previous `reasoning_content` and `tool_calls` back
unchanged (the template renders them to the generated tokens, so the prefix cache hits).

## Measurements

4× RTX PRO 6000 Blackwell (400 W), TP4, 512 output tokens per request, `ignore_eos`, temperature 0; aggregate tok/s
in the shared decode window / per request / engine steps per second. Image `-r2`/`-r3`/`-r4`, step 1 on unless noted.

| configuration | code 1 | code 8 | code 32 | prose 1 | prose 8 | prose 32 |
|---|---|---|---|---|---|---|
| stock vLLM, k 7 (NCCL) | 416 @ 63.1 | 2270 / 283 @ 43.1 | — | 169 @ 61.8 | 915 / 115 @ 42.8 | — |
| step 1, k 4 | 293 @ 64.6 | 1990 / 249 @ 55.4 | 4060 / 127 @ 28.3 | 147 @ 64.0 | 1108 / 137 @ 55.3 | 2320 / 72 @ 28.6 |
| step 1, k 7 | 425 @ 64.5 | 2617 / 327 @ 49.3 | 5030 / 158 @ 23.9 | 172 @ 63.6 | 1052 / 130 @ 48.8 | 2060 / 64 @ 24.0 |
| step 1, k 11 | 573 @ 63.7 | 2890 / 360 @ 41.1 | 4700 / 147 @ 16.9 | 173 @ 62.6 | 902 / 111 @ 40.7 | 1500 / 47 @ 17.0 |
| step 1, k 15 | 614 @ 64.2 | 2940 / 365 @ 37.9 | 4430 / 138 @ 14.1 | 169 @ 62.6 | 833 / 104 @ 37.5 | 1276 / 39 @ 14.1 |
| step 1 (≤ 64 rows), dynamic 15/11/7 | 589 @ 63.2 | 2838 / 354 @ 40.8 | 4632 / 145 @ 21.7 | 169 @ 62.0 | 890 / 110 @ 40.6 | 1910 / 59 @ 21.8 |
| step 1 (≤ 256 rows), dynamic 15/11/7 | 630 @ 63.2 | 2993 / 376 @ 42.5 | 4438 / 139 @ 21.0 | 171 @ 62.4 | 963 / 121 @ 42.6 | 1845 / 58 @ 21.1 |
| step 1 (≤ 128 rows), dynamic 15/11/7 — served `-r3` (2026-11-05 23:26Z) | 600–613 @ 63.3 | 3036–3112 / 380 @ 43.7 | 4738 / 148 @ 22.4 | 167–196 @ 60.8 | 965 / 121 @ 43.1 | 1950–1970 / 61 @ 22.5 |
| **+ step 2 (drafter at the schedule's k) — served `-r4`** | 626–640 @ 63.6 | 3010–3097 / 384 @ 43.5 | **5041–5046 / 158 @ 24.0** | 166–171 @ 62.3 | 984 / 122 @ 43.3 | **2074–2089 / 65 @ 24.1** |

Reading: at one request the step takes ~15.6 ms whatever the draft length, so tokens per step (6.6 at k 7, 9.6 at
k 15 on code) set the single-stream speed; at 32 requests the verification of 32 × (k + 1) rows is compute-bound and
k 7 is the best static choice. The dynamic schedule costs ~8 % at 32 requests against static k 7 (21.7 vs 23.9
steps/s) and gains 40 % at one; a static k 11 is the simple alternative (573 / 2890 / 4700). The all-reduce push pays
at 8–16 requests (96 rows: 40.8 → 42.5 steps/s) and loses at 256 rows. Step 2 removes the schedule's cost at 32 requests: the
drafter had been running its 16-wide block for every request whatever k (512 rows for 7 drafts each), now 8 — 24.0
steps/s, the static-k7 level, with k 15 kept at one request.

Prefill does not depend on the draft length: 19K tokens 2.3–2.5 s (7.9–8.4K tok/s), 163K 29–30 s (5.4–5.6K), 391K
110 s (3.5K) — the 16 full-attention layers over 391K tokens on FlashAttention 2 (`FLASH_ATTN`, vLLM's choice on
sm_120). Memory: weights 16.6 GiB per GPU (drafter included), KV 69 GiB per GPU = 3.1–3.2M tokens (bf16; the hybrid
KV manager keeps the 64 sliding layers at 1024 tokens each), peak 90.8 GiB per GPU at 0.92.

Quality (baseline configuration): GSM8K-200 at 8 requests 193/200 (the model card: 0.916 completions / 0.971 chat on
the full set, NVFP4 on B300); arithmetic battery 4/5 (one wrong sum when the model answers without reasoning);
needle 6/6 at 28K / 105K / 399K; tool call + round trip and strict JSON schema pass. Step 1 reproduces NCCL's bits
by construction (the summation order is measured on the TP communicator at start-up and each row count is checked
against NCCL; row counts that do not reproduce stay on NCCL): three prompts × up to 768 greedy tokens with per-token
logprobs (`tools/ripple_sm120/parity_probe.py`) are identical token for token and logprob for logprob between the
P2P and the NCCL arm (1724 tokens, max |Δlogprob| 0), as are two runs on the same server.

## Where the decode step goes (profile, 2026-11-06)

Torch-profiler traces of the served configuration, rank 0, mean over the middle steps of a 256-token decode
(`tools/ripple_sm120/step_anatomy.py` on a rank trace; GEMM shapes in isolation: `tools/ripple_sm120/decode_gemm_bench.py`). The cards run their memory at
16.5 GHz (2.11 TB/s), so a TP rank's 16.1 GB of weights per step have a 7.6 ms floor.

| class (per step, rank 0) | 1 request, code (16 rows) | 8 requests, code (96 rows) | 32 requests, code (256 rows, static k 7) |
|---|---|---|---|
| step | 16.2 ms, 2375 kernels, GPU busy 96 % | 22.7 ms | 40.4 ms |
| BF16 GEMMs (cuBLAS: qkg 5376×5120, v, o, drafter, lm_head) | 6.1 ms (38 %) — qkg 33 µs = 79 % of bandwidth, o 17 µs | 7.1 ms | 9.5 ms (compute-bound) |
| NVFP4 GEMMs (FlashInfer-CUTLASS: gate_up, down) | 4.5 ms (28 %) — 33.5 / 22 µs | 4.9 ms | 6.6 ms |
| all-reduce (172 per step) | 2.45 ms (15 %) — P2P two-shot 14.2 µs each | 7.4 ms (33 %) — 43 µs each, 1.47 MB per rank = 34 GB/s over PCIe | 18.9 ms (47 %) — NCCL 109 µs each, 3.9 MB per rank = 36 GB/s |
| norms / gating / fp4 input quant (Inductor Triton, 652 launches) | 1.4 ms | 1.6 ms | 2.0 ms |
| attention (FA2 split-KV, 85 launches) | 0.7 ms | 0.7 ms | 1.9 ms |
| idle (launch granularity inside the graphs; async scheduling is already on) | 0.66 ms | 0.3 ms | 0.8 ms |

Reading: at one request the step is weight-bandwidth-bound (GEMMs at 58–84 % of 2.11 TB/s) plus the all-reduce
latency; at 8–32 requests the all-reduce is PCIe-bandwidth-bound (the two-shot push and NCCL both sit at 34–36 GB/s
per rank) and takes a third to a half of the step. ≥ 100 steps/s at one request is out of reach without changing
bytes: the BF16 attention projections are 6.6 GB of the 16.1 GB a rank reads per step.

Closed by measurement (do not revisit without new evidence):

- One-shot push for small `M` (`VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS`): never faster here — two-shot 12.1 / 13.4 / 17.1 /
  33.4 / 44.4 / 58.6 µs for M = 8 / 16 / 32 / 64 / 96 / 128 against one-shot 13.3 / 23.2 / 40 / 80 / 123 / 166 and
  NCCL 13.2 / 18.1 / 28.3 / 51.6 / 50.2 / 60.3 (4-rank graph bench, bits = NCCL for all).
- NVFP4 GEMM backend (`--linear-backend`): in a microbench every backend reproduces FlashInfer-CUTLASS's bits and
  `flashinfer_cudnn` is 2–10 µs faster per layer; in situ it is 10 % slower at one request (57 vs 63 steps/s) and
  its outputs differ (parity diverges after 326 / 88 tokens). `b12x` is 19 % faster at M = 256 only. Kept: `auto`.
- `--attention-backend FLASHINFER` (target; the drafter keeps FA2): prefill −8 % at 163K (27.0 s) and −9 % at 391K
  (100.5 s), KV cache +13 %, but decode −10 % at one request (57 steps/s) and −3 % at eight. Kept: FA2.
- Dynamic draft length at 32 requests: the −6 % against static k 7 (42.9 vs 40.4 ms per step) is the DFlash2 drafter,
  which drafts the configured maximum (15, so 512 rows at 32 requests) whatever the schedule says for the step
  (`DFlashSpeculator.num_query_per_req = 1 + num_speculative_steps`; `propose()` only special-cases 0) — its gate_up
  GEMM runs 4.8 × 331 µs instead of 1.2 × 214 µs. Fixed by step 2 (`1 + k` rows per request with graphs per width;
  the drafter's grouped convolutions had to learn each row's offset in its block — their `row % 16` made the first
  arm's drafts collapse to 0.24 acceptance at C8): 22.4 → 24.0 steps/s at 32 requests, the schedule stays.
- BF16 GEMMs through cuBLASLt instead of cuBLAS: no gain, different bits for most shapes.

## Candidate: attention projections in MXFP8 (a numerics change — the owner's decision)

`tools/ripple_sm120/make_mxfp8_attn_checkpoint.py` derives `Ripple-65B-Preview-NVFP4-attnMXFP8`: q/k/g/o_proj of all
80 layers in MXFP8 (e4m3 values, one e8m0 scale per 32 elements, vLLM's own quantization convention), v_proj
(256-wide per rank: faster in BF16 than any MXFP8 kernel), embeddings, lm_head and the NVFP4 MLPs unchanged;
`quant_algo: MIXED_PRECISION` with a `quantized_layers` map, loaded by stock vLLM as `modelopt_mixed`
(FlashInfer-CUTLASS MXFP8 GEMM on sm_120). Weights per rank 16.6 → 13.6 GiB, KV cache 3.14M → 3.27M tokens.

Measured on this host (2026-11-06 00:14–00:25Z, same served configuration otherwise):

| | NVFP4 checkpoint (served) | + attention in MXFP8 |
|---|---|---|
| code, 1 request | 600–613 tok/s @ 63.3 steps/s | **731 tok/s @ 70.0 steps/s** |
| prose, 1 request | 167–196 @ 60.5–61 | 176–193 @ 68.3–69.0 |
| code, 8 requests | 3036–3112 / 382–387 @ 43.7 | **3218–3248 / 405–407 @ 45.5** |
| code, 32 requests | 4738 / 148 @ 22.4 | 4785–4802 / 150 @ 22.3 (all-reduce-bound) |
| fresh prefill 19K / 163K / 391K | 2.45 s / 29–30 s / 110 s | 2.24 s / 28.5 s / 106.3 s |
| GSM8K-200, 8 requests | 193 / 200 (wrong: 66 100 147 158 159 160 164) | 191 / 200 (wrong: 3 29 76 100 147 158 159 160 164) |
| GSM8K full (1319), 8 requests, paired | **1282 / 1319 = 97.19 %** (the model card: 0.971) | **1278 / 1319 = 96.89 %** — 32 errors shared, 9 only here, 5 only on the served arm; exact McNemar p = 0.42 (not significant; point estimate −0.3 pp) |
| needle 28K / 105K / 399K, tool call + round trip, JSON schema, arithmetic 4/5 | pass (same answers) | pass (same answers) |
| greedy parity against the served arm | — | diverges after 67 / 59 / 94 tokens (code / prose / math), as any numerics change does |

Per-item results: `ripple65b/results/gsm8k_full_{served-y2,attn-mxfp8}.json` (`wrong_runs[].run_index`). Not served: the owner decides whether +20 % single-stream speed is worth a numerics change the 1319-item paired test cannot distinguish from noise (−4 items).

## The 2026-11-06 nightly (vLLM 4fab5a07, Ripple native) — built as `-r5`, not served

`vllm/vllm-openai:nightly-4fab5a07…` (vLLM 0.33.0rc1.dev910, FlashInfer 0.7.3, torch 2.14.0+cu130) has `RicursiveForCausalLM`
and `DFlash2RicursiveDraftModel` natively, so step 0 is off there (`--build-arg VLLM_MOET_RIPPLE_STEP0_WHEEL=0 --build-arg
BASE_IMAGE=vllm/vllm-openai:nightly-x86_64@sha256:8b45a8be…`); steps 1 and 2 apply unchanged. Measured in situ against
`-r4` (same configuration): the same speed (code 616–653 @ 63.7 / 2996–3034 @ 43.6 / 5064–5079 @ 24.2; prose 2100–2122 @
24.1), the same smoke / needle answers, GSM8K-200 192/200 (r4 194, r3 193) — but other numerics: the parity probe diverges
from the served bits after 369 code / 83 prose tokens (math identical), max |Δlogprob| 0.36 on the common prefix, KV
3.147M tokens (−0.9 %), start 211 s (compile cache miss). With no speed to gain the served image stays `-r4` (bits of
03bd3c68 + FlashInfer 0.7.2); `-r5` is the base to move to when a newer vLLM brings something, with the quality gates.

## Not done, next

- Multi-request throughput: the all-reduce is 33–47 % of the step at 8–32 requests and sits on the PCIe floor. The
  structural fix, two micro-batches with one half's all-reduce under the other half's GEMMs, was measured in
  isolation (`tools/ripple_sm120/decode_overlap_bench.py`, 2026-11-06): the platform hides 85–89 % of the P2P push
  under GEMMs on another stream, but two halves stream the layer's weights twice (the decode GEMMs are
  weight-bandwidth-bound: +43 % at 256 rows to +93 % at 96 for two halves, +24…+51 % when the halves alternate per
  attention / MLP block so the second finds the block's ~80 MB in L2 — vLLM's DBO schedule). Net on the GEMMs + 2
  all-reduces of a layer against today's serial step: **−16.5 % at 256 rows (32 requests × k 7), −14 % at 128,
  −6.5 % at 96 (8 × k 11)**, i.e. about −12 % of the whole step at 32 requests and −5 % at 8 — not the −30…−40 %
  the all-reduce share suggested. vLLM's DBO needs DP > 1 and a DeepEP / NIXL all-to-all backend and is marked
  under development for the V2 runner, so it would be our own runner work (days) for that gain; the owner decides.
- Fusing v_proj into the qkg GEMM (one launch and one weight stream less per layer: 4.4 µs of 16 rows; bits of V may
  move); fused add + norm + norm + fp4-quant kernels for PeriNorm (652 small launches per step, 1.4 ms).
- KV offload (host RAM + disk tier, `KV_OFFLOAD_GIB`) for multi-turn agent sessions beyond the 3.1M-token GPU cache,
  with the prefix-hit fidelity probe; `--kv-cache-dtype fp8` only if capacity is ever short.
- A BF16 reference (the 130 GB checkpoint, TP4 on the test host) for a paired GSM8K / parity gate of the NVFP4 and
  the MXFP8-attention checkpoints.
