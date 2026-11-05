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

**Baseline** (stock vLLM `03bd3c68`, k = 7, 512 output tokens, `glm_spec_matrix.py`): code 416 tok/s single stream
at 63 steps/s (6.6 tokens per step), 2270 tok/s aggregate at 8 requests (283 per request, 43 steps/s); prose 169 / 915
(the drafter accepts 1.7 of 7 on prose); fresh prefill 7.9K tok/s at 19K tokens, 5.3K at 163K, 3.5K at 391K; GPU KV
cache 3.17M tokens (bf16); GSM8K-200 193/200 at 8 requests; needle 6/6 at 28K / 105K / 399K tokens; start 4 min.

**As served** (`-r3`: step 1 up to 128 rows, dynamic draft length 15 / 11 / 7, 32 requests, 8192-token chunks, bf16 KV,
no KV offload): see the table in [Measurements](#measurements) — code **~600 tok/s single stream, ~3000 tok/s
aggregate at 8 requests, ~4600 at 32**; prose 170 / 960 / 1900; prefill and quality as the baseline.

On the same four cards GLM-5.3-Flash (the slot's previous model, TP4, MTP k = 3, FP8 KV) did code 244–258 / prose
175–190 tok/s single stream at 71–75 steps/s, ~720 / ~550 aggregate at 8 requests, prefill 8.3K / 7.9K / 7.1K tok/s,
GSM8K-200 192/200.

## Build and serve

```bash
hf download ricursiveAI/Ripple-65B-Preview-NVFP4 --local-dir /srv/models/Ripple-65B-Preview-NVFP4     # 57 GB
hf download ricursiveAI/Ripple-65B-Preview-DFlash2 --local-dir /srv/models/Ripple-65B-Preview-DFlash2  # 8 GB
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-ripple -t vllm-moet-sm120:ripple-nightly-a2bcb565-r3 .   # ~3 min
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
in the shared decode window / per request / engine steps per second. Image `-r2`/`-r3`, step 1 on unless noted.

| configuration | code 1 | code 8 | code 32 | prose 1 | prose 8 | prose 32 |
|---|---|---|---|---|---|---|
| stock vLLM, k 7 (NCCL) | 416 @ 63.1 | 2270 / 283 @ 43.1 | — | 169 @ 61.8 | 915 / 115 @ 42.8 | — |
| step 1, k 4 | 293 @ 64.6 | 1990 / 249 @ 55.4 | 4060 / 127 @ 28.3 | 147 @ 64.0 | 1108 / 137 @ 55.3 | 2320 / 72 @ 28.6 |
| step 1, k 7 | 425 @ 64.5 | 2617 / 327 @ 49.3 | 5030 / 158 @ 23.9 | 172 @ 63.6 | 1052 / 130 @ 48.8 | 2060 / 64 @ 24.0 |
| step 1, k 11 | 573 @ 63.7 | 2890 / 360 @ 41.1 | 4700 / 147 @ 16.9 | 173 @ 62.6 | 902 / 111 @ 40.7 | 1500 / 47 @ 17.0 |
| step 1, k 15 | 614 @ 64.2 | 2940 / 365 @ 37.9 | 4430 / 138 @ 14.1 | 169 @ 62.6 | 833 / 104 @ 37.5 | 1276 / 39 @ 14.1 |
| step 1 (≤ 64 rows), dynamic 15/11/7 | 589 @ 63.2 | 2838 / 354 @ 40.8 | 4632 / 145 @ 21.7 | 169 @ 62.0 | 890 / 110 @ 40.6 | 1910 / 59 @ 21.8 |
| step 1 (≤ 256 rows), dynamic 15/11/7 | 630 @ 63.2 | 2993 / 376 @ 42.5 | 4438 / 139 @ 21.0 | 171 @ 62.4 | 963 / 121 @ 42.6 | 1845 / 58 @ 21.1 |

Reading: at one request the step takes ~15.6 ms whatever the draft length, so tokens per step (6.6 at k 7, 9.6 at
k 15 on code) set the single-stream speed; at 32 requests the verification of 32 × (k + 1) rows is compute-bound and
k 7 is the best static choice. The dynamic schedule costs ~8 % at 32 requests against static k 7 (21.7 vs 23.9
steps/s) and gains 40 % at one; a static k 11 is the simple alternative (573 / 2890 / 4700). The all-reduce push pays
at 8–16 requests (96 rows: 40.8 → 42.5 steps/s) and loses at 256 rows.

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

## Not done, next

- Prefill: `--attention-backend FLASHINFER` and 16K chunks against FA2 at 163K / 391K; MXFP8 attention projections
  (an output change — gate it) — attention is half the weight bytes.
- The decode step at one request (15.6 ms for 80 layers + the drafter): a profile first; candidates are the
  one-shot push for `M ≤ 16` (`VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS`, neutral on DeepSeek), `--async-scheduling`, the
  NVFP4 GEMM backend (`VLLM_NVFP4_GEMM_BACKEND`; FlashInfer-CUTLASS is vLLM's choice here).
- The dynamic schedule's 8 % at 32 requests (graph sizes vs truncated drafts); a 9–16 band.
- KV offload (host RAM + disk tier, `KV_OFFLOAD_GIB`) for multi-turn agent sessions beyond the 3.1M-token GPU cache,
  with the prefix-hit fidelity probe; `--kv-cache-dtype fp8` only if capacity is ever short.
- A BF16 reference (the 130 GB checkpoint, TP4 on the test host) for a paired GSM8K / parity gate of the NVFP4
  checkpoint.
