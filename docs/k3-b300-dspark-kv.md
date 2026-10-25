# Kimi-K3 + DSpark on 8x B300: where the KV cache goes, and getting it back

Served configuration (hgx-02, `docker/b300/run-k3.sh`): `moonshotai/Kimi-K3` (MXFP4, 69 KDA + 24 MLA layers, 1M
context) on `vllm/vllm-openai:v0.32.0`, TP8 + DCP8, fp8 KV, prefix caching (`--prefix-match-unit 128`),
`SimpleCPUOffloadConnector` on an NVMe slab through the page cache, DSpark (`Inferact/Kimi-K3-DSpark`, 5 MLA
layers) with k = 4, `--max-num-seqs 128`, `--max-num-batched-tokens 8192`, `--gpu-memory-utilization 0.95`. Since
2026-10-25 the image is `vllm-moet-b300:k3-v0.32.0-r1` (the official image plus §2's patch) with `USE_RECOVERSSM=1`.

The 2026-10-23 A/B measured the GPU KV pool at **29.4M tokens without speculation and 19.26M with DSpark k = 4**
(`GPU KV cache size` in the engine log). This note explains the 35 % with v0.32.0's accounting, reproduces the
numbers (`tools/k3_b300/kv_model.py`), and recovers most of the loss with one flag that the official image refuses to
combine with KV offload (`tools/k3_b300/patch_vllm_recoverssm_offload.py`, `Dockerfile.b300-k3`).

## 1. How v0.32.0 lays out K3's KV cache

K3 is a hybrid model, so the hybrid KV cache manager is in play (`vllm/v1/core/kv_cache_utils.py`,
`vllm/platforms/interface.py`):

* **One page size for every layer.** A KDA layer's "page" is its whole recurrent state for one sequence:
  conv state `4608 x (3 + k)` bf16 + recurrent state `12 x 128 x 128` fp32 per GPU = 850,944 B at k = 4. The
  MLA page must be at least that big, so the attention block is inflated to `128 x ceil(850,944 / (128 x 576 B))
  = 1536` tokens (576 B = fp8 latent 512 + rope 64), the KDA page is padded 3.97 % to the same 884,736 B, and DCP8
  makes the logical attention block 12,288 tokens (`kv cache group sizes (1536, 1536, 1536, 12288)`).
* **Layers grouped in equal-sized groups.** The 24 target MLA layers and the draft's 5 MLA layers share one spec, so
  they form one bucket of 29; the 69 KDA layers form the other. `group_size = min = 29` (the `max < 1.5 x min`
  escape does not fire: 69 > 43.5), so KDA is split into three groups of 29 and the last one holds 11 real layers
  plus **18 padding layers** (`Add 18 padding layers, may waste at most 26.09% KV cache memory`). Without the
  draft the buckets are 24 / 69 and the padding is 3 layers.
* **One block = 29 pages = 25.66 MB on every group**, whichever group owns it (`bytes_per_block = group_size x
  page`). 46.98 GiB of KV memory is 1966 blocks.
* **Per-request need.** A 1M-token request needs `ceil(1,048,576 / 8 / 1536) = 86` attention blocks, plus, per KDA
  group in align mode, `2 + num_speculative_blocks + 1 (prefill checkpoint)` state blocks
  (`MambaSpec.max_memory_usage_bytes`). `num_speculative_blocks = num_speculative_tokens` unless ReplaySSM/
  RecoverSSM is on (`model_executor/layers/mamba/abstract.py`), i.e. **one full KDA state slot per draft token on
  every KDA layer**: at k = 4 that is 7 blocks per group, 21 per request. 86 + 21 = 107 blocks per request,
  1966 / 107 = 18.36 concurrent 1M requests = the logged **19,256,559 tokens**.

So the DSpark penalty at k = 4 decomposes as (per GPU, `kv_model.py --kv-gib`):

| cause | effect on the 19.26M-token pool |
| --- | --- |
| fixed memory: draft weights + buffers ~3 GiB, peak activation 9.18 vs 5.9 GiB, CUDA graphs 2.94 vs 0.55 GiB (640-token graphs) | 52.7 -> 46.98 GiB of KV memory, -10 % |
| 5 draft MLA layers in every block (29 instead of 24 pages) | -17 % on every block, inherent to the draft |
| KDA speculative state slots: 3 groups x 4 blocks x 25.66 MB = **308 MB per request** | -12 % at 1M-token requests; dominant at agent-sized contexts: a 32K request holds 21 blocks (539 MB), 3 of them attention |
| 18 instead of 3 padding layers | 62 % of the third KDA group's blocks are empty: ~96 MB per request at k = 4 |

## 2. RecoverSSM: the speculative state slots go away

`--use-replayssm` on K3 selects the KDA **RecoverSSM** verify (`models/kimi_k3/nvidia/ops/recoverssm.py`,
`kda_metadata.py`): the k + 1 verify tokens run off one checkpoint state, the kernel keeps a small fp32 correction
record and a key/gate record per layer (61 KB per page, i.e. the KDA page grows to 912,384 B and the attention block
to 1664 tokens, 5.05 % padding), and `KDARecoverSSMCommitContext.commit` restores the accepted state into the single
slot in the step's postprocess. `MambaSpec.num_speculative_blocks` becomes 0: per request 9 instead of 21 KDA blocks
at the 1M estimate, and at run time one or two per group.

v0.32.0 refuses the flag next to any KV connector ("--use-replayssm is incompatible with KV connectors (P/D
disaggregation, KV cache offload)", `VllmConfig.validate_mamba_cached_kernel`). That rule is written for Mamba2
ReplaySSM, which defers the state store to a flush; K3's RecoverSSM commits in-step
(`MambaHybridModelState.postprocess_state` -> `RecoverSSMState.commit_step`), before the V2 runner calls
`kv_connector.post_forward`, and the offload connector only copies boundary states the scheduler has finalized
(`MambaManager.finalize_partial_tail_offload`). `patch_vllm_recoverssm_offload.py` lifts the check when
`cache_config.use_kda_recoverssm` is set and leaves Mamba2 alone; `Dockerfile.b300-k3` applies it to the official
image.

Also qualified in the recipe's InferenceX AgentX lanes, orthogonal to DSpark: `--mamba-ssm-cache-dtype bfloat16`
halves the recurrent state, so the KDA page and the attention block halve as well (1024 tokens with RecoverSSM;
`MAMBA_SSM_CACHE_DTYPE=bfloat16` in the launcher). The `latency_specialized` decode path needs fp32, and it is a
numerics trade-off, so it is measured below but not the default.

## 3. Measurements (hgx-02, 2026-10-24, same memory budget, prod = F of the 2026-10-23 A/B)

Engine accounting (`GPU KV cache size`, the model's numbers in brackets):

| configuration | attention block | KDA blocks / request (1M estimate) | GPU KV tokens | 32K request: blocks (MB) | 32K requests that fit |
| --- | --- | --- | --- | --- | --- |
| A no speculation (A/B 2026-10-23) | 1536 | 9 | 29.4M | 9 (191) | 263 |
| F DSpark k = 4 (prod) | 1536 | 21 | 19,256,559 [19.27M] | 21 (539) | 93 |
| T1 F + RecoverSSM | 1664 | 9 | 21,614,964 [21.61M] | 9 (250) | 201 |
| T2 T1 + bf16 SSM state | 1024 | 9 | 22,563,518 [22.57M] | 10 (171) | 294 |

The fixed costs are unchanged by the flags (198.19 GiB weights + non-torch, 9.18 GiB peak activation, 2.94 GiB
CUDA graphs in every run); the KV memory is 46.98 GiB per GPU in all three.

Serving, same host and image, prod F re-measured the same day (`tools/k3_b300/`; the A/B rows are from
2026-10-23):

| | F prod | T1 RecoverSSM | T2 + bf16 state |
| --- | --- | --- | --- |
| Fraser code, 1 stream: tok/s (accepted length, steps/s) | 227.2 (3.35, 67.8) | 219.6 (3.34, 65.8) | 217.2 (3.29, 66.1) |
| per-position acceptance | 0.83 / 0.65 / 0.49 / 0.38 | 0.85 / 0.63 / 0.49 / 0.37 | 0.84 / 0.63 / 0.47 / 0.36 |
| 8K prompt, C = 32 / 64 / 128: tok/s per session | 60.7 / 42.8 / queue (A/B) | 57.6 / 40.7 / 25.5 | - / 40.8 / - |
| 32K prompt, C = 32 / 64: tok/s per session | 56.2 / 37.8 (A/B) | 54.2 / 37.3 | - / 36.1 |
| 128 x ~22K-token requests, 3000 output tokens each (`capacity_probe.py`) | **KV 100 %, 34 preemptions, at most 115 running**, TTFT median 264 s, 803 tok/s aggregate, 478 s | all 128 running, KV 36 %, 0 preemptions, TTFT median 71 s, **1418 tok/s, 271 s** | all 128 running, KV 27 %, 0 preemptions, 76 s, 1346 tok/s, 285 s |
| GSM8K-200 (greedy, `reasoning_effort: low`) | 196/200 = 98.0 % | 196/200 = 98.0 % | 197/200 = 98.5 % |
| needle 28K / 94K tokens, depth 0.2 / 0.7 | 4/4 PASS | 4/4 PASS | 4/4 PASS |
| 30K prompt: cold -> GPU-warm TTFT; multi-turn continuation | 0.98 -> 0.11 s; repeat identical | - | 0.99 -> 0.11 s; repeat identical |
| greedy output vs prod F (4 prompts, 400 tokens) | identical run-to-run | same answers, token-identical for 70-240 chars then diverges (the verify kernel's own fp32 recurrence) | same, 40-380 chars |

RecoverSSM costs 3-4 % single-stream decode (its verify kernel replaces the fused recurrent KDA step; the
per-session rates at C = 32-64 are within 2-5 %) and removes the KV ceiling that the agent-sized workload hits on
prod: at 128 sessions of ~22K tokens the F pool is full and the scheduler preempts, T1 runs them all at 36 %.
The cold-vs-warm greedy divergence on a long prompt is the baseline's behaviour too (F: cold != warm, warm == warm);
RecoverSSM does not add non-determinism (T1 and T2 repeat their warm answers exactly).

T2 (`--mamba-ssm-cache-dtype bfloat16`) adds another 4 % of pool and 30 % less memory per 32K request over T1 at
no further speed cost in these runs; the quality probes do not separate it from T1, but they are short-context
probes and the recipe's low-latency decode path requires fp32, so T1 is the served default and T2 stays a knob.

### KV offload round trip (RecoverSSM)

The patch's claim is that a connector never copies a stale KDA state. Checked end to end on the served image with
`--kv-cache-memory 6 GiB` (a 2.75M-token pool) so that eviction is quick (`tools/k3_b300/evict_probe.py`,
2026-10-25), greedy, `reasoning_effort: low`:

| step | TTFT | prefix hits (GPU / external) | output |
| --- | --- | --- | --- |
| 30K-token prompt, cold | 1.33 s | 0 / 0 | reference |
| same prompt, GPU-warm | 0.12 s | 20,224 / 0 | differs from cold (baseline behaviour, see above) |
| continuation (turn 2) on the warm answer | 0.21 s | 20,224 / 0 | reference for turn 2 |
| 30 distinct ~71K-token prompts, 2.12M tokens in 105 s | | | evicts the pool (KV usage 5 % after) |
| same prompt, restored from the connector | 0.14 s | **0 / 20,224** | **token-identical to GPU-warm** |
| continuation (turn 2) after the restore | 0.12 s | 13,312 / 7,040 | **token-identical to turn 2** |

The MLA KV and the block-aligned KDA states came back from the NVMe slab through the page cache and reproduced both
the answer and a continuation that depends on the restored KDA state; nothing was recomputed (TTFT 0.14 s vs
1.33 s cold).

## 4. What is left, and why it is not patched here

* **The 5 draft layers** (+21 % per block): the draft attends over the whole context, so its KV is per token. Only a
  smaller draft KV format (FP4 latents) would shrink it; FLASHINFER_MLA has no such path on B300.
* **18 padding layers**: with 29 MLA and 69 KDA layers every uniform group size pads somewhere, and the cheapest
  place is the KDA side (its blocks are per request, not per token). Removing it needs a second block pool with its
  own stride for the short group - a change across `BlockPool`, the coordinator and the worker's tensor layout for
  ~35 MB per request with RecoverSSM (2 blocks x 18/29 x 27.8 MB); ~5 % of the pool at 64 sessions.
* **Fixed costs (8.7 GiB, 16 % of the pool)**: the profile's peak includes the five aux hidden states the draft
  reads (5 x 8192 x 7168 bf16 = 587 MB, concatenated once more by `combine_hidden_states`) and the 640-token
  rejection sampler; the graph pool is the 640-token decode graphs. Each is a 1-2 % item.
* **Two resident KDA state blocks per request** (the last block-aligned state and the running one) is align mode's
  design: the aligned state is what a later turn's prefix hit resumes from.

## 5. Files

* `tools/k3_b300/patch_vllm_recoverssm_offload.py` - the patch (idempotent, `--check`).
* `Dockerfile.b300-k3` - `vllm/vllm-openai:v0.32.0` + the patch (`vllm-moet-b300:k3-v0.32.0-r1`).
* `docker/b300/run-k3.sh` - the launcher with `USE_RECOVERSSM` and `MAMBA_SSM_CACHE_DTYPE`.
* `tools/k3_b300/kv_model.py` - the budget model (`--kv-gib 46.98` reproduces every row above).
* `tools/k3_b300/k3_code_bench.py`, `k3_matrix.py`, `capacity_probe.py`, `gsm8k_probe.py`, `ref_outputs.py` - the
  measurements.
