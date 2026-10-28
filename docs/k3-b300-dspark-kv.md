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

## 3b. The other draft: RedHatAI/Kimi-K3-speculator.dspark (2026-10-25)

The Inferact draft's 5 MLA layers attend over the whole context, so their KV is per token (+21 % on every block).
RedHat's speculator is a 5-layer Qwen3-style GQA draft (16 KV heads x head_dim 64, 2 per GPU at TP8) with a
**2048-token sliding window**, trained for k = 8 on the aux hidden states of layers 24/48/72/88/92: its KV is a
`SlidingWindowSpec`, a few blocks per request whatever the context. v0.32.0 cannot start it next to the DCP-sharded
MLA target; `tools/k3_b300/patch_vllm_swa_draft_dcp.py` (image `-r2`; `-r3` adds step 4 below) fixes the three
stops, in order:

1. `resolve_kv_cache_layout` sees a sharded target and a replicated draft with different page sizes and demands a
   block-outer layout (BLHNC), which TOKENSPEED_MLA cannot split into its kernel blocks ("packed 1664-token KV cache
   blocks cannot be split into"). The hybrid grouping pads the draft's page to the MLA page anyway, so the patch
   decides the requirement on the unified page sizes: one page size, layer-compact LBHNC, the split as today.
2. The target's CUDA graph support was computed over every attention group including the draft's; FlashInfer under
   the target's config (DCP 8) answers UNIFORM_SINGLE_TOKEN_DECODE and the target's graphs collapse to NONE. The
   patch asks the target's layers only (the draft has its own graph manager and a DCP 1 config).
3. The sliding-window layer picked 64-token blocks because FlashInfer advertises its trtllm-gen pages (128..1024) from
   the *current* model config's head ratio (the MLA target's: 1). Each 64-token block then padded to the 958 KB MLA
   page cost ~290 admission blocks (1.4 GB) per request (`GPU KV cache size: 15,624,648`) and `--prefix-match-unit
   128` rejected the 64-token group. The patch judges the trtllm-gen pages on the layer's own heads: 1024-token blocks,
   13 admission blocks, 2-3 held in decode.

Layer groups become 24 MLA / 69 KDA / 5 SWA -> group size 5: 5 MLA groups (1 padding layer), 14 KDA groups (1), one
SWA group - **20 KV cache groups** of 5 pages (4.79 MB blocks), 1 + 1 padding layers instead of 18.

Same host, same flags (RecoverSSM, fp8 KV, prefix caching, disk offload), one stream, greedy, two reps each; `code`
is the Fraser prompt, `prose` a 1500-word essay, `math` a number-theory proof at `reasoning_effort: high`, `longctx`
a 30K-token design document followed by a coding task (`tools/k3_b300/accept_suite.sh`):

| draft, k | GPU KV tokens | code: tok/s (accepted) | prose | math | longctx 30K | steps/s (code) |
| --- | --- | --- | --- | --- | --- | --- |
| Inferact k = 4 (served) | 21,614,964 | 205.2 (3.33); 219.6 (3.34) in §3 | 142.4 (2.16) | 215.0 (3.27) | 244.1 (3.74) | 61.7-65.8 |
| **RedHat k = 4** | **24,482,869** | 230.3 (3.76) | 162.5 (2.48) | 213.6 (3.27) | 265.1 (4.08) | 61.4 |
| Inferact k = 7 | 19,885,043 | 220.6 (3.96) | 130.5 (2.19) | 215.0 (3.62) | 265.2 (4.51) | 55.6 |
| **RedHat k = 8** | 22,102,814 | 236.3 (4.42) | 145.2 (2.53) | 215.5 (3.79) | 299.5 (5.29) | 53.5 |

Per-position acceptance, code: RedHat k = 4 0.89 / 0.75 / 0.61 / 0.50 vs Inferact k = 4 0.83 / 0.63 / 0.49 / 0.37;
longctx: RedHat k = 4 0.93 / 0.82 / 0.72 / 0.62 vs Inferact 0.90 / 0.75 / 0.61 / 0.48; RedHat k = 8
0.91 / 0.79 / 0.65 / 0.53 / 0.46 / 0.38 / 0.31 / 0.26. The 2048-token window did not hurt the 30K-context case; the
RedHat draft accepts more at every position on every prompt type except `math`, where the two tie at k = 4, and its
step is marginally slower at the same k (5 dense Qwen3 layers plus the 20 groups' bookkeeping). Net per stream at
k = 4: code +12 %, prose +14 %, math 0 %, longctx +9 %, with 13 % more KV pool.

### k sweep and the gates (2026-10-25/26)

Same suite plus the 8K matrix at C = 32 / 64 for each k (`matrix-sweep.log`); Inferact k = 4 rows from §3:

| draft, k | pool M | code | prose | math | longctx | 8K C = 32: /session (total) | 8K C = 64: /session (total) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Inferact k = 4 (served) | 21.61 | 205-220 (3.33) | 142 (2.16) | 215 (3.27) | 244 (3.74) | 57.6 (1869) | 40.7 (2604) |
| RedHat k = 4 | 24.48 | 230 (3.76) | 162 (2.48) | 214 (3.27) | 265 (4.08) | 61.6 (1988) | **44.6 (2840)** |
| RedHat k = 5 | 24.02 | 237 (4.02) | **170 (2.66)** | 225 (3.56) | 284 (4.50) | 60.2 (1939) | 42.5 (2721) |
| RedHat k = 6 | 23.54 | 242 (4.23) | 156 (2.54) | 219 (3.58) | 305 (5.04) | 61.0 (1915) | 40.3 (2574) |
| RedHat k = 7 | 22.53 | 245 (4.42) | 153 (2.55) | **227 (3.81)** | **311 (5.23)** | 58.5 (1851) | 37.4 (2413) |
| RedHat k = 8 | 22.10 | 236 (4.42) | 145 (2.53) | 216 (3.79) | 300 (5.29) | - | - |

Per session at C = 64 falls monotonically with k; k = 4 (pool, concurrency) or k = 5 (prose, math) would be the
served choice. On RedHat k = 4: GSM8K-200 196/200, needle 4/4, 128 x 22K sessions all running at 33 % KV, 0
preemptions, 1486 tok/s aggregate (Inferact: 36 %, 1418); 32K at C = 64: 45.3 tok/s per session.

**Blocker: prefix caching.** With the draft's sliding-window group in prefix caching, `SlidingWindowManager`
"requires block-aligned lookups" and the whole model loses fine-grained hits; measured on RedHat k = 4, a 30K prompt
resent hit **0 of 20,348 tokens** (TTFT 1.10 s instead of 0.11 s), a second turn 0 as well. Three variants were
tried (`patch_vllm_swa_draft_dcp.py` step 4 and its `VLLM_MOET_SWA_DRAFT_*` switches):

| variant | pool | resend of a 30K prompt | agent turns 2-4 on a 16-20K context |
| --- | --- | --- | --- |
| Inferact k = 4 (served), for reference | 21.61M | 20,224 of 20,354 hits, TTFT 0.11 s | 15,872 / 16,512 / 17,152 of ~17K; accepted length 3.7-4.7 |
| window in prefix cache, 1024-token blocks (default of steps 1-3) | 24.48M | 0 hits | 0 hits |
| window in prefix cache, 128-token blocks (`VLLM_MOET_SWA_DRAFT_BLOCK=128 VLLM_MOET_SWA_DRAFT_REPLAY=0`) | 19.18M | 0 hits | 0 / 15,872 / 15,872 of ~17K |
| window as per-request scratch, hits capped 2048 tokens below the prompt end (step 4, default) | 24.48M | 0 hits | 0 / 11,648 / 15,872 of ~17K; accepted length 4.0-4.9 after a hit |

The reason is K3-specific: the KDA groups retain their aligned state sparsely (at a finished request's prompt-end
hash boundary and a few block boundaries), so a hit must land exactly there; the draft's window blocks (variants 1-2)
or the 2048-token cap (variant 3) move the admissible position away from it.

**Where the groups keep state** (read off the managers with a CPU-side emulation of the scheduler loop, 2026-10-26;
MLA block 13,312 tokens = 1,664 x DCP 8, KDA block 1,664, window block 1,024, hash unit 128, EAGLE-style drop on
every group - with a draft whose spec carries no eagle marker the coordinator flags all of them). After a 10,000-token
prompt decoded to 14,000 tokens: MLA `9984` (the prompt-end partial entry, kept as a secondary key after the block
is promoted) and `13312` (a full block); KDA `6656` (a chunk end), `9984` (the prefill checkpoint), `13312` (a block
boundary crossed in decode); so a hit exists only at a prompt-end boundary (the EAGLE drop then lands it one unit
below: `9856`) or at a scheduler-block multiple crossed in decode (and only without the drop). Every group retains
the prompt-end boundary; the window must be reachable exactly there.

**Step 4 (image `-r3`, 2026-10-26): `SlidingWindowManager` with fine-grained lookups.** The manager declares
`supports_fine_grained_hash_lookup`, so the coordinator keeps the hash-unit alignment, and its lookup accepts a hit
ending at any hash boundary P whose window blocks are cached: whole blocks under their full-block hash, the block P
falls in under any prefix hash at or past P (its full-block hash, or the partial entry that `cache_blocks` now
registers at the prompt's final hash boundary, the way `FullAttentionManager._cache_partial_tail_block` does, and
re-registers after the block is promoted). Under the EAGLE drop the proof must reach one unit past the hit (the
coordinator re-queries a verified eagle group without the drop; the same keys satisfy it). The partial tail block is
shared copy-on-write through the base class's `_partial_hit_reqs` path - one reserved block, one block copy on the
worker, as for the MLA and KDA partial tails. The window a replaying request needs (the blocks before the prompt-end
boundary) stays referenced by its request until it finishes, instead of being nulled when it leaves the attention
window, so it ages out of the LRU together with the request's MLA/KDA blocks and not during a long decode; it fits
the per-request admission cap (window + one in-flight chunk = 13 blocks, 2-3 held in decode). The fine-grained
branch is entered only when the coordinator aligns hits below the block size, which needs a Mamba "align" group;
block-aligned models and Gemma-style full+SWA hybrids keep the upstream code path. Open on the offload tier:
`SimpleCPUOffload` copies the prompt-end partial tail only for the full-attention group (`fa_gidx`), so after a GPU
eviction the external hit with this draft lands on the prompt-end boundary only once the producer's decode has
promoted the window's tail block to a full one (its secondary key travels with it); otherwise on the window's last
full block below. GPU hits are unaffected.

Emulated against the real `KVCacheManager` (the K3 group layout, chunked prefill through the scheduler's Mamba
split, spec-decode steps with rejections, 220-block pools with overlapping prefixes): a 20,354-token prompt resent
while its producer still decodes hits 20,224 (= prod's position) with CoW copies in the MLA, KDA and window groups;
a second turn hits the same boundary; its resend hits its own boundary (21,120); pool accounting balances after
every request is freed.

**Measured 2026-10-26/27 (`-r3`, `/root/work/k3-test/night-rh5.sh`, k = 5 then k = 4; prod = Inferact k = 4):**

| | prod Inferact k = 4 | RedHat k = 5 `-r3` | RedHat k = 4 `-r3` |
| --- | --- | --- | --- |
| GPU KV pool (graphs 640 / 768 / 640) | 21.61M | 24.02M | 24.48M |
| 30K resend: hits, warm TTFT | 20,096 / 20,347, 0.09 s | 20,096 / 20,350, 0.12 s | 20,096 / 20,350, 0.12 s |
| turn 2 after a 120-token answer | 20,096, 0.15 s | 20,096, 0.44 s | 20,096, 0.39 s |
| agent turns 2 / 3 / 4 on 16K (600-token answers) | 15,872 / 16,512 / 17,152 | **0 / 14,976 / 17,152** | **0 / 14,976 / 17,152** |
| accepted length after a hit (turns 2-4) | 4.81 / 4.29 / 3.79 | 5.82 / 5.50 / 3.95 | 4.90 / 4.74 / 3.94 |
| code / prose / math / 30K+code (tok/s, accepted) | 205-220 (3.3) / 142 / 215 / 244 | 263 (4.13) / 162 / 226 / 292 (4.62) | 243 (3.67) / 165 / 222 / 269 |
| 8K C = 32 / 64 per session | 57.6 / 40.7 | 61.7 / 42.6 | 62.7 / 43.8 |
| 32K C = 64 end-to-end (64 x 1,024 tokens) | - | 535 tok/s (122 s) | 540 tok/s (121 s) |
| 128 x 15K, 3,000 output tokens | (22K: 1,418 tok/s, 36 % KV) | 1,654 tok/s, 35 % KV, 0 preemptions | 1,775 tok/s, 35 % KV, 0 |
| GSM8K-200 / needle | 196 / 4/4 | 195 / 4/4 | 196 / 4/4 |

The window's own hits worked (resend, turn 2 after a short answer, turn 4), but the agent session lost turns 2-3.
The instrumented run of 2026-10-27 (per-group cached positions at every lookup, `/root/work/k3-test/dbg-r3/`)
showed the window group complete at every position (`11264P .. 15360P, 16000S, 16384P`: full keys plus the
secondary partial key after promotion) and the **KDA groups empty**: no prompt-end checkpoint for the 16,006-token
turn 1, no chunk-end states, and for turn 2 one state keyed at 14,976 that held the state after 14,336 tokens.
Cause (step 5): with the window group prefix-cacheable, `cache_config.block_size` becomes the minimum over the
cacheable groups - the window's 1,024 instead of the 1,664-token Mamba/attention block - and the scheduler's Mamba
"align" chunking reads it: chunks ended at 7,168 / 14,336 / 15,360 instead of 6,656 / 13,312, so chunk-end states
no longer sat on Mamba block boundaries (and one was hashed under the next boundary's key), and the scheduler's
checkpoint-validity check (on 1,024) disagreed with the manager's (on 1,664) for that chunk geometry, so the
prompt-end checkpoint was not taken. A resend of a 20,350-token prompt happened to have a valid geometry, which is
why the resend probes passed. The worker's `mamba_hybrid` state already reads `cache_config.mamba_block_size` for
exactly this reason; `-r4` makes the scheduler do the same (one hunk). In the emulation, the engine's chunk budget
(8,187 = 8,192 - 5 draft slots) with a 1,024-token split reproduces the loss (turn 2: KDA 14,976, not 15,872); with
the Mamba block it does not. This is an upstream bug for any Mamba "align" model with a prefix-cacheable draft
group below the Mamba block (the DFlash case the worker comment names).

**Verdict (2026-10-28):** the RedHat draft is faster and leaner in KV; with steps 4-5 its window takes the same
prefix hits as the MLA/KDA groups and the KDA keeps its checkpoints. The served draft stays Inferact (k = 4,
RecoverSSM) until the `-r4` battery (2026-10-28/29 night window) confirms the agent-turn hits on the GPUs; the
planned successor is RedHat k = 5 (code / math / long-context best, C = 64 within 3 % of k = 4).

## 4. What is left, and why it is not patched here

* **The 5 draft layers** (+21 % per block) with the Inferact draft: it attends over the whole context, so its KV is
  per token. The RedHat draft (§3b) is the way around it (+13 % pool at k = 4); with step 4 (`-r3`) its
  sliding-window group takes fine-grained prefix hits, pending the night measurement.
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
* `tools/k3_b300/patch_vllm_swa_draft_dcp.py` - the RedHat draft under DCP (§3b steps 1-3), the fine-grained
  `SlidingWindowManager` (step 4) and the scheduler's Mamba chunking on the Mamba block (step 5); idempotent, `--check`.
* `Dockerfile.b300-k3` - `vllm/vllm-openai:v0.32.0` + the patches (`vllm-moet-b300:k3-v0.32.0-r1` = RecoverSSM
  only, served; `-r4` = both, the RedHat draft candidate; `-r3` lacks step 5).
* `docker/b300/run-k3.sh` - the launcher with `USE_RECOVERSSM` and `MAMBA_SSM_CACHE_DTYPE`.
* `tools/k3_b300/kv_model.py` - the budget model (`--kv-gib 46.98` reproduces every row above).
* `tools/k3_b300/k3_code_bench.py`, `k3_matrix.py`, `capacity_probe.py`, `gsm8k_probe.py`, `ref_outputs.py` - the
  measurements.
