# Kimi-K3 + DSpark on 8x B300 (HGX, sm_103)

The official `vllm/vllm-openai:v0.32.0` serves K3 on B300 as is; this directory holds the two patches the served
configuration needs (RecoverSSM next to the KV connector; the RedHat sliding-window draft with prefix caching, served at
k = 5 since 2026-10-28) and the tools behind `docs/k3-b300-dspark-kv.md` (why DSpark cost 35 % of the GPU KV pool and
how the KDA speculative state slots are removed). Image: `Dockerfile.b300-k3`; launcher: `docker/b300/run-k3.sh`.

| file | role |
|---|---|
| `patch_vllm_recoverssm_offload.py` | idempotent, anchored patcher for `vllm/config/vllm.py`: admits `--use-replayssm` (K3's KDA RecoverSSM verify, `use_kda_recoverssm`) next to a KV connector. v0.32.0 rejects the pair for Mamba2 ReplaySSM's deferred state store; K3 commits the accepted KDA state in the step's postprocess, before the connector runs. `--check` verifies without writing. |
| `patch_vllm_swa_draft_dcp.py` | (images `-r2`..`-r5`) a replicated sliding-window GQA draft (`RedHatAI/Kimi-K3-speculator.dspark`) next to the DCP-sharded MLA target: the KV layout decided on the unified page sizes (layer-compact, TOKENSPEED_MLA keeps its kernel-block split), the target's CUDA graph support judged on the target's layers only, the draft's sliding-window block on trtllm-gen's 1024-token pages instead of 64-token blocks padded to the MLA page (`VLLM_MOET_SWA_DRAFT_BLOCK` caps it), and (`-r3`) a `SlidingWindowManager` with hash-unit (fine-grained) lookups and a copy-on-write partial tail, so the window's prefix hits land on the prompt-end hash boundary where K3's KDA groups keep their state - the `-r2` image lost every prefix hit with this draft (docs §3b) - and (`-r5`) the scheduler's Mamba "align" chunking on the Mamba group's block from the KV cache config: `cache_config.block_size` had been pulled down to the window's 1024 (the KDA lost its checkpoints on `-r3`) and `cache_config.mamba_block_size` is stale in the engine core (`-r4` split every 16 tokens). Seven anchored hunks over five files; the first three only reached when such a draft is configured, the manager's fine-grained branch only next to a Mamba "align" group. |
| `multiturn_accept.py` | agent-style turns on a long context: prefix hits, TTFT, accepted length and tok/s per turn (whether a draft keeps its acceptance after a prefix hit). |
| `accept_suite.sh`, `make_longctx_prompt.py`, `prompts/` | DSpark acceptance per prompt type (Fraser code, 1500-word prose, a number-theory proof with `reasoning_effort: high`, a 30K-token document + coding task): accepted length, per-position acceptance and tok/s through `k3_code_bench.py`. |
| `kv_model.py` | the engine's KV accounting for K3 (+ DSpark) in closed form: hybrid block inflation, layer grouping and padding, bytes per block, per-request blocks, `GPU KV cache size`. `--kv-gib 46.98` reproduces the served runs to the token. |
| `k3_code_bench.py` | single-stream code decode with DSpark counters (the `fraser_bench.py` measure through the chat API; accepted length = 1 + accepted / drafts). |
| `k3_matrix.py` | burst matrix (prompt size x concurrency) through the chat API, as `tools/sm120_perf/openai_matrix.py`. |
| `capacity_probe.py` | N concurrent long decodes with P-token prompts: peak requests running at once, preemptions, KV usage - the capacity measure the matrix cannot give when prefill outlasts decode. |
| `gsm8k_probe.py` | GSM8K first-N accuracy (greedy, concurrent) over the chat API; caches the test set next to itself. |
| `ref_outputs.py` | greedy reference outputs for four fixed prompts and a `--compare` against a saved run (token parity / divergence point). |
| `prefix_probe.py` | cold vs GPU-warm TTFT and output on a long prompt, prefix-hit counters, and a multi-turn continuation (with `reasoning_content` sent back, as K3 requires). |
| `prefill_bench.py` | fresh-prefill throughput over the chat API: unique prompts (nonce + random words, no prefix hits), `max_tokens=1`, single stream at 1K/2K/8K/16K/64K tokens (median of 2) and C=4 / C=8 streams of 16K (aggregate tok/s, TTFT median); one summary line + JSON. The prefill counterpart of `k3_code_bench.py`'s steps/s. |
| `profile_prefill.py` | torch-profiler capture of one fresh prefill (server started with `--profiler-config`): `/start_profile`, one 16K and one 64K prompt, `/stop_profile`, then the rank-0 trace by category (MoE / dense GEMM / sparse MLA / indexer / KDA / collectives / norm), GPU busy vs idle, kernels per chunk, top kernels, us of GPU time per prompt token. `--analyze-only TRACE` re-reads a saved trace. |
| `evict_probe.py` | KV offload round trip: cold, GPU-warm, evicted by a stream of large prompts, then restored through the connector; external hit counters, TTFT and output / continuation parity. Run against `--kv-cache-memory <small>` so the eviction is quick. |

Build and serve (hgx-02 paths; every knob is documented in the launcher header):

```bash
DOCKER_BUILDKIT=0 docker build -f Dockerfile.b300-k3 -t vllm-moet-b300:k3-v0.32.0-r5 .
IMAGE=vllm-moet-b300:k3-v0.32.0-r5 MODEL_DIR=/data/models/Kimi-K3 CACHE_DIR=/data/cache/vllm-k3 \
  GPU_MEM_UTIL=0.95 KV_OFFLOAD=disk KV_OFFLOAD_GIB=12288 KV_OFFLOAD_DIR=/data/kvcache \
  SPEC_TOKENS=5 DRAFT_DIR=/data/models/Kimi-K3-speculator-redhat DRAFT_ATTENTION_BACKEND="" \
  MAX_CUDAGRAPH_CAPTURE_SIZE=768 USE_RECOVERSSM=1 docker/b300/run-k3.sh
```

## Kimi-K3.1-Pro (`Dockerfile.b300-k31`)

The K3.1 image is `vllm/vllm-openai:kimi-k31` plus three steps, each described in the Dockerfile header. The K3
patchers above do not apply to it: their fixes are upstream in that line.

| file | role |
|---|---|
| `patch_vllm_k31_experts_grid.py` | step 1: idempotent, anchored patcher for the specialized decode's persistent routed-experts kernel. Its grid table is sized for 152 SMs and the path refuses to start on a 148-SM B300; the patch caps the table at the device's SM count. `--check` verifies without writing. |
| `k31/r2.patch` | step 2 (build arg `VLLM_MOET_K31_R2_PATCH`, images `-r2` and later): the DSpark drafter and the MoE tail of the specialized path, ten files, one of them new. |
| `k31/r3.patch` | step 3 (build arg `VLLM_MOET_K31_R3_PATCH`, images `-r3`): the Markov head sharded across the TP ranks and the sampler chain, eight files; cut against the files step 2 leaves, so it needs step 2. |
| `apply_file_patches.py` | applies a file patch set: unified diffs under a manifest with the sha256 of every file before and after. A file whose hash is neither of the two stops the build instead of being patched approximately; hunks apply without fuzz; nothing is written unless the whole set applies and every result has the recorded hash; a second run changes nothing; `--check` verifies without writing. |
| `make_file_patches.py` | cuts a set from two directory trees with the same relative paths (`vllm/` copied out of an image before and after a change) and emits it only when the applier reproduces the second tree from the first, byte for byte. |

Runtime switches of steps 2 and 3. They are read from the server's environment at start (through the launcher:
`EXTRA_DOCKER_ARGS="-e NAME=0"`), default to on, and `=0` runs the stock code of that part:

| switch | step | default | `=0` |
|---|---|---|---|
| `VLLM_MOET_DRAFT_SWIGLU` | 2 | the gate_up GEMM of the drafter's MLP with a SwiGLU epilogue, one launch per layer | the stock GEMM + `SiluAndMul` |
| `VLLM_MOET_DRAFT_CONTEXT_KV` | 2 | the drafter's context KV (grouped norm, RoPE, FP8 quantization, paged insert) in one launch | the stock chain of six launches |
| `VLLM_MOET_MARKOV_SHARD` | 3 | the vocab-sharded Markov step wherever its conditions hold (a drafter mailbox, BF16 draft logits, FP32 Gumbel keys, the lm_head layout it expects); the server log names the reason otherwise | step 2's replicated kernel |
| `VLLM_MOET_FUSED_PREP` | 3 | positions, sequence lengths, input ids and logits indices of a step in one launch | the two stock launches |

The rest of the two sets has no switch (the causal draft block on the DCP drafter attention, the schedule of the
Markov step, the 3-4-row MoE tail, the sampler's device-side parameters and its fused resample tail); a build with
`--build-arg VLLM_MOET_K31_R3_PATCH=0`, and `VLLM_MOET_K31_R2_PATCH=0` as well, leaves them out. The image's
`com.vllm-moet.k31.r2` / `.r3` labels say which sets it carries.

```bash
DOCKER_BUILDKIT=0 docker build -f Dockerfile.b300-k31 -t vllm-moet-b300:k31-266e88c0-r3 .
python3 -m unittest bench/tests/test_file_patches.py     # applier, maker, the two sets against the Dockerfile (CPU)
# a further set: vllm/ copied out of the image before and after the change (docker cp), then
python3 tools/k3_b300/make_file_patches.py --name k31-r4 --before before/ --after after/ \
  --out tools/k3_b300/k31/r4.patch
```
