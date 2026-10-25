# Kimi-K3 + DSpark on 8x B300 (HGX, sm_103)

The official `vllm/vllm-openai:v0.32.0` serves K3 on B300 as is; this directory holds the one patch the served
configuration needs and the tools behind `docs/k3-b300-dspark-kv.md` (why DSpark cost 35 % of the GPU KV pool and
how the KDA speculative state slots are removed). Image: `Dockerfile.b300-k3`; launcher: `docker/b300/run-k3.sh`.

| file | role |
|---|---|
| `patch_vllm_recoverssm_offload.py` | idempotent, anchored patcher for `vllm/config/vllm.py`: admits `--use-replayssm` (K3's KDA RecoverSSM verify, `use_kda_recoverssm`) next to a KV connector. v0.32.0 rejects the pair for Mamba2 ReplaySSM's deferred state store; K3 commits the accepted KDA state in the step's postprocess, before the connector runs. `--check` verifies without writing. |
| `kv_model.py` | the engine's KV accounting for K3 (+ DSpark) in closed form: hybrid block inflation, layer grouping and padding, bytes per block, per-request blocks, `GPU KV cache size`. `--kv-gib 46.98` reproduces the served runs to the token. |
| `k3_code_bench.py` | single-stream code decode with DSpark counters (the `fraser_bench.py` measure through the chat API; accepted length = 1 + accepted / drafts). |
| `k3_matrix.py` | burst matrix (prompt size x concurrency) through the chat API, as `tools/sm120_perf/openai_matrix.py`. |
| `capacity_probe.py` | N concurrent long decodes with P-token prompts: peak requests running at once, preemptions, KV usage - the capacity measure the matrix cannot give when prefill outlasts decode. |
| `gsm8k_probe.py` | GSM8K first-N accuracy (greedy, concurrent) over the chat API; caches the test set next to itself. |
| `ref_outputs.py` | greedy reference outputs for four fixed prompts and a `--compare` against a saved run (token parity / divergence point). |
| `prefix_probe.py` | cold vs GPU-warm TTFT and output on a long prompt, prefix-hit counters, and a multi-turn continuation (with `reasoning_content` sent back, as K3 requires). |
| `evict_probe.py` | KV offload round trip: cold, GPU-warm, evicted by a stream of large prompts, then restored through the connector; external hit counters, TTFT and output / continuation parity. Run against `--kv-cache-memory <small>` so the eviction is quick. |

Build and serve (hgx-02 paths; every knob is documented in the launcher header):

```bash
DOCKER_BUILDKIT=0 docker build -f Dockerfile.b300-k3 -t vllm-moet-b300:k3-v0.32.0-r1 .
IMAGE=vllm-moet-b300:k3-v0.32.0-r1 MODEL_DIR=/data/models/Kimi-K3 CACHE_DIR=/data/cache/vllm-k3 \
  GPU_MEM_UTIL=0.95 KV_OFFLOAD=disk KV_OFFLOAD_GIB=12288 KV_OFFLOAD_DIR=/data/kvcache \
  SPEC_TOKENS=4 DRAFT_DIR=/data/models/Kimi-K3-DSpark MAX_CUDAGRAPH_CAPTURE_SIZE=640 \
  USE_RECOVERSSM=1 docker/b300/run-k3.sh
```
