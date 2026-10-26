#!/bin/bash
# In-container entry of the hybrid (CPU expert tier) serve: apply the patcher to the image's vLLM / FlashInfer, then
# exec vllm serve. Started by docker/sm120/run-dsv41-5090.sh; the tier's own switches arrive as VLLM_MOET_CPU_TIER_*
# environment (see vllm_moet_cpu_tier.py).
#   usage: serve_hybrid.sh MODEL_DIR PORT [extra vllm serve args...]
#   env: TP (4) SERVED_NAME SPEC_TOKENS (5; 0 = no DSpark) GPU_MEM_UTIL (0.92) MAX_MODEL_LEN (65536) MAX_NUM_SEQS (1)
#        MAX_NUM_BATCHED_TOKENS (512) CAPTURE_SIZE (8) LANGUAGE_ONLY (1) LOAD_FORMAT (auto) HYBRID_PATCH_AT_START (1)
set -u
MODEL_DIR="${1:?model dir}"
PORT="${2:?port}"
shift 2
# the patcher: from the mounted worktree at every start (the dev flow; idempotent, rebuilds libcpu_tier.so, ~60 s) unless
# HYBRID_PATCH_AT_START=0 (the image has the layer of Dockerfile step 28: /opt/vllm-moet/cpu_tier + the patched packages)
if [ "${HYBRID_PATCH_AT_START:-1}" = "1" ]; then
  python3 "${HYBRID_SRC:-/repo/tools/dsv41_5090}/cpu_tier/patch_vllm_moe_cpu_tier.py" || exit 1
else
  python3 -c "import vllm_moet_cpu_tier" || { echo "the image has no baked cpu tier (Dockerfile step 28); set HYBRID_PATCH_AT_START=1 with the worktree mounted"; exit 1; }
fi
ARGS=()
SPEC_TOKENS="${SPEC_TOKENS:-5}"
if [ "$SPEC_TOKENS" != "0" ]; then
  ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false,\"moe_backend\":\"b12x\"}")
fi
[ "${LANGUAGE_ONLY:-1}" = "1" ] && ARGS+=(--language-model-only)
[ "${LOAD_FORMAT:-auto}" = "auto" ] || ARGS+=(--load-format "$LOAD_FORMAT")
exec vllm serve "$MODEL_DIR" --served-model-name "${SERVED_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}" --seed 0 \
  --tensor-parallel-size "${TP:-4}" \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --kv-cache-dtype nvfp4_ds_mla --attention-config '{"indexer_kv_dtype":"mxfp4"}' --moe-backend b12x \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}" --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-512}" \
  --max-cudagraph-capture-size "${CAPTURE_SIZE:-8}" \
  "${ARGS[@]}" "$@" --host 0.0.0.0 --port "$PORT"
