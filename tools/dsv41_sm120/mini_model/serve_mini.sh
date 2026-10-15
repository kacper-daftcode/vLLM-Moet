#!/bin/bash
# Serve the 1/5-scale dummy-weight DeepSeek-V4.1 (make_mini_config.py) inside the serving image on 4 GPUs next to a
# serving job: TP4, nvfp4 KV, MXFP4 indexer, DSpark, the served flags at small sizes, ~3 GiB per GPU plus the KV pool
# (GPU_MEM_UTIL, default 0.065 = 6.2 GiB on a 96 GB card; the profile run needs ~4.7 GiB before any KV).
#   usage (container): serve_mini.sh MODEL_DIR PORT [extra vllm serve args...]
#   e.g. docker run --rm --gpus '"device=0,1,2,3"' --ipc host -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS \
#          -v $MINI:/model -v $CACHE:/root/.cache -v $REPO/tools/dsv41_sm120:/t:ro -p 127.0.0.1:8011:8011 \
#          --entrypoint bash IMAGE /t/mini_model/serve_mini.sh /model 8011 --enable-dbo --dbo-prefill-token-threshold 512
# PATCH=/t/patch_vllm_prefill_dbo_sm120.py applies a patcher to the container's tree first (for images built without it).
set -u
MODEL_DIR="${1:?model dir}"
PORT="${2:?port}"
shift 2
[[ -n "${PATCH:-}" ]] && { python3 "$PATCH" || exit 1; }
exec vllm serve "$MODEL_DIR" --load-format dummy --served-model-name mini \
  --tensor-parallel-size 4 --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --kv-cache-dtype nvfp4_ds_mla \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.065}" --max-model-len 8192 --max-num-seqs 4 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" --max-cudagraph-capture-size 16 \
  --language-model-only --attention-config '{"indexer_kv_dtype":"mxfp4"}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' \
  --profiler-config.profiler torch --profiler-config.torch_profiler_dir "${PROFILE_DIR:-/profiles}" \
  --profiler-config.torch_profiler_with_stack false \
  "$@" --host 0.0.0.0 --port "$PORT"
