#!/bin/bash
# Serve the TP1 mini model (make_mini_config.py + moe_intermediate_size 576, dummy weights) with the CPU expert tier
# inside the sm120 serving image, one GPU. The patcher runs first (idempotent).
#   usage (container): serve_mini_hybrid.sh MODEL_DIR PORT [extra vllm serve args...]
#   env: DSPARK=0 serves without the drafter; VLLM_MOET_MOE_CPU_TIER=1 VLLM_MOET_CPU_TIER_RESIDENT=8 VLLM_MOET_CPU_TIER_DUMMY_WEIGHTS=1
#        VLLM_MOET_CPU_TIER_TABLES=/parity VLLM_MOET_CPU_TIER_CPUS=8-15 VLLM_MOET_CPU_TIER_LOG=1
#   e.g. flock -w 3600 /tmp/exl3diff-gpu1.lock docker run --rm --gpus '"device=1"' --name moet-mini-hyb \
#          -e VLLM_MOET_MOE_CPU_TIER=1 -e VLLM_MOET_CPU_TIER_RESIDENT=8 -e VLLM_MOET_CPU_TIER_DUMMY_WEIGHTS=1 \
#          -e VLLM_MOET_CPU_TIER_TABLES=/parity -e VLLM_MOET_CPU_TIER_CPUS=8-15 -e VLLM_MOET_CPU_TIER_LOG=1 \
#          -v /mnt/nvme/models/mini-hyb576:/model -v /mnt/nvme/cache/ds41/dot-cache:/root/.cache \
#          -v /mnt/nvme/vLLM-Moet-cpu:/repo:ro -v /mnt/nvme/vLLM-Moet-cpu/internal/dsv41-5090-cpu/parity:/parity:ro \
#          -p 127.0.0.1:8031:8031 --entrypoint bash vllm-moet-sm120:dsv41-nightly-df3bd094-hyb1 \
#          /repo/tools/dsv41_5090/cpu_tier/serve_mini_hybrid.sh /model 8031
set -u
MODEL_DIR="${1:?model dir}"
PORT="${2:?port}"
shift 2
python3 /repo/tools/dsv41_5090/cpu_tier/patch_vllm_moe_cpu_tier.py || exit 1
SPEC=(--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false,"moe_backend":"b12x"}')
[ "${DSPARK:-1}" = "1" ] || SPEC=()
exec vllm serve "$MODEL_DIR" --load-format dummy --served-model-name mini --seed 0 \
  --tensor-parallel-size 1 --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --kv-cache-dtype nvfp4_ds_mla \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.3}" --max-model-len 4096 --max-num-seqs 1 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-512}" --max-cudagraph-capture-size 16 \
  --language-model-only --attention-config '{"indexer_kv_dtype":"mxfp4"}' --moe-backend b12x \
  "${SPEC[@]}" "$@" --host 0.0.0.0 --port "$PORT"
