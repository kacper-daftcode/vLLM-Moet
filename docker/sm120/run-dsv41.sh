#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 4x RTX PRO 6000 (sm_120), TP4, DSpark k=5, fp8 KV, 512K context -- the
# validated deployment (docs/sm120-deploy.md, docs/dsv41-sm120-port.md). Image: Dockerfile.sm120-dsv41.
#
# Required:  MODEL_DIR   directory with the official DeepSeek-V4.1-Flash checkpoint
# Optional:  IMAGE (vllm-moet-sm120:dsv41-0909)  NAME (ds41-flash)  GPUS (0,1,2,3)  TP (4)  PORT (8001)
#            BIND (127.0.0.1: only local; "" = all interfaces)
#            CACHE_DIR   host dir with two subdirs mounted at /root/.cache and /root/.deep_gemm
#                        (FlashInfer autotune + JIT, DeepGEMM JIT; first start ~25 min, later ~17 min)
#            MAX_MODEL_LEN (524288)  MAX_NUM_SEQS (8)  MAX_NUM_BATCHED_TOKENS (4096)
#            GPU_MEM_UTIL (0.94)     0.94 with a populated FlashInfer autotune cache; use 0.92 for
#                        the very first start (the autotune sweep after KV allocation OOMs at 0.95)
#            SPEC_TOKENS (5)         DSpark draft tokens; 0 disables the drafter (-2.4 GiB/GPU, ~106 tok/s)
#            LANGUAGE_ONLY (0)       1 = --language-model-only (no vision encoder, +0.3 GiB KV)
#            PROFILER (0)            1 = torch profiler endpoints, traces in PROFILE_DIR
#            NCCL_P2P_LEVEL (SYS)    P2P over PCIe works in the KVM guests NCCL classifies as PHB
#            EXTRA_ARGS / EXTRA_DOCKER_ARGS   appended to `vllm serve` / `docker run`
# Runtime kill switches (env, no rebuild): VLLM_MOET_SM120_GEMV=0 (CUTLASS instead of the dense MXFP8
# GEMV), VLLM_MOET_GEMV_IMPL=v1 (the pre-2026-09-20 GEMV kernel instead of v3), VLLM_MOET_SM120_GEMV_BMM=0
# (BF16 emulation for wo_a). enable_adaptive_verification must stay
# false on this path (DeepseekV4IndexerBackend does not support it on sm_120).
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR (DeepSeek-V4.1-Flash checkpoint directory) is required}"
IMAGE="${IMAGE:-vllm-moet-sm120:dsv41-0909}"
NAME="${NAME:-ds41-flash}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
PORT="${PORT:-8001}"
BIND="${BIND-127.0.0.1:}"
CACHE_DIR="${CACHE_DIR:-$PWD/cache-$NAME}"
SERVED_NAME="${SERVED_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
LANGUAGE_ONLY="${LANGUAGE_ONLY:-0}"
PROFILER="${PROFILER:-0}"
PROFILE_DIR="${PROFILE_DIR:-$PWD/profiles-$NAME}"
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"

[[ -f "$MODEL_DIR/config.json" ]] || { echo "no config.json in MODEL_DIR=$MODEL_DIR" >&2; exit 1; }
mkdir -p "$CACHE_DIR/dot-cache" "$CACHE_DIR/deep_gemm"

ARGS=()
if [[ "$SPEC_TOKENS" != "0" ]]; then
  ARGS+=(--speculative-config
    "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false}")
fi
[[ "$LANGUAGE_ONLY" == "1" ]] && ARGS+=(--language-model-only)
MOUNTS=(-v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR/dot-cache:/root/.cache" -v "$CACHE_DIR/deep_gemm:/root/.deep_gemm")
if [[ "$PROFILER" == "1" ]]; then
  mkdir -p "$PROFILE_DIR"
  MOUNTS+=(-v "$PROFILE_DIR:/profiles")
  ARGS+=(--profiler-config.profiler torch --profiler-config.torch_profiler_dir /profiles
         --profiler-config.torch_profiler_with_stack false)
fi

if docker ps -q -f name="^${NAME}$" -f status=running | grep -q .; then
  echo "container $NAME already running - stop it first (docker stop -t 60 $NAME)"; exit 0
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true

# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart no \
  --gpus "\"device=${GPUS}\"" \
  --ipc host --shm-size 32g \
  -p "${BIND}${PORT}:${PORT}" \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  "${MOUNTS[@]}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
  ${EXTRA_DOCKER_ARGS} \
  "$IMAGE" \
  --model /model --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-cudagraph-capture-size 64 \
  "${ARGS[@]}" \
  ${EXTRA_ARGS} \
  --host 0.0.0.0 --port "$PORT"

echo "container $NAME started; logs: docker logs -f $NAME ; API: http://${BIND:-0.0.0.0:}${PORT}/v1"
echo "expect in the log: 'Using Sm120GemvMxfp8BmmLinearKernel for MXFP8 GEMM', 'wo_a stays MXFP8 on sm_120'"
