#!/usr/bin/env bash
# Qwen3.8-Flash-Next-FP8 on 4x RTX PRO 6000 (sm_120), TP4, MTP k=3 -- the validated deployment
# (docs/sm120-deploy.md, tools/qwen38_sm120/README.md). Image: Dockerfile.sm120-qwen38.
#
# Required:  MODEL_DIR   directory with the Qwen3.8-Flash-Next-FP8 checkpoint
# Optional:  IMAGE (vllm-moet-sm120:qwen38-20073)  NAME (qwen38)  GPUS (0,1,2,3)  TP (4)  PORT (8000)
#            BIND (127.0.0.1: only local; "" = all interfaces)
#            CACHE_DIR   host dir mounted at /root/.cache (vLLM torch.compile cache, FlashInfer JIT;
#                        the first start compiles ~10 min, later starts ~4 min). Keep one cache dir
#                        per image tag: the compile-cache hash does not see the patched GEMM path.
#            MAX_MODEL_LEN (262144)  MAX_NUM_SEQS (32)  MAX_NUM_BATCHED_TOKENS (8192)
#            GPU_MEM_UTIL (0.92)     KV_CACHE_MEMORY (34359738368 = 32 GiB; "" lets vLLM decide --
#                        do not: the start-up profile of the GPU-PLE path over-reports 35 GiB of
#                        activations and leaves ~5 GiB of KV)
#            SPEC_TOKENS (3)         MTP draft tokens; 0 disables speculative decoding
#            PROFILER (0)            1 = torch profiler endpoints (/start_profile, /stop_profile), traces in PROFILE_DIR
#            NCCL_P2P_LEVEL (SYS)    P2P over PCIe works in the KVM guests NCCL classifies as PHB; SHM otherwise
#            EXTRA_ARGS / EXTRA_DOCKER_ARGS   appended to `vllm serve` / `docker run`
# Runtime kill switches (env, no rebuild): VLLM_MOET_SM120_LL_GEMM=0 (cuBLAS instead of the skinny
# GEMM -- use a fresh CACHE_DIR), VLLM_MOET_SM120_MOE_GEMV=0 (Triton fused_moe), VLLM_PLE_CPU_OFFLOAD=1
# (PLE table in the CPU worker; then KV_CACHE_MEMORY="" and --gpu-memory-utilization decides).
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR (Qwen3.8-Flash-Next-FP8 checkpoint directory) is required}"
IMAGE="${IMAGE:-vllm-moet-sm120:qwen38-20073}"
NAME="${NAME:-qwen38}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
PORT="${PORT:-8000}"
BIND="${BIND-127.0.0.1:}"
CACHE_DIR="${CACHE_DIR:-$PWD/cache-$NAME}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY-34359738368}"
SPEC_TOKENS="${SPEC_TOKENS:-3}"
PROFILER="${PROFILER:-0}"
PROFILE_DIR="${PROFILE_DIR:-$PWD/profiles-$NAME}"
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"

[[ -f "$MODEL_DIR/config.json" ]] || { echo "no config.json in MODEL_DIR=$MODEL_DIR" >&2; exit 1; }
mkdir -p "$CACHE_DIR"

ARGS=()
[[ -n "$KV_CACHE_MEMORY" ]] && ARGS+=(--kv-cache-memory "$KV_CACHE_MEMORY")
if [[ "$SPEC_TOKENS" != "0" ]]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}")
fi
MOUNTS=(-v "$MODEL_DIR:/models/Qwen3.8-Flash-Next-FP8:ro" -v "$CACHE_DIR:/root/.cache")
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
docker run -d --init --name "$NAME" --restart no \
  --gpus "\"device=${GPUS}\"" \
  --ipc host --shm-size 32g \
  -p "${BIND}${PORT}:${PORT}" \
  --cap-add=SYS_PTRACE \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  "${MOUNTS[@]}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e HF_HUB_OFFLINE=1 \
  -e NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
  ${EXTRA_DOCKER_ARGS} \
  "$IMAGE" \
  /models/Qwen3.8-Flash-Next-FP8 \
  --served-model-name Qwen3.8-Flash-Next Qwen3.8-Flash-Next-FP8 Qwen/Qwen3.8-Flash-Next-FP8 \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  "${ARGS[@]}" \
  ${EXTRA_ARGS}

echo "container $NAME started; logs: docker logs -f $NAME ; API: http://${BIND:-0.0.0.0:}${PORT}/v1"
echo "expect in the log: 'low-latency skinny GEMM (sm_120)', 'sm_120 MoE GEMV', 'Using configuration from /opt/vllm-moet/qwen38_sm120/moe_configs'"
