#!/bin/bash
# Boot the shared 1M diagnostic server (recipe pro6000x4-tp4-dcp4-1m) as
# container moet-dcp1m on port 8123, GPUs 0-3, image vllm-moet-sm120:v024-r5.
#
# Coordination (see COORDINATION.md): run ONLY after the GPQA bench client
# is gone; if glm-bench sits idle it must be removed first to free ~600 GiB
# of host RAM (the user's harness re-creates it on demand).
#
# Usage: boot_dcp1m.sh [--dcp 4] [--mml 1048576] [--kv-bytes 9126805504]
#                      [--tau 0.60] [--spec-k 2] [--name moet-dcp1m]
#                      [--port 8123] [--gpus 0,1,2,3] [--extra-env K=V ...]
# Variants:
#   Phase 1.1 DCP1@500K:  boot_dcp1m.sh --dcp 1 --mml 524288 --kv-bytes 17179869184
#   Phase 1.3 k=1/k=5:    boot_dcp1m.sh --spec-k 1   /   --spec-k 5
set -euo pipefail

DCP=4; MML=1048576; KV_BYTES=9126805504; TAU=0.60; SPEC_K=2
NAME=moet-dcp1m; PORT=8123; GPUS=0,1,2,3
IMAGE=${MOET_IMAGE:-vllm-moet-sm120:v024-r5}
MODEL_DIR=/root/models/GLM-5.2-NVFP4
PLANES_DIR=/root/moet-planes-glm
EXTRA_ENV=()
while [ $# -gt 0 ]; do case "$1" in
  --dcp) DCP=$2; shift 2;;
  --mml) MML=$2; shift 2;;
  --kv-bytes) KV_BYTES=$2; shift 2;;
  --tau) TAU=$2; shift 2;;
  --spec-k) SPEC_K=$2; shift 2;;
  --name) NAME=$2; shift 2;;
  --port) PORT=$2; shift 2;;
  --gpus) GPUS=$2; shift 2;;
  --extra-env) EXTRA_ENV+=(-e "$2"); shift 2;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done

# Safety: never boot while the GPQA bench client is alive.
if pgrep -f "llm_decode_bench.py" >/dev/null; then
  echo "FATAL: GPQA bench still running (llm_decode_bench.py); see COORDINATION.md" >&2
  exit 1
fi

free_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if [ "$free_gb" -lt 150 ]; then
  echo "FATAL: only ${free_gb} GiB host RAM available; the 1M recipe needs ~160." >&2
  echo "If glm-bench is idle (bench finished, results saved), remove it first:" >&2
  echo "  docker rm -f glm-bench" >&2
  exit 1
fi

DCP_ARGS=()
[ "$DCP" -gt 1 ] && DCP_ARGS=(--decode-context-parallel-size "$DCP")

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --init --name "$NAME" \
  --gpus "\"device=$GPUS\"" --network host --ipc host --shm-size 64g \
  -v "$MODEL_DIR":/model:ro -v "$PLANES_DIR":/planes \
  -v /root/ab-cache-vllm:/root/.cache/vllm \
  -v /root/ab-cache-inductor:/tmp/torchinductor_root \
  -e NCCL_P2P_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_MOE_W2=1 -e VLLM_MOE_W2_DELTA_GB=auto -e VLLM_MOE_W2_DELTA_RESERVE_GB=3 \
  -e VLLM_MOE_W2_GATE=1 -e VLLM_MOE_W2_GATE_TAU="$TAU" \
  -e VLLM_MOE_W2_GATE_TAU_FILE=/tmp/gate_tau \
  -e VLLM_MOE_W2_PLANES_CACHE=/planes \
  "${EXTRA_ENV[@]}" \
  "$IMAGE" \
  --model /model --served-model-name glm-5.2 --trust-remote-code \
  --tensor-parallel-size 4 --disable-custom-all-reduce "${DCP_ARGS[@]}" \
  --kv-cache-dtype nvfp4 --block-size 256 --max-model-len "$MML" \
  --kv-cache-memory-bytes "$KV_BYTES" \
  --gpu-memory-utilization 0.93 --max-num-batched-tokens 2048 --max-num-seqs 2 \
  --no-scheduler-reserve-full-isl \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_K}" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port "$PORT" >/dev/null

echo "[boot] $NAME on port $PORT (dcp=$DCP mml=$MML kv=$KV_BYTES spec_k=$SPEC_K tau=$TAU image=$IMAGE)"
echo "[boot] waiting for readiness (planes boot ~7-15 min)"
for i in $(seq 1 240); do
  st=$(docker inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null || echo gone)
  [ "$st" != running ] && { echo "FATAL: container died; docker logs $NAME" >&2; exit 1; }
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models" && { echo "[boot] ready"; exit 0; }
  sleep 10
done
echo "FATAL: not ready after 40 min" >&2; exit 1
