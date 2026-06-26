#!/usr/bin/env bash
# Generic launcher for vLLM-Moet (DeepSeek-V4-Flash, 2-bit MoE + FP4 delta, SM120).
# Runs an image with the patch BAKED IN (no source mounts). No environment-specific
# paths: pass everything via env / args.
#
# Usage:
#   MODEL=/path/to/DeepSeek-V4-Flash \
#   CUBINS=/path/to/vLLM-Moet/kernels/cubins-sm120 \
#   IMAGE=vllm-moet-sm120:base \
#   ./serve.sh [GPUS] [PORT]
#
#   GPUS  comma list. PP=1 (default) -> tensor-parallel ("0,1" = TP2).
#                     PP>1           -> pipeline-parallel ("4,5,6,7" + PP=4, e.g. 4x 32GB cards).
#   PORT  default 8000.
set -euo pipefail

GPUS="${1:-0}"
PORT="${2:-8000}"
PP="${PP:-1}"
NAME="${NAME:-vllm-moet}"
IMAGE="${IMAGE:-vllm-moet-sm120:base}"
MODEL="${MODEL:?set MODEL=/path/to/DeepSeek-V4-Flash checkpoint}"
# CUBINS is OPTIONAL: the image already bakes the SM120 cubins. Set it only to
# override with a local cubins dir (e.g. rebuilt/modified kernels).
CUBINS="${CUBINS:-}"

# Serving knobs (override via env)
MAX_LEN="${MAX_LEN:-262144}"
UTIL="${UTIL:-0.97}"
DELTA_GB="${DELTA_GB:-1.0}"                 # FP4 hot-expert delta pool (GiB); 0 disables
DELTA_TRACE="${DELTA_TRACE:-0}"             # delta observability: 1=coverage/churn summary, 2=+promote/evict events
DELTA_TRACE_EVERY="${DELTA_TRACE_EVERY:-64}"  # ticks between summaries (see docker logs)
DELTA_POLICY="${DELTA_POLICY:-freq}"        # promotion/eviction: freq (default) or lru
NUM_LAYERS="${NUM_LAYERS:-44}"              # 44 = include the MTP drafter on 2-bit planes
PREFILL_CUBIT="${PREFILL_CUBIT:-1}"
BATCHED="${BATCHED:-1024}"
SEQS="${SEQS:-16}"
MTP="${MTP:-1}"                             # 0 disables speculative decoding
# Confidence-gated FP4 re-forward: low-confidence 2-bit decode steps are re-run with
# routed experts promoted to FP4. GATE=1 to enable (TP/single-GPU only -- see note).
GATE="${GATE:-0}"; GATE_TAU="${GATE_TAU:-}"; GATE_SIGNAL="${GATE_SIGNAL:-max_prob}"
GATE_MAX_PROMOTE="${GATE_MAX_PROMOTE:-0}"; GATE_TRACE="${GATE_TRACE:-0}"
PP_PARTITION="${PP_PARTITION:-}"           # PP only: custom per-rank layer split, e.g. "11,12,12,8"
# Brace inside ${VAR:-{...}} mis-parses, so compute the default separately.
CHAT_KWARGS="${CHAT_KWARGS:-}"
[ -z "$CHAT_KWARGS" ] && CHAT_KWARGS='{"thinking": true}'

# PP>1 pins TP=1 (whole layers per rank); else TP = #GPUs.
if [ "$PP" -gt 1 ]; then TP=1; else TP="$(awk -F, '{print NF}' <<<"$GPUS")"; fi

SPEC=(--speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}')
[ "$MTP" = 1 ] || SPEC=()
XENV=()
[ -n "$GATE_TAU" ]      && XENV+=(-e "VLLM_MOE_W2_GATE_TAU=$GATE_TAU")
[ -n "$PP_PARTITION" ]  && XENV+=(-e "VLLM_PP_LAYER_PARTITION=$PP_PARTITION")
# Multi-GPU without NVLink (e.g. consumer cards over PCIe) needs the custom
# all-reduce disabled; harmless on single GPU.
[ "$TP" -gt 1 ] && CAR=(--disable-custom-all-reduce) || CAR=()
# Cubins are baked into the image; only mount + point at a dir if CUBINS is set.
CUBIN_ARGS=()
[ -n "$CUBINS" ] && CUBIN_ARGS=(-v "$CUBINS:/cubit-share:ro"
  -e VLLM_MOE_W2_CUBIT_DIR=/cubit-share
  -e VLLM_SPARSE_MLA_PREFILL_CUBIN=/cubit-share/mla_prefill_state.cubin)

# Robust teardown: a multi-GB container can take minutes to release GPU memory; re-issue the
# force-remove and wait so the next `docker run` never races into a name conflict.
docker rm -f "$NAME" >/dev/null 2>&1 || true
for i in $(seq 1 360); do
  docker inspect "$NAME" >/dev/null 2>&1 || break
  if [ $((i % 30)) -eq 0 ]; then docker rm -f "$NAME" >/dev/null 2>&1 || true; fi
  sleep 1
done

docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" --ipc=host --shm-size=32g -p "$PORT:$PORT" \
  -v "$MODEL:/model:ro" \
  "${CUBIN_ARGS[@]}" \
  -e VLLM_MOE_W2=1 \
  -e VLLM_MOE_W2_NUM_LAYERS="$NUM_LAYERS" -e VLLM_MOE_W2_DELTA_GB="$DELTA_GB" \
  -e VLLM_MOE_W2_DELTA_TRACE="$DELTA_TRACE" -e VLLM_MOE_W2_DELTA_TRACE_EVERY="$DELTA_TRACE_EVERY" \
  -e VLLM_MOE_W2_DELTA_POLICY="$DELTA_POLICY" \
  -e VLLM_MOE_W2_GATE="$GATE" -e VLLM_MOE_W2_GATE_SIGNAL="$GATE_SIGNAL" \
  -e VLLM_MOE_W2_GATE_MAX_PROMOTE="$GATE_MAX_PROMOTE" -e VLLM_MOE_W2_GATE_TRACE="$GATE_TRACE" \
  -e VLLM_SPARSE_MLA_PREFILL_CUBIT="$PREFILL_CUBIT" \
  -e VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${XENV[@]}" \
  "$IMAGE" \
  --model /model --served-model-name deepseek-v4-flash --trust-remote-code \
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" "${CAR[@]}" \
  --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$UTIL" \
  --max-num-batched-tokens "$BATCHED" --max-num-seqs "$SEQS" \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --default-chat-template-kwargs "$CHAT_KWARGS" --no-scheduler-reserve-full-isl \
  "${SPEC[@]}" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --host 0.0.0.0 --port "$PORT"

echo "launched '$NAME' (image $IMAGE) TP=$TP PP=$PP GATE=$GATE on GPU(s) $GPUS, port $PORT"

# NOTE: the confidence GATE is correct under tensor-parallel / single-GPU. Under
# pipeline-parallel (PP>1) WITH MTP it is auto-disabled (the spec-decode verify
# step's re-forward is not yet re-entrant across pipeline stages); PP without MTP
# gets the full gate. See docs/quality.md.
