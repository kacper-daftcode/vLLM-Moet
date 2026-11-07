#!/usr/bin/env bash
# Ripple-65B Preview (ricursiveAI) on 4x RTX PRO 6000 (sm_120), TP4, with the DFlash2 drafter, the `ripple` reasoning and
# tool-call parsers, optional KV offload to host RAM and disk. Image: Dockerfile.sm120-ripple (docs/ripple-sm120.md).
#
# Required:  MODEL_DIR   the checkpoint directory (ricursiveAI/Ripple-65B-Preview-NVFP4: MLPs in ModelOpt NVFP4, attention,
#                        embeddings and lm_head in BF16 - 57 GB; or the BF16 checkpoint, 130 GB)
#
# Drafter:   DRAFT_DIR ("")  ricursiveAI/Ripple-65B-Preview-DFlash2 (4B BF16, blocks of 16; "" = no speculative decoding)
#            SPEC_TOKENS (7)  draft tokens per step with DRAFT_DIR set (vLLM's recipe: 7; the model card validates up to
#                        15 - GSM8K mean acceptance 7.34 at k = 15 on B300); 0 = no drafter even with DRAFT_DIR
#            SPEC_SCHEDULE ("")  vLLM's dynamic speculative decoding: a JSON list of [batch_from, batch_to, k] (inclusive
#                        batch-size ranges, k <= SPEC_TOKENS), e.g. [[1,2,15],[3,8,11],[9,16,7],[17,32,4]] - the step drafts
#                        as many tokens as the schedule gives the number of running requests (Model Runner V2 truncates
#                        the drafts centrally, full CUDA graphs stay)
#
# Container: IMAGE (vllm-moet-sm120:ripple-nightly-a2bcb565-r6)  NAME (ripple-65b)  GPUS (4,5,6,7; indices or UUIDs)
#            TP (4)  PORT (8002)  BIND (127.0.0.1:)  "" publishes on all interfaces - put an authenticating proxy in front
#            CPUS ("")  --cpuset-cpus for the container (the NUMA node of GPUS)   MEMORY ("")  --memory cap
#            CACHE_DIR ($PWD/cache-$NAME)  torch.compile / FlashInfer caches; the first start fills it
#            SERVED_NAME (ricursiveAI/Ripple-65B-Preview)  the model name the API answers to; SERVED_ALIASES
#                        (ricursiveAI/Ripple-65B-Preview-NVFP4) further names, space-separated
#            NCCL_P2P_LEVEL (SYS)    P2P over PCIe also in KVM guests NCCL classifies as PHB
#            EXTRA_ARGS / EXTRA_DOCKER_ARGS   appended to `vllm serve` / `docker run`
#            DECODE_UBATCH (0)  2 = two-microbatch decode (step 3) for uniform decodes >= DECODE_UBATCH_MIN_TOKENS (128) tokens
#
# Engine:    MAX_MODEL_LEN (524288)  the checkpoint allows 2,097,152; the drafter's positions are configured to 131,072
#            MAX_NUM_SEQS (32)       MAX_NUM_BATCHED_TOKENS (8192: the prefill chunk)  GPU_MEM_UTIL (0.92)
#            CUDAGRAPH_MAX ((SPEC_TOKENS + 1) * MAX_NUM_SEQS)  --max-cudagraph-capture-size: graphs up to the full decode
#                        batch with every draft token accepted
#            KV_CACHE_DTYPE (auto)   auto = the checkpoint's BF16; fp8 halves the KV bytes (an output change - gate it)
#            ATTN_BACKEND ("")       --attention-backend; "" = vLLM's choice for sm_120
#            PREFIX_CACHING (1)      0 = --no-enable-prefix-caching
#
# KV cache:  KV_OFFLOAD_GIB (0)      > 0: evicted KV kept in N GiB of pinned host RAM and restored on a prefix hit
#            KV_OFFLOAD_FS_DIR ("")  with KV_OFFLOAD_GIB > 0: a disk tier behind the RAM tier (one file per block, survives
#                        restarts; vLLM never deletes the files - run docker/sm120/kvcache-ttl.sh from cron)
#            KV_OFFLOAD_PROMPT_ONLY (0)  1 = offload prompts only (0 = generated tokens too: the next turn hits the answer)
#
# Diagnostics: PROFILER (0)  1 = torch profiler endpoints (/start_profile, /stop_profile), traces in PROFILE_DIR
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR (Ripple-65B Preview checkpoint directory) is required}"
IMAGE="${IMAGE:-vllm-moet-sm120:ripple-nightly-a2bcb565-r6}"
NAME="${NAME:-ripple-65b}"
GPUS="${GPUS:-4,5,6,7}"
TP="${TP:-4}"
PORT="${PORT:-8002}"
BIND="${BIND-127.0.0.1:}"
CPUS="${CPUS:-}"
MEMORY="${MEMORY:-}"
CACHE_DIR="${CACHE_DIR:-$PWD/cache-$NAME}"
SERVED_NAME="${SERVED_NAME:-ricursiveAI/Ripple-65B-Preview}"
SERVED_ALIASES="${SERVED_ALIASES-ricursiveAI/Ripple-65B-Preview-NVFP4}"
DRAFT_DIR="${DRAFT_DIR:-}"
SPEC_TOKENS="${SPEC_TOKENS:-7}"
SPEC_SCHEDULE="${SPEC_SCHEDULE:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ATTN_BACKEND="${ATTN_BACKEND:-}"
PREFIX_CACHING="${PREFIX_CACHING:-1}"
KV_OFFLOAD_GIB="${KV_OFFLOAD_GIB:-0}"
KV_OFFLOAD_FS_DIR="${KV_OFFLOAD_FS_DIR:-}"
KV_OFFLOAD_PROMPT_ONLY="${KV_OFFLOAD_PROMPT_ONLY:-0}"
PROFILER="${PROFILER:-0}"
PROFILE_DIR="${PROFILE_DIR:-$PWD/profiles-$NAME}"
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
# Step 3 (image -r6+): DECODE_UBATCH=2 runs uniform decodes of >= DECODE_UBATCH_MIN_TOKENS tokens as two microbatches with one
# half's TP all-reduces under the other half's GEMMs (vLLM's DBO without DP; needs the V2 runner pinned). Default off.
DECODE_UBATCH="${DECODE_UBATCH:-0}"
DECODE_UBATCH_MIN_TOKENS="${DECODE_UBATCH_MIN_TOKENS:-128}"
if [[ "$DECODE_UBATCH" == "2" ]]; then
  EXTRA_DOCKER_ARGS="$EXTRA_DOCKER_ARGS -e VLLM_MOET_DECODE_UBATCH=2 -e VLLM_MOET_DECODE_UBATCH_MIN_TOKENS=$DECODE_UBATCH_MIN_TOKENS -e VLLM_USE_V2_MODEL_RUNNER=1"
fi

[[ -f "$MODEL_DIR/config.json" ]] || { echo "no config.json in MODEL_DIR=$MODEL_DIR" >&2; exit 1; }
if [[ -n "$DRAFT_DIR" && "$SPEC_TOKENS" != "0" ]]; then
  [[ -f "$DRAFT_DIR/config.json" ]] || { echo "no config.json in DRAFT_DIR=$DRAFT_DIR" >&2; exit 1; }
else
  SPEC_TOKENS=0
fi
for v in SPEC_TOKENS MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_MODEL_LEN TP KV_OFFLOAD_GIB; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "$v must be a non-negative integer, got ${!v}" >&2; exit 1; }
done
case "$PREFIX_CACHING$KV_OFFLOAD_PROMPT_ONLY$PROFILER" in
  [01][01][01]) ;;
  *) echo "PREFIX_CACHING, KV_OFFLOAD_PROMPT_ONLY and PROFILER must be 0 or 1" >&2; exit 1 ;;
esac
CUDAGRAPH_MAX="${CUDAGRAPH_MAX:-$(( (SPEC_TOKENS + 1) * MAX_NUM_SEQS ))}"
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "container $NAME exists; remove it first (docker stop -t 120 $NAME && docker rm $NAME)" >&2; exit 1
fi
mkdir -p "$CACHE_DIR/dot-cache"

ARGS=(--model /model --served-model-name "$SERVED_NAME")
for alias in $SERVED_ALIASES; do ARGS+=("$alias"); done
ARGS+=(--tensor-parallel-size "$TP"
       --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
       --gpu-memory-utilization "$GPU_MEM_UTIL" --max-cudagraph-capture-size "$CUDAGRAPH_MAX"
       --kv-cache-dtype "$KV_CACHE_DTYPE"
       --reasoning-parser ripple --tool-call-parser ripple --enable-auto-tool-choice
       --enable-prompt-tokens-details)
[[ -n "$ATTN_BACKEND" ]] && ARGS+=(--attention-backend "$ATTN_BACKEND")
[[ "$PREFIX_CACHING" == "0" ]] && ARGS+=(--no-enable-prefix-caching)
MOUNTS=(-v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR/dot-cache:/root/.cache")
if [[ "$SPEC_TOKENS" != "0" ]]; then
  MOUNTS+=(-v "$DRAFT_DIR:/draft:ro")
  SPEC_EXTRA=""
  [[ -n "$SPEC_SCHEDULE" ]] && SPEC_EXTRA=",\"num_speculative_tokens_per_batch_size\":${SPEC_SCHEDULE}"
  ARGS+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":${SPEC_TOKENS}${SPEC_EXTRA}}")
fi
if [[ "$KV_OFFLOAD_GIB" != "0" ]]; then
  ARGS+=(--kv-offloading-size "$KV_OFFLOAD_GIB")
  OFFLOAD_EXTRA=""
  [[ "$KV_OFFLOAD_PROMPT_ONLY" == "0" ]] && OFFLOAD_EXTRA="\"offload_prompt_only\":false"
  if [[ -n "$KV_OFFLOAD_FS_DIR" ]]; then
    mkdir -p "$KV_OFFLOAD_FS_DIR"
    MOUNTS+=(-v "$KV_OFFLOAD_FS_DIR:$KV_OFFLOAD_FS_DIR")
    OFFLOAD_EXTRA="\"spec_name\":\"TieringOffloadingSpec\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$KV_OFFLOAD_FS_DIR\",\"n_read_threads\":16,\"n_write_threads\":16,\"locality\":\"LOCAL\"}]${OFFLOAD_EXTRA:+,$OFFLOAD_EXTRA}"
  fi
  [[ -n "$OFFLOAD_EXTRA" ]] && ARGS+=(--kv-transfer-config
    "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$OFFLOAD_EXTRA}}")
  # vLLM leaves its /dev/shm region behind on every stop (vllm#57303): drop the regions no process maps.
  for region in /dev/shm/vllm_offload_*.mmap; do
    [[ -e "$region" ]] || continue
    grep -qs -- "$region" /proc/[0-9]*/maps || { echo "removing stale offload region $region"; rm -f "$region"; }
  done
fi
if [[ "$PROFILER" == "1" ]]; then
  mkdir -p "$PROFILE_DIR"
  MOUNTS+=(-v "$PROFILE_DIR:/profiles")
  ARGS+=(--profiler-config.profiler torch --profiler-config.torch_profiler_dir /profiles
         --profiler-config.torch_profiler_with_stack false)
fi
DOCKER_OPTS=()
[[ -n "$CPUS" ]] && DOCKER_OPTS+=(--cpuset-cpus "$CPUS")
[[ -n "$MEMORY" ]] && DOCKER_OPTS+=(--memory "$MEMORY")

docker run -d --name "$NAME" --restart no \
  --gpus "\"device=${GPUS}\"" \
  "${DOCKER_OPTS[@]}" \
  --ipc host \
  -p "${BIND}${PORT}:${PORT}" \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  "${MOUNTS[@]}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
  ${EXTRA_DOCKER_ARGS} \
  "$IMAGE" \
  "${ARGS[@]}" \
  ${EXTRA_ARGS} \
  --host 0.0.0.0 --port "$PORT"
echo "container $NAME started; logs: docker logs -f $NAME ; API: http://127.0.0.1:${PORT}/v1"
