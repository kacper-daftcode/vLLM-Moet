#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 4x RTX PRO 6000 (sm_120), TP4, 512K context, DSpark, FP4 compressed KV + MXFP4 indexer
# cache, optional KV offload to host RAM and disk. Image: Dockerfile.sm120-dsv41-nightly (docs/dsv41-sm120.md).
#
# Required:  MODEL_DIR   the official DeepSeek-V4.1-Flash checkpoint directory
#
# Profile:   SERVED (0)  1 = the served configuration: --moe-backend b12x, DSpark k = 7 on the b12x backend,
#                        --enable-prompt-tokens-details and the runtime environment the reference host serves with (the
#                        list is SERVED_SWITCHES below; docs/dsv41-sm120.md explains each). EXTRA_DOCKER_ARGS="-e VAR=0"
#                        after it overrides a single switch. 0 = vLLM's defaults plus the image's build-time changes only.
#                        The list is the environment of the -r32 image (the default IMAGE): the runtime keys, the second
#                        attention cubin and the idle freeze (steps 42b, 46b, 49) came with it; step 42's _WS1=9 (FC1 on
#                        the warpgroup kernel) and its five _SUM_* (the top-k sum as FC2 work items) with -r28.
#            RUNTIME_AB_KEEP_FILE (0)  with the reader of the runtime keys on (VLLM_MOET_RUNTIME_AB=1; SERVED=1 sets it) the
#                        launcher refuses to start while the reader's control file exists: /dev/shm/vllm_moet_ab.json, or
#                        the /dev/shm path VLLM_MOET_RUNTIME_AB_FILE names. The container shares the host's /dev/shm, so a
#                        file left behind would be applied at the first prefill and could select the second attention
#                        cubin. 1 = start anyway (the file's keys then apply).
#
# Container: IMAGE (vllm-moet-sm120:dsv41-nightly-df3bd094-r32)  NAME (ds41-flash)  GPUS (0,1,2,3)  TP (4)  PORT (8001)
#            BIND (127.0.0.1:)  "" publishes on all interfaces - put an authenticating proxy in front
#            CACHE_DIR ($PWD/cache-$NAME)  FlashInfer autotune results + DeepGEMM JIT; the first start fills it (~22 min)
#            SERVED_NAME (deepseek-ai/DeepSeek-V4.1-Flash)  the model name the API answers to
#            NCCL_P2P_LEVEL (SYS)    P2P over PCIe also in KVM guests NCCL classifies as PHB
#            EXTRA_ARGS / EXTRA_DOCKER_ARGS   appended to `vllm serve` / `docker run`
#
# Engine:    MAX_MODEL_LEN (524288)  MAX_NUM_SEQS (8)  MAX_NUM_BATCHED_TOKENS (4096: the prefill chunk; 8192 with PREFILL_PIPE=2)
#            PREFILL_PIPE (0; 2 with SERVED=1)  2 = step 47, the pipelined sequence-parallel prefill (two 4096-row groups of one
#                        8192-token step one layer apart; needs the steps SERVED=1 turns on); MAX_NUM_BATCHED_TOKENS then defaults to 8192
#            PREFILL_PIPE_REPLAY_GRAPHS (1 with PREFILL_PIPE=2, else 0)  the 128-row replay layers 20-39 of every step as CUDA graphs
#                        (vLLM's decoder-replay graphs, captured per row-group slot as well): the pipelined step's two replay tails
#                        are ~1,400 launches each and CPU-bound when the host's cores are loaded - the 2026-11-07 production window
#                        (+31 % at 163K, rolled back) was that; measured under 100 nice'd CPU hogs +3.3 % with the graphs vs +26.6 %
#                        without, bits identical, -1 GiB KV per GPU. Passes --compilation-config decoder_replay_cudagraph_capture_sizes
#                        [REPLAY_CAPTURE_SIZES (128,256)]; vLLM's default captures none under --max-cudagraph-capture-size 64
#            GPU_MEM_UTIL (0.94)     0.92 for the very first start (the autotune sweep OOMs at 0.95 on a cold CACHE_DIR)
#            SPEC_TOKENS (5; 7 with SERVED=1)  DSpark draft tokens, 0 = no drafter (-2.4 GiB per GPU)
#            MOE_BACKEND ("" ; b12x with SERVED=1)  --moe-backend; b12x = FlashInfer's W4A8 compact path the fused
#                        decode MoE and the prefill MoE kernels (steps 17, 22, 27, 42) attach to
#            LANGUAGE_ONLY (0)       1 = --language-model-only (no vision encoder, +0.3 GiB KV)
#
# KV cache:  KV_RECORD (nvfp4)       compressed KV record: nvfp4 (288 B/state, the checkpoint's own FP4 format; 9.6M GPU
#                        tokens at 0.94), fp8_ds_mla (584 B, vLLM's stock record) or fp8_v41 (528 B, the 0909 image only)
#            INDEXER_KV_DTYPE (mxfp4) indexer K cache: mxfp4 (68 B/key, the indexer's training format) or fp8 (132 B/key)
#            KV_MODE (auto)          how KV_RECORD reaches vLLM: auto reads the image label com.vllm-moet.kv-mode
#                        (upstream = --kv-cache-dtype nvfp4_ds_mla / fp8_ds_mla; moet = the 0909 image's plumbing)
#            INDEXER_TP_SHARD_MIN_ROWS ("")  from this many prefill rows per step the indexer shards its rows over the TP
#                        ranks; "" keeps vLLM's 65,536 (never reached with 4096-token chunks); production runs 2048
#            KV_OFFLOAD_GIB (0)      > 0: evicted KV kept in N GiB of pinned host RAM (one /dev/shm region for all ranks,
#                        896 B per token) and restored on a prefix hit; with 64 GiB the host needs 253.0 GiB of shared
#                        memory in total under SERVED=1 (328.3 GiB without its allocator variable) - on a host with
#                        less RAM keep 0 (nothing changes below the GPU KV capacity)
#            KV_OFFLOAD_FS_DIR ("")  with KV_OFFLOAD_GIB > 0: a disk tier behind the RAM tier (one file per 128-token
#                        block, survives restarts). vLLM never deletes the files: run docker/sm120/kvcache-ttl.sh from
#                        cron. Images labeled com.vllm-moet.v41-kv-records=interleaved write under DIR/v41-interleaved;
#                        INDEXER_KV_DTYPE=fp8 under DIR/indexer-fp8 (the formats must not share a directory)
#            KV_OFFLOAD_PROMPT_ONLY (1)  0 = offload generated tokens too (the next turn hits the previous answer)
#
# Experiments: PREFILL_DBO (0)  1 = --enable-dbo (two prefill microbatches; needs an image built with step 15; measured
#                        slower on 4x RTX PRO 6000), DBO_PREFILL_THRESHOLD (1024), DBO_DECODE_THRESHOLD (4096)
#            PROFILER (0)            1 = torch profiler endpoints (/start_profile, /stop_profile), traces in PROFILE_DIR
#
# Reasoning effort: the image renders the checkpoint's tiers (low 50 / high 75 / max 100, default high; the OpenAI
# aliases minimal 25 / medium 62 / xhigh 87 are accepted) - chat_template_kwargs {"reasoning_effort": 1..100} or the
# OpenAI reasoning_effort field; "none" or {"thinking": false} turns thinking off. enable_adaptive_verification stays
# false (DeepseekV4IndexerBackend does not support it on sm_120).
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR (DeepSeek-V4.1-Flash checkpoint directory) is required}"
IMAGE="${IMAGE:-vllm-moet-sm120:dsv41-nightly-df3bd094-r32}"
NAME="${NAME:-ds41-flash}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
PORT="${PORT:-8001}"
BIND="${BIND-127.0.0.1:}"
CACHE_DIR="${CACHE_DIR:-$PWD/cache-$NAME}"
SERVED_NAME="${SERVED_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
SERVED="${SERVED:-0}"
case "$SERVED" in
  0|1) ;;
  *) echo "SERVED must be 0 or 1, got $SERVED" >&2; exit 1 ;;
esac
# Step 47 (PREFILL_PIPE=2 with 8192 batched tokens) is -22.6 % at 163K on an idle host; its eager replay tails are launch-bound, so a
# host whose CPUs other processes load on the server's NUMA node pays +31 % (not the KV offload tier: with it +0.5 %). The -r26b
# image's PREFILL_PIPE_REPLAY_GRAPHS=1 runs those tails as CUDA graphs (+3.3 % under 100 nice'd hogs instead of +26.6 % on the second
# host; on the reference host 163K 6.6 -> 5.6-5.8 s, +1.7 % under 100 hogs, KV 12.5 -> 10.5 GiB). SERVED=1 turns it on (PREFILL_PIPE=0
# keeps the one-group path).
PREFILL_PIPE="${PREFILL_PIPE:-$([[ "$SERVED" == "1" ]] && echo 2 || echo 0)}"
case "$PREFILL_PIPE" in
  0|2) ;;
  *) echo "PREFILL_PIPE must be 0 or 2, got $PREFILL_PIPE" >&2; exit 1 ;;
esac
if [[ "$PREFILL_PIPE" == "2" ]]; then
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
  PREFILL_PIPE_REPLAY_GRAPHS="${PREFILL_PIPE_REPLAY_GRAPHS:-1}"
else
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
  PREFILL_PIPE_REPLAY_GRAPHS="${PREFILL_PIPE_REPLAY_GRAPHS:-0}"
fi
REPLAY_CAPTURE_SIZES="${REPLAY_CAPTURE_SIZES:-128,256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"
# The served configuration (docs/dsv41-sm120.md, "Switches"): the runtime switches of the image's steps that are
# off by default, in their served state. EXTRA_DOCKER_ARGS comes later on the docker command line, so
# "-e VAR=0" there overrides one of these. Step 42's stages: FC2 on the producer-warp kernel (_WS2=3), FC1 on the
# warpgroup kernel (_WS1=9), the top-k sum as work items of the FC2 kernel (the five _SUM_*); _WS1=9 and the _SUM_*
# need the -r28 image or a later one. Three of the last four entries need the -r32 image: the reader of the runtime
# keys (step 42b; its control file /dev/shm/vllm_moet_ab.json is a file of the HOST, because the container runs
# --ipc host: the launcher refuses to start over one, see RUNTIME_AB_KEEP_FILE); a second attention cubin for that
# facility (step 46b: loaded and compared with the shipped cubin, never selected - v43b, v43 and v42 must not be
# selected, docs/dsv41-sm120.md says why); the idle freeze (step 49). The fourth is PyTorch's own pinned-allocator
# variable (no power-of-two rounding above 256 MiB: the loader's pinned tables take 189.0 instead of 264.3 GiB of
# host memory).
SERVED_SWITCHES=(
  VLLM_MOET_B12X_MOE_FUSED=1 VLLM_MOET_SPARSE_MLA_LDG=1 VLLM_MOET_TARGET_LM_HEAD_MXFP8=1
  VLLM_MOET_P2P_AR=1 VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=8
  VLLM_MOET_DFLASH_PREP_GROUPS=1 VLLM_MOET_CANDIDATE_DET_TOPK=1 VLLM_MOET_SWA_DECODE_BUILD=1
  VLLM_MOET_MOE_TOPKSUM_ADD=1 VLLM_MOET_SWA_DRAFT_BUILD=1 B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1
  VLLM_MOET_MHC_MXFP8=1 VLLM_MOET_MHC_SIDE_DEFER=1
  VLLM_MOET_SPARSE_MLA_PREFILL_BF16=1 VLLM_MOET_PREFILL_SP=1 VLLM_MOET_B12X_MOE_PREFILL_FAST=1
  VLLM_MOET_B12X_MOE_PREFILL_FAST_WS2=3 VLLM_MOET_B12X_MOE_PREFILL_FAST_WS1=9 VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_ITEMS=1
  VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_DISCARD=1 VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_ROUTE_HINT=1
  VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_TOKENS=128 VLLM_MOET_B12X_MOE_PREFILL_FAST_SUM_MIN_ROWS=512
  VLLM_MOET_PREFILL_SP_GATHER=mxfp8 VLLM_MOET_DET_TOPK=1 VLLM_MOET_SPARSE_MLA_PREFILL_SASS=1
  VLLM_MOET_PREFILL_PIPE=${PREFILL_PIPE} VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS=${PREFILL_PIPE_REPLAY_GRAPHS}
  VLLM_MOET_RUNTIME_AB=1
  VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN_B=/opt/vllm-moet/kernels/cubins-sm120/sparse_mla_prefill_v43b.cubin
  VLLM_MOET_GC_IDLE_FREEZE=1 PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:256
)
if [[ "$SERVED" == "1" ]]; then
  : "${SPEC_TOKENS:=7}"
  : "${MOE_BACKEND:=b12x}"
fi
SPEC_TOKENS="${SPEC_TOKENS:-5}"
MOE_BACKEND="${MOE_BACKEND:-}"
INDEXER_KV_DTYPE="${INDEXER_KV_DTYPE:-mxfp4}"
KV_RECORD="${KV_RECORD:-nvfp4}"
LANGUAGE_ONLY="${LANGUAGE_ONLY:-0}"
PREFILL_DBO="${PREFILL_DBO:-0}"
INDEXER_TP_SHARD_MIN_ROWS="${INDEXER_TP_SHARD_MIN_ROWS:-}"
DBO_PREFILL_THRESHOLD="${DBO_PREFILL_THRESHOLD:-1024}"
DBO_DECODE_THRESHOLD="${DBO_DECODE_THRESHOLD:-4096}"
KV_OFFLOAD_GIB="${KV_OFFLOAD_GIB:-0}"
KV_OFFLOAD_FS_DIR="${KV_OFFLOAD_FS_DIR:-}"
KV_OFFLOAD_PROMPT_ONLY="${KV_OFFLOAD_PROMPT_ONLY:-1}"
PROFILER="${PROFILER:-0}"
PROFILE_DIR="${PROFILE_DIR:-$PWD/profiles-$NAME}"
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"

[[ -f "$MODEL_DIR/config.json" ]] || { echo "no config.json in MODEL_DIR=$MODEL_DIR" >&2; exit 1; }
mkdir -p "$CACHE_DIR/dot-cache" "$CACHE_DIR/deep_gemm"

ARGS=()
SERVED_ENV=()
if [[ "$SERVED" == "1" ]]; then
  for kv in "${SERVED_SWITCHES[@]}"; do SERVED_ENV+=(-e "$kv"); done
  ARGS+=(--enable-prompt-tokens-details)
fi
[[ -n "$MOE_BACKEND" ]] && ARGS+=(--moe-backend "$MOE_BACKEND")
if [[ "$SPEC_TOKENS" != "0" ]]; then
  SPEC_MOE=""
  [[ -n "$MOE_BACKEND" ]] && SPEC_MOE=",\"moe_backend\":\"$MOE_BACKEND\""
  ARGS+=(--speculative-config
    "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false${SPEC_MOE}}")
fi
[[ "$LANGUAGE_ONLY" == "1" ]] && ARGS+=(--language-model-only)
case "$PREFILL_PIPE_REPLAY_GRAPHS" in
  0) ;;
  1) ARGS+=(--compilation-config "{\"decoder_replay_cudagraph_capture_sizes\":[${REPLAY_CAPTURE_SIZES}]}") ;;  # EXTRA_ARGS must not carry a second one
  *) echo "PREFILL_PIPE_REPLAY_GRAPHS must be 0 or 1, got $PREFILL_PIPE_REPLAY_GRAPHS" >&2; exit 1 ;;
esac
case "$PREFILL_DBO" in
  0) ;;
  1)
    if [[ "$(docker image inspect --format '{{index .Config.Labels "com.vllm-moet.prefill-dbo"}}' "$IMAGE" 2>/dev/null)" != "1" ]]; then
      echo "PREFILL_DBO=1 needs an image built with Dockerfile step 15 (label com.vllm-moet.prefill-dbo=1); $IMAGE has none" >&2; exit 1
    fi
    ARGS+=(--enable-dbo --dbo-prefill-token-threshold "$DBO_PREFILL_THRESHOLD"
           --dbo-decode-token-threshold "$DBO_DECODE_THRESHOLD") ;;
  *) echo "PREFILL_DBO must be 0 or 1, got $PREFILL_DBO" >&2; exit 1 ;;
esac
case "$KV_OFFLOAD_PROMPT_ONLY" in
  0|1) ;;
  *) echo "KV_OFFLOAD_PROMPT_ONLY must be 0 or 1, got $KV_OFFLOAD_PROMPT_ONLY" >&2; exit 1 ;;
esac
if [[ "$KV_OFFLOAD_GIB" != "0" ]]; then
  # --kv-offloading-size adds cpu_bytes_to_use to the connector's extra config given here.
  ARGS+=(--kv-offloading-size "$KV_OFFLOAD_GIB")
  OFFLOAD_EXTRA=""
  [[ "$KV_OFFLOAD_PROMPT_ONLY" == "0" ]] && OFFLOAD_EXTRA="\"offload_prompt_only\":false"
  # The fs tier's directory name covers the model path, TP, KV dtype and layer names but not the indexer
  # format, and a block file is read up to the block size whatever its length: files written with the other
  # INDEXER_KV_DTYPE would load as garbage, so that format gets its own subdirectory. So do images whose vLLM
  # writes the V4.1 records interleaved (vllm#61371, label com.vllm-moet.v41-kv-records): same bytes per block,
  # another order.
  FS_ROOT="$KV_OFFLOAD_FS_DIR"
  [[ "$INDEXER_KV_DTYPE" != "mxfp4" ]] && FS_ROOT="$KV_OFFLOAD_FS_DIR/indexer-$INDEXER_KV_DTYPE"
  if [[ "$(docker image inspect --format '{{index .Config.Labels "com.vllm-moet.v41-kv-records"}}' "$IMAGE" 2>/dev/null)" == "interleaved" ]]; then
    FS_ROOT="$FS_ROOT/v41-interleaved"
  fi
  [[ -n "$KV_OFFLOAD_FS_DIR" ]] && OFFLOAD_EXTRA="\"spec_name\":\"TieringOffloadingSpec\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$FS_ROOT\",\"n_read_threads\":16,\"n_write_threads\":16,\"locality\":\"LOCAL\"}]${OFFLOAD_EXTRA:+,$OFFLOAD_EXTRA}"
  [[ -n "$OFFLOAD_EXTRA" ]] && ARGS+=(--kv-transfer-config
    "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$OFFLOAD_EXTRA}}")
fi
case "$KV_RECORD" in
  fp8_ds_mla|nvfp4|fp8_v41) ;;
  *) echo "KV_RECORD must be fp8_ds_mla, nvfp4 or fp8_v41, got $KV_RECORD" >&2; exit 1 ;;
esac
KV_MODE="${KV_MODE:-auto}"
if [[ "$KV_MODE" == "auto" ]]; then
  KV_MODE="$(docker image inspect --format '{{index .Config.Labels "com.vllm-moet.kv-mode"}}' "$IMAGE" 2>/dev/null || true)"
  KV_MODE="${KV_MODE:-moet}"
fi
KV_ENV=()
case "$KV_MODE" in
  moet)
    KV_CACHE_DTYPE=fp8
    KV_ENV=(-e VLLM_MOET_KV_RECORD="$KV_RECORD") ;;
  upstream)
    case "$KV_RECORD" in
      nvfp4) KV_CACHE_DTYPE=nvfp4_ds_mla ;;
      fp8_ds_mla) KV_CACHE_DTYPE=fp8_ds_mla ;;
      *) echo "KV_RECORD=$KV_RECORD has no upstream (vLLM main) equivalent; use nvfp4 or fp8_ds_mla" >&2; exit 1 ;;
    esac ;;
  *) echo "KV_MODE must be auto, moet or upstream, got $KV_MODE" >&2; exit 1 ;;
esac
ATTN_CONFIG=()
case "$INDEXER_KV_DTYPE" in
  fp8) ;;
  mxfp4) ATTN_CONFIG+=("\"indexer_kv_dtype\":\"mxfp4\"") ;;
  *) echo "INDEXER_KV_DTYPE must be fp8 or mxfp4, got $INDEXER_KV_DTYPE" >&2; exit 1 ;;
esac
if [[ -n "$INDEXER_TP_SHARD_MIN_ROWS" ]]; then
  [[ "$INDEXER_TP_SHARD_MIN_ROWS" =~ ^[1-9][0-9]*$ ]] || { echo "INDEXER_TP_SHARD_MIN_ROWS must be a positive integer, got $INDEXER_TP_SHARD_MIN_ROWS" >&2; exit 1; }
  ATTN_CONFIG+=("\"indexer_tp_shard_min_rows\":$INDEXER_TP_SHARD_MIN_ROWS")
fi
if (( ${#ATTN_CONFIG[@]} )); then
  ARGS+=(--attention-config "{$(IFS=,; echo "${ATTN_CONFIG[*]}")}")
fi
MOUNTS=(-v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR/dot-cache:/root/.cache" -v "$CACHE_DIR/deep_gemm:/root/.deep_gemm")
if [[ "$KV_OFFLOAD_GIB" != "0" && -n "$KV_OFFLOAD_FS_DIR" ]]; then
  mkdir -p "$KV_OFFLOAD_FS_DIR"
  MOUNTS+=(-v "$KV_OFFLOAD_FS_DIR:$KV_OFFLOAD_FS_DIR")
fi
if [[ "$PROFILER" == "1" ]]; then
  mkdir -p "$PROFILE_DIR"
  MOUNTS+=(-v "$PROFILE_DIR:/profiles")
  ARGS+=(--profiler-config.profiler torch --profiler-config.torch_profiler_dir /profiles
         --profiler-config.torch_profiler_with_stack false)
fi

if docker ps -q -f name="^${NAME}$" -f status=running | grep -q .; then
  echo "container $NAME already running - stop it first (docker stop -t 60 $NAME)"; exit 0
fi

# The value the container gets for a variable that SERVED_SWITCHES or EXTRA_DOCKER_ARGS may set: docker keeps the last
# -e of a name, and EXTRA_DOCKER_ARGS comes after the profile's list on the command line.
# shellcheck disable=SC2086
passed_env() {
  local name="$1" val="" prev="" w
  for w in "${SERVED_ENV[@]}" ${EXTRA_DOCKER_ARGS}; do
    if [[ ( "$prev" == "-e" || "$prev" == "--env" ) && "$w" == "$name="* ]]; then val="${w#*=}"; fi
    if [[ "$w" == -e"$name="* || "$w" == --env="$name="* ]]; then val="${w#*"$name="}"; fi
    prev="$w"
  done
  printf '%s' "$val"
}
# Step 42b's control file is a file of the host's /dev/shm (the container runs --ipc host) and outlives the container: one
# left behind would be applied by the new server at its first prefill - under SERVED=1 it could select the second
# attention cubin, which is loaded and must not be selected. With the reader on, do not start over such a file.
RUNTIME_AB_KEEP_FILE="${RUNTIME_AB_KEEP_FILE:-0}"
case "$RUNTIME_AB_KEEP_FILE" in
  0|1) ;;
  *) echo "RUNTIME_AB_KEEP_FILE must be 0 or 1, got $RUNTIME_AB_KEEP_FILE" >&2; exit 1 ;;
esac
if [[ "$(passed_env VLLM_MOET_RUNTIME_AB)" == "1" ]]; then
  ab_file="$(passed_env VLLM_MOET_RUNTIME_AB_FILE)"
  ab_file="${ab_file:-/dev/shm/vllm_moet_ab.json}"
  if [[ "$ab_file" == /dev/shm/* && -e "$ab_file" ]]; then
    if [[ "$RUNTIME_AB_KEEP_FILE" == "1" ]]; then
      echo "the runtime keys' control file $ab_file exists and RUNTIME_AB_KEEP_FILE=1: its keys apply from the first prefill on"
    else
      echo "refusing to start $NAME: the runtime keys' control file $ab_file exists on this host, and with" >&2
      echo "  VLLM_MOET_RUNTIME_AB=1 the new server would apply it at its first prefill (content: $(head -c 200 "$ab_file" 2>/dev/null | tr '\n' ' '))." >&2
      echo "  Remove it (rm $ab_file) and start again, or keep it on purpose with RUNTIME_AB_KEEP_FILE=1." >&2
      exit 1
    fi
  fi
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
if [[ "$KV_OFFLOAD_GIB" != "0" ]]; then
  # An offload region its server did not remove on exit stays in the host's /dev/shm (RAM) until deleted;
  # drop the ones no process maps any more.
  for region in /dev/shm/vllm_offload_*.mmap; do
    [[ -e "$region" ]] || continue
    grep -qs -- "$region" /proc/[0-9]*/maps || { echo "removing stale offload region $region"; rm -f "$region"; }
  done
fi

# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart no \
  --gpus "\"device=${GPUS}\"" \
  --ipc host --shm-size 32g \
  -p "${BIND}${PORT}:${PORT}" \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  "${MOUNTS[@]}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
  "${KV_ENV[@]}" \
  "${SERVED_ENV[@]}" \
  ${EXTRA_DOCKER_ARGS} \
  "$IMAGE" \
  --model /model --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-cudagraph-capture-size 64 \
  "${ARGS[@]}" \
  ${EXTRA_ARGS} \
  --host 0.0.0.0 --port "$PORT"

echo "container $NAME started; logs: docker logs -f $NAME ; API: http://${BIND:-0.0.0.0:}${PORT}/v1"
echo "expect in the log: 'Using Sm120GemvMxfp8BmmLinearKernel for MXFP8 GEMM', 'wo_a stays MXFP8 on sm_120',"
echo "  'Using ${INDEXER_KV_DTYPE^^} indexer cache for Lightning Indexer'"
[[ "$PREFILL_DBO" == "1" ]] && echo "  'vllm-moet prefill DBO: eager steps of >= ${DBO_PREFILL_THRESHOLD} tokens ... run as 2 microbatches'"
exit 0
