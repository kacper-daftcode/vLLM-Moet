#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 4x RTX PRO 6000 (sm_120), TP4, DSpark k=5, FP4 compressed KV + fp8 SWA KV, MXFP4
# indexer cache, 512K context -- the validated deployment (docs/sm120-deploy.md, docs/dsv41-sm120-port.md).
# Image: Dockerfile.sm120-dsv41-nightly (the vLLM main line, served since 2026-09-23: 3.45M-token KV, 74 steps/s);
# IMAGE=vllm-moet-sm120:dsv41-0909 (Dockerfile.sm120-dsv41, the recipe's 0909 image) is the rollback -- KV_MODE=auto
# picks the right KV plumbing for either.
#
# Required:  MODEL_DIR   directory with the official DeepSeek-V4.1-Flash checkpoint
# Optional:  IMAGE (vllm-moet-sm120:dsv41-nightly-20260923)  NAME (ds41-flash)  GPUS (0,1,2,3)  TP (4)  PORT (8001)
#            BIND (127.0.0.1: only local; "" = all interfaces)
#            CACHE_DIR   host dir with two subdirs mounted at /root/.cache and /root/.deep_gemm
#                        (FlashInfer autotune + JIT, DeepGEMM JIT; first start ~25 min, later ~17 min)
#            MAX_MODEL_LEN (524288)  MAX_NUM_SEQS (8)  MAX_NUM_BATCHED_TOKENS (4096)
#            GPU_MEM_UTIL (0.94)     0.94 with a populated FlashInfer autotune cache; use 0.92 for
#                        the very first start (the autotune sweep after KV allocation OOMs at 0.95)
#            SPEC_TOKENS (5)         DSpark draft tokens; 0 disables the drafter (-2.4 GiB/GPU, ~106 tok/s)
#            INDEXER_KV_DTYPE (mxfp4) indexer K cache: mxfp4 (68 B/key, the format the indexer was trained
#                        with; validated 2026-09-20: greedy outputs, GSM8K-200 and needle identical to fp8,
#                        2.29M instead of 1.49M KV tokens at 0.94) or fp8 (132 B/key, V3.2 layout, the
#                        pre-2026-09-20 default). Passed as --attention-config '{"indexer_kv_dtype":...}';
#                        mxfp4 needs the 2026-09-20 image (DeepGEMM sm120_fp4 paged logits on 128-key
#                        pages + the lifted sm_10x gate). Peak GPU memory with mxfp4 at 0.94 is
#                        97.0/97.9 GB (docs/sm120-deploy.md); GPU_MEM_UTIL=0.93 restores the fp8 margin.
#            KV_RECORD (nvfp4)       compressed (main) KV record: nvfp4 (288 B/state: the checkpoint's own FP4
#                        e2m1 + e4m3/16 format, validated 2026-09-21: GSM8K-200 / needle unchanged, greedy
#                        outputs differ from fp8 the way two serving stacks differ, -1 % decode, -3 %
#                        prefill, 3.06M instead of 2.29M KV tokens at 0.94), fp8_ds_mla (584 B, the stock
#                        record served until 2026-09-21) or fp8_v41 (528 B, op-level validated only).
#                        Needs the 2026-09-21 image (tools/dsv41_sm120/nvfp4_kv/); no rebuild to switch.
#                        VLLM_MOET_KV_PREFILL_POOL_STATES (default MAX_MODEL_LEN states = 306 MB at 512K)
#                        sizes the prefill dequant pool; VLLM_MOET_KV_GATHER_ROWS (64) the decode scratch.
#            KV_MODE (auto)          how KV_RECORD reaches vLLM: "moet" = the 0909 image's plumbing
#                        (--kv-cache-dtype fp8 + VLLM_MOET_KV_RECORD, our scratch/pool kernels);
#                        "upstream" = the vLLM-main image (Dockerfile.sm120-dsv41-nightly): nvfp4 ->
#                        --kv-cache-dtype nvfp4_ds_mla read by FlashInfer's DSv4.1 dual cache, fp8_ds_mla ->
#                        --kv-cache-dtype fp8_ds_mla (fp8_v41 has no upstream equivalent). "auto" reads the
#                        image label com.vllm-moet.kv-mode and falls back to "moet".
#            KV_OFFLOAD_GIB (0)      > 0 = --kv-offloading-size N: vLLM main's native OffloadingConnector keeps
#                        evicted KV in pinned host RAM (a /dev/shm region of N GiB, total over the TP ranks;
#                        the container runs with --ipc host, so /dev/shm is the host's). Only the block-128
#                        group (compressed MLA + indexer) is offloaded; the 128-token SWA window is replayed
#                        on a hit. Measured 2026-09-23 (4x RTX PRO 6000): 179K-token context restored in
#                        0.49 s instead of a 17.3 s prefill, decode and prefill unchanged. Host RAM per token:
#                        896 B (one copy of the TP-replicated KV: the -kvdedup and -ced images,
#                        patch_vllm_offload_replicated_mla.py; 64 GiB = 76.7M tokens), 3.58 KB on the plain
#                        vLLM-main image (one copy per rank). Needs the vLLM-main image (Dockerfile.sm120-dsv41-nightly).
#            KV_OFFLOAD_FS_DIR ("")  with KV_OFFLOAD_GIB > 0: a disk tier behind the host-RAM tier (spec
#                        TieringOffloadingSpec, secondary tier "fs", bind-mounted at the same path). Every block
#                        stored to RAM is also written there, one file per 128-token block named by its content
#                        hash, so the KV outlives evictions from RAM and server restarts. vLLM never deletes
#                        these files: run docker/sm120/kvcache-ttl.sh from cron. Measured 2026-09-23: a
#                        221K-token context read back from disk (virtio on this KVM host) in 1.6 s instead of
#                        a 22 s prefill, 896 B/token on disk (-kvdedup / -ced), prefill unchanged while the
#                        writes run. INDEXER_KV_DTYPE other than mxfp4 writes to DIR/indexer-<dtype>: vLLM's
#                        directory name does not cover the indexer format.
#            KV_OFFLOAD_PROMPT_ONLY (1)  0 = offload generated tokens too (offload_prompt_only=false), so the
#                        next turn also hits the previous answer. The V4.1 encoder keeps earlier reasoning and
#                        tool calls in the history when the request has tools (agents) and drops the reasoning
#                        otherwise; the dropped blocks only take room in the tiers.
#            LANGUAGE_ONLY (0)       1 = --language-model-only (no vision encoder, +0.3 GiB KV)
#            PROFILER (0)            1 = torch profiler endpoints, traces in PROFILE_DIR
#            NCCL_P2P_LEVEL (SYS)    P2P over PCIe works in the KVM guests NCCL classifies as PHB
#            EXTRA_ARGS / EXTRA_DOCKER_ARGS   appended to `vllm serve` / `docker run`
# Runtime kill switches (env, no rebuild): VLLM_MOET_SM120_GEMV=0 (CUTLASS instead of the dense MXFP8
# GEMV), VLLM_MOET_GEMV_IMPL=v1 (the pre-2026-09-20 GEMV kernel instead of v3), VLLM_MOET_SM120_GEMV_BMM=0
# (BF16 emulation for wo_a), VLLM_MOET_DECODER_REPLAY=0 (the -ced image prefills layers 21-39 on every
# prompt token again instead of the last 128). enable_adaptive_verification must stay
# false on this path (DeepseekV4IndexerBackend does not support it on sm_120).
# Reasoning effort: the image renders the checkpoint's tiers (low 50 / high 75 / max 100,
# default high = 75; OpenAI aliases minimal 25 / medium 62 / xhigh 87 are accepted). Pin a
# budget per request with chat_template_kwargs {"reasoning_effort": <1..100>} or the OpenAI
# `reasoning_effort` field; "none" or {"thinking": false} turns thinking off.
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR (DeepSeek-V4.1-Flash checkpoint directory) is required}"
IMAGE="${IMAGE:-vllm-moet-sm120:dsv41-nightly-20260923-ced}"
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
INDEXER_KV_DTYPE="${INDEXER_KV_DTYPE:-mxfp4}"
KV_RECORD="${KV_RECORD:-nvfp4}"
LANGUAGE_ONLY="${LANGUAGE_ONLY:-0}"
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
if [[ "$SPEC_TOKENS" != "0" ]]; then
  ARGS+=(--speculative-config
    "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false}")
fi
[[ "$LANGUAGE_ONLY" == "1" ]] && ARGS+=(--language-model-only)
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
  # INDEXER_KV_DTYPE would load as garbage, so that format gets its own subdirectory.
  FS_ROOT="$KV_OFFLOAD_FS_DIR"
  [[ "$INDEXER_KV_DTYPE" != "mxfp4" ]] && FS_ROOT="$KV_OFFLOAD_FS_DIR/indexer-$INDEXER_KV_DTYPE"
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
case "$INDEXER_KV_DTYPE" in
  fp8) ;;
  mxfp4) ARGS+=(--attention-config "{\"indexer_kv_dtype\":\"mxfp4\"}") ;;
  *) echo "INDEXER_KV_DTYPE must be fp8 or mxfp4, got $INDEXER_KV_DTYPE" >&2; exit 1 ;;
esac
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
