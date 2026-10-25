#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 4x RTX 5090 (sm_120), TP4, as the MoET hybrid: the resident experts (a static placement
# from routing frequencies, plus promote-on-miss into spare slots) in VRAM through b12x's compact W4A8 path, every
# other routed expert of a decode step computed on the CPU from the pinned expert store, bit-identical to the
# all-resident b12x path (tools/dsv41_5090/cpu_tier/, /root/work/HANDOFF-moet-hybrid.md). Phase 2.4 bring-up:
# prompts <= 8 tokens take the hybrid path; longer prefills run b12x's non-compact kernels with the missing experts
# as zero rows (phase 2.5 adds the streamed prefill) - this launcher is for the decode bring-up, not for serving yet.
#
# Required:  MODEL_DIR   the official DeepSeek-V4.1-Flash checkpoint
# Optional:  IMAGE (vllm-moet-sm120:dsv41-nightly-df3bd094-hyb1)  NAME (ds41-hyb)  GPUS (0,1,2,3)  TP (4)  PORT (8041)
#            BIND (127.0.0.1:)       CACHE_DIR (/mnt/nvme/cache/ds41)  REPO (/mnt/nvme/vLLM-Moet-cpu: patcher + runtime)
#            STORE (/mnt/nvme/packs/dsv41-hybrid)   cpu_moe/build_store.py's files (269 GB on NVMe)
#            STORE_MODE (nonresident)  which records the tier pins: nonresident (52 GB per rank) or all (72 GB per rank)
#            PLACEMENT ($REPO/internal/dsv41-5090-cpu/routes/placement-0.28-uniform.json)  routing/placement.py's file;
#                        RESIDENT=<int> instead = the first N experts of every layer (a test placement)
#            SPARE (0)               empty slots per layer for promote-on-miss; PROMOTIONS (4) per layer per step
#            TIMEOUT_MS (500)        the in-graph wait's bound (fail-closed)
#            SPEC_TOKENS (5)         DSpark draft tokens (0 = no drafter)
#            GPU_MEM_UTIL (0.92)  MAX_MODEL_LEN (65536)  MAX_NUM_SEQS (1)  MAX_NUM_BATCHED_TOKENS (512)  CAPTURE_SIZE (8)
#            LOG_EVERY (100)         KPI line period (steps);  TIER_DEBUG (0)
#            SCRATCH (0)             slots of the prefill scratch pool (phase 2.5; 128 = 1.2 GB per rank; 0 = the missing
#                        experts of > 8-token calls are not computed: decode bring-up only)
#            DEBUG_COMPACT (0)       1 + EXTRA_ARGS="--enforce-eager" + SCRATCH >= 64: every compact call recomputed with all
#                        its routed experts in the scratch pool and compared bit for bit (the within-boot identity check)
#            DEBUG_PREFILL (0)       1: the same for the > 8-token calls (SCRATCH >= the call's distinct experts)
#            POOL_SPIN_US ("")       the kernel pool's spin before it sleeps on the futex (the pool's default 500; the cores
#                        then drop into C2 and the kernel runs up to 2x slower after ms-long gaps: 5000 spins through a
#                        decode step's gaps and sleeps only when the API is idle; -1 = spin forever)
#            NCCL_P2P_LEVEL (SYS)    NCCL_P2P_DISABLE (0)   NCCL_CHANNELS ("" = NCCL's choice; 4 pins NCCL_MIN/MAX_NCHANNELS: on
#                        this box NCCL picks 2 or 4 channels per boot, and the all-reduce decomposition changes the
#                        outputs by ulps -> bit-identity across boots needs the pin)
#            EXTRA_ARGS / EXTRA_DOCKER_ARGS appended to `vllm serve` / `docker run`
# GPU protocol: this script does NOT take the locks - run it as
#   flock -w 3600 /tmp/exl3diff-gpu0.lock flock -w 3600 /tmp/exl3diff-gpu1.lock flock -w 3600 /tmp/exl3diff-gpu2.lock \
#     flock -w 3600 /tmp/exl3diff-gpu3.lock bash -c 'MODEL_DIR=... docker/sm120/run-dsv41-5090.sh && docker wait ds41-hyb'
# and announce the window on /root/work/GPU-BOARD.md.
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR (DeepSeek-V4.1-Flash checkpoint directory) is required}"
IMAGE="${IMAGE:-vllm-moet-sm120:dsv41-nightly-df3bd094-hyb1}"
NAME="${NAME:-ds41-hyb}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
PORT="${PORT:-8041}"
BIND="${BIND-127.0.0.1:}"
CACHE_DIR="${CACHE_DIR:-/mnt/nvme/cache/ds41}"
REPO="${REPO:-/mnt/nvme/vLLM-Moet-cpu}"
STORE="${STORE:-/mnt/nvme/packs/dsv41-hybrid}"
STORE_MODE="${STORE_MODE:-nonresident}"
PLACEMENT="${PLACEMENT:-$REPO/internal/dsv41-5090-cpu/routes/placement-0.28-uniform.json}"
RESIDENT="${RESIDENT:-}"
SPARE="${SPARE:-0}"
PROMOTIONS="${PROMOTIONS:-4}"
TIMEOUT_MS="${TIMEOUT_MS:-500}"
LOG_EVERY="${LOG_EVERY:-100}"
TIER_DEBUG="${TIER_DEBUG:-0}"
SCRATCH="${SCRATCH:-0}"
DEBUG_COMPACT="${DEBUG_COMPACT:-0}"
DEBUG_PREFILL="${DEBUG_PREFILL:-0}"
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
NCCL_CHANNELS="${NCCL_CHANNELS:-}"
POOL_SPIN_US="${POOL_SPIN_US:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
TABLES="$REPO/internal/dsv41-5090-cpu/parity"

[[ -f "$MODEL_DIR/config.json" ]] || { echo "no config.json in MODEL_DIR=$MODEL_DIR" >&2; exit 1; }
[[ -f "$STORE/manifest.json" ]] || { echo "no manifest.json in STORE=$STORE (cpu_moe/build_store.py)" >&2; exit 1; }
[[ -f "$TABLES/b12x_act_m_limit10.f32" && -f "$TABLES/b12x_layout_w13.npy" ]] || { echo "kernel tables missing in $TABLES" >&2; exit 1; }
mkdir -p "$CACHE_DIR/dot-cache"

TIER_ENV=(-e VLLM_MOET_MOE_CPU_TIER=1 -e VLLM_MOET_CPU_TIER_STORE=/store -e VLLM_MOET_CPU_TIER_STORE_MODE="$STORE_MODE"
          -e VLLM_MOET_CPU_TIER_TABLES=/parity -e VLLM_MOET_CPU_TIER_SPARE="$SPARE" -e VLLM_MOET_CPU_TIER_MAX_PROMOTIONS="$PROMOTIONS"
          -e VLLM_MOET_CPU_TIER_TIMEOUT_MS="$TIMEOUT_MS" -e VLLM_MOET_CPU_TIER_LOG=1 -e VLLM_MOET_CPU_TIER_LOG_EVERY="$LOG_EVERY"
          -e VLLM_MOET_CPU_TIER_DEBUG="$TIER_DEBUG" -e VLLM_MOET_DET_TOPK=1
          -e VLLM_MOET_CPU_TIER_SCRATCH="$SCRATCH" -e VLLM_MOET_CPU_TIER_DEBUG_COMPACT="$DEBUG_COMPACT"
          -e VLLM_MOET_CPU_TIER_DEBUG_PREFILL="$DEBUG_PREFILL")
MOUNTS=(-v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR/dot-cache:/root/.cache" -v "$REPO:/repo:ro" -v "$TABLES:/parity:ro" -v "$STORE:/store:ro")
# the store may be a directory of symlinks into another store (the mini-real test): mount the packs root too
MOUNTS+=(-v /mnt/nvme/packs:/mnt/nvme/packs:ro)
if [[ -n "$RESIDENT" ]]; then
  TIER_ENV+=(-e VLLM_MOET_CPU_TIER_RESIDENT="$RESIDENT")
else
  [[ -f "$PLACEMENT" ]] || { echo "no placement file $PLACEMENT" >&2; exit 1; }
  MOUNTS+=(-v "$PLACEMENT:/placement.json:ro")
  TIER_ENV+=(-e VLLM_MOET_CPU_TIER_PLACEMENT=/placement.json)
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
  -e NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}" -e NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
  ${NCCL_CHANNELS:+-e NCCL_MIN_NCHANNELS=$NCCL_CHANNELS -e NCCL_MAX_NCHANNELS=$NCCL_CHANNELS} \
  ${POOL_SPIN_US:+-e MOE_B12X_POOL_SPIN_US=$POOL_SPIN_US} \
  -e TP="$TP" -e SPEC_TOKENS="${SPEC_TOKENS:-5}" -e GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}" -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}" -e CAPTURE_SIZE="${CAPTURE_SIZE:-8}" \
  "${TIER_ENV[@]}" \
  ${EXTRA_DOCKER_ARGS} \
  --entrypoint bash "$IMAGE" \
  /repo/tools/dsv41_5090/cpu_tier/serve_hybrid.sh /model "$PORT" ${EXTRA_ARGS}

echo "container $NAME started; logs: docker logs -f $NAME ; API: http://${BIND:-0.0.0.0:}${PORT}/v1"
echo "expect in the log: '[moet-cpu-tier] rank N: store /store (...) records = ... GB pinned', 'tier started: 40 layers x 384 experts',"
echo "  then 'compact path with N tokens' at the first decode and 'step 100: ... pair hit rate ...' KPI lines"
exit 0
