#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 4x RTX 5090 (sm_120), TP4, as the MoET hybrid: the resident experts (a static placement
# from routing frequencies, plus promote-on-miss into spare slots) in VRAM through b12x's compact W4A8 path, every
# other routed expert of a decode step computed on the CPU from the pinned expert store, bit-identical to the
# all-resident b12x path (tools/dsv41_5090/cpu_tier/, /root/work/HANDOFF-moet-hybrid.md). Defaults = the serving
# configuration measured 2026-10-25 (40 static + 80 LRU expert slots per layer per rank, promotion admission window
# 12, exact-size Engram tables, the streamed prefill in 4096-token chunks through a 64-slot scratch pool, no DSpark):
# decode 143 / 120 tok/s on short prompts, prefill 2.1-2.4K tok/s at 2K and 3.9K tok/s at 14K, MemAvailable ~45 GB while up,
# boot ~6 min; HumanEval 159 vs native 160, MATH-500 L5 55 vs 55 (paired).
#
# Required:  MODEL_DIR   the official DeepSeek-V4.1-Flash checkpoint
# Optional:  IMAGE (vllm-moet-sm120:dsv41-nightly-df3bd094-hyb2: the hybrid baked in, see BAKED)  NAME (ds41-hyb)  GPUS (0,1,2,3)  TP (4)  PORT (8041)
#            BIND (127.0.0.1:)       CACHE_DIR (/mnt/nvme/cache/ds41)  REPO (/mnt/nvme/vLLM-Moet-cpu: patcher + runtime)
#            STORE (/mnt/nvme/packs/dsv41-hybrid)   cpu_moe/build_store.py's files (269 GB on NVMe)
#            STORE_MODE (nonresident)  which records the tier pins: nonresident (65 GB per rank with 40 static) or all (72 GB per rank)
#            STORE_EARLY (0)         1 = pin the records at the last MoE layer's create_weights, i.e. before the weights are
#                        loaded (needs PLACEMENT). Measured 10-28 (k0-baked-early2): it boots, but the loader then streams
#                        290 GB of shards through the ~45 GB of page cache the pinned memory leaves - boot 527 s against
#                        348-394 with the late pin + the window script's two cache drops; more compaction afterwards. Off.
#            PLACEMENT ($REPO/internal/dsv41-5090-cpu/routes/placement-0.1042-uniform.json = 40 static experts per layer;
#                        0.28 = 107, 0.2526 = 97)  routing/placement.py's file;
#                        RESIDENT=<int> instead = the first N experts of every layer (a test placement)
#            SPARE (80)              LRU slots per layer for promote-on-miss (40 + 80 = the 120 slots of the 0.92 VRAM budget);
#                        PROMOTIONS (4) per layer per step
#            TIMEOUT_MS (500)        the in-graph wait's bound (fail-closed)
#            SPEC_TOKENS (0)         DSpark draft tokens (0 = no drafter: k = 5 is CPU-bound with the m > 1 kernel)
#            GPU_MEM_UTIL (0.94; 0.92 until 10-29 - the second prefill scratch set needs the 0.64 GiB)  MAX_MODEL_LEN (65536)  MAX_NUM_SEQS (1)  MAX_NUM_BATCHED_TOKENS (4096: the prefill chunk; 14K prompts
#                        prefill at 3.9K tok/s against 2.2K with 2048)  CAPTURE_SIZE (8)
#            LOG_EVERY (100)         KPI line period (steps);  TIER_DEBUG (0)
#            SCRATCH (64)            slots of the prefill scratch pool (phase 2.5; 64 = 0.6 GB per rank, 128 = 1.2; 0 = the missing
#                        experts of > 8-token calls are not computed: decode-only diagnostics)
#            DEBUG_COMPACT (0)       1 + EXTRA_ARGS="--enforce-eager" + SCRATCH >= 64: every compact call recomputed with all
#                        its routed experts in the scratch pool and compared bit for bit (the within-boot identity check)
#            DEBUG_PREFILL (0)       1: the same for the > 8-token calls (SCRATCH >= the call's distinct experts)
#            ENGRAM_THP (1)          1 = --engram-config {"use_thp":true}: vLLM backs the Engram tables with an exact-size mmap
#                        (+ MADV_HUGEPAGE, prefaulted, cudaHostRegister) instead of torch's pinned allocator, which rounds
#                        each 24.6 GB table up to 32 GiB (10-25: 2 tables x 4 ranks = ~80 GB of host RAM lost); needs free,
#                        unfragmented memory at boot (DROP_CACHES=1 on the window script)
#            PREFIX_CACHE (1)        0 = --no-enable-prefix-caching (measurements: repeated / shared-prefix prompts would hit
#                        vLLM's prefix cache and skip the prefill)
#            PROMOTE_WINDOW (12)     promotion admission: 0 = every miss; W > 0 = only an expert whose previous miss was <= W
#                        steps ago (the traces: W = 8-16 keeps the hit rate within 0.6 pt at a quarter of the promotions / DMA)
#            POOL_THREADS (6)        kernel pool threads per rank = cores of its CCD: 6 (8r+2..8r+7), 7 (8r+1..8r+7, the dispatcher on
#                        main's SMT sibling) or 8 (the whole CCD; main + dispatcher on siblings) - 10-28: the GPU waits
#                        ~2.6 ms per decode step for the tier, the bench kernel is 14-17 % faster on 8 threads than on 6
#            POOL_SPIN_HOT (1)       the kernel pool's spin does ALU work instead of `pause` (keeps the cores' clock up between a
#                        step's jobs: +3 % decode on 10-25); 0 = pause
#            HOST_EMPTY_CACHE (0)    1 = torch._C._host_emptyCache() after the weights are loaded (the RAM item: the loader's
#                        pinned staging blocks), with MemAvailable logged before / after
#            POOL_SPIN_US (5000)     the kernel pool's spin before it sleeps on the futex (the pool's own default 500; the
#                        cores then drop into C2 and the kernel runs up to 2x slower after ms-long gaps: 5000 spins
#                        through a decode step's gaps and sleeps only when the API is idle; -1 = spin forever). Default
#                        since 10-29 (k0-pf, with PREFETCH=1: in-kernel 127-135 us per expert against 148-154, the wait
#                        2.06-2.12 ms against 2.62-3.11, step 7.61 against 8.40 ms, decode 148 / 129 against 143 / 120)
#            PREFETCH (1)            MOE_B12X_PREFETCH: the kernel's software prefetch, D rows / row-groups ahead (the expert
#                        from DRAM costs 119 us in-kernel against 69 from L3; bits untouched by construction, parity PASS);
#                        PREFETCH_HINT (1 = L2). Default since 10-29 (k0-pf, see POOL_SPIN_US); 0 = off
#            PREFILL_OVERLAP (1)     a second prefill scratch set + a side stream: a non-compact call's round 0 is staged
#                        during the stock b12x run and round r + 1 during round r's extra run - bit-identical to the
#                        serial path (mini-real A/B), TTFT -15 % at 2K / 14K / 30K (k0-pfill vs k0-pfill-base 10-29: 0.94 ->
#                        0.80 s, 3.8 -> 3.25 s, 7.7 -> 6.4 s; 0.12 -> 0.09 ms per streamed expert). Costs SCRATCH x 4.7 MB x 2
#                        of VRAM - hence GPU_MEM_UTIL 0.94 (0.92 would leave the KV at the 64K minimum). 0 = off;
#                        DEBUG_OVERLAP=1 re-runs every call serially and compares bit for bit
#            SPIN_TOUCH_STRIDE ("")  MOE_B12X_POOL_SPIN_TOUCH_STRIDE for POOL_SPIN_HOT=2 (the kernel's default is 64 since
#                        10-29 = every worker streams its 32 MB buffer sequentially while idle; 4160 = the old page stride).
#                        POOL_SPIN_HOT=2 measured 10-29 (k0-pf-touch): in-kernel 115-126 us per expert on every rank, the
#                        wait 1.84-1.90 ms, decode 150.6 / 131.6 - at ~19 GB/s per CCD of idle DRAM reads and a slower
#                        host under the profiler (rank 0's step 10.1 ms profiled); not the default until k0-new-touch
#            NCCL_P2P_LEVEL (SYS)    NCCL_P2P_DISABLE (0)   NCCL_CHANNELS ("" = NCCL's choice; 4 pins NCCL_MIN/MAX_NCHANNELS: on
#                        this box NCCL picks 2 or 4 channels per boot, and the all-reduce decomposition changes the
#                        outputs by ulps -> bit-identity across boots needs the pin)
#            BAKED (1)               1 = run the image's own copy of the hybrid (Dockerfile step 28: patched packages, runtime,
#                        libcpu_tier.so) without mounting the worktree or re-running the patcher at start (IMAGE must have the
#                        layer: vllm-moet-sm120:dsv41-nightly-df3bd094-hyb2; verified in the TP4 boot k0-baked-early2, 10-28).
#                        After a change under tools/dsv41_5090/ rebuild the image (layer cache, seconds):
#                          DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly \
#                            --label com.vllm-moet.cpu-tier-commit=$(git rev-parse --short HEAD) -t vllm-moet-sm120:dsv41-nightly-df3bd094-hyb2 .
#                        0 = the dev flow (IMAGE …-hyb1 or the base, the worktree mounted at /repo, the patcher at start, ~60 s)
#            POWER_CAP_W (CAP_W of /etc/default/gpu-power-cap, else 400)  the GPUs' power limit this launcher re-applies before
#                        the boot when a card sits above it (the box's cap after GPU 0 fell off the bus twice under full
#                        power, 2026-10-25; gpu-power-cap.service sets it at boot); POWER_CAP_W=0 skips the check
#            EXTRA_ARGS / EXTRA_DOCKER_ARGS appended to `vllm serve` / `docker run`
# GPU protocol: this script does NOT take the locks - run it as
#   flock -w 3600 /tmp/exl3diff-gpu0.lock flock -w 3600 /tmp/exl3diff-gpu1.lock flock -w 3600 /tmp/exl3diff-gpu2.lock \
#     flock -w 3600 /tmp/exl3diff-gpu3.lock bash -c 'MODEL_DIR=... docker/sm120/run-dsv41-5090.sh && docker wait ds41-hyb'
# and announce the window on /root/work/GPU-BOARD.md.
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR (DeepSeek-V4.1-Flash checkpoint directory) is required}"
BAKED="${BAKED:-1}"
# the baked image by default; the dev flow (BAKED=0) defaults to the image without step 28 (the patcher runs at start)
if [[ "$BAKED" == "1" ]]; then
  IMAGE="${IMAGE:-vllm-moet-sm120:dsv41-nightly-df3bd094-hyb2}"
else
  IMAGE="${IMAGE:-vllm-moet-sm120:dsv41-nightly-df3bd094-hyb1}"
fi
NAME="${NAME:-ds41-hyb}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
PORT="${PORT:-8041}"
BIND="${BIND-127.0.0.1:}"
CACHE_DIR="${CACHE_DIR:-/mnt/nvme/cache/ds41}"
REPO="${REPO:-/mnt/nvme/vLLM-Moet-cpu}"
STORE="${STORE:-/mnt/nvme/packs/dsv41-hybrid}"
STORE_MODE="${STORE_MODE:-nonresident}"
STORE_EARLY="${STORE_EARLY:-0}"
PLACEMENT="${PLACEMENT:-$REPO/internal/dsv41-5090-cpu/routes/placement-0.1042-uniform.json}"
RESIDENT="${RESIDENT:-}"
SPARE="${SPARE:-80}"
PROMOTIONS="${PROMOTIONS:-4}"
TIMEOUT_MS="${TIMEOUT_MS:-500}"
LOG_EVERY="${LOG_EVERY:-100}"
TIER_DEBUG="${TIER_DEBUG:-0}"
SCRATCH="${SCRATCH:-64}"
DEBUG_COMPACT="${DEBUG_COMPACT:-0}"
DEBUG_PREFILL="${DEBUG_PREFILL:-0}"
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
NCCL_CHANNELS="${NCCL_CHANNELS:-}"
POOL_SPIN_US="${POOL_SPIN_US:-5000}"
PREFETCH="${PREFETCH:-1}"
PREFETCH_HINT="${PREFETCH_HINT:-1}"
SPIN_TOUCH_STRIDE="${SPIN_TOUCH_STRIDE:-}"
PREFILL_OVERLAP="${PREFILL_OVERLAP:-1}"
DEBUG_OVERLAP="${DEBUG_OVERLAP:-0}"
POOL_SPIN_HOT="${POOL_SPIN_HOT:-1}"
POOL_THREADS="${POOL_THREADS:-6}"
PROMOTE_WINDOW="${PROMOTE_WINDOW:-12}"
HOST_EMPTY_CACHE="${HOST_EMPTY_CACHE:-0}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
ENGRAM_THP="${ENGRAM_THP:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
[[ "$ENGRAM_THP" == "1" ]] && EXTRA_ARGS="--engram-config {\"use_thp\":true} $EXTRA_ARGS"
[[ "$PREFIX_CACHE" == "1" ]] || EXTRA_ARGS="--no-enable-prefix-caching $EXTRA_ARGS"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
TABLES="$REPO/internal/dsv41-5090-cpu/parity"

[[ -f "$MODEL_DIR/config.json" ]] || { echo "no config.json in MODEL_DIR=$MODEL_DIR" >&2; exit 1; }
[[ -f "$STORE/manifest.json" ]] || { echo "no manifest.json in STORE=$STORE (cpu_moe/build_store.py)" >&2; exit 1; }
[[ -f "$TABLES/b12x_act_m_limit10.f32" && -f "$TABLES/b12x_layout_w13.npy" ]] || { echo "kernel tables missing in $TABLES" >&2; exit 1; }
mkdir -p "$CACHE_DIR/dot-cache"

TIER_ENV=(-e VLLM_MOET_MOE_CPU_TIER=1 -e VLLM_MOET_CPU_TIER_STORE=/store -e VLLM_MOET_CPU_TIER_STORE_MODE="$STORE_MODE" -e VLLM_MOET_CPU_TIER_STORE_EARLY="$STORE_EARLY"
          -e VLLM_MOET_CPU_TIER_TABLES=/parity -e VLLM_MOET_CPU_TIER_SPARE="$SPARE" -e VLLM_MOET_CPU_TIER_MAX_PROMOTIONS="$PROMOTIONS"
          -e VLLM_MOET_CPU_TIER_TIMEOUT_MS="$TIMEOUT_MS" -e VLLM_MOET_CPU_TIER_LOG=1 -e VLLM_MOET_CPU_TIER_LOG_EVERY="$LOG_EVERY"
          -e VLLM_MOET_CPU_TIER_DEBUG="$TIER_DEBUG" -e VLLM_MOET_DET_TOPK=1
          -e VLLM_MOET_CPU_TIER_SCRATCH="$SCRATCH" -e VLLM_MOET_CPU_TIER_DEBUG_COMPACT="$DEBUG_COMPACT"
          -e VLLM_MOET_CPU_TIER_DEBUG_PREFILL="$DEBUG_PREFILL" -e VLLM_MOET_CPU_TIER_PROMOTE_WINDOW="$PROMOTE_WINDOW"
          -e VLLM_MOET_CPU_TIER_HOST_EMPTY_CACHE="$HOST_EMPTY_CACHE" -e MOE_B12X_POOL_SPIN_HOT="$POOL_SPIN_HOT"
          -e VLLM_MOET_CPU_TIER_POOL_THREADS="$POOL_THREADS")
MOUNTS=(-v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR/dot-cache:/root/.cache" -v "$TABLES:/parity:ro" -v "$STORE:/store:ro")
if [[ "$BAKED" == "1" ]]; then
  ENTRY=/opt/vllm-moet/dsv41_5090/cpu_tier/serve_hybrid.sh
  TIER_ENV+=(-e HYBRID_PATCH_AT_START=0)
else
  MOUNTS+=(-v "$REPO:/repo:ro")
  ENTRY=/repo/tools/dsv41_5090/cpu_tier/serve_hybrid.sh
fi
# the store may be a directory of symlinks into another store (the mini-real test): mount the packs root too
MOUNTS+=(-v /mnt/nvme/packs:/mnt/nvme/packs:ro)
if [[ -n "$RESIDENT" ]]; then
  TIER_ENV+=(-e VLLM_MOET_CPU_TIER_RESIDENT="$RESIDENT")
else
  [[ -f "$PLACEMENT" ]] || { echo "no placement file $PLACEMENT" >&2; exit 1; }
  MOUNTS+=(-v "$PLACEMENT:/placement.json:ro")
  TIER_ENV+=(-e VLLM_MOET_CPU_TIER_PLACEMENT=/placement.json)
fi

# the GPUs' power cap (see the header): re-applied here when a card is above it (the launcher runs as root on the host)
if [[ -z "${POWER_CAP_W:-}" ]]; then
  POWER_CAP_W=$( (. /etc/default/gpu-power-cap 2>/dev/null; echo "${CAP_W:-400}") )
fi
if [[ "$POWER_CAP_W" != "0" ]]; then
  over=$(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v cap="$POWER_CAP_W" '$2 > cap + 0.5 {printf "%s ", $1}')
  if [[ -n "$over" ]]; then
    echo "GPU(s) $over above the ${POWER_CAP_W} W cap - re-applying it (nvidia-smi -pl ${POWER_CAP_W})"
    nvidia-smi -pl "$POWER_CAP_W" > /dev/null 2>&1 || echo "WARNING: nvidia-smi -pl ${POWER_CAP_W} failed (not root?)"
  fi
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
  ${PREFETCH:+-e MOE_B12X_PREFETCH=$PREFETCH -e MOE_B12X_PREFETCH_HINT=$PREFETCH_HINT} \
  ${SPIN_TOUCH_STRIDE:+-e MOE_B12X_POOL_SPIN_TOUCH_STRIDE=$SPIN_TOUCH_STRIDE} \
  -e VLLM_MOET_CPU_TIER_PREFILL_OVERLAP="$PREFILL_OVERLAP" -e VLLM_MOET_CPU_TIER_DEBUG_OVERLAP="$DEBUG_OVERLAP" \
  -e TP="$TP" -e SPEC_TOKENS="${SPEC_TOKENS:-0}" -e GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}" -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}" -e CAPTURE_SIZE="${CAPTURE_SIZE:-8}" \
  "${TIER_ENV[@]}" \
  ${EXTRA_DOCKER_ARGS} \
  --entrypoint bash "$IMAGE" \
  "$ENTRY" /model "$PORT" ${EXTRA_ARGS}

echo "container $NAME started; logs: docker logs -f $NAME ; API: http://${BIND:-0.0.0.0:}${PORT}/v1"
echo "expect in the log: '[moet-cpu-tier] rank N: store /store (...) records = ... GB pinned', 'tier started: 40 layers x 384 experts',"
echo "  then 'compact path with N tokens' at the first decode and 'step 100: ... pair hit rate ...' KPI lines"
exit 0
