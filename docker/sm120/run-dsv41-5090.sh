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
#            SPARE (70)              LRU slots per layer for promote-on-miss (40 + 70 = 110 slots since 11-03: ten slots = 1.75 GiB
#                        per card traded for the 512K context's KV - short-prompt decode 144.7 / 125.3 tok/s, inside the
#                        120-slot band; 40 + 80 = 120 was the default 10-25 .. 11-03);  PROMOTIONS (4) per layer per step
#            TIMEOUT_MS (500)        the in-graph wait's bound (fail-closed)
#            SPEC_TOKENS (0)         DSpark draft tokens (0 = no drafter: k = 5 is CPU-bound with the m > 1 kernel)
#            GPU_MEM_UTIL (0.90 since 11-03 with SPARE=70: KV 1.28 GiB = 1.98 x a 512K request and ~1.3 GiB outside the budget for the
#                        indexer's runtime peak; 0.94 from 10-29 to 11-03 with 120 slots and the 64K max-model-len; 0.92 before)
#            MAX_MODEL_LEN (524288 since 11-03: needle 3/3 at 512-519K, cold prefill 4.3-4.6K tok/s = TTFT ~2 min, decode 82 tok/s behind
#                        512K; 65536 before)  INDEXER_LOGITS_MB (256)  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB: the sparse-attention indexer's
#                        prefill logits budget (vLLM default 512) - with the 512K max-model-len at util 0.94 the indexer OOMed on the 4th
#                        chunk of a 128K prompt (its runtime peak is ~2x the budget: DeepGEMM output + fp32 upcast + top-k, and the
#                        per-request buffers sized by max_model_len eat the ~190 MiB outside the budget); 256 + util 0.90 ran 128K / 256K /
#                        512K prompts and the 512K needles clean
#            MAX_NUM_SEQS (1)  MAX_NUM_BATCHED_TOKENS (4096: the prefill chunk; 14K prompts prefill at 3.9K tok/s against 2.2K with 2048)
#            CAPTURE_SIZE (8)
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
#            POOL_SPIN_HOT (1)       the kernel pool's idle spin: 1 = ALU work instead of `pause` (keeps the cores' clock up between
#                        a step's jobs: +3 % decode on 10-25); 2 = also every idle worker streams a private 32 MB buffer from
#                        DRAM (SPIN_TOUCH_STRIDE 64) - the kernel's after-a-gap penalty is a per-core memory-path state that
#                        only continuous DRAM-missing loads keep away (note §20.2): in situ -8 % per expert on every rank in
#                        10-min windows (k0-pf-touch, k0-new-touch: 128-138 -> 113-134 us, decode 148 / 129 -> 150 / 132) -
#                        BUT NOT FOR LONG: in a 30-min single-stream soak the kernel jumps 118 -> 158 us after ~18 min and the
#                        stream's per-stream tok/s ends 14 % below the plain spin's (k0-therm-new vs k0-therm-cool, note §20.11:
#                        ~77 GB/s of idle DRAM reads heat the memory side until it throttles; the plain spin is flat). 2 was the
#                        default 10-30..11-03; 1 again since 11-03. 0 = pause
#            HOST_EMPTY_CACHE (0)    1 = torch._C._host_emptyCache() after the weights are loaded (the RAM item: the loader's
#                        pinned staging blocks), with MemAvailable logged before / after
#            POOL_SPIN_US (5000)     the kernel pool's spin before it sleeps on the futex (the pool's own default is 500; the
#                        cores then drop into C2 and the kernel runs up to 2x slower after ms-long gaps; 5000 spins through
#                        a decode step's gaps and sleeps only when the API is idle; -1 = spin forever). 5000 since 10-29
#                        (k0-pf, with PREFETCH=1: decode 148 / 129), back to 500 for a day while the 11-02 thermal pair was
#                        read - the cliff after 18 min was the DRAM stream of POOL_SPIN_HOT=2, not the spin: k0-therm-spin
#                        (11-03, 5000 + POOL_SPIN_HOT=1, a 30-min single-stream soak with the thermal trace) is as flat as
#                        k0-therm-cool (500): the kernel 127-132 us per expert throughout, per-minute tok/s -0.1 % first vs
#                        second half, Tctl max 85 C, and decode 148.7 / 129.3 vs 145.4 / 124.2 at the start (note §20.12;
#                        the soak aggregates 86.3 vs 87.6 tok/s are inside the +-3.5 % boot-to-boot band and on a different
#                        prompt mix). Under load both settings sit at the socket's 350 W PPT; idle, the pool sleeps after 5 ms
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
#                        10-29 = every worker streams its 32 MB buffer sequentially while idle; 4160 = the old page stride,
#                        which lost in situ on 10-29)
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
#   ---- the low-RAM tiers (2026-11-04, branch moet-hybrid-engram-v2; all inert at their defaults = the 512 GB configuration) ----
#            ENGRAM_MODE (ram)       ram = vLLM's pinned host tables (2 x 23.6 GiB per rank, 189 GiB); nvme = Engram v2: no pinned
#                        tables - the rows come from the checkpoint's own safetensors shards on NVMe through a DRAM row cache
#                        and an io_uring fetcher thread per rank (tools/dsv41_5090/engram_v2/), vLLM's own lookup kernel
#                        dequantizes them from a device landing buffer (bit-identical rows). RAM 457 -> ~275 GiB used. Needs
#                        the worktree mounted (the engram patcher runs at container start, ~30 s: the image has no engram_v2
#                        layer) and docker/sm120/seccomp-io_uring.json (docker's default seccomp denies io_uring) - both added
#                        here automatically. The 384 GB tier.
#            ENGRAM_CACHE_GIB (2)    the DRAM row cache per rank (plain memory, split over the two Engram layers; 8M rows at 2 GiB)
#            ENGRAM_FETCH_CPU ("")   the fetcher thread's CPU ("" = 8*rank+33, the dispatcher core's SMT sibling)
#            ENGRAM_QD (64)          io_uring depth per rank;  ENGRAM_TIMEOUT_MS (2000) the in-stream wait's bound (fail-closed)
#            ENGRAM_LOOKAHEAD (1)    the prompt's rows hashed on the host and the next ENGRAM_LOOKAHEAD_CHUNKS (4) chunks' rows
#                        prefetched into the row cache during a prefill (only the first chunk of a prompt pays the NVMe latency)
#            STORE_NVME (0)          1 = the expert store's cold tail on NVMe: only STORE_DRAM_RECORDS records per layer per rank
#                        stay pinned in DRAM (an LRU), the rest are read from the store files on a miss (io_uring, 4.7 MB reads
#                        into a pinned staging record, kernel v2 computes from it; the prefill streaming and the promotions
#                        read their cold records the same way). The 256 GB (N ~ 230) and 128 GB (N ~ 110) tiers.
#            STORE_DRAM_RECORDS (0)  records per layer per rank kept in DRAM when STORE_NVME=1 (0 = all 344 = today's store)
#            STORE_NVME_QD (8)       io_uring depth of the record reads per rank
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
# the worktree this launcher lives in (= /mnt/nvme/vLLM-Moet-cpu in the main worktree; another worktree's copy mounts itself)
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENGRAM_MODE="${ENGRAM_MODE:-ram}"
ENGRAM_CACHE_GIB="${ENGRAM_CACHE_GIB:-2}"
ENGRAM_FETCH_CPU="${ENGRAM_FETCH_CPU:-}"
ENGRAM_QD="${ENGRAM_QD:-64}"
ENGRAM_TIMEOUT_MS="${ENGRAM_TIMEOUT_MS:-2000}"
ENGRAM_LOOKAHEAD="${ENGRAM_LOOKAHEAD:-1}"
ENGRAM_LOOKAHEAD_CHUNKS="${ENGRAM_LOOKAHEAD_CHUNKS:-4}"
STORE_NVME="${STORE_NVME:-0}"
STORE_DRAM_RECORDS="${STORE_DRAM_RECORDS:-0}"
STORE_NVME_QD="${STORE_NVME_QD:-8}"
STORE="${STORE:-/mnt/nvme/packs/dsv41-hybrid}"
STORE_MODE="${STORE_MODE:-nonresident}"
STORE_EARLY="${STORE_EARLY:-0}"
PLACEMENT="${PLACEMENT:-$REPO/internal/dsv41-5090-cpu/routes/placement-0.1042-uniform.json}"
RESIDENT="${RESIDENT:-}"
SPARE="${SPARE:-70}"
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
# the low-RAM tiers (ENGRAM_MODE=nvme / STORE_NVME=1): io_uring needs a seccomp profile that allows it; the engram_v2
# runtime is not baked into any image - the worktree is mounted and its patcher runs at container start
SECCOMP_ARGS=()
if [[ "$ENGRAM_MODE" == "nvme" || "$STORE_NVME" == "1" ]]; then
  [[ -f "$REPO/docker/sm120/seccomp-io_uring.json" ]] || { echo "missing $REPO/docker/sm120/seccomp-io_uring.json" >&2; exit 1; }
  SECCOMP_ARGS=(--security-opt "seccomp=$REPO/docker/sm120/seccomp-io_uring.json")
fi
if [[ "$ENGRAM_MODE" == "nvme" ]]; then
  [[ "$BAKED" == "1" ]] && MOUNTS+=(-v "$REPO:/repo:ro")
  # the weight iterator's skip of the 2 x 98 GB engram.embed tensors must be in the workers' environment before vLLM's
  # loader modules import (the runtime sets it too, but engram.py may import after ep_weight_filter.py)
  TIER_ENV+=(-e ENGRAM_PATCH_AT_START=1 -e HYBRID_SRC=/repo/tools/dsv41_5090 -e 'VLLM_MOET_SKIP_WEIGHT_RE=(^|\.)engram\.embed\.(weight|scale)$'
             -e VLLM_MOET_ENGRAM_MODE=nvme -e VLLM_MOET_ENGRAM_CACHE_GIB="$ENGRAM_CACHE_GIB" -e VLLM_MOET_ENGRAM_QD="$ENGRAM_QD"
             -e VLLM_MOET_ENGRAM_TIMEOUT_MS="$ENGRAM_TIMEOUT_MS" -e VLLM_MOET_ENGRAM_MODEL_DIR=/model -e VLLM_MOET_ENGRAM_LOG=1
             -e VLLM_MOET_ENGRAM_LOG_EVERY="$LOG_EVERY" -e VLLM_MOET_ENGRAM_LOOKAHEAD="$ENGRAM_LOOKAHEAD"
             -e VLLM_MOET_ENGRAM_LOOKAHEAD_CHUNKS="$ENGRAM_LOOKAHEAD_CHUNKS")
  [[ -n "$ENGRAM_FETCH_CPU" ]] && TIER_ENV+=(-e VLLM_MOET_ENGRAM_FETCH_CPU="$ENGRAM_FETCH_CPU")
fi
if [[ "$STORE_NVME" == "1" ]]; then
  TIER_ENV+=(-e VLLM_MOET_CPU_TIER_STORE_NVME=1 -e VLLM_MOET_CPU_TIER_STORE_DRAM_RECORDS="$STORE_DRAM_RECORDS"
             -e VLLM_MOET_CPU_TIER_STORE_NVME_QD="$STORE_NVME_QD")
fi
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
  ${SECCOMP_ARGS[@]+"${SECCOMP_ARGS[@]}"} \
  "${MOUNTS[@]}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}" -e NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
  ${NCCL_CHANNELS:+-e NCCL_MIN_NCHANNELS=$NCCL_CHANNELS -e NCCL_MAX_NCHANNELS=$NCCL_CHANNELS} \
  ${POOL_SPIN_US:+-e MOE_B12X_POOL_SPIN_US=$POOL_SPIN_US} \
  ${PREFETCH:+-e MOE_B12X_PREFETCH=$PREFETCH -e MOE_B12X_PREFETCH_HINT=$PREFETCH_HINT} \
  ${SPIN_TOUCH_STRIDE:+-e MOE_B12X_POOL_SPIN_TOUCH_STRIDE=$SPIN_TOUCH_STRIDE} \
  -e VLLM_MOET_CPU_TIER_PREFILL_OVERLAP="$PREFILL_OVERLAP" -e VLLM_MOET_CPU_TIER_DEBUG_OVERLAP="$DEBUG_OVERLAP" \
  -e TP="$TP" -e SPEC_TOKENS="${SPEC_TOKENS:-0}" -e GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}" -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}" \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${INDEXER_LOGITS_MB:-256}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}" -e CAPTURE_SIZE="${CAPTURE_SIZE:-8}" \
  "${TIER_ENV[@]}" \
  ${EXTRA_DOCKER_ARGS} \
  --entrypoint bash "$IMAGE" \
  "$ENTRY" /model "$PORT" ${EXTRA_ARGS}

echo "container $NAME started; logs: docker logs -f $NAME ; API: http://${BIND:-0.0.0.0:}${PORT}/v1"
echo "expect in the log: '[moet-cpu-tier] rank N: store /store (...) records = ... GB pinned', 'tier started: 40 layers x 384 experts',"
echo "  then 'compact path with N tokens' at the first decode and 'step 100: ... pair hit rate ...' KPI lines"
exit 0
