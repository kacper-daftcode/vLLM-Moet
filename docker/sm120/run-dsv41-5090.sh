#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 4x RTX 5090 (sm_120), TP4, as the MoET hybrid: the resident experts (a static placement from
# routing frequencies plus promote-on-miss into spare slots) in VRAM through b12x's compact W4A8 path, every other
# routed expert of a decode step computed on the CPU from the pinned expert store, bit-identical to the all-resident
# b12x path (tools/dsv41_5090/, docs/moet-hybrid.md). Defaults = the serving configuration.
#
# TP=2 / TP=1 (2x / 1x GPU; docs/moet-hybrid.md "One or two GPUs"): the same hybrid through VIRTUAL RANKS - on one GPU every
# expert stays its 4 / TP slices of 576 rows (the TP4 rank slices the store holds), b12x's compact W4A8 path runs on the
# slices at N = 576 as at TP4, one CPU pool (CCD) + mailbox set per slice computes the misses with kernel v2 unchanged,
# the router's top-6 becomes 6 x (4 / TP) (token, slice) pairs and the compact TopKSum adds a token's rows in route order
# (fp32, one bf16 rounding - the fixed order that replaces TP4's NCCL ring). The store files are the same (rank r's
# process pins store ranks r*V .. r*V+V-1), the host RAM is the same ~450 GB, the expert slots per layer follow the
# VRAM (one slot = 4 / TP slices): 46 on 2x 32 GB, 16 on 1x 32 GB; ~210-235 / ~100 on 2x / 1x 96 GB. The TP-dependent
# defaults below (GPUS, SPARE, PLACEMENT, MAX_MODEL_LEN, PROMOTIONS) are the 32 GB-card settings; every one is a knob.
# The runtime derives the slices from the layer geometry (VLLM_MOET_CPU_TIER_VRANKS=auto; VRANKS=1 forces plain ranks);
# the CPU layout of a process with V pools is automatic (process r: main thread on core 8rV, per CCD g = rV+v: dispatcher
# 8g+1, pool 8g+2..8g+7; VLLM_MOET_CPU_TIER_VRANK_CPUS / _VRANK_DISPATCHERS override it for another CCD geometry).
#
# Required:  MODEL_DIR   the official DeepSeek-V4.1-Flash checkpoint
#            STORE       cpu_moe/build_store.py's expert store (269 GB on NVMe; manifest.json inside)
#
# Container: IMAGE (vllm-moet-sm120:dsv41-nightly-df3bd094-r23-hyb: built with --build-arg VLLM_MOET_CPU_TIER_PATCH=1)
#            NAME (ds41-hyb)  TP (4: 1 | 2 | 4 GPUs)  GPUS (0,1,2,3 / 0,1 / 0 by TP)  PORT (8041)  BIND (127.0.0.1:)
#            VRANKS ("" = VLLM_MOET_CPU_TIER_VRANKS auto: 4 / TP slices per expert; 1 = plain ranks, the TP4 path)
#            VRANK_CPUS / VRANK_DISPATCHERS ("" = the CCD layout above)  the pools' cores per virtual rank (";"-separated
#                        ranges, e.g. "2-7;10-15;18-23;26-31") and their dispatcher cores ("1,9,17,25") for another CPU geometry:
#                        TP x (4 / TP) entries = every rank's (rank r takes entries r*V..), or 4 / TP entries = each process's own
#            CACHE_DIR ($PWD/cache-$NAME)  FlashInfer autotune results, the kernel tables (below), profiles
#            REPO (this checkout)    mounted at /repo with BAKED=0
#            BAKED (1)               1 = the image's own copy of the hybrid (Dockerfile step 43: patched packages, runtime,
#                        libcpu_tier.so). 0 = the dev flow: the checkout mounted at /repo, the patcher re-run at start (~60 s)
#            STORE_ROOT ("")         a second read-only mount for a store of symlinks into another store (the mini-real tests)
#            EXTRA_ARGS / EXTRA_DOCKER_ARGS   appended to `vllm serve` / `docker run`
#
# Experts:   PLACEMENT ($REPO/tools/dsv41_5090/routing/placements/placement-0.1042-uniform.json)  routing/placement.py's file:
#                        the static experts per layer (40 at 0.1042); RESIDENT=<int> instead = the first N experts of every
#                        layer (a test placement). TP1: tp12-placement-0.026-uniform.json (9 static per layer) from the same
#                        directory
#            SPARE (70)              LRU slots per layer for promote-on-miss (40 + 70 = 110 slots at GPU_MEM_UTIL 0.90, 512K)
#                        TP2 (2x 32 GB): 6 = 46 slots with the 40-static placement (an expert costs 9.4 MB per card there:
#                        measured 11-04, 50 slots left no KV at util 0.90 / 64K and the boot refused; traces: 40+10 -> hit
#                        0.565, 20+30 -> 0.658, 0+50 -> 0.708 - the all-LRU split is better, but every static expert made
#                        dynamic adds its record to the pinned store: +14 GiB for 20 static, +28 for 0 - a 512 GB+ box)
#                        TP1 (1x 32 GB): 7 = 16 slots with the 9-static placement (18.8 MB per expert; KV ~2 GiB at util 0.90 /
#                        32K; traces: hit 0.42, ~145 CPU experts per step; the host RAM is +23 GB over the 40-static store)
#            PROMOTIONS (4)          promotions per layer per step;  PROMOTE_WINDOW (12)  admission: a missed expert is
#                        promoted only if its previous miss was <= W steps ago (0 = every miss). TP1: 1 - a promotion moves
#                        4 / TP records (18.8 MB at TP1) and the traces give the same hit rate at cap 1 as at cap 4 (0.420 vs
#                        0.426) for 31 instead of 53 promotions per step
#            STORE_MODE (nonresident) which records the tier pins: nonresident (65 GB per rank with 40 static) or all (72 GB)
#            STORE_EARLY (0)         1 = pin the records at the last MoE layer's create_weights, before the weights are loaded
#                        (measured slower: the loader then streams the shards through the page cache the pinned memory leaves)
#            CPU_TIER_TABLES ($CACHE_DIR/cpu-tier-tables)  the kernel tables: cpu_moe/qmma_probe --tables DIR and
#                        cpu_tier/b12x_layout_probe.py --out DIR, once per GPU model (docs/moet-hybrid.md)
#            TIMEOUT_MS (500)        the in-graph wait's bound (fail-closed)
#            SCRATCH (64)            slots of the prefill scratch pool (64 = 0.6 GB per rank; 0 = the missing experts of
#                        > 8-token calls are not computed: decode-only diagnostics). With virtual ranks a slot holds one
#                        SLICE (an expert takes 4 / TP slots): 64 = 32 experts per round at TP2, 16 at TP1; DEBUG_COMPACT
#                        needs SCRATCH >= the routed experts of a compact call x slices (8 x 6 = 48 -> 96 at TP2, 192 at TP1)
#            PREFILL_OVERLAP (1)     a second scratch set + a side stream: the next round's records staged during the current
#                        run, bit-identical to the serial path (TTFT -15 %); costs SCRATCH x 4.7 MB x 2 of VRAM. 0 = off;
#                        DEBUG_OVERLAP=1 re-runs every call serially and compares bit for bit
#
# CPU pool:  POOL_THREADS (6)        kernel pool threads per rank = cores of its CCD (6, 7 or 8)
#            POOL_SPIN_US (5000)     the pool's spin before it sleeps on the futex (-1 = spin forever); 5000 spins through a
#                        decode step's gaps and sleeps when the API is idle
#            POOL_SPIN_HOT (1)       the idle spin: 1 = ALU work instead of `pause` (keeps the cores' clock up), 2 = also a
#                        private 32 MB DRAM stream per worker (faster per expert, but heats the memory side until it
#                        throttles after ~18 min of load), 0 = pause;  SPIN_TOUCH_STRIDE ("") its stride for mode 2
#            PREFETCH (1)            the kernel's software prefetch of the next rows;  PREFETCH_HINT (1 = L2)
#            HOST_EMPTY_CACHE (0)    1 = torch._C._host_emptyCache() after the weights are loaded
#
# Engine:    MAX_MODEL_LEN (524288; TP2 65536, TP1 32768: the KV and the indexer's buffers compete with the expert slots on
#                        32 GB cards)  MAX_NUM_SEQS (1: several streams are correct but slow - every stream brings its own
#                        misses)  MAX_NUM_BATCHED_TOKENS (4096: the prefill chunk; at TP1 on 32 GB its workspaces cost ~0.6 GB
#                        of KV against 2048, but a 14K prompt prefills at 944 against 576 tok/s)  CAPTURE_SIZE (8)
#            GPU_MEM_UTIL (0.90)     INDEXER_LOGITS_MB (256: VLLM_SPARSE_INDEXER_MAX_LOGITS_MB; the indexer OOMed at 0.94 on
#                        the 4th chunk of a 128K prompt)
#            MOE_DETERMINISTIC (0)   B12X_DYNAMIC_DETERMINISTIC_OUTPUT for the > 8-token (prefill) MoE calls. The image bakes 1
#                        (step 27a, the 4x PRO 6000 line's choice: identical logits at any batch size); the hybrid's measured
#                        serving path is 0: on the -r23-hyb image the 30K-token prefill takes 6.5 s with 0 and 6.8 s with 1
#                        (+4 %), 14K / 2K unchanged, decode unchanged (k0-merge vs k0-merge-det0, 2026-11-04). The debug checks
#                        of the streamed prefill (DEBUG_OVERLAP, DEBUG_PREFILL) need 1
#            SPEC_TOKENS (0)         DSpark draft tokens (0 = no drafter: k = 5 is CPU-bound with the m > 1 kernel)
#            ENGRAM_THP (1)          --engram-config {"use_thp":true}: exact-size mmap for the Engram tables instead of torch's
#                        pinned allocator, which rounds each 24.6 GB table to 32 GiB; needs free, unfragmented memory at boot
#            PREFIX_CACHE (1)        0 = --no-enable-prefix-caching (for measurements)
#            NCCL_P2P_LEVEL (SYS)  NCCL_P2P_DISABLE (0)  NCCL_CHANNELS ("" = NCCL's choice; a pin gives bit-identity across
#                        boots - the all-reduce decomposition changes the outputs by ulps)
#
# Low RAM:   (384 / 256 / 128 GB hosts; 2026-11-04, docs/moet-hybrid.md "Hosts with less RAM"; all inert at the defaults = the 512 GB
#            configuration. An NVMe mode needs an image with Dockerfile step 44 (--build-arg VLLM_MOET_ENGRAM_V2_PATCH=1, the
#            -hyb-engram tags) with BAKED=1, or BAKED=0 with the checkout mounted; io_uring needs docker/sm120/seccomp-io_uring.json,
#            which the launcher passes from its own checkout - docker's default seccomp profile denies io_uring_setup)
#            ENGRAM_MODE (ram)       ram = vLLM's pinned Engram tables (2 x 23.6 GiB per rank); nvme = Engram v2: no pinned tables, the
#                        rows come from the checkpoint's safetensors shards on NVMe through a DRAM row cache + an io_uring fetcher
#                        per rank (tools/dsv41_5090/engram_v2/), vLLM's own lookup kernel dequantizes them from a device landing
#                        buffer (bit-identical rows). Measured 11-04: 457 -> 271 GB used, decode unchanged, TTFT +0.1-0.5 s
#            ENGRAM_CACHE_GIB (2)    the DRAM row cache per rank (plain memory, split over the two Engram layers)
#            ENGRAM_QD (64)          io_uring depth per rank (128 equal, 256 slower in this implementation);  ENGRAM_TIMEOUT_MS (2000)
#                        the in-stream wait's bound (fail-closed)
#            ENGRAM_RESIDENT_SCALES (1)  this rank's ue8m0 scales stay in RAM (8 B per row: 1.5 GB per rank for the two tables), so a
#                        row costs one NVMe read instead of two - Fraser Price's BRRR recipe does the same (bench: a decode step's
#                        rows 133 -> 92 us median, a cold 4096-token chunk 101 -> 46 ms); 0 on the tightest hosts
#            ENGRAM_POLLED (1)       the fetcher runs on the CPU tier's dispatcher thread (its dedicated core); 0 = a thread of its own
#                        on ENGRAM_FETCH_CPU ("" = 8*rank+33)
#            ENGRAM_LOOKAHEAD (1)    the prompt's rows hashed on the host and the next ENGRAM_LOOKAHEAD_CHUNKS (4) chunks' rows prefetched
#                        into the row cache during a prefill (only a prompt's first chunk pays the NVMe latency)
#            STORE_NVME (0)          1 = the expert store's cold tail on NVMe: only STORE_DRAM_RECORDS records per layer per rank stay
#                        pinned (an LRU), the rest are read from the store files on a miss (io_uring, 4.7 MB reads into a spare
#                        slot; kernel v2 computes from it - bit-identical rows); the prefill streaming and the promotions read their
#                        cold records the same way. Measured 11-04: N = 230 -> 194 GB used, decode -10 / -14 %, TTFT 2.1-2.2x;
#                        N = 110 -> 106 GB, -24 / -35 %, TTFT 3.3-3.6x (identity PASS at N = 110)
#            STORE_DRAM_RECORDS (0)  records per layer per rank kept in DRAM when STORE_NVME=1 (0 = all 344 = today's store); **auto** = fill
#                        the host: N from the footprint model below with the budget MemTotal - HEADROOM_GIB (20); N >= the store's records
#                        per layer (344 with the 40-static placement, 384 with STORE_MODE=all) turns the tail off (everything in DRAM)
#            STORE_DRAM_GIB ("")     the budget in GiB instead of the host's memory (implies auto): N = floor((budget - base) / 0.72)
#            HEADROOM_GIB (20)       what auto leaves free of MemTotal (the loader's page cache, the API server; the 512 GiB box runs with 41)
#                        The footprint model (measured 11-04 on 4x 5090, `free` used in GiB, the same at TP2 / TP1 - the 4 slices of an
#                        expert are in DRAM whatever the GPU count): footprint = base + 0.72 GiB x N, one record per layer = 4 slices x 40
#                        layers x 4.7 MB; base = 35 GiB with Engram v2 (workers / engine / kernel, the resident scales, the prefill arenas,
#                        the 2 GiB row cache per store rank; +4 x (ENGRAM_CACHE_GIB - 2) for another cache size) or 216 GiB with the
#                        pinned Engram tables (ENGRAM_MODE=ram: +181). Measured points: N = 344 + the full LRU 307 GiB, 344 -> 278,
#                        280 -> 236, 230 -> 200, 120 (cache 1) -> 120, 110 (cache 1) -> 112. Per host: 384 GiB -> everything in DRAM + the
#                        full LRU (ENGRAM_MODE=nvme STORE_MODE=all RESIDENT=0 SPARE=110: 307 GiB, 147.4 / 125.1 tok/s, TTFT unchanged);
#                        256 GiB -> N ~ 270 (~229 GiB, ~20 free; auto gives 279 at HEADROOM 20: 236 GiB, 137.6 / 119.6, TTFT 1.4-1.5x);
#                        128 GiB -> N ~ 105 with ENGRAM_CACHE_GIB=1 (~111 GiB; auto gives 106: 114 / 81 tok/s, TTFT 3.0-3.1x)
#            STORE_NVME_QD (8)       record reads in flight per rank (also the prefill's reader threads)
#            With TP=2 / TP=1 both compose with the virtual ranks: the store tail is PER SLICE (STORE_DRAM_RECORDS counts per layer
#                        per store rank, so a process pins N x 4.7 MB x 40 x (4 / TP) = the same host RAM per GPU card as at TP4; a
#                        cold expert is 4 / TP concurrent record reads), Engram v2 is per TP-rank process (vLLM's own head sharding:
#                        12 heads at TP2, 24 at TP1; one fetcher, polled by virtual rank 0's dispatcher). The image needs steps 43
#                        AND 44 from a checkout with the virtual ranks (the -hyb-next tag); serve_hybrid.sh refuses an older one
#
# Diagnostics: LOG_EVERY (100)  KPI line period (steps);  TIER_DEBUG (0)
#            DEBUG_COMPACT (0)       1 + EXTRA_ARGS="--enforce-eager" + SCRATCH >= 64: every compact call recomputed with all
#                        its routed experts in the scratch pool and compared bit for bit (the within-boot identity check)
#            DEBUG_PREFILL (0)       1: the same for the > 8-token calls (SCRATCH >= the call's distinct experts)
#            POWER_CAP_W (CAP_W of /etc/default/gpu-power-cap, else 400)  the GPUs' power limit re-applied before the boot
#                        when a card sits above it (docker/sm120/host/); 0 skips the check
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR (DeepSeek-V4.1-Flash checkpoint directory) is required}"
: "${STORE:?STORE (the expert store written by tools/dsv41_5090/cpu_moe/build_store.py) is required}"
BAKED="${BAKED:-1}"
ENGRAM_MODE="${ENGRAM_MODE:-ram}"
ENGRAM_CACHE_GIB="${ENGRAM_CACHE_GIB:-2}"
ENGRAM_FETCH_CPU="${ENGRAM_FETCH_CPU:-}"
ENGRAM_QD="${ENGRAM_QD:-64}"
ENGRAM_TIMEOUT_MS="${ENGRAM_TIMEOUT_MS:-2000}"
ENGRAM_POLLED="${ENGRAM_POLLED:-1}"
ENGRAM_RESIDENT_SCALES="${ENGRAM_RESIDENT_SCALES:-1}"
ENGRAM_LOOKAHEAD="${ENGRAM_LOOKAHEAD:-1}"
ENGRAM_LOOKAHEAD_CHUNKS="${ENGRAM_LOOKAHEAD_CHUNKS:-4}"
STORE_NVME="${STORE_NVME:-0}"
STORE_DRAM_RECORDS="${STORE_DRAM_RECORDS:-0}"
STORE_DRAM_GIB="${STORE_DRAM_GIB:-}"
HEADROOM_GIB="${HEADROOM_GIB:-20}"
STORE_NVME_QD="${STORE_NVME_QD:-8}"
NAME="${NAME:-ds41-hyb}"
TP="${TP:-4}"
# the TP-dependent defaults (virtual ranks on 32 GB cards; see the header): GPUs, slots, placement, context, promotions
case "$TP" in
  4) D_GPUS=0,1,2,3; D_SPARE=70; D_PLACEMENT=placement-0.1042-uniform.json;      D_MAX_MODEL_LEN=524288; D_PROMOTIONS=4 ;;
  2) D_GPUS=0,1;     D_SPARE=6;  D_PLACEMENT=placement-0.1042-uniform.json;      D_MAX_MODEL_LEN=65536;  D_PROMOTIONS=4 ;;
  1) D_GPUS=0;       D_SPARE=7;  D_PLACEMENT=tp12-placement-0.026-uniform.json; D_MAX_MODEL_LEN=32768;  D_PROMOTIONS=1 ;;
  *) echo "TP=$TP is not 1, 2 or 4" >&2; exit 1 ;;
esac
GPUS="${GPUS:-$D_GPUS}"
if [[ $(echo "$GPUS" | tr ',' '\n' | wc -l) != "$TP" ]]; then
  echo "GPUS=$GPUS does not list $TP GPUs for TP=$TP" >&2; exit 1
fi
PORT="${PORT:-8041}"
BIND="${BIND-127.0.0.1:}"
CACHE_DIR="${CACHE_DIR:-$PWD/cache-$NAME}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STORE_ROOT="${STORE_ROOT:-}"
STORE_MODE="${STORE_MODE:-nonresident}"
STORE_EARLY="${STORE_EARLY:-0}"
PLACEMENT="${PLACEMENT:-$REPO/tools/dsv41_5090/routing/placements/$D_PLACEMENT}"
RESIDENT="${RESIDENT:-}"
# STORE_DRAM_RECORDS=auto / STORE_DRAM_GIB=<budget>: N from the footprint model of the header (base + 0.72 GiB x N); the store's
# records per layer (the cap: everything in DRAM at or above it) from the placement file / RESIDENT / STORE_MODE
if [[ "$STORE_DRAM_RECORDS" == "auto" || -n "$STORE_DRAM_GIB" ]]; then
  if [[ -n "$STORE_DRAM_GIB" ]]; then
    BUDGET_GIB="$STORE_DRAM_GIB"; BUDGET_SRC="STORE_DRAM_GIB=$STORE_DRAM_GIB"
  else
    MEM_GIB=$(awk '/^MemTotal:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
    BUDGET_GIB=$((MEM_GIB - HEADROOM_GIB)); BUDGET_SRC="MemTotal $MEM_GIB GiB - HEADROOM_GIB $HEADROOM_GIB"
  fi
  if [[ "$STORE_MODE" == "all" ]]; then STORE_CAP=384
  elif [[ -n "$RESIDENT" ]]; then STORE_CAP=$((384 - RESIDENT))
  else STORE_CAP=$(python3 -c 'import json, sys; p = json.load(open(sys.argv[1])); print(384 - min(l["count"] for l in p["layers"]))' "$PLACEMENT" 2>/dev/null || echo 344); fi
  if [[ "$ENGRAM_MODE" == "nvme" ]]; then BASE_GIB=$(awk -v c="$ENGRAM_CACHE_GIB" 'BEGIN {printf "%.1f", 35 + 4 * (c - 2)}'); BASE_SRC="$BASE_GIB GiB with Engram v2 (35 at its default 2 GiB row cache per rank; ENGRAM_CACHE_GIB=$ENGRAM_CACHE_GIB)"
  else BASE_GIB=216; BASE_SRC="216 GiB with the pinned Engram tables (ENGRAM_MODE=ram)"; fi
  N_AUTO=$(awk -v b="$BUDGET_GIB" -v base="$BASE_GIB" 'BEGIN {n = int((b - base) / 0.72); if (n < 0) n = 0; print n}')
  if (( N_AUTO >= STORE_CAP )); then
    echo "STORE_DRAM_RECORDS=auto: $BUDGET_SRC -> budget $BUDGET_GIB GiB; base $BASE_SRC; N = floor(($BUDGET_GIB - $BASE_GIB) / 0.72) = $N_AUTO >= the store's $STORE_CAP records per layer: the tail is OFF, every record in DRAM (expected footprint $(awk -v base="$BASE_GIB" -v n="$STORE_CAP" 'BEGIN {printf "%.0f", base + 0.72 * n}') GiB)"
    STORE_NVME=0; STORE_DRAM_RECORDS=0
  else
    echo "STORE_DRAM_RECORDS=auto: $BUDGET_SRC -> budget $BUDGET_GIB GiB; base $BASE_SRC; N = floor(($BUDGET_GIB - $BASE_GIB) / 0.72) = $N_AUTO of the store's $STORE_CAP records per layer in DRAM (expected footprint $(awk -v base="$BASE_GIB" -v n="$N_AUTO" 'BEGIN {printf "%.0f", base + 0.72 * n}') GiB); the tail is on"
    STORE_NVME=1; STORE_DRAM_RECORDS=$N_AUTO
  fi
fi
NVME_TIER=0; [[ "$ENGRAM_MODE" == "nvme" || "$STORE_NVME" == "1" ]] && NVME_TIER=1
# the baked image by default (Dockerfile step 43 built in; the NVMe modes need step 44 too: the -hyb-engram tag); the dev flow
# (BAKED=0) mounts the checkout and runs the patcher(s) at start
if [[ "$BAKED" == "1" ]]; then
  if [[ "$NVME_TIER" == "1" ]]; then
    IMAGE="${IMAGE:-vllm-moet-sm120:dsv41-nightly-df3bd094-r23-hyb-engram}"
  else
    IMAGE="${IMAGE:-vllm-moet-sm120:dsv41-nightly-df3bd094-r23-hyb}"
  fi
else
  IMAGE="${IMAGE:-vllm-moet-sm120:dsv41-nightly-df3bd094-r23}"
fi
SPARE="${SPARE:-$D_SPARE}"
PROMOTIONS="${PROMOTIONS:-$D_PROMOTIONS}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$D_MAX_MODEL_LEN}"
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
TABLES="${CPU_TIER_TABLES:-$CACHE_DIR/cpu-tier-tables}"

[[ -f "$MODEL_DIR/config.json" ]] || { echo "no config.json in MODEL_DIR=$MODEL_DIR" >&2; exit 1; }
[[ -f "$STORE/manifest.json" ]] || { echo "no manifest.json in STORE=$STORE (cpu_moe/build_store.py)" >&2; exit 1; }
[[ -f "$TABLES/b12x_act_m_limit10.f32" && -f "$TABLES/b12x_layout_w13.npy" ]] || { echo "kernel tables missing in $TABLES (CPU_TIER_TABLES; cpu_moe/qmma_probe --tables + cpu_tier/b12x_layout_probe.py --out, docs/moet-hybrid.md)" >&2; exit 1; }
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
# the store may be a directory of symlinks into another store (the mini-real test): STORE_ROOT mounts that root too
[[ -n "$STORE_ROOT" ]] && MOUNTS+=(-v "$STORE_ROOT:$STORE_ROOT:ro")
# the low-RAM tiers (ENGRAM_MODE=nvme / STORE_NVME=1): io_uring needs a seccomp profile that allows it (the checkout's);
# BAKED=0 runs the engram_v2 patcher at start from the mounted checkout, BAKED=1 expects the image's step 44
SECCOMP_ARGS=()
if [[ "$NVME_TIER" == "1" ]]; then
  [[ -f "$REPO/docker/sm120/seccomp-io_uring.json" ]] || { echo "missing $REPO/docker/sm120/seccomp-io_uring.json (io_uring for the NVMe tiers)" >&2; exit 1; }
  SECCOMP_ARGS=(--security-opt "seccomp=$REPO/docker/sm120/seccomp-io_uring.json")
fi
if [[ "$ENGRAM_MODE" == "nvme" ]]; then
  # the weight iterator's skip of the 2 x 98 GB engram.embed tensors must be in the workers' environment before vLLM's loader
  # modules import (the runtime sets it too, but engram.py may import after ep_weight_filter.py)
  TIER_ENV+=(-e ENGRAM_PATCH_AT_START="$([[ "$BAKED" == "1" ]] && echo 0 || echo 1)" -e 'VLLM_MOET_SKIP_WEIGHT_RE=(^|\.)engram\.embed\.(weight|scale)$'
             -e VLLM_MOET_ENGRAM_MODE=nvme -e VLLM_MOET_ENGRAM_CACHE_GIB="$ENGRAM_CACHE_GIB" -e VLLM_MOET_ENGRAM_QD="$ENGRAM_QD"
             -e VLLM_MOET_ENGRAM_TIMEOUT_MS="$ENGRAM_TIMEOUT_MS" -e VLLM_MOET_ENGRAM_MODEL_DIR=/model -e VLLM_MOET_ENGRAM_LOG=1
             -e VLLM_MOET_ENGRAM_LOG_EVERY="$LOG_EVERY" -e VLLM_MOET_ENGRAM_POLLED="$ENGRAM_POLLED" -e VLLM_MOET_ENGRAM_LOOKAHEAD="$ENGRAM_LOOKAHEAD"
             -e VLLM_MOET_ENGRAM_LOOKAHEAD_CHUNKS="$ENGRAM_LOOKAHEAD_CHUNKS" -e VLLM_MOET_ENGRAM_RESIDENT_SCALES="$ENGRAM_RESIDENT_SCALES")
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
  -e TP="$TP" -e SPEC_TOKENS="${SPEC_TOKENS:-0}" -e GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}" -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${INDEXER_LOGITS_MB:-256}" -e B12X_DYNAMIC_DETERMINISTIC_OUTPUT="${MOE_DETERMINISTIC:-0}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}" -e CAPTURE_SIZE="${CAPTURE_SIZE:-8}" \
  ${VRANKS:+-e VLLM_MOET_CPU_TIER_VRANKS=$VRANKS} \
  ${VRANK_CPUS:+-e VLLM_MOET_CPU_TIER_VRANK_CPUS=$VRANK_CPUS} ${VRANK_DISPATCHERS:+-e VLLM_MOET_CPU_TIER_VRANK_DISPATCHERS=$VRANK_DISPATCHERS} \
  "${TIER_ENV[@]}" \
  ${EXTRA_DOCKER_ARGS} \
  --entrypoint bash "$IMAGE" \
  "$ENTRY" /model "$PORT" ${EXTRA_ARGS}

echo "container $NAME started; logs: docker logs -f $NAME ; API: http://${BIND:-0.0.0.0:}${PORT}/v1"
echo "expect in the log: '[moet-cpu-tier] rank N: store /store (...) records = ... GB pinned', 'tier started: 40 layers x 384 experts',"
echo "  then 'compact path with N tokens' at the first decode and 'step 100: ... pair hit rate ...' KPI lines"
exit 0
