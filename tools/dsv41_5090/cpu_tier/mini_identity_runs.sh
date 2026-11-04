#!/bin/bash
# The mini-model identity test of phase 2.3 on one GPU: serve the TP1/576 mini model with the CPU tier at several
# resident counts, query the same prompts twice per server, compare everything against the all-resident run.
# Deterministic settings: VLLM_MOET_DET_TOPK=1 (the sm120 line's exact indexer top-k), no prefix caching, greedy, seed 0.
#   usage: mini_identity_runs.sh GPU "16 8 1" [PORT] [extra vllm args, e.g. --speculative-config '{}' to disable DSpark]
#   env:   MODEL_DIR (/mnt/nvme/models/mini-hyb576: dummy weights; LOAD_FORMAT=dummy)
#          STORE=/mnt/nvme/packs/dsv41-hybrid-mini576 + LOAD_FORMAT=auto + MODEL_DIR=/mnt/nvme/models/mini-real576:
#                 the real-checkpoint path (store mode: R-row parameters, loader skip, records from the store files)
#          SPARE (0) spare slots per layer; PROMOTIONS (4) per layer per step; DSPARK (1); DEBUG (0); CPUS (unset = the
#          runtime's per-rank layout; 9-15 pins one pool); GPU=1,2 serves TP2 (TP from the count; both locks taken)
#          GPU_MEM_UTIL (0.3; the 384-expert mini-real needs ~0.85); R=placement uses the model dir's placement.json
#          EXTRA_DOCKER_ENV ("-e K=V ...") extra container environment, e.g. -e VLLM_MOET_CPU_TIER_STORE_EARLY=1
#          NO_LOCK (0) 1 = do not take the GPU's flock (the caller holds it)
#          STORE_ROOT (unset = STORE) a second mount for a store of symlinks into another store (the mini-real stores point
#          into the full store: STORE_ROOT=<packs root>; without it the records fail to load, rc=-2)
#          REPO (this checkout) the worktree mounted at /repo (patcher + runtime); NAME (moet-mini-hyb) the
#          container; CACHE_DIR ($PWD/cache-ds41) the JIT caches; IMAGE (…-r23); PY (python3); CPU_TIER_TABLES (required:
#          the kernel tables); TP (the GPU count) - e.g. TP=1 on the full-expert
#          mini-real (mini-real384-tp4 / mini-real2304-e16) = 4 virtual ranks, GPU=1,2 TP=2 = 2 per GPU (11-04)
#   (holds the GPU's flock for each serve; logs /tmp/moet-mini-r<R>.log)
set -u
GPU="${1:?gpu index (or a list 1,2 for TP2)}"
# The GPU locks are taken by THIS script (fd-based flocks, held across each serve) before the container and the client
# start: with the lock taken inside the backgrounded docker-run subshell (as until 11-04) a contended lock let the client
# run against a container that did not exist yet, and the serve it finally started was never stopped (11-04 00:47).
# NO_LOCK=1: the caller already holds the GPU's lock (e.g. a TP4 chain that runs this test under its four locks before
# the full-model boot) - a second flock on the same file would block against it
LOCK_FDS=()
take_locks() {
  [ "${NO_LOCK:-0}" = "1" ] && return 0
  local fd=9
  for g in ${GPU//,/ }; do
    eval "exec $fd>/tmp/exl3diff-gpu$g.lock"
    flock -w 3600 $fd || { echo "could not take /tmp/exl3diff-gpu$g.lock within 3600 s"; exit 1; }
    LOCK_FDS+=($fd); fd=$((fd + 1))
  done
}
release_locks() {
  for fd in "${LOCK_FDS[@]}"; do flock -u $fd; eval "exec $fd>&-"; done
  LOCK_FDS=()
}
TP="${TP:-$(echo $GPU | tr "," "\n" | wc -l)}"
RS="${2:-16 8 1}"
PORT="${3:-8031}"
shift 3 2>/dev/null || shift $#
EXTRA="$*"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
NAME="${NAME:-moet-mini-hyb}"
IMG="${IMAGE:-vllm-moet-sm120:dsv41-nightly-df3bd094-r23}"
PY="${PY:-python3}"
CACHE_DIR="${CACHE_DIR:-$PWD/cache-ds41}"
TABLES="${CPU_TIER_TABLES:?CPU_TIER_TABLES (the kernel tables directory) is required}"
CLIENT=$REPO/tools/dsv41_5090/cpu_tier/compare_mini_outputs.py
[ "${LONG:-0}" = "1" ] && CLIENT="$CLIENT --long"
: "${MODEL_DIR:?MODEL_DIR (make_mini_config.py's mini model, or make_mini_real.py's) is required}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
STORE="${STORE:-}"
TAG="${TAG:-}"
STORE_ARGS=()
DUMMY_W=1
if [ -n "$STORE" ]; then
  STORE_ARGS=(-v "${STORE_ROOT:-$STORE}:${STORE_ROOT:-$STORE}:ro" -e VLLM_MOET_CPU_TIER_STORE=$STORE -e VLLM_MOET_CPU_TIER_STORE_MODE=${STORE_MODE:-nonresident})
  DUMMY_W=0
fi
for R in $RS; do
  echo "== R=$R  $(date -u +%T)  model $MODEL_DIR load $LOAD_FORMAT store '${STORE}' spare ${SPARE:-0}; waiting for the GPU lock(s) $GPU"
  take_locks
  docker ps --format '{{.Names}}' | grep -q "^$NAME$" && { echo "container still up"; exit 1; }
  # R=placement: the model dir's placement.json (make_mini_real.py) instead of the first R experts
  RES_ARGS=(-e VLLM_MOET_CPU_TIER_RESIDENT=$R)
  [ "$R" = "placement" ] && RES_ARGS=(-e VLLM_MOET_CPU_TIER_PLACEMENT=/model/placement.json)
  ( docker run --rm --gpus "\"device=$GPU\"" --name $NAME --ulimit memlock=-1:-1 --ipc host --shm-size 16g -e TP=$TP ${EXTRA_DOCKER_ENV:-} \
      -e VLLM_MOET_MOE_CPU_TIER=1 "${RES_ARGS[@]}" -e VLLM_MOET_CPU_TIER_DUMMY_WEIGHTS=$DUMMY_W \
      -e VLLM_MOET_CPU_TIER_TABLES=/parity ${CPUS:+-e VLLM_MOET_CPU_TIER_CPUS=$CPUS} -e VLLM_MOET_CPU_TIER_LOG=1 -e VLLM_MOET_CPU_TIER_LOG_EVERY=${LOG_EVERY:-50} \
      -e VLLM_MOET_DET_TOPK=1 -e DSPARK=${DSPARK:-1} -e LOAD_FORMAT=$LOAD_FORMAT -e GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.3} -e MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-512} -e VLLM_MOET_CPU_TIER_SPARE=${SPARE:-0} \
      -e VLLM_MOET_CPU_TIER_MAX_PROMOTIONS=${PROMOTIONS:-4} -e VLLM_MOET_CPU_TIER_DEBUG=${DEBUG:-0} -e VLLM_MOET_CPU_TIER_SCRATCH=${SCRATCH:-0} -e VLLM_MOET_CPU_TIER_DEBUG_PREFILL=${DEBUG_PREFILL:-0} -e VLLM_MOET_CPU_TIER_DEBUG_COMPACT=${DEBUG_COMPACT:-0} "${STORE_ARGS[@]}" \
      -v $MODEL_DIR:/model -v "$CACHE_DIR/dot-cache:/root/.cache" \
      -v $REPO:/repo:ro -v "$TABLES:/parity:ro" \
      -p 127.0.0.1:$PORT:$PORT --entrypoint bash $IMG \
      /repo/tools/dsv41_5090/cpu_tier/serve_mini_hybrid.sh /model $PORT --no-enable-prefix-caching --max-cudagraph-capture-size 8 $EXTRA \
      > /tmp/moet-mini${TAG}-r$R.log 2>&1 ) &
  BG=$!
  # the second client only after a successful first one (pipefail: a client that gave up must not leave the next one waiting)
  set -o pipefail
  $PY $CLIENT --port $PORT --container $NAME --out /tmp/moet-mini${TAG}-run-r${R}a.json | tail -1 && \
  $PY $CLIENT --port $PORT --container $NAME --out /tmp/moet-mini${TAG}-run-r${R}b.json | tail -1
  set +o pipefail
  grep "moet-cpu-tier\] rank 0: step" /tmp/moet-mini${TAG}-r$R.log | tail -2
  # the serve may still be queued on the flock (the client gave up): kill the waiter, not a running container's flock
  docker ps -q -f name="^$NAME$" | grep -q . || { pkill -P $BG 2>/dev/null; kill $BG 2>/dev/null; }
  docker stop -t 20 $NAME > /dev/null 2>&1
  wait
  release_locks
done
echo "== comparisons"
for R in $RS; do
  echo "-- R=$R: a vs b (determinism)"; $PY $CLIENT --compare /tmp/moet-mini${TAG}-run-r${R}a.json /tmp/moet-mini${TAG}-run-r${R}b.json | tail -1
done
set -- $RS
BASE=$1
for R in $RS; do
  [ "$R" = "$BASE" ] && continue
  echo "-- R=$BASE vs R=$R"; $PY $CLIENT --compare /tmp/moet-mini${TAG}-run-r${BASE}a.json /tmp/moet-mini${TAG}-run-r${R}a.json | tail -9
done
