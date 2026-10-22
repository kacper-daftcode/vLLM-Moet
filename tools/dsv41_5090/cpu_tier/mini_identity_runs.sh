#!/bin/bash
# The mini-model identity test of phase 2.3 on one GPU: serve the TP1/576 mini model (dummy weights, seeded) with the
# CPU tier at several resident counts, query the same prompts twice per server, compare everything against the
# all-resident run. Deterministic settings: VLLM_MOET_DET_TOPK=1 (the sm120 line's exact indexer top-k), no prefix
# caching, greedy, seed 0.
#   usage: mini_identity_runs.sh GPU "16 8 1" [PORT] [extra vllm args, e.g. --speculative-config '{}' to disable DSpark]
#   (holds the GPU's flock for each serve; logs /tmp/moet-mini-r<R>.log)
set -u
GPU="${1:?gpu index}"
RS="${2:-16 8 1}"
PORT="${3:-8031}"
shift 3 2>/dev/null || shift $#
EXTRA="$*"
REPO=/mnt/nvme/vLLM-Moet-cpu
IMG=vllm-moet-sm120:dsv41-nightly-df3bd094-hyb1
PY=/root/work/venv/bin/python
CLIENT=$REPO/tools/dsv41_5090/cpu_tier/compare_mini_outputs.py
for R in $RS; do
  echo "== R=$R  $(date -u +%T)"
  docker ps --format '{{.Names}}' | grep -q '^moet-mini-hyb$' && { echo "container still up"; exit 1; }
  ( flock -w 3600 /tmp/exl3diff-gpu$GPU.lock docker run --rm --gpus "\"device=$GPU\"" --name moet-mini-hyb \
      -e VLLM_MOET_MOE_CPU_TIER=1 -e VLLM_MOET_CPU_TIER_RESIDENT=$R -e VLLM_MOET_CPU_TIER_DUMMY_WEIGHTS=1 \
      -e VLLM_MOET_CPU_TIER_TABLES=/parity -e VLLM_MOET_CPU_TIER_CPUS=8-15 -e VLLM_MOET_CPU_TIER_LOG=1 -e VLLM_MOET_DET_TOPK=1 -e DSPARK=${DSPARK:-1} -e VLLM_MOET_CPU_TIER_MAX_PROMOTIONS=${PROMOTIONS:-4} -e VLLM_MOET_CPU_TIER_DEBUG=${DEBUG:-0} \
      -v /mnt/nvme/models/mini-hyb576:/model -v /mnt/nvme/cache/ds41/dot-cache:/root/.cache \
      -v $REPO:/repo:ro -v $REPO/internal/dsv41-5090-cpu/parity:/parity:ro \
      -p 127.0.0.1:$PORT:$PORT --entrypoint bash $IMG \
      /repo/tools/dsv41_5090/cpu_tier/serve_mini_hybrid.sh /model $PORT --no-enable-prefix-caching --max-cudagraph-capture-size 8 $EXTRA \
      > /tmp/moet-mini-r$R.log 2>&1 ) &
  $PY $CLIENT --port $PORT --out /tmp/moet-mini-run-r${R}a.json | tail -1
  $PY $CLIENT --port $PORT --out /tmp/moet-mini-run-r${R}b.json | tail -1
  grep "moet-cpu-tier\] step" /tmp/moet-mini-r$R.log | tail -1
  docker stop -t 20 moet-mini-hyb > /dev/null 2>&1
  wait
done
echo "== comparisons"
for R in $RS; do
  echo "-- R=$R: a vs b (determinism)"; $PY $CLIENT --compare /tmp/moet-mini-run-r${R}a.json /tmp/moet-mini-run-r${R}b.json | tail -1
done
set -- $RS
BASE=$1
for R in $RS; do
  [ "$R" = "$BASE" ] && continue
  echo "-- R=$BASE vs R=$R"; $PY $CLIENT --compare /tmp/moet-mini-run-r${BASE}a.json /tmp/moet-mini-run-r${R}a.json | tail -9
done
