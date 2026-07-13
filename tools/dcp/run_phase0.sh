#!/bin/bash
# Phase 0 of the MTP-decay task: acceptance-vs-depth on a REAL corpus vs the
# legacy random-word filler, both on the SAME 1M server boot (A/B without an
# autotune lottery). Runs under the shared probe lock (COORDINATION.md).
#
# Usage: run_phase0.sh [port]
set -euo pipefail
PORT=${1:-8123}
STAMP=$(date -u +%Y%m%d-%H%M)
OUT_DIR=/root/bench-results/$STAMP-mtpdecay-phase0
PY=/opt/hfenv/bin/python3
PROBE=/root/vLLM-Moet/tools/dcp/decode_at_depth_real.py
DEPTHS=8192,200000,500000,700000,891000
mkdir -p "$OUT_DIR"

curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null \
  || { echo "FATAL: no server on :$PORT (boot_dcp1m.sh first)" >&2; exit 1; }

# Server fingerprint for the record.
NAME=$(docker ps --format '{{.Names}} {{.Ports}}' | awk -v p=$PORT '$0 ~ p {print $1; exit}')
NAME=${NAME:-moet-dcp1m}
docker inspect "$NAME" \
  --format '{"image":"{{.Config.Image}}","cmd":{{json .Config.Cmd}},"env":{{json .Config.Env}},"created":"{{.Created}}"}' \
  > "$OUT_DIR/server.json" 2>/dev/null || echo '{}' > "$OUT_DIR/server.json"

echo "[phase0] results -> $OUT_DIR"

run_probe() {  # tag, extra args...
  local tag=$1; shift
  echo "[phase0] probe $tag"
  flock /tmp/moet-dcp1m-probe.lock \
    $PY "$PROBE" --port "$PORT" --depths "$DEPTHS" \
    --warmups 1 --runs 3 --max-tokens 512 \
    --output "$OUT_DIR/$tag.jsonl" "$@" \
    2>&1 | tee "$OUT_DIR/$tag.log"
}

# 1. The validation run: real corpus, story task (original probe character).
run_probe real-story --source-tree /root/vllm-moet-src --task story

# 2. Same boot, legacy junk filler: must reproduce the 2026-07-12 decay
#    (acc ~1.8 @891K) or the original observation itself is in question.
run_probe random-story --random-words --task story

# 3. Cheap secondary task on the real corpus: code task after code context.
run_probe real-refactor --source-tree /root/vllm-moet-src --task refactor

echo "[phase0] done: $OUT_DIR"
