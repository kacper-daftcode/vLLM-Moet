#!/bin/bash
# One boot of the full model on 4x 5090 (phase 2.4) inside a 4-GPU window: start the hybrid serve, wait for it,
# measure (short-prompt identity outputs, decode tok/s, VRAM, host RSS, the tier's KPI lines), stop it. Meant to run
# under the four GPU locks, taken in order 0 -> 1 -> 2 -> 3 (the same order the other lines use):
#   flock -w 7200 /tmp/exl3diff-gpu0.lock flock -w 7200 /tmp/exl3diff-gpu1.lock flock -w 7200 /tmp/exl3diff-gpu2.lock \
#     flock -w 7200 /tmp/exl3diff-gpu3.lock bash tools/dsv41_5090/cpu_tier/tp4_window.sh TAG [env for run-dsv41-5090.sh ...]
#   e.g. ... tp4_window.sh static SPARE=0 PROMOTIONS=0
#        ... tp4_window.sh promote SPARE=13 PROMOTIONS=4
# Outputs: /tmp/ds41-hyb-<TAG>.log (container log), /tmp/ds41-hyb-<TAG>-outputs.json (the identity client's run),
# /tmp/ds41-hyb-<TAG>-summary.txt. Appends start / end lines to /root/work/GPU-BOARD.md.
set -u
TAG="${1:?tag}"
shift
for kv in "$@"; do export "$kv"; done
REPO=/mnt/nvme/vLLM-Moet-cpu
PY=/root/work/venv/bin/python
NAME="${NAME:-ds41-hyb}"
PORT="${PORT:-8041}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"
GEN_TOKENS="${GEN_TOKENS:-256}"
LOG=/tmp/ds41-hyb-$TAG.log
SUMMARY=/tmp/ds41-hyb-$TAG-summary.txt
BOARD=/root/work/GPU-BOARD.md
export MODEL_DIR="${MODEL_DIR:-/mnt/nvme/models/DeepSeek-V4.1-Flash}"

note() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }
: > "$SUMMARY"
# whatever ends this script (a kill included) stops the container first: the GPU locks (inherited from the flock
# chain, held through this process's fds) must outlive the container
cleanup() {
  if docker ps -q -f name="^${NAME}$" | grep -q .; then
    docker logs "$NAME" > "$LOG" 2>&1
    docker stop -t 20 "$NAME" > /dev/null 2>&1
    docker rm -f "$NAME" > /dev/null 2>&1
    echo "- $(date -u '+%Y-%m-%d %H:%M') UTC — MoET hybrid: window ended (boot \`$TAG\` stopped by the script's exit trap; GPUs released)." >> "$BOARD"
  fi
}
trap cleanup EXIT TERM INT

# 1. nothing foreign on the GPUs
if docker ps --format '{{.Names}}' | grep -q .; then
  note "ABORT: containers running: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; exit 2
fi
if nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -q .; then
  note "ABORT: compute processes on the GPUs: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"; exit 2
fi
echo "- $(date -u '+%Y-%m-%d %H:%M') UTC — MoET hybrid: **4-GPU window started** (TP4 boot \`$TAG\`: $*; container \`$NAME\`, port $PORT, log $LOG)." >> "$BOARD"
note "window start: boot $TAG ($*)"

# 2. boot
cd "$REPO" || exit 1
bash docker/sm120/run-dsv41-5090.sh || { note "launcher failed"; exit 1; }
t0=$(date +%s)
ready=0
while (( $(date +%s) - t0 < READY_TIMEOUT )); do
  if ! docker ps -q -f name="^${NAME}$" | grep -q .; then
    note "container exited during startup after $(( $(date +%s) - t0 )) s"; break
  fi
  if curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then ready=1; break; fi
  sleep 10
done
docker logs "$NAME" > "$LOG" 2>&1
if (( ready == 0 )); then
  note "NOT READY after $(( $(date +%s) - t0 )) s; last log lines:"
  grep -v "it/s\]" "$LOG" | tail -40 | tee -a "$SUMMARY"
  docker stop -t 30 "$NAME" > /dev/null 2>&1; docker logs "$NAME" > "$LOG" 2>&1; docker rm -f "$NAME" > /dev/null 2>&1
  echo "- $(date -u '+%Y-%m-%d %H:%M') UTC — MoET hybrid: window ended (boot \`$TAG\` did not come up; GPUs released)." >> "$BOARD"
  exit 1
fi
note "ready after $(( $(date +%s) - t0 )) s"
if [[ "${PIN_ENGINE:-1}" == "1" ]]; then
  # the engine core and the API server off the kernel pools' cores (the workers pin their own threads)
  for p in $(pgrep -f "VLLM::EngineCore" ; docker inspect --format '{{.State.Pid}}' "$NAME"); do
    taskset -a -cp "${ENGINE_CPUS:-32,33,40,41,48,49,56,57}" "$p" > /dev/null 2>&1 && note "pinned pid $p ($(cat /proc/$p/comm 2>/dev/null)) to ${ENGINE_CPUS:-32,33,40,41,48,49,56,57}"
  done
fi
grep -E "moet-cpu-tier\] rank 0: (store|tier started|layer language_model.model.layers.0.ffn)|Model loading took|Available KV cache memory|GPU KV cache size|init engine" "$LOG" | cut -c1-220 | tee -a "$SUMMARY"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee -a "$SUMMARY"
ps -eo pid,rss,comm,args | grep -E "VLLM|EngineCore|vllm" | grep -v grep | awk '{printf "  pid %s rss %.1f GB %s\n", $1, $2/1048576, $3}' | tee -a "$SUMMARY"
free -g | head -2 | tee -a "$SUMMARY"

# 3. measurements: the identity client (10 short prompts x 48 tokens, saved), then decode tok/s on longer generations
MODEL="deepseek-ai/DeepSeek-V4.1-Flash"
$PY "$REPO/tools/dsv41_5090/cpu_tier/compare_mini_outputs.py" --port "$PORT" --model "$MODEL" --out "/tmp/ds41-hyb-$TAG-outputs.json" 2>&1 | tee -a "$SUMMARY"
for p in "Hello" "def fib(n):"; do
  $PY - "$PORT" "$MODEL" "$p" "$GEN_TOKENS" <<'EOF' 2>&1 | tee -a "$SUMMARY"
import json, sys, time, urllib.request
port, model, prompt, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
body = dict(model=model, prompt=prompt, max_tokens=n, temperature=0, seed=0, ignore_eos=True)
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=1800))
dt = time.time() - t0
u = r.get("usage", {})
print(f"  decode {prompt!r}: {u.get('completion_tokens')} tokens in {dt:.1f} s = {u.get('completion_tokens', 0) / dt:.1f} tok/s (prompt {u.get('prompt_tokens')} tokens); text {r['choices'][0]['text'][:60]!r}")
EOF
done
docker logs "$NAME" > "$LOG" 2>&1
grep "moet-cpu-tier\] rank 0: step" "$LOG" | tail -3 | cut -c1-300 | tee -a "$SUMMARY"
grep -c "moet-cpu-tier\] rank 0: WARNING" "$LOG" | xargs -I{} note "tier warnings in the log: {}"
grep -i "fail\|error word\|Traceback" "$LOG" | grep -v "errors 0" | head -5 | tee -a "$SUMMARY"

# 4. stop, release
docker stop -t 60 "$NAME" > /dev/null 2>&1
docker logs "$NAME" > "$LOG" 2>&1
docker rm -f "$NAME" > /dev/null 2>&1
note "window end: $TAG stopped; log $LOG"
echo "- $(date -u '+%Y-%m-%d %H:%M') UTC — MoET hybrid: window ended (boot \`$TAG\` stopped; GPUs released; summary $SUMMARY)." >> "$BOARD"
exit 0
