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
# where the host RAM goes (the RAM budget is an open item): meminfo + the workers' smaps_rollup
grep -E "^(MemAvailable|Shmem|AnonPages|Mapped|Cached|Unevictable|Mlocked|KernelStack|PageTables|VmallocUsed):" /proc/meminfo | awk '{printf "  %s %.1f GB\n", $1, $2/1048576}' | tr '\n' ' ' | tee -a "$SUMMARY"; echo | tee -a "$SUMMARY"
for p in $(pgrep -f "VLLM::Worker_TP" | head -4); do
  awk -v p="$p" '/^(Rss|Pss|Shared_Clean|Shared_Dirty|Private_Dirty|Anonymous|Shmem.*Rss|Locked):/ {printf "%s %.1f GB  ", $1, $2/1048576} END {print "(pid " p ")"}' "/proc/$p/smaps_rollup" 2>/dev/null | tee -a "$SUMMARY"
done

# 3. measurements: the identity client (10 short prompts x 48 tokens, saved), then decode tok/s on longer generations
MODEL="deepseek-ai/DeepSeek-V4.1-Flash"
$PY "$REPO/tools/dsv41_5090/cpu_tier/compare_mini_outputs.py" --port "$PORT" --model "$MODEL" --out "/tmp/ds41-hyb-$TAG-outputs.json" 2>&1 | tee -a "$SUMMARY"
# who runs on which CPU while the first decode runs (foreign threads on the pool cores / their SMT siblings slow the kernel)
( sleep 2; bash "$REPO/tools/dsv41_5090/cpu_tier/cpu_placement_probe.sh" "${PLACEMENT_SAMPLES:-40}" 0.1 > "/tmp/ds41-hyb-$TAG-placement.txt" 2>&1 ) &
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
grep -o "worker-0 in-kernel [0-9]* us per expert" "$LOG" | sort | uniq -c | sort -rn | head -3 | tee -a "$SUMMARY"
wait  # the placement probe
if [[ -s "/tmp/ds41-hyb-$TAG-placement.txt" ]]; then
  note "CPU placement during decode (pool cores and their siblings; full list /tmp/ds41-hyb-$TAG-placement.txt):"
  # pool cores 8r+2..8r+7 and their siblings 8r+34..8r+39 are exactly the CPUs with cpu % 8 >= 2
  awk '/^  cpu/ {cpu = $2 + 0; if (cpu % 8 >= 2) print}' "/tmp/ds41-hyb-$TAG-placement.txt" | head -48 | tee -a "$SUMMARY"
fi
# optional (phase 2.5): TTFT / prefill rate on long prompts, e.g. TTFT_LENGTHS=2048,14336,30720 (needs SCRATCH > 0 for
# correct outputs; MAX_NUM_BATCHED_TOKENS sets the chunk)
if [[ -n "${TTFT_LENGTHS:-}" ]]; then
  note "TTFT probe: lengths $TTFT_LENGTHS, gen ${TTFT_GEN:-64}, repeat ${TTFT_REPEAT:-2}"
  $PY "$REPO/tools/dsv41_5090/cpu_tier/ttft_probe.py" --port "$PORT" --model "$MODEL" --lengths "$TTFT_LENGTHS" \
    --gen "${TTFT_GEN:-64}" --repeat "${TTFT_REPEAT:-2}" --out "/tmp/ds41-hyb-$TAG-ttft.json" 2>&1 | tee -a "$SUMMARY"
  docker logs "$NAME" > "$LOG" 2>&1
  grep -E "moet-cpu-tier\] rank 0: (prefill call|scratch pool|DEBUG prefill|[0-9]+-token MoE call)" "$LOG" | tail -8 | cut -c1-600 | tee -a "$SUMMARY"
  grep "moet-cpu-tier\] rank 0: step" "$LOG" | tail -1 | cut -c1-300 | tee -a "$SUMMARY"
fi
grep -c "moet-cpu-tier\] rank 0: WARNING" "$LOG" | xargs -I{} note "tier warnings in the log: {}"
if [[ "${DEBUG_COMPACT:-0}" == "1" ]]; then
  # the within-boot identity check (every compact call vs all its experts in one b12x run): totals per rank, mismatches
  for r in 0 1 2 3; do grep -o "rank $r: step [0-9]*:.*compact check: .*" "$LOG" | tail -1 | sed -E "s/(rank $r: step [0-9]+):.*compact check:/\1 compact check:/" | tee -a "$SUMMARY"; done
  grep -c "DEBUG compact.*MISMATCH" "$LOG" | xargs -I{} note "compact check MISMATCH lines (all ranks): {}"
  grep "DEBUG compact.*MISMATCH" "$LOG" | head -5 | cut -c1-300 | tee -a "$SUMMARY"
  grep -c "DEBUG compact.*check skipped" "$LOG" | xargs -I{} note "compact check skipped lines: {}"
fi
grep -i "fail\|error word\|Traceback" "$LOG" | grep -v "errors 0" | head -5 | tee -a "$SUMMARY"

# 4. stop, release
docker stop -t 60 "$NAME" > /dev/null 2>&1
docker logs "$NAME" > "$LOG" 2>&1
docker rm -f "$NAME" > /dev/null 2>&1
note "window end: $TAG stopped; log $LOG"
echo "- $(date -u '+%Y-%m-%d %H:%M') UTC — MoET hybrid: window ended (boot \`$TAG\` stopped; GPUs released; summary $SUMMARY)." >> "$BOARD"
exit 0
