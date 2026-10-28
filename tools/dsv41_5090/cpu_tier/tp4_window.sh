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
# /tmp is tmpfs (three reboots on 10-25 took the day's logs): the window's artefacts are copied to NVMe at the end
KEEP="${KEEP_DIR:-$REPO/internal/dsv41-5090-cpu/handoff/boots}"
keep_artefacts() {
  mkdir -p "$KEEP" 2>/dev/null || return 0
  for f in "$SUMMARY" "/tmp/ds41-hyb-$TAG-outputs.json" "/tmp/ds41-hyb-$TAG-ttft.json" "/tmp/ds41-hyb-$TAG-needle.json" "/tmp/ds41-hyb-$TAG-soak.json" "/tmp/ds41-hyb-$TAG-placement.txt"; do
    [[ -s "$f" ]] && cp -f "$f" "$KEEP/" 2>/dev/null
  done
  [[ -s "$LOG" ]] && gzip -c "$LOG" > "$KEEP/$(basename "$LOG").gz" 2>/dev/null
  return 0
}
# whatever ends this script (a kill included) stops the container first: the GPU locks (inherited from the flock
# chain, held through this process's fds) must outlive the container
cleanup() {
  if docker ps -q -f name="^${NAME}$" | grep -q .; then
    docker logs "$NAME" > "$LOG" 2>&1
    docker stop -t 20 "$NAME" > /dev/null 2>&1
    docker rm -f "$NAME" > /dev/null 2>&1
    echo "- $(date -u '+%Y-%m-%d %H:%M') UTC — MoET hybrid: window ended (boot \`$TAG\` stopped by the script's exit trap; GPUs released)." >> "$BOARD"
  fi
  keep_artefacts
}
trap cleanup EXIT TERM INT

# 1. nothing foreign on the GPUs. Only containers that can touch a GPU count (a --gpus device request, the nvidia
# runtime, or NVIDIA_VISIBLE_DEVICES in their environment): the other lines' CPU-only containers (SWE-bench task and
# scoring containers, `minisweagent-*` / `sweb.eval.*`) run for hours and must not block a window (10-26 14:24: a
# window aborted on eight scoring containers after waiting 2 h 45 for the locks)
gpu_containers=""
for c in $(docker ps --format '{{.Names}}'); do
  info=$(docker inspect -f '{{len .HostConfig.DeviceRequests}} {{.HostConfig.Runtime}}{{range .Config.Env}} {{.}}{{end}}' "$c" 2>/dev/null) || continue
  devreq=${info%% *}
  if [[ "$devreq" != "0" || "$info" == *" nvidia"* || "$info" == *"NVIDIA_VISIBLE_DEVICES="* && "$info" != *"NVIDIA_VISIBLE_DEVICES=void"* ]]; then
    gpu_containers+="$c "
  fi
done
if [[ -n "$gpu_containers" ]]; then
  note "ABORT: GPU containers running: $gpu_containers"; exit 2
fi
if nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -q .; then
  note "ABORT: compute processes on the GPUs: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"; exit 2
fi
# host RAM: the hybrid pins ~396 GB (4 x 52 GB store + 4 x 47 GiB Engram) and needs ~480 of the 498 GB while up; a boot
# into a box with less than RAM_MIN_GB available dies in the OOM killer at the JIT warmup (10-24) - fail fast instead
avail_gb=$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
if (( avail_gb < ${RAM_MIN_GB:-465} )); then
  note "ABORT: MemAvailable ${avail_gb} GB < ${RAM_MIN_GB:-465} GB needed for the boot; top RSS: $(ps -eo rss,comm --sort=-rss | awk 'NR>1 && NR<5 {printf "%s %.0f GB; ", $2, $1/1048576}')"; exit 2
fi
echo "- $(date -u '+%Y-%m-%d %H:%M') UTC — MoET hybrid: **4-GPU window started** (TP4 boot \`$TAG\`: $*; container \`$NAME\`, port $PORT, log $LOG)." >> "$BOARD"
note "window start: boot $TAG ($*); MemAvailable ${avail_gb} GB; GPU power limits: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | awk '{printf "%d ", $1}')W (cap $( (. /etc/default/gpu-power-cap 2>/dev/null; echo "${CAP_W:-400}") ) W, re-applied by the launcher if above)"
# the box's memory-management activity across the window (compaction / page migration -> TLB shootdowns on the workers'
# cores; kcompactd / khugepaged CPU time): snapshots at start, ready, end
memsnap() {
  awk '/^(compact_stall|compact_fail|compact_success|compact_daemon_wake|pgmigrate_success|pgmigrate_fail|thp_fault_alloc|thp_collapse_alloc|pgscan_kswapd|pgsteal_kswapd|pgscan_direct|allocstall_normal) / {printf "%s=%s ", $1, $2}' /proc/vmstat
  awk '/^ *TLB:/ {s = 0; for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]+$/) s += $i; printf "tlb_shootdowns=%d ", s}' /proc/interrupts
  for p in $(pgrep -x kcompactd0) $(pgrep -x khugepaged); do awk -v p="$p" '{printf "%s_cpu_s=%.0f ", $2, ($14 + $15) / 100}' "/proc/$p/stat" 2>/dev/null; done
  echo
}
memdelta() {  # memdelta "<snap a>" "<snap b>" -> the differences of every counter
  awk -v a="$1" -v b="$2" 'BEGIN {
    n = split(a, A, " "); for (i = 1; i <= n; i++) { split(A[i], kv, "="); va[kv[1]] = kv[2] }
    n = split(b, B, " "); for (i = 1; i <= n; i++) { split(B[i], kv, "="); if (kv[1] in va) printf "%s %+d  ", kv[1], kv[2] - va[kv[1]] }
    print "" }'
}
SNAP0=$(memsnap)
# DROP_CACHES=1: the boot pins ~400 GB and evicts the page cache anyway; doing it first spares the loader the direct-reclaim
# storm (10-25 boot: pgscan_direct +162 M pages, allocstall +100 K, compact_stall +68 K; the 48 safetensors shards
# re-read from NVMe at 36-40 GB/s cost nothing against the loader's own pace)
if [[ "${DROP_CACHES:-0}" == "1" ]]; then
  sync; echo 3 > /proc/sys/vm/drop_caches; note "page cache dropped before the boot: $(free -g | awk '/^Mem:/ {print $6 " GB buff/cache, " $7 " GB available"}')"
fi

# 1b. BAKED=1 (the launcher's default since 10-28) runs the image's copy of tools/dsv41_5090: warn when it is not the worktree's
# (a change committed or not, without `docker build` of the hyb2 image) - the window would measure stale code
cd "$REPO" || exit 1
if [[ "${BAKED:-1}" == "1" ]]; then
  img="${IMAGE:-vllm-moet-sm120:dsv41-nightly-df3bd094-hyb2}"
  files='cpu_tier/*.py cpu_tier/*.cpp cpu_tier/*.cu cpu_tier/*.h cpu_moe/*.cpp cpu_moe/*.h cpu_moe/*.py'
  stale=$(diff <(docker run --rm --entrypoint bash "$img" -c "cd /opt/vllm-moet/dsv41_5090 && md5sum $files 2>/dev/null" 2>/dev/null) \
               <(cd tools/dsv41_5090 && eval md5sum $files 2>/dev/null) | grep '^>' | awk '{print $3}' | tr '\n' ' ')
  if [[ -n "$stale" ]]; then
    note "WARNING: the baked image $img differs from the worktree in: $stale(rebuild it, or run BAKED=0)"
  else
    note "baked image $img = the worktree's tools/dsv41_5090 (commit label: $(docker image inspect "$img" --format '{{index .Config.Labels "com.vllm-moet.cpu-tier-commit"}}' 2>/dev/null), HEAD $(git rev-parse --short HEAD))"
  fi
fi

# 2. boot
bash docker/sm120/run-dsv41-5090.sh || { note "launcher failed"; exit 1; }
t0=$(date +%s)
ready=0
dropped2=0
while (( $(date +%s) - t0 < READY_TIMEOUT )); do
  if ! docker ps -q -f name="^${NAME}$" | grep -q .; then
    note "container exited during startup after $(( $(date +%s) - t0 )) s"; break
  fi
  if curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then ready=1; break; fi
  # the second drop: the weight loading leaves ~290 GB of clean shard pages in the cache and the store pinning (260 GB
  # of cudaHostAlloc) then reclaims them under pressure - 10-26: 16 min stuck in direct reclaim with another line's
  # docker pulls dirtying the cache in parallel; dropping right after the weights are read frees the pages at once
  if [[ "${DROP_CACHES:-0}" == "1" && $dropped2 == 0 ]] && docker logs "$NAME" 2>&1 | grep -q "Loading weights took"; then
    sync; echo 3 > /proc/sys/vm/drop_caches; dropped2=1
    note "page cache dropped again after the weights were read ($(( $(date +%s) - t0 )) s): $(free -g | awk '/^Mem:/ {print $7 " GB available"}')"
  fi
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
SNAP1=$(memsnap)
note "memory management during the boot: $(memdelta "$SNAP0" "$SNAP1")"
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
# the RAM item: Shmem (458 GB) exceeds the pinned store + Engram (411) and the workers' RSS differ by rank - the biggest
# mappings of the first and the third worker (size / rss / name) show what the rest is
for p in $(pgrep -f "VLLM::Worker_TP" | sed -n '1p;3p'); do
  echo "  pid $p largest mappings (size GB / rss GB / path):" | tee -a "$SUMMARY"
  awk '/^[0-9a-f]+-[0-9a-f]+ / {path = (NF >= 6) ? $6 : "[anon]"} /^Size:/ {sz = $2} /^Rss:/ {if (sz > 1048576) printf "    %6.1f %6.1f %s\n", sz / 1048576, $2 / 1048576, path}' "/proc/$p/smaps" 2>/dev/null | sort -rn | head -8 | tee -a "$SUMMARY"
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
SNAP2=$(memsnap)
note "memory management during the measurements: $(memdelta "$SNAP1" "$SNAP2")"
grep "moet-cpu-tier\] rank 0: step" "$LOG" | tail -3 | cut -c1-300 | tee -a "$SUMMARY"
for r in 0 1 2 3; do printf "  rank $r in-kernel us per expert (KPI lines): " | tee -a "$SUMMARY"; grep -o "rank $r: step [0-9]*:.*in-kernel [0-9]* us" "$LOG" | grep -v "step [0-9]:" | sed -E 's/.*in-kernel ([0-9]+) us/\1/' | tr '\n' ' ' | tee -a "$SUMMARY"; echo | tee -a "$SUMMARY"; done
wait  # the placement probe
if [[ -s "/tmp/ds41-hyb-$TAG-placement.txt" ]]; then
  note "CPU placement during decode (pool cores and their siblings; full list /tmp/ds41-hyb-$TAG-placement.txt):"
  # pool cores 8r+2..8r+7 and their siblings 8r+34..8r+39 are exactly the CPUs with cpu % 8 >= 2
  awk '/^  cpu/ {cpu = $2 + 0; if (cpu % 8 >= 2) print}' "/tmp/ds41-hyb-$TAG-placement.txt" | head -48 | tee -a "$SUMMARY"
fi
# optional (phase 3, G3): HARD_PROBE=humaneval,math500 - the W2 line's hard_probe.py (copied to internal/…/probe/ with the
# native fin01 references): greedy, thinking off; HumanEval executed (164 items), MATH-500 level 5 (60 items); paired
# comparison against native (McNemar). ~8 + ~12 min at ~100 tok/s; decode-dominated (prompts of a few hundred tokens)
if [[ -n "${HARD_PROBE:-}" ]]; then
  PROBE=$REPO/internal/dsv41-5090-cpu/probe
  for suite in ${HARD_PROBE//,/ }; do
    # math500: the native fin01 reference = the first 60 level-5 items at --max-tokens 2500 (the probe's default 1200
    # truncates a third of them: 10-26); humaneval: all 164 at the default 768
    lv=(); [[ "$suite" == "math500" ]] && lv=(--levels 5 --n "${PROBE_MATH_N:-60}" --max-tokens "${PROBE_MATH_TOKENS:-2500}")
    note "hard probe $suite (tag $TAG) ..."
    ( cd "$PROBE" && $PY hard_probe.py run "http://127.0.0.1:$PORT" "$MODEL" "$suite" "$TAG" --conc "${PROBE_CONC:-1}" "${lv[@]}" 2>&1 | tail -4 | cut -c1-300 ) | tee -a "$SUMMARY"
    note "hard probe $suite vs native fin01:"
    ( cd "$PROBE" && $PY hard_probe.py compare "hard_${suite}_native_fin01.jsonl" "hard_${suite}_${TAG}.jsonl" 2>&1 | tail -12 | cut -c1-300 ) | tee -a "$SUMMARY"
    docker logs "$NAME" > "$LOG" 2>&1
    grep "moet-cpu-tier\] rank 0: step" "$LOG" | tail -1 | cut -c1-300 | tee -a "$SUMMARY"
  done
fi
# optional (phase 3): NEEDLE_LENGTHS=32768,65536 - a code word planted at NEEDLE_DEPTHS (0.1,0.5,0.9) of long fillers,
# asked back (needle_probe.py); SOAK_MINUTES=120 SOAK_CONC=4 - concurrent mixed-prompt streams (soak_probe.py; the
# server needs MAX_NUM_SEQS >= SOAK_CONC); the tier's error counter and hit rate from the KPI lines after each
if [[ -n "${NEEDLE_LENGTHS:-}" ]]; then
  note "needle probe: lengths $NEEDLE_LENGTHS, depths ${NEEDLE_DEPTHS:-0.1,0.5,0.9}"
  $PY "$REPO/tools/dsv41_5090/cpu_tier/needle_probe.py" --port "$PORT" --model "$MODEL" --lengths "$NEEDLE_LENGTHS" \
    --depths "${NEEDLE_DEPTHS:-0.1,0.5,0.9}" --out "/tmp/ds41-hyb-$TAG-needle.json" 2>&1 | tee -a "$SUMMARY"
  docker logs "$NAME" > "$LOG" 2>&1
  grep "moet-cpu-tier\] rank 0: step" "$LOG" | tail -1 | cut -c1-300 | tee -a "$SUMMARY"
fi
if [[ -n "${SOAK_MINUTES:-}" ]]; then
  note "soak: $SOAK_MINUTES min x ${SOAK_CONC:-4} streams, max_tokens ${SOAK_TOKENS:-256}"
  $PY "$REPO/tools/dsv41_5090/cpu_tier/soak_probe.py" --port "$PORT" --model "$MODEL" --minutes "$SOAK_MINUTES" \
    --conc "${SOAK_CONC:-4}" --max-tokens "${SOAK_TOKENS:-256}" --out "/tmp/ds41-hyb-$TAG-soak.json" 2>&1 | tail -8 | tee -a "$SUMMARY"
  docker logs "$NAME" > "$LOG" 2>&1
  grep "moet-cpu-tier\] rank 0: step" "$LOG" | tail -1 | cut -c1-300 | tee -a "$SUMMARY"
  note "server log: error / traceback lines after the soak: $(grep -ciE "error word|fail.*closed|Traceback" "$LOG")"
  free -g | head -2 | tail -1 | tee -a "$SUMMARY"
fi
# optional (phase 2.5): TTFT / prefill rate on long prompts, e.g. TTFT_LENGTHS=2048,14336,30720 (needs SCRATCH > 0 for
# correct outputs; MAX_NUM_BATCHED_TOKENS sets the chunk)
if [[ -n "${TTFT_LENGTHS:-}" ]]; then
  note "TTFT probe: lengths $TTFT_LENGTHS, gen ${TTFT_GEN:-64}, repeat ${TTFT_REPEAT:-2}"
  $PY "$REPO/tools/dsv41_5090/cpu_tier/ttft_probe.py" --port "$PORT" --model "$MODEL" --lengths "$TTFT_LENGTHS" \
    --gen "${TTFT_GEN:-64}" --repeat "${TTFT_REPEAT:-2}" --vary-repeats --out "/tmp/ds41-hyb-$TAG-ttft.json" 2>&1 | tee -a "$SUMMARY"
  docker logs "$NAME" > "$LOG" 2>&1
  grep -E "moet-cpu-tier\] rank 0: (prefill call|scratch pool|DEBUG prefill|[0-9]+-token MoE call)" "$LOG" | tail -8 | cut -c1-600 | tee -a "$SUMMARY"
  grep "moet-cpu-tier\] rank 0: step" "$LOG" | tail -1 | cut -c1-300 | tee -a "$SUMMARY"
fi
note "tier warnings in the log: $(grep -c "moet-cpu-tier\] rank 0: WARNING" "$LOG")"
if [[ "${DEBUG_COMPACT:-0}" == "1" ]]; then
  # the within-boot identity check (every compact call vs all its experts in one b12x run): totals per rank, mismatches
  for r in 0 1 2 3; do grep -o "rank $r: step [0-9]*:.*compact check: .*" "$LOG" | tail -1 | sed -E "s/(rank $r: step [0-9]+):.*compact check:/\1 compact check:/" | tee -a "$SUMMARY"; done
  note "compact check MISMATCH lines (all ranks): $(grep -c "DEBUG compact.*MISMATCH" "$LOG")"
  grep "DEBUG compact.*MISMATCH" "$LOG" | head -5 | cut -c1-300 | tee -a "$SUMMARY"
  note "compact check skipped lines: $(grep -c "DEBUG compact.*check skipped" "$LOG")"
fi
grep -i "fail\|error word\|Traceback" "$LOG" | grep -v "errors 0" | head -5 | tee -a "$SUMMARY"

# 4. stop, release
docker stop -t 60 "$NAME" > /dev/null 2>&1
docker logs "$NAME" > "$LOG" 2>&1
docker rm -f "$NAME" > /dev/null 2>&1
note "window end: $TAG stopped; log $LOG"
echo "- $(date -u '+%Y-%m-%d %H:%M') UTC — MoET hybrid: window ended (boot \`$TAG\` stopped; GPUs released; summary $SUMMARY)." >> "$BOARD"
exit 0
