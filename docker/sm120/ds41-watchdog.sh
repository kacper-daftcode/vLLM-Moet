#!/usr/bin/env bash
# Watchdog for the served DeepSeek-V4.1 container (and the Qwen3.8 one) on the serving host. The containers run with
# --restart no on purpose (maintenance windows stop and replace them), so a crash - the host's OOM killer took the
# production worker on 2026-10-14 23:02Z while a second server was starting - stayed unnoticed for 14 hours. Run from
# cron every minute:
#   * * * * * /bin/bash /mnt/k3x/vllm-moet/docker/sm120/ds41-watchdog.sh >> /mnt/k3x/logs/watchdog.log 2>&1
# What it does, per container:
#   - not running (exited / dead / missing) -> saves its log and `docker inspect` to LOG_DIR (<name>-crash-<UTC>.log),
#     renames it <name>-crash-<UTC> (the log stays readable with `docker logs`), starts a new one with the wrapper
#     (ds41-flash: WRAPPER_DIR/run.sh; qwen38: `docker start`, there is no rename - the same container comes back);
#   - running longer than STARTUP_GRACE_MIN minutes but GET /health fails PROBE_FAILS times in a row (one probe per
#     run, 20 s timeout; vLLM answers 500 once its engine is dead) -> the same, after `docker stop -t 60`;
#   - at most one restart per COOLDOWN_MIN minutes and MAX_RESTARTS per 6 h per container; above that it only logs
#     ("needs a human") - a server that cannot stay up should not be restarted in a loop.
# A maintenance window: `touch WRAPPER_DIR/MAINTENANCE` before stopping anything by hand (w8_stop.sh ds41-flash does
# it), `rm` it when the window is over; while it exists the watchdog does nothing but remind every hour.
# Output only when something is wrong or done (a healthy minute prints nothing), one line per event, UTC.
# DRY_RUN=1 prints the actions instead of running them.
set -uo pipefail

WRAPPER_DIR="${WRAPPER_DIR:-/mnt/k3x/ds41flash}"
LOG_DIR="${LOG_DIR:-/mnt/k3x/logs}"
STATE_DIR="${STATE_DIR:-$WRAPPER_DIR/.watchdog}"
MAINTENANCE="${MAINTENANCE:-$WRAPPER_DIR/MAINTENANCE}"
NAME="${NAME:-ds41-flash}"; PORT="${PORT:-8001}"
QWEN="${QWEN:-qwen38}"; QWEN_PORT="${QWEN_PORT:-8000}"
STARTUP_GRACE_MIN="${STARTUP_GRACE_MIN:-35}"   # a DeepSeek start takes 17-25 min on this host
PROBE_FAILS="${PROBE_FAILS:-3}"
COOLDOWN_MIN="${COOLDOWN_MIN:-30}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"              # per 6 h, per container
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 9>"$STATE_DIR/lock"
flock -n 9 || exit 0   # the previous minute's run is still going (a restart in progress)

now_s=$(date -u +%s)
stamp() { date -u +%Y%m%dT%H%M%SZ; }
log() { echo "$(date -u +%FT%TZ) $*"; }
run() { if [[ "$DRY_RUN" == 1 ]]; then log "DRY_RUN: $*"; else "$@"; fi; }

if [[ -e "$MAINTENANCE" ]]; then
  age_min=$(( (now_s - $(stat -c %Y "$MAINTENANCE")) / 60 ))
  # a reminder once an hour, from the sixth hour on
  if (( age_min >= 360 && age_min % 60 == 0 )); then log "MAINTENANCE flag $MAINTENANCE is $((age_min / 60)) h old - still a window? (rm it when done)"; fi
  exit 0
fi

# restart bookkeeping: $STATE_DIR/<name>.restarts holds one epoch per restart (last 6 h kept)
restarts_recent() {  # name -> count in the last 6 h; trims the file
  local f="$STATE_DIR/$1.restarts"
  [[ -e "$f" ]] || { echo 0; return; }
  awk -v now="$now_s" 'NF && $1 > now - 6*3600' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  wc -l < "$f"
}
last_restart_age_min() {  # name -> minutes since the last restart (9999 = never)
  local f="$STATE_DIR/$1.restarts" last
  last=$(awk 'NF { l = $1 } END { print l + 0 }' "$f" 2>/dev/null); last="${last:-0}"
  (( last > 0 )) || { echo 9999; return; }
  echo $(( (now_s - last) / 60 ))
}
may_restart() {  # name -> 0 if a restart is allowed now, else logs why not and returns 1
  local n="$1" recent age
  recent=$(restarts_recent "$n"); age=$(last_restart_age_min "$n")
  if (( age < COOLDOWN_MIN )); then log "$n: down, but restarted $age min ago (cooldown $COOLDOWN_MIN) - waiting"; return 1; fi
  if (( recent >= MAX_RESTARTS )); then log "$n: down, but $recent restarts in 6 h (max $MAX_RESTARTS) - NEEDS A HUMAN, not restarting"; return 1; fi
  return 0
}
save_and_rename() {  # name -> saves docker logs + inspect, renames the container to <name>-crash-<UTC>
  local n="$1" ts; ts=$(stamp)
  if docker inspect "$n" >/dev/null 2>&1; then
    run bash -c "docker logs '$n' > '$LOG_DIR/$n-crash-$ts.log' 2>&1; docker inspect '$n' > '$LOG_DIR/$n-crash-$ts.inspect.json'"
    [[ "$DRY_RUN" == 1 ]] || log "$n: log saved to $LOG_DIR/$n-crash-$ts.log ($(grep -ac Traceback "$LOG_DIR/$n-crash-$ts.log" 2>/dev/null || echo 0) Traceback, OOMKilled $(docker inspect "$n" --format '{{.State.OOMKilled}}' 2>/dev/null))"
    run docker rename "$n" "$n-crash-$ts"
  fi
}
note_restart() { [[ "$DRY_RUN" == 1 ]] || echo "$now_s" >> "$STATE_DIR/$1.restarts"; }

# --- the production server: wrapper start (a fresh container, the launcher clears stale /dev/shm regions)
check_ds41() {
  local st started age_min fails f="$STATE_DIR/$NAME.fails"
  st=$(docker inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null || echo missing)
  if [[ "$st" != running ]]; then
    log "$NAME: $st (exit $(docker inspect "$NAME" --format '{{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} finished {{.State.FinishedAt}}' 2>/dev/null || echo '-'))"
    may_restart "$NAME" || return
    save_and_rename "$NAME"
    log "$NAME: starting a new container with $WRAPPER_DIR/run.sh"
    run bash -c "cd '$WRAPPER_DIR' && ./run.sh > '$LOG_DIR/$NAME-watchdog-start-$(stamp).log' 2>&1 < /dev/null"
    note_restart "$NAME"; rm -f "$f"; return
  fi
  started=$(docker inspect "$NAME" --format '{{.State.StartedAt}}')
  age_min=$(( (now_s - $(date -u -d "$started" +%s)) / 60 ))
  (( age_min < STARTUP_GRACE_MIN )) && return   # still starting (weights, torch.compile, graphs)
  if curl -sf -m 20 -o /dev/null "http://127.0.0.1:$PORT/health"; then rm -f "$f"; return; fi
  fails=$(( $(cat "$f" 2>/dev/null || echo 0) + 1 )); echo "$fails" > "$f"
  log "$NAME: running for $age_min min but GET /health failed ($fails/$PROBE_FAILS)"
  (( fails < PROBE_FAILS )) && return
  may_restart "$NAME" || return
  log "$NAME: unhealthy $fails times - stopping and starting a new container"
  run docker stop -t 60 "$NAME"
  save_and_rename "$NAME"
  run bash -c "cd '$WRAPPER_DIR' && ./run.sh > '$LOG_DIR/$NAME-watchdog-start-$(stamp).log' 2>&1 < /dev/null"
  note_restart "$NAME"; rm -f "$f"
}

# --- qwen38: the same container comes back with `docker start` (its wrapper would recreate it; not needed)
check_qwen() {
  local st started age_min fails f="$STATE_DIR/$QWEN.fails"
  docker inspect "$QWEN" >/dev/null 2>&1 || return   # not deployed on this host
  st=$(docker inspect "$QWEN" --format '{{.State.Status}}')
  if [[ "$st" != running ]]; then
    log "$QWEN: $st (exit $(docker inspect "$QWEN" --format '{{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} finished {{.State.FinishedAt}}'))"
    may_restart "$QWEN" || return
    run bash -c "docker logs '$QWEN' > '$LOG_DIR/$QWEN-crash-$(stamp).log' 2>&1"
    log "$QWEN: docker start"
    run docker start "$QWEN"
    note_restart "$QWEN"; rm -f "$f"; return
  fi
  started=$(docker inspect "$QWEN" --format '{{.State.StartedAt}}')
  age_min=$(( (now_s - $(date -u -d "$started" +%s)) / 60 ))
  (( age_min < 20 )) && return
  if curl -sf -m 20 -o /dev/null "http://127.0.0.1:$QWEN_PORT/health"; then rm -f "$f"; return; fi
  fails=$(( $(cat "$f" 2>/dev/null || echo 0) + 1 )); echo "$fails" > "$f"
  log "$QWEN: running for $age_min min but GET /health failed ($fails/$PROBE_FAILS)"
  (( fails < PROBE_FAILS )) && return
  may_restart "$QWEN" || return
  log "$QWEN: unhealthy $fails times - docker restart"
  run bash -c "docker logs '$QWEN' > '$LOG_DIR/$QWEN-crash-$(stamp).log' 2>&1"
  run docker restart -t 60 "$QWEN"
  note_restart "$QWEN"; rm -f "$f"
}

check_ds41
check_qwen
