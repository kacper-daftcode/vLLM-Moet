#!/usr/bin/env bash
# Retention for the KV offload disk tier of run-dsv41.sh (KV_OFFLOAD_FS_DIR). vLLM's "fs" tier writes one file per
# offloaded block, named by its content hash, and never deletes any, so the tree only grows. Run from cron, e.g.
#   0 * * * * /srv/vllm-moet/docker/sm120/kvcache-ttl.sh /srv/kvcache >> /var/log/kvcache-ttl.log 2>&1
# 1. deletes block files not read for TTL_HOURS (72, the retention the V4.1 report gives its SSD tier). It goes by
#    atime, which a relatime mount still advances on the first read of a day, so a file read at least every
#    48 h stays. Blocks served from the RAM tier do not touch their file: a prefix that stays hot in RAM
#    for three days loses its disk copy and is recomputed once if RAM evicts it later.
# 2. then, while the tree is above MAX_GB (200), deletes the least recently read files down to 90 % of MAX_GB,
# 3. and removes *.tmp files older than an hour (a write interrupted by a crash).
# The server treats a missing file as a miss and recomputes the block, so this is safe to run while it serves.
set -euo pipefail

DIR="${1:?usage: kvcache-ttl.sh DIR   (env: TTL_HOURS=72 MAX_GB=200)}"
TTL_HOURS="${TTL_HOURS:-72}"
MAX_GB="${MAX_GB:-200}"
[[ -d "$DIR" ]] || { echo "no such directory: $DIR" >&2; exit 1; }

tree_bytes() { find "$DIR" -type f -printf '%s\n' | awk '{ s += $1 } END { print s + 0 }'; }

before=$(tree_bytes)
n_ttl=$(find "$DIR" -type f -name '*.bin' -amin +$((TTL_HOURS * 60)) -print -delete | wc -l)
find "$DIR" -type f -name '*.tmp' -mmin +60 -delete

n_cap=0
max=$((MAX_GB * 1000 * 1000 * 1000))
size=$(tree_bytes)
if (( size > max )); then
  list=$(mktemp)
  find "$DIR" -type f -name '*.bin' -printf '%A@ %s %p\n' | sort -n |
    awk -v excess=$((size - max * 9 / 10)) '
      freed < excess { freed += $2; sub(/^[^ ]+ [^ ]+ /, ""); print }' > "$list"
  n_cap=$(wc -l < "$list")
  xargs -r -d '\n' rm -f -- < "$list"
  rm -f "$list"
fi

echo "$(date -u +%FT%TZ) $DIR: $((before / 1000000)) -> $(( $(tree_bytes) / 1000000 )) MB;" \
     "not read for ${TTL_HOURS} h: $n_ttl files removed; over ${MAX_GB} GB: $n_cap files removed"
