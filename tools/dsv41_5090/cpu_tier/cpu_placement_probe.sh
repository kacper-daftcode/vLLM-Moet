#!/bin/bash
# Who runs where while the hybrid decodes: SAMPLES snapshots (every INTERVAL s) of the threads in state R and the CPU
# they are on (ps psr), accumulated per CPU. The tier's layout per rank r: main thread 8r, dispatcher 8r+1, pool
# 8r+2..8r+7, their SMT siblings 8r+34..8r+39 must stay idle (a busy sibling halves the AVX-512 kernel's core, and the
# job's barrier waits for the slowest thread: 125 -> 185-266 us per expert in the bench when a foreign thread shares
# the core). Usage: cpu_placement_probe.sh [SAMPLES=40] [INTERVAL=0.1]
SAMPLES="${1:-40}"
INTERVAL="${2:-0.1}"
TMP=$(mktemp)
for ((i = 0; i < SAMPLES; i++)); do
  ps -eL -o psr=,stat=,pid=,comm= | awk '$2 ~ /^R/ {print $1, $3, $4}' >> "$TMP"
  sleep "$INTERVAL"
done
echo "running threads per CPU over $SAMPLES samples (count comm[pid] ...; pool cores 8r+2..8r+7, siblings 8r+34..8r+39):"
awk '{c[$1]++; k[$1 " " $3 "[" $2 "]"]++} END {
  for (cpu in c) {
    line = sprintf("  cpu %2d: %3d", cpu, c[cpu]);
    for (key in k) { split(key, a, " "); if (a[1] == cpu) line = line sprintf("  %s:%d", a[2], k[key]); }
    print line
  }
}' "$TMP" | sort -k2 -n
rm -f "$TMP"
