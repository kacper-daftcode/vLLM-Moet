#!/bin/bash
# The MoET hybrid's host: keep the kernel's memory-management threads off the CPU-tier pool cores and their SMT siblings.
# The hybrid pins ~450 GiB of unmovable pages; after a few pin / unpin cycles the kernel's compaction daemon grinds against them
# (pgmigrate_fail in the millions) and, if the scheduler puts it on a pool core's sibling, halves that core - the pool's 6-thread
# expert job waits at its barrier for the slowest thread: +10-20 % per expert, the first sample of a window 3-5 % slow (2026-11-04).
# Pin kcompactd / kswapd / khugepaged to the siblings of the ranks' main + dispatcher cores (8r+32, 8r+33), where the window script
# already pins the engine core and the API server. Installed by moet-kthread-affinity.service (runs once at boot; kthread PIDs are
# stable until the next boot); re-run by hand after a boot: bash kthread-affinity.sh [CPUS]
CPUS="${1:-${ENGINE_CPUS:-32,33,40,41,48,49,56,57}}"
for name in kcompactd0 kcompactd1 kswapd0 kswapd1 khugepaged; do
  for pid in $(pgrep -x "$name"); do
    taskset -a -cp "$CPUS" "$pid" > /dev/null 2>&1 && echo "$name[$pid] -> $CPUS"
  done
done
