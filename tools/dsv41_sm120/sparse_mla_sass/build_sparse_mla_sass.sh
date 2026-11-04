#!/usr/bin/env bash
# Session V: generate + assemble the sparse-MLA prefill SASS kernel (sm_120).
#
#   build_sparse_mla_sass.sh [out.cubin]            (default: kernels/cubins-sm120/sparse_mla_prefill_v41.cubin)
#
# Requirements (see kernels/MANIFEST.md, "Sparse-MLA prefill"):
#   CUBIT  — a cubit binary with the BUG-421 (path-following barrier allocation) and BUG-422 (BAR.SYNC stall
#            floor 6) fixes; patches: /workspace/tmp/par31/V/cubit/0001-*.patch, 0002-*.patch on cubit 53e9ed8.
#   TABLE  — tables/sm120.json(.gz) of the same cubit tree.
# Scheduling policy (the measured 0.66-0.72x-of-reference build): CUBIT_SB_SHARE=1 (scoreboards shared as counters,
# long uses on the top two, far WAR uses released early at 32 walked instructions), CUBIT_MMA_WAIT=1 +
# CUBIT_MMA_WB_DRAIN=13 (ptxas discipline for the HMMA chain). Without CUBIT_SB_SHARE the kernel is still bit-exact
# but 1.09-1.13x of the reference.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
CUBIT=${CUBIT:-/workspace/tmp/par31/V/cubit-V/target/release/cubit}
TABLE=${TABLE:-/workspace/tmp/par31/V/cubit-V/tables/sm120.json.gz}
OUT=${1:-$REPO/kernels/cubins-sm120/sparse_mla_prefill_v41.cubin}
SASS=$REPO/kernels/sass/sparse_mla_prefill_v41.sass

python3 "$REPO/kernels/gen/gen_sparse_mla_prefill_v41.py" >/dev/null
echo "generated $SASS ($(grep -c ';' "$SASS") statements)"
env CUBIT_SB_SHARE=${CUBIT_SB_SHARE:-1} CUBIT_RB_EARLY=${CUBIT_RB_EARLY:-32} CUBIT_MMA_WAIT=1 CUBIT_MMA_WB_DRAIN=13 \
  "$CUBIT" asm "$SASS" -t "$TABLE" -o "$OUT" --kernel sparse_mla_prefill_v41 \
  --mercury-stub "$REPO/kernels/sass/qmma_e4m3.merc.stub"
echo "wrote $OUT"
