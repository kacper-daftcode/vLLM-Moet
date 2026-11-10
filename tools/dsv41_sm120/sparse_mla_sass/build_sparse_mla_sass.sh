#!/usr/bin/env bash
# Generate + assemble the sparse-MLA prefill SASS kernel (sm_120).
#
#   build_sparse_mla_sass.sh [out.cubin]            (default: kernels/cubins-sm120/sparse_mla_prefill_v41.cubin)
#   SCHEDULE=s1 build_sparse_mla_sass.sh [out.cubin] (default: kernels/cubins-sm120/sparse_mla_prefill_v41s1.cubin)
#
# Requirements (see kernels/MANIFEST.md, "Sparse-MLA prefill"):
#   CUBIT  — cubit `main` at or after ed4d316: the required fixes are BUG-478 (path-following
#            barrier allocation), BUG-479 (BAR.SYNC stall floor 6) and BUG-480 (R128-R254 tracking) there.
#            Default: the main checkout's release build; the validated cubin (sha256 c896234bc76d6938...)
#            reproduces byte for byte from main up to cubit 52767f9 (checked 2026-11-05 at ed4d316, f79a48a,
#            27102a8, 60b196a and 2026-11-06 at 52767f9). From f064bed (BUG-485, the allocator's WAW rule) main emits
#            the same instruction text with two extra wait control words (@P0 VIMNMX R18/R19): the shipped cubin stays
#            pinned to its digest; a rebuild from a newer main is a new schedule and needs the in-situ A/B again.
#   TABLE  — tables/sm120.json(.gz) of the same cubit tree.
# Scheduling policy (the measured 0.66-0.72x-of-reference build): CUBIT_SB_SHARE=1 (scoreboards shared as counters,
# long uses on the top two, far WAR uses released early at 32 walked instructions), CUBIT_MMA_WAIT=1 +
# CUBIT_MMA_WB_DRAIN=13 (ptxas discipline for the HMMA chain). Without CUBIT_SB_SHARE the kernel is still bit-exact
# but 1.09-1.13x of the reference.
# SCHEDULE=s1 — the dependency-aware stall schedule of the same source: the instruction words are those of the default
# build, the control codes are new, the 128 MMA writeback drains are gone (2 540 instructions). On top of the policy
# above: CUBIT_SCHED2=1 (a stall is charged where a consumer needs it; a waiter is never closer than two issue cycles
# behind the instruction that set its barrier), CUBIT_SCHED2_LAT=sm120 (cubit's pair-latency table for sm_120, measured
# on RTX PRO 6000 silicon and enforced on every control-flow path), CUBIT_MMA_PTXAS=1 (ptxas's issue distances for the
# HMMA chain: 16 between two MMAs, 29 into the accumulate operand, 28 into any other reader — no drains). Needs cubit
# `main` at or after 49f8340; reproduces sha256 5924622b6289fca4... from 49f8340 and 38e4cc8. The schedule is a
# separate cubin, selected at run time with VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN=<path>; the default cubin, its
# digest and the Dockerfile's check of it are untouched. See kernels/MANIFEST.md for its gate.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
CUBIT=${CUBIT:-/workspace/cubit/target/release/cubit}
TABLE=${TABLE:-/workspace/cubit/tables/sm120.json.gz}
SCHEDULE=${SCHEDULE:-}
case "$SCHEDULE" in
  "") SCHED_ENV=(); DEFAULT_OUT=$REPO/kernels/cubins-sm120/sparse_mla_prefill_v41.cubin ;;
  s1) SCHED_ENV=(CUBIT_SCHED2=1 CUBIT_SCHED2_LAT=sm120 CUBIT_MMA_PTXAS=1)
      DEFAULT_OUT=$REPO/kernels/cubins-sm120/sparse_mla_prefill_v41s1.cubin ;;
  *) echo "unknown SCHEDULE '$SCHEDULE' (known: s1)" >&2; exit 2 ;;
esac
OUT=${1:-$DEFAULT_OUT}
SASS=$REPO/kernels/sass/sparse_mla_prefill_v41.sass

python3 "$REPO/kernels/gen/gen_sparse_mla_prefill_v41.py" >/dev/null
echo "generated $SASS ($(grep -c ';' "$SASS") statements)"
env CUBIT_SB_SHARE=${CUBIT_SB_SHARE:-1} CUBIT_RB_EARLY=${CUBIT_RB_EARLY:-32} CUBIT_MMA_WAIT=1 CUBIT_MMA_WB_DRAIN=13 \
  ${SCHED_ENV[@]+"${SCHED_ENV[@]}"} \
  "$CUBIT" asm "$SASS" -t "$TABLE" -o "$OUT" --kernel sparse_mla_prefill_v41 \
  --mercury-stub "$REPO/kernels/sass/qmma_e4m3.merc.stub"
echo "wrote $OUT"
