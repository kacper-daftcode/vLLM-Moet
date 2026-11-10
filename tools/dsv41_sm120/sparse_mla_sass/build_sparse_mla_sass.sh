#!/usr/bin/env bash
# Generate + assemble the sparse-MLA prefill SASS kernel (sm_120).
#
#   build_sparse_mla_sass.sh [out.cubin]            (default: kernels/cubins-sm120/sparse_mla_prefill_v41.cubin)
#   SCHEDULE=s1 build_sparse_mla_sass.sh [out.cubin] (default: kernels/cubins-sm120/sparse_mla_prefill_v41s1.cubin)
#   KERNEL=v42 build_sparse_mla_sass.sh [out.cubin]  (default: kernels/cubins-sm120/sparse_mla_prefill_v42.cubin)
#   KERNEL=v43 build_sparse_mla_sass.sh [out.cubin]  (default: kernels/cubins-sm120/sparse_mla_prefill_v43.cubin)
#   KERNEL=v43b build_sparse_mla_sass.sh [out.cubin] (default: kernels/cubins-sm120/sparse_mla_prefill_v43b.cubin)
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
# KERNEL=v42 — the next source: v41's text with the dequant of the staged records done by sm_120's scaled BF16 unpack
# (kernels/gen/gen_sparse_mla_prefill_v42.py -> kernels/sass/sparse_mla_prefill_v42.sass, 1 454 instructions), always
# with the s1 schedule switches. The entry symbol stays sparse_mla_prefill_v41 (the cubin loads wherever a v41 cubin
# does). Needs cubit `main` at or after 89564ac (the instruction's table row); reproduces sha256 dfdc0279ea5a3388...
# from db4cfd1. A separate cubin as well; see kernels/MANIFEST.md for its gate.
# KERNEL=v43 — v42 with two load waits taken out (kernels/gen/gen_sparse_mla_prefill_v43.py ->
# kernels/sass/sparse_mla_prefill_v43.sass, 1 489 instructions): the candidate index of chunk c+2 is loaded one chunk
# ahead, and the prologue's global loads are issued back to back. The s1 schedule switches plus CUBIT_SB_LONG_KEEP=1
# (cubit's opt-in colouring rule: a prefetch keeps its scoreboard to itself — without it the same text assembles to a
# legal, bit-identical kernel whose gather prefetch shares a scoreboard with a wait at the start of QK). Needs a cubit
# that has the rule (commit df4c557; the script refuses a binary without it); reproduces sha256 43f04db5cc9e9874...
# The entry symbol stays sparse_mla_prefill_v41. A separate cubin as well; see kernels/MANIFEST.md for its gate.
# KERNEL=v43b — v43's source and v43's instruction words with one stall raised: the last HMMA of PV at least 14 issue
# cycles in front of the two conditional branches behind it (v41s1 / v42 / v43: 7 and 12-13; the default cubin, with its
# writeback drains: 39 and 44). ptxas never issues a conditional branch closer than 14 cycles behind an MMA; cubit's
# table has no such row, and removing the drains (CUBIT_MMA_PTXAS) took the distance away. The row is added here: the
# script copies cubit's tables/dep_latency_sm120.json (the file next to TABLE), adds "an MMA holds every BRA 14 cycles
# off" to its resource rows and passes the copy as CUBIT_SCHED2_LAT; everything else is KERNEL=v43's environment.
# One control word differs from v43 (BAR.SYNC behind PV: S06 -> S13). Reproduces sha256 a163aea6d869c55e...
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
CUBIT=${CUBIT:-/workspace/cubit/target/release/cubit}
TABLE=${TABLE:-/workspace/cubit/tables/sm120.json.gz}
SCHEDULE=${SCHEDULE:-}
KERNEL=${KERNEL:-v41}
KEEP_ENV=()
MMA_BRA=
SRC_KERNEL=
case "$KERNEL" in
  v41) ;;
  v42) SCHEDULE=${SCHEDULE:-s1}
       [[ "$SCHEDULE" == s1 ]] || { echo "KERNEL=v42 is built with SCHEDULE=s1 only" >&2; exit 2; } ;;
  v43) SCHEDULE=${SCHEDULE:-s1}
       [[ "$SCHEDULE" == s1 ]] || { echo "KERNEL=v43 is built with SCHEDULE=s1 only" >&2; exit 2; }
       grep -q -a CUBIT_SB_LONG_KEEP "$CUBIT" || { echo "KERNEL=v43 needs a cubit with CUBIT_SB_LONG_KEEP (commit df4c557 or later): $CUBIT has none" >&2; exit 2; }
       KEEP_ENV=(CUBIT_SB_LONG_KEEP=1) ;;
  v43b) SCHEDULE=${SCHEDULE:-s1}
       [[ "$SCHEDULE" == s1 ]] || { echo "KERNEL=v43b is built with SCHEDULE=s1 only" >&2; exit 2; }
       grep -q -a CUBIT_SB_LONG_KEEP "$CUBIT" || { echo "KERNEL=v43b needs a cubit with CUBIT_SB_LONG_KEEP (commit df4c557 or later): $CUBIT has none" >&2; exit 2; }
       KEEP_ENV=(CUBIT_SB_LONG_KEEP=1); MMA_BRA=14; SRC_KERNEL=v43 ;;
  *) echo "unknown KERNEL '$KERNEL' (known: v41, v42, v43, v43b)" >&2; exit 2 ;;
esac
case "$SCHEDULE" in
  "") SCHED_ENV=(); DEFAULT_OUT=$REPO/kernels/cubins-sm120/sparse_mla_prefill_v41.cubin ;;
  s1) SCHED_ENV=(CUBIT_SCHED2=1 CUBIT_SCHED2_LAT=sm120 CUBIT_MMA_PTXAS=1)
      DEFAULT_OUT=$REPO/kernels/cubins-sm120/sparse_mla_prefill_v41s1.cubin ;;
  *) echo "unknown SCHEDULE '$SCHEDULE' (known: s1)" >&2; exit 2 ;;
esac
[[ "$KERNEL" == v42 || "$KERNEL" == v43 || "$KERNEL" == v43b ]] && DEFAULT_OUT=$REPO/kernels/cubins-sm120/sparse_mla_prefill_$KERNEL.cubin
OUT=${1:-$DEFAULT_OUT}
SRC_KERNEL=${SRC_KERNEL:-$KERNEL}
SASS=$REPO/kernels/sass/sparse_mla_prefill_$SRC_KERNEL.sass
if [[ -n "$MMA_BRA" ]]; then
  # cubit's dependency table + one resource row: a member of `mma` holds every BRA MMA_BRA issue cycles off
  DEP_TABLE=${DEP_TABLE:-$(dirname "$TABLE")/dep_latency_sm120.json}
  [[ -f "$DEP_TABLE" ]] || { echo "KERNEL=$KERNEL needs cubit's dependency table: $DEP_TABLE not found (set DEP_TABLE)" >&2; exit 2; }
  VARIANT=$(mktemp --suffix=.json); trap 'rm -f "$VARIANT"' EXIT
  python3 - "$DEP_TABLE" "$VARIANT" "$MMA_BRA" <<'PY'
import json
import sys

doc = json.load(open(sys.argv[1]))
mma = next(r for r in doc["resources"] if r["name"] == "mma")
assert "then" not in mma and not any(r["name"] == "bra" for r in doc["resources"]), "the table already has a row for MMA -> BRA"
mma["then"] = {"bra": int(sys.argv[3])}
doc["resources"].append({"name": "bra", "members": ["BRA"], "spacing": 1, "basis": "holder of the `then` row of mma"})
json.dump(doc, open(sys.argv[2], "w"))
PY
  SCHED_ENV=(CUBIT_SCHED2=1 "CUBIT_SCHED2_LAT=$VARIANT" CUBIT_MMA_PTXAS=1)
fi

python3 "$REPO/kernels/gen/gen_sparse_mla_prefill_$SRC_KERNEL.py" >/dev/null
echo "generated $SASS ($(grep -c ';' "$SASS") statements)"
env CUBIT_SB_SHARE=${CUBIT_SB_SHARE:-1} CUBIT_RB_EARLY=${CUBIT_RB_EARLY:-32} CUBIT_MMA_WAIT=1 CUBIT_MMA_WB_DRAIN=13 \
  ${SCHED_ENV[@]+"${SCHED_ENV[@]}"} ${KEEP_ENV[@]+"${KEEP_ENV[@]}"} \
  "$CUBIT" asm "$SASS" -t "$TABLE" -o "$OUT" --kernel sparse_mla_prefill_v41 \
  --mercury-stub "$REPO/kernels/sass/qmma_e4m3.merc.stub"
echo "wrote $OUT"
