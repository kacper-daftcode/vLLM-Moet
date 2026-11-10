#!/usr/bin/env bash
# Rebuild the `s1` schedule set of the live MoE GEMM kernels (kernels/cubins-sm120/s1/) from kernels/sass/ and compare
# every file with the committed one.
#
#   kernels/gen/build_moe_s1.sh              rebuild into a temporary directory, compare, one line per file and a verdict
#   kernels/gen/build_moe_s1.sh <out-dir>    the same, keeping the rebuilt files in <out-dir>
#
# The set is the complete `_a32` MoE GEMM family under the loader's file names, so that VLLM_MOE_W2_CUBIT_DIR can point
# at it (see kernels/MANIFEST.md, "Dependency-aware schedule set"):
#   * a file that differs from the pinned cubin of the same name (kernels/cubins-sm120/<name>) is the dependency-aware
#     stall schedule of the same source: the source's policy environment (MANIFEST: CUBIT_RB_CHAIN=1 CUBIT_SB_SHARE=1
#     for the MC4 K >= 4096 sources, nothing for the others) plus
#         CUBIT_SCHED2=1 CUBIT_SCHED2_LAT=sm120 CUBIT_MMA_PTXAS=1
#     -- the instructions of the source, other control words, no MMA writeback drains;
#   * a file that equals the pinned cubin is that kernel's pinned schedule (the new one was not faster there).
# Requirements: CUBIT -- cubit `main` at or after 49f8340 (the schedule switches); TABLE -- tables/sm120.json(.gz) of the
# same tree. The committed files were built with cubit ef8ff88 and reproduce byte for byte at ef7d335.
# Exit status 0 when every file reproduces and SHA256SUMS matches.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
K=$(cd "$HERE/.." && pwd)
CUBIT=${CUBIT:-/workspace/cubit/target/release/cubit}
TABLE=${TABLE:-/workspace/cubit/tables/sm120.json.gz}
PIN=$K/cubins-sm120
SET=$PIN/s1
OUT=${1:-$(mktemp -d)}
mkdir -p "$OUT"
bad=0; new=0; same=0
for f in "$SET"/*.cubin; do
  name=$(basename "$f"); stem=${name%.cubin}
  if cmp -s "$f" "$PIN/$name"; then
    cp "$PIN/$name" "$OUT/$name"; same=$((same + 1))
    echo "$name: the pinned schedule"
    continue
  fi
  case "$stem" in
    moe_w4q_mm_*) kernel=moe_w4q_mm ;;
    moe_w4_mm_*)  kernel=moe_w4_mm ;;
    *)            kernel=moe_w2_mm ;;
  esac
  policy=()
  case "$stem" in
    moe_w2_mm_mc4_k4096_a32|moe_w2_mm_mc4_k6144_a32|moe_w2_mm_mc4_k7168_a32) policy=(CUBIT_RB_CHAIN=1 CUBIT_SB_SHARE=1) ;;
  esac
  env ${policy[@]+"${policy[@]}"} CUBIT_SCHED2=1 CUBIT_SCHED2_LAT=sm120 CUBIT_MMA_PTXAS=1 \
    "$CUBIT" asm "$K/sass/$stem.sass" -t "$TABLE" -o "$OUT/$name" --kernel "$kernel" \
    --mercury-stub "$K/sass/qmma_e4m3.merc.stub" > "$OUT/$stem.log" 2>&1 || { echo "$name: ASSEMBLY FAILED (see $OUT/$stem.log)"; bad=$((bad + 1)); continue; }
  new=$((new + 1))
  if cmp -s "$OUT/$name" "$f"; then echo "$name: s1, reproduces $(sha256sum "$f" | cut -c1-16)"; else echo "$name: s1, DIFFERS from the committed file"; bad=$((bad + 1)); fi
done
(cd "$SET" && sha256sum -c --quiet SHA256SUMS) || { echo "SHA256SUMS does not match the committed files"; bad=$((bad + 1)); }
[[ $(ls "$SET"/*.cubin | wc -l) -eq $(grep -c . "$SET/SHA256SUMS") ]] || { echo "SHA256SUMS does not list every file of the set"; bad=$((bad + 1)); }
echo "s1 set: $new re-scheduled + $same pinned files; $bad problem(s); rebuilt into $OUT"
[[ $bad -eq 0 ]]
