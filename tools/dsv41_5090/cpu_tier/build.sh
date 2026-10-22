#!/usr/bin/env bash
# Build libcpu_tier.so = the host dispatcher (cpu_tier.cpp) + the b12x-exact expert kernel (../cpu_moe/cpu_moe_b12x.cpp).
# ARCH=znver5 on the host's GCC 15; the serving image's GCC takes znver4 (same AVX-512 BW/VNNI/BF16 subset).
set -euo pipefail
cd "$(dirname "$0")"
ARCH=${ARCH:-znver5}
CXX=${CXX:-g++}
OUT=${OUT:-libcpu_tier.so}
KERNEL_SRC=${KERNEL_SRC:-../cpu_moe/cpu_moe_b12x.cpp}
$CXX -O3 -std=c++17 -march=$ARCH -mavx512f -mavx512bw -mavx512vbmi -mavx512vnni -mavx512bf16 -fno-trapping-math \
  -ffp-contract=off -fPIC -shared -pthread -Wall -Wextra -Wno-unused-parameter -I. \
  -o "$OUT" cpu_tier.cpp "$KERNEL_SRC"
echo "built $OUT ($ARCH)"
