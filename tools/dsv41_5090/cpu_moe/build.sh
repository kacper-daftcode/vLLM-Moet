#!/usr/bin/env bash
# Build libcpu_moe_mxfp4.so (v1: the DeepGEMM line's numerics, pack layout) and libcpu_moe_b12x.so (v2: b12x's
# compact W4A8 numerics, the checkpoint layout). AVX-512 BW/VBMI/VNNI/BF16; -march=znver5 for the TR PRO 9975WX.
set -euo pipefail
cd "$(dirname "$0")"
ARCH=${ARCH:-znver5}
CXX=${CXX:-g++}
FLAGS="-O3 -std=c++17 -march=$ARCH -mavx512f -mavx512bw -mavx512vbmi -mavx512vnni -mavx512bf16 -fno-trapping-math \
  -ffp-contract=off -fPIC -shared -pthread -Wall -Wextra -Wno-unused-parameter"
$CXX $FLAGS -o libcpu_moe_mxfp4.so cpu_moe_mxfp4.cpp
$CXX $FLAGS -o libcpu_moe_b12x.so cpu_moe_b12x.cpp
echo "built $(pwd)/libcpu_moe_mxfp4.so and libcpu_moe_b12x.so ($ARCH)"
