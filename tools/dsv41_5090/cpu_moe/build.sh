#!/usr/bin/env bash
# Build libcpu_moe_mxfp4.so (AVX-512 BW/VBMI/BF16; -march=znver5 for the Threadripper PRO 9975WX).
set -euo pipefail
cd "$(dirname "$0")"
ARCH=${ARCH:-znver5}
CXX=${CXX:-g++}
$CXX -O3 -std=c++17 -march=$ARCH -mavx512f -mavx512bw -mavx512vbmi -mavx512bf16 -fno-trapping-math \
  -fPIC -shared -pthread -Wall -Wextra -Wno-unused-parameter \
  -o libcpu_moe_mxfp4.so cpu_moe_mxfp4.cpp
echo "built $(pwd)/libcpu_moe_mxfp4.so ($ARCH)"
