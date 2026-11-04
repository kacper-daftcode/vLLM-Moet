#!/usr/bin/env bash
# Build libengram_v2.so = the Engram v2 host fetcher (engram_v2.cpp: the DRAM row cache + io_uring NVMe reads +
# the mailbox protocol). No AVX-512 / arch flags needed (memcpy + syscalls); -O2 generic.
set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-g++}
OUT=${OUT:-libengram_v2.so}
$CXX -O2 -std=c++17 -fPIC -shared -pthread -Wall -Wextra -Wno-unused-parameter -I. -o "$OUT" engram_v2.cpp
echo "built $OUT"
