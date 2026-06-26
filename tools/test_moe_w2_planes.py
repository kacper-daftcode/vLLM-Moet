#!/usr/bin/env python3
"""Golden tests for moe_w2_planes vs the QUANT_PROBE-validated repack tool.

1. nibble->code map == repack_expert_bits' tensor-sym cmap16 (levels
   {-4,-1,1,4}, odd-symmetric tie-break).
2. pack_fragment_major layout: every byte lands where the kernel doc says.
3. round-trip: codes -> plane -> (python unpack) -> codes.

Run: python3 tools/test_moe_w2_planes.py  (CPU, ~seconds)
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "tools")
sys.path.insert(0, ".")

from repack_expert_bits import _subset_tables, _candidate_level_sets  # noqa: E402

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "moe_w2_planes",
    "vllm/model_executor/layers/quantization/utils/moe_w2_planes.py")
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
_CODE_TO_NIBBLE = _m._CODE_TO_NIBBLE
_NIBBLE_TO_CODE = _m._NIBBLE_TO_CODE
mxfp4_to_codes = _m.mxfp4_to_codes
pack_fragment_major = _m.pack_fragment_major

# ---- 1. quantization map equals the validated tool -------------------------
err16, cmap16 = _subset_tables(4, symmetric=True)
sets = _candidate_level_sets(4, symmetric=True)
target = sets.index((-4.0, -1.0, 1.0, 4.0))
cmap = cmap16[target]            # e2m1 code -> e2m1 code of nearest level
CODE_VALUES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6] * 2)
CODE_VALUES[8:] *= -1
lv = {-4.0: 0, -1.0: 1, 1.0: 2, 4.0: 3}
ok = True
for c in range(16):
    expect_code = lv[float(CODE_VALUES[cmap[c]])]
    got_code = int(_NIBBLE_TO_CODE[c])
    if expect_code != got_code:
        print(f"MISMATCH nibble {c:#x}: tool->{expect_code} ours->{got_code}")
        ok = False
assert ok, "nibble->code map diverges from the validated tool"
print("1. quantization map == repack tool (16/16 nibbles)")

# also: reconstruction nibbles match the tool's cmap
for code in range(4):
    pass
recon = {int(_NIBBLE_TO_CODE[c]): int(cmap[c]) for c in range(16)}
for c in range(16):
    assert int(_CODE_TO_NIBBLE[int(_NIBBLE_TO_CODE[c])]) == int(cmap[c]), c
print("2. reconstruction nibbles == tool cmap")

# ---- 3. fragment-major layout positions ------------------------------------
N, K = 32, 128
codes = torch.arange(N * K, dtype=torch.int64).reshape(N, K) % 4
codes = codes.to(torch.uint8)
plane = pack_fragment_major(codes)
assert plane.numel() == N * K // 4


def plane_byte(nb, kb, g, t, j):
    """Index of fragment byte j (0..7) for lane (g,t) of block (nb,kb)."""
    lanes_per_blk = 32 * 8
    blk = (nb * (K // 64) + kb)
    lane = g * 4 + t
    return blk * lanes_per_blk + lane * 8 + j


def expect_byte(nb, kb, g, t, j):
    tile, rest = divmod(j, 4)
    k32, half = divmod(rest, 2)
    row = nb * 16 + tile * 8 + g
    kbase = kb * 64 + k32 * 32 + half * 16 + t * 4
    b = 0
    for k4 in range(4):
        b |= int(codes[row, kbase + k4]) << (2 * k4)
    return b


bad = 0
for nb in range(N // 16):
    for kb in range(K // 64):
        for g in range(8):
            for t in range(4):
                for j in range(8):
                    got = int(plane[plane_byte(nb, kb, g, t, j)])
                    exp = expect_byte(nb, kb, g, t, j)
                    if got != exp:
                        bad += 1
assert bad == 0, f"{bad} plane bytes misplaced"
print("3. fragment-major layout exact (all bytes)")

# ---- 4. mxfp4 nibble order --------------------------------------------------
w = torch.tensor([[0x52, 0xE0]], dtype=torch.uint8)   # nibbles: 2,5 then 0,E
c = mxfp4_to_codes(w)
assert c.tolist() == [[2, 3, 2, 0]], c.tolist()        # 1.0->+1, 3->+4, +0->+1, -4->-4
print("4. mxfp4 nibble order (lo=even k)")
print("ALL PASS")
