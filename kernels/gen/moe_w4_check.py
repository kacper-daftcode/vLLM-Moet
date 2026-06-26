#!/usr/bin/env python3
"""Op-level validation of moe_w4_mm (full e2m1 planes, delta tier).

Same harness shape as moe_w2_check.py; B = random e2m1 nibbles packed
fragment-major as one 8-nibble WORD per (tile, k32) per lane:
  word nibbles 0-3 = codes k = ks*32 + 4t + j   (lo quad)
        nibbles 4-7 = codes k = ks*32 + 16 + 4t + j  (hi quad)
  16B per lane per k64, word order [t0a, t0b, t1a, t1b].

Env: CUBIN, K, N, E, M, RUNS.
"""
import ctypes
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/cubit/tools")
from culaunch import Cuda  # noqa: E402

CUBIN = os.environ["CUBIN"]
NWARP = int(os.environ.get("NWARP", "8"))
N = int(os.environ.get("N", "4096"))
K = int(os.environ.get("K", "4096"))
E = int(os.environ.get("E", "5"))
M = int(os.environ.get("M", "4"))
RUNS = int(os.environ.get("RUNS", "4"))
torch.manual_seed(int(os.environ.get("SEED", "3")))

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2)
E2M1[8:] *= -1


def pack_fp4_fragment_major(codes):
    """[N, K] u8 e2m1 codes (0..15) -> fragment-major nibble plane."""
    n, k = codes.shape
    c = codes.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    # dims: nb, tile, g, kb, k32, half(lo|hi), t, j
    # target word order per lane: [tile, k32] -> 4 words; nibble idx =
    # half*4 + j
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous()
    # now [nb, kb, g, t, tile, k32, half, j] -> pack 8 nibbles LE per word
    c = c.view(-1, 8).to(torch.int64)
    w = (c[:, 0] | (c[:, 1] << 4) | (c[:, 2] << 8) | (c[:, 3] << 12)
         | (c[:, 4] << 16) | (c[:, 5] << 20) | (c[:, 6] << 24)
         | (c[:, 7] << 28))
    return w.to(torch.int32).numpy().view(np.uint8)


def pack_scales(s):
    n, ks = s.shape
    return s.view(n // 16, 16, ks).transpose(1, 2).contiguous().flatten()


cu = Cuda()
fn = cu.load_kernel(CUBIN, "moe_w4_mm")

descs = np.zeros((E, 6), dtype=np.uint64)
refs, d_cs = [], []
for e in range(E):
    codes = torch.randint(0, 16, (N, K), dtype=torch.uint8)
    sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)
    a = torch.randn(M, K) * 0.5
    ab = a.view(M, K // 128, 128)
    a_s = (ab.abs().amax(-1).clamp_min(1e-10) / 448.0)
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(M, K)

    w_deq = E2M1[codes.long()] * torch.exp2(sexp.float() - 127.0).repeat_interleave(32, 1)
    ref = (a8.float() * a_s.float().repeat_interleave(128, 1)) @ w_deq.T
    refs.append(ref)

    d_a = cu.to_device(a8.view(torch.uint8).numpy())
    d_as = cu.to_device(a_s.float().numpy().astype(np.float32).view(np.uint8))
    d_b = cu.to_device(pack_fp4_fragment_major(codes))
    d_bs = cu.to_device(pack_scales(sexp).numpy())
    d_c = cu.alloc(M * N * 2)
    d_cs.append(d_c)
    descs[e] = [d_a.value, d_as.value, d_b.value, d_bs.value, d_c.value,
                np.uint64(M)]

d_desc = cu.to_device(descs.view(np.uint8))
args = [d_desc, ctypes.c_uint32(K), ctypes.c_uint32(K // 64),
        ctypes.c_uint32(N * 2), ctypes.c_uint32(K // 128)]

outs, worst = [], 0.0
for r in range(RUNS):
    for d_c in d_cs:
        cu.memset32(d_c, 0, M * N // 2)
    cu.launch(fn, (N // 16, E, 1), (NWARP * 32, 1, 1), args)
    cu.synchronize()
    blob = b""
    for e, d_c in enumerate(d_cs):
        raw = cu.from_device(d_c, M * N * 2, dtype=np.uint16).copy()
        blob += raw.tobytes()
        got = torch.from_numpy(raw.reshape(M, N).copy()).view(torch.bfloat16).float()
        rel = (got - refs[e]).abs().max().item() / refs[e].abs().max().item()
        worst = max(worst, rel)
    outs.append(blob)

ok = worst < 2.5e-2 and len(set(outs)) == 1
print(f"moe_w4 NWARP={NWARP} N={N} K={K} E={E} M={M}: worst_rel={worst:.3e} "
      f"distinct={len(set(outs))}")
print(f"RESULT: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
