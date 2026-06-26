#!/usr/bin/env python3
"""Op-level validation of moe_w2_mm: 2-bit planes + block-32 UE8M0 vs f32 ref.

Builds random codes/scales for E "experts", random fp8 A per (expert,
token-group) pair, runs ONE launch with a desc table covering all pairs,
checks every pair's bf16 output against the dequantized f32 reference.
Multi-run determinism included (address-WAR regression guard).

Env: CUBIN, K (=4096|2048), N (=4096), E (pairs), M (<=4), RUNS.
Run inside cubit-dev: CUDA_VISIBLE_DEVICES=0 python3 tools/moe_w2_check.py
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
torch.manual_seed(int(os.environ.get("SEED", "7")))

LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0])


def pack_fragment_major(codes):
    n, k = codes.shape
    c = codes.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 4).to(torch.int32)
    return (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).to(torch.uint8)


def pack_scales(s):
    n, ks = s.shape
    return s.view(n // 16, 16, ks).transpose(1, 2).contiguous().flatten()


cu = Cuda()
fn = cu.load_kernel(CUBIN, "moe_w2_mm")

descs = np.zeros((E, 6), dtype=np.uint64)
refs, d_cs = [], []
for e in range(E):
    codes = torch.randint(0, 4, (N, K), dtype=torch.uint8)
    sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)  # e8m0 around 1.0
    a = torch.randn(M, K) * 0.5
    ab = a.view(M, K // 128, 128)
    a_s = (ab.abs().amax(-1).clamp_min(1e-10) / 448.0)
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(M, K)

    w_deq = LEVELS[codes.long()] * torch.exp2(sexp.float() - 127.0).repeat_interleave(32, 1)
    ref = (a8.float() * a_s.float().repeat_interleave(128, 1)) @ w_deq.T
    refs.append(ref)

    d_a = cu.to_device(a8.view(torch.uint8).numpy())
    d_as = cu.to_device(a_s.float().numpy().astype(np.float32).view(np.uint8))
    d_b = cu.to_device(pack_fragment_major(codes).numpy())
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
print(f"moe_w2 NWARP={NWARP} N={N} K={K} E={E} M={M}: worst_rel={worst:.3e} "
      f"distinct={len(set(outs))}")
print(f"RESULT: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
