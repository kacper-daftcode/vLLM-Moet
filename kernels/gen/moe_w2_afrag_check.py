#!/usr/bin/env python3
"""AFRAG-vs-MC4 bit-exactness check for moe_w2_mm at one K.

Same harness as moe_w2_check.py, but runs BOTH the mc4 cubin (row-major A)
and the mc4afrag cubin (fragment-major A) on identical planes/scales/inputs
and requires bit-identical bf16 outputs plus <=2.5e-2 rel vs the f32
reference (the AFRAG contract: same QMMA inputs, different A load order).

Env: CUBIN_MC4, CUBIN_AFRAG, K, N, E, RUNS.
"""
import ctypes
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from culaunch import Cuda  # noqa: E402

CUBIN_MC4 = os.environ["CUBIN_MC4"]
CUBIN_AFRAG = os.environ["CUBIN_AFRAG"]
NWARP = int(os.environ.get("NWARP", "8"))
N = int(os.environ.get("N", "4096"))
K = int(os.environ.get("K", "7168"))
E = int(os.environ.get("E", "5"))
M = 16                                   # AFRAG = full m16 tile
RUNS = int(os.environ.get("RUNS", "3"))
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


def pack_a_fragment_major(a8: torch.Tensor) -> torch.Tensor:
    """[16, K] fp8 rows -> AFRAG layout (mirrors moe_w2_cubit._to_fragment_major
    for a single 16-row tile): [g2, g, j, quad, t, b] -> [j, g, t, quad, g2, b]."""
    v = a8.view(torch.uint8).view(2, 8, K // 64, 4, 4, 4)
    v = v.permute(2, 1, 4, 3, 0, 5).reshape(16, K)
    return v.contiguous()


cu = Cuda()
fn_mc4 = cu.load_kernel(CUBIN_MC4, "moe_w2_mm")
fn_afr = cu.load_kernel(CUBIN_AFRAG, "moe_w2_mm")

descs_mc4 = np.zeros((E, 6), dtype=np.uint64)
descs_afr = np.zeros((E, 6), dtype=np.uint64)
refs, d_cs_mc4, d_cs_afr = [], [], []
for e in range(E):
    codes = torch.randint(0, 4, (N, K), dtype=torch.uint8)
    sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)
    a = torch.randn(M, K) * 0.5
    ab = a.view(M, K // 128, 128)
    a_s = (ab.abs().amax(-1).clamp_min(1e-10) / 448.0)
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(M, K)

    w_deq = LEVELS[codes.long()] * torch.exp2(sexp.float() - 127.0).repeat_interleave(32, 1)
    refs.append((a8.float() * a_s.float().repeat_interleave(128, 1)) @ w_deq.T)

    d_b = cu.to_device(pack_fragment_major(codes).numpy())
    d_bs = cu.to_device(pack_scales(sexp).numpy())
    d_as = cu.to_device(a_s.float().numpy().astype(np.float32).view(np.uint8))
    d_a_row = cu.to_device(a8.view(torch.uint8).numpy())
    d_a_frg = cu.to_device(pack_a_fragment_major(a8).numpy())
    for descs, d_a, d_cs in ((descs_mc4, d_a_row, d_cs_mc4),
                             (descs_afr, d_a_frg, d_cs_afr)):
        d_c = cu.alloc(M * N * 2)
        d_cs.append(d_c)
        descs[e] = [d_a.value, d_as.value, d_b.value, d_bs.value, d_c.value,
                    np.uint64(M)]

args_of = {id(descs_mc4): cu.to_device(descs_mc4.view(np.uint8)),
           id(descs_afr): cu.to_device(descs_afr.view(np.uint8))}

worst = 0.0
mismatch = 0
for r in range(RUNS):
    outs = {}
    for name, fn, descs, d_cs in (("mc4", fn_mc4, descs_mc4, d_cs_mc4),
                                  ("afrag", fn_afr, descs_afr, d_cs_afr)):
        for d_c in d_cs:
            cu.memset32(d_c, 0, M * N // 2)
        args = [args_of[id(descs)], ctypes.c_uint32(K),
                ctypes.c_uint32(K // 64), ctypes.c_uint32(N * 2),
                ctypes.c_uint32(K // 128)]
        cu.launch(fn, (N // 16, E, 1), (NWARP * 32, 1, 1), args)
        cu.synchronize()
        outs[name] = [cu.from_device(d_c, M * N * 2, dtype=np.uint16).copy()
                      for d_c in d_cs]
    for e in range(E):
        if not np.array_equal(outs["mc4"][e], outs["afrag"][e]):
            mismatch += 1
        got = torch.from_numpy(outs["afrag"][e].reshape(M, N).copy()) \
            .view(torch.bfloat16).float()
        rel = (got - refs[e]).abs().max().item() / refs[e].abs().max().item()
        worst = max(worst, rel)

ok = worst < 2.5e-2 and mismatch == 0
print(f"moe_w2 AFRAG K={K} N={N} E={E}: worst_rel={worst:.3e} "
      f"bit-mismatch pairs={mismatch}/{E * RUNS}")
print(f"RESULT: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
