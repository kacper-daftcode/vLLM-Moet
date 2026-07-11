#!/usr/bin/env python3
"""Op-level validation of moe_w4s_mm (SPLIT FP4: resident 2-bit base +
2-bit refinement plane = FP4 quality at half the delta-slot bytes).

Random e2m1 nibbles -> base codes (NIBBLE_TO_CODE) + refinement codes
(nested encoder; mag 0 merges into 0.5). Both planes pack with the w2
fragment-major 2-bit packer. Reference dequant uses the DECODER's value
table (so the kernel must match to bf16 rounding); the true-FP4 delta of
the merge is reported separately.

Desc ABI (64B/pair): {a, as, base, ref, bs, c, m_rows, pad}.
Env: CUBIN, K, N, E, M, RUNS.
"""
import ctypes
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/cubit/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
NIBBLE_TO_CODE = torch.tensor([2] * 5 + [3] * 3 + [1] * 5 + [0] * 3,
                              dtype=torch.uint8)
# nested encoder: mag idx -> ref (mag 0 merged into 0.5)
MAG_TO_REF = torch.tensor([0, 0, 1, 2, 3, 0, 1, 2], dtype=torch.uint8)
# decoder value tables (what the kernel reconstructs)
SMALL_VAL = torch.tensor([.5, 1.0, 1.5, 2.0])
BIG_VAL = torch.tensor([3.0, 4.0, 6.0, 6.0])


def nibbles_to_split(nibs):
    """[N,K] e2m1 nibbles -> (base codes, ref codes, decoded values)."""
    mag = (nibs & 7).long()
    code = NIBBLE_TO_CODE[nibs.long()]
    ref = MAG_TO_REF[mag]
    big = (code == 0) | (code == 3)
    val = torch.where(big, BIG_VAL[ref.long()], SMALL_VAL[ref.long()])
    val = torch.where((code <= 1), -val, val)
    return code, ref, val


def pack_fragment_major(codes):
    n, k = codes.shape
    c = codes.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 4).to(torch.int32)
    return (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).to(torch.uint8)


def pack_scales(s):
    n, ks = s.shape
    return s.view(n // 16, 16, ks).transpose(1, 2).contiguous().flatten()


cu = Cuda()
fn = cu.load_kernel(CUBIN, "moe_w4s_mm")

descs = np.zeros((E, 8), dtype=np.uint64)
refs, d_cs = [], []
merge_delta = 0.0
for e in range(E):
    nibs = torch.randint(0, 16, (N, K), dtype=torch.uint8)
    sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)
    a = torch.randn(M, K) * 0.5
    ab = a.view(M, K // 128, 128)
    a_s = (ab.abs().amax(-1).clamp_min(1e-10) / 448.0)
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(M, K)

    code, ref, val = nibbles_to_split(nibs)
    scale = torch.exp2(sexp.float() - 127.0).repeat_interleave(32, 1)
    w_deq = val * scale                       # decoder's values (merge incl.)
    w_true = E2M1[nibs.long()] * scale        # true FP4
    merge_delta = max(merge_delta,
                      (w_deq - w_true).abs().max().item())
    a_full = a8.float() * a_s.float().repeat_interleave(128, 1)
    refs.append(a_full @ w_deq.T)

    d_a = cu.to_device(a8.view(torch.uint8).numpy())
    d_as = cu.to_device(a_s.float().numpy().astype(np.float32).view(np.uint8))
    d_base = cu.to_device(pack_fragment_major(code).numpy())
    d_ref = cu.to_device(pack_fragment_major(ref).numpy())
    d_bs = cu.to_device(pack_scales(sexp).numpy())
    d_c = cu.alloc(M * N * 2)
    d_cs.append(d_c)
    descs[e] = [d_a.value, d_as.value, d_base.value, d_ref.value, d_bs.value,
                d_c.value, np.uint64(M), 0]

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
print(f"moe_w4s NWARP={NWARP} N={N} K={K} E={E} M={M}: worst_rel={worst:.3e} "
      f"distinct={len(set(outs))} (merge |w-w_fp4| max {merge_delta:.3f} "
      f"unit-scaled)")
print(f"RESULT: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
