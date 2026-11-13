#!/usr/bin/env python3
"""The operand layouts of the three block-scaled QMMAs the Inkling2 expert GEMM uses on sm_120, established by
experiment: one instruction per case with every lane's registers built by this script from full 16 x K and K x 8
operands, the result compared with exact arithmetic.

    python3 test_qmma_layout_sm120.py            # one sm_120 GPU; exit 0 = the layouts the kernel assumes hold

Layouts under test (lane l, q = l / 4, c = l % 4):
  A values   a0 / a2: row q, bytes 4c.. / 16 + 4c.. of the row's 32 bytes; a1 / a3: row q + 8
  B values   b0 / b1: column q, bytes 4c.. / 16 + 4c..
  A scales   even lanes of a quad: row q; odd lanes: row q + 8  (alternatives are tried and must fail)
  B scales   every lane of a quad: column q
  D          d0, d1: (row q, columns 2c, 2c + 1); d2, d3: (row q + 8, the same columns)
With power-of-two scales every product and sum is exact in fp32, so the comparison is bitwise.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import inkling2_bmm_sm120 as m  # noqa: E402

dev = torch.device("cuda")
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)


def words(row_bytes: torch.Tensor) -> torch.Tensor:
    """[..., 4X] uint8 -> [..., X] little-endian int64 words"""
    b = row_bytes.reshape(*row_bytes.shape[:-1], -1, 4).to(torch.int64)
    return b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)


def to_i32(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32)


def run(kind: int, a_bytes, b_bytes, sfa_rows, sfb_cols, a_scale_rule: str):
    """a_bytes [16, 32] uint8, b_bytes [8, 32] uint8 (column n's K bytes), sfa_rows [16] int64 scale words,
    sfb_cols [8]; returns D [16, 8] float32"""
    lanes = torch.arange(32)
    q, c = lanes // 4, lanes % 4
    aw, bw = words(a_bytes), words(b_bytes)          # [16, 8], [8, 8] words of 4 bytes
    a = torch.stack((aw[q, c], aw[q + 8, c], aw[q, 4 + c], aw[q + 8, 4 + c]), dim=1)
    b = torch.stack((bw[q, c], bw[q, 4 + c]), dim=1)
    if a_scale_rule == "parity":      # the layout the kernel assumes
        rs = torch.where(c % 2 == 1, q + 8, q)
    elif a_scale_rule == "upper":     # lanes 2, 3 of a quad carry row q + 8
        rs = torch.where(c >= 2, q + 8, q)
    elif a_scale_rule == "low":       # every lane carries row q
        rs = q
    else:
        raise ValueError(a_scale_rule)
    sfa, sfb = sfa_rows[rs], sfb_cols[q]
    d = torch.empty((32, 4), dtype=torch.float32, device=dev)
    m._ext().qmma_probe(kind, to_i32(a).to(dev).contiguous(), to_i32(b).to(dev).contiguous(),
                        to_i32(sfa).to(dev).contiguous(), to_i32(sfb).to(dev).contiguous(), d)
    torch.cuda.synchronize()
    d = d.cpu()
    out = torch.empty((16, 8), dtype=torch.float32)
    out[q, 2 * c], out[q, 2 * c + 1] = d[:, 0], d[:, 1]
    out[q + 8, 2 * c], out[q + 8, 2 * c + 1] = d[:, 2], d[:, 3]
    return out


def case(kind: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    if kind == m.NVFP4:
        a_bytes = torch.randint(0, 256, (16, 32), generator=g, dtype=torch.uint8)
        b_bytes = torch.randint(0, 256, (8, 32), generator=g, dtype=torch.uint8)
        a_val = m.e2m1_values(a_bytes)                                   # [16, 64]
        b_val = m.e2m1_values(b_bytes)                                   # [8, 64]
        ja = torch.randint(-2, 3, (16, 4), generator=g)                  # 2^j per 16 k: every sum exact in fp32
        jb = torch.randint(-2, 3, (8, 4), generator=g)
        code_a, code_b = (ja + 7) << 3, (jb + 7) << 3                    # e4m3 codes of 2^j
        sfa = code_a[:, 0] | (code_a[:, 1] << 8) | (code_a[:, 2] << 16) | (code_a[:, 3] << 24)
        sfb = code_b[:, 0] | (code_b[:, 1] << 8) | (code_b[:, 2] << 16) | (code_b[:, 3] << 24)
        sa = (2.0 ** ja.to(torch.float64)).repeat_interleave(16, dim=1)
        sb = (2.0 ** jb.to(torch.float64)).repeat_interleave(16, dim=1)
    else:
        # e4m3 codes of the binades [1, 4) (multiples of 1/8): with scales 2^-1 .. 2^1 every sum is exact in fp32
        codes = torch.tensor([c for c in range(256) if (c >> 3) & 0xF in (7, 8)], dtype=torch.uint8)
        a_bytes = codes[torch.randint(0, len(codes), (16, 32), generator=g)]
        a_val = m.e4m3_values(a_bytes)
        if kind == m.MXFP8:
            b_bytes = codes[torch.randint(0, len(codes), (8, 32), generator=g)]
            b_val = m.e4m3_values(b_bytes)
        else:
            b_bytes = torch.randint(0, 64, (8, 32), generator=g, dtype=torch.uint8)   # e2m3 in byte containers
            s, ee, mmm = b_bytes >> 5, (b_bytes >> 3) & 3, b_bytes & 7
            mag = torch.where(ee == 0, mmm / 8.0, (1 + mmm / 8.0) * 2.0 ** (ee.to(torch.float64) - 1).clamp(min=0))
            b_val = torch.where(s == 1, -mag, mag).to(torch.float64)
        ja = torch.randint(-1, 2, (16,), generator=g)
        jb = torch.randint(-1, 2, (8,), generator=g)
        sfa, sfb = (127 + ja).to(torch.int64), (127 + jb).to(torch.int64)
        sa = (2.0 ** ja.to(torch.float64))[:, None].expand(16, 32)
        sb = (2.0 ** jb.to(torch.float64))[:, None].expand(8, 32)
    ref = ((a_val * sa) @ (b_val * sb).T).to(torch.float32)
    return a_bytes, b_bytes, sfa.to(torch.int64), sfb.to(torch.int64), ref


def main() -> int:
    assert m.supported(), "needs an sm_120 / sm_121 GPU"
    names = {m.NVFP4: "nvfp4 (mxf4nvf4, 4X, k64)", m.MXFP8: "mxfp8 (mxf8f6f4, 1X, k32)", m.MXFP6: "mxfp6 (e4m3 x e2m3)"}
    bad = 0
    for kind in (m.NVFP4, m.MXFP8, m.MXFP6):
        res = {}
        for rule in ("parity", "upper", "low"):
            eq = tot = 0
            for seed in range(20):
                a_bytes, b_bytes, sfa, sfb, ref = case(kind, 1000 * kind + seed)
                out = run(kind, a_bytes, b_bytes, sfa, sfb, rule)
                eq += int((out == ref).sum())
                tot += out.numel()
            res[rule] = (eq, tot)
        ok = res["parity"][0] == res["parity"][1] and res["upper"][0] < res["upper"][1] and res["low"][0] < res["low"][1]
        bad += not ok
        print(f"{names[kind]}: " + ", ".join(f"A scales by {r}: {e}/{t} identical" for r, (e, t) in res.items())
              + ("  OK" if ok else "  FAILED"))
    print("QMMA LAYOUT " + ("OK" if not bad else "FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
