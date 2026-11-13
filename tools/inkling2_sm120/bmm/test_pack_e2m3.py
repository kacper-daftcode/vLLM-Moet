#!/usr/bin/env python3
"""CPU test of the MXFP8 -> e2m3 packing (inkling2_bmm_sm120.pack_e2m3) and of the prepared weight layout.

    python3 test_pack_e2m3.py [--model-dir DIR]        # no GPU needed

  1. every e4m3 code as a block maximum, with every e4m3 code beside it: the packed value times the moved block scale
     equals the original exactly when the value lies in the block's top three binades, and is within half a step of
     e2m3's subnormal grid (1/64 of the block's top binade's lower end) otherwise - checked against an independent
     decoder (a table of the 64 e2m3 codes);
  2. random blocks: the same bound, the block scale moves by one power of two per block, an all-zero block stays zero;
  3. prepare / unprepare are inverse byte permutations for every format's step size;
  4. with --model-dir: the same on real MXFP8 expert weights (the share kept exactly, error / weight energy).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import inkling2_bmm_sm120 as m  # noqa: E402

E2M3_TABLE = torch.tensor(
    [(-1.0 if c >> 5 else 1.0) * ((c & 7) / 8 if (c >> 3) & 3 == 0 else (1 + (c & 7) / 8) * 2.0 ** (((c >> 3) & 3) - 1))
     for c in range(64)], dtype=torch.float64)
CODES = torch.tensor([c for c in range(256) if c & 0x7F != 0x7F], dtype=torch.uint8)


def table_decode(packed: torch.Tensor) -> torch.Tensor:
    """independent decoder: bit i * 6 of each 12-byte group, through the 64-entry table"""
    b = packed.reshape(-1, 12).to(torch.int64)
    out = torch.empty((b.shape[0], 16), dtype=torch.float64)
    for i in range(16):
        bit = 6 * i
        byte, sh = bit // 8, bit % 8
        v = (b[:, byte] >> sh) | (b[:, byte + 1] << (8 - sh) if byte + 1 < 12 else 0)
        out[:, i] = E2M3_TABLE[v & 0x3F]
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 4 // 3)


def check_blocks(w: torch.Tensor, label: str) -> bool:
    """w: e4m3 codes [B, 32]"""
    packed, delta = m.pack_e2m3(w)
    v6 = m.e2m3_values(packed)
    assert torch.equal(v6, table_decode(packed)), "the two decoders disagree"
    orig = m.e4m3_values(w)
    delta = delta.reshape(-1)
    rec = v6 * (2.0 ** delta.to(torch.float64))[:, None]
    top = orig.abs().amax(dim=1, keepdim=True)
    binade_low = 2.0 ** torch.floor(torch.log2(top.clamp(min=2.0**-30)))   # lower end of the block's top binade
    exact_zone = orig.abs() >= binade_low / 4                              # the top three binades
    err = (rec - orig).abs()
    ok_exact = bool((err[exact_zone.expand_as(err)] == 0).all())
    ok_bound = bool((err <= binade_low / 64 + 0).all())                    # half a subnormal step = top / 64
    ok_sign = bool(((rec == 0) | (torch.sign(rec) == torch.sign(orig))).all())
    zero_blocks = (top.squeeze(1) == 0)
    ok_zero = bool((rec[zero_blocks] == 0).all()) and bool((delta[zero_blocks] == 0).all())
    ok = ok_exact and ok_bound and ok_sign and ok_zero
    snr = 10 * torch.log10((orig**2).sum() / (err**2).sum().clamp(min=1e-300))
    print(f"  {label}: {w.shape[0]} blocks; exact {100 * float((err == 0).double().mean()):.2f} % of the values "
          f"(top three binades all exact: {ok_exact}); max error / top binade {float((err / binade_low).max()):.5f} "
          f"(bound 1/64 = 0.01563: {ok_bound}); error / energy {-float(snr):.1f} dB; deltas {int(delta.min())}..{int(delta.max())}"
          f"  {'OK' if ok else 'FAILED'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir")
    a = ap.parse_args()
    ok = True
    # 1. every code as the block's maximum-magnitude member next to every other code
    blocks = []
    for top in CODES.tolist():
        row = CODES[torch.randperm(len(CODES))[:31]].clone()
        blocks.append(torch.cat((torch.tensor([top], dtype=torch.uint8), row)))
    grid = torch.stack(blocks)
    pair = torch.stack(torch.meshgrid(CODES, CODES, indexing="ij"), dim=-1).reshape(-1, 2)
    pair = torch.cat((pair, torch.zeros((pair.shape[0], 30), dtype=torch.uint8)), dim=1)
    ok &= check_blocks(grid, "every code with 31 random codes")
    ok &= check_blocks(pair, "every pair of codes (30 zeros)")
    g = torch.Generator().manual_seed(0)
    ok &= check_blocks(CODES[torch.randint(0, len(CODES), (20000, 32), generator=g)], "random codes")
    z = CODES[torch.randint(0, len(CODES), (64, 32), generator=g)]
    z[::2] = 0
    z[1::4] = 0x80
    ok &= check_blocks(z, "zero and minus-zero blocks")
    # 3. the layout permutation
    for bps in (32, 64, 48, 4, 2):
        x = torch.randint(0, 256, (3, 256, bps * 9), generator=g, dtype=torch.uint8)
        y = m.prepare(x, bps)
        same = torch.equal(m.unprepare(y, bps), x)
        # row r, step s of block b sits at [b][s][r % 128]
        probe = torch.equal(y.reshape(3, 2, 9, 128, bps)[1, 1, 4, 77], x.reshape(3, 256, 9, bps)[1, 128 + 77, 4])
        ok &= same and probe
        print(f"  prepare / unprepare, {bps} bytes per step: inverse {same}, element check {probe}")
    if a.model_dir:
        idx = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))["weight_map"]
        for layer in range(40, 48):
            name = f"model.llm.layers.{layer}.mlp.experts.w13_weight"
            path = os.path.join(a.model_dir, idx[name])
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                hl = struct.unpack("<Q", f.read(8))[0]
                t = json.loads(f.read(hl))[name]
                f.seek(8 + hl + t["data_offsets"][0] + 5 * t["shape"][1] * t["shape"][2])
                buf = f.read(2 * t["shape"][1] * t["shape"][2])
            w = torch.frombuffer(bytearray(buf), dtype=torch.uint8).reshape(-1, 32)
            ok &= check_blocks(w, f"real weights, layer {layer} w13, experts 5-6")
            break
    print("PACK E2M3 " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
