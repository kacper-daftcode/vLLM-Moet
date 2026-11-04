"""Engram rows of DeepSeek-V4.1-Flash: statistics and the trellis-4 (trellis-kv) re-quantization error.

    python engram_rows_probe.py [--layer 1] [--rows 4096] [--blocks 64] [--trellis-kv /mnt/nvme/qwen256k/trellis-kv]

Samples `blocks` runs of `rows/blocks` consecutive rows at random positions of the layer's table (fp8 e4m3
[N, 256] + e8m0 scale per 32 [N, 8]), dequantizes them exactly, and reports: row RMS distribution, zero
fraction, kurtosis of the raw and FWHT-rotated unit-RMS rows (how Gaussian the trellis codebook's input is),
and the relative RMS error of trellis-4 (L = 12, Lloyd codebook `lloyd_kv12a` trained on Qwen K/V rows) next
to int4 + Hadamard g32 / g256 and NVFP4 / MXFP4 fake-quantizers from trellis_kv.baselines — the same
yardstick as the trellis-kv README's CPU table. The checkpoint's own fp8 record has no reference here (the
bf16 original is not in the release), so the numbers are errors relative to the fp8 rows.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

MODEL_DIR = os.environ.get("DSV41_MODEL_DIR", "/mnt/nvme/models/DeepSeek-V4.1-Flash")
HEAD_DIM = 256
SCALE_GROUP = 32


def e4m3_to_f32_table() -> np.ndarray:
    t = np.zeros(256, dtype=np.float32)
    for c in range(256):
        s, E, m = c >> 7, (c >> 3) & 0xF, c & 7
        if E == 15 and m == 7:
            v = np.nan
        elif E == 0:
            v = m * 2.0 ** -9
        else:
            v = (1 + m / 8.0) * 2.0 ** (E - 7)
        t[c] = -v if s else v
    return t


def tensor_location(name: str) -> tuple[str, int, int, list[int]]:
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][name]
    path = os.path.join(MODEL_DIR, shard)
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    ent = hdr[name]
    base = 8 + n + ent["data_offsets"][0]
    return path, base, ent["data_offsets"][1] - ent["data_offsets"][0], ent["shape"]


def sample_rows(layer: int, rows: int, blocks: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    wpath, wbase, _, wshape = tensor_location(f"layers.{layer}.engram.embed.weight")
    spath, sbase, _, sshape = tensor_location(f"layers.{layer}.engram.embed.scale")
    n_rows = wshape[0]
    per = rows // blocks
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.integers(0, n_rows - per, size=blocks))
    codes = np.empty((blocks * per, HEAD_DIM), dtype=np.uint8)
    scales = np.empty((blocks * per, HEAD_DIM // SCALE_GROUP), dtype=np.uint8)
    with open(wpath, "rb") as fw, open(spath, "rb") as fs:
        for i, r0 in enumerate(starts):
            fw.seek(wbase + int(r0) * HEAD_DIM)
            codes[i * per : (i + 1) * per] = np.frombuffer(fw.read(per * HEAD_DIM), dtype=np.uint8).reshape(per, HEAD_DIM)
            fs.seek(sbase + int(r0) * (HEAD_DIM // SCALE_GROUP))
            scales[i * per : (i + 1) * per] = np.frombuffer(fs.read(per * (HEAD_DIM // SCALE_GROUP)), dtype=np.uint8).reshape(
                per, HEAD_DIM // SCALE_GROUP)
    return codes, scales


def dequant(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    vals = e4m3_to_f32_table()[codes]
    sc = np.ldexp(np.float32(1.0), scales.astype(np.int32) - 127).astype(np.float32)
    return vals * np.repeat(sc, SCALE_GROUP, axis=1)


def kurtosis(x: np.ndarray) -> float:
    x = x.astype(np.float64).ravel()
    x = x - x.mean()
    return float((x**4).mean() / (x**2).mean() ** 2)


def rel_rms(xhat: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(xhat - x) / np.linalg.norm(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trellis-kv", default="/mnt/nvme/qwen256k/trellis-kv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    sys.path.insert(0, args.trellis_kv)
    from trellis_kv import baselines as B  # noqa: E402
    from trellis_kv import reference as R  # noqa: E402

    t0 = time.time()
    codes, scales = sample_rows(args.layer, args.rows, args.blocks, args.seed)
    x = dequant(codes, scales)
    print(f"layer {args.layer}: {x.shape[0]} rows sampled in {time.time() - t0:.1f}s")
    row_rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=1))
    zero_rows = int((row_rms == 0).sum())
    print(f"  row RMS: median {np.median(row_rms):.4f}  p5 {np.percentile(row_rms, 5):.4f}  p95 {np.percentile(row_rms, 95):.4f}  "
          f"max {row_rms.max():.4f}; zero rows {zero_rows}; zero elements {(x == 0).mean() * 100:.2f} %")
    print(f"  scale bytes: min {scales.min()} max {scales.max()} (2^{int(scales.min()) - 127}..2^{int(scales.max()) - 127}); "
          f"distinct e4m3 codes used {len(np.unique(codes))}")
    nz = x[row_rms > 0]
    unit = nz / np.sqrt((nz**2).mean(axis=1, keepdims=True))
    rot = R.rotate(unit.astype(np.float32))
    print(f"  kurtosis (3 = Gaussian): raw unit-RMS rows {kurtosis(unit):.2f}, FWHT-rotated {kurtosis(rot):.2f}, "
          f"N(0,1) reference 3.00")
    res = {"layer": args.layer, "rows": int(nz.shape[0]), "row_rms_median": float(np.median(row_rms))}
    # the yardstick: relative RMS error of each format on the rows (per-vector scales as in the README)
    specs = [
        ("trellis-4 L=12 lloyd_kv12a (4.09 b/elem)", lambda v: B.trellis(v, "lloyd_kv12a", workers=args.workers)),
        ("int4 + Hadamard g32 (5.0 b/elem)", lambda v: B.int_hadamard(v, 4, 32)),
        ("int4 + Hadamard g256 (4.125 b/elem)", lambda v: B.int_hadamard(v, 4, 256)),
        ("NVFP4 + Hadamard (4.5 b/elem)", lambda v: B.fp4(v, "nv", had=True)),
        ("MXFP4 + Hadamard (4.25 b/elem)", lambda v: B.fp4(v, "mx", had=True)),
        ("MXFP4, no rotation (4.25 b/elem)", lambda v: B.fp4(v, "mx", had=False)),
    ]
    for name, fn in specs:
        t1 = time.time()
        xhat = fn(nz.astype(np.float32))
        err = rel_rms(xhat, nz)
        res[name] = err
        print(f"  {name:44s} rel RMS {err:.4f}   ({time.time() - t1:.1f}s)")
    # reference point: the fp8 record's own granularity (what an extra fp8 round trip costs)
    e4 = B.e4m3(nz.astype(np.float32) / 2.0 ** np.floor(np.log2(np.abs(nz).max(axis=1, keepdims=True) / 448 + 1e-30)))
    print(f"  (for scale: re-quantizing the rows to e4m3 with one per-row power-of-two scale: rel RMS "
          f"{rel_rms(e4 * 2.0 ** np.floor(np.log2(np.abs(nz).max(axis=1, keepdims=True) / 448 + 1e-30)), nz):.4f})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
