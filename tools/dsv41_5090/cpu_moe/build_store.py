"""The pinned expert store's files: every expert of every MoE layer, sliced per TP rank, in the checkpoint's own bytes
(the kernel's record: w1 nibbles | w1 scales | w3 nibbles | w3 scales | w2 nibbles | w2 scales, 4,700,160 B).

    python build_store.py [--out /mnt/nvme/packs/dsv41-hybrid] [--tp 4] [--layers 0-39] [--workers 8] [--check]

Writes <out>/rank<r>/layer<LL>.bin (384 records x 4,700,160 B = 1.68 GiB per file; 40 layers = 67 GiB per rank) and
<out>/manifest.json (geometry, checkpoint path, record layout, per-file sha1 of the first and last record). One pass
over the checkpoint per layer (w1 / w3 / w2 of each expert read once, all ranks written); idempotent: finished files
(size right and the manifest entry present) are skipped. The loader of the hybrid mmaps / reads a rank's 40 files
into its pinned arena. --check re-reads a few random records against the checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv41_b12x as B  # noqa: E402
import dsv41_mxfp4 as D1  # noqa: E402

E_PER_LAYER = 384


def parse_range(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def layer_records(layer: int, tp: int) -> list[np.ndarray]:
    """All experts of one layer -> per rank a [384, RECORD_BYTES] array (one read of each expert's tensors)."""
    out = [np.empty((E_PER_LAYER, B.RECORD_BYTES), dtype=np.uint8) for _ in range(tp)]
    inter_r = D1.INTER // tp
    for e in range(E_PER_LAYER):
        base = f"layers.{layer}.ffn.experts.{e}"
        w1n = D1.get_tensor(f"{base}.w1.weight").view(torch.uint8).numpy()
        w1s = D1.get_tensor(f"{base}.w1.scale").view(torch.uint8).numpy()
        w3n = D1.get_tensor(f"{base}.w3.weight").view(torch.uint8).numpy()
        w3s = D1.get_tensor(f"{base}.w3.scale").view(torch.uint8).numpy()
        w2n = D1.get_tensor(f"{base}.w2.weight").view(torch.uint8).numpy()
        w2s = D1.get_tensor(f"{base}.w2.scale").view(torch.uint8).numpy()
        for r in range(tp):
            r0, r1 = inter_r * r, inter_r * (r + 1)
            rec = out[r][e]
            o = 0
            for arr in (w1n[r0:r1], w1s[r0:r1], w3n[r0:r1], w3s[r0:r1], w2n[:, r0 // 2:r1 // 2], w2s[:, r0 // 32:r1 // 32]):
                flat = np.ascontiguousarray(arr).reshape(-1)
                rec[o:o + flat.size] = flat
                o += flat.size
            assert o == B.RECORD_BYTES
    return out


def sha1_of(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _build_layer(job):
    l, tp, out = job
    t0 = time.perf_counter()
    recs = layer_records(l, tp)
    t1 = time.perf_counter()
    entries = {}
    for r in range(tp):
        p = os.path.join(out, f"rank{r}/layer{l:02d}.bin")
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(recs[r].tobytes())
        os.replace(tmp, p)
        entries[f"rank{r}/layer{l:02d}.bin"] = dict(layer=l, rank=r, first_sha1=sha1_of(recs[r][0]), last_sha1=sha1_of(recs[r][-1]))
    return l, entries, t1 - t0, time.perf_counter() - t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/nvme/packs/dsv41-hybrid")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--workers", type=int, default=4, help="layer-building processes (each holds 4 x 1.7 GiB)")
    ap.add_argument("--check", type=int, default=0, help="re-read N random records per rank against the checkpoint")
    args = ap.parse_args()
    layers = parse_range(args.layers)
    os.makedirs(args.out, exist_ok=True)
    for r in range(args.tp):
        os.makedirs(os.path.join(args.out, f"rank{r}"), exist_ok=True)
    mpath = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    manifest.update(dict(checkpoint=D1.MODEL_DIR, tp=args.tp, experts=E_PER_LAYER, record_bytes=B.RECORD_BYTES,
                         record_layout="w1n[576,2560] w1s[576,160] w3n[576,2560] w3s[576,160] w2n[5120,288] w2s[5120,18]",
                         file_bytes=E_PER_LAYER * B.RECORD_BYTES))
    files = manifest.setdefault("files", {})

    def fname(r, l):
        return f"rank{r}/layer{l:02d}.bin"

    def done(l):
        return all(fname(r, l) in files and os.path.exists(os.path.join(args.out, fname(r, l))) and
                   os.path.getsize(os.path.join(args.out, fname(r, l))) == E_PER_LAYER * B.RECORD_BYTES for r in range(args.tp))

    todo = [l for l in layers if not done(l)]
    print(f"store {args.out}: {len(layers) - len(todo)} layers present, {len(todo)} to build", flush=True)
    t_all = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for l, entries, t_read, t_write in ex.map(_build_layer, [(l, args.tp, args.out) for l in todo]):
            files.update(entries)
            with open(mpath + ".tmp", "w") as f:
                json.dump(manifest, f, indent=1)
            os.replace(mpath + ".tmp", mpath)
            print(f"  layer {l:2d}: read+slice {t_read:5.1f} s, write {t_write:4.1f} s "
                  f"({args.tp * E_PER_LAYER * B.RECORD_BYTES / 2**30:.2f} GiB)", flush=True)
    print(f"built {len(todo)} layers in {time.perf_counter() - t_all:.0f} s", flush=True)

    if args.check:
        rng = np.random.default_rng(0)
        bad = 0
        for r in range(args.tp):
            for _ in range(args.check):
                l = int(rng.choice(layers))
                e = int(rng.integers(E_PER_LAYER))
                p = os.path.join(args.out, fname(r, l))
                with open(p, "rb") as f:
                    f.seek(e * B.RECORD_BYTES)
                    rec = np.frombuffer(f.read(B.RECORD_BYTES), dtype=np.uint8)
                ref = B.record_from_checkpoint(l, e, r)
                ok = np.array_equal(rec, ref)
                bad += not ok
                print(f"  check rank {r} layer {l} expert {e}: {'ok' if ok else 'MISMATCH'}", flush=True)
        print(f"check: {bad} mismatches")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
