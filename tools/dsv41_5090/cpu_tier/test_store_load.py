"""The expert store loader: one rank's records into a pinned arena (phase 2.2), timing and spot checks.

    ... test_store_load.py --store /store --rank 0 [--layers 40] [--resident-json /parity/placement.json] [--pinned 1]

Loads every expert of the given layers (or only the NON-resident ones of a placement file: the static-store option)
into a cudaHostAlloc'd arena through libcpu_tier.so's pread loader, reports GB/s and the pinned footprint, and checks
16 random records against the files. Runs in the serving image (any GPU, for the pinned allocation).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def ensure_libs(args):
    if os.environ.get("CPU_TIER_LIB"):
        return
    out = os.path.join(args.build_dir, "libcpu_tier.so")
    if not os.path.exists(out) or args.rebuild:
        subprocess.run(["bash", os.path.join(_HERE, "build.sh")], check=True, env=dict(os.environ, ARCH=args.arch, OUT=out))
    os.environ["CPU_TIER_LIB"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--resident-json", default="", help="placement.py output: load only the non-resident experts")
    ap.add_argument("--pinned", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--build-dir", default="/tmp")
    ap.add_argument("--arch", default="znver4")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    ensure_libs(args)
    import torch  # noqa: F401  (the extension needs a CUDA context for the pinned allocation)
    from cpu_tier import RECORD_BYTES, ExpertStore

    mask = None
    if args.resident_json:
        P = json.load(open(args.resident_json))
        mask = np.ones((args.layers, args.experts), dtype=np.uint8)
        for L in P["layers"][: args.layers]:
            for e in L["resident"]:
                mask[L["layer"], e] = 0
    t0 = time.perf_counter()
    store = ExpertStore(args.store, args.rank, args.layers, args.experts, mask=mask, pinned=bool(args.pinned),
                        nthreads=args.threads)
    t_all = time.perf_counter() - t0
    gb = store.nbytes / 1e9
    print(f"rank {args.rank}: {store.n_records} records = {gb:.1f} GB {'pinned' if args.pinned else 'malloc'} arena; "
          f"load {store.load_seconds:.1f} s ({gb / store.load_seconds:.1f} GB/s read), alloc+load {t_all:.1f} s", flush=True)
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(16):
        l = int(rng.integers(args.layers))
        cands = np.nonzero(store.slot_of[l] >= 0)[0]
        e = int(rng.choice(cands))
        with open(os.path.join(args.store, f"rank{args.rank}", f"layer{l:02d}.bin"), "rb") as f:
            f.seek(e * RECORD_BYTES)
            ref = np.frombuffer(f.read(RECORD_BYTES), dtype=np.uint8)
        got = store.record(l, e).numpy()
        bad += int(not np.array_equal(got, ref))
    print(f"spot check: {bad} of 16 records differ from the files")
    # RSS / pinned footprint as the kernel sees it
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS", "VmLck", "VmPin")):
                print("  " + line.strip())
    print("PASS" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
