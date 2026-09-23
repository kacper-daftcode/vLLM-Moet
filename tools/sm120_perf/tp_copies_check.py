#!/usr/bin/env python3
"""Compare the TP-rank copies inside the block files of vLLM's KV offload disk tier.

With one host copy per rank (vLLM's per-rank layout), each file the TieringOffloadingSpec "fs" tier writes is one
CPU chunk row: world_size equal slots, rank after rank. For a TP-replicated cache (the MLA latent, the DSA indexer K)
the slots should be byte-identical - which is what makes one shared copy (replicated_layout) lossless. Reports how
many sampled files are, and how many bytes differ in the others. Files are opened with O_NOATIME, so the
atime-based retention (docker/sm120/kvcache-ttl.sh) is not disturbed.

usage: tp_copies_check.py DIR [--world-size 4] [--sample 400] [--seed 0]
       (DIR = a <root_dir>/_model_<hash>_r<rank> directory of the per-rank layout)
"""

import argparse
import os
import random
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = []
    for dirpath, _, names in os.walk(args.root):
        files += [os.path.join(dirpath, n) for n in names if n.endswith(".bin")]
    files.sort()
    random.Random(args.seed).shuffle(files)
    files = files[: args.sample]
    print(f"{len(files)} files sampled under {args.root}")

    sizes: dict[int, int] = {}
    identical = 0
    diff_files = []
    for path in files:
        fd = os.open(path, os.O_RDONLY | os.O_NOATIME)
        try:
            data = os.read(fd, 1 << 24)
        finally:
            os.close(fd)
        sizes[len(data)] = sizes.get(len(data), 0) + 1
        slot = len(data) // args.world_size
        slots = [data[i * slot : (i + 1) * slot] for i in range(args.world_size)]
        if all(s == slots[0] for s in slots[1:]):
            identical += 1
            continue
        ndiff = [sum(a != b for a, b in zip(slots[0], s)) for s in slots[1:]]
        diff_files.append((path, ndiff, slot))

    print("file sizes:", sizes)
    print(f"identical slots: {identical}/{len(files)}")
    for path, ndiff, slot in diff_files[:20]:
        print(f"  differs: {os.path.basename(path)} bytes != slot 0: {ndiff} of {slot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
