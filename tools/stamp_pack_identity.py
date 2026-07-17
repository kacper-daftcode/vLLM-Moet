#!/usr/bin/env python3
"""Stamp EXISTING pack-store sidecars with the checkpoint identity.

The pack sidecar gate now includes `ckpt_id` (checkpoint path + sha1 of the
safetensors index — the planes cache convention). Pre-fix sidecars lack the
field and are treated as STALE, forcing a full pack rebuild on next boot.
For packs an operator KNOWS were written from a given checkpoint, this tool
injects the correct ckpt_id in place, skipping the rebuild.

IMPORTANT: ckpt_id hashes the model path STRING AS THE SERVER SEES IT
(planes-cache convention). Containers usually mount the checkpoint at
/model, so pass --model-as /model and point --checkpoint at the HOST copy
for the index hash.

  python3 tools/stamp_pack_identity.py \
      --store-dir /workspace/moet-serve/packs-glm \
      --checkpoint /workspace/glm-5.2-nvfp4 --model-as /model

Refuses to overwrite a DIFFERENT existing ckpt_id without --force (that
means the dir served another checkpoint at some point — rebuild instead).
"""
import argparse
import glob
import hashlib
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="host path of the checkpoint (for the index hash)")
    ap.add_argument("--model-as", required=True,
                    help="model path AS SERVED (e.g. /model in docker)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    h = hashlib.sha1(args.model_as.encode())
    idx = os.path.join(args.checkpoint, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, "rb") as f:
            h.update(f.read())
    else:
        print(f"note: {idx} absent — ckpt_id covers the path string only")
    ckpt_id = h.hexdigest()
    print(f"ckpt_id = {ckpt_id}  (model-as {args.model_as!r})")

    sidecars = sorted(glob.glob(os.path.join(args.store_dir, "*.json")))
    sidecars = [p for p in sidecars
                if not os.path.basename(p).startswith("pool-heat")
                and not p.endswith(".heat.json")]
    if not sidecars:
        sys.exit(f"no pack sidecars under {args.store_dir}")
    for p in sidecars:
        with open(p) as f:
            meta = json.load(f)
        if "slot_bytes" not in meta or "layers" not in meta:
            print(f"  skip {os.path.basename(p)} (not a pack sidecar)")
            continue
        old = meta.get("ckpt_id")
        if old == ckpt_id:
            print(f"  ok   {os.path.basename(p)} (already stamped)")
            continue
        if old is not None and not args.force:
            sys.exit(f"  REFUSE {p}: existing ckpt_id {old} != {ckpt_id} "
                     "(another checkpoint served this dir; rebuild or "
                     "--force if you are certain)")
        meta["ckpt_id"] = ckpt_id
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, p)
        print(f"  stamp {os.path.basename(p)} "
              f"({len(meta.get('layers', []))} layers)")
    print("done")


if __name__ == "__main__":
    main()
