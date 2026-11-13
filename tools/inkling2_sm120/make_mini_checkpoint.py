#!/usr/bin/env python3
"""A few-layer Inkling2 checkpoint cut from the released one, for kernel-path tests on ONE card.

    python3 make_mini_checkpoint.py SRC DST [--layers 0,1,2,42,43,47] [--draft]

The full model needs four 96 GB cards; every kernel path of it (routing, both expert formats, local and full
attention, the short convolutions, AttnRes, the vision tower) can be driven on one card by a model of six real
layers: three NVFP4 layers and three MXFP8 layers, the last a full-attention layer. DST gets

  - config.json with `num_hidden_layers`, `local_layer_ids` of the kept layers (renumbered 0..) and no `mtp_config`
    (the DSpark draft reads target layers that are not there) - or, with --draft, the draft attached to what is
    there: the draft mixes the inputs of the target's layers 5, 11, ..., 41 and its final-norm output through
    `fuse_weight` (draft layers + 1 rows, one column per feature); `target_layer_ids` keeps the kept layers among
    them (renumbered) and the output, `fuse_weight` in DST's mtp.safetensors keeps their columns, every other draft
    tensor is copied byte for byte. The cut must keep at least one of those layers (e.g. --layers 0,1,2,42,43,41).
    The draft of a cut predicts nothing useful: it drives the draft's code paths - load, K / V write, block
    attention, Markov head, argmax, verification and rejection - on one card;
  - hf_quant_config.json with the kept tensors renamed;
  - the kept tensors copied byte for byte under their new names (layers and AttnRes queries renumbered - an
    attention query belongs to the layer after its index; embeddings, norms, unembedding and the vision tower as they
    are), in shards of at most 8 GB with an index;
  - copies of the tokenizer / processor / template files.

The text it generates is meaningless (42 layers are missing); what it serves for is that nothing crashes, every
value is finite and the same request returns the same tokens. Copies run through bounded writes (the page cache of
the output is dropped as it is written).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import re
import struct
import sys

SHARD_BYTES = 8 << 30
CHUNK = 64 << 20


def read_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def write_draft(src: str, dst: str, cols: list[int]) -> None:
    """mtp.safetensors with `fuse_weight` cut to the columns `cols`; every other tensor byte for byte."""
    base, h = read_header(src)
    meta = h.pop("__metadata__", None)
    header: dict = {} if meta is None else {"__metadata__": meta}
    plan, off = [], 0
    for name in sorted(h, key=lambda n: h[n]["data_offsets"][0]):
        t = dict(h[name])
        b0, b1 = t["data_offsets"]
        cut = None
        if name == "fuse_weight":
            rows, width = t["shape"]
            cut = (rows, width, (b1 - b0) // (rows * width))
            t["shape"] = [rows, len(cols)]
            nbytes = rows * len(cols) * cut[2]
        else:
            nbytes = b1 - b0
        t["data_offsets"] = [off, off + nbytes]
        off += nbytes
        header[name] = t
        plan.append((base + b0, b1 - b0, cut))
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    fd = os.open(src, os.O_RDONLY)
    try:
        with open(dst + ".part", "wb") as w:
            w.write(struct.pack("<Q", len(hb)))
            w.write(hb)
            for pos, nbytes, cut in plan:
                if cut is not None:
                    rows, width, elem = cut
                    raw = os.pread(fd, nbytes, pos)
                    assert len(raw) == nbytes, "short read in mtp.safetensors"
                    w.write(b"".join(raw[(r * width + c) * elem:(r * width + c + 1) * elem] for r in range(rows) for c in cols))
                    continue
                done = 0
                while done < nbytes:
                    b = os.pread(fd, min(CHUNK, nbytes - done), pos + done)
                    assert b, "short read in mtp.safetensors"
                    w.write(b)
                    done += len(b)
            w.flush()
            os.fdatasync(w.fileno())
            os.posix_fadvise(w.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    os.rename(dst + ".part", dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", default="0,1,2,42,43,47")
    ap.add_argument("--draft", action="store_true", help="attach the DSpark draft (the cut must keep one of its target layers)")
    ap.add_argument("--draft-only", action="store_true", help="write only config.json's mtp_config and mtp.safetensors into an existing cut of --layers")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    new_id = {old: i for i, old in enumerate(layers)}
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    text = cfg["text_config"]
    assert all(0 <= l < text["num_hidden_layers"] for l in layers) and len(set(layers)) == len(layers)
    local = set(text["local_layer_ids"])
    text["local_layer_ids"] = [new_id[l] for l in layers if l in local]
    text["num_hidden_layers"] = len(layers)
    mtp = cfg.pop("mtp_config", None)
    os.makedirs(a.dst, exist_ok=True)
    if a.draft or a.draft_only:
        if not mtp:
            raise SystemExit("--draft: the checkpoint's config.json has no mtp_config")
        ids = mtp["target_layer_ids"]
        keep = [j for j, old in enumerate(ids[:-1]) if old in new_id]
        new_ids = [new_id[ids[j]] for j in keep]
        if not keep or new_ids != sorted(new_ids):
            raise SystemExit(f"--draft: the cut must keep, in ascending positions, at least one of the draft's target layers {ids[:-1]} "
                             f"(e.g. --layers 0,1,2,42,43,41); it keeps {[ids[j] for j in keep]} at {new_ids}")
        mtp["target_layer_ids"] = new_ids + [len(layers)]
        cfg["mtp_config"] = mtp
        write_draft(os.path.join(a.src, "mtp.safetensors"), os.path.join(a.dst, "mtp.safetensors"), keep + [len(ids) - 1])
        print(f"draft: target_layer_ids {ids} -> {mtp['target_layer_ids']} (fuse_weight columns {keep + [len(ids) - 1]})", flush=True)
        if a.draft_only:
            json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
            return 0

    def rename(name: str) -> str | None:
        """The tensor's name in the mini checkpoint, or None if it is dropped."""
        m = re.match(r"(model\.llm\.layers\.)(\d+)(\..*)$", name)
        if m:
            return f"{m.group(1)}{new_id[int(m.group(2))]}{m.group(3)}" if int(m.group(2)) in new_id else None
        m = re.match(r"(model\.llm\.attn_res\.)(\w+)\.(\d+)$", name)
        if m:
            # mlp_queries.N belongs to layer N; attn_queries.N to layer N + 1 (the first layer has none)
            layer = int(m.group(3)) + (1 if m.group(2) == "attn_queries" else 0)
            if layer not in new_id or (m.group(2) == "attn_queries" and new_id[layer] == 0):
                return None
            return f"{m.group(1)}{m.group(2)}.{new_id[layer] - (1 if m.group(2) == 'attn_queries' else 0)}"
        return name

    index = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    plan = []  # (new name, source file, data offset, nbytes, dtype, shape)
    headers = {}
    for name, fname in index.items():
        new = rename(name)
        if new is None:
            continue
        if fname not in headers:
            headers[fname] = read_header(os.path.join(a.src, fname))
        base, h = headers[fname]
        t = h[name]
        plan.append((new, fname, base + t["data_offsets"][0], t["data_offsets"][1] - t["data_offsets"][0], t["dtype"], t["shape"]))
    plan.sort(key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p[0])])
    shards, cur, size = [], [], 0
    for p in plan:
        if cur and size + p[3] > SHARD_BYTES:
            shards.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += p[3]
    shards.append(cur)
    total = sum(p[3] for p in plan)
    print(f"{len(plan)} tensors, {total / 1e9:.2f} GB in {len(shards)} shards -> {a.dst}", flush=True)
    weight_map = {}
    for i, shard in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        header, off = {"__metadata__": {"format": "pt"}}, 0
        for new, _, _, nbytes, dtype, shape in shard:
            header[new] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nbytes]}
            off += nbytes
            weight_map[new] = fname
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        out = os.path.join(a.dst, fname)
        with open(out + ".part", "wb") as w:
            w.write(struct.pack("<Q", len(hb)))
            w.write(hb)
            for _, src_name, src_off, nbytes, _, _ in shard:
                fd = os.open(os.path.join(a.src, src_name), os.O_RDONLY)
                try:
                    done = 0
                    while done < nbytes:
                        b = os.pread(fd, min(CHUNK, nbytes - done), src_off + done)
                        assert b, f"short read in {src_name}"
                        w.write(b)
                        done += len(b)
                        if done % (16 * CHUNK) < CHUNK:
                            w.flush()
                            os.fdatasync(w.fileno())
                            os.posix_fadvise(w.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    os.posix_fadvise(fd, src_off, nbytes, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            w.flush()
            os.fdatasync(w.fileno())
            os.posix_fadvise(w.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        os.rename(out + ".part", out)
        print(f"  {fname}: {len(shard)} tensors, {off / 1e9:.2f} GB", flush=True)
    json.dump({"metadata": {"format": "pt"}, "weight_map": weight_map}, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=1)
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
    quant = json.load(open(os.path.join(a.src, "hf_quant_config.json")))
    for group in quant["model"]["parameter_groups"]:
        group["tensors"] = sorted(n for n in (rename(t) for t in group["tensors"]) if n is not None)
    if not a.draft:
        quant.pop("mtp", None)
    json.dump(quant, open(os.path.join(a.dst, "hf_quant_config.json"), "w"), indent=1)
    skip = {"config.json", "hf_quant_config.json", "model.safetensors.index.json", "mtp.safetensors", "README.md", ".gitattributes"}
    for name in sorted(os.listdir(a.src)):
        src = os.path.join(a.src, name)
        if name in skip or name.endswith((".safetensors", ".part")):
            continue
        dst = os.path.join(a.dst, name)
        if not os.path.lexists(dst):
            # copies, not links: the directory is served from a container that mounts it alone
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
    print("layers", layers, "-> local", text["local_layer_ids"], "| mxfp8 layers:",
          sorted({int(re.search(r"layers\.(\d+)\.", t).group(1)) for g in quant["model"]["parameter_groups"]
                  if g["compute_dtype"] == "mxfp8" for t in g["tensors"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
