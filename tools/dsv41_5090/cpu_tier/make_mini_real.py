#!/usr/bin/env python3
"""A "mini-real" DeepSeek-V4.1-Flash checkpoint for the hybrid's single-GPU tests of the real-checkpoint path.

Eight real decoder layers in the mini config's structure (make_mini_config.py: SWA / SWA / ratio-2 KV+index source /
ratio 2 / ratio-1 KV+index+candidate source / ratio 1 / ratio-1 index source / ratio 1 = real layers
0 1 2 3 20 21 24 23), the first E routed experts of each with their TP4 rank-0 slice (w1/w3 rows 0..575, w2 columns
0..575 = exactly the hybrid store's rank-0 records, so `VLLM_MOET_CPU_TIER_STORE=/mnt/nvme/packs/dsv41-hybrid` serves
it with n_experts = E), the shared expert and the router sliced the same way, no Engram (engram_layer_ids = []), no
MTP / vision tensors. Served at TP1 with moe_intermediate_size 576 the per-rank expert slice is the full model's,
so the CPU kernel and b12x's compact path run exactly as at TP4. The outputs are not the model's (a 1/5-depth,
1/24-width MoE); the point is the loader, the store and the identity hybrid == all-resident on real bytes.

    make_mini_real.py [--src /mnt/nvme/models/DeepSeek-V4.1-Flash] [--mini /mnt/nvme/models/mini-hyb576]
                      [--out /mnt/nvme/models/mini-real576] [--layers 0,1,2,3,20,21,24,23] [--experts 16]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INTER_R = 576


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/nvme/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--mini", default="/mnt/nvme/models/mini-hyb576", help="make_mini_config.py's directory (config + tokenizer)")
    ap.add_argument("--out", default="/mnt/nvme/models/mini-real576")
    ap.add_argument("--layers", default="0,1,2,3,20,21,24,23", help="real layer of every mini layer, in order")
    ap.add_argument("--experts", type=int, default=16)
    args = ap.parse_args()
    real_layers = [int(x) for x in args.layers.split(",")]
    E = args.experts
    os.makedirs(args.out, exist_ok=True)

    cfg = json.load(open(os.path.join(args.mini, "config.json")))
    t = cfg["text_config"]
    assert t["num_hidden_layers"] == len(real_layers), (t["num_hidden_layers"], real_layers)
    assert t["moe_intermediate_size"] == INTER_R, t["moe_intermediate_size"]
    t["n_routed_experts"] = E
    t["engram_layer_ids"] = []
    t["engram_num_embeddings"] = []
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    # the real placement (routing/placement.py) renumbered to the mini layers, for tests at the full per-layer scale
    pl_path = os.environ.get("REAL_PLACEMENT", "/mnt/nvme/vLLM-Moet-cpu/internal/dsv41-5090-cpu/routes/placement-0.28-uniform.json")
    if os.path.exists(pl_path):
        P = json.load(open(pl_path))
        mini = dict(source=pl_path, real_layers=real_layers,
                    layers=[dict(layer=i, resident=[e for e in P["layers"][L]["resident"] if e < E],
                                 count=len([e for e in P["layers"][L]["resident"] if e < E])) for i, L in enumerate(real_layers)])
        json.dump(mini, open(os.path.join(args.out, "placement.json"), "w"))
        print(f"placement.json: {[l['count'] for l in mini['layers']]} resident per mini layer (from {pl_path})")
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(os.path.join(args.mini, f), os.path.join(args.out, f))

    wmap = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))["weight_map"]
    shards: dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        s = wmap[name]
        f = shards.get(s)
        if f is None:
            f = shards[s] = safe_open(os.path.join(args.src, s), framework="pt")
        return f.get_tensor(name)

    def slice_tensor(name: str, x: torch.Tensor) -> torch.Tensor | None:
        """The mini model's slice of a layer tensor (None = drop)."""
        m = re.search(r"\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$", name)
        if m:
            e, w = int(m.group(1)), m.group(2)
            if e >= E:
                return None
            if w in ("w1", "w3"):
                return x[:INTER_R]                         # nibbles [2304, 2560] / e8m0 scales [2304, 160]
            return x[:, : x.shape[1] // 4]                 # w2 nibbles [5120, 1152] -> 288; scales [5120, 72] -> 18
        if ".ffn.shared_experts." in name:
            w = name.rsplit(".", 2)[-2]
            if w in ("w1", "w3"):
                return x[: x.shape[0] // 4]                 # fp8 [2304, 5120] -> 576; scales [72, 160] -> 18
            return x[:, : x.shape[1] // 4]                 # w2 [5120, 2304] -> 576; scales [160, 72] -> 18
        if ".ffn.gate." in name:
            return x[:E]                                   # router weight [384, 5120] / bias [384]
        if ".engram." in name:
            return None
        return x

    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    t_all = time.perf_counter()
    for mini_l, real_l in enumerate(real_layers):
        t0 = time.perf_counter()
        prefix = f"layers.{real_l}."
        names = sorted(n for n in wmap if n.startswith(prefix))
        out: dict[str, torch.Tensor] = {}
        for n in names:
            # skip the experts we drop without reading them
            m = re.search(r"\.ffn\.experts\.(\d+)\.", n)
            if m and int(m.group(1)) >= E:
                continue
            if ".engram." in n:
                continue
            x = slice_tensor(n, get(n))
            if x is None:
                continue
            out[f"layers.{mini_l}." + n[len(prefix):]] = x.contiguous()
        fname = f"model-layer{mini_l:02d}.safetensors"
        save_file(out, os.path.join(args.out, fname), metadata={"format": "pt"})
        nbytes = sum(v.numel() * v.element_size() for v in out.values())
        for k in out:
            index["weight_map"][k] = fname
        index["metadata"]["total_size"] += nbytes
        print(f"  mini layer {mini_l} <- real layer {real_l}: {len(out)} tensors, {nbytes / 2**20:.0f} MiB, "
              f"{time.perf_counter() - t0:.1f} s", flush=True)
    rest = {}
    for n in ("embed.weight", "head.weight", "norm.weight"):
        rest[n] = get(n).contiguous()
    fname = "model-rest.safetensors"
    save_file(rest, os.path.join(args.out, fname), metadata={"format": "pt"})
    for k, v in rest.items():
        index["weight_map"][k] = fname
        index["metadata"]["total_size"] += v.numel() * v.element_size()
    json.dump(index, open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=1)
    print(f"{args.out}: {len(index['weight_map'])} tensors, {index['metadata']['total_size'] / 2**30:.2f} GiB, "
          f"{time.perf_counter() - t_all:.0f} s; experts = rank-0 records of the hybrid store for layers {real_layers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
