#!/usr/bin/env python3
"""The full-depth dummy model: DeepSeek-V4.1-Flash's own 40-layer structure (compress ratios, KV / index / candidate
sources, Engram layers 1 and 14, DSpark targets 37-39, hidden sizes) with 16 routed experts and ~60K-row Engram
tables, for activation / CUDA-graph memory at the full depth (~7 GiB of weights per GPU at TP4).
usage, inside the serving image: make_deep_config.py FULL_CONFIG OUT_DIR, then copy the checkpoint's small files into
OUT_DIR as for make_mini_config.py."""
import json
import sys
from pathlib import Path

from vllm.models.deepseek_v41.common.engram import find_next_prime

src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads(src.read_text())
t = cfg["text_config"]
t["engram_vocab_size"] = 10007
t["n_routed_experts"] = 16
t["dspark_n_routed_experts"] = 16
t["max_position_embeddings"] = 16384
seen: set[int] = set()
nums = []
for _ in t["engram_layer_ids"]:
    total = 0
    for _ in range(t["engram_max_ngram_size"] - 1):
        cur = t["engram_vocab_size"] - 1
        for _ in range(t["engram_n_heads"]):
            cur = find_next_prime(cur, seen)
            seen.add(cur)
            total += cur
    nums.append(total)
t["engram_num_embeddings"] = nums
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
print(f"{out_dir / 'config.json'}: {t['num_hidden_layers']} layers, 16 experts, engram tables {nums}")
