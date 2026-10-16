#!/usr/bin/env python3
"""A 1/5-scale DeepSeek-V4.1-Flash config for dummy-weight dry runs next to the serving job.

Eight decoder layers that keep every structural feature of the 40-layer model — compress ratios 0/0/2/2/1/1/1/1
(two KV-source layers, three index-source layers, the last KV source at ratio 1 and the candidate source like layer 20
of the full model - vLLM since vllm#59894 replays from it and asserts it - and three decoder-replay layers behind it),
two Engram layers before the last KV source, the DSpark drafter's three target layers, the same hidden sizes — with
16 routed experts and Engram tables of ~60K rows per rank, so `vllm serve --load-format dummy`
(random weights) fits in ~3 GiB per GPU at TP4 and starts in ~3 minutes with a warm JIT cache. The outputs are
noise; the point is the plumbing (runner, model state, attention metadata, replay batches, Engram, drafter) and the
GPU/CPU timeline of a step, which are the full model's.

usage: make_mini_config.py /path/to/DeepSeek-V4.1-Flash/config.json OUT_DIR
Then copy (not symlink: the container cannot follow host paths) the checkpoint's small files into OUT_DIR —
tokenizer.json, tokenizer_config.json, model.safetensors.index.json, encoding/, assets/, inference/, evaluation/ —
and serve OUT_DIR with serve_mini.sh. Needs vLLM importable (the serving image) for the Engram prime layout."""

import json
import sys
from pathlib import Path


def main() -> int:
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    from vllm.models.deepseek_v41.common.engram import find_next_prime

    cfg = json.loads(src.read_text())
    t = cfg["text_config"]
    t["num_hidden_layers"] = 8
    t["compress_ratios"] = [0, 0, 2, 2, 1, 1, 1, 1] + [0] * t["num_nextn_predict_layers"]
    t["kv_source_layer_ids"] = [2, 4]
    t["index_source_layer_ids"] = [2, 4, 6]
    t["candidate_source_layer_id"] = 4
    t["engram_layer_ids"] = [1, 3]
    t["engram_vocab_size"] = 10007
    t["n_routed_experts"] = 16
    t["dspark_n_routed_experts"] = 16
    t["dspark_target_layer_ids"] = [5, 6, 7]
    t["max_position_embeddings"] = 16384
    # engram_num_embeddings = the sum of the per-layer primes EngramLayout draws (unique across layers).
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
    print(f"{out_dir / 'config.json'}: 8 layers, {t['n_routed_experts']} experts, engram tables {nums}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
