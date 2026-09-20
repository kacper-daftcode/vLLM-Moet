#!/usr/bin/env python3
"""Offline probe of vLLM's KV cache layout for DeepSeek-V4.1-Flash (no model weights, one GPU for
platform detection): which layers share a physical block, the per-layer page sizes, the packed
block stride each kernel sees, and the token capacity for a given KV budget.

vLLM packs the pages of every kv-source layer (compressed cache) and its indexer K cache into one
block (block-outermost layout, `BLHNC`), so the block stride = the sum of those pages, padded to
each spec's alignment (576 B for fp8_ds_mla). DeepGEMM's paged MQA logits and FlashInfer's SM120
sparse MLA address pages through that stride; this script prints it and checks the alignments
the kernels rely on, for the FP8 (132 B/key) or MXFP4 (68 B/key) indexer record and, optionally,
another main-KV record width (e.g. 288 B for an NVFP4 compressed cache).

The specs are built from the same constructor arguments the model uses
(`vllm/models/deepseek_v4_1/attention.py`, `sparse_swa.py`); check them against those files after
a vLLM rebase. With the served configuration (block 128, fp8_ds_mla, 3.14 GiB available) it
reproduces the log's `GPU KV cache size: 1,490,870 tokens` to 0.1 %.

usage (inside the serving image):
  python3 kv_layout_probe.py --model-dir /model [--indexer-bytes 132|68] [--main-bytes 584]
                             [--avail-gib 3.14] [--max-model-len 524288]
"""
from __future__ import annotations

import argparse
import json

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--indexer-bytes", type=int, default=132, help="indexer K record: 132 (fp8) or 68 (mxfp4)")
    ap.add_argument("--main-bytes", type=int, default=584, help="compressed/SWA record: 584 (fp8_ds_mla)")
    ap.add_argument("--avail-gib", type=float, default=3.14, help="'Available KV cache memory' from the log")
    ap.add_argument("--max-model-len", type=int, default=524288)
    ap.add_argument("--block-size", type=int, default=128)
    args = ap.parse_args()

    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_capacity,
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )
    from vllm.v1.kv_cache_interface import (
        KVCacheTensor,
        MLAAttentionSpec,
        SlidingWindowMLASpec,
        UniformTypeKVCacheSpecs,
        create_kv_cache_views,
        get_kv_quant_mode,
    )

    vc = EngineArgs(
        model=args.model_dir,
        tokenizer_mode="deepseek_v41",
        tensor_parallel_size=1,  # MLA has one KV head: page sizes do not depend on TP
        kv_cache_dtype="fp8",
        max_model_len=args.max_model_len,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        block_size=args.block_size,
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": 5,
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "block",
            "enable_adaptive_verification": False,
        },
    ).create_engine_config()
    # the attention layer writes the canonical string back at model init; the engine core resolves
    # the layout from DeepseekV4IndexerBackend.supported_kv_cache_layouts()[0]
    vc.cache_config.cache_dtype = "fp8_ds_mla"
    vc.cache_config.kv_cache_layout = "BLHNC"
    layout = vc.cache_config.get_resolved_kv_cache_layout()
    block_size = vc.cache_config.block_size

    cfg = json.load(open(f"{args.model_dir}/config.json"))["text_config"]
    crs, kv_sources = cfg["compress_ratios"], cfg["kv_source_layer_ids"]
    qm = get_kv_quant_mode("fp8_ds_mla")
    main_bytes, idx_bytes = args.main_bytes, args.indexer_bytes

    specs = {}
    for li in kv_sources:
        cr = crs[li]
        specs[f"model.layers.{li}.self_attn"] = MLAAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=512, dtype=torch.uint8, tokens_per_state=cr,
            cache_dtype_str="fp8_ds_mla", alignment=576, model_version="deepseek_v4", kv_quant_mode=qm,
            state_content_bytes=main_bytes,
        )
        specs[f"model.layers.{li}.self_attn.indexer.k_cache"] = MLAAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=idx_bytes, dtype=torch.uint8, tokens_per_state=cr,
            alignment=576,
        )
    for li in range(len(crs)):
        specs[f"model.layers.{li}.self_attn.swa_cache"] = SlidingWindowMLASpec(
            block_size=32, num_kv_heads=1, head_size=512, dtype=torch.uint8, sliding_window=128,
            cache_dtype_str="fp8_ds_mla", state_content_bytes=584, alignment=576, model_version="deepseek_v4",
            kv_quant_mode=qm,
        )

    print(f"block_size={block_size} layout={layout.name} indexer_bytes={idx_bytes} main_bytes={main_bytes}")
    print("pages (unpadded -> padded):")
    seen = set()
    for s in specs.values():
        key = (type(s).__name__, s.head_size, s.tokens_per_state, s.block_size)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {type(s).__name__:22s} head={s.head_size:4d} tokens/state={s.tokens_per_state} block={s.block_size:3d}: "
              f"{s.unpadded_page_size_bytes:6d} -> {s.page_size_bytes:6d}")

    groups = get_kv_cache_groups(vc, specs)
    print(f"{len(groups)} kv cache groups:")
    for gi, g in enumerate(groups):
        gs = g.kv_cache_spec
        per = gs.kv_cache_specs if isinstance(gs, UniformTypeKVCacheSpecs) else {ln: gs for ln in g.layer_names}
        tot = sum(per[ln].page_size_bytes for ln in g.layer_names)
        if len(g.layer_names) > 1 or gi < 2:
            names = [ln.replace("model.layers.", "L").replace(".self_attn", "") for ln in g.layer_names]
            print(f"  group {gi}: {len(g.layer_names)} layers, block bytes {tot}: {names[:8]}{' ...' if len(names) > 8 else ''}")
    print(f"  (groups of one SWA layer each: {sum(1 for g in groups if len(g.layer_names) == 1)})")

    kvc = get_kv_cache_config_from_groups(vc, groups, int(args.avail_gib * (1 << 30)))
    toks, conc = get_kv_cache_capacity(vc, kvc)
    print(f"num_blocks={kvc.num_blocks}  capacity {toks:,} tokens, {conc:.2f}x concurrency at {args.max_model_len}")
    print("packed tensors (compressed + indexer caches):")
    for t in kvc.kv_cache_tensors:
        if not any("indexer" in l or l.endswith("self_attn") for l in t.layers):
            continue
        names = [l.replace("model.layers.", "L").replace(".self_attn", "") for l in t.layers]
        group = next(g for g in kvc.kv_cache_groups if t.layers[0] in g.layer_names)
        spec = group.kv_cache_spec.kv_cache_specs[t.layers[0]]
        small = 4
        raw = torch.zeros(t.block_stride * small, dtype=torch.int8)
        tt = KVCacheTensor(size=t.block_stride * small, layers=t.layers, layer_stride=t.layer_stride,
                           block_stride=t.block_stride, offset=t.offset)
        v = create_kv_cache_views(raw, spec, small, layout, tt)[0].squeeze(1)
        print(f"  {str(names):48s} offset={t.offset:7d} page={t.layer_stride:6d} block_stride={t.block_stride}"
              f"  view {tuple(v.shape[1:])} strides {v.stride()}")
    bs = {t.block_stride for t in kvc.kv_cache_tensors}
    for b in sorted(bs):
        print(f"block stride {b}: %16={b % 16} (TMA) %64={b % 64} %512={b % 512} %576={b % 576} %{main_bytes}={b % main_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
