#!/usr/bin/env python3
"""One host copy of the DeepSeek-V4.1 KV instead of one per TP rank in vLLM's KV offload.

vLLM main's OffloadingConnector deduplicates TP-replicated MLA KV (`replicated_layout`,
vllm#48906 / #50301): one slot in the shared /dev/shm region, rank 0 stores, every rank
loads the same bytes, and the fs tier names that single copy TP-independently. The gate
in `distributed/kv_transfer/kv_connector/v1/offloading/config.py` admits only a single
bare `MLAAttentionSpec` group, so DeepSeek-V4.1 falls back to world_size copies: its KV
config has ten groups (8 sliding-window MLA groups of 32 tokens, the compressed group of
128 = MLA latent + DSA indexer K in one UniformTypeKVCacheSpecs, and the compressor ring,
a CircularBufferSpec of 8 rows) in a packed block-outermost layout.

Only the compressed group is offloaded (`prefix_cacheable_group_ids`, vllm#57145: the
bounded-replay SWA groups and the ring are scratch and never leave the GPU), and a packed
block holds the pages of the one group that owns it. So the gate should look at what is
copied: every layer of every offloaded group an MLA spec with one KV head. vllm#57652
widens the gate to multi-group MLA but requires *all* groups to be MLA, which V4.0 passes
(its compressor state is a SlidingWindowMLASpec) and V4.1 does not (CircularBufferSpec);
if that PR is already in the file, its call is narrowed to the offloaded groups.

Measured on DeepSeek-V4.1-Flash, TP4, nvfp4_ds_mla + MXFP4 indexer: 18,219 of 18,219
fs-tier block files written by three server processes hold four byte-identical rank
slots (tp_copies_check.py); the host cost drops from 3.58 KB to 896 B per token.

Idempotent, anchor-based. Usage:
    python3 patch_vllm_offload_replicated_mla.py [--file PATH] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/"
    "kv_connector/v1/offloading/config.py"
)
MARKER = "# [vllm-moet] replicated MLA layout gated on the offloaded groups"

HELPER_ANCHOR = "\n\ndef build_offloading_config(\n"
HELPER = (
    "\n\n"
    f"{MARKER}\n"
    "_REPLICATED_MLA_LAYER_TYPES = (MLAAttentionSpec, SlidingWindowMLASpec)\n"
    "\n"
    "\n"
    "def _offloaded_groups_are_replicated_mla(\n"
    "    selected_groups: tuple[tuple[int, \"KVCacheGroupSpec\"], ...],\n"
    ") -> bool:\n"
    "    \"\"\"Whether every layer of every offloaded group holds an MLA latent.\n"
    "\n"
    "    MLA keeps one latent KV head that each TP rank computes in full, so the\n"
    "    ranks' pages are the same bytes and one host copy serves all of them.\n"
    "    Exact types: MLA subclasses with other contents fail closed.\n"
    "    \"\"\"\n"
    "    layer_specs = [\n"
    "        spec\n"
    "        for _, group in selected_groups\n"
    "        for spec in iter_layer_specs(group.kv_cache_spec)\n"
    "    ]\n"
    "    return bool(layer_specs) and all(\n"
    "        isinstance(spec, AttentionSpec)\n"
    "        and type(spec) in _REPLICATED_MLA_LAYER_TYPES\n"
    "        and spec.num_kv_heads == 1\n"
    "        for spec in layer_specs\n"
    "    )\n"
    "\n"
    "\n"
    "def build_offloading_config(\n"
)

# vLLM main 0961bbae .. f9ad9dd6 (2026-09-23): single bare MLAAttentionSpec group.
OLD_GATE = (
    "        vllm_config.model_config.use_mla\n"
    "        # Exact type: fail closed on wrappers and sliding-window variants.\n"
    "        and type(single_group_spec) is MLAAttentionSpec\n"
    "        # Page accounting: one MLA page per layer, no packed/mixed rows.\n"
    "        and worker_kv_bytes_per_block > 0\n"
    "        and worker_kv_bytes_per_block\n"
    "        == single_group_spec.page_size_bytes\n"
    "        * len(kv_cache_config.kv_cache_groups[0].layer_names)\n"
)
NEW_GATE = (
    "        vllm_config.model_config.use_mla\n"
    f"        {MARKER}:\n"
    "        # only they are copied, and a packed block holds the pages of the one\n"
    "        # group that owns it; scratch groups (the V4.1 compressor ring,\n"
    "        # bounded-replay SWA) never leave the GPU.\n"
    "        and _offloaded_groups_are_replicated_mla(selected_groups)\n"
    "        and worker_kv_bytes_per_block > 0\n"
)

# vllm#57652 (open 2026-09-23): every KV cache group MLA, scratch groups included.
PR57652_GATE = "        and _all_groups_are_replicated(kv_cache_config.kv_cache_groups)\n"
PR57652_NEW = (
    f"        {MARKER}:\n"
    "        # scratch groups (the V4.1 compressor ring) never leave the GPU.\n"
    "        and _all_groups_are_replicated([group for _, group in selected_groups])\n"
)


def patch_text(src: str) -> tuple[str, str]:
    if MARKER in src:
        return src, "already patched"
    if src.count(PR57652_GATE) == 1:
        return src.replace(PR57652_GATE, PR57652_NEW, 1), "vllm#57652 gate narrowed"
    for anchor in (OLD_GATE, HELPER_ANCHOR):
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"anchor found {n} times (expected 1):\n{anchor}")
    src = src.replace(OLD_GATE, NEW_GATE, 1)
    src = src.replace(HELPER_ANCHOR, HELPER, 1)
    return src, "gate on the offloaded groups"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    patched, what = patch_text(src)
    if patched is src:
        print(f"{args.file}: {what}")
        return 0
    compile(patched, str(args.file), "exec")
    if args.check:
        print(f"{args.file}: patch applies cleanly ({what}; not written)")
        return 0
    args.file.write_text(patched)
    print(f"{args.file}: patched ({what})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
