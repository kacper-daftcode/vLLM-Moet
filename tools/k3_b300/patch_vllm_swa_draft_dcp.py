#!/usr/bin/env python3
"""A replicated sliding-window draft (RedHatAI/Kimi-K3-speculator.dspark) next to a DCP-sharded MLA target.

The RedHat K3 speculator is a 5-layer Qwen3-style GQA draft with a 2048-token sliding window. Under a
DCP-sharded MLA target (K3, TP8 + DCP8, TOKENSPEED_MLA) vLLM v0.32.0 cannot start it:

1. `resolve_kv_cache_layout` sees a sharded target cache and a replicated draft cache with different page
   sizes and requires a block-outer layout (BLHNC) "to pack those groups without equalizing pages". But
   TOKENSPEED_MLA's 32/64-token kernel blocks cannot be carved out of a packed 1664-token block of a
   token-level cache (`AttentionGroup.map_kv_cache_to_kernel_blocks`: "packed ... blocks cannot be split
   into"). The hybrid grouping later pads the draft's page to the MLA page anyway
   (`unify_kv_cache_spec_page_size`: non-MLA attention pages that do not divide the maximum are padded and
   read through a strided view), so after unification there is one page size and a layer-compact layout
   (the one the target runs today) is admissible. The patch decides the block-outer requirement on the
   unified page sizes.

2. The target runner computes its CUDA graph support over every attention group, the draft's included.
   The draft's FlashInfer builder, asked under the target's config (DCP 8), answers
   UNIFORM_SINGLE_TOKEN_DECODE, and the target's FULL_AND_PIECEWISE graphs collapse to NONE ("is not
   supported with spec-decode for attention backend FlashInferBackend") - although the draft's attention
   never runs in the target's graphs (the speculator has its own graph manager and its own config, DCP 1
   for a replicated draft). The patch narrows the target's support query to the target's layers, the set
   the runner already derives for adaptive verification and CP checks.

3. The sliding-window draft layer picks its block size from the kernel blocks its backend advertises
   (`Attention.get_kv_cache_spec`). FlashInfer advertises the trtllm-gen pages (128..1024) only when the
   *current* model config is GQA on Blackwell; at KV-spec time that is the MLA target (ratio 1), so the
   draft gets 64-token blocks. Each is then padded to the 958 KB MLA page (`unify_kv_cache_spec_page_size`),
   and `--prefix-match-unit 128` rejects a 64-token group: with it the per-request admission for the
   2048-token window is ~290 padded blocks (1.4 GB) and the pool reads 15.6M tokens instead of ~25M. The
   patch lets a GQA FlashInfer layer on the SM100 family use the largest trtllm-gen page that fits the
   primary block, judged on the layer's own heads (1024 here: 13 admission blocks, 2-3 held in decode).

Idempotent, anchor-based. Usage:
    python3 patch_vllm_swa_draft_dcp.py [--root PATH] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "# [vllm-moet] replicated SWA draft under a DCP-sharded MLA target"

LAYOUT_FILE = "v1/attention/backends/utils.py"
LAYOUT_OLD = (
    "    page_sizes = {spec.page_size_bytes for spec in kv_cache_specs}\n"
    "    if len(dcp_sharding) > 1 and len(page_sizes) > 1:\n"
)
LAYOUT_NEW = (
    "    page_sizes = {spec.page_size_bytes for spec in kv_cache_specs}\n"
    f"    {MARKER}:\n"
    "    # pages the hybrid grouping pads to one size (a GQA draft page next to the\n"
    "    # MLA page) need no block-outer packing, and a layer-compact layout keeps\n"
    "    # the MLA backend's kernel-block split, which packed blocks cannot offer.\n"
    "    if len(dcp_sharding) > 1 and len(page_sizes) > 1:\n"
    "        from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size\n"
    "\n"
    "        try:\n"
    "            unified = unify_kv_cache_spec_page_size(\n"
    "                {str(i): spec for i, spec in enumerate(kv_cache_specs)}\n"
    "            )\n"
    "        except NotImplementedError:\n"
    "            pass\n"
    "        else:\n"
    "            page_sizes = {spec.page_size_bytes for spec in unified.values()}\n"
    "    if len(dcp_sharding) > 1 and len(page_sizes) > 1:\n"
)

RUNNER_FILE = "v1/worker/gpu/model_runner.py"
RUNNER_OLD = (
    "        additional_attn_cg_support = self.model_state.get_additional_cg_support()\n"
    "        attn_cg_support = attn_cg_support.narrow(*additional_attn_cg_support)\n"
)
RUNNER_NEW = (
    f"        {MARKER}:\n"
    "        # the draft's attention runs under the speculator's own graph manager\n"
    "        # and config (DCP 1 for a replicated draft); its backend's support under\n"
    "        # the target's config must not decide the target's graph mode.\n"
    "        if target_attn_layer_names is not None:\n"
    "            from vllm.v1.worker.gpu.attn_utils import get_attn_cg_support\n"
    "\n"
    "            attn_cg_support = get_attn_cg_support(\n"
    "                self.attn_groups, self.vllm_config, target_attn_layer_names\n"
    "            )\n"
    "        additional_attn_cg_support = self.model_state.get_additional_cg_support()\n"
    "        attn_cg_support = attn_cg_support.narrow(*additional_attn_cg_support)\n"
)

ATTN_FILE = "model_executor/layers/attention/attention.py"
ATTN_OLD = (
    "            sw_block_size = _largest_kernel_block_within(\n"
    "                self.attn_backend,\n"
    "                sw_per_token,\n"
    "                page_budget,\n"
    "                block_size,\n"
    "                kv_cache_spec,\n"
    "            )\n"
    "            return SlidingWindowSpec(\n"
    "                block_size=sw_block_size,\n"
)
ATTN_NEW = (
    "            sw_block_size = _largest_kernel_block_within(\n"
    "                self.attn_backend,\n"
    "                sw_per_token,\n"
    "                page_budget,\n"
    "                block_size,\n"
    "                kv_cache_spec,\n"
    "            )\n"
    f"            {MARKER}:\n"
    "            # FlashInfer advertises its trtllm-gen pages (128..1024) from the\n"
    "            # *current* model config's head ratio, which at KV-spec time is the\n"
    "            # target's (MLA, ratio 1), not this GQA draft layer's; a 64-token block\n"
    "            # padded to the MLA page costs ~290 admission blocks per request.\n"
    "            if (\n"
    "                self.attn_backend.get_name() == \"FLASHINFER\"\n"
    "                and self.num_kv_heads > 0\n"
    "                and self.num_heads // self.num_kv_heads > 1\n"
    "                and current_platform.is_device_capability_family(100)\n"
    "            ):\n"
    "                from vllm.config import set_current_vllm_config\n"
    "                from vllm.utils.flashinfer import can_use_trtllm_attention\n"
    "\n"
    "                # No config is current at KV-spec time; the check reads\n"
    "                # attention_config.use_trtllm_attention off the current one.\n"
    "                with set_current_vllm_config(vllm_config):\n"
    "                    large_pages = can_use_trtllm_attention(\n"
    "                        self.num_heads, self.num_kv_heads\n"
    "                    )\n"
    "                if large_pages:\n"
    "                    max_block = page_budget // sw_per_token\n"
    "                    for candidate in (1024, 512, 256, 128):\n"
    "                        if sw_block_size < candidate <= max_block:\n"
    "                            sw_block_size = candidate\n"
    "                            break\n"
    "            return SlidingWindowSpec(\n"
    "                block_size=sw_block_size,\n"
)

PATCHES = (
    (LAYOUT_FILE, LAYOUT_OLD, LAYOUT_NEW, "layout decided on unified page sizes"),
    (RUNNER_FILE, RUNNER_OLD, RUNNER_NEW, "target graph support over target layers"),
    (ATTN_FILE, ATTN_OLD, ATTN_NEW, "GQA sliding-window draft on trtllm-gen pages"),
)


def patch_text(src: str, old: str, new: str) -> str | None:
    if MARKER in src:
        return None
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"anchor found {n} times (expected 1):\n{old}")
    return src.replace(old, new, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    for rel, old, new, what in PATCHES:
        path = args.root / rel
        src = path.read_text()
        patched = patch_text(src, old, new)
        if patched is None:
            print(f"{path}: already patched")
            continue
        compile(patched, str(path), "exec")
        if args.check:
            print(f"{path}: patch applies cleanly ({what}; not written)")
            continue
        path.write_text(patched)
        print(f"{path}: patched ({what})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
