#!/usr/bin/env python3
"""A replicated sliding-window draft (RedHatAI/Kimi-K3-speculator.dspark) next to a DCP-sharded MLA target.

The RedHat K3 speculator is a 5-layer Qwen3-style GQA draft with a 2048-token sliding window. Under a
DCP-sharded MLA target (K3, TP8 + DCP8, TOKENSPEED_MLA) vLLM v0.32.0 cannot start it, and once started it
loses prefix caching:

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

4. `SlidingWindowManager` only hits on whole blocks ("requires block-aligned lookups"), so with the
   draft's window in prefix caching the hybrid coordinator gives up hash-unit (fine-grained) hits for the
   whole model and every hit must land on a multiple of the scheduler block (LCM of 13,312-token MLA
   blocks, 1,664-token KDA blocks and 1,024-token window blocks = 13,312): the K3 target got no prefix
   hit at all (a 30K prompt resent: 0 of 20,348 tokens). K3's KDA groups retain their aligned state only
   where the engine can commit it - the prompt-end hash boundary (`get_mamba_prefill_checkpoint_position`)
   and the 1,664-token chunk ends - and the MLA groups retain the same prompt-end boundary as a partial
   entry; a hit exists only where every group has state, i.e. at a finished request's prompt-end hash
   boundary. So the window must be reachable exactly there. The patch gives `SlidingWindowManager`
   fine-grained lookups (`supports_fine_grained_hash_lookup`): a hit may end at any hash boundary P whose
   window blocks are cached - whole blocks under their full-block hash, the block P falls in under the
   prefix hash at P, which `cache_blocks` registers at the prompt's final hash boundary the way
   `FullAttentionManager._cache_partial_tail_block` does (and re-registers after the block is promoted to
   a full one). The partial tail block is shared copy-on-write through the base class's
   `_partial_hit_reqs` path (one reserved block, one block copy on the worker), exactly like the MLA
   groups' partial tails. The window that a replaying request needs (the blocks before the prompt-end
   boundary) is kept referenced by its request for the request's lifetime instead of being nulled once it
   leaves the attention window, so it ages out of the LRU with the request's MLA/KDA blocks and not during
   a long decode; it fits inside the per-request admission cap (`SlidingWindowSpec.
   max_admission_blocks_per_request`: window + one in-flight chunk, 13 blocks here, 2-3 held in decode).
   Everything else (block-aligned models, the dense-attention path, Gemma-style full+SWA hybrids without a
   Mamba "align" group) keeps the upstream block-aligned code path: the fine-grained branch is entered only
   when the coordinator aligns hits below the block size, which needs a Mamba "align" group.
   Supersedes the 2026-10-25/26 variant (the window as per-request scratch with every hit capped
   `sliding_window` tokens below the prompt end), which found no KDA state at the capped position.

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
    "                    import os\n"
    "\n"
    "                    max_block = page_budget // sw_per_token\n"
    "                    # VLLM_MOET_SWA_DRAFT_BLOCK caps the choice (e.g. 128 = the\n"
    "                    # prefix-match unit; the default 1024 is the largest page).\n"
    "                    cap = int(os.environ.get(\"VLLM_MOET_SWA_DRAFT_BLOCK\", \"1024\"))\n"
    "                    for candidate in (1024, 512, 256, 128):\n"
    "                        if candidate <= cap and sw_block_size < candidate <= max_block:\n"
    "                            sw_block_size = candidate\n"
    "                            break\n"
    "            return SlidingWindowSpec(\n"
    "                block_size=sw_block_size,\n"
)

# 4. SlidingWindowManager with hash-unit (fine-grained) hits and a copy-on-write partial tail.
SWA_FILE = "v1/core/single_type_kv_cache_manager.py"
SWA_CLASS_OLD = (
    "class SlidingWindowManager(SingleTypeKVCacheManager):\n"
    "    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:\n"
    "        super().__init__(kv_cache_spec, **kwargs)\n"
    "        self.sliding_window = kv_cache_spec.sliding_window\n"
)
SWA_CLASS_NEW = (
    "class SlidingWindowManager(SingleTypeKVCacheManager):\n"
    f"    {MARKER}:\n"
    "    # Hash-unit (fine-grained) lookups. Next to a Mamba \"align\" group the\n"
    "    # hybrid coordinator aligns hits to the hash unit; a manager that only\n"
    "    # hits on whole blocks would force every group back to scheduler-block\n"
    "    # aligned hits (the LCM of the group block sizes), which for Kimi-K3 with\n"
    "    # a sliding-window draft meant no hit at all. A hit may end inside a\n"
    "    # block; its partial tail is shared copy-on-write, as for full attention\n"
    "    # (_find_longest_fine_grained_hit, cache_blocks, remove_skipped_blocks).\n"
    "    supports_fine_grained_hash_lookup: ClassVar[bool] = True\n"
    "\n"
    "    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:\n"
    "        super().__init__(kv_cache_spec, **kwargs)\n"
    "        self.sliding_window = kv_cache_spec.sliding_window\n"
)

SWA_FIND_OLD = (
    "        # The number of contiguous blocks needed for a prefix cache hit.\n"
    "        sliding_window_contiguous_blocks = cls._contiguous_blocks_for_hit(\n"
    "            kv_cache_spec.sliding_window, kv_cache_spec.block_size, drop_eagle_block\n"
    "        )\n"
)
SWA_FIND_NEW = (
    f"        {MARKER}:\n"
    "        # fine-grained mode (hash unit below the block): resolve_block_hashes\n"
    "        # kept the raw hash-unit list, and a hit may end at any hash boundary.\n"
    "        if (\n"
    "            alignment_tokens < kv_cache_spec.block_size\n"
    "            and kv_cache_spec.block_size % alignment_tokens == 0\n"
    "        ):\n"
    "            # list or lazy BlobBlockHashes view\n"
    "            assert isinstance(block_hashes, Sequence)\n"
    "            return cls._find_longest_fine_grained_hit(\n"
    "                block_hashes,\n"
    "                max_length,\n"
    "                kv_cache_group_ids,\n"
    "                block_pool,\n"
    "                kv_cache_spec,\n"
    "                drop_eagle_block,\n"
    "                alignment_tokens,\n"
    "            )\n" + SWA_FIND_OLD
)

SWA_METHODS_OLD = (
    "    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:\n"
    '        """Get the number of tokens that will be skipped for attention computation.\n'
    "\n"
    "        For sliding window, this corresponds to the tokens that are prior to\n"
    "        the current sliding window.\n"
)
SWA_METHODS_NEW = (
    f"    {MARKER}\n"
    "    @classmethod\n"
    "    def _find_longest_fine_grained_hit(\n"
    "        cls,\n"
    "        block_hashes: Sequence,\n"
    "        max_length: int,\n"
    "        kv_cache_group_ids: list[int],\n"
    "        block_pool: BlockPool,\n"
    "        kv_cache_spec: SlidingWindowSpec,\n"
    "        drop_eagle_block: bool,\n"
    "        alignment_tokens: int,\n"
    "    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:\n"
    '        """Longest hit ending at a hash boundary, not only at a block boundary.\n'
    "\n"
    "        A hit of ``hit_length`` tokens needs the blocks holding the window\n"
    "        ``[hit_length - sliding_window + 1, hit_length)`` cached: whole blocks\n"
    "        under their full-block hash, and the block the hit ends in under a\n"
    "        prefix hash at or past ``hit_length`` (its full-block hash, or a\n"
    "        partial entry ``cache_blocks`` registered at a prompt-end boundary);\n"
    "        any such key proves the block holds the prefix below it. With the\n"
    "        EAGLE drop the proof must reach one unit past the hit (the key at\n"
    "        ``hit_length + unit``, or the next block when the hit is block\n"
    "        aligned); the coordinator re-queries an already verified eagle group\n"
    "        without the drop, which the same keys satisfy. Candidates are probed\n"
    "        from the longest down, each block's keys once (``scale`` lookups per\n"
    "        block, i.e. one per hash unit), then one per window block on a hit.\n"
    '        """\n'
    "        block_size = kv_cache_spec.block_size\n"
    "        unit = alignment_tokens\n"
    "        scale = block_size // unit\n"
    "        drop = unit if drop_eagle_block else 0\n"
    "        window = kv_cache_spec.sliding_window\n"
    "        num_groups = len(kv_cache_group_ids)\n"
    "        num_units = len(block_hashes)\n"
    "        empty: tuple[list[KVCacheBlock], ...] = tuple([] for _ in range(num_groups))\n"
    "        max_hit = min(max_length - drop, num_units * unit) // unit * unit\n"
    "        if max_hit <= 0:\n"
    "            return empty, 0\n"
    "\n"
    "        # Highest registered prefix hash on a block, (tokens, blocks per group),\n"
    "        # None when the block is not cached; probed once per block.\n"
    "        block_keys: dict[int, tuple[int, list[KVCacheBlock]] | None] = {}\n"
    "\n"
    "        def block_key(idx: int) -> tuple[int, list[KVCacheBlock]] | None:\n"
    "            if idx in block_keys:\n"
    "                return block_keys[idx]\n"
    "            found = None\n"
    "            for u in range(min((idx + 1) * scale, num_units), idx * scale, -1):\n"
    "                cached = block_pool.get_cached_block(\n"
    "                    block_hashes[u - 1], kv_cache_group_ids\n"
    "                )\n"
    "                if cached is not None:\n"
    "                    found = (u * unit, cached)\n"
    "                    break\n"
    "            block_keys[idx] = found\n"
    "            return found\n"
    "\n"
    "        for hit_length in range(max_hit, 0, -unit):\n"
    "            last_block = (hit_length - 1) // block_size\n"
    "            key = block_key(last_block)\n"
    "            if key is None or key[0] < hit_length:\n"
    "                continue\n"
    "            if drop and key[0] < hit_length + unit:\n"
    "                # A block-aligned hit is proven by any key on the next block.\n"
    "                if (\n"
    "                    hit_length != (last_block + 1) * block_size\n"
    "                    or block_key(last_block + 1) is None\n"
    "                ):\n"
    "                    continue\n"
    "            first_block = max(0, hit_length - window + 1) // block_size\n"
    "            window_blocks: list[list[KVCacheBlock]] = []\n"
    "            for idx in range(first_block, last_block):\n"
    "                full = block_key(idx)\n"
    "                if full is None or full[0] < (idx + 1) * block_size:\n"
    "                    break\n"
    "                window_blocks.append(full[1])\n"
    "            else:\n"
    "                window_blocks.append(key[1])\n"
    "                computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(\n"
    "                    [block_pool.null_block] * first_block for _ in range(num_groups)\n"
    "                )\n"
    "                for cached in window_blocks:\n"
    "                    for computed, block in zip(computed_blocks, cached):\n"
    "                        computed.append(block)\n"
    "                return computed_blocks, hit_length\n"
    "        return empty, 0\n"
    "\n"
    "    @property\n"
    "    def _fine_grained_hits(self) -> bool:\n"
    '        """Whether the coordinator serves hash-unit hits inside this group\'s blocks."""\n'
    "        return (\n"
    "            self.enable_caching\n"
    "            and self.cache_hit_alignment_tokens < self.block_size\n"
    "            and self.block_size % self.cache_hit_alignment_tokens == 0\n"
    "        )\n"
    "\n"
    "    def _replay_window(self, num_prompt_tokens: int) -> tuple[int, int] | None:\n"
    '        """(proof boundary, first window block) of a prompt\'s replayable tail.\n'
    "\n"
    "        The proof boundary is the prompt's final hash boundary (the one a\n"
    "        resend or a longer sibling matches; under the EAGLE drop the hit then\n"
    "        ends one unit below it), the first window block the lowest block that\n"
    "        hit needs. None when the prompt has no such boundary.\n"
    '        """\n'
    "        hash_block_size = self.block_pool.hash_block_size\n"
    "        token_limit = num_prompt_tokens if self.use_eagle else num_prompt_tokens - 1\n"
    "        boundary = token_limit // hash_block_size * hash_block_size\n"
    "        hit_end = boundary - (hash_block_size if self.use_eagle else 0)\n"
    "        if hit_end <= 0:\n"
    "            return None\n"
    "        first_block = max(0, hit_end - self.sliding_window + 1) // self.block_size\n"
    "        return boundary, first_block\n"
    "\n"
    "    def cache_blocks(\n"
    "        self,\n"
    "        request: Request,\n"
    "        num_tokens: int,\n"
    "        retention_interval: int | None = None,\n"
    "        *,\n"
    "        replay_boundaries: Sequence[int],\n"
    "    ) -> None:\n"
    "        super().cache_blocks(\n"
    "            request,\n"
    "            num_tokens,\n"
    "            retention_interval=retention_interval,\n"
    "            replay_boundaries=replay_boundaries,\n"
    "        )\n"
    "        if self.kv_cache_spec.prefix_cacheable and self._fine_grained_hits:\n"
    "            self._cache_partial_tail_block(request, num_tokens)\n"
    "\n"
    "    def _cache_partial_tail_block(self, request: Request, num_tokens: int) -> None:\n"
    '        """Register the prompt\'s final hash boundary as a partial entry of the\n'
    "        block it falls in, so a resend or a longer sibling hits there (as\n"
    "        ``FullAttentionManager._cache_partial_tail_block`` does). Re-registered\n"
    "        on every call: promotion to a full block drops the entry.\n"
    '        """\n'
    "        replay = self._replay_window(request.num_prompt_tokens)\n"
    "        if replay is None:\n"
    "            return\n"
    "        boundary, _ = replay\n"
    "        if boundary > num_tokens or boundary % self.block_size == 0:\n"
    "            return\n"
    "        blocks = self.req_to_blocks[request.request_id]\n"
    "        block_idx = boundary // self.block_size\n"
    "        if block_idx >= len(blocks) or blocks[block_idx].is_null:\n"
    "            return\n"
    "        self.block_pool.cache_partial_block(\n"
    "            request=request,\n"
    "            block=blocks[block_idx],\n"
    "            num_tokens=boundary,\n"
    "            kv_cache_group_id=self.kv_cache_group_id,\n"
    "            block_size=self.block_size,\n"
    "        )\n"
    "\n"
    "    def remove_skipped_blocks(\n"
    "        self,\n"
    "        request_id: str,\n"
    "        processed_computed_tokens: int,\n"
    "        num_prompt_tokens: int | None = None,\n"
    "    ) -> None:\n"
    "        replay = (\n"
    "            self._replay_window(num_prompt_tokens)\n"
    "            if num_prompt_tokens is not None and self._fine_grained_hits\n"
    "            else None\n"
    "        )\n"
    "        if replay is None:\n"
    "            super().remove_skipped_blocks(\n"
    "                request_id, processed_computed_tokens, num_prompt_tokens\n"
    "            )\n"
    "            return\n"
    "        # Keep the window a replaying request needs (the blocks before the\n"
    "        # prompt-end boundary) referenced for the request's lifetime: it then\n"
    "        # enters the eviction queue with the request's other groups' blocks\n"
    "        # instead of ageing out during a long decode. Within the admission\n"
    "        # cap: the window plus one in-flight chunk, of which decode holds 2-3.\n"
    "        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)\n"
    "        if num_skipped_tokens <= 0 or request_id not in self.req_to_blocks:\n"
    "            return\n"
    "        boundary, keep_start = replay\n"
    "        keep_end = cdiv(boundary, self.block_size)\n"
    "        num_skipped_blocks = min(\n"
    "            num_skipped_tokens // self.block_size,\n"
    "            len(self.req_to_blocks[request_id]),\n"
    "        )\n"
    "        self._remove_blocks_in_range(request_id, keep_end, num_skipped_blocks)\n"
    "        self._remove_blocks_in_range(\n"
    "            request_id, 0, min(keep_start, num_skipped_blocks)\n"
    "        )\n"
    "\n" + SWA_METHODS_OLD
)

PATCHES = (
    (LAYOUT_FILE, LAYOUT_OLD, LAYOUT_NEW, "layout decided on unified page sizes"),
    (RUNNER_FILE, RUNNER_OLD, RUNNER_NEW, "target graph support over target layers"),
    (ATTN_FILE, ATTN_OLD, ATTN_NEW, "GQA sliding-window draft on trtllm-gen pages"),
    (SWA_FILE, SWA_CLASS_OLD, SWA_CLASS_NEW, "SlidingWindowManager: fine-grained lookups"),
    (SWA_FILE, SWA_FIND_OLD, SWA_FIND_NEW, "SlidingWindowManager: fine-grained hit branch"),
    (SWA_FILE, SWA_METHODS_OLD, SWA_METHODS_NEW, "SlidingWindowManager: partial tail, CoW, retention"),
)


def patch_text(src: str, old: str, new: str) -> str | None:
    # Several hunks share a file: a hunk is applied when its own text is absent.
    if new in src:
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
    texts: dict[Path, str] = {}
    for rel, old, new, what in PATCHES:
        path = args.root / rel
        src = texts.get(path) or path.read_text()
        patched = patch_text(src, old, new)
        if patched is None:
            print(f"{path}: already patched ({what})")
            continue
        compile(patched, str(path), "exec")
        texts[path] = patched
        print(f"{path}: {'patch applies cleanly' if args.check else 'patched'} ({what})")
    if not args.check:
        for path, text in texts.items():
            path.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
