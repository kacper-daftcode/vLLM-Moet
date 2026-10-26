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

4. With the draft's window in prefix caching, `SlidingWindowManager` ("requires block-aligned lookups")
   switches the whole model to block-aligned hits, and the K3 target got **no prefix hit at all** (a 30K
   prompt resent: 0 of 20,348 tokens; multi-turn TTFT = a full prefill). The draft's window is a few blocks of
   per-request state, so the patch takes it out of prefix caching instead (`DraftSlidingWindowSpec`,
   `prefix_cacheable = False`) and caps every prefix hit - local (`KVCacheManager.get_computed_blocks`, the
   connector-path variant) and external (`SimpleCPUOffload` manager) - `sliding_window` tokens below the
   prompt end (`draft_replay_window`). The target recomputes that tail (2048-2175 tokens, ~0.1 s of prefill)
   and the DFlash precompute writes the draft's window from the recomputed hidden states; fine-grained
   hits are back for the MLA / KDA groups. UNRESOLVED (2026-10-26): K3's KDA groups retain their aligned
   state sparsely - at the prompt-end hash boundary of a finished request and at a few block boundaries -
   so a hit capped 2048 tokens below the prompt end finds no KDA state: a 30K prompt resent hits 0 of
   20,350 tokens, later agent turns hit 11,648-15,872 of ~17-20K (where an earlier turn's end state
   happens to lie). The target stays correct and the pool stays at 24.48M tokens, but multi-turn TTFT
   is a prefill. `VLLM_MOET_SWA_DRAFT_REPLAY=0` keeps the window in prefix caching instead: with the
   default 1024-token blocks the window's and the KDA's hit positions never coincide (0 hits); with
   `VLLM_MOET_SWA_DRAFT_BLOCK=128` (block = prefix-match unit) hits land where the retained window
   still covers the KDA boundary - 0 on a resend, 15,872 of 17,307 on a third turn - and the pool drops
   to 19.2M tokens (each 128-token block is padded to the 958 KB MLA page). A proper fix is a
   `SlidingWindowManager` with fine-grained (hash-unit) lookups and copy-on-write for the partial
   block, or the draft window rebuilt from a retained KDA state; neither is attempted here. Until then
   this draft is for single-session / no-prefix-cache serving only.

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
    "                    # prefix-match unit, so the window's blocks hit fine-grained).\n"
    "                    cap = int(os.environ.get(\"VLLM_MOET_SWA_DRAFT_BLOCK\", \"1024\"))\n"
    "                    for candidate in (1024, 512, 256, 128):\n"
    "                        if candidate <= cap and sw_block_size < candidate <= max_block:\n"
    "                            sw_block_size = candidate\n"
    "                            break\n"
    "            return SlidingWindowSpec(\n"
    "                block_size=sw_block_size,\n"
)

# 4. The draft's window as per-request scratch, replayed after a prefix hit.
IFACE_FILE = "v1/kv_cache_interface.py"
IFACE_OLD = (
    "@dataclass(frozen=True, kw_only=True)\n"
    "class SlidingWindowMLASpec(SlidingWindowSpec):\n"
)
IFACE_NEW = (
    f"{MARKER}\n"
    "@dataclass(frozen=True, kw_only=True)\n"
    "class DraftSlidingWindowSpec(SlidingWindowSpec):\n"
    '    """A speculative draft\'s sliding-window cache, kept out of prefix caching.\n'
    "\n"
    "    A SlidingWindowManager only hits on whole blocks, which disables the\n"
    "    fine-grained (hash-unit) hits a hybrid target relies on, and in practice\n"
    "    left the K3 target without any prefix hit. The draft's window is instead\n"
    "    per-request scratch: every prefix hit is capped ``sliding_window`` tokens\n"
    "    below the prompt end (``draft_replay_window``), the target recomputes that\n"
    "    tail and the draft writes its window from the recomputed hidden states.\n"
    '    """\n'
    "\n"
    "    @property\n"
    "    def prefix_cacheable(self) -> bool:\n"
    "        return False\n"
    "\n"
    "\n"
    'def draft_replay_window(kv_cache_config: "KVCacheConfig") -> int:\n'
    '    """Tokens a prefix hit must leave for recomputation so a non-cacheable\n'
    '    sliding-window draft rebuilds its window (0 without such a draft)."""\n'
    "    return max(\n"
    "        (\n"
    "            spec.sliding_window\n"
    "            for group in kv_cache_config.kv_cache_groups\n"
    "            for spec in iter_layer_specs(group.kv_cache_spec)\n"
    "            if isinstance(spec, DraftSlidingWindowSpec)\n"
    "        ),\n"
    "        default=0,\n"
    "    )\n"
    "\n"
    "\n"
    "@dataclass(frozen=True, kw_only=True)\n"
    "class SlidingWindowMLASpec(SlidingWindowSpec):\n"
)

RUNNER_SPEC_OLD = (
    "            for name in self.speculator.draft_attn_layer_names & kv_cache_spec.keys():\n"
    "                kv_cache_spec[name] = replace(kv_cache_spec[name], dcp_sharded=False)\n"
)
RUNNER_SPEC_NEW = (
    "            for name in self.speculator.draft_attn_layer_names & kv_cache_spec.keys():\n"
    "                kv_cache_spec[name] = replace(kv_cache_spec[name], dcp_sharded=False)\n"
    f"                {MARKER}:\n"
    "                # a sliding-window draft cache is per-request scratch, replayed\n"
    "                # after a prefix hit (see DraftSlidingWindowSpec).\n"
    "                import os\n"
    "                from dataclasses import fields\n"
    "\n"
    "                from vllm.v1.kv_cache_interface import (\n"
    "                    DraftSlidingWindowSpec,\n"
    "                    SlidingWindowSpec,\n"
    "                )\n"
    "\n"
    "                spec = kv_cache_spec[name]\n"
    "                # VLLM_MOET_SWA_DRAFT_REPLAY=0 keeps the draft's window in prefix\n"
    "                # caching (needs a block of one prefix-match unit to hit fine-grained).\n"
    "                if (\n"
    "                    type(spec) is SlidingWindowSpec\n"
    "                    and os.environ.get(\"VLLM_MOET_SWA_DRAFT_REPLAY\", \"1\") == \"1\"\n"
    "                ):\n"
    "                    kv_cache_spec[name] = DraftSlidingWindowSpec(\n"
    "                        **{f.name: getattr(spec, f.name) for f in fields(spec) if f.init}\n"
    "                    )\n"
)

MANAGER_FILE = "v1/core/kv_cache_manager.py"
MANAGER_OLD_A = (
    "        max_cache_hit_length = request.num_tokens - 1\n"
    "        computed_blocks, num_new_computed_tokens, num_uncached = (\n"
)
MANAGER_NEW_A = (
    "        max_cache_hit_length = request.num_tokens - 1\n"
    f"        {MARKER}:\n"
    "        # leave the draft's window to be recomputed (DraftSlidingWindowSpec).\n"
    "        max_cache_hit_length -= self._draft_replay_window()\n"
    "        if max_cache_hit_length <= 0:\n"
    "            return self.empty_kv_cache_blocks, 0, 0\n"
    "        computed_blocks, num_new_computed_tokens, num_uncached = (\n"
)
MANAGER_OLD_B = (
    "        computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(\n"
    "            request.block_hashes, request.num_tokens - 1\n"
    "        )\n"
)
MANAGER_NEW_B = (
    f"        {MARKER}\n"
    "        max_cache_hit_length = request.num_tokens - 1 - self._draft_replay_window()\n"
    "        if max_cache_hit_length <= 0:\n"
    "            return self.empty_kv_cache_blocks, 0, 0, False\n"
    "        computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(\n"
    "            request.block_hashes, max_cache_hit_length\n"
    "        )\n"
)
MANAGER_OLD_C = "    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:\n"
MANAGER_NEW_C = (
    f"    {MARKER}\n"
    "    def _draft_replay_window(self) -> int:\n"
    "        window = getattr(self, \"_draft_replay_window_tokens\", None)\n"
    "        if window is None:\n"
    "            from vllm.v1.kv_cache_interface import draft_replay_window\n"
    "\n"
    "            window = draft_replay_window(self.kv_cache_config)\n"
    "            self._draft_replay_window_tokens = window\n"
    "        return window\n"
    "\n"
    "    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:\n"
)

OFFLOAD_FILE = "v1/simple_kv_offload/manager.py"
OFFLOAD_OLD = (
    "        max_hit_len = request.num_tokens - 1 - num_computed_tokens\n"
    "        if max_hit_len <= 0:\n"
    "            return 0, False\n"
)
OFFLOAD_NEW = (
    "        max_hit_len = request.num_tokens - 1 - num_computed_tokens\n"
    f"        {MARKER}:\n"
    "        # the external hit observes the same replay window as the local one.\n"
    "        from vllm.v1.kv_cache_interface import draft_replay_window\n"
    "\n"
    "        max_hit_len -= draft_replay_window(self.cpu_kv_cache_config)\n"
    "        if max_hit_len <= 0:\n"
    "            return 0, False\n"
)

PATCHES = (
    (LAYOUT_FILE, LAYOUT_OLD, LAYOUT_NEW, "layout decided on unified page sizes"),
    (RUNNER_FILE, RUNNER_OLD, RUNNER_NEW, "target graph support over target layers"),
    (ATTN_FILE, ATTN_OLD, ATTN_NEW, "GQA sliding-window draft on trtllm-gen pages"),
    (IFACE_FILE, IFACE_OLD, IFACE_NEW, "DraftSlidingWindowSpec + draft_replay_window"),
    (RUNNER_FILE, RUNNER_SPEC_OLD, RUNNER_SPEC_NEW, "draft sliding-window specs as scratch"),
    (MANAGER_FILE, MANAGER_OLD_C, MANAGER_NEW_C, "replay window helper"),
    (MANAGER_FILE, MANAGER_OLD_A, MANAGER_NEW_A, "local prefix hit capped below the draft window"),
    (MANAGER_FILE, MANAGER_OLD_B, MANAGER_NEW_B, "connector-path local hit capped below the draft window"),
    (OFFLOAD_FILE, OFFLOAD_OLD, OFFLOAD_NEW, "external prefix hit capped below the draft window"),
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
