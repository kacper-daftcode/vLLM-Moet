#!/usr/bin/env python3
"""The KV offload dedup gate on the DeepSeek-V4.1 cache layout (patch_vllm_offload_replicated_mla.py).

Builds the KV cache config the served engine logs (group sizes [32 x 8, 128, 8]: eight
bounded-replay sliding-window MLA groups, the compressed group = MLA latent + DSA indexer K
of the kv-source layers 2/8/14/20 in one UniformTypeKVCacheSpecs, the compressor ring as a
CircularBufferSpec; one packed block-outermost allocation of 114,688 B per block) and runs
vLLM's own build_offloading_config, TieringOffloadingSpec sizing and fs-tier FileMapper on it:

  gate      replicated_layout at TP 2/4/8 (only the compressed group is offloaded);
  closed    stays off for use_mla=False, TP1, a non-MLA or a HiddenStateCacheSpec layer in
            an offloaded group, a GQA group offloaded next to the MLA one, PP2 and the ray
            executor;
  sizing    64 GiB of host buffer = world_size times the chunks of the per-rank layout;
  identity  the fs tier names the single copy TP-independently (no files of the four-copy
            layout are read back as one-copy rows).

Needs no GPU. Run inside the serving image:
  python3 test_offload_replicated_mla.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import torch

from vllm.config import KVTransferConfig, ParallelConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

BLOCK_BYTES = 114_688  # per rank and 128-token block on the served image (fs-tier slot)
KV_SOURCE_LAYERS = (2, 8, 14, 20)
HOST_BYTES = 64 << 30


def vllm_config(
    tp: int = 4,
    pp: int = 1,
    use_mla: bool = True,
    backend: Any = "mp",
) -> VllmConfig:
    config = MagicMock()
    config.cache_config.block_size = 128
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_match_unit = None
    config.cache_config.cache_dtype = "nvfp4_ds_mla"
    config.cache_config.prefix_cache_retention_interval = None
    config.model_config.model = "/model"
    config.model_config.use_mla = use_mla
    config.model_config.get_total_num_kv_heads.return_value = 1
    with patch.object(current_platform, "device_count", return_value=tp * pp):
        config.parallel_config = ParallelConfig(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        )
    config.parallel_config.distributed_executor_backend = backend
    config.parallel_config.nnodes = 1
    config.kv_events_config = None
    config.use_v2_model_runner = True
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        engine_id="test-engine",
        kv_connector_extra_config={
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": HOST_BYTES,
            "offload_prompt_only": False,
            "secondary_tiers": [{"type": "fs", "root_dir": "/tmp/kvcache"}],
        },
    )
    return cast(VllmConfig, config)


def compressed_specs(extra: dict[str, KVCacheSpec] | None = None) -> dict[str, KVCacheSpec]:
    specs: dict[str, KVCacheSpec] = {}
    for layer in KV_SOURCE_LAYERS:
        prefix = f"language_model.model.layers.{layer}.attn"
        specs[prefix] = MLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=512, dtype=torch.uint8,
            tokens_per_state=2,
        )
        specs[f"{prefix}.indexer.k_cache"] = MLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=68, dtype=torch.uint8,
            tokens_per_state=2,
        )
    specs.update(extra or {})
    return specs


def v41_kv_cache_config(
    compressed: dict[str, KVCacheSpec] | None = None,
    extra_groups: list[KVCacheGroupSpec] | None = None,
    num_blocks: int = 64,
) -> KVCacheConfig:
    groups = []
    layer = 0
    for _ in range(8):
        names = [f"language_model.model.layers.{layer + i}.attn.swa_cache" for i in range(5)]
        layer += 5
        spec = SlidingWindowMLASpec(
            block_size=32, num_kv_heads=1, head_size=528, dtype=torch.uint8,
            sliding_window=128, bounded_replay=True,
        )
        groups.append(KVCacheGroupSpec(names, spec))
    compressed = compressed if compressed is not None else compressed_specs()
    groups.append(
        KVCacheGroupSpec(
            list(compressed), UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=compressed)
        )
    )
    groups += extra_groups or []
    ring = {
        f"language_model.model.layers.{i}.attn.compressor.state_cache": CircularBufferSpec(
            block_size=8, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32
        )
        for i in KV_SOURCE_LAYERS
    }
    groups.append(
        KVCacheGroupSpec(list(ring), UniformTypeKVCacheSpecs(block_size=8, kv_cache_specs=ring))
    )
    layers = [name for group in groups for name in group.layer_names]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=BLOCK_BYTES * num_blocks,
                layers=layers,
                layer_stride=0,
                block_stride=BLOCK_BYTES,
            )
        ],
        kv_cache_groups=groups,
    )


def main() -> int:
    results: list[tuple[bool, str]] = []

    def check(ok: bool, what: str) -> None:
        results.append((ok, what))
        print(f"  {'[ok ]' if ok else '[BAD]'} {what}")

    kv = v41_kv_cache_config()
    print(f"groups {[g.kv_cache_spec.block_size for g in kv.kv_cache_groups]}, "
          f"offloaded {list(kv.prefix_cacheable_group_ids)}")

    print("gate")
    for tp in (2, 4, 8):
        cfg = build_offloading_config(vllm_config(tp=tp), kv)
        check(cfg.replicated_layout, f"TP{tp}: replicated_layout")
    cfg = build_offloading_config(vllm_config(), kv)
    check([g.group_id for g in cfg.groups] == [8], "only the compressed group is offloaded")
    check(cfg.worker_kv_bytes_per_block == BLOCK_BYTES, f"{BLOCK_BYTES} B per block and rank")

    print("closed")
    gqa = KVCacheGroupSpec(
        ["mtp.attn"], FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    )
    hidden = HiddenStateCacheSpec(block_size=128, num_kv_heads=1, head_size=512, dtype=torch.bfloat16)
    non_mla = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=512, dtype=torch.uint8)
    for label, config, kv_config in (
        ("use_mla=False", vllm_config(use_mla=False), kv),
        ("TP1", vllm_config(tp=1), kv),
        ("PP2", vllm_config(tp=2, pp=2), kv),
        ("ray executor", vllm_config(backend="ray"), kv),
        ("GQA group offloaded too", vllm_config(), v41_kv_cache_config(extra_groups=[gqa])),
        ("non-MLA layer in the compressed group",
         vllm_config(), v41_kv_cache_config(compressed_specs({"x.attn": non_mla}))),
        ("HiddenStateCacheSpec layer in the compressed group",
         vllm_config(), v41_kv_cache_config(compressed_specs({"x.hidden": hidden}))),
    ):
        check(not build_offloading_config(config, kv_config).replicated_layout, f"{label}: per-rank layout")

    print("sizing")
    offloading_config = build_offloading_config(vllm_config(), kv)
    single = TieringOffloadingSpec(offloading_config)
    per_rank = TieringOffloadingSpec(replace(offloading_config, replicated_layout=False))
    check(per_rank.num_chunks == 149_796, f"per-rank layout: {per_rank.num_chunks} chunks (served log: 149796)")
    check(single.num_chunks >= 4 * per_rank.num_chunks,
          f"single copy: {single.num_chunks} chunks = {single.num_chunks * 128 / 1e6:.1f}M tokens "
          f"(per rank: {per_rank.num_chunks * 128 / 1e6:.1f}M)")
    check(single.cpu_page_size_per_worker == single.kv_bytes_per_chunk == BLOCK_BYTES,
          f"row = one {BLOCK_BYTES} B copy")

    print("identity")
    m_single = FileMapper.from_offloading_spec("/tmp/kvcache", single, 1, parallel_agnostic=True)
    m_rank = FileMapper.from_offloading_spec("/tmp/kvcache", per_rank, 1, parallel_agnostic=True)
    run = m_single.get_run_config()
    check(run.get("replicated_layout") is True and run["tp_size"] == 1,
          "single copy: TP-independent namespace with replicated_layout")
    check(m_single.base_path != m_rank.base_path,
          f"new directory {m_single.base_path.rsplit('/', 1)[-1]} "
          f"(per-rank: {m_rank.base_path.rsplit('/', 1)[-1]})")

    ok = all(r for r, _ in results)
    print("\nALL OK" if ok else "\nFAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
