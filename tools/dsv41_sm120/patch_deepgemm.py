#!/usr/bin/env python3
"""Let DeepGEMM's SM120 FP8 paged MQA logits run on 128-row KV pages.

DeepSeek-V4.1's sparse indexer (vLLM DeepseekV41IndexerBackend, kernel block
128 on every non-Hopper arch) hands DeepGEMM an indexer K cache paged at
128 states/block on the compress_ratio=1 layers (20..39). The SM120 host
paths were ported with a 64-row page only:

  csrc/apis/attention.hpp         get_paged_mqa_logits_metadata: arch 12 -> block_kv in {32, 64}
  csrc/apis/attention.hpp         fp8_fp4_paged_mqa_logits:      arch 12, fp8 -> block_kv == 64
  csrc/jit_kernels/impls/sm120_mqa_logits.hpp  sm120_fp8_paged_mqa_logits launcher: block_kv == 64

The device kernel (deep_gemm/include/deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh)
and its scheduler are templated on BLOCK_KV with kNumGroups = SPLIT_KV / BLOCK_KV
math-warp groups; 128 rows = one group of eight 16-row MMA warps, SPLIT_KV stays
128, and the smem budget (2 Q stages + 3 KV stages of 128x128 B) is ~71 KB.
Nothing else refers to the page size, so the change is host-side only: the
three asserts admit 128 for the FP8 cache.

The MXFP4 indexer cache (vLLM `--attention-config '{"indexer_kv_dtype":"mxfp4"}'`,
68 B per key: 64 B of e2m1 pairs + 4 UE8M0 scales) goes through the sibling
kernel sm120_fp4_paged_mqa_logits.cuh, which is templated on BLOCK_KV the same
way (kWarpsPerGroup = BLOCK_KV / 16, SPLIT_KV = BLOCK_KV * groups) and stages
128 x 64 B + 128 x 4 B per KV stage, i.e. less smem than the FP8 kernel at the
same page. Its two host asserts (fp8_fp4_paged_mqa_logits and the
sm120_fp4_paged_mqa_logits launcher) were also written for 32/64-row pages only;
this patcher admits 128 for both (2026-09-20). vLLM's own sm_100 gate on the
MXFP4 indexer is lifted by patch_vllm_indexer_fp4_sm120.py.

Validated by test_deepgemm_sm120_paged_mqa.py (torch reference + parity of the
same logical cache re-paged 64 vs 128; `--fmt mxfp4` for the FP4 kernel) on
RTX 5090 / RTX PRO 6000.

Idempotent; hard anchors (assert). Usage:
  python3 patch_deepgemm.py /path/to/DeepGEMM   (checkout of deepseek-ai/DeepGEMM
                                                 8b1392b978f5a03c828dd1711090d7fb50958b8a,
                                                 the pin vendored in vllm-openai:deepseekv41-flash-0909)
"""
import pathlib
import sys


def patch(path: pathlib.Path, old: str, new: str, count: int = 1):
    s = path.read_text()
    if new in s:
        return
    assert old in s, f"ANCHOR NOT FOUND in {path}:\n{old[:200]}"
    assert s.count(old) == count, f"anchor x{s.count(old)} != {count} in {path}"
    path.write_text(s.replace(old, new))
    print(f"patched: {path.name}: {old.strip()[:70]!r}")


def main() -> None:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    attn = root / "csrc/apis/attention.hpp"
    host = root / "csrc/jit_kernels/impls/sm120_mqa_logits.hpp"
    assert attn.exists() and host.exists(), f"not a DeepGEMM checkout: {root}"

    # 1. metadata: arch 12 admits the 128-row indexer page
    patch(
        attn,
        """    } else if (arch_major == 12) {
        DG_HOST_ASSERT(block_kv == 32 or block_kv == 64);
        const int next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1;""",
        """    } else if (arch_major == 12) {
        // DSV4.1 (vLLM-Moet): 128-row pages on the compress_ratio=1 indexer layers.
        DG_HOST_ASSERT(block_kv == 32 or block_kv == 64 or block_kv == 128);
        const int next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1;""",
    )
    # 2. logits: arch 12 admits 128 for both the FP8 and the MXFP4 indexer cache
    patch(
        attn,
        """        (arch_major == 12 and ((is_fp4 and (block_kv == 32 or block_kv == 64)) or
                               (not is_fp4 and block_kv == 64))));""",
        """        (arch_major == 12 and ((is_fp4 and (block_kv == 32 or block_kv == 64 or block_kv == 128)) or
                               (not is_fp4 and (block_kv == 64 or block_kv == 128)))));""",
    )
    # 3. SM120 FP8 paged launcher: one 8-warp group per 128-row page
    patch(
        host,
        """    const int num_groups = split_kv / block_kv;
    const int next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1;
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 12);
    DG_HOST_ASSERT(block_kv == 64);
    DG_HOST_ASSERT(split_kv == 128 and logits_stride % split_kv == 0);""",
        """    const int num_groups = split_kv / block_kv;
    const int next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1;
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 12);
    // DSV4.1 (vLLM-Moet): 128-row pages -> kNumGroups = 1 (eight 16-row MMA warps).
    DG_HOST_ASSERT(block_kv == 64 or block_kv == 128);
    DG_HOST_ASSERT(split_kv == 128 and logits_stride % split_kv == 0);""",
    )
    # 4. SM120 MXFP4 paged launcher: same warp grouping, 64 B rows + int32 UE8M0 quads
    patch(
        host,
        """    DG_HOST_ASSERT(device_runtime->get_arch_major() == 12);
    DG_HOST_ASSERT(split_kv == 128 and logits_stride % split_kv == 0);
    DG_HOST_ASSERT(block_kv == 32 or block_kv == 64);
    DG_HOST_ASSERT(head_dim == 128);""",
        """    DG_HOST_ASSERT(device_runtime->get_arch_major() == 12);
    DG_HOST_ASSERT(split_kv == 128 and logits_stride % split_kv == 0);
    // DSV4.1 (vLLM-Moet): 128-row MXFP4 indexer pages -> kNumGroups = 1.
    DG_HOST_ASSERT(block_kv == 32 or block_kv == 64 or block_kv == 128);
    DG_HOST_ASSERT(head_dim == 128);""",
    )
    print("PATCH-DEEPGEMM-DSV41-DONE")


if __name__ == "__main__":
    main()
