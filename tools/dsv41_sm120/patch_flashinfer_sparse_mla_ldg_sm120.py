#!/usr/bin/env python3
"""DeepSeek-V4.1 sparse MLA decode on sm_120 with the candidate rows gathered by plain loads and a merge kernel whose
loads are issued up front (step 28), behind a runtime switch; output, LSE and the split scratch bit-identical to
FlashInfer 0.7.2's decode + merge.

FlashInfer's mixed-cache decode kernel (sparse_mla_sm120: 128 math + 128 convert + 128 gather threads per CTA) has
its gather warp issue one cp.async.bulk per candidate row (32 fp8 or 64 fp4 copies of 288-528 B per tile); the TMA
unit works them off one by one and a tile's bytes land ~2.5 us after its indices. The extension
(sparse_mla_ldg/sparse_mla_ldg_sm120.cu) compiles FlashInfer's own kernel template from the installed headers, with
FlashInfer's flags, instantiated with a gather schedule whose 128 threads load the same rows as 16-byte words (one
DRAM round trip per tile) and store them into the same shared-memory slots before completing the same mbarrier; the
consumers (conversion, math, every barrier) are FlashInfer's code. Its merge kernel is FlashInfer's expression by
expression, with the partial rows and the attention sink loaded before the LSE reductions. Measured at the served C1
shape (6 tokens, 128 SWA + 512 compressed candidates, 10 splits, cold L2, CUDA graph): decode 10.5 -> 9.1 us, merge
2.6-2.9 -> 2.2 us per layer.

Patched file (flashinfer/mla/_sparse_mla_sm120/_prepared.py): prepare() records the resolved plan's facts
(plan.inspect(), the interleaved-scales and extra-fp4 flags of the metadata) and hands out an execute function that
runs the extension's decode + merge when one function, _moet_sparse_mla_ldg_on, says so: VLLM_MOET_SPARSE_MLA_LDG=1
(default 0: off), sm_12x, the plan is the mixed-cache decode (fp8 SWA + fp4 compressed, 16 heads x 512, <= 16
splits) and the extension is importable from VLLM_MOET_SPARSE_MLA_LDG_DIR (default
/opt/vllm-moet/dsv41_sm120/sparse_mla_ldg). Every other plan (SWA-only layers, prefill, NVFP4) and the switch off run
FlashInfer's execute_attention unchanged. VLLM_MOET_SPARSE_MLA_LDG_MERGE=0 keeps FlashInfer's merge kernel.

Usage: python3 patch_flashinfer_sparse_mla_ldg_sm120.py [--root DIR] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_sparse_mla_sm120")
MARKER = "[vllm-moet] sm_120 sparse MLA load-based gather"

HELPERS = '''

# {marker}: the mixed-cache decode kernel with its rows gathered by plain loads, and the merge with its loads
# issued up front (bits unchanged), decided once per process; the plan facts are recorded at prepare time.
_MOET_LDG_PLANS: dict = {{}}


@functools.lru_cache(maxsize=None)
def _moet_sparse_mla_ldg_module():
    import os
    import sys

    path = os.environ.get("VLLM_MOET_SPARSE_MLA_LDG_DIR", "/opt/vllm-moet/dsv41_sm120/sparse_mla_ldg")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import sparse_mla_ldg_sm120 as module
    except ImportError:
        return None
    return module


@functools.lru_cache(maxsize=None)
def _moet_sparse_mla_ldg_on() -> bool:
    import os

    if os.environ.get("VLLM_MOET_SPARSE_MLA_LDG", "0") != "1":
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        return False
    return _moet_sparse_mla_ldg_module() is not None


def _moet_sparse_mla_ldg_record(plan, metadata) -> None:
    if not _moet_sparse_mla_ldg_on():
        return
    try:
        facts = dict(plan.inspect())
    except Exception:  # noqa: BLE001 - a plan without inspect(): FlashInfer's path
        return
    _MOET_LDG_PLANS[id(plan)] = (plan, facts, bool(getattr(metadata, "interleaved_scales", False)),
                                 bool(getattr(metadata, "extra_fp4", False)))


def _moet_sparse_mla_ldg_execute_fn(is_dsv4_nvfp4):
    orig = _execute_fn(is_dsv4_nvfp4)
    if is_dsv4_nvfp4 or not _moet_sparse_mla_ldg_on():
        return orig
    module = _moet_sparse_mla_ldg_module()

    def execute(plan, q, cache, indices, mid, mlse, output, lse, scale, lengths, sink, extra, extra_indices,
                extra_lengths, lse_scale):
        rec = _MOET_LDG_PLANS.get(id(plan))
        if (rec is not None and rec[0] is plan and rec[3] and lse is not None
                and module.supported(rec[1], q, extra, extra_indices)):
            module.decode(rec[1], q, cache, indices, mid, mlse, output, lse, scale, lengths, sink, extra,
                          extra_indices, extra_lengths, lse_scale, rec[2])
            return None
        return orig(plan, q, cache, indices, mid, mlse, output, lse, scale, lengths, sink, extra, extra_indices,
                    extra_lengths, lse_scale)

    return execute
'''

RESOLVE_ANCHOR = '''    plan = resolve_execution(
        metadata, device, precision, is_dsv4_nvfp4, preference, decode_only
    )
'''
RESOLVE = RESOLVE_ANCHOR + "    _moet_sparse_mla_ldg_record(plan, metadata)\n"
EXECUTE_ANCHOR = "    execute_fn = _execute_fn(is_dsv4_nvfp4)\n"
EXECUTE = "    execute_fn = _moet_sparse_mla_ldg_execute_fn(is_dsv4_nvfp4)\n"
HELPERS_ANCHOR = '''def _execute_fn(is_dsv4_nvfp4: bool):
    module = get_sparse_mla_sm120_module()
    return (
        module.dsv4_nvfp4_execute_attention
        if is_dsv4_nvfp4
        else module.execute_attention
    )
'''


def patch_prepared(src: str) -> str:
    if MARKER in src:
        return src
    if "import functools" not in src or "import torch" not in src:
        raise SystemExit("_prepared.py: functools / torch imports not found - FlashInfer changed")
    for anchor, what in ((HELPERS_ANCHOR, "_execute_fn"), (RESOLVE_ANCHOR, "resolve_execution call"),
                         (EXECUTE_ANCHOR, "execute_fn assignment")):
        if src.count(anchor) != 1:
            raise SystemExit(f"_prepared.py: anchor '{what}' not found once - FlashInfer changed")
    src = src.replace(HELPERS_ANCHOR, HELPERS_ANCHOR + HELPERS.format(marker=MARKER), 1)
    src = src.replace(RESOLVE_ANCHOR, RESOLVE, 1)
    return src.replace(EXECUTE_ANCHOR, EXECUTE, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = a.root / "_prepared.py"
    src = path.read_text()
    out = patch_prepared(src)
    if a.check:
        print(f"{path}: {'patched' if out == src else 'applies'}")
        return 0
    if out != src:
        compile(out, str(path), "exec")
        path.write_text(out)
        print(f"patched {path}")
    else:
        print(f"{path} already patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
