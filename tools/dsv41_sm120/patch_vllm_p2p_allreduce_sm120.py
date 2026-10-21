#!/usr/bin/env python3
"""The decode step's TP4 all-reduces on a push over PCIe P2P instead of NCCL's ring, on sm_120 (DeepSeek-V4.1-Flash).

Where: the three all-reduce sites of the overlap path (step 10) in vllm/models/deepseek_v41/nvidia/{ops/mhc.py,model.py}:
the sublayer boundary in front of mhc_post_norm, the Engram layers' post, the last layer's post. The module p2p_ar/
(p2p_allreduce_sm120.cu + .py) picks by shape: rows <= VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS (default 0) one-shot push,
rows <= VLLM_MOET_P2P_AR_MAX_ROWS (default 64) two-shot (reduce-scatter + all-gather, each phase to one peer at a time -
on this host a kernel writing to several peers concurrently is ~16x slower), anything else and every failure NCCL.
The sum is taken in fixed source order in fp32 and rounded once, so the result is deterministic and identical on every
rank - but its bits differ from NCCL's ring order: an output change, gated like every all-reduce change (GSM8K-200 at
concurrency 4 exercises the two-shot path, needle, tools).

Measured (4x RTX PRO 6000, 325 W, memory +5000, 80 all-reduces replayed from a graph): [48, 5120] bf16 two-shot 23.5 us
vs NCCL 69.8; [24] 14.6 vs 37.3; [6] one-shot 10.6 vs 18.5 - neutral in the full C1 graph (NCCL and the push both sit on
the rank skew + flag latency floor there), hence one-shot off by default.

VLLM_MOET_P2P_AR=1 enables it at runtime (default off); VLLM_MOET_P2P_AR_DIR points at the module directory
(default /opt/vllm-moet/dsv41_sm120/p2p_ar). Needs step 10 (mhc_overlap) applied first.

Usage: python3 patch_vllm_p2p_allreduce_sm120.py [--mhc PATH] [--model PATH] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia")
MARKER = "[vllm-moet] sm_120 P2P all-reduce"

MHC_HELPER_ANCHOR = "def _moet_mhc_overlap_supported("
MHC_HELPER = f'''# {MARKER}: the boundary all-reduce of the decode graphs on a push over PCIe P2P
# (tools/dsv41_sm120/p2p_ar; VLLM_MOET_P2P_AR=1 enables it, larger shapes and every failure fall back to NCCL).
_MOET_P2P_AR = None  # None = not resolved, False = off / unavailable, else the module


def _moet_p2p_all_reduce(x: torch.Tensor) -> torch.Tensor:
    global _MOET_P2P_AR
    if _MOET_P2P_AR is None:
        import os
        import sys

        _MOET_P2P_AR = False
        if os.environ.get("VLLM_MOET_P2P_AR", "0") == "1" and current_platform.is_cuda() and current_platform.is_device_capability_family(120):
            d = os.environ.get("VLLM_MOET_P2P_AR_DIR", "/opt/vllm-moet/dsv41_sm120/p2p_ar")
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                import p2p_allreduce_sm120 as _m

                _MOET_P2P_AR = _m
            except Exception as exc:  # noqa: BLE001
                _moet_logger.warning("vllm-moet sm_120 P2P all-reduce module unavailable, NCCL stays: %r", exc)
    if _MOET_P2P_AR:
        return _MOET_P2P_AR.all_reduce(x)
    return get_tp_group().all_reduce(x)


'''
MHC_SITE_OLD = (
    "            if reduce_results:\n"
    "                x = get_tp_group().all_reduce(x)\n"
    "            return _moet_mhc_shifted_post_pre_overlap(\n"
)
MHC_SITE_NEW = (
    "            if reduce_results:\n"
    f"                x = _moet_p2p_all_reduce(x)  # {MARKER} (NCCL when not applicable)\n"
    "            return _moet_mhc_shifted_post_pre_overlap(\n"
)
MODEL_IMPORT_ANCHOR = "from .ops.mhc import (\n"
MODEL_IMPORT = f"    _moet_p2p_all_reduce,  # {MARKER}\n"
MODEL_SITES = (
    (
        "            if self.fuse_mhc_all_reduce:\n"
        "                x = tensor_model_parallel_all_reduce(x)\n"
        "            previous_post = mhc_post_tilelang(x, residual, post_mix, res_mix)\n",
        "            if self.fuse_mhc_all_reduce:\n"
        f"                x = _moet_p2p_all_reduce(x)  # {MARKER}\n"
        "            previous_post = mhc_post_tilelang(x, residual, post_mix, res_mix)\n",
    ),
    (
        "        if self.fuse_mhc_all_reduce:\n"
        "            hidden_states = tensor_model_parallel_all_reduce(hidden_states)\n"
        "        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)\n",
        "        if self.fuse_mhc_all_reduce:\n"
        f"            hidden_states = _moet_p2p_all_reduce(hidden_states)  # {MARKER}\n"
        "        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)\n",
    ),
)


def _once(src: str, anchor: str, what: str, name: str) -> None:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"{name}: {what} anchor found {n} times (expected 1)")


def patch_mhc(src: str) -> str:
    if MARKER in src:
        return src
    for what, anchor in (("overlap helper (step 10)", MHC_HELPER_ANCHOR), ("boundary all-reduce", MHC_SITE_OLD)):
        _once(src, anchor, what, "ops/mhc.py")
    src = src.replace(MHC_HELPER_ANCHOR, MHC_HELPER + MHC_HELPER_ANCHOR, 1)
    return src.replace(MHC_SITE_OLD, MHC_SITE_NEW, 1)


def patch_model(src: str) -> str:
    if MARKER in src:
        return src
    _once(src, MODEL_IMPORT_ANCHOR, "ops.mhc import", "model.py")
    for old, _ in MODEL_SITES:
        _once(src, old, "all-reduce site", "model.py")
    src = src.replace(MODEL_IMPORT_ANCHOR, MODEL_IMPORT_ANCHOR + MODEL_IMPORT, 1)
    for old, new in MODEL_SITES:
        src = src.replace(old, new, 1)
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhc", type=Path, default=VLLM / "ops" / "mhc.py")
    ap.add_argument("--model", type=Path, default=VLLM / "model.py")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    for path, fn in ((args.mhc, patch_mhc), (args.model, patch_model)):
        src = path.read_text()
        if MARKER in src:
            print(f"{path}: already patched")
            continue
        out = fn(src)
        compile(out, str(path), "exec")
        if args.check:
            print(f"{path}: patch applies cleanly (not written)")
            continue
        path.write_text(out)
        print(f"{path}: patched ({MARKER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
