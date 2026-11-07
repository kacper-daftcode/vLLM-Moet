#!/usr/bin/env python3
"""Ripple step 1: the TP group's all-reduces on the sm_120 P2P push (tools/dsv41_sm120/p2p_ar) instead of NCCL's ring.

Where: `GroupCoordinator._all_reduce_out_place` in vllm/distributed/parallel_state.py - the runtime behind every
`tensor_model_parallel_all_reduce` (RowParallelLinear's o_proj and down_proj: 160 per decode step on Ripple's 80 dense
layers, plus the drafter's 10), reached both through the `vllm.all_reduce` custom op (torch.compile) and directly. On
4x RTX PRO 6000 vLLM has only PyNCCL for TP4 (its custom all-reduce refuses >2 PCIe GPUs, FlashInfer's refuses
world_size 4, symmetric memory is not available on capability 12.0).

What: for the TP group only, a contiguous bf16 [M, H] tensor with H the module's compiled row width (5120 = Ripple's
d_model, the default) and M <= VLLM_MOET_P2P_AR_MAX_ROWS (64) goes through p2p_allreduce_sm120 (one-shot push for
M <= VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS, default 0; two-shot reduce-scatter + all-gather above it), which adds the four
partials in NCCL's own order with NCCL's bf16 rounding (measured on the TP communicator at the first call, each M checked
against NCCL on random data; an M that does not reproduce NCCL stays on NCCL) - the bits are NCCL's. Every other shape,
group or failure takes the device communicator as before. The first call is collective (IPC handles, order tables): it
is the first TP all-reduce of the first forward (the profiling run), which every rank reaches together.

VLLM_MOET_P2P_AR=1 enables it at runtime (default off); VLLM_MOET_P2P_AR_DIR points at the module directory
(default /opt/vllm-moet/dsv41_sm120/p2p_ar); TORCH_EXTENSIONS_DIR holds the compiled extension (built in the image).

Usage: python3 patch_vllm_p2p_allreduce_tp_sm120.py [--path PATH] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.py")
MARKER = "[vllm-moet] sm_120 P2P all-reduce (TP, generic)"

CLASS_ANCHOR = "class GroupCoordinator:\n"
HELPERS = f'''# {MARKER}: the TP group's bf16 [M, H] all-reduces of M <= 64 rows on a push over PCIe P2P with NCCL's
# summation order (tools/dsv41_sm120/p2p_ar; VLLM_MOET_P2P_AR=1 enables it). Other shapes, other groups and every failure
# stay on the device communicator.
_MOET_P2P_AR = None  # None = not resolved, False = off / unavailable, else the module
_MOET_P2P_TP_NAME: str | None = None


def _moet_p2p_resolve():
    global _MOET_P2P_AR
    if _MOET_P2P_AR is None:
        import os

        from vllm.platforms import current_platform

        _MOET_P2P_AR = False
        if (
            os.environ.get("VLLM_MOET_P2P_AR", "0") == "1"
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        ):
            d = os.environ.get("VLLM_MOET_P2P_AR_DIR", "/opt/vllm-moet/dsv41_sm120/p2p_ar")
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                import p2p_allreduce_sm120 as _m

                _MOET_P2P_AR = _m
            except Exception as exc:  # noqa: BLE001
                logger.warning("vllm-moet sm_120 P2P all-reduce module unavailable, NCCL stays: %r", exc)
    return _MOET_P2P_AR


def _moet_p2p_is_tp(group) -> bool:
    global _MOET_P2P_TP_NAME
    if _MOET_P2P_TP_NAME is None:
        if _TP is None:
            return False
        _MOET_P2P_TP_NAME = _TP.unique_name
    return group.unique_name == _MOET_P2P_TP_NAME


def _moet_p2p_try_all_reduce(group, x: torch.Tensor):
    """The P2P all-reduce of x when it applies (TP group, enabled, shape + measured order), else None."""
    m = _moet_p2p_resolve()
    if not m or group.world_size != 4 or not _moet_p2p_is_tp(group):
        return None
    st = m._init()  # collective on the first call: the first TP all-reduce of the first forward on every rank
    if st is None:
        return None
    how = m.mode(x)
    if how == "nccl":
        return None
    x = x.contiguous()
    return m._run(st, x, st["orders"][x.shape[0]], how)


'''
SITE_OLD = (
    "    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:\n"
    "        if self.device_communicator is None:\n"
    '            raise ValueError("No device communicator found")\n'
    "        return self.device_communicator.all_reduce(input_)\n"
)
SITE_NEW = (
    "    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:\n"
    "        if self.device_communicator is None:\n"
    '            raise ValueError("No device communicator found")\n'
    f"        out = _moet_p2p_try_all_reduce(self, input_)  # {MARKER}\n"
    "        if out is not None:\n"
    "            return out\n"
    "        return self.device_communicator.all_reduce(input_)\n"
)
IMPORT_ANCHOR = "import torch\n"
IMPORT_ADD = "import sys  # [vllm-moet] step 1 (P2P all-reduce module path)\n"


def _once(src: str, anchor: str, what: str) -> None:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"parallel_state.py: {what} anchor found {n} times (expected 1)")


def patch(src: str) -> str:
    for what, anchor in (("class", CLASS_ANCHOR), ("all-reduce site", SITE_OLD), ("import", IMPORT_ANCHOR)):
        _once(src, anchor, what)
    if "import sys\n" not in src:
        src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_ADD, 1)
    src = src.replace(CLASS_ANCHOR, HELPERS + CLASS_ANCHOR, 1)
    return src.replace(SITE_OLD, SITE_NEW, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=PATH)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.path.read_text()
    if MARKER in src:
        print(f"{args.path}: already patched")
        return 0
    out = patch(src)
    compile(out, str(args.path), "exec")
    if args.check:
        print(f"{args.path}: patch applies cleanly (not written)")
        return 0
    args.path.write_text(out)
    print(f"{args.path}: patched ({MARKER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
