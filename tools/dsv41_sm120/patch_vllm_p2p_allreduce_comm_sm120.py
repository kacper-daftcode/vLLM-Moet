#!/usr/bin/env python3
"""The TP group's decode-sized all-reduces on the sm_120 PCIe P2P push, hooked at the communicator (GLM-5.3-Flash line, step G1).

The DeepSeek-V4.1 patcher (patch_vllm_p2p_allreduce_sm120.py) hooks the three all-reduce sites of the mHC overlap path in
deepseek_v41/nvidia/ops/mhc.py. GLM-5.3-Flash on stock vLLM has no such sites: its ~100 all-reduces per decode step
(o_proj's RowParallelLinear, the MoE combine, the embedding, the drafter's) all end in
vllm/distributed/device_communicators/cuda_communicator.py: CudaCommunicator.all_reduce, whose stock dispatch on 4 PCIe
GPUs is PyNccl's ring (13.5 us for [4, 4096] bf16, 31.5 us for [32, 4096]; 10-12 % of the step). This patcher puts the
P2P module (tools/dsv41_sm120/p2p_ar: one-shot push / two-shot reduce-scatter + all-gather, summed in NCCL's own measured
order so the bits are PyNccl's) in front of that dispatch for the TP group only: 2-D bf16 [M, H] tensors with
M <= VLLM_MOET_P2P_AR_MAX_ROWS (64) and H = VLLM_MOET_P2P_AR_H (4096 for GLM); every other tensor, and every failure to
initialise, falls through to the stock backends. The module's first use is collective (IPC handles, NCCL order tables for
M = 1..64 on the group's PyNccl communicator) - it happens at the first TP all-reduce of the profile run, eager, on every
rank together.

Runtime: VLLM_MOET_P2P_AR=1 (default off), VLLM_MOET_P2P_AR_H=4096, VLLM_MOET_P2P_AR_DIR (module directory, default
/opt/vllm-moet/dsv41_sm120/p2p_ar), VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS (0), VLLM_MOET_P2P_AR_MAX_ROWS (64). The start-up
log lists 'MOET_P2P_SM120' first among the TP group's all-reduce backends when the hook is armed.

Usage: python3 patch_vllm_p2p_allreduce_comm_sm120.py [--path PATH] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "[vllm-moet] sm_120 P2P all-reduce (communicator hook)"

HELPER_ANCHOR = "logger = init_logger(__name__)\n"
HELPER = f'''

# {MARKER}: the TP group's decode-sized bf16 all-reduces ([M, H], M <= VLLM_MOET_P2P_AR_MAX_ROWS,
# H = VLLM_MOET_P2P_AR_H) on the PCIe P2P push of tools/dsv41_sm120/p2p_ar, summed in NCCL's own order (bits = PyNccl).
# VLLM_MOET_P2P_AR=1 enables it; every other tensor and every failure to initialise continue to the stock dispatch.
_MOET_P2P = None  # None = unresolved, False = off / unavailable, else the module


def _moet_p2p_wanted(comm) -> bool:
    import os

    return (
        os.environ.get("VLLM_MOET_P2P_AR", "0") == "1"
        and (comm.unique_name or "").split(":")[0] == "tp"
        and comm.world_size == 4
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
    )


def _moet_p2p_module(comm):
    global _MOET_P2P
    if _MOET_P2P is None:
        import os
        import sys

        _MOET_P2P = False
        if _moet_p2p_wanted(comm):
            d = os.environ.get("VLLM_MOET_P2P_AR_DIR", "/opt/vllm-moet/dsv41_sm120/p2p_ar")
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                import p2p_allreduce_sm120 as _m

                _MOET_P2P = _m
            except Exception as exc:  # noqa: BLE001
                logger.warning("vllm-moet sm_120 P2P all-reduce module unavailable, the stock backends stay: %r", exc)
    return _MOET_P2P


def _moet_p2p_all_reduce(comm, x):
    """The P2P all-reduce of x, or None when the fast path does not apply (the caller continues with the stock dispatch)."""
    m = _moet_p2p_module(comm)
    if not m:
        return None
    st = m._init()  # collective on first use (IPC handles + NCCL's summation order per M); cheap afterwards
    if st is None:
        return None
    how = m.mode(x)
    if how == "nccl":
        return None
    x = x.contiguous()
    return m._run(st, x, st["orders"][x.shape[0]], how)
'''

SITE_OLD = (
    "    def all_reduce(self, input_):\n"
    "        fi_ar_comm = self.fi_ar_comm\n"
)
SITE_NEW = (
    "    def all_reduce(self, input_):\n"
    f"        if _MOET_P2P is not False:  # {MARKER}\n"
    "            out = _moet_p2p_all_reduce(self, input_)\n"
    "            if out is not None:\n"
    "                return out\n"
    "        fi_ar_comm = self.fi_ar_comm\n"
)
LOG_OLD = (
    "        logger.info_once(\n"
    '            "Using %s all-reduce backends (in dispatch order) for group "\n'
)
LOG_NEW = (
    f"        if _moet_p2p_wanted(self):  # {MARKER}\n"
    '            enabled_ar_backends.insert(0, "MOET_P2P_SM120")\n'
    "        logger.info_once(\n"
    '            "Using %s all-reduce backends (in dispatch order) for group "\n'
)


def _once(src: str, anchor: str, what: str) -> None:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"cuda_communicator.py: {what} anchor found {n} times (expected 1)")


def patch(src: str) -> str:
    if MARKER in src:
        return src
    for what, anchor in (("logger", HELPER_ANCHOR), ("all_reduce entry", SITE_OLD), ("backends log", LOG_OLD)):
        _once(src, anchor, what)
    src = src.replace(HELPER_ANCHOR, HELPER_ANCHOR + HELPER, 1)
    src = src.replace(SITE_OLD, SITE_NEW, 1)
    return src.replace(LOG_OLD, LOG_NEW, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=VLLM / "distributed" / "device_communicators" / "cuda_communicator.py")
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
