#!/usr/bin/env python3
"""MK-4 (program megakernel, stage 3, milestone 1): the sequence-parallel collectives of the pipelined prefill step from the
device - a 4th stage on top of patch_vllm_prefill_sp_sm120.py / patch_vllm_prefill_sp_gather_sm120.py / patch_vllm_prefill_pipe_sm120.py.

Switch: VLLM_MOET_PREFILL_PIPE_DEVICE_COMM=0|1|verify (default 0 = every patched line is a no-op: NCCL as today). Runtime module
tools/dsv41_sm120/sp_pipeline/moet_device_comm.py (+ moet_device_comm_sm120.cu, built on first use), installed next to the
pipeline module as vllm/v1/worker/moet_device_comm.py - its docstring has the design (one small kernel instance per collective,
PCIe P2P pushes phased per peer, NCCL's summation order from a probed per-element table, every device wait watchdog-bounded).

Patched files (anchor-based, idempotent; the text the earlier patchers installed is the anchor, nothing else accepted):
  vllm/models/deepseek_v41/nvidia/model.py   _moet_sp_reduce_scatter: the device reduce-scatter first (inside an open comm phase
                                             of a pipelined group), NCCL's otherwise; the bf16 MoE-input gather fallback of
                                             _moet_sp_moe_gather through the device all-gather hook
  vllm/models/deepseek_v41/attention.py      _moet_sp_input_projections: the full-input gather (AUX=full layers) and the shard
                                             projections' gathers through the device all-gather hook
  (tools/dsv41_sm120/sp_moe_gather/sp_moe_gather_sm120.py at HEAD routes the e4m3 / small gathers through the same hook itself)

Usage:
    python3 patch_vllm_prefill_device_comm_sm120.py [--dist DIR] [--src DIR] [--check]
  --dist   the vllm package directory (default /usr/local/lib/python3.12/dist-packages/vllm)
  --src    the directory with moet_device_comm.py + moet_device_comm_sm120.cu (default sp_pipeline/ next to this file)
  --check  dry run: every anchor present (or already patched), nothing written
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm")
DEFAULT_SRC = Path(__file__).resolve().parent / "sp_pipeline"
HELPERS = ("moet_device_comm.py", "moet_device_comm_sm120.cu")
HELPER_DIR = "v1/worker"
MARKER = "[vllm-moet] sm_120 prefill device comm"

MODEL_EDITS = [
    (
        "reduce-scatter hook",
        "    _moet_p2p_rs_resolve()\n"
        "    _moet_pipe.record_for_comm(x)  # [vllm-moet] sm_120 prefill pipeline: born on the compute stream, read by the comm stream\n"
        "    if _MOET_P2P_RS:\n",
        "    _moet_p2p_rs_resolve()\n"
        "    _moet_pipe.record_for_comm(x)  # [vllm-moet] sm_120 prefill pipeline: born on the compute stream, read by the comm stream\n"
        f"    _moet_dc_out = _moet_pipe.device_reduce_scatter(x, sp_reduce_scatter)  # {MARKER} (MK-4): the device RS inside a pipelined phase\n"
        "    if _moet_dc_out is not None:\n"
        "        return _moet_dc_out\n"
        "    if _MOET_P2P_RS:\n",
    ),
    (
        "bf16 MoE gather fallback",
        "        out = m.gather_moe_input(layer.ffn, x, positions.shape[0], input_ids)\n"
        "        if out is not None:\n"
        "            return out\n"
        "    return sp_all_gather(x)[: positions.shape[0]]\n",
        "        out = m.gather_moe_input(layer.ffn, x, positions.shape[0], input_ids)\n"
        "        if out is not None:\n"
        "            return out\n"
        f"    return _moet_pipe.device_all_gather(x, sp_all_gather)[: positions.shape[0]]  # {MARKER} (MK-4)\n",
    ),
]

ATTN_EDITS = [
    (
        "full-input gather",
        "    if _moet_sp_aux_mode() == \"full\" and (attn.compressor is not None or attn.indexer is not None):\n"
        "        full = sp_all_gather(hidden_states)[:num_tokens]\n",
        "    if _moet_sp_aux_mode() == \"full\" and (attn.compressor is not None or attn.indexer is not None):\n"
        f"        full = _moet_pipe.device_all_gather(hidden_states, sp_all_gather)[:num_tokens]  # {MARKER} (MK-4)\n",
    ),
    (
        "shard projections gather",
        "    gathered = tuple(\n"
        "        None if t is None else sp_all_gather(t.contiguous())[:num_tokens] for t in (qr_kv, kv_score, indexer_weights)\n"
        "    )\n",
        "    gathered = tuple(\n"
        f"        None if t is None else _moet_pipe.device_all_gather(t.contiguous(), sp_all_gather)[:num_tokens]  # {MARKER} (MK-4)\n"
        "        for t in (qr_kv, kv_score, indexer_weights)\n"
        "    )\n",
    ),
]

FILES = [
    ("models/deepseek_v41/nvidia/model.py", MODEL_EDITS),
    ("models/deepseek_v41/attention.py", ATTN_EDITS),
]


def patch_text(src: str, edits) -> tuple[str, str]:
    if MARKER in src:
        return src, "already"
    for name, old, new in edits:
        if old not in src:
            return src, f"missing:{name}"
        src = src.replace(old, new, 1)
    return src, "applied"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    rc = 0
    for h in HELPERS:
        if not (args.src / h).is_file():
            print(f"helper missing: {args.src / h}")
            return 1
    compile((args.src / HELPERS[0]).read_text(), str(args.src / HELPERS[0]), "exec")
    outputs = []
    for rel, edits in FILES:
        path = args.dist / rel
        if not path.exists():
            print(f"{rel}: not found under {args.dist}")
            rc = 1
            continue
        src = path.read_text()
        if "_moet_pipe" not in src:
            print(f"{rel}: the pipeline patcher has not run on this file (no _moet_pipe) - apply patch_vllm_prefill_pipe_sm120.py first")
            rc = 1
            continue
        new, status = patch_text(src, edits)
        if status.startswith("missing"):
            print(f"{rel}: anchor {status} - not patched")
            rc = 1
            continue
        if status == "already":
            print(f"{rel}: already patched")
            continue
        if args.check:
            print(f"{rel}: all {len(edits)} anchors present (dry run)")
        else:
            compile(new, str(path), "exec")
            outputs.append((path, new, len(edits)))
    if rc or args.check:
        return rc
    for path, new, n in outputs:
        path.write_text(new)
        print(f"{path.relative_to(args.dist)}: patched ({n} edits)")
    dst_dir = args.dist / HELPER_DIR
    dst_dir.mkdir(parents=True, exist_ok=True)
    for h in HELPERS:
        shutil.copyfile(args.src / h, dst_dir / h)
        print(f"helper installed at {dst_dir / h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
