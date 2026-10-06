#!/usr/bin/env python3
"""vLLM's captured CUDA graphs without NCCL's per-collective serialEvent record nodes (all but the last per event).

With graph mixing on (NCCL's default), every collective NCCL captures leaves an event-record node behind its kernel
that the next NCCL kernel of the graph also depends on. On the 4x RTX PRO 6000 host the graph executor then starts
each NCCL kernel ~2-3 us after its predecessor ends instead of ~0.3-0.5 us (80 all-reduces + 4 Engram all-gathers
per decode step). Only the graph's last record of the event matters to later graphs and eager NCCL calls, and the
NCCL kernels stay chained by direct edges, so NCCL's mixing guarantees are unchanged (nccl_graph_records.py).

Every capture site vLLM uses captures with keep_graph=True, drops the records and instantiates the graph itself
(moet_nccl_graph_records.finish_capture; a failing rewrite keeps the graph as captured):
  vllm/v1/worker/gpu/cudagraph_utils.py   CudaGraphManager.capture - the GPU model runner's FULL graphs (the served
                                          decode graphs and the DSpark drafter's)
  vllm/compilation/breakable_cudagraph.py BreakableCUDAGraphCapture segments (PIECEWISE, VLLM_USE_BREAKABLE_CUDAGRAPH)
  vllm/compilation/cuda_graph.py          CUDAGraphWrapper (the v1 gpu_model_runner's graphs)
Runtime switch: VLLM_MOET_NCCL_GRAPH_RECORDS=0 captures and instantiates exactly as before.

Installs nccl_graph_records.py as vllm/compilation/moet_nccl_graph_records.py. Idempotent, anchor-based. Usage:
    python3 patch_vllm_nccl_graph_records.py [--vllm-dir DIR] [--check]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "# [vllm-moet] NCCL serialEvent records reduced in captured graphs (tools/dsv41_sm120/nccl_graph)"
MODULE = "moet_nccl_graph_records.py"
IMPORT = f"import vllm.compilation.moet_nccl_graph_records as _moet_nccl  {MARKER}\n"

# (file, [(anchor, replacement), ...]); every anchor must occur exactly once
SITES = [
    ("v1/worker/gpu/cudagraph_utils.py", [
        ("from vllm.compilation.counter import compilation_counter\n",
         "from vllm.compilation.counter import compilation_counter\n" + IMPORT),
        ("                        graph = torch.cuda.CUDAGraph()\n",
         "                        moet_keep = _moet_nccl.records_enabled()\n"
         "                        graph = torch.cuda.CUDAGraph(keep_graph=moet_keep)\n"),
        ("                        self.graphs[desc] = graph\n",
         "                        if moet_keep:\n"
         "                            _moet_nccl.finish_capture(graph)\n"
         "                        self.graphs[desc] = graph\n"),
    ]),
    ("compilation/breakable_cudagraph.py", [
        ("from vllm.compilation.monitor import validate_cudagraph_capturing_enabled\n",
         "from vllm.compilation.monitor import validate_cudagraph_capturing_enabled\n" + IMPORT),
        ("        g = torch.cuda.CUDAGraph()\n",
         "        self._moet_keep = _moet_nccl.records_enabled()\n"
         "        g = torch.cuda.CUDAGraph(keep_graph=self._moet_keep)\n"),
        ("        self._current_graph.capture_end()\n",
         "        self._current_graph.capture_end()\n"
         "        if self._moet_keep:\n"
         "            _moet_nccl.finish_capture(self._current_graph)\n"),
    ]),
    ("compilation/cuda_graph.py", [
        ("from vllm.compilation.counter import compilation_counter\n",
         "from vllm.compilation.counter import compilation_counter\n" + IMPORT),
        ("            cudagraph = torch.cuda.CUDAGraph()\n",
         "            moet_keep = _moet_nccl.records_enabled()\n"
         "            cudagraph = torch.cuda.CUDAGraph(keep_graph=moet_keep)\n"),
        ("            # here we always use weak ref for the output\n"
         "            # to save memory\n"
         "            entry.output = weak_ref_tensors(output)\n",
         "            if moet_keep:\n"
         "                _moet_nccl.finish_capture(cudagraph)\n"
         "            # here we always use weak ref for the output\n"
         "            # to save memory\n"
         "            entry.output = weak_ref_tensors(output)\n"),
    ]),
]


def patch_text(src: str, edits, name: str) -> str:
    if MARKER in src:
        return src
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"{name}: anchor found {n} times (expected 1): {old.strip()[:70]!r}")
        src = src.replace(old, new, 1)
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-dir", type=Path, default=DEFAULT_VLLM)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    results = []
    for rel, edits in SITES:
        path = args.vllm_dir / rel
        src = path.read_text()
        patched = patch_text(src, edits, rel)
        compile(patched, str(path), "exec")
        results.append((path, src, patched))
    if args.check:
        for path, src, patched in results:
            print(f"{path}: {'already patched' if patched == src else 'patch applies cleanly (not written)'}")
        return 0
    shutil.copyfile(HERE / "nccl_graph_records.py", args.vllm_dir / "compilation" / MODULE)
    for path, src, patched in results:
        if patched != src:
            path.write_text(patched)
        print(f"{path}: {'already patched' if patched == src else 'patched'}")
    print(f"installed {args.vllm_dir / 'compilation' / MODULE} (NCCL serialEvent records reduced in captured graphs; "
          "VLLM_MOET_NCCL_GRAPH_RECORDS=0 turns it off)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
