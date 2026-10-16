#!/usr/bin/env python3
"""Backport of vllm#61647 (24f632ca7e): the indexer's TP row-sharding threshold as an attention-config field.

vllm#54951 (3e8e1364ff) splits a step's prefill rows of the sparse-attention indexer across the TP ranks
(DeepseekV41's `SparseAttnIndexer`; each rank scores its share and the top-k rows are gathered), but only
from MIN_TP_SHARD_ROWS_PER_RANK * TP = 65,536 rows in the step. A chunked prefill of 4096 tokens never gets
there, so on vLLM 3f1bcfd82b the sharding is dead code for this deployment. vllm#61647 turned the constant
into `AttentionConfig.indexer_tp_shard_min_rows` (default 65,536 total rows: the same threshold at TP4);
the launcher passes it with INDEXER_TP_SHARD_MIN_ROWS. Two files, both or neither:

  vllm/config/attention.py                    the field
  vllm/v1/attention/backends/mla/indexer.py   balanced_prefill_row_shard(..., min_rows) and its caller

Idempotent; refuses a vLLM that already has the field (vllm#61647 merged) with "already upstream".
Usage: python3 patch_vllm_indexer_shard_rows.py [--vllm-dir DIR] [--check]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "[vllm-moet] vllm#61647 backport"

CONFIG: list[tuple[str, str]] = [
    (
        "    hisparse_config: HiSparseConfig | None = None\n",
        f"    # {MARKER}\n"
        "    indexer_tp_shard_min_rows: int = Field(default=65536, ge=1)\n"
        '    """Minimum total prefill rows for the sparse-attention indexer to split\n'
        '    them across TP ranks instead of scoring every row on each rank."""\n'
        "\n"
        "    hisparse_config: HiSparseConfig | None = None\n",
    ),
]

INDEXER: list[tuple[str, str]] = [
    (
        "# Below this many rows per rank, stay replicated (TP4 engages at 64K rows).\n"
        "MIN_TP_SHARD_ROWS_PER_RANK = 16_384\n"
        "\n"
        "\n"
        "def balanced_prefill_row_shard(\n"
        "    seq_lens_cpu: torch.Tensor,\n"
        "    query_lens_cpu: torch.Tensor,\n"
        "    compress_ratio: int,\n"
        "    tp_size: int,\n"
        ") -> list[int] | None:\n"
        '    """Return contiguous TP row counts balanced by indexer MQA work."""\n'
        "    num_rows = int(query_lens_cpu.sum())\n"
        "    if tp_size < 2 or num_rows < MIN_TP_SHARD_ROWS_PER_RANK * tp_size:\n"
        "        return None\n",
        f"# {MARKER}: the threshold is AttentionConfig.indexer_tp_shard_min_rows.\n"
        "def balanced_prefill_row_shard(\n"
        "    seq_lens_cpu: torch.Tensor,\n"
        "    query_lens_cpu: torch.Tensor,\n"
        "    compress_ratio: int,\n"
        "    tp_size: int,\n"
        "    min_rows: int,\n"
        ") -> list[int] | None:\n"
        '    """Return contiguous TP row counts balanced by indexer MQA work."""\n'
        "    num_rows = int(query_lens_cpu.sum())\n"
        "    if tp_size < 2 or num_rows < max(min_rows, tp_size):\n"
        "        return None\n",
    ),
    (
        "                    prefill_query_lens_cpu,\n"
        "                    self.compress_ratio,\n"
        "                    self.tp_size,\n"
        "                )\n",
        "                    prefill_query_lens_cpu,\n"
        "                    self.compress_ratio,\n"
        "                    self.tp_size,\n"
        "                    self.vllm_config.attention_config.indexer_tp_shard_min_rows,\n"
        "                )\n",
    ),
]

FILES = {
    "config/attention.py": CONFIG,
    "v1/attention/backends/mla/indexer.py": INDEXER,
}


def patched_text(path: Path, edits: list[tuple[str, str]]) -> str | None:
    s = path.read_text()
    for old, new in edits:
        if new in s:
            continue
        n = s.count(old)
        if n != 1:
            print(f"{path}: anchor x{n} != 1:\n{old[:200]}", file=sys.stderr)
            return None
        s = s.replace(old, new)
    compile(s, str(path), "exec")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-dir", type=Path, default=DEFAULT_VLLM)
    ap.add_argument("--check", action="store_true", help="verify anchors, write nothing")
    args = ap.parse_args()
    config = (args.vllm_dir / "config/attention.py").read_text()
    if "indexer_tp_shard_min_rows" in config and MARKER not in config:
        print("indexer_tp_shard_min_rows is already upstream (vllm#61647): nothing to backport")
        return 0
    out = {}
    for rel, edits in FILES.items():
        out[args.vllm_dir / rel] = patched_text(args.vllm_dir / rel, edits)
    if any(s is None for s in out.values()):
        print("nothing written", file=sys.stderr)
        return 1
    for path, s in out.items():
        if s == path.read_text():
            print(f"{path}: already patched")
        elif args.check:
            print(f"{path}: patch applies cleanly (not written)")
        else:
            path.write_text(s)
            print(f"{path}: patched (indexer_tp_shard_min_rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
