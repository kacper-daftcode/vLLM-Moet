#!/usr/bin/env python3
"""DeepSeek-V4.1 decoder-side SWA bounded replay (CED prefill) from vllm#58132, on the vLLM-main image.

The report's Causal Encoder-Decoder: the decoder layers' global KV is projected from the encoder output, so in
prefill the layers after the last KV-source layer (21-39 on V4.1-Flash) only need each request's last 128 tokens
(their sliding window), with the window clamped to that segment - an approximation the model was post-trained
with. vllm#58132 (ivanium, open; head 9a86c2c9, base 496c6472) implements it confined to vllm/models/deepseek_v41/;
nothing under that directory changed between our pin 0961bbae and its base, and none of our patchers touches the
files it changes.

This applies decoder_replay/vllm-pr58132-9a86c2c9.patch (the PR's vllm/ diff) to the installed package with GNU
patch and adds a runtime kill switch: VLLM_MOET_DECODER_REPLAY=0 keeps the decoder layers on every row (the encoder
side, swa_bounded_replay, stays as configured). The PR's unit tests land in decoder_replay/tests/ next to this file.

Measured on 4x RTX PRO 6000 (TP4, DSpark k=5, nvfp4_ds_mla, MXFP4 indexer, vision, KV offload), against the same
image without it: fresh prefill 1.63x at 19K tokens, 1.68x at 163K, 1.75x at 391K; GSM8K-200 193 vs 194 (McNemar
p=1), greedy agreement 24/24, needle 8/8 fresh and 6/6 from the disk tier, long-prompt deviation inside the
stack's run-to-run spread, decode and peak memory unchanged, GPU KV -1.2 %.

Idempotent. Usage:
    python3 patch_vllm_decoder_replay.py [--site DIR] [--tests-dir DIR] [--check]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / "decoder_replay"
PATCH = HERE / "vllm-pr58132-9a86c2c9.patch"
TESTS_PATCH = HERE / "vllm-pr58132-9a86c2c9-tests.patch"
DEFAULT_SITE = Path("/usr/local/lib/python3.12/dist-packages")
MODEL = Path("vllm/models/deepseek_v41/nvidia/model.py")
MARKER = "# [vllm-moet] VLLM_MOET_DECODER_REPLAY=0 keeps the decoder layers on every row"

IMPORT_OLD = "import typing\n"
IMPORT_NEW = "import os\nimport typing\n"
GATE_OLD = "        if self.start_layer > cut or self.end_layer < self.config.num_hidden_layers:\n"
GATE_NEW = (
    '        if os.environ.get("VLLM_MOET_DECODER_REPLAY", "1") == "0":\n'
    f"            {MARKER}\n"
    '            reason = "VLLM_MOET_DECODER_REPLAY=0"\n'
    "        elif self.start_layer > cut or self.end_layer < self.config.num_hidden_layers:\n"
)


def run_patch(patch: Path, directory: Path, dry_run: bool) -> None:
    cmd = ["patch", "-p1", "--forward", "--batch", "-d", str(directory), "-i", str(patch)]
    if dry_run:
        cmd.append("--dry-run")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{res.stdout}{res.stderr}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, default=DEFAULT_SITE)
    ap.add_argument("--tests-dir", type=Path, default=HERE)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    model = args.site / MODEL
    if MARKER in model.read_text():
        print(f"{model}: already patched")
        return 0
    run_patch(PATCH, args.site, dry_run=True)
    if args.check:
        print(f"{PATCH.name} applies cleanly to {args.site} (not written)")
        return 0
    run_patch(PATCH, args.site, dry_run=False)

    src = model.read_text()
    for old in (IMPORT_OLD, GATE_OLD):
        if src.count(old) != 1:
            raise SystemExit(f"{model}: anchor found {src.count(old)} times (expected 1):\n{old}")
    src = src.replace(IMPORT_OLD, IMPORT_NEW, 1).replace(GATE_OLD, GATE_NEW, 1)
    compile(src, str(model), "exec")
    model.write_text(src)

    args.tests_dir.mkdir(parents=True, exist_ok=True)
    if not (args.tests_dir / "tests/models/test_deepseek_v41_replay_batch.py").exists():
        run_patch(TESTS_PATCH, args.tests_dir, dry_run=False)
    print(f"{args.site}/vllm: vllm#58132 applied (decoder-side SWA bounded replay, kill switch "
          f"VLLM_MOET_DECODER_REPLAY=0); tests in {args.tests_dir}/tests/models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
