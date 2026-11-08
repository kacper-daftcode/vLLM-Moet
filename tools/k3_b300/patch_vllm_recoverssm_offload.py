#!/usr/bin/env python3
"""Kimi-K3 RecoverSSM (`--use-replayssm`) next to a KV connector (KV offload, P/D).

vLLM v0.32.0's `VllmConfig.validate_mamba_cached_kernel` rejects `--use-replayssm`
whenever a KV connector is configured: "--use-replayssm is incompatible with KV
connectors (P/D disaggregation, KV cache offload)". The rule is right for Mamba2
ReplaySSM, which caches recent SSM inputs and writes the SSM state back only on a
flush, so a connector could copy a stale page. On Kimi-K3 the same flag selects a
different kernel, RecoverSSM (`cache_config.use_kda_recoverssm`): the KDA verify
runs off one checkpoint state, and `KDARecoverSSMCommitContext.commit` restores the
accepted state into that same slot in the step's postprocess
(`MambaHybridModelState.postprocess_state` -> `RecoverSSMState.commit_step`), which
the V2 runner runs before `kv_connector.post_forward`. The pages a connector sees are
therefore always current; the two extra per-layer records (the fp32 correction and
the key/gate buffer) are per-step scratch and can be copied or dropped freely.

Why it matters: with DSpark on K3 every KDA layer keeps one full state slot per draft
token (`MambaSpec.num_speculative_blocks = num_speculative_tokens`, align mode:
2 + k + 1 blocks per request per KDA group; a block is 29 pages of 0.9 MB). At k = 4
that is 12 extra blocks of 25.7 MB per request on every GPU - 308 MB, the equivalent
of ~150K tokens of MLA KV - and it is what pushed the 8x B300 pool from 29.4M to
19.3M tokens. RecoverSSM sets `num_speculative_blocks = 0`. Measured on hgx-02
(TP8 / DCP8, fp8 KV, DSpark k = 4, 2026-10-24): 19,256,559 -> 21,614,964 tokens of
KV at the same memory, a 32K-token request holds 9 instead of 21-24 blocks
(250 MB instead of 539 MB), single-stream code decode 227 -> 220 tok/s at the same
accepted length (3.35 / 3.34), GSM8K-200 98.0%, greedy output semantically
unchanged (not token-identical: the verify kernel's fp32 recurrence differs).

This lifts the check only when `use_kda_recoverssm` is set (K3 + speculative
decoding); Mamba2 ReplaySSM keeps the restriction.

Idempotent, anchor-based. Usage:
    python3 patch_vllm_recoverssm_offload.py [--file PATH] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py")
MARKER = "# [vllm-moet] K3 RecoverSSM commits its state in-step; connectors are safe"

OLD = (
    "        if (\n"
    "            self.kv_transfer_config is not None\n"
    "            and self.kv_transfer_config.is_kv_transfer_instance\n"
    "        ):\n"
    "            raise ValueError(\n"
    '                "--use-replayssm is incompatible with KV connectors "\n'
    '                "(P/D disaggregation, KV cache offload)"\n'
    "            )\n"
)
NEW = (
    "        if (\n"
    "            self.kv_transfer_config is not None\n"
    "            and self.kv_transfer_config.is_kv_transfer_instance\n"
    f"            {MARKER}:\n"
    "            # KDARecoverSSMCommitContext.commit restores the accepted KDA\n"
    "            # state into its single slot in the same step's postprocess,\n"
    "            # before kv_connector.post_forward; only Mamba2 ReplaySSM defers\n"
    "            # the state store to a flush.\n"
    "            and not self.cache_config.use_kda_recoverssm\n"
    "        ):\n"
    "            raise ValueError(\n"
    '                "--use-replayssm is incompatible with KV connectors "\n'
    '                "(P/D disaggregation, KV cache offload)"\n'
    "            )\n"
)


def patch_text(src: str) -> tuple[str, str]:
    if MARKER in src:
        return src, "already patched"
    n = src.count(OLD)
    if n != 1:
        raise SystemExit(f"anchor found {n} times (expected 1):\n{OLD}")
    return src.replace(OLD, NEW, 1), "K3 RecoverSSM admitted next to KV connectors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    patched, what = patch_text(src)
    if patched is src:
        print(f"{args.file}: {what}")
        return 0
    compile(patched, str(args.file), "exec")
    if args.check:
        print(f"{args.file}: patch applies cleanly ({what}; not written)")
        return 0
    args.file.write_text(patched)
    print(f"{args.file}: patched ({what})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
