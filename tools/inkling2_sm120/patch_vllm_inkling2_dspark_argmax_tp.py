#!/usr/bin/env python3
"""Inkling2 step 6: the DSpark draft's argmax without vLLM's multicast kernel.

Where: `Inkling2DSparkModel` in vllm/models/inkling2/dspark.py. The draft's Markov head takes the argmax of the
softcapped logits, whose vocabulary is sharded over the TP ranks; vLLM gathers each rank's per-block (max, index)
pairs through multicast Lamport buffers (`Inkling2DSparkArgmax`: FlashInfer MNNVL memory), and the workspace asserts
NVLink multicast when the TP group has more than one rank - four PCIe cards have none, so the draft does not load.

What: where multicast is not available and the group has more than one rank - and on every capability-12 device,
whose image of vLLM's compiled library has no image of the argmax kernel, a single rank included - no workspace is
made and `markov_argmax` computes the same value another way: each rank takes the first argmax of its own shard of
`add_softcap(base, bias)` (NaN as -inf), the ranks all-gather their (maximum, global index) pair - two fp32 per row -
and the first maximum over the ranks wins; the shards are in vocabulary order, so that is the first argmax of the
full row, what `markov_logits(...).argmax(-1)` returns. Other devices with multicast, and their single-rank
groups, keep vLLM's path. VLLM_MOET_INKLING2_DSPARK_ARGMAX_TP=0 turns the step off (the draft then needs vLLM's
kernel and, with several ranks, multicast).

Usage: python3 patch_vllm_inkling2_dspark_argmax_tp.py [--root DIR] [--check]   (idempotent; refuses any other text)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/inkling2")
REL = "dspark.py"
MARKER = "[vllm-moet] inkling2 DSpark argmax over the TP group"

WORKSPACE_ANCHOR = '''        self.argmax_workspace = Inkling2DSparkArgmax(
            get_tp_group().cpu_group, self.max_num_reqs, num_cols
        )
'''
WORKSPACE_NEW = '''        self.argmax_workspace = (
            None
            if _moet_argmax_without_multicast()
            else Inkling2DSparkArgmax(get_tp_group().cpu_group, self.max_num_reqs, num_cols)
        )
'''
ARGMAX_ANCHOR = '''        """``markov_logits(...).argmax(-1)``, NaN as -inf."""
        return dspark_argmax(
'''
ARGMAX_NEW = '''        """``markov_logits(...).argmax(-1)``, NaN as -inf."""
        if self.argmax_workspace is None:
            return _moet_argmax_tp(
                base_logits,
                bias,
                self.config.final_logit_softcapping,
                self.valid_vocab,
                self.vocab_offset,
            )
        return dspark_argmax(
'''
HELPERS_ANCHOR = '''class Inkling2DSparkModel(nn.Module):
'''
HELPERS = f'''def _moet_argmax_without_multicast() -> bool:
    """{MARKER}: where vLLM's argmax kernel cannot run - a TP group of several ranks on cards without NVLink
    multicast (PCIe), and any capability-12 device (the compiled library of the base image has no image of the
    kernel for it, also for a single rank)."""
    import os

    if os.environ.get("VLLM_MOET_INKLING2_DSPARK_ARGMAX_TP", "1") == "0":
        return False
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
        return True
    if get_tensor_model_parallel_world_size() == 1:
        return False
    try:
        from flashinfer.comm.mnnvl import is_multicast_supported

        return not is_multicast_supported(torch.accelerator.current_device_index())
    except Exception:  # noqa: BLE001
        return True


def _moet_argmax_tp(
    base: torch.Tensor, bias: torch.Tensor, softcap: float, valid: int, offset: int
) -> torch.Tensor:
    """The first argmax (NaN as -inf) of the full softcapped row from the ranks' shards: every rank's first local
    maximum and its global index, all-gathered (two fp32 per row and rank), then the first maximum over the ranks."""
    local = add_softcap(base, bias, softcap)[:, :valid].float()
    local = torch.where(torch.isnan(local), torch.full_like(local, float("-inf")), local)
    value, index = local.max(dim=1)
    pair = torch.stack((value, (index + offset).float()), dim=1)
    pairs = tensor_model_parallel_all_gather(pair, dim=1)
    values, indices = pairs[:, 0::2], pairs[:, 1::2]
    best = values.argmax(dim=1, keepdim=True)
    return indices.gather(1, best).squeeze(1).long()


class Inkling2DSparkModel(nn.Module):
'''

EDITS = [(HELPERS_ANCHOR, HELPERS), (WORKSPACE_ANCHOR, WORKSPACE_NEW), (ARGMAX_ANCHOR, ARGMAX_NEW)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = a.root / REL
    text = path.read_text()
    if a.check:
        print("patched" if MARKER in text else "not patched")
        return 0 if MARKER in text else 1
    if MARKER in text:
        print(f"already patched {path}")
        return 0
    for old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"{REL}: anchor not found exactly once (another vLLM version?): {old.strip().splitlines()[0]!r}")
        text = text.replace(old, new)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
