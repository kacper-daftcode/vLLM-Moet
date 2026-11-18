#!/usr/bin/env python3
"""Inkling2 step 9: the draft's CUDA graphs follow the draft schedule.

Where: `DFlashSpeculator.init_cudagraph_manager` in vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (the DSpark
speculators derive from it). The draft's query pass is captured as a FULL CUDA graph per batch size, up to
`max_num_seqs`. With vLLM's draft length by batch size (`num_speculative_tokens_per_batch_size`) the query pass runs
only at batch sizes whose length is not 0 - at the others `propose` stores the context K / V and returns - and the
target's own graphs are already captured by that rule (a query length only up to the largest batch size that drafts
it). The draft's are not: a graph is captured, and its memory kept, for every batch size up to `max_num_seqs`. A
DSpark draft graph holds logits over the vocabulary per request and Markov step; on a card with 86 GiB of weights
that is the room of the paged cache (1.39 GiB of graphs with the draft at 32 sequences against 0.19 GiB without).

What: the draft's graphs are captured up to the largest batch size the schedule drafts at. Larger batches never run
the query pass; a call that would (vLLM's warm-up) runs it without a graph. No schedule: nothing changes.
VLLM_MOET_DRAFT_GRAPHS_ALL=1 keeps vLLM's rule.

Usage: python3 patch_vllm_draft_graphs_by_schedule.py [--root DIR] [--check]   (idempotent; refuses any other file text)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash")
REL = "speculator.py"
MARKER = "[vllm-moet] draft graphs by the draft schedule"

ANCHOR = '''        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_lens={self.num_query_per_req: self.max_num_reqs},
        )
'''
NEW = '''        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_lens={
                self.num_query_per_req: _moet_max_draft_reqs(
                    self.vllm_config, self.max_num_reqs
                )
            },
        )
'''
HELPER_ANCHOR = '''class DFlashSpeculator('''
HELPER = f'''def _moet_max_draft_reqs(vllm_config, max_num_reqs: int) -> int:
    """{MARKER}: the largest batch size at which the draft's query pass runs. With a draft length by batch size
    it is the largest batch size whose length is not 0 - graphs of larger batches would never be replayed. Without a
    schedule, or with VLLM_MOET_DRAFT_GRAPHS_ALL=1, it is max_num_seqs (vLLM's rule)."""
    import os

    spec = vllm_config.speculative_config
    if (
        os.environ.get("VLLM_MOET_DRAFT_GRAPHS_ALL", "0") == "1"
        or spec is None
        or not spec.uses_dynamic_speculative_decoding()
    ):
        return max_num_reqs
    from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

    dense = build_dynamic_sd_schedule_lookup(
        spec.num_speculative_tokens_per_batch_size,
        vllm_max_batch_size=max_num_reqs,
        vllm_num_speculative_tokens=spec.num_speculative_tokens,
    )
    drafting = [batch for batch in range(1, max_num_reqs + 1) if dense[batch] > 0]
    return max(drafting) if drafting else 1


''' + HELPER_ANCHOR

EDITS = ((HELPER_ANCHOR, HELPER), (ANCHOR, NEW))


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
    for anchor, new in EDITS:
        if text.count(anchor) != 1:
            raise SystemExit(f"{REL}: anchor not found exactly once (another vLLM version?):\\n{anchor}")
        text = text.replace(anchor, new)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
