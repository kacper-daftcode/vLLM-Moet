#!/usr/bin/env python3
"""Step 9 without a card: the largest batch size the draft's CUDA graphs are captured for, by the draft schedule.

Run in the image: python3 test_draft_graphs_by_schedule.py   (imports the patched speculator module; no GPU)
With --source FILE --utils FILE it reads the function from a patched copy of dflash/speculator.py and vLLM's
v1/spec_decode/dynamic/utils.py instead (a host without vLLM).
"""
import argparse
import importlib.util
import os
import sys
import types

ap = argparse.ArgumentParser()
ap.add_argument("--source")
ap.add_argument("--utils")
a = ap.parse_args()
if a.source:
    for name in ("vllm", "vllm.v1", "vllm.v1.spec_decode", "vllm.v1.spec_decode.dynamic"):
        sys.modules.setdefault(name, types.ModuleType(name))
    spec = importlib.util.spec_from_file_location("vllm.v1.spec_decode.dynamic.utils", a.utils)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    sys.modules["vllm.v1.spec_decode.dynamic.utils"] = utils
    text = open(a.source).read()
    start = text.index("def _moet_max_draft_reqs(")
    ns: dict = {}
    exec(text[start:text.index("\nclass DFlashSpeculator(", start)], ns)
    fn = ns["_moet_max_draft_reqs"]
else:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import _moet_max_draft_reqs as fn


def config(schedule, k=7):
    spec = types.SimpleNamespace(
        num_speculative_tokens_per_batch_size=schedule, num_speculative_tokens=k,
        uses_dynamic_speculative_decoding=lambda: schedule is not None)
    return types.SimpleNamespace(speculative_config=spec)


CASES = [
    (None, 32, 32, "no schedule: every batch size"),
    ([[1, 8, 7], [9, 32, 0]], 32, 8, "the draft up to 8 requests"),
    ([[1, 2, 7], [3, 8, 0]], 8, 2, "the draft up to 2 requests"),
    ([[1, 4, 7], [5, 8, 3], [9, 64, 0]], 32, 8, "a shorter draft still drafts"),
    ([[1, 8, 7]], 32, 32, "the last range's length is carried to max_num_seqs"),
    ([[1, 16, 7], [17, 64, 0]], 8, 8, "a schedule wider than max_num_seqs"),
    ([[1, 32, 0]], 32, 1, "no batch size drafts"),
]
ok = True
for schedule, max_reqs, want, what in CASES:
    got = fn(config(schedule), max_reqs)
    ok &= got == want
    print(f"{'ok  ' if got == want else 'FAIL'} {str(schedule):38s} max_num_seqs {max_reqs:2d} -> {got:2d} (want {want}): {what}")
os.environ["VLLM_MOET_DRAFT_GRAPHS_ALL"] = "1"
got = fn(config([[1, 8, 7], [9, 32, 0]]), 32)
ok &= got == 32
print(f"{'ok  ' if got == 32 else 'FAIL'} VLLM_MOET_DRAFT_GRAPHS_ALL=1 -> {got} (want 32): vLLM's rule")
got = fn(types.SimpleNamespace(speculative_config=None), 16)
ok &= got == 16
print(f"{'ok  ' if got == 16 else 'FAIL'} no speculative config -> {got} (want 16)")
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
