#!/usr/bin/env python3
"""Offline functional check + demo of the FP4 delta tier's observability.

Drives the *real* `DeltaTier` manager (promote / evict / periodic summary /
JSON dump / `precision_map`) with a synthetic but realistic MoE routing stream
— Zipfian top-k experts per layer at the true DeepSeek-V4 shape (44 layers x
256 experts, ~170 hot slots) — so you can see exactly what
`VLLM_MOE_W2_DELTA_TRACE` logs *without* loading the 159B model.

The per-slot byte size is shrunk to a few bytes; the manager logic is
byte-size-agnostic (it works on slot indices and (layer, expert) pairs), so
this runs on any CUDA device in ~1 s and a few MB while exercising the exact
code path used in production.

Run inside the vLLM image on any free GPU:

    docker run --rm --gpus '"device=0"' --entrypoint python3 \
        -v "$PWD/tools/delta_trace_demo.py:/demo.py" \
        <vllm-moet-image> /demo.py

Override the trace level/cadence the same way production does, e.g.
`-e VLLM_MOE_W2_DELTA_TRACE=1 -e VLLM_MOE_W2_DELTA_TRACE_EVERY=50`.
"""
import os

# These knobs are read at import time, so set loud defaults before importing
# the module. A real `-e VLLM_MOE_W2_DELTA_TRACE=...` from the environment wins.
os.environ.setdefault("VLLM_MOE_W2_DELTA_TRACE", "2")
os.environ.setdefault("VLLM_MOE_W2_DELTA_TRACE_EVERY", "25")
os.environ.setdefault("VLLM_MOE_W2_DELTA_DUMP", "/tmp/delta_demo.json")
os.environ.setdefault("VLLM_MOE_W2_DELTA_GB", "2.0")

import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch

import vllm.model_executor.layers.quantization.utils.moe_w2_delta as d

N_LAYERS, N_EXPERTS, TOPK, STEPS = 44, 256, 8, 200
DEV = torch.device("cuda", 0)

# Shrink per-slot bytes (logic unchanged) and size the pool to ~170 slots, the
# count a ~2 GiB production pool holds at the real 12.6 MiB/expert.
d.W13_BYTES, d.W2_BYTES, d.SLOT_BYTES = 96, 32, 128
d._GB = 170 * d.SLOT_BYTES / 2**30

tier = d.DeltaTier(N_LAYERS, N_EXPERTS, DEV)
for li in range(N_LAYERS):  # register tiny dummy host planes for every layer
    tier.add_layer_host_planes(
        li,
        torch.zeros(N_EXPERTS, d.W13_BYTES, dtype=torch.uint8, device=DEV),
        torch.zeros(N_EXPERTS, d.W2_BYTES, dtype=torch.uint8, device=DEV))

# Zipfian expert hotness with a per-layer permutation (each layer has its own
# hot set, like real routing): a minority of experts carry most of the mass.
ranks_w = 1.0 / torch.arange(1, N_EXPERTS + 1).float() ** 1.1
ranks_w /= ranks_w.sum()
perm = torch.stack([torch.randperm(N_EXPERTS) for _ in range(N_LAYERS)])

print(f"# driving the real DeltaTier: {N_LAYERS}x{N_EXPERTS} experts, "
      f"{tier.n_slots} slots, top-{TOPK}/layer, {STEPS} steps\n", flush=True)
for _ in range(STEPS):
    sel = torch.zeros(N_LAYERS, N_EXPERTS, dtype=torch.uint8, device=DEV)
    for li in range(N_LAYERS):
        ranks = torch.multinomial(ranks_w, TOPK, replacement=False)
        sel[li, perm[li][ranks].to(DEV)] = 1
    tier.seen.copy_(sel)
    tier._tick_once()

print("\n# === final state via the public API ===", flush=True)
pm = tier.precision_map()
print("stats():", tier.stats())
print(f"layers covered: {len(pm)}/{N_LAYERS}; "
      f"experts in FP4: {sum(len(v) for v in pm.values())}")
hot0, cold0 = int(perm[0][0]), int(perm[0][-1])
print(f"precision_of(L0, E{hot0}) = {tier.precision_of(0, hot0)!r}  (hottest)")
print(f"precision_of(L0, E{cold0}) = {tier.precision_of(0, cold0)!r}  (coldest)")

dump = json.load(open(os.environ["VLLM_MOE_W2_DELTA_DUMP"]))
print(f"\n# === JSON dump @ {os.environ['VLLM_MOE_W2_DELTA_DUMP']} ===")
print({k: dump[k] for k in
       ("tick", "n_slots", "cached", "promoted_total", "evicted_total")})
print("fp4_by_layer['0'] =", dump["fp4_by_layer"].get("0"))
