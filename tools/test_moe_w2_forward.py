#!/usr/bin/env python3
"""GPU unit test: moe_w2_cubit._moe_w2_forward vs torch reference.

Synthetic 2-layer-free test: builds random e2m1 expert weights, quantizes
through the SAME plane builder the loader uses, then checks the full glue
path (align/gather/desc/GEMM/silu/quant/GEMM/scatter) against an f32
reference computed from the dequantized 2-bit levels.

Run (inside the vllm image): python3 test_moe_w2_forward.py
"""
import os
import sys

import torch

os.environ.setdefault("VLLM_MOE_W2", "1")

from vllm.model_executor.layers.quantization.utils import moe_w2_cubit  # noqa: E402
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (  # noqa: E402
    mxfp4_to_codes, pack_fragment_major, pack_scales,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8,
)

assert moe_w2_cubit._ensure_ready(), "cubins not found"
dev = torch.device("cuda")
torch.manual_seed(11)

E = int(os.environ.get("E", "32"))
H = int(os.environ.get("H", "4096"))     # 4096 DS4, 6144 GLM-5.x, 7168 Kimi-K2.x
I = int(os.environ.get("I", "2048"))     # per-rank I under TP (1024 TP2, 512 TP4)
T = int(os.environ.get("T", "9"))        # T>96 exercises the PREFILL tier (mc4/afrag)
TOPK = 6
LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0], device=dev)

w13_pack = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
w2_pack = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, device=dev)

st = dict(N13=2 * I, K13=H, N2=H, K2=I, E=E)
st["planes13"] = torch.stack([pack_fragment_major(mxfp4_to_codes(w13_pack[e])) for e in range(E)])
st["sc13"] = torch.stack([pack_scales(s13[e]) for e in range(E)])
st["planes2"] = torch.stack([pack_fragment_major(mxfp4_to_codes(w2_pack[e])) for e in range(E)])
st["sc2"] = torch.stack([pack_scales(s2[e]) for e in range(E)])
moe_w2_cubit._LAYERS[0] = st


def dequant(pack, sc):
    codes = mxfp4_to_codes(pack)
    return LEVELS[codes.long()] * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1)


x = (torch.randn(T, H, device=dev) * 0.3).to(torch.bfloat16)
topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).to(torch.int32)
topk_w = torch.rand(T, TOPK, device=dev) * 0.5

got = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)

# ---- reference with the same activation-quant numerics
a8, as8 = per_token_group_quant_fp8(x, 128)
a_deq = a8.float() * as8.repeat_interleave(128, 1)
ref = torch.zeros(T, H, device=dev)
for t in range(T):
    for j in range(TOPK):
        e = int(topk_ids[t, j])
        w13d = dequant(w13_pack[e], s13[e])
        c13 = a_deq[t] @ w13d.T
        act = torch.nn.functional.silu(c13[:I]) * c13[I:]
        q2, qs2 = per_token_group_quant_fp8(act.to(torch.bfloat16).unsqueeze(0), 128)
        act_deq = q2.float() * qs2.repeat_interleave(128, 1)
        w2d = dequant(w2_pack[e], s2[e])
        ref[t] += float(topk_w[t, j]) * (act_deq[0] @ w2d.T)

rel = (got.float() - ref).abs().max().item() / ref.abs().max().item()
cos = torch.nn.functional.cosine_similarity(
    got.float().flatten(), ref.flatten(), dim=0).item()
print(f"T={T} E={E}: max_rel={rel:.3e} cos={cos:.6f}")
ok = rel < 0.06 and cos > 0.999

# ---- delta tier: promote half the experts to full FP4, expect the mixed
# reference (FP4 dequant for promoted, 2-bit for the rest).
# DECODE-ONLY by design: the production forward routes every pair to the
# 2-bit tier at prefill (T > 96), so the mixed reference does not apply there.
if T > 96:
    print("DELTA mixed: skipped (prefill tier is 2-bit-only by design)")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
from vllm.model_executor.layers.quantization.utils import moe_w2_delta
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
    mxfp4_to_nibbles, pack_fp4_fragment_major)

os.environ["VLLM_MOE_W2_DELTA_GB"] = "1"
# per-expert FP4 plane bytes for THIS model's shapes (the module defaults are
# the DS4 TP1 sizes; production passes these via _fp4_tier_for_build)
tier = moe_w2_delta.DeltaTier(1, E, dev,
                              w13_bytes=2 * I * H // 2,
                              w2_bytes=H * I // 2)
moe_w2_delta._TIER = tier
fp13 = torch.stack([pack_fp4_fragment_major(mxfp4_to_nibbles(w13_pack[e]))
                    for e in range(E)])
fp2 = torch.stack([pack_fp4_fragment_major(mxfp4_to_nibbles(w2_pack[e]))
                   for e in range(E)])
tier.add_layer_host_planes(0, fp13, fp2)
promoted = list(range(0, E, 2))
for e in promoted:
    slot = tier._take_slot(set())
    tier._promote(0, e, slot)
torch.cuda.synchronize()

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2, device=dev)
E2M1[8:] *= -1


def dequant_fp4(pack, sc):
    nib = mxfp4_to_nibbles(pack)
    return E2M1[nib.long()] * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1)


got2 = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
ref2 = torch.zeros(T, H, device=dev)
for t in range(T):
    for j in range(TOPK):
        e = int(topk_ids[t, j])
        dq13 = dequant_fp4 if e in promoted else dequant
        dq2 = dequant_fp4 if e in promoted else dequant
        w13d = (dq13(w13_pack[e], s13[e]))
        c13 = a_deq[t] @ w13d.T
        act = torch.nn.functional.silu(c13[:I]) * c13[I:]
        q2, qs2 = per_token_group_quant_fp8(act.to(torch.bfloat16).unsqueeze(0), 128)
        act_deq = q2.float() * qs2.repeat_interleave(128, 1)
        w2d = dq2(w2_pack[e], s2[e])
        ref2[t] += float(topk_w[t, j]) * (act_deq[0] @ w2d.T)

rel2 = (got2.float() - ref2).abs().max().item() / ref2.abs().max().item()
cos2 = torch.nn.functional.cosine_similarity(
    got2.float().flatten(), ref2.flatten(), dim=0).item()
print(f"DELTA mixed ({len(promoted)}/{E} promoted): max_rel={rel2:.3e} "
      f"cos={cos2:.6f}")
ok = ok and rel2 < 0.06 and cos2 > 0.999

# ---- SPLIT FP4 (VLLM_MOE_W2_DELTA_SPLIT): the delta slots hold 2-bit
# REFINEMENT planes and moe_w4s_mm reads them alongside the resident base.
# Reference: split_fp4_dequant (true FP4 modulo the mag-0 -> 0.5 merge).
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
    nibbles_to_refinement, split_fp4_dequant)

moe_w2_delta._SPLIT = True          # env is read at import; force for test
assert moe_w2_delta.split_enabled()
tier_s = moe_w2_delta.DeltaTier(1, E, dev,
                                w13_bytes=2 * I * H // 4,
                                w2_bytes=H * I // 4)
moe_w2_delta._TIER = tier_s
rf13 = torch.stack([pack_fragment_major(
    nibbles_to_refinement(mxfp4_to_nibbles(w13_pack[e]))) for e in range(E)])
rf2 = torch.stack([pack_fragment_major(
    nibbles_to_refinement(mxfp4_to_nibbles(w2_pack[e]))) for e in range(E)])
assert rf13.shape[1] == 2 * I * H // 4 and rf2.shape[1] == H * I // 4
tier_s.add_layer_host_planes(0, rf13, rf2)
for e in promoted:
    slot = tier_s._take_slot(set())
    tier_s._promote(0, e, slot)
torch.cuda.synchronize()


def dequant_split(pack, sc):
    nib = mxfp4_to_nibbles(pack)
    return (split_fp4_dequant(nib)
            * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1))


got3 = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
ref3 = torch.zeros(T, H, device=dev)
for t in range(T):
    for j in range(TOPK):
        e = int(topk_ids[t, j])
        dq = dequant_split if e in promoted else dequant
        w13d = dq(w13_pack[e], s13[e])
        c13 = a_deq[t] @ w13d.T
        act = torch.nn.functional.silu(c13[:I]) * c13[I:]
        q2, qs2 = per_token_group_quant_fp8(act.to(torch.bfloat16).unsqueeze(0), 128)
        act_deq = q2.float() * qs2.repeat_interleave(128, 1)
        w2d = dq(w2_pack[e], s2[e])
        ref3[t] += float(topk_w[t, j]) * (act_deq[0] @ w2d.T)

rel3 = (got3.float() - ref3).abs().max().item() / ref3.abs().max().item()
cos3 = torch.nn.functional.cosine_similarity(
    got3.float().flatten(), ref3.flatten(), dim=0).item()
print(f"SPLIT mixed ({len(promoted)}/{E} promoted, slots {rf13.shape[1]}+"
      f"{rf2.shape[1]} B): max_rel={rel3:.3e} cos={cos3:.6f}")
ok = ok and rel3 < 0.06 and cos3 > 0.999
moe_w2_delta._SPLIT = False
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
