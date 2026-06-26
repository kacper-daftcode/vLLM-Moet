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

E, H, I = 32, 4096, 2048
T, TOPK = 9, 6
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
# reference (FP4 dequant for promoted, 2-bit for the rest)
from vllm.model_executor.layers.quantization.utils import moe_w2_delta
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
    mxfp4_to_nibbles, pack_fp4_fragment_major)

os.environ["VLLM_MOE_W2_DELTA_GB"] = "1"
tier = moe_w2_delta.DeltaTier(1, E, dev)
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
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
