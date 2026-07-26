#!/usr/bin/env python3
"""GPU repro for the 2x5090 TP2 base-cache x FP4-pool x spec crash
(CUDA illegal memory access surfacing at miss_count.item()).

Serving evidence (2026-07-24): TP2 base+pool+gate WITHOUT spec = stable
for hours; TP1 base+pool+gate WITH MTP-1 = stable; TP2 WITH MTP-1 =
illegal access within minutes. The only decode path never op-tested is
_desc_build_kernel_base_delta (base cache + coexisting FP4 need-pool)
- this test drives exactly that at TP2 rank geometry (I_rank = I/2)
for T = 1..4 decode shapes with partial FP4 residency, checking
correctness AND surfacing the illegal access under compute-sanitizer.

Run (inside the vllm image, one GPU):
  VLLM_MOE_W2_CUBIT_DIR=/cubit-share VLLM_MOE_W2_BASE_CACHE_GB=1 \
  VLLM_MOE_W2_DELTA_GB=0.2 python3 tools/test_moe_w2_base_delta_tp2.py
Optionally under: compute-sanitizer --tool memcheck python3 ...
"""
import os
import sys

import torch

os.environ.setdefault("VLLM_MOE_W2", "1")
os.environ.setdefault("VLLM_MOE_W2_BASE_CACHE_GB", "1")
os.environ.setdefault("VLLM_MOE_W2_DELTA_GB", "0.2")

from vllm.model_executor.layers.quantization.utils import moe_w2_cubit  # noqa: E402
from vllm.model_executor.layers.quantization.utils import moe_w2_delta  # noqa: E402
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (  # noqa: E402
    mxfp4_to_codes, mxfp4_to_nibbles, pack_fp4_fragment_major,
    pack_fragment_major, pack_scales,
)

assert moe_w2_cubit._ensure_ready(), "cubins not found"
assert moe_w2_delta.base_enabled(), "BASE_CACHE_GB env expected"
dev = torch.device("cuda")
torch.manual_seed(11)

# TP2 rank geometry: H unsharded, expert I sharded in half.
E, H = 32, 4096
I_FULL = 2048
I = I_FULL // 2                      # per-rank intermediate (K2 = 1024)
TOPK = 6
LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0], device=dev)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2, device=dev)
E2M1[8:] *= -1

w13_pack = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
w2_pack = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, device=dev)

planes13 = torch.stack([pack_fragment_major(mxfp4_to_codes(w13_pack[e])) for e in range(E)])
sc13 = torch.stack([pack_scales(s13[e]) for e in range(E)])
planes2 = torch.stack([pack_fragment_major(mxfp4_to_codes(w2_pack[e])) for e in range(E)])
sc2 = torch.stack([pack_scales(s2[e]) for e in range(E)])
fp13 = torch.stack([pack_fp4_fragment_major(mxfp4_to_nibbles(w13_pack[e])) for e in range(E)])
fp2 = torch.stack([pack_fp4_fragment_major(mxfp4_to_nibbles(w2_pack[e])) for e in range(E)])

c13len, s13len = planes13.shape[1], sc13.shape[1]
c2len, s2len = planes2.shape[1], sc2.shape[1]
f13len, f2len = fp13.shape[1], fp2.shape[1]

N_KEYS = 6
btier = moe_w2_delta.get_base_tier(N_KEYS, E, dev,
                                   w13_bytes=c13len + s13len,
                                   w2_bytes=c2len + s2len)
# koegzystujaca pula FP4 (need-pool nad baza): slot = [fp4_13|sc13|fp4_2|sc2]
tier = moe_w2_delta.get_tier(N_KEYS, E, dev,
                             w13_bytes=f13len + s13len,
                             w2_bytes=f2len + s2len)
assert tier is not None, "FP4 need-pool expected (DELTA_GB set)"

for k in range(N_KEYS):
    btier.add_layer_host_planes(k, torch.cat((planes13, sc13), dim=1),
                                torch.cat((planes2, sc2), dim=1))
    tier.add_layer_host_sections(k, (fp13, sc13), (fp2, sc2))
    moe_w2_cubit._LAYERS[k] = dict(
        N13=2 * I, K13=H, N2=H, K2=I, E=E, base=True, tl_idx=k,
        off_s13=c13len, off_c2=c13len + s13len,
        off_s2=c13len + s13len + c2len,
        off4_s13=f13len, off4_c2=f13len + s13len,
        off4_s2=f13len + s13len + f2len,
    )
_key = iter(range(N_KEYS))


def dequant_w2(pack, sc):
    codes = mxfp4_to_codes(pack)
    return LEVELS[codes.long()] * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1)


def dequant_fp4(pack, sc):
    nib = mxfp4_to_nibbles(pack)
    return E2M1[nib.long()] * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1)


def reference(x, topk_w, topk_ids, fp4set, dropped=frozenset()):
    T = x.shape[0]
    a_deq = moe_w2_cubit.a32_dequant_ref(x, gemm=1)
    ref = torch.zeros(T, H, device=dev)
    for t in range(T):
        for j in range(TOPK):
            e = int(topk_ids[t, j])
            if e in dropped:
                continue
            dq = dequant_fp4 if e in fp4set else dequant_w2
            c13 = a_deq[t] @ dq(w13_pack[e], s13[e]).T
            act = torch.nn.functional.silu(c13[:I]) * c13[I:]
            act_deq = moe_w2_cubit.a32_dequant_ref(
                act.to(torch.bfloat16).unsqueeze(0), gemm=2)
            ref[t] += float(topk_w[t, j]) * (act_deq[0] @ dq(w2_pack[e], s2[e]).T)
    return ref


def check(tag, got, ref, tol=0.06):
    rel = (got.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref.flatten(), dim=0).item()
    okc = rel < tol and cos > 0.999
    print(f"{tag}: max_rel={rel:.3e} cos={cos:.6f} -> {'PASS' if okc else 'FAIL'}")
    return okc


ok = True
for T in (1, 2, 3, 4):
    x = (torch.randn(T, H, device=dev) * 0.3).to(torch.bfloat16)
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).to(torch.int32)
    topk_w = torch.rand(T, TOPK, device=dev) * 0.5
    routed = sorted(set(topk_ids.flatten().tolist()))
    base_missing = set(routed[::3])           # co trzeci routowany: MISS bazy
    fp4_resident = set(routed[1::3])          # co trzeci: FP4-resident

    k = next(_key)
    btier.ensure_resident(k, torch.tensor(
        sorted(set(range(E)) - base_missing), device=dev))
    for e in fp4_resident:
        tier._promote(k, e, tier._take_slot(set()))
    torch.cuda.synchronize()

    btier.miss_count.zero_()
    got = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, k)
    torch.cuda.synchronize()                  # <- tu wyplynie illegal access
    m = int(btier.miss_count.item())
    exp_miss = len(base_missing - fp4_resident)
    print(f"T={T}: miss={m} (oczekiwane >= {exp_miss} z {len(routed)} routed; "
          f"fp4={len(fp4_resident)})")
    ok &= (m >= exp_miss)
    ok &= check(f"T={T} base+delta mixed (missing zeroed)",
                got, reference(x, topk_w, topk_ids, fp4_resident,
                               dropped=base_missing - fp4_resident))

print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
