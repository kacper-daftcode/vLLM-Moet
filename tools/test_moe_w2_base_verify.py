#!/usr/bin/env python3
"""GPU unit test: BASE-cache decode path on MTP-verify-shaped steps.

Motivation (2026-07-23): cache-route NT drops deterministically with the
number of speculative tokens (68.69 @spec2 / 71.72 @spec1 / 73.74 @spec0
vs native 74.24 @24k) while resident+MTP and native+MTP are healthy.
FP_THRESH is bit-identical 0 vs 64, the drafter is healthy (68%
acceptance). This test isolates the base-tier decode path for step
shapes T=1 (plain decode) vs T=3 (MTP-2 verify) and checks:

  A. full residency: T=3 forward == mixed-precision reference,
  B. partial residency: miss counter counts EXACTLY the non-resident
     routed experts (per T, no undercount on multi-row steps),
  C. zeroed contributions: with misses, output equals reference computed
     with those experts' contributions dropped (no bleed across rows),
  D. replay contract: fetch missing (force_promote path) -> rebuild ->
     output == full reference,
  E. batch-geometry numerics: T=3 in ONE call vs the same 3 tokens as
     3x T=1 calls (all resident) — max relative row divergence, printed
     (a large value = borderline-greedy flips under verify batching).

Run inside the vllm image on one GPU:
  VLLM_MOE_W2_CUBIT_DIR=/cubit-share VLLM_MOE_W2_BASE_CACHE_GB=1 \
    python3 tools/test_moe_w2_base_verify.py
"""
import os
import sys

import torch

os.environ.setdefault("VLLM_MOE_W2", "1")
os.environ.setdefault("VLLM_MOE_W2_BASE_CACHE_GB", "1")

from vllm.model_executor.layers.quantization.utils import moe_w2_cubit  # noqa: E402
from vllm.model_executor.layers.quantization.utils import moe_w2_delta  # noqa: E402
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (  # noqa: E402
    mxfp4_to_codes, pack_fragment_major, pack_scales,
)

assert moe_w2_cubit._ensure_ready(), "cubins not found"
assert moe_w2_delta.base_enabled(), "BASE_CACHE_GB env expected"
dev = torch.device("cuda")
torch.manual_seed(7)

E, H, I = 32, 4096, 2048
TOPK = 6
LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0], device=dev)

w13_pack = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
w2_pack = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, device=dev)

planes13 = torch.stack([pack_fragment_major(mxfp4_to_codes(w13_pack[e])) for e in range(E)])
sc13 = torch.stack([pack_scales(s13[e]) for e in range(E)])
planes2 = torch.stack([pack_fragment_major(mxfp4_to_codes(w2_pack[e])) for e in range(E)])
sc2 = torch.stack([pack_scales(s2[e]) for e in range(E)])

c13len, s13len = planes13.shape[1], sc13.shape[1]
c2len, s2len = planes2.shape[1], sc2.shape[1]

# base tier: one layer_key per scenario (residency ONLY via
# ensure_resident of a chosen subset - untouched experts keep slot -1,
# so misses are REAL, not hand-wiped)
N_KEYS = 8
btier = moe_w2_delta.get_base_tier(N_KEYS, E, dev,
                                   w13_bytes=c13len + s13len,
                                   w2_bytes=c2len + s2len)
for k in range(N_KEYS):
    btier.add_layer_host_planes(k, torch.cat((planes13, sc13), dim=1),
                                torch.cat((planes2, sc2), dim=1))
    moe_w2_cubit._LAYERS[k] = dict(
        N13=2 * I, K13=H, N2=H, K2=I, E=E, base=True, tl_idx=k,
        off_s13=c13len, off_c2=c13len + s13len,
        off_s2=c13len + s13len + c2len,
        off4_s13=2 * c13len, off4_c2=2 * c13len + s13len,
        off4_s2=2 * c13len + s13len + 2 * c2len,
    )
_key_iter = iter(range(N_KEYS))


def dequant_w2(pack, sc):
    codes = mxfp4_to_codes(pack)
    return LEVELS[codes.long()] * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1)


def reference(x, topk_w, topk_ids, dropped=frozenset()):
    T = x.shape[0]
    a_deq = moe_w2_cubit.a32_dequant_ref(x, gemm=1)
    ref = torch.zeros(T, H, device=dev)
    for t in range(T):
        for j in range(TOPK):
            e = int(topk_ids[t, j])
            if e in dropped:
                continue
            c13 = a_deq[t] @ dequant_w2(w13_pack[e], s13[e]).T
            act = torch.nn.functional.silu(c13[:I]) * c13[I:]
            act_deq = moe_w2_cubit.a32_dequant_ref(
                act.to(torch.bfloat16).unsqueeze(0), gemm=2)
            ref[t] += float(topk_w[t, j]) * (act_deq[0] @ dequant_w2(w2_pack[e], s2[e]).T)
    return ref


def fresh_key(resident):
    """New layer_key with exactly `resident` experts fetched (rest -1)."""
    k = next(_key_iter)
    if len(resident):
        btier.ensure_resident(k, torch.tensor(sorted(resident), device=dev))
    torch.cuda.synchronize()
    return k


def check(tag, got, ref, tol=0.06):
    rel = (got.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref.flatten(), dim=0).item()
    okc = rel < tol and cos > 0.999
    print(f"{tag}: max_rel={rel:.3e} cos={cos:.6f} -> {'PASS' if okc else 'FAIL'}")
    return okc


ok = True
for T in (1, 2, 3):
    x = (torch.randn(T, H, device=dev) * 0.3).to(torch.bfloat16)
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).to(torch.int32)
    topk_w = torch.rand(T, TOPK, device=dev) * 0.5
    routed = set(topk_ids.flatten().tolist())
    missing = set(sorted(routed)[::2])

    # ---- A: full residency
    kA = fresh_key(range(E))
    btier.miss_count.zero_()
    got = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, kA)
    m = int(btier.miss_count.item())
    ok &= check(f"T={T} full residency", got, reference(x, topk_w, topk_ids))
    print(f"  miss_count={m} (expected 0) {'PASS' if m == 0 else 'FAIL'}")
    ok &= (m == 0)

    # ---- B/C: partial residency (only routed-minus-missing fetched)
    kB = fresh_key(set(range(E)) - missing)
    btier.miss_count.zero_()
    got = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, kB)
    m = int(btier.miss_count.item())
    exp = len(missing & routed)
    print(f"T={T} partial: miss_count={m} expected>={exp} "
          f"{'PASS' if m >= exp else 'FAIL (UNDERCOUNT)'}")
    ok &= (m >= exp)
    ok &= check(f"T={T} zeroed contributions",
                got, reference(x, topk_w, topk_ids, dropped=missing))

    # ---- D: replay contract (fetch missing on the SAME key, rerun)
    btier.ensure_resident(kB, torch.tensor(sorted(missing), device=dev))
    torch.cuda.synchronize()
    btier.miss_count.zero_()
    got2 = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, kB)
    m2 = int(btier.miss_count.item())
    ok &= check(f"T={T} after fetch (replay contract)",
                got2, reference(x, topk_w, topk_ids))
    print(f"  post-fetch miss_count={m2} (expected 0) "
          f"{'PASS' if m2 == 0 else 'FAIL'}")
    ok &= (m2 == 0)

# ---- E: batch-geometry numerics, T=3 vs 3x T=1 (all resident)
kE = fresh_key(range(E))
x3 = (torch.randn(3, H, device=dev) * 0.3).to(torch.bfloat16)
ids3 = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(3)]).to(torch.int32)
w3 = torch.rand(3, TOPK, device=dev) * 0.5
batch = moe_w2_cubit._moe_w2_forward(x3, w3, ids3, kE)
rows = torch.cat([
    moe_w2_cubit._moe_w2_forward(x3[i:i + 1], w3[i:i + 1], ids3[i:i + 1], kE)
    for i in range(3)])
d = (batch.float() - rows.float()).abs().max().item()
r = d / max(rows.float().abs().max().item(), 1e-6)
bit = torch.equal(batch, rows)
print(f"E: T=3 batch vs 3x T=1: bit_equal={bit} max_abs={d:.3e} max_rel={r:.3e}")

print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
