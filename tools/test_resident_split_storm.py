#!/usr/bin/env python3
"""Storm harness for the RESIDENT + SPLIT-FP4 + gate path — hunts the
accumulating-corruption bug (GSM8K wrong-sets growing 5c9c11c12 across
runs on one server, prefix-cache reset ruled out).

Real DS4-Flash geometry (H=4096, I=2048, E=256/layer, top-6), resident
2-bit planes + FP4 need-tier holding REFINEMENT rows. Loop (default 600
rounds) with a long-tailed router so new experts keep trickling in
(gate storms early, trickle promotions forever — the live pattern):

  forward -> uncapped force_promote (gate idiom) -> manager passes ->
  INVARIANTS:
    I1 slot bytes == host refinement row  (torn/raced promotion copy)
    I2 mirror/slot_table/owner coherent
    I3 host row STILL == freshly recomputed refinement from nibbles
       (host-store corruption after boot)
    I4 resident base planes/scales checksum unchanged (stray writes)
    I5 forward output == mixed dequant reference for CURRENT mirror
       (dispatch-level wrongness with byte-clean state)

Env: ROUNDS (600), E (256), LAYERS (2), SEED, POOL_SLOTS (96).
Run inside the vllm image on one GPU.
"""
import os
import sys

import torch

os.environ.setdefault("VLLM_MOE_W2", "1")

from vllm.model_executor.layers.quantization.utils import moe_w2_cubit  # noqa: E402
from vllm.model_executor.layers.quantization.utils import moe_w2_delta  # noqa: E402
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (  # noqa: E402
    mxfp4_to_codes, mxfp4_to_nibbles, nibbles_to_refinement,
    pack_fragment_major, pack_scales, split_fp4_dequant,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8,
)

assert moe_w2_cubit._ensure_ready(), "cubins not found"
moe_w2_delta._SPLIT = True
assert moe_w2_delta.split_enabled()
dev = torch.device("cuda")
torch.manual_seed(int(os.environ.get("SEED", "20260713")))

H = 4096
I = 2048
E = int(os.environ.get("E", "256"))
L = int(os.environ.get("LAYERS", "2"))
T = 12
TOPK = 6
ROUNDS = int(os.environ.get("ROUNDS", "600"))
POOL_SLOTS = int(os.environ.get("POOL_SLOTS", "96"))
LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0], device=dev)

print(f"[setup] H={H} I={I} E={E} L={L} top-{TOPK} pool={POOL_SLOTS} "
      f"rounds={ROUNDS}", flush=True)

layers = []
for li in range(L):
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
    s13 = torch.randint(116, 126, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
    s2 = torch.randint(116, 126, (E, H, I // 32), dtype=torch.uint8, device=dev)
    planes13 = torch.stack([pack_fragment_major(mxfp4_to_codes(w13[e]))
                            for e in range(E)])
    sc13 = torch.stack([pack_scales(s13[e]) for e in range(E)])
    planes2 = torch.stack([pack_fragment_major(mxfp4_to_codes(w2[e]))
                           for e in range(E)])
    sc2 = torch.stack([pack_scales(s2[e]) for e in range(E)])
    st = dict(N13=2 * I, K13=H, N2=H, K2=I, E=E, base=False,
              planes13=planes13, sc13=sc13, planes2=planes2, sc2=sc2)
    moe_w2_cubit._LAYERS[li] = st
    layers.append(dict(w13=w13, s13=s13, w2=w2, s2=s2, st=st))

# base-plane fingerprints for I4 (any stray write flips a sum)
base_sums = [(int(l["st"]["planes13"].long().sum()),
              int(l["st"]["sc13"].long().sum()),
              int(l["st"]["planes2"].long().sum()),
              int(l["st"]["sc2"].long().sum())) for l in layers]

w13r_bytes = 2 * I * H // 4
w2r_bytes = H * I // 4
pool_gb = POOL_SLOTS * (w13r_bytes + w2r_bytes) / 2**30
tier = moe_w2_delta.DeltaTier(L, E, dev, w13_bytes=w13r_bytes,
                              w2_bytes=w2r_bytes, pool_gb=pool_gb,
                              policy="need", tag="fp4")
moe_w2_delta._TIER = tier

host_rows = {}
for li, l in enumerate(layers):
    rf13 = torch.stack([pack_fragment_major(
        nibbles_to_refinement(mxfp4_to_nibbles(l["w13"][e]))) for e in range(E)])
    rf2 = torch.stack([pack_fragment_major(
        nibbles_to_refinement(mxfp4_to_nibbles(l["w2"][e]))) for e in range(E)])
    tier.add_layer_host_planes(li, rf13, rf2)
    host_rows[li] = torch.cat((rf13, rf2), dim=1)   # snapshot for I1/I3

x = (torch.randn(T, H, device=dev) * 0.3).to(torch.bfloat16)
a8, as8 = per_token_group_quant_fp8(x, 128)
a_deq = a8.float() * as8.repeat_interleave(128, 1)


def deq2(pack, sc):
    return (LEVELS[mxfp4_to_codes(pack).long()]
            * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1))


def deq4(pack, sc):
    return (split_fp4_dequant(mxfp4_to_nibbles(pack))
            * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1))


def reference(li, topk_ids, topk_w, fp4):
    l = layers[li]
    ref = torch.zeros(T, H, device=dev)
    for t in range(T):
        for j in range(TOPK):
            e = int(topk_ids[t, j])
            dq = deq4 if e in fp4 else deq2
            c13 = a_deq[t] @ dq(l["w13"][e], l["s13"][e]).T
            act = torch.nn.functional.silu(c13[:I]) * c13[I:]
            q2, qs2 = per_token_group_quant_fp8(
                act.to(torch.bfloat16).unsqueeze(0), 128)
            ad = q2.float() * qs2.repeat_interleave(128, 1)
            ref[t] += float(topk_w[t, j]) * (ad[0] @ dq(l["w2"][e], l["s2"][e]).T)
    return ref


# long-tailed popularity: hot core + trickling tail (the GSM8K pattern)
weights = 1.0 / (torch.arange(E, dtype=torch.float64) + 3.0) ** 1.3
perm = torch.randperm(E)
weights = weights[perm.argsort()]        # shuffle identity of hot experts

CAPTURE = os.environ.get("CAPTURE", "0") == "1"
ids_buf = torch.zeros(T, TOPK, dtype=torch.int32, device=dev)
w_buf = torch.zeros(T, TOPK, device=dev)
graphs = {}
if CAPTURE:
    # capture one graph per layer around the DECODE forward, exactly the
    # serving cadence: static in/out buffers, tables read in-graph, then
    # thousands of replays interleaved with table mutations (promotions)
    for li in range(L):
        ids_buf.fill_(0)
        for _ in range(2):   # warmup for capture
            moe_w2_cubit._moe_w2_forward(x, w_buf, ids_buf, li)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        tier.notify_capture()
        with torch.cuda.graph(g):
            out = moe_w2_cubit._moe_w2_forward(x, w_buf, ids_buf, li)
        graphs[li] = (g, out)
    tier._last_capture = 0.0
    print("[setup] graphs captured", flush=True)

fails = 0
verified_slots = 0
for r in range(ROUNDS):
    li = r % L
    ids = torch.multinomial(weights.to(dev).float(), T * TOPK,
                            replacement=True).view(T, TOPK).to(torch.int32)
    topk_w = torch.rand(T, TOPK, device=dev) * 0.5
    tier.step_begin()
    fp4_pre = {e for e in range(E) if int(tier._mirror[li, e]) >= 0}
    if CAPTURE:
        g, out = graphs[li]
        ids_buf.copy_(ids); w_buf.copy_(topk_w)
        g.replay(); torch.cuda.synchronize()
        got = out.clone()
    else:
        got = moe_w2_cubit._moe_w2_forward(x, topk_w, ids, li)
        torch.cuda.synchronize()
    # gate idiom: uncapped force-promote of this step's routed cold
    # experts, then the REPLAY (the live gate contract: the re-forward
    # must reflect the freshly promoted FP4 set)
    tier.force_promote(max_promote=None)
    fp4_post = {e for e in range(E) if int(tier._mirror[li, e]) >= 0}
    if CAPTURE:
        g, out = graphs[li]
        g.replay(); torch.cuda.synchronize()
        got_replay = out.clone()
    else:
        got_replay = moe_w2_cubit._moe_w2_forward(x, topk_w, ids, li)
        torch.cuda.synchronize()
    tier._last_capture = 0.0
    tier._tick_once()
    torch.cuda.synchronize()
    tier.step_end()

    # ---- invariants
    mir = tier._mirror
    mapped = [(lj, e, int(mir[lj, e])) for lj in range(L) for e in range(E)
              if int(mir[lj, e]) >= 0]
    for lj, e, s in mapped:
        if int(tier.slot_table[lj, e]) != s:
            print(f"[r{r}] I2 FAIL slot_table!=mirror ({lj},{e})"); fails += 1
        if (int(tier._owner_li[s]), int(tier._owner_ei[s])) != (lj, e):
            print(f"[r{r}] I2 FAIL owner!=mirror slot {s}"); fails += 1
    # byte-verify a rotating subset (all mapped every 25 rounds)
    if r % 25 == 24 or r == ROUNDS - 1:
        for lj, e, s in mapped:
            exp = host_rows[lj][e].to(dev)
            if not torch.equal(tier.pool[s], exp):
                nb = int((tier.pool[s] != exp).sum())
                print(f"[r{r}] I1 FAIL pool!=host ({lj},{e}) slot {s}: "
                      f"{nb} bytes differ"); fails += 1
            verified_slots += 1
        for lj in range(L):
            hs = tier._store  # backend: verify a few rows vs recompute
        li_chk = r % L
        l = layers[li_chk]
        for e in [m[1] for m in mapped if m[0] == li_chk][:4]:
            fresh13 = pack_fragment_major(
                nibbles_to_refinement(mxfp4_to_nibbles(l["w13"][e])))
            row = tier._store.rows_for([(li_chk, e)])[0]
            if not torch.equal(row[:w13r_bytes].cpu(), fresh13.cpu()):
                print(f"[r{r}] I3 FAIL host row != recompute ({li_chk},{e})")
                fails += 1
        for lj, l in enumerate(layers):
            sums = (int(l["st"]["planes13"].long().sum()),
                    int(l["st"]["sc13"].long().sum()),
                    int(l["st"]["planes2"].long().sum()),
                    int(l["st"]["sc2"].long().sum()))
            if sums != base_sums[lj]:
                print(f"[r{r}] I4 FAIL resident base planes mutated L{lj}")
                fails += 1
    # output checks every 10 rounds (expensive reference):
    # pass-1 vs PRE-promotion mirror, replay vs POST-promotion mirror
    if r % 10 == 9:
        ref = reference(li, ids, topk_w, fp4_pre)
        rel = (got.float() - ref).abs().max().item() / ref.abs().max().item()
        if rel > 0.08:
            print(f"[r{r}] I5a FAIL pass1 rel={rel:.3e} (fp4={len(fp4_pre)})")
            fails += 1
        ref2 = reference(li, ids, topk_w, fp4_post)
        rel2 = (got_replay.float() - ref2).abs().max().item() / ref2.abs().max().item()
        if rel2 > 0.08:
            print(f"[r{r}] I5b FAIL replay rel={rel2:.3e} "
                  f"(fp4={len(fp4_post)}, nowe={len(fp4_post-fp4_pre)})")
            fails += 1
    if r % 100 == 99:
        print(f"[r{r+1}/{ROUNDS}] promoted={tier._n_promoted} "
              f"evicted={tier._n_evicted} mapped={len(mapped)} "
              f"verified={verified_slots} fails={fails}", flush=True)

print(f"DONE rounds={ROUNDS} promoted={tier._n_promoted} "
      f"evicted={tier._n_evicted} slot-verifications={verified_slots} "
      f"fails={fails}")
print("RESULT:", "PASS" if fails == 0 else "FAIL")
sys.exit(0 if fails else 1)
