#!/usr/bin/env python3
"""b12x W4A8 MoE (MXFP4 x MXFP8) through vLLM's tensor-based call path at DeepSeek-V4.1-Flash TP4 shapes, one GPU.

E=384 experts, hidden 5120, intermediate 576 per rank (2304 / 4; 576 % 128 == 64 selects the compact N64 kernels),
top-6, SiLU with swiglu_limit 10, random MXFP4 weights. The plan / bind / run sequence is
vllm/model_executor/layers/fused_moe/b12x.py's.
  1. the bind as installed: on a stock FlashInfer 0.7.2 it stops with "compact W4A8 execution requires its prepared
     native launchers"; with patch_flashinfer_b12x_compact_bind.py it runs
  2. the compact launchers attached (the patcher's code, monkey-patched unless --no-monkeypatch): output vs a
     dequantized float32 reference (expect a relative L2 error of ~0.05 from the MXFP8 activations, cosine > 0.998)
  3. CUDA-graph replay time per call at decode token counts, rotating over --layers weight sets (L2 stays cold);
     --sweep instead races decode configs at M=6 with random and with correlated (24-expert pool) routing
Measured 2026-10-20 on one RTX PRO 6000 (GPU 4 of rtx): rel L2 0.047 / cosine 0.9989; M=6 110 us with random
routing, 73-83 us with the 24-expert pool (the in-model decode graph: 77 us per layer).

usage (in the serving image, B12X_*_CACHE_DIR pointing at a writable cache):
  python3 test_b12x_moe_compact_bind.py [--no-monkeypatch] [--tokens 1,6,64] [--sweep]
"""
import argparse
import dataclasses
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--experts", type=int, default=384)
ap.add_argument("--hidden", type=int, default=5120)
ap.add_argument("--inter", type=int, default=576)
ap.add_argument("--topk", type=int, default=6)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--tokens", default="1,2,4,6,8,12,16,24,32,48,64")
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--replays", type=int, default=50)
ap.add_argument("--w13-layout", default="w31")
ap.add_argument("--no-monkeypatch", action="store_true")
ap.add_argument("--sweep", action="store_true", help="M=6: decode-config sweep instead of the token sweep")
args = ap.parse_args()

dev = torch.device("cuda", 0)
torch.manual_seed(0)
from b12x.moe import fused_moe  # noqa: E402
from b12x.moe.fused_moe import _impl, _preparation  # noqa: E402

E, K, N, TOPK = args.experts, args.hidden, args.inter, args.topk
FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=dev)


def make_layer():
    w1 = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device=dev)
    # E8M0 scales around 2^-7 .. 2^-5 keep the activations in range.
    s1 = torch.randint(120, 123, (E, 2 * N, K // 32), dtype=torch.uint8, device=dev)
    s2 = torch.randint(120, 123, (E, K, N // 32), dtype=torch.uint8, device=dev)
    return w1, w2, s1, s2


def canonicalize(packed):  # vLLM's _canonicalize_fp4_zero_signs_
    magnitude = packed & 0x77
    nonzero = (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11
    packed.bitwise_and_(0x77 | (nonzero << 3))


def dequant(w, s):  # [.., R, C/2] uint8 + [.., R, C/32] e8m0 -> float32 [.., R, C]
    lo, hi = FP4[(w & 0xF).long()], FP4[(w >> 4).long()]
    v = torch.stack((lo, hi), dim=-1).flatten(-2)
    scale = torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=-1)
    return v * scale


def prepare(w1, w2, s1, s2):
    canonicalize(w1)
    canonicalize(w2)
    plan = fused_moe.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
                                  params_dtype=torch.bfloat16, num_experts=E, hidden_size=K, intermediate_size=N,
                                  w13_layout=args.w13_layout)
    one = torch.ones(E, dtype=torch.float32, device=dev)
    return fused_moe.prepare_weights(plan=plan, w1_fp4=w1, w1_blockscale=s1, w1_global_scale=one, a1_gscale=one,
                                     w2_fp4=w2, w2_blockscale=s2, w2_global_scale=one, a2_gscale=one,
                                     params_dtype=torch.bfloat16)


def make_caps(prepared, tokens, decode_config=None):
    return fused_moe.Caps(max_tokens=tokens, num_topk=TOPK, device=dev, weight_plan=prepared.plan,
                          core_token_counts=(tokens,), route_num_experts=0, quant_mode="w4a8_mx",
                          apply_router_weight_on_input=False, swiglu_limit=10.0, swiglu_alpha=None,
                          swiglu_beta=None, frozen=True, decode_config=decode_config)


def make_plan(prepared, tokens, decode_config=None):
    return fused_moe.plan(make_caps(prepared, tokens, decode_config))


def run(plan, prepared, a, ids, wts, out):
    scratch = run.scratch.get(id(plan))
    if scratch is None:
        spec = plan.scratch_specs()[0]
        scratch = run.scratch[id(plan)] = torch.empty(spec.shape, dtype=spec.dtype, device=dev)
    b = fused_moe.bind(plan, scratch=scratch, a=a, experts=prepared, topk_weights=wts, topk_ids=ids, output=out,
                       input_scales_static=True, unit_scale_contract=False)
    fused_moe.run(binding=b)


run.scratch = {}


def reference(a, ids, wts, layer, layout):
    w1, w2, s1, s2 = layer
    out = torch.zeros(a.shape[0], K, dtype=torch.float32, device=dev)
    for t in range(a.shape[0]):
        for j in range(TOPK):
            e = int(ids[t, j])
            w13 = dequant(w1[e], s1[e])
            first, second = w13[:N], w13[N:]
            gate, up = (second, first) if layout == "w31" else (first, second)
            x = a[t].float()
            g = (gate @ x).clamp(max=10.0)
            u = (up @ x).clamp(-10.0, 10.0)
            h = torch.nn.functional.silu(g) * u
            out[t] += wts[t, j] * (dequant(w2[e], s2[e]) @ h)
    return out


_orig_bind = _impl.TPMoEScratchPlan.bind
_compact = {}


def bind_with_compact(self, **kw):
    b = _orig_bind(self, **kw)
    hit = _compact.get(id(self))
    if hit is None or hit[0] is not self:
        hit = _compact[id(self)] = (self, _preparation._compact_launches(self.launch_plan, self.caps))
    return dataclasses.replace(b, compact_launches=hit[1]) if hit[1] is not None else b


layers = [make_layer() for _ in range(args.layers)]
ref_layer = tuple(t.clone() for t in layers[0])
prepared = [prepare(*l) for l in layers]
print(f"prepared {len(prepared)} layers: E={E} K={K} N={N} top-{TOPK} layout={args.w13_layout}, "
      f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

M = 6
a = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
wts = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1).float()
out = torch.empty(M, K, dtype=torch.bfloat16, device=dev)
p6 = make_plan(prepared[0], M)
print("launch plan:", p6.launch_plan.implementation, p6.launch_plan.quant_mode, "n", p6.launch_plan.n, flush=True)
try:
    run(p6, prepared[0], a, ids, wts, out)
    torch.cuda.synchronize()
    print("1. unpatched bind: ran without error (the compact path was not selected)")
except RuntimeError as exc:
    print("1. unpatched bind:", exc)

if not args.no_monkeypatch:
    _impl.TPMoEScratchPlan.bind = bind_with_compact
run(p6, prepared[0], a, ids, wts, out)
torch.cuda.synchronize()
for layout in ("w31", "w13"):
    ref = reference(a, ids, wts, ref_layer, layout)
    err = (out.float() - ref).norm() / ref.norm()
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0)
    print(f"2. patched bind vs dequantized reference ({layout}): rel L2 err {err:.4f}, cosine {cos:.5f}")

def bench(M, decode_config=None, pool=None):
    a = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
    calls = []
    for li, pr in enumerate(prepared):
        if pool:  # correlated routing: every token picks its top-k from one pool of experts
            p = torch.randperm(E, device=dev)[:pool]
            ids = torch.stack([p[torch.randperm(pool, device=dev)[:TOPK]] for _ in range(M)]).to(torch.int32)
        else:
            ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        wts = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1).float()
        calls.append((make_plan(pr, M, decode_config), pr, ids, wts, torch.empty(M, K, dtype=torch.bfloat16, device=dev)))
    for c in calls:
        run(c[0], c[1], a, c[2], c[3], c[4])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for c in calls:
                run(c[0], c[1], a, c[2], c[3], c[4])
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.replays):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / args.replays / len(calls) * 1e6, calls[0][0].launch_plan.implementation


if args.sweep:
    base = make_caps(prepared[0], 6).decode_config
    print("default decode config at M=6:", base, flush=True)
    for pool in (None, 24):
        for mac in (None, 16, 32, 47, 64, 94, 128, 141, 188):
            for backend in ("micro", "dynamic"):
                cfg = dataclasses.replace(base, backend=backend, max_active_clusters=mac)
                if backend == "dynamic":
                    cfg = dataclasses.replace(cfg, dynamic_tile_m=16, dynamic_route_mode="grouped")
                try:
                    us, impl = bench(6, cfg, pool)
                    print(f"   pool={pool} backend={backend:7s} max_active_clusters={mac}: {us:7.1f} us ({impl})", flush=True)
                except Exception as exc:
                    print(f"   pool={pool} backend={backend:7s} max_active_clusters={mac}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    raise SystemExit(0)

print("3. CUDA-graph replay, us per MoE call (mean over", args.layers, "rotating layers):", flush=True)
for M in [int(x) for x in args.tokens.split(",")]:
    a = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
    calls = []
    for li, pr in enumerate(prepared):
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        wts = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1).float()
        calls.append((make_plan(pr, M), pr, ids, wts, torch.empty(M, K, dtype=torch.bfloat16, device=dev)))
    for c in calls:  # compile + first launch outside the graph
        run(c[0], c[1], a, c[2], c[3], c[4])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for c in calls:
                run(c[0], c[1], a, c[2], c[3], c[4])
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.replays):
        g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / args.replays / len(calls) * 1e6
    print(f"   M={M:4d}: {us:7.1f} us  impl={calls[0][0].launch_plan.implementation}", flush=True)
