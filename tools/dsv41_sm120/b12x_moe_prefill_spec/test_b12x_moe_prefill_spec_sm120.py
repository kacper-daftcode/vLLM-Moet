#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU test (one sm_120 GPU, serving image) of the prefill MoE spec B against b12x's served `dynamic` path, bit for bit.

  python3 test_b12x_moe_prefill_spec_sm120.py [--dumps DIR] [--synthetic-only] [--rows N]

1. Synthetic: the decode test's synthetic experts (384 x 5120 x 576 MXFP4, UE8M0 scales) and random bf16 tokens for
   T in (16, 64, 300, 1024) with random top-6 routings (plus -1 padding pairs): b12x's planned API with
   B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1 (the served setting since step 27a) runs the dynamic path (T > 8); its route
   rows (workspace.route_output, token-major pair rows, spied from _impl._launch_dynamic) and its output must equal
   ours (`torch.equal` on the int16 views).
2. Dumps (--dumps, default /workspace/tmp/par31/L/win/dumps): the served model's MoE layer (weights_layer2.pt: the exact
   tensors handed to b12x prepare_weights on rank 0; call1_layer2.pt: hidden_states / topk_ids / topk_weights / output of
   a real 4096-row prefill chunk). b12x re-run here must reproduce the dumped output exactly (the served arithmetic
   recorded), and ours must equal both.
Prints per case: route rows equal / differing rows, output equal / differing elements (first few with values), and the
timings of b12x's launch sequence vs the spec kernels (the spec is not a fast kernel; the number is a sanity bound).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b12x_moe_fused"))

E, H, N, TOPK, LIMIT = 384, 5120, 576, 6, 10.0


def _served(torch, dev, raw):
    """b12x prepared experts + plan for up to `max_tokens` tokens (dynamic path)."""
    import b12x.moe.fused_moe as fm

    class S:
        pass

    s = S()
    s.fm = fm
    ones = torch.ones(E, dtype=torch.float32, device=dev)
    plan_w = fm.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
                             params_dtype=torch.bfloat16, num_experts=E, hidden_size=H, intermediate_size=N,
                             w13_layout="w31")
    s.prepared = fm.prepare_weights(plan=plan_w, w1_fp4=raw["w1_fp4"].to(dev), w1_blockscale=raw["w1_blockscale"].to(dev),
                                    w1_global_scale=raw.get("w1_global_scale", ones).to(dev),
                                    a1_gscale=raw.get("a1_gscale", ones).to(dev), w2_fp4=raw["w2_fp4"].to(dev),
                                    w2_blockscale=raw["w2_blockscale"].to(dev),
                                    w2_global_scale=raw.get("w2_global_scale", ones).to(dev),
                                    a2_gscale=raw.get("a2_gscale", ones).to(dev), params_dtype=torch.bfloat16)
    s.cap = 4096  # the served plan capacity (--max-num-batched-tokens); every case runs through it, padded with -1 rows,
    # so the CuTe compile cache of the served configuration is reused and the kernels are the served ones
    plan = fm.plan(fm.Caps(max_tokens=s.cap, num_topk=TOPK, device=s.prepared.w1_fp4.device,
                           weight_plan=s.prepared.plan, core_token_counts=(s.cap,), route_num_experts=0,
                           quant_mode="w4a8_mx", apply_router_weight_on_input=False, swiglu_limit=LIMIT,
                           swiglu_alpha=None, swiglu_beta=None, deterministic_output=True, frozen=True))
    (spec,) = plan.scratch_specs()
    s.plan, s.scratch = plan, torch.zeros(int(spec.shape[0]), dtype=torch.uint8, device=dev)

    def run(x, ids, w, out):
        T = int(x.shape[0])
        if T < s.cap:
            xp = torch.zeros(s.cap, H, dtype=x.dtype, device=dev); xp[:T] = x
            idp = torch.full((s.cap, TOPK), -1, dtype=torch.int32, device=dev); idp[:T] = ids
            wp = torch.zeros(s.cap, TOPK, dtype=torch.float32, device=dev); wp[:T] = w
            op = torch.empty(s.cap, H, dtype=out.dtype, device=dev)
        else:
            xp, idp, wp, op = x, ids, w, out
        b = fm.bind(s.plan, scratch=s.scratch, a=xp, experts=s.prepared, topk_weights=wp, topk_ids=idp, output=op,
                    input_scales_static=True, unit_scale_contract=False)
        fm.run(binding=b)
        if T < s.cap:
            out.copy_(op[:T])

    s.run = run
    return s


def _planes(torch, prepared, spied):
    """The n64-repacked planes and the per-expert scales our kernels read, from the prepared object / the spied launch."""
    g = lambda *names: next((getattr(prepared, n) for n in names if getattr(prepared, n, None) is not None), None)
    planes = dict(w13=g("w13_rp", "w13"), w13_sf=g("w13_sfb", "w13_scales", "w13_sf"), w2=g("w2_rp", "w2"),
                  w2_sf=g("w2_sfb", "w2_scales", "w2_sf"))
    wv = spied.get("weights")
    for k, attr in (("w13", "w13"), ("w13_sf", "w13_sf"), ("w2", "down"), ("w2_sf", "down_sf")):
        if planes[k] is None and wv is not None:
            planes[k] = getattr(wv, attr, None)
    dev = planes["w13"].device
    ones = torch.ones(E, dtype=torch.float32, device=dev)
    planes["alpha"] = g("alpha1", "alpha", "g1_alphas", "w1_alpha") if g("alpha1", "alpha", "g1_alphas", "w1_alpha") is not None else ones
    planes["input_global_scale"] = g("input_scale", "input_global_scale", "a1_gscale") if g("input_scale", "input_global_scale", "a1_gscale") is not None else ones[:1]
    planes["down_alpha"] = g("alpha2", "down_alpha", "g2_alphas", "w2_alpha") if g("alpha2", "down_alpha", "g2_alphas", "w2_alpha") is not None else ones
    planes["global_scale"] = g("down_scale", "global_scale", "a2_gscale") if g("down_scale", "global_scale", "a2_gscale") is not None else ones[:1]
    for k in ("alpha", "input_global_scale", "down_alpha", "global_scale"):
        planes[k] = planes[k].reshape(-1).to(torch.float32)
    return planes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="/workspace/tmp/par31/L/win/dumps")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--rows", type=int, default=0, help="limit the dump case to the first N rows (0 = all 4096)")
    a = ap.parse_args()
    os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
    import torch
    from b12x_moe_prefill_spec_sm120 import moe_prefill_spec
    import b12x.moe.fused_moe._impl as impl

    dev = torch.device("cuda")
    torch.manual_seed(0)
    spied = {}
    orig_dyn = impl._launch_dynamic

    def spy_dyn(**kw):
        spied.clear()
        spied.update(kw)
        return orig_dyn(**kw)

    impl._launch_dynamic = spy_dyn

    def compare(tag, x, ids, w, served, out_ref=None):
        T = int(x.shape[0])
        out_b = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        served.run(x, ids, w, out_b)
        torch.cuda.synchronize()
        tb = (time.perf_counter() - t0) * 1e3
        route_b = spied["workspace"].route_output.reshape(-1)[: T * TOPK * H].view(T * TOPK, H).clone() if spied else None
        planes = _planes(torch, served.prepared, spied)
        t0 = time.perf_counter()
        out_m, route_m, a_values, a_scales, inter, inter_sf = moe_prefill_spec(x, ids, w, planes, LIMIT)
        torch.cuda.synchronize()
        tm = (time.perf_counter() - t0) * 1e3
        flat = ids.reshape(-1)
        valid = torch.nonzero(flat >= 0).flatten()
        msg = [f"{tag}: T={T}"]
        if route_b is not None:
            same_rows = (route_m[valid].view(torch.int16) == route_b[valid].view(torch.int16)).all(dim=1)
            msg.append(f"route rows {int(same_rows.sum())}/{valid.numel()} identical")
            if not bool(same_rows.all()):
                bad = valid[~same_rows][:3].tolist()
                for p in bad:
                    d = (route_m[p].float() - route_b[p].float()).abs()
                    msg.append(f"  pair {p} (tok {p // TOPK} slot {p % TOPK} expert {int(flat[p])}): {int((d > 0).sum())} cols differ, max {d.max().item():.3e}")
        ok = torch.equal(out_m.view(torch.int16), out_b.view(torch.int16))
        msg.append(f"output vs b12x {'IDENTICAL' if ok else 'DIFFER'}")
        if not ok:
            d = (out_m.float() - out_b.float()).abs()
            msg.append(f"  {int((d > 0).sum())}/{d.numel()} elements, max {d.max().item():.3e}; first rows {torch.nonzero(d.sum(1) > 0).flatten()[:5].tolist()}")
        if out_ref is not None:
            okr = torch.equal(out_b.view(torch.int16), out_ref.view(torch.int16))
            okm = torch.equal(out_m.view(torch.int16), out_ref.view(torch.int16))
            msg.append(f"served dump: b12x re-run {'IDENTICAL' if okr else 'DIFFER'}, ours {'IDENTICAL' if okm else 'DIFFER'}")
        msg.append(f"time b12x {tb:.1f} ms, spec {tm:.1f} ms")
        print(" | ".join(msg))
        sys.stdout.flush()
        return ok

    results = []
    # ---- synthetic
    from test_b12x_moe_fused_sm120 import _weights
    w1, w1s, w2, w2s = _weights(torch, dev)
    raw = dict(w1_fp4=w1, w1_blockscale=w1s, w2_fp4=w2, w2_blockscale=w2s)
    served = _served(torch, dev, raw)
    g = torch.Generator(device=dev).manual_seed(1)
    for T in (16, 64, 300, 1024):
        x = (torch.randn(T, H, device=dev, dtype=torch.float32, generator=g) * 0.5).to(torch.bfloat16)
        ids = torch.randint(0, E, (T, TOPK), device=dev, dtype=torch.int32, generator=g)
        if T >= 64:
            ids[5, 1] = -1
            ids[7] = -1
        w = (torch.rand(T, TOPK, device=dev, generator=g) * 0.4 + 0.05).to(torch.float32)
        results.append(compare("synthetic", x, ids, w, served))
    if a.synthetic_only:
        print(f"{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} cases identical")
        return 0 if all(results) else 1
    # ---- the served model's dumps
    d = Path(a.dumps)
    raw = torch.load(d / "weights_layer2.pt", map_location="cpu", weights_only=False)
    served = _served(torch, dev, raw)
    for name in sorted(p.name for p in d.glob("call*_layer2.pt")):
        c = torch.load(d / name, map_location="cpu", weights_only=False)
        x, ids, w, out_ref = c["hidden_states"].to(dev), c["topk_ids"].to(dev), c["topk_weights"].to(dev), c["output"].to(dev)
        if a.rows:
            x, ids, w, out_ref = x[: a.rows], ids[: a.rows], w[: a.rows], out_ref[: a.rows]
        results.append(compare(f"dump {name}", x, ids, w, served, out_ref))
    print(f"{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} cases identical")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
