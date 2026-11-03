"""GPU test (one sm_120 GPU, serving image) of b12x_moe_mega_sm120: the routing phase against bf16_gemm's spread GEMM
(plain launch, the same slice width) and dsv4_topk_sm120 (logits, weights, ids bit for bit), the MoE phases against the
served b12x path on the routing the kernel itself produced (projections, intermediate, route_output bit for bit),
50 replays of one case, 500 back-to-back launches from a CUDA graph (a race in the work distribution shows only after
many launches), and the kernel with routing off against b12x_moe_fused's inputs.

  python3 test_b12x_moe_mega_sm120.py [--iters 20]

Synthetic MXFP4 experts of one TP4 rank of DeepSeek-V4.1-Flash (384 experts, hidden 5120, intermediate 576, top-6,
SwiGLU limit 10) as in test_b12x_moe_fused_sm120.py; router inputs x [8][5120] bf16 and weights [384][5120] bf16 drawn
so that the routings have 20-40 distinct experts, several seeds and scales (incl. wide logits), padding tokens, 1..8
tokens, slice widths 192 / 320 / 640 (12 / 20 / 40 steps). Prints cold-L2 times of spread + top-k + fused MoE vs the
megakernel at the served shape.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

E, H, N, TOPK, LIMIT, TOKENS = 384, 5120, 576, 6, 10.0, 8
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "b12x_moe_fused"))
sys.path.insert(0, os.path.join(HERE, "..", "bf16_gemm"))
sys.path.insert(0, os.path.join(HERE, "..", "dsv4_topk"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    os.environ["VLLM_MOET_B12X_MOE_FUSED"] = "0"  # the served b12x path is the MoE reference
    import torch
    from torch.profiler import ProfilerActivity, profile

    import b12x.moe._shared.kernels.w4a8_compact_micro as cm
    import b12x_moe_fused_sm120 as mf
    import b12x_moe_mega_sm120 as mg
    import bf16_replica_gemm_sm120 as bf
    import dsv4_topk_sm120 as tk
    from test_b12x_moe_fused_sm120 import _weights

    class Served:  # test_b12x_moe_fused_sm120._Served planned for 8 tokens
        def __init__(self, weights):
            import b12x.moe.fused_moe as fm

            self.fm = fm
            w1, w1s, w2, w2s = weights
            ones = torch.ones(E, dtype=torch.float32, device=dev)
            plan_w = fm.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
                                     params_dtype=torch.bfloat16, num_experts=E, hidden_size=H, intermediate_size=N,
                                     w13_layout="w31")
            self.prepared = fm.prepare_weights(plan=plan_w, w1_fp4=w1, w1_blockscale=w1s, w1_global_scale=ones,
                                               a1_gscale=ones, w2_fp4=w2, w2_blockscale=w2s, w2_global_scale=ones,
                                               a2_gscale=ones, params_dtype=torch.bfloat16)
            self.plan = fm.plan(fm.Caps(max_tokens=TOKENS, num_topk=TOPK, device=self.prepared.w1_fp4.device,
                                        weight_plan=self.prepared.plan, core_token_counts=(TOKENS,), route_num_experts=0,
                                        quant_mode="w4a8_mx", apply_router_weight_on_input=False, swiglu_limit=LIMIT,
                                        swiglu_alpha=None, swiglu_beta=None, frozen=True))
            (spec,) = self.plan.scratch_specs()
            self.scratch = torch.zeros(int(spec.shape[0]), dtype=torch.uint8, device=dev)

        def run(self, x, ids, w, out):
            # always 8 rows: tokens past the case's are padding (ids -1, weight 0, zero input)
            m = x.shape[0]
            if m < TOKENS:
                x = torch.cat([x, torch.zeros(TOKENS - m, H, dtype=x.dtype, device=dev)])
                ids = torch.cat([ids, torch.full((TOKENS - m, TOPK), -1, dtype=ids.dtype, device=dev)])
                w = torch.cat([w, torch.zeros(TOKENS - m, TOPK, dtype=w.dtype, device=dev)])
            b = self.fm.bind(self.plan, scratch=self.scratch, a=x.contiguous(), experts=self.prepared, topk_weights=w.contiguous(),
                             topk_ids=ids.contiguous(), output=out, input_scales_static=True, unit_scale_contract=False)
            self.fm.run(binding=b)

    dev = torch.device("cuda")
    served = Served(_weights(torch, dev))
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    got = {}
    orig = cm.launch_w4a8_compact_micro

    def spy(**kw):
        got.update(kw)
        return orig(**kw)

    cm.launch_w4a8_compact_micro = spy
    bfe = bf._ext()
    tke = tk._ext()
    g = torch.Generator(device=dev).manual_seed(5)
    wr = (torch.randn(E, H, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    bias = (torch.randn(E, device=dev, generator=g) * 0.05).float().contiguous()

    def ref_routing(x, step, is_padding, scale):
        m = x.shape[0]
        logits = torch.empty(m, E, dtype=torch.float32, device=dev)
        slices = -(-H // step)
        ws = torch.empty(E // 8 * slices * 128 + 2 * 16 * E, dtype=torch.float32, device=dev)
        cnt = torch.zeros(E // 8 + 1, dtype=torch.int32, device=dev)
        bfe.spread_out(x, wr, logits, step, False, ws, cnt, 4, False)
        w = torch.empty(m, TOPK, dtype=torch.float32, device=dev)
        ids = torch.empty(m, TOPK, dtype=torch.int32, device=dev)
        tke.dsv4_topk_out(logits, bias, is_padding, w, ids, float(scale), None, None, 0, False)
        return logits, w, ids

    proj_mine = torch.empty(TOKENS * TOPK, 2 * N, dtype=torch.bfloat16, device=dev)
    route_mine = torch.empty(TOKENS * TOPK, H, dtype=torch.bfloat16, device=dev)
    inter_mine = None

    def regions():
        lay = cm._layout(int(got["max_tokens"]), TOPK, H, N)
        st = got["scratch"].view(torch.uint8).reshape(-1)
        r = {name: st.narrow(0, *lay[name]) for name in ("a_values", "a_scales", "projections", "intermediate", "route_output")}
        return r, int(got["max_tokens"]) * TOPK

    def same_moe_bits(r, ids, rows_cap, m):
        flat = ids.reshape(-1)
        rows = torch.nonzero((flat >= 0) & (flat < E)).flatten()
        proj_ref = r["projections"].view(torch.bfloat16).view(-1, 2 * N)
        wpr = 5 * 32
        iv = lambda t: t[: rows_cap * wpr * 4].view(torch.int32).view(rows_cap, wpr)
        isc = lambda t: t[rows_cap * wpr * 4: rows_cap * wpr * 4 + 5 * rows_cap * 4].view(torch.int32).view(5, rows_cap)
        okp = torch.equal(proj_mine[rows].view(torch.int16), proj_ref[rows].view(torch.int16))
        oki = torch.equal(iv(inter_mine)[rows], iv(r["intermediate"])[rows]) and torch.equal(isc(inter_mine)[:, rows], isc(r["intermediate"])[:, rows])
        route_ref = r["route_output"].view(torch.bfloat16).view(-1, H)[: m * TOPK]
        oko = torch.equal(route_mine[: m * TOPK].view(torch.int16), route_ref.view(torch.int16))
        return okp, oki, oko

    # cases: (tokens, seed, x scale, step, padding tail)
    cases = [(8, s, sc, st, 0) for s in (1, 2, 3) for sc in (1.0, 6.0) for st in (320,)]
    cases += [(8, 4, 1.0, 192, 0), (8, 5, 1.0, 640, 0), (8, 6, 30.0, 320, 0), (8, 7, 1.0, 320, 2), (8, 8, 1.0, 320, 5)]
    cases += [(m, 10 + m, 1.0, 320, 0) for m in (1, 2, 3, 5, 6, 7)]
    cases += [(6, 20, 1.0, 320, 1), (7, 21, 2.0, 192, 0), (8, 22, 0.3, 640, 0)]
    ok_route = ok_moe = 0
    times = []
    served_shape = None
    for i, (m, seed, sc, step, pad) in enumerate(cases):
        gi = torch.Generator(device=dev).manual_seed(100 + seed)
        x = (torch.randn(TOKENS, H, device=dev, generator=gi) * sc).to(torch.bfloat16)
        x = x[:m].contiguous()
        is_padding = None
        if pad:
            is_padding = torch.zeros(m, dtype=torch.bool, device=dev)
            is_padding[m - pad:] = True
        scale = 2.5
        logits_ref, w_ref, ids_ref = ref_routing(x, step, is_padding, scale)
        # the served b12x MoE on the reference routing -> the reference projections / intermediate / route_output
        out = torch.empty(TOKENS, H, dtype=torch.bfloat16, device=dev)
        served.run(x, ids_ref, w_ref, out)
        torch.cuda.synchronize()
        r, rows_cap = regions()
        if inter_mine is None:
            inter_mine = torch.zeros_like(r["intermediate"])
        # the megakernel: routing from x / wr itself, then the MoE on the same quantized tokens
        logits_mine = torch.full((m, E), float("nan"), dtype=torch.float32, device=dev)
        w_mine = torch.full((m, TOPK), float("nan"), dtype=torch.float32, device=dev)
        ids_mine = torch.full((m, TOPK), -7, dtype=torch.int32, device=dev)
        proj_mine.fill_(float("nan")); inter_mine.fill_(0xFF); route_mine.fill_(float("nan"))
        routing = dict(x=x, w=wr, logits=logits_mine, step=step, bias=bias, scale=scale, is_padding=is_padding)
        pdl = bool(i % 2)  # both launches: plain and programmatic stream serialization (L2 prefetch before the wait)
        mg.moe_mega(r["a_values"], r["a_scales"], got["w13"], got["w13_scales"], proj_mine, inter_mine, got["w2"],
                    got["w2_scales"], ids_mine.reshape(-1), w_mine.reshape(-1), got["alpha1"], got["input_scale"],
                    got["alpha2"], got["down_scale"], route_mine, m * TOPK, TOPK, rows_cap, H, N, LIMIT, routing=routing, pdl=pdl)
        torch.cuda.synchronize()
        okl = torch.equal(logits_mine.view(torch.int32), logits_ref.view(torch.int32))
        okw = torch.equal(w_mine.view(torch.int32), w_ref.view(torch.int32))
        oki_ = torch.equal(ids_mine, ids_ref)
        okp, oki, oko = same_moe_bits(r, ids_ref, rows_cap, m)
        ok_route += okl and okw and oki_
        ok_moe += okp and oki and oko
        uniq = len(set(ids_ref[ids_ref >= 0].tolist()))
        print(f"tokens {m} seed {seed:2d} x{sc:4} step {step} pad {pad} {'pdl ' if pdl else 'plain'}: distinct {uniq:2d}; logits {'ok' if okl else 'DIFFER'}, weights"
              f" {'ok' if okw else 'DIFFER'}, ids {'ok' if oki_ else 'DIFFER'}; projections {'ok' if okp else 'DIFFER'}, intermediate"
              f" {'ok' if oki else 'DIFFER'}, route_output {'ok' if oko else 'DIFFER'}")
        if m == TOKENS and step == 320 and sc == 1.0 and seed == 1:
            served_shape = (x, is_padding, scale, step, r, rows_cap, ids_ref, w_ref, logits_mine, w_mine, ids_mine)
    # replay stability on the served shape: 50 launches (the kernel resets its control block)
    x, is_padding, scale, step, r, rows_cap, ids_ref, w_ref, logits_mine, w_mine, ids_mine = served_shape
    logits_ref, _, _ = ref_routing(x, step, is_padding, scale)
    out = torch.empty(TOKENS, H, dtype=torch.bfloat16, device=dev)
    served.run(x, ids_ref, w_ref, out)
    torch.cuda.synchronize()
    r, rows_cap = regions()
    rep = 0
    routing = dict(x=x, w=wr, logits=logits_mine, step=step, bias=bias, scale=scale, is_padding=is_padding)

    def run_mega(pdl=False):
        mg.moe_mega(r["a_values"], r["a_scales"], got["w13"], got["w13_scales"], proj_mine, inter_mine, got["w2"],
                    got["w2_scales"], ids_mine.reshape(-1), w_mine.reshape(-1), got["alpha1"], got["input_scale"],
                    got["alpha2"], got["down_scale"], route_mine, TOKENS * TOPK, TOPK, rows_cap, H, N, LIMIT, routing=routing, pdl=pdl)

    def run_separate():
        lg, w_, i_ = ref_routing(x, step, is_padding, scale)
        mf.moe_fused(r["a_values"], r["a_scales"], got["w13"], got["w13_scales"], proj_mine, inter_mine, got["w2"],
                     got["w2_scales"], i_.reshape(-1), w_.reshape(-1), got["alpha1"], got["input_scale"], got["alpha2"],
                     got["down_scale"], route_mine, TOKENS * TOPK, TOPK, rows_cap, H, N, LIMIT)

    for it in range(50):
        route_mine.fill_(float("nan")); ids_mine.fill_(-7); w_mine.fill_(float("nan"))
        run_mega(pdl=bool(it % 2))
        torch.cuda.synchronize()
        rep += torch.equal(ids_mine, ids_ref) and torch.equal(w_mine.view(torch.int32), w_ref.view(torch.int32)) and same_moe_bits(r, ids_ref, rows_cap, TOKENS)[2]
    # stress: the served shape replayed from a CUDA graph many times back to back (a race in the work distribution shows
    # up as a wrong row, a hang or an illegal address only after many launches)
    gs = torch.cuda.CUDAGraph()
    ss = torch.cuda.Stream()
    with torch.cuda.stream(ss):
        run_mega(); torch.cuda.synchronize()
        with torch.cuda.graph(gs, stream=ss):
            for _ in range(20):
                run_mega()
    stress_ok = 0
    for it in range(25):
        route_mine.fill_(float("nan")); ids_mine.fill_(-7); w_mine.fill_(float("nan"))
        gs.replay()
        torch.cuda.synchronize()
        stress_ok += torch.equal(ids_mine, ids_ref) and torch.equal(w_mine.view(torch.int32), w_ref.view(torch.int32)) and same_moe_bits(r, ids_ref, rows_cap, TOKENS)[2]
    # routing off = b12x_moe_fused on given ids
    route_mine.fill_(float("nan"))
    mg.moe_mega(r["a_values"], r["a_scales"], got["w13"], got["w13_scales"], proj_mine, inter_mine, got["w2"],
                got["w2_scales"], ids_ref.reshape(-1), w_ref.reshape(-1), got["alpha1"], got["input_scale"],
                got["alpha2"], got["down_scale"], route_mine, TOKENS * TOPK, TOPK, rows_cap, H, N, LIMIT)
    torch.cuda.synchronize()
    off_ok = same_moe_bits(r, ids_ref, rows_cap, TOKENS)[2]
    # timing, cold L2
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(a.iters):
            flush.view(torch.int32).sum()
            run_separate()
            flush.view(torch.int32).sum()
            run_mega()
        torch.cuda.synchronize()
    per = {}
    for e in prof.events():
        if e.device_type.name == "CUDA":
            per.setdefault(e.name, []).append(e.device_time)

    def pick(subs):
        for k, v in per.items():
            if any(s in k for s in subs):
                return statistics.median(v)
        return float("nan")

    sp, tkk, fu, me = pick(("spread_kernel",)), pick(("dsv4_topk_kernel",)), pick(("moe_fused_kernel",)), pick(("moe_mega_kernel",))
    # in a CUDA graph behind a small predecessor kernel (as in the served graph: the mHC boundary kernel precedes the
    # sublayer): the separate launches (spread with PDL, as served) vs the megakernel plain vs with PDL
    pre = torch.zeros(8, 5120, dtype=torch.bfloat16, device=dev)

    def predecessor():
        pre.mul_(1.0001)  # one small elementwise kernel

    def sep_pdl(spread_pdl=True):
        lg = torch.empty(TOKENS, E, dtype=torch.float32, device=dev)
        slices = -(-H // step)
        ws = torch.empty(E // 8 * slices * 128 + 2 * 16 * E, dtype=torch.float32, device=dev)
        cnt = torch.zeros(E // 8 + 1, dtype=torch.int32, device=dev)
        bfe.spread_out(x, wr, lg, step, False, ws, cnt, 4, spread_pdl)
        w_ = torch.empty(TOKENS, TOPK, dtype=torch.float32, device=dev)
        i_ = torch.empty(TOKENS, TOPK, dtype=torch.int32, device=dev)
        tke.dsv4_topk_out(lg, bias, is_padding, w_, i_, float(scale), None, None, 0, False)
        mf.moe_fused(r["a_values"], r["a_scales"], got["w13"], got["w13_scales"], proj_mine, inter_mine, got["w2"],
                     got["w2_scales"], i_.reshape(-1), w_.reshape(-1), got["alpha1"], got["input_scale"], got["alpha2"],
                     got["down_scale"], route_mine, TOKENS * TOPK, TOPK, rows_cap, H, N, LIMIT)

    def graph_time(fn, reps=50):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                predecessor(); fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                for _ in range(10):
                    predecessor(); fn()
        g.replay(); torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(reps):
            g.replay()
        t1.record(); torch.cuda.synchronize()
        return t0.elapsed_time(t1) * 1000 / (reps * 10)

    # the arms interleaved over several rounds, the median per arm: one graph timing of ~0.1 s drifts by several us
    # (the third arm measured was the slowest in two runs, whichever it was)
    arms = {"sep_pdl": sep_pdl, "sep_plain": lambda: sep_pdl(False), "mega_plain": lambda: run_mega(pdl=False),
            "mega_pdl": lambda: run_mega(pdl=True)}
    samples = {k: [] for k in arms}
    for _ in range(5):
        for k, fn in arms.items():
            samples[k].append(graph_time(fn, reps=20))
    g_sep, g_sep_plain, g_plain, g_pdl = (statistics.median(samples[k]) for k in ("sep_pdl", "sep_plain", "mega_plain", "mega_pdl"))
    print(f"served shape (8 tokens, step 320), cold L2: spread {sp:6.2f} + top-k {tkk:5.2f} + fused MoE {fu:6.2f} = {sp + tkk + fu:6.2f} us,"
          f" megakernel {me:6.2f} us")
    print(f"in a CUDA graph behind a small kernel (warm L2, per sublayer; median of 5 interleaved rounds): separate launches (spread PDL)"
          f" {g_sep:6.2f} us, separate (spread plain) {g_sep_plain:6.2f} us, megakernel plain {g_plain:6.2f} us, megakernel PDL {g_pdl:6.2f} us")
    print("   rounds: " + "; ".join(f"{k} " + "/".join(f"{v:.1f}" for v in samples[k]) for k in arms))
    cm.launch_w4a8_compact_micro = orig
    print(f"routing: {ok_route}/{len(cases)} identical (logits, weights, ids); MoE: {ok_moe}/{len(cases)} identical (projections,"
          f" intermediate, route_output); 50 replays: {rep}/50; stress 25 x 20 graph launches: {stress_ok}/25; routing off: {'ok' if off_ok else 'DIFFER'}")
    passed = ok_route == len(cases) and ok_moe == len(cases) and rep == 50 and stress_ok == 25 and off_ok
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
