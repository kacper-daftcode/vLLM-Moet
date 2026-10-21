"""GPU test (one sm_120 GPU, serving image) of b12x_moe_fused_sm120 and patch_flashinfer_b12x_moe_fused_sm120.py.

  python3 test_b12x_moe_fused_sm120.py [--iters 20]

Synthetic MXFP4 experts of one TP4 rank of DeepSeek-V4.1-Flash (384 experts, hidden 5120, intermediate 576, top-6,
SwiGLU limit 10), prepared and run through b12x's planned API as vLLM's B12xExperts does; 6 tokens (a C1 decode step
with five drafts), routings with 6, 12, 22 and 36 distinct experts (three seeds, two input scales), padding tokens,
and - when real_c1_ids.pt is next to this file - 160 routings recorded from the served model.
1. Kernel: with VLLM_MOET_B12X_MOE_FUSED=0 the served run leaves b12x's projections, intermediate and route_output in
   the scratch; the fused kernel on the same quantized tokens must give all three with the same bits, 50 replays of
   one routing included (the control block is reset by the kernel itself). Prints cold-L2 times (L2 refilled by a
   256 MB read first) of the fused kernel and of b12x's projection + activation + FC2.
2. Served path: a child process with VLLM_MOET_B12X_MOE_FUSED=1 runs the patched b12x MoE on the same inputs, eagerly
   and from a CUDA graph replayed 20 times; its MoE outputs must equal the parent's bit for bit, and its launches must
   hold the fused kernel and none of b12x's projection / activation / phase 2 or the grouped FC2.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import tempfile

E, H, N, TOPK, LIMIT, TOKENS = 384, 5120, 576, 6, 10.0, 6
CASES = [(u, seed, sc) for u in (6, 12, 22, 36) for seed in (1, 2, 3) for sc in ((0.5,) if seed != 3 else (4.0,))]
CASES += [(22, -1, 0.5), (12, -2, 0.5), (36, -5, 0.5)]  # seed < 0: the last -seed tokens are padding


def _weights(torch, dev):
    g = torch.Generator(device=dev).manual_seed(0)
    w1 = torch.randint(0, 256, (E, 2 * N, H // 2), dtype=torch.uint8, device=dev, generator=g)
    w2 = torch.randint(0, 256, (E, H, N // 2), dtype=torch.uint8, device=dev, generator=g)
    w1s = torch.randint(119, 123, (E, 2 * N, H // 32), dtype=torch.uint8, device=dev, generator=g)
    w2s = torch.randint(119, 123, (E, H, N // 32), dtype=torch.uint8, device=dev, generator=g)
    for packed in (w1, w2):  # vLLM's _canonicalize_fp4_zero_signs_
        magnitude = packed & 0x77
        nonzero = (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11
        packed.bitwise_and_(0x77 | (nonzero << 3))
    return w1, w1s, w2, w2s


def _routing(torch, unique, seed, scale, dev):
    pad = seed < 0  # the last -seed tokens are padding: ids -1, as the router's top-k writes for them
    seed = abs(seed)
    g = torch.Generator().manual_seed(seed)
    pool = torch.randperm(E, generator=g)[:unique].tolist()
    ids, used = [], set()
    for _ in range(TOKENS):
        must = [e for e in pool if e not in used][:TOPK]
        rest = [e for e in (pool[i] for i in torch.randperm(unique, generator=g).tolist()) if e not in must]
        row = (must + rest)[:TOPK]
        used.update(row)
        ids.append(row)
    if pad:
        for t in range(TOKENS - seed, TOKENS):
            ids[t] = [-1] * TOPK
    w = torch.rand(TOKENS, TOPK, generator=g) + 0.1
    x = torch.randn(TOKENS, H, generator=g) * scale
    return (torch.tensor(ids, dtype=torch.int32, device=dev), (w / w.sum(-1, keepdim=True) * 1.5).to(dev),
            x.to(torch.bfloat16).to(dev))


def _recorded(torch, dev):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_c1_ids.pt")
    if not os.path.exists(path):
        return []
    real = torch.load(path).to(dev)
    g = torch.Generator().manual_seed(9)
    out = []
    for i in range(real.shape[0]):
        w = torch.rand(TOKENS, TOPK, generator=g) + 0.1
        x = torch.randn(TOKENS, H, generator=g) * 0.5
        out.append((real[i].contiguous(), w.to(dev), x.to(torch.bfloat16).to(dev)))
    return out


class _Served:
    def __init__(self, torch, weights, dev):
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
        b = self.fm.bind(self.plan, scratch=self.scratch, a=x, experts=self.prepared, topk_weights=w, topk_ids=ids,
                         output=out, input_scales_static=True, unit_scale_contract=False)
        self.fm.run(binding=b)


def _all_cases(torch, dev):
    cases = [_routing(torch, u, seed, sc, dev) for u, seed, sc in CASES]
    return cases + _recorded(torch, dev)


def _child(path):
    import torch
    from torch.profiler import ProfilerActivity, profile

    dev = "cuda"
    served = _Served(torch, _weights(torch, dev), dev)
    cases = _all_cases(torch, dev)
    outs = []
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for ids, w, x in cases:
            out = torch.empty(TOKENS, H, dtype=torch.bfloat16, device=dev)
            served.run(x, ids, w, out)
            outs.append(out)
        torch.cuda.synchronize()
    names = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
    # the served run from a CUDA graph, replayed on the first recorded (or last synthetic) routing 20 times
    ids, w, x = cases[-1]
    xi, idi, wi = x.clone(), ids.clone(), w.clone()
    gout = torch.empty(TOKENS, H, dtype=torch.bfloat16, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            served.run(xi, idi, wi, gout)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        served.run(xi, idi, wi, gout)
    graph_outs = []
    for _ in range(20):
        gout.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        graph_outs.append(gout.clone().cpu())
    torch.save({"outs": [o.cpu() for o in outs], "names": names, "graph_outs": graph_outs}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--child", default=None)
    a = ap.parse_args()
    if a.child:
        return _child(a.child)
    os.environ["VLLM_MOET_B12X_MOE_FUSED"] = "0"  # this process: b12x's kernels as the reference
    import torch
    from torch.profiler import ProfilerActivity, profile

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import b12x.moe._shared.kernels.w4a8_compact_micro as cm
    import b12x_moe_fused_sm120 as mf

    dev = "cuda"
    served = _Served(torch, _weights(torch, dev), dev)
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    got = {}
    orig = cm.launch_w4a8_compact_micro

    def spy(**kw):
        got.update(kw)
        return orig(**kw)

    cm.launch_w4a8_compact_micro = spy
    cases = _all_cases(torch, dev)
    n_rec = len(cases) - len(CASES)
    ok, ref_outs, times = 0, [], []
    proj_mine = torch.empty(TOKENS * TOPK, 2 * N, dtype=torch.bfloat16, device=dev)
    route_mine = torch.empty(TOKENS * TOPK, H, dtype=torch.bfloat16, device=dev)
    inter_mine = None

    def regions():
        lay = cm._layout(int(got["max_tokens"]), TOPK, H, N)
        st = got["scratch"].view(torch.uint8).reshape(-1)
        r = {name: st.narrow(0, *lay[name]) for name in ("a_values", "a_scales", "projections", "intermediate", "route_output")}
        return r, int(got["max_tokens"]) * TOPK

    def run_mine(r, ids, w, rows_cap):
        mf.moe_fused(r["a_values"], r["a_scales"], got["w13"], got["w13_scales"], proj_mine, inter_mine, got["w2"],
                     got["w2_scales"], ids.reshape(-1), w.reshape(-1), got["alpha1"], got["input_scale"], got["alpha2"],
                     got["down_scale"], route_mine, TOKENS * TOPK, TOPK, rows_cap, H, N, LIMIT)

    def same_bits(r, ids, rows_cap):
        flat = ids.reshape(-1)
        rows = torch.nonzero((flat >= 0) & (flat < E)).flatten()
        proj_ref = r["projections"].view(torch.bfloat16).view(-1, 2 * N)
        wpr = 5 * 32
        iv = lambda t: t[: rows_cap * wpr * 4].view(torch.int32).view(rows_cap, wpr)
        isc = lambda t: t[rows_cap * wpr * 4: rows_cap * wpr * 4 + 5 * rows_cap * 4].view(torch.int32).view(5, rows_cap)
        okp = torch.equal(proj_mine[rows].view(torch.int16), proj_ref[rows].view(torch.int16))
        oki = torch.equal(iv(inter_mine)[rows], iv(r["intermediate"])[rows]) and torch.equal(isc(inter_mine)[:, rows], isc(r["intermediate"])[:, rows])
        route_ref = r["route_output"].view(torch.bfloat16).view(-1, H)[: TOKENS * TOPK]
        oko = torch.equal(route_mine.view(torch.int16), route_ref.view(torch.int16))
        return okp, oki, oko

    for i, (ids, w, x) in enumerate(cases):
        out = torch.empty(TOKENS, H, dtype=torch.bfloat16, device=dev)
        served.run(x, ids, w, out)
        torch.cuda.synchronize()
        ref_outs.append(out.cpu())
        r, rows_cap = regions()
        if inter_mine is None:
            inter_mine = torch.zeros_like(r["intermediate"])
        proj_mine.fill_(float("nan")); inter_mine.fill_(0xFF); route_mine.fill_(float("nan"))
        run_mine(r, ids, w, rows_cap)
        torch.cuda.synchronize()
        okp, oki, oko = same_bits(r, ids, rows_cap)
        ok += okp and oki and oko
        if i < len(CASES):
            u, seed, sc = CASES[i]
            print(f"distinct experts {u:2d} seed {seed:2d} x{sc}: projections {'ok' if okp else 'DIFFER'}, intermediate"
                  f" {'ok' if oki else 'DIFFER'}, route_output {'ok' if oko else 'DIFFER'}")
            if seed == 1:  # timing
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(a.iters):
                        flush.view(torch.int32).sum()
                        served.run(x, ids, w, out)
                        flush.view(torch.int32).sum()
                        run_mine(r, ids, w, rows_cap)
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

                times.append((u, pick(("CompactMicroProjection",)), pick(("CompactMicroActivation",)),
                              pick(("fc2_grouped_kernel", "Phase2Kernel")), pick(("moe_fused_kernel",))))
    print(f"recorded routings: {n_rec} cases in the count above")
    # replay stability: one routing 50 times (the kernel resets its control block)
    ids, w, x = cases[-1]
    served.run(x, ids, w, out)
    r, rows_cap = regions()
    rep = 0
    for _ in range(50):
        route_mine.fill_(float("nan"))
        run_mine(r, ids, w, rows_cap)
        torch.cuda.synchronize()
        rep += same_bits(r, ids, rows_cap)[2]
    cm.launch_w4a8_compact_micro = orig
    for u, pj, ac, f2, mk in times:
        print(f"  {u:2d} distinct experts, cold L2: b12x projection {pj:6.2f} + activation {ac:5.2f} + FC2 {f2:6.2f} ="
              f" {pj + ac + f2:6.2f} us, fused {mk:6.2f} us")
    print(f"kernel: {ok}/{len(cases)} identical (projections, intermediate, route_output); 50 replays: {rep}/50")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "child.pt")
        env = dict(os.environ, VLLM_MOET_B12X_MOE_FUSED="1")
        subprocess.run([sys.executable, os.path.abspath(__file__), "--child", path], env=env, check=True)
        child = torch.load(path)
    same = sum(torch.equal(r.view(torch.int16), c.view(torch.int16)) for r, c in zip(ref_outs, child["outs"]))
    gsame = sum(torch.equal(ref_outs[-1].view(torch.int16), c.view(torch.int16)) for c in child["graph_outs"])
    fused = any("moe_fused_kernel" in n for n in child["names"])
    others = [n for n in child["names"] if any(s in n for s in ("CompactMicroProjection", "CompactMicroActivation", "Phase2Kernel", "fc2_grouped_kernel"))]
    print(f"served path with VLLM_MOET_B12X_MOE_FUSED=1: MoE outputs {same}/{len(cases)} identical to b12x's; graph replays"
          f" {gsame}/20 identical; fused kernel launched {fused}; b12x MoE kernels launched {len(others)}")
    passed = ok == len(cases) and rep == 50 and same == len(cases) and gsame == 20 and fused and not others
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
