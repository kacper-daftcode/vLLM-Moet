"""GPU test (one sm_120 GPU, serving image) of b12x_fc2_grouped_sm120 and patch_flashinfer_b12x_fc2_grouped_sm120.py.

  python3 test_b12x_fc2_grouped_sm120.py [--iters 30]

Synthetic MXFP4 experts of one TP4 rank of DeepSeek-V4.1-Flash (384 experts, hidden 5120, intermediate 576, top-6,
SwiGLU limit 10), prepared and run through b12x's planned API as vLLM's B12xExperts does; 6 tokens (a C1 decode step
with five drafts), routings with 6, 12, 22 and 36 distinct experts, three seeds each.
1. Kernel: with VLLM_MOET_B12X_FC2_GROUPED=0 (b12x's own phase 2) the grouped FC2 on the same scratch must give
   route_output with the same bits, also with padding tokens (ids -1: zeros). Prints cold-L2 kernel times of both (L2 refilled by a 256 MB read first).
2. Served path: a child process with VLLM_MOET_B12X_FC2_GROUPED=1 runs the patched b12x MoE on the same inputs; its MoE
   outputs must equal the parent's bit for bit, and its launches must hold the grouped kernel and no phase 2 kernel.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import tempfile

E, H, N, TOPK, LIMIT, TOKENS = 384, 5120, 576, 6, 10.0, 6
CASES = [(u, seed) for u in (6, 12, 22, 36) for seed in (1, 2, 3)] + [(22, -1), (12, -2)]  # seed < 0: padding tokens


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


def _routing(torch, unique, seed, dev):
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
    x = torch.randn(TOKENS, H, generator=g) * 0.5
    return (torch.tensor(ids, dtype=torch.int32, device=dev), (w / w.sum(-1, keepdim=True) * 1.5).to(dev),
            x.to(torch.bfloat16).to(dev))


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


def _child(path):
    import torch
    from torch.profiler import ProfilerActivity, profile

    dev = "cuda"
    served = _Served(torch, _weights(torch, dev), dev)
    outs = []
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for u, seed in CASES:
            ids, w, x = _routing(torch, u, seed, dev)
            out = torch.empty(TOKENS, H, dtype=torch.bfloat16, device=dev)
            served.run(x, ids, w, out)
            outs.append(out)
        torch.cuda.synchronize()
    names = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
    torch.save({"outs": [o.cpu() for o in outs], "names": names}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--child", default=None)
    a = ap.parse_args()
    if a.child:
        return _child(a.child)
    os.environ["VLLM_MOET_B12X_FC2_GROUPED"] = "0"  # this process: b12x's phase 2 as the reference
    import torch
    from torch.profiler import ProfilerActivity, profile

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import b12x.moe._shared.kernels.w4a8_compact_micro as cm
    import b12x_fc2_grouped_sm120 as fc2

    dev = "cuda"
    served = _Served(torch, _weights(torch, dev), dev)
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    got = {}
    orig = cm.launch_w4a8_compact_micro

    def spy(**kw):
        got.update(kw)
        return orig(**kw)

    cm.launch_w4a8_compact_micro = spy
    ok, ref_outs, times = 0, [], []
    route_mine = torch.empty(TOKENS * TOPK, H, dtype=torch.bfloat16, device=dev)
    for u, seed in CASES:
        ids, w, x = _routing(torch, u, seed, dev)
        out = torch.empty(TOKENS, H, dtype=torch.bfloat16, device=dev)
        served.run(x, ids, w, out)
        torch.cuda.synchronize()
        ref_outs.append(out.cpu())
        lay = cm._layout(int(got["max_tokens"]), TOPK, H, N)
        st = got["scratch"].view(torch.uint8).reshape(-1)
        inter = st.narrow(0, *lay["intermediate"])
        route = st.narrow(0, *lay["route_output"]).view(torch.bfloat16).view(-1, H)[: TOKENS * TOPK]
        args = (inter, got["w2"], got["w2_scales"], ids.reshape(-1), w.reshape(-1), got["alpha2"], got["down_scale"],
                route_mine, TOKENS * TOPK, int(got["max_tokens"]) * TOPK)
        route_mine.fill_(float("nan"))
        fc2.fc2_grouped(*args)
        torch.cuda.synchronize()
        same = torch.equal(route_mine.view(torch.int16), route.view(torch.int16))
        ok += same
        if seed == 1:  # timing
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(a.iters):
                    flush.view(torch.int32).sum()
                    served.run(x, ids, w, out)
                    flush.view(torch.int32).sum()
                    fc2.fc2_grouped(*args)
                torch.cuda.synchronize()
            per = {}
            for e in prof.events():
                if e.device_type.name == "CUDA":
                    per.setdefault(e.name, []).append(e.device_time)
            p2 = statistics.median(next(v for k, v in per.items() if "Phase2Kernel" in k))
            mk = statistics.median(next(v for k, v in per.items() if "fc2_grouped_kernel" in k))
            times.append((u, p2, mk))
        print(f"distinct experts {u:2d} seed {seed}: route_output bits {'identical' if same else 'DIFFER'}")
    cm.launch_w4a8_compact_micro = orig
    for u, p2, mk in times:
        print(f"  {u:2d} distinct experts, cold L2: b12x phase 2 {p2:6.2f} us, grouped FC2 {mk:6.2f} us")
    print(f"kernel: {ok}/{len(CASES)} identical")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "child.pt")
        env = dict(os.environ, VLLM_MOET_B12X_FC2_GROUPED="1")
        subprocess.run([sys.executable, os.path.abspath(__file__), "--child", path], env=env, check=True)
        child = torch.load(path)
    same = sum(torch.equal(r.view(torch.int16), c.view(torch.int16)) for r, c in zip(ref_outs, child["outs"]))
    grouped = any("fc2_grouped_kernel" in n for n in child["names"])
    phase2 = any("Phase2Kernel" in n for n in child["names"])
    print(f"served path with VLLM_MOET_B12X_FC2_GROUPED=1: MoE outputs {same}/{len(CASES)} identical to b12x's phase 2;"
          f" grouped kernel launched {grouped}, phase 2 kernel launched {phase2}")
    passed = ok == len(CASES) and same == len(CASES) and grouped and not phase2
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
