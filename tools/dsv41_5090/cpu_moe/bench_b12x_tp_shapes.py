"""GPU cost of one decode-step MoE layer under b12x at the TP4 rank shape and at the TP2 / TP1 shapes - the GPU side of
the "2x / 1x PRO 6000 support" question (handoff/tp1-feasibility-20261030.md). Runs in the vLLM base image on one
sm_120 GPU under its lock (minutes), synthetic MXFP4 records (the timing does not depend on the values):

    flock -w 3600 /tmp/exl3diff-gpu3.lock docker run --rm --gpus '"device=3"' --name moet-bench-tp \
      -v /mnt/nvme/cache/ds41/dot-cache:/root/.cache -v <cpu_moe dir>:/work:ro --entrypoint python3 \
      vllm/vllm-openai:nightly-df3bd094aeea829f27baa0a5282830464f601d9f /work/bench_b12x_tp_shapes.py [--iters 200]

Configurations (T tokens, top-k pairs per token, E experts of N intermediate rows each):
  tp4      N = 576  (2304 / 4), top-6, E = 8   - today's per-rank decode call: the compact N64 micro path (N % 128 == 64)
  tp1v     N = 576, top-24, E = 32            - TP1 as 4 "virtual ranks" on one GPU: every routed expert = its 4 rank
                                                slices as 4 pairs (same kernels and numerics as tp4, 4x the rows; the
                                                TopKSum adds the 4 partials in a fixed order instead of NCCL)
  tp2v     N = 576, top-12, E = 16            - TP2 as 2 virtual ranks per GPU
  tp1n     N = 2304, top-6, E = 8             - TP1 native: b12x's own path for N % 128 == 0 (NOT the compact micro
                                                kernels: _compact_launches returns None) - other kernels, other numerics
  tp2n     N = 1152, top-6, E = 8             - TP2 native, same remark
Each: the plan's implementation / compact flag, then GPU time per call from a CUDA graph of `iters` calls (replayed
3x, the median) and the eager wall time per call. The decode profile of 10-28 puts the TP4 layer at ~77 us of MoE
kernels (b12x MoE 3.1 ms / 40 layers, incl. the rows quant and the TopKSum); compare against that.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import torch

K = 5120  # hidden


def build(E: int, N: int, dev, g: torch.Generator):
    """Random MXFP4 records in b12x's input format: w13 [E, 2N, K/2] nibbles + [E, 2N, K/32] e8m0, w2 [E, K, N/2] + [E, K, N/32]."""
    def nib(*shape):
        return torch.randint(0, 256, shape, generator=g, dtype=torch.int64).to(torch.uint8).to(dev)

    def sc(*shape):
        return torch.randint(118, 127, shape, generator=g, dtype=torch.int64).to(torch.uint8).to(dev)

    return nib(E, 2 * N, K // 2), sc(E, 2 * N, K // 32), nib(E, K, N // 2), sc(E, K, N // 32)


def one_config(name: str, E: int, N: int, T: int, TOPK: int, iters: int, dev) -> dict:
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe import _impl, _preparation

    g = torch.Generator().manual_seed(7)
    w13, w13s, w2, w2s = build(E, N, dev, g)
    plan_w = fused_moe.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
                                    params_dtype=torch.bfloat16, num_experts=E, hidden_size=K, intermediate_size=N,
                                    w13_layout="w31")
    one = torch.ones(E, dtype=torch.float32, device=dev)
    prepared = fused_moe.prepare_weights(plan=plan_w, w1_fp4=w13, w1_blockscale=w13s, w1_global_scale=one, a1_gscale=one,
                                         w2_fp4=w2, w2_blockscale=w2s, w2_global_scale=one, a2_gscale=one,
                                         params_dtype=torch.bfloat16)
    caps = fused_moe.Caps(max_tokens=T, num_topk=TOPK, device=dev, weight_plan=prepared.plan, core_token_counts=(T,),
                          route_num_experts=0, quant_mode="w4a8_mx", apply_router_weight_on_input=False, swiglu_limit=10.0,
                          swiglu_alpha=None, swiglu_beta=None, frozen=True)
    plan = fused_moe.plan(caps)
    compact = _preparation._compact_launches(plan.launch_plan, caps)
    impl = plan.launch_plan.implementation
    # attach the compact launchers the way the serve does (patch_flashinfer_b12x_compact_bind.py)
    orig_bind = _impl.TPMoEScratchPlan.bind

    def bind_with_compact(self, **kw):
        b = orig_bind(self, **kw)
        return dataclasses.replace(b, compact_launches=compact) if compact is not None else b

    _impl.TPMoEScratchPlan.bind = bind_with_compact
    try:
        (spec,) = plan.scratch_specs()
        scratch = torch.zeros(int(spec.shape[0]), dtype=torch.uint8, device=dev)
        x = (torch.randn(T, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
        # every call routes each token to TOPK distinct experts, a DIFFERENT random subset per call, so that across the
        # graph's calls the experts stream from VRAM as in the decode (one fixed subset would sit in the 96 MB L2)
        ids_list = [torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).to(torch.int32).to(dev)
                    for _ in range(iters)]
        wts = torch.full((T, TOPK), 1.0 / TOPK, dtype=torch.float32, device=dev)
        out = torch.empty(T, K, dtype=torch.bfloat16, device=dev)

        def call(i: int):
            b = fused_moe.bind(plan, scratch=scratch, a=x, experts=prepared, topk_weights=wts, topk_ids=ids_list[i % iters],
                               output=out, input_scales_static=True, unit_scale_contract=False)
            fused_moe.run(binding=b)

        for i in range(5):
            call(i)
        torch.cuda.synchronize()
        # eager wall per call
        t0 = time.perf_counter()
        for i in range(iters):
            call(i)
        torch.cuda.synchronize()
        eager_us = (time.perf_counter() - t0) / iters * 1e6
        # CUDA graph of `iters` calls: GPU time per call
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                call(i)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                for i in range(iters):
                    call(i)
        torch.cuda.synchronize()
        times = []
        for _ in range(3):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graph.replay()
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) / iters * 1e3)
        graph_us = sorted(times)[1]
    finally:
        _impl.TPMoEScratchPlan.bind = orig_bind
    r = dict(name=name, E=E, N=N, T=T, topk=TOPK, implementation=str(impl), compact=compact is not None,
             graph_us_per_call=round(graph_us, 1), eager_us_per_call=round(eager_us, 1),
             weight_bytes_per_call=T * TOPK * (2 * N * K // 2 + 2 * N * K // 32 + K * N // 2 + K * N // 32))
    print(f"{name:5s} E={E:3d} N={N:4d} T={T} top-{TOPK:<2d} impl={impl} compact={compact is not None}: "
          f"graph {graph_us:7.1f} us/call, eager {eager_us:7.1f} us/call, "
          f"{r['weight_bytes_per_call'] / 1e6:6.1f} MB of expert bytes per call", flush=True)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--configs", default="tp4,tp1v,tp2v,tp1n,tp2n")
    ap.add_argument("--experts", type=int, default=120, help="experts per layer in VRAM (120 = today's slots per layer per rank; "
                    "the virtual-rank configs get 4x / 2x that many 576-row slices)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    print(torch.cuda.get_device_name(dev), flush=True)
    E0 = args.experts
    cfgs = dict(tp4=(E0, 576, 6), tp1v=(4 * E0, 576, 24), tp2v=(2 * E0, 576, 12), tp1n=(E0, 2304, 6), tp2n=(E0, 1152, 6))
    results = []
    for name in args.configs.split(","):
        E, N, TOPK = cfgs[name]
        try:
            results.append(one_config(name, E, N, args.tokens, TOPK, args.iters, dev))
        except Exception as ex:  # noqa: BLE001 - report and go on with the other shapes
            print(f"{name}: FAILED: {type(ex).__name__}: {str(ex)[:300]}", flush=True)
            results.append(dict(name=name, error=f"{type(ex).__name__}: {str(ex)[:300]}"))
    if args.out:
        import json
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
