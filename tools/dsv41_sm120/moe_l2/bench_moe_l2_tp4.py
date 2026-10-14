#!/usr/bin/env python3
"""What the padding-row L2 discard saves per MoE layer in a CUDA graph, through the patched DeepGemmFP4Experts.

One graph runs --layers decoder-like layers on --tokens tokens: an attention stand-in (three GEMV v3
[T, 5120 <- 2048], 31 MB, then vLLM's all-reduce), then the MoE as served - the router stand-in (~10 us), the patched
DeepGemmFP4Experts prepare -> apply (fused quant+scatter, FC1, silu*mul+quant, FC2, [discard], gather) on the main
stream next to the shared expert on an aux stream (GEMV v3 w13 [1152 <- 5120], silu*mul, quant, GEMV v1 w2
[5120 <- 576]), the add and vLLM's all-reduce. Every layer routes to fresh random experts (top-6 of --experts), so
weights are cold as in the served graph. Variants: the discard off (VLLM_MOET_MOE_FC2_DISCARD=0 equivalent), FC2's
padding rows discarded inline on the model stream (fc2; VLLM_MOET_MOE_FC2_DISCARD_SIDE=0, as served since
2026-10-13), the same discard forked onto the patch's side stream after FC2 and joined after the gather (fc2side;
the patch's default), and one experiment: fc2+fc1 (FC1's padding rows too, right after the FC1 call, before
silu*mul reads the real rows; FC2 later writes over the same workspace).

Every variant's graph is captured first, then the graphs are replayed round-robin. Printed per variant (rank-0
medians over the layers after the first and --replays profiled replays; torch profiler): FC1, FC2, the discard
kernels, the gather, the layer span (FC1 to FC1), and the replay time per layer (CUDA events, --rounds replays);
"paired" is the median over rounds of the difference to the same round's "off" replay.

usage (serving image, the patch applied):
    torchrun --nproc_per_node=4 bench_moe_l2_tp4.py --tokens 6 [--experts 288] [--topk 6] [--variants off,fc2,fc2side]
(--experts 256 keeps a 4-rank run under 3 GB per GPU; the DSpark drafter's MoE: --tokens 5 --experts 128 --topk 3.)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _d in (HERE.parent / "nccl_graph", Path("/opt/vllm-moet/dsv41_sm120/nccl_graph")):
    if (_d / "nccl_gap_tp4.py").exists():
        sys.path.insert(0, str(_d))
        break
for _d in (HERE.parent / "sm120_gemv", Path("/opt/vllm-moet/dsv41_sm120/sm120_gemv")):
    if (_d / "mxfp8_gemv_sm120.py").exists():
        sys.path.insert(0, str(_d))
        break

K, I = 5120, 640
N1 = 2 * I


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--topk", type=int, default=6, help="6 = the target model; the DSpark drafter: 128 experts, top-3")
    ap.add_argument("--distinct", type=int, default=0,
                    help="route each layer's pairs over exactly this many experts (0: top-k of all, per token)")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--replays", type=int, default=8, help="profiled replays per variant")
    ap.add_argument("--rounds", type=int, default=40, help="timed replays per variant (CUDA events)")
    ap.add_argument("--variants", default="off,fc2,fc2side")
    args = ap.parse_args()
    for key, value in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29551"), ("RANK", "0"), ("WORLD_SIZE", "1"),
                       ("LOCAL_RANK", "0")):
        os.environ.setdefault(key, value)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda", torch.cuda.current_device())

    import vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe as dgm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_reduce,
    )
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig, FusedMoEQuantDesc
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import DeepGemmFP4Experts
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _pack_deepgemm_mxfp4_scales
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import set_default_torch_dtype

    import moe_l2_discard_sm120 as l2
    from mxfp8_gemv_sm120 import mxfp8_gemv
    from nccl_gap_tp4 import mx_weight

    assert hasattr(dgm, "_moet_drop_padding_rows") and hasattr(dgm, "_moet_l2_side_stream"), \
        "deep_gemm_moe.py is not patched with the side-stream discard"
    T, E, L, TOPK = args.tokens, args.experts, args.layers, args.topk
    with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(torch.bfloat16), torch.device(dev):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        cpu_group = get_tp_group().cpu_group if world > 1 else None
        g = torch.Generator(device=dev).manual_seed(9)
        w13 = torch.randint(0, 256, (E, N1, K // 2), device=dev, dtype=torch.uint8, generator=g)
        w2 = torch.randint(0, 256, (E, K, I // 2), device=dev, dtype=torch.uint8, generator=g)
        w13_s = torch.randint(118, 126, (E, N1, K // 32), device=dev, dtype=torch.uint8, generator=g)
        w2_s = torch.randint(118, 126, (E, K, I // 32), device=dev, dtype=torch.uint8, generator=g)
        w13_s, w2_s = _pack_deepgemm_mxfp4_scales(w13, w2, w13_s, w2_s)
        fp8, block = current_platform.fp8_dtype(), GroupShape(128, 128)
        experts = DeepGemmFP4Experts.__new__(DeepGemmFP4Experts)
        experts.quant_config = FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(fp8, block, None, None, None, None),
            _a2=FusedMoEQuantDesc(fp8, block, None, None, None, None),
            _w1=FusedMoEQuantDesc("mxfp4", None, w13_s, None, None, None),
            _w2=FusedMoEQuantDesc("mxfp4", None, w2_s, None, None, None),
        )
        experts.moe_config = SimpleNamespace(in_dtype=torch.bfloat16)
        experts.gemm1_clamp_limit = None
        prepare = MoEPrepareAndFinalizeNoDPEPModular()
        s13, s2, _ = experts.workspace_shapes(T, N1, K, TOPK, E, E, None, MoEActivation.SILU)
        ws13 = torch.empty(s13, device=dev, dtype=torch.bfloat16)
        ws2 = torch.empty(s2, device=dev, dtype=torch.bfloat16)
        def route() -> torch.Tensor:
            if not args.distinct:
                return torch.stack([torch.randperm(E, device=dev, generator=g)[:TOPK] for _ in range(T)])
            # the T x TOPK pairs over exactly D experts: token t takes TOPK consecutive entries of a random pool
            pool = torch.randperm(E, device=dev, generator=g)[: args.distinct]
            idx = (torch.arange(T, device=dev)[:, None] * TOPK + torch.arange(TOPK, device=dev)) % args.distinct
            return pool[idx]

        assert not args.distinct or TOPK <= args.distinct <= min(E, T * TOPK)
        routes = [route().to(torch.int32) for _ in range(L)]
        weights = torch.softmax(torch.randn(T, TOPK, device=dev, generator=g), dim=-1)
        x = (torch.randn(T, K, device=dev, generator=g) * 0.7).to(torch.bfloat16)
        moe_out = torch.empty(T, K, device=dev, dtype=torch.bfloat16)
        att_w = [mx_weight(5120, 2048, dev, g) for _ in range(6)]
        aq, aq_sf = mxfp8_e4m3_quantize(torch.randn(T, 2048, device=dev, generator=g).to(torch.bfloat16),
                                        is_sf_swizzled_layout=True)
        sh_w13 = [mx_weight(1152, K, dev, g) for _ in range(2)]
        sh_w2 = [mx_weight(K, 576, dev, g) for _ in range(2)]
        hq, hq_sf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        resid = torch.zeros(T, K, device=dev, dtype=torch.bfloat16)
        aux = torch.cuda.Stream()

        fc1_discard = {"on": False}
        real_gemm = dgm.m_grouped_fp8_fp4_gemm_nt_contiguous

        def gemm_with_fc1_discard(*a, **k):
            out = real_gemm(*a, **k)
            if fc1_discard["on"] and a[2].size(1) == N1:  # FC1's output, before silu*mul reads its real rows
                l2.discard_padding_rows(a[2], a[3], 64)
            return out

        dgm.m_grouped_fp8_fp4_gemm_nt_contiguous = gemm_with_fc1_discard

        def shared(j: int) -> torch.Tensor:
            h = mxfp8_gemv(hq, hq_sf, *sh_w13[j % 2])
            h = torch.nn.functional.silu(h[:, :576]) * h[:, 576:]
            q, q_sf = mxfp8_e4m3_quantize(h.contiguous(), is_sf_swizzled_layout=True)
            return mxfp8_gemv(q, q_sf, *sh_w2[j % 2])

        def layer(j: int) -> None:
            for k in range(3):
                resid.add_(mxfp8_gemv(aq, aq_sf, *att_w[(j % 2) * 3 + k]))
            resid.add_(tensor_model_parallel_all_reduce(resid))
            aux.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(aux):
                sh = shared(j)
            torch.cuda._sleep(20000)  # router + top-k in front of the MoE (~10 us)
            a1q, a1q_scale, _, _, _ = prepare.prepare(x, weights, routes[j], E, None, False, experts.quant_config,
                                                      defer_input_quant=experts.expects_unquantized_inputs)
            experts.apply(moe_out, a1q, w13, w2, weights, routes[j], MoEActivation.SILU, E, None, a1q_scale, None,
                          ws13, ws2, None, False)
            torch.cuda.current_stream().wait_stream(aux)
            resid.add_(tensor_model_parallel_all_reduce(moe_out + sh))

        def barrier() -> None:
            if cpu_group is not None:
                dist.barrier(group=cpu_group)

        torch.cuda.empty_cache()  # the unpacked scales etc.: the PRO 6000 budget next to qwen38 is 3 GB per GPU
        # one graph per variant (the switch is read at capture), then every measurement round-robin over the
        # variants, so clock / thermal drift hits all of them alike
        variants = args.variants.split(",")
        graphs = {}
        for variant in variants:
            dgm._MOET_L2 = None
            dgm._MOET_L2_SIDE = None
            if variant == "off":
                dgm._MOET_L2 = False
            else:
                assert dgm._moet_moe_l2() is not None
                dgm._MOET_L2 = (dgm._MOET_L2[0], 1, 64)
                if "side" in variant:
                    assert dgm._moet_l2_side_stream() is not None, "VLLM_MOET_MOE_FC2_DISCARD_SIDE=0 in this environment"
                else:
                    dgm._MOET_L2_SIDE = False  # inline, as served
            fc1_discard["on"] = "fc1" in variant
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for j in range(L):
                    layer(j)
                stream.wait_stream(aux)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    aux.wait_stream(stream)
                    for j in range(L):
                        layer(j)
                    stream.wait_stream(aux)
            torch.cuda.synchronize()
            graphs[variant] = graph
        dgm.m_grouped_fp8_fp4_gemm_nt_contiguous = real_gemm
        for _ in range(3):
            for variant in variants:
                barrier()
                graphs[variant].replay()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts: dict = {v: [] for v in variants}
        for _ in range(args.rounds):
            for variant in variants:
                barrier()
                st.record()
                graphs[variant].replay()
                en.record()
                en.synchronize()
                ts[variant].append(st.elapsed_time(en) * 1000 / L)
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.replays):
                for variant in variants:
                    barrier()
                    graphs[variant].replay()
                    torch.cuda.synchronize()
        path = os.path.join(tempfile.gettempdir(), f"bench_moe_l2_r{rank}.json")
        prof.export_chrome_trace(path)
        ev = json.load(open(path))["traceEvents"]
        launch_ev = sorted((e for e in ev if e.get("cat") == "cuda_runtime" and "GraphLaunch" in e.get("name", "")),
                           key=lambda e: e["ts"])
        variant_of = {e["args"]["correlation"]: variants[i % len(variants)] for i, e in enumerate(launch_ev)}
        by: dict = {}
        for e in ev:
            if e.get("cat") == "kernel" and e["args"].get("correlation") in variant_of:
                by.setdefault(e["args"]["correlation"], []).append(e)
        results = {}
        for variant in variants:
            st_k = {"fc1": [], "fc2": [], "discard": [], "gather": [], "span": []}
            for corr, ks in by.items():
                if variant_of[corr] != variant:
                    continue
                ks.sort(key=lambda k: k["ts"])
                fc1 = [k for k in ks if "1d1d_impl<0u, 1280u, 5120u" in k["name"]]
                fc2 = [k for k in ks if "1d1d_impl<0u, 5120u, 640u" in k["name"]]
                st_k["fc1"] += [k["dur"] for k in fc1[1:]]
                st_k["fc2"] += [k["dur"] for k in fc2[1:]]
                st_k["discard"] += [k["dur"] for k in ks if "discard_padding_rows" in k["name"]]
                st_k["gather"] += [k["dur"] for k in ks if "ep_gather" in k["name"]]
                if len(fc1) >= 2:
                    st_k["span"].append((fc1[-1]["ts"] - fc1[0]["ts"]) / (len(fc1) - 1))
            med = {k: (statistics.median(v) if v else 0.0) for k, v in st_k.items()}
            med["replay"] = statistics.median(ts[variant])
            # paired with the round's "off" replay: drift between rounds cancels
            med["paired"] = (statistics.median(a - b for a, b in zip(ts[variant], ts["off"]))
                             if "off" in ts and variant != "off" else 0.0)
            med["n_discard"] = len(st_k["discard"]) / max(1, args.replays) / L
            med["peak_gib"] = torch.cuda.max_memory_reserved() / 2**30
            results[variant] = med
        del graphs
        torch.cuda.synchronize()
        gathered = [None] * world
        if world > 1:
            dist.all_gather_object(gathered, results, group=cpu_group)
        else:
            gathered = [results]
        if rank == 0:
            m_sum = s2[0]
            routing = f"top-{TOPK}" + (f" over {args.distinct} experts per layer" if args.distinct else "")
            print(f"{torch.cuda.get_device_name(dev)} x {world}: {T} tokens, {E} experts, {routing}, M_sum {m_sum}; "
                  f"{L} layers per graph; us (rank-0 medians; replay = CUDA events per layer)")
            base = gathered[0].get("off")
            for variant, r in gathered[0].items():
                d = "" if base is None or variant == "off" else (
                    f"   per layer vs off: replay {r['paired']:+6.2f} (paired), span {r['span'] - base['span']:+6.2f}")
                print(f"  {variant:8s} FC1 {r['fc1']:6.1f}  FC2 {r['fc2']:6.1f}  discard {r['discard']:4.1f} "
                      f"(x{r['n_discard']:.0f}/layer)  gather {r['gather']:4.1f}  layer span {r['span']:6.1f}  "
                      f"replay {r['replay']:6.1f}{d}")
            other = [f"rank {rk}: " + ", ".join(f"{v} {res[v]['replay']:.1f} ({res[v]['paired']:+.2f})" for v in res)
                     for rk, res in enumerate(gathered[1:], start=1)]
            if other:
                print("  replay per layer on the other ranks: " + "; ".join(other))
            peaks = [max(r["peak_gib"] for r in res.values()) for res in gathered]
            print("  peak memory reserved by torch per rank (GiB): " + ", ".join(f"{p:.2f}" for p in peaks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
