#!/usr/bin/env python3
"""What an NCCL collective costs inside a CUDA graph on 4 real TP ranks, through vLLM's own path.

vLLM's distributed state is initialised the way test_engram_wkv_tp4.py does it, and every collective goes through
tensor_model_parallel_all_reduce / tensor_model_parallel_all_gather (pynccl on the current stream, as served).
Each pattern is one "layer", captured `--layers` times into one CUDA graph:

  ar_gemv   MXFP8 GEMV v3 [T, 5120 <- 2048] (wo_b-like, a fresh weight per layer) -> all-reduce -> add into the residual
  ar_add    elementwise add [T, 5120] -> all-reduce -> add into the residual (the MoE output -> all-reduce pattern)
  ag_gemv   GEMV v3 [T, 6400 <- 6144] (the Engram wkv slice) -> all-gather (dim -1, pynccl + layout copy) -> a small
            elementwise kernel on the [T, 25600] result
  ar_fork   ar_gemv with an mHC-like side stream: the all-reduce joins the side stream (its last kernel is a second
            parent, as with fuse_mhc_all_reduce), the add after it forks two small kernels onto the side stream
  engram    the patched Engram forward (patch_vllm_engram_wkv_tp.py, split at decode sizes: lookup, all-gather of the
            hash-head rows, wkv slice, all-gather of the output, post kernel), synthetic weights

and the same chain without the collective (`*_local`; not for engram) as the kernel-only baseline. Printed per pattern: the captured
topology (node types, the NCCL kernel nodes' parents and children), the replay time per layer (CUDA events around
the replay, median), and from a torch-profiler trace of the replays on every rank the GPU-idle gap in front of each
NCCL kernel (no kernel of the launch running on any stream), its duration and the gap behind it.

--rewrite events|dedupe|both|records modifies the captured graph before instantiation: drop the event record / wait
nodes NCCL adds for graph mixing (experiment only), and/or the edges into NCCL nodes that another path already
implies, or (records) only NCCL's intermediate serialEvent records - the graph's first wait and last record stay.

usage (serving image, 4 GPUs; NCCL environment variants are separate runs):
    torchrun --nproc_per_node=4 nccl_gap_tp4.py [--patterns ar_gemv,ar_add,ag_gemv] [--layers 40] [--tokens 6]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GEMV_DIRS = [HERE.parent / "sm120_gemv", Path("/opt/vllm-moet/dsv41_sm120/sm120_gemv")]

NCCL_ENV = ("NCCL_GRAPH_MIXING_SUPPORT", "NCCL_GRAPH_STREAM_ORDERING", "NCCL_LAUNCH_MODE", "NCCL_PROTO", "NCCL_ALGO",
            "NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS", "NCCL_P2P_LEVEL", "NCCL_P2P_DISABLE", "NCCL_LAUNCH_ORDER_IMPLICIT")


def mx_weight(n: int, k: int, dev, g):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale

    w = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=dev, generator=g)
    w[(w & 0x7F) == 0x7F] = 0
    rows = torch.randint(118, 124, (n, k // 32), dtype=torch.uint8, device=dev, generator=g)
    return w.view(torch.float8_e4m3fn), swizzle_mxfp8_scale(rows, M=n, K=k)


def trace_stats(path: str, n_layers: int) -> dict:
    """Per graph launch: gaps before / after each NCCL kernel (by collective), NCCL durations, time per layer."""
    ev = json.load(open(path))["traceEvents"]
    kern = [e for e in ev if e.get("cat") == "kernel"]
    launches = {e["args"]["correlation"] for e in ev if e.get("cat") == "cuda_runtime" and "GraphLaunch" in e.get("name", "")}
    by: dict = {}
    for k in kern:
        c = k["args"].get("correlation")
        if c in launches:
            by.setdefault(c, []).append(k)
    per_coll: dict = {}
    per_layer, streams = [], set()
    for ks in by.values():
        ks.sort(key=lambda k: k["ts"])
        nccl = [i for i, k in enumerate(ks) if k["name"].startswith("ncclDevKernel")]
        for j, i in enumerate(nccl):
            k = ks[i]
            streams.add(k["args"].get("stream"))
            if j == 0:
                continue  # the first one absorbs the skew between the ranks' launches
            coll = "AllGather" if "AllGather" in k["name"] else "AllReduce" if "AllReduce" in k["name"] else "other"
            st = per_coll.setdefault(coll, {"pre": [], "dur": [], "post": [], "stream_switch": []})
            end_before = max((p["ts"] + p["dur"] for p in ks[:i]), default=k["ts"])
            last = max(ks[:i], key=lambda p: p["ts"] + p["dur"])
            st["pre"].append(k["ts"] - end_before)
            st["dur"].append(k["dur"])
            st["stream_switch"].append(last["args"].get("stream") != k["args"].get("stream"))
            later = [q["ts"] for q in ks[i + 1:]]
            if later:
                st["post"].append(min(later) - (k["ts"] + k["dur"]))
        if len(nccl) >= 2:
            first_end = ks[nccl[0]]["ts"] + ks[nccl[0]]["dur"]
            last_end = max(k["ts"] + k["dur"] for k in ks)
            per_layer.append((last_end - first_end) / (n_layers - 1))
    med = lambda v: statistics.median(v) if v else float("nan")
    other_streams = {k["args"].get("stream") for ks in by.values() for k in ks if not k["name"].startswith("ncclDevKernel")}
    out = {"per_layer": med(per_layer), "nccl_streams": len(streams), "other_streams": len(other_streams),
           "launches": len(by), "colls": {}}
    for coll, st in per_coll.items():
        out["colls"][coll] = {"pre_gap": med(st["pre"]), "nccl_dur": med(st["dur"]), "post_gap": med(st["post"]),
                              "switch": sum(st["stream_switch"]) / len(st["stream_switch"]), "n": len(st["pre"])}
    return out


ENGRAM_CFG = dict(
    hidden_size=5120, hc_mult=4, rms_norm_eps=1e-20, engram_layer_ids=[1, 14], engram_num_embeddings=[20000] * 2,
    engram_max_ngram_size=4, engram_vocab_size=64, engram_n_heads=8, engram_head_dim=256,
    engram_pad_token_id=2, engram_compressed_vocab_size=99,
)


def build_engram(dev, rank: int, tokens: int):
    """The patched Engram on synthetic weights (as test_engram_wkv_tp4.py builds it) and inputs for `tokens`."""
    from types import SimpleNamespace

    import vllm.models.deepseek_v41.common.engram as E
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

    cfg = SimpleNamespace(**ENGRAM_CFG)
    qc = DeepseekV4FP8Config.from_config({"quant_method": "fp8", "activation_scheme": "dynamic",
                                          "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"})
    layout = E.EngramLayout(cfg)
    engram = E.Engram(cfg, qc, layout, 0, use_sequence_parallel=False, prefix="model.layers.1.engram")
    g_rank = torch.Generator(device=dev).manual_seed(100 + rank)
    g_all = torch.Generator(device=dev).manual_seed(7)
    emb = engram.embed_tokens
    table = torch.randint(0, 256, tuple(emb.weight.shape), dtype=torch.uint8, device=dev, generator=g_rank)
    table[(table & 0x7F) == 0x7F] = 0
    emb.weight.data.copy_(table.view(torch.float8_e4m3fn))
    emb.weight_scale_inv.data.copy_(torch.randint(127 - 8, 127 - 4, tuple(emb.weight_scale_inv.shape),
                                                  dtype=torch.uint8, device=dev, generator=g_rank))
    for p in (engram.q_weight, engram.k_weight):
        p.data.copy_((torch.randn(cfg.hc_mult, cfg.hidden_size, device=dev, generator=g_all) * 0.05).to(torch.bfloat16))
    w = torch.randint(0, 256, (25600, 6144), dtype=torch.uint8, device=dev, generator=g_all)
    w[(w & 0x7F) == 0x7F] = 0
    s = torch.randint(127 - 14, 127 - 3, (800, 192), dtype=torch.uint8, device=dev, generator=g_all)
    for param, tensor in ((engram.wkv.weight, w.view(torch.float8_e4m3fn)), (engram.wkv.weight_scale, s.view(torch.float8_e8m0fnu))):
        param.weight_loader(param, tensor)
    engram.wkv.quant_method.process_weights_after_loading(engram.wkv)
    del w, s, table
    sizes = torch.tensor([p for per_ngram in layout.primes[0] for p in per_ngram], device=dev)
    ids = ((torch.rand(tokens, sizes.numel(), device=dev, generator=g_all) * sizes).long()
           + layout.offsets[0].to(dev)).to(torch.int32)
    hidden = (torch.randn(tokens, cfg.hc_mult, cfg.hidden_size, device=dev, generator=g_all) * 2).to(torch.bfloat16)
    E._MOET_WKV_TP = (True, 64)
    return engram, ids, hidden


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", default="ar_gemv,ar_add,ag_gemv")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--ag-layers", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--rewrite", choices=("none", "events", "dedupe", "both", "records"), default="none")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda", torch.cuda.current_device())

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils.torch_utils import set_default_torch_dtype

    for d in GEMV_DIRS:
        if (d / "mxfp8_gemv_sm120.py").exists():
            sys.path.insert(0, str(d))
            break
    from mxfp8_gemv_sm120 import mxfp8_gemv
    from graph_topology import Topology

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    T = args.tokens
    with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(torch.bfloat16), torch.device(dev):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        tp = get_tp_group()
        comm = tp.device_communicator
        cpu_group = tp.cpu_group
        pyn = getattr(comm, "pynccl_comm", None)
        log(f"{torch.cuda.get_device_name(dev)} x {world}; tag={args.tag!r}; NCCL "
            f"{pyn.nccl.ncclGetVersion() if pyn is not None else '?'}; env "
            + " ".join(f"{k}={os.environ[k]}" for k in NCCL_ENV if k in os.environ))
        log("all-reduce backends: " + ", ".join(
            f"{a}={'on' if getattr(comm, a, None) is not None and not getattr(getattr(comm, a), 'disabled', False) else 'off'}"
            for a in ("ca_comm", "symm_mem_comm", "fi_ar_comm", "fi_pcie_ipc_ar_comm", "qr_comm", "pynccl_comm")))

        g = torch.Generator(device=dev).manual_seed(11)
        x = torch.randn(T, 2048, device=dev, generator=g).to(torch.bfloat16)
        a, a_sf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        xe = torch.randn(T, 6144, device=dev, generator=g).to(torch.bfloat16)
        ae, ae_sf = mxfp8_e4m3_quantize(xe, is_sf_swizzled_layout=True)
        resid = torch.zeros(T, 5120, device=dev, dtype=torch.bfloat16)
        bias = torch.randn(T, 5120, device=dev, generator=g).to(torch.bfloat16)
        sink = torch.zeros(T, 6400 * world, device=dev, dtype=torch.bfloat16)
        pats = args.patterns.split(",")
        w_ar = ([mx_weight(5120, 2048, dev, g) for _ in range(args.layers)]
                if {"ar_gemv", "ar_fork"} & set(pats) else [])
        w_ag = [mx_weight(6400, 6144, dev, g) for _ in range(args.ag_layers)] if "ag_gemv" in pats else []
        side = torch.cuda.Stream()
        side_buf = torch.zeros(T, 5120, device=dev, dtype=torch.bfloat16)
        engram = build_engram(dev, rank, T) if "engram" in pats else None

        def layer(pat: str, i: int, collective: bool) -> None:
            if pat == "ar_gemv":
                y = mxfp8_gemv(a, a_sf, *w_ar[i])
                resid.add_(tensor_model_parallel_all_reduce(y) if collective else y)
            elif pat == "ar_add":
                u = resid + bias
                resid.add_(tensor_model_parallel_all_reduce(u) if collective else u)
            elif pat == "ag_gemv":
                y = mxfp8_gemv(ae, ae_sf, *w_ag[i])
                full = tensor_model_parallel_all_gather(y, dim=-1) if collective else y.repeat(1, world)
                sink.mul_(0.5).add_(full)
            elif pat == "ar_fork":
                y = mxfp8_gemv(a, a_sf, *w_ar[i])
                torch.cuda.current_stream().wait_stream(side)  # the join lands on the all-reduce
                resid.add_(tensor_model_parallel_all_reduce(y) if collective else y)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    side_buf.mul_(0.5).add_(resid)
                    side_buf.mul_(0.25)
            elif pat == "engram":
                ids, hidden = engram[1], engram[2]
                engram[0].prepare_embeddings(ids)
                sink[:, : 5120].add_(engram[0](hidden, ids)[:, 0])

        def barrier() -> None:
            dist.barrier(group=cpu_group)

        results = {}
        for pat in pats:
            n_layers = args.ag_layers if pat in ("ag_gemv", "engram") else args.layers
            for collective in ((True,) if pat == "engram" else (True, False)):
                name = pat if collective else pat + "_local"
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for i in range(n_layers):
                        layer(pat, i, collective)  # warm-up outside capture
                    stream.wait_stream(side)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph, stream=stream):
                        side.wait_stream(stream)  # bring the side stream into the capture
                        for i in range(n_layers):
                            layer(pat, i, collective)
                        stream.wait_stream(side)
                topo = Topology(graph.raw_cuda_graph())
                note = ""
                if collective and args.rewrite in ("events", "both"):
                    note += f" dropped {topo.drop_event_nodes()} event nodes;"
                if collective and args.rewrite == "records":
                    note += f" dropped {topo.drop_intermediate_records()} intermediate event-record nodes;"
                if collective and args.rewrite in ("dedupe", "both"):
                    note += f" dropped {topo.drop_redundant_nccl_edges()} redundant edges into NCCL nodes;"
                if collective:
                    log(f"[{name}] topology:{note}\n" + topo.summary())
                graph.instantiate()
                torch.cuda.synchronize()
                for _ in range(3):
                    barrier()
                    graph.replay()
                torch.cuda.synchronize()
                ts = []
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for _ in range(args.reps):
                    barrier()
                    st.record()
                    graph.replay()
                    en.record()
                    en.synchronize()
                    ts.append(st.elapsed_time(en) * 1000 / n_layers)
                res = {"replay_per_layer": statistics.median(ts)}
                if collective and not args.no_profile:
                    from torch.profiler import ProfilerActivity, profile

                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        for _ in range(8):
                            barrier()
                            graph.replay()
                            torch.cuda.synchronize()
                    path = os.path.join(tempfile.gettempdir(), f"nccl_gap_{name}_r{rank}.json")
                    prof.export_chrome_trace(path)
                    res.update(trace_stats(path, n_layers))
                results[name] = res
                del graph
                torch.cuda.synchronize()

        gathered = [None] * world
        dist.all_gather_object(gathered, results, group=cpu_group)
        if rank == 0:
            print(f"\nper layer, us (replay: CUDA events around the replay / layers, median of {args.reps}; "
                  "profile: median over the NCCL kernels after the first of each launch, 8 launches)")
            for pat in pats:
                rows = [r[pat] for r in gathered]
                rep = statistics.median(r["replay_per_layer"] for r in rows)
                if pat + "_local" in gathered[0]:
                    loc = statistics.median(r[pat + "_local"]["replay_per_layer"] for r in gathered)
                    print(f"  {pat:8s} replay {rep:7.2f}  without the collective {loc:7.2f}  -> collective costs "
                          f"{rep - loc:6.2f}")
                else:
                    print(f"  {pat:8s} replay {rep:7.2f}")
                if "colls" in rows[0]:
                    for rk, r in enumerate(rows):
                        for coll, c in sorted(r["colls"].items()):
                            print(f"    rank {rk} {coll:9s}: idle before {c['pre_gap']:5.2f}  kernel {c['nccl_dur']:6.2f}  "
                                  f"idle after {c['post_gap']:5.2f}  (stream switch into it {c['switch']:.0%}, n={c['n']})"
                                  f"  steady per layer {r['per_layer']:7.2f}  "
                                  f"[NCCL on {r['nccl_streams']} stream(s), others on {r['other_streams']}]")
            print(json.dumps({"tag": args.tag, "rewrite": args.rewrite,
                              "env": {k: os.environ[k] for k in NCCL_ENV if k in os.environ},
                              "results": gathered[0]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
