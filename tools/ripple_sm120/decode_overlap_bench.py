#!/usr/bin/env python3
"""How much of a TP4 all-reduce hides under GEMMs on another stream: the two-micro-batch overlap question for
Ripple's decode step at 8-32 requests, where the all-reduce is 33-47 % of the step on the PCIe floor.

Measured on rtx (4x RTX PRO 6000, PCIe Gen5 through the root complex, 2026-11-06; --ms 96,128,256 --layers 20): 85-89 %
of the halves' all-reduces hide under the other half's GEMMs, but two halves stream the layer's weights twice (the
decode GEMMs are weight-bandwidth-bound: two halves of M/2 cost +43 % (M 256) to +93 % (M 96) over one batch of M,
+24 % to +51 % when the halves alternate per attention / MLP block so the second finds the block's ~80 MB in the 128 MB
L2 - vLLM's DBO schedule). Net per layer of GEMMs + 2 all-reduces against today's serial step: the block schedule
-6.5 % at M 96 (8 requests x k 11), -14 % at M 128, -16.5 % at M 256 (32 requests x k 7); the naive two-stream split
+15 % / +4 % / -14 %. On the whole step (attention, norms, drafter and lm_head do not shrink) about -5 % at 8 requests
and -12 % at 32: days of runner work for that, not the -30...-40 % estimated from the all-reduce share alone.

Usage inside the image on 4 GPUs: VLLM_MOET_P2P_AR_M_MAX=128 PYTHONPATH=/opt/vllm-moet/dsv41_sm120/p2p_ar \
  torchrun --nproc_per_node=4 decode_overlap_bench.py --ms 96,128,256 --layers 20 [--replays 20]   (OUT=<json>)

Per rank (torchrun, 4 GPUs, the Ripple image): one decode "layer" of Ripple's per-rank GEMMs at M rows (BF16 cuBLAS
qkg 5376x5120 and o 5120x2560, NVFP4 FlashInfer-CUTLASS gate_up 20480x5120 and down 5120x10240, weight copies rotated
past L2) and the all-reduces of [M, 5120] bf16 (the P2P two-shot push for M <= 128, NCCL for any M). CUDA graphs of
LAYERS layers:
  serial     : per layer GEMMs(M) then 2 ARs(M) on one stream (today's step shape: o_proj + down_proj all-reduces)
  split      : per layer two halves of M/2 rows, serial (what splitting costs without overlap)
  overlap    : per layer two halves of M/2 rows, half h's ARs on the comm stream while half 1-h's GEMMs run (the
               two-micro-batch pattern: GEMMs_0 | AR_0 under GEMMs_1 | AR_1 under the next layer's GEMMs_0)
  gemm-only, ar-only: the parts alone
Prints per-layer times and the hidden share of the all-reduce. Exit 0 always (a measurement).
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2p_allreduce_sm120 import H, _build, nccl_order_tables  # noqa: E402
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator  # noqa: E402
from vllm.model_executor.kernels.linear.nvfp4 import flashinfer as fi  # noqa: E402
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ms", default="96,128,256")
ap.add_argument("--layers", type=int, default=20)
ap.add_argument("--replays", type=int, default=20)
ap.add_argument("--max-ctas", type=int, default=64)
a = ap.parse_args()

dist.init_process_group("gloo")
rank, ws = dist.get_rank(), dist.get_world_size()
assert ws == 4
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
M_MAX = int(os.environ.get("VLLM_MOET_P2P_AR_M_MAX", "128"))
ext = _build()
NSLOTS = 4
data2 = ext.alloc_shared(NSLOTS * 4 * M_MAX * H * 2)
flags2 = ext.alloc_shared(NSLOTS * 4 * 32 * 4)
h2 = [None] * ws
dist.all_gather_object(h2, (ext.ipc_handle(data2), ext.ipc_handle(flags2)))
peers = [r for r in range(ws) if r != rank]
peers_data2 = [ext.ipc_open(h2[r][0]) for r in peers]
peers_flags2 = [ext.ipc_open(h2[r][1]) for r in peers]
ctrl2 = torch.zeros(8, dtype=torch.int32, device=dev)
timeout = torch.zeros(1, dtype=torch.int32, device=dev)
dist.barrier()
comm = PyNcclCommunicator(dist.group.WORLD, device=dev)
MS = [int(m) for m in a.ms.split(",")]
p2p_rows = sorted({m for m in MS if m <= M_MAX} | {m // 2 for m in MS if m // 2 <= M_MAX})
orders = nccl_order_tables(comm.all_reduce, rank, dev, p2p_rows) if p2p_rows else {}


def ar_p2p(x):
    ext.all_reduce_twoshot(x, x, M_MAX, rank, data2, flags2, peers_data2, peers_flags2, ctrl2, orders[x.shape[0]],
                           timeout, a.max_ctas)


def ar_nccl(x):
    comm.all_reduce(x, out_tensor=x, stream=torch.cuda.current_stream())


# ---------------- the layer's GEMMs (per rank) ----------------
def make_nvfp4_layer(n, k, gen):
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev, generator=gen),
                                requires_grad=False)
    wsc = (torch.rand(n, k // 16, device=dev, generator=gen) * 0.5 + 0.25).to(torch.float8_e4m3fn)
    layer.weight_scale = nn.Parameter(wsc, requires_grad=False)
    layer.weight_global_scale = nn.Parameter(torch.tensor(0.0123, device=dev), requires_grad=False)
    layer.input_global_scale = nn.Parameter(torch.tensor(0.0456, device=dev), requires_grad=False)
    layer.input_global_scale_inv = nn.Parameter(1.0 / layer.input_global_scale.data, requires_grad=False)
    layer.alpha = nn.Parameter(layer.input_global_scale.data * layer.weight_global_scale.data, requires_grad=False)
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.params_dtype = torch.bfloat16
    layer.weight_scale_2 = nn.Parameter(layer.weight_global_scale.data.clone(), requires_grad=False)
    layer.input_scale = nn.Parameter(layer.input_global_scale.data.clone(), requires_grad=False)
    return layer


NCOPIES = 3  # 3 x ~160 MB per layer set > the 128 MB L2
kern = fi.FlashInferCutlassNvFp4LinearKernel(NvFp4LinearLayerConfig())
sets = []
for c in range(NCOPIES):
    g = torch.Generator(device=dev).manual_seed(100 + c)
    qkg = torch.randn(5376, 5120, device=dev, dtype=torch.bfloat16, generator=g)
    o = torch.randn(5120, 2560, device=dev, dtype=torch.bfloat16, generator=g)
    gu = make_nvfp4_layer(20480, 5120, g); kern.process_weights_after_loading(gu)
    dn = make_nvfp4_layer(5120, 10240, g); kern.process_weights_after_loading(dn)
    sets.append((qkg, o, gu, dn))


def gemms(xs, c):
    """One layer's GEMMs on rows xs = (x_h [m,5120], x_o [m,2560], x_d [m,10240]) with weight set c."""
    qkg, o, gu, dn = sets[c % NCOPIES]
    torch.nn.functional.linear(xs[0], qkg)
    torch.nn.functional.linear(xs[1], o)
    kern.apply_weights(gu, xs[0])
    kern.apply_weights(dn, xs[2])


def inputs(m):
    g = torch.Generator(device=dev).manual_seed(7 + m)
    return (torch.randn(m, 5120, device=dev, dtype=torch.bfloat16, generator=g),
            torch.randn(m, 2560, device=dev, dtype=torch.bfloat16, generator=g),
            torch.randn(m, 10240, device=dev, dtype=torch.bfloat16, generator=g))


def warm(fn):
    from flashinfer.autotuner import autotune as fi_autotune
    with fi_autotune(True):
        fn()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()


def time_graph(build, label):
    """build(stream_main, stream_comm) records LAYERS layers; returns µs per layer (median of replays)."""
    s = torch.cuda.Stream(); sc = torch.cuda.Stream()
    with torch.cuda.stream(s):
        build(s, sc)  # warm-up (eager)
    torch.cuda.synchronize(); dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        build(s, sc)
    torch.cuda.synchronize(); dist.barrier()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize(); dist.barrier()
    ts = []
    for _ in range(a.replays):
        dist.barrier()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / a.layers)
    t = int(timeout.item())
    if t:
        print(f"[rank {rank}] TIMEOUT flags {t:#x} in {label}", flush=True); timeout.zero_()
    ts.sort()
    return ts[len(ts) // 2]


results = []
for m in MS:
    ar_full = ar_p2p if m <= M_MAX else ar_nccl
    ar_half = ar_p2p if m // 2 <= M_MAX else ar_nccl
    xf = inputs(m); xh = [inputs(m // 2), inputs(m // 2)]
    arf = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
    arh = [torch.randn(m // 2, H, device=dev, dtype=torch.bfloat16) for _ in range(2)]
    warm(lambda: gemms(xf, 0)); warm(lambda: gemms(xh[0], 0))
    for fn, x in ((ar_full, arf), (ar_half, arh[0]), (ar_half, arh[1])):
        fn(x); torch.cuda.synchronize()
    dist.barrier()

    def b_gemm(s, sc):
        for L in range(a.layers):
            gemms(xf, L)

    def b_gemm_half(s, sc):
        for L in range(a.layers):
            gemms(xh[0], L); gemms(xh[1], L)

    def b_ar(s, sc):
        for L in range(a.layers):
            ar_full(arf); ar_full(arf)

    def b_ar_half(s, sc):
        for L in range(a.layers):
            for h in range(2):
                ar_half(arh[h]); ar_half(arh[h])

    def b_serial(s, sc):
        for L in range(a.layers):
            gemms(xf, L); ar_full(arf); ar_full(arf)

    def b_split(s, sc):
        for L in range(a.layers):
            for h in range(2):
                gemms(xh[h], L); ar_half(arh[h]); ar_half(arh[h])

    def b_overlap(s, sc):
        # half h's ARs on the comm stream after its GEMMs; the main stream goes on with the other half's GEMMs and
        # waits for a half's ARs only before that half's next GEMMs (one layer later).
        done = [None, None]
        sc.wait_stream(s)
        for L in range(a.layers):
            for h in range(2):
                if done[h] is not None:
                    s.wait_event(done[h])
                gemms(xh[h], L)
                ev = torch.cuda.Event(); ev.record(s)
                sc.wait_event(ev)
                with torch.cuda.stream(sc):
                    ar_half(arh[h]); ar_half(arh[h])
                    done[h] = torch.cuda.Event(); done[h].record(sc)
        s.wait_stream(sc)

    # block-level alternation (vLLM's DBO schedule): the attention block's GEMMs of half 0 then of half 1 (the second
    # half may find the block's ~80 MB of weights in the 128 MB L2), each half's all-reduce on the comm stream as soon
    # as its block is done, the MLP block likewise; a half waits for its own all-reduce only before its next block.
    def attn(xs, c):
        qkg, o, gu, dn = sets[c % NCOPIES]
        torch.nn.functional.linear(xs[0], qkg); torch.nn.functional.linear(xs[1], o)

    def mlp(xs, c):
        qkg, o, gu, dn = sets[c % NCOPIES]
        kern.apply_weights(gu, xs[0]); kern.apply_weights(dn, xs[2])

    def b_gemm_blk(s, sc):
        for L in range(a.layers):
            attn(xh[0], L); attn(xh[1], L); mlp(xh[0], L); mlp(xh[1], L)

    def b_split_blk(s, sc):
        for L in range(a.layers):
            attn(xh[0], L); attn(xh[1], L); ar_half(arh[0]); ar_half(arh[1])
            mlp(xh[0], L); mlp(xh[1], L); ar_half(arh[0]); ar_half(arh[1])

    def b_overlap_blk(s, sc):
        done = [None, None]
        sc.wait_stream(s)
        for L in range(a.layers):
            for block in (attn, mlp):
                for h in range(2):
                    if done[h] is not None:
                        s.wait_event(done[h])
                    block(xh[h], L)
                    ev = torch.cuda.Event(); ev.record(s)
                    sc.wait_event(ev)
                    with torch.cuda.stream(sc):
                        ar_half(arh[h])
                        done[h] = torch.cuda.Event(); done[h].record(sc)
        s.wait_stream(sc)

    r = dict(m=m, ar_full=ar_full.__name__, ar_half=ar_half.__name__)
    r["gemm"] = time_graph(b_gemm, "gemm")
    r["gemm_half2"] = time_graph(b_gemm_half, "gemm-half")
    r["gemm_blk"] = time_graph(b_gemm_blk, "gemm-blk")
    r["ar"] = time_graph(b_ar, "ar")
    r["ar_half2"] = time_graph(b_ar_half, "ar-half")
    r["serial"] = time_graph(b_serial, "serial")
    r["split"] = time_graph(b_split, "split")
    r["overlap"] = time_graph(b_overlap, "overlap")
    r["split_blk"] = time_graph(b_split_blk, "split-blk")
    r["overlap_blk"] = time_graph(b_overlap_blk, "overlap-blk")
    results.append(r)
    if rank == 0:
        hidden = (r["gemm_half2"] + r["ar_half2"] - r["overlap"]) / r["ar_half2"] * 100
        hidden_blk = (r["gemm_blk"] + r["ar_half2"] - r["overlap_blk"]) / r["ar_half2"] * 100
        print(f"M={m:4d} ({r['ar_full']} full / {r['ar_half']} halves), µs per layer: GEMMs {r['gemm']:.1f} (2 halves "
              f"{r['gemm_half2']:.1f}, halves alternating per block {r['gemm_blk']:.1f}), 2 ARs {r['ar']:.1f} (2x2 halves "
              f"{r['ar_half2']:.1f}) | serial {r['serial']:.1f}, split {r['split']:.1f}, overlap {r['overlap']:.1f} "
              f"({100 * (r['overlap'] / r['serial'] - 1):+.1f} % vs serial, hidden {hidden:.0f} % of the halves' ARs) | "
              f"per block: split {r['split_blk']:.1f}, overlap {r['overlap_blk']:.1f} "
              f"({100 * (r['overlap_blk'] / r['serial'] - 1):+.1f} % vs serial, hidden {hidden_blk:.0f} %)", flush=True)
dist.barrier()
if rank == 0:
    import json
    json.dump(results, open(os.environ.get("OUT", "/out/overlap_bench.json"), "w"), indent=1)
    print("OVERLAP_BENCH_DONE", flush=True)
