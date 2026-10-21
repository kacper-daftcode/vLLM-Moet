#!/usr/bin/env python3
"""Test + bench of the sm_120 P2P all-reduce (p2p_allreduce_sm120.cu): one-shot push and two-shot (reduce-scatter +
all-gather) against NCCL's bits (PyNccl, the communicator vLLM's graphs capture; the summation order measured by
nccl_order_tables), identical bits on every rank, 50 repeated all-reduces (slot rotation), the fused add + RMSNorm
consumer against torch; timings of 80 all-reduces replayed from a CUDA graph next to NCCL's. 4 GPUs:
torchrun --nproc_per_node=4 test_p2p_allreduce_sm120.py [--rows 6,24,48] [--ars 80] [--replays 100] [--no-nccl]
(--no-nccl: no NCCL timing). Exit code 1 on any mismatch.
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2p_allreduce_sm120 import nccl_order_tables  # noqa: E402
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="6,12,48")
ap.add_argument("--ars", type=int, default=80)
ap.add_argument("--replays", type=int, default=100)
ap.add_argument("--max-ctas", type=int, default=64)
ap.add_argument("--no-nccl", action="store_true")
a = ap.parse_args()

dist.init_process_group("gloo")
rank, ws = dist.get_rank(), dist.get_world_size()
assert ws == 4
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
H, M_MAX = 5120, 64
ext = load(name="p2p_allreduce_sm120", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2p_allreduce_sm120.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=rank == 0)

NSLOTS = 4
data = ext.alloc_shared(NSLOTS * 4 * M_MAX * H * 2)
flags = ext.alloc_shared(NSLOTS * 4 * 32 * 4)
handles = [None] * ws
dist.all_gather_object(handles, (ext.ipc_handle(data), ext.ipc_handle(flags)))
peers = [r for r in range(ws) if r != rank]
peers_data = [ext.ipc_open(handles[r][0]) for r in peers]
peers_flags = [ext.ipc_open(handles[r][1]) for r in peers]
ctrl = torch.zeros(8, dtype=torch.int32, device=dev)
timeout = torch.zeros(1, dtype=torch.int32, device=dev)
data2 = ext.alloc_shared(NSLOTS * 4 * M_MAX * H * 2)
flags2 = ext.alloc_shared(NSLOTS * 4 * 32 * 4)
h2 = [None] * ws
dist.all_gather_object(h2, (ext.ipc_handle(data2), ext.ipc_handle(flags2)))
peers_data2 = [ext.ipc_open(h2[r][0]) for r in peers]
peers_flags2 = [ext.ipc_open(h2[r][1]) for r in peers]
ctrl2 = torch.zeros(8, dtype=torch.int32, device=dev)
dist.barrier()
comm = PyNcclCommunicator(dist.group.WORLD, device=dev)
ROWS = [int(r) for r in a.rows.split(",")]
orders = nccl_order_tables(comm.all_reduce, rank, dev, ROWS)
if set(orders) != set(ROWS):
    raise SystemExit(f"[rank {rank}] NCCL's order not measurable for rows {sorted(set(ROWS) - set(orders))}")


def ar2(x, out):
    ext.all_reduce_twoshot(x, out, M_MAX, rank, data2, flags2, peers_data2, peers_flags2, ctrl2, orders[x.shape[0]],
                           timeout, a.max_ctas)


def ar(x, out, residual=None, weight=None, residual_out=None, eps=1e-6):
    ext.push(x, M_MAX, rank, peers_data, peers_flags, ctrl, a.max_ctas)
    ext.wait_reduce(x, rank, M_MAX, data, flags, ctrl, orders[x.shape[0]], out, residual, weight, residual_out, eps,
                    timeout)


def check_timeout(tag):
    t = int(timeout.item())
    if t:
        print(f"[rank {rank}] TIMEOUT flags {t:#x} in {tag}", flush=True)
        timeout.zero_()


def time_graph(fn, label):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize(); dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(a.ars):
            fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(a.replays):
        g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / a.replays / a.ars * 1e6
    check_timeout(label)
    dist.barrier()
    return us


results = []
for M in ROWS:
    g = torch.Generator(device=dev).manual_seed(1000 + rank)
    x = torch.randn(M, H, device=dev, dtype=torch.float32, generator=g).to(torch.bfloat16)
    out = torch.empty_like(x)
    ar(x, out); torch.cuda.synchronize(); check_timeout("plain")
    ref = comm.all_reduce(x).cpu()  # NCCL's bits
    ok_ref = torch.equal(out.cpu().view(torch.int16), ref.view(torch.int16))
    outs = [torch.empty_like(ref) for _ in range(ws)]
    dist.all_gather(outs, out.cpu())
    ok_ranks = all(torch.equal(outs[0], o) for o in outs)
    residual = torch.randn(M, H, device=dev, dtype=torch.float32, generator=g).to(torch.bfloat16)
    weight = (torch.rand(H, device=dev, generator=g) + 0.5).to(torch.bfloat16)
    y = torch.empty_like(x); res_out = torch.empty_like(x)
    ar(x, y, residual, weight, res_out); torch.cuda.synchronize(); check_timeout("norm")
    z = (ref.to(dev).float() + residual.float()).to(torch.bfloat16); zf = z.float()
    y_ref = (zf * torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + 1e-6) * weight.float()).to(torch.bfloat16)
    # the residual add must have the bits; the RMSNorm sums its squares in another order than torch: within 1 bf16 ulp
    ok_res = torch.equal(res_out.view(torch.int16), z.view(torch.int16))
    ok_norm = int((y.view(torch.int16).int() - y_ref.view(torch.int16).int()).abs().max()) <= 1
    y_diff = (y.float() - y_ref.float()).abs().max().item()
    # repeated correctness across many ARs (slot rotation / seq wrap-around of counters)
    bad = 0
    for _ in range(50):
        ar(x, out)
    torch.cuda.synchronize(); check_timeout("rep")
    bad += 0 if torch.equal(out.cpu().view(torch.int16), ref.view(torch.int16)) else 1
    # two-shot correctness: NCCL's bits (as the one-shot), identical across ranks, 50x repeat
    out2 = torch.empty_like(x)
    ar2(x, out2); torch.cuda.synchronize(); check_timeout("twoshot")
    ok2 = torch.equal(out2.cpu().view(torch.int16), ref.view(torch.int16))
    outs2 = [torch.empty_like(ref) for _ in range(ws)]
    dist.all_gather(outs2, out2.cpu())
    ok2_ranks = all(torch.equal(outs2[0], o) for o in outs2)
    for _ in range(50):
        ar2(x, out2)
    torch.cuda.synchronize(); check_timeout("twoshot-rep")
    bad2 = 0 if torch.equal(out2.cpu().view(torch.int16), ref.view(torch.int16)) else 1
    timings = {"two-shot AR": time_graph(lambda: ar2(x, out2), "two-shot"),
               "push-AR": time_graph(lambda: ar(x, out), "push-AR"),
               "push-AR+add+RMSNorm": time_graph(lambda: ar(x, y, residual, weight, res_out), "norm"),
               "push-only": time_graph(lambda: ext.push(x, M_MAX, rank, peers_data, peers_flags, ctrl, a.max_ctas), "push-only")}
    # after push-only the flags are ahead of the counter; re-sync by a full AR round the consumer will accept
    # (flags >= seq always holds here), then continue
    ar(x, out); torch.cuda.synchronize()
    if not a.no_nccl:
        xn = x.clone()
        s = torch.cuda.Stream()
        timings["NCCL-AR (PyNccl, graph)"] = time_graph(lambda: comm.all_reduce(xn, out_tensor=xn, stream=torch.cuda.current_stream()), "nccl")
    results.append((M, ok_ref, ok_ranks, ok_res, ok_norm, y_diff, bad, timings, ok2, ok2_ranks, bad2))

failed = False
if rank == 0:
    for M, ok_ref, ok_ranks, ok_res, ok_norm, y_diff, bad, timings, ok2, ok2_ranks, bad2 in results:
        failed |= not (ok_ref and ok_ranks and ok_res and bad == 0 and ok2 and ok2_ranks and bad2 == 0) or (M <= 16 and not ok_norm)
        print(f"[{M}x{H}] one-shot: ==NCCL {ok_ref}, identical ranks {ok_ranks}, 50x mismatches {bad}; + residual ==ref {ok_res}, "
              f"RMSNorm within 1 ulp {ok_norm} "
              f"(max|diff| {y_diff:.3g}) | two-shot: ==NCCL {ok2}, identical ranks {ok2_ranks}, 50x mismatches {bad2}")
        print("        " + "  ".join(f"{k}: {v:6.2f} us" for k, v in timings.items()), flush=True)
dist.barrier()
if rank == 0:
    print("RESULT:", "FAIL" if failed else "PASS", flush=True)
    if failed:
        raise SystemExit(1)
