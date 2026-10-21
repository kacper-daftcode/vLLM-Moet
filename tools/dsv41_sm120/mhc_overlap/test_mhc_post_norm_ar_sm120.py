#!/usr/bin/env python3
"""Fused all-reduce consumer in mhc_post_norm (step 24) vs the two-kernel path (one-shot P2P all-reduce, then the plain
boundary kernel): every output bit-identical (residual_out, layer_input, aux, reduced x), identical across ranks, 50
repeats; timings from a CUDA graph of 77 boundaries: NCCL AR + post_norm, P2P one-shot AR + post_norm, push + fused.
usage: torchrun --nproc_per_node=4 test_mhc_post_norm_ar_sm120.py [--rows 6,12] [--iters 77] [--replays 100]
"""
import argparse, os, sys, time
import torch, torch.distributed as dist
from torch.utils.cpp_extension import load

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="6,12")
ap.add_argument("--iters", type=int, default=77)
ap.add_argument("--replays", type=int, default=100)
ap.add_argument("--p2p-dir", default=os.environ.get("P2P_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "p2p_ar")))
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); sys.path.insert(0, a.p2p_dir)

dist.init_process_group("gloo")
rank, ws = dist.get_rank(), dist.get_world_size(); assert ws == 4
torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
H, HC, M_MAX, NSLOTS = 5120, 4, 64, 4
p2p = load(name="p2p_allreduce_sm120", sources=[os.path.join(a.p2p_dir, "p2p_allreduce_sm120.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=False)
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
mhc = load(name="vllm_moet_mhc_post_norm_sm120_ar", sources=[os.path.join(here, "mhc_post_norm_sm120.cu")],
           extra_cuda_cflags=["-O3", "-std=c++20"], extra_cflags=["-O3", "-std=c++20"], verbose=False)

def shared_set():
    data = p2p.alloc_shared(NSLOTS * 4 * M_MAX * H * 2); flags = p2p.alloc_shared(NSLOTS * 4 * 32 * 4)
    hs = [None] * ws; dist.all_gather_object(hs, (p2p.ipc_handle(data), p2p.ipc_handle(flags)))
    peers = [r for r in range(ws) if r != rank]
    return dict(data=data, flags=flags, pd=[p2p.ipc_open(hs[r][0]) for r in peers], pf=[p2p.ipc_open(hs[r][1]) for r in peers],
                ctrl=torch.zeros(8, dtype=torch.int32, device=dev))
A_ = shared_set()   # for the two-kernel reference
B_ = shared_set()   # for the fused path
timeout = torch.zeros(1, dtype=torch.int32, device=dev)
dist.barrier()
EPS = 1e-6

def inputs(T, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(T, H, device=dev, dtype=torch.float32, generator=g).to(torch.bfloat16)        # this rank's partial
    g2 = torch.Generator(device=dev).manual_seed(777)                                             # identical on all ranks
    residual = torch.randn(T, HC, H, device=dev, dtype=torch.float32, generator=g2).to(torch.bfloat16)
    post = torch.rand(T, HC, device=dev, generator=g2) * 0.5 + 0.5
    comb = torch.rand(T, HC, HC, device=dev, generator=g2) * 0.4 + 0.1
    pre = torch.rand(T, HC, device=dev, generator=g2) * 0.5 + 0.25
    w = (torch.rand(H, device=dev, generator=g2) + 0.5).to(torch.bfloat16)
    return x, residual, post, comb, pre, w

def ref_path(x, residual, post, comb, pre, w, S, out):  # one-shot AR kernel, then the plain boundary kernel
    xr = torch.empty_like(x)
    p2p.push(x, M_MAX, rank, S["pd"], S["pf"], S["ctrl"], 64)
    p2p.wait_reduce(x, rank, M_MAX, S["data"], S["flags"], S["ctrl"], xr, None, None, None, 0.0, timeout)
    ro, li, aux = out
    mhc.mhc_post_norm(xr, residual, post, comb, pre, w, ro, li, aux, EPS, 1, True)
    return xr

def fused_path(x, residual, post, comb, pre, w, S, out, xo):
    p2p.push(x, M_MAX, rank, S["pd"], S["pf"], S["ctrl"], 64)
    ro, li, aux = out
    mhc.mhc_post_norm(x, residual, post, comb, pre, w, ro, li, aux, EPS, 1, True, S["data"], S["flags"], S["ctrl"], rank, M_MAX, xo)

def time_graph(fn, n):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.synchronize(); dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(n): fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(a.replays): g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / a.replays / n * 1e6
    dist.barrier(); return us

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
comm = PyNcclCommunicator(dist.group.WORLD, device=dev)
failed = False
for T in [int(r) for r in a.rows.split(",")]:
    x, residual, post, comb, pre, w = inputs(T, 1000 + rank)
    mk = lambda: (torch.empty_like(residual), torch.empty(T, H, dtype=torch.bfloat16, device=dev), torch.empty(T, H, dtype=torch.bfloat16, device=dev))
    outA, outB = mk(), mk()
    xr = ref_path(x, residual, post, comb, pre, w, A_, outA)
    xo = torch.empty_like(x)
    fused_path(x, residual, post, comb, pre, w, B_, outB, xo)
    torch.cuda.synchronize()
    same = [torch.equal(p, q) for p, q in zip(outA, outB)] + [torch.equal(xr, xo)]
    # identical across ranks
    flat = torch.cat([t.flatten() for t in outB] + [xo.flatten()]).cpu()
    gathered = [torch.empty_like(flat) for _ in range(ws)]; dist.all_gather(gathered, flat)
    across = all(torch.equal(gathered[0], g_) for g_ in gathered)
    # 50 repeats (slot rotation), fused vs ref again
    for _ in range(50):
        ref_path(x, residual, post, comb, pre, w, A_, outA); fused_path(x, residual, post, comb, pre, w, B_, outB, xo)
    torch.cuda.synchronize()
    same50 = all(torch.equal(p, q) for p, q in zip(outA, outB))
    to = int(timeout.item()); timeout.zero_()
    ok = all(same) and across and same50 and to == 0
    failed |= not ok
    # timings
    xn = x.clone()
    def nccl_path():
        comm.all_reduce(xn, out_tensor=xn, stream=torch.cuda.current_stream())
        ro, li, aux = outA
        mhc.mhc_post_norm(xn, residual, post, comb, pre, w, ro, li, aux, EPS, 1, True)
    t_nccl = time_graph(nccl_path, a.iters)
    t_ref = time_graph(lambda: ref_path(x, residual, post, comb, pre, w, A_, outA), a.iters)
    t_fused = time_graph(lambda: fused_path(x, residual, post, comb, pre, w, B_, outB, xo), a.iters)
    if rank == 0:
        print(f"[T={T}] fused == ref (residual, layer_input, aux, x): {same}; identical across ranks {across}; 50x {same50}; "
              f"timeouts {to}\n       per boundary: NCCL AR + post_norm {t_nccl:6.2f} us | P2P one-shot AR + post_norm {t_ref:6.2f} us | "
              f"push + fused post_norm {t_fused:6.2f} us", flush=True)
dist.barrier()
if rank == 0:
    print("RESULT:", "FAIL" if failed else "PASS", flush=True)
    if failed: raise SystemExit(1)
