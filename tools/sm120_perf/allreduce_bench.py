#!/usr/bin/env python3
"""4-rank all-reduce latency on PCIe-only GPUs: NCCL (pynccl, P2P) vs vLLM CustomAllreduce
one-shot with the "fully connected" gate forced open.

    torchrun --nproc_per_node=4 car_bench.py [--iters 100] [--sizes 2560,20480,...]

Numbers are per call; "graph" rows replay ITERS captured calls per replay (the in-situ
usage pattern of both servers), "eager" rows include the memcpy into the IPC buffer
(custom AR) or a plain ncclAllReduce (pynccl).
"""
import argparse
import os
import time

import torch
import torch.distributed as dist


def timeit(fn, iters, reps=3):
    best = None
    for _ in range(reps):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        fn(iters)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1e6
        best = dt if best is None else min(best, dt)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument(
        "--sizes",
        default="2560,5120,10240,20480,30720,40960,61440,81920,163840,327680,1310720",
        help="element counts (bf16)",
    )
    ap.add_argument("--max-size", type=int, default=512 * 1024, help="custom AR max bytes")
    args = ap.parse_args()

    dist.init_process_group("gloo")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    gloo_group = dist.group.WORLD

    from vllm.platforms import current_platform

    fc_orig = current_platform.is_fully_connected
    type(current_platform).is_fully_connected = classmethod(lambda cls, ids: True)
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    ca = CustomAllreduce(group=gloo_group, device=dev, max_size=args.max_size)
    assert not ca.disabled, "custom allreduce disabled"
    nccl = PyNcclCommunicator(group=gloo_group, device=dev)
    assert not nccl.disabled

    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []
    for n in sizes:
        nbytes = n * 2
        # values whose 4-rank sums stay exact in bf16 (< 256), so ok=False means a real error
        x = (torch.arange(n, device=dev, dtype=torch.float32) % 50 + rank).to(torch.bfloat16)
        ref = nccl.all_reduce(x.clone())
        torch.cuda.synchronize()

        # --- NCCL eager / graph ---
        def nccl_eager(iters):
            for _ in range(iters):
                nccl.all_reduce(x)

        t_nccl_eager = timeit(nccl_eager, args.iters)

        g_nccl = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                nccl.all_reduce(x)
        torch.cuda.synchronize()
        with torch.cuda.graph(g_nccl, stream=s):
            for _ in range(args.iters):
                y_nccl = nccl.all_reduce(x)

        def nccl_graph(iters):
            g_nccl.replay()

        t_nccl_graph = timeit(nccl_graph, args.iters)
        torch.cuda.synchronize()
        ok_nccl_graph = torch.equal(y_nccl, ref)

        # --- custom AR eager / graph ---
        use_ca = ca.should_custom_ar(x)
        t_ca_eager = t_ca_graph = float("nan")
        ok_ca_eager = ok_ca_graph = None
        if use_ca:
            y = ca.custom_all_reduce(x)
            torch.cuda.synchronize()
            ok_ca_eager = torch.equal(y, ref)

            def ca_eager(iters):
                for _ in range(iters):
                    ca.custom_all_reduce(x)

            t_ca_eager = timeit(ca_eager, args.iters)

            g_ca = torch.cuda.CUDAGraph()
            with ca.capture():
                with torch.cuda.graph(g_ca, stream=s):
                    for _ in range(args.iters):
                        y_ca = ca.custom_all_reduce(x)
            torch.cuda.synchronize()
            dist.barrier()
            g_ca.replay()
            torch.cuda.synchronize()
            ok_ca_graph = torch.equal(y_ca, ref)

            def ca_graph(iters):
                g_ca.replay()

            t_ca_graph = timeit(ca_graph, args.iters)
            del g_ca
        del g_nccl
        rows.append((nbytes, t_nccl_eager, t_nccl_graph, ok_nccl_graph, use_ca, t_ca_eager, t_ca_graph, ok_ca_eager, ok_ca_graph))

    # gather per-rank timings, report rank 0 and the max over ranks (skew-free lockstep)
    all_rows = [None] * ws
    dist.all_gather_object(all_rows, rows)
    if rank == 0:
        print(
            f"world={ws} iters={args.iters} NCCL_P2P_LEVEL={os.environ.get('NCCL_P2P_LEVEL')} "
            f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')} custom max_size={args.max_size}"
        )
        print(f"{'bytes':>9} | {'nccl eager':>10} {'nccl graph':>10} | {'CA eager':>9} {'CA graph':>9} | ok(nccl-g, ca-e, ca-g) | max-over-ranks CA graph")
        for i, r in enumerate(rows):
            nbytes, te, tg, okg, use_ca, ce, cg, oke, okc = r
            cg_max = max(rr[i][6] for rr in all_rows)
            print(
                f"{nbytes/1024:7.1f}KiB | {te:10.1f} {tg:10.1f} | {ce:9.1f} {cg:9.1f} | {okg!s:>6} {oke!s:>6} {okc!s:>6} | {cg_max:8.1f}"
            )
    type(current_platform).is_fully_connected = fc_orig
    ca.close() if hasattr(ca, "close") else None
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
