#!/usr/bin/env python3
"""What a sequence-parallel mHC boundary costs on the ring: one all-reduce of a sublayer output vs a
reduce-scatter of it plus an all-gather of the next sublayer's input (the same bytes, two collectives).

The prefill sublayer output is [T, 5120] bf16 per rank (T = 4096 per chunk); the sequence-parallel boundary
reduces it to the rank's T/4 rows, runs the mHC post / pre / norm there, and gathers the next sublayer's
[T, 5120] input. The gathered input may also travel as MXFP8 (e4m3 + one UE8M0 byte per 32 values, 1.03 B per
element instead of 2) when its consumers quantize it anyway. Times are medians of CUDA-event-timed eager calls
(NCCL through torch.distributed, the library vLLM's pynccl wraps).

  torchrun --nproc_per_node=4 nccl_rs_ag_bench.py [--tokens 512,1024,2048,4096,8192] [--iters 30]
"""
from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist


def timed(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    t = torch.tensor([statistics.median(times)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="512,1024,2048,4096,8192")
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    h = args.hidden
    if rank == 0:
        print(f"world {world}, NCCL {torch.cuda.nccl.version()}, hidden {h}; us, max over ranks of the per-rank median")
        print(f"{'T':>6} {'AR bf16':>9} {'RS bf16':>9} {'AG bf16':>9} {'RS+AG':>9} {'AG mxfp8':>9} {'RS+AGfp8':>9}")
    for t in (int(x) for x in args.tokens.split(",")):
        assert t % world == 0
        x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
        shard = torch.empty(t // world, h, device="cuda", dtype=torch.bfloat16)
        full = torch.empty(t, h, device="cuda", dtype=torch.bfloat16)
        q_shard = torch.empty(t // world, h + h // 32, device="cuda", dtype=torch.uint8)
        q_full = torch.empty(t, h + h // 32, device="cuda", dtype=torch.uint8)
        ar = timed(lambda: dist.all_reduce(x), args.iters)
        rs = timed(lambda: dist.reduce_scatter_tensor(shard, x), args.iters)
        ag = timed(lambda: dist.all_gather_into_tensor(full, shard), args.iters)
        ag8 = timed(lambda: dist.all_gather_into_tensor(q_full, q_shard), args.iters)

        def rs_ag():
            dist.reduce_scatter_tensor(shard, x)
            dist.all_gather_into_tensor(full, shard)

        def rs_ag8():
            dist.reduce_scatter_tensor(shard, x)
            dist.all_gather_into_tensor(q_full, q_shard)

        both = timed(rs_ag, args.iters)
        both8 = timed(rs_ag8, args.iters)
        if rank == 0:
            print(f"{t:>6} {ar:9.1f} {rs:9.1f} {ag:9.1f} {both:9.1f} {ag8:9.1f} {both8:9.1f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
