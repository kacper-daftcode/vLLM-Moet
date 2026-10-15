#!/usr/bin/env python3
"""NCCL all-reduce time per protocol / algorithm / tuning knob on the 4 PCIe GPUs of the serving host, for the
DeepSeek-V4.1 TP4 message sizes: decode (6 x 5120 bf16, 60 KB), the CED-trimmed prefill rows (128 x 5120, 1.3 MB),
a quarter chunk (1024 x 5120) and a prefill chunk (4096 x 5120, 42 MB). Each configuration is one process group
(NCCL reads its environment at init), e.g. inside the serving image on GPUs 0-3:
    for cfg in "" NCCL_PROTO=LL NCCL_PROTO=Simple NCCL_ALGO=Tree NCCL_BUFFSIZE=16777216; do
      env $cfg NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node=4 nccl_proto_bench.py; done
Rank 0 prints: configuration, size, us per all-reduce (median of ITERS after 5 warm-ups), GB/s of payload. The
allocator is capped at 2 % of the GPU so it runs next to a serving job. Result on 4x RTX PRO 6000 (PCIe Gen5 x16, one
host bridge), 2026-10-15: the default equals Simple at 1,600 us for 42 MB (26 GB/s payload, 39 GB/s per link and
direction), LL 3,293, LL128 1,711, Tree 27,800; buffer size, thread count, channel count change nothing
(docs/dsv41-sm120-port.md, "Where a 163K-token prefill spends its 9 s")."""
import os, statistics, time
import torch, torch.distributed as dist

dist.init_process_group("nccl")
rank, ws = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
torch.cuda.set_per_process_memory_fraction(0.02)  # ~1.9 GB cap next to the serving job
cfg = {k: v for k, v in sorted(os.environ.items()) if k.startswith("NCCL_")}
sizes = [(6, 5120), (128, 5120), (1024, 5120), (4096, 5120)]
iters = int(os.environ.get("ITERS", "20"))
out = []
for shape in sizes:
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    med = statistics.median(ts)
    nbytes = x.numel() * 2
    out.append((shape, med, nbytes / med / 1e3))  # GB/s of payload
    dist.barrier()
if rank == 0:
    tag = " ".join(f"{k.replace('NCCL_', '')}={v}" for k, v in cfg.items()) or "defaults"
    for shape, med, gbps in out:
        print(f"{tag:40s} {shape[0]:5d}x{shape[1]} bf16 {shape[0]*shape[1]*2/1e6:7.2f} MB: {med:9.1f} us  ({gbps:5.1f} GB/s payload)")
dist.destroy_process_group()
