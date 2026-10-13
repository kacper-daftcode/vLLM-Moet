#!/usr/bin/env python3
"""What fuse_mhc_all_reduce does to an eager prefill step that trims (decoder SWA bounded replay), measured on the
TP group's all-reduce: the bits of a row subset's all-reduce against the same rows of the full all-reduce.

With fuse_mhc_all_reduce (patch_vllm_mhc_overlap_sm120.py, VLLM_MOET_MHC_FUSE_ALLREDUCE=1) the MoE of layer 20 (the
last KV source) returns its unreduced partial sum, DecoderReplayLayers selects the replay rows of it, and layer 21's
first boundary all-reduces those rows only; without it the MoE all-reduces every row and the replay selects reduced
rows. NCCL is deterministic for a given message, but its rounding depends on the message size and the rows' offset in
it. The launcher's batch shapes: decodes of 6 rows first, prefill chunks after them, window 128, <= 4096 tokens.
Control: the unfused path's own bits for the same prefill rows in batches of another composition. Last: eager
all-reduce times of [T, 5120] bf16.

Run in the image (or the bare vLLM base) on 4 GPUs, ~1.9 GB each, ~15 s:
    docker run --rm --gpus '"device=0,1,2,3"' --ipc host -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS \
        -v $PWD/nccl_replay_subset_tp4.py:/t.py:ro --entrypoint python3 <image> /t.py
"""

import os
import sys

import torch
import torch.multiprocessing as mp

H = 5120
W = 128


def span_rows(spans):
    return torch.cat([torch.arange(a, b) for a, b in spans])


# (label, tokens in the step, kept rows): decodes first, prefill chunks after them (the batch order)
CASES = [
    ("one 4096-token chunk", 4096, [(4096 - W, 4096)]),
    ("1 decode + 4090-token chunk", 4096, [(0, 6), (4096 - W, 4096)]),
    ("7 decodes + 4054-token chunk", 4096, [(0, 42), (4096 - W, 4096)]),
    ("two 2048-token prompts", 4096, [(2048 - W, 2048), (4096 - W, 4096)]),
    ("last chunk, 1000 tokens", 1000, [(1000 - W, 1000)]),
    ("300-token prompt", 300, [(300 - W, 300)]),
    ("200-token prompt", 200, [(200 - W, 200)]),
    ("129-token prompt", 129, [(1, 129)]),
]


def ulp_key(t: torch.Tensor) -> torch.Tensor:
    """bf16 bit patterns mapped to integers whose order is the float order (distance = ulps)."""
    u = t.contiguous().view(torch.int16).int() & 0xFFFF
    return torch.where(u >= 0x8000, 0xFFFF - u, u | 0x8000)


def compare(a: torch.Tensor, b: torch.Tensor) -> str:
    d = (ulp_key(a) - ulp_key(b)).abs()
    n = int((d != 0).sum())
    if n == 0:
        return "identical"
    rel = float(((a.float() - b.float()).abs().max() / a.float().abs().max()))
    return (f"{n} of {d.numel()} elements differ ({100 * n / d.numel():.2f} %; 1 ulp {int((d == 1).sum())}, "
            f"2 ulp {int((d == 2).sum())}, >2 ulp {int((d > 2).sum())}, max {int(d.max())} ulp; "
            f"max |diff| / max |x| {rel:.1e}); rows touched {int((d != 0).any(dim=1).sum())} of {d.shape[0]}")


def worker(rank, world, port, out):
    torch.cuda.set_device(rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        tp = get_tp_group()
        dc = tp.device_communicator
        lines = []
        if rank == 0:
            def state(comm):
                return "none" if comm is None else ("disabled" if getattr(comm, "disabled", False) else "on")

            lines.append(f"communicator: custom AR {state(getattr(dc, 'ca_comm', None))}, "
                         f"symm mem {state(getattr(dc, 'symm_mem_comm', None))}, "
                         f"pynccl {state(getattr(dc, 'pynccl_comm', None))}")

        def partials(n, seed):
            # a rank's partial sum of an FFN output: shared + routed experts, |x| ~ 0.1-1
            gen = torch.Generator(device="cuda").manual_seed(seed + rank)
            return (torch.randn(n, H, device="cuda", generator=gen) * 0.3).to(torch.bfloat16)

        lines.append("-- fused path (subset all-reduce) vs reference (full all-reduce, then the rows):")
        for label, T, spans in CASES:
            x = partials(T, 1000 * T + 17 * len(spans))
            rows = span_rows(spans).cuda()
            full = tp.all_reduce(x.clone())
            again = tp.all_reduce(x.clone())
            sub = tp.all_reduce(x.index_select(0, rows))
            ref = full.index_select(0, rows)
            torch.cuda.synchronize()
            lines.append(f"{label:30s} T={T:4d} R={rows.numel():3d}: repeat {'identical' if torch.equal(full, again) else 'DIFFERS'}; "
                         f"subset vs full: {compare(ref, sub)}")
            del x, full, again, sub, ref

        lines.append("-- control: the reference on the same prefill rows in batches of another composition:")
        P = 4096 - 42
        prefill = partials(P, 7)
        dec42, dec6 = partials(42, 11), partials(6, 13)
        alone = tp.all_reduce(prefill.clone())
        with7 = tp.all_reduce(torch.cat([dec42, prefill]))[42:]
        with1 = tp.all_reduce(torch.cat([dec6, prefill]))[6:]
        torch.cuda.synchronize()
        lines.append(f"{P}-token chunk alone vs after 7 decodes (all rows): {compare(alone, with7)}")
        lines.append(f"{P}-token chunk after 1 decode vs after 7 decodes (all rows): {compare(with1, with7)}")
        last = slice(P - W, P)
        lines.append(f"  the same, its last {W} rows only: {compare(with1[last], with7[last])}")
        del prefill, alone, with7, with1

        lines.append("-- eager all-reduce of [T, 5120] bf16, median of 20 (CUDA events, ranks lined up by a barrier):")
        for T in (4096, 2048, 1000, 256, 170, 128):
            x = partials(T, 3)
            for _ in range(3):
                tp.all_reduce(x)
            times = []
            for _ in range(20):
                torch.cuda.synchronize()
                torch.distributed.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                tp.all_reduce(x)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            times.sort()
            lines.append(f"  T={T:4d} ({T * H * 2 / 2**20:5.1f} MiB): {times[len(times) // 2] * 1000:8.1f} us")
            del x
        if rank == 0:
            out.put(lines)
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main() -> int:
    world = 4
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + os.getpid() % 1000
    procs = [ctx.Process(target=worker, args=(r, world, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    lines = q.get(timeout=600)
    for p in procs:
        p.join(timeout=120)
    print("\n".join(lines))
    return 0 if all(p.exitcode == 0 for p in procs) else 1


if __name__ == "__main__":
    sys.exit(main())
