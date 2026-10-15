#!/usr/bin/env python3
"""The prefill DBO comm region on 4 real ranks (NCCL), without the model.

Two microbatch threads under vLLM's own `UBatchContext`s (the protocol the patched V2 runner drives) run a stand-in
layer stack - a bf16 GEMM, the TP all-reduce of its output through `moet_prefill_dbo.all_reduce`, an all-gather
through `moet_prefill_dbo.all_gather`, the shared row buffer written and read across the yields, a replay-style
forward-context swap with a collective inside - and the result of every microbatch is compared bit for bit with the
same microbatch run alone on one stream (the same NCCL calls, so the same bits). The full batch's single all-reduce
is also computed, to show that halving changes bits (the ring chunks differ): that is why the serving gate is a
quality window. Then the wall time of L layers: one batch against the two microbatches.

Needs the patched tree (patch_vllm_prefill_dbo_sm120.py). usage, inside the serving image on GPUs 0-3:
    python3 patch_vllm_prefill_dbo_sm120.py && torchrun --nproc_per_node=4 prefill_dbo/test_prefill_dbo_tp4.py
        [--half-tokens 2048] [--layers 8] [--iters 5]
The allocator is capped at 3 % of the GPU so it runs next to a serving job."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time

import torch
import torch.distributed as dist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-tokens", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()
    for key, value in (
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29553"),
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
    ):
        os.environ.setdefault(key, value)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.cuda.set_per_process_memory_fraction(0.03)
    dev = torch.device("cuda", torch.cuda.current_device())

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group, init_distributed_environment, initialize_model_parallel
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.utils.torch_utils import current_stream
    from vllm.v1.worker import moet_prefill_dbo as dbo
    from vllm.v1.worker import ubatching as ub

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    vllm_config = VllmConfig()
    failures: list[str] = []
    with set_current_vllm_config(vllm_config), torch.device(dev):
        init_distributed_environment(
            world_size=world, rank=rank, distributed_init_method="env://", local_rank=rank, backend="nccl"
        )
        initialize_model_parallel(tensor_model_parallel_size=world)
        tp = get_tp_group()
        cpu_group = tp.cpu_group

        def all_ok(flag: bool) -> bool:
            t = torch.tensor([0 if flag else 1])
            if world > 1:
                dist.all_reduce(t, group=cpu_group)
            return t.item() == 0

        H, D = args.half_tokens, args.hidden
        T = 2 * H
        L = args.layers
        log(f"{torch.cuda.get_device_name(dev)} x {world}; H={H} tokens, D={D}, {L} layers")
        g = torch.Generator(device=dev).manual_seed(11 + rank)  # different per rank, like TP partial sums
        W = (torch.randn(D, D, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        x_full = torch.randn(T, D, device=dev, generator=g).to(torch.bfloat16)
        halves = [slice(0, H), slice(H, T)]
        row_buf = torch.zeros(T, 4, dtype=torch.int32, device=dev)
        switch = dbo.RowBufferSwitch([row_buf], 2)
        replay_context = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})

        def stack(x: torch.Tensor, tag: int, layers: int, check_rows: bool = True) -> torch.Tensor:
            """A stand-in layer stack on rows x: GEMM -> all-reduce -> all-gather; the row buffer written
            before and checked after the collectives (each of which yields on a microbatch thread)."""
            n = x.shape[0]
            for layer in range(layers):
                y = torch.matmul(x, W)
                if check_rows:
                    row_buf[:n].fill_(tag * 1000 + layer)
                if layer == 1 and dbo.enabled():
                    # The decoder replay layers run under another forward context: installed on the
                    # microbatch context too, so the yield inside the collective accepts it.
                    ctx = dbo.context()
                    saved = ctx.forward_context
                    ctx.forward_context = replay_context
                    try:
                        with override_forward_context(replay_context):
                            y = dbo.all_reduce(y)
                    finally:
                        ctx.forward_context = saved
                else:
                    y = dbo.all_reduce(y)
                gathered = dbo.all_gather(x[:, :64].contiguous(), 1)  # [n, 64 * world]
                if check_rows and not bool((row_buf[:n] == tag * 1000 + layer).all()):
                    raise AssertionError(f"row buffer clobbered in layer {layer} (tag {tag})")
                x = y + gathered[:, :1].to(y.dtype)
            return x

        # 1. Each microbatch alone, one stream: the reference bits.
        ref = [stack(x_full[sl].clone(), i, L) for i, sl in enumerate(halves)]
        full = stack(x_full.clone(), 7, L)
        torch.cuda.synchronize()

        def run_dbo(check_rows: bool = True) -> list[torch.Tensor]:
            compute_stream = current_stream()
            comm_stream = torch.cuda.Stream(device=dev, priority=dbo.comm_stream_priority())
            contexts = [
                ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={}) for _ in range(2)
            ]
            barrier = threading.Barrier(3)
            ctxs = ub.make_ubatch_contexts(
                2, compute_stream, comm_stream, contexts, barrier, token_offsets=[0, H]
            )
            outs: dict[int, torch.Tensor] = {}
            errs: dict[int, BaseException] = {}

            def worker(ctx, i: int) -> None:
                try:
                    torch.accelerator.set_device_index(dev.index)
                    with ctx:
                        assert dbo.enabled() and dbo.ubatch_id() == i
                        assert dbo.token_offset() == halves[i].start, dbo.token_offset()
                        assert switch.current == i, (switch.current, i)
                        outs[i] = stack(x_full[halves[i]].clone(), i, L, check_rows)
                except BaseException as e:  # noqa: BLE001
                    errs[i] = e

            threads = [threading.Thread(target=worker, args=(ctx, i)) for i, ctx in enumerate(ctxs)]
            for t in threads:
                t.start()
            barrier.wait()
            ctxs[0].cpu_wait_event.set()
            for t in threads:
                t.join()
            switch.reset()
            if errs:
                i = min(errs)
                raise RuntimeError(f"microbatch {i} failed") from errs[i]
            torch.cuda.synchronize()
            return [outs[0], outs[1]]

        # 2. The two microbatches under the ubatch protocol: the same bits as alone.
        for it in range(3):
            outs = run_dbo()
            for i in range(2):
                same = torch.equal(outs[i], ref[i])
                if not same:
                    diff = (outs[i].float() - ref[i].float()).abs().max().item()
                    failures.append(f"iteration {it}: microbatch {i} differs from its single-stream run (max {diff})")
        same_as_full = all(torch.equal(torch.cat(ref), full) for _ in (0,))
        max_diff_full = (torch.cat(ref).float() - full.float()).abs().max().item()
        log(
            f"microbatches under the ubatch protocol: bit-identical to their single-stream runs: "
            f"{'yes' if not failures else 'NO'}; against the one-batch all-reduce: "
            f"{'identical' if same_as_full else f'max |diff| {max_diff_full:.3g} (ring chunking)'}"
        )
        if not all_ok(not failures):
            log("FAILED:\n  " + "\n  ".join(failures))
            return 1

        # 3. Wall time: one batch vs two microbatches (the threads' CPU handoffs included).
        def timed(fn) -> float:
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            if world > 1:
                dist.barrier(group=cpu_group)
            ts = []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                if world > 1:
                    dist.barrier(group=cpu_group)
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(ts)

        t_full = timed(lambda: stack(x_full, 7, L, check_rows=False))
        t_dbo = timed(lambda: run_dbo(check_rows=False))
        log(
            f"{L} layers x (GEMM [{T}x{D}x{D}] + all-reduce [{T}x{D}] bf16 + all-gather): one batch "
            f"{t_full:.2f} ms, two microbatches {t_dbo:.2f} ms ({(t_dbo / t_full - 1) * 100:+.1f} %)"
        )
        log("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
