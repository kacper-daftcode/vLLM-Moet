#!/usr/bin/env python3
"""vLLM's NCCL usage pattern with graph mixing off (NCCL_GRAPH_MIXING_SUPPORT=0), on 4 real TP ranks.

NCCL's documentation does not support, with graph mixing off, "launching a non-captured NCCL collective during an
outstanding graph launch that uses the same communicator, regardless of stream ordering" - which is what vLLM does
every step (the decode graph is replayed and the host goes on to eager collectives on the same communicator while
the graph still runs). In NCCL 2.30.7 the rule protects the host-side ordering of proxy operations (network / CE
transports post them from the host at launch time, regardless of stream order) and the device-side serialization of
NCCL kernels issued on different streams; neither applies here: the P2P/CUMEM transport has no proxy progress
function (no proxy ops), and vLLM issues every collective of the TP group from one thread on one stream.
This test runs exactly that pattern through vLLM's own collectives (tensor_model_parallel_all_reduce /
tensor_model_parallel_all_gather -> pynccl):

  per iteration: refill the inputs, replay a decode-like graph (40 all-reduces [6, 5120] + 2 all-gathers [6, 6400]),
  then without any host synchronisation eager collectives on the same stream (an all-reduce [6, 5120], an
  all-gather [6, 6400], a prefill-sized all-reduce [2048, 5120], a one-element all-reduce), replay a drafter-like
  graph (7 all-reduces), another eager all-reduce; the host synchronises only every 10 iterations.

Inputs are small integers (sums over 4 ranks exact in bf16), so every output is checked against its exact value;
a second pass with wide-range random inputs checksums every output (int64 sums over the bit patterns) so two runs
(mixing on / off) can be compared bit for bit. A watchdog fails the test if anything hangs.

usage (serving image, 4 GPUs):
    torchrun --nproc_per_node=4 test_nccl_graph_mixing_tp4.py [--iters 400] [--hash-out FILE]
    run once with NCCL_GRAPH_MIXING_SUPPORT=0 and once without; the printed hashes must be equal.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time

import torch
import torch.distributed as dist

T, H, NS, PREFILL = 6, 5120, 6400, 2048
L_DECODE, L_DRAFT = 40, 7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--sync-every", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--hash-out", default="")
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda", torch.cuda.current_device())

    def watchdog():
        time.sleep(args.timeout)
        print(f"rank {rank}: TIMEOUT after {args.timeout:.0f} s (hang?)", flush=True)
        os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    with set_current_vllm_config(VllmConfig()), torch.device(dev):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        cpu_group = get_tp_group().cpu_group
        pyn = getattr(get_tp_group().device_communicator, "pynccl_comm", None)
        log(f"{torch.cuda.get_device_name(dev)} x {world}, NCCL {pyn.nccl.ncclGetVersion() if pyn else '-'}, "
            f"NCCL_GRAPH_MIXING_SUPPORT={os.environ.get('NCCL_GRAPH_MIXING_SUPPORT', '(unset = 1)')}")

        bf = torch.bfloat16
        dec_in = torch.zeros(L_DECODE, T, H, dtype=bf)
        ag_in = torch.zeros(2, T, NS, dtype=bf)
        dft_in = torch.zeros(L_DRAFT, T, H, dtype=bf)

        def decode_body():
            outs = [tensor_model_parallel_all_reduce(dec_in[i] * 1) for i in range(L_DECODE)]
            gs = [tensor_model_parallel_all_gather(ag_in[j] * 1, dim=-1) for j in range(2)]
            return torch.stack(outs), torch.stack(gs)

        def draft_body():
            return torch.stack([tensor_model_parallel_all_reduce(dft_in[i] * 1) for i in range(L_DRAFT)])

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            decode_body(), draft_body()
            stream.synchronize()
            g_dec, g_dft = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_dec, stream=stream):
                dec_out, ag_out = decode_body()
            with torch.cuda.graph(g_dft, stream=stream):
                dft_out = draft_body()
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)

        def inputs(it: int, salt: int, shape, exact: bool, r: int) -> torch.Tensor:
            g = torch.Generator(device=dev).manual_seed(((it * 131 + salt) * 7 + r) * 1009 + (0 if exact else 5))
            if exact:
                return torch.randint(-16, 17, shape, generator=g, device=dev).to(bf)
            x = torch.randn(shape, generator=g, device=dev)
            return (x * torch.exp2(torch.randint(-8, 9, shape, generator=g, device=dev).float())).to(bf)

        def reduced(it, salt, shape):  # the exact all-reduce result
            return sum(inputs(it, salt, shape, True, r).float() for r in range(world)).to(bf)

        def gathered(it, salt, shape):  # the all-gather result along the last dim, one per leading index
            per_rank = [inputs(it, salt, shape, True, r) for r in range(world)]
            return torch.cat(per_rank, dim=-1)

        weights = {}

        def checksum(t: torch.Tensor) -> torch.Tensor:
            v = t.contiguous().view(torch.int16).reshape(-1).to(torch.int64)
            w = weights.get(v.numel())
            if w is None:
                w = weights[v.numel()] = (torch.arange(v.numel(), device=dev, dtype=torch.int64) * 2654435761) % 2**31 + 1
            return torch.stack([(v * w).sum(), (v * v).sum()])

        ok = True
        for exact in (True, False):
            sums, pending, bad = [], [], 0
            t0 = time.time()
            with torch.cuda.stream(stream):
                for it in range(args.iters):
                    dec_in.copy_(inputs(it, 1, dec_in.shape, exact, rank))
                    ag_in.copy_(inputs(it, 2, ag_in.shape, exact, rank))
                    g_dec.replay()
                    outs = [dec_out.clone(), ag_out.clone()]
                    # non-captured collectives while the decode graph is outstanding, same stream, no host sync
                    outs.append(tensor_model_parallel_all_reduce(inputs(it, 3, (T, H), exact, rank)))
                    outs.append(tensor_model_parallel_all_gather(inputs(it, 4, (T, NS), exact, rank), dim=-1))
                    outs.append(tensor_model_parallel_all_reduce(inputs(it, 5, (PREFILL, H), exact, rank)))
                    outs.append(tensor_model_parallel_all_reduce(inputs(it, 6, (1,), exact, rank)))
                    dft_in.copy_(inputs(it, 7, dft_in.shape, exact, rank))
                    g_dft.replay()
                    outs.append(dft_out.clone())
                    outs.append(tensor_model_parallel_all_reduce(inputs(it, 8, (T, H), exact, rank)))
                    if exact:
                        pending.append((it, outs))
                    else:
                        sums.append(torch.stack([checksum(t) for t in outs]))
                    if (it + 1) % args.sync_every == 0 or it + 1 == args.iters:
                        stream.synchronize()
                        for pit, o in pending:
                            want = [reduced(pit, 1, dec_in.shape),
                                    gathered(pit, 2, ag_in.shape),
                                    reduced(pit, 3, (T, H)), gathered(pit, 4, (T, NS)), reduced(pit, 5, (PREFILL, H)),
                                    reduced(pit, 6, (1,)), reduced(pit, 7, dft_in.shape), reduced(pit, 8, (T, H))]
                            bad += sum(int(not torch.equal(g, w)) for g, w in zip(o, want))
                        pending.clear()
            digest = hashlib.sha256(torch.stack(sums).cpu().numpy().tobytes()) if sums else None
            elapsed = time.time() - t0
            n_bad = torch.tensor([bad])
            dist.all_reduce(n_bad, group=cpu_group)
            if exact:
                log(f"exact-integer pass: {args.iters} iterations ({elapsed:.1f} s; graph replays interleaved with "
                    f"eager collectives, host sync every {args.sync_every}): {n_bad.item()} wrong outputs")
                ok &= n_bad.item() == 0
            else:
                hashes = [None] * world
                dist.all_gather_object(hashes, digest.hexdigest(), group=cpu_group)
                log(f"random pass ({elapsed:.1f} s): output hash per rank " + " ".join(h[:16] for h in hashes))
                if rank == 0 and args.hash_out:
                    with open(args.hash_out, "w") as f:
                        f.write("\n".join(hashes) + "\n")
    ok_t = torch.tensor([0 if ok else 1])
    dist.all_reduce(ok_t, group=cpu_group)
    log("ALL OK" if ok_t.item() == 0 else "FAILED")
    return 0 if ok_t.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
