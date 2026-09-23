#!/usr/bin/env python3
"""4-rank all-reduce latency on PCIe-only GPUs: NCCL (pynccl, P2P) vs vLLM CustomAllreduce
one-shot with the "fully connected" gate forced open vs FlashInfer's PCIe CUDA-IPC all-reduce
(push-based; `PcieIpcAllReduceWorkspace`, FlashInfer >= 0.7.0 nightlies; vLLM enables it with
VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 for decode-shaped [tokens, hidden] tensors).

    torchrun --nproc_per_node=4 allreduce_bench.py [--iters 100] [--sizes 2560,20480,...] [--hidden 5120]

Numbers are per call; "graph" rows replay ITERS captured calls per replay (the in-situ
usage pattern of both servers), "eager" rows include the memcpy into the IPC buffer
(custom AR) or a plain ncclAllReduce (pynccl). The FlashInfer column is measured only for
sizes that are whole [batch, --hidden] tensors (its workspace is tuned per batch and
sized to the largest one, as vLLM sizes it to the largest CUDA-graph capture size); tuning
runs once per process (--fi-tune-cache persists it like vLLM's autotune dir does).
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
    ap.add_argument("--hidden", type=int, default=5120, help="hidden size for the FlashInfer PCIe IPC column")
    ap.add_argument("--skip-fi", action="store_true", help="skip the FlashInfer PCIe IPC column")
    ap.add_argument("--fi-tune-cache", default="", help="tune cache file for the FlashInfer workspace (persisted)")
    ap.add_argument("--fi-no-tune", action="store_true", help="use FlashInfer's seed launch configs (no tune())")
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

    # --- FlashInfer PCIe IPC workspace: one per process, sized to the largest [batch, hidden] ---
    fi_ws = None
    fi_batches = sorted({n // args.hidden for n in sizes if n % args.hidden == 0 and n // args.hidden > 0})
    if not args.skip_fi and fi_batches:
        try:
            import flashinfer.comm as fi_comm

            if not hasattr(fi_comm, "PcieIpcAllReduceWorkspace"):
                raise RuntimeError("this FlashInfer has no PcieIpcAllReduceWorkspace")
            t0 = time.perf_counter()
            fi_ws = fi_comm.PcieIpcAllReduceWorkspace(
                group=gloo_group,
                max_numel=fi_batches[-1] * args.hidden,
                dtype=torch.bfloat16,
                tune_batches=fi_batches,
                tune_cache=args.fi_tune_cache or None,
            )
            torch.cuda.synchronize()
            fi_ws.rebind_stream()
            if not args.fi_no_tune:
                fi_ws.tune([args.hidden], dtype=torch.bfloat16, tune_group=gloo_group)
            fi_ws.prepare([(b, args.hidden) for b in fi_batches], dtype=torch.bfloat16)
            torch.cuda.synchronize()
            fi_ws.rebind_stream()
            if rank == 0:
                print(
                    f"flashinfer PCIe IPC workspace: profile={fi_ws.profile!r} memop={fi_ws.memop_supported} "
                    f"batches={fi_batches} hidden={args.hidden} setup {time.perf_counter()-t0:.1f}s "
                    f"(tune={'off' if args.fi_no_tune else 'on'})"
                )
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                print("flashinfer PCIe IPC unavailable:", repr(e)[:300])
            fi_ws = None

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

        # --- FlashInfer PCIe IPC eager / graph (2-D [batch, hidden] only) ---
        t_fi_eager = t_fi_graph = float("nan")
        ok_fi_eager = ok_fi_graph = None
        use_fi = fi_ws is not None and n % args.hidden == 0
        if use_fi:
            x2 = x.view(n // args.hidden, args.hidden)
            use_fi = bool(fi_ws.supports(x2))
        if use_fi:
            torch.cuda.synchronize()
            fi_ws.rebind_stream()
            y = fi_ws.all_reduce(x2)
            torch.cuda.synchronize()
            ok_fi_eager = torch.equal(y.view(-1), ref)

            def fi_eager(iters):
                for _ in range(iters):
                    fi_ws.all_reduce(x2)

            t_fi_eager = timeit(fi_eager, args.iters)

            # capture on stream s: the workspace serves one stream, so re-bind around the capture
            # (vLLM does the same in FlashInferPcieIpcAllReduce.capture())
            torch.cuda.synchronize()
            fi_ws.rebind_stream()
            g_fi = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_fi, stream=s):
                for _ in range(args.iters):
                    y_fi = fi_ws.all_reduce(x2)
            torch.cuda.synchronize()
            fi_ws.rebind_stream()
            dist.barrier()
            g_fi.replay()
            torch.cuda.synchronize()
            ok_fi_graph = torch.equal(y_fi.view(-1), ref)

            def fi_graph(iters):
                g_fi.replay()

            t_fi_graph = timeit(fi_graph, args.iters)
            del g_fi
        rows.append((nbytes, t_nccl_eager, t_nccl_graph, ok_nccl_graph, use_ca, t_ca_eager, t_ca_graph, ok_ca_eager,
                     ok_ca_graph, use_fi, t_fi_eager, t_fi_graph, ok_fi_eager, ok_fi_graph))

    # gather per-rank timings, report rank 0 and the max over ranks (skew-free lockstep)
    all_rows = [None] * ws
    dist.all_gather_object(all_rows, rows)
    if rank == 0:
        print(
            f"world={ws} iters={args.iters} NCCL_P2P_LEVEL={os.environ.get('NCCL_P2P_LEVEL')} "
            f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')} custom max_size={args.max_size} "
            f"flashinfer PCIe IPC={'on' if fi_ws is not None else 'off'} hidden={args.hidden}"
        )
        print(
            f"{'bytes':>9} | {'nccl eager':>10} {'nccl graph':>10} | {'CA eager':>9} {'CA graph':>9} | "
            f"{'FI eager':>9} {'FI graph':>9} | ok(nccl-g, ca-e, ca-g, fi-e, fi-g) | max-over-ranks CA graph, FI graph"
        )
        for i, r in enumerate(rows):
            nbytes, te, tg, okg, use_ca, ce, cg, oke, okc, use_fi, fe, fg, okfe, okfg = r
            cg_max = max(rr[i][6] for rr in all_rows)
            fg_max = max(rr[i][11] for rr in all_rows)
            print(
                f"{nbytes/1024:7.1f}KiB | {te:10.1f} {tg:10.1f} | {ce:9.1f} {cg:9.1f} | {fe:9.1f} {fg:9.1f} | "
                f"{okg!s:>6} {oke!s:>6} {okc!s:>6} {okfe!s:>6} {okfg!s:>6} | {cg_max:8.1f} {fg_max:8.1f}"
            )
    type(current_platform).is_fully_connected = fc_orig
    ca.close() if hasattr(ca, "close") else None
    if fi_ws is not None:
        torch.cuda.synchronize()
        dist.barrier()
        fi_ws.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
