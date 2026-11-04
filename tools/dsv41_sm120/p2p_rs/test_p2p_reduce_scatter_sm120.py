#!/usr/bin/env python3
"""Bit test, order-table probe and bandwidth probe of the sm_120 P2P reduce-scatter in NCCL's all-reduce order
(p2p_reduce_scatter_sm120.{cu,py}) on 4 real ranks, through vLLM's TP group (the communicator the served prefill
all-reduces on). Sections:

  tables   nccl_order_tables for every M in --tokens with diagnostics: is NCCL's all-reduce sum a chain, the same for
           the 8 elements of every 16-B group (the kernel's table condition)? how many distinct orders, how long the
           runs of constant order (the ring chunks), how many rows have one order throughout; probe time per M;
  bits     the kernel's rows against the same rows of vLLM's tp.all_reduce of random bf16 partials (torch.equal on
           every rank, zero pad rows checked), 3 repeats (slot rotation / determinism); NCCL's reduce-scatter against
           the all-reduce for scale (block B: 32 % of the elements differ);
  bw       push-only bandwidth at M = 4096 (3 x 10.5 MB per rank, serialised per peer) over CTAs-per-peer x unroll,
           and the concurrent-destinations control arm; GB/s per rank = bytes pushed / time;
  time     the full reduce-scatter (push + reduce) against NCCL's reduce_scatter and all_reduce, medians of --iters,
           max over ranks, per M.

usage, inside the serving image on 4 GPUs (NCCL_P2P_LEVEL=SYS as the server):
    torchrun --nproc_per_node=4 test_p2p_reduce_scatter_sm120.py [--tokens 4096,4092,3063,2048,1024,896,128]
             [--iters 20] [--json out.json] [--skip-bw]
Exit code 1 when a served M (every M whose table passed) is not bit-identical with the all-reduce on some rank.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

os.environ.setdefault("VLLM_MOET_PREFILL_SP_RS", "p2p")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2p_reduce_scatter_sm120 as prs  # noqa: E402

H = prs.H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="4096,4092,3063,2048,1024,896,128")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-bw", action="store_true")
    ap.add_argument("--ctas", default="64,128,256,512")
    ap.add_argument("--unrolls", default="1,2,4,8")
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group, init_distributed_environment, initialize_model_parallel
    from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter

    def log(msg: str = "") -> None:
        if rank == 0:
            print(msg, flush=True)

    def allmax(v: float) -> float:
        t = torch.tensor([v], device=dev, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    def allmin_bool(b: bool) -> bool:
        t = torch.tensor([1.0 if b else 0.0], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    def resync(st: dict) -> None:
        """After push-only launches (no consumer): the peers' flags of the current slot carry seq + 1 already, so
        the sequence is advanced past it (every rank alike) and the phase / done counters cleared - the next
        reduce-scatter then waits for its own pushes again."""
        torch.cuda.synchronize()
        st["ctrl"][0] += 1
        st["ctrl"][1:5].zero_()
        torch.cuda.synchronize()
        dist.barrier()

    def timed(fn, iters: int) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        ts = []
        for _ in range(iters):
            dist.barrier()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        return allmax(statistics.median(ts))

    report: dict = {"tables": {}, "bits": {}, "bw": [], "time": {}}
    bad: list[str] = []
    with set_current_vllm_config(VllmConfig()), torch.device(dev):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        tp = get_tp_group()
        comm = tp.device_communicator.pynccl_comm
        log(f"{torch.cuda.get_device_name(dev)} x {world}; NCCL {torch.cuda.nccl.version()}; "
            f"NCCL_P2P_LEVEL={os.environ.get('NCCL_P2P_LEVEL')}; M_MAX {prs.M_MAX}")
        st = prs._init()
        if st is None:
            log(f"module unavailable: {prs._disabled_reason}")
            return 2
        ext = st["ext"]
        log(f"extension built; slots {ext.data_bytes() / 1e6:.1f} MB per rank, {prs.NSRC - 1} peers")
        sizes = [int(v) for v in args.tokens.split(",")]

        # --- tables ----------------------------------------------------------------------------------------------
        log("\n== order tables (NCCL all-reduce order per 8-element group; the kernel needs a uniform chain per group)")
        for m in sizes:
            torch.cuda.synchronize(); dist.barrier()
            t0 = time.perf_counter()
            diag: dict = {}
            tables = prs.nccl_order_tables(comm.all_reduce, rank, dev, [m], diag)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            d = diag[m]
            passed = m in tables
            passed_all = allmin_bool(passed)
            codes = ", ".join(f"{prs.decode_code(k)}:{v}" for k, v in sorted(d["codes"].items()))
            log(f"  M={m}: {'PASS' if passed_all else 'FAIL'} ({ms:.1f} ms); groups not a chain {d['frac_not_chain']:.5f}, "
                f"not uniform within 8 el {d['frac_not_uniform']:.5f}; {d['n_codes']} orders [{codes}]; {d['n_runs']} runs, "
                f"groups/run min {d['run_groups_min']} med {d['run_groups_median']:.0f} max {d['run_groups_max']} "
                f"(median run {d['run_bytes_median'] / 1e3:.0f} KB); rows with one order {d['rows_uniform']}/{m}; "
                f"order changes {d['change_points']} ({d['change_points_row_aligned']} at a row boundary)")
            report["tables"][m] = dict(d, passed=passed_all, probe_ms=ms)
            if passed:
                st["orders"][m] = tables[m]  # the test uses the kernel at this M below
            else:
                st["failed"].add(m)

        # --- bits --------------------------------------------------------------------------------------------------
        log("\n== bits: kernel rows vs tp.all_reduce rows (random bf16 partials, sigma 0.5), 3 repeats")
        for m in sizes:
            if m not in st["orders"]:
                log(f"  M={m}: skipped (no table)")
                continue
            g = torch.Generator(device=dev).manual_seed(77 + rank + 131 * m)
            partial = (0.5 * torch.randn(m, H, device=dev, dtype=torch.float32, generator=g)).to(torch.bfloat16)
            ref = tp.all_reduce(partial.clone())
            c = (m + world - 1) // world
            n = max(0, min(c, m - rank * c))
            oks = []
            for rep in range(3):
                out = prs.reduce_scatter(partial)
                assert out is not None and out.shape == (c, H)
                ok = bool(torch.equal(out[:n].view(torch.int16), ref[rank * c:rank * c + n].view(torch.int16)))
                ok &= bool((out[n:] == 0).all())
                oks.append(ok)
            torch.cuda.synchronize()
            to = int(st["timeout"].item())
            ok_all = allmin_bool(all(oks))
            nccl_rs = sp_reduce_scatter(partial)[:n]
            frac = float((nccl_rs != ref[rank * c:rank * c + n]).float().mean()) if n else 0.0
            frac = allmax(frac)
            log(f"  M={m}: kernel == all_reduce rows on every rank: {ok_all} (3 repeats{', TIMEOUT flags %#x' % to if to else ''}); "
                f"NCCL reduce_scatter vs all_reduce: {frac:.2%} of the elements differ (expected ~32 %)")
            report["bits"][m] = dict(equal=ok_all, repeats=3, timeouts=to, nccl_rs_frac_differ=frac)
            if not ok_all:
                bad.append(f"M={m}: kernel rows != all_reduce rows")
            if to:
                st["timeout"].zero_()

        # --- bandwidth -----------------------------------------------------------------------------------------------
        if not args.skip_bw:
            m = 4096
            c = m // world
            nbytes = 3 * c * H * 2
            log(f"\n== push bandwidth at M={m} ({nbytes / 1e6:.1f} MB per rank to 3 peers, serialised per peer): GB/s per rank")
            x = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
            best = None
            for ctas, unr, split in itertools.product((int(v) for v in args.ctas.split(",")), (int(v) for v in args.unrolls.split(",")),
                                                      (True, False)):
                def push_ser():
                    ext.push(x, rank, st["peers_data"], st["peers_flags"], st["ctrl"], ctas, unr, True, split)
                us = timed(push_ser, args.iters)
                gbps = nbytes / (us * 1e-6) / 1e9
                mode = "3 launches" if split else "1 launch "
                log(f"  ctas/peer {ctas:4d} x unroll {unr}, {mode}: {us:7.1f} us = {gbps:5.1f} GB/s")
                report["bw"].append(dict(mode="split" if split else "single", ctas=ctas, unroll=unr, us=us, GBps=gbps))
                if best is None or us < best[0]:
                    best = (us, ctas, unr, split)
            us, ctas, unr, split = best
            log(f"  best: ctas {ctas} x unroll {unr}, {'3 launches' if split else '1 launch'}: {us:.1f} us = {nbytes / (us * 1e-6) / 1e9:.1f} GB/s")

            def push_conc():
                ext.push(x, rank, st["peers_data"], st["peers_flags"], st["ctrl"], ctas, unr, False, False)
                st["ctrl"][1:4].zero_()
            us_c = timed(push_conc, args.iters)
            log(f"  concurrent destinations (control arm, same ctas/unroll): {us_c:.1f} us = {nbytes / (us_c * 1e-6) / 1e9:.1f} GB/s")
            report["bw"].append(dict(mode="concurrent", ctas=ctas, unroll=unr, us=us_c, GBps=nbytes / (us_c * 1e-6) / 1e9))
            # one peer only (phase 0 alone): the single-link write bandwidth
            report["bw_best"] = dict(ctas=ctas, unroll=unr, split=split, us=us)
            resync(st)

        # --- time -------------------------------------------------------------------------------------------------
        log(f"\n== time (us, median of {args.iters}, max over ranks): P2P reduce-scatter as served (push {prs.ctas_per_dest()} CTAs/peer x "
            f"unroll {prs.unroll()}, {'one launch per peer' if prs.split_launches() else 'one launch'}, consumer {prs.reduce_ctas()} CTAs) "
            f"| the single-launch / one-CTA-per-row variant | NCCL reduce_scatter | NCCL all_reduce")
        for m in sizes:
            if m not in st["orders"]:
                continue
            partial = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
            t_p2p = timed(lambda: prs.reduce_scatter(partial), args.iters)
            t_push = timed(lambda: ext.push(partial, rank, st["peers_data"], st["peers_flags"], st["ctrl"],
                                            prs.ctas_per_dest(), prs.unroll(), True, prs.split_launches()), args.iters)
            resync(st)
            c = (m + world - 1) // world
            out_v = torch.empty((c, H), dtype=torch.bfloat16, device=dev)

            def rs_variant():
                ext.reduce_scatter(partial, out_v, rank, st["data"], st["flags"], st["peers_data"], st["peers_flags"], st["ctrl"],
                                   st["orders"][m], st["timeout"], 256, 4, False, 0)
            t_var = timed(rs_variant, args.iters)
            t_rs = timed(lambda: sp_reduce_scatter(partial), args.iters)
            t_ar = timed(lambda: tp.all_reduce(partial), args.iters)
            gb = 3 * c * H * 2 / 1e9
            log(f"  M={m}: P2P RS {t_p2p:7.0f} (push alone {t_push:6.0f} = {gb / (t_push * 1e-6):4.1f} GB/s) | single-launch variant "
                f"{t_var:6.0f} | NCCL RS {t_rs:6.0f} | NCCL AR {t_ar:6.0f} | P2P RS - NCCL RS {t_p2p - t_rs:+6.0f} us")
            report["time"][m] = dict(p2p_rs=t_p2p, p2p_push=t_push, p2p_rs_single_launch=t_var, nccl_rs=t_rs, nccl_ar=t_ar)
            to = int(st["timeout"].item())
            if to:
                log(f"    TIMEOUT flags {to:#x}")
                st["timeout"].zero_()

    if rank == 0:
        print("\nSUMMARY:", "ALL OK (every served M bit-identical with NCCL's all-reduce rows on every rank)" if not bad
              else "FAILURES:\n  " + "\n  ".join(bad), flush=True)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(dict(report=report, failures=bad, nccl=torch.cuda.nccl.version(),
                               env={k: v for k, v in os.environ.items() if k.startswith(("NCCL", "VLLM_MOET"))}),
                          f, indent=1, default=str)
    dist.barrier()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
