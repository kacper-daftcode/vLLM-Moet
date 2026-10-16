#!/usr/bin/env python3
"""A sequence-parallel mHC boundary on 4 real ranks: what it changes in the bits and what it saves in time.

The served eager prefill boundary (fuse_mhc_all_reduce, sm_120): the sublayer's partial output [T, 5120] is
all-reduced, then `mhc_shifted_post_pre` (TileLang post kernel + DeepGEMM TF32 pre-norm GEMM + fused epilogue) maps the
residual streams [T, 4, 5120] and produces the next sublayer's normed input on every rank for all T rows. The
sequence-parallel boundary reduce-scatters the partial output instead, runs the same `mhc_shifted_post_pre` on this
rank's T/4 rows, and all-gathers the next sublayer's input (the residual streams and mixes stay sharded).

Checks, every output compared bit for bit with the served path's rows:
  RS      the reduce-scattered rows against the same rows of the all-reduce (NCCL's summation order);
  native  the boundary on T/4 rows as vLLM would run it (its split-k count follows the token count);
  exact   the same with the split-k count of the full T (compute_mhc_pre_num_splits pinned), i.e. what
          per-row identity costs;
  input   the all-gathered next-sublayer input against the served one.
Then the time of one boundary (collective + mHC), served vs sequence parallel (both variants), median of
CUDA-event-timed eager calls, max over ranks.

usage, inside the serving image on GPUs 0-3:
    torchrun --nproc_per_node=4 test_sp_boundary_tp4.py [--tokens 1024,4096,4093] [--iters 20]
"""
from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist

HC, H = 4, 5120
MIX = HC * (HC + 2)


def make_inputs(t: int, rank: int, dev):
    g = torch.Generator(device=dev).manual_seed(1234)
    residual = torch.randn(t, HC, H, device=dev, dtype=torch.bfloat16, generator=g)
    post = torch.rand(t, HC, 1, device=dev, dtype=torch.float32, generator=g) * 2
    comb = torch.rand(t, HC, HC, device=dev, dtype=torch.float32, generator=g)
    comb = comb / comb.sum(-1, keepdim=True)
    pre = torch.softmax(torch.randn(t, HC, device=dev, dtype=torch.float32, generator=g), -1).contiguous()
    fn = torch.randn(MIX, HC * H, device=dev, dtype=torch.float32, generator=g) * 0.01
    scale = torch.tensor([0.5, 0.5, 0.5], device=dev, dtype=torch.float32)
    base = torch.randn(MIX, device=dev, dtype=torch.float32, generator=g) * 0.1
    norm_w = (1 + 0.1 * torch.randn(H, device=dev, dtype=torch.float32, generator=g)).to(torch.bfloat16)
    gr = torch.Generator(device=dev).manual_seed(77 + rank)  # each rank's sublayer partial differs
    partial = (0.5 * torch.randn(t, H, device=dev, dtype=torch.float32, generator=gr)).to(torch.bfloat16)
    return residual, post, comb, pre, fn, scale, base, norm_w, partial


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1024,4096,4093")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
    )
    import vllm.model_executor.kernels.mhc.warmup as mhc_warmup
    from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter
    from vllm.models.deepseek_v41.nvidia.ops.mhc import mhc_shifted_post_pre

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    bad = 0
    with set_current_vllm_config(VllmConfig()), torch.device(dev):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        tp = get_tp_group()
        native_splits = mhc_warmup.compute_mhc_pre_num_splits
        log(f"{torch.cuda.get_device_name(dev)} x {world}")

        def boundary(x, residual, post, comb, pre, fn, scale, base, norm_w):
            return mhc_shifted_post_pre(x, residual, post, comb, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20,
                                        pre_mix=pre, norm_weight=norm_w, norm_eps=1e-6)

        def pinned_splits(full_t):
            return lambda input_size, num_tokens: native_splits(input_size, full_t)

        def frac_diff(a, b):
            return float((a != b).float().mean()) if a.numel() else 0.0

        for t in (int(v) for v in args.tokens.split(",")):
            residual, post, comb, pre, fn, scale, base, norm_w, partial = make_inputs(t, rank, dev)
            c = (t + world - 1) // world
            lo, hi = rank * c, min((rank + 1) * c, t)
            n = hi - lo

            # served: all-reduce (pynccl, as get_tp_group().all_reduce in the boundary) + mHC on all rows
            x_ar = tp.all_reduce(partial.clone())
            ref = boundary(x_ar, residual, post, comb, pre, fn, scale, base, norm_w)
            ref_rows = [o[lo:hi] for o in ref[:5]]

            # sequence parallel: reduce-scatter (padded to a multiple of TP) + mHC on this rank's rows
            x_rs = sp_reduce_scatter(partial)[:n]
            shard = lambda z: z[lo:hi].contiguous()  # noqa: E731
            rs_diff = frac_diff(x_rs, x_ar[lo:hi])
            results = {}
            for variant in ("native", "exact"):
                mhc_warmup.compute_mhc_pre_num_splits = native_splits if variant == "native" else pinned_splits(t)
                outs = boundary(x_rs, shard(residual), shard(post), shard(comb), shard(pre), fn, scale, base, norm_w)
                gathered = tensor_model_parallel_all_gather(
                    torch.nn.functional.pad(outs[3], (0, 0, 0, c - n)), 0)[:t]
                results[variant] = ([frac_diff(o, r) for o, r in zip(outs[:5], ref_rows)],
                                    frac_diff(gathered, ref[3]))
                # isolate the mHC: the same rows of the all-reduced input, no reduce-scatter
                outs_ar = boundary(x_ar[lo:hi].contiguous(), shard(residual), shard(post), shard(comb), shard(pre),
                                   fn, scale, base, norm_w)
                results[variant + "_ar"] = [frac_diff(o, r) for o, r in zip(outs_ar[:5], ref_rows)]
            mhc_warmup.compute_mhc_pre_num_splits = native_splits

            stats = torch.tensor([rs_diff] + results["native"][0] + [results["native"][1]] + results["exact"][0]
                                 + [results["exact"][1]] + results["native_ar"] + results["exact_ar"], device=dev)
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            s = stats.tolist()
            names = ("residual", "post", "comb", "input", "pre")
            log(f"T={t}: rank rows {c} (last {t - (world - 1) * c}); splits full {native_splits(HC * H, t)}, "
                f"shard {native_splits(HC * H, c)}")
            log(f"  RS vs AR rows: {s[0]:.4%} of elements differ")
            log("  native (RS + mHC as vLLM picks it):  " + ", ".join(f"{k} {v:.4%}" for k, v in zip(names, s[1:6]))
                + f"; gathered input {s[6]:.4%}")
            log("  exact  (RS + mHC, full-T splits):    " + ", ".join(f"{k} {v:.4%}" for k, v in zip(names, s[7:12]))
                + f"; gathered input {s[12]:.4%}")
            log("  mHC alone on AR rows, native splits: " + ", ".join(f"{k} {v:.4%}" for k, v in zip(names, s[13:18])))
            log("  mHC alone on AR rows, full-T splits: " + ", ".join(f"{k} {v:.4%}" for k, v in zip(names, s[18:23])))
            if max(s[18:23]) > 0:
                bad += 1
                log("  [BAD] the boundary on a row subset with the full-T split count is not per-row identical")

            # timing: one boundary
            def served():
                xr = tp.all_reduce(partial)
                boundary(xr, residual, post, comb, pre, fn, scale, base, norm_w)

            res_s, post_s, comb_s, pre_s = shard(residual), shard(post), shard(comb), shard(pre)

            def seqpar(variant):
                def run():
                    mhc_warmup.compute_mhc_pre_num_splits = native_splits if variant == "native" else pinned_splits(t)
                    xs = sp_reduce_scatter(partial)[:n]
                    o = boundary(xs, res_s, post_s, comb_s, pre_s, fn, scale, base, norm_w)
                    tensor_model_parallel_all_gather(torch.nn.functional.pad(o[3], (0, 0, 0, c - n)), 0)
                return run

            def mhc_only(rows_x, rs, ps, cs, prs):
                return lambda: boundary(rows_x, rs, ps, cs, prs, fn, scale, base, norm_w)

            def timed(fn_):
                for _ in range(3):
                    fn_()
                torch.cuda.synchronize()
                dist.barrier()
                ts = []
                for _ in range(args.iters):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    fn_()
                    b.record()
                    b.synchronize()
                    ts.append(a.elapsed_time(b) * 1e3)
                v = torch.tensor([statistics.median(ts)], device=dev)
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
                return float(v.item())

            t_served = timed(served)
            t_native = timed(seqpar("native"))
            t_exact = timed(seqpar("exact"))
            mhc_warmup.compute_mhc_pre_num_splits = native_splits
            t_mhc_full = timed(mhc_only(x_ar, residual, post, comb, pre))
            t_mhc_shard = timed(mhc_only(x_ar[lo:hi].contiguous(), res_s, post_s, comb_s, pre_s))
            mhc_warmup.compute_mhc_pre_num_splits = pinned_splits(t)
            t_mhc_shard_exact = timed(mhc_only(x_ar[lo:hi].contiguous(), res_s, post_s, comb_s, pre_s))
            mhc_warmup.compute_mhc_pre_num_splits = native_splits
            log(f"  time per boundary (us): served AR+mHC {t_served:.1f} | SP native {t_native:.1f} "
                f"({t_native - t_served:+.1f}) | SP exact {t_exact:.1f} ({t_exact - t_served:+.1f}); "
                f"mHC alone: all rows {t_mhc_full:.1f}, rank rows {t_mhc_shard:.1f} native / "
                f"{t_mhc_shard_exact:.1f} full-T splits")
    log("ALL OK" if bad == 0 else f"FAILURES: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
