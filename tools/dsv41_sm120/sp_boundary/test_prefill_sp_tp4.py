#!/usr/bin/env python3
"""Bit test of the prefill sequence-parallel boundary (patch_vllm_prefill_sp_sm120.py) on 4 real ranks, no model.

Everything the patch moves onto this rank's row shard is checked against the served path on the same rows, with
exact equality counts (torch.equal / elements that differ / max |ulp| in bf16):

  RS        the rows of NCCL's reduce-scatter (sp_reduce_scatter, padded to a multiple of TP) against the same rows
            of NCCL's all-reduce of the same partials; and all_gather(reduce_scatter(x)) against all_reduce(x);
  boundary  mhc_shifted_post_pre (post + TF32 split-k pre-mix projection + Sinkhorn + RMSNorm) on the shard against
            the served call on all rows: "native" (the split-k count vLLM picks for the shard) and "pinned" (the
            unsharded step's count, what the patch does); on the all-reduced rows to isolate the mHC from the RS;
  engram    the Engram layer's pair (mhc_post_tilelang + mhc_pre_delayed_tilelang) the same way; and the first layer's
            mhc_pre_delayed_tilelang on the broadcast embedding;
  gemm      row independence of the replicated input projections: CUTLASS mm_mxfp8 (fused_wqa_wkv 5120 -> 1792,
            Engram wkv 6144 -> 25600; the 29.10 run used K = 1536), cuBLAS bf16 with fp32 output (compressor 5120 -> 1024) and bf16 output (indexer
            weights_proj 5120 -> 32): the full-M result's rows against the same rows computed with M = T/4;
  time      one boundary: served (AR + mHC) vs SP as the patch runs it at the attention input (RS + mHC on the shard +
            all-gather of the [T, 1792] projection) and at the MoE input (RS + mHC + all-gather of [T, 5120]).

usage, inside the serving image on 4 GPUs:
    torchrun --nproc_per_node=4 test_prefill_sp_tp4.py [--tokens 4096,4092,3063,1024,896,128] [--iters 20]
             [--skip-mhc] [--skip-gemm] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
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
    fn_b = torch.randn(MIX, H, device=dev, dtype=torch.float32, generator=g) * 0.01
    scale = torch.tensor([0.5, 0.5, 0.5], device=dev, dtype=torch.float32)
    base = torch.randn(MIX, device=dev, dtype=torch.float32, generator=g) * 0.1
    norm_w = (1 + 0.1 * torch.randn(H, device=dev, dtype=torch.float32, generator=g)).to(torch.bfloat16)
    gr = torch.Generator(device=dev).manual_seed(77 + rank)  # each rank's sublayer partial differs
    partial = (0.5 * torch.randn(t, H, device=dev, dtype=torch.float32, generator=gr)).to(torch.bfloat16)
    return residual, post, comb, pre, fn, fn_b, scale, base, norm_w, partial


def cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Exact comparison: equal?, differing elements, max |ulp| (bf16/fp32 bit patterns) and max |diff|."""
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.numel() == 0:
        return dict(equal=True, diff=0, n=0, max_ulp=0, max_abs=0.0)
    ne = a != b
    out = dict(equal=bool(torch.equal(a, b)), diff=int(ne.sum()), n=a.numel(), max_abs=float((a.float() - b.float()).abs().max()))
    if a.dtype == torch.bfloat16:
        ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
    elif a.dtype == torch.float32:
        ia, ib = a.view(torch.int32).long(), b.view(torch.int32).long()
    else:
        ia = ib = None
    out["max_ulp"] = int((ia - ib).abs().max()) if ia is not None else -1
    return out


def fmt(c: dict) -> str:
    if c["equal"]:
        return f"EQUAL ({c['n']} el)"
    return f"differ {c['diff']}/{c['n']} = {c['diff'] / max(c['n'], 1):.2%}, max |ulp| {c['max_ulp']}, max |d| {c['max_abs']:.3g}"


def gmax(stats: list[dict], dev) -> list[dict]:
    """Worst case over ranks (diff counts summed, equal = all ranks equal)."""
    out = []
    for c in stats:
        v = torch.tensor([0.0 if c["equal"] else 1.0, c["diff"], c["max_ulp"], c["max_abs"]], device=dev, dtype=torch.float64)
        dist.all_reduce(v, op=dist.ReduceOp.MAX)
        s = torch.tensor([float(c["diff"]), float(c["n"])], device=dev, dtype=torch.float64)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        out.append(dict(equal=v[0].item() == 0.0, diff=int(s[0].item()), n=int(s[1].item()), max_ulp=int(v[2].item()), max_abs=v[3].item()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="4096,4092,3063,1024,896,128")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--skip-mhc", action="store_true")
    ap.add_argument("--skip-gemm", action="store_true")
    ap.add_argument("--json", default=None)
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
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_pre_delayed_tilelang
    from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter
    from vllm.models.deepseek_v41.nvidia.ops.mhc import mhc_shifted_post_pre

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    report: dict = {"tokens": {}, "gemm": {}}
    bad: list[str] = []
    with set_current_vllm_config(VllmConfig()), torch.device(dev):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        tp = get_tp_group()
        native_splits = mhc_warmup.compute_mhc_pre_num_splits
        log(f"{torch.cuda.get_device_name(dev)} x {world}; NCCL {torch.cuda.nccl.version()}")

        def boundary(x, residual, post, comb, pre, fn, scale, base, norm_w):
            return mhc_shifted_post_pre(x, residual, post, comb, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20,
                                        pre_mix=pre, norm_weight=norm_w, norm_eps=1e-6)

        def engram_pair(x, residual, post, comb, pre, fn, scale, base, norm_w):
            res = mhc_post_tilelang(x, residual, post, comb)
            outs = mhc_pre_delayed_tilelang(res, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20, pre_mix=pre,
                                            norm_weight=norm_w, norm_eps=1e-6)
            return (res, *outs)

        def first_layer(emb, fn_b, scale, base, norm_w):
            res = emb.unsqueeze(1).expand(-1, HC, -1).contiguous()
            return mhc_pre_delayed_tilelang(res, fn_b, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20, x=emb,
                                            norm_weight=norm_w, norm_eps=1e-6)

        def pinned(full_t):
            return lambda input_size, num_tokens: native_splits(input_size, full_t)

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

        for t in (int(v) for v in args.tokens.split(",")):
            rep: dict = {}
            residual, post, comb, pre, fn, fn_b, scale, base, norm_w, partial = make_inputs(t, rank, dev)
            c = (t + world - 1) // world
            lo, hi = rank * c, min((rank + 1) * c, t)
            n = hi - lo
            shard = lambda z: z[lo:hi].contiguous()  # noqa: E731
            pad_rows = lambda z: torch.nn.functional.pad(z, (0, 0, 0, c - n))  # noqa: E731
            log(f"\nT={t}: rows per rank {c} (last rank {t - (world - 1) * c}); mHC pre-mix split-k: full {native_splits(HC * H, t)}, "
                f"shard {native_splits(HC * H, c)}; first layer (K={H}): full {native_splits(H, t)}, shard {native_splits(H, c)}")

            # --- RS vs AR -------------------------------------------------------------------------------------
            x_ar = tp.all_reduce(partial.clone())
            x_rs = sp_reduce_scatter(partial)[:n]
            rs_cmp = gmax([cmp(x_rs, x_ar[lo:hi])], dev)[0]
            ag_rs = tensor_model_parallel_all_gather(pad_rows(x_rs), 0)[:t]
            ag_cmp = gmax([cmp(ag_rs, x_ar)], dev)[0]
            rows_equal = int((ag_rs == x_ar).all(dim=1).sum())
            # determinism of each collective (3 repeats)
            det_ar = all(torch.equal(tp.all_reduce(partial.clone()), x_ar) for _ in range(3))
            det_rs = all(torch.equal(sp_reduce_scatter(partial)[:n], x_rs) for _ in range(3))
            det = torch.tensor([float(det_ar), float(det_rs)], device=dev)
            dist.all_reduce(det, op=dist.ReduceOp.MIN)
            log(f"  RS rows vs AR rows:        {fmt(rs_cmp)}")
            log(f"  all_gather(RS) vs AR:      {fmt(ag_cmp)}; rows identical {rows_equal}/{t}; "
                f"deterministic: AR {bool(det[0])}, RS {bool(det[1])}")
            rep["rs_vs_ar"] = rs_cmp
            rep["ag_rs_vs_ar"] = dict(ag_cmp, rows_equal=rows_equal, rows=t, det_ar=bool(det[0]), det_rs=bool(det[1]))

            if not args.skip_mhc:
                # --- the boundary on the shard ------------------------------------------------------------------
                ref = boundary(x_ar, residual, post, comb, pre, fn, scale, base, norm_w)
                ref_rows = [o[lo:hi] for o in ref[:5]]
                names = ("residual", "post", "comb", "input", "pre")
                for variant in ("native", "pinned"):
                    mhc_warmup.compute_mhc_pre_num_splits = native_splits if variant == "native" else pinned(t)
                    # the mHC alone: the all-reduced rows (no RS) -> isolates the kernel's row independence
                    outs_ar = boundary(x_ar[lo:hi].contiguous(), shard(residual), shard(post), shard(comb), shard(pre),
                                       fn, scale, base, norm_w)
                    cs = gmax([cmp(o, r) for o, r in zip(outs_ar[:5], ref_rows)], dev)
                    log(f"  mHC on the AR rows, {variant:6s} splits: " + "; ".join(f"{k} {fmt(v)}" for k, v in zip(names, cs)))
                    rep[f"mhc_{variant}_on_ar_rows"] = dict(zip(names, cs))
                    if variant == "pinned" and not all(v["equal"] for v in cs):
                        if c <= 32:
                            # <= 32 rows take the fused post + pre kernel (another arithmetic); the patch engages
                            # from VLLM_MOET_PREFILL_SP_MIN_TOKENS = 2048 rows, i.e. shards of >= 512 rows
                            log(f"  (T={t}: a {c}-row shard takes the <= 32-token fused kernel - outside the patch's range, not a failure)")
                        else:
                            bad.append(f"T={t}: the boundary on a row subset with the pinned split count is not per-row identical")
                    # the patch's path: RS rows -> boundary -> gathered input
                    outs = boundary(x_rs, shard(residual), shard(post), shard(comb), shard(pre), fn, scale, base, norm_w)
                    gathered = tensor_model_parallel_all_gather(pad_rows(outs[3]), 0)[:t]
                    cs2 = gmax([cmp(o, r) for o, r in zip(outs[:5], ref_rows)] + [cmp(gathered, ref[3])], dev)
                    log(f"  RS + mHC, {variant:6s} splits:        " + "; ".join(f"{k} {fmt(v)}" for k, v in zip(names, cs2[:5]))
                        + f"; gathered input {fmt(cs2[5])}")
                    rep[f"rs_mhc_{variant}"] = dict(zip((*names, "gathered_input"), cs2))
                mhc_warmup.compute_mhc_pre_num_splits = native_splits

                # --- the Engram layer's pair and the first layer ------------------------------------------------
                ref_e = engram_pair(x_ar, residual, post, comb, pre, fn, scale, base, norm_w)
                mhc_warmup.compute_mhc_pre_num_splits = pinned(t)
                out_e = engram_pair(x_ar[lo:hi].contiguous(), shard(residual), shard(post), shard(comb), shard(pre),
                                    fn, scale, base, norm_w)
                mhc_warmup.compute_mhc_pre_num_splits = native_splits
                cs = gmax([cmp(o, r[lo:hi]) for o, r in zip(out_e, ref_e)], dev)
                en = ("post_residual", "post_mix", "comb", "input", "pre")
                log("  Engram pair (post + pre) on the AR rows, pinned: " + "; ".join(f"{k} {fmt(v)}" for k, v in zip(en, cs)))
                rep["engram_pair_pinned"] = dict(zip(en, cs))
                if not all(v["equal"] for v in cs):
                    bad.append(f"T={t}: Engram pair on a row subset (pinned) differs")
                emb = residual[:, 0].contiguous()
                ref_0 = first_layer(emb, fn_b, scale, base, norm_w)
                mhc_warmup.compute_mhc_pre_num_splits = pinned(t)
                out_0 = first_layer(shard(emb), fn_b, scale, base, norm_w)
                mhc_warmup.compute_mhc_pre_num_splits = native_splits
                cs = gmax([cmp(o, r[lo:hi]) for o, r in zip(out_0, ref_0)], dev)
                f0 = ("post_mix", "comb", "input", "pre")
                log("  first layer pre on the embedding shard, pinned:  " + "; ".join(f"{k} {fmt(v)}" for k, v in zip(f0, cs)))
                rep["first_layer_pinned"] = dict(zip(f0, cs))
                if not all(v["equal"] for v in cs):
                    bad.append(f"T={t}: first-layer pre on the embedding shard (pinned) differs")

                # --- time ---------------------------------------------------------------------------------------
                res_s, post_s, comb_s, pre_s = shard(residual), shard(post), shard(comb), shard(pre)
                proj = torch.empty(c, 1792, device=dev, dtype=torch.bfloat16)

                def served():
                    xr = tp.all_reduce(partial)
                    boundary(xr, residual, post, comb, pre, fn, scale, base, norm_w)

                def sp_attn_input():  # boundary 1 as patched: RS, mHC on the shard, the [T,1792] projection gathered
                    mhc_warmup.compute_mhc_pre_num_splits = pinned(t)
                    xs = sp_reduce_scatter(partial)[:n]
                    boundary(xs, res_s, post_s, comb_s, pre_s, fn, scale, base, norm_w)
                    tensor_model_parallel_all_gather(proj, 0)

                def sp_moe_input():  # boundary 2 as patched: RS, mHC on the shard, the [T,5120] input gathered
                    mhc_warmup.compute_mhc_pre_num_splits = pinned(t)
                    xs = sp_reduce_scatter(partial)[:n]
                    o = boundary(xs, res_s, post_s, comb_s, pre_s, fn, scale, base, norm_w)
                    tensor_model_parallel_all_gather(pad_rows(o[3]), 0)

                t_served, t_b1, t_b2 = timed(served), timed(sp_attn_input), timed(sp_moe_input)
                t_ar = timed(lambda: tp.all_reduce(partial))
                t_rs = timed(lambda: sp_reduce_scatter(partial))
                t_ag_full = timed(lambda: tensor_model_parallel_all_gather(pad_rows(x_rs), 0))
                t_ag_proj = timed(lambda: tensor_model_parallel_all_gather(proj, 0))
                mhc_warmup.compute_mhc_pre_num_splits = native_splits
                t_mhc_full = timed(lambda: boundary(x_ar, residual, post, comb, pre, fn, scale, base, norm_w))
                mhc_warmup.compute_mhc_pre_num_splits = pinned(t)
                t_mhc_shard = timed(lambda: boundary(x_ar[lo:hi].contiguous(), res_s, post_s, comb_s, pre_s, fn, scale, base, norm_w))
                mhc_warmup.compute_mhc_pre_num_splits = native_splits
                log(f"  time (us, median, max over ranks): served AR+mHC {t_served:.0f} | SP attention input RS+mHC/4+AG[T,1792] "
                    f"{t_b1:.0f} ({t_b1 - t_served:+.0f}) | SP MoE input RS+mHC/4+AG[T,5120] {t_b2:.0f} ({t_b2 - t_served:+.0f}); "
                    f"AR {t_ar:.0f}, RS {t_rs:.0f}, AG[T,5120] {t_ag_full:.0f}, AG[T,1792] {t_ag_proj:.0f}, "
                    f"mHC all rows {t_mhc_full:.0f}, mHC shard (pinned) {t_mhc_shard:.0f}")
                rep["time_us"] = dict(served=t_served, sp_attn_input=t_b1, sp_moe_input=t_b2, ar=t_ar, rs=t_rs,
                                      ag_full=t_ag_full, ag_proj=t_ag_proj, mhc_full=t_mhc_full, mhc_shard_pinned=t_mhc_shard)
            report["tokens"][t] = rep

        if not args.skip_gemm:
            # --- row independence of the replicated input projections ---------------------------------------------
            log("\nGEMM row independence (rows of the M=4096 result vs the same rows computed with M=1024 / 1023):")
            from flashinfer import mm_mxfp8, mxfp8_quantize

            g = torch.Generator(device=dev).manual_seed(4321)
            m = 4096
            for k, nn_, what in ((H, 1792, "fused_wqa_wkv (CUTLASS mxfp8)"), (6144, 25600, "Engram wkv (CUTLASS mxfp8)")):
                a = (torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g) / 8).clamp(-2, 2)
                w = (torch.randn(nn_, k, device=dev, dtype=torch.bfloat16, generator=g) / 8).clamp(-2, 2)
                w_q, w_s = mxfp8_quantize(w, is_sf_swizzled_layout=True)

                def run(rows):
                    a_q, a_s = mxfp8_quantize(rows, is_sf_swizzled_layout=True)
                    return mm_mxfp8(a_q, w_q.t(), a_s, w_s, out_dtype=torch.bfloat16, backend="cutlass")

                full = run(a)
                res = {}
                for mm in (1024, 1023, 766):
                    part = run(a[1024:1024 + mm].contiguous())
                    res[mm] = cmp(part, full[1024:1024 + mm])
                log(f"  {what}: " + "; ".join(f"M={mm}: {fmt(cv)}" for mm, cv in res.items()))
                report["gemm"][what] = {str(mm): cv for mm, cv in res.items()}
                if not all(cv["equal"] for cv in res.values()):
                    bad.append(f"{what}: rows depend on M")
            for nn_, out_f32, what in ((1024, True, "compressor fused_wkv_wgate (cuBLAS bf16 -> fp32)"),
                                       (512, True, "compressor wkv only (cuBLAS bf16 -> fp32)"),
                                       (32, False, "indexer weights_proj (cuBLAS bf16 -> bf16)")):
                a = torch.randn(m, H, device=dev, dtype=torch.bfloat16, generator=g)
                w = torch.randn(nn_, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02

                def run(rows):
                    if out_f32:
                        return torch.mm(rows, w.T, out_dtype=torch.float32)
                    return torch.nn.functional.linear(rows, w)

                full = run(a)
                res = {}
                for mm in (1024, 1023, 766):
                    res[mm] = cmp(run(a[1024:1024 + mm].contiguous()), full[1024:1024 + mm])
                log(f"  {what}: " + "; ".join(f"M={mm}: {fmt(cv)}" for mm, cv in res.items()))
                report["gemm"][what] = {str(mm): cv for mm, cv in res.items()}

    if rank == 0:
        print("\nSUMMARY:", "ALL OK (every pinned row-subset computation identical)" if not bad else "FAILURES:\n  " + "\n  ".join(bad), flush=True)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(dict(report=report, failures=bad), f, indent=1, default=str)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
