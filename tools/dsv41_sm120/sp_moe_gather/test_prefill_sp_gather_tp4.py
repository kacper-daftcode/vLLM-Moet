#!/usr/bin/env python3
"""Bit / time test of the MXFP8 all-gather of the MoE input (sp_moe_gather_sm120.py) on 4 real ranks, no model.

Every row-local step the design moves onto this rank's row shard is checked against the full-row path on the same rows
with exact equality (torch.equal on the byte / bit views), the pattern of ../sp_boundary/test_prefill_sp_tp4.py:

  quant     FlashInfer's mxfp8 quantizer (the shared expert's, row-major scales) vs the same kernel with swizzled scales
            (unswizzled back) vs the fast prefill MoE frontend's quant_row (b12x's quantize_block_fp8_mx) vs a torch
            emulation of b12x's recipe: e4m3 bytes and UE8M0 scales on random rows of realistic and wide magnitude, on
            edge rows (zeros, +-448 * 2^k, bf16 extremes) and, when the served dumps are at hand (--dumps), on the real
            MoE input rows of a 4096-token prefill chunk;
  rows      the fast MoE kernels on bf16 rows vs on the pre-quantized rows (route rows EQUAL): the new entry point;
  router    torch.mm(x, W.T, out_dtype=fp32) on all T rows vs (a) the shard's rows alone (M = T/4: cuBLAS may pick another
            kernel) and (b) the shard's rows inside a T-row buffer (the design: the same M, hence the same kernel);
  top-6     dsv4_topk_sm120 on the shard's rows (input_ids and padding mask sliced) vs on all rows;
  time      boundary 2 at T = 4096 / 3063 / 2048 / 1024: today's RS + mHC on the shard + AG bf16 [T, 5120] vs the design's
            RS + mHC + quant + router GEMM (M = T) + top-6 on the shard + pack + AG e4m3 + AG small + unpack + swizzle,
            and every piece alone (medians of --iters, max over ranks).

usage, inside the serving image on 4 GPUs (the fast kernels from VLLM_MOET_B12X_MOE_PREFILL_FAST_DIR, built on first use):
    torchrun --nproc_per_node=4 test_prefill_sp_gather_tp4.py [--tokens 4096,3063,2048,1024] [--iters 20] [--dumps DIR]
             [--skip-time] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sp_boundary"))
sys.path.insert(0, os.environ.get("VLLM_MOET_B12X_MOE_PREFILL_FAST_DIR", str(HERE.parent / "b12x_moe_prefill_fast")))
sys.path.insert(0, os.environ.get("VLLM_MOET_DSV4_TOPK_DIR", str(HERE.parent / "dsv4_topk")))

H, E_ROUTED, TOPK, N_INTER, RSF = 5120, 384, 6, 576, 1.5


def cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Exact comparison on the bit patterns: equal?, differing elements, max |diff| where a float view exists."""
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype in (torch.float8_e4m3fn,):
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    ne = a != b
    out = dict(equal=bool(torch.equal(a, b)), diff=int(ne.sum()), n=a.numel())
    if a.dtype.is_floating_point:
        d = (a.float() - b.float()).abs()
        out["max_abs"] = float(d[torch.isfinite(d)].max()) if torch.isfinite(d).any() else float("nan")
        bits = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}[a.dtype]
        out["max_ulp"] = int((a.view(bits).long() - b.view(bits).long()).abs().max())
    return out


def fmt(c: dict) -> str:
    if c["equal"]:
        return f"EQUAL ({c['n']} el)"
    extra = f", max |ulp| {c['max_ulp']}, max |d| {c['max_abs']:.3g}" if "max_ulp" in c else ""
    return f"differ {c['diff']}/{c['n']} = {c['diff'] / max(c['n'], 1):.2%}{extra}"


def gmax(c: dict, dev) -> dict:
    """Worst case over ranks."""
    v = torch.tensor([0.0 if c["equal"] else 1.0, float(c["diff"]), float(c.get("max_ulp", 0)), float(c.get("max_abs", 0.0))],
                     device=dev, dtype=torch.float64)
    dist.all_reduce(v, op=dist.ReduceOp.MAX)
    out = dict(equal=v[0].item() == 0.0, diff=int(v[1].item()), n=c["n"])
    if "max_ulp" in c:
        out["max_ulp"], out["max_abs"] = int(v[2].item()), v[3].item()
    return out


def b12x_quant_torch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """b12x's quantize_block_fp8_mx in torch (documentation of the recipe): amax * fp32(1/448), pow2-ceil to the UE8M0
    byte (mantissa != 0 -> exponent + 1), inv = 2^(127 - byte), values * inv -> e4m3 RN (satfinite is never reached: the
    scale bounds |v * inv| by 448 up to fp32 rounding)."""
    t, k = x.shape
    v = x.float().view(t, k // 32, 32)
    amax = v.abs().amax(dim=-1)
    sc = amax * torch.tensor(0x3B124925, dtype=torch.int32).view(torch.float32).item()
    bits = sc.view(torch.int32)
    bump = (bits & 0x007FFFFF) != 0
    byte = ((bits >> 23) & 0xFF) + bump.int()
    byte = torch.where(amax > 0, byte, torch.zeros_like(byte)).clamp_(0, 255)
    inv = torch.where(byte > 0, torch.exp2((127 - byte).float()), torch.zeros_like(amax))
    q = (v * inv.unsqueeze(-1)).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(t, k).view(torch.uint8), byte.to(torch.uint8)


def dummy_planes(e: int, dev) -> dict:
    """Planes of the n64-repacked layout with the byte sizes the fast kernels check; scale bytes 127 (x 2^0) keep the
    outputs finite. The values are random: only the identity of two launches on the same rows is tested."""
    g = torch.Generator(device=dev).manual_seed(99)
    u8 = lambda n: torch.randint(0, 256, (n,), device=dev, dtype=torch.int32, generator=g).to(torch.uint8)  # noqa: E731
    return dict(
        w13=u8(e * 2 * N_INTER * H // 2),
        w13_sf=torch.full((e * 2 * N_INTER * H // 32,), 127, dtype=torch.uint8, device=dev),
        w2=u8(e * H * N_INTER // 2),
        w2_sf=torch.full((e * (H // 128) * 128 * 18,), 127, dtype=torch.uint8, device=dev),
        alpha=torch.full((e,), 0.5, device=dev), input_global_scale=torch.ones(1, device=dev),
        down_alpha=torch.full((e,), 0.25, device=dev), global_scale=torch.ones(1, device=dev),
    )


def realistic_rows(t: int, dev, seed: int) -> torch.Tensor:
    """RMSNorm-like rows (N(0,1) x a norm weight around 1), a quarter of them of wide magnitude (x 2^-6 .. 2^6), the first
    rows with edge blocks: zeros, +-448 * 2^k, bf16 max / min normal, mixed signs."""
    g = torch.Generator(device=dev).manual_seed(seed)
    w = 1 + 0.3 * torch.randn(H, device=dev, generator=g)
    x = torch.randn(t, H, device=dev, generator=g) * w
    wide = torch.exp2(torch.randint(-6, 7, (t // 4, H), device=dev, generator=g).float())
    x[: t // 4] *= wide
    x = x.to(torch.bfloat16)
    x[0] = 0
    x[1, :32] = 448.0
    x[1, 32:64] = -448.0 * 2.0 ** 10
    x[1, 64:96] = 448.0 * 2.0 ** -10
    x[2, :32] = torch.finfo(torch.bfloat16).max
    x[2, 32:64] = torch.finfo(torch.bfloat16).tiny  # the smallest normal bf16: normalized = tiny / 448 is subnormal
    x[2, 64:96] = 449.0
    x[2, 96:128] = 1e-20
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="4096,3063,2048,1024")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dumps", default=os.environ.get("VLLM_MOET_DUMPS", ""), help="served MoE dumps (call1_layer2.pt): real rows")
    ap.add_argument("--experts", type=int, default=8, help="experts of the dummy planes for the fast-kernel checks")
    ap.add_argument("--skip-time", action="store_true")
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
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter
    from vllm.models.deepseek_v41.nvidia.ops.mhc import mhc_shifted_post_pre

    import sp_moe_gather_sm120 as spg
    import test_prefill_sp_tp4 as spt  # the SP boundary test: its inputs and the boundary call
    import b12x_moe_prefill_fast_sm120 as fast
    import dsv4_topk_sm120 as topk_mod

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    report: dict = {"quant": {}, "rows": {}, "router": {}, "topk": {}, "time": {}}
    bad: list[str] = []
    with set_current_vllm_config(VllmConfig()), torch.device(dev):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        tp = get_tp_group()
        log(f"{torch.cuda.get_device_name(dev)} x {world}; NCCL {torch.cuda.nccl.version()}; torch {torch.__version__}")
        fast._ext()
        log(f"fast prefill MoE extension: {fast._ext().__file__}; pre-quantized rows entry: {fast.prequant_rows_supported()}")
        if not fast.prequant_rows_supported():
            bad.append("the fast prefill MoE extension has no pre-quantized rows entry (build from the snapshot with it)")

        # ------------------------------------------------------------------------------------------ quantizers
        def check_quant(tag: str, x: torch.Tensor) -> dict:
            """FlashInfer row-major (the design's shard quant) vs FlashInfer swizzled (today's shared expert), vs the fast
            frontend's quant_row (today's routed experts), vs the torch emulation of b12x's recipe; and the fast MoE on
            the pre-quantized rows vs on the bf16 rows (route rows), twice (determinism)."""
            t = x.shape[0]
            q_rm, s_rm = spg.quant_rows(x)
            q_sw, s_sw = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            res = dict(
                fi_values_swizzled_vs_rowmajor=cmp(q_sw.view(torch.uint8), q_rm),
                fi_scales_unswizzled_vs_rowmajor=cmp(spg.unswizzle_scales(s_sw, t, H), s_rm),
                fi_scales_swizzled_vs_relaid=cmp(s_sw, spg.swizzle_scales(s_rm)),
            )
            q_em, s_em = b12x_quant_torch(x)
            res["torch_b12x_values_vs_fi"] = cmp(q_em, q_rm)
            res["torch_b12x_scales_vs_fi"] = cmp(s_em, s_rm)
            planes = dummy_planes(args.experts, dev)
            g = torch.Generator(device=dev).manual_seed(5 + t)
            ids = torch.randint(0, args.experts, (t, TOPK), device=dev, dtype=torch.int32, generator=g)
            wts = torch.rand(t, TOPK, device=dev, generator=g)
            outs = fast.moe_prefill_fast(x, ids, wts, planes, 10.0, N_INTER, topk_sum=False)
            res["fast_frontend_values_vs_fi"] = cmp(outs[2], q_rm)
            res["fast_frontend_scales_vs_fi"] = cmp(outs[3], s_rm)
            if fast.prequant_rows_supported():
                outs_q = fast.moe_prefill_fast(None, ids, wts, planes, 10.0, N_INTER, topk_sum=False, a_values=q_rm, a_scales=s_rm)
                res["route_rows_prequant_vs_bf16"] = cmp(outs_q[1].view(torch.int16), outs[1].view(torch.int16))
                det = all(torch.equal(fast.moe_prefill_fast(None, ids, wts, planes, 10.0, N_INTER, topk_sum=False,
                                                            a_values=q_rm, a_scales=s_rm)[1].view(torch.int16), outs_q[1].view(torch.int16))
                          for _ in range(2))  # bit patterns: the edge rows' random-weight FC1 may carry NaN (NaN != NaN)
                res["route_rows_prequant_deterministic"] = dict(equal=det, diff=0, n=outs_q[1].numel())
            res = {k: gmax(v, dev) for k, v in res.items()}
            log(f"  [{tag}, T={t}]:\n    " + "\n    ".join(f"{k:40s} {fmt(v)}" for k, v in res.items()))
            return res

        log("\nQuantizer identity (FlashInfer mxfp8 row-major = swizzled = fast frontend quant_row = torch b12x recipe) and the "
            "fast MoE on pre-quantized rows:")
        x_real = realistic_rows(4096, dev, 11 + rank)
        # the edge rows apart: a block whose amax * (1/448) is an fp32 subnormal (bf16 'tiny') is the one place FlashInfer's
        # recipe (no exponent bump there) and b12x's (bump) differ - unreachable for a normed activation; reported, not judged
        rq_edge = check_quant("realistic rows + 3 edge rows (zeros, +-448 * 2^k, bf16 max / tiny, 449, 1e-20)", x_real)
        report["quant"]["realistic_with_edge_rows"] = rq_edge
        for k in ("fi_values_swizzled_vs_rowmajor", "fi_scales_unswizzled_vs_rowmajor", "fi_scales_swizzled_vs_relaid",
                  "route_rows_prequant_vs_bf16", "route_rows_prequant_deterministic"):
            if k in rq_edge and not rq_edge[k]["equal"]:
                bad.append(f"edge rows: {k} differs")
        rq_main = check_quant("realistic rows only", x_real[3:].contiguous())
        report["quant"]["realistic"] = rq_main
        for k, v in rq_main.items():
            if not v["equal"]:
                bad.append(f"quantizer on realistic rows: {k} differs")
        if args.dumps:
            p = Path(args.dumps) / "call1_layer2.pt"
            if p.exists():
                c = torch.load(p, map_location="cpu", weights_only=False)
                xr = c["hidden_states"].to(dev).to(torch.bfloat16).contiguous()
                log(f"  served dump {p.name}: hidden_states {tuple(xr.shape)}, |x| max {xr.float().abs().max().item():.3g}, "
                    f"|x| min non-zero {xr.float().abs()[xr != 0].min().item():.3g}, zero rows {int((xr == 0).all(dim=1).sum())}")
                rd = check_quant("served dump call1_layer2 (real MoE input rows)", xr)
                report["quant"]["served_dump"] = rd
                for k, v in rd.items():
                    if not v["equal"]:
                        bad.append(f"quantizer on the served dump: {k} differs")
                del c
            else:
                log(f"  (no dumps at {p})")

        # ------------------------------------------------------------------------------------------ router + top-6
        log("\nRouter GEMM 5120 -> 384 bf16 -> fp32 (torch.mm out_dtype) and the top-6 on the shard vs all rows:")
        gw = torch.Generator(device=dev).manual_seed(2024)
        w_gate = (torch.randn(E_ROUTED, H, device=dev, generator=gw) * 0.02).to(torch.bfloat16)
        bias = torch.randn(E_ROUTED, device=dev, generator=gw) * 0.1
        bias_vl = torch.randn(E_ROUTED, device=dev, generator=gw) * 0.1
        image_lo = 100
        for t in (int(v) for v in args.tokens.split(",")):
            c = (t + world - 1) // world
            lo, hi = rank * c, min((rank + 1) * c, t)
            n = hi - lo
            x = realistic_rows(t, dev, 31)  # the same rows on every rank (a full-row tensor)
            ref = torch.mm(x, w_gate.t(), out_dtype=torch.float32)
            direct = torch.mm(x[lo:hi].contiguous(), w_gate.t(), out_dtype=torch.float32)
            garbage = (torch.randn(t, H, device=dev, generator=gw) * 3).to(torch.bfloat16)
            padded, _ = spg.router_logits_same_kernel(x[lo:hi], lo, w_gate, t, buf=garbage)
            zeros_buf = torch.zeros(t, H, device=dev, dtype=torch.bfloat16)
            padded0, _ = spg.router_logits_same_kernel(x[lo:hi], lo, w_gate, t, buf=zeros_buf)
            r = dict(direct_M_shard=gmax(cmp(direct, ref[lo:hi]), dev), buffer_M_T_garbage=gmax(cmp(padded[lo:hi], ref[lo:hi]), dev),
                     buffer_M_T_zeros=gmax(cmp(padded0[lo:hi], ref[lo:hi]), dev),
                     det=gmax(cmp(torch.mm(x, w_gate.t(), out_dtype=torch.float32), ref), dev))
            log(f"  T={t} (shard {n} rows): " + "; ".join(f"{k} {fmt(v)}" for k, v in r.items()))
            report["router"][t] = r
            if not r["buffer_M_T_garbage"]["equal"] or not r["buffer_M_T_zeros"]["equal"]:
                bad.append(f"T={t}: the router GEMM on the T-row buffer is not the full-row GEMM's rows")
            # top-6: the sm_120 kernel on the shard's rows vs on all rows (ids / padding sliced); with and without a mask
            ids_tok = torch.randint(0, 1000, (t,), device=dev, dtype=torch.int32, generator=gw)
            ids_tok[lo:lo + 3] = image_lo + 2  # a few image-sentinel rows in every shard
            mask = torch.zeros(t, dtype=torch.bool, device=dev)
            mask[-5:] = True
            for tag, m in (("no mask", None), ("padding mask", mask)):
                w_full, i_full = topk_mod.dsv4_topk_cuda(ref, bias, torch.int32, RSF, is_padding=m, input_ids=ids_tok,
                                                         bias_vl=bias_vl, image_sentinel_lo=image_lo)
                w_sh, i_sh = topk_mod.dsv4_topk_cuda(ref[lo:hi].contiguous(), bias, torch.int32, RSF,
                                                     is_padding=None if m is None else m[lo:hi].contiguous(),
                                                     input_ids=ids_tok[lo:hi].contiguous(), bias_vl=bias_vl, image_sentinel_lo=image_lo)
                cw, ci = gmax(cmp(w_sh, w_full[lo:hi]), dev), gmax(cmp(i_sh, i_full[lo:hi]), dev)
                log(f"    top-6 on the shard ({tag}): weights {fmt(cw)}; ids {fmt(ci)}")
                report["topk"][f"{t}:{tag}"] = dict(weights=cw, ids=ci)
                if not cw["equal"] or not ci["equal"]:
                    bad.append(f"T={t}: top-6 on the shard differs ({tag})")
            # pack / unpack round trip
            q_rm, s_rm = spg.quant_rows(x[lo:lo + c] if hi == lo + c else torch.nn.functional.pad(x[lo:hi], (0, 0, 0, c - n)))
            small = spg.pack_small(s_rm, i_full[lo:hi], w_full[lo:hi], c)
            s2, i2, w2 = spg.unpack_small(small, H, TOPK)
            ok = torch.equal(s2, s_rm) and torch.equal(i2[:n], i_full[lo:hi]) and torch.equal(w2[:n], w_full[lo:hi])
            okv = torch.tensor([float(ok)], device=dev)
            dist.all_reduce(okv, op=dist.ReduceOp.MIN)
            log(f"    pack/unpack of scales + top-6 ({small.shape[1]} B/row): {'EQUAL' if okv.item() == 1 else 'DIFFER'}")
            if okv.item() != 1:
                bad.append(f"T={t}: pack/unpack round trip differs")

        # ------------------------------------------------------------------------------------------ time
        if not args.skip_time:
            log("\nBoundary 2 time (us, median of %d, max over ranks): today RS + mHC/4 + AG bf16 vs the MXFP8 gather" % args.iters)
            native_splits = mhc_warmup.compute_mhc_pre_num_splits

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

            def boundary(xs, residual, post, comb, pre, fn, scale, base, norm_w):
                return mhc_shifted_post_pre(xs, residual, post, comb, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20,
                                            pre_mix=pre, norm_weight=norm_w, norm_eps=1e-6)

            planes = dummy_planes(args.experts, dev)
            for t in (int(v) for v in args.tokens.split(",")):
                residual, post, comb, pre, fn, fn_b, scale, base, norm_w, partial = spt.make_inputs(t, rank, dev)
                c = (t + world - 1) // world
                lo, hi = rank * c, min((rank + 1) * c, t)
                n = hi - lo
                shard = lambda z: z[lo:lo + c].contiguous() if hi == lo + c else torch.nn.functional.pad(z[lo:hi], (0, 0) * (z.dim() - 1) + (0, c - n))  # noqa: E731
                res_s, post_s, comb_s, pre_s = shard(residual), shard(post), shard(comb), shard(pre)
                mhc_warmup.compute_mhc_pre_num_splits = lambda input_size, num_tokens, _t=t: native_splits(input_size, _t)
                xs0 = sp_reduce_scatter(partial)
                o = boundary(xs0, res_s, post_s, comb_s, pre_s, fn, scale, base, norm_w)
                x_shard = o[3].contiguous()  # [c, H] bf16 the MoE input rows of this rank
                ids_tok = torch.randint(0, 1000, (t,), device=dev, dtype=torch.int32)

                def today():
                    xs = sp_reduce_scatter(partial)
                    oo = boundary(xs, res_s, post_s, comb_s, pre_s, fn, scale, base, norm_w)
                    return tensor_model_parallel_all_gather(oo[3].contiguous(), 0)[:t]

                def design_tail(xq):
                    q, s_rm = spg.quant_rows(xq)
                    logits, buf = spg.router_logits_same_kernel(xq[:n], lo, w_gate, t)
                    tw, ti = topk_mod.dsv4_topk_cuda(logits[lo:lo + n], bias, torch.int32, RSF, is_padding=None,
                                                     input_ids=ids_tok[lo:lo + n], bias_vl=bias_vl, image_sentinel_lo=image_lo)
                    q_full = tensor_model_parallel_all_gather(q, 0)[:t]
                    small = spg.pack_small(s_rm, ti, tw, c)
                    small_full = tensor_model_parallel_all_gather(small, 0)[:t]
                    s_full, i_full, w_full = spg.unpack_small(small_full, H, TOPK)
                    sw = spg.swizzle_scales(s_full)
                    return q_full, s_full, i_full, w_full, sw, buf

                def design():
                    xs = sp_reduce_scatter(partial)
                    oo = boundary(xs, res_s, post_s, comb_s, pre_s, fn, scale, base, norm_w)
                    return design_tail(oo[3].contiguous())

                q, s_rm = spg.quant_rows(x_shard)
                logits_s, buf = spg.router_logits_same_kernel(x_shard[:n], lo, w_gate, t)
                tw, ti = topk_mod.dsv4_topk_cuda(logits_s[lo:lo + n], bias, torch.int32, RSF, input_ids=ids_tok[lo:lo + n],
                                                 bias_vl=bias_vl, image_sentinel_lo=image_lo)
                small = spg.pack_small(s_rm, ti, tw, c)
                q_full = tensor_model_parallel_all_gather(q, 0)[:t]
                small_full = tensor_model_parallel_all_gather(small, 0)[:t]
                s_full = spg.unpack_small(small_full, H, TOPK)[0]
                x_full = tensor_model_parallel_all_gather(x_shard, 0)[:t]
                tm = dict(
                    today=timed(today), design=timed(design),
                    rs=timed(lambda: sp_reduce_scatter(partial)),
                    mhc_shard=timed(lambda: boundary(xs0, res_s, post_s, comb_s, pre_s, fn, scale, base, norm_w)),
                    ag_bf16=timed(lambda: tensor_model_parallel_all_gather(x_shard, 0)),
                    ag_e4m3=timed(lambda: tensor_model_parallel_all_gather(q, 0)),
                    ag_small=timed(lambda: tensor_model_parallel_all_gather(small, 0)),
                    quant_shard=timed(lambda: spg.quant_rows(x_shard)),
                    router_gemm_M_T=timed(lambda: spg.router_logits_same_kernel(x_shard[:n], lo, w_gate, t, buf=buf)),
                    router_gemm_M_shard=timed(lambda: torch.mm(x_shard[:n], w_gate.t(), out_dtype=torch.float32)),
                    router_gemm_full_rows=timed(lambda: torch.mm(x_full, w_gate.t(), out_dtype=torch.float32)),
                    topk_shard=timed(lambda: topk_mod.dsv4_topk_cuda(logits_s[lo:lo + n], bias, torch.int32, RSF,
                                                                     input_ids=ids_tok[lo:lo + n], bias_vl=bias_vl, image_sentinel_lo=image_lo)),
                    topk_full=timed(lambda: topk_mod.dsv4_topk_cuda(logits_s, bias, torch.int32, RSF,
                                                                    input_ids=ids_tok, bias_vl=bias_vl, image_sentinel_lo=image_lo)),
                    pack_unpack_swizzle=timed(lambda: (spg.pack_small(s_rm, ti, tw, c), spg.unpack_small(small_full, H, TOPK),
                                                       spg.swizzle_scales(s_full))),
                    quant_full_rows_fi_swizzled=timed(lambda: mxfp8_e4m3_quantize(x_full, is_sf_swizzled_layout=True)),
                )
                if fast.prequant_rows_supported():
                    g = torch.Generator(device=dev).manual_seed(77)
                    ids_e = torch.randint(0, args.experts, (t, TOPK), device=dev, dtype=torch.int32, generator=g)
                    wts = torch.rand(t, TOPK, device=dev, generator=g)
                    s_full_c = s_full.contiguous()
                    tm["fast_moe_bf16_rows"] = timed(lambda: fast.moe_prefill_fast(x_full, ids_e, wts, planes, 10.0, N_INTER, topk_sum=False))
                    tm["fast_moe_prequant_rows"] = timed(lambda: fast.moe_prefill_fast(None, ids_e, wts, planes, 10.0, N_INTER, topk_sum=False,
                                                                                       a_values=q_full, a_scales=s_full_c))
                mhc_warmup.compute_mhc_pre_num_splits = native_splits
                log(f"  T={t}: today {tm['today']:.0f} -> design {tm['design']:.0f} ({tm['design'] - tm['today']:+.0f} us); "
                    f"RS {tm['rs']:.0f}, mHC shard {tm['mhc_shard']:.0f}, AG bf16 {tm['ag_bf16']:.0f} vs AG e4m3 {tm['ag_e4m3']:.0f} + AG small {tm['ag_small']:.0f}; "
                    f"quant shard {tm['quant_shard']:.0f}, router GEMM M=T {tm['router_gemm_M_T']:.0f} (M=shard {tm['router_gemm_M_shard']:.0f}, "
                    f"full rows {tm['router_gemm_full_rows']:.0f}), top-6 shard {tm['topk_shard']:.0f} (full {tm['topk_full']:.0f}), "
                    f"pack+unpack+swizzle {tm['pack_unpack_swizzle']:.0f}, FI quant of full rows (today's shared expert) {tm['quant_full_rows_fi_swizzled']:.0f}"
                    + (f"; fast MoE E={args.experts}: bf16 rows {tm['fast_moe_bf16_rows']:.0f} vs pre-quantized {tm['fast_moe_prequant_rows']:.0f}"
                       if "fast_moe_bf16_rows" in tm else ""))
                report["time"][t] = tm

    if rank == 0:
        print("\nSUMMARY:", "ALL OK (every shard-side step bit-identical with the full-row path)" if not bad else "FAILURES:\n  " + "\n  ".join(bad), flush=True)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(dict(report=report, failures=bad), f, indent=1, default=str)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
