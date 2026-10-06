#!/usr/bin/env python3
"""Why the FFN right after an Engram layer got slower with the wkv split (segments 1 and 26 of the served graph).

One CUDA graph mirrors segments 0 -> 1 of the decode graph, several times over (alternating weight sets, all cold):

  layer-0 FFN : DeepGEMM FP8 x FP4 FC1 (36 experts x 64-row tiles, 118 MB) -> silu*up + quant -> FC2 (59 MB, writes
                its padded [2304, 5120] bf16 output, 23.6 MB, of which the gather reads 36 rows) -> gather -> all-reduce
  Engram      : the patched Engram forward, wkv replicated (157 MB) or split (39 MB + all-gather); on one GPU the
                wkv GEMV on the full weight or on one 6400-row slice instead
  attention   : three GEMV v3 [6, 5120 <- 2048] (31 MB) -> all-reduce
  layer-1 FFN : the shared expert on an aux stream (GEMV v3 w13 [1152 <- 5120], silu*mul, quant, GEMV v1 w2
                [5120 <- 576]) next to FC1 -> silu*up + quant -> FC2 -> gather on the main stream -> all-reduce

Printed (torch profiler, median over the repetitions and replays): layer-1 FC1 / FC2 / shared w2 GEMV and the
layer-0 FC1, per variant:
  --engram rep|split        the wkv as before the split / as served
  --prev full|fc1|none      the layer-0 FFN as served / without FC2 (no padded output written) / absent
  --discard                 after the layer-0 gather, discard.global.L2 over FC2's output buffer (its lines leave L2
                            without a write-back; the gather has read the 36 rows it needs)

usage (serving image): torchrun --nproc_per_node=4 engram_ffn_l2_tp4.py --engram split --prev full
       (one GPU: python3 engram_ffn_l2_tp4.py ..., the wkv GEMV stands in for the Engram forward)
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _d in (HERE.parent / "sm120_gemv", Path("/opt/vllm-moet/dsv41_sm120/sm120_gemv")):
    if (_d / "mxfp8_gemv_sm120.py").exists():
        sys.path.insert(0, str(_d))
        break

T, TOPK, K, I = 6, 6, 5120, 640
N1 = 2 * I
E_ALLOC = 40  # experts allocated per weight set; the 36 (token, expert) pairs hit 36 distinct ones


def discard_l2(buf: torch.Tensor) -> None:
    import triton
    import triton.language as tl

    global _discard_kernel
    if "_discard_kernel" not in globals():
        @triton.jit
        def _k(ptr, n_lines, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            p = ptr + i.to(tl.int64) * 128
            tl.inline_asm_elementwise("discard.global.L2 [$1], 128;\n mov.u32 $0, 0;", "=r,l", [p],
                                      dtype=tl.int32, is_pure=False, pack=1)

        _discard_kernel = _k
    n_lines = buf.numel() * buf.element_size() // 128
    _discard_kernel[(triton.cdiv(n_lines, 1024),)](buf.view(torch.uint8), n_lines, BLOCK=1024)


def discard_padding_rows(out: torch.Tensor, m_indices: torch.Tensor) -> None:
    """discard.global.L2 over the rows of FC2's output whose expert id is -1 (written by the epilogue, never read)."""
    import triton
    import triton.language as tl

    global _discard_rows_kernel
    if "_discard_rows_kernel" not in globals():
        @triton.jit
        def _discard_rows(ptr, ids, row_bytes, LINES: tl.constexpr, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            if tl.load(ids + row) < 0:
                line = tl.minimum(tl.arange(0, BLOCK), LINES - 1)  # spare lanes repeat the row's last line
                p = ptr + row.to(tl.int64) * row_bytes + line.to(tl.int64) * 128
                tl.inline_asm_elementwise("discard.global.L2 [$1], 128;\n mov.u32 $0, 0;", "=r,l", [p],
                                          dtype=tl.int32, is_pure=False, pack=1)

        _discard_rows_kernel = _discard_rows
    row_bytes = out.shape[1] * out.element_size()
    assert row_bytes % 128 == 0 and out.data_ptr() % 128 == 0 and out.is_contiguous()
    lines = row_bytes // 128
    _discard_rows_kernel[(out.shape[0],)](out.view(torch.uint8), m_indices, row_bytes, LINES=lines,
                                          BLOCK=triton.next_power_of_2(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engram", choices=("rep", "split"), default="split")
    ap.add_argument("--prev", choices=("full", "fc1", "none"), default="full")
    ap.add_argument("--discard", action="store_true")
    ap.add_argument("--discard-all", action="store_true", help="discard FC2's output after the gather in both FFNs")
    ap.add_argument("--discard-pad", action="store_true",
                    help="discard only FC2's padding rows (expert id -1), right after FC2, in both FFNs")
    ap.add_argument("--reps", type=int, default=6, help="segment pairs per graph")
    ap.add_argument("--replays", type=int, default=6)
    ap.add_argument("--router-cycles", type=int, default=20000)
    args = ap.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda", torch.cuda.current_device())

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_reduce,
    )
    import vllm.model_executor.layers.fused_moe.deep_gemm_utils as dgu
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _pack_deepgemm_mxfp4_scales
    from vllm.model_executor.layers.quantization.utils.fp8_utils import silu_mul_quant_fp8_packed_triton
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils.deep_gemm import m_grouped_fp8_fp4_gemm_nt_contiguous, mk_alignment_scope
    from vllm.utils.torch_utils import set_default_torch_dtype

    from mxfp8_gemv_sm120 import mxfp8_gemv
    from nccl_gap_tp4 import build_engram, mx_weight

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    for key, value in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29533"), ("RANK", "0"), ("WORLD_SIZE", "1"),
                       ("LOCAL_RANK", "0")):
        os.environ.setdefault(key, value)
    with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(torch.bfloat16), torch.device(dev):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        g = torch.Generator(device=dev).manual_seed(5)

        def experts():
            w13 = torch.randint(0, 256, (E_ALLOC, N1, K // 2), device=dev, dtype=torch.uint8, generator=g)
            w2 = torch.randint(0, 256, (E_ALLOC, K, I // 2), device=dev, dtype=torch.uint8, generator=g)
            w13_s = torch.randint(118, 126, (E_ALLOC, N1, K // 32), device=dev, dtype=torch.uint8, generator=g)
            w2_s = torch.randint(118, 126, (E_ALLOC, K, I // 32), device=dev, dtype=torch.uint8, generator=g)
            w13_s, w2_s = _pack_deepgemm_mxfp4_scales(w13, w2, w13_s, w2_s)
            return w13.view(torch.int8), w2.view(torch.int8), w13_s, w2_s

        sets = [experts(), experts()]  # layer 0 / layer 1 of each segment pair
        x = (torch.randn(T, K, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
        x_s = torch.exp2(torch.randint(-8, -4, (T, K // 128), device=dev, generator=g).float())
        topk_ids = torch.randperm(E_ALLOC, device=dev, generator=g)[: T * TOPK].view(T, TOPK).to(torch.int32)
        topk_w = torch.softmax(torch.randn(T, TOPK, device=dev, generator=g), dim=-1)

        orig = dgu.compute_aligned_M_and_alignment

        def capped(M, num_topk, local_num_experts, alignment, expert_tokens_meta):
            m_sum, used = orig(M, num_topk, local_num_experts, alignment, expert_tokens_meta)
            if used > 64:  # the SM120 BLOCK_M the served path uses
                used = 64
                m_sum = M * num_topk + min(M * num_topk, local_num_experts) * (used - 1)
                m_sum = (m_sum + used - 1) // used * used
            return m_sum, used

        dgu.compute_aligned_M_and_alignment = capped
        try:
            res = dgu.deepgemm_moe_permute(aq=x, aq_scale=x_s, topk_ids=topk_ids, local_num_experts=E_ALLOC,
                                           expert_map=None, expert_tokens_meta=None)
        finally:
            dgu.compute_aligned_M_and_alignment = orig
        a1q, a1q_s, expert_ids, inv_perm = res[:4]
        m_sum = a1q.shape[0]
        mm1 = torch.empty(m_sum, N1, dtype=torch.bfloat16, device=dev)
        a2q_buf = torch.empty(m_sum, I, dtype=torch.float8_e4m3fn, device=dev)
        mm2 = torch.empty(m_sum, K, dtype=torch.bfloat16, device=dev)
        moe_out = torch.empty(T, K, dtype=torch.bfloat16, device=dev)
        act_kw = ({"m_indices": expert_ids}
                  if "m_indices" in inspect.signature(silu_mul_quant_fp8_packed_triton).parameters else {})
        log(f"{torch.cuda.get_device_name(dev)} x {world}: contiguous layout {m_sum} rows for {T * TOPK} pairs; "
            f"engram={args.engram} prev={args.prev} discard={args.discard} discard_all={args.discard_all} "
            f"discard_pad={args.discard_pad}; "
            f"rows with expert id -1: {(expert_ids < 0).sum().item()}")

        def ffn(ws, fc2: bool = True) -> None:
            w13, w2, w13_s, w2_s = ws
            m_grouped_fp8_fp4_gemm_nt_contiguous((a1q, a1q_s), (w13, w13_s), mm1, expert_ids,
                                                 recipe_a=(1, 128), recipe_b=(1, 32))
            if not fc2:
                return
            a2q, a2q_s = silu_mul_quant_fp8_packed_triton(mm1, group_size=128, output_q=a2q_buf, **act_kw)
            m_grouped_fp8_fp4_gemm_nt_contiguous((a2q, a2q_s), (w2, w2_s), mm2, expert_ids,
                                                 recipe_a=(1, 128), recipe_b=(1, 32))
            if args.discard_pad:
                discard_padding_rows(mm2, expert_ids)
            dgu.deepgemm_unpermute_and_reduce(a=mm2, topk_ids=topk_ids, topk_weights=topk_w, inv_perm=inv_perm,
                                              expert_map=None, output=moe_out)

        # shared expert, attention stand-in, Engram
        sh_w13 = [mx_weight(1152, K, dev, g) for _ in range(2)]
        sh_w2 = [mx_weight(K, 576, dev, g) for _ in range(2)]
        hx = torch.randn(T, K, device=dev, generator=g).to(torch.bfloat16)
        hq, hq_sf = mxfp8_e4m3_quantize(hx, is_sf_swizzled_layout=True)
        att_w = [mx_weight(5120, 2048, dev, g) for _ in range(3 * 2)]
        ax = torch.randn(T, 2048, device=dev, generator=g).to(torch.bfloat16)
        aq, aq_sf = mxfp8_e4m3_quantize(ax, is_sf_swizzled_layout=True)
        resid = torch.zeros(T, K, dtype=torch.bfloat16, device=dev)
        aux = torch.cuda.Stream()
        if world > 1:
            import vllm.models.deepseek_v41.common.engram as E

            engram, e_ids, e_hidden = build_engram(dev, rank, T)
            E._MOET_WKV_TP = (args.engram == "split", 64)

            def engram_fwd():
                engram.prepare_embeddings(e_ids)
                resid.add_(engram(e_hidden, e_ids)[:, 0])
        else:
            wkv = mx_weight(25600, 6144, dev, g)
            wkv_part = (wkv[0][:6400], wkv[1][: wkv[1].numel() // 4])
            ex = torch.randn(T, 6144, device=dev, generator=g).to(torch.bfloat16)
            eq, eq_sf = mxfp8_e4m3_quantize(ex, is_sf_swizzled_layout=True)

            def engram_fwd():
                w = wkv if args.engram == "rep" else wkv_part
                resid.add_(mxfp8_gemv(eq, eq_sf, *w)[:, :K])

        def shared(j: int) -> torch.Tensor:
            h = mxfp8_gemv(hq, hq_sf, *sh_w13[j])
            h = torch.nn.functional.silu(h[:, :576]) * h[:, 576:]
            q, q_sf = mxfp8_e4m3_quantize(h.contiguous(), is_sf_swizzled_layout=True)
            return mxfp8_gemv(q, q_sf, *sh_w2[j])

        def segment_pair(j: int) -> None:
            s0, s1 = sets[j % 2], sets[(j + 1) % 2]
            if args.prev != "none":
                ffn(s0, fc2=args.prev == "full")
                if (args.discard or args.discard_all) and args.prev == "full":
                    discard_l2(mm2)
            resid.add_(tensor_model_parallel_all_reduce(moe_out))
            engram_fwd()
            for k in range(3):
                resid.add_(mxfp8_gemv(aq, aq_sf, *att_w[(j % 2) * 3 + k]))
            resid.add_(tensor_model_parallel_all_reduce(resid))
            aux.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(aux):
                sh = shared(j % 2)
            torch.cuda._sleep(args.router_cycles)  # router + top-k + quant/scatter in front of FC1 (~10 us)
            ffn(s1)
            if args.discard_all:
                discard_l2(mm2)
            torch.cuda.current_stream().wait_stream(aux)
            resid.add_(tensor_model_parallel_all_reduce(moe_out + sh))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        scope = mk_alignment_scope(64)
        with scope, torch.cuda.stream(stream):
            if args.discard_pad:
                same = True
                for ws in sets:
                    args.discard_pad = False
                    ffn(ws)
                    ref = moe_out.clone()
                    args.discard_pad = True
                    ffn(ws)
                    same &= torch.equal(moe_out.view(torch.int16), ref.view(torch.int16))
                log(f"MoE output with the padding-row discard bit-identical to without: {same}")
            for j in range(2):
                segment_pair(j)
            stream.wait_stream(aux)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                aux.wait_stream(stream)
                for j in range(args.reps):
                    segment_pair(j)
                stream.wait_stream(aux)
        torch.cuda.synchronize()
        cpu_group = get_tp_group().cpu_group if world > 1 else None
        for _ in range(3):
            if cpu_group is not None:
                dist.barrier(group=cpu_group)
            graph.replay()
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.replays):
                if cpu_group is not None:
                    dist.barrier(group=cpu_group)
                graph.replay()
                torch.cuda.synchronize()
        path = os.path.join(tempfile.gettempdir(), f"engram_ffn_r{rank}.json")
        prof.export_chrome_trace(path)

        ev = json.load(open(path))["traceEvents"]
        launches = {e["args"]["correlation"] for e in ev if e.get("cat") == "cuda_runtime" and "GraphLaunch" in e.get("name", "")}
        stats: dict = {"fc1_l1": [], "fc2_l1": [], "sh_w2": [], "fc1_l0": [], "wkv": [], "pair": [], "discard": []}
        by: dict = {}
        for e in ev:
            if e.get("cat") == "kernel" and e["args"].get("correlation") in launches:
                by.setdefault(e["args"]["correlation"], []).append(e)
        for ks in by.values():
            ks.sort(key=lambda k: k["ts"])
            fc1 = [k for k in ks if "1d1d_impl<0u, 1280u, 5120u" in k["name"]]
            fc2 = [k for k in ks if "1d1d_impl<0u, 5120u, 640u" in k["name"]]
            w2 = [k for k in ks if "mxfp8_mma_gemv_kernel<1, 0, 0>" in k["name"]]
            wkv_k = [k for k in ks if "gemv_v3" in k["name"] and k["args"].get("grid", [0])[0] in (800, 3200)]
            n1 = 2 if args.prev != "none" else 1  # FC1 launches per segment pair
            n2 = 2 if args.prev == "full" else 1  # FC2 launches per segment pair
            for j in range(1, args.reps):  # the first pair follows the previous replay's tail
                stats["fc1_l1"].append(fc1[j * n1 + n1 - 1]["dur"])
                if args.prev != "none":
                    stats["fc1_l0"].append(fc1[j * n1]["dur"])
                stats["fc2_l1"].append(fc2[j * n2 + n2 - 1]["dur"])
                stats["sh_w2"].append(w2[j]["dur"])
            stats["wkv"] += [k["dur"] for k in wkv_k]
            stats["discard"] += [k["dur"] for k in ks if k["name"] in ("_k", "_discard_rows")]
            stats["pair"].append((ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]) / args.reps)
        med = {k: (statistics.median(v) if v else float("nan")) for k, v in stats.items()}
        allm = [None] * world
        if world > 1:
            dist.all_gather_object(allm, med, group=cpu_group)
        else:
            allm = [med]
        if rank == 0:
            for r, m in enumerate(allm):
                print(f"  rank {r}: layer-1 FC1 {m['fc1_l1']:6.1f}  FC2 {m['fc2_l1']:5.1f}  shared w2 GEMV {m['sh_w2']:5.1f}"
                      f"   | layer-0 FC1 {m['fc1_l0']:6.1f}   wkv GEMV {m['wkv']:6.1f}   per segment pair {m['pair']:7.1f} us"
                      f"   discard kernel {m['discard']:4.1f}")
            print(json.dumps({"engram": args.engram, "prev": args.prev, "discard": args.discard, "ranks": allm}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
