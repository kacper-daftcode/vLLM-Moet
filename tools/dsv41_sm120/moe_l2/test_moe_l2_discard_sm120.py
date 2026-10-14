#!/usr/bin/env python3
"""The MoE padding-row L2 discard (moe_l2_discard_sm120, patch_vllm_moe_fc2_discard_sm120.py) on one sm_120 GPU.

kernel part (no patch needed): a buffer with 64 KiB guards on both sides holds a grouped-GEMM-shaped output (FC2:
[2304, 5120] bf16, FC1: [2304, 1280]; 64-row tiles, real rows a prefix of each tile, some tiles unused). The whole
buffer is written with pattern A, pushed out of L2, overwritten with pattern B (dirty in L2), then the padding rows
are discarded. Afterwards every byte of the guards, of the real rows and of the unused tiles must still be B -
nothing outside the discarded rows was touched - while the discarded lines read back A (DRAM's stale copy: they left
L2 without a write-back). Layouts the kernel must refuse (rows not whole 128-byte lines, a misaligned base, a
non-contiguous view, int64 ids) return False without launching.

integration part (patched deep_gemm_moe.py): DeepGemmFP4Experts driven the way FusedMoEModularKernel drives it
(MoEPrepareAndFinalizeNoDPEPModular.prepare -> apply, vLLM's own workspaces) with the switch on and off, inline and
on the side stream (VLLM_MOET_MOE_FC2_DISCARD_SIDE): the MoE output must be bit-identical for 1..64 tokens and
prefill sizes (65..2048 tokens), with the default token range (2..64) and with it widened so the discard also runs
at 1 token and at the prefill sizes, eagerly and replayed from CUDA graphs (discard captured at 1/6/16/48/64 tokens
in both modes); the discard must run exactly inside the range, and on the stream its mode says. Two MoE layers
back to back on the same workspaces (the next FC1 overwrites what the discard dropped: the join has to order them),
eagerly and from a graph. The patched apply() must join the side stream after the gather.

usage (serving image, the patch applied):
    python3 test_moe_l2_discard_sm120.py [--experts 384] [--prefill 65,128,512,2048] [--part kernel|integration|all]
(--experts 256 keeps a whole run under 3 GB of GPU memory.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import moe_l2_discard_sm120 as l2  # noqa: E402

dev = torch.device("cuda")
GUARD = 64 << 10
A, B = 0x3C, 0x5A


def layout(rows: int, block_m: int, g: torch.Generator, unused_every: int = 5) -> torch.Tensor:
    """int32 expert id per row: per tile a random number (1..block_m) of real rows first, every n-th tile unused."""
    ids = torch.full((rows,), -1, dtype=torch.int32)
    for t in range(rows // block_m):
        if t % unused_every == unused_every - 1:
            continue
        n_real = int(torch.randint(1, block_m + 1, (1,), generator=g))
        ids[t * block_m : t * block_m + n_real] = t
    return ids.to(dev)


def kernel_part() -> bool:
    ok = True
    g = torch.Generator().manual_seed(1)
    evict = torch.empty(256 << 20, dtype=torch.uint8, device=dev)  # > L2 of every sm_120 part
    for rows, n, block_m in ((2304, 5120, 64), (2304, 1280, 64), (18432, 1280, 64), (1024, 5120, 128)):
        row_bytes = n * 2
        buf = torch.empty(GUARD + rows * row_bytes + GUARD, dtype=torch.uint8, device=dev)
        out = buf[GUARD : GUARD + rows * row_bytes].view(torch.bfloat16).view(rows, n)
        ids = layout(rows, block_m, g)
        buf.fill_(A)
        torch.cuda.synchronize()
        evict.fill_(1)  # A goes to DRAM
        buf.fill_(B)  # B dirty in L2
        launched = l2.discard_padding_rows(out, ids, block_m)
        torch.cuda.synchronize()
        rows_b = buf[GUARD : GUARD + rows * row_bytes].view(rows, row_bytes)
        tile_used = (ids.view(-1, block_m)[:, 0] >= 0).repeat_interleave(block_m)
        real, pad = ids >= 0, (ids < 0) & tile_used
        keep_ok = (bool((buf[:GUARD] == B).all()) and bool((buf[-GUARD:] == B).all())
                   and bool((rows_b[real | ~tile_used] == B).all()))
        lines = rows_b[pad].view(-1, 128)
        reverted = float((lines == A).all(dim=1).float().mean()) if lines.numel() else 0.0
        good = launched and keep_ok and set(lines.unique().tolist()) <= {A, B}
        print(f"[kernel] [{rows}, {n}] bf16, BLOCK_M {block_m}: {int(real.sum())} real / {int(pad.sum())} padding rows "
              f"in used tiles / {int((~tile_used).sum())} rows of unused tiles: guards, real rows and unused tiles "
              f"untouched: {keep_ok}; discarded lines that read back DRAM's stale copy: {reverted:.1%} "
              f"-> {'OK' if good else 'FAILED'}")
        ok &= good
    base = torch.empty(GUARD + 2304 * 10240, dtype=torch.uint8, device=dev)
    ids = layout(2304, 64, g)
    refused = {
        "rows of 2000 bytes": torch.empty(2304, 1000, dtype=torch.bfloat16, device=dev),
        "base 64 bytes off a line": base[64 : 64 + 2304 * 10240].view(torch.bfloat16).view(2304, 5120),
        "non-contiguous view": torch.empty(2304, 10240, dtype=torch.bfloat16, device=dev)[:, :5120],
    }
    for name, t in refused.items():
        r = l2.discard_padding_rows(t, ids, 64)
        print(f"[kernel] refused ({name}): {not r}")
        ok &= not r
    r = l2.discard_padding_rows(torch.empty(2304, 5120, dtype=torch.bfloat16, device=dev), ids.long(), 64)
    print(f"[kernel] refused (int64 ids): {not r}")
    ok &= not r
    del evict, buf, base
    torch.cuda.empty_cache()
    return ok


def integration_part(n_experts: int, prefill: list[int]) -> bool:
    import vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe as dgm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig, FusedMoEQuantDesc
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import DeepGemmFP4Experts
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _pack_deepgemm_mxfp4_scales
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.platforms import current_platform

    if not hasattr(dgm, "_moet_drop_padding_rows") or not hasattr(dgm, "_moet_l2_side_stream"):
        print("[integration] deep_gemm_moe.py is not patched with the side-stream discard "
              "(patch_vllm_moe_fc2_discard_sm120.py)")
        return False
    import inspect

    apply_src = inspect.getsource(DeepGemmFP4Experts.apply)
    fork, gather = apply_src.find("_moet_l2_side = _moet_drop_padding_rows("), apply_src.find("deepgemm_unpermute_and_reduce(")
    join = apply_src.find("torch.cuda.current_stream().wait_stream(_moet_l2_side)")
    src_ok = 0 <= fork < gather < join and apply_src.count("_moet_l2_side") == 3
    print(f"[integration] patched apply(): fork after FC2, gather, then the join of the side stream: {src_ok}")
    E, K, I, TOPK = n_experts, 5120, 640, 6
    N1 = 2 * I
    g = torch.Generator(device=dev).manual_seed(3)
    w13 = torch.randint(0, 256, (E, N1, K // 2), device=dev, dtype=torch.uint8, generator=g)
    w2 = torch.randint(0, 256, (E, K, I // 2), device=dev, dtype=torch.uint8, generator=g)
    w13_s = torch.randint(118, 126, (E, N1, K // 32), device=dev, dtype=torch.uint8, generator=g)
    w2_s = torch.randint(118, 126, (E, K, I // 32), device=dev, dtype=torch.uint8, generator=g)
    w13_s, w2_s = _pack_deepgemm_mxfp4_scales(w13, w2, w13_s, w2_s)
    fp8, block = current_platform.fp8_dtype(), GroupShape(128, 128)
    experts = DeepGemmFP4Experts.__new__(DeepGemmFP4Experts)
    experts.quant_config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(fp8, block, None, None, None, None),
        _a2=FusedMoEQuantDesc(fp8, block, None, None, None, None),
        _w1=FusedMoEQuantDesc("mxfp4", None, w13_s, None, None, None),
        _w2=FusedMoEQuantDesc("mxfp4", None, w2_s, None, None, None),
    )
    experts.moe_config = SimpleNamespace(in_dtype=torch.bfloat16)
    experts.gemm1_clamp_limit = None
    prepare = MoEPrepareAndFinalizeNoDPEPModular()
    launches = []  # the stream each discard launch went to
    real_discard = l2.discard_padding_rows

    def counting(*a, **k):
        launches.append(torch.cuda.current_stream().cuda_stream)
        return real_discard(*a, **k)

    l2.discard_padding_rows = counting

    def launched_on(side: bool) -> bool:
        """Every recorded launch went to the mode's stream: the side stream, or the stream apply() ran on."""
        want = dgm._MOET_L2_SIDE.cuda_stream if side else torch.cuda.current_stream().cuda_stream
        return all(s == want for s in launches)

    def inputs(m: int):
        x = (torch.randn(m, K, device=dev, generator=g) * 0.7).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=dev, generator=g)[:TOPK] for _ in range(m)]).to(torch.int32)
        w = torch.softmax(torch.randn(m, TOPK, device=dev, generator=g), dim=-1)
        return x, w, ids

    def moe(x, w, ids, ws):
        m = x.shape[0]
        a1q, a1q_scale, _, _, _ = prepare.prepare(x, w, ids, E, None, False, experts.quant_config,
                                                  defer_input_quant=experts.expects_unquantized_inputs)
        out = torch.empty(m, K, device=dev, dtype=torch.bfloat16)
        experts.apply(out, a1q, w13, w2, w, ids, MoEActivation.SILU, E, None, a1q_scale, None, ws[0], ws[1],
                      None, False)
        return out

    def workspaces(m: int):
        s13, s2, _ = experts.workspace_shapes(m, N1, K, TOPK, E, E, None, MoEActivation.SILU)
        return torch.empty(s13, device=dev, dtype=torch.bfloat16), torch.empty(s2, device=dev, dtype=torch.bfloat16)

    def switch(on: bool, limits: tuple[int, int] | None = None, side: bool = True) -> None:
        """off, on with the environment's token range (default 2..64), or on with limits = (min, max); side = the
        discard forked onto the side stream (the default), else inline on the model stream."""
        dgm._MOET_L2 = None
        dgm._MOET_L2_SIDE = None  # resolved again from the environment (default: a side stream)
        if on:
            assert dgm._moet_moe_l2() is not None, "the discard did not load"
            if limits is not None:
                dgm._MOET_L2 = (dgm._MOET_L2[0], *limits)
            if side:
                assert dgm._moet_l2_side_stream() is not None, "the side stream is off in this environment"
            else:
                dgm._MOET_L2_SIDE = False
        else:
            dgm._MOET_L2 = False

    ok = src_ok
    modes = ((False, "inline"), (True, "side stream"))
    with set_current_vllm_config(VllmConfig()):
        bad = []
        default_range = None
        for m in list(range(1, 65)) + prefill:
            x, w, ids = inputs(m)
            ws = workspaces(m)
            switch(False)
            ref = moe(x, w, ids, ws)
            for limits in (None, (1, 64) if m <= 64 else (1, 4096)):
                for side, _name in modes:
                    switch(True, limits, side)
                    lo, hi = dgm._MOET_L2[1], dgm._MOET_L2[2]
                    default_range = default_range or (lo, hi)
                    launches.clear()
                    ws[1].fill_(7)  # stale workspace content: the discarded rows read back whatever was there
                    got = moe(x, w, ids, ws)
                    torch.cuda.synchronize()
                    expected = 1 if lo <= m <= hi else 0
                    if (not torch.equal(got.view(torch.int16), ref.view(torch.int16)) or len(launches) != expected
                            or not launched_on(side)):
                        bad.append((m, (lo, hi), _name, len(launches)))
            del x, w, ids, ws, ref, got
            torch.cuda.empty_cache()  # one token count's workspaces at a time (the PRO 6000 test budget is 3 GB)
        sizes = "/".join(map(str, prefill))
        print(f"[integration] eager, {n_experts} experts, 1..64 tokens + {sizes}, token range "
              f"{default_range[0]}..{default_range[1]} (default) and widened to 1..64 / 1..4096, inline and side "
              f"stream: switch on != off in {len(bad)} cases {bad[:6]}; discard launched exactly inside the range, "
              f"on the mode's stream")
        ok &= not bad
        gbad = []
        for side_mode, name in modes:
            for m in (1, 6, 16, 48, 64):
                x, w, ids = inputs(m)
                ws = workspaces(m)
                switch(True, (1, 64), side_mode)
                launches.clear()
                cap = torch.cuda.Stream()
                cap.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(cap):
                    moe(x, w, ids, ws)
                    cap.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=cap):
                        out = moe(x, w, ids, ws)
                torch.cuda.synchronize()
                if len(launches) != 2:  # the warm-up call and the captured one
                    gbad.append((name, m, "not captured"))
                for _ in range(3):
                    nx, nw, nids = inputs(m)
                    x.copy_(nx)
                    w.copy_(nw)
                    ids.copy_(nids)
                    graph.replay()
                    torch.cuda.synchronize()
                    got = out.clone()
                    switch(False)
                    ref = moe(x, w, ids, workspaces(m))
                    switch(True, (1, 64), side_mode)
                    torch.cuda.synchronize()
                    if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                        gbad.append((name, m))
                del graph
        print(f"[integration] CUDA graph replays with the discard (1/6/16/48/64 tokens, inline and side stream) vs "
              f"eager switch off: differ in {len(gbad)} {gbad}")
        ok &= not gbad

        # two MoE layers back to back on the same workspaces, as in the decoder: the second layer's FC1 overwrites
        # the rows the first layer's discard dropped, so the join after the gather has to order them
        def two_layers(a, b, ws):
            return moe(*a, ws), moe(*b, ws)

        cbad = []
        for m in (6, 48):
            a, b = inputs(m), inputs(m)
            ws = workspaces(m)
            switch(False)
            ref = [t.clone() for t in two_layers(a, b, ws)]
            switch(True, (1, 64), True)
            for _ in range(5):
                ws[1].fill_(7)
                got = two_layers(a, b, ws)
                torch.cuda.synchronize()
                if not all(torch.equal(r.view(torch.int16), g.view(torch.int16)) for r, g in zip(ref, got)):
                    cbad.append((m, "eager"))
                    break
            cap = torch.cuda.Stream()
            cap.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(cap):
                two_layers(a, b, ws)
                cap.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=cap):
                    outs = two_layers(a, b, ws)
            torch.cuda.synchronize()
            for _ in range(20):
                graph.replay()
                torch.cuda.synchronize()
                if not all(torch.equal(r.view(torch.int16), g.view(torch.int16)) for r, g in zip(ref, outs)):
                    cbad.append((m, "graph"))
                    break
            del graph
        print(f"[integration] two layers back to back on one workspace, side stream (6 / 48 tokens), eager x5 and "
              f"20 graph replays vs switch off: differ in {len(cbad)} {cbad}")
        ok &= not cbad
    l2.discard_padding_rows = real_discard
    dgm._MOET_L2 = None
    dgm._MOET_L2_SIDE = None
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("kernel", "integration", "all"), default="all")
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--prefill", default="65,128,512,2048", help="token counts above the decode range")
    args = ap.parse_args()
    print(f"device={torch.cuda.get_device_name(0)}")
    ok = True
    if args.part in ("kernel", "all"):
        ok &= kernel_part()
    if args.part in ("integration", "all"):
        ok &= integration_part(args.experts, [int(v) for v in args.prefill.split(",") if v])
    print(f"peak memory reserved by torch: {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
