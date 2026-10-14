#!/usr/bin/env python3
"""The DSpark drafter's mHC boundaries once its layers are given mhc_stream.

DSpark builds three DeepseekV4DecoderLayers. The first call enters with the embedding
already expanded to [T, hc, H] and pre_mix=None, so that boundary is upstream's
mhc_pre_delayed_overlap (input on the model stream, TF32 prenorm + stats on the side
stream). The next five are mhc_shifted_post_pre, the same call the target's decoder
makes between sublayers, joined after the sublayer. patch_vllm_dspark_mhc_stream_sm120.py
is what passes the stream in; this test checks that patch against nvidia/dspark.py and
checks the two calls the way the drafter chains them.

Bit-identity: the first boundary with pre_mix=None, then five shifted boundaries threaded
the way the decoder threads mixes, against the TileLang path (stream=None), eagerly and
from a CUDA graph. Then the drafter's all-reduce wiring: with fuse_mhc_all_reduce the
layers leave every sublayer output unreduced and the boundary that consumes it all-reduces
(reduce_results=True), and the last layer's FFN output has no successor boundary - the
patched forward all-reduces it in front of the final mhc_post_tilelang, as the target's
_collapse does. A stand-in all-reduce (x + x, exact in bf16, counted) drives the 3-layer
drafter in all four wirings (reduce inside the sublayer vs. fused, TileLang vs. the side
stream): the same bits, six all-reduces each; the fused side-stream drafter also from a graph.
Then the timing of the boundary chain with one weight-streaming GEMM between boundaries,
about one drafter sublayer (~60-80 us).

Run inside the ds41 image on one sm_120 GPU (the image already has the mhc.py patch):
    python3 mhc_overlap/test_dspark_mhc_stream.py [--bench]
Stays under 2 GB. --check-patch only compiles the patcher against --dspark PATH (no GPU).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

H, HC, MIX = 5120, 4, 24
RMS_EPS, HC_EPS, POST_MULT, SINKHORN, NORM_EPS = 1e-20, 1e-6, 2.0, 20, 1e-20
N_SHIFTED = 5  # three layers x two boundaries, minus the first layer's mhc_pre


def check_patch(dspark: Path) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import patch_vllm_dspark_mhc_stream_sm120 as p

    src = dspark.read_text()
    if p.MARKER in src:
        print(f"[ok ] {dspark}: already carries the drafter stream")
        return 0
    patched = p.patch_text(src)
    compile(patched, str(dspark), "exec")
    ok = (
        "mhc_stream=mhc_stream" in patched
        and "fuse_mhc_all_reduce=fuse_mhc_all_reduce" in patched
        and "supports_mhc_overlap(current_vllm_config)" in patched
        and "layer.ffn.defer_finalize()" in patched
        and p.MARKER in patched
    )
    print(f"[{'ok ' if ok else 'BAD'}] patch applies to {dspark} and passes the stream through")
    # the last layer's FFN output: all-reduced in forward() right in front of the final post-mix,
    # inside the fuse_mhc_all_reduce branch, with the import that call needs
    forward = patched[patched.index("    def forward(") :]
    final_post = forward.find("hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)")
    reduce = forward.find("hidden_states = tensor_model_parallel_all_reduce(hidden_states)")
    guard = forward.find("if self.fuse_mhc_all_reduce:")
    ok2 = (
        0 <= guard < reduce < final_post
        and "from vllm.distributed import tensor_model_parallel_all_reduce" in patched
        and patched.count("tensor_model_parallel_all_reduce(hidden_states)") == 1
    )
    print(f"[{'ok ' if ok2 else 'BAD'}] the patched forward all-reduces the last FFN output in front of "
          "mhc_post_tilelang when fuse_mhc_all_reduce")
    return (0 if ok else 1) + (0 if ok2 else 1)


def params(T, dev):
    residual = (torch.randn(T, HC, H, device=dev) * 0.8).to(torch.bfloat16)
    fn = (torch.randn(MIX, HC * H, device=dev) * 0.01).float()
    scale = torch.tensor([0.02, 0.02, 0.05], device=dev)
    base = (torch.randn(MIX, device=dev) * 0.5).float()
    norm_w = (torch.rand(H, device=dev) + 0.5).to(torch.bfloat16)
    return residual, fn, scale, base, norm_w


def cloned(ts):
    """A second call must not see tensors the first call wrote in place."""
    return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in ts)


def shifted_inputs(T, dev):
    x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
    post = (torch.rand(T, HC, 1, device=dev) * 2).float()
    comb = torch.softmax(torch.randn(T, HC, HC, device=dev), dim=-1).float().contiguous()
    pre = (torch.rand(T, HC, device=dev) * 0.9 + 0.05).float()
    residual = (torch.randn(T, HC, H, device=dev) * 0.8).to(torch.bfloat16)
    return x, post, comb, pre, residual


def bench_one(fn, flush) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        s.synchronize()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(8):
        flush.zero_()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--check-patch", action="store_true")
    ap.add_argument("--dspark", type=Path, default=Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia/dspark.py"))
    args = ap.parse_args()
    fails = check_patch(args.dspark)
    if args.check_patch:
        print("ALL OK" if fails == 0 else f"{fails} FAILURES")
        return fails

    import os

    os.environ.setdefault("VLLM_MOET_MHC_OVERLAP_DIR", str(Path(__file__).resolve().parent))
    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_delayed_tilelang
    from vllm.models.deepseek_v41.nvidia.ops import mhc as ops

    assert hasattr(ops, "_moet_mhc_shifted_post_pre_overlap"), "ops/mhc.py is not patched"
    dev = torch.device("cuda")
    torch.manual_seed(7)
    side = torch.cuda.Stream()
    names_pre = ("post_mix", "comb_mix", "layer_input", "next_pre")
    names_shift = ("residual", "post_mix", "comb_mix", "layer_input", "next_pre", "aux")

    def first(stream, p, T):
        residual, fn, scale, base, norm_w = p
        if stream is None:
            return mhc_pre_delayed_tilelang(
                residual, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
                norm_weight=norm_w, norm_eps=NORM_EPS)
        outs = ops.mhc_pre_delayed_overlap(
            residual, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
            norm_weight=norm_w, norm_eps=NORM_EPS, stream=side)
        torch.cuda.current_stream().wait_stream(side)
        return outs

    def shift(stream, ins, fns):
        x, post, comb, pre, residual = ins
        fn, scale, base, norm_w = fns
        outs = ops.mhc_shifted_post_pre(
            x, residual, post, comb, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS,
            POST_MULT, SINKHORN, pre_mix=pre, norm_weight=norm_w, norm_eps=NORM_EPS, stream=stream)
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
        return outs

    for T in (1, 6, 16, 32):
        p = params(T, dev)
        ref = first(None, cloned(p), T)
        eager = first(side, cloned(p), T)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            first(side, p, T)
            s.synchronize()
            with torch.cuda.graph(g, stream=s):
                graphed = first(side, p, T)
        g.replay()
        torch.cuda.synchronize()
        report, ok = [], True
        for name, r, e, gr in zip(names_pre, ref, eager, graphed):
            same = torch.equal(r, e) and torch.equal(r, gr)
            ok &= same
            report.append(f"{name} {'=' if same else '!='}")
        fails += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] first boundary pre_mix=None T={T:2d} eager+graph vs TileLang: "
              + ", ".join(report))

    for T in (1, 6, 16, 32):
        fns = params(T, dev)[1:]  # fn, scale, base, norm_w
        ins = shifted_inputs(T, dev)
        ref = shift(None, cloned(ins), cloned(fns))
        eager = shift(side, cloned(ins), cloned(fns))
        torch.cuda.synchronize()
        ok = all(torch.equal(r, e) for r, e in zip(ref, eager))
        fails += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] shifted boundary T={T:2d} eager vs TileLang: "
              + ", ".join(f"{n} {'=' if torch.equal(r, e) else '!='}" for n, r, e in zip(names_shift, ref, eager)))

    def chain(stream, T, p, work):
        """One drafter: mhc_pre (pre_mix=None) then five shifted boundaries, join after each sublayer."""
        residual, fn, scale, base, norm_w = p
        if stream is None:
            post, comb, x, pre = mhc_pre_delayed_tilelang(
                residual, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
                norm_weight=norm_w, norm_eps=NORM_EPS)
        else:
            post, comb, x, pre = ops.mhc_pre_delayed_overlap(
                residual, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
                norm_weight=norm_w, norm_eps=NORM_EPS, stream=stream)
            # the decoder joins after the sublayer, not before it
        work(x)
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
        hc = residual
        for _ in range(N_SHIFTED):
            outs = ops.mhc_shifted_post_pre(
                x, hc, post, comb, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
                pre_mix=pre, norm_weight=norm_w, norm_eps=NORM_EPS, stream=stream)
            hc, post, comb, x, pre, _aux = outs
            work(x)
            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)
        return hc, post, comb, x, pre

    for T in (6,):
        p = params(T, dev)
        # identity stand-in: the chain's mhc state is what has to match
        ref = chain(None, T, cloned(p), lambda _x: None)
        eager = chain(side, T, cloned(p), lambda _x: None)
        torch.cuda.synchronize()
        names = ("residual", "post_mix", "comb_mix", "layer_input", "next_pre")
        ok = all(torch.equal(r, e) for r, e in zip(ref, eager))
        fails += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] drafter chain T={T} (1 pre + {N_SHIFTED} shifted) eager vs TileLang: "
              + ", ".join(f"{n} {'=' if torch.equal(r, e) else '!='}" for n, r, e in zip(names, ref, eager)))

        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        pg = cloned(p)
        with torch.cuda.stream(s):
            chain(side, T, pg, lambda _x: None)
            s.synchronize()
            with torch.cuda.graph(g, stream=s):
                graphed = chain(side, T, pg, lambda _x: None)
        g.replay()
        torch.cuda.synchronize()
        ok = all(torch.equal(r, gr) for r, gr in zip(ref, graphed))
        fails += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] drafter chain T={T} graph replay vs TileLang: "
              + ", ".join(f"{n} {'=' if torch.equal(r, gr) else '!='}" for n, r, gr in zip(names, ref, graphed)))

    # The all-reduce wiring of the patched drafter. fuse_mhc_all_reduce builds the layers with
    # reduce_results=False on wo_b and the MoE: a sublayer's output is this rank's partial sum until the
    # boundary that consumes it all-reduces (reduce_results=True), and the last layer's FFN output has no
    # successor boundary - the patched forward all-reduces it in front of the final mhc_post_tilelang (what
    # DeepseekV4Model._collapse does for the target). A stand-in all-reduce (x + x, exact in bf16, counted)
    # drives the 3-layer drafter in all four wirings; the bits and the number of all-reduces must agree.
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
    from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton

    class StandInAllReduce:
        def __init__(self):
            self.calls = 0

        def all_reduce(self, x):
            self.calls += 1
            return x + x

    def drafter(stream, fuse, p, ws, tp):
        """DSparkDeepseekV4Model.forward at the op level: mhc_pre, then 6 sublayers (a GEMM each, its output
        this rank's partial sum) with 5 shifted boundaries between them, then the final post-mix + collapse."""
        residual, fn, scale, base, norm_w = p
        if stream is None:
            post, comb, x, pre = mhc_pre_delayed_tilelang(
                residual, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
                norm_weight=norm_w, norm_eps=NORM_EPS)
        else:
            post, comb, x, pre = ops.mhc_pre_delayed_overlap(
                residual, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
                norm_weight=norm_w, norm_eps=NORM_EPS, stream=stream)
        hc = residual
        for i in range(N_SHIFTED + 1):
            y = torch.mm(x, ws[i])  # the sublayer: this rank's partial output
            if not fuse:
                y = tp.all_reduce(y)  # reduce_results=True inside wo_b / the MoE
            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)  # the layer joins after the sublayer
            if i < N_SHIFTED:
                hc, post, comb, x, pre, _aux = ops.mhc_shifted_post_pre(
                    y, hc, post, comb, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
                    pre_mix=pre, norm_weight=norm_w, norm_eps=NORM_EPS, stream=stream, reduce_results=fuse)
            else:
                if fuse:
                    y = tp.all_reduce(y)  # the patched forward: no successor boundary
                hidden = mhc_post_tilelang(y, hc, post, comb)
                return hidden, hc_collapse_triton(hidden, pre)

    for T in (1, 6):
        p = params(T, dev)
        ws = [(torch.randn(H, H, device=dev) * 0.02).to(torch.bfloat16) for _ in range(N_SHIFTED + 1)]
        real_tp = ops.get_tp_group
        outs, calls = {}, {}
        try:
            for stream, fuse in ((None, False), (None, True), (side, False), (side, True)):
                tp = StandInAllReduce()
                ops.get_tp_group = lambda tp=tp: tp
                outs[(stream, fuse)] = drafter(stream, fuse, cloned(p), ws, tp)
                torch.cuda.synchronize()
                calls[(stream, fuse)] = tp.calls
            tp = StandInAllReduce()
            ops.get_tp_group = lambda tp=tp: tp
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            pg = cloned(p)
            with torch.cuda.stream(s):
                drafter(side, True, pg, ws, tp)
                s.synchronize()
                with torch.cuda.graph(g, stream=s):
                    graphed = drafter(side, True, pg, ws, tp)
            g.replay()
            torch.cuda.synchronize()
        finally:
            ops.get_tp_group = real_tp
        ref = outs[(None, False)]
        report, ok = [], True
        for key, label in (((None, True), "fused TileLang"), ((side, False), "side stream"),
                           ((side, True), "fused side stream")):
            same = all(torch.equal(r, o) for r, o in zip(ref, outs[key]))
            ok &= same
            report.append(f"{label} {'=' if same else '!='}")
        same = all(torch.equal(r, o) for r, o in zip(ref, graphed))
        ok &= same
        report.append(f"fused side stream from graph {'=' if same else '!='}")
        n_sub = N_SHIFTED + 1
        ok &= all(c == n_sub for c in calls.values())
        print(f"[{'ok ' if ok else 'BAD'}] drafter wiring T={T} (mhc_pre + 6 sublayers + final post-mix) vs reduce in "
              f"the sublayer on TileLang: " + ", ".join(report) + "; all-reduces "
              + "/".join(str(calls[k]) for k in ((None, False), (None, True), (side, False), (side, True)))
              + f" (expected {n_sub} each)")
        fails += 0 if ok else 1

    if args.bench:
        # one GEMM of the drafter sublayer's size between boundaries; L2 flushed between replays
        T = 6
        p = params(T, dev)
        w = torch.randn(H, 8192, device=dev, dtype=torch.bfloat16)  # 80 MB, ~one attention sublayer
        a = torch.empty(T, 8192, device=dev, dtype=torch.bfloat16)
        flush = torch.empty(128 * 1024 * 1024, device=dev, dtype=torch.int8)

        def work(x, _w=w, _a=a):
            _a.copy_(torch.mm(x, _w))

        def served():
            chain(None, T, p, work)

        def overlap():
            chain(side, T, p, work)

        t_served = bench_one(served, flush)
        t_overlap = bench_one(overlap, flush)
        print(f"bench T={T} 1 pre + {N_SHIFTED} shifted + {H}x8192 GEMM between them: "
              f"TileLang chain {t_served:.1f} us, with the side stream {t_overlap:.1f} us "
              f"({t_overlap - t_served:+.1f} us)")
    print("ALL OK" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
