#!/usr/bin/env python3
"""The patched `mhc_shifted_post_pre` (patch_vllm_mhc_overlap_sm120.py) against the served path, driven
like the decoder layer does: with a side stream (the critical path in one kernel, the projection and the
coefficients on the side stream, joined afterwards) against `stream=None` (the TileLang pair), eagerly and
inside a CUDA graph, for 1-64 tokens with and without the draft aux; every output bit-identical. Then the
timing that matters: one boundary followed by a stand-in for the sublayer (a GEMM), with the join where
the decoder layer puts it.

On the vLLM main 21d93d0d base it also checks what the patch has to keep away from sm_120 and what else
crosses the boundary there:
  - the gates on a TP4 DeepSeek-V4.1-Flash config: supports_mhc_overlap / supports_mhc_all_reduce true
    (VLLM_MOET_MHC_FUSE_ALLREDUCE=0: upstream's MNNVL check decides), init_mhc_all_reduce builds no CuTe DSL
    AllReduceMHC, the TRTLLM-Gen experts that could defer the MoE finalize are not available on this GPU,
    and the model module sees MHC_OVERLAP_MAX_TOKENS = 64;
  - reduce_results=True (what fuse_mhc_all_reduce passes): the all-reduce runs once, in front of the
    kernel, on both paths - a stand-in that doubles x replaces the TP group's all-reduce;
  - the first layer's and the Engram layers' boundary (upstream's mhc_pre_delayed_overlap on the side
    stream) against mhc_pre_delayed_tilelang, which those layers run without the side stream;
  - VLLM_MOET_MHC_OVERLAP=0 in a fresh process: no kernel, MHC_OVERLAP_MAX_TOKENS 16, no side stream.

Run inside the ds41 image on one sm_120 GPU after the patch has been applied (the image does that at
build; on an unpatched image pass --patch to apply it in place first):
    python3 test_mhc_overlap_integration.py [--patch] [--bench]
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

H, HC, MIX = 5120, 4, 24
RMS_EPS, HC_EPS, POST_MULT, SINKHORN, NORM_EPS = 1e-20, 1e-6, 2.0, 20, 1e-20
CUTE_DSL = "vllm.models.deepseek_v41.nvidia.ops.cute_dsl"


def apply_patch_in_place() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import patch_vllm_mhc_overlap_sm120 as p

    src = p.DEFAULT_PATH.read_text()
    if p.MARKER not in src:
        p.DEFAULT_PATH.write_text(p.patch_text(src, str(Path(__file__).resolve().parent)))
        print(f"patched {p.DEFAULT_PATH}")


def inputs(T, dev):
    x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
    residual = (torch.randn(T, HC, H, device=dev) * 0.8).to(torch.bfloat16)
    post_layer_mix = (torch.rand(T, HC, 1, device=dev) * 2).float()
    comb_res_mix = torch.softmax(torch.randn(T, HC, HC, device=dev), dim=-1).float().contiguous()
    fn = (torch.randn(MIX, HC * H, device=dev) * 0.01).float()
    scale = torch.tensor([0.02, 0.02, 0.05], device=dev)
    base = (torch.randn(MIX, device=dev) * 0.5).float()
    norm_w = (torch.rand(H, device=dev) + 0.5).to(torch.bfloat16)
    pre_mix = (torch.rand(T, HC, device=dev) * 0.9 + 0.05).float()
    return x, residual, post_layer_mix, comb_res_mix, fn, scale, base, norm_w, pre_mix


def flash_config(tp=4):
    """The fields of a DeepSeek-V4.1-Flash TP4 VllmConfig the gates read."""
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(hc_mult=HC, hidden_size=H, num_experts_per_tok=6)),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, enable_expert_parallel=False, use_ubatching=False),
    )


class StandInAllReduce:
    """get_tp_group() stand-in: all_reduce doubles x (exact in bf16) and counts its calls."""

    def __init__(self):
        self.calls = 0
        self.device_communicator = SimpleNamespace(ca_comm=None)

    def all_reduce(self, x):
        self.calls += 1
        return x * 2


def bench_graph(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        s.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(reps):
                fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / reps)
    return statistics.median(ts)


def graphed(fn):
    """fn() captured in a CUDA graph (after one eager warm-up run) and replayed twice; returns its outputs."""
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        s.synchronize()
        with torch.cuda.graph(g, stream=s):
            outs = fn()
    g.replay()
    g.replay()
    torch.cuda.synchronize()
    return outs


def check_gates(ops) -> int:
    """What the patch must enable on sm_120 and what it must keep off it."""
    fails = 0

    def check(ok, what):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] {what}")

    cfg = flash_config()
    check(ops.supports_mhc_overlap(cfg) is True, "supports_mhc_overlap(TP4 Flash) on sm_120")
    check(ops.supports_mhc_all_reduce(cfg) is True, "supports_mhc_all_reduce(TP4 Flash) on sm_120 (fuse_mhc_all_reduce)")
    real_tp = ops.get_tp_group
    os.environ["VLLM_MOET_MHC_FUSE_ALLREDUCE"] = "0"
    try:
        ops.get_tp_group = lambda: StandInAllReduce()  # no custom all-reduce: upstream's check says no
        check(ops.supports_mhc_all_reduce(cfg) is False,
              "VLLM_MOET_MHC_FUSE_ALLREDUCE=0: upstream's MNNVL check decides (no ca_comm -> False)")
    finally:
        ops.get_tp_group = real_tp
        del os.environ["VLLM_MOET_MHC_FUSE_ALLREDUCE"]
    check(ops.supports_mhc_all_reduce(flash_config(tp=2)) is False, "supports_mhc_all_reduce(TP2) stays False")
    had_cute = CUTE_DSL in sys.modules
    ops.init_mhc_all_reduce(cfg)
    check(ops._all_reduce_mhc is None and (had_cute or CUTE_DSL not in sys.modules),
          "init_mhc_all_reduce builds no CuTe DSL AllReduceMHC on sm_120"
          + ("" if had_cute else " (cute_dsl not even imported)"))
    from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import TrtLlmMxfp4ExpertsModular

    check(not TrtLlmMxfp4ExpertsModular._supports_current_device(),
          "TRTLLM-Gen MXFP4 experts unavailable here: DeepseekV4MoE.defer_finalize() cannot defer, "
          "no MoEOutput reaches the boundary")
    try:
        from vllm.models.deepseek_v41.nvidia import model as dsv41_model

        seen = dsv41_model.MHC_OVERLAP_MAX_TOKENS
    except Exception as exc:  # noqa: BLE001
        seen = f"model.py import failed: {exc!r}"
    check(seen == ops.MHC_OVERLAP_MAX_TOKENS == 64, f"MHC_OVERLAP_MAX_TOKENS as the decoder layer sees it: {seen}")
    return fails


def check_pre_boundaries(ops, side, dev) -> int:
    """The first layer (broadcast embedding, no carried pre-mix) and the Engram layers (full streams, carried
    pre-mix) take upstream's mhc_pre_delayed_overlap on the side stream; without it they run
    mhc_pre_delayed_tilelang. Every output bit-identical, eagerly and in a graph."""
    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_delayed_tilelang

    fails = 0
    names = ("post_mix", "comb_mix", "layer_input", "next_pre_mix")
    for label in ("first layer", "Engram layer"):
        for T in (1, 6, 16, 32, 48, 64):
            x, residual, _, _, fn, scale, base, norm_w, pre_mix = inputs(T, dev)
            if label == "first layer":
                residual = x.unsqueeze(1).expand(-1, HC, -1).contiguous()
                fn = (torch.randn(MIX, H, device=dev) * 0.02).float()
                kw = dict(x=x, norm_weight=norm_w, norm_eps=NORM_EPS)
            else:
                kw = dict(pre_mix=pre_mix, norm_weight=norm_w, norm_eps=NORM_EPS)
            args = (residual, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN)

            def overlap():
                outs = ops.mhc_pre_delayed_overlap(*args, **kw, stream=side)
                torch.cuda.current_stream().wait_stream(side)
                return outs

            ref = mhc_pre_delayed_tilelang(*args, **kw)
            eager = overlap()
            torch.cuda.synchronize()
            graph = graphed(overlap)
            report, ok = [], True
            for name, r, e, g in zip(names, ref, eager, graph):
                same = torch.equal(r, e) and torch.equal(r, g)
                ok &= same
                report.append(f"{name} {'=' if same else '!='}")
            fails += 0 if ok else 1
            print(f"[{'ok ' if ok else 'BAD'}] {label:12s} T={T:2d} mhc_pre_delayed_overlap eager+graph vs "
                  f"mhc_pre_delayed_tilelang: " + ", ".join(report))
    return fails


def switch_off_probe() -> int:
    """Run in a fresh process with VLLM_MOET_MHC_OVERLAP=0."""
    from vllm.models.deepseek_v41.nvidia.ops import mhc as ops

    ok = (ops._moet_mhc_post_norm() is None and ops.MHC_OVERLAP_MAX_TOKENS == 16
          and ops.supports_mhc_overlap(flash_config()) is False)
    print(f"[{'ok ' if ok else 'BAD'}] VLLM_MOET_MHC_OVERLAP=0: kernel {ops._moet_mhc_post_norm()}, "
          f"MHC_OVERLAP_MAX_TOKENS {ops.MHC_OVERLAP_MAX_TOKENS}, supports_mhc_overlap "
          f"{ops.supports_mhc_overlap(flash_config())} (the TileLang pair, no side stream)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="store_true", help="apply patch_vllm_mhc_overlap_sm120.py in place first")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--switch-off-probe", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.switch_off_probe:
        return switch_off_probe()
    if args.patch:
        apply_patch_in_place()

    os.environ.setdefault("VLLM_MOET_MHC_OVERLAP_DIR", str(Path(__file__).resolve().parent))
    import vllm
    from vllm.models.deepseek_v41.nvidia.ops import mhc as ops

    assert hasattr(ops, "_moet_mhc_shifted_post_pre_overlap"), "ops/mhc.py is not patched (use --patch)"
    assert ops._moet_mhc_post_norm() is not None, "the sm_120 kernel did not load"
    kernels = ops._moet_mhc_post_norm()
    print(f"vLLM {vllm.__version__}, {torch.cuda.get_device_name()}, MHC_OVERLAP_MAX_TOKENS = "
          f"{ops.MHC_OVERLAP_MAX_TOKENS}, boundary kernel PDL (VLLM_MOET_MHC_PDL) = {kernels.PDL_DEFAULT}")
    dev = torch.device("cuda")
    torch.manual_seed(5)
    side = torch.cuda.Stream()
    fails = check_gates(ops)

    def run(stream, T, capture_aux, ins, reduce_results=False):
        x, residual, post_layer_mix, comb_res_mix, fn, scale, base, norm_w, pre_mix = ins
        outs = ops.mhc_shifted_post_pre(
            x, residual, post_layer_mix, comb_res_mix, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
            pre_mix=pre_mix, norm_weight=norm_w, norm_eps=NORM_EPS, capture_aux=capture_aux, stream=stream,
            reduce_results=reduce_results)
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
        return outs

    names = ("residual", "post_mix", "comb_mix", "layer_input", "next_pre_mix", "aux")
    cases = ((1, False), (6, False), (6, True), (16, True), (32, False), (33, False), (48, True), (64, False))
    for proj in ("fused", "ours", "tf32"):
        os.environ["VLLM_MOET_MHC_PROJ"] = proj
        for T, capture_aux in cases:
            ins = inputs(T, dev)
            ref = run(None, T, capture_aux, ins)
            eager = run(side, T, capture_aux, ins)
            torch.cuda.synchronize()
            # the same inside a CUDA graph (fork / join / references under capture), replayed twice
            graph = graphed(lambda: run(side, T, capture_aux, ins))
            report, ok = [], True
            for name, r, e, gr in zip(names, ref, eager, graph):
                same_e = torch.equal(r, e)
                same_g = torch.equal(r, gr)
                if not same_e or not same_g:
                    # tf32 below 33 tokens is the served path's >32-token numerics: close, not identical
                    close = torch.allclose(r.float(), e.float(), rtol=2e-2, atol=1e-3) and torch.equal(e, gr)
                    ok &= close and proj == "tf32" and T <= 32
                    report.append(f"{name} {'~' if close else '!='}")
                else:
                    report.append(f"{name} =")
            fails += 0 if ok else 1
            path = "post+tf32" if (T > 32 or proj == "tf32") else proj
            print(f"[{'ok ' if ok else 'BAD'}] T={T:2d} aux={capture_aux} proj={proj:5s} ({path}) eager+graph vs served: "
                  + ", ".join(report))
    os.environ["VLLM_MOET_MHC_PROJ"] = "ours"

    # reduce_results=True (fuse_mhc_all_reduce): the TP all-reduce in front of the kernel on both paths
    real_tp = ops.get_tp_group
    try:
        for T, capture_aux in cases:
            ins = inputs(T, dev)
            stand_in = StandInAllReduce()
            ops.get_tp_group = lambda: stand_in
            ref = run(None, T, capture_aux, ins, reduce_results=True)
            calls_ref, stand_in.calls = stand_in.calls, 0
            eager = run(side, T, capture_aux, ins, reduce_results=True)
            calls_ours, stand_in.calls = stand_in.calls, 0
            torch.cuda.synchronize()
            graph = graphed(lambda: run(side, T, capture_aux, ins, reduce_results=True))
            plain = run(None, T, capture_aux, ins)  # x not doubled: the outputs must differ
            same = [torch.equal(r, e) and torch.equal(r, g) for r, e, g in zip(ref, eager, graph)]
            ok = all(same) and calls_ref == calls_ours == 1 and not torch.equal(ref[0], plain[0])
            fails += 0 if ok else 1
            print(f"[{'ok ' if ok else 'BAD'}] T={T:2d} aux={capture_aux} reduce_results=True: all-reduce calls "
                  f"served {calls_ref} / overlap {calls_ours}, eager+graph vs served: "
                  + ", ".join(f"{n} {'=' if s else '!='}" for n, s in zip(names, same)))
    finally:
        ops.get_tp_group = real_tp

    fails += check_pre_boundaries(ops, side, dev)

    env = dict(os.environ, VLLM_MOET_MHC_OVERLAP="0")
    probe = subprocess.run([sys.executable, __file__, "--switch-off-probe"], env=env, capture_output=True, text=True)
    print("\n".join(line for line in probe.stdout.splitlines() if line.startswith("[")) or probe.stdout + probe.stderr[-2000:])
    fails += 0 if probe.returncode == 0 else 1

    if args.bench:
        # one boundary + a stand-in for the sublayer, the join after it as in the decoder layer. Two stand-ins:
        # a compute-bound GEMM that fills every SM (the pessimistic case: the side stream can only share) and
        # a weight-streaming small-M GEMM like the decode GEMVs (bandwidth-bound, SM slots free).
        w_big = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        a_big = torch.randn(64, 8192, device=dev, dtype=torch.bfloat16)
        w_gemv = [torch.randn(5120, 8192, device=dev, dtype=torch.bfloat16) for _ in range(4)]  # 4 x 84 MB, cold
        for T in (1, 6, 16, 32, 48):
            ins = inputs(T, dev)
            a_gemv = torch.randn(T, 5120, device=dev, dtype=torch.bfloat16)
            for label, work in (("compute-bound GEMM", lambda: torch.mm(a_big, w_big)),
                                ("weight-streaming GEMVs", lambda: [torch.mm(a_gemv, w) for w in w_gemv])):

                def served():
                    ops.mhc_shifted_post_pre(*ins[:7], RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN, pre_mix=ins[8],
                                             norm_weight=ins[7], norm_eps=NORM_EPS, stream=None)
                    work()

                def overlap():
                    ops.mhc_shifted_post_pre(*ins[:7], RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN, pre_mix=ins[8],
                                             norm_weight=ins[7], norm_eps=NORM_EPS, stream=side)
                    work()
                    torch.cuda.current_stream().wait_stream(side)

                t_work = bench_graph(work)
                t_served = bench_graph(served)
                line = (f"  T={T:2d} {label:22s} {t_work:6.2f} us | served boundary +{t_served - t_work:5.2f} | overlapped:")
                for proj in ("fused", "ours", "tf32"):
                    os.environ["VLLM_MOET_MHC_PROJ"] = proj
                    t_overlap = bench_graph(overlap)
                    line += f" {proj} +{t_overlap - t_work:5.2f}"
                os.environ["VLLM_MOET_MHC_PROJ"] = "ours"
                print(line)
    print("ALL OK" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
