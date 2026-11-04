"""The load-based gather + hoisted merge (sparse_mla_ldg_sm120) against FlashInfer 0.7.2's sparse MLA decode on
sm_120: output / LSE / split-scratch bits on random mixed-cache inputs (full and partial candidate lists, -1 padding,
tokens 1..8), the patched execute path end to end (a subprocess with VLLM_MOET_SPARSE_MLA_LDG=1, SWA-only layers
falling through to FlashInfer), and the time of the served call in a CUDA graph.

  python3 test_sparse_mla_ldg_sm120.py [--cases 40] [--iters 50] [--no-subprocess]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

H, D, PAGE = 16, 512, 128
SWA_W, TOPK = 128, 512
E4M3_NAN = (0x7F, 0xFF)


def fp8_rows(n, gen, dev):
    """n V4.1 FP8 records (512 e4m3 bytes + 16 ue8m0 scale bytes), random codes, scales 2^-10..2^-6."""
    data = torch.randint(0, 256, (n, 512), dtype=torch.uint8, device=dev, generator=gen)
    for nan in E4M3_NAN:
        data[data == nan] = 0x40
    scales = torch.randint(117, 122, (n, 16), dtype=torch.uint8, device=dev, generator=gen)
    return torch.cat([data, scales], dim=1)


def fp4_rows(n, gen, dev):
    """n V41_FP4 records (256 packed e2m1 bytes + 32 e4m3 scale bytes); scales ~0.25..1."""
    data = torch.randint(0, 256, (n, 256), dtype=torch.uint8, device=dev, generator=gen)
    scales = torch.randint(0x34, 0x39, (n, 32), dtype=torch.uint8, device=dev, generator=gen)
    return torch.cat([data, scales], dim=1)


def make_inputs(tokens, gen, dev, swa_lens, extra_lens, swa_pages=64, extra_pages=256):
    q = torch.randn(tokens, H, D, device=dev, generator=gen).to(torch.bfloat16)
    swa = fp8_rows(swa_pages * PAGE, gen, dev).view(swa_pages, PAGE, 1, 528)
    extra = fp4_rows(extra_pages * PAGE, gen, dev).view(extra_pages, PAGE, 1, 288)
    swa_idx = torch.randint(0, swa_pages * PAGE, (tokens, SWA_W), dtype=torch.int32, device=dev, generator=gen)
    extra_idx = torch.randint(0, extra_pages * PAGE, (tokens, TOPK), dtype=torch.int32, device=dev, generator=gen)
    swa_idx[torch.arange(SWA_W, device=dev)[None, :] >= swa_lens.to(dev)[:, None]] = -1
    extra_idx[torch.arange(TOPK, device=dev)[None, :] >= extra_lens.to(dev)[:, None]] = -1
    sinks = torch.randn(H, device=dev, generator=gen) * 2
    return dict(q=q, swa=swa, extra=extra, swa_idx=swa_idx, extra_idx=extra_idx, swa_lens=swa_lens.to(dev),
                extra_lens=extra_lens.to(dev), sinks=sinks)


def case_lens(case, tokens):
    swa_lens = torch.full((tokens,), SWA_W, dtype=torch.int32)
    extra_lens = torch.full((tokens,), TOPK, dtype=torch.int32)
    g = torch.Generator().manual_seed(500 + case)
    if case % 4 == 1:  # partial lists
        swa_lens = torch.randint(1, SWA_W + 1, (tokens,), dtype=torch.int32, generator=g)
        extra_lens = torch.randint(0, TOPK + 1, (tokens,), dtype=torch.int32, generator=g)
    if case % 4 == 2:  # a token with nothing in the compressed cache, another with a short window
        extra_lens[0] = 0
        swa_lens[-1] = 7
    return swa_lens, extra_lens


def case_inputs(case, dev):
    gen = torch.Generator(device=dev).manual_seed(100 + case)
    # tokens: the served decode batches (1-8), and every 16th case a batch of 40 or 64 tokens, where FlashInfer's plan
    # gives a CTA several tiles per slot (the producers reuse the shared-memory slots: the handoff protocol's waits)
    tokens = [6, 6, 6, 1, 2, 8, 5, 3][case % 8]
    if case % 16 == 4:
        tokens = 40
    if case % 16 == 12:
        tokens = 64
    swa_lens, extra_lens = case_lens(case, tokens)
    inp = make_inputs(tokens, gen, dev, swa_lens, extra_lens)
    if case % 8 == 7:  # SWA-only layer (no compressed cache): FlashInfer's ordinary schedule, not the extension's
        inp["extra"] = None
    return inp


def run_flashinfer(inp, out, workspace, scale):
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

    if inp["extra"] is None:
        return trtllm_batch_decode_sparse_mla_dsv4(
            query=inp["q"], swa_kv_cache=inp["swa"], workspace_buffer=workspace,
            sparse_indices=inp["swa_idx"].view(-1, 1, SWA_W), compressed_kv_cache=None, out=out, bmm1_scale=scale,
            sinks=inp["sinks"], kv_layout="NHD", swa_topk_lens=inp["swa_lens"], extra_sparse_indices=None,
            extra_sparse_topk_lens=None, kv_cache_format="fp8_dsv41", kv_scale_placement="interleaved")
    return trtllm_batch_decode_sparse_mla_dsv4(
        query=inp["q"], swa_kv_cache=inp["swa"], workspace_buffer=workspace,
        sparse_indices=inp["swa_idx"].view(-1, 1, SWA_W), compressed_kv_cache=inp["extra"], out=out, bmm1_scale=scale,
        sinks=inp["sinks"], kv_layout="NHD", swa_topk_lens=inp["swa_lens"],
        extra_sparse_indices=inp["extra_idx"].view(-1, 1, TOPK), extra_sparse_topk_lens=inp["extra_lens"],
        kv_cache_format="fp8_dsv41_fp4_ca", kv_scale_placement="interleaved")


class Spy:
    """Captures the arguments of the prepared plan's execute (what reaches the FlashInfer C++ binding)."""

    def __init__(self):
        import flashinfer.mla._sparse_mla_sm120._prepared as P

        self.P = P
        self.orig = P.PreparedCall.execute
        self.last = None

    def __enter__(self):
        spy = self

        def execute(this, q, cache, indices, output, scale, lengths, sink, extra, extra_indices, extra_lengths,
                    mid=None, mlse=None, lse=None, lse_scale=1.0):
            spy.last = dict(plan=dict(this.plan.inspect()), q=q, cache=cache, indices=indices, output=output,
                            scale=scale, lengths=lengths, sink=sink, extra=extra, extra_indices=extra_indices,
                            extra_lengths=extra_lengths, mid=this.mid if mid is None else mid,
                            mlse=this.mlse if mlse is None else mlse, lse=this.lse if lse is None else lse,
                            lse_scale=lse_scale)
            return spy.orig(this, q, cache, indices, output, scale, lengths, sink, extra, extra_indices,
                            extra_lengths, mid, mlse, lse, lse_scale)

        self.P.PreparedCall.execute = execute
        return self

    def __exit__(self, *a):
        self.P.PreparedCall.execute = self.orig


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def subprocess_outputs(cases, path):
    """Child: the public API only, with the patched execute routing to the extension; the outputs go to `path`."""
    dev = torch.device("cuda")
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device=dev)
    import flashinfer.mla._sparse_mla_sm120._prepared as P

    assert hasattr(P, "_moet_sparse_mla_ldg_on"), "FlashInfer's _prepared.py is not patched (step 28)"
    assert P._moet_sparse_mla_ldg_on(), "VLLM_MOET_SPARSE_MLA_LDG=1 did not enable the extension (import failed?)"
    outs = {}
    for case in range(cases):
        inp = case_inputs(case, dev)
        out = torch.zeros(inp["q"].shape[0], H, D, dtype=torch.bfloat16, device=dev)
        run_flashinfer(inp, out, workspace, D ** -0.5)
        torch.cuda.synchronize()
        outs[str(case)] = out.view(torch.int16).cpu()
    torch.save(outs, path)
    import sparse_mla_ldg_sm120 as M

    mixed = sum(1 for c in range(cases) if c % 8 != 7)
    assert M.CALLS == mixed, f"the extension served {M.CALLS} of the {mixed} mixed-cache calls"
    print(f"child: the patched execute routed {M.CALLS}/{mixed} mixed-cache calls to the extension, "
          f"{cases - mixed} SWA-only calls to FlashInfer; plans recorded: {len(P._MOET_LDG_PLANS)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-subprocess", action="store_true")
    ap.add_argument("--child", type=str, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        subprocess_outputs(a.cases, a.child)
        return
    import sparse_mla_ldg_sm120 as M

    ext = M._ext()
    dev = torch.device("cuda")
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device=dev)
    scale = D ** -0.5
    refs = {}
    ok = tot = 0
    first_bad = None
    plans_seen = set()
    for case in range(a.cases):
        inp = case_inputs(case, dev)
        tokens = inp["q"].shape[0]
        out_ref = torch.zeros(tokens, H, D, dtype=torch.bfloat16, device=dev)
        with Spy() as spy:
            run_flashinfer(inp, out_ref, workspace, scale)
            torch.cuda.synchronize()
        refs[str(case)] = out_ref.view(torch.int16).cpu()
        last = spy.last
        if inp["extra"] is None:
            assert not M.supported(last["plan"], last["q"], last["extra"], last["extra_indices"])
            continue  # the extension does not serve SWA-only plans; the subprocess checks the fall-through
        assert M.supported(last["plan"], last["q"], last["extra"], last["extra_indices"]), last["plan"]
        ns = last["plan"]["scratch_split_stride"]
        plans_seen.add((tokens, int(last["plan"]["cpb"]), ns))
        mid_ref, mlse_ref, lse_ref = last["mid"].clone(), last["mlse"].clone(), last["lse"].clone()
        out = torch.full_like(out_ref, float("nan"))
        out_lse = torch.full_like(lse_ref, float("nan"))
        mid = torch.full_like(mid_ref, float("nan"))
        mlse = torch.full_like(mlse_ref, float("nan"))
        M.decode(last["plan"], last["q"], last["cache"], last["indices"], mid, mlse, out, out_lse, last["scale"],
                 last["lengths"], last["sink"], last["extra"], last["extra_indices"], last["extra_lengths"],
                 last["lse_scale"], True)
        torch.cuda.synchronize()
        n_mid, n_lse = tokens * H * ns * D, tokens * H * ns
        valid = (mlse_ref[:n_lse] > -1e29).view(tokens, H, ns, 1).expand(tokens, H, ns, D).reshape(-1)
        same = (torch.equal(out.view(torch.int16), out_ref.view(torch.int16)),
                torch.equal(out_lse.view(torch.int32), lse_ref.view(torch.int32)),
                torch.equal(mid[:n_mid].view(torch.int16)[valid], mid_ref[:n_mid].view(torch.int16)[valid]),
                torch.equal(mlse[:n_lse].view(torch.int32), mlse_ref[:n_lse].view(torch.int32)))
        tot += 1
        ok += all(same)
        if not all(same) and first_bad is None:
            first_bad = (case, same)
    print(f"bits: {ok}/{tot} mixed-cache cases identical to FlashInfer (output, LSE, mid_out of the non-empty splits, "
          f"mid_lse); plans seen: {sorted(plans_seen)}" + (f"; first difference: case {first_bad[0]} (output, lse, mid, mlse) equal = {first_bad[1]}"
                         if first_bad else ""))
    assert ok == tot, "bit mismatch"

    if not a.no_subprocess:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "child_outputs.pt")
            env = dict(os.environ, VLLM_MOET_SPARSE_MLA_LDG="1", VLLM_MOET_SPARSE_MLA_LDG_DIR=str(HERE))
            cases = min(a.cases, 16)
            r = subprocess.run([sys.executable, __file__, "--cases", str(cases), "--child", path], env=env,
                               capture_output=True, text=True)
            print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-2000:])
            assert r.returncode == 0, r.stderr[-4000:]
            child = torch.load(path)
            same = sum(torch.equal(child[str(c)], refs[str(c)]) for c in range(cases))
            print(f"patched execute path (subprocess, VLLM_MOET_SPARSE_MLA_LDG=1): {same}/{cases} cases identical to "
                  f"the unpatched run, incl. the SWA-only layers served by FlashInfer")
            assert same == cases

    # timing: the served C1 shape in CUDA graphs, cold L2 before each replay
    inp = case_inputs(0, dev)
    out_ref = torch.zeros(6, H, D, dtype=torch.bfloat16, device=dev)
    with Spy() as spy:
        run_flashinfer(inp, out_ref, workspace, scale)
        torch.cuda.synchronize()
    last = spy.last
    out = torch.zeros_like(out_ref)
    out_lse = torch.zeros_like(last["lse"])
    mid = torch.zeros_like(last["mid"])
    mlse = torch.zeros_like(last["mlse"])
    g_ref = capture(lambda: run_flashinfer(inp, out_ref, workspace, scale))
    g_new = capture(lambda: M.decode(last["plan"], last["q"], last["cache"], last["indices"], mid, mlse, out, out_lse,
                                     last["scale"], last["lengths"], last["sink"], last["extra"],
                                     last["extra_indices"], last["extra_lengths"], last["lse_scale"], True))
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    from torch.profiler import ProfilerActivity, profile

    def timed(g):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(a.iters):
                flush.view(torch.int32).sum()
                g.replay()
            torch.cuda.synchronize()
        per = {}
        for e in prof.events():
            if e.device_type.name == "CUDA" and "sparse_mla" in e.name:
                per.setdefault("merge" if "merge" in e.name else "decode", []).append(e.device_time)
        return {k: statistics.median(v) for k, v in per.items()}

    k_ref, k_new = timed(g_ref), timed(g_new)
    print(f"graph replay (6 tokens, 128 + 512 candidates, 10 splits, cold L2): FlashInfer decode {k_ref['decode']:.2f} "
          f"+ merge {k_ref['merge']:.2f} us; load-based gather decode {k_new['decode']:.2f} + hoisted merge "
          f"{k_new['merge']:.2f} us")
    print("  outputs after the graph replays identical:", torch.equal(out.view(torch.int16), out_ref.view(torch.int16)))
    assert torch.equal(out.view(torch.int16), out_ref.view(torch.int16))


if __name__ == "__main__":
    main()
