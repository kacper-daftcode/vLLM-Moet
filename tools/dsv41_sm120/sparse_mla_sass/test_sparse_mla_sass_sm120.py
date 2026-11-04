#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bit test + timing of the cubit-assembled `sparse_mla_prefill_v41` against spec B (FlashInfer 0.7.2
`sparse_mla_prefill_dsv41_bf16_kernel<16,true>` via SparseMLASm120Wrapper(compute_precision="bf16")).

  --synthetic   bench-like data (256K-token pools of 528-B fp8 / 288-B FP4 records, interleaved scales, random indices,
                partial lengths, -1 holes, sink; shapes 128/192/1152 + 512, pbs 128/64, SWA-only): output and lse torch.equal
  dumps         attn_*.pt from patch_vllm_sparse_mla_prefill_dump_sm120.py (+ the spec-B .ref.pt of replay_specB_dumps.py)
  --time        CUDA-graph medians (20 replays, cold 256K pools) of spec B and the SASS kernel per prod shape

usage (inside the image, one GPU): python3 test_sparse_mla_sass_sm120.py --cubin X.cubin [--synthetic] [--time] [dumps...]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("NVFP4_KV_DIR", os.path.join(HERE, "..", "nvfp4_kv")))
from sass_launcher import SparseMlaSass  # noqa: E402

D_QK = D_V = 512
HEADS = 16


def make_wrapper(has_extra: bool):
    from flashinfer.mla import SparseMLASm120Wrapper

    return SparseMLASm120Wrapper(kv_scale_format="ue8m0_g32", kv_cache_format="fp8", extra_kv_fp4=bool(has_extra),
                                 compute_precision="bf16", backend="sparse", kv_scale_placement="interleaved")


def run_specB(w, q, swa, idx, lens, sink, sm_scale, ext=None, eidx=None, elens=None):
    out = torch.zeros(q.shape[0], HEADS, D_V, dtype=torch.bfloat16, device=q.device)
    lse = w.run(q, swa, idx, out, sm_scale, topk_length=lens, attn_sink=sink, extra_kv_cache=ext, extra_indices=eidx,
                extra_topk_length=elens, return_lse=True)
    torch.cuda.synchronize()
    return out, lse.clone()


def run_sass(k, q, swa, idx, lens, sink, sm_scale, ext=None, eidx=None, elens=None):
    out = torch.zeros(q.shape[0], HEADS, D_V, dtype=torch.bfloat16, device=q.device)
    lse = torch.full((q.shape[0], HEADS), 12345.0, dtype=torch.float32, device=q.device)
    k.run(q, swa, idx, out, sm_scale, topk_length=lens, attn_sink=sink, extra_kv_cache=ext, extra_indices=eidx,
          extra_topk_length=elens, out_lse=lse)
    torch.cuda.synchronize()
    return out, lse


def compare(name, out_s, lse_s, out_r, lse_r) -> bool:
    eq_o, eq_l = torch.equal(out_s, out_r), torch.equal(lse_s, lse_r)
    d = (out_s.float() - out_r.float()).abs()
    dl = (lse_s - lse_r).abs()
    bad = int((d > 0).sum())
    rows_bad = int(((d > 0).flatten(1).any(1)).sum())
    print(f"{'PASS' if eq_o and eq_l else 'FAIL'} {name}: output {'IDENTICAL' if eq_o else 'DIFFERS'} "
          f"(max|d| {d.max().item():.3e}, {bad} elements in {rows_bad} rows), lse {'IDENTICAL' if eq_l else 'DIFFERS'} "
          f"(max|d| {dl.max().item():.3e}, {int((dl > 0).sum())} values)")
    return eq_o and eq_l


def to_interleaved(footer, pbs, data_bytes, scale_bytes):
    pages = footer.view(-1, pbs * (data_bytes + scale_bytes))
    data = pages[:, : pbs * data_bytes].view(-1, pbs, data_bytes)
    scales = pages[:, pbs * data_bytes:].view(-1, pbs, scale_bytes)
    return torch.cat([data, scales], dim=-1).contiguous().view(-1, pbs, 1, data_bytes + scale_bytes)


def time_us(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def synthetic(k, dev, do_time: bool, pool_log2: int, quick: bool = False) -> bool:
    from test_flashinfer_dsv41_fp4_extra import our_insert  # noqa: E402

    torch.manual_seed(11)
    n_pool = 1 << pool_log2
    main_bf16 = (torch.randn(n_pool, D_QK, device=dev, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    extra_bf16 = (torch.randn(n_pool, D_QK, device=dev, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    swa = {pbs: to_interleaved(our_insert(main_bf16, pbs, 528).view(-1, pbs, 1, 528), pbs, 512, 16) for pbs in (128, 32)}
    ext = {pbs: to_interleaved(our_insert(extra_bf16, pbs, 288).view(-1, pbs, 1, 288), pbs, 256, 32) for pbs in (128, 64)}
    del main_bf16, extra_bf16
    torch.cuda.synchronize()
    sm_scale = D_QK ** -0.5
    ok = True
    # (label, rows, topk, swa pbs, extra pbs or None, partial lengths, holes, sink)
    cases = [
        ("text 128+512 p128 full", 4096, 128, 128, 128, False, False, True),
        ("text 128+512 p128 partial+holes", 1536, 128, 128, 128, True, True, True),
        ("dspark 192+512 p128", 1024, 192, 128, 128, True, False, True),
        ("vision 1152+512 p128", 512, 1152, 128, 128, True, False, True),
        ("text 128+512 p64 (cr=2)", 1024, 128, 128, 64, True, True, True),
        ("swa-only 128 p32", 1100, 128, 32, None, True, True, True),
        ("no sink 128+512", 700, 128, 128, 128, True, False, False),
        ("empty rows (len 0) 128+512", 300, 128, 128, 128, "zero", True, True),
    ]
    if quick:
        cases = [("quick 128+512 p128", 16, 128, 128, 128, True, True, True)]
    for label, rows, topk, spbs, epbs, partial, holes, with_sink in cases:
        s_cache = swa[spbs]
        q = (torch.randn(rows, HEADS, D_QK, device=dev, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
        idx = torch.randint(0, n_pool, (rows, topk), device=dev, dtype=torch.int32)
        if holes:
            idx[torch.rand(rows, topk, device=dev) < 0.02] = -1
        lens = torch.full((rows,), topk, dtype=torch.int32, device=dev)
        if partial:
            lens = torch.randint(0, topk + 1, (rows,), device=dev, dtype=torch.int32)
            lens[::3] = topk
        if partial == "zero":
            lens[::2] = 0
        sink = torch.randn(HEADS, device=dev, dtype=torch.float32) if with_sink else None
        if epbs is not None:
            e_cache = ext[epbs]
            eidx = torch.randint(0, n_pool, (rows, 512), device=dev, dtype=torch.int32)
            if holes:
                eidx[torch.rand(rows, 512, device=dev) < 0.01] = -1
            elens = torch.full((rows,), 512, dtype=torch.int32, device=dev)
            if partial:
                elens = torch.randint(0, 513, (rows,), device=dev, dtype=torch.int32)
                elens[1::2] = 512
            if partial == "zero":
                elens[::2] = 0
            extra = (e_cache, eidx, elens)
        else:
            extra = (None, None, None)
        w = make_wrapper(epbs is not None)
        out_r, lse_r = run_specB(w, q, s_cache, idx, lens, sink, sm_scale, *extra)
        out_s, lse_s = run_sass(k, q, s_cache, idx, lens, sink, sm_scale, *extra)
        out_s2, lse_s2 = run_sass(k, q, s_cache, idx, lens, sink, sm_scale, *extra)
        det = torch.equal(out_s, out_s2) and torch.equal(lse_s, lse_s2)
        ok &= compare(f"synthetic {label} (rows {rows})", out_s, lse_s, out_r, lse_r) and det
        if not det:
            print(f"FAIL {label}: SASS kernel not deterministic")
    if do_time:
        nt = 4096
        sink = torch.randn(HEADS, device=dev, dtype=torch.float32)
        for label, topk, epbs in (("128+512 p128", 128, 128), ("192+512 p128", 192, 128), ("1152+512 p128", 1152, 128),
                                  ("128+512 p64", 128, 64)):
            q = (torch.randn(nt, HEADS, D_QK, device=dev, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
            idx = torch.randint(0, n_pool, (nt, topk), device=dev, dtype=torch.int32)
            eidx = torch.randint(0, n_pool, (nt, 512), device=dev, dtype=torch.int32)
            lens = torch.full((nt,), topk, dtype=torch.int32, device=dev)
            elens = torch.full((nt,), 512, dtype=torch.int32, device=dev)
            w = make_wrapper(True)
            out = torch.zeros(nt, HEADS, D_V, dtype=torch.bfloat16, device=dev)
            lse = torch.zeros(nt, HEADS, dtype=torch.float32, device=dev)
            e_cache = ext[epbs]

            def f_ref():
                w.run(q, swa[128], idx, out, sm_scale, topk_length=lens, attn_sink=sink, extra_kv_cache=e_cache,
                      extra_indices=eidx, extra_topk_length=elens)

            def f_sass():
                k.run(q, swa[128], idx, out, sm_scale, topk_length=lens, attn_sink=sink, extra_kv_cache=e_cache,
                      extra_indices=eidx, extra_topk_length=elens, out_lse=lse)

            us_r, us_s = time_us(f_ref), time_us(f_sass)
            print(f"TIME {label}: spec B {us_r:.0f} us, SASS {us_s:.0f} us, ratio {us_s / us_r:.3f}")
    return ok


def dumps(k, dev, paths) -> bool:
    ok = True
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        ref_path = p[:-3] + ".ref.pt"
        has_extra = "extra_cache" in d
        q = d["q"].to(dev)
        swa_cache, swa_idx, swa_lens = d["swa_cache"].to(dev), d["swa_indices"].to(dev), d["swa_lens"].to(dev)
        sink = None if d["attn_sink"] is None else d["attn_sink"].to(dev)
        extra = (d["extra_cache"].to(dev), d["extra_indices"].to(dev), d["extra_lens"].to(dev)) if has_extra else (None, None, None)
        if os.path.exists(ref_path):
            r = torch.load(ref_path, map_location="cpu", weights_only=False)
            out_r, lse_r = r["output"].to(dev), r["lse"].to(dev)
        else:
            out_r, lse_r = run_specB(make_wrapper(has_extra), q, swa_cache, swa_idx, swa_lens, sink, d["sm_scale"], *extra)
        out_s, lse_s = run_sass(k, q, swa_cache, swa_idx, swa_lens, sink, d["sm_scale"], *extra)
        ok &= compare(f"dump {os.path.basename(p)} (f{d['forward']} l{d['layer']} cr{d['compress_ratio']} rows {d['rows']})",
                      out_s, lse_s, out_r, lse_r)
        ok &= torch.equal(out_s, d["output"].to(dev)) or print(f"FAIL {p}: SASS output != dumped output") is not None and False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--pool-log2", type=int, default=18)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("dumps", nargs="*")
    args = ap.parse_args()
    dev = torch.device("cuda")
    k = SparseMlaSass(args.cubin)
    ok = True
    if args.synthetic or args.time:
        ok &= synthetic(k, dev, args.time, args.pool_log2, args.quick)
    if args.dumps:
        ok &= dumps(k, dev, args.dumps)
    print("PASS all" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
