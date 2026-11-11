#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU test (one sm_120 GPU, serving image) of the fast prefill MoE kernels against b12x's served `dynamic` path AND
against spec B (../b12x_moe_prefill_spec), bit for bit, with the spec harness's cases:

  python3 test_b12x_moe_prefill_fast_sm120.py [--dumps DIR] [--synthetic-only] [--rows N] [--cfg1 C] [--cfg2 C]
                                              [--dynamic 0|1] [--repeats 3] [--all-cfgs]

1. Synthetic (the decode test's experts, random bf16 tokens, random top-6 routings with -1 pairs) for T in
   (16, 64, 300, 1024) through the served plan (cap 4096, B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1).
2. The served model's dumps (weights_layer2.pt + call*_layer2.pt, 4096 rows): b12x's re-run must reproduce the dumped
   output; ours must equal b12x and the spec.
Per case: route rows of the valid pairs (torch.equal on int16 views) vs b12x and vs spec, output vs b12x / spec / dump,
the intermediates vs spec (a_values, a_scales, the MXFP8 intermediate rows and scale bytes of the valid pairs, and the
expert-sorted row_pair table vs the spec's expert_sorted_rows), determinism: `--repeats` runs identical (route rows,
output, intermediate), and - when the extension has the entry - the pre-quantized rows path (the run's own a_values /
a_scales fed back, the quant stage skipped) gives the same route rows / output / intermediate. Exit 0 only if everything
is identical.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b12x_moe_prefill_spec"))
sys.path.insert(0, str(HERE.parent / "b12x_moe_fused"))

E, H, N, TOPK, LIMIT = 384, 5120, 576, 6, 10.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=os.environ.get("VLLM_MOET_DUMPS", "dumps"),
                    help="directory with the served model's MoE dumps (patch_vllm_b12x_moe_dump_sm120.py); env VLLM_MOET_DUMPS")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--cfg1", type=int, default=None)
    ap.add_argument("--cfg2", type=int, default=None)
    ap.add_argument("--dynamic", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--all-cfgs", action="store_true", help="run the dump case through every (cfg1, cfg2, dynamic)")
    a = ap.parse_args()
    os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
    import torch
    from b12x_moe_prefill_fast_sm120 import moe_prefill_fast, prequant_rows_supported, strided_rows_supported
    try:  # K-12: with SUM_ITEMS=1 + SUM_DISCARD=1 the route rows are summed and dropped in-kernel (scratch): compare the output only
        from b12x_moe_prefill_fast_sm120 import route_rows_consumed
        route_scratch = bool(route_rows_consumed())
    except ImportError:
        route_scratch = False
    from b12x_moe_prefill_spec_sm120 import expert_sorted_rows, moe_prefill_spec
    from test_b12x_moe_prefill_spec_sm120 import _planes, _served
    import b12x.moe.fused_moe._impl as impl

    dev = torch.device("cuda")
    torch.manual_seed(0)
    spied = {}
    orig_dyn = impl._launch_dynamic

    def spy_dyn(**kw):
        spied.clear()
        spied.update(kw)
        return orig_dyn(**kw)

    impl._launch_dynamic = spy_dyn
    kw_fast = dict(cfg1=a.cfg1, cfg2=a.cfg2, dynamic=a.dynamic)

    def same(x, y):
        return torch.equal(x.view(torch.int16) if x.dtype == torch.bfloat16 else x,
                           y.view(torch.int16) if y.dtype == torch.bfloat16 else y)

    def sentinel_route(T):
        # A route buffer of our own, every element a bf16 pattern no route row holds in all 5 120 columns: a call that does not write a
        # route row cannot pass the comparison on what an earlier call left in a recycled allocation (the extension's own buffer for the
        # pre-quantized / strided entries is often the allocation of the temporary `route_m[valid]` freed just before it - the same rows).
        r = torch.empty(T * TOPK, H, dtype=torch.bfloat16, device=dev)
        r.view(torch.int16).fill_(0x5A5A)
        return r

    def compare(tag, x, ids, w, served, out_ref=None, kw=None):
        kw = kw_fast if kw is None else kw
        T = int(x.shape[0])
        ok = True
        msg = [f"{tag}: T={T}"]
        out_b = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        served.run(x, ids, w, out_b)
        torch.cuda.synchronize()
        tb = (time.perf_counter() - t0) * 1e3
        route_b = spied["workspace"].route_output.reshape(-1)[: T * TOPK * H].view(T * TOPK, H).clone()
        planes = _planes(torch, served.prepared, spied)
        out_s, route_s, av_s, as_s, inter_s, isf_s = moe_prefill_spec(x, ids, w, planes, LIMIT)
        torch.cuda.synchronize()
        runs = []
        for r in range(max(1, a.repeats)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = moe_prefill_fast(x, ids, w, planes, LIMIT, **kw)
            torch.cuda.synchronize()
            tm = (time.perf_counter() - t0) * 1e3
            runs.append((res, tm))
        (out_m, route_m, av_m, as_m, inter_m, isf_m, row_pair_m, counts_m), tm = runs[0]
        flat = ids.reshape(-1)
        valid = torch.nonzero(flat >= 0).flatten()
        # routing table vs the spec's host-side sort (same order: pairs grouped by expert in pair order, padded to 16)
        rp_s, _ = expert_sorted_rows(ids.to(torch.int32), E)
        rows_padded = int(counts_m[0].item())
        rp_ok = rows_padded == rp_s.numel() and torch.equal(row_pair_m[:rows_padded], rp_s)
        if rows_padded != rp_s.numel():
            msg.append(f"row_pair rows {rows_padded} != spec {rp_s.numel()}")
        msg.append(f"row_pair {'identical' if rp_ok else 'DIFFER'} ({rows_padded} rows, {int(counts_m[1])} FC1 / {int(counts_m[2])} FC2 items)")
        ok &= rp_ok
        # the front-end quant and the intermediate vs the spec
        q_ok = same(av_m, av_s) and same(as_m, as_s)
        inter_ok = same(inter_m[valid, :N], inter_s[valid, :N])
        isf_m8 = isf_m.view(torch.uint8).view(-1, 20)[valid, :18]
        isf_s8 = isf_s.view(torch.uint8).view(-1, 20)[valid, :18]
        isf_ok = same(isf_m8, isf_s8)
        msg.append(f"vs spec: quant {'ok' if q_ok else 'DIFFER'}, intermediate {'ok' if inter_ok else 'DIFFER'}, scales {'ok' if isf_ok else 'DIFFER'}")
        ok &= q_ok and inter_ok and isf_ok
        if not inter_ok:
            d = (inter_m[valid, :N].int() != inter_s[valid, :N].int())
            bad = valid[d.any(dim=1)][:3].tolist()
            msg.append(f"  intermediate rows differing: {int(d.any(dim=1).sum())}, first {bad}")
        # route rows and output vs b12x and vs the spec
        for name, route_r, out_r in (("b12x", route_b, out_b), ("spec", route_s, out_s)):
            same_rows = (route_m[valid].view(torch.int16) == route_r[valid].view(torch.int16)).all(dim=1)
            r_ok = bool(same_rows.all()) or route_scratch
            o_ok = same(out_m, out_r)
            rr = "route rows consumed in-kernel (scratch, not compared)" if route_scratch else f"route rows {int(same_rows.sum())}/{valid.numel()}"
            msg.append(f"vs {name}: {rr}, output {'IDENTICAL' if o_ok else 'DIFFER'}")
            if not r_ok:
                for p in valid[~same_rows][:3].tolist():
                    d = (route_m[p].float() - route_r[p].float()).abs()
                    msg.append(f"  pair {p} (tok {p // TOPK} slot {p % TOPK} expert {int(flat[p])}): {int((d > 0).sum())} cols differ, max {d.max().item():.3e}")
            if not o_ok:
                d = (out_m.float() - out_r.float()).abs()
                msg.append(f"  {int((d > 0).sum())}/{d.numel()} elements, max {d.max().item():.3e}; first rows {torch.nonzero(d.sum(1) > 0).flatten()[:5].tolist()}")
            ok &= r_ok and o_ok
        if out_ref is not None:
            okr = same(out_b, out_ref)
            okm = same(out_m, out_ref)
            msg.append(f"served dump: b12x re-run {'IDENTICAL' if okr else 'DIFFER'}, ours {'IDENTICAL' if okm else 'DIFFER'}")
            ok &= okm
        # determinism (per component, with the first differing pair)
        det = True
        for ri, (r, _) in enumerate(runs[1:], 1):
            parts = []
            for name, x, y in (("a_values", r[2], av_m), ("a_scales", r[3], as_m), ("row_pair", r[6][:rows_padded], row_pair_m[:rows_padded]),
                               ("counts", r[7][:3], counts_m[:3]), ("inter", r[4][valid, :N], inter_m[valid, :N]),
                               ("inter_sf", r[5].view(-1, 20)[valid, :18], isf_m8), ("route", r[1][valid], route_m[valid]),
                               ("out", r[0], out_m)):
                if name == "route" and route_scratch:
                    continue
                if not same(x, y):
                    xi = x.view(torch.int16) if x.dtype == torch.bfloat16 else x
                    yi = y.view(torch.int16) if y.dtype == torch.bfloat16 else y
                    dm = (xi != yi)
                    nrows = int(dm.any(dim=-1).sum()) if dm.dim() > 1 else int(dm.sum())
                    first = torch.nonzero(dm.any(dim=-1) if dm.dim() > 1 else dm).flatten()[:4].tolist()
                    extra = ""
                    if name in ("route", "inter") and first:
                        p0 = int(valid[first[0]])
                        extra = f" pair {p0} tok {p0 // TOPK} slot {p0 % TOPK} expert {int(flat[p0])}"
                        if name == "route":
                            d = (r[1][p0].float() - route_m[p0].float()).abs()
                            extra += f" cols differ {int((d > 0).sum())} max {d.max().item():.3e} first cols {torch.nonzero(d > 0).flatten()[:4].tolist()}"
                    parts.append(f"{name} {nrows} rows differ (first {first}{extra})")
            if parts:
                det = False
                msg.append(f"run {ri}: " + "; ".join(parts))
        msg.append(f"{len(runs)} runs {'identical' if det else 'DIFFER'}")
        ok &= det
        # the pre-quantized-rows entry (the MXFP8 all-gather): the same rows fed back as a_values / a_scales must give
        # the same route rows and output (the quant stage is skipped, everything else is the same code)
        if prequant_rows_supported():
            res_q = moe_prefill_fast(None, ids, w, planes, LIMIT, route_out=sentinel_route(T), a_values=av_m, a_scales=as_m, **kw)
            torch.cuda.synchronize()
            pq_ok = same(res_q[0], out_m) and (route_scratch or same(res_q[1][valid], route_m[valid])) and same(res_q[4][valid, :N], inter_m[valid, :N])
            msg.append(f"pre-quantized rows entry {'identical' if pq_ok else 'DIFFER'}")
            ok &= pq_ok
            # the same rows as row-strided views of ONE buffer (the one-all-gather layout: per row the K e4m3 bytes,
            # the K/32 scale bytes, then 48 bytes of top-k ids / weights - here random garbage the kernels must not read)
            if strided_rows_supported():
                Tq, Kq = av_m.shape
                buf = torch.randint(0, 256, (Tq, Kq + Kq // 32 + 48), device=av_m.device, dtype=torch.uint8)
                buf[:, :Kq] = av_m
                buf[:, Kq:Kq + Kq // 32] = as_m
                res_s = moe_prefill_fast(None, ids, w, planes, LIMIT, route_out=sentinel_route(T), a_values=buf[:, :Kq],
                                         a_scales=buf[:, Kq:Kq + Kq // 32], **kw)
                torch.cuda.synchronize()
                st_ok = same(res_s[0], out_m) and (route_scratch or same(res_s[1][valid], route_m[valid])) and same(res_s[4][valid, :N], inter_m[valid, :N])
                msg.append(f"strided rows (one-buffer layout, row stride {Kq + Kq // 32 + 48}) {'identical' if st_ok else 'DIFFER'}")
                ok &= st_ok
        msg.append(f"time b12x {tb:.2f} ms, ours {min(t for _, t in runs):.2f} ms (eager, host included)")
        print(" | ".join(msg))
        sys.stdout.flush()
        return ok

    results = []
    if not a.all_cfgs:
        from test_b12x_moe_fused_sm120 import _weights
        w1, w1s, w2, w2s = _weights(torch, dev)
        raw = dict(w1_fp4=w1, w1_blockscale=w1s, w2_fp4=w2, w2_blockscale=w2s)
        served = _served(torch, dev, raw)
        g = torch.Generator(device=dev).manual_seed(1)
        for T in (16, 64, 300, 1024):
            x = (torch.randn(T, H, device=dev, dtype=torch.float32, generator=g) * 0.5).to(torch.bfloat16)
            ids = torch.randint(0, E, (T, TOPK), device=dev, dtype=torch.int32, generator=g)
            if T >= 64:
                ids[5, 1] = -1
                ids[7] = -1
            w = (torch.rand(T, TOPK, device=dev, generator=g) * 0.4 + 0.05).to(torch.float32)
            results.append(compare("synthetic", x, ids, w, served))
        if a.synthetic_only:
            print(f"{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} cases identical")
            return 0 if all(results) else 1
    d = Path(a.dumps)
    raw = torch.load(d / "weights_layer2.pt", map_location="cpu", weights_only=False)
    served = _served(torch, dev, raw)
    for name in sorted(p.name for p in d.glob("call*_layer2.pt")):
        c = torch.load(d / name, map_location="cpu", weights_only=False)
        x, ids, w, out_ref = c["hidden_states"].to(dev), c["topk_ids"].to(dev), c["topk_weights"].to(dev), c["output"].to(dev)
        if a.rows:
            x, ids, w, out_ref = x[: a.rows], ids[: a.rows], w[: a.rows], out_ref[: a.rows]
        if a.all_cfgs:
            if name.startswith("call0"):
                continue
            for cfg1 in range(2):
                for cfg2 in range(2):
                    for dyn in (1, 0):
                        results.append(compare(f"dump {name} cfg1={cfg1} cfg2={cfg2} dyn={dyn}", x, ids, w,
                                               served, out_ref, kw=dict(cfg1=cfg1, cfg2=cfg2, dynamic=dyn)))
        else:
            results.append(compare(f"dump {name}", x, ids, w, served, out_ref))
    print(f"{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} cases identical")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
