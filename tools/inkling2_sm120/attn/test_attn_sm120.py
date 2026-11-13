#!/usr/bin/env python3
"""Op-level test of the sm_120 relative attention kernel (inkling2_attn_sm120.cu) against the operation's definition
in float64 (`inkling2_attn_sm120.reference`).

    python3 test_attn_sm120.py [--bench] [--quick]

Every case builds a paged K/V cache with a shuffled page table, query rows, the sheared relative-logit table as the
attention prologue lays it out, gates and sinks, and runs the kernel through the two entries vLLM calls:

  decode   `decode_attention` - one split (the kernel writes the gated bf16 output) and several splits (fp32
           partials merged and gated by the combine kernel); full and window layers; the draft mode (no relative
           logits, gate or sink, every query at the request's last position);
  prefill  `prefill_attention` - fp32 output and natural-log LSE; several requests of different lengths in one
           batch, with and without earlier context.

What must not be read is poison: cache slots past a request's last key and the pages of other requests' padding
are NaN, the table's columns of invisible keys are NaN, block-table entries past a request's pages point at a NaN
page. One NaN in an output fails the case.

Criterion. The kernel rounds the softmax weights to bf16 before P @ V (as vLLM's kernels and FA4 do), so an output
element may differ from the float64 value by 2^-9 of sum(p |v|) <= 2^-9 max|V| plus the output's own rounding; the
LSE is computed in fp32 from unrounded weights. A case passes when
  |out - ref| <= 2^-8 max|V| + 2^-8 |ref|     for every element (the gated outputs: times the gate's factor <= 1),
  |lse - ref| <= 2e-3                         for every row,
and the run is repeated once: the two results are bit-identical.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inkling2_attn_sm120 as attn  # noqa: E402

DEV = torch.device("cuda")
D = 128


def build(seed, q_lens, pasts, heads, kv_heads, page, extent, window, draft, extra_reqs=0):
    """A batch: (q, key_cache, value_cache, rel, gate, sinks, block_table, seqlens, cu_q)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_reqs = len(q_lens) + extra_reqs
    seqlens = [p + n for p, n in zip(pasts, q_lens)] + [0] * extra_reqs
    q_lens = list(q_lens) + [0] * extra_reqs
    pasts = list(pasts) + [0] * extra_reqs
    num_tokens = sum(q_lens)
    pages = [-(-s // page) for s in seqlens]
    max_pages = max(max(pages), 1) + 2
    num_blocks = sum(pages) + 3
    perm = torch.randperm(num_blocks - 1, generator=g) + 1  # page 0 is the poison page
    kv = torch.randn((num_blocks, kv_heads, page, 2 * D), generator=g) * 0.5
    kv[0] = float("nan")
    block_table = torch.zeros((num_reqs, max_pages), dtype=torch.int32)
    used = 0
    for r in range(num_reqs):
        block_table[r, : pages[r]] = perm[used : used + pages[r]].to(torch.int32)
        used += pages[r]
        if pages[r] and seqlens[r] % page:
            kv[block_table[r, pages[r] - 1].item(), :, seqlens[r] % page :] = float("nan")
    kv = kv.to(DEV, torch.bfloat16)
    key_cache, value_cache = kv.transpose(1, 2).split(D, dim=-1)
    q = (torch.randn((num_tokens, heads, D), generator=g) * 0.7).to(DEV, torch.bfloat16)
    gate = torch.randn((num_tokens, heads * D), generator=g).to(DEV, torch.bfloat16)
    sinks = (torch.randn((num_tokens, heads), generator=g) * 2).to(DEV, torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
    rel = None
    if not draft:
        pad = extent + 2 * attn.FA_BLOCK
        rel = torch.full((num_tokens + 128, heads, pad), float("nan"))
        t = 0
        for r in range(num_reqs):
            for i in range(q_lens[r]):
                pos = pasts[r] + i
                tile_end = min((i // 128 + 1) * 128, q_lens[r])
                base = -(-(pasts[r] + tile_end) // 128) * 128 - pad
                lo = max(base, 0)
                k = torch.arange(lo, pos + 1)
                vals = torch.randn((heads, k.numel()), generator=g)
                dist = pos - k
                if window:
                    vals[:, dist >= window] = float("nan")  # the kernel masks these keys by position
                else:
                    vals[:, dist >= extent] = 0.0  # the table's padding past the extent
                rel[t, :, k - base] = vals
                t += 1
        rel = rel.to(DEV, torch.bfloat16)
    return dict(
        q=q, key_cache=key_cache, value_cache=value_cache, rel=rel, gate=gate, sinks=sinks,
        block_table=block_table.to(DEV), seqlens=torch.tensor(seqlens, dtype=torch.int32, device=DEV), cu=cu.to(DEV),
        heads=heads, kv_heads=kv_heads, window=window, draft=draft, vmax=0.5 * 6, max_q=max(q_lens),
    )  # fmt: skip


def expect(b):
    scale = D**-0.5
    o, lse = attn.reference(
        b["q"], b["key_cache"], b["value_cache"], b["rel"], b["block_table"], b["seqlens"], b["cu"], scale,
        b["window"], b["draft"], dtype=torch.float64,
    )  # fmt: skip
    return o, lse


def check(name, got, ref, bound, lse=None, lse_ref=None):
    assert not torch.isnan(got).any(), f"{name}: NaN in the output"
    err = (got.double() - ref).abs()
    worst = (err - bound).max().item()
    rms = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-30)).item()
    msg = f"max err {err.max().item():.3e} (bound slack {-worst:.3e}), rel rms {rms:.2e}"
    ok = worst <= 0
    if lse is not None:
        finite = torch.isfinite(lse_ref)
        same_inf = bool((torch.isinf(lse) == ~finite).all())
        lerr = (lse.double() - lse_ref)[finite].abs().max().item() if finite.any() else 0.0
        msg += f", lse max err {lerr:.2e}"
        ok = ok and same_inf and lerr <= 2e-3
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: {msg}")
    return ok


def run_decode(b, num_splits, name):
    num_tokens = b["q"].shape[0]
    scale = D**-0.5
    outs = []
    for _ in range(2):
        out = torch.full((num_tokens, b["heads"] * D), float("nan"), dtype=torch.bfloat16, device=DEV)
        attn.decode_attention(
            b["q"], b["key_cache"], b["value_cache"], out, None if b["draft"] else (b["rel"], b["gate"], b["sinks"]),
            b["block_table"], b["seqlens"], b["cu"], b["max_q"], num_tokens, scale, b["window"], num_splits,
        )  # fmt: skip
        outs.append(out)
    torch.cuda.synchronize()
    o, lse = expect(b)
    if b["draft"]:
        ref = o.reshape(num_tokens, -1)
        factor = torch.ones_like(ref)
    else:
        ref = attn.gate_sink(o, lse, b["gate"], b["sinks"])
        factor = (torch.sigmoid(b["gate"].double()).view_as(o) * torch.sigmoid(lse - b["sinks"].double())[:, :, None]).reshape(num_tokens, -1)
    bound = 2.0**-8 * b["vmax"] * factor + 2.0**-8 * ref.abs() + 1e-30
    ok = check(name, outs[0].float(), ref, bound)
    same = torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
    if not same:
        print(f"  FAIL {name}: two runs differ")
    return ok and same


def run_prefill(b, name):
    num_tokens = b["q"].shape[0]
    scale = D**-0.5
    window_size = (b["window"] - 1, 0) if b["window"] else (-1, -1)
    res = []
    for _ in range(2):
        out = torch.full((num_tokens, b["heads"], D), float("nan"), dtype=torch.float32, device=DEV)
        lse_full = torch.full((b["heads"], num_tokens + 7), float("nan"), dtype=torch.float32, device=DEV)
        attn.prefill_attention(
            b["q"], b["key_cache"], b["value_cache"], block_table=b["block_table"], cache_seqlens=b["seqlens"],
            cu_seqlens_q=b["cu"], softmax_scale=scale, window_size=window_size, sheared_bias=b["rel"], out=out,
            lse=lse_full[:, :num_tokens],
        )  # fmt: skip
        res.append((out, lse_full))
    torch.cuda.synchronize()
    o, lse = expect(b)
    bound = 2.0**-8 * b["vmax"] + 2.0**-8 * o.abs()
    ok = check(name, res[0][0], o, bound, res[0][1][:, :num_tokens].t(), lse)
    untouched = bool(torch.isnan(res[0][1][:, num_tokens:]).all())
    same = torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1][:, :num_tokens], res[1][1][:, :num_tokens])
    if not (untouched and same):
        print(f"  FAIL {name}: padding untouched {untouched}, two runs identical {same}")
    return ok and untouched and same


def cases(quick):
    full = dict(heads=12, kv_heads=2, extent=1024, window=0, draft=False)
    local = dict(heads=12, kv_heads=2, extent=1024, window=1024, draft=False)
    ok = True
    print("decode, full layers")
    for seed, (q_lens, pasts, page, splits) in enumerate(
        [
            ([1], [0], 16, [1, 2]),
            ([1], [1], 16, [1]),
            ([1, 1, 1], [63, 64, 127], 16, [1, 3]),
            ([1, 8, 10, 3], [128, 1000, 2047, 5], 16, [1, 2, 7]),
            ([8] * 4, [4000, 3, 700, 1280], 32, [1, 5, 47]),
            ([1] * 6, [9000, 120, 255, 256, 1023, 1024], 128, [1, 4, 94]),
        ][: 3 if quick else None]
    ):
        b = build(seed, q_lens, pasts, page=page, **full)
        for s in splits:
            ok &= run_decode(b, s, f"q {q_lens} past {pasts} page {page} splits {s}")
    print("decode, window layers")
    for seed, (q_lens, pasts, page, splits) in enumerate(
        [
            ([1], [0], 16, [1, 9]),
            ([1, 8, 10], [1023, 1024, 3000], 16, [1, 9, 5]),
            ([8] * 3 + [1], [700, 5000, 1100, 2049], 128, [1, 9, 3, 2]),
        ][: 2 if quick else None]
    ):
        b = build(100 + seed, q_lens, pasts, page=page, **local)
        for s in splits:
            ok &= run_decode(b, s, f"q {q_lens} past {pasts} page {page} splits {s}")
    print("decode, padding requests in the batch")
    b = build(200, [1, 8], [300, 1500], page=16, extra_reqs=3, **full)
    for s in (1, 6):
        ok &= run_decode(b, s, f"two requests + three empty, splits {s}")
    print("draft (no relative logits, gate or sink; every query at the last position)")
    for seed, (pasts, window, page, splits) in enumerate(
        [([0, 100, 3000], 0, 16, [1, 4]), ([0, 1500, 1016, 40], 1024, 16, [1, 9]), ([5000, 7], 1024, 128, [1, 3]),
         ([2500, 1030, 9], 1032, 128, [1, 10])][: 2 if quick else None]  # 1032: the draft's window (1024 + its block)
    ):
        b = build(300 + seed, [8] * len(pasts), pasts, heads=12, kv_heads=2, page=page, extent=0, window=window, draft=True)
        for s in splits:
            ok &= run_decode(b, s, f"past {pasts} window {window} page {page} splits {s}")
    print("prefill")
    for seed, (q_lens, pasts, page, kind) in enumerate(
        [
            ([1], [0], 16, full),
            ([37], [0], 16, full),
            ([300, 1, 7, 129, 64], [0, 500, 2000, 127, 1], 16, full),
            ([300, 1, 7, 129, 64], [0, 500, 2000, 127, 1], 16, local),
            ([1500, 10], [0, 3000], 32, local),
            ([700], [900], 128, full),
            ([257, 2, 2, 2, 2, 2, 1], [1, 0, 0, 0, 0, 0, 0], 16, full),
        ][: 4 if quick else None]
    ):
        b = build(400 + seed, q_lens, pasts, page=page, **kind)
        ok &= run_prefill(b, f"{'window' if kind['window'] else 'full'} q {q_lens} past {pasts} page {page}")
    print("other head counts (TP1: 48 heads / 8 KV heads; a group of 4)")
    b = build(500, [3, 10], [200, 1300], heads=48, kv_heads=8, page=16, extent=1024, window=0, draft=False)
    ok &= run_decode(b, 1, "48 / 8 heads, splits 1") & run_decode(b, 3, "48 / 8 heads, splits 3")
    b = build(501, [5, 16], [200, 1300], heads=8, kv_heads=2, page=16, extent=1024, window=1024, draft=False)
    ok &= run_decode(b, 2, "8 / 2 heads, window, splits 2") & run_prefill(b, "8 / 2 heads, window, prefill")
    return ok


def bench():
    print("timings (median of 20 launches after 5, torch.cuda.synchronize around each; microseconds)")
    scale = D**-0.5

    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        return sorted(ts)[len(ts) // 2]

    full = dict(heads=12, kv_heads=2, extent=1024, window=0, draft=False)
    local = dict(heads=12, kv_heads=2, extent=1024, window=1024, draft=False)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for label, kind, q_lens, pasts in [
        ("decode 1 request x 1 token", local, [1], [4000]),
        ("decode 8 requests x 1 token", local, [1] * 8, [4000] * 8),
        ("verify 1 request x 8 tokens", local, [8], [4000]),
        ("verify 8 requests x 8 tokens", local, [8] * 8, [4000] * 8),
        ("decode 1 request x 1 token", full, [1], [4000]),
        ("decode 1 request x 1 token", full, [1], [32000]),
        ("decode 8 requests x 1 token", full, [1] * 8, [32000] * 8),
        ("verify 8 requests x 8 tokens", full, [8] * 8, [32000] * 8),
    ]:
        b = build(900, q_lens, pasts, page=16, **kind)
        work = len(q_lens) * 2
        if kind["window"]:
            splits = 9 if work <= (16 if max(q_lens) <= 2 else 8) else 5 if work <= 16 else 3 if work <= 32 else 2 if work <= 64 else 1
        else:
            splits = max(sms // work, min(sms // 8, 6 * sms // work))
        out = torch.empty((b["q"].shape[0], 12 * D), dtype=torch.bfloat16, device=DEV)
        us = timed(lambda: attn.decode_attention(
            b["q"], b["key_cache"], b["value_cache"], out, (b["rel"], b["gate"], b["sinks"]), b["block_table"],
            b["seqlens"], b["cu"], max(q_lens), b["q"].shape[0], scale, b["window"], splits))  # fmt: skip
        keys = sum(min(p + n, 1024 + n) if kind["window"] else p + n for p, n in zip(pasts, q_lens))
        print(f"  {'window' if kind['window'] else 'full  '} {label:30s} past {pasts[0]:6d} splits {splits:3d}: {us:8.1f} us"
              f"  ({keys * 2 * 512 / us / 1e6:.2f} TB/s of K|V)")  # fmt: skip
    for label, kind, q_lens, pasts in [
        ("prefill 1 x 2048", local, [2048], [0]),
        ("prefill 1 x 8192", local, [8192], [0]),
        ("prefill 1 x 2048", full, [2048], [0]),
        ("prefill 1 x 8192", full, [8192], [0]),
        ("prefill 1 x 2048 after 30000", full, [2048], [30000]),
    ]:
        b = build(901, q_lens, pasts, page=16, **kind)
        nt = b["q"].shape[0]
        out = torch.empty((nt, 12, D), dtype=torch.float32, device=DEV)
        lse = torch.empty((12, nt), dtype=torch.float32, device=DEV)
        ws = (kind["window"] - 1, 0) if kind["window"] else (-1, -1)
        us = timed(lambda: attn.prefill_attention(
            b["q"], b["key_cache"], b["value_cache"], block_table=b["block_table"], cache_seqlens=b["seqlens"],
            cu_seqlens_q=b["cu"], softmax_scale=scale, window_size=ws, sheared_bias=b["rel"], out=out, lse=lse))  # fmt: skip
        if kind["window"]:
            pairs = sum(min(pasts[0] + i + 1, 1024) for i in range(q_lens[0]))
        else:
            pairs = sum(pasts[0] + i + 1 for i in range(q_lens[0]))
        print(f"  {'window' if kind['window'] else 'full  '} {label:30s}: {us:10.1f} us  ({pairs * 12 * D * 4 / us / 1e6:.1f} TFLOP/s on the visible pairs)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    assert attn.supported(), "needs a capability-12 device"
    torch.manual_seed(0)
    print(f"{torch.cuda.get_device_name(0)}, kernel facts {attn.kernel_facts()}")
    ok = cases(a.quick)
    print("ALL OK" if ok else "FAILED")
    if a.bench and ok:
        bench()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
