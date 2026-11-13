#!/usr/bin/env python3
"""Op-level test of the sm_120 batched expert GEMM (inkling2_bmm_sm120) against exact arithmetic.

    python3 test_bmm_sm120.py [--model-dir DIR] [--bench]      # one sm_120 GPU

For every format (nvfp4, mxfp8, mxfp6) and every token tile (8 ... 256): random tiles (experts, row limits, tiles in
use) on a small shape and on the model's shapes (N 4608, K 4608 and 2304); the kernel's rows are compared with the
exact result (float64 products and sums of the decoded operands, the fp32 row scale, one rounding to bf16):
  - at least 99.5 % of the elements identical to the exact result; the others differ by what an fp32 accumulation
    in K order may: beyond the final rounding the error stays below one unit of 2^-24 of the terms' magnitude sum
    per QMMA (a result near zero from large cancelling terms can therefore differ by more than one bf16 unit);
  - rows beyond a tile's limit and tiles not in use are not written;
  - the output does not depend on the grid (1 or 2 CTAs per SM) nor on the run.
With --model-dir (the checkpoint directory) the same on real expert weights of an NVFP4 and an MXFP8 layer, and the
6-bit packing of the MXFP8 weights (pack_e2m3) is characterized: values kept exactly, relative error of the rest,
and the kernel's mxfp6 output against its mxfp8 output on the same rows.
--bench times decode- and prefill-shaped launches (bursts of at most 40 ms with a second of idle between them).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import inkling2_bmm_sm120 as m  # noqa: E402

dev = torch.device("cuda")
NAMES = {m.NVFP4: "nvfp4", m.MXFP8: "mxfp8", m.MXFP6: "mxfp6"}
E4M3_CODES = torch.tensor([c for c in range(256) if c & 0x7F != 0x7F], dtype=torch.uint8)


def rand_e4m3(shape, g, lo=None, hi=None):
    if lo is not None:
        return torch.randint(lo, hi + 1, shape, generator=g, dtype=torch.uint8)
    return E4M3_CODES[torch.randint(0, len(E4M3_CODES), shape, generator=g)]


def make_weights(fmt, experts, n, k, g):
    """row-major weight / scale bytes (the checkpoint's form)"""
    if fmt == m.NVFP4:
        w = torch.randint(0, 256, (experts, n, k // 2), generator=g, dtype=torch.uint8)
        s = rand_e4m3((experts, n, k // 16), g, 97, 126)
        return w, s
    w = rand_e4m3((experts, n, k), g)
    s = torch.randint(112, 119, (experts, n, k // 32), generator=g, dtype=torch.uint8)
    return w, s


def pow2(exponent):
    """2^exponent as float64, built from its bits: a float pow is not exact on every device, and the packing's
    figures compare values for equality"""
    return ((exponent.to(torch.int64) + 1023) << 52).view(torch.float64)


def to_kernel(fmt, w, s):
    """row-major checkpoint bytes -> (kernel operands in the prepared layout, row-major operands for the reference)"""
    w, s = w.to(dev), s.to(dev)
    if fmt == m.MXFP6:
        parts = [m.pack_e2m3(w[i : i + 4]) for i in range(0, w.shape[0], 4)]   # the intermediates are 40 B per value
        s = (s.to(torch.int16) + torch.cat([d for _, d in parts]).to(torch.int16)).to(torch.uint8)
        w = torch.cat([p for p, _ in parts])
        del parts
    return (m.prepare(w, m.WEIGHT_BYTES_PER_STEP[fmt]), m.prepare(s, m.SCALE_BYTES_PER_STEP[fmt])), (w, s)


def make_acts(fmt, rows, k, g):
    if fmt == m.NVFP4:
        act = torch.randint(0, 256, (rows, k // 2), generator=g, dtype=torch.uint8)
        sc = rand_e4m3((rows, k // 16), g, 40, 126)
        rs = (torch.rand(rows, generator=g) + 0.5) * 1e-6
        return act.to(dev), sc.to(dev), rs.to(dev)
    act = rand_e4m3((rows, k), g)
    sc = torch.randint(116, 125, (rows, k // 32), generator=g, dtype=torch.uint8)
    return act.to(dev), sc.to(dev), None


def make_tiles(tile, max_ctas, experts, g, full=False):
    num = max_ctas if full else int(torch.randint(1, max_ctas + 1, (1,), generator=g))
    cta_expert = torch.randint(0, experts, (max_ctas,), generator=g, dtype=torch.int32)
    used = torch.randint(1, tile + 1, (max_ctas,), generator=g, dtype=torch.int32)
    if full:
        used[:] = tile
    used[0] = tile
    cta_limit = torch.arange(max_ctas, dtype=torch.int32) * tile + used
    return cta_expert.to(dev), cta_limit.to(dev), torch.tensor([num], dtype=torch.int32, device=dev)


def bf16_order(x: torch.Tensor) -> torch.Tensor:
    b = x.view(torch.int16).to(torch.int32)
    return torch.where(b < 0, -(b & 0x7FFF), b)


def check(fmt, tile, n, k, experts, max_ctas, seed, weights=None, label=""):
    g = torch.Generator().manual_seed(seed)
    if weights is None:
        weights = to_kernel(fmt, *make_weights(fmt, experts, n, k, g))
    (wp, sp), (wr, sr) = weights
    rows = tile * max_ctas
    act, asc, rs = make_acts(fmt, rows, k, g)
    cta_expert, cta_limit, num_ctas = make_tiles(tile, max_ctas, experts, g)
    sentinel = float("nan")
    outs = []
    for cps in ("2", "1", "2"):
        os.environ["VLLM_MOET_INKLING2_BMM_CPS"] = cps
        out = torch.full((rows, n), sentinel, dtype=torch.bfloat16, device=dev)
        m.bmm(fmt, act, asc, rs, wp, sp, cta_expert, cta_limit, num_ctas, tile, n, k, out=out)
        torch.cuda.synchronize()
        outs.append(out)
    os.environ["VLLM_MOET_INKLING2_BMM_CPS"] = "2"
    ctrl = m._ctrl(dev)
    assert ctrl.tolist() == [0, 0], f"control block not reset: {ctrl.tolist()}"
    r = torch.arange(rows, device=dev)
    t = r // tile
    valid = (t < num_ctas) & (r < cta_limit[t])
    ref, exact, mag = m.reference(fmt, act[valid], asc[valid], rs[valid] if rs is not None else None, wr, sr,
                                  cta_expert[t][valid].long(), with_bound=True)
    got = outs[0][valid]
    d = (bf16_order(got) - bf16_order(ref)).abs()
    same = float((d == 0).float().mean())
    near = float((d <= 1).float().mean())
    # the accumulation's absolute error in units of 2^-24 of the terms' magnitude sum, beyond the final rounding
    # (half a bf16 unit of the result): an fp32 accumulation of Q QMMAs stays below Q of these units
    qmmas = (k // 64) * (1 if fmt == m.NVFP4 else 2)
    half_ulp = got.double().abs() * 2.0**-8
    excess = ((got.double() - exact).abs() - half_ulp).clamp(min=0) / (mag * 2.0**-24).clamp(min=1e-300)
    worst = float(excess.max())
    untouched = bool(outs[0][~valid].isnan().all()) if bool((~valid).any()) else True
    stable = all(torch.equal(outs[0].view(torch.int16), o.view(torch.int16)) for o in outs[1:])
    finite = bool(torch.isfinite(got.float()).all())
    ok = worst <= qmmas and same >= 0.995 and untouched and stable and finite
    print(f"  {NAMES[fmt]} tile {tile:3d} n {n} k {k} tiles {int(num_ctas)}/{max_ctas} rows {int(valid.sum())}{label}: "
          f"identical {100 * same:.3f} %, within one bf16 unit {100 * near:.3f} %, accumulation error at most {worst:.2f} "
          f"x 2^-24 of the magnitude sum (bound {qmmas}), padding untouched {untouched}, "
          f"grid- and run-independent {stable}  {'OK' if ok else 'FAILED'}")
    return ok


def read_tensor(model_dir, name, experts):
    """experts [e0, e1) of tensor `name` from the checkpoint, as uint8 [e1 - e0, N, X]"""
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    path = os.path.join(model_dir, idx[name])
    with open(path, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        t = json.loads(f.read(hl))[name]
        e_all, n, x = t["shape"]
        e0, e1 = experts
        f.seek(8 + hl + t["data_offsets"][0] + e0 * n * x)
        buf = f.read((e1 - e0) * n * x)
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return torch.frombuffer(bytearray(buf), dtype=torch.uint8).reshape(e1 - e0, n, x)


def real_weights(model_dir):
    ok = True
    print("real expert weights:")
    w = read_tensor(model_dir, "model.llm.layers.0.mlp.experts.w13_weight", (0, 4))
    s = read_tensor(model_dir, "model.llm.layers.0.mlp.experts.w13_weight_scale", (0, 4))
    wk = to_kernel(m.NVFP4, w, s)
    for tile in (8, 16, 128):
        ok &= check(m.NVFP4, tile, 4608, 4608, 4, 6 if tile < 128 else 3, 77 + tile, weights=wk, label=" [layer 0 w13]")
    for proj, k in (("w13", 4608), ("w2", 2304)):
        layer = next(l for l in range(40, 48) if os.path.exists(os.path.join(model_dir, json.load(open(os.path.join(
            model_dir, "model.safetensors.index.json")))["weight_map"][f"model.llm.layers.{l}.mlp.experts.{proj}_weight"])))
        w = read_tensor(model_dir, f"model.llm.layers.{layer}.mlp.experts.{proj}_weight", (0, 4))
        s = read_tensor(model_dir, f"model.llm.layers.{layer}.mlp.experts.{proj}_weight_scale", (0, 4))
        w8 = to_kernel(m.MXFP8, w, s)
        w6 = to_kernel(m.MXFP6, w, s)
        for tile in (8, 128):
            ok &= check(m.MXFP8, tile, 4608, k, 4, 4 if tile < 128 else 2, 99 + tile, weights=w8, label=f" [layer {layer} {proj}]")
            ok &= check(m.MXFP6, tile, 4608, k, 4, 4 if tile < 128 else 2, 99 + tile, weights=w6, label=f" [layer {layer} {proj}, packed]")
        # the packing itself: e4m3 -> e2m3 on the real weights
        v8 = m.e4m3_values(w.to(dev)) * pow2(s.to(dev).to(torch.int64) - 127).repeat_interleave(32, dim=-1)
        v6 = (m.e2m3_values(w6[1][0]) * 8).round() / 8 * pow2(w6[1][1].to(torch.int64) - 127).repeat_interleave(32, dim=-1)
        exact = float((v8 == v6).double().mean())
        err = v6 - v8
        snr = 10 * torch.log10((v8**2).sum() / (err**2).sum())
        print(f"  pack_e2m3 layer {layer} {proj} (4 experts): {100 * exact:.2f} % of the weights exact, "
              f"error / weight energy {-float(snr):.1f} dB, max |error| / block max {float((err.abs().reshape(-1, 32).amax(1) / v8.abs().reshape(-1, 32).amax(1).clamp(min=1e-30)).max()):.4f}")
        # the kernel's outputs on the same activations: packed against unpacked
        g = torch.Generator().manual_seed(5)
        tile, max_ctas = 16, 8
        act, asc, _ = make_acts(m.MXFP8, tile * max_ctas, k, g)
        ce, cl, nc = make_tiles(tile, max_ctas, 4, g, full=True)
        o8 = m.bmm(m.MXFP8, act, asc, None, *w8[0], ce, cl, nc, tile, 4608, k)
        o6 = m.bmm(m.MXFP6, act, asc, None, *w6[0], ce, cl, nc, tile, 4608, k)
        d = (bf16_order(o8) - bf16_order(o6)).abs()
        rel = (o6.double() - o8.double()).norm() / o8.double().norm()
        print(f"  mxfp6 against mxfp8 outputs, layer {layer} {proj}, random activations: identical {100 * float((d == 0).float().mean()):.2f} %, "
              f"within 1 unit {100 * float((d <= 1).float().mean()):.2f} %, relative error {float(rel):.2e} ({20 * torch.log10(rel).item():.1f} dB)")
    return ok


def bench():
    print("timing (one card, cold weights; bursts of <= 40 ms, 1 s idle between):")
    for fmt, k in ((m.NVFP4, 4608), (m.NVFP4, 2304), (m.MXFP8, 4608), (m.MXFP6, 4608), (m.MXFP8, 2304), (m.MXFP6, 2304)):
        g = torch.Generator().manual_seed(1)
        experts = 96
        (wp, sp), _ = to_kernel(fmt, *make_weights(fmt, experts, 4608, k, g))
        wbytes = 4608 * (k // 64) * (m.WEIGHT_BYTES_PER_STEP[fmt] + m.SCALE_BYTES_PER_STEP[fmt])
        for tile, max_ctas, label in ((8, 16, "decode, 16 experts x 8 rows"), (8, 64, "decode, 64 experts x 8 rows"),
                                      (16, 96, "96 experts x 16 rows"), (128, 32, "prefill, 32 tiles x 128 rows")):
            rows = tile * max_ctas
            act, asc, rs = make_acts(fmt, rows, k, g)
            sets = []
            for i in range(6):
                ce = ((torch.arange(max_ctas) + i * max_ctas) % experts).to(torch.int32).to(dev)
                cl = ((torch.arange(max_ctas, dtype=torch.int32) + 1) * tile).to(dev)
                sets.append((ce, cl))
            nc = torch.tensor([max_ctas], dtype=torch.int32, device=dev)
            out = torch.empty((rows, 4608), dtype=torch.bfloat16, device=dev)
            times = []
            for rep in range(4):
                time.sleep(1.0)
                calls = 12 if tile < 128 else 3
                evs = [torch.cuda.Event(enable_timing=True) for _ in range(calls + 1)]
                evs[0].record()
                for i in range(calls):
                    ce, cl = sets[(rep * calls + i) % len(sets)]
                    m.bmm(fmt, act, asc, rs, wp, sp, ce, cl, nc, tile, 4608, k, out=out)
                    evs[i + 1].record()
                torch.cuda.synchronize()
                if rep:
                    times += [evs[i].elapsed_time(evs[i + 1]) * 1e3 for i in range(calls)]
            times.sort()
            med = times[len(times) // 2]
            uniq = min(max_ctas, experts)
            print(f"  {NAMES[fmt]} k {k} {label}: median {med:8.1f} us, min {times[0]:8.1f} us; "
                  f"{uniq * wbytes / 1e6:7.1f} MB of weights per call = {uniq * wbytes / med / 1e6:5.2f} TB/s")
        del wp, sp
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    assert m.supported(), "needs an sm_120 / sm_121 GPU"
    print(torch.cuda.get_device_name(0), "| kernels (shared memory, CTAs per SM, registers per thread):")
    for fmt in NAMES:
        print("  " + NAMES[fmt] + ": " + "; ".join(
            f"tile {t}: {f['smem']} B, {f['ctas_per_sm']}, {f['registers']}" for t in (8, 16, 32, 64) for f in [m.kernel_facts(fmt, t)]))
    ok = True
    print("synthetic weights:")
    for fmt in NAMES:
        for tile in (8, 16, 32, 64, 128, 256):
            ok &= check(fmt, tile, 256, 1024, 6, 5, 10 * fmt + tile)
        if not a.quick:
            for tile, max_ctas in ((8, 12), (16, 9), (32, 5), (64, 3), (128, 2), (256, 1)):
                ok &= check(fmt, tile, 4608, 4608, 3, max_ctas, 1 + tile)
            ok &= check(fmt, 8, 4608, 2304, 3, 12, 3)
            ok &= check(fmt, 128, 4608, 2304, 3, 2, 4)
    if a.model_dir:
        ok &= real_weights(a.model_dir)
    if a.bench:
        bench()
    print("BMM SM120 " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
