#!/usr/bin/env python3
"""Op-level test of the sm_120 small-batch bf16 linear (inkling2_linear_sm120.cu) against exact arithmetic.

    python3 test_linear_sm120.py [--bench]

Cases: every row count class (1 .. 32 rows), the model's shapes on one rank of TP4 (qkvr 2896 x 4608, the full
layers' query side 2512 x 4608 and K / V 256 x 4608, the draft's MLP 4608 x 4608), a short-K output projection
4608 x 1152 (a rank's `wo` is 4608 x 1536) and odd ones (a partial last column block, K of one and five steps). x and the weight are row slices of larger tensors whose other
rows are NaN: what the kernel must not read is poison.

Criterion. The kernel accumulates K / 16 products in fp32 and rounds once to bf16. With `exact` the float64 product
of the bf16 operands, an element passes when it is the bf16 nearest to `exact`, or when its distance from `exact` is
within half a bf16 unit at `exact` plus the accumulation bound (K / 16) 2^-24 sum|x_k w_k|; at least 99 % of the
elements must be the nearest bf16. The run is repeated: bit-identical.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inkling2_linear_sm120 as lin  # noqa: E402

DEV = torch.device("cuda")


def poisoned(rows: int, cols: int, gen: torch.Generator, scale: float) -> torch.Tensor:
    """[rows, cols] bf16 values as the middle rows of a NaN tensor."""
    full = torch.full((rows + 4, cols), float("nan"))
    full[2 : 2 + rows] = torch.randn((rows, cols), generator=gen) * scale
    return full.to(DEV, torch.bfloat16)[2 : 2 + rows]


def case(seed: int, m: int, n: int, k: int) -> bool:
    gen = torch.Generator().manual_seed(seed)
    x, w = poisoned(m, k, gen, 1.0), poisoned(n, k, gen, 0.05)
    assert lin.takes(x, w, True), (m, n, k)
    out = lin.linear(x, w, True)
    again = lin.linear(x, w, True)
    torch.cuda.synchronize()
    exact = x.double() @ w.double().t()
    mass = x.double().abs() @ w.double().abs().t()
    nearest = exact.to(torch.bfloat16)
    same = out.view(torch.int16) == nearest.view(torch.int16)
    ulp = torch.maximum(exact.abs(), torch.tensor(1e-37, dtype=torch.float64, device=DEV)).log2().floor().exp2() * 2.0**-7
    bound = 0.5 * ulp * (1 + 1e-9) + (k / 16) * 2.0**-24 * mass
    err = (out.double() - exact).abs()
    ok = bool((same | (err <= bound)).all()) and not bool(torch.isnan(out).any())
    frac = same.double().mean().item()
    repeat = torch.equal(out.view(torch.int16), again.view(torch.int16))
    good = ok and frac >= 0.99 and repeat
    print(f"  {'OK  ' if good else 'FAIL'} m {m:2d} n {n:5d} k {k:5d}: nearest bf16 {100 * frac:.3f} %, worst err / bound "
          f"{(err / bound).max().item():.3f}, two runs identical {repeat}")  # fmt: skip
    return good


def bench() -> None:
    print("timings (median of 30 launches after 10, torch.cuda.synchronize around each; microseconds)")

    def timed(fn):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        return sorted(ts)[len(ts) // 2]

    gen = torch.Generator().manual_seed(1)
    for n, k, name in [(2896, 4608, "qkvr"), (4608, 1152, "wo"), (2512, 4608, "q side"), (256, 4608, "k / v"), (4608, 4608, "draft mlp")]:
        w = (torch.randn((n, k), generator=gen) * 0.05).to(DEV, torch.bfloat16)
        # other weights between launches, as in a decode step: the weight is not in L2 when its turn comes
        others = [(torch.randn((4608, 4608), generator=gen) * 0.05).to(DEV, torch.bfloat16) for _ in range(6)]
        for m in (1, 8, 16, 32):
            x = torch.randn((m, k), generator=gen).to(DEV, torch.bfloat16)
            warm = timed(lambda: lin.linear(x, w, True))
            ref = timed(lambda: torch.nn.functional.linear(x, w))

            def cold(fn):
                ts = []
                for i in range(12):
                    for o in others:
                        o.add_(0)  # 254 MB through L2
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1e6)
                return sorted(ts[2:])[len(ts[2:]) // 2]

            c_k, c_ref = cold(lambda: lin.linear(x, w, True)), cold(lambda: torch.nn.functional.linear(x, w))
            mb = n * k * 2 / 1e6
            print(f"  {name:9s} {n:5d} x {k:5d} ({mb:5.1f} MB) m {m:2d}: kernel {warm:7.1f} us warm, {c_k:7.1f} us cold ({mb / c_k:.2f} TB/s)"
                  f" | cuBLAS {ref:7.1f} us warm, {c_ref:7.1f} us cold ({mb / c_ref:.2f} TB/s)")  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    assert lin.supported(), "needs a capability-12 device"
    print(f"{torch.cuda.get_device_name(0)}, kernel facts m <= 16 {lin.kernel_facts(16)}, m <= 32 {lin.kernel_facts(32)}")
    ok = True
    seed = 0
    for n, k in [(2896, 4608), (2512, 4608), (256, 4608), (4608, 1152), (4608, 4608), (34, 64), (30, 320), (2, 128), (130, 2304)]:
        for m in (1, 2, 7, 8, 9, 16, 17, 31, 32):
            if n * k > 4000 * 4608 and m not in (1, 8, 17, 32):
                continue
            ok &= case(seed, m, n, k)
            seed += 1
    x = torch.randn((3, 100), device=DEV).to(torch.bfloat16)
    w = torch.randn((7, 100), device=DEV).to(torch.bfloat16)
    assert not lin.takes(x, w, True) and lin.linear(x, w, True).shape == (3, 7)  # the fallback
    print("ALL OK" if ok else "FAILED")
    if a.bench and ok:
        bench()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
