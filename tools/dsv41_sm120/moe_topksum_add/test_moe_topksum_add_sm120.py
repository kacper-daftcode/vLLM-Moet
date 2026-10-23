#!/usr/bin/env python3
"""Bit test of step 33 (tools/dsv41_sm120/patch_vllm_moe_topksum_add_sm120.py): `_moet_topksum_add_kernel` against
b12x's plain W4A16TopKSumKernel followed by torch's bf16 add of the shared-expert output, inside the image with b12x's
_impl.py patched. Random route rows (bf16, wide range, a few inf / NaN), route ids with -1 (unrouted) entries, 1-8
tokens, top-k 6 (and 4, 8), hidden 5120 (and 2048): every output element must be bit-identical. Then the served shape
(8 tokens x top-6 x 5120) is timed: the two original launches vs the one fused.
usage: python3 test_moe_topksum_add_sm120.py [--cases N]
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from b12x.moe._shared.kernels.w4a16.kernel import _w4a16_topk_sum_launch_flat  # noqa: E402
import flashinfer.experimental.b12x.moe.fused_moe._impl as impl  # noqa: E402

assert hasattr(impl, "_moet_topksum_add_launch"), "patched _impl.py not in place"
dev = torch.device("cuda")


def reference(route, ids, shared, m, topk, k):
    out = torch.empty((m, k), dtype=torch.bfloat16, device=dev)
    _w4a16_topk_sum_launch_flat(route, out, m, topk, k, "bf16", int(torch.cuda.current_stream().cuda_stream), route_expert_ids=ids)
    return shared + out


def fused(route, ids, shared, m, topk, k):
    out = torch.empty((m, k), dtype=torch.bfloat16, device=dev)
    impl._moet_topksum_add_launch(route, ids, shared, out, m, topk, k)
    return out


def make(gen, m, topk, k, wild):
    route = (torch.randn((m * topk, k), generator=gen) * (8.0 if wild else 1.0)).to(torch.bfloat16)
    if wild:
        flat = route.view(-1)
        pick = torch.randint(0, flat.numel(), (16,), generator=gen)
        flat[pick[:6]] = float("inf")
        flat[pick[6:10]] = -float("inf")
        flat[pick[10:]] = float("nan")
        flat[torch.randint(0, flat.numel(), (64,), generator=gen)] = 65504.0
    ids = torch.randint(0, 384, (m * topk,), generator=gen).to(torch.int32)
    ids[torch.rand(m * topk, generator=gen) < 0.2] = -1
    shared = (torch.randn((m, k), generator=gen) * (4.0 if wild else 1.0)).to(torch.bfloat16)
    return route.to(dev), ids.to(dev), shared.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=48)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(33)
    shapes = [(8, 6, 5120), (1, 6, 5120), (6, 6, 5120), (3, 6, 5120), (8, 4, 2048), (2, 8, 2048)]
    fails = 0
    for i in range(a.cases):
        m, topk, k = shapes[i % len(shapes)]
        route, ids, shared = make(gen, m, topk, k, wild=(i % 3 == 2))
        ref = reference(route, ids, shared, m, topk, k)
        new = fused(route, ids, shared, m, topk, k)
        torch.cuda.synchronize()
        if not torch.equal(ref.view(torch.int16), new.view(torch.int16)):
            fails += 1
            d = (ref.view(torch.int16) != new.view(torch.int16)).sum().item()
            print(f"case {i} m={m} topk={topk} k={k}: {d} elements differ")
    print(f"{a.cases - fails}/{a.cases} cases bit-identical (top-k sum + shared add)")
    m, topk, k = 8, 6, 5120
    route, ids, shared = make(gen, m, topk, k, wild=False)
    out = torch.empty((m, k), dtype=torch.bfloat16, device=dev)
    def orig():
        _w4a16_topk_sum_launch_flat(route, out, m, topk, k, "bf16", int(torch.cuda.current_stream().cuda_stream), route_expert_ids=ids)
        torch.add(shared, out, out=out)
    def new():
        impl._moet_topksum_add_launch(route, ids, shared, out, m, topk, k)
    for name, fn in (("original (top-k sum + add)", orig), ("step 33 fused", new)):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(300):
            fn()
        torch.cuda.synchronize()
        eager = (time.perf_counter() - t0) / 300 * 1e6
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            fn()
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                for _ in range(40):
                    fn()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            g.replay()
        torch.cuda.synchronize()
        print(f"   {name}: eager {eager:.1f} us, in a CUDA graph {(time.perf_counter() - t0) / 50 / 40 * 1e6:.2f} us per layer")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
