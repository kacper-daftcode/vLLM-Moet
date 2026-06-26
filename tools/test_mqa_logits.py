#!/usr/bin/env python3
"""Validator for the SM120 triton FP8 MQA-logits kernel (vllm/utils/deep_gemm.py).

Checks the triton kernel + the fp8_fp4_mqa_logits dispatch against the f32 torch
reference (_fp8_mqa_logits_torch_reference) across prefill shapes, plus the
clean_logits masking path, and reports the speedup. Run inside the SM120 vLLM
image with deep_gemm.py mounted, e.g.:

  docker run --rm --gpus '"device=7"' --entrypoint python3 \
    -v $PWD/vllm/utils/deep_gemm.py:/build/vllm/vllm/utils/deep_gemm.py \
    -v $PWD/tools:/tools vllm-moet-sm120:base /tools/test_mqa_logits.py

Gate: rel < 2.5e-3 (kernel is bit-faithful, normally ~3e-7).
"""
import time

import torch

from vllm.utils.deep_gemm import (
    _fp8_mqa_logits_torch_reference,
    _fp8_mqa_logits_triton,
    fp8_fp4_mqa_logits,
)

H, D = 64, 128
dev = "cuda"
REL_GATE = 2.5e-3


def _inputs(M, N, seed=0):
    torch.manual_seed(seed)
    q = (torch.randn(M, H, D, device=dev) * 0.5).clamp(-4, 4).to(torch.float8_e4m3fn)
    k = (torch.randn(N, D, device=dev) * 0.5).clamp(-4, 4).to(torch.float8_e4m3fn)
    ks = (torch.rand(N, device=dev) * 0.1 + 0.01).float()
    w = (torch.randn(M, H, device=dev) * 0.5).float()
    return q, k, ks, w


def _bench(fn, iters=20):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


def main():
    worst = 0.0
    cu0 = None
    for M, N in [(512, 4096), (2048, 16384), (2048, 65536), (1, 8192), (37, 4096)]:
        q, k, ks, w = _inputs(M, N)
        cu = torch.zeros(M, dtype=torch.int32, device=dev)
        r = _fp8_mqa_logits_torch_reference((q, None), (k, ks), w, cu, cu, False)
        t = _fp8_mqa_logits_triton((q, None), (k, ks), w, cu, cu, False)
        d = fp8_fp4_mqa_logits((q, None), (k, ks), w, cu, cu, clean_logits=False)
        rel_t = ((t - r).abs().max() / r.abs().max().clamp_min(1e-9)).item()
        rel_d = ((d - r).abs().max() / r.abs().max().clamp_min(1e-9)).item()
        worst = max(worst, rel_t, rel_d)
        msg = f"M={M:<5} N={N:<6} rel_kernel={rel_t:.2e} rel_dispatch={rel_d:.2e}"
        if M >= 2048 and N >= 16384:
            tr = _bench(lambda: _fp8_mqa_logits_torch_reference(
                (q, None), (k, ks), w, cu, cu, False))
            tt = _bench(lambda: _fp8_mqa_logits_triton(
                (q, None), (k, ks), w, cu, cu, False))
            msg += f"  | torch {tr:6.1f}ms  triton {tt:5.2f}ms  {tr / tt:4.1f}x"
        print(msg)

    # clean_logits=True (per-row valid window -> -inf mask)
    M, N = 128, 8192
    q, k, ks, w = _inputs(M, N, seed=1)
    klo = torch.randint(0, N // 2, (M,), dtype=torch.int32, device=dev)
    khi = torch.randint(N // 2, N, (M,), dtype=torch.int32, device=dev)
    r = _fp8_mqa_logits_torch_reference((q, None), (k, ks), w, klo, khi, True)
    t = _fp8_mqa_logits_triton((q, None), (k, ks), w, klo, khi, True)
    # compare only finite entries (both -inf where masked)
    fin = torch.isfinite(r)
    rel_c = ((t[fin] - r[fin]).abs().max() / r[fin].abs().max().clamp_min(1e-9)).item()
    same_mask = (torch.isinf(t) == torch.isinf(r)).all().item()
    worst = max(worst, rel_c)
    print(f"clean_logits=True rel={rel_c:.2e} mask_match={same_mask}")

    ok = worst < REL_GATE and same_mask
    print(f"RESULT: {'PASS' if ok else 'FAIL'} (worst_rel={worst:.2e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
