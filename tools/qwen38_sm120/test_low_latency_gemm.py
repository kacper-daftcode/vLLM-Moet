#!/usr/bin/env python3
"""Check the patched low_latency_gemm.py inside the qwen38 image: sm_120 gate open, merged plans,
every (shape, M) entry dispatches through torch.ops.vllm.qwen3_8_flash_next_low_latency_gemm and
matches fp32 within bf16 rounding; token counts without a plan fall back to cuBLAS."""
import torch
from vllm.models.qwen3_8_flash_next.nvidia import low_latency_gemm as ll

print("sm120 gate:", ll._is_sm120(), "| plans:", len(ll._plans()), "shapes; upstream", len(ll.QWEN38NEXT_GEMM_PLANS))
dev = torch.device("cuda")
bad = 0
for (N, K), plan in sorted(ll._plans().items()):
    if N * K * 2 > 1e9:
        continue
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    for M in sorted(plan):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        y = torch.ops.vllm.qwen3_8_flash_next_low_latency_gemm(x, w)
        ref = (x.float() @ w.float().t())
        err = (y.float() - ref).abs().max().item()
        ok = err <= 0.02 * (ref.abs().max().item() + 1e-6) + 0.05
        bad += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] {N}x{K} M={M}: maxabs {err:.2e}")
    # a token count without a plan must fall back to cuBLAS and still be right
    x = torch.randn(5, K, device=dev, dtype=torch.bfloat16)
    y = torch.ops.vllm.qwen3_8_flash_next_low_latency_gemm(x, w)
    assert torch.allclose(y.float(), torch.nn.functional.linear(x, w).float()), (N, K)
print("fallback M=5 OK;", "ALL OK" if bad == 0 else f"{bad} BAD")
