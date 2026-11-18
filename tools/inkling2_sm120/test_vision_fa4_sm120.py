#!/usr/bin/env python3
"""Op-level check of step 4: the vision tower's attention (tml_fa4's `flash_attn_varlen_func`, plain variable-length
self-attention without a causal mask) on a capability-12 card, against the float64 definition.

    python3 test_vision_fa4_sm120.py        (inside the image, after patch_vllm_inkling2_vision_fa4_sm120.py)

Without the step the first call fails in the kernel's compile (the forward class takes a TMA output path it has no
atom for). The shapes are the tower's: 16 heads of 64, bf16, images of different token counts in one batch; the
softmax weights are rounded to bf16 before P @ V, so an element may differ from the float64 value by 2^-8 max|V| plus
2^-8 of itself.
"""

from __future__ import annotations

import sys

import torch

from vllm.third_party.tml_fa4 import flash_attn_varlen_func

DEV = torch.device("cuda")


def main() -> int:
    assert torch.cuda.get_device_capability()[0] == 12, "needs a capability-12 device"
    gen = torch.Generator().manual_seed(0)
    heads, dim = 16, 64
    ok = True
    for lens in ([1], [17], [257, 64, 1025, 3], [4096]):
        total = sum(lens)
        q, k, v = ((torch.randn((total, heads, dim), generator=gen) * s).to(DEV, torch.bfloat16) for s in (0.7, 0.7, 0.5))
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=DEV)
        out, _ = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lens), max_seqlen_k=max(lens),
            softmax_scale=dim**-0.5, causal=False,
        )  # fmt: skip
        torch.cuda.synchronize()
        ref = torch.empty((total, heads, dim), dtype=torch.float64, device=DEV)
        start = 0
        for n in lens:
            qq, kk, vv = (t[start : start + n].double().transpose(0, 1) for t in (q, k, v))
            p = torch.softmax(qq @ kk.transpose(1, 2) * dim**-0.5, -1)
            ref[start : start + n] = (p @ vv).transpose(0, 1)
            start += n
        err = (out.double() - ref).abs()
        bound = 2.0**-8 * v.double().abs().max() + 2.0**-8 * ref.abs()
        good = bool((err <= bound).all()) and not bool(torch.isnan(out).any())
        rms = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(f"  {'OK  ' if good else 'FAIL'} images of {lens} tokens: max err {err.max().item():.3e}, rel rms {rms:.2e}")
        ok &= good
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
