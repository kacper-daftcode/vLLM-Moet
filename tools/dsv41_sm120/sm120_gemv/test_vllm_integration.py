#!/usr/bin/env python3
"""Integration check for patch_vllm_mxfp8_gemv.py: FlashInferCutlassMxfp8LinearKernel
must take the GEMV path for M <= 16 and CUTLASS beyond, with matching results."""

from __future__ import annotations

import sys

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.kernels.linear.mxfp8 import flashinfer as fi_mod
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutlassMxfp8LinearKernel
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize


def main() -> int:
    assert hasattr(fi_mod, "_sm120_gemv_for"), "patch not applied"
    dev = torch.device("cuda")
    torch.manual_seed(0)
    N, K = 4096, 1280
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    w_q, w_sf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    layer = torch.nn.Module()
    layer.weight = Parameter(w_q, requires_grad=False)
    layer.weight_scale = Parameter(w_sf.view(N, -1), requires_grad=False)
    kernel = FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    fails = 0
    for M in (1, 6, 16, 17, 64):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        out = kernel.apply_weights(layer, x)
        # reference: force CUTLASS by disabling the GEMV
        fi_mod._SM120_GEMV = False
        ref = kernel.apply_weights(layer, x)
        fi_mod._SM120_GEMV = None
        used_gemv = fi_mod._sm120_gemv_for(M, torch.bfloat16) is not None
        ok = torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=1e-2) and used_gemv == (M <= 16)
        fails += 0 if ok else 1
        maxdiff = (out.float() - ref.float()).abs().max().item()
        print(f"[{'ok ' if ok else 'BAD'}] M={M:3d} path={'gemv' if used_gemv else 'cutlass'} maxdiff vs cutlass={maxdiff:.3e}")
    print("ALL OK" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
