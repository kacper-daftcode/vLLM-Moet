#!/usr/bin/env python3
"""Checks for patch_vllm_query_quant_gate.py (vllm#57679) on one sm_120 GPU inside the image.

1. `can_fuse_query_quant` sees the consumer capability the way vLLM exposes it (`_input_quant_key`
   set by expose_input_quant_key): True for MXFP8 linears on the FlashInfer CUTLASS kernel class
   (our GEMV patch keeps that class), False for a BF16 linear.
2. The fused kernel `fused_q_kv_rmsnorm_quant` against the separate path `fused_q_kv_rmsnorm` +
   `mxfp8_e4m3_quantize(is_sf_swizzled_layout=True)` on the served geometry (q_lora 1280, kv 512,
   1..170 tokens): identical FP8 bytes, identical F8_128x4 scale bytes of every real (token, group),
   identical normalized KV.
3. The consumer: the (patched) FlashInfer CUTLASS MXFP8 linear kernel fed the QuantizedActivation
   returns the same output as when fed the normalized bf16 Q (GEMV path at <= 16 rows, CUTLASS above).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutlassMxfp8LinearKernel
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.common.ops.fused_qk_rmsnorm import fused_q_kv_rmsnorm
from vllm.models.deepseek_v41.common.ops import query_quant as qq


def swizzled_offsets(tokens: int, groups: int, padded_groups: int, dev) -> torch.Tensor:
    """F8_128x4 offsets of the real (token, group) scale bytes, as the fused kernel lays them out."""
    row = torch.arange(tokens, device=dev).view(-1, 1)
    g = torch.arange(groups, device=dev).view(1, -1)
    return (row // 128 * (128 * padded_groups) + g // 4 * 512 + row % 32 * 16 + row % 128 // 32 * 4 + g % 4).flatten()


def main() -> int:
    dev = torch.device("cuda")
    torch.manual_seed(11)
    fails = 0

    # 1. the gate
    kernel = FlashInferCutlassMxfp8LinearKernel.__new__(FlashInferCutlassMxfp8LinearKernel)
    mxfp8_linear = SimpleNamespace(_input_quant_key=kMxfp8Dynamic, quant_method=SimpleNamespace(kernel=kernel))
    bf16_linear = SimpleNamespace(quant_method=SimpleNamespace(kernel=object()))
    gate_on = qq.can_fuse_query_quant([mxfp8_linear, mxfp8_linear])
    gate_off = qq.can_fuse_query_quant([mxfp8_linear, bf16_linear])
    ok = gate_on and not gate_off
    fails += 0 if ok else 1
    print(f"[{'ok ' if ok else 'BAD'}] can_fuse_query_quant: MXFP8 consumers -> {gate_on}, with a BF16 consumer -> {gate_off}")

    # 2. fused kernel vs separate path
    Q, KV, eps = 1280, 512, 1e-20
    qw = (torch.rand(Q, device=dev, dtype=torch.bfloat16) + 0.5)
    kvw = (torch.rand(KV, device=dev, dtype=torch.bfloat16) + 0.5)
    for T in (1, 6, 7, 16, 17, 48, 64, 128, 129, 170):
        qr_kv = torch.randn(T, Q + KV, device=dev, dtype=torch.bfloat16) * float(torch.rand(1).item() * 3 + 0.1)
        qr, kv = qr_kv.split([Q, KV], dim=-1)  # strided views, as in _split_qkv_and_norm
        qa, kv_f = qq.fused_q_kv_rmsnorm_quant(qr, kv, qw, kvw, eps)
        qr_n, kv_n = fused_q_kv_rmsnorm(qr, kv, qw, kvw, eps)
        q_sep, s_sep = mxfp8_e4m3_quantize(qr_n, is_sf_swizzled_layout=True)
        torch.cuda.synchronize()
        same_q = torch.equal(qa.data.view(torch.uint8), q_sep.view(torch.uint8).view(qa.data.shape))
        groups, padded_groups = Q // 32, (Q // 32 + 3) // 4 * 4
        off = swizzled_offsets(T, groups, padded_groups, dev)
        same_s = torch.equal(qa.scale.flatten()[off], s_sep.flatten()[off])
        same_kv = torch.equal(kv_f, kv_n)
        ok = same_q and same_s and same_kv and qa.quant_key == kMxfp8Dynamic and qa.orig_shape == qr.shape
        fails += 0 if ok else 1
        nq = int((qa.data.view(torch.uint8) != q_sep.view(torch.uint8).view(qa.data.shape)).sum())
        print(f"[{'ok ' if ok else 'BAD'}] T={T:3d}: fp8 bytes {'identical' if same_q else f'{nq} differ'}, "
              f"scale bytes {'identical' if same_s else 'DIFFER'}, kv {'identical' if same_kv else 'DIFFER'}")

    # 3. the consumer with the pre-quantized activation
    N = 8192  # wq_b per rank
    w = torch.randn(N, Q, device=dev, dtype=torch.bfloat16) * 0.05
    w_q, w_sf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    layer = torch.nn.Module()
    layer.weight = Parameter(w_q, requires_grad=False)
    layer.weight_scale = Parameter(w_sf.view(N, -1), requires_grad=False)
    lk = FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    lk.process_weights_after_loading(layer)
    for T in (1, 6, 16, 17, 64):
        qr = torch.randn(T, Q, device=dev, dtype=torch.bfloat16)
        kv = torch.randn(T, KV, device=dev, dtype=torch.bfloat16)
        qa, _ = qq.fused_q_kv_rmsnorm_quant(qr, kv, qw, kvw, eps)
        qr_n, _ = fused_q_kv_rmsnorm(qr, kv, qw, kvw, eps)
        out_qa = lk.apply_weights(layer, qa)
        out_bf = lk.apply_weights(layer, qr_n)
        torch.cuda.synchronize()
        ok = torch.equal(out_qa, out_bf) and out_qa.shape == (T, N)
        fails += 0 if ok else 1
        d = (out_qa.float() - out_bf.float()).abs().max().item()
        print(f"[{'ok ' if ok else 'BAD'}] consumer T={T:3d}: wq_b(QuantizedActivation) vs wq_b(bf16) "
              f"{'identical' if ok else f'max diff {d:.3e}'}")
    print("ALL OK" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
