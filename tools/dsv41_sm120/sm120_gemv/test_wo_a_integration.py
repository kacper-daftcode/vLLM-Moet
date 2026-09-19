#!/usr/bin/env python3
"""Integration check for patch_vllm_wo_a_sm120.py inside the serving image:
init_mxfp8_linear_kernel(bmm_batch_size=2) must pick Sm120GemvMxfp8BmmLinearKernel, the
patched deep_gemm_fp8_o_proj must dispatch to it, decode batches (<= 64) must match the
fp32 reference on the fp8 operands and prefill batches must reproduce the bf16 bmm path."""

from __future__ import annotations

import sys

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant
from vllm.models.deepseek_v4.nvidia.ops import o_proj as o_proj_mod

N_GROUPS, HEADS_PER_GROUP, NOPE, ROPE, O_LORA = 2, 8, 448, 64, 1024
HEAD_DIM = NOPE + ROPE
K = HEADS_PER_GROUP * HEAD_DIM
N = N_GROUPS * O_LORA


class Identity(nn.Module):
    def forward(self, x):
        return x


def main() -> int:
    assert "sm120_gemv_bmm" in open(o_proj_mod.__file__).read(), "o_proj patch not applied"
    dev = torch.device("cuda")
    torch.manual_seed(1)
    kernel = init_mxfp8_linear_kernel(bmm_batch_size=N_GROUPS)
    print("kernel class:", type(kernel).__name__)
    assert type(kernel).__name__ == "Sm120GemvMxfp8BmmLinearKernel", type(kernel)

    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    w_q, w_sf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    layer = nn.Module()
    layer.weight = Parameter(w_q, requires_grad=False)
    layer.weight_scale = Parameter(w_sf.view(N, K // MXFP8_BLOCK_SIZE).contiguous(), requires_grad=False)
    layer.bmm_batch_size = N_GROUPS
    kernel.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.float8_e4m3fn and getattr(layer, "sm120_gemv_bmm", None) is kernel
    assert layer.weight_block_size == [1, 32]
    w_deq = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale)

    max_pos = 4096
    inv_freq = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=dev, dtype=torch.float32) / ROPE))
    freqs = torch.outer(torch.arange(max_pos, device=dev, dtype=torch.float32), inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()
    wo_b = Identity()
    fails = 0
    for T in (1, 6, 16, 48, 64, 65, 300):
        o = torch.randn(T, N_GROUPS * HEADS_PER_GROUP, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        positions = torch.randint(0, max_pos, (T,), device=dev, dtype=torch.int64)
        out = o_proj_mod.deep_gemm_fp8_o_proj(
            o, positions, cos_sin, layer, wo_b, n_groups=N_GROUPS, heads_per_group=HEADS_PER_GROUP,
            nope_dim=NOPE, rope_dim=ROPE, o_lora_rank=O_LORA, einsum_recipe=(1, 1, 32), tma_aligned_scales=True,
        )
        assert out.shape == (T, N_GROUPS * O_LORA), out.shape
        # bf16 path (what sm_120 ran before the patch)
        x_bf16, _ = fused_inv_rope_fp8_quant(
            o, positions, cos_sin, n_groups=N_GROUPS, heads_per_group=HEADS_PER_GROUP, nope_dim=NOPE,
            rope_dim=ROPE, quant_group_size=32, tma_aligned_scales=True, quantize=False)
        z_bf16 = torch.bmm(x_bf16.transpose(0, 1), w_deq.view(N_GROUPS, O_LORA, K).transpose(1, 2)).transpose(0, 1)
        z_bf16 = z_bf16.reshape(T, -1)
        if T <= 64:
            # fp32 reference on the quantized activations the GEMV consumed
            q, sf = fused_inv_rope_fp8_quant(
                o, positions, cos_sin, n_groups=N_GROUPS, heads_per_group=HEADS_PER_GROUP, nope_dim=NOPE,
                rope_dim=ROPE, quant_group_size=32, tma_aligned_scales=True, quantize=True)
            words = sf.contiguous().to(torch.int64)
            e = torch.stack([(words >> (8 * j)) & 0xFF for j in range(4)], dim=-1).reshape(T, N_GROUPS, -1)[..., : K // 32]
            a = (q.float().view(T, N_GROUPS, K // 32, 32) * torch.exp2(e.float() - 127.0).unsqueeze(-1)).view(T, N_GROUPS, K)
            ref = torch.einsum("tgk,gnk->tgn", a, w_deq.float().view(N_GROUPS, O_LORA, K)).reshape(T, -1)
            err = ((out.float() - ref).abs() / (ref.abs() + 1e-2)).max().item()
            rel_fro = ((out.float() - z_bf16.float()).norm() / z_bf16.float().norm()).item()
            ok = err < 1.5e-2
            print(f"[{'ok ' if ok else 'BAD'}] T={T:3d} gemv path: maxrel vs fp32-ref {err:.2e}; ||gemv - bf16 path||/||bf16|| = {rel_fro:.3e} (fp8 activation quantization)")
        else:
            err = (out.float() - z_bf16.float()).abs().max().item()
            ok = torch.allclose(out.float(), z_bf16.float(), rtol=1e-2, atol=1e-2)
            print(f"[{'ok ' if ok else 'BAD'}] T={T:3d} bf16 fallback: maxdiff vs bf16 bmm {err:.2e}")
        fails += 0 if ok else 1
    print("ALL OK" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
