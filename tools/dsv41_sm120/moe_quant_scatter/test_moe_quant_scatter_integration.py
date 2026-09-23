#!/usr/bin/env python3
"""Integration check for patch_vllm_moe_quant_scatter_sm120.py: the patched DeepGemmFP4Experts,
driven the way FusedMoEModularKernel drives it (MoEPrepareAndFinalizeNoDPEPModular.prepare ->
apply), must produce the bit-identical MoE output with the fused quant+permute (decode shapes),
with the deferred-quantization fallback (> 1024 pairs) and with the switch off
(VLLM_MOET_MOE_QUANT_SCATTER=0: quantization in prepare(), vLLM's permute) - and must take the
intended path in each case.

Run inside the ds41 image on one sm_120 GPU after the patch is applied:
    python3 test_moe_quant_scatter_integration.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

import vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe as dgm
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig, FusedMoEQuantDesc
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import DeepGemmFP4Experts
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _pack_deepgemm_mxfp4_scales
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform


def deepgemm_mxfp4_quant_config(w13_s, w2_s) -> FusedMoEQuantConfig:
    """What oracle/mxfp4.py builds for the DEEPGEMM_MXFP4 backend: fp8 activations quantized
    per (token, 128-group), mxfp4 weights."""
    fp8 = current_platform.fp8_dtype()
    block = GroupShape(128, 128)
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(fp8, block, None, None, None, None),
        _a2=FusedMoEQuantDesc(fp8, block, None, None, None, None),
        _w1=FusedMoEQuantDesc("mxfp4", None, w13_s, None, None, None),
        _w2=FusedMoEQuantDesc("mxfp4", None, w2_s, None, None, None),
    )


def make_experts(E, N1, K, I, dev):
    w13 = torch.randint(0, 256, (E, N1, K // 2), device=dev, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, K, I // 2), device=dev, dtype=torch.uint8)
    w13_s = torch.randint(118, 126, (E, N1, K // 32), device=dev, dtype=torch.uint8)
    w2_s = torch.randint(118, 126, (E, K, I // 32), device=dev, dtype=torch.uint8)
    w13_s, w2_s = _pack_deepgemm_mxfp4_scales(w13, w2, w13_s, w2_s)
    return w13, w2, w13_s, w2_s


def run(experts, prepare, x, topk_w, topk_ids, w13, w2, E, calls):
    """prepare -> apply like FusedMoEKernelModularImpl, recording which permute ran."""
    M, K = x.shape
    N1 = w13.size(1)
    a1q, a1q_scale, _, _, _ = prepare.prepare(
        x, topk_w, topk_ids, E, None, False, experts.quant_config,
        defer_input_quant=experts.expects_unquantized_inputs)
    ws13_shape, ws2_shape, out_shape = experts.workspace_shapes(
        M, N1, K, topk_ids.size(1), E, E, None, MoEActivation.SILU)
    ws13 = torch.empty(ws13_shape, device=x.device, dtype=torch.bfloat16)
    ws2 = torch.empty(ws2_shape, device=x.device, dtype=torch.bfloat16)
    out = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
    calls.clear()
    experts.apply(out, a1q, w13, w2, topk_w, topk_ids, MoEActivation.SILU, E, None, a1q_scale, None,
                  ws13, ws2, None, False)
    torch.cuda.synchronize()
    return out, a1q.dtype


def main() -> int:
    assert hasattr(dgm, "_moet_quant_scatter"), "patch not applied"
    dev = torch.device("cuda")
    torch.manual_seed(3)
    E, K, I, TOPK = 384, 5120, 640, 6
    N1 = 2 * I
    w13, w2, w13_s, w2_s = make_experts(E, N1, K, I, dev)
    quant_config = deepgemm_mxfp4_quant_config(w13_s, w2_s)
    experts = DeepGemmFP4Experts.__new__(DeepGemmFP4Experts)
    experts.quant_config = quant_config
    experts.moe_config = SimpleNamespace(in_dtype=torch.bfloat16)
    experts.gemm1_clamp_limit = None
    prepare = MoEPrepareAndFinalizeNoDPEPModular()

    # record which permute runs
    calls: list[str] = []
    qs = dgm._moet_quant_scatter()
    assert qs is not None, "fused quant+scatter did not load"
    orig_fused, orig_perm = qs.fused_quant_permute, dgm.deepgemm_moe_permute

    def rec_fused(*a, **k):
        calls.append("fused")
        return orig_fused(*a, **k)

    def rec_perm(*a, **k):
        calls.append("vllm")
        return orig_perm(*a, **k)

    qs.fused_quant_permute = rec_fused
    dgm.deepgemm_moe_permute = rec_perm

    fails = 0
    for M in (1, 6, 12, 48, 64, 170, 171, 300):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.7
        topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        topk_w = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)
        expect_fused = M * TOPK <= 1024
        # patched (default): deferred quantization, fused at decode shapes
        dgm._MOET_QS = None
        out_on, in_dtype_on = run(experts, prepare, x, topk_w, topk_ids, w13, w2, E, calls)
        path_on = calls[0] if calls else "?"
        # switch off: vLLM's quantization in prepare(), vLLM's permute
        dgm._MOET_QS = False
        out_off, in_dtype_off = run(experts, prepare, x, topk_w, topk_ids, w13, w2, E, calls)
        path_off = calls[0] if calls else "?"
        dgm._MOET_QS = None
        same = torch.equal(out_on, out_off)
        ok = (same and path_on == ("fused" if expect_fused else "vllm") and path_off == "vllm"
              and in_dtype_on == torch.bfloat16 and in_dtype_off == torch.float8_e4m3fn)
        fails += 0 if ok else 1
        d = (out_on.float() - out_off.float()).abs().max().item()
        print(f"[{'ok ' if ok else 'BAD'}] M={M:3d} pairs={M * TOPK:4d}: on -> {path_on} (input {in_dtype_on}), "
              f"off -> {path_off} (input {in_dtype_off}); output {'bit-identical' if same else f'DIFFERS max {d:.3e}'}")
    print("ALL OK" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
