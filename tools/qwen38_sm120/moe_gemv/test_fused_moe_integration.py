#!/usr/bin/env python3
"""Integration check of the patched vLLM fused_moe.py (tools/qwen38_sm120/patch_fused_moe_sm120.py):
`invoke_fused_moe_triton_kernel` must dispatch decode-sized fp8 [32,32] launches to the sm_120
MoE GEMV and leave everything else on Triton, with identical outputs.

Run inside the built image (Dockerfile.sm120-qwen38), e.g.
    docker run --rm --gpus '"device=0"' --ipc host --entrypoint bash vllm-moet-sm120:qwen38-20073 -c \
      "python3 /opt/vllm-moet/qwen38_sm120/moe_gemv/test_fused_moe_integration.py"
(or against the official image with the patched fused_moe.py / triton_moe.py and the moe_gemv/
directory bind-mounted in; VLLM_MOET_SM120_MOE_GEMV_DIR points the patch at the module directory)
"""

from __future__ import annotations

import os
import sys

import torch

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.fused_moe import fused_moe as fm
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from vllm.triton_utils import tl

E, INTER, K, TOPK = 512, 160, 2560, 10
N1 = 2 * INTER
BLOCK = [32, 32]


def main() -> int:
    assert hasattr(fm, "_sm120_moe_gemv_module"), "fused_moe.py is not the patched file"
    mod = fm._sm120_moe_gemv_module()
    assert mod is not None, "GEMV module did not load (see log above)"
    print(f"GEMV module: {mod.__file__}  max_pairs={mod.max_pairs()} aligned_min_k={mod.aligned_min_k()}")
    from vllm.model_executor.layers.fused_moe.experts import triton_moe

    if hasattr(triton_moe, "_sm120_moe_gemv_module"):
        print(f"triton_moe.py patched (fused activation path): hook -> {triton_moe._sm120_moe_gemv_module() is mod}")
    else:
        print("triton_moe.py NOT patched: down GEMM keeps act_and_mul + quant + GEMV")

    calls: list[tuple[int, int, bool]] = []
    real = mod.fused_moe_gemv

    def counting(A, B, C, *args, **kw):
        calls.append((A.size(0), B.size(2), args[3] is not None))  # (rows, K, aligned)
        return real(A, B, C, *args, **kw)

    mod.fused_moe_gemv = counting
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3)
    w1 = (torch.randn(E, N1, K, device=dev, generator=gen) * 32).to(torch.float8_e4m3fn)
    w1_s = torch.exp2(torch.empty(E, N1 // 32, K // 32, device=dev).uniform_(-9, -5, generator=gen))
    w2 = (torch.randn(E, K, INTER, device=dev, generator=gen) * 32).to(torch.float8_e4m3fn)
    w2_s = torch.exp2(torch.empty(E, K // 32, INTER // 32, device=dev).uniform_(-9, -5, generator=gen))

    def run(M: int, force_triton: bool):
        config = fm.try_get_optimal_moe_config(w1.shape, w2.shape, TOPK, "fp8_w8a8", M, block_shape=BLOCK)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=gen)
        a1, a1_s = per_token_group_quant_fp8(x, 32)
        ids = torch.stack([torch.randperm(E, device=dev, generator=gen)[:TOPK] for _ in range(M)]).to(torch.int32)
        tw = torch.softmax(torch.randn(M, TOPK, device=dev, generator=gen), dim=-1).contiguous()
        s_ids, e_ids, npp = fm._prepare_expert_assignment(ids, config, M, TOPK, E, None, block_shape=BLOCK)
        c1 = torch.zeros(M, TOPK, N1, device=dev, dtype=torch.bfloat16)
        if force_triton:
            os.environ["VLLM_MOET_SM120_MOE_GEMV"] = "0"
        try:
            fm.invoke_fused_moe_triton_kernel(a1, w1, c1, a1_s, w1_s, None, s_ids, e_ids, npp, False, TOPK, config,
                                              compute_type=tl.bfloat16, use_fp8_w8a8=True, use_int8_w8a8=False,
                                              use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False,
                                              block_shape=BLOCK, B_bias=None)
            h = torch.empty(M * TOPK, INTER, device=dev, dtype=torch.bfloat16)
            torch.ops._C.silu_and_mul(h, c1.view(-1, N1))
            a2, a2_s = per_token_group_quant_fp8(h, 32)
            c3 = torch.zeros(M, TOPK, K, device=dev, dtype=torch.bfloat16)
            fm.invoke_fused_moe_triton_kernel(a2, w2, c3, a2_s, w2_s, tw, s_ids, e_ids, npp, True, 1, config,
                                              compute_type=tl.bfloat16, use_fp8_w8a8=True, use_int8_w8a8=False,
                                              use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False,
                                              block_shape=BLOCK, B_bias=None)
        finally:
            os.environ.pop("VLLM_MOET_SM120_MOE_GEMV", None)
        torch.cuda.synchronize()
        return c1, c3

    ok = True
    for M, expect in ((1, [(1, K, False), (10, INTER, False)]),
                      (4, [(4, K, False), (40, INTER, False)]),
                      (16, [(16, K, True)]),          # aligned: only the K=2560 GEMM goes to the GEMV
                      (32, [(32, K, True)]),
                      (64, [])):                       # 640 pairs > max_pairs -> Triton
        gen.manual_seed(100 + M)
        calls.clear()
        c1, c3 = run(M, False)
        got = list(calls)
        gen.manual_seed(100 + M)
        c1_t, c3_t = run(M, True)
        assert not calls[len(got):], "dispatch happened with VLLM_MOET_SM120_MOE_GEMV=0"
        d1 = (c1.float() - c1_t.float()).abs().max().item()
        d3 = (c3.float() - c3_t.float()).abs().max().item()
        same = got == expect and d1 <= 0.0625 and d3 <= 0.0625  # <= 1 bf16 ulp of O(1) values
        ok &= same
        print(f"[{'ok ' if same else 'BAD'}] M={M:2d}: GEMV calls {got} (expected {expect}); max|patched - triton| w13 {d1:.3e} w2 {d3:.3e}")
    # ---- whole TritonExperts.apply with vLLM's workspace layout (cache1/cache3 share
    # workspace2, the output aliases workspace13): fused activation path vs vLLM's sequence
    if hasattr(triton_moe, "_sm120_moe_gemv_module"):
        from math import prod

        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEConfig,
            FusedMoEParallelConfig,
            RoutingMethodType,
            fp8_w8a8_moe_quant_config,
        )
        from vllm.model_executor.layers.fused_moe.utils import _resize_cache

        moe_cfg = FusedMoEConfig(
            num_experts=E, experts_per_token=TOPK, hidden_dim=K, intermediate_size=INTER, num_local_experts=E,
            num_logical_experts=E, activation=MoEActivation.SILU, device=dev, routing_method=RoutingMethodType.Default,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(), in_dtype=torch.bfloat16,
        )
        experts = triton_moe.TritonExperts(moe_cfg, fp8_w8a8_moe_quant_config(w1_scale=w1_s, w2_scale=w2_s, block_shape=BLOCK))
        for M in (1, 4, 8):
            gen.manual_seed(200 + M)
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=gen)
            a1, a1_s = per_token_group_quant_fp8(x, 32)
            ids = torch.stack([torch.randperm(E, device=dev, generator=gen)[:TOPK] for _ in range(M)]).to(torch.int32)
            tw = torch.softmax(torch.randn(M, TOPK, device=dev, generator=gen), dim=-1).contiguous()
            outs = []
            for fuse in ("1", "0"):
                ws13_shape, ws2_shape, out_shape = experts.workspace_shapes(M, N1, K, TOPK, E, E, None, MoEActivation.SILU)
                common = torch.full((max(prod(ws13_shape), prod(out_shape)),), float("nan"), device=dev, dtype=torch.bfloat16)
                ws2 = torch.full(ws2_shape, float("nan"), device=dev, dtype=torch.bfloat16)
                ws13 = _resize_cache(common, ws13_shape)
                out = _resize_cache(common, out_shape)  # aliases workspace13, as in vLLM's modular kernel
                calls.clear()
                os.environ["VLLM_MOET_SM120_MOE_GEMV_FUSE_ACT"] = fuse
                try:
                    experts.apply(out, a1, w1, w2, tw, ids, MoEActivation.SILU, E, None, a1_s, None, ws13, ws2, None, False)
                finally:
                    os.environ.pop("VLLM_MOET_SM120_MOE_GEMV_FUSE_ACT", None)
                torch.cuda.synchronize()
                outs.append((out.clone(), len(calls)))
            (o_f, n_f), (o_u, n_u) = outs
            d = (o_f.float() - o_u.float()).abs().max().item()
            same = d == 0.0 and n_f == 1 and n_u == 2 and torch.isfinite(o_f).all().item()
            ok &= same
            print(f"[{'ok ' if same else 'BAD'}] TritonExperts.apply M={M}: fused path GEMV calls {n_f} (expect 1: w13 only), "
                  f"unfused {n_u} (expect 2); max|fused - unfused| = {d:.3e}")

    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
