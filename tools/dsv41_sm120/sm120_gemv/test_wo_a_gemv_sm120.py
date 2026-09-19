#!/usr/bin/env python3
"""Correctness + timing of mxfp8_gemv_grouped for DeepSeek-V4.1-Flash's wo_a at TP4
(2 head groups x [1024 <- 4096] per rank) against an fp32 reference on the same
quantized operands, plus the production sm_120 path (BF16 weights + cuBLAS bmm).

Run inside the serving image on one sm_120 GPU:
    python3 test_wo_a_gemv_sm120.py [--ts 1,6,16,32,48,64] [--iters 50]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mxfp8_gemv_sm120 import mxfp8_gemv_grouped  # noqa: E402

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    MXFP8_BLOCK_SIZE,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (  # noqa: E402
    fused_inv_rope_fp8_quant,
)

# DeepSeek-V4.1-Flash per TP4 rank: 16 local heads in 2 groups, head_dim 512 (448 + 64),
# o_lora_rank 1024 -> wo_a weight [2 * 1024, 8 * 512].
N_GROUPS, HEADS_PER_GROUP, NOPE, ROPE, O_LORA = 2, 8, 448, 64, 1024
HEAD_DIM = NOPE + ROPE
K = HEADS_PER_GROUP * HEAD_DIM
N = N_GROUPS * O_LORA


def unpack_scales(sf_tgs: torch.Tensor, k_blocks: int) -> torch.Tensor:
    """[T, G, S] int32 packed ue8m0 -> [T, G, k_blocks] float32 scale values."""
    words = sf_tgs.contiguous().to(torch.int64)
    bytes_ = torch.stack([(words >> (8 * j)) & 0xFF for j in range(4)], dim=-1)
    e = bytes_.reshape(*words.shape[:-1], -1)[..., :k_blocks].to(torch.float32)
    return torch.exp2(e - 127.0)


def bench(fn, rotate, iters: int) -> float:
    per_graph = len(rotate)
    for i in range(3):
        fn(*rotate[i % per_graph])
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for i in range(2):
            fn(*rotate[i % per_graph])
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for i in range(per_graph):
                fn(*rotate[i])
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(max(3, iters // 5)):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) * 1000 / per_graph)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--ts", default="1,6,16,32,48,64")
    args = ap.parse_args()
    torch.manual_seed(7)
    dev = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")

    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    w_q, w_sf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    w_sf = w_sf.view(N, K // MXFP8_BLOCK_SIZE).contiguous()
    w_deq = dequant_mxfp8_to_bf16(w_q, w_sf)  # bf16 weights of the production path
    w_deq_f32 = w_deq.float()
    max_pos = 8192
    inv_freq = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=dev, dtype=torch.float32) / ROPE))
    t = torch.arange(max_pos, device=dev, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()  # [max_pos, ROPE]

    n_fp8 = max(2, (160 << 20) // (N * K) + 1)
    n_bf16 = max(2, (160 << 20) // (N * K * 2) + 1)
    w_q_copies = [w_q.clone() for _ in range(n_fp8)]
    w_deq_copies = [w_deq.view(N_GROUPS, O_LORA, K).clone() for _ in range(n_bf16)]

    fails = 0
    for T in [int(x) for x in args.ts.split(",")]:
        o = torch.randn(T, N_GROUPS * HEADS_PER_GROUP, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        positions = torch.randint(0, max_pos, (T,), device=dev, dtype=torch.int64)
        q, sf = fused_inv_rope_fp8_quant(
            o, positions, cos_sin, n_groups=N_GROUPS, heads_per_group=HEADS_PER_GROUP,
            nope_dim=NOPE, rope_dim=ROPE, quant_group_size=MXFP8_BLOCK_SIZE,
            tma_aligned_scales=True, quantize=True,
        )
        x_bf16, _ = fused_inv_rope_fp8_quant(
            o, positions, cos_sin, n_groups=N_GROUPS, heads_per_group=HEADS_PER_GROUP,
            nope_dim=NOPE, rope_dim=ROPE, quant_group_size=MXFP8_BLOCK_SIZE,
            tma_aligned_scales=True, quantize=False,
        )
        assert q.shape == (T, N_GROUPS, K) and sf.shape[:2] == (T, N_GROUPS), (q.shape, sf.shape)
        # fp32 reference on the dequantized fp8 operands
        scales = unpack_scales(sf, K // MXFP8_BLOCK_SIZE)  # [T, G, KB]
        a_deq = (q.float().view(T, N_GROUPS, K // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
                 * scales.unsqueeze(-1)).view(T, N_GROUPS, K)
        ref = torch.einsum("tgk,gnk->tgn", a_deq, w_deq_f32.view(N_GROUPS, O_LORA, K))
        # production path: bf16 activations x bf16 (dequantized) weights
        z_prod = torch.empty((T, N_GROUPS, O_LORA), device=dev, dtype=torch.bfloat16)
        torch.bmm(x_bf16.transpose(0, 1), w_deq.view(N_GROUPS, O_LORA, K).transpose(1, 2),
                  out=z_prod.transpose(0, 1))
        ours = mxfp8_gemv_grouped(q.transpose(0, 1), sf.transpose(0, 1), w_q, w_sf)
        assert ours.shape == (T, N_GROUPS, O_LORA)
        err_ref = ((ours.float() - ref).abs() / (ref.abs() + 1e-2)).max().item()
        err_prod = ((ours.float() - z_prod.float()).abs() / (z_prod.float().abs() + 1e-2)).max().item()
        ok = err_ref < 1.5e-2  # bf16 output rounding of an fp32-exact reference
        fails += 0 if ok else 1

        t_prod = bench(
            lambda wc: torch.bmm(x_bf16.transpose(0, 1), wc.transpose(1, 2), out=z_prod.transpose(0, 1)),
            [(c,) for c in w_deq_copies], args.iters,
        )
        t_ours = bench(
            lambda wc: mxfp8_gemv_grouped(q.transpose(0, 1), sf.transpose(0, 1), wc, w_sf),
            [(c,) for c in w_q_copies], args.iters,
        )
        gbps = N * K / t_ours / 1e3
        print(
            f"[{'ok ' if ok else 'BAD'}] T={T:2d}  maxrel vs fp32-ref={err_ref:.2e}  vs bf16 path={err_prod:.2e}  "
            f"bf16 bmm {t_prod:6.1f} us  grouped gemv {t_ours:6.1f} us  x{t_prod/t_ours:4.1f}  ({gbps:5.0f} GB/s fp8)"
        )
    # the two fused_inv_rope variants cost the same (informational)
    T = 6
    o = torch.randn(T, N_GROUPS * HEADS_PER_GROUP, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (T,), device=dev, dtype=torch.int64)
    for quant in (False, True):
        tq = bench(
            lambda: fused_inv_rope_fp8_quant(
                o, positions, cos_sin, n_groups=N_GROUPS, heads_per_group=HEADS_PER_GROUP,
                nope_dim=NOPE, rope_dim=ROPE, quant_group_size=MXFP8_BLOCK_SIZE,
                tma_aligned_scales=True, quantize=quant),
            [()] * 10, args.iters,
        )
        print(f"fused_inv_rope_fp8_quant(quantize={quant}) T={T}: {tq:5.1f} us")
    print(f"\n{'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
