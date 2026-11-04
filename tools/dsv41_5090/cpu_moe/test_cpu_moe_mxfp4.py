"""Exactness of the CPU MXFP4 expert kernel against the torch emulation of the GPU's W4A8 math.

    python test_cpu_moe_mxfp4.py [--layer 20] [--experts 0,1,2,3] [--rank 0] [--tokens 13] [--cpus 0-7]

Checks, on real expert slices from the checkpoint:
  1. pack round trip (the nibble permutation),
  2. prepare_x: fp8 codes + UE8M0 scales -> bf16 with the scale folded (bit-exact),
  3. act_quant: FC1 outputs -> clamp-SwiGLU -> bf16 -> fp8 per 128 (codes, scales, folded bf16; bit-exact up to
     expf vs torch.exp ulps that the bf16 rounding of y does not absorb — counted),
  4. the whole expert for m = 1..8 and a multi-expert batch with mixed m (incl. m > 8): the kernel's bf16 rows
     against the fp32 emulation and an fp64 reference, next to the fp32-vs-fp64 floor (summation order).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv41_mxfp4 as D  # noqa: E402


def parse_cpus(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def realistic_hidden(T: int, gen: torch.Generator) -> torch.Tensor:
    """bf16 hidden states with a few heavy channels (the MoE input is post-RMSNorm, with outliers)."""
    x = torch.randn(T, D.HID, generator=gen)
    heavy = torch.randperm(D.HID, generator=gen)[:48]
    x[:, heavy] *= 12.0
    x[:, heavy[:4]] *= 8.0
    return x.to(torch.bfloat16)


def compare(name: str, a: torch.Tensor, b: torch.Tensor, floor: float | None = None) -> float:
    """max |a-b| / rms(b), and counts of elements off by more than 1 / 2 bf16 ulps of b."""
    a = a.float()
    b = b.float()
    rms = b.pow(2).mean().sqrt().item()
    diff = (a - b).abs()
    ulp = (b.abs().clamp(min=1e-30).log2().floor().exp2() * 2.0 ** -7)  # bf16 ulp of b
    n1 = int((diff > ulp * 1.0001).sum())
    n2 = int((diff > ulp * 2.0001).sum())
    rel = diff.max().item() / max(rms, 1e-30)
    mean_rel = (diff.mean() / max(rms, 1e-30)).item()
    tag = "" if floor is None else f"   (fp32-vs-fp64 floor: {floor:.2e})"
    print(f"  {name:38s} max|d|/rms {rel:.2e}  mean|d|/rms {mean_rel:.2e}  >1ulp {n1:6d}  >2ulp {n2:5d}  of {a.numel()}{tag}")
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--experts", default="0,1,2,3")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=13)
    ap.add_argument("--cpus", default="0-7")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    experts = [int(e) for e in args.experts.split(",")]
    gen = torch.Generator().manual_seed(args.seed)
    lib = D.load_lib()
    torch.set_num_threads(8)

    # --- 1. packs -----------------------------------------------------------------------------------------
    t0 = time.time()
    raws = [D.expert_slice_raw(args.layer, e, args.rank) for e in experts]
    arena = D.PackArena(len(experts), lib)
    for i, raw in enumerate(raws):
        pack = D.build_pack(raw)
        back = D.unpack_pack(pack)
        for k in raw:
            assert np.array_equal(back[k], raw[k]), f"pack round trip failed on {k}"
        arena.slot(i)[:] = pack
    print(f"[1] {len(experts)} expert slices read + packed + round-tripped in {time.time() - t0:.1f}s "
          f"(pack {D.PACK_BYTES} B, record {D.RAW_BYTES} B)")

    # --- 2. prepare_x -------------------------------------------------------------------------------------
    T = args.tokens
    x = realistic_hidden(T, gen)
    xq, xe = D.quant_fp8_ue8m0(x.float())
    xq_b, xs_b = D.to_kernel_inputs(xq, xe)
    xb16 = np.zeros((T, D.HID), dtype=np.uint16)
    lib.moe_mxfp4_prepare_x(xq_b.ctypes.data, xs_b.ctypes.data, xb16.ctypes.data, T)
    xd_ref = D.dequant_fp8_ue8m0(xq, xe)
    xd_k = D.bf16_bits_to_float(xb16).view(T, D.HID)
    exact = torch.equal(xd_k, xd_ref)
    print(f"[2] prepare_x bit-exact vs emulation: {exact}  (max |d| {((xd_k - xd_ref).abs().max()).item():.3e})")
    assert exact

    # --- 3. act_quant -------------------------------------------------------------------------------------
    emu32 = [D.ExpertEmulator(r, torch.float32) for r in raws]
    emu64 = [D.ExpertEmulator(r, torch.float64) for r in raws]
    o_ref, inter = emu32[0].forward_from_q(xq, xe, return_intermediates=True)
    g = inter["g"].contiguous().numpy()
    u = inter["u"].contiguous().numpy()
    yb16 = np.zeros((T, D.K2), dtype=np.uint16)
    yq_k = np.zeros((T, D.K2), dtype=np.uint8)
    ys_k = np.zeros((T, 5), dtype=np.uint8)
    lib.moe_mxfp4_act_quant(g.ctypes.data, u.ctypes.data, yb16.ctypes.data, yq_k.ctypes.data, ys_k.ctypes.data, T)
    yq_ref = inter["yq"].view(torch.uint8).numpy()
    ys_ref = (inter["ye"] + 127).to(torch.uint8).numpy()
    n_code = int((yq_k != yq_ref).sum())
    n_scale = int((ys_k != ys_ref).sum())
    yd_ref = D.dequant_fp8_ue8m0(inter["yq"], inter["ye"])
    yd_k = D.bf16_bits_to_float(yb16).view(T, D.K2)
    print(f"[3] act_quant: fp8 code mismatches {n_code} of {yq_k.size}, scale mismatches {n_scale} of {ys_k.size}, "
          f"folded bf16 equal {torch.equal(yd_k, yd_ref)}")

    # --- 4. the expert, m = 1..8 on expert 0 -------------------------------------------------------------
    print("[4] expert forward, kernel vs emulation (expert %d, rank %d, layer %d)" % (experts[0], args.rank, args.layer))
    pool = D.Pool(parse_cpus(args.cpus), lib)
    o64 = emu64[0].forward_from_q(xq, xe)
    floor = compare("emulation fp32 vs fp64 (floor)", o_ref, o64)
    worst = 0.0
    for m in range(1, 9):
        toks = list(range(m))
        out = np.zeros((m, D.HID), dtype=np.uint16)
        pool.experts_forward([arena.ptr(0)], [m], toks, xq_b, xs_b, out)
        o_k = D.bf16_bits_to_float(out).view(m, D.HID)
        worst = max(worst, compare(f"m={m} kernel vs fp32 emulation", o_k, o_ref[:m], floor))
        compare(f"m={m} kernel vs fp64 reference", o_k, o64[:m])
    # --- multi-expert batch, mixed m, shuffled token lists, m > 8 ----------------------------------------
    rng = np.random.default_rng(args.seed)
    ms = [T, 1, 5, 3][: len(experts)]
    token_lists = [list(rng.permutation(T)[:m]) for m in ms]
    out = np.zeros((sum(ms), D.HID), dtype=np.uint16)
    pool.experts_forward([arena.ptr(i) for i in range(len(ms))], ms, [int(t) for tl in token_lists for t in tl],
                         xq_b, xs_b, out)
    o_k = D.bf16_bits_to_float(out).view(sum(ms), D.HID)
    row = 0
    for i, (m, tl) in enumerate(zip(ms, token_lists)):
        ref = emu32[i].forward_from_q(xq[tl], xe[tl])
        ref64 = emu64[i].forward_from_q(xq[tl], xe[tl])
        fl = compare(f"batch expert {experts[i]} m={m} fp32 vs fp64", ref, ref64)
        worst = max(worst, compare(f"batch expert {experts[i]} m={m} kernel vs fp32", o_k[row:row + m], ref, fl))
        row += m
    pool.close()
    arena.close()
    verdict = "PASS" if worst < 10 * max(floor, 1e-6) else "CHECK"
    print(f"[4] worst kernel-vs-emulation max|d|/rms {worst:.2e} vs floor {floor:.2e} -> {verdict}")


if __name__ == "__main__":
    main()
