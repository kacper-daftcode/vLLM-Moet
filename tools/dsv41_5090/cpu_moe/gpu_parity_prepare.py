"""Host side of the CPU-vs-GPU expert parity check (PLAN-dsv41-5090-cpu 3.1, run early).

    python gpu_parity_prepare.py --out /path/inputs.npz [--layer 20] [--expert 7] [--rank 0] [--tokens 5]

Writes the expert slice's raw MXFP4 record, T quantized hidden states (the fp8 codes + UE8M0 scale bytes the GPU
path would hand to the CPU), the torch emulation's intermediates and the CPU kernel's bf16 output. gpu_parity_run.py
(inside the vLLM image, one GPU) runs DeepGEMM's fp8 x fp4 grouped GEMMs and vLLM's activation kernel on the same
bytes and compares.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv41_mxfp4 as D  # noqa: E402
from test_cpu_moe_mxfp4 import realistic_hidden  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--expert", type=int, default=7)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    gen = torch.Generator().manual_seed(args.seed)
    raw = D.expert_slice_raw(args.layer, args.expert, args.rank)
    x = realistic_hidden(args.tokens, gen)
    xq, xe = D.quant_fp8_ue8m0(x.float())
    xq_b, xs_b = D.to_kernel_inputs(xq, xe)
    emu = D.ExpertEmulator(raw)
    o, inter = emu.forward_from_q(xq, xe, return_intermediates=True)
    lib = D.load_lib()
    arena = D.PackArena(1, lib)
    arena.slot(0)[:] = D.build_pack(raw)
    pool = D.Pool(list(range(8)), lib)
    out = np.zeros((args.tokens, D.HID), dtype=np.uint16)
    pool.experts_forward([arena.ptr(0)], [args.tokens], list(range(args.tokens)), xq_b, xs_b, out)
    pool.close()
    arena.close()
    cpu_o = D.bf16_bits_to_float(out).view(args.tokens, D.HID)
    print(f"layer {args.layer} expert {args.expert} rank {args.rank}: kernel vs emulation max |d| "
          f"{(cpu_o - o).abs().max().item():.3e}")
    np.savez(args.out,
             layer=args.layer, expert=args.expert, rank=args.rank,
             w1n=raw["w1n"], w1s=raw["w1s"], w3n=raw["w3n"], w3s=raw["w3s"], w2n=raw["w2n"], w2s=raw["w2s"],
             x_bf16=x.view(torch.int16).numpy(), xq=xq_b, xs=xs_b,
             emu_g=inter["g"].numpy(), emu_u=inter["u"].numpy(), emu_y=inter["y"].numpy(),
             emu_yq=inter["yq"].view(torch.uint8).numpy(), emu_ye=inter["ye"].numpy(),
             emu_out=o.numpy(), cpu_out_bf16=out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
