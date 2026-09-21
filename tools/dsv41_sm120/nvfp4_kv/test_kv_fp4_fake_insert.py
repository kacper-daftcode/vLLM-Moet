#!/usr/bin/env python3
"""A0 insert kernel check: patched `rope_quant_insert` (VLLM_MOET_KV_FP4_FAKE=1) writes
fp8_ds_mla records of the FP4-quantized RoPE'd latent; with the switch off it is byte-identical
to the stock kernel.

Reference for the FP4 step: the checkpoint's `inference/kernel.py::fp4_act_quant` (inplace, e4m3/16)
applied to the RoPE'd bf16 latent exactly as DeepSeek's `_compress_kv` does; the fp8_ds_mla step is
vLLM's own UE8M0/64 recipe re-implemented in torch (amax floor 1e-4, exponent = ceil(log2(amax/448)),
e4m3 round-to-nearest).

usage (inside the serving image, /model = checkpoint):
  python3 test_kv_fp4_fake_insert.py --model-dir /model --patched /path/to/patched_fused_compress_quant_cache.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import torch


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rope_bf16(latent: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, cr: int) -> torch.Tensor:
    """vLLM kernel's RoPE on the last 64 dims (fp32 math, bf16 cos/sin), rounded to bf16 -> fp32 [T, 512]."""
    x = latent.float()
    cpos = (positions // cr) * cr
    cs = cos_sin[cpos.long()].float()
    c, s = cs[:, :32], cs[:, 32:]
    even, odd = x[:, 448::2], x[:, 449::2]
    r_even = (even * c - odd * s).to(torch.bfloat16).float()
    r_odd = (odd * c + even * s).to(torch.bfloat16).float()
    out = x.clone()
    out[:, 448::2], out[:, 449::2] = r_even, r_odd
    return out


def fp8_ds_mla_bytes(full: torch.Tensor):
    """torch emulation of vLLM's NoPE quantization: returns (fp8 bytes [T, 448] uint8, scale bytes [T, 8])."""
    nope = full[:, :448].reshape(-1, 7, 64)
    amax = nope.abs().amax(dim=-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(amax * (1.0 / 448.0)))
    scaled = (nope * torch.exp2(-exponent).unsqueeze(-1)).clamp(-448.0, 448.0)
    fp8 = scaled.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1, 448)
    scales = torch.zeros(full.shape[0], 8, dtype=torch.uint8, device=full.device)
    scales[:, :7] = (exponent + 127.0).clamp(0, 255).to(torch.uint8)
    return fp8, scales


def read_records(cache: torch.Tensor, slots: torch.Tensor, block: int):
    """fp8_ds_mla page layout -> (fp8 bytes [T,448], rope bf16 [T,64], scale bytes [T,8]) for the given slots."""
    pg, pos = slots // block, slots % block
    flat = cache.view(cache.shape[0], -1)
    data = flat[pg, : block * 576].view(-1, block, 576)[torch.arange(slots.numel(), device=cache.device), pos]
    sc = flat[pg, block * 576:].view(-1, block, 8)[torch.arange(slots.numel(), device=cache.device), pos]
    return data[:, :448], data[:, 448:].contiguous().view(torch.bfloat16), sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--patched", required=True, help="fused_compress_quant_cache.py after patch_vllm_kv_fp4_fake.py")
    ap.add_argument("--tokens", type=int, default=4096)
    args = ap.parse_args()
    sys.path.insert(0, f"{args.model_dir}/inference")
    import kernel as ref

    import vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache as stock

    torch.manual_seed(9)
    dev = torch.device("cuda")
    T, block = args.tokens, 128
    max_pos = T + 8
    inv_freq = 1.0 / (160000.0 ** (torch.arange(0, 64, 2, device=dev).float() / 64))
    freqs = torch.outer(torch.arange(max_pos, device=dev).float(), inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)
    ok = True
    for cr in (1, 2):
        latent = torch.randn(T, 512, device=dev, dtype=torch.bfloat16)
        positions = torch.arange(T, device=dev, dtype=torch.int64)
        num_states = T // cr
        num_pages = (num_states + block - 1) // block + 1
        perm = torch.randperm(num_pages, device=dev)[: (num_states + block - 1) // block]
        state = positions // cr
        slots = perm[(state // block).long()] * block + state % block
        boundary = (positions + 1) % cr == 0
        slot_mapping = torch.where(boundary, slots, torch.full_like(slots, -1))
        slot_mapping[3] = -1  # a padded token

        for fake in (False, True):
            os.environ["VLLM_MOET_KV_FP4_FAKE"] = "1" if fake else "0"
            mod = load_module(args.patched, f"fcqc_patched_{int(fake)}_{cr}")
            assert mod._KV_FP4_FAKE is fake
            cache = torch.randint(0, 256, (num_pages, block, 584), device=dev, dtype=torch.uint8)
            mod.rope_quant_insert(latent, positions, cos_sin, cache, slot_mapping, cr)
            torch.cuda.synchronize()
            written = slot_mapping >= 0
            fp8_got, rope_got, sc_got = read_records(cache, slot_mapping[written], block)

            full = rope_bf16(latent, positions, cos_sin, cr)  # fp32, RoPE'd, bf16-valued
            if fake:
                bf = full.to(torch.bfloat16).clone()
                ref.fp4_act_quant(bf, 16, True, scale_dtype=torch.float8_e4m3fn)  # the checkpoint's quantizer, in place
                full = bf.float()
            fp8_ref, sc_ref = fp8_ds_mla_bytes(full)
            rope_ref = full[:, 448:].to(torch.bfloat16)
            n_fp8 = (fp8_got != fp8_ref[written]).sum().item()
            n_sc = (sc_got != sc_ref[written]).sum().item()
            n_rope = (rope_got.view(torch.int16) != rope_ref[written].view(torch.int16)).sum().item()
            good = n_fp8 == 0 and n_sc == 0 and n_rope == 0
            ok &= good
            print(f"[{'ok ' if good else 'BAD'}] cr={cr} FP4_FAKE={int(fake)}: fp8 bytes differ {n_fp8}/{fp8_got.numel()}, "
                  f"scale bytes differ {n_sc}/{sc_got.numel()}, rope bf16 differ {n_rope}/{rope_got.numel()}")
            if not fake:
                # the switch off must reproduce the stock kernel byte for byte
                cache2 = torch.randint(0, 256, (num_pages, block, 584), device=dev, dtype=torch.uint8)
                stock.rope_quant_insert(latent, positions, cos_sin, cache2, slot_mapping, cr)
                torch.cuda.synchronize()
                a = read_records(cache, slot_mapping[written], block)
                b = read_records(cache2, slot_mapping[written], block)
                same = all(torch.equal(x.view(torch.uint8) if x.dtype != torch.uint8 else x, y.view(torch.uint8) if y.dtype != torch.uint8 else y) for x, y in zip(a, b))
                ok &= same
                print(f"      stock kernel parity with the switch off: {'BIT-EXACT' if same else 'DIFFERS'}")
            else:
                deq_err = ((full - rope_bf16(latent, positions, cos_sin, cr)).norm() / rope_bf16(latent, positions, cos_sin, cr).norm()).item()
                print(f"      info: FP4 quantization rel rms error on this latent {deq_err:.3e}")
    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
