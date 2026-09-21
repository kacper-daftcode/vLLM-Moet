#!/usr/bin/env python3
"""Op-level validation of the packed compressed-KV kernels (variant A) on one SM120 GPU.

  INSERT 288  rope_quant_insert_packed into the NVFP4 record == the checkpoint's
              fp4_act_quant(RoPE'd latent, 16, e4m3): codes and scales bit-exact
  INSERT 528  the V4.1 fp8 record == the checkpoint's act_quant(RoPE'd latent, 32, "ue8m0")
  GATHER 288  gather_requant_to_ds_mla(nvfp4 cache) == experiment A0's fp8_ds_mla records for the
              same tokens (patch_vllm_kv_fp4_fake.py) -- byte for byte, so the A0 end-to-end
              numbers are variant A's numbers
  GATHER 528  gather_requant_to_ds_mla(fp8_v41 cache) == torch dequant + vLLM's fp8_ds_mla recipe
  ATTENTION   FlashInfer SM120 sparse MLA (dual cache) on the scratch + remapped indices ==
              the same kernel on a real fp8_ds_mla cache holding the A0 records at the original
              slots (bit-exact), and close to the fp32 reference on the dequantized keys

usage (inside the serving image; /model = checkpoint, /t = tools/dsv41_sm120):
  python3 test_nvfp4_kv_kernels.py --model-dir /model --a0-patched /path/to/fcqc_patched.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from nvfp4_kv_kernels import (  # noqa: E402
    DS_MLA_PAGE,
    gather_requant_to_ds_mla,
    rope_quant_insert_packed,
    scratch_pages,
)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rope_bf16(latent, positions, cos_sin, cr):
    x = latent.float()
    cpos = (positions // cr) * cr
    cs = cos_sin[cpos.long()].float()
    c, s = cs[:, :32], cs[:, 32:]
    even, odd = x[:, 448::2], x[:, 449::2]
    out = x.clone()
    out[:, 448::2] = (even * c - odd * s).to(torch.bfloat16).float()
    out[:, 449::2] = (odd * c + even * s).to(torch.bfloat16).float()
    return out


def records(cache: torch.Tensor, slots: torch.Tensor, value_bytes: int, scale_bytes: int):
    """Segregated page layout -> (values [T, value_bytes], scales [T, scale_bytes]) for the given slots."""
    block = cache.shape[1]
    pg, pos = slots // block, slots % block
    flat = cache.view(cache.shape[0], -1)
    ar = torch.arange(slots.numel(), device=cache.device)
    vals = flat[pg, : block * value_bytes].view(-1, block, value_bytes)[ar, pos]
    sc = flat[pg, block * value_bytes:].view(-1, block, scale_bytes)[ar, pos]
    return vals, sc


def ds_mla_ref(full: torch.Tensor):
    nope = full[:, :448].reshape(-1, 7, 64)
    amax = nope.abs().amax(dim=-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(amax * (1.0 / 448.0)))
    fp8 = (nope * torch.exp2(-exponent).unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1, 448)
    rope = full[:, 448:].to(torch.bfloat16).view(torch.uint8)
    scales = torch.zeros(full.shape[0], 8, dtype=torch.uint8, device=full.device)
    scales[:, :7] = (exponent + 127.0).clamp(0, 255).to(torch.uint8)
    return torch.cat([fp8, rope], dim=1), scales


def dequant_528(vals: torch.Tensor, sc: torch.Tensor):
    fp8 = vals.view(torch.float8_e4m3fn).float().view(-1, 16, 32)
    return (fp8 * torch.exp2(sc.float() - 127.0).unsqueeze(-1)).view(-1, 512)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--a0-patched", required=True)
    ap.add_argument("--tokens", type=int, default=8192)
    args = ap.parse_args()
    sys.path.insert(0, f"{args.model_dir}/inference")
    import kernel as ref

    torch.manual_seed(21)
    dev = torch.device("cuda")
    T = args.tokens
    inv_freq = 1.0 / (160000.0 ** (torch.arange(0, 64, 2, device=dev).float() / 64))
    freqs = torch.outer(torch.arange(T + 8, device=dev).float(), inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)
    ok = True

    os.environ["VLLM_MOET_KV_FP4_FAKE"] = "1"
    a0 = load_module(args.a0_patched, "fcqc_a0")
    assert a0._KV_FP4_FAKE

    for cr, block in ((1, 128), (2, 64)):
        latent = torch.randn(T, 512, device=dev, dtype=torch.bfloat16)
        positions = torch.arange(T, device=dev, dtype=torch.int64)
        num_states = T // cr
        pages = (num_states + block - 1) // block + 1
        perm = torch.randperm(pages, device=dev)[: (num_states + block - 1) // block]
        state = positions // cr
        slots = perm[(state // block).long()] * block + state % block
        boundary = (positions + 1) % cr == 0
        slot_mapping = torch.where(boundary, slots, torch.full_like(slots, -1))
        full = rope_bf16(latent, positions, cos_sin, cr)[boundary]
        sl = slot_mapping[boundary]

        # ---- INSERT 288 vs the checkpoint quantizer
        c288 = torch.randint(0, 256, (pages, block, 288), device=dev, dtype=torch.uint8)
        rope_quant_insert_packed(latent, positions, cos_sin, c288, slot_mapping, cr)
        torch.cuda.synchronize()
        vals, sc = records(c288, sl, 256, 32)
        ref_codes, ref_scales = ref.fp4_act_quant(full.to(torch.bfloat16), 16, False, scale_dtype=torch.float8_e4m3fn)
        n_c = (vals != ref_codes.view(torch.uint8)).sum().item()
        n_s = (sc != ref_scales.view(torch.uint8)).sum().item()
        good = n_c == 0 and n_s == 0
        ok &= good
        print(f"[{'ok ' if good else 'BAD'}] INSERT 288 cr={cr} page={block}: e2m1 code bytes differ {n_c}/{vals.numel()}, e4m3 scales differ {n_s}/{sc.numel()}")

        # ---- INSERT 528 vs the checkpoint's act_quant(kv, 32, ue8m0)
        c528 = torch.randint(0, 256, (pages, block, 528), device=dev, dtype=torch.uint8)
        rope_quant_insert_packed(latent, positions, cos_sin, c528, slot_mapping, cr)
        torch.cuda.synchronize()
        vals5, sc5 = records(c528, sl, 512, 16)
        ref_y, ref_s = ref.act_quant(full.to(torch.bfloat16), 32, "ue8m0", torch.float8_e8m0fnu, False)
        n_c = (vals5 != ref_y.view(torch.uint8)).sum().item()
        n_s = (sc5 != ref_s.view(torch.uint8)).sum().item()
        good = n_c == 0 and n_s == 0
        ok &= good
        print(f"[{'ok ' if good else 'BAD'}] INSERT 528 cr={cr} page={block}: fp8 bytes differ {n_c}/{vals5.numel()}, ue8m0 scales differ {n_s}/{sc5.numel()}")

        # ---- A0 records for the same tokens (fp8_ds_mla of the FP4-quantized latent)
        c584 = torch.randint(0, 256, (pages, block, 584), device=dev, dtype=torch.uint8)
        a0.rope_quant_insert(latent, positions, cos_sin, c584, slot_mapping, cr)
        torch.cuda.synchronize()

        # ---- GATHER 288 -> scratch == A0 records
        rows, topk = 6, 512
        idx = torch.randint(0, num_states, (rows, topk), device=dev)
        idx = slots[boundary][idx.view(-1)].view(rows, topk).to(torch.int32)  # global slots of real states
        idx[:, topk * 3 // 4:] = -1
        idx[2, :] = -1  # an empty row
        scratch = torch.randint(0, 256, (scratch_pages(rows, topk), DS_MLA_PAGE, 584), device=dev, dtype=torch.uint8)
        remap = gather_requant_to_ds_mla(c288, idx, scratch)
        torch.cuda.synchronize()
        valid = idx >= 0
        exp_remap = torch.where(valid, torch.arange(rows * topk, device=dev, dtype=torch.int32).view(rows, topk), torch.full_like(idx, -1))
        remap_ok = torch.equal(remap, exp_remap)
        got_v, got_s = records(scratch, remap[valid], 576, 8)
        a0_v, a0_s = records(c584, idx[valid], 576, 8)
        n_v = (got_v != a0_v).sum().item()
        n_s = (got_s != a0_s).sum().item()
        good = remap_ok and n_v == 0 and n_s == 0
        ok &= good
        print(f"[{'ok ' if good else 'BAD'}] GATHER 288 cr={cr}: remap {'ok' if remap_ok else 'BAD'}, value bytes differ {n_v}/{got_v.numel()}, scale bytes differ {n_s}/{got_s.numel()} vs A0 records")

        # ---- CONTEXT dequant (prefill path): logical state i -> scratch slot i == A0 record of state i
        from nvfp4_kv_kernels import dequant_context_to_ds_mla

        bt_row = perm.to(torch.int32).contiguous()  # logical block j -> physical page perm[j]
        ctx_states = num_states - 5
        ctx_scratch = torch.randint(0, 256, ((ctx_states + 127) // 128, DS_MLA_PAGE, 584), device=dev, dtype=torch.uint8)
        dequant_context_to_ds_mla(c288, bt_row, ctx_states, ctx_scratch)
        torch.cuda.synchronize()
        st_ids = torch.arange(ctx_states, device=dev)
        got_v, got_s = records(ctx_scratch, st_ids, 576, 8)
        a0_v, a0_s = records(c584, slots[boundary][st_ids], 576, 8)  # state i lives at the i-th boundary token's slot
        n_v = (got_v != a0_v).sum().item()
        n_s = (got_s != a0_s).sum().item()
        good = n_v == 0 and n_s == 0
        ok &= good
        print(f"[{'ok ' if good else 'BAD'}] CONTEXT 288 cr={cr}: {ctx_states} states dequantized in logical order, value bytes differ {n_v}, scale bytes differ {n_s} vs A0 records")

        # ---- GATHER 528 -> scratch == torch reference
        scratch5 = torch.randint(0, 256, (scratch_pages(rows, topk), DS_MLA_PAGE, 584), device=dev, dtype=torch.uint8)
        remap5 = gather_requant_to_ds_mla(c528, idx, scratch5)
        torch.cuda.synchronize()
        got_v, got_s = records(scratch5, remap5[valid], 576, 8)
        src_v, src_s = records(c528, idx[valid], 512, 16)
        ref_v, ref_s2 = ds_mla_ref(dequant_528(src_v, src_s))
        n_v = (got_v != ref_v).sum().item()
        n_s = (got_s != ref_s2).sum().item()
        good = n_v == 0 and n_s == 0
        ok &= good
        print(f"[{'ok ' if good else 'BAD'}] GATHER 528 cr={cr}: value bytes differ {n_v}/{got_v.numel()}, scale bytes differ {n_s}/{got_s.numel()} vs torch reference")

        # ---- ATTENTION: FlashInfer dual-cache sparse MLA on (real A0 cache, idx) vs (scratch, remap)
        if cr == 1:
            import test_sparse_mla_sm120_dsv41 as h

            swa_deq, swa_packed = h.make_kv(64 * 32, dev, (32,))
            q = (torch.randn(rows, 16, 512, device=dev, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
            swa_idx = torch.randint(0, 64 * 32, (rows, 128), device=dev, dtype=torch.int32)
            swa_idx[:, 96:] = -1
            sm_scale = 512**-0.5
            out_a, lse_a = h.run_kernel(q, swa_packed[32], swa_idx, sm_scale, sink=None, topk_length=None,
                                        extra_kv=c584.unsqueeze(2), extra_idx=idx, extra_topk_length=None, decode_scratch=True)
            out_b, lse_b = h.run_kernel(q, swa_packed[32], swa_idx, sm_scale, sink=None, topk_length=None,
                                        extra_kv=scratch.unsqueeze(2), extra_idx=remap, extra_topk_length=None, decode_scratch=True)
            exact = torch.equal(out_a, out_b) and torch.equal(lse_a, lse_b)
            ok &= exact
            # fp32 reference over [swa | dequantized A0 compressed states]
            comp_deq = h.dequantize_kv_dsv4(c584.unsqueeze(2)).reshape(-1, 512)
            virtual = torch.cat([swa_deq.reshape(-1, 512), comp_deq], 0).reshape(-1, 1, 1, 512)
            e_shift = torch.where(idx < 0, idx, idx + 64 * 32)
            ref_out, ref_lse = h._ref_sparse_attn(q, virtual, torch.cat([swa_idx, e_shift], -1), sm_scale, 512)
            md = (out_b.float() - ref_out.float()).abs().max().item()
            ml = (lse_b - ref_lse).abs().max().item()
            close = md <= 5e-2 and ml <= 5e-2
            ok &= close
            print(f"[{'ok ' if exact and close else 'BAD'}] ATTENTION (dual, 16 heads, 6 rows, swa 128 + compressed 512): scratch vs real cache "
                  f"{'BIT-EXACT' if exact else 'DIFFERS'}; vs fp32 reference max|dout| {md:.2e}, max|dlse| {ml:.2e}")

    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
