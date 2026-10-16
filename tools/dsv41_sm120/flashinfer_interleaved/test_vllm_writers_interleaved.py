#!/usr/bin/env python3
"""vLLM's own V4.1 writers (vllm#61371: each token's scales right after its data) against FlashInfer's footer writers.

The same BF16 latents go through vLLM's quantize_and_insert_k_cache (528-byte MXFP8 SWA record) and rope_quant_insert
(288-byte NVFP4 compressed record, identity RoPE: cos 1 / sin 0) and through FlashInfer's
dsv41_fp{8,4}_quantize_pack_sparse_mla_cache (footer). Re-laid out, the footer pages must equal vLLM's bytes - then
the formats agree record for record - and FlashInfer's attention with kv_scale_placement="interleaved" over vLLM's
caches must equal its footer attention over its own caches, bit for bit. Run inside an image on a vLLM that writes
the interleaved records (the 49cb2bbf base on) with FlashInfer patched by patch_flashinfer_dsv41_interleaved.py.
usage: test_vllm_writers_interleaved.py"""
import sys

import torch
from flashinfer.mla import (
    dsv41_fp4_quantize_pack_sparse_mla_cache,
    dsv41_fp8_quantize_pack_sparse_mla_cache,
    trtllm_batch_decode_sparse_mla_dsv4,
)

from vllm.models.deepseek_v41.common.ops.cache_utils import quantize_and_insert_k_cache
from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import rope_quant_insert


def interleave(footer: torch.Tensor, page: int, data: int, scale: int) -> torch.Tensor:
    pages = footer.shape[0]
    flat = footer.reshape(pages, page * (data + scale))
    rows = torch.cat([flat[:, : page * data].view(pages, page, data), flat[:, page * data :].view(pages, page, scale)], 2)
    return rows.reshape(pages, page * (data + scale))


def compare(name: str, ours: torch.Tensor, theirs: torch.Tensor, page: int, data: int, scale: int) -> bool:
    a = ours.view(-1, page, data + scale)
    b = theirs.view(-1, page, data + scale)
    data_eq = (a[..., :data] == b[..., :data]).float().mean().item()
    scale_eq = (a[..., data:] == b[..., data:]).float().mean().item()
    same = torch.equal(a, b)
    print(f"{name:44s} bytes {'EQUAL' if same else 'DIFFER'} (data {data_eq:.6f}, scales {scale_eq:.6f} equal)")
    return same


def main() -> int:
    torch.manual_seed(0)
    dev = torch.device("cuda")
    ok = True

    pages, page = 16, 64
    latent = torch.randn(pages * page, 512, device=dev, dtype=torch.bfloat16)
    vllm_main = torch.zeros(pages, page * 528, dtype=torch.uint8, device=dev)
    slots = torch.arange(pages * page, device=dev, dtype=torch.int64)
    quantize_and_insert_k_cache(latent, vllm_main, slots, block_size=page, bytes_per_token=528)
    fi_main = dsv41_fp8_quantize_pack_sparse_mla_cache(latent.view(pages, page, 512), kv_layout="NHD").contiguous()
    ok &= compare("SWA 528 B (MXFP8): vLLM vs FlashInfer", vllm_main, interleave(fi_main, page, 512, 16), page, 512, 16)

    cpages, cpage = 24, 32
    clatent = torch.randn(cpages * cpage, 512, device=dev, dtype=torch.bfloat16)
    vllm_comp = torch.zeros(cpages, cpage, 288, dtype=torch.uint8, device=dev)
    n = cpages * cpage
    cos_sin = torch.cat([torch.ones(n, 32), torch.zeros(n, 32)], 1).to(dev)
    positions = torch.arange(n, device=dev, dtype=torch.int64)
    rope_quant_insert(clatent, positions, cos_sin, vllm_comp, positions, compress_ratio=1)
    fi_comp = dsv41_fp4_quantize_pack_sparse_mla_cache(clatent.view(cpages, cpage, 512), kv_layout="NHD").contiguous()
    ok &= compare("compressed 288 B (NVFP4): vLLM vs FlashInfer", vllm_comp, interleave(fi_comp, cpage, 256, 32),
                  cpage, 256, 32)

    heads, tokens = 16, 8
    q = torch.randn(tokens, heads, 512, device=dev, dtype=torch.bfloat16)
    swa = torch.randint(0, pages * page, (tokens, 128), device=dev, dtype=torch.int32)
    extra = torch.randint(0, n, (tokens, 512), device=dev, dtype=torch.int32)
    lens = torch.full((tokens,), 128, device=dev, dtype=torch.int32)
    extra_lens = torch.full((tokens,), 512, device=dev, dtype=torch.int32)
    ws = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    sink = torch.zeros(heads, device=dev)

    def attend(main, comp, placement):
        out = torch.empty(tokens, heads, 512, device=dev, dtype=torch.bfloat16)
        trtllm_batch_decode_sparse_mla_dsv4(
            query=q, swa_kv_cache=main.view(pages, page, 1, 528), workspace_buffer=ws, sparse_indices=swa,
            compressed_kv_cache=comp.view(cpages, cpage, 1, 288), out=out, bmm1_scale=512 ** -0.5, sinks=sink,
            kv_layout="NHD", swa_topk_lens=lens, extra_sparse_indices=extra, extra_sparse_topk_lens=extra_lens,
            kv_cache_format="fp8_dsv41_fp4_ca", kv_scale_placement=placement)
        torch.cuda.synchronize()
        return out

    ours = attend(vllm_main, vllm_comp, "interleaved")
    theirs = attend(fi_main, fi_comp, "footer")
    same = torch.equal(ours, theirs)
    diff = (ours.float() - theirs.float()).abs().max().item()
    print(f"{'attention: vLLM caches interleaved vs FlashInfer footer':44s} {'BIT-EXACT' if same else f'max diff {diff:.3e}'}")
    ok &= same
    print("ALL OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
