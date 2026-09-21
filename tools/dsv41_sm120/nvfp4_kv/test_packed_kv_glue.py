#!/usr/bin/env python3
"""The vLLM glue of the packed compressed KV (moet_packed_kv.py as installed by
patch_vllm_packed_kv_sm120.py) with stand-in layer / metadata objects, on one GPU:

  * packed_gather_for_attention: the index-source layer gathers, its consumers get the same
    (scratch, remap) objects back without a new gather; a different index set gathers again
  * packed_prefill_segments, pool fits: one segment per chunk, pool slot base_k + i holds state i of
    request k (== gather_requant of that state), indices = local + base_k, out-of-range/-1 -> -1;
    the kv-source layer fills the pool, a consumer layer reuses it (no new dequant)
  * packed_prefill_segments, pool too small: one segment per request, base 0, every layer dequantizes

usage: python3 test_packed_kv_glue.py --module /path/to/moet_packed_kv.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from types import SimpleNamespace

import torch


def load(path, name="moet_packed_kv"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def records(cache, slots, value_bytes, scale_bytes):
    block = cache.shape[1]
    pg, pos = slots // block, slots % block
    flat = cache.view(cache.shape[0], -1)
    ar = torch.arange(slots.numel(), device=cache.device)
    return flat[pg, : block * value_bytes].view(-1, block, value_bytes)[ar, pos], flat[pg, block * value_bytes:].view(-1, block, scale_bytes)[ar, pos]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    args = ap.parse_args()
    os.environ["VLLM_MOET_KV_RECORD"] = "nvfp4"
    os.environ["VLLM_MOET_KV_PREFILL_POOL_STATES"] = "4096"  # small pool so the fallback is reachable
    m = load(args.module)
    dev = torch.device("cuda")
    torch.manual_seed(3)
    ok = True

    # a compressed cache with random records (contents irrelevant for the glue; use the insert kernel for realism)
    pages, block = 64, 128
    cache = torch.zeros(pages, block, 288, device=dev, dtype=torch.uint8)
    T = pages * block
    latent = torch.randn(T, 512, device=dev, dtype=torch.bfloat16)
    pos = torch.arange(T, device=dev)
    cs = torch.randn(T + 8, 64, device=dev).to(torch.bfloat16)
    m.rope_quant_insert_packed(latent, pos, cs, cache, torch.arange(T, device=dev), 1)

    # ---------------- decode sharing
    L20 = SimpleNamespace(layer_id=20, kv_source_layer_id=20, index_source_layer_id=20)
    L21 = SimpleNamespace(layer_id=21, kv_source_layer_id=20, index_source_layer_id=20)
    L24 = SimpleNamespace(layer_id=24, kv_source_layer_id=20, index_source_layer_id=24)
    idx = torch.randint(0, T, (6, 1, 512), device=dev, dtype=torch.int32)
    idx[:, :, 400:] = -1
    kv_a, remap_a = m.packed_gather_for_attention(cache, idx, layer=L20)
    torch.cuda.synchronize()
    ref_scratch = torch.empty_like(kv_a.squeeze(2))
    ref_remap = m.gather_requant_to_ds_mla(cache, idx.view(6, 512), ref_scratch)
    same = torch.equal(kv_a.squeeze(2)[: (6 * 512 + 127) // 128], ref_scratch[: (6 * 512 + 127) // 128]) and torch.equal(remap_a.view(6, 512), ref_remap)
    # consumer: identical objects back, no gather (poison the cache copy to prove nothing recomputes)
    kv_b, remap_b = m.packed_gather_for_attention(cache, idx, layer=L21)
    reuse = kv_b is kv_a and remap_b is remap_a
    # another index set: recomputed (different remap object)
    idx2 = torch.randint(0, T, (6, 1, 512), device=dev, dtype=torch.int32)
    kv_c, remap_c = m.packed_gather_for_attention(cache, idx2, layer=L24)
    recomputed = remap_c is not remap_a
    # decode scratch pages: for the default 64 rows
    print(f"[{'ok ' if same and reuse and recomputed else 'BAD'}] decode gather: == gather_requant {same}; consumer reuse {reuse}; new index set recomputes {recomputed}; scratch {tuple(kv_a.shape)}")
    ok &= same and reuse and recomputed

    # ---------------- prefill, pool fits (2 prefill requests after 1 decode request)
    cr = 1
    seq_lens = [1000, 700]  # prefill requests' total lengths
    q_lens = [300, 700]  # tokens in this step
    num_decodes, num_decode_tokens = 1, 3
    qsl = torch.tensor([0, 3, 3 + q_lens[0], 3 + q_lens[0] + q_lens[1]], dtype=torch.int32)  # cpu
    # block tables: request rows in the compressed cache (decode req 0, prefill reqs 1, 2)
    max_blocks = (max(seq_lens) + block - 1) // block
    bt = torch.randperm(pages, device=dev)[: 3 * max_blocks].view(3, max_blocks).to(torch.int32)
    swa_md = SimpleNamespace(num_decodes=num_decodes, num_decode_tokens=num_decode_tokens, num_prefills=2,
                             query_start_loc_cpu=qsl, prefill_seq_lens_cpu=torch.tensor(seq_lens), prefill_max_model_len=8192)
    attn_md = SimpleNamespace(block_table=bt)
    topk = 512
    topk_buf = torch.randint(0, 2000, (num_decode_tokens + sum(q_lens), topk), device=dev, dtype=torch.int32)
    topk_buf[:, 450:] = -1
    layer_src = SimpleNamespace(compress_ratio=cr, layer_id=20, kv_source_layer_id=20, index_source_layer_id=20, is_kv_source=True, topk_indices_buffer=topk_buf)
    layer_con = SimpleNamespace(compress_ratio=cr, layer_id=23, kv_source_layer_id=20, index_source_layer_id=20, is_kv_source=False, topk_indices_buffer=topk_buf)
    segs = list(m.packed_prefill_segments(layer_src, cache, attn_md, swa_md, 0, 2, prefill_token_base=3))
    torch.cuda.synchronize()
    one_segment = len(segs) == 1 and segs[0][0] == 0 and segs[0][1] == sum(q_lens)
    r0, r1, pool_kv, sidx = segs[0]
    pool = pool_kv.squeeze(2)
    bases = [0, 1024]  # 1000 states padded to 1024
    good_pool = True
    for k, (n_states, base) in enumerate(zip(seq_lens, bases)):
        st = torch.arange(n_states, device=dev)
        phys = bt[num_decodes + k][(st // block).long()].long() * block + st % block
        ref_s = torch.empty(((n_states + 127) // 128), 128, 584, device=dev, dtype=torch.uint8)
        ref_remap = m.gather_requant_to_ds_mla(cache, phys.to(torch.int32).view(1, -1), ref_s)  # -> ref rows 0..n-1
        got_v, got_s = records(pool, base + st, 576, 8)
        ref_v, ref_sc = records(ref_s, st, 576, 8)
        good_pool &= torch.equal(got_v, ref_v) and torch.equal(got_s, ref_sc)
    # indices: local + base, masked
    local = topk_buf[num_decode_tokens:]
    row_states = torch.tensor([seq_lens[0]] * q_lens[0] + [seq_lens[1]] * q_lens[1], device=dev).view(-1, 1)
    row_base = torch.tensor([bases[0]] * q_lens[0] + [bases[1]] * q_lens[1], device=dev, dtype=torch.int32).view(-1, 1)
    exp = torch.where((local >= 0) & (local < row_states), local + row_base, torch.full_like(local, -1))
    good_idx = torch.equal(sidx, exp)
    # consumer reuses the pool: poison it, call as a consumer, the pool must stay poisoned (no dequant)
    marker = pool[0, 0, :8].clone()
    pool[0, 0, :8] = 0xEE
    segs_c = list(m.packed_prefill_segments(layer_con, cache, attn_md, swa_md, 0, 2, prefill_token_base=3))
    torch.cuda.synchronize()
    reused = bool((pool[0, 0, :8] == 0xEE).all()) and len(segs_c) == 1
    pool[0, 0, :8] = marker
    print(f"[{'ok ' if one_segment and good_pool and good_idx and reused else 'BAD'}] prefill (pool fits): one segment {one_segment}, pool == per-state requant {good_pool}, indices = local + base masked {good_idx}, consumer reuse {reused}")
    ok &= one_segment and good_pool and good_idx and reused

    # ---------------- prefill fallback: pool too small for both requests together (4096 states pool, 1000 + 700 fits -> make them larger)
    seq_lens2 = [3000, 2500]
    swa_md2 = SimpleNamespace(**{**vars(swa_md), "prefill_seq_lens_cpu": torch.tensor(seq_lens2)})
    max_blocks2 = (max(seq_lens2) + block - 1) // block
    bt2 = torch.randint(0, pages, (3, max_blocks2), device=dev, dtype=torch.int32)  # pages may repeat: only addressing is tested
    attn_md2 = SimpleNamespace(block_table=bt2)
    segs2 = list(m.packed_prefill_segments(layer_con, cache, attn_md2, swa_md2, 0, 2, prefill_token_base=3))
    torch.cuda.synchronize()
    two = len(segs2) == 2 and segs2[0][:2] == (0, q_lens[0]) and segs2[1][:2] == (q_lens[0], sum(q_lens))
    # the pool after the generator ends holds request 1 (the last dequantized) at base 0
    st = torch.arange(seq_lens2[1], device=dev)
    phys = bt2[num_decodes + 1][(st // block).long()].long() * block + st % block
    ref_s = torch.empty((seq_lens2[1] + 127) // 128, 128, 584, device=dev, dtype=torch.uint8)
    m.gather_requant_to_ds_mla(cache, phys.to(torch.int32).view(1, -1), ref_s)
    got_v, got_s = records(segs2[1][2].squeeze(2), st, 576, 8)
    ref_v, ref_sc = records(ref_s, st, 576, 8)
    last_ok = torch.equal(got_v, ref_v) and torch.equal(got_s, ref_sc)
    idx_ok = bool((segs2[1][3] < seq_lens2[1]).all()) and torch.equal(segs2[1][3] >= 0, topk_buf[num_decode_tokens + q_lens[0]:] >= 0)
    print(f"[{'ok ' if two and last_ok and idx_ok else 'BAD'}] prefill (fallback): two segments {two}, last request's context at base 0 {last_ok}, indices local/masked {idx_ok}")
    ok &= two and last_ok and idx_ok

    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
