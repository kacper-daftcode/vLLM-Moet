#!/usr/bin/env python3
"""Step 8 on the card: `vllm._custom_ops.reshape_and_cache_flash` (what `FlashAttentionImpl.do_kv_cache_update` and
through it the DSpark draft's K / V write call) against the operation's definition - cache[slot // page, slot % page]
= (key, value) for every token of the slot mapping, every other byte of the cache untouched.

Cases: both cache layouts vLLM hands it (two contiguous (blocks, page, heads, dim) tensors; the two views of a
(blocks, heads, page, 2 * dim) cache the flash backend splits), 1 to 1024 tokens, 2 and 8 KV heads, a key / value
batch longer than the slot mapping (the padded batch of a captured graph). The cache starts as random data, so a
write outside the mapped slots shows. Run in the image: python3 test_cache_flash_sm120.py   (needs one sm_120 card)
"""
import sys

import torch

from vllm._custom_ops import reshape_and_cache_flash

dev = "cuda"; bf = torch.bfloat16
torch.manual_seed(0)
one = torch.ones(1, device=dev, dtype=torch.float32)
ok = True; n = 0
for layout in ("separate", "views"):
    for heads in (2, 8):
        for tokens, extra in ((1, 0), (8, 0), (8, 5), (64, 0), (1024, 0), (1024, 24)):
            dim, page = 128, 128
            blocks = -(-tokens // page) + 3
            if layout == "separate":
                key_cache = torch.randn(blocks, page, heads, dim, device=dev, dtype=bf)
                value_cache = torch.randn(blocks, page, heads, dim, device=dev, dtype=bf)
            else:
                kv = torch.randn(blocks, heads, page, 2 * dim, device=dev, dtype=bf)
                key_cache, value_cache = kv.transpose(1, 2).split(dim, dim=-1)
            key = torch.randn(tokens + extra, heads, dim, device=dev, dtype=bf)
            value = torch.randn(tokens + extra, heads, dim, device=dev, dtype=bf)
            slot = torch.randperm(blocks * page, device=dev)[:tokens].long()
            want_k, want_v = key_cache.clone(), value_cache.clone()
            want_k[slot // page, slot % page] = key[:tokens]; want_v[slot // page, slot % page] = value[:tokens]
            reshape_and_cache_flash(key, value, key_cache, value_cache, slot, "auto", one, one)
            torch.cuda.synchronize()
            good = torch.equal(key_cache, want_k) and torch.equal(value_cache, want_v)
            ok &= good; n += 1
            print(f"{'ok  ' if good else 'FAIL'} {layout:8s} heads {heads} tokens {tokens:4d} (+{extra} padded rows)")
print(f"{n} cases: {'ALL OK' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
