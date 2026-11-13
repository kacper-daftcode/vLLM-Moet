"""Inkling2's relative attention on sm_120.

vLLM computes this attention with two sm_100 kernels: `_rel_verify_kernel` (Triton Gluon: tcgen05 MMA, tensor memory,
TMA) for decode, verify and the DSpark draft, and tml_fa4's sheared-bias forward for prefill. Neither instruction set
exists on capability 12, and the fallback vLLM has for sm_103 (upstream cute FA4 with a score_mod bias) runs on
sm_120 through a forward that ignores the page table. This module is the same operation for sm_120 behind vLLM's two
entries (`_decode_attention` and `Inkling2FA4RelAttentionKernel.kernel`):

  logits[t, h, k] = scale * (q[t, h] . K[k, h // G]) + bias[t, h, pos(t) - k]      over the keys visible to token t
  out[t, h]       = softmax(logits[t, h]) @ V[:, h // G]          lse[t, h] = log(sum(exp(logits[t, h])))

K and V come from the paged cache of the token's request (`block_table`, `cache_seqlens`), a key is visible when it
is not after the token's position and, in a window layer, less than `window` positions before it. `bias` is read
from the sheared table the attention prologue writes (`rel_logits[t, h, k - base(t)]`, `base` the key of column 0
for the token's 128-token query tile); keys left of the table have no relative logit. The draft mode has no bias and
puts every query of a request at the request's last position.

`reference` is that definition in torch (the tests' oracle, and the server's path with VLLM_MOET_INKLING2_ATTN=ref);
the kernel is in inkling2_attn_sm120.cu.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
FA_BLOCK = 128  # the query tile and key block of the sheared table
ROWS = 64  # query rows (tokens x GQA group) of a CTA
PART, FUSE_GATE, FUSE_PLAIN, PREFILL = 0, 1, 2, 3


@functools.lru_cache(maxsize=None)
def _ext():
    """The compiled kernels (torch.utils.cpp_extension.load; cached under $TORCH_EXTENSIONS_DIR, built with the image)."""
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_INKLING2_ATTN_ARCH", "12.0a")
    return load(
        name="vllm_moet_inkling2_attn_sm120",
        sources=[str(_HERE / "inkling2_attn_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def kernel_facts() -> dict[str, int]:
    smem, regs, resident = _ext().kernel_facts()
    return {"shared_memory_bytes": smem, "registers": regs, "ctas_per_sm": resident}


def supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def mode() -> str:
    """kernel (default) | ref: which implementation the server's entries take."""
    return os.environ.get("VLLM_MOET_INKLING2_ATTN", "kernel")


def reference(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor | None,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    window: int,
    draft: bool = False,
    dtype: torch.dtype = torch.float32,
    chunk: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The attention of every request's query tokens: `(out [T, H, D], lse [T, H])` in `dtype` (natural-log lse; a
    token without visible keys gives zeros and -inf). `window` 0 is a full layer; `rel_logits` None is no bias."""
    num_tokens, num_heads, head_dim = q.shape
    page, num_kv_heads = key_cache.shape[1:3]
    group = num_heads // num_kv_heads
    out = torch.zeros((num_tokens, num_heads, head_dim), dtype=dtype, device=q.device)
    lse = torch.full((num_tokens, num_heads), float("-inf"), dtype=dtype, device=q.device)
    starts = cu_seqlens_q.tolist()
    seqlens = cache_seqlens.tolist()
    extent_pad = rel_logits.shape[2] if rel_logits is not None else 0
    for req, seqlen in enumerate(seqlens):
        t0, q_len = starts[req], starts[req + 1] - starts[req]
        if q_len <= 0 or seqlen <= 0:
            continue
        past = seqlen - q_len
        pages = block_table[req, : -(-seqlen // page)].long()
        keys = key_cache[pages].reshape(-1, num_kv_heads, head_dim)[:seqlen]
        values = value_cache[pages].reshape(-1, num_kv_heads, head_dim)[:seqlen]
        for c0 in range(0, q_len, chunk):
            idx = torch.arange(c0, min(c0 + chunk, q_len), device=q.device)
            n = idx.numel()
            pos = torch.full_like(idx, seqlen - 1) if draft else past + idx
            lo = max(int(pos.min()) - (window - 1), 0) if window else 0
            hi = int(pos.max()) + 1
            k = torch.arange(lo, hi, device=q.device)
            kk = keys[lo:hi].to(dtype)
            scores = torch.einsum(
                "nhgd,khd->nhgk", q[t0 + idx].to(dtype).view(n, num_kv_heads, group, head_dim), kk
            ) * softmax_scale
            dist = pos[:, None] - k[None, :]
            visible = dist >= 0
            if window:
                visible &= dist < window
            if rel_logits is not None:
                tile_end = torch.clamp((idx // FA_BLOCK + 1) * FA_BLOCK, max=q_len)
                base = (past + tile_end + FA_BLOCK - 1) // FA_BLOCK * FA_BLOCK - extent_pad
                col = k[None, :] - base[:, None]
                inside = (col >= 0) & (col < extent_pad) & visible
                bias = rel_logits[t0 + idx].to(dtype).gather(
                    2, col.clamp(0, extent_pad - 1)[:, None, :].expand(-1, num_heads, -1)
                )
                bias = torch.where(inside[:, None, :], bias, torch.zeros((), dtype=dtype, device=q.device))
                scores = scores + bias.view(n, num_kv_heads, group, -1)
            scores = scores.masked_fill(~visible[:, None, None, :], float("-inf"))
            row_lse = torch.logsumexp(scores, -1)
            probs = torch.nan_to_num(torch.softmax(scores, -1), nan=0.0)
            out[t0 + idx] = torch.einsum("nhgk,khd->nhgd", probs, values[lo:hi].to(dtype)).reshape(
                n, num_heads, head_dim
            )
            lse[t0 + idx] = row_lse.reshape(n, num_heads)
    return out, lse


def gate_sink(out: torch.Tensor, lse: torch.Tensor, gate: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
    """`out * sigmoid(gate) * sigmoid(lse - sink)` as `[T, H * D]` in out's dtype."""
    num_tokens, num_heads, head_dim = out.shape
    gated = out * torch.sigmoid(gate.to(out.dtype)).view(num_tokens, num_heads, head_dim)
    gated = gated * torch.sigmoid(lse - sinks.to(out.dtype))[:, :, None]
    return gated.reshape(num_tokens, num_heads * head_dim)


# ------------------------------------------------------------------------------------------------ vLLM's entries
def _cache_layout(key_cache: torch.Tensor, value_cache: torch.Tensor) -> list[int]:
    """[slot_stride, page, page_rows, head_rows, head_cols] of the paged cache as rows of (K | V) per KV head: a
    page's slots are rows, the KV heads sit side by side in a row or in row blocks (vLLM's `_decode_attention`)."""
    page, head_dim = key_cache.shape[1], key_cache.shape[3]
    page_stride, slot_stride, head_stride = key_cache.stride()[:3]
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
    assert value_cache.data_ptr() == key_cache.data_ptr() + head_dim * 2
    assert page_stride % slot_stride == 0 and slot_stride >= 2 * head_dim
    if head_stride < slot_stride:
        head_rows, head_cols = 0, head_stride
    else:
        assert head_stride % slot_stride == 0
        head_rows, head_cols = head_stride // slot_stride, 0
    # cp.async copies 16-byte chunks
    assert key_cache.data_ptr() % 16 == 0 and slot_stride % 8 == 0 and head_cols % 8 == 0
    return [slot_stride, page, page_stride // slot_stride, head_rows, head_cols]


def decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    rel_gate_sinks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
) -> None:
    """vLLM's `_decode_attention`: the gated bf16 `(T, H * D)` output of small batches into `out` (the draft mode,
    `rel_gate_sinks` None, has no relative logits, gate or sink); tokens from `num_actual_tokens` on are zeroed.
    Split s of a request takes the 128-key blocks s, s + num_splits, ... of its keys (of the window if `window`)."""
    num_tokens, num_heads, head_dim = q.shape
    draft = rel_gate_sinks is None
    num_reqs = cache_seqlens.shape[0]
    if mode() == "ref":
        rel_logits, gate, sinks = (None, None, None) if draft else rel_gate_sinks
        cu = cu_seqlens_q[: num_reqs + 1]
        o, lse = reference(
            q, key_cache, value_cache, rel_logits, block_table, cache_seqlens, cu, softmax_scale, window, draft
        )
        result = o.reshape(num_tokens, -1) if draft else gate_sink(o, lse, gate, sinks)
        out.copy_(result.to(out.dtype))
        out[num_actual_tokens:].zero_()
        return
    num_kv_heads = key_cache.shape[2]
    assert head_dim == 128 and max_query_len <= ROWS // (num_heads // num_kv_heads)
    assert q.stride(1) == head_dim and q.stride(2) == 1 and q.stride(0) % 2 == 0
    assert out.is_contiguous() and out.dtype == torch.bfloat16 and out.shape == (num_tokens, num_heads * head_dim)
    if draft:
        rel = gate = sinks = q  # placeholders
        rel_stride = rel_pad = 0
    else:
        rel, gate, sinks = rel_gate_sinks
        rel_stride, rel_pad = rel.stride(0), rel.shape[2]
        assert rel.stride(1) == rel_pad and rel.stride(2) == 1 and (rel_pad - 2 * FA_BLOCK) % FA_BLOCK == 0
        assert gate.stride(1) == sinks.stride(1) == 1 and gate.stride(0) % 2 == 0
    fuse = num_splits == 1
    o_part = q.new_empty((0 if fuse else num_splits, num_tokens, num_heads, head_dim), dtype=torch.float32)
    lse_part = q.new_empty((0 if fuse else num_splits, num_tokens, num_heads), dtype=torch.float32)
    slot_stride, page, page_rows, head_rows, head_cols = _cache_layout(key_cache, value_cache)
    ext = _ext()
    ext.attn_run(
        [q, key_cache, rel, block_table, cache_seqlens, cu_seqlens_q, o_part, lse_part, gate, sinks, out],
        [
            PART if not fuse else FUSE_PLAIN if draft else FUSE_GATE,
            int(draft),
            q.stride(0),
            slot_stride,
            page,
            page_rows,
            head_rows,
            head_cols,
            rel_stride,
            rel_pad,
            block_table.stride(0),
            gate.stride(0),
            sinks.stride(0),
            0,
            num_tokens,
            num_reqs,
            num_splits,
            num_heads,
            num_kv_heads,
            window,
        ],
        softmax_scale,
    )
    if fuse:
        out[num_actual_tokens:].zero_()
        return
    ext.combine_run(
        o_part,
        lse_part,
        gate,
        sinks,
        out,
        [num_tokens, num_actual_tokens, num_heads, num_splits, int(not draft), gate.stride(0), sinks.stride(0)],
    )


def prefill_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    window_size: tuple[int, int],
    sheared_bias: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    """vLLM's `Inkling2FA4RelAttentionKernel.kernel`: the fp32 `(T, H, D)` output into `out` and the natural-log
    `(H, T)` LSE into `lse`."""
    window = 0 if window_size == (-1, -1) else window_size[0] + 1
    num_tokens, num_heads, head_dim = q.shape
    if mode() == "ref":
        o, row_lse = reference(
            q, key_cache, value_cache, sheared_bias, block_table, cache_seqlens, cu_seqlens_q, softmax_scale, window
        )
        out.copy_(o)
        lse.copy_(row_lse.t())
        return
    num_kv_heads = key_cache.shape[2]
    rel_pad = sheared_bias.shape[2]
    assert head_dim == 128 and q.stride(1) == head_dim and q.stride(2) == 1 and q.stride(0) % 2 == 0
    assert sheared_bias.stride(1) == rel_pad and sheared_bias.stride(2) == 1
    assert out.is_contiguous() and out.dtype == torch.float32 and out.shape == q.shape
    assert lse.dtype == torch.float32 and lse.shape == (num_heads, num_tokens) and (num_tokens <= 1 or lse.stride(1) == 1)
    slot_stride, page, page_rows, head_rows, head_cols = _cache_layout(key_cache, value_cache)
    _ext().attn_run(
        [q, key_cache, sheared_bias, block_table, cache_seqlens, cu_seqlens_q, out, lse, q, q, q],
        [
            PREFILL,
            0,
            q.stride(0),
            slot_stride,
            page,
            page_rows,
            head_rows,
            head_cols,
            sheared_bias.stride(0),
            rel_pad,
            block_table.stride(0),
            0,
            0,
            lse.stride(0),
            num_tokens,
            cache_seqlens.shape[0],
            1,
            num_heads,
            num_kv_heads,
            window,
        ],
        softmax_scale,
    )
