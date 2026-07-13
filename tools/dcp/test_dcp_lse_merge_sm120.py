"""E2E numeric check: sharded sparse-MLA + log2 LSE merge == full attention.

Runs inside the vllm-moet image on one SM120 GPU. Emulates DCP ranks
sequentially: rank r attends only to its owned subset of the top-k indices,
then the partial outputs are merged with correct_attn_out
(is_lse_base_on_e=False) and summed - exactly what cp_lse_ag_out_rs does
(all-gather LSE + per-rank correction + reduce-scatter sum over heads).
"""
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.ops.common import correct_attn_out, CPTritonContext
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

torch.manual_seed(7)
dev = "cuda"

B = 3            # decode tokens
H = 64           # heads (post all-gather head count)
TOPK = 2048
BLOCK = 64
T = 300          # context tokens
WORLD = 4
INTERLEAVE = 1

kv_lora, rope = 512, 64

# ---- build a packed fp8_ds_mla cache via the production write op ----
num_blocks = (T + BLOCK - 1) // BLOCK + 1
kv_cache = torch.zeros(num_blocks, BLOCK, 656, dtype=torch.uint8, device=dev)
kv_c = (torch.randn(T, kv_lora, device=dev, dtype=torch.bfloat16) * 0.5)
k_pe = (torch.randn(T, rope, device=dev, dtype=torch.bfloat16) * 0.5)
slot_mapping = torch.arange(T, dtype=torch.long, device=dev)  # identity blocks
scale = torch.tensor(1.0, device=dev)
ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
                         kv_cache_dtype="fp8_ds_mla", scale=scale)

q = torch.randn(B, 1, H, kv_lora + rope, device=dev, dtype=torch.bfloat16) * 0.3
workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
sm_scale = (kv_lora + rope) ** -0.5


def run(indices):
    out = torch.empty(B, 1, H, kv_lora, dtype=torch.bfloat16, device=dev)
    res = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace,
        qk_nope_head_dim=128,
        kv_lora_rank=kv_lora,
        qk_rope_head_dim=rope,
        block_tables=indices.unsqueeze(1),
        seq_lens=None,
        max_seq_len=TOPK,
        out=out,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        sparse_mla_top_k=TOPK,
        kv_scale_format="arbitrary_fp32",
        return_lse=True,
    )
    o, lse = res
    return o.squeeze(1).float(), lse.float()


# global top-k = wszystkie T pozycji (T < TOPK) w losowej kolejnosci per token
perm = torch.stack([torch.randperm(T, device=dev) for _ in range(B)])
full_idx = torch.full((B, TOPK), -1, dtype=torch.int32, device=dev)
full_idx[:, :T] = perm.to(torch.int32)

out_full, lse_full = run(full_idx)

# ---- sharded runs ----
outs, lses = [], []
for r in range(WORLD):
    owned = ((full_idx // INTERLEAVE) % WORLD == r) & (full_idx >= 0)
    shard = torch.where(owned, full_idx, torch.full_like(full_idx, -1))
    o_r, lse_r = run(shard)
    outs.append(o_r)
    lses.append(lse_r)
    n_owned = int(owned.sum())
    print(f"rank {r}: owned={n_owned} lse[min,max]=({lse_r.min():.2f},{lse_r.max():.2f})"
          f" out_nan={int(o_r.isnan().sum())}")

# ---- merge jak cp_lse_ag_out_rs (log2!) ----
lses_stacked = torch.stack(lses)  # [N, B, H]
merged = torch.zeros_like(out_full)
for r in range(WORLD):
    o_corr, _ = correct_attn_out(
        outs[r].clone(), lses_stacked, r, CPTritonContext(), is_lse_base_on_e=False
    )
    merged += o_corr

err = (merged - out_full).abs()
rel = err.max() / out_full.abs().max()
print(f"max_abs_err={err.max():.6f} max_val={out_full.abs().max():.4f} rel={rel:.6f}")
assert not merged.isnan().any(), "NaN in merged output"
assert rel < 2e-2, f"merge mismatch: rel={rel}"

# ---- edge: krotki kontekst, rank bez tokenow ----
T2 = 2
idx2 = torch.full((B, TOPK), -1, dtype=torch.int32, device=dev)
idx2[:, 0] = 0
idx2[:, 1] = 1
out_f2, _ = run(idx2)
outs2, lses2 = [], []
for r in range(WORLD):
    owned = ((idx2 // INTERLEAVE) % WORLD == r) & (idx2 >= 0)
    shard = torch.where(owned, idx2, torch.full_like(idx2, -1))
    o_r, lse_r = run(shard)
    outs2.append(o_r)
    lses2.append(lse_r)
    print(f"[short] rank {r}: owned={int(owned.sum())} "
          f"lse[min,max]=({lse_r.min():.2f},{lse_r.max():.2f}) "
          f"out_absmax={o_r.abs().max():.4f} nan={int(o_r.isnan().sum())}")
l2 = torch.stack(lses2)
m2 = torch.zeros_like(out_f2)
for r in range(WORLD):
    o_corr, _ = correct_attn_out(outs2[r].clone(), l2, r, CPTritonContext(), is_lse_base_on_e=False)
    m2 += o_corr
err2 = (m2 - out_f2).abs().max()
print(f"[short] max_abs_err={err2:.6f}")
assert not m2.isnan().any(), "NaN in short-context merge"
assert err2 < 1e-2 * max(1.0, out_f2.abs().max().item())
print("LSE-MERGE-OK")
