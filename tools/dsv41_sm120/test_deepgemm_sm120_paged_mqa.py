#!/usr/bin/env python3
"""Op-level validation of DeepGEMM SM120 FP8 paged MQA logits on 128-row pages.

DeepSeek-V4.1's indexer hands DeepGEMM 128-state pages on its compress_ratio=1
layers; stock DeepGEMM 8b1392b9 admits only 64 on sm_120 (patch_deepgemm.py).
For every case this script checks

  REF    - kernel logits vs the fp8-simulated torch reference (upstream
           tests/test_attention.py::ref_paged_mqa_logits), relative diff
           < 1e-3 exactly like upstream's FP8 gate, plus the self-consistency
           re-run (bitwise identical across launches);
  PARITY - the same logical KV re-paged 64 vs 128 rows/page, same context
           lengths, block tables remapped: logits must agree bit-for-bit
           (the per-token math is independent of the page grouping).

Shapes follow vLLM's V4.1 indexer: 32 heads, head_dim 128, fp32 weights,
2-D context_lens, next_n 1 (native decode) and 2.

Run inside the serving image on one SM120 GPU with the patched _C.so in place:
  python3 test_deepgemm_sm120_paged_mqa.py [--module vllm.third_party.deep_gemm]

ref_paged_mqa_logits / kv_cache_cast_to_fp8 / calc_diff are vendored from
DeepGEMM tests (MIT, DeepSeek).
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import sys
import time

import torch


def ref_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, max_model_len, use_2d_context_lens):
    batch_size, next_n, num_heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full([batch_size * next_n, max_model_len], float("-inf"), device=q.device, dtype=torch.float32)
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        if context_len == 0:
            continue
        q_offsets = (torch.full((next_n,), context_len, device="cuda", dtype=torch.int32) if use_2d_context_lens
                     else torch.arange(context_len - next_n, context_len, device="cuda"))
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()
        num_blocks = (context_len + block_size - 1) // block_size
        block_idxs = block_tables[i][:num_blocks]
        kv_slice = kv_cache[block_idxs]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(kv_slice.size(2), dim, -1)
        qx = q[i].transpose(0, 1)
        s = torch.matmul(qx, kx).to(logits.dtype)
        total_len = num_blocks * block_size
        k_offsets = torch.arange(0, total_len, device=q.device)
        mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
        s = torch.where(mask[None, :, :], s, float("-inf"))
        s = torch.relu(s) * weight_slice[..., None]
        s = s.sum(dim=0)
        logits[i * next_n:(i + 1) * next_n, :total_len] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


def kv_cache_cast_to_fp8(x: torch.Tensor):
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_cast_back = x_scaled.float() * sf
    x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    x_fp8[:, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(torch.uint8)
    x_fp8[:, block_size * head_dim:] = sf.view(num_blocks, block_size).view(torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4), x_cast_back.to(x.dtype)


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim.item()


def paged_views(kv_tokens: torch.Tensor, seq_lens: torch.Tensor, block_kv: int, max_model_len: int, device):
    """Lay the per-sequence token KV [B, max_model_len, D] into a shuffled block pool.

    Returns (kv_cache [num_blocks, block_kv, 1, D], block_table [B, max_blocks]).
    The same kv_tokens laid out at two block sizes hold identical per-token values.
    """
    bsz, _, dim = kv_tokens.shape
    max_blocks = max_model_len // block_kv
    num_total = bsz * max_blocks
    perm = torch.randperm(num_total, device=device, dtype=torch.int32)
    block_table = perm.view(bsz, max_blocks).contiguous()
    kv_cache = torch.empty((num_total, block_kv, 1, dim), device=device, dtype=torch.bfloat16)
    src = kv_tokens.view(bsz, max_blocks, block_kv, 1, dim)
    kv_cache[block_table.view(-1).long()] = src.reshape(num_total, block_kv, 1, dim)
    return kv_cache, block_table


def run_case(dg, bsz: int, next_n: int, avg_kv: int, device) -> dict:
    torch.manual_seed(7)
    heads, dim = 32, 128
    max_model_len = ((int(1.3 * avg_kv) + 127) // 128) * 128
    context_lens = torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv), (bsz,), device=device, dtype=torch.int32)
    context_lens[bsz // 2] = 0  # an empty request in the middle (scheduler skip path)
    q = torch.randn((bsz, next_n, heads, dim), device=device, dtype=torch.bfloat16)
    weights = torch.randn((bsz * next_n, heads), device=device, dtype=torch.float)
    kv_tokens = torch.randn((bsz, max_model_len, dim), device=device, dtype=torch.bfloat16)

    # 2-D context lens (B, next_n): earlier draft rows see a shorter context, last row the full one.
    ctx2d = ((context_lens.unsqueeze(1) + 1) * torch.rand(bsz, next_n, device=device)).int()
    ctx2d[:, -1] = context_lens
    positions = torch.arange(max_model_len, device=device).unsqueeze(0).expand(bsz * next_n, -1)
    neginf_mask = ~(positions < ctx2d.view(-1, 1))

    q_in = q.to(torch.float8_e4m3fn)
    q_sim = q_in.to(torch.bfloat16)
    res = dict(case=f"bsz={bsz} next_n={next_n} avg_kv={avg_kv}")
    out_by_pbs = {}
    for block_kv in (64, 128):
        kv_cache, block_table = paged_views(kv_tokens, context_lens, block_kv, max_model_len, device)
        kv_in, kv_sim = kv_cache_cast_to_fp8(kv_cache)
        sim_logits = ref_paged_mqa_logits(q_sim, kv_sim, weights, context_lens, block_table, max_model_len, True)
        meta = dg.get_paged_mqa_logits_metadata(ctx2d, block_kv, dg.get_num_sms())
        kw = dict(q=(q_in, None), kv_cache=kv_in, weights=weights, context_lens=ctx2d, block_table=block_table,
                  schedule_meta=meta, max_context_len=max_model_len, clean_logits=False, logits_dtype=torch.float)
        logits = dg.fp8_fp4_paged_mqa_logits(**kw)
        torch.cuda.synchronize()
        again = dg.fp8_fp4_paged_mqa_logits(**kw)
        torch.cuda.synchronize()
        lm = logits.masked_fill(neginf_mask, 0)
        res[f"self_consistent_{block_kv}"] = bool(torch.equal(lm, again.masked_fill(neginf_mask, 0)))
        res[f"ref_diff_{block_kv}"] = calc_diff(lm, sim_logits.masked_fill(neginf_mask, 0))
        out_by_pbs[block_kv] = lm
    res["parity_maxdiff"] = (out_by_pbs[64] - out_by_pbs[128]).abs().max().item()
    res["parity"] = "BIT-EXACT" if torch.equal(out_by_pbs[64], out_by_pbs[128]) else f"maxdiff={res['parity_maxdiff']:.3e}"
    res["ok"] = (res["ref_diff_64"] < 1e-3 and res["ref_diff_128"] < 1e-3 and res["self_consistent_64"]
                 and res["self_consistent_128"] and res["parity_maxdiff"] <= 1e-3)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="vllm.third_party.deep_gemm")
    args = ap.parse_args()
    dg = importlib.import_module(args.module)
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} deep_gemm={dg.__version__} from {dg.__file__}")
    cases = list(itertools.product((16, 256), (1, 2), (2048, 8192)))
    fails = 0
    t0 = time.time()
    for bsz, next_n, avg_kv in cases:
        try:
            r = run_case(dg, bsz, next_n, avg_kv, device)
        except Exception as e:  # noqa: BLE001
            r = dict(case=f"bsz={bsz} next_n={next_n} avg_kv={avg_kv}", ok=False, err=repr(e)[:200])
        fails += 0 if r["ok"] else 1
        if "err" in r:
            print(f"[BAD] {r['case']:<34} ERROR {r['err']}")
        else:
            print(f"[{'ok ' if r['ok'] else 'BAD'}] {r['case']:<34} ref64={r['ref_diff_64']:.2e} ref128={r['ref_diff_128']:.2e} "
                  f"selfc={int(r['self_consistent_64'])}{int(r['self_consistent_128'])} parity={r['parity']}")
        sys.stdout.flush()
    print(f"\n{len(cases) - fails}/{len(cases)} passed in {time.time() - t0:.0f}s")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
