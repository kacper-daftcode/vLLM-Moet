#!/usr/bin/env python3
"""Analytical KV-cache budget model for Kimi-K3 (+ DSpark) on vLLM v0.32.0's hybrid KV cache manager.

Reproduces the engine's accounting: hybrid block-size inflation (platforms/interface.py), layer grouping with
padding (kv_cache_utils._get_kv_cache_groups_uniform_page_size), bytes per block, and the per-request block
need (FullAttentionSpec / MambaSpec.max_memory_usage_bytes) behind "GPU KV cache size: N tokens".
"""
import argparse
from math import ceil

GiB = 1024 ** 3
TP = 8
DCP = 8
# K3 text config
MLA_LAYERS = 24
KDA_LAYERS = 69
KV_LORA, ROPE = 512, 64
KDA_HEADS, KDA_HEAD_DIM, CONV_K = 96, 128, 4
DRAFT_MLA_LAYERS = 5  # Inferact/Kimi-K3-DSpark: 5 MLA layers, same kv_lora_rank/rope as K3
MLA_ALIGN = 128  # MLA hybrid: manager block a multiple of 128 (platforms/interface.py)
HIDDEN = 7168


def cdiv(a, b):
    return -(-a // b)


def kda_page(num_spec, ssm_fp32, recoverssm):
    conv_dim = 3 * KDA_HEADS * KDA_HEAD_DIM // TP
    conv = conv_dim * (CONV_K - 1 + num_spec) * 2  # bf16 conv state
    state = (KDA_HEADS // TP) * KDA_HEAD_DIM * KDA_HEAD_DIM * (4 if ssm_fp32 else 2)
    page = conv + state
    if recoverssm:
        h = KDA_HEADS // TP
        q = 1 + num_spec
        page += h * q * KDA_HEAD_DIM * 4  # fp32 correction
        page += h * q * 2 * KDA_HEAD_DIM * 2  # bf16 key/gate
    return page


def model(kv_bytes, k, ssm_fp32=True, recoverssm=False, draft=True, max_model_len=1 << 20,
          req_len=None, max_num_seqs=128, group_override=None):
    mla_tok = KV_LORA + ROPE  # fp8: 1 byte
    mpage = kda_page(k, ssm_fp32, recoverssm)
    block = MLA_ALIGN * cdiv(mpage, MLA_ALIGN * mla_tok)
    page = block * mla_tok
    n_mla = MLA_LAYERS + (DRAFT_MLA_LAYERS if (draft and k > 0) else 0)
    n_kda = KDA_LAYERS
    g = min(n_mla, n_kda)
    if max(n_mla, n_kda) < 1.5 * g:
        g = max(n_mla, n_kda)
    if group_override:
        g = group_override
    pad_mla = (-n_mla) % g
    pad_kda = (-n_kda) % g
    groups_mla = cdiv(n_mla, g)
    groups_kda = cdiv(n_kda, g)
    bytes_per_block = g * page
    num_blocks = kv_bytes // bytes_per_block
    spec_blocks = 0 if (recoverssm or k == 0) else k
    kda_per_req_max = 2 + spec_blocks + 1  # align: 2 + num_speculative_blocks + prefill checkpoint
    kda_per_req_decode = 2 + spec_blocks
    mla_per_req = cdiv(cdiv(max_model_len, DCP), block) * groups_mla
    per_req = mla_per_req + groups_kda * kda_per_req_max
    conc = num_blocks / per_req
    out = dict(block=block, logical_block=block * DCP, page=page, mamba_page=mpage,
               mamba_pad_pct=100 * (page - mpage) / mpage, group_size=g, pad_mla=pad_mla, pad_kda=pad_kda,
               groups=(groups_mla, groups_kda), bytes_per_block=bytes_per_block, num_blocks=num_blocks,
               blocks_per_1M_req=per_req, kda_blocks_per_req_max=groups_kda * kda_per_req_max,
               kda_blocks_per_req_decode=groups_kda * kda_per_req_decode,
               max_concurrency_1M=conc, reported_tokens=conc * max_model_len)
    if req_len:
        mla_req = cdiv(cdiv(req_len, DCP), block) * groups_mla
        dec = mla_req + groups_kda * kda_per_req_decode
        out["req_len"] = req_len
        out["blocks_per_req_decode"] = dec
        out["MB_per_req_decode"] = dec * bytes_per_block / 1e6
        out["concurrent_reqs_decode"] = num_blocks // dec
        # memory wasted in padding slots of the padded KDA group (per decoding request)
        out["pad_waste_MB_per_req"] = (pad_kda / g) * kda_per_req_decode * bytes_per_block / 1e6 if pad_kda else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-gib", type=float, default=46.98, help="Available KV cache memory per GPU (GiB)")
    ap.add_argument("--req-len", type=int, default=32768)
    a = ap.parse_args()
    kv = int(a.kv_gib * GiB)
    rows = [
        ("A  no spec", dict(k=0, draft=False)),
        ("F  DSpark k=4 (prod)", dict(k=4)),
        ("T1 F + RecoverSSM", dict(k=4, recoverssm=True)),
        ("T2 F + RecoverSSM + bf16 ssm state", dict(k=4, recoverssm=True, ssm_fp32=False)),
        ("   F + bf16 ssm state only", dict(k=4, ssm_fp32=False)),
        ("   k=7 (B)", dict(k=7)),
        ("   k=7 + RecoverSSM", dict(k=7, recoverssm=True)),
    ]
    for name, kw in rows:
        m = model(kv, req_len=a.req_len, **kw)
        print(f"{name:38s} block={m['block']:5d} (x{DCP}={m['logical_block']:6d}) grp={m['group_size']:2d} "
              f"pad_kda={m['pad_kda']:2d} B/blk={m['bytes_per_block']/1e6:5.1f}MB blocks={m['num_blocks']:5d} "
              f"1M-req blocks={m['blocks_per_1M_req']:4d} -> {m['reported_tokens']/1e6:5.2f}M tok | "
              f"{a.req_len}-tok req: {m['blocks_per_req_decode']:3d} blk = {m['MB_per_req_decode']:6.0f} MB, "
              f"fits {m['concurrent_reqs_decode']:4d} reqs, pad waste {m['pad_waste_MB_per_req']:4.0f} MB/req")


if __name__ == "__main__":
    main()
