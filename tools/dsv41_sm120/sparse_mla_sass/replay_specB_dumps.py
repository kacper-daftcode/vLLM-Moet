#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay spec B (FlashInfer 0.7.2 `sparse_mla_prefill_dsv41_bf16_kernel<16,true>`, `SparseMLASm120Wrapper(
compute_precision="bf16")`) on the dumps of `patch_vllm_sparse_mla_prefill_dump_sm120.py` and require the dumped output
`torch.equal` (front "cubit + prefill", Phase 1b, stage 2). The dumps hold the paged caches compacted to the touched pages and
the indices renumbered accordingly; this replay proves that transformation keeps the bits, and writes next to each dump the
spec-B LSE (`return_lse=True`) as the second reference output for the SASS kernel (`<dump>.ref.pt`: output, lse).

  python3 replay_specB_dumps.py /dumps/attn_f0_l2.pt [...]      replay + torch.equal + write <dump>.ref.pt
  python3 replay_specB_dumps.py --selftest                       synthetic pools (256K tokens): spec B on the full caches vs
                                                                 spec B on the compacted caches of the hook's function
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import torch

D_QK = D_V = 512
HEADS = 16


def load_compact_fn():
    """The hook's `_moet_attn_compact` (same source as the patched module)."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("dump_patch", os.path.join(here, "patch_vllm_sparse_mla_prefill_dump_sm120.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns = {"torch": torch}
    src = mod.HELPER  # an f-string at import: the {{ }} of the source are already single braces here
    start = src.index("def _moet_attn_compact(")
    end = src.index("def _moet_attn_dump(")
    exec(src[start:end], ns)  # noqa: S102 - our own source
    return ns["_moet_attn_compact"]


def make_wrapper(has_extra: bool):
    from flashinfer.mla import SparseMLASm120Wrapper

    return SparseMLASm120Wrapper(kv_scale_format="ue8m0_g32", kv_cache_format="fp8", extra_kv_fp4=bool(has_extra),
                                 compute_precision="bf16", backend="sparse", kv_scale_placement="interleaved")


def run_specB(w, q, swa_cache, swa_idx, swa_lens, sink, sm_scale, extra_cache=None, extra_idx=None, extra_lens=None,
              return_lse=False):
    out = torch.zeros(q.shape[0], HEADS, D_V, dtype=torch.bfloat16, device=q.device)
    lse = w.run(q, swa_cache, swa_idx, out, sm_scale, topk_length=swa_lens, attn_sink=sink, extra_kv_cache=extra_cache,
                extra_indices=extra_idx, extra_topk_length=extra_lens, return_lse=return_lse)
    torch.cuda.synchronize()
    return out, (None if lse is None else lse.clone())


def replay(path: str, dev) -> bool:
    d = torch.load(path, map_location="cpu", weights_only=False)
    has_extra = "extra_cache" in d
    q = d["q"].to(dev)
    swa_cache = d["swa_cache"].to(dev)
    swa_idx = d["swa_indices"].to(dev)
    swa_lens = d["swa_lens"].to(dev)
    sink = None if d["attn_sink"] is None else d["attn_sink"].to(dev)
    extra = (d["extra_cache"].to(dev), d["extra_indices"].to(dev), d["extra_lens"].to(dev)) if has_extra else (None, None, None)
    w = make_wrapper(has_extra)
    out, lse = run_specB(w, q, swa_cache, swa_idx, swa_lens, sink, d["sm_scale"], *extra, return_lse=True)
    out2, _ = run_specB(w, q, swa_cache, swa_idx, swa_lens, sink, d["sm_scale"], *extra)
    ref = d["output"].to(dev)
    eq = torch.equal(out, ref)
    det = torch.equal(out, out2)
    diff = (out.float() - ref.float()).abs()
    print(f"{os.path.basename(path)}: forward {d['forward']} layer {d['layer']} cr {d['compress_ratio']} rows {d['rows']} "
          f"swa pbs {d['swa_pbs']} pages {tuple(swa_cache.shape)} lens [{int(swa_lens.min())}, {int(swa_lens.max())}] "
          f"{'extra pbs %d pages %s lens [%d, %d] ' % (d['extra_pbs'], tuple(extra[0].shape), int(extra[2].min()), int(extra[2].max())) if has_extra else ''}"
          f"sm_scale {d['sm_scale']:.6g} -> replay {'IDENTICAL' if eq else 'DIFFERS'} (max|d| {diff.max().item():.3e}, "
          f"{int((diff > 0).sum())} elements), deterministic {det}")
    if eq:
        torch.save(dict(output=out.cpu(), lse=lse.cpu(), source=os.path.basename(path)), path[:-3] + ".ref.pt")
    return eq and det


def selftest(dev) -> bool:
    sys.path.insert(0, os.environ.get("NVFP4_KV_DIR", "/opt/vllm-moet/dsv41_sm120/nvfp4_kv"))
    from test_flashinfer_dsv41_fp4_extra import our_insert  # noqa: E402

    compact = load_compact_fn()

    def to_interleaved(footer, pbs, data_bytes, scale_bytes):
        pages = footer.view(-1, pbs * (data_bytes + scale_bytes))
        data = pages[:, : pbs * data_bytes].view(-1, pbs, data_bytes)
        scales = pages[:, pbs * data_bytes:].view(-1, pbs, scale_bytes)
        return torch.cat([data, scales], dim=-1).contiguous().view(-1, pbs, 1, data_bytes + scale_bytes)

    torch.manual_seed(7)
    n_pool = 1 << 18
    swa_pbs, e_pbs = 128, 64
    main_bf16 = (torch.randn(n_pool, D_QK, device=dev, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    extra_bf16 = (torch.randn(n_pool, D_QK, device=dev, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    swa = to_interleaved(our_insert(main_bf16, swa_pbs, 528).view(-1, swa_pbs, 1, 528), swa_pbs, 512, 16)
    ext = to_interleaved(our_insert(extra_bf16, e_pbs, 288).view(-1, e_pbs, 1, 288), e_pbs, 256, 32)
    del main_bf16, extra_bf16
    ok = True
    for rows, topk, with_extra in ((1536, 128, True), (1024, 192, True), (1100, 128, False)):
        q = (torch.randn(rows, HEADS, D_QK, device=dev, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
        # indices concentrated on a sparse subset of pages (so the compaction really renumbers), with -1 holes inside
        # the valid length (the kernel skips them: valid = index >= 0) and partial lengths
        swa_sub = torch.randperm(n_pool // swa_pbs, device=dev)[:37].to(torch.int32)
        idx = swa_sub[torch.randint(0, 37, (rows, topk), device=dev)] * swa_pbs + torch.randint(0, swa_pbs, (rows, topk), device=dev, dtype=torch.int32)
        idx[torch.rand(rows, topk, device=dev) < 0.02] = -1
        lens = torch.randint(1, topk + 1, (rows,), device=dev, dtype=torch.int32)
        lens[::3] = topk
        e_sub = torch.randperm(n_pool // e_pbs, device=dev)[:301].to(torch.int32)
        eidx = e_sub[torch.randint(0, 301, (rows, 512), device=dev)] * e_pbs + torch.randint(0, e_pbs, (rows, 512), device=dev, dtype=torch.int32)
        eidx[torch.rand(rows, 512, device=dev) < 0.01] = -1
        elens = torch.randint(0, 513, (rows,), device=dev, dtype=torch.int32)
        elens[1::2] = 512
        sink = torch.randn(HEADS, device=dev, dtype=torch.float32)
        sm_scale = D_QK ** -0.5
        w = make_wrapper(with_extra)
        full, lse_full = run_specB(w, q, swa, idx, lens, sink, sm_scale, *((ext, eidx, elens) if with_extra else (None, None, None)),
                                   return_lse=True)
        c_swa, r_idx, _, pages = compact(swa, idx, lens)
        if with_extra:
            c_ext, r_eidx, _, epages = compact(ext, eidx, elens)
            extra_args = (c_ext, r_eidx, elens)
        else:
            extra_args = (None, None, None)
        rep, lse_rep = run_specB(w, q, c_swa, r_idx, lens, sink, sm_scale, *extra_args, return_lse=True)
        eq = torch.equal(full, rep) and torch.equal(lse_full, lse_rep)
        ok &= eq
        print(f"selftest rows {rows} topk {topk} extra {with_extra}: swa pages {int(pages.numel())}/{swa.shape[0]}"
              f"{', extra pages %d/%d' % (int(epages.numel()), ext.shape[0]) if with_extra else ''} -> "
              f"{'IDENTICAL' if eq else 'DIFFERS'} (output and lse)")
    print("PASS selftest" if ok else "FAIL selftest")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    ok = True
    if args.selftest:
        ok &= selftest(dev)
    for p in args.dumps:
        ok &= replay(p, dev)
    if args.dumps:
        print(f"{'PASS' if ok else 'FAIL'} replay: {len(args.dumps)} dumps")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
