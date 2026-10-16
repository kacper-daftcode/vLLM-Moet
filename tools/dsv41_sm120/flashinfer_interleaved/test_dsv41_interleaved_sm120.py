#!/usr/bin/env python3
"""FlashInfer SM120 sparse MLA, DeepSeek-V4.1 rows: the interleaved placement (each row's scales right after its
data, FlashMLA 2026.09.30 / vLLM after vllm#61371) must read exactly what the footer placement reads.

Caches come from FlashInfer's own footer writers (528-byte FP8 main, 288-byte V41_FP4 compressed), are re-laid out
byte for byte into the interleaved order (optionally inside padded pages), and every route is run on both:
  - trtllm_batch_decode_sparse_mla_dsv4 (the call vLLM makes; its policy picks decode / prefill kernels by tokens),
  - SparseMLASm120Wrapper with compute_precision fp8 and bf16 (MixedCache / ordinary and the full-BF16 kernels).
Outputs must be bit-identical; reading the interleaved bytes as a footer must not be (the flag has to matter).
--footer-only --dump PATH runs the footer calls alone without the new argument (works on an unpatched FlashInfer) and
saves them, so two FlashInfer builds can be compared on the footer path with --compare A B.
usage: test_dsv41_interleaved_sm120.py [--quick] [--footer-only --dump PATH] | --compare A B"""
import argparse
import itertools
import sys

import torch

FP8 = (512, 16)  # data bytes, scale bytes per row
FP4 = (256, 32)


def packed(latent, fp4, page_stride):
    """(footer cache, interleaved cache), both [P, ps, 1, row] NHD views over pages of page_stride bytes."""
    from flashinfer.mla import dsv41_fp4_quantize_pack_sparse_mla_cache, dsv41_fp8_quantize_pack_sparse_mla_cache

    pack = dsv41_fp4_quantize_pack_sparse_mla_cache if fp4 else dsv41_fp8_quantize_pack_sparse_mla_cache
    footer = pack(latent, kv_layout="NHD").contiguous()
    pages, ps = latent.shape[:2]
    data, scale = FP4 if fp4 else FP8
    row = data + scale
    stride = page_stride or ps * row
    flat = footer.view(pages, ps * row)
    rows = torch.cat([flat[:, : ps * data].view(pages, ps, data), flat[:, ps * data :].view(pages, ps, scale)], 2)

    def place(page_bytes):
        buf = torch.full((pages, stride), 0xA5, dtype=torch.uint8, device=latent.device)
        buf[:, : ps * row] = page_bytes
        return buf.as_strided((pages, ps, 1, row), (stride, row, row, 1))

    return place(flat), place(rows.reshape(pages, ps * row))


def indices(tokens, width, pool, gen):
    idx = torch.randint(0, pool, (tokens, width), generator=gen, dtype=torch.int32)
    lens = torch.randint(1, width + 1, (tokens,), generator=gen, dtype=torch.int32)
    lens[0] = width
    idx[torch.arange(width)[None, :] >= lens[:, None]] = -1
    return idx.cuda(), lens.cuda()


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    bad = [k for k in a if k not in b or not torch.equal(a[k], b[k])]
    for k in a:
        print(f"{k:72s} {'BIT-EXACT' if k not in bad else 'DIFF'}")
    print(f"\nfooter path, {len(a)} outputs: {len(bad)} differ between the two builds")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--footer-only", action="store_true")
    ap.add_argument("--dump")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    import flashinfer
    from flashinfer.mla import SparseMLASm120Wrapper, trtllm_batch_decode_sparse_mla_dsv4

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    dev = torch.device("cuda")
    heads, swa_topk, extra_topk, main_ps = 16, 128, 512, 64
    scale = 512 ** -0.5
    ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    sink = torch.randn(heads, device=dev) * 0.5
    fails, runs, dump, wrapper_routes = [], 0, {}, set()
    placement_kw = (lambda p: {}) if args.footer_only else (lambda p: {"kv_scale_placement": p})

    def check(name, a, b, misread=None):
        nonlocal runs
        runs += 1
        same = torch.equal(a, b)
        moved = None if misread is None else not torch.equal(a, misread)
        status = "BIT-EXACT" if same else f"DIFF max {(a.float() - b.float()).abs().max().item():.3e}"
        if moved is False:
            status += "  (!) the misread layout gives the same output"
        print(f"{name:72s} {status}", flush=True)
        if not same or moved is False:
            fails.append(name)

    token_counts = (1, 4) if args.quick else (1, 4, 33, 256)
    extra_forms = ((False, None), (True, 32)) if args.quick else ((False, None), (True, 32), (True, 64))
    for (fp4_extra, extra_ps), tokens, padded in itertools.product(extra_forms, token_counts, (False, True)):
        main_pages, extra_pages = 24, 40
        main_lat = torch.randn(main_pages, main_ps, 512, device=dev, dtype=torch.bfloat16)
        main_f, main_i = packed(main_lat, False, main_ps * 528 + 512 if padded else None)
        q = torch.randn(tokens, heads, 512, device=dev, dtype=torch.bfloat16)
        swa_idx, swa_len = indices(tokens, swa_topk, main_pages * main_ps, gen)
        extra = {}
        fmt = "fp8_dsv41"
        if fp4_extra:
            fmt = "fp8_dsv41_fp4_ca"
            extra_lat = torch.randn(extra_pages, extra_ps, 512, device=dev, dtype=torch.bfloat16)
            extra_f, extra_i = packed(extra_lat, True, extra_ps * 288 + 256 if padded else None)
            ex_idx, ex_len = indices(tokens, extra_topk, extra_pages * extra_ps, gen)
            extra = dict(f=extra_f, i=extra_i, idx=ex_idx, len=ex_len)

        def call(main, comp, placement):
            out = torch.empty(tokens, heads, 512, device=dev, dtype=torch.bfloat16)
            trtllm_batch_decode_sparse_mla_dsv4(
                query=q, swa_kv_cache=main, workspace_buffer=ws, sparse_indices=swa_idx,
                compressed_kv_cache=comp, out=out, bmm1_scale=scale, sinks=sink, kv_layout="NHD",
                swa_topk_lens=swa_len,
                extra_sparse_indices=extra.get("idx"), extra_sparse_topk_lens=extra.get("len"),
                kv_cache_format=fmt, **placement_kw(placement),
            )
            torch.cuda.synchronize()
            return out

        name = f"vllm-call {fmt:17s} extra_ps={extra_ps} T={tokens:3d} padded={padded}"
        ref = call(main_f, extra.get("f"), "footer")
        dump[name] = ref.cpu()
        if not args.footer_only:
            check(name, ref, call(main_i, extra.get("i"), "interleaved"), call(main_i, extra.get("i"), "footer"))

        for precision in ("fp8", "bf16"):
            outs = []
            forms = (("footer", main_f, extra.get("f")),)
            if not args.footer_only:
                forms += (("interleaved", main_i, extra.get("i")),)
            for placement, main, comp in forms:
                try:
                    w = SparseMLASm120Wrapper(kv_scale_format="ue8m0_g32", extra_kv_fp4=fp4_extra,
                                              compute_precision=precision, **placement_kw(placement))
                    out = torch.empty(tokens, heads, 512, device=dev, dtype=torch.bfloat16)
                    w.run(q, main.squeeze(2), swa_idx, out, scale, topk_length=swa_len, attn_sink=sink,
                          extra_kv_cache=None if comp is None else comp.squeeze(2),
                          extra_indices=extra.get("idx"), extra_topk_length=extra.get("len"))
                    torch.cuda.synchronize()
                    outs.append(out)
                    for call in w._prepared_calls.values():
                        info = call.plan.inspect()
                        wrapper_routes.add(f"{precision}:{info['implementation']}/{info['numeric_route']}@T{tokens}")
                except (ValueError, RuntimeError) as e:
                    outs.append(str(e).splitlines()[0][:90])
            name = f"wrapper {precision:4s} {fmt:17s} extra_ps={extra_ps} T={tokens:3d} padded={padded}"
            if not isinstance(outs[0], str):
                dump[name] = outs[0].cpu()
            if args.footer_only:
                continue
            if isinstance(outs[0], str) or isinstance(outs[1], str):
                same = outs[0] == outs[1]
                print(f"{name:72s} {'both refused: ' + outs[0] if same else 'MISMATCH ' + str(outs)}", flush=True)
                if not same:
                    fails.append(name)
                continue
            check(name, outs[0], outs[1])

    from flashinfer.mla._sparse_mla_sm120 import _prepared

    routes = {}
    for key, call in _prepared._functional_plans.items():
        info = call.plan.inspect()
        routes.setdefault(f"{info['implementation']}/{info['numeric_route']}", set()).add(key[-1][0][0][0])
    print("functional routes (implementation/numeric: query token counts):",
          {k: sorted(v) for k, v in sorted(routes.items())})
    print("wrapper routes:", sorted(wrapper_routes))
    if args.dump:
        torch.save(dump, args.dump)
        print(f"saved {len(dump)} footer outputs to {args.dump}")
    if not args.footer_only:
        print(f"\n{runs} comparisons, {len(fails)} failed" + (": " + "; ".join(fails) if fails else ""))
    print(f"flashinfer {flashinfer.__version__} from {flashinfer.__file__}, {torch.cuda.get_device_name()}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
