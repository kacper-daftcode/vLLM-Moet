#!/usr/bin/env python3
"""Dump hook for the sparse-MLA prefill attention of a real DeepSeek-V4.1-Flash prefill on sm_120 (front "cubit + prefill",
Phase 1b: the bit oracle of the SASS kernel; same style as patch_vllm_b12x_moe_dump_sm120.py).

Every prefill chunk of the V4.1-record configuration (`nvfp4_ds_mla`: 528-byte fp8 SWA records + 288-byte FP4 compressed
records, interleaved scales) runs, since step 40, through `SparseMLASm120Wrapper(compute_precision="bf16").run(...)` in
`deepseek_v41/nvidia/flashinfer_sparse.py::_forward_prefill` = spec B (`sparse_mla_prefill_dsv41_bf16_kernel<16,true>`).
This hook records, on TP rank 0, exactly what that call consumed and produced:

  <dir>/attn_f<forward>_l<layer>.pt   q (bf16 [rows, 16, 512]), the SWA pages the chunk touches (compact copy of the paged
                                      cache, uint8 [pages, pbs, 1, 528]) + the chunk's SWA indices remapped to that compact
                                      cache (int32 [rows, topk]; entries beyond the row's length = -1) + the original indices,
                                      the lengths, the same for the compressed FP4 cache when the layer has one
                                      (compress_ratio > 0; [pages, pbs_e, 1, 288], top-k 512), sm_scale, attn_sink (fp32 [16]),
                                      the output written by spec B (bf16 [rows, 16, 512]), page sizes, compress_ratio, rows.

Compacting the paged caches to the touched pages keeps every byte the kernel reads and only renumbers pages, so the replay
(`replay_specB_dumps.py`) must reproduce the dumped output `torch.equal` before anything else is compared to it.

Env (worker): VLLM_MOET_SPARSE_MLA_DUMP_DIR (enables; must be writable from the worker), the file <dir>/ENABLE must exist
for a call to be considered (touch it after the server is up so warm-up / profiling forwards are not counted),
VLLM_MOET_SPARSE_MLA_DUMP_MIN_ROWS (1024), VLLM_MOET_SPARSE_MLA_DUMP_FORWARDS ("0,3,5,40": indices of counted forwards =
forwards whose first attention call has >= MIN_ROWS rows while enabled), VLLM_MOET_SPARSE_MLA_DUMP_LAYERS ("0,2,20": the
attention-layer ordinal within a forward, by order of first call: 0 = SWA-only, 2 = compress_ratio 2, 20 = compress_ratio 1).
Debug-only; never served. Usage: python3 patch_vllm_sparse_mla_prefill_dump_sm120.py [--dist DIR] [--check]   (idempotent;
requires the step-40 patch `patch_vllm_sparse_mla_prefill_bf16_sm120.py` to be applied first, as in the r22/r23 images).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "[vllm-moet] sparse-MLA prefill dump hook (sm_120)"
STEP40_MARKER = "[vllm-moet] sparse-MLA prefill BF16 route (sm_120)"

IMPORT_ANCHOR = "from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4\n"
HELPER = f'''

# {MARKER}: records the inputs and the output of spec B's prefill calls (rank 0) for the bit oracle of the SASS kernel.
_MOET_ATTN_DUMP = None  # None = unresolved, False = off, dict = config


def _moet_attn_dump_cfg():
    global _MOET_ATTN_DUMP
    if _MOET_ATTN_DUMP is None:
        import os

        d = os.environ.get("VLLM_MOET_SPARSE_MLA_DUMP_DIR", "")
        if not d:
            _MOET_ATTN_DUMP = False
        else:
            rank = 0
            try:
                from vllm.distributed import get_tensor_model_parallel_rank

                rank = get_tensor_model_parallel_rank()
            except Exception:  # noqa: BLE001
                pass
            ints = lambda s: {{int(x) for x in s.split(",") if x.strip()}}  # noqa: E731
            _MOET_ATTN_DUMP = False if rank != 0 else dict(
                dir=d, min_rows=int(os.environ.get("VLLM_MOET_SPARSE_MLA_DUMP_MIN_ROWS", "1024")),
                forwards=ints(os.environ.get("VLLM_MOET_SPARSE_MLA_DUMP_FORWARDS", "0,3,5,40")),
                layers=ints(os.environ.get("VLLM_MOET_SPARSE_MLA_DUMP_LAYERS", "0,2,20")),
                ordinals={{}}, forward=-1, enabled_seen=False)
    return _MOET_ATTN_DUMP


def _moet_attn_compact(cache, indices, lengths):
    """Touched pages of a paged cache [pages, pbs, 1, bytes] -> (compact copy, remapped indices [rows, topk], pbs, pages).
    indices may be [rows, topk] or [rows, 1, topk] (both accepted by the wrapper); lengths [rows] or [rows, 1]."""
    pbs = int(cache.shape[1])
    indices = indices.reshape(indices.shape[0], -1)
    lengths = lengths.reshape(-1)
    rows, topk = indices.shape
    valid = torch.arange(topk, device=indices.device)[None, :] < lengths[:, None].to(indices.device)
    ix = torch.where(valid & (indices >= 0), indices, torch.full_like(indices, -1))
    pages = torch.unique(ix[ix >= 0] // pbs)
    compact = cache[pages].contiguous()
    pos = torch.searchsorted(pages, torch.clamp(ix, min=0) // pbs)
    remapped = torch.where(ix >= 0, (pos * pbs + torch.clamp(ix, min=0) % pbs).to(torch.int32), ix)
    return compact, remapped, pbs, pages


def _moet_attn_dump(self, *args):
    try:
        _moet_attn_dump_impl(self, *args)
    except Exception as e:  # noqa: BLE001 - a debug hook must never take the server down
        import traceback

        from vllm.logger import init_logger

        init_logger(__name__).error("[vllm-moet] sparse-MLA dump hook failed (ignored): %r -- %s", e, traceback.format_exc())


def _moet_attn_dump_impl(self, q_chunk, swa_kv_paged, swa_indices_chunk, swa_lens_chunk, extra_kv_paged,
                         extra_indices_chunk, extra_lens_chunk, output_chunk):
    cfg = _moet_attn_dump_cfg()
    if not cfg:
        return
    import os

    if not os.path.exists(os.path.join(cfg["dir"], "ENABLE")):
        return
    rows = int(q_chunk.shape[0])
    if rows < cfg["min_rows"]:
        return
    key = id(self)
    if key not in cfg["ordinals"]:
        cfg["ordinals"][key] = len(cfg["ordinals"])
    layer = cfg["ordinals"][key]
    if layer == 0:
        cfg["forward"] += 1
    fwd = cfg["forward"]
    if fwd not in cfg["forwards"] or layer not in cfg["layers"]:
        return
    torch.cuda.synchronize()
    swa_compact, swa_remap, swa_pbs, swa_pages = _moet_attn_compact(swa_kv_paged, swa_indices_chunk, swa_lens_chunk)
    payload = dict(
        forward=fwd, layer=layer, rows=rows, compress_ratio=int(self.compress_ratio), sm_scale=float(self.scale),
        q=q_chunk.detach().cpu(), output=output_chunk.detach().cpu(),
        attn_sink=None if self.attn_sink is None else self.attn_sink.detach().float().cpu(),
        swa_cache=swa_compact.cpu(), swa_indices=swa_remap.cpu(), swa_indices_orig=swa_indices_chunk.detach().cpu(),
        swa_pbs=swa_pbs, swa_pages=swa_pages.cpu(),
        swa_cache_shape=tuple(swa_kv_paged.shape), swa_cache_stride=tuple(swa_kv_paged.stride()),
        q_shape=tuple(q_chunk.shape), q_stride=tuple(q_chunk.stride()), out_stride=tuple(output_chunk.stride()),
        swa_indices_stride=tuple(swa_indices_chunk.stride()), swa_indices_shape=tuple(swa_indices_chunk.shape),
        swa_lens_shape=tuple(swa_lens_chunk.shape), swa_lens=swa_lens_chunk.detach().reshape(-1).cpu(),
    )
    if extra_kv_paged is not None:
        e_compact, e_remap, e_pbs, e_pages = _moet_attn_compact(extra_kv_paged, extra_indices_chunk, extra_lens_chunk)
        payload.update(
            extra_cache=e_compact.cpu(), extra_indices=e_remap.cpu(), extra_indices_orig=extra_indices_chunk.detach().cpu(),
            extra_lens=extra_lens_chunk.detach().reshape(-1).cpu(), extra_pbs=e_pbs, extra_pages=e_pages.cpu(),
            extra_indices_shape=tuple(extra_indices_chunk.shape), extra_lens_shape=tuple(extra_lens_chunk.shape),
            extra_cache_shape=tuple(extra_kv_paged.shape), extra_cache_stride=tuple(extra_kv_paged.stride()),
            extra_indices_stride=tuple(extra_indices_chunk.stride()),
        )
    os.makedirs(cfg["dir"], exist_ok=True)
    path = os.path.join(cfg["dir"], f"attn_f{{fwd}}_l{{layer}}.pt")
    torch.save(payload, path)
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "[vllm-moet] sparse-MLA dump: forward %d layer %d (%d rows, cr %d, swa pages %d%s) -> %s", fwd, layer, rows,
        int(self.compress_ratio), int(swa_pages.numel()),
        "" if extra_kv_paged is None else f", extra pages {{int(e_pages.numel())}}", path)
'''

CALL_OLD = (
    "                _moet_prefill_bf16_wrapper(extra_kv_paged is not None, q.device).run(\n"
    "                    q_chunk,\n"
    "                    swa_kv_paged,\n"
    "                    swa_indices_chunk,\n"
    "                    output[query_start:query_end],\n"
    "                    self.scale,\n"
    "                    topk_length=swa_lens_chunk,\n"
    "                    attn_sink=self.attn_sink,\n"
    "                    extra_kv_cache=extra_kv_paged,\n"
    "                    extra_indices=extra_sparse_indices_chunk,\n"
    "                    extra_topk_length=extra_sparse_lengths_chunk,\n"
    "                )\n"
    "                continue\n"
)
CALL_NEW = CALL_OLD.replace(
    "                continue\n",
    f"                _moet_attn_dump(  # {MARKER}\n"
    "                    self, q_chunk, swa_kv_paged, swa_indices_chunk, swa_lens_chunk, extra_kv_paged,\n"
    "                    extra_sparse_indices_chunk, extra_sparse_lengths_chunk, output[query_start:query_end],\n"
    "                )\n"
    "                continue\n",
)


def _once(src: str, anchor: str, what: str, name: str) -> None:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"{name}: {what} anchor found {n} times (expected 1)")


def patch(src: str) -> str:
    if MARKER in src:
        return src
    if STEP40_MARKER not in src:
        raise SystemExit("flashinfer_sparse.py: step-40 BF16 route not present - apply patch_vllm_sparse_mla_prefill_bf16_sm120.py first")
    for what, anchor in (("import", IMPORT_ANCHOR), ("bf16 prefill call", CALL_OLD)):
        _once(src, anchor, what, "deepseek_v41/nvidia/flashinfer_sparse.py")
    src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + HELPER, 1)
    return src.replace(CALL_OLD, CALL_NEW, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=DIST)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--src", type=Path, default=None, help="patch this copy of flashinfer_sparse.py instead of --dist")
    ap.add_argument("--out", type=Path, default=None, help="write the patched copy here (with --src)")
    args = ap.parse_args()
    path = args.src or (args.dist / "models" / "deepseek_v41" / "nvidia" / "flashinfer_sparse.py")
    src = path.read_text()
    if MARKER in src:
        print(f"{path}: already patched")
        return 0
    out = patch(src)
    compile(out, str(path), "exec")
    if args.check:
        print(f"{path}: patch applies cleanly (not written)")
        return 0
    dst = args.out or path
    dst.write_text(out)
    print(f"{dst}: patched ({MARKER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
