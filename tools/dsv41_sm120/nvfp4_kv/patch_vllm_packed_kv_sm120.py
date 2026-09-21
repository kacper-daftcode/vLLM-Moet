#!/usr/bin/env python3
"""vLLM 0909 patch: DeepSeek-V4.1's compressed (main) KV cache in a packed record on sm_120.

Selected at start by VLLM_MOET_KV_RECORD (read once per worker process):
  fp8_ds_mla  (default) today's 584 B record, nothing changes
  nvfp4       288 B: e2m1 pairs + one e4m3 scale per 16 dims -- the checkpoint's own compressed-KV
              format (`fp4_act_quant(latent, 16, e4m3)` after RoPE); -49 % main-KV bytes
  fp8_v41     528 B: fp8 with one UE8M0 scale per 32 dims over all 512 dims (DeepSeek's V4.1
              sliding-window record `act_quant(kv, 32, "ue8m0")`); -10 %

Three files change (anchored, idempotent) and one module is installed:
  * models/deepseek_v4_1/attention.py::get_kv_cache_spec  -- state_content_bytes of the kv-source
    layers' compressed cache (the SWA cache and the indexer cache are untouched; the page alignment
    stays 576 B, 128 x 288 and 64 x 288 are multiples of it)
  * models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py::rope_quant_insert -- dispatches
    288/528-byte caches to `moet_packed_kv.rope_quant_insert_packed`
  * models/deepseek_v4_1/nvidia/flashinfer_sparse.py::DeepseekV4FlashInferSM120Attention
    ._forward_decode / _forward_prefill -- before the FlashInfer SM120 sparse-MLA call, the states a
    step attends to are gathered from the packed cache, dequantized and re-quantized into an
    fp8_ds_mla scratch (a 128-state paged cache) whose indices replace the compressed indices; the
    kernels themselves are unchanged. Decode gathers the top-k records of its rows (one gather per
    (kv source, index set): the index-source layer computes it, its consumer layers reuse it; scratch
    sized for VLLM_MOET_KV_GATHER_ROWS=64 rows, 19 MB). Prefill dequantizes the whole compressed
    context of the step's prefill requests into a pool (VLLM_MOET_KV_PREFILL_POOL_STATES, default
    max_model_len/compress_ratio states = 306 MB at 512K) once per kv source and addresses it with
    the request-local top-k indices plus a per-request base; requests that do not fit the pool
    together are dequantized one at a time by every layer.
  * models/deepseek_v4_1/common/ops/moet_packed_kv.py -- the Triton kernels
    (tools/dsv41_sm120/nvfp4_kv/{fp4_kv_quant,nvfp4_kv_kernels}.py merged)

Numerics: what the attention kernel sees is fp8_ds_mla(dequant(record)) -- for nvfp4 exactly what
experiment A0 (patch_vllm_kv_fp4_fake.py) produces with today's storage (bit-exact,
test_nvfp4_kv_kernels.py), i.e. the reference's FP4 keys plus vLLM's fp8 re-quantization.

Usage: python3 patch_vllm_packed_kv_sm120.py [--vllm-dir DIR] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "[vllm-moet] packed compressed KV (VLLM_MOET_KV_RECORD)"

# ------------------------------------------------------------------ attention.py: the spec
ATTN_OLD = (
    "        uses_fp8_ds_mla_layout = self.kv_cache_dtype == \"fp8_ds_mla\"\n"
    "        return MLAAttentionSpec(\n"
    "            block_size=vllm_config.cache_config.block_size,\n"
    "            num_kv_heads=1,\n"
    "            head_size=self.head_dim,\n"
    "            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,\n"
    "            tokens_per_state=self.compress_ratio,\n"
    "            cache_dtype_str=self.kv_cache_dtype,\n"
    "            alignment=576 if uses_fp8_ds_mla_layout else 512,\n"
    "            model_version=\"deepseek_v4\",\n"
    "            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),\n"
    "            # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token;\n"
    "            # head_size stays semantic (512).\n"
    "            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,\n"
    "        )\n"
)
ATTN_NEW = (
    "        uses_fp8_ds_mla_layout = self.kv_cache_dtype == \"fp8_ds_mla\"\n"
    f"        # {MARKER}: 584 (fp8_ds_mla) / 288 (nvfp4) / 528 (fp8_v41) bytes per compressed state;\n"
    "        # the SM120 attention path re-quantizes packed records into an fp8_ds_mla scratch.\n"
    "        from vllm.models.deepseek_v4_1.common.ops.moet_packed_kv import packed_record_bytes\n"
    "        return MLAAttentionSpec(\n"
    "            block_size=vllm_config.cache_config.block_size,\n"
    "            num_kv_heads=1,\n"
    "            head_size=self.head_dim,\n"
    "            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,\n"
    "            tokens_per_state=self.compress_ratio,\n"
    "            cache_dtype_str=self.kv_cache_dtype,\n"
    "            alignment=576 if uses_fp8_ds_mla_layout else 512,\n"
    "            model_version=\"deepseek_v4\",\n"
    "            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),\n"
    "            # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token;\n"
    "            # head_size stays semantic (512).\n"
    "            state_content_bytes=packed_record_bytes() if uses_fp8_ds_mla_layout else None,\n"
    "        )\n"
)

# ------------------------------------------------------------------ insert dispatch
INSERT_OLD = (
    "    if kv_cache.dtype == torch.uint8:\n"
    "        assert kv_cache.shape[-1] == 584\n"
    "        _rope_quant_insert_kernel[(num_tokens,)](\n"
)
INSERT_NEW = (
    "    if kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] in (288, 528):\n"
    f"        # {MARKER}\n"
    "        from vllm.models.deepseek_v4_1.common.ops.moet_packed_kv import rope_quant_insert_packed\n"
    "        rope_quant_insert_packed(latent, positions, cos_sin_cache, kv_cache, slot_mapping, compress_ratio)\n"
    "        return\n"
    "    if kv_cache.dtype == torch.uint8:\n"
    "        assert kv_cache.shape[-1] == 584\n"
    "        _rope_quant_insert_kernel[(num_tokens,)](\n"
)

# ------------------------------------------------------------------ flashinfer_sparse.py
FI_IMPORT_OLD = "from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4\n"
FI_IMPORT_NEW = (
    "from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4\n"
    f"# {MARKER}\n"
    "from vllm.models.deepseek_v4_1.common.ops.moet_packed_kv import (\n"
    "    packed_gather_for_attention as _moet_packed_gather,\n"
    "    packed_prefill_segments as _moet_packed_prefill,\n"
    "    reserve_packed_buffers as _moet_reserve_packed,\n"
    ")\n"
)
FI_RESERVE_OLD = (
    "    def _reserve_empty_forward_workspace(self) -> None:\n"
    "        self._get_workspace(\n"
    "            torch.device(\"cuda\", torch.accelerator.current_device_index())\n"
    "        )\n"
)
FI_RESERVE_NEW = (
    "    def _reserve_empty_forward_workspace(self) -> None:\n"
    "        self._get_workspace(\n"
    "            torch.device(\"cuda\", torch.accelerator.current_device_index())\n"
    "        )\n"
    f"        # {MARKER}: the profile run has no attention metadata, so reserve the packed-KV\n"
    "        # scratch and pool here for the memory accounting\n"
    "        if self.compress_ratio > 0:\n"
    "            _moet_reserve_packed(\n"
    "                torch.device(\"cuda\", torch.accelerator.current_device_index()),\n"
    "                self.max_model_len // self.compress_ratio,\n"
    "                self.topk_indices_buffer.shape[-1] if self.topk_indices_buffer is not None else 512,\n"
    "            )\n"
)
FI_DECODE_OLD = (
    "        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None\n"
    "        if extra_cache is not None and extra_sparse_indices is None:\n"
    "            raise RuntimeError(\n"
    "                \"Compressed sparse MLA decode requires compressed sparse indices.\"\n"
    "            )\n"
    "        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(\n"
)
FI_DECODE_NEW = (
    "        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None\n"
    "        if extra_cache is not None and extra_sparse_indices is None:\n"
    "            raise RuntimeError(\n"
    "                \"Compressed sparse MLA decode requires compressed sparse indices.\"\n"
    "            )\n"
    "        if extra_cache is not None and extra_cache.shape[-1] != 584:\n"
    f"            # {MARKER}: packed record -> fp8_ds_mla scratch of the attended states\n"
    "            extra_cache, extra_sparse_indices = _moet_packed_gather(\n"
    "                kv_cache, extra_sparse_indices, layer=self\n"
    "            )\n"
    "        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(\n"
)
FI_PREFILL_OLD = (
    "            if extra_kv_paged is not None and extra_sparse_indices_chunk is None:\n"
    "                raise RuntimeError(\n"
    "                    \"Compressed sparse MLA prefill requires compressed sparse indices.\"\n"
    "                )\n"
    "            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(\n"
    "                query=q_chunk,\n"
    "                swa_kv_cache=swa_kv_paged,\n"
    "                workspace_buffer=self._get_workspace(q.device),\n"
    "                sparse_indices=swa_indices_chunk,\n"
    "                compressed_kv_cache=extra_kv_paged,\n"
    "                out=output[query_start:query_end],\n"
    "                bmm1_scale=self.scale,\n"
    "                sinks=self.attn_sink,\n"
    "                kv_layout=\"NHD\",\n"
    "                swa_topk_lens=swa_lens_chunk,\n"
    "                extra_sparse_indices=extra_sparse_indices_chunk,\n"
    "                extra_sparse_topk_lens=extra_sparse_lengths_chunk,\n"
    "            )\n"
)
FI_PREFILL_NEW = (
    "            if extra_kv_paged is not None and extra_sparse_indices_chunk is None:\n"
    "                raise RuntimeError(\n"
    "                    \"Compressed sparse MLA prefill requires compressed sparse indices.\"\n"
    "                )\n"
    "            if extra_kv_paged is not None and extra_kv_paged.shape[-1] != 584:\n"
    f"                # {MARKER}: the requests' compressed contexts, dequantized into an fp8_ds_mla\n"
    "                # pool shared by the layers reading this kv source; local top-k indices + base\n"
    "                for r0, r1, scratch_kv, scratch_idx in _moet_packed_prefill(\n"
    "                    self, compressed_k_cache, attn_metadata, swa_metadata,\n"
    "                    chunk_start, chunk_end, prefill_token_base,\n"
    "                ):\n"
    "                    flashinfer_trtllm_batch_decode_sparse_mla_dsv4(\n"
    "                        query=q_chunk[r0:r1],\n"
    "                        swa_kv_cache=swa_kv_paged,\n"
    "                        workspace_buffer=self._get_workspace(q.device),\n"
    "                        sparse_indices=swa_indices_chunk[r0:r1],\n"
    "                        compressed_kv_cache=scratch_kv,\n"
    "                        out=output[query_start + r0:query_start + r1],\n"
    "                        bmm1_scale=self.scale,\n"
    "                        sinks=self.attn_sink,\n"
    "                        kv_layout=\"NHD\",\n"
    "                        swa_topk_lens=swa_lens_chunk[r0:r1],\n"
    "                        extra_sparse_indices=scratch_idx,\n"
    "                        extra_sparse_topk_lens=extra_sparse_lengths_chunk[r0:r1],\n"
    "                    )\n"
    "                continue\n"
    "            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(\n"
    "                query=q_chunk,\n"
    "                swa_kv_cache=swa_kv_paged,\n"
    "                workspace_buffer=self._get_workspace(q.device),\n"
    "                sparse_indices=swa_indices_chunk,\n"
    "                compressed_kv_cache=extra_kv_paged,\n"
    "                out=output[query_start:query_end],\n"
    "                bmm1_scale=self.scale,\n"
    "                sinks=self.attn_sink,\n"
    "                kv_layout=\"NHD\",\n"
    "                swa_topk_lens=swa_lens_chunk,\n"
    "                extra_sparse_indices=extra_sparse_indices_chunk,\n"
    "                extra_sparse_topk_lens=extra_sparse_lengths_chunk,\n"
    "            )\n"
)

GLUE = f'''

# ------------------------------------------------------------------ vLLM glue ({MARKER})
import os as _os

_RECORD_BYTES = {{"fp8_ds_mla": 584, "nvfp4": 288, "fp8_v41": 528}}
_RECORD = _os.environ.get("VLLM_MOET_KV_RECORD", "fp8_ds_mla")
if _RECORD not in _RECORD_BYTES:
    raise ValueError(f"VLLM_MOET_KV_RECORD must be one of {{sorted(_RECORD_BYTES)}}, got {{_RECORD!r}}")
# decode: rows x topk gathered records per (kv source, index set); the scratch is sized for this many rows
MOET_PACKED_KV_ROWS = int(_os.environ.get("VLLM_MOET_KV_GATHER_ROWS", "64"))
# prefill: the whole compressed context of the step's prefill requests is dequantized into one pool
# (state -> slot), reused by every layer reading the same kv source; capacity in states (default:
# the model's max length, i.e. 306 MB for 512K); requests that do not fit together fall back to a
# per-request, per-layer dequant into the same pool
_PREFILL_POOL_STATES = int(_os.environ.get("VLLM_MOET_KV_PREFILL_POOL_STATES", "0"))


def packed_record_bytes() -> int:
    return _RECORD_BYTES[_RECORD]


_scratch: dict = {{}}
_pool: dict = {{}}
# decode gathers shared by the layers reading the same (kv source, index set); written by the
# index-source layer of the group, which always runs before its consumers within a step
_decode_shared: dict = {{}}
# prefill context pools shared by the layers reading the same kv source; written by the kv-source layer
_prefill_shared: dict = {{}}


def _get_scratch(device: torch.device, pages: int) -> torch.Tensor:
    buf = _scratch.get(device)
    if buf is None or buf.shape[0] < pages:
        pages = max(pages, buf.shape[0] if buf is not None else 0)
        buf = torch.empty((pages, DS_MLA_PAGE, DS_MLA_BYTES), dtype=torch.uint8, device=device)
        _scratch[device] = buf
    return buf


def _get_pool(device: torch.device, states: int) -> torch.Tensor:
    pages = (states + DS_MLA_PAGE - 1) // DS_MLA_PAGE
    buf = _pool.get(device)
    if buf is None or buf.shape[0] < pages:
        pages = max(pages, buf.shape[0] if buf is not None else 0)
        buf = torch.empty((pages, DS_MLA_PAGE, DS_MLA_BYTES), dtype=torch.uint8, device=device)
        _pool[device] = buf
    return buf


def reserve_packed_buffers(device: torch.device, max_states: int, topk: int = 512) -> None:
    """Allocate the decode scratch and the prefill pool up front (vLLM's memory profile run executes
    the attention layers without metadata, so a lazy first-use allocation would come out of the
    run-time margin instead of the accounted budget)."""
    if packed_record_bytes() == 584:
        return
    _get_scratch(device, scratch_pages(MOET_PACKED_KV_ROWS, topk))
    _get_pool(device, _PREFILL_POOL_STATES or max_states)


def packed_gather_for_attention(kv_cache: torch.Tensor, indices: torch.Tensor, layer=None):
    """Decode: kv_cache [P, PBS, 288|528] (the layer's compressed cache), indices [rows, topk] or
    [rows, 1, topk] int32 -> (scratch [pages, 128, 1, 584] for the FlashInfer call, remapped indices
    of the same shape). With `layer`, the gather is shared: the group's index-source layer computes
    it, the consumer layers below it (same kv source, same top-k buffer) reuse the result."""
    key = None
    if layer is not None and layer.index_source_layer_id is not None:
        key = (kv_cache.device, layer.kv_source_layer_id, layer.index_source_layer_id, tuple(indices.shape))
        if layer.layer_id != layer.index_source_layer_id:
            hit = _decode_shared.get(key)
            if hit is not None:
                return hit
    idx2 = indices.reshape(indices.shape[0], -1)
    if not idx2.is_contiguous():
        idx2 = idx2.contiguous()
    rows, topk = idx2.shape
    scratch = _get_scratch(kv_cache.device, max(scratch_pages(rows, topk), scratch_pages(MOET_PACKED_KV_ROWS, topk)))
    remap = gather_requant_to_ds_mla(kv_cache, idx2, scratch)
    result = (scratch.unsqueeze(2), remap.view(indices.shape))
    if key is not None:
        _decode_shared[key] = result
    return result


def packed_prefill_segments(layer, compressed_k_cache, attn_metadata, swa_metadata, chunk_start, chunk_end,
                            prefill_token_base):
    """Prefill: yields (row0, row1, scratch_kv [pages, 128, 1, 584], scratch_indices [rows, topk] int32)
    for the query rows of prefill requests chunk_start..chunk_end-1 (rows relative to the chunk).

    The compressed context of each request is dequantized in logical order (state i -> pool slot
    base_k + i), so the request-local top-k indices address the pool with a per-request base. When
    every prefill request of the step fits the pool, the kv-source layer dequantizes them all once
    and the consumer layers reuse the pool; otherwise each request is dequantized on its own for
    every layer."""
    cr = layer.compress_ratio
    num_decodes = swa_metadata.num_decodes
    num_decode_tokens = swa_metadata.num_decode_tokens
    qsl = swa_metadata.query_start_loc_cpu
    seq_lens_cpu = swa_metadata.prefill_seq_lens_cpu
    assert seq_lens_cpu is not None, "packed KV prefill needs prefill_seq_lens_cpu"
    device = compressed_k_cache.device
    capacity = _PREFILL_POOL_STATES or (swa_metadata.prefill_max_model_len // max(cr, 1))
    pool = _get_pool(device, capacity)
    n_pref = swa_metadata.num_prefills
    states = [int(seq_lens_cpu[k]) // cr for k in range(n_pref)]
    padded = [((s + DS_MLA_PAGE - 1) // DS_MLA_PAGE) * DS_MLA_PAGE for s in states]
    fits = sum(padded) <= pool.shape[0] * DS_MLA_PAGE
    topk_buf = layer.topk_indices_buffer
    if fits:
        key = (device, layer.kv_source_layer_id)
        bases = [0]
        for p in padded[:-1]:
            bases.append(bases[-1] + p)
        if layer.is_kv_source or _prefill_shared.get(key) != (id(swa_metadata), tuple(states)):
            for k in range(n_pref):
                if states[k] > 0:
                    dequant_context_to_ds_mla(compressed_k_cache, attn_metadata.block_table[num_decodes + k].contiguous(),
                                              states[k], pool[bases[k] // DS_MLA_PAGE:])
            _prefill_shared[key] = (id(swa_metadata), tuple(states))
        r0 = int(qsl[num_decodes + chunk_start]) - prefill_token_base
        r1 = int(qsl[num_decodes + chunk_end]) - prefill_token_base
        local = topk_buf[num_decode_tokens + r0: num_decode_tokens + r1]
        # per row: the base of its request's context in the pool and the request's state count
        row_base = torch.empty(r1 - r0, dtype=torch.int32, device=device)
        row_states = torch.empty(r1 - r0, dtype=torch.int32, device=device)
        for k in range(chunk_start, chunk_end):
            a = int(qsl[num_decodes + k]) - prefill_token_base - r0
            b = int(qsl[num_decodes + k + 1]) - prefill_token_base - r0
            row_base[a:b] = bases[k]
            row_states[a:b] = states[k]
        valid = (local >= 0) & (local < row_states[:, None])
        sidx = torch.where(valid, local + row_base[:, None], torch.full_like(local, -1))
        yield 0, r1 - r0, pool.unsqueeze(2), sidx
        return
    # fallback: one request at a time, dequantized by every layer (rows relative to the chunk)
    r0 = int(qsl[num_decodes + chunk_start]) - prefill_token_base
    for k in range(chunk_start, chunk_end):
        a = int(qsl[num_decodes + k]) - prefill_token_base
        b = int(qsl[num_decodes + k + 1]) - prefill_token_base
        if states[k] > 0:
            dequant_context_to_ds_mla(compressed_k_cache, attn_metadata.block_table[num_decodes + k].contiguous(),
                                      states[k], pool)
        local = topk_buf[num_decode_tokens + a: num_decode_tokens + b]
        sidx = torch.where((local >= 0) & (local < states[k]), local, torch.full_like(local, -1))
        yield a - r0, b - r0, pool.unsqueeze(2), sidx
'''


def build_module() -> str:
    quant = (HERE / "fp4_kv_quant.py").read_text()
    kern = (HERE / "nvfp4_kv_kernels.py").read_text()
    kern = kern.replace("from fp4_kv_quant import _e2m1_code_to_f32, _fp32x2_to_fp4x2\n", "")
    kern = kern.replace("from __future__ import annotations\n", "")  # already at the top of the merged file
    header = (
        "# SPDX-License-Identifier: Apache-2.0\n"
        f"# {MARKER} -- generated by tools/dsv41_sm120/nvfp4_kv/patch_vllm_packed_kv_sm120.py from\n"
        "# fp4_kv_quant.py + nvfp4_kv_kernels.py; do not edit here.\n"
    )
    return header + quant + "\n\n# ============================== nvfp4_kv_kernels.py ==============================\n" + kern + GLUE


def apply(path: Path, old: str, new: str, check: bool) -> bool:
    src = path.read_text()
    if MARKER in src:
        print(f"{path}: already patched")
        return False
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"{path}: anchor found {n} times (expected 1):\n{old[:300]}")
    if check:
        print(f"{path}: patch applies cleanly (not written)")
        return False
    out = src.replace(old, new, 1)
    compile(out, str(path), "exec")
    path.write_text(out)
    print(f"{path}: patched")
    return True


def apply_multi(path: Path, pairs, check: bool) -> bool:
    src = path.read_text()
    if MARKER in src:
        print(f"{path}: already patched")
        return False
    out = src
    for old, new in pairs:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"{path}: anchor found {n} times (expected 1):\n{old[:300]}")
        out = out.replace(old, new, 1)
    if check:
        print(f"{path}: patch applies cleanly (not written)")
        return False
    compile(out, str(path), "exec")
    path.write_text(out)
    print(f"{path}: patched")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-dir", type=Path, default=DEFAULT_VLLM)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    v = args.vllm_dir
    ops = v / "models/deepseek_v4_1/common/ops"
    module = ops / "moet_packed_kv.py"
    text = build_module()
    compile(text, str(module), "exec")
    if not args.check:
        module.write_text(text)
        print(f"{module}: installed ({len(text.splitlines())} lines)")
    apply(v / "models/deepseek_v4_1/attention.py", ATTN_OLD, ATTN_NEW, args.check)
    apply(ops / "fused_compress_quant_cache.py", INSERT_OLD, INSERT_NEW, args.check)
    apply_multi(
        v / "models/deepseek_v4_1/nvidia/flashinfer_sparse.py",
        [(FI_IMPORT_OLD, FI_IMPORT_NEW), (FI_RESERVE_OLD, FI_RESERVE_NEW), (FI_DECODE_OLD, FI_DECODE_NEW),
         (FI_PREFILL_OLD, FI_PREFILL_NEW)],
        args.check,
    )
    print("PATCH-PACKED-KV-DONE" if not args.check else "PATCH-PACKED-KV-CHECK-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
