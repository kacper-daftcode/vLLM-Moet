#!/usr/bin/env python3
"""DeepSeek-V4.1 sparse-MLA prefill attention through the cubit-assembled SASS kernel (front "cubit + prefill", Phase 1b).

Requires the step-40 patch (`patch_vllm_sparse_mla_prefill_bf16_sm120.py`: the V4.1-record prefill chunks go through
FlashInfer's full-BF16 SM120 kernel `sparse_mla_prefill_dsv41_bf16_kernel<16, extra_fp4>` = spec B). With
VLLM_MOET_SPARSE_MLA_PREFILL_SASS=1 those chunks run `kernels/sass/sparse_mla_prefill_v41.sass` instead — the same
arithmetic (bit-identical with spec B: `torch.equal` on output and lse, 8 synthetic cases + 8 dumps of a real 19K/163K
prefill, `tools/dsv41_sm120/sparse_mla_sass/test_sparse_mla_sass_sm120.py`), 0.66-0.72x of spec B's time on the
harness (GPU 4, 4096 rows). Chunks with <= 64 rows stay on the FlashInfer wrapper (it routes them to its decode kernel,
a different arithmetic; the served bits must not depend on the chunk size), as does anything the SASS path cannot
take (page size not a power of two, cubin missing, launch error) — logged once, never fatal.

Cubin: VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN (default /opt/vllm-moet/kernels/cubins-sm120/sparse_mla_prefill_v41.cubin,
built by tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh). The launcher (driver API through ctypes in
PyTorch's context and current stream, captured by CUDA graphs like any kernel) is embedded below — a copy of
tools/dsv41_sm120/sparse_mla_sass/sass_launcher.py, so the patched file has no import outside the image.

Usage: python3 patch_vllm_sparse_mla_prefill_sass_sm120.py [--dist DIR] [--check] [--src FILE --out FILE]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "[vllm-moet] sparse-MLA prefill SASS kernel (sm_120)"
STEP40_MARKER = "[vllm-moet] sparse-MLA prefill BF16 route (sm_120)"

IMPORT_ANCHOR = "from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4\n"
HELPER = f'''

# {MARKER}: VLLM_MOET_SPARSE_MLA_PREFILL_SASS=1 runs the V4.1-record prefill chunks (> 64 rows) on the cubit-assembled
# sparse_mla_prefill_v41 kernel (bit-identical with the FlashInfer BF16 prefill kernel of the step-40 route).
_MOET_SASS_STATE: dict = {{"launchers": {{}}, "disabled": False, "warned": False}}


class _MoetSparseMlaSass:
    """cuModuleLoadData -> cuModuleGetFunction -> cuFuncSetAttribute -> cuLaunchKernel with a packed .param buffer."""

    SMEM_BYTES = 87040
    BLOCK = 256

    def __init__(self, cubin: str):
        import ctypes as C
        from pathlib import Path

        import torch

        self.C = C
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        self.lib = C.CDLL("libcuda.so.1")
        sigs = {{
            "cuCtxGetCurrent": ([C.POINTER(C.c_void_p)], C.c_int),
            "cuModuleLoadData": ([C.POINTER(C.c_void_p), C.c_void_p], C.c_int),
            "cuModuleGetFunction": ([C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p], C.c_int),
            "cuFuncSetAttribute": ([C.c_void_p, C.c_int, C.c_int], C.c_int),
            "cuLaunchKernel": ([C.c_void_p, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint,
                                C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)], C.c_int),
            "cuGetErrorName": ([C.c_int, C.POINTER(C.c_char_p)], C.c_int),
            "cuGetErrorString": ([C.c_int, C.POINTER(C.c_char_p)], C.c_int),
        }}
        for name, (args, res) in sigs.items():
            f = getattr(self.lib, name)
            f.argtypes, f.restype = args, res
        ctx = C.c_void_p()
        self._ck(self.lib.cuCtxGetCurrent(C.byref(ctx)))
        if not ctx.value:
            raise RuntimeError("no current CUDA context")
        self._image = C.create_string_buffer(Path(cubin).read_bytes())
        self.module = C.c_void_p()
        self._ck(self.lib.cuModuleLoadData(C.byref(self.module), C.cast(self._image, C.c_void_p)))
        self.fn = C.c_void_p()
        self._ck(self.lib.cuModuleGetFunction(C.byref(self.fn), self.module, b"sparse_mla_prefill_v41"))
        self._ck(self.lib.cuFuncSetAttribute(self.fn, 8, self.SMEM_BYTES))  # MAX_DYNAMIC_SHARED_SIZE_BYTES

    def _ck(self, st):
        if st:
            C = self.C
            n, d = C.c_char_p(), C.c_char_p()
            self.lib.cuGetErrorName(st, C.byref(n))
            self.lib.cuGetErrorString(st, C.byref(d))
            raise RuntimeError(f"CUDA {{st}}: {{n.value!r}}: {{d.value!r}}")

    @staticmethod
    def _page_shift(pbs: int) -> int:
        s = pbs.bit_length() - 1
        if (1 << s) != pbs:
            raise ValueError(f"page size {{pbs}} is not a power of two")
        return s

    def run(self, q, kv_cache, indices, output, sm_scale, topk_length, attn_sink, extra_kv_cache, extra_indices,
            extra_topk_length):
        import ctypes as C
        import struct

        import torch

        nt, topk, pbs = q.shape[0], indices.shape[1], kv_cache.shape[1]
        has_extra = extra_kv_cache is not None
        if has_extra:
            etopk, epbs, estride = extra_indices.shape[1], extra_kv_cache.shape[1], extra_kv_cache.stride(0)
        else:
            etopk, epbs, estride = 0, 1, 0
        ptr = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
        buf = struct.pack(
            "<10Q10I2f",
            ptr(q), ptr(kv_cache), ptr(indices), ptr(topk_length), ptr(extra_kv_cache) if has_extra else 0,
            ptr(extra_indices) if has_extra else 0, ptr(extra_topk_length) if has_extra else 0, ptr(output), 0,
            ptr(attn_sink),
            nt, topk, etopk, self._page_shift(pbs), self._page_shift(epbs), kv_cache.stride(0), estride,
            indices.stride(0), extra_indices.stride(0) if has_extra else 0, 0, float(sm_scale), 1.0)
        cbuf = C.create_string_buffer(buf, len(buf))
        size = C.c_size_t(len(buf))
        extra = (C.c_void_p * 5)(C.c_void_p(1), C.cast(cbuf, C.c_void_p), C.c_void_p(2),
                                 C.cast(C.pointer(size), C.c_void_p), C.c_void_p(0))
        stream = torch.cuda.current_stream().cuda_stream
        self._ck(self.lib.cuLaunchKernel(self.fn, nt, 1, 1, self.BLOCK, 1, 1, self.SMEM_BYTES, C.c_void_p(stream),
                                         None, extra))


def _moet_prefill_sass_enabled() -> bool:
    import os

    return os.environ.get("VLLM_MOET_SPARSE_MLA_PREFILL_SASS", "0") == "1" and not _MOET_SASS_STATE["disabled"]


def _moet_prefill_sass_run(self, q_chunk, swa_kv_paged, swa_indices_chunk, swa_lens_chunk, extra_kv_paged,
                           extra_indices_chunk, extra_lens_chunk, out) -> bool:
    """Run the chunk on the SASS kernel; False = not taken (caller falls back to the FlashInfer wrapper)."""
    import os

    import torch

    rows = q_chunk.shape[0]
    if rows <= 64 or not _moet_prefill_sass_enabled():
        return False
    try:
        dev = q_chunk.device
        k = _MOET_SASS_STATE["launchers"].get(dev.index)
        if k is None:
            cubin = os.environ.get("VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN",
                                   "/opt/vllm-moet/kernels/cubins-sm120/sparse_mla_prefill_v41.cubin")
            with torch.cuda.device(dev):
                k = _MoetSparseMlaSass(cubin)
            _MOET_SASS_STATE["launchers"][dev.index] = k
            from vllm.logger import init_logger

            init_logger(__name__).info("{MARKER}: loaded %s on %s", cubin, dev)
        q = q_chunk if q_chunk.is_contiguous() else q_chunk.contiguous()
        o = out if out.is_contiguous() else torch.empty_like(q)
        idx = swa_indices_chunk.view(rows, -1)
        lens = None if swa_lens_chunk is None else swa_lens_chunk.view(-1)
        has_extra = extra_kv_paged is not None
        eidx = extra_indices_chunk.view(rows, -1) if has_extra else None
        elens = extra_lens_chunk.view(-1) if (has_extra and extra_lens_chunk is not None) else None
        sink = None if self.attn_sink is None else self.attn_sink.to(torch.float32).contiguous()
        k.run(q, swa_kv_paged, idx, o, self.scale, lens, sink, extra_kv_paged if has_extra else None, eidx, elens)
        if o is not out:
            out.copy_(o)
        return True
    except Exception as e:  # noqa: BLE001 — never take the server down; fall back to the FlashInfer wrapper
        _MOET_SASS_STATE["disabled"] = True
        if not _MOET_SASS_STATE["warned"]:
            _MOET_SASS_STATE["warned"] = True
            from vllm.logger import init_logger

            init_logger(__name__).warning("{MARKER}: disabled after an error (falling back to FlashInfer): %r", e)
        return False
'''

CALL_ANCHOR = "                _moet_prefill_bf16_wrapper(extra_kv_paged is not None, q.device).run(\n"
CALL_INSERT = (
    f"                if _moet_prefill_sass_run(  # {MARKER}\n"
    "                    self, q_chunk, swa_kv_paged, swa_indices_chunk, swa_lens_chunk, extra_kv_paged,\n"
    "                    extra_sparse_indices_chunk, extra_sparse_lengths_chunk, output[query_start:query_end],\n"
    "                ):\n"
    "                    continue\n"
)


def _once(src: str, anchor: str, what: str, name: str) -> None:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"{name}: {what} anchor found {n} times (expected 1)")


def patch(src: str) -> str:
    if MARKER in src:
        return src
    name = "deepseek_v41/nvidia/flashinfer_sparse.py"
    if STEP40_MARKER not in src:
        raise SystemExit(f"{name}: the step-40 patch (patch_vllm_sparse_mla_prefill_bf16_sm120.py) must be applied first")
    for what, anchor in (("import", IMPORT_ANCHOR), ("bf16 wrapper call", CALL_ANCHOR)):
        _once(src, anchor, what, name)
    src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + HELPER, 1)
    return src.replace(CALL_ANCHOR, CALL_INSERT + CALL_ANCHOR, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=DIST)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--src", type=Path, help="patch this file instead of the dist copy")
    ap.add_argument("--out", type=Path, help="write the patched copy here (with --src)")
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
    dest = args.out or path
    dest.write_text(out)
    print(f"{dest}: patched ({MARKER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
