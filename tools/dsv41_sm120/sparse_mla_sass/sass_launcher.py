# SPDX-License-Identifier: Apache-2.0
"""Launch the cubit-assembled `sparse_mla_prefill_v41` cubin on PyTorch tensors (front "cubit + prefill", Phase 1b).

Driver API through ctypes in PyTorch's own context and current stream (pattern: H/sass_perf/common/cuda_driver.py):
cuModuleLoadData -> cuModuleGetFunction -> cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) -> cuLaunchKernel with a
packed parameter buffer (CU_LAUNCH_PARAM_BUFFER_POINTER/SIZE), so the kernel's `.param` layout is owned here, not by
PyTorch. Captured by CUDA graphs like any kernel on the capturing stream.
"""
from __future__ import annotations

import ctypes as C
import math
import struct
from pathlib import Path

import torch

KERNEL = "sparse_mla_prefill_v41"
import os
SMEM_BYTES = int(os.environ.get("SMLA_SMEM_BYTES", "87040"))
BLOCK = 256
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
CU_LAUNCH_PARAM_END = C.c_void_p(0)
CU_LAUNCH_PARAM_BUFFER_POINTER = C.c_void_p(1)
CU_LAUNCH_PARAM_BUFFER_SIZE = C.c_void_p(2)


class DriverError(RuntimeError):
    pass


class SparseMlaSass:
    def __init__(self, cubin: str | Path, kernel: str = KERNEL, smem: int = SMEM_BYTES):
        torch.cuda.init()
        torch.zeros(1, device="cuda")  # make sure the primary context is current
        self.lib = C.CDLL("libcuda.so.1")
        sigs = {
            "cuCtxGetCurrent": ([C.POINTER(C.c_void_p)], C.c_int),
            "cuModuleLoadData": ([C.POINTER(C.c_void_p), C.c_void_p], C.c_int),
            "cuModuleGetFunction": ([C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p], C.c_int),
            "cuFuncSetAttribute": ([C.c_void_p, C.c_int, C.c_int], C.c_int),
            "cuLaunchKernel": ([C.c_void_p, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint,
                                C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)], C.c_int),
            "cuGetErrorName": ([C.c_int, C.POINTER(C.c_char_p)], C.c_int),
            "cuGetErrorString": ([C.c_int, C.POINTER(C.c_char_p)], C.c_int),
        }
        for name, (args, res) in sigs.items():
            f = getattr(self.lib, name)
            f.argtypes, f.restype = args, res
        ctx = C.c_void_p()
        self._ck(self.lib.cuCtxGetCurrent(C.byref(ctx)))
        if not ctx.value:
            raise DriverError("no current CUDA context")
        self._image = C.create_string_buffer(Path(cubin).read_bytes())
        self.module = C.c_void_p()
        self._ck(self.lib.cuModuleLoadData(C.byref(self.module), C.cast(self._image, C.c_void_p)))
        self.fn = C.c_void_p()
        self._ck(self.lib.cuModuleGetFunction(C.byref(self.fn), self.module, kernel.encode()))
        self.smem = smem
        self._ck(self.lib.cuFuncSetAttribute(self.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))

    def _ck(self, st):
        if st:
            n, d = C.c_char_p(), C.c_char_p()
            self.lib.cuGetErrorName(st, C.byref(n))
            self.lib.cuGetErrorString(st, C.byref(d))
            raise DriverError(f"CUDA {st}: {n.value!r}: {d.value!r}")

    @staticmethod
    def _page_shift(pbs: int) -> int:
        s = int(math.log2(pbs))
        if (1 << s) != pbs:
            raise ValueError(f"page size {pbs} is not a power of two (the SASS kernel decodes pages by shift)")
        return s

    def run(self, q, kv_cache, indices, output, sm_scale, *, topk_length=None, attn_sink=None, extra_kv_cache=None,
            extra_indices=None, extra_topk_length=None, out_lse=None, lse_scale=1.0):
        """Same contract as SparseMLASm120Wrapper.run for the V4.1 record pair (interleaved scales, 16 heads)."""
        assert q.dtype == torch.bfloat16 and q.is_contiguous() and q.shape[1:] == (16, 512), q.shape
        assert output.dtype == torch.bfloat16 and output.shape == q.shape and output.stride() == q.stride()
        assert kv_cache.dtype == torch.uint8 and kv_cache.dim() == 4 and kv_cache.shape[-1] == 528, kv_cache.shape
        assert indices.dtype == torch.int32 and indices.stride(1) == 1 and indices.shape[0] == q.shape[0]
        nt = q.shape[0]
        topk = indices.shape[1]
        pbs = kv_cache.shape[1]
        page_stride = kv_cache.stride(0)
        has_extra = extra_kv_cache is not None
        if has_extra:
            assert extra_kv_cache.dtype == torch.uint8 and extra_kv_cache.dim() == 4 and extra_kv_cache.shape[-1] == 288
            assert extra_indices is not None and extra_indices.dtype == torch.int32 and extra_indices.stride(1) == 1
            etopk, epbs, estride = extra_indices.shape[1], extra_kv_cache.shape[1], extra_kv_cache.stride(0)
        else:
            etopk, epbs, estride = 0, 1, 0
        ptr = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
        buf = struct.pack(
            "<10Q10I2f",
            ptr(q), ptr(kv_cache), ptr(indices), ptr(topk_length), ptr(extra_kv_cache) if has_extra else 0,
            ptr(extra_indices) if has_extra else 0, ptr(extra_topk_length) if has_extra else 0, ptr(output), ptr(out_lse),
            ptr(attn_sink),
            nt, topk, etopk, self._page_shift(pbs), self._page_shift(epbs), page_stride, estride, indices.stride(0),
            extra_indices.stride(0) if has_extra else 0, out_lse.stride(0) if out_lse is not None else 0,
            float(sm_scale), float(lse_scale))
        assert len(buf) == 128
        cbuf = C.create_string_buffer(buf, len(buf))
        size = C.c_size_t(len(buf))
        extra = (C.c_void_p * 5)(CU_LAUNCH_PARAM_BUFFER_POINTER, C.cast(cbuf, C.c_void_p), CU_LAUNCH_PARAM_BUFFER_SIZE,
                                 C.cast(C.pointer(size), C.c_void_p), CU_LAUNCH_PARAM_END)
        stream = torch.cuda.current_stream().cuda_stream
        if nt == 0:
            return
        self._ck(self.lib.cuLaunchKernel(self.fn, nt, 1, 1, BLOCK, 1, 1, self.smem, C.c_void_p(stream), None, extra))
