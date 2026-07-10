"""Minimal ctypes CUDA-driver harness for the kernel check scripts.

Provides the `Cuda` API the checks were written against (load_kernel,
to_device, alloc, memset32, launch, from_device, synchronize) with no
dependency on torch's CUDA context. Device selection via
CUDA_VISIBLE_DEVICES; kernels launch with static shared memory only.
"""

import ctypes

import numpy as np

_CU = None


def _drv():
    global _CU
    if _CU is None:
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuInit.argtypes = [ctypes.c_uint]
        cu.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cu.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        cu.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        cu.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.c_void_p, ctypes.c_char_p]
        cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.c_size_t]
        cu.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t]
        cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t]
        cu.cuMemsetD32_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                      ctypes.c_size_t]
        cu.cuCtxSynchronize.argtypes = []
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 \
            + [ctypes.c_uint, ctypes.c_void_p,
               ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _CU = cu
    return _CU


def _ck(r, what):
    if r:
        raise RuntimeError(f"culaunch: CUDA error {r} in {what}")


class Cuda:
    def __init__(self, device: int = 0):
        cu = _drv()
        _ck(cu.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        _ck(cu.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        _ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
            "cuDevicePrimaryCtxRetain")
        _ck(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        self._cu = cu
        self._mods = []

    def load_kernel(self, cubin: str, kernel: str):
        mod = ctypes.c_void_p()
        _ck(self._cu.cuModuleLoad(ctypes.byref(mod), cubin.encode()),
            f"cuModuleLoad {cubin}")
        self._mods.append(mod)
        fn = ctypes.c_void_p()
        _ck(self._cu.cuModuleGetFunction(ctypes.byref(fn), mod,
                                         kernel.encode()),
            f"cuModuleGetFunction {kernel}")
        return fn

    def alloc(self, nbytes: int) -> ctypes.c_void_p:
        d = ctypes.c_void_p()
        _ck(self._cu.cuMemAlloc_v2(ctypes.byref(d), nbytes), "cuMemAlloc")
        return d

    def to_device(self, arr: np.ndarray) -> ctypes.c_void_p:
        arr = np.ascontiguousarray(arr)
        d = self.alloc(arr.nbytes)
        _ck(self._cu.cuMemcpyHtoD_v2(d, arr.ctypes.data_as(ctypes.c_void_p),
                                     arr.nbytes), "cuMemcpyHtoD")
        return d

    def from_device(self, d, nbytes: int, dtype=np.uint8) -> np.ndarray:
        out = np.empty(nbytes // np.dtype(dtype).itemsize, dtype=dtype)
        _ck(self._cu.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), d,
                                     nbytes), "cuMemcpyDtoH")
        return out

    def memset32(self, d, value: int, n_words: int):
        _ck(self._cu.cuMemsetD32_v2(d, value, n_words), "cuMemsetD32")

    def launch(self, fn, grid, block, args, smem: int = 0):
        argv = (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
        _ck(self._cu.cuLaunchKernel(fn, grid[0], grid[1], grid[2],
                                    block[0], block[1], block[2],
                                    smem, None, argv, None), "cuLaunchKernel")

    def synchronize(self):
        _ck(self._cu.cuCtxSynchronize(), "cuCtxSynchronize")
