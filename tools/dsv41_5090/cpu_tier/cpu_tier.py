"""The CPU expert tier of one TP rank, Python side: mapped mailboxes, the host dispatcher (libcpu_tier.so: the b12x-exact
kernel's pool on one CCD + the request/ack protocol) and the capturable GPU ops (cpu_tier_cuda.cu: step counter,
publish, wait + inject).

    tier = CpuTier(n_layers, n_experts, cpu_ids=range(8, 16), dispatcher_cpu=40, device=torch.device("cuda", 0))
    tier.set_record(layer, expert, host_ptr)          # the pinned expert store, record = cpu_moe/dsv41_b12x layout
    tier.start()
    # per forward (eager or captured), in order:
    tier.step_begin()
    for layer: remapped = tier.publish(layer, topk_ids, topk_weights, slot_table, a_values, a_scales, n_tokens, remapped)
               ... the GPU's MoE on `remapped` (missing pairs -1 -> zero rows) ...
               tier.wait_inject(layer, route_output)   # before TopKSum
    tier.error()  # the engine's host loop: nonzero = a step failed closed (timeout / missing record)
"""
from __future__ import annotations

import ctypes
import functools
import os

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
HID, SCALES, MAX_PAIRS, MAX_TOKENS = 5120, 160, 64, 8


@functools.lru_cache(maxsize=None)
def cuda_ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("MOET_CPU_TIER_ARCH", "12.0")
    return load(name="moet_cpu_tier_cuda", sources=[os.path.join(_HERE, "cpu_tier_cuda.cu")],
                extra_include_paths=[_HERE], extra_cuda_cflags=["-O3", "-std=c++20"], extra_cflags=["-O3", "-std=c++20"],
                verbose=False)


class _Config(ctypes.Structure):
    _fields_ = [("n_layers", ctypes.c_int), ("n_experts", ctypes.c_int), ("n_cpus", ctypes.c_int),
                ("cpu_ids", ctypes.POINTER(ctypes.c_int)), ("dispatcher_cpu", ctypes.c_int),
                ("vs1", ctypes.c_float), ("vs2", ctypes.c_float)]


@functools.lru_cache(maxsize=None)
def host_lib(path: str | None = None):
    lib = ctypes.CDLL(path or os.environ.get("CPU_TIER_LIB") or os.path.join(_HERE, "libcpu_tier.so"))
    lib.cpu_tier_create.restype = ctypes.c_void_p
    lib.cpu_tier_create.argtypes = [ctypes.POINTER(_Config)]
    lib.cpu_tier_destroy.argtypes = [ctypes.c_void_p]
    lib.cpu_tier_set_mailboxes.restype = ctypes.c_int
    lib.cpu_tier_set_mailboxes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.cpu_tier_set_record.restype = ctypes.c_int
    lib.cpu_tier_set_record.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.cpu_tier_start.restype = ctypes.c_int
    lib.cpu_tier_start.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.cpu_tier_stop.restype = ctypes.c_int
    lib.cpu_tier_stop.argtypes = [ctypes.c_void_p]
    lib.cpu_tier_totals.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
    lib.cpu_tier_kpis.restype = ctypes.c_int
    lib.cpu_tier_kpis.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
    lib.cpu_tier_request_bytes.restype = ctypes.c_int64
    lib.cpu_tier_ack_bytes.restype = ctypes.c_int64
    lib.cpu_tier_staging_bytes.restype = ctypes.c_int64
    lib.cpu_tier_now_ns.restype = ctypes.c_int64
    lib.moe_b12x_init_tables.restype = ctypes.c_int
    lib.moe_b12x_init_tables.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return lib


def init_tables(lib, tables_dir: str):
    act = np.ascontiguousarray(np.fromfile(os.path.join(tables_dir, "b12x_act_m_limit10.f32"), dtype=np.float32))
    inv = np.ascontiguousarray(np.fromfile(os.path.join(tables_dir, "b12x_ue8m0_inv.f32"), dtype=np.float32))
    assert act.shape == (65536,) and inv.shape == (256,)
    assert lib.moe_b12x_init_tables(act.ctypes.data, inv.ctypes.data) == 0


class CpuTier:
    def __init__(self, n_layers: int, n_experts: int, cpu_ids, dispatcher_cpu: int, device: torch.device,
                 tables_dir: str, timeout_ms: float = 500.0, vs1: float = 1.0, vs2: float = 1.0, lib_path: str | None = None):
        self.ext = cuda_ext()
        self.lib = host_lib(lib_path)
        init_tables(self.lib, tables_dir)
        assert self.lib.cpu_tier_request_bytes() == self.ext.request_bytes()
        assert self.lib.cpu_tier_ack_bytes() == self.ext.ack_bytes()
        self.n_layers, self.n_experts, self.device = n_layers, n_experts, device
        self.timeout_ns = int(timeout_ms * 1e6)
        cpus = list(cpu_ids)
        arr = (ctypes.c_int * len(cpus))(*cpus)
        cfg = _Config(n_layers, n_experts, len(cpus), arr, dispatcher_cpu, vs1, vs2)
        self.handle = self.lib.cpu_tier_create(ctypes.byref(cfg))
        assert self.handle, "cpu_tier_create failed"
        self._cfg_keep = (cfg, arr)
        with torch.cuda.device(device):
            rb, ab, sb = self.ext.request_bytes(), self.ext.ack_bytes(), self.ext.staging_bytes()
            self.req = [self.ext.alloc_mapped(rb) for _ in range(n_layers)]
            self.ack = [self.ext.alloc_mapped(ab) for _ in range(n_layers)]
            self.staging = [self.ext.alloc_mapped(sb) for _ in range(n_layers)]
            self.err = self.ext.alloc_mapped(64)
            self.req_d = [self.ext.device_ptr(p) for p in self.req]
            self.ack_d = [self.ext.device_ptr(p) for p in self.ack]
            self.staging_d = [self.ext.device_ptr(p) for p in self.staging]
            self.err_d = self.ext.device_ptr(self.err)
            self.step_dev = torch.zeros(1, dtype=torch.int32, device=device)
        for L in range(n_layers):
            assert self.lib.cpu_tier_set_mailboxes(self.handle, L, self.req[L], self.ack[L], self.staging[L]) == 0
        self._err_view = (ctypes.c_uint32 * 16).from_address(self.err)
        self.started = False

    # ---- store ------------------------------------------------------------------------------------------------
    def set_record(self, layer: int, expert: int, host_ptr: int) -> None:
        assert self.lib.cpu_tier_set_record(self.handle, layer, expert, host_ptr) == 0

    # ---- lifecycle --------------------------------------------------------------------------------------------
    def start(self, first_step: int = 1) -> None:
        assert self.lib.cpu_tier_start(self.handle, first_step) == 0
        self.started = True

    def stop(self) -> None:
        if self.started:
            self.lib.cpu_tier_stop(self.handle)
            self.started = False

    def close(self) -> None:
        self.stop()
        if self.handle:
            self.lib.cpu_tier_destroy(self.handle)
            self.handle = None
            for p in self.req + self.ack + self.staging + [self.err]:
                self.ext.free_mapped(p)

    # ---- the captured ops -------------------------------------------------------------------------------------
    def step_begin(self) -> None:
        self.ext.step_begin(self.step_dev)

    def publish(self, layer: int, topk_ids: torch.Tensor, topk_weights: torch.Tensor, slot_table: torch.Tensor,
                a_values: torch.Tensor, a_scales: torch.Tensor, n_tokens: int, remapped: torch.Tensor) -> torch.Tensor:
        self.ext.publish(self.req_d[layer], self.step_dev, topk_ids, topk_weights, slot_table, remapped, a_values,
                         a_scales, int(n_tokens))
        return remapped

    def wait_inject(self, layer: int, route_output: torch.Tensor) -> None:
        self.ext.wait_inject(self.ack_d[layer], self.req_d[layer], self.staging_d[layer], self.step_dev, route_output,
                             self.err_d, self.timeout_ns)

    # ---- host-side observation (no device syncs) -------------------------------------------------------------
    def error(self) -> int:
        return int(self._err_view[0])

    def clear_error(self) -> None:
        self._err_view[0] = 0

    def totals(self) -> dict:
        buf = (ctypes.c_int64 * 7)()
        self.lib.cpu_tier_totals(self.handle, buf)
        return dict(steps=buf[0], layers=buf[1], pairs=buf[2], busy_ns=buf[3], errors=buf[4], expected_step=buf[5],
                    layer_cursor=buf[6])

    def kpis(self, cap: int = 4096) -> np.ndarray:
        """[n, 8]: step, layer, pairs, experts, t_publish_gpu_ns, t_seen_ns, t_done_ns, status (oldest first)."""
        buf = (ctypes.c_int64 * (8 * cap))()
        n = self.lib.cpu_tier_kpis(self.handle, buf, cap)
        return np.frombuffer(buf, dtype=np.int64, count=8 * n).reshape(n, 8).copy()

    def now_ns(self) -> int:
        return int(self.lib.cpu_tier_now_ns())
