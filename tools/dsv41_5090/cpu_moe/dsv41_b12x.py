"""DeepSeek-V4.1-Flash MXFP4 experts on the CPU, b12x line: the checkpoint-layout slice record, the GPU tables,
the ctypes bindings of libcpu_moe_b12x.so (kernel, scalar reference, rows quant) and an approximate fp32 reference.

The GPU side this mirrors is FlashInfer 0.7.2's b12x compact W4A8 micro path (`--moe-backend b12x`, TP4: 384 experts,
hidden 5120, intermediate 576 per rank, top-6, SwiGLU limit 10): MXFP8 per-32 activations, zero-C block QMMAs added
with fadd.rn (FC1), accumulate-in-C QMMAs (FC2) with the alignment / truncation rules measured by qmma_probe.cu, the
router weight applied in FC2's epilogue. See cpu_moe_b12x.cpp for the numerics.

Slice record (one TP rank's slice of one expert, 4,700,160 B, the checkpoint's own bytes):
  w1n [576, 2560] | w1s [576, 160] | w3n [576, 2560] | w3s [576, 160] | w2n [5120, 288] | w2s [5120, 18]
"""
from __future__ import annotations

import ctypes
import os

import numpy as np
import torch

import dsv41_mxfp4 as D1  # checkpoint access, dequant helpers, geometry

HID, INTER_R, N13, K1, K2, N2 = D1.HID, D1.INTER_R, D1.N13, D1.K1, D1.K2, D1.N2
B1, B2 = K1 // 32, K2 // 32
RECORD_BYTES = 2 * (INTER_R * K1 // 2 + INTER_R * B1) + N2 * K2 // 2 + N2 * B2  # 4,700,160
CLAMP = 10.0

_HERE = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.environ.get(
    "DSV41_B12X_TABLES", os.path.join(_HERE, "..", "..", "..", "internal", "dsv41-5090-cpu", "parity"))
ACT_TABLE = "b12x_act_m_limit10.f32"
INV_TABLE = "b12x_ue8m0_inv.f32"


# ---------------------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------------------
def build_record(raw: dict) -> np.ndarray:
    """expert_slice_raw(...) dict -> the contiguous slice record."""
    parts = [raw["w1n"], raw["w1s"], raw["w3n"], raw["w3s"], raw["w2n"], raw["w2s"]]
    rec = np.concatenate([np.ascontiguousarray(p).reshape(-1) for p in parts])
    assert rec.nbytes == RECORD_BYTES, (rec.nbytes, RECORD_BYTES)
    return rec


def split_record(rec: np.ndarray) -> dict:
    o = 0
    out = {}
    for name, shape in (("w1n", (INTER_R, K1 // 2)), ("w1s", (INTER_R, B1)), ("w3n", (INTER_R, K1 // 2)),
                        ("w3s", (INTER_R, B1)), ("w2n", (N2, K2 // 2)), ("w2s", (N2, B2))):
        n = shape[0] * shape[1]
        out[name] = rec[o:o + n].reshape(shape)
        o += n
    return out


def random_record(gen: torch.Generator, scale_lo: int = 118, scale_hi: int = 126) -> np.ndarray:
    """Synthetic slice: uniform nibbles (vLLM's -0 canonicalization applied) and e8m0 scales in [lo, hi]."""
    def nib(shape):
        v = torch.randint(0, 256, shape, generator=gen, dtype=torch.int32)
        mag = v & 0x77
        nonzero = (mag | (mag >> 1) | (mag >> 2)) & 0x11
        return (v & (0x77 | (nonzero << 3))).to(torch.uint8).numpy()

    def sc(shape):
        return torch.randint(scale_lo, scale_hi + 1, shape, generator=gen, dtype=torch.int32).to(torch.uint8).numpy()

    raw = dict(w1n=nib((INTER_R, K1 // 2)), w1s=sc((INTER_R, B1)), w3n=nib((INTER_R, K1 // 2)), w3s=sc((INTER_R, B1)),
               w2n=nib((N2, K2 // 2)), w2s=sc((N2, B2)))
    return build_record(raw)


def record_from_checkpoint(layer: int, expert: int, rank: int, prefix: str = "layers") -> np.ndarray:
    return build_record(D1.expert_slice_raw(layer, expert, rank, prefix))


# ---------------------------------------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------------------------------------
_LIB = None
_TABLES_LOADED = False


def lib_path() -> str:
    # DSV41_B12X_LIB may point at any library with the moe_b12x_* API (libcpu_tier.so carries it too)
    return os.environ.get("DSV41_B12X_LIB") or os.path.join(_HERE, "libcpu_moe_b12x.so")


def load_tables(tables_dir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    d = tables_dir or TABLES_DIR
    act = np.fromfile(os.path.join(d, ACT_TABLE), dtype=np.float32)
    inv = np.fromfile(os.path.join(d, INV_TABLE), dtype=np.float32)
    assert act.shape == (65536,) and inv.shape == (256,), (act.shape, inv.shape)
    return act, inv


def load_lib(path: str | None = None, tables_dir: str | None = None):
    global _LIB, _TABLES_LOADED
    if _LIB is None:
        lib = ctypes.CDLL(path or lib_path())
        lib.moe_b12x_record_bytes.restype = ctypes.c_int64
        lib.moe_b12x_alloc.restype = ctypes.c_void_p
        lib.moe_b12x_alloc.argtypes = [ctypes.c_int64]
        lib.moe_b12x_free.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        lib.moe_b12x_init_tables.restype = ctypes.c_int
        lib.moe_b12x_init_tables.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.moe_b12x_pool_create.restype = ctypes.c_void_p
        lib.moe_b12x_pool_create.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.moe_b12x_pool_destroy.argtypes = [ctypes.c_void_p]
        lib.moe_b12x_experts_forward.restype = ctypes.c_int
        lib.moe_b12x_experts_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_float, ctypes.c_float,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.moe_b12x_reference_pair.restype = ctypes.c_int
        lib.moe_b12x_reference_pair.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.moe_b12x_rows_quant.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.moe_b12x_pool_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
        assert lib.moe_b12x_record_bytes() == RECORD_BYTES
        _LIB = lib
    if not _TABLES_LOADED:
        act, inv = load_tables(tables_dir)
        act = np.ascontiguousarray(act)
        inv = np.ascontiguousarray(inv)
        rc = _LIB.moe_b12x_init_tables(act.ctypes.data, inv.ctypes.data)
        assert rc == 0
        _TABLES_LOADED = True
    return _LIB


def rows_quant(x_bf16: torch.Tensor, zero_byte: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """b12x's MXFP8 rows quant of bf16 [T, 5120]: (e4m3 codes uint8 [T, 5120], scale bytes uint8 [T, 160]).
    zero_byte = the byte of an all-zero group (0 on b12x's <= 8-row specialization, 127 above)."""
    lib = load_lib()
    xb = np.ascontiguousarray(x_bf16.contiguous().view(torch.int16).numpy()).view(np.uint16)
    T = xb.shape[0]
    q = np.zeros((T, K1), dtype=np.uint8)
    s = np.zeros((T, B1), dtype=np.uint8)
    lib.moe_b12x_rows_quant(xb.ctypes.data, T, q.ctypes.data, s.ctypes.data, int(zero_byte))
    return q, s


def reference_pair(record: np.ndarray, aq: np.ndarray, as_: np.ndarray, weight: float, vs1: float = 1.0,
                   vs2: float = 1.0) -> dict:
    """The scalar reference for one (token, expert): proj bf16 bits [1152], q [576], sbytes [18], out bf16 bits [5120]."""
    lib = load_lib()
    proj = np.zeros(N13, dtype=np.uint16)
    q = np.zeros(K2, dtype=np.uint8)
    sb = np.zeros(B2, dtype=np.uint8)
    out = np.zeros(N2, dtype=np.uint16)
    rec = np.ascontiguousarray(record)
    aq = np.ascontiguousarray(aq)
    as_ = np.ascontiguousarray(as_)
    rc = lib.moe_b12x_reference_pair(rec.ctypes.data, aq.ctypes.data, as_.ctypes.data, float(weight), float(vs1),
                                     float(vs2), proj.ctypes.data, q.ctypes.data, sb.ctypes.data, out.ctypes.data)
    assert rc == 0
    return dict(proj=proj, q=q, sbytes=sb, out=out)


class Pool:
    """A worker pool pinned to the given CPUs (one CCD for one rank)."""

    def __init__(self, cpu_ids: list[int]):
        self.lib = load_lib()
        arr = (ctypes.c_int * len(cpu_ids))(*cpu_ids)
        self.handle = self.lib.moe_b12x_pool_create(len(cpu_ids), arr)
        assert self.handle, "pool creation failed (tables loaded?)"
        self.cpu_ids = list(cpu_ids)

    def close(self):
        if self.handle:
            self.lib.moe_b12x_pool_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def stats(self, reset: bool = True) -> dict:
        """FC2 blocks / blocks through the general (truncating) path; worker 0's TSC cycles per phase."""
        buf = (ctypes.c_int64 * 7)()
        self.lib.moe_b12x_pool_stats(self.handle, buf, 1 if reset else 0)
        return dict(fc2_blocks=buf[0], fc2_general=buf[1], cyc_prepare=buf[2], cyc_fc1=buf[3], cyc_act=buf[4], cyc_fc2=buf[5], fc2_q0=buf[6])

    def experts_forward(self, record_ptrs: list[int], m_per_expert: list[int], token_idx: list[int],
                        pair_weight: list[float], out_row: list[int], aq: np.ndarray, as_: np.ndarray,
                        out: np.ndarray, vs1: float = 1.0, vs2: float = 1.0, debug: bool = False) -> dict | None:
        """aq [T, 5120] uint8 e4m3, as_ [T, 160] uint8; out [rows, 5120] uint16 bf16 (rows out_row[p] written).
        debug=True also returns the projections / intermediate codes / scale bytes per pair."""
        n = len(record_ptrs)
        P = sum(m_per_expert)
        assert len(m_per_expert) == n and P == len(token_idx) == len(pair_weight) == len(out_row)
        assert aq.dtype == np.uint8 and as_.dtype == np.uint8 and out.dtype == np.uint16
        assert aq.flags.c_contiguous and as_.flags.c_contiguous and out.flags.c_contiguous
        assert aq.shape == (aq.shape[0], K1) and as_.shape == (aq.shape[0], B1) and out.shape[1] == N2
        recs = (ctypes.c_void_p * n)(*record_ptrs)
        ms = (ctypes.c_int * max(1, n))(*m_per_expert)
        ti = (ctypes.c_int * max(1, P))(*token_idx)
        pw = (ctypes.c_float * max(1, P))(*pair_weight)
        orow = (ctypes.c_int * max(1, P))(*out_row)
        dbg = None
        pp = qq = ss = None
        if debug:
            dbg = dict(proj=np.zeros((P, N13), dtype=np.uint16), q=np.zeros((P, K2), dtype=np.uint8),
                       sbytes=np.zeros((P, B2), dtype=np.uint8))
            pp, qq, ss = dbg["proj"].ctypes.data, dbg["q"].ctypes.data, dbg["sbytes"].ctypes.data
        rc = self.lib.moe_b12x_experts_forward(self.handle, n, recs, ms, ti, pw, orow, aq.shape[0], aq.ctypes.data,
                                               as_.ctypes.data, out.ctypes.data, out.shape[1], float(vs1), float(vs2),
                                               pp, qq, ss)
        if rc != 0:
            raise RuntimeError(f"moe_b12x_experts_forward rc={rc}")
        return dbg


class RecordArena:
    """Records in a hugepage-advised arena (the shape of the pinned expert store)."""

    def __init__(self, n_records: int):
        self.lib = load_lib()
        self.n = n_records
        self.nbytes = n_records * RECORD_BYTES
        self.base = self.lib.moe_b12x_alloc(self.nbytes)
        assert self.base, "arena allocation failed"
        self.view = np.ctypeslib.as_array((ctypes.c_uint8 * self.nbytes).from_address(self.base))

    def slot(self, i: int) -> np.ndarray:
        return self.view[i * RECORD_BYTES:(i + 1) * RECORD_BYTES]

    def ptr(self, i: int) -> int:
        return self.base + i * RECORD_BYTES

    def close(self):
        if self.base:
            self.lib.moe_b12x_free(self.base, self.nbytes)
            self.base = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------------------
# approximate fp32 reference (layout sanity; not bit-exact: fp32 dot products, exact sigmoid)
# ---------------------------------------------------------------------------------------------------------
def dequant_e4m3_mx(q: np.ndarray, s: np.ndarray) -> torch.Tensor:
    """codes [T, K] + scale bytes [T, K/32] -> fp32 [T, K]."""
    v = torch.from_numpy(q.copy()).view(torch.float8_e4m3fn).float()
    sc = torch.ldexp(torch.ones(s.shape, dtype=torch.float32), torch.from_numpy(s.astype(np.int32)) - 127)
    return v * sc.repeat_interleave(32, dim=1)


def approx_forward(record: np.ndarray, aq: np.ndarray, as_: np.ndarray, weights: np.ndarray) -> dict:
    """fp32 math on the dequantized operands for the given tokens: proj [T, 1152], out [T, 5120] (weighted)."""
    r = split_record(record)
    w1 = D1.dequant_mxfp4(r["w1n"], r["w1s"])
    w3 = D1.dequant_mxfp4(r["w3n"], r["w3s"])
    w2 = D1.dequant_mxfp4(r["w2n"], r["w2s"])
    x = dequant_e4m3_mx(aq, as_)
    g = x @ w1.T
    u = x @ w3.T
    gb = g.to(torch.bfloat16).float()
    ub = u.to(torch.bfloat16).float()
    gc = gb.clamp(max=CLAMP)
    uc = ub.clamp(-CLAMP, CLAMP)
    y = (uc * gc * torch.sigmoid(gc)).to(torch.bfloat16).float()
    # per-32 MXFP8 requant (pow2-ceil scale) to stay close to the GPU's intermediate
    T = y.shape[0]
    yg = y.view(T, B2, 32)
    amax = yg.abs().amax(dim=2)
    sc = torch.ldexp(torch.ones_like(amax), torch.ceil(torch.log2(torch.clamp(amax / 448.0, min=1e-30))).int())
    yq = torch.clamp(yg / sc[:, :, None], -448, 448).to(torch.float8_e4m3fn).float() * sc[:, :, None]
    o = yq.view(T, K2) @ w2.T
    return dict(proj=torch.cat([gb, ub], dim=1), out=o * torch.from_numpy(np.asarray(weights, dtype=np.float32))[:, None])


def bf16_bits_to_float(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.astype(np.int32) << 16).view(torch.float32)
