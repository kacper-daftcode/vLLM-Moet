"""DeepSeek-V4.1-Flash MXFP4 experts on the CPU — checkpoint access, the GPU-math emulation, the kernel pack.

Geometry (TP4, one rank): hidden 5120, routed intermediate 2304 -> 576 per rank; an expert slice is
  w1 (gate) [576, 5120], w3 (up) [576, 5120], w2 (down) [5120, 576]
in the checkpoint's MXFP4 record: e2m1 nibbles (low nibble = even k) + one UE8M0 scale per 32 k.

What the GPU does (vLLM `DeepGemmFP4Experts`, nightly-49cb2bbf, sm_120), which this module emulates:
  1. hidden states (bf16) -> FP8 e4m3 per token with one UE8M0 scale per 128 k
     (scale = 2^ceil(log2(max(amax/448, 1e-10))), q = clamp(x/scale, +-448) -> e4m3 RNE)
  2. FC1 = DeepGEMM fp8 x fp4 grouped GEMM, fp32 accumulate, bf16 output (gate | up)
  3. SwiGLU with clamp (swiglu_limit 10): g = min(g, 10), u = clamp(u, -10, 10), y = u * g * sigmoid(g)
     in fp32 from the bf16 FC1 output, rounded through bf16, then step 1 again on the 576 columns
     (groups of 128: 4 full + one of 64)
  4. FC2 = the same GEMM, fp32 accumulate, bf16 output; the router weights are applied by the gather.
All scales are powers of two, so the CPU result equals the GPU's up to fp32 summation order (plus the
GPU's approximate exp/log2 in the activation, which the bf16 rounding of y absorbs except at ties).

Kernel pack (one expert slice, see cpu_moe_mxfp4.cpp): row-major, each row = [scales | nibbles],
  W13: 1152 rows (gate rows then up rows) x 2752 B = 192 B scales (160 used) + 40 chunks x 64 B
  W2:  5120 rows x 320 B = 32 B scales (18 used) + 4 full chunks x 64 B + one half chunk of 32 B
A 64-B chunk holds 128 consecutive k: the low nibbles are k 0..63, the high nibbles k 64..127, with the
byte order chosen so that vpunpck{l,h}bw after a byte LUT yields the 32-element scale blocks in order.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
from functools import lru_cache

import numpy as np
import torch

MODEL_DIR = os.environ.get("DSV41_MODEL_DIR", "/mnt/nvme/models/DeepSeek-V4.1-Flash")

HID = 5120
INTER = 2304
TP = 4
INTER_R = INTER // TP          # 576
N13 = 2 * INTER_R              # 1152
K1 = HID
K2 = INTER_R
N2 = HID
ACT_GROUP = 128                # FP8 activation group (DeepGemmFP4Experts._ACT_BLOCK_K)
W_GROUP = 32                   # MXFP4 weight group
CLAMP = 10.0                   # swiglu_limit
FP8_MAX = 448.0

CH = 128                       # k per 64-B chunk
W13_SCALE_BYTES = 192          # 160 used
W13_ROW_BYTES = W13_SCALE_BYTES + K1 // 2          # 2752
W2_SCALE_BYTES = 32            # 18 used
W2_ROW_BYTES = W2_SCALE_BYTES + K2 // 2            # 320
W13_BYTES = N13 * W13_ROW_BYTES                    # 3,170,304
W2_BYTES = N2 * W2_ROW_BYTES                       # 1,638,400
PACK_BYTES = W13_BYTES + W2_BYTES                  # 4,808,704
RAW_BYTES = (2 * INTER_R * K1 + N2 * K2) // 2 + 2 * INTER_R * (K1 // 32) + N2 * (K2 // 32)  # 4,700,160

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


# ---------------------------------------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def weight_map() -> dict:
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


_open_shards: dict = {}


def get_tensor(name: str) -> torch.Tensor:
    from safetensors import safe_open

    shard = weight_map()[name]
    f = _open_shards.get(shard)
    if f is None:
        f = safe_open(os.path.join(MODEL_DIR, shard), framework="pt")
        _open_shards[shard] = f
    return f.get_tensor(name)


def expert_slice_raw(layer: int, expert: int, rank: int, prefix: str = "layers") -> dict:
    """The rank's slice of one expert in the checkpoint's record: uint8 nibbles + uint8 e8m0 scales.

    Returns w1n [576, 2560], w1s [576, 160], w3n, w3s (same), w2n [5120, 288], w2s [5120, 18].
    `prefix="mtp.N"` reads a DSpark layer's expert (same shapes)."""
    base = f"{prefix}.{layer}.ffn.experts.{expert}"
    r0, r1 = INTER_R * rank, INTER_R * (rank + 1)
    out = {}
    for w in ("w1", "w3"):
        out[w + "n"] = get_tensor(f"{base}.{w}.weight").view(torch.uint8)[r0:r1].numpy()
        out[w + "s"] = get_tensor(f"{base}.{w}.scale").view(torch.uint8)[r0:r1].numpy()
    out["w2n"] = get_tensor(f"{base}.w2.weight").view(torch.uint8)[:, r0 // 2 : r1 // 2].numpy()
    out["w2s"] = get_tensor(f"{base}.w2.scale").view(torch.uint8)[:, r0 // 32 : r1 // 32].numpy()
    return {k: np.ascontiguousarray(v) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------------------
# exact dequant / quant helpers (torch, fp32)
# ---------------------------------------------------------------------------------------------------------
def dequant_mxfp4(nib: np.ndarray, scale: np.ndarray) -> torch.Tensor:
    """[N, K/2] nibbles + [N, K/32] e8m0 bytes -> fp32 [N, K] (exact)."""
    nib_t = torch.from_numpy(nib)
    lo = nib_t & 0xF
    hi = nib_t >> 4
    idx = torch.stack([lo, hi], dim=-1).reshape(nib.shape[0], -1)       # k = 2i (low), 2i+1 (high)
    mag = torch.from_numpy(E2M1)[(idx & 7).long()]
    val = torch.where((idx & 8).bool(), -mag, mag)
    sc = torch.ldexp(torch.ones(scale.shape, dtype=torch.float32), torch.from_numpy(scale).int() - 127)
    sc = sc.repeat_interleave(W_GROUP, dim=1)
    return val * sc


def ue8m0_exponent(amax: torch.Tensor) -> torch.Tensor:
    """exponent e with scale = 2^e, exactly ceil(log2(max(amax/448, 1e-10))) (integer-exact, no log2)."""
    raw = torch.clamp(amax.double() / FP8_MAX, min=1e-10)
    m, e = torch.frexp(raw)                       # raw = m * 2^e, m in [0.5, 1)
    e = e - (m == 0.5).int()                      # exact power of two -> ceil(log2) = e - 1
    return e.int()


def quant_fp8_ue8m0(x: torch.Tensor, group: int = ACT_GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """[M, K] fp32 -> (e4m3fn [M, K], exponent int32 [M, ceil(K/group)]); the last group may be short."""
    M, K = x.shape
    ngroups = (K + group - 1) // group
    pad = ngroups * group - K
    xp = torch.nn.functional.pad(x.float(), (0, pad)).view(M, ngroups, group)
    e = ue8m0_exponent(xp.abs().amax(dim=2))
    scale = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)
    q = torch.clamp(xp / scale[:, :, None], -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(M, -1)[:, :K].contiguous(), e


def dequant_fp8_ue8m0(q: torch.Tensor, e: torch.Tensor, group: int = ACT_GROUP) -> torch.Tensor:
    M, K = q.shape
    scale = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e).repeat_interleave(group, dim=1)[:, :K]
    return q.float() * scale


def bf16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


class ExpertEmulator:
    """The GPU's W4A8 math for one expert slice in torch fp32 (or fp64 for the summation-order floor)."""

    def __init__(self, raw: dict, dtype: torch.dtype = torch.float32):
        self.w1 = dequant_mxfp4(raw["w1n"], raw["w1s"]).to(dtype)
        self.w3 = dequant_mxfp4(raw["w3n"], raw["w3s"]).to(dtype)
        self.w2 = dequant_mxfp4(raw["w2n"], raw["w2s"]).to(dtype)
        self.dtype = dtype

    def forward_from_q(self, xq: torch.Tensor, xe: torch.Tensor, return_intermediates: bool = False):
        xd = dequant_fp8_ue8m0(xq, xe).to(self.dtype)
        g = bf16_round((xd @ self.w1.T).float())
        u = bf16_round((xd @ self.w3.T).float())
        gc = torch.clamp(g, max=CLAMP)
        uc = torch.clamp(u, -CLAMP, CLAMP)
        y = bf16_round(uc * (gc / (1.0 + torch.exp(-gc))))
        yq, ye = quant_fp8_ue8m0(y)
        yd = dequant_fp8_ue8m0(yq, ye).to(self.dtype)
        o = bf16_round((yd @ self.w2.T).float())
        if return_intermediates:
            return o, dict(g=g, u=u, y=y, yq=yq, ye=ye)
        return o

    def forward(self, x_bf16: torch.Tensor):
        xq, xe = quant_fp8_ue8m0(x_bf16.float())
        return self.forward_from_q(xq, xe)


# ---------------------------------------------------------------------------------------------------------
# kernel pack
# ---------------------------------------------------------------------------------------------------------
def _chunk_perm(nbytes: int) -> tuple[np.ndarray, np.ndarray]:
    """For a 64-B (128 k) or 32-B (64 k) chunk: k index held by the low / high nibble of byte b."""
    b = np.arange(nbytes)
    lane, pos = b // 16, b % 16
    half = nbytes // 2  # 32 k per nibble-plane for a full chunk, 16 for a half chunk
    k_lo = np.where(pos < 8, lane * 8 + pos, half + lane * 8 + (pos - 8))
    return k_lo, k_lo + nbytes  # high nibbles hold k + nbytes (64 for a full chunk, 32 for a half)


def _pack_rows(nib: np.ndarray, scale: np.ndarray, scale_bytes: int) -> np.ndarray:
    """[N, K/2] checkpoint nibbles (low = even k) + [N, K/32] scales -> [N, scale_bytes + K/2] kernel rows."""
    N, half_k = nib.shape
    K = 2 * half_k
    # unpack to one nibble index per k
    k_vals = np.empty((N, K), dtype=np.uint8)
    k_vals[:, 0::2] = nib & 0xF
    k_vals[:, 1::2] = nib >> 4
    out = np.zeros((N, scale_bytes + half_k), dtype=np.uint8)
    out[:, : scale.shape[1]] = scale
    pos = scale_bytes
    k0 = 0
    while k0 < K:
        n_k = min(CH, K - k0)
        nbytes = n_k // 2
        klo, khi = _chunk_perm(nbytes)
        blk = k_vals[:, k0 : k0 + n_k]
        out[:, pos : pos + nbytes] = blk[:, klo] | (blk[:, khi] << 4)
        pos += nbytes
        k0 += n_k
    return out


def build_pack(raw: dict) -> np.ndarray:
    """One expert slice -> the kernel's PACK_BYTES byte array."""
    w13n = np.concatenate([raw["w1n"], raw["w3n"]], axis=0)
    w13s = np.concatenate([raw["w1s"], raw["w3s"]], axis=0)
    p13 = _pack_rows(w13n, w13s, W13_SCALE_BYTES)
    p2 = _pack_rows(raw["w2n"], raw["w2s"], W2_SCALE_BYTES)
    pack = np.concatenate([p13.reshape(-1), p2.reshape(-1)])
    assert pack.nbytes == PACK_BYTES, (pack.nbytes, PACK_BYTES)
    return pack


def unpack_pack(pack: np.ndarray) -> dict:
    """Inverse of build_pack (for tests)."""
    p13 = pack[:W13_BYTES].reshape(N13, W13_ROW_BYTES)
    p2 = pack[W13_BYTES:].reshape(N2, W2_ROW_BYTES)

    def rows(p: np.ndarray, scale_bytes: int, K: int) -> tuple[np.ndarray, np.ndarray]:
        N = p.shape[0]
        scale = p[:, : K // 32]
        k_vals = np.empty((N, K), dtype=np.uint8)
        pos = scale_bytes
        k0 = 0
        while k0 < K:
            n_k = min(CH, K - k0)
            nbytes = n_k // 2
            klo, khi = _chunk_perm(nbytes)
            chunk = p[:, pos : pos + nbytes]
            blk = np.empty((N, n_k), dtype=np.uint8)
            blk[:, klo] = chunk & 0xF
            blk[:, khi] = chunk >> 4
            k_vals[:, k0 : k0 + n_k] = blk
            pos += nbytes
            k0 += n_k
        nib = (k_vals[:, 0::2] | (k_vals[:, 1::2] << 4)).astype(np.uint8)
        return nib, np.ascontiguousarray(scale)

    w13n, w13s = rows(p13, W13_SCALE_BYTES, K1)
    w2n, w2s = rows(p2, W2_SCALE_BYTES, K2)
    return dict(w1n=w13n[:INTER_R], w1s=w13s[:INTER_R], w3n=w13n[INTER_R:], w3s=w13s[INTER_R:], w2n=w2n, w2s=w2s)


# ---------------------------------------------------------------------------------------------------------
# the native library
# ---------------------------------------------------------------------------------------------------------
_LIB = None


def lib_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcpu_moe_mxfp4.so")


def load_lib(path: str | None = None):
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(path or lib_path())
        lib.moe_mxfp4_pack_bytes.restype = ctypes.c_int64
        lib.moe_mxfp4_pool_create.restype = ctypes.c_void_p
        lib.moe_mxfp4_pool_create.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.moe_mxfp4_pool_destroy.argtypes = [ctypes.c_void_p]
        lib.moe_mxfp4_experts_forward.restype = ctypes.c_int
        lib.moe_mxfp4_experts_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.moe_mxfp4_prepare_x.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.moe_mxfp4_act_quant.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_void_p, ctypes.c_int]
        lib.moe_mxfp4_read_bench.restype = ctypes.c_uint64
        lib.moe_mxfp4_read_bench.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        lib.moe_mxfp4_alloc.restype = ctypes.c_void_p
        lib.moe_mxfp4_alloc.argtypes = [ctypes.c_int64]
        lib.moe_mxfp4_free.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        assert lib.moe_mxfp4_pack_bytes() == PACK_BYTES, (lib.moe_mxfp4_pack_bytes(), PACK_BYTES)
        _LIB = lib
    return _LIB


class Pool:
    """A worker pool pinned to the given CPUs (one CCD for one rank)."""

    def __init__(self, cpu_ids: list[int], lib=None):
        self.lib = lib or load_lib()
        arr = (ctypes.c_int * len(cpu_ids))(*cpu_ids)
        self.handle = self.lib.moe_mxfp4_pool_create(len(cpu_ids), arr)
        assert self.handle, "pool creation failed"
        self.cpu_ids = list(cpu_ids)

    def close(self):
        if self.handle:
            self.lib.moe_mxfp4_pool_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def experts_forward(self, pack_ptrs: list[int], m_per_expert: list[int], token_idx: list[int],
                        xq: np.ndarray, xs: np.ndarray, out: np.ndarray) -> None:
        """xq [T, 5120] uint8 e4m3, xs [T, 40] uint8 ue8m0 bytes; out [sum(m), 5120] uint16 bf16 rows in
        (expert, token) order; token_idx = concatenated token lists (len sum(m))."""
        n = len(pack_ptrs)
        assert len(m_per_expert) == n and sum(m_per_expert) == len(token_idx) == out.shape[0]
        assert xq.dtype == np.uint8 and xs.dtype == np.uint8 and out.dtype == np.uint16
        assert xq.flags.c_contiguous and xs.flags.c_contiguous and out.flags.c_contiguous
        packs = (ctypes.c_void_p * n)(*pack_ptrs)
        ms = (ctypes.c_int * n)(*m_per_expert)
        ti = (ctypes.c_int * max(1, len(token_idx)))(*token_idx)
        rc = self.lib.moe_mxfp4_experts_forward(self.handle, n, packs, ms, ti, xq.shape[0],
                                                xq.ctypes.data, xs.ctypes.data, out.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"moe_mxfp4_experts_forward rc={rc}")

    def read_bench(self, pack_ptrs: list[int]) -> int:
        """Stream the packs through the workers with the kernel's striping, no compute (DRAM ceiling)."""
        n = len(pack_ptrs)
        packs = (ctypes.c_void_p * n)(*pack_ptrs)
        return int(self.lib.moe_mxfp4_read_bench(self.handle, n, packs))


class PackArena:
    """Packs in a hugepage-advised arena (the shape of the pinned expert store of phase 2)."""

    def __init__(self, n_experts: int, lib=None):
        self.lib = lib or load_lib()
        self.n = n_experts
        self.nbytes = n_experts * PACK_BYTES
        self.base = self.lib.moe_mxfp4_alloc(self.nbytes)
        assert self.base, "arena allocation failed"
        self.view = np.ctypeslib.as_array((ctypes.c_uint8 * self.nbytes).from_address(self.base))

    def slot(self, i: int) -> np.ndarray:
        return self.view[i * PACK_BYTES : (i + 1) * PACK_BYTES]

    def ptr(self, i: int) -> int:
        return self.base + i * PACK_BYTES

    def close(self):
        if self.base:
            self.lib.moe_mxfp4_free(self.base, self.nbytes)
            self.base = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def to_kernel_inputs(xq: torch.Tensor, xe: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """torch (e4m3fn [T, K], exponent int [T, G]) -> kernel bytes (uint8 codes, uint8 e+127)."""
    q = xq.view(torch.uint8).numpy().copy()
    s = (xe + 127).clamp(0, 255).to(torch.uint8).numpy().copy()
    return q, s


def bf16_bits_to_float(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.astype(np.int32) << 16).view(torch.float32)
