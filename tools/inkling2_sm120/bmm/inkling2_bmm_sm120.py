"""The batched expert GEMM of the Inkling2 MoE on sm_120 (inkling2_bmm_sm120.cu): loader, the prepared weight layout,
the 6-bit packing of MXFP8 expert weights, and reference decoders for the tests.

    from inkling2_bmm_sm120 import NVFP4, MXFP8, MXFP6, bmm, prepare, pack_e2m3

    w = prepare(weight_bytes, 32)                      # [E, N, K/2] row-major NVFP4 -> the kernel's layout, once
    out = bmm(NVFP4, act, act_scale, row_scale, w, w_scale, cta_expert, cta_limit, num_ctas, tile, n, k)

`bmm` is vLLM's `trtllm_bmm.bmm` for sm_120: bf16 [tiles * tile, N], row r = row_scale[r] * act[r] @ weight[e(r)]^T
for the rows of the tiles in use; tile t holds rows [t * tile, cta_limit[t]) of local expert cta_expert[t].
Activations and their block scales are row-major; weights and their block scales are in the prepared layout.

Formats
  NVFP4  act uint8 [R, K/2] (two e2m1 per byte, the lower column in the low nibble), act_scale e4m3 [R, K/16],
         row_scale fp32 [R]; weight uint8 [E, N, K/2], weight_scale e4m3 [E, N, K/16]
  MXFP8  act e4m3 [R, K], act_scale uint8 [R, K/32] (ue8m0); weight e4m3 [E, N, K], weight_scale uint8 [E, N, K/32]
  MXFP6  activations as MXFP8; weight uint8 [E, N, 3K/4]: e2m3, six bits per value, value i of a group of 16 at
         bit 6 i of three little-endian 32-bit words; weight_scale uint8 [E, N, K/32] (`pack_e2m3`)

The first import compiles the extension (torch.utils.cpp_extension.load, ~1 min, cached under $TORCH_EXTENSIONS_DIR;
the image compiles it at build time). The block-scaled QMMAs need sm_120a. Tuning: VLLM_MOET_INKLING2_BMM_CPS (CTAs
per SM, default 2) and VLLM_MOET_INKLING2_BMM_PF (K64 steps of weights prefetched into L2 ahead of the ring, default 4).
VLLM_MOET_INKLING2_BMM=ref computes `bmm` by `reference` instead (float64 on decoded operands; eager only, slow).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
NVFP4, MXFP8, MXFP6 = 0, 1, 2
PREP_ROWS = 128
#: weight bytes per row and K64 step / scale bytes per row and K64 step of each format
WEIGHT_BYTES_PER_STEP = {NVFP4: 32, MXFP8: 64, MXFP6: 48}
SCALE_BYTES_PER_STEP = {NVFP4: 4, MXFP8: 2, MXFP6: 2}
_CTRL: dict[int, torch.Tensor] = {}


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_INKLING2_BMM_ARCH", "12.0a")
    return load(
        name="vllm_moet_inkling2_bmm_sm120",
        sources=[str(_HERE / "inkling2_bmm_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def supported(device: torch.device | int | None = None) -> bool:
    """sm_120 / sm_121: the cards FlashInfer's TRT-LLM-gen MoE cubins do not exist for."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] == 12


def _ctrl(device: torch.device) -> torch.Tensor:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    c = _CTRL.get(idx)
    if c is None:
        c = torch.zeros(2, dtype=torch.int32, device=torch.device("cuda", idx))
        _CTRL[idx] = c
    return c


_FILE_MAPS: list[tuple[int, int, int, str]] = []  # (start, end, file offset, path) of this process's file mappings


def _file_mapping(addr: int):
    for known in (True, False):
        if not known:
            _FILE_MAPS.clear()
            with open("/proc/self/maps") as maps:
                for line in maps:
                    parts = line.split(None, 5)
                    if len(parts) == 6 and parts[5].startswith("/"):
                        lo, hi = (int(v, 16) for v in parts[0].split("-"))
                        _FILE_MAPS.append((lo, hi, int(parts[2], 16), parts[5].rstrip("\n")))
        for lo, hi, offset, path in _FILE_MAPS:
            if lo <= addr < hi:
                return lo, hi, offset, path
    return None


def drop_file_pages(t: torch.Tensor) -> int:
    """For a slice of a memory-mapped checkpoint tensor that has been copied to the card: unmap its pages and tell
    the kernel that the file range will not be read again (madvise + posix_fadvise DONTNEED), so that the load of a
    394 GB checkpoint leaves no page cache behind it - a model start beside another server otherwise pushes the host
    into reclaim. Returns the bytes released; 0 for a tensor that is not file-backed, with
    VLLM_MOET_INKLING2_DROP_PAGES=0, or on any error (the load does not depend on it)."""
    if os.environ.get("VLLM_MOET_INKLING2_DROP_PAGES", "1") == "0" or t.device.type != "cpu" or t.numel() == 0:
        return 0
    try:
        import ctypes

        addr, size = t.data_ptr(), t.numel() * t.element_size()
        found = _file_mapping(addr)
        if found is None or not t.is_contiguous():
            return 0
        lo, hi, offset, path = found
        page = os.sysconf("SC_PAGE_SIZE")
        first = (addr + page - 1) // page * page  # whole pages inside the slice: its neighbours may be another rank's
        last = min(addr + size, hi) // page * page
        if last <= first:
            return 0
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise(ctypes.c_void_p(first), ctypes.c_size_t(last - first), 4)  # MADV_DONTNEED: file pages, nothing is lost
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, offset + first - lo, last - first, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return last - first
    except Exception:  # noqa: BLE001
        return 0


def prepare(x: torch.Tensor, bytes_per_step: int) -> torch.Tensor:
    """Row-major bytes [E, N, X] of expert weights (or of their block scales) -> the kernel's layout, the same
    shape: within each block of 128 rows the K64 steps come first, [E, N/128, steps, 128, bytes_per_step]."""
    experts, n, x_bytes = x.shape
    assert n % PREP_ROWS == 0 and x_bytes % bytes_per_step == 0, (x.shape, bytes_per_step)
    steps = x_bytes // bytes_per_step
    y = x.reshape(experts, n // PREP_ROWS, PREP_ROWS, steps, bytes_per_step).permute(0, 1, 3, 2, 4)
    return y.contiguous().reshape(experts, n, x_bytes)


def unprepare(y: torch.Tensor, bytes_per_step: int) -> torch.Tensor:
    """The inverse of `prepare`."""
    experts, n, x_bytes = y.shape
    steps = x_bytes // bytes_per_step
    x = y.reshape(experts, n // PREP_ROWS, steps, PREP_ROWS, bytes_per_step).permute(0, 1, 3, 2, 4)
    return x.contiguous().reshape(experts, n, x_bytes)


def _e4m3_magnitude(codes: torch.Tensor) -> torch.Tensor:
    """|value| of e4m3 codes in units of 2^-9 (int32): subnormals m, normals (8 + m) << (e - 1)."""
    c = codes.to(torch.int32)
    e, m = (c >> 3) & 0xF, c & 7
    return torch.where(e == 0, m, (8 + m) << (e - 1).clamp(min=0))


def pack_e2m3(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 expert weights (e4m3 codes [..., K], K % 32 == 0) -> e2m3 packed 6 bits per value (uint8 [..., 3K/4])
    and the int8 change of each block's ue8m0 scale exponent ([..., K/32]; new scale byte = old + delta).

    Per block of 32 the scale moves by a power of two so that the block's largest magnitude lies in e2m3's top
    binade [4, 8). Values in the block's top three binades keep all three mantissa bits (exact); smaller ones are
    rounded to nearest even on e2m3's subnormal grid (an eighth of the third binade's lower end)."""
    codes = weight.view(torch.uint8) if weight.dtype != torch.uint8 else weight
    *lead, k = codes.shape
    assert k % 32 == 0
    c = codes.reshape(-1, k // 32, 32)
    assert not bool(((c & 0x7F) == 0x7F).any()), "NaN codes in the weights"
    mag = _e4m3_magnitude(c)
    mag_max = mag.amax(dim=2, keepdim=True)
    # p = floor(log2(max)) in integers (a floating log2 is not exact at powers of two on every device); 0 for an
    # all-zero block
    p = torch.zeros_like(mag_max)
    for bit in range(1, 18):
        p += (mag_max >= (1 << bit)).to(torch.int32)
    shift = p - 5
    left = (-shift).clamp(min=0)
    right = shift.clamp(min=0)
    g = mag << left
    # round half to even at 2^right
    half = torch.where(right > 0, torch.ones_like(right) << (right - 1).clamp(min=0), torch.zeros_like(right))
    floor = g >> right
    rem = g - (floor << right)
    up = (rem > half) | ((rem == half) & (right > 0) & ((floor & 1) == 1))
    g = floor + up.to(torch.int32)
    assert int(g.max()) <= 60, int(g.max())
    ee = (g >= 8).to(torch.int32) + (g >= 16).to(torch.int32) + (g >= 32).to(torch.int32)
    mmm = torch.where(ee <= 1, g - 8 * (ee == 1).to(torch.int32), (g >> (ee - 1).clamp(min=0)) - 8)
    code6 = ((c.to(torch.int32) >> 7) << 5) | (ee << 3) | mmm
    delta = torch.where(mag_max > 0, p - 11, torch.zeros_like(p)).squeeze(2)
    # 16 values -> 96 bits -> three little-endian words
    v = code6.reshape(-1, k // 16, 4, 4).to(torch.int64)
    v24 = v[..., 0] | (v[..., 1] << 6) | (v[..., 2] << 12) | (v[..., 3] << 18)
    w0 = (v24[..., 0] | (v24[..., 1] << 24)) & 0xFFFFFFFF
    w1 = ((v24[..., 1] >> 8) | (v24[..., 2] << 16)) & 0xFFFFFFFF
    w2 = ((v24[..., 2] >> 16) | (v24[..., 3] << 8)) & 0xFFFFFFFF
    words = torch.stack((w0, w1, w2), dim=-1)
    shifts = torch.tensor([0, 8, 16, 24], device=words.device, dtype=torch.int64)
    packed = ((words.unsqueeze(-1) >> shifts) & 0xFF).to(torch.uint8)
    return packed.reshape(*lead, k * 3 // 4), delta.to(torch.int8).reshape(*lead, k // 32)


def bmm(
    fmt: int,
    act: torch.Tensor,
    act_scale: torch.Tensor,
    row_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    cta_expert: torch.Tensor,
    cta_limit: torch.Tensor,
    num_ctas: torch.Tensor,
    tile: int,
    n: int,
    k: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    max_ctas = cta_expert.numel()
    if out is None:
        out = torch.empty((max_ctas * tile, n), dtype=torch.bfloat16, device=act.device)
    if os.environ.get("VLLM_MOET_INKLING2_BMM", "1") == "ref":
        return _bmm_reference(fmt, act, act_scale, row_scale, weight, weight_scale, cta_expert, cta_limit, num_ctas, tile, out)
    cps = int(os.environ.get("VLLM_MOET_INKLING2_BMM_CPS", "2"))
    pf = int(os.environ.get("VLLM_MOET_INKLING2_BMM_PF", "4"))
    _ext().bmm_run(
        int(fmt), int(tile), weight, weight_scale, act, act_scale, row_scale, out, cta_expert, cta_limit, num_ctas,
        _ctrl(out.device), int(n), int(k), cps, pf,
    )
    return out


def _bmm_reference(fmt, act, act_scale, row_scale, weight, weight_scale, cta_expert, cta_limit, num_ctas, tile, out):
    """`bmm` by the definition (`reference`: the operands decoded to float64, one rounding) - the server's path with
    VLLM_MOET_INKLING2_BMM=ref, for telling the kernel from the rest of a model in a paired run. It reads the tile
    table on the host, so it needs --enforce-eager, and it decodes every used expert: slow."""
    tiles = int(num_ctas.reshape(-1)[0])
    experts, limits = cta_expert[:tiles].tolist(), cta_limit[:tiles].tolist()
    for t, (e, limit) in enumerate(zip(experts, limits)):
        rows = slice(t * tile, limit)
        if limit <= t * tile:
            continue
        w = unprepare(weight[e : e + 1].view(torch.uint8), WEIGHT_BYTES_PER_STEP[fmt])
        ws = unprepare(weight_scale[e : e + 1].view(torch.uint8), SCALE_BYTES_PER_STEP[fmt])
        out[rows] = reference(
            fmt, act[rows], act_scale[rows], None if row_scale is None else row_scale[rows], w, ws,
            torch.zeros(limit - t * tile, dtype=torch.long, device=act.device),
        )
    return out


def kernel_facts(fmt: int, tile: int) -> dict[str, int]:
    smem, resident, regs = _ext().kernel_facts(int(fmt), int(tile))
    return {"smem": int(smem), "ctas_per_sm": int(resident), "registers": int(regs)}


# ---------------------------------------------------------------------------------------------- reference decoders
def e4m3_values(codes: torch.Tensor) -> torch.Tensor:
    """float64 values of e4m3 codes (uint8)."""
    c = codes.to(torch.int32)
    mag = _e4m3_magnitude(c).to(torch.float64) * 2.0**-9
    return torch.where((c >> 7) == 1, -mag, mag)


def e2m1_values(packed: torch.Tensor) -> torch.Tensor:
    """float64 [..., 2X] values of packed e2m1 bytes [..., X] (the lower column in the low nibble)."""
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64,
                         device=packed.device)
    b = packed.to(torch.int64)
    return torch.stack((table[b & 0xF], table[b >> 4]), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def e2m3_values(packed: torch.Tensor) -> torch.Tensor:
    """float64 [..., 4X/3] values of e2m3 packed 6 bits per value (`pack_e2m3`'s layout)."""
    *lead, x = packed.shape
    w = packed.reshape(-1, x // 12, 3, 4).to(torch.int64)
    words = w[..., 0] | (w[..., 1] << 8) | (w[..., 2] << 16) | (w[..., 3] << 24)
    w0, w1, w2 = words[..., 0], words[..., 1], words[..., 2]
    v24 = torch.stack((w0 & 0xFFFFFF, ((w0 >> 24) | (w1 << 8)) & 0xFFFFFF, ((w1 >> 16) | (w2 << 16)) & 0xFFFFFF,
                       (w2 >> 8) & 0xFFFFFF), dim=-1)
    shifts = torch.tensor([0, 6, 12, 18], device=packed.device, dtype=torch.int64)
    code = (v24.unsqueeze(-1) >> shifts) & 0x3F
    s, ee, mmm = code >> 5, (code >> 3) & 3, code & 7
    mag = torch.where(ee == 0, mmm.to(torch.float64) / 8, (1 + mmm.to(torch.float64) / 8) * 2.0 ** (ee - 1).clamp(min=0))
    return torch.where(s == 1, -mag, mag).reshape(*lead, x * 4 // 3)


def reference(
    fmt: int,
    act: torch.Tensor,
    act_scale: torch.Tensor,
    row_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_of_row: torch.Tensor,
    with_bound: bool = False,
):
    """The exact result (float64 products and sums, then the fp32 row scale and one rounding to bf16) for
    row-major operands: act [R, .], weight [E, N, .] with their block scales; expert_of_row [R]. With `with_bound`
    also the unrounded float64 result and the sum of the terms' magnitudes (the scale of an fp32 accumulation's
    absolute error)."""
    if fmt == NVFP4:
        a = e2m1_values(act.view(torch.uint8)) * e4m3_values(act_scale.view(torch.uint8)).repeat_interleave(16, dim=-1)
        w = e2m1_values(weight.view(torch.uint8)) * e4m3_values(weight_scale.view(torch.uint8)).repeat_interleave(16, dim=-1)
    else:
        a = e4m3_values(act.view(torch.uint8)) * (2.0 ** (act_scale.to(torch.float64) - 127)).repeat_interleave(32, dim=-1)
        wv = e4m3_values(weight.view(torch.uint8)) if fmt == MXFP8 else e2m3_values(weight)
        w = wv * (2.0 ** (weight_scale.to(torch.float64) - 127)).repeat_interleave(32, dim=-1)
    out = torch.empty((a.shape[0], w.shape[1]), dtype=torch.float64, device=a.device)
    mag = torch.empty_like(out) if with_bound else None
    for e in torch.unique(expert_of_row).tolist():
        rows = (expert_of_row == e).nonzero().squeeze(1)
        out[rows] = a[rows] @ w[e].T
        if mag is not None:
            mag[rows] = a[rows].abs() @ w[e].abs().T
    if fmt == NVFP4:
        assert row_scale is not None
        scale = row_scale[: out.shape[0], None].to(torch.float64)
        exact = out * scale
        rounded = (out.to(torch.float32) * row_scale[: out.shape[0], None]).to(torch.bfloat16)
        if mag is not None:
            mag = mag * scale.abs()
    else:
        exact = out
        rounded = out.to(torch.float32).to(torch.bfloat16)
    return (rounded, exact, mag) if with_bound else rounded
