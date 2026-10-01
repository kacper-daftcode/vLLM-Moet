"""JIT loader, calibration and dispatch for the sm_120 small-M BF16 GEMM that reproduces cuBLAS's bits
(bf16_replica_gemm_sm120.cu).

    from bf16_replica_gemm_sm120 import replica_mm
    out = replica_mm(x, w, out_f32, site)   # == torch.mm(x, w.T, out_dtype=torch.float32) when out_f32,
                                            #    torch.nn.functional.linear(x, w) otherwise - bit for bit

cuBLAS picks its kernel and split-K slice count per shape, token count and GPU (an RTX 5090 and an RTX PRO 6000
choose differently), and the result bits follow from the slice width alone (see the .cu header). The first eager
call with a weight shape - vLLM's profile run - calibrates every token count 2..16 against cuBLAS with that weight:
the first slice width whose result matches cuBLAS bit for bit on wide-range random activations (>= 60k outputs,
fp32 or bf16 bit patterns) is kept; token counts without a match, or outside the kernels' range, stay on cuBLAS.
Shapes that are first called inside a CUDA-graph capture (the indexer's wk: the profile run skips it) are registered
up front and calibrated on a random weight at the first eager call of any site. During capture only calibrated cases
take the kernels, so a captured graph computes the same bits as the cuBLAS one either way.

Sites (VLLM_MOET_BF16_GEMM_SITES, comma-separated; default indexer,wk,compressor):
  indexer     lightning-indexer weights_proj 5120 -> 32, bf16 out (cuBLAS runs it on 2 CTAs: ~35 us on the side
              stream in the served graph, the main stream waits ~26 us for it in every index-source layer)
  wk          indexer K projection 512 -> 128, bf16 out
  compressor  compressor fused_wkv_wgate 5120 -> 1024 / 512, fp32 out
  router      MoE gate (GateLinear tier 4) 5120 -> 384, drafter 5120 -> 128, fp32 out
VLLM_MOET_BF16_GEMM=0 turns all of it off (checked by the patched call sites). Where cuBLAS splits K the kernels
are used up to VLLM_MOET_BF16_GEMM_SPLITK_MAX_M rows (8; above that they lose to cuBLAS), the one-slice chain up to 16.

The first import compiles the extension with torch.utils.cpp_extension.load (~60 s, cached under
$TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions; the serving image compiles it at build time).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
MAX_M = 16
# The split-K kernels lose to cuBLAS above 8 rows on the RTX PRO 6000 (rows 9-16 double the activation fragments:
# compressor 5120->1024 at 12 tokens 16.9 vs 12.1 us); the one-slice chain wins up to 16.
SPLITK_MAX_M = int(os.environ.get("VLLM_MOET_BF16_GEMM_SPLITK_MAX_M", "8"))
MIN_OUTPUTS = 60_000
MAX_TRIALS = 256
SPREAD_WARPS = 4
DEFAULT_SITES = "indexer,wk,compressor"
# kernel tried first when cuBLAS splits K (the other one if the first cannot take the case); measured on the
# RTX PRO 6000 with cold weights: compressor split 9.2 vs spread 9.7 us (cuBLAS 11.8), router spread 5.1 vs split 5.7
SPLIT_ORDER = {"compressor": ("split", "spread"), "router": ("spread", "split")}

try:
    from vllm.logger import init_logger

    _logger = init_logger("vllm.vllm_moet.bf16_gemm")  # under vllm.*: vLLM's handler prints it
except Exception:  # noqa: BLE001 - standalone use (tests)
    import logging

    _logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_BF16_GEMM_ARCH", "12.0")
    cuda_flags = ["-O3", "-std=c++17"]  # no fast-math: the reduction adds must keep denormals
    if os.environ.get("VLLM_MOET_BF16_GEMM_PTXAS", "0") == "1":
        cuda_flags += ["-Xptxas", "-v"]
    return load(
        name="vllm_moet_bf16_replica_gemm_sm120",
        sources=[str(_HERE / "bf16_replica_gemm_sm120.cu")],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=bool(int(os.environ.get("VLLM_MOET_BF16_GEMM_VERBOSE", "0"))),
    )


@functools.lru_cache(maxsize=None)
def enabled_sites() -> frozenset[str]:
    return frozenset(s.strip() for s in os.environ.get("VLLM_MOET_BF16_GEMM_SITES", DEFAULT_SITES).split(",") if s.strip())


def cublas_mm(x: torch.Tensor, w: torch.Tensor, out_f32: bool) -> torch.Tensor:
    """The op the call sites ran before: GateLinear tier 4 / the compressor (fp32) or ReplicatedLinear (bf16)."""
    return torch.mm(x, w.T, out_dtype=torch.float32) if out_f32 else F.linear(x, w)


@dataclass(frozen=True)
class Case:
    kind: str  # "chain" (one slice), "split" or "spread"
    step: int  # slice width (k); == K for "chain"
    pbf16: bool  # partials rounded to bf16 before the reduction (cuBLAS's bf16-workspace splitKreduce)


class Shape:
    """Calibrated cases of one weight shape / output dtype / activation row stride on one device."""

    def __init__(self, n: int, k: int, out_f32: bool, ldx: int, device: torch.device):
        self.n, self.k, self.out_f32, self.ldx, self.device = n, k, out_f32, ldx, device
        self.cases: dict[int, Case] = {}
        # (slice count, stream, capture) -> (partials, tile counters) of the spread kernel. Per stream: two calls of
        # one shape run concurrently only from different streams (also as nodes of a captured graph). Per capture:
        # a workspace first touched inside a capture is zeroed by a node of that graph only, so no other graph -
        # which might replay first - may reuse it.
        self.ws: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def workspace(self, slices: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (slices, torch.cuda.current_stream(self.device).cuda_stream, _ext().capture_id())
        if key not in self.ws:
            self.ws[key] = (
                torch.empty(self.n // 8 * slices * 128, device=self.device, dtype=torch.float32),
                torch.zeros(self.n // 8, device=self.device, dtype=torch.int32),
            )
        return self.ws[key]

    def describe(self) -> str:
        runs: list[list] = []
        for m in range(1, MAX_M + 1):
            c = self.cases.get(m)
            label = "cuBLAS" if c is None else (c.kind if c.kind == "chain" else f"{c.kind} {self.k // c.step + (self.k % c.step > 0)}x{c.step}")
            if runs and runs[-1][2] == label:
                runs[-1][1] = m
            else:
                runs.append([m, m, label])
        return ", ".join(f"M {a}{'' if a == b else f'-{b}'} {label}" for a, b, label in runs)


_SHAPES: dict[tuple, Shape] = {}
_PENDING: dict[tuple, str] = {}  # (n, k, out_f32, ldx) -> site, calibrated on a random weight


def _key(device: torch.device, n: int, k: int, out_f32: bool, ldx: int) -> tuple:
    return (device.index, n, k, out_f32, ldx)


def _candidates(k: int):
    """Slice widths to try: slice counts 1 (no split-K) .. 32, each width K / S when that divides, else rounded up
    to a multiple of 16, 32 or 64. The bits tell the right one apart."""
    seen = []
    for s in range(1, 33):
        widths = [k // s] if k % s == 0 else []
        widths += [-(-(-(-k // s)) // a) * a for a in (16, 32, 64)]
        for wd in widths:
            if wd not in seen:
                seen.append(wd)
                yield wd


def _supported(kind: str, m: int, n: int, k: int, step: int) -> bool:
    ext = _ext()
    if kind == "chain":
        return step == k and ext.chain_supported(m, n, k)
    if kind == "split":
        return ext.split_supported(m, n, k, step)
    return ext.spread_supported(m, n, k, step)


def _run(shape: Shape, case: Case, x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    ext = _ext()
    if case.kind == "chain":
        ext.chain_out(x, w, out)
    elif case.kind == "split":
        ext.split_out(x, w, out, case.step, case.pbf16)
    else:
        slices = -(-shape.k // case.step)
        ws, cnt = shape.workspace(slices)
        ext.spread_out(x, w, out, case.step, case.pbf16, ws, cnt, SPREAD_WARPS)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def _random_x(m: int, k: int, ldx: int, device: torch.device, seed: int) -> torch.Tensor:
    """Wide-range activations (x 2^-6 .. 2^6) with row stride ldx: accumulation-order differences show up in the
    bits far more often than with unit-range data."""
    g = torch.Generator(device=device).manual_seed(seed)
    base = torch.empty(m, ldx, device=device, dtype=torch.bfloat16)
    v = torch.randn(m, k, device=device, generator=g) * torch.exp2(
        torch.randint(-6, 7, (m, k), device=device, generator=g).float()
    )
    base[:, :k] = v.to(torch.bfloat16)
    return base[:, :k]


def calibrate(w: torch.Tensor, out_f32: bool, ldx: int, site: str) -> Shape:
    """Find, for every token count 2..16, the slice width that reproduces cuBLAS's bits for this weight shape."""
    n, k = w.shape
    shape = Shape(n, k, out_f32, ldx, w.device)
    order = ("chain",) + SPLIT_ORDER.get(site, ("spread", "split"))
    for m in range(2, MAX_M + 1):
        trials = max(2, min(MAX_TRIALS, -(-MIN_OUTPUTS // (m * n))))
        seeds = [(0x6EED << 20) ^ (n << 8) ^ (k << 3) ^ (m << 40) ^ (i << 48) for i in range(trials)]
        refs = [_bits(cublas_mm(_random_x(m, k, ldx, w.device, sd), w, out_f32)) for sd in seeds]
        found = None
        for step in _candidates(k):
            slices = -(-k // step)
            kinds = ("chain",) if slices == 1 else (order[1:] if m <= SPLITK_MAX_M else ())
            kind = next((kd for kd in kinds if _supported(kd, m, n, k, step)), None)
            if kind is None:
                continue
            for pb in (False,) if (out_f32 or slices == 1) else (False, True):
                case = Case(kind, step, pb)
                out = torch.empty(m, n, device=w.device, dtype=torch.float32 if out_f32 else torch.bfloat16)
                ok = True
                for sd, ref in zip(seeds, refs):
                    _run(shape, case, _random_x(m, k, ldx, w.device, sd), w, out)
                    if not torch.equal(_bits(out), ref):
                        ok = False
                        break
                if ok:
                    found = case
                    break
            if found is not None:
                break
        if found is not None:
            shape.cases[m] = found
    _logger.info(
        "vllm-moet sm_120 BF16 GEMM %d->%d %s (%s): %s",
        k, n, "fp32" if out_f32 else "bf16", site, shape.describe(),
    )
    return shape


def register(n: int, k: int, out_f32: bool, ldx: int, site: str) -> None:
    """A shape whose first call happens inside graph capture: calibrate it (random weight) at the next eager call."""
    if site in enabled_sites():
        _PENDING.setdefault((n, k, out_f32, ldx), site)


def _calibrate_pending(device: torch.device) -> None:
    while _PENDING:
        (n, k, out_f32, ldx), site = _PENDING.popitem()
        key = _key(device, n, k, out_f32, ldx)
        if key not in _SHAPES:
            g = torch.Generator(device=device).manual_seed(n * 131 + k)
            w = (torch.randn(n, k, device=device, generator=g) * 0.05).to(torch.bfloat16)
            _SHAPES[key] = calibrate(w, out_f32, ldx, site)


def replica_mm(x: torch.Tensor, w: torch.Tensor, out_f32: bool, site: str) -> torch.Tensor:
    """torch.mm(x, w.T, out_dtype=torch.float32) (out_f32) / F.linear(x, w) with cuBLAS's bits."""
    if (
        site not in enabled_sites()
        or x.dim() != 2
        or x.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or not x.is_cuda
        or x.stride(1) != 1
        or not w.is_contiguous()
    ):
        return cublas_mm(x, w, out_f32)
    n, k = w.shape
    key = _key(x.device, n, k, out_f32, x.stride(0))
    shape = _SHAPES.get(key)
    if shape is None or _PENDING:
        if not (torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()):
            if shape is None:
                shape = _SHAPES[key] = calibrate(w, out_f32, x.stride(0), site)
            _calibrate_pending(x.device)
        elif shape is None:
            return cublas_mm(x, w, out_f32)
    case = shape.cases.get(x.shape[0])
    if case is None or x.data_ptr() % 16 or w.data_ptr() % 16 or x.stride(0) % 8:
        return cublas_mm(x, w, out_f32)
    out = torch.empty(x.shape[0], n, device=x.device, dtype=torch.float32 if out_f32 else torch.bfloat16)
    _run(shape, case, x, w, out)
    return out


def table() -> dict[tuple, dict[int, Case]]:
    """Calibrated cases per (device, N, K, out_f32, ldx) - for tests and logs."""
    return {key: dict(shape.cases) for key, shape in _SHAPES.items()}
