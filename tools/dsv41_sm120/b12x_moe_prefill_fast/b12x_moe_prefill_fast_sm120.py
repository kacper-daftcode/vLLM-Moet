"""JIT loader + host side of b12x_moe_prefill_fast_sm120.cu: the prefill MoE of b12x's `integration.tp_moe.dynamic`
deterministic path (FC1 + SwiGLU + MXFP8 quant, FC2, top-k sum) as pipelined CUDA kernels, bit for bit with spec B
(../b12x_moe_prefill_spec) and with b12x.

    from b12x_moe_prefill_fast_sm120 import moe_prefill_fast
    out, route, a_values, a_scales, inter, inter_sf, row_pair, counts = moe_prefill_fast(hidden, ids, weights, planes)

`planes` as in the spec loader: dict(w13=prepared.w13_rp, w13_sf=prepared.w13_sfb, w2=prepared.w2_rp, w2_sf=prepared.w2_sfb,
alpha=[E], input_global_scale=[E]|[1], down_alpha=[E], global_scale=[E]|[1]). `route_out` (optional, [>= T*topk, K] bf16):
the route rows go there (b12x's workspace.route_output when integrated); `topk_sum=False` skips the sum (b12x's own
TopKSum runs on the route rows instead). The routing (pair counts, padded expert offsets, the stable scatter, the CTA
work lists) is done on the GPU by the front-end kernel: no host synchronization. Tuning knobs (env, default = measured
best): VLLM_MOET_B12X_MOE_PREFILL_FAST_CFG1 / _CFG2 (stages, CTAs per SM: 0 = (3, 2), 1 = (4, 2)), _DYNAMIC (1 = atomic
work counter, 0 = static round-robin), _GRID1 / _GRID2 (CTA caps, 0 = occupancy). Every configuration computes the same bits.
The block-scaled QMMA needs sm_120a; the extension compiles on first import (~60 s) without fast-math / ftz.

Pre-quantized rows (the sequence-parallel MXFP8 all-gather of the MoE input, ../sp_moe_gather): `moe_prefill_fast(None, ids,
weights, planes, ..., a_values=q, a_scales=s)` with q uint8 [T, K] (e4m3, row-major) and s uint8 [T, K/32] (ue8m0, row-major)
skips the front-end's rows quant stage and feeds FC1 the given rows (the same bytes quant_row writes when they come from
FlashInfer's mxfp8 quantizer, see sp_moe_gather/test_prefill_sp_gather_tp4.py). `prequant_rows_supported()` tells whether the
built extension has that entry; without it the caller must pass bf16 rows. VLLM_MOET_B12X_MOE_PREFILL_FAST_NAME overrides the
extension's build name (a kernel variant next to the image's prebuilt one).

v9 (K-4): the producer-warp kernels (one warp issues the TMA / cp.async copies, 8 consumer warps never stop for a barrier or
address work; bit-identical to v8 by the harness) are selected per kernel by VLLM_MOET_B12X_MOE_PREFILL_FAST_WS1 (fc1) /
_WS2 (fc2): 0 = v8 (default), a mode = the ring (stages, CTAs per SM) - see the .cu; measured best fc1 2 (6 stages, 1 CTA/SM,
-4 % at 4096 rows), fc2 3 (4 stages, 2 CTAs/SM, -16 %); fc1 below _WS1_MIN_ROWS (256) rows stays on v8 (the 128-row replay
layers are at the DRAM floor).

v10 (K-5): fc1 as three warpgroups (`fc1_kernel_wg`: 8 consumer warps + 4 producer warps, one LDS.128 per A fragment from a layout
8-B cp.asyncs fill; at 2 CTAs/SM `setmaxnreg` 32 / 104) = WS1 modes 8 (3 stages, 2 CTAs/SM), 9 (4, 1), 10 (6, 1), 11 (7, 1); measured
best mode 9: fc1 -10.6 % at 4096 rows in the harness (913 vs 1 021 us), bit-identical; in situ (pipelined r26 step, WS1=9 WS2=3)
the 4096-row fc1 launches are -10.5 % (887 -> 794 us) but the 163K / 19K walls move <= 0.3 % (K/REPORT.md 7.4).

MK-2 (megakernel stage 1, milestone 1): `VLLM_MOET_B12X_MOE_PREFILL_FAST_WS` >= 20 runs fc1 + SwiGLU/quant + fc2 as ONE persistent launch
(`moe_persistent_kernel`: v10's CTA, 1 CTA/SM x 12 warps, one work list with the fc2 items gated by per-expert fc1_done counters, every wait
watchdog-bounded - an expired wait records to pinned host memory and traps; the separate topk_sum_kernel follows). 20-25 = fc2 items of one
128-column tile with 4 / 6 / 7 / 3 / 5 / 2 ring stages, 30-35 = fc2 items of two tiles (the fc1-shaped stage; the faster form). Knobs:
_WS_LEAD (the work list's lead in experts, default 64; >= the expert count = every fc1 item before the first fc2 item), _WS_MIN_ROWS (default 0),
_WS_WATCHDOG_MS (2000), _WS_STATS=1 (per-item %globaltimer statistics, moe_prefill_ws_stats()), _WS_ROUTE_CS (streaming route-row stores, 1).
Bit-identical to v8/v9/v10 on W's harness (6/6 at 4096 and 128 rows, 3 runs); measured +2.6 % vs WS1=9 + WS2=3 at call1 4096 rows
(MK/REPORT.md 2) - off by default.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
N_INTERMEDIATE = 576


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_B12X_MOE_PREFILL_FAST_ARCH", "12.0a")
    return load(
        name=os.environ.get("VLLM_MOET_B12X_MOE_PREFILL_FAST_NAME", "vllm_moet_b12x_moe_prefill_fast_sm120"),
        sources=[str(_HERE / "b12x_moe_prefill_fast_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20"] + os.environ.get("VLLM_MOET_B12X_MOE_PREFILL_FAST_NVCC_EXTRA", "").split(),
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def prequant_rows_supported() -> bool:
    """Whether the built extension takes pre-quantized rows (a_values / a_scales) in place of the bf16 rows."""
    return hasattr(_ext(), "prequant_rows_supported")


def strided_rows_supported() -> bool:
    """Whether the built extension takes a_values / a_scales as row-strided views (unit column stride, row strides and bases a
    multiple of 4 bytes) - e.g. buf[:, :K] and buf[:, K:K + K // 32] of one gathered [T, K + K // 32 + ...] buffer (session
    U's one-all-gather follow-up); older builds get contiguous copies."""
    return hasattr(_ext(), "strided_rows_supported")


def _rows_arg(x: torch.Tensor) -> torch.Tensor:
    if x.is_contiguous():
        return x
    if strided_rows_supported() and x.dim() == 2 and x.stride(1) == 1 and x.stride(0) % 4 == 0 and x.data_ptr() % 4 == 0:
        return x
    return x.contiguous()


def moe_prefill_fast(hidden: torch.Tensor | None, ids: torch.Tensor, weights: torch.Tensor, planes: dict, limit: float = 10.0,
                     n: int = N_INTERMEDIATE, route_out: torch.Tensor | None = None, out: torch.Tensor | None = None,
                     topk_sum: bool = True, cfg1: int | None = None, cfg2: int | None = None,
                     dynamic: int | None = None, grid1: int | None = None, grid2: int | None = None,
                     a_values: torch.Tensor | None = None, a_scales: torch.Tensor | None = None):
    cfg1 = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_CFG1", 0) if cfg1 is None else cfg1
    cfg2 = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_CFG2", 0) if cfg2 is None else cfg2
    dynamic = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_DYNAMIC", 1) if dynamic is None else dynamic
    grid1 = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_GRID1", 0) if grid1 is None else grid1
    grid2 = _env_int("VLLM_MOET_B12X_MOE_PREFILL_FAST_GRID2", 0) if grid2 is None else grid2
    ids = ids.to(torch.int32).contiguous()
    args = (ids, weights.to(torch.float32).contiguous(), planes["w13"],
            planes["w13_sf"], planes["w2"], planes["w2_sf"],
            planes["alpha"].to(torch.float32).contiguous(),
            planes["input_global_scale"].to(torch.float32).contiguous(),
            planes["down_alpha"].to(torch.float32).contiguous(),
            planes["global_scale"].to(torch.float32).contiguous(), float(limit), int(n),
            route_out, out,
            bool(topk_sum), int(cfg1), int(cfg2), int(dynamic), int(grid1), int(grid2))
    if a_values is not None:
        if not prequant_rows_supported():
            raise RuntimeError("this build of the fast prefill MoE extension takes bf16 rows only (no a_values / a_scales entry)")
        assert a_scales is not None, "a_scales must come with a_values"
        return _ext().moe_prefill_fast(None if hidden is None else hidden.contiguous(), *args,
                                       _rows_arg(a_values), _rows_arg(a_scales))
    assert hidden is not None, "hidden (bf16 rows) or a_values + a_scales (pre-quantized rows)"
    if prequant_rows_supported():
        return _ext().moe_prefill_fast(hidden.contiguous(), *args, None, None)
    return _ext().moe_prefill_fast(hidden.contiguous(), *args)


def supported(*, n: int, k: int, num_topk: int, experts: int, ids_dtype) -> bool:
    """The shapes the kernels handle: intermediate 576, hidden a multiple of 128 (>= 640), <= 512 experts, int32 ids."""
    return int(n) == N_INTERMEDIATE and int(k) % 128 == 0 and int(k) >= 640 and 1 <= int(experts) <= 512 and \
        1 <= int(num_topk) <= 64 and ids_dtype == torch.int32
