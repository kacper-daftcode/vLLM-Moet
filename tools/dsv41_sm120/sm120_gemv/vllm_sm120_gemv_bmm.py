# SPDX-License-Identifier: Apache-2.0
"""sm_120 MXFP8 kernel for DeepSeek-V4's grouped o-projection (`wo_a`, `is_bmm=True`).

Installed by tools/dsv41_sm120/patch_vllm_wo_a_sm120.py as
vllm/model_executor/kernels/linear/mxfp8/sm120_gemv_bmm.py and put first in
init_mxfp8_linear_kernel()'s BMM candidate list.

Why: DeepGemmMxfp8BmmLinearKernel is sm_100-only, so on RTX PRO 6000 vLLM falls back to
EmulationMxfp8LinearKernel, which dequantizes wo_a to BF16 at load (2x the bytes) and
runs cuBLAS bmm: 26 us + 3 us split-K reduce per layer at decode, 1.0 ms of a 15.7 ms
step (profile 2026-09-18). This kernel keeps the checkpoint's MXFP8 weight + ue8m0
scales and runs the vLLM-Moet tensor-core GEMV (tools/dsv41_sm120/sm120_gemv/) on the
FP8 activations that fused_inv_rope_fp8_quant already produces for the sm_100 path,
so decode reads half the bytes and skips the BF16 bmm. Batches above 64 tokens
(prefill) dequantize the weight on the fly and use the original bf16 bmm.

Runtime switch: VLLM_MOET_SM120_GEMV_BMM=0 -> EmulationMxfp8LinearKernel as before.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

logger = init_logger(__name__)

DEFAULT_GEMV_DIR = "/opt/vllm-moet/dsv41_sm120/sm120_gemv"
_QUANT_GROUP = MXFP8_BLOCK_SIZE  # 32
_PIPE_ROW_BLOCKS = None  # None = unresolved, False = no pipeline module, else moet_prefill_pipe.row_blocks


def _row_blocks(num_rows: int):
    """K-9 (sp_pipeline shared replay tail): the row blocks a cuBLAS call over `num_rows` tokens is issued in - one per group
    inside the pipelined prefill's shared tail (the bits of the group's own 128-token call), one block everywhere else or
    without the pipeline module."""
    global _PIPE_ROW_BLOCKS
    if _PIPE_ROW_BLOCKS is None:
        try:
            from vllm.v1.worker import moet_prefill_pipe as _pipe  # noqa: PLC0415

            _PIPE_ROW_BLOCKS = _pipe.row_blocks
        except Exception:  # noqa: BLE001 - an image without the pipeline patch
            _PIPE_ROW_BLOCKS = False
    if not _PIPE_ROW_BLOCKS:
        return [(0, num_rows)]
    return _PIPE_ROW_BLOCKS(num_rows)


# The prefill branch of o_proj (tokens > MAX_GROUPED_GEMV_T) dequantizes the MXFP8 wo_a to bf16 for cuBLAS on every call; vLLM's
# dequant_mxfp8_to_bf16 does that with six PyTorch kernels (fp8 -> fp32 copy, scale copy, - 127, exp2, the fp32 broadcast multiply,
# fp32 -> bf16 copy: ~50 us of GPU time alone and ~98 us next to the other stream's kernels, ~160 MB of memory traffic per layer call;
# 60 calls per 8 192-token step = ~6 ms and ~9.6 GB per step). VLLM_MOET_WOA_DEQUANT_FUSED=1 does the same arithmetic in one Triton kernel:
# the e4m3 value to fp32 (exact), descale = libdevice exp2(scale - 127) (the same __nv_exp2f that torch.exp2 calls), one fp32 multiply (RN),
# cvt.rn to bf16 - bit for bit dequant_mxfp8_to_bf16 for every scale code except 0: a code-0 block has the descale 2^-127, an fp32
# subnormal that the six kernels carry into a bf16 subnormal and the Triton kernel flushes to zero (7 808 of the 2^16 (value, scale) codes
# differ, all at scale byte 0 - test_dequant_fused_sm120.py). The guard: at the first call of a layer, its scales are scanned once for
# code 0 next to a non-zero value byte (a CPU-side check on the device tensor, cached per layer); such a layer stays on the six kernels,
# so the fused path is bit-identical by construction on any weights (the served DeepSeek-V4.1 checkpoint carries codes 114-118 only).
# Reads 8.4 MB + 0.26 MB, writes 16.8 MB. _CACHE=1 keeps the bf16 weight per layer after the first call (no kernel at all; +16.8 MB per
# layer of device memory = 0.63 GiB per rank for 40 layers - a KV-budget decision).
# Runtime A/B: with VLLM_MOET_RUNTIME_AB=1 the key WOA_DEQUANT_FUSED (0 / 1) of the control file VLLM_MOET_RUNTIME_AB_FILE (default
# /dev/shm/vllm_moet_ab.json; one flat JSON object switch -> value, the same file every module reads for its own keys) overrides the
# env value from the next o_proj call on: polled at most once per 100 ms at the prefill branch's entry (a flip between two calls changes no
# bits - both paths are bit-identical on guarded weights), re-read only on an mtime / size change; a malformed or missing file or an invalid
# value keeps the state (one warning per cause), a dropped key goes back to the env value, one line per rank per change. Counters (calls /
# rows per path) print with the fast-MoE module's lines on `touch /dev/shm/vllm_moet_stats` (VLLM_MOET_RUNTIME_STATS_FILE, checked every 64
# calls) or every VLLM_MOET_WOA_DEQUANT_STATS_EVERY calls.
_WOA_DEQUANT_FUSED = os.environ.get("VLLM_MOET_WOA_DEQUANT_FUSED", "0") == "1"
_WOA_DEQUANT_CACHE = os.environ.get("VLLM_MOET_WOA_DEQUANT_CACHE", "0") == "1"
_dequant_kernel = None
_WOA_RT: dict = {"on": None, "fused": None, "next_poll": 0.0, "sig": None, "warned": set(), "calls": 0, "stats_mtime": None,
                 "stats_every": None, "n_fused": 0, "rows_fused": 0, "n_plain": 0, "rows_plain": 0, "n_guarded": 0}


def _woa_log(msg: str) -> None:
    try:
        dev = torch.cuda.current_device()
    except Exception:  # noqa: BLE001
        dev = "?"
    print(f"[vllm-moet sm120 wo_a, device {dev}] {msg}", file=sys.stderr, flush=True)


def _woa_warn_once(key: str, msg: str) -> None:
    if key not in _WOA_RT["warned"]:
        _WOA_RT["warned"].add(key)
        _woa_log("WARNING: " + msg)


def woa_dequant_fused_effective() -> bool:
    """The fused-dequant switch in effect: the runtime A/B override when set, else the env value."""
    f = _WOA_RT["fused"]
    return _WOA_DEQUANT_FUSED if f is None else bool(f)


def _woa_runtime_ab_poll() -> None:
    import time  # noqa: PLC0415

    now = time.monotonic()
    if now < _WOA_RT["next_poll"]:
        return
    _WOA_RT["next_poll"] = now + 0.1
    path = os.environ.get("VLLM_MOET_RUNTIME_AB_FILE", "/dev/shm/vllm_moet_ab.json")
    try:
        st = os.stat(path)
    except OSError:
        _woa_warn_once("missing", f"runtime A/B: control file {path} missing or unreadable - the current state is kept until it appears")
        return
    sig = (st.st_mtime_ns, st.st_size)
    if sig == _WOA_RT["sig"]:
        return
    _WOA_RT["sig"] = sig
    import json  # noqa: PLC0415

    try:
        with open(path, encoding="utf-8") as f:
            values = json.load(f)
        if not isinstance(values, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        _woa_warn_once("malformed", f"runtime A/B: control file {path} malformed ({exc}) - the current state is kept")
        return
    _WOA_RT["warned"].discard("missing")
    _WOA_RT["warned"].discard("malformed")
    old = woa_dequant_fused_effective()
    if "WOA_DEQUANT_FUSED" in values:
        try:
            v = int(values["WOA_DEQUANT_FUSED"])
        except (TypeError, ValueError):
            v = -1
        if v in (0, 1):
            _WOA_RT["fused"] = v
        else:
            _woa_warn_once("val", f"runtime A/B: WOA_DEQUANT_FUSED={values['WOA_DEQUANT_FUSED']!r} out of range (0 / 1) - kept {int(old)}")
    else:
        _WOA_RT["fused"] = None  # the key dropped -> the env value
    new = woa_dequant_fused_effective()
    if new != old:
        _woa_log(f"runtime A/B: WOA_DEQUANT_FUSED {int(old)} -> {int(new)}")


def woa_dequant_counters() -> dict:
    """calls / rows (weight rows) through the fused kernel and through the six-kernel path, and the layers the guard kept on the six kernels."""
    r = _WOA_RT
    return {"fused": (r["n_fused"], r["rows_fused"]), "six_kernels": (r["n_plain"], r["rows_plain"]), "guarded_layers": r["n_guarded"]}


def _woa_stats_line(reason: str) -> None:
    c = woa_dequant_counters()
    _woa_log(f"activation counters ({reason}; calls / weight rows): wo_a dequant fused {c['fused'][0]} / {c['fused'][1]}, six kernels "
             f"{c['six_kernels'][0]} / {c['six_kernels'][1]}, layers kept on the six kernels by the code-0 guard {c['guarded_layers']}; "
             f"switches now WOA_DEQUANT_FUSED={int(woa_dequant_fused_effective())} CACHE={int(_WOA_DEQUANT_CACHE)}")


def _woa_runtime_hooks() -> None:
    r = _WOA_RT
    if r["on"] is None:
        r["on"] = os.environ.get("VLLM_MOET_RUNTIME_AB", "0") == "1"
        try:
            r["stats_every"] = int(os.environ.get("VLLM_MOET_WOA_DEQUANT_STATS_EVERY", "0"))
        except ValueError:
            r["stats_every"] = 0
    if r["on"]:
        _woa_runtime_ab_poll()
    r["calls"] += 1
    n = r["calls"]
    if r["stats_every"] > 0 and n % r["stats_every"] == 0:
        _woa_stats_line(f"every {r['stats_every']} calls, call {n}")
    if n % 64 == 0:
        try:
            m = os.stat(os.environ.get("VLLM_MOET_RUNTIME_STATS_FILE", "/dev/shm/vllm_moet_stats")).st_mtime_ns
        except OSError:
            m = None
        if m is not None and m != r["stats_mtime"]:
            r["stats_mtime"] = m
            _woa_stats_line(f"on demand, call {n}")


def _fused_dequant_kernel():
    global _dequant_kernel
    if _dequant_kernel is None:
        import triton  # noqa: PLC0415
        import triton.language as tl  # noqa: PLC0415
        from triton.language.extra import libdevice  # noqa: PLC0415

        @triton.jit
        def _k(x_ptr, s_ptr, out_ptr, K, n_blocks, BLOCK_K: tl.constexpr):
            row = tl.program_id(0).to(tl.int64)
            offs = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(x_ptr + row * K + offs, mask=mask, other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            s = tl.load(s_ptr + row * n_blocks + offs // 32, mask=mask, other=127).to(tl.float32)
            d = libdevice.exp2(s - 127.0)
            tl.store(out_ptr + row * K + offs, (x * d).to(tl.bfloat16), mask=mask)

        _dequant_kernel = _k
    return _dequant_kernel


def dequant_mxfp8_to_bf16_fused(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """dequant_mxfp8_to_bf16 as one kernel (bit for bit for scale codes >= 1; see the note above). x: fp8 e4m3 [..., K] contiguous,
    scales: uint8 [..., K/32]."""
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    s2 = scales.reshape(-1, K // MXFP8_BLOCK_SIZE)
    assert x2.is_contiguous() and s2.is_contiguous() and s2.shape[0] == x2.shape[0]
    out = torch.empty(x2.shape, dtype=torch.bfloat16, device=x.device)
    block = 1024 if K % 1024 == 0 else 512 if K % 512 == 0 else 256
    grid = (x2.shape[0], triton_cdiv(K, block))
    _fused_dequant_kernel()[grid](x2.view(torch.uint8), s2.view(torch.uint8), out, K, K // MXFP8_BLOCK_SIZE, BLOCK_K=block)
    return out.view(x.shape)


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def fused_dequant_safe(x: torch.Tensor, scales: torch.Tensor) -> bool:
    """The code-0 guard: True when no 32-block has scale code 0 together with a non-zero value byte (then the fused kernel equals the
    six kernels bit for bit). One pass over the scales (and, only if a code 0 exists, over its blocks' value bytes)."""
    s2 = scales.reshape(-1, scales.shape[-1]).view(torch.uint8)
    zero = s2 == 0
    if not bool(zero.any()):
        return True
    K = x.shape[-1]
    vb = x.reshape(-1, K // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE).view(torch.uint8) & 0x7F  # +-0 = byte 0x00 / 0x80
    nonzero_block = (vb != 0).any(dim=2)
    return not bool((zero.reshape(nonzero_block.shape) & nonzero_block).any())


def dequant_mxfp8_to_bf16_guarded(x: torch.Tensor, scales: torch.Tensor, cache_key=None) -> torch.Tensor:
    """The fused kernel when the guard allows it, vLLM's six kernels otherwise; the guard's verdict is cached per `cache_key` (a layer)."""
    if cache_key is not None and hasattr(cache_key, "_moet_woa_fused_safe"):
        safe = cache_key._moet_woa_fused_safe
    else:
        safe = fused_dequant_safe(x, scales)
        if cache_key is not None:
            cache_key._moet_woa_fused_safe = safe
            if not safe:
                _WOA_RT["n_guarded"] += 1
                _woa_log("wo_a dequant: a 32-block with scale code 0 and a non-zero value in this layer's weight - the fused kernel is not "
                         "bit-identical there; this layer stays on the six-kernel path")
    return dequant_mxfp8_to_bf16_fused(x, scales) if safe else dequant_mxfp8_to_bf16(x, scales)


def _woa_dequant(wo_a: nn.Module) -> torch.Tensor:
    """The bf16 wo_a weight of the prefill branch: cached per layer (_CACHE=1), fused with the code-0 guard (_FUSED=1 or the runtime A/B
    key), or today's six-kernel chain."""
    _woa_runtime_hooks()
    rows = int(wo_a.weight.shape[0]) if wo_a.weight.dim() >= 1 else 0
    if _WOA_DEQUANT_CACHE:
        w = getattr(wo_a, "_moet_woa_bf16", None)
        if w is None:
            w = (dequant_mxfp8_to_bf16_guarded(wo_a.weight, wo_a.weight_scale, wo_a) if woa_dequant_fused_effective()
                 else dequant_mxfp8_to_bf16(wo_a.weight, wo_a.weight_scale))
            wo_a._moet_woa_bf16 = w
        return w
    if woa_dequant_fused_effective():
        w = dequant_mxfp8_to_bf16_guarded(wo_a.weight, wo_a.weight_scale, wo_a)
        if getattr(wo_a, "_moet_woa_fused_safe", True):
            _WOA_RT["n_fused"] += 1; _WOA_RT["rows_fused"] += rows
        else:
            _WOA_RT["n_plain"] += 1; _WOA_RT["rows_plain"] += rows
        return w
    _WOA_RT["n_plain"] += 1; _WOA_RT["rows_plain"] += rows
    return dequant_mxfp8_to_bf16(wo_a.weight, wo_a.weight_scale)


def _gemv_module():
    gemv_dir = os.environ.get("VLLM_MOET_SM120_GEMV_DIR", DEFAULT_GEMV_DIR)
    if gemv_dir not in sys.path:
        sys.path.insert(0, gemv_dir)
    import mxfp8_gemv_sm120  # noqa: PLC0415

    return mxfp8_gemv_sm120


class Sm120GemvMxfp8BmmLinearKernel(Mxfp8LinearKernel):
    """Grouped MXFP8 GEMV (tensor cores) for wo_a on sm_120; bf16 bmm above 64 tokens."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if os.environ.get("VLLM_MOET_SM120_GEMV_BMM", "1") != "1":
            return False, "disabled by VLLM_MOET_SM120_GEMV_BMM=0"
        if not current_platform.is_cuda():
            return False, "CUDA only"
        if not current_platform.is_device_capability_family(120):
            return False, "sm_120 only"
        try:
            _gemv_module()
        except Exception as exc:  # noqa: BLE001
            return False, f"vllm-moet sm_120 GEMV extension unavailable: {exc!r}"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if c.bmm_batch_size is None or c.bmm_batch_size <= 0:
            return False, "grouped GEMV requires a positive bmm_batch_size"
        return True, None

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        weight = layer.weight.data
        if weight.ndim != 2:
            raise ValueError(f"expected a 2D MXFP8 weight, got {tuple(weight.shape)}")
        N, K = weight.shape
        groups = self.config.bmm_batch_size
        assert groups is not None
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"expected float8_e4m3fn weight, got {weight.dtype}")
        if K % 128 != 0 or N % groups != 0:
            raise ValueError(f"unsupported wo_a shape N={N} K={K} groups={groups}")
        scale = layer.weight_scale.data[:N, : K // _QUANT_GROUP].contiguous()
        if scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(f"expected {MXFP8_SCALE_DTYPE} weight_scale, got {scale.dtype}")
        replace_parameter(layer, "weight", weight.contiguous())
        replace_parameter(layer, "weight_scale", scale)
        layer.weight_block_size = [1, _QUANT_GROUP]
        # Build/load the extension now (precompiled in the image) instead of inside
        # the first forward, and mark the layer for the o-projection dispatch.
        self._mod = _gemv_module()
        self._mod._ext()
        layer.sm120_gemv_bmm = self
        logger.info_once(
            "vllm-moet: wo_a stays MXFP8 on sm_120 (grouped tensor-core GEMV, %d groups)",
            groups,
        )

    # -- generic entry (not used by the DeepSeek-V4 o-projection, kept for completeness)
    def apply_weights(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            raise NotImplementedError("pre-quantized input is only supported via o_proj()")
        w = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale).to(x.dtype)
        groups = self.config.bmm_batch_size
        assert groups is not None
        if x.ndim == 3:
            tokens = x.shape[0]
            w3 = w.view(groups, -1, w.shape[-1])
            out = torch.bmm(x.transpose(0, 1), w3.transpose(1, 2)).transpose(0, 1)
            out = out.reshape(tokens, -1)
        else:
            out = torch.nn.functional.linear(x, w)
        if bias is not None:
            out = out + bias
        return out

    # -- DeepSeek-V4 o-projection (called from the patched deep_gemm_fp8_o_proj)
    def o_proj(
        self,
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        wo_a: nn.Module,
        wo_b: nn.Module,
        *,
        n_groups: int,
        heads_per_group: int,
        nope_dim: int,
        rope_dim: int,
        o_lora_rank: int,
    ) -> torch.Tensor:
        from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (  # noqa: PLC0415
            fused_inv_rope_fp8_quant,
        )

        tokens = o.shape[0]
        if tokens <= self._mod.MAX_GROUPED_GEMV_T:
            q, sf = fused_inv_rope_fp8_quant(
                o,
                positions,
                cos_sin_cache,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                quant_group_size=_QUANT_GROUP,
                tma_aligned_scales=True,
                quantize=True,
            )
            # q/sf are [T, G, *] views of group-major buffers; hand the buffers over.
            z = self._mod.mxfp8_gemv_grouped(
                q.transpose(0, 1), sf.transpose(0, 1), wo_a.weight, wo_a.weight_scale
            )
        else:
            x, _ = fused_inv_rope_fp8_quant(
                o,
                positions,
                cos_sin_cache,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                quant_group_size=_QUANT_GROUP,
                tma_aligned_scales=True,
                quantize=False,
            )
            w = _woa_dequant(wo_a)  # VLLM_MOET_WOA_DEQUANT_FUSED (+ the runtime A/B key) / _CACHE; default: dequant_mxfp8_to_bf16 as before
            grouped_weight = w.view(n_groups, o_lora_rank, -1)
            z = torch.empty(
                (tokens, n_groups, o_lora_rank), device=o.device, dtype=torch.bfloat16
            )
            # K-9 (sp_pipeline shared replay tail): inside the pipelined prefill's shared tail one bmm per group row block -
            # cuBLAS picks its kernel per (batch, M, N, K), so the bits of a 256-token call are not the 128-token call's
            for lo, hi in _row_blocks(tokens):
                torch.bmm(
                    x.transpose(0, 1)[:, lo:hi],
                    grouped_weight.transpose(1, 2),
                    out=z.transpose(0, 1)[:, lo:hi],
                )
        return wo_b(z.flatten(1))
