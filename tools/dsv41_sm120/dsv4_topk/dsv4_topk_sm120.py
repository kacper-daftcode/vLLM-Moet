"""JIT loader for dsv4_topk_sm120.cu: DeepSeek-V4 routing top-6, bit-identical to vLLM's Triton dsv4_topk.

    from dsv4_topk_sm120 import dsv4_topk_cuda
    w, ids = dsv4_topk_cuda(gating_output, correction_bias, indices_dtype, routed_scaling_factor,
                            is_padding=None, input_ids=None, bias_vl=None, image_sentinel_lo=0, pdl=True)

Same signature and results as vllm.model_executor.layers.fused_moe.router.dsv4_topk.dsv4_topk for fp32 logits with
256 or 384 experts. VLLM_MOET_DSV4_TOPK_PDL=1 launches with programmatic stream serialization; the default is a plain
launch: in the served decode graph the PDL launch ends the top-k ~3.5 us later (see the README).

The first import compiles the extension with torch.utils.cpp_extension.load (~30 s, cached under
$TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions; the serving image compiles it at build time).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_DSV4_TOPK_ARCH", "12.0")
    # libdevice logf with flush-to-zero (Triton's reflect); no fast-math (logf must stay libdevice's)
    return load(
        name="vllm_moet_dsv4_topk_sm120",
        sources=[str(_HERE / "dsv4_topk_sm120.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20", "-ftz=true"],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def dsv4_topk_cuda(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    indices_dtype: torch.dtype,
    routed_scaling_factor: float,
    is_padding: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
    pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = gating_output.shape[0]
    w = gating_output.new_empty((num_tokens, 6), dtype=torch.float32)
    ids = gating_output.new_empty((num_tokens, 6), dtype=indices_dtype)
    if pdl is None:
        pdl = os.environ.get("VLLM_MOET_DSV4_TOPK_PDL", "0") == "1"
    has_vl = bias_vl is not None and image_sentinel_lo > 0
    # step 35: the router GEMM that produced these logits may be pending in the BF16 GEMM module - run both in one launch;
    # step 37 (VLLM_MOET_MOE_MEGA=1): neither runs here - the MoE launch computes the logits and the top-6 itself
    rec = _pending_router_gemm(gating_output)
    if rec is not None and moe_mega_enabled() and indices_dtype == torch.int32:
        _defer_routing(rec, gating_output, correction_bias, is_padding, w, ids, float(routed_scaling_factor),
                       input_ids if has_vl else None, bias_vl if has_vl else None, int(image_sentinel_lo) if has_vl else 0)
        return w, ids
    if rec is not None:
        _BF16.run_pending_router_topk(
            rec, gating_output, correction_bias, is_padding, w, ids, float(routed_scaling_factor),
            input_ids if has_vl else None, bias_vl if has_vl else None, int(image_sentinel_lo) if has_vl else 0,
        )
        return w, ids
    _ext().dsv4_topk_out(
        gating_output, correction_bias, is_padding, w, ids, float(routed_scaling_factor),
        input_ids if has_vl else None, bias_vl if has_vl else None, int(image_sentinel_lo) if has_vl else 0, bool(pdl),
    )
    return w, ids


_BF16 = None  # the BF16 GEMM module (bf16_replica_gemm_sm120), resolved once; False when unavailable


def _pending_router_gemm(gating_output: torch.Tensor):
    """The deferred router spread launch whose output is `gating_output` (step 35), or None."""
    global _BF16
    if _BF16 is None:
        _BF16 = False
        if os.environ.get("VLLM_MOET_ROUTER_TOPK_FUSED", "0") == "1" or os.environ.get("VLLM_MOET_MOE_MEGA", "0") == "1":
            import sys

            d = os.environ.get("VLLM_MOET_BF16_GEMM_DIR", "/opt/vllm-moet/dsv41_sm120/bf16_gemm")
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                import bf16_replica_gemm_sm120 as _m

                _BF16 = _m
            except Exception:  # noqa: BLE001
                _BF16 = False
    if not _BF16:
        return None
    return _BF16.take_pending_router(gating_output)


# ---- step 37: the routing deferred into the MoE launch (b12x_moe_mega) -----------------------------------------------
_MOE_MEGA: bool | None = None
_PENDING_ROUTING: dict[int, dict] = {}  # ids.data_ptr() -> routing record (the MoE launch takes it)
_PENDING_ROUTING_LATE = [0]


def moe_mega_enabled() -> bool:
    """VLLM_MOET_MOE_MEGA=1 and the MoE launch site able to take the routing (step 37's patch of w4a8_compact_micro.py):
    deferring the routing without a consumer would hand the MoE unwritten ids."""
    global _MOE_MEGA
    if _MOE_MEGA is None:
        _MOE_MEGA = False
        if os.environ.get("VLLM_MOET_MOE_MEGA", "0") == "1":
            try:
                from b12x.moe._shared.kernels import w4a8_compact_micro as _cm

                _MOE_MEGA = hasattr(_cm, "_moet_mega_take")
            except Exception:  # noqa: BLE001
                _MOE_MEGA = False
            if not _MOE_MEGA:
                import logging

                logging.getLogger(__name__).warning(
                    "vllm-moet step 37: VLLM_MOET_MOE_MEGA=1 but the MoE launch site is not patched (image built without "
                    "VLLM_MOET_MOE_MEGA_PATCH=1); the routing runs as separate launches"
                )
    return _MOE_MEGA


def _defer_routing(rec, logits, bias, is_padding, out_w, out_ids, scale, input_ids, bias_vl, image_lo) -> None:
    """Record the router GEMM + top-6 for the MoE launch; `out_w` / `out_ids` stay unwritten until it runs."""
    if _PENDING_ROUTING:
        run_pending_routing_late()
    x, w, out, case, slices, ws, cnt, pdl = rec
    assert out.data_ptr() == logits.data_ptr()
    _PENDING_ROUTING[out_ids.data_ptr()] = dict(
        x=x, w=w, logits=logits, step=int(case.step), bias=bias, scale=float(scale), is_padding=is_padding,
        input_ids=input_ids, bias_vl=bias_vl, image_lo=int(image_lo), w_out=out_w, ids_out=out_ids,
        fallback=(rec, logits, bias, is_padding, out_w, out_ids, scale, input_ids, bias_vl, image_lo),
    )


def take_pending_routing(ids: torch.Tensor):
    """The deferred routing whose ids are `ids`, or None. Used by the step-27 launch site (w4a8_compact_micro)."""
    return _PENDING_ROUTING.pop(ids.data_ptr(), None)


def run_routing_now(record) -> None:
    """A taken routing the MoE launch cannot fuse: step 35's spread + top-6 launch, before anything reads the ids."""
    _BF16.run_pending_router_topk(*record["fallback"])


def run_pending_routing_late() -> None:
    """Deferred routings nobody took: run them now (their reader, if any, saw stale ids - reported once)."""
    while _PENDING_ROUTING:
        _, record = _PENDING_ROUTING.popitem()
        run_routing_now(record)
        _PENDING_ROUTING_LATE[0] += 1
        if _PENDING_ROUTING_LATE[0] == 1:
            import logging

            logging.getLogger(__name__).warning(
                "vllm-moet step 37: a deferred routing was not consumed by the MoE launch; launched late"
            )


def supported(gating_output: torch.Tensor, indices_dtype: torch.dtype) -> bool:
    return (
        gating_output.dtype == torch.float32
        and gating_output.dim() == 2
        and gating_output.shape[1] in (256, 384)
        and gating_output.is_contiguous()
        and indices_dtype in (torch.int32, torch.int64)
        and torch.cuda.get_device_capability(gating_output.device)[0] == 12
    )
