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
            w = dequant_mxfp8_to_bf16(wo_a.weight, wo_a.weight_scale)
            grouped_weight = w.view(n_groups, o_lora_rank, -1)
            z = torch.empty(
                (tokens, n_groups, o_lora_rank), device=o.device, dtype=torch.bfloat16
            )
            torch.bmm(
                x.transpose(0, 1), grouped_weight.transpose(1, 2), out=z.transpose(0, 1)
            )
        return wo_b(z.flatten(1))
