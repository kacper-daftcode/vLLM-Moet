"""The MoE input of a sequence-parallel prefill step all-gathered in MXFP8 instead of bf16 (DeepSeek-V4.1-Flash, sm_120, TP4).

Boundary 2 of the eager prefill step (patch_vllm_prefill_sp_sm120.py) reduce-scatters the attention output, runs the mHC
boundary on this rank's row shard and today all-gathers the normed MoE input in bf16 ([T, 5120] = 10 240 B/row, 738 us per
4096-row chunk on the PCIe ring). Its three consumers (../sp_moe_gather/REPORT in /workspace/tmp/par31/U, section 1) are
row-local in a way that lets every rank finish its own rows first and gather the compact results:

  routed experts  b12x's dynamic path quantizes the rows to MXFP8 (e4m3 + UE8M0 per 32 columns) in its front-end; our fast
                  kernels (step 42) do the same in quant_row. The quantizer is per row and per 32-column block, and it is the
                  recipe FlashInfer's mxfp8 quantizer applies (amax * fp32(1/448), exponent rounded up unless the mantissa is 0,
                  2^(127 - sf), cvt.rn.satfinite e4m3): quantize on the shard with FlashInfer's kernel, all-gather the bytes
                  and scales (5 120 + 160 B/row), hand the fast kernels the rows pre-quantized (their frontend skips its quant).
  shared expert   its first GEMM (MXFP8 CUTLASS) quantizes the same bf16 input with the same FlashInfer kernel, scales in the
                  F8_128x4 swizzled layout; step 38's consumer hook takes a pre-quantized pair hung on the input tensor as
                  `_moet_mxfp8`. The gathered scales are re-laid to the swizzled layout (a permutation of bytes).
  router          GateLinear's cuBLAS tier, torch.mm(x, W.T, out_dtype=fp32) on all T rows; cuBLAS picks its kernel and split-K
                  by M, so the shard's rows are computed with the same M: a [T, 5120] bf16 buffer holds this rank's rows at
                  their positions (the other rows are whatever the buffer held: finite, never read), the GEMM runs on it with
                  M = T (the kernel the full-row path picks), the shard's rows of the result are taken. The top-6 (one program
                  per row: sqrt-softplus + bias, 6 arg-max, renormalize, padding and image-sentinel rows) runs on the shard
                  through the layer's own router (the padding mask sliced to the shard) and ids + weights (48 B/row) are
                  gathered with the scales in one small all-gather.

Bytes per row: 10 240 -> 5 120 (e4m3) + 208 (scales 160 + ids 24 + weights 24) = 0.52x. Two NCCL all-gathers instead of one.
Bits: every step is the served arithmetic on the same rows (test_prefill_sp_gather_tp4.py: EQUAL at T = 4096 / 3063 / 2048 / 1024).

The MoE call then receives the [T, 5120] bf16 buffer as a stand-in (every consumer reads its shape / dtype / device only) and
three hooks (patch_vllm_prefill_sp_gather_sm120.py) take the gathered pieces off a side channel keyed by the stand-in's data
pointer and shape: GateLinear.forward (the logits), MoERunner._apply_quant_method (the top-k), the fast prefill MoE launch
in b12x's _launch_dynamic (the pre-quantized rows). `finish()` asserts that all three took their piece: a consumer that
silently read the stand-in would be a wrong result, so the check is loud.

Switch: VLLM_MOET_PREFILL_SP_GATHER=mxfp8 (default bf16 = today's gather). Used only when the layer is eligible (b12x routed
experts behind the fast kernels with the pre-quantized-rows entry, a FusedTopKBiasRouter without EPLB / fused shared slots,
GateLinear's cuBLAS tier, a MXFP8 CUTLASS shared expert with step 38's hook) - decided once per layer, identically on every rank.
"""
from __future__ import annotations

import os
from typing import Any

import torch

__all__ = [
    "quant_rows", "swizzle_scales", "unswizzle_scales", "router_logits_same_kernel", "pack_small", "unpack_small",
    "gather_moe_input", "finish", "take_router_logits", "take_topk", "take_prequant", "eligible",
]

SCALE_BLOCK = 32


# ------------------------------------------------------------------------------------------------------ pure pieces
def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer's MXFP8 quantizer on bf16 rows: e4m3 bytes (uint8 view, [n, K]) and UE8M0 scales row-major [n, K/32]."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    return q.view(torch.uint8), s.view(x.shape[0], x.shape[1] // SCALE_BLOCK)


def swizzle_scales(s_rm: torch.Tensor) -> torch.Tensor:
    """Row-major UE8M0 scales [T, K/32] -> FlashInfer's F8_128x4 swizzled layout (uint8 [ceil(T/128)*128 * K/32]):
    offset(row, kc) = (((row//128 * nkt + kc//4) * 32 + row%32) * 4 + row%128//32) * 4 + kc%4, nkt = K/128."""
    t, nkc = s_rm.shape
    assert nkc % 4 == 0, "K must be a multiple of 128"
    tp = (t + 127) // 128 * 128
    if tp != t:
        s_rm = torch.nn.functional.pad(s_rm, (0, 0, 0, tp - t))
    return s_rm.view(tp // 128, 4, 32, nkc // 4, 4).permute(0, 3, 2, 1, 4).contiguous().view(-1)


def unswizzle_scales(s_sw: torch.Tensor, t: int, k: int) -> torch.Tensor:
    """The inverse of swizzle_scales: [T, K/32] row-major."""
    nkc = k // SCALE_BLOCK
    tp = (t + 127) // 128 * 128
    return s_sw.view(tp // 128, nkc // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).contiguous().view(tp, nkc)[:t]


def router_logits_same_kernel(x_rows: torch.Tensor, lo: int, weight: torch.Tensor, t: int,
                              buf: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.mm(x, W.T, out_dtype=fp32) for rows [lo, lo + n) of a T-row input of which only those rows are at hand, with
    the kernel cuBLAS picks for M = T: the rows are placed into a [T, K] bf16 buffer and the GEMM runs on the whole buffer.
    Returns (the [T, N] logits - rows outside [lo, lo + n) are meaningless -, the buffer)."""
    n, k = x_rows.shape
    if buf is None:
        buf = torch.empty((t, k), dtype=x_rows.dtype, device=x_rows.device)
    buf[lo:lo + n].copy_(x_rows)
    return torch.mm(buf, weight.t(), out_dtype=torch.float32), buf


def pack_small(s_rm: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, rows: int) -> torch.Tensor:
    """[rows, K/32 + 24 + 24] uint8: the scales, the int32 top-6 ids and the fp32 weights of n <= rows rows (pad rows 0)."""
    n = ids.shape[0]
    ids_u8 = ids.contiguous().view(torch.uint8).view(n, -1)
    w_u8 = weights.contiguous().view(torch.uint8).view(n, -1)
    if n < rows:
        ids_u8 = torch.nn.functional.pad(ids_u8, (0, 0, 0, rows - n))
        w_u8 = torch.nn.functional.pad(w_u8, (0, 0, 0, rows - n))
    return torch.cat([s_rm, ids_u8, w_u8], dim=1)


def unpack_small(small: torch.Tensor, k: int, topk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The gathered small rows -> (scales [T, K/32] uint8, ids [T, topk] int32, weights [T, topk] fp32), each contiguous."""
    nkc = k // SCALE_BLOCK
    t = small.shape[0]
    s_rm = small[:, :nkc].contiguous()
    ids = small[:, nkc:nkc + 4 * topk].contiguous().view(torch.int32).view(t, topk)
    w = small[:, nkc + 4 * topk:nkc + 8 * topk].contiguous().view(torch.float32).view(t, topk)
    return s_rm, ids, w


# ------------------------------------------------------------------------------------------------- the runtime path
def _one_all_gather() -> bool:
    """VLLM_MOET_PREFILL_SP_GATHER_ONE_AG=1: one all-gather of [c, K + K/32 + 8 topk] rows instead of two (needs the fast prefill
    MoE build with strided_rows_supported(); checked once in _check_eligible). Default 0 = the two gathers of step 45.
    Measured in situ (session K, test host 4x RTX PRO 6000 SE, image r25, 6.11): bits identical (long_agree x2 + agree_client ALL
    IDENTICAL vs the two-gather arm), fresh prefill 163K 6973 / 6987 / 6975 ms vs 6946 / 6949 / 6947 (+0.4 %), 19K 826 / 824 / 822 vs
    818 / 819 / 821 - the shared expert's CUTLASS GEMM still needs a contiguous e4m3 operand and that 21 MB copy costs more than the
    second NCCL launch it saves; it pays only once that consumer takes the operand with a row stride. Kept opt-in for that follow-up."""
    global _ONE_AG
    if _ONE_AG is None:
        _ONE_AG = os.environ.get("VLLM_MOET_PREFILL_SP_GATHER_ONE_AG", "0") == "1"
    return _ONE_AG


_ONE_AG: bool | None = None
_CTX: dict[tuple, dict[str, Any]] = {}  # (stand-in data_ptr, shape) -> the gathered pieces + consumption flags
_ELIGIBLE: dict[int, bool] = {}  # id(ffn) -> decided once
_STATS = {"calls": 0, "bf16_fallback": 0}


def _log():
    from vllm.logger import init_logger

    return init_logger("vllm.vllm_moet.sp_moe_gather")


def _fast_module():
    """The fast prefill MoE loader module, if step 42's hook has it active (it decides once per process)."""
    try:
        import b12x.moe.fused_moe._impl as impl  # the served b12x package (flashinfer.experimental.b12x is aliased to it)
    except Exception:  # noqa: BLE001
        try:
            import flashinfer.experimental.b12x.moe.fused_moe._impl as impl
        except Exception:  # noqa: BLE001
            return None
    fn = getattr(impl, "_moet_prefill_fast_module", None)
    if fn is None or not getattr(impl, "_MOET_SP_GATHER_PREQUANT_HOOK", False):
        return None
    module = fn()
    return module or None


def eligible(ffn) -> bool:
    """Whether this MoE layer can take the MXFP8 gather (static properties: the same answer on every rank)."""
    key = id(ffn)
    hit = _ELIGIBLE.get(key)
    if hit is not None:
        return hit
    ok, why = _check_eligible(ffn)
    if not ok:
        _log().warning("vllm-moet sm_120 prefill SP gather=mxfp8: layer %s stays on the bf16 gather: %s",
                       getattr(ffn, "prefix", "?"), why)
    _ELIGIBLE[key] = ok
    return ok


def _check_eligible(ffn) -> tuple[bool, str]:
    try:
        if getattr(ffn, "use_native_mega_moe", False):
            return False, "native mega MoE"
        gate = ffn.gate
        if gate.tid2eid is not None:
            return False, "hash routing"
        if not getattr(gate, "allow_cublas_router_gemm", False) or gate.weight.dtype != torch.bfloat16 \
                or gate.out_dtype != torch.float32:
            return False, "the router is not GateLinear's cuBLAS bf16 -> fp32 tier"
        runner = ffn.experts
        if runner.gate is not gate or getattr(runner, "_fse_fuse_gate", False):
            return False, "the runner does not hold the plain gate"
        router = runner.router
        if type(router).__name__ != "FusedTopKBiasRouter" or getattr(router, "num_fused_shared_experts", 0) \
                or router.eplb_state is not None or getattr(router, "capture_fn", None) is not None:
            return False, f"router {type(router).__name__} (EPLB / fused shared slots / capture)"
        experts = runner._quant_method.moe_kernel.fused_experts
        if type(experts).__name__ != "B12xExperts":
            return False, f"routed experts {type(experts).__name__} (b12x only)"
        if runner._shared_experts is None or ffn.shared_experts is None:
            return False, "no shared expert"
        qm = ffn.shared_experts.gate_up_proj.quant_method
        kernel = getattr(qm, "fp8_linear", None) or getattr(qm, "kernel", None)
        if type(kernel).__name__ != "FlashInferCutlassMxfp8LinearKernel":
            return False, f"shared expert kernel {type(kernel).__name__} (MXFP8 CUTLASS with step 38's hook only)"
        if "_moet_mxfp8" not in type(kernel).apply_weights.__code__.co_consts:
            return False, "the shared expert's kernel has no step-38 pre-quantized input hook"
        import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as ftb
        import vllm.model_executor.layers.fused_moe.runner.moe_runner as mr
        import vllm.model_executor.layers.fused_moe.router.gate_linear as gl

        for mod, flag in ((ftb, "_MOET_SP_GATHER_PADDING"), (mr, "_MOET_SP_GATHER_TOPK_HOOK"), (gl, "_MOET_SP_GATHER_ROUTER_HOOK")):
            if not hasattr(mod, flag):
                return False, f"{mod.__name__} is not patched for the MXFP8 gather"
        fast = _fast_module()
        if fast is None:
            return False, "the fast prefill MoE kernels are not active (VLLM_MOET_B12X_MOE_PREFILL_FAST=1 + step 42's hook)"
        if not fast.prequant_rows_supported():
            return False, "the fast prefill MoE extension has no pre-quantized rows entry"
        if _one_all_gather() and not getattr(fast, "strided_rows_supported", lambda: False)():
            return False, "VLLM_MOET_PREFILL_SP_GATHER_ONE_AG=1 needs the fast prefill MoE build with strided a_values / a_scales"
        if not fast.supported(n=ffn.moe_intermediate_size // ffn.tp_size, k=ffn.hidden_size, num_topk=ffn.n_activated_experts,
                              experts=ffn.n_routed_experts, ids_dtype=torch.int32):
            return False, "the fast prefill MoE kernels do not serve this MoE shape"
        if runner._quant_method.topk_indices_dtype not in (torch.int32, None):
            return False, f"top-k ids dtype {runner._quant_method.topk_indices_dtype}"
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"eligibility check failed: {exc!r}"


def gather_moe_input(ffn, x_shard: torch.Tensor, num_tokens: int, input_ids: torch.Tensor | None) -> torch.Tensor | None:
    """The MXFP8 gather of boundary 2 for an eligible layer: returns the [T, 5120] bf16 stand-in to hand to ffn(...),
    or None (the caller then all-gathers bf16 as today)."""
    if not eligible(ffn):
        _STATS["bf16_fallback"] += 1
        return None
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.distributed import tensor_model_parallel_all_gather as all_gather
    from vllm.forward_context import get_forward_context, is_forward_context_available
    import vllm.envs as envs
    import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as ftb

    tp_rank, tp_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    c, k = x_shard.shape  # ceil(T / TP) rows (sp_reduce_scatter pads T to a multiple of TP)
    lo = tp_rank * c
    n = max(0, min(c, num_tokens - lo))  # this rank's real rows
    runner = ffn.experts
    topk = ffn.n_activated_experts

    # 1. MXFP8 rows of the shard (the pad rows too: dropped after the gather)
    q, s_rm = quant_rows(x_shard)

    # 2. router logits with the full-row kernel; the stand-in buffer is the GEMM input
    logits, buf = router_logits_same_kernel(x_shard[:n], lo, ffn.gate.weight, num_tokens)

    # 3. the top-6 of this rank's rows through the layer's router (its padding mask sliced to the shard)
    is_padding = None
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        mask = get_forward_context().is_padding
        if mask is not None:
            is_padding = mask[lo:lo + n]
    ftb._MOET_SP_GATHER_PADDING = (True, is_padding)
    try:
        topk_weights, topk_ids = runner.router.select_experts(
            hidden_states=buf[lo:lo + n],
            router_logits=logits[lo:lo + n],
            topk_indices_dtype=runner._quant_method.topk_indices_dtype,
            input_ids=None if input_ids is None else input_ids[lo:lo + n],
        )
    finally:
        ftb._MOET_SP_GATHER_PADDING = (False, None)
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)

    # 4. the all-gathers: the e4m3 bytes, and the small rows (scales + ids + weights) - two gathers (default), or ONE gather of
    #    [c, K + K/32 + 8 topk] rows (VLLM_MOET_PREFILL_SP_GATHER_ONE_AG=1; the fast prefill MoE kernels read a_values / a_scales
    #    as row-strided views of the gathered buffer since K's commit of 5.11 - the same bytes, one NCCL launch less per layer;
    #    the shared expert's CUTLASS operand is the one consumer that still needs a contiguous [T, K] copy)
    small = pack_small(s_rm, topk_ids, topk_weights, c)
    if _one_all_gather():
        full = all_gather(torch.cat([q, small], dim=1), 0)[:num_tokens]  # [T, K + 208], row stride K + 208 (a multiple of 4)
        q_full, small_full = full[:, :k], full[:, k:]
        q_shared = q_full.contiguous()
    else:
        q_full = all_gather(q, 0)[:num_tokens]
        small_full = all_gather(small, 0)[:num_tokens]
        q_shared = q_full
    s_full, ids_full, w_full = unpack_small(small_full, k, topk)
    if _one_all_gather():
        s_full = small_full[:, :k // SCALE_BLOCK]  # the strided view for the fast kernels (unpack_small's copy serves the swizzle)

    # 5. the consumers' views: the shared expert's pair on the stand-in (step 38's hook), the rest on the side channel
    buf._moet_mxfp8 = (q_shared.view(torch.float8_e4m3fn), swizzle_scales(s_full.contiguous() if _one_all_gather() else s_full))
    key = (buf.data_ptr(), tuple(buf.shape))
    assert not _CTX, "prefill SP gather: a previous MoE call did not finish()"
    _CTX[key] = dict(router=logits, topk=(w_full, ids_full), prequant=(q_full, s_full),
                     taken={"router": False, "topk": False, "prequant": False})
    _STATS["calls"] += 1
    if _STATS["calls"] == 1:
        _log().info("vllm-moet sm_120 prefill SP gather=mxfp8: the MoE input is all-gathered as e4m3 + UE8M0 scales + the "
                    "shard's top-%d (%d + %d B/row instead of %d); the router GEMM runs on the stand-in with M = T",
                    topk, k, small.shape[1], 2 * k)
    return buf


def finish(standin: torch.Tensor) -> None:
    """After ffn(standin): every consumer must have taken its piece off the side channel."""
    key = (standin.data_ptr(), tuple(standin.shape))
    entry = _CTX.pop(key, None)
    if entry is None:
        return
    missing = [name for name, taken in entry["taken"].items() if not taken]
    if missing:
        raise RuntimeError(
            f"prefill SP gather=mxfp8: the MoE consumers {missing} did not take the gathered pieces - the stand-in's bytes "
            "would have been read; the layer is not eligible for the MXFP8 gather (set VLLM_MOET_PREFILL_SP_GATHER=bf16)")


def _take(x: torch.Tensor, what: str):
    if not _CTX:
        return None
    entry = _CTX.get((x.data_ptr(), tuple(x.shape)))
    if entry is None:
        return None
    entry["taken"][what] = True
    return entry[what]


def take_router_logits(x: torch.Tensor):
    """GateLinear.forward(x): the logits of the stand-in (the shard's rows are the real ones; nobody reads the others)."""
    return _take(x, "router")


def take_topk(hidden_states: torch.Tensor):
    """MoERunner._apply_quant_method: (topk_weights [T, 6] fp32, topk_ids [T, 6] int32) gathered from the shards."""
    return _take(hidden_states, "topk")


def take_prequant(a: torch.Tensor):
    """The fast prefill MoE launch: (a_values uint8 [T, K], a_scales uint8 [T, K/32]) gathered from the shards."""
    return _take(a, "prequant")


def stats() -> dict:
    return dict(_STATS)
