#!/usr/bin/env python3
"""Step 33: the shared-expert add fused into b12x's top-k sum of the decode MoE (bit-identical).

Every MoE layer of the decode step ends with b12x's W4A16TopKSumKernel (the routed rows of a token summed in route
order: fp32 adds of the bf16 route values, one bf16 rounding) followed by vLLM's `shared_output + fused_output`
(torch: bf16(float(a) + float(b))) in model_executor/layers/fused_moe/runner/moe_runner.py - 40 x 1.2 us of
elementwise add plus a graph node per layer on the critical path, right before the sublayer's all-reduce.
`_moet_topksum_add_kernel` (Triton, in b12x's _impl.py next to the compact-micro top-k sum) computes the same sum with
the same operations and order, rounds it to bf16 as the original kernel does, then adds the shared-expert row the way
torch does and stores once: every output element bit-identical to the two kernels.

The shared experts run on vLLM's aux stream (SharedExperts.maybe_forward_async); the fused launch waits on their
output-ready event exactly where the runner would have waited before the add, and the runner's add is skipped for that
layer. Only the compact-micro W4A8 path (the served DeepSeek-V4.1 decode MoE), only when the runner applies no routed
scaling, no padding trim and no output transform and the routed output is not already reduced; anything else keeps the
original launches. Runtime switch VLLM_MOET_MOE_TOPKSUM_ADD=1 (default off).

Usage: python3 patch_vllm_moe_topksum_add_sm120.py [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages")
IMPL = SITE / "flashinfer/experimental/b12x/moe/fused_moe/_impl.py"
RUNNER = SITE / "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
MARKER = "[vllm-moet] step 33 shared-expert add fused into the top-k sum"

# ---- b12x _impl.py: the compact-micro top-k sum site -----------------------------------------------------------------
IMPL_SITE_OLD = (
    "        from b12x.moe._shared.kernels.w4a16.kernel import _w4a16_topk_sum_launch_flat\n"
    "\n"
    "        _w4a16_topk_sum_launch_flat(\n"
    "            route_output,\n"
    "            scatter_output,\n"
    "            m,\n"
    "            num_topk,\n"
    "            k,\n"
    '            "bf16",\n'
    "            int(stream),\n"
    "            route_expert_ids=_int32_route_ids(flat_ids),\n"
    "            launcher=compact.topk_sum,\n"
    "        )\n"
    "        return scatter_output\n"
)
IMPL_SITE_NEW = (
    "        from b12x.moe._shared.kernels.w4a16.kernel import _w4a16_topk_sum_launch_flat\n"
    "\n"
    f"        _moet_ids = _int32_route_ids(flat_ids)  # {MARKER}\n"
    "        if _moet_topksum_add(route_output, _moet_ids, scatter_output, m, num_topk, k):\n"
    "            return scatter_output\n"
    "        _w4a16_topk_sum_launch_flat(\n"
    "            route_output,\n"
    "            scatter_output,\n"
    "            m,\n"
    "            num_topk,\n"
    "            k,\n"
    '            "bf16",\n'
    "            int(stream),\n"
    "            route_expert_ids=_moet_ids,\n"
    "            launcher=compact.topk_sum,\n"
    "        )\n"
    "        return scatter_output\n"
)
IMPL_DEF_ANCHOR = "def _launch_dynamic_topk_sum(\n"
IMPL_CODE = f'''# {MARKER} (tools/dsv41_sm120/patch_vllm_moe_topksum_add_sm120.py): the plain
# W4A16TopKSumKernel (per element: fp32 adds of the token's valid route values in route order, one bf16 rounding) with
# vLLM's shared-expert add (bf16(float(sum) + float(shared))) in one launch; the runner (patched too) publishes the
# shared-experts helper for the layer and skips its own add when this ran. VLLM_MOET_MOE_TOPKSUM_ADD=1 enables it.
_MOET_TOPKSUM_ADD_ON = None


def _moet_topksum_add_enabled() -> bool:
    global _MOET_TOPKSUM_ADD_ON
    if _MOET_TOPKSUM_ADD_ON is None:
        import os

        _MOET_TOPKSUM_ADD_ON = os.environ.get("VLLM_MOET_MOE_TOPKSUM_ADD", "0") == "1"
    return _MOET_TOPKSUM_ADD_ON


def _moet_topksum_add_kernel_fn():
    import triton
    import triton.language as tl

    @triton.jit
    def _moet_topksum_add_kernel(
        route_ptr,
        ids_ptr,
        shared_ptr,
        out_ptr,
        k,
        shared_stride,
        out_stride,
        TOPK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < k
        rid = tl.arange(0, 8)
        ids = tl.load(ids_ptr + token * TOPK + rid, mask=rid < TOPK, other=-1)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for r in tl.static_range(TOPK):
            row = token * TOPK + r
            expert = tl.max(tl.where(rid == r, ids, -1), axis=0)
            # The original skips unrouted entries (id < 0); here they add +0.0 through the load mask. acc starts at
            # +0.0 and a sum never yields -0.0 (x + (-x) is +0.0 in round-to-nearest), so acc + 0.0 is acc bit for
            # bit - the same value, and the six row loads issue together instead of behind six branches.
            v = tl.load(route_ptr + row * k + cols, mask=mask & (expert >= 0), other=0.0).to(tl.float32)
            acc = acc + v  # fp32 add.rn in route order, as the original kernel
        s = acc.to(tl.bfloat16).to(tl.float32)  # the top-k sum's bf16 output
        sh = tl.load(shared_ptr + token * shared_stride + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + token * out_stride + cols, (s + sh).to(tl.bfloat16), mask=mask)

    return _moet_topksum_add_kernel


_MOET_TOPKSUM_ADD_KERNEL = None


def _moet_topksum_add(route_output, route_ids, output, m, num_topk, k) -> bool:
    """The top-k sum with the layer's shared-expert output added, or False (the caller runs the original sum)."""
    global _MOET_TOPKSUM_ADD_KERNEL
    if not _moet_topksum_add_enabled():
        return False
    try:
        from vllm.model_executor.layers.fused_moe.runner import moe_runner as _mr
    except Exception:  # noqa: BLE001
        return False
    ctx = getattr(_mr, "_MOET_TOPKSUM_ADD_CTX", None)
    if ctx is None or ctx.get("helper") is None:
        return False
    if ctx.get("consumed"):
        ctx["again"] = True  # a second sum in the same forward: the original path, and the runner adds as before
        return False
    helper = ctx["helper"]
    idx = helper._output_idx
    shared = helper._output[idx] if idx < len(helper._output) else None
    if shared is None:  # the shared experts did not run ahead on the aux stream
        return False
    if not (
        output.dtype == torch.bfloat16
        and output.is_contiguous()
        and tuple(output.shape) == (m, k)
        and shared.dtype == torch.bfloat16
        and shared.dim() == 2
        and tuple(shared.shape) == (m, k)
        and shared.stride(1) == 1
        and route_output.dtype == torch.bfloat16
        and route_output.is_contiguous()
        and route_output.numel() >= m * num_topk * k
        and route_ids.dtype == torch.int32
        and route_ids.is_contiguous()
        and route_ids.numel() >= m * num_topk
        and 1 <= num_topk <= 8
        and k % 8 == 0
    ):
        return False
    helper.wait()  # the main stream joins the aux stream here instead of before the runner's add
    _moet_topksum_add_launch(route_output, route_ids, shared, output, m, num_topk, k)
    ctx["consumed"] = True
    return True


def _moet_topksum_add_launch(route_output, route_ids, shared, output, m, num_topk, k) -> None:
    """One launch: 128 columns per program, one warp (served shape 8 x 6 x 5120 in a graph: 1.03 us, the same as
    b12x's sum alone; the sum + torch's add take 1.8)."""
    global _MOET_TOPKSUM_ADD_KERNEL
    if _MOET_TOPKSUM_ADD_KERNEL is None:
        _MOET_TOPKSUM_ADD_KERNEL = _moet_topksum_add_kernel_fn()
    BLOCK = 128
    _MOET_TOPKSUM_ADD_KERNEL[(m, (k + BLOCK - 1) // BLOCK)](
        route_output,
        route_ids,
        shared,
        output,
        k,
        shared.stride(0),
        output.stride(0),
        TOPK=num_topk,
        BLOCK=BLOCK,
        num_warps=1,
    )


'''

# ---- vLLM moe_runner.py: publish the helper before the experts run, skip the add when the sum did it ------------------
RUNNER_ENTRY_OLD = (
    "        result = self._forward_entry(\n"
    "            hidden_states,\n"
    "            router_logits,\n"
    "            shared_experts_input,\n"
    "            input_ids,\n"
    "            self._encode_layer_name(),\n"
)
RUNNER_ENTRY_NEW = (
    f"        _moet_topksum_add_begin(self, og_hidden_dim_pre_xform)  # {MARKER}\n"
    "        result = self._forward_entry(\n"
    "            hidden_states,\n"
    "            router_logits,\n"
    "            shared_experts_input,\n"
    "            input_ids,\n"
    "            self._encode_layer_name(),\n"
)
RUNNER_ADD_OLD = (
    "        if shared_output is not None:\n"
    "            result = shared_output + fused_output\n"
    "        else:\n"
    "            result = fused_output\n"
)
RUNNER_ADD_NEW = (
    "        if shared_output is not None:\n"
    f"            if _moet_topksum_add_end():  # {MARKER}: the top-k sum already added shared_output\n"
    "                result = fused_output\n"
    "            else:\n"
    "                result = shared_output + fused_output\n"
    "        else:\n"
    f"            _moet_topksum_add_end()  # {MARKER}\n"
    "            result = fused_output\n"
)
RUNNER_CLASS_ANCHOR = "class MoERunner("
RUNNER_CODE = f'''# {MARKER} (tools/dsv41_sm120/patch_vllm_moe_topksum_add_sm120.py): the layer's shared-experts
# helper is published for b12x's compact-micro top-k sum, which adds the shared output itself when it can; the runner
# then skips its add. VLLM_MOET_MOE_TOPKSUM_ADD=1 enables it (the b12x side reads the same switch).
_MOET_TOPKSUM_ADD_CTX: dict = {{"helper": None, "consumed": False, "again": False}}
_MOET_TOPKSUM_ADD_ON = None


def _moet_topksum_add_begin(runner, og_hidden_dim_pre_xform) -> None:
    global _MOET_TOPKSUM_ADD_ON
    if _MOET_TOPKSUM_ADD_ON is None:
        import os

        _MOET_TOPKSUM_ADD_ON = os.environ.get("VLLM_MOET_MOE_TOPKSUM_ADD", "0") == "1"
    ctx = _MOET_TOPKSUM_ADD_CTX
    ctx["helper"] = None
    ctx["consumed"] = False
    ctx["again"] = False
    if not _MOET_TOPKSUM_ADD_ON:
        return
    helper = getattr(runner, "_shared_experts", None)
    if helper is None or getattr(helper, "_stream", None) is None:
        return
    if og_hidden_dim_pre_xform is not None:
        return
    if getattr(runner, "routed_scaling_factor", 1.0) != 1.0:
        return
    if getattr(runner, "routed_output_transform", None) is not None:
        return
    if runner.apply_routed_output_transform.__func__ is not MoERunner.apply_routed_output_transform:
        return
    try:
        if runner._fused_output_is_reduced:
            return
    except Exception:  # noqa: BLE001
        return
    ctx["helper"] = helper


def _moet_topksum_add_end() -> bool:
    ctx = _MOET_TOPKSUM_ADD_CTX
    fused = bool(ctx["consumed"]) and not ctx["again"]
    ctx["helper"] = None
    ctx["consumed"] = False
    ctx["again"] = False
    return fused


'''


def patch_file(path: Path, edits: list[tuple[str, str]], anchor: str, code: str, check: bool) -> int:
    text = path.read_text()
    if MARKER in text:
        print(f"already patched: {path}")
        return 0
    if check:
        print(f"NOT patched: {path}")
        return 1
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(f"anchor not found exactly once in {path} (found {text.count(old)}):\n{old}")
        text = text.replace(old, new)
    if text.count(anchor) != 1:
        sys.exit(f"'{anchor.strip()}' not found exactly once in {path}")
    text = text.replace(anchor, code + anchor)
    path.write_text(text)
    print(f"patched: {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--impl", type=Path, default=IMPL)
    ap.add_argument("--runner", type=Path, default=RUNNER)
    ap.add_argument("--check", action="store_true", help="exit 1 if the patch is not applied")
    a = ap.parse_args()
    rc = patch_file(a.impl, [(IMPL_SITE_OLD, IMPL_SITE_NEW)], IMPL_DEF_ANCHOR, IMPL_CODE, a.check)
    rc |= patch_file(a.runner, [(RUNNER_ENTRY_OLD, RUNNER_ENTRY_NEW), (RUNNER_ADD_OLD, RUNNER_ADD_NEW)], RUNNER_CLASS_ANCHOR, RUNNER_CODE, a.check)
    return rc


if __name__ == "__main__":
    sys.exit(main())
