#!/usr/bin/env python3
"""Step 34: the decode MoE's rows quantization off the critical path (bit-identical).

b12x's compact-micro launcher (w4a8_compact_micro.py) quantizes the MoE input rows to MXFP8 (compact.quantize, ~1.2 us)
on the main stream right before the fused MoE kernel - after the router's GEMM and top-k, although it depends on
neither. The patched vLLM runner launches the same prepared quantize kernel on a side stream before select_experts
(fork event -> quantize into vllm-moet-owned buffers of the scratch layout -> done event) and records a marker; the
launcher, when the fused MoE path is taken and the marker matches the input tensor (pointer, shape, capacity, K), waits
on the done event and hands the fused kernel those buffers instead of quantizing again. Same kernel, same input, same
bytes: the fused MoE reads identical operands. Any mismatch (other path, other tensor, prefill, no compact launchers)
quantizes as before into the scratch.

Runtime switch VLLM_MOET_MOE_PREQUANT=1 (default off).

Usage: python3 patch_vllm_moe_prequant_sm120.py [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages")
RUNNER = SITE / "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
MICRO = SITE / "flashinfer/experimental/b12x/moe/_shared/kernels/w4a8_compact_micro.py"
MARKER = "[vllm-moet] step 34 MoE rows quantization off the critical path"

# ---- vLLM moe_runner.py --------------------------------------------------------------------------------------------
RUNNER_SITE_OLD = (
    "            # Modular kernels: select experts first, then call routed_experts\n"
    "            topk_weights, topk_ids = self.router.select_experts(\n"
)
RUNNER_SITE_NEW = (
    "            # Modular kernels: select experts first, then call routed_experts\n"
    f"            _moet_prequant_begin(self, hidden_states)  # {MARKER}\n"
    "            topk_weights, topk_ids = self.router.select_experts(\n"
)
RUNNER_CLASS_ANCHOR = "class MoERunner("
RUNNER_CODE = f'''# {MARKER} (tools/dsv41_sm120/patch_vllm_moe_prequant_sm120.py): b12x's prepared
# MXFP8 rows quantize of the MoE input launched on a side stream before the router, into vllm-moet-owned buffers of the
# compact scratch layout; the compact-micro launcher takes them when its fused path runs for the same tensor.
# VLLM_MOET_MOE_PREQUANT=1 enables it.
_MOET_PREQUANT_CTX: dict = {{"on": None, "stream": None, "ring": [], "slot": 0, "marker": None, "misses": 0}}


def _moet_prequant_join_pending() -> None:
    """Join a side-stream quantize nobody took (the launch went another way): keeps a capture consistent."""
    ctx = _MOET_PREQUANT_CTX
    m = ctx["marker"]
    ctx["marker"] = None
    if m is not None:
        ctx["misses"] += 1
        torch.cuda.current_stream(m[4]["values"].device).wait_event(m[4]["done"])


def _moet_prequant_begin(runner, hidden_states) -> None:
    ctx = _MOET_PREQUANT_CTX
    _moet_prequant_join_pending()
    if ctx["on"] is None:
        import os

        ctx["on"] = os.environ.get("VLLM_MOET_MOE_PREQUANT", "0") == "1"
    if not ctx["on"]:
        return
    try:
        experts = runner._quant_method.moe_kernel.fused_experts
        if type(experts).__name__ != "B12xExperts":
            return
        if not (hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16 and hidden_states.dim() == 2
                and hidden_states.is_contiguous()):
            return
        tokens = max(int(hidden_states.shape[0]), 1)
        plans = [p for key, p in experts._plans.items() if key[0] == tokens]
        if len(plans) != 1:
            return
        plan = plans[0]
        from flashinfer.experimental.b12x.moe.fused_moe import _impl as _bi

        compact = _bi._moet_compact_launches(plan)
        if compact is None or getattr(compact, "quantize", None) is None:
            return
        lp = plan.launch_plan
        cap, k = int(lp.max_rows) // int(lp.num_topk), int(lp.k)
        if hidden_states.shape[0] > cap or int(hidden_states.shape[1]) != k or k % 128:
            return
        if int(hidden_states.shape[0]) * int(lp.num_topk) < 32:
            return  # step 27's fused MoE asks for >= 32 pairs; below it the launcher quantizes into the scratch itself
        dev = hidden_states.device
        if ctx["stream"] is None:
            ctx["stream"] = torch.cuda.Stream(device=dev)
        ring = ctx["ring"]
        if not ring or ring[0]["key"] != (cap, k, dev):
            # two buffer sets, alternated per layer (the previous layer's MoE has been enqueued before this fork)
            mma = ((cap + 127) // 128) * ((k + 127) // 128) * 512
            ring[:] = [
                dict(key=(cap, k, dev),
                     values=torch.empty((cap, k), dtype=torch.uint8, device=dev),
                     scale_rows=torch.empty((cap, k // 32), dtype=torch.uint8, device=dev),
                     scale_mma=torch.empty(mma, dtype=torch.uint8, device=dev),
                     fork=torch.cuda.Event(), done=torch.cuda.Event())
                for _ in range(2)
            ]
            ctx["slot"] = 0
        slot = ring[ctx["slot"]]
        ctx["slot"] ^= 1
        main = torch.cuda.current_stream(dev)
        side = ctx["stream"]
        slot["fork"].record(main)
        with torch.cuda.stream(side):
            side.wait_event(slot["fork"])
            compact.quantize(hidden_states, slot["values"], slot["scale_rows"], slot["scale_mma"])
            slot["done"].record(side)
        ctx["marker"] = (hidden_states.data_ptr(), tuple(hidden_states.shape), cap, k, slot)
    except Exception:  # noqa: BLE001 - never break the forward; the launcher quantizes as before
        ctx["marker"] = None


def _moet_prequant_take(a, cap, k):
    """The pre-quantized (values, scale_rows) of `a` for this launch, after joining the side stream; None = none."""
    ctx = _MOET_PREQUANT_CTX
    m = ctx["marker"]
    ctx["marker"] = None
    if m is None:
        return None
    slot = m[4]
    torch.cuda.current_stream(a.device).wait_event(slot["done"])  # join the side stream either way
    if m[0] != a.data_ptr() or m[1] != tuple(a.shape) or m[2] != int(cap) or m[3] != int(k):
        ctx["misses"] += 1
        return None
    return slot["values"], slot["scale_rows"]


'''

# ---- b12x w4a8_compact_micro.py (on top of step 27's text) ----------------------------------------------------------
MICRO_SITE_OLD = (
    "    if _prepared_quantize is None:\n"
    "        quantize_mxfp8_rows_cute(a, values, scale_rows, scale_mma, expected_m=cap)\n"
    "    else:\n"
    "        _prepared_quantize(a, values, scale_rows, scale_mma)\n"
    "\n"
    "    if _moet_moe_fused_on(\n"
    "        n=n, k=k, num_topk=num_topk, max_tokens=cap, ids_dtype=topk_ids.dtype, csf_inline=csf_inline, experts=experts,\n"
    "        has_limit=swiglu_limit is not None,\n"
    "    ):\n"
)
MICRO_SITE_NEW = (
    f"    _moet_fused = _moet_moe_fused_on(  # {MARKER}: decided first, the pre-quantized rows serve the fused path only\n"
    "        n=n, k=k, num_topk=num_topk, max_tokens=cap, ids_dtype=topk_ids.dtype, csf_inline=csf_inline, experts=experts,\n"
    "        has_limit=swiglu_limit is not None,\n"
    "    )\n"
    "    _moet_pre = _moet_prequant_take(a, cap, k) if _moet_fused else None\n"
    "    if _moet_pre is not None:\n"
    "        values, scale_rows = _moet_pre\n"
    "    elif _prepared_quantize is None:\n"
    "        quantize_mxfp8_rows_cute(a, values, scale_rows, scale_mma, expected_m=cap)\n"
    "    else:\n"
    "        _prepared_quantize(a, values, scale_rows, scale_mma)\n"
    "\n"
    "    if _moet_fused:\n"
)
MICRO_DEF_ANCHOR = "def launch_w4a8_compact_micro(\n"
MICRO_CODE = f'''# {MARKER}: the runner may have quantized this launch's rows on a side stream already.
def _moet_prequant_take(a, cap, k):
    try:
        from vllm.model_executor.layers.fused_moe.runner import moe_runner as _mr
    except Exception:  # noqa: BLE001
        return None
    take = getattr(_mr, "_moet_prequant_take", None)
    return None if take is None else take(a, cap, k)


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


RUNNER_JOIN_OLD = (
    "        # Extract outputs from result\n"
    "        shared_output, fused_output = _unpack(result)\n"
)
RUNNER_JOIN_NEW = (
    f"        _moet_prequant_join_pending()  # {MARKER}\n"
    "        # Extract outputs from result\n"
    "        shared_output, fused_output = _unpack(result)\n"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runner", type=Path, default=RUNNER)
    ap.add_argument("--micro", type=Path, default=MICRO)
    ap.add_argument("--check", action="store_true", help="exit 1 if the patch is not applied")
    a = ap.parse_args()
    rc = patch_file(a.runner, [(RUNNER_SITE_OLD, RUNNER_SITE_NEW), (RUNNER_JOIN_OLD, RUNNER_JOIN_NEW)], RUNNER_CLASS_ANCHOR, RUNNER_CODE, a.check)
    rc |= patch_file(a.micro, [(MICRO_SITE_OLD, MICRO_SITE_NEW)], MICRO_DEF_ANCHOR, MICRO_CODE, a.check)
    return rc


if __name__ == "__main__":
    sys.exit(main())
