#!/usr/bin/env python3
"""The sequence-parallel prefill step's MoE input all-gathered in MXFP8 instead of bf16 (DeepSeek-V4.1-Flash, sm_120, TP4).

Switch: VLLM_MOET_PREFILL_SP_GATHER=mxfp8 (default bf16 = the served gather of step 41, untouched). Applies on top of
patch_vllm_prefill_sp_sm120.py (step 41: the sequence-parallel boundaries) and patch_vllm_b12x_moe_prefill_fast_sm120.py
(step 42: the fast prefill MoE kernels, whose launch takes the pre-quantized rows); the runtime lives in
tools/dsv41_sm120/sp_moe_gather/sp_moe_gather_sm120.py (VLLM_MOET_SP_MOE_GATHER_DIR, default
/opt/vllm-moet/dsv41_sm120/sp_moe_gather), whose header is the design document.

What changes at boundary 2 of an eager prefill step (attention output -> MoE input) with the switch on, for a layer whose MoE
is eligible (b12x routed experts on the fast kernels with the pre-quantized-rows entry, GateLinear's cuBLAS tier, a
FusedTopKBiasRouter without EPLB / fused shared slots, the shared expert on the MXFP8 CUTLASS kernel with step 38's
pre-quantized-input hook; decided once per layer, identically on every rank):
  today   RS -> mHC on the shard -> all-gather bf16 [T, 5120] (10 240 B/row, 738 us per 4096-row chunk) -> ffn(x)
  mxfp8   RS -> mHC on the shard -> on the shard: FlashInfer's mxfp8 quantizer (e4m3 + UE8M0 per 32 columns, row-major -
          the recipe of b12x's quantize_block_fp8_mx and of the shared expert's own quantization: identical bytes), the
          router GEMM with the full-row kernel (this rank's rows inside a [T, 5120] buffer, M = T), the top-6 through the
          layer's router on the shard's rows (the padding mask sliced) -> two all-gathers: e4m3 [T, 5120] and
          [T, 160 + 24 + 24] (scales, ids, weights) -> ffn(stand-in): the [T, 5120] bf16 buffer whose rows are never read -
          GateLinear returns the gathered logits (the shard rows are the real ones), the runner takes the gathered top-k
          instead of select_experts, the fast MoE launch takes the e4m3 rows + scales, the shared expert's first GEMM takes
          the e4m3 rows + swizzled scales through step 38's hook. After ffn(): every consumer must have taken its piece
          (a loud RuntimeError otherwise - a consumer that read the stand-in would be a wrong result).
Bits: every shard-side step is the served arithmetic on the same rows (sp_moe_gather/test_prefill_sp_gather_tp4.py: EQUAL).

Patched files (anchor-based, idempotent, --check):
  vllm/models/deepseek_v41/nvidia/model.py                              boundary 2's gather -> _moet_sp_moe_gather(), finish()
  vllm/model_executor/layers/fused_moe/router/gate_linear.py            GateLinear.forward: the gathered logits for a stand-in
  vllm/model_executor/layers/fused_moe/runner/moe_runner.py             _apply_quant_method: the gathered top-k for a stand-in
  vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py _get_padding_mask: the shard's slice while the top-k
                                                                        runs on the shard
  flashinfer/experimental/b12x/moe/fused_moe/_impl.py                   step 42's launch (v1 = the r23 image, or v2 of
                                                                        4.11): a_values / a_scales for a stand-in

Usage:
    python3 patch_vllm_prefill_sp_gather_sm120.py [--dist DIR] [--flashinfer DIR] [--check]
  --dist        the vllm package directory (default /usr/local/lib/python3.12/dist-packages/vllm)
  --flashinfer  the flashinfer package directory (default /usr/local/lib/python3.12/dist-packages/flashinfer)
  --check       dry run: report whether every anchor is present (or the patch is already applied), change nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm")
DEFAULT_FLASHINFER = Path("/usr/local/lib/python3.12/dist-packages/flashinfer")
MARKER = "[vllm-moet] sm_120 prefill SP gather"
SP_MARKER = "[vllm-moet] sm_120 prefill SP"
FAST_MARKER = "[vllm-moet] sm_120 fast prefill MoE"


def resolver(flag: str, piece: str) -> str:
    return f'''
# {MARKER} (tools/dsv41_sm120/patch_vllm_prefill_sp_gather_sm120.py): with VLLM_MOET_PREFILL_SP_GATHER=mxfp8 the
# sequence-parallel prefill step's MoE input arrives as e4m3 rows + scales + the shards' top-k on a side channel keyed by the
# stand-in tensor (sp_moe_gather_sm120). Here: {piece}. Off (the default): nothing is imported, a dictionary miss per call.
{flag} = True
_MOET_SPG = None  # None = unresolved, False = off / unavailable, else the sp_moe_gather_sm120 module


def _moet_spg():
    global _MOET_SPG
    if _MOET_SPG is None:
        import os
        import sys

        _MOET_SPG = False
        if os.environ.get("VLLM_MOET_PREFILL_SP_GATHER", "bf16") == "mxfp8":
            d = os.environ.get("VLLM_MOET_SP_MOE_GATHER_DIR", "/opt/vllm-moet/dsv41_sm120/sp_moe_gather")
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                import sp_moe_gather_sm120 as _moet_spg_module

                _MOET_SPG = _moet_spg_module
            except ImportError:
                _MOET_SPG = False
    return _MOET_SPG
'''


# ---------------------------------------------------------------------------------------------------------------
# model.py (after patch_vllm_prefill_sp_sm120.py)
# ---------------------------------------------------------------------------------------------------------------
MODEL_HELPERS_ANCHOR = (
    "    return replay._run(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n"
    "\n"
    "\n"
    "class DeepseekV4MoE(DeepseekV4MoEBase):\n"
)
MODEL_HELPERS = (
    "    return replay._run(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n"
    + resolver("_MOET_SP_GATHER_MODEL_HOOK", "boundary 2 runs gather_moe_input() and, after the MoE, finish()")
    + '''

def _moet_sp_moe_gather(layer, x: torch.Tensor, positions: torch.Tensor, input_ids: torch.Tensor | None) -> torch.Tensor:
    """Boundary 2's all-gather of the MoE input for the sequence-parallel step: bf16 as in step 41 (the default), or with
    VLLM_MOET_PREFILL_SP_GATHER=mxfp8 the shard quantized / routed first and gathered as e4m3 + scales + top-k, the MoE
    then called on a [T, H] stand-in (sp_moe_gather_sm120.gather_moe_input; None = the layer is not eligible)."""
    m = _moet_spg()
    if m:
        out = m.gather_moe_input(layer.ffn, x, positions.shape[0], input_ids)
        if out is not None:
            return out
    return sp_all_gather(x)[: positions.shape[0]]


def _moet_sp_moe_gather_finish(standin: torch.Tensor) -> None:
    """After ffn(): every consumer took its piece of the MXFP8 gather (a no-op for the bf16 gather)."""
    m = _moet_spg()
    if m:
        m.finish(standin)


class DeepseekV4MoE(DeepseekV4MoEBase):
'''
)
MODEL_B2_ANCHOR = (
    f"        if moet_sp:  # {SP_MARKER}: the TP experts, the router and the shared expert read every row\n"
    "            x = sp_all_gather(x)[: positions.shape[0]]\n"
    "        if self.fuse_mhc_all_reduce and self.ffn.defers_finalize(x.shape[0]):\n"
    "            # The next layer's first boundary finalizes it with the all-reduce.\n"
    "            x = self.ffn.forward_unfinalized(x, input_ids)\n"
    "        else:\n"
    "            x = self.ffn(x, input_ids, mega_gate_metadata)\n"
)
MODEL_B2_NEW = (
    f"        moet_sp_standin = None  # {MARKER}\n"
    f"        if moet_sp:  # {SP_MARKER}: the TP experts, the router and the shared expert read every row\n"
    f"            x = _moet_sp_moe_gather(self, x, positions, input_ids)  # {MARKER}: bf16, or MXFP8 + the shard's top-k\n"
    "            moet_sp_standin = x\n"
    "        if self.fuse_mhc_all_reduce and self.ffn.defers_finalize(x.shape[0]):\n"
    "            # The next layer's first boundary finalizes it with the all-reduce.\n"
    "            x = self.ffn.forward_unfinalized(x, input_ids)\n"
    "        else:\n"
    "            x = self.ffn(x, input_ids, mega_gate_metadata)\n"
    "        if moet_sp_standin is not None:\n"
    f"            _moet_sp_moe_gather_finish(moet_sp_standin)  # {MARKER}: every consumer took its piece\n"
)
MODEL_EDITS = [("helpers", MODEL_HELPERS_ANCHOR, MODEL_HELPERS), ("boundary 2 gather", MODEL_B2_ANCHOR, MODEL_B2_NEW)]

# ---------------------------------------------------------------------------------------------------------------
# gate_linear.py (after patch_vllm_bf16_gemm_sm120.py, which is in the served image; the anchors do not depend on it)
# ---------------------------------------------------------------------------------------------------------------
GATE_HELPER_ANCHOR = (
    "logger = init_logger(__name__)\n"
    "\n"
    "\n"
    '@PluggableLayer.register("gate_linear")\n'
    "class GateLinear(ReplicatedLinear):\n"
)
GATE_HELPER = (
    "logger = init_logger(__name__)\n"
    + resolver("_MOET_SP_GATHER_ROUTER_HOOK", "GateLinear.forward returns a stand-in's gathered logits (the shard rows are the real ones; nobody reads the others)")
    + '''

def _moet_sp_gather_take_router_logits(x):
    m = _moet_spg()
    return m.take_router_logits(x) if m else None


@PluggableLayer.register("gate_linear")
class GateLinear(ReplicatedLinear):
'''
)
GATE_FWD_ANCHOR = (
    "    def forward(\n"
    "        self, x: torch.Tensor\n"
    "    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:\n"
    "        # Tier 1: cuteDSL ll_bf16_gemm (SM90+, bf16 in, fp32 out,\n"
)
GATE_FWD_NEW = (
    "    def forward(\n"
    "        self, x: torch.Tensor\n"
    "    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:\n"
    f"        _moet_spg_logits = _moet_sp_gather_take_router_logits(x)  # {MARKER}: a stand-in carries its logits\n"
    "        if _moet_spg_logits is not None:\n"
    "            return self._return(_moet_spg_logits)\n"
    "        # Tier 1: cuteDSL ll_bf16_gemm (SM90+, bf16 in, fp32 out,\n"
)
GATE_EDITS = [("helpers", GATE_HELPER_ANCHOR, GATE_HELPER), ("forward", GATE_FWD_ANCHOR, GATE_FWD_NEW)]

# ---------------------------------------------------------------------------------------------------------------
# moe_runner.py (after patch_vllm_moe_prequant_sm120.py: step 34's line is the anchor's first line)
# ---------------------------------------------------------------------------------------------------------------
RUNNER_HELPER_ANCHOR = "\n\nclass MoERunner(MoERunnerInterface):\n"
RUNNER_HELPER = (
    resolver("_MOET_SP_GATHER_TOPK_HOOK", "_apply_quant_method takes the top-k weights and ids gathered from the shards")
    + '''

def _moet_sp_gather_take_topk(hidden_states):
    m = _moet_spg()
    return m.take_topk(hidden_states) if m else None


class MoERunner(MoERunnerInterface):
'''
)
RUNNER_TOPK_ANCHOR = (
    "            _moet_prequant_begin(self, hidden_states)  # [vllm-moet] step 34 MoE rows quantization off the critical path\n"
    "            topk_weights, topk_ids = self.router.select_experts(\n"
    "                hidden_states=hidden_states,\n"
    "                router_logits=router_logits,\n"
    "                topk_indices_dtype=self._quant_method.topk_indices_dtype,\n"
    "                input_ids=input_ids,\n"
    "            )\n"
)
RUNNER_TOPK_NEW = (
    "            _moet_prequant_begin(self, hidden_states)  # [vllm-moet] step 34 MoE rows quantization off the critical path\n"
    f"            _moet_spg_topk = _moet_sp_gather_take_topk(hidden_states)  # {MARKER}: the shards' top-k for a stand-in\n"
    "            if _moet_spg_topk is not None:\n"
    "                topk_weights, topk_ids = _moet_spg_topk\n"
    "            else:\n"
    "                topk_weights, topk_ids = self.router.select_experts(\n"
    "                    hidden_states=hidden_states,\n"
    "                    router_logits=router_logits,\n"
    "                    topk_indices_dtype=self._quant_method.topk_indices_dtype,\n"
    "                    input_ids=input_ids,\n"
    "                )\n"
)
RUNNER_EDITS = [("helpers", RUNNER_HELPER_ANCHOR, RUNNER_HELPER), ("select_experts", RUNNER_TOPK_ANCHOR, RUNNER_TOPK_NEW)]

# ---------------------------------------------------------------------------------------------------------------
# fused_topk_bias_router.py
# ---------------------------------------------------------------------------------------------------------------
TOPK_MASK_ANCHOR = (
    "def _get_padding_mask(num_tokens: int) -> torch.Tensor | None:\n"
    "    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():\n"
)
TOPK_MASK_NEW = (
    f"# {MARKER}: while the top-k runs on this rank's row shard (sp_moe_gather_sm120.gather_moe_input), the padding mask is\n"
    "# the shard's slice, set here by the caller (active?, mask); otherwise the forward context's rows as before.\n"
    "_MOET_SP_GATHER_PADDING = (False, None)\n"
    "\n"
    "\n"
    "def _get_padding_mask(num_tokens: int) -> torch.Tensor | None:\n"
    f"    if _MOET_SP_GATHER_PADDING[0]:  # {MARKER}\n"
    "        return _MOET_SP_GATHER_PADDING[1]\n"
    "    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():\n"
)
TOPK_EDITS = [("padding mask", TOPK_MASK_ANCHOR, TOPK_MASK_NEW)]

# ---------------------------------------------------------------------------------------------------------------
# flashinfer _impl.py (after patch_vllm_b12x_moe_prefill_fast_sm120.py)
# ---------------------------------------------------------------------------------------------------------------
IMPL_HELPER_ANCHOR = f"# {FAST_MARKER} (tools/dsv41_sm120/patch_vllm_b12x_moe_prefill_fast_sm120.py): "
IMPL_HELPER = (
    resolver("_MOET_SP_GATHER_PREQUANT_HOOK", "the fast prefill MoE launch takes the pre-quantized rows (e4m3 bytes + row-major UE8M0 scales)")
    + '''

def _moet_sp_gather_take_prequant(a):
    m = _moet_spg()
    return m.take_prequant(a) if m else None


''' + IMPL_HELPER_ANCHOR
)
# the launch inside step 42's _moet_prefill_fast_launch: v1 of that patch (the r23 image: route rows only) or v2
# (4.11, W's Phase 1a v8 commit: the kernels may also write the top-k sum) - whichever the file carries
IMPL_LAUNCH_V1_ANCHOR = (
    "    module.moe_prefill_fast(\n"
    "        a,\n"
    "        flat_ids.view(m, num_topk),\n"
    "        flat_weights.view(m, num_topk),\n"
    "        planes,\n"
    "        float(swiglu_limit),\n"
    "        n,\n"
    "        route_out=workspace.route_output,\n"
    "        topk_sum=False,\n"
    "    )\n"
    "    return True\n"
)
IMPL_LAUNCH_V2_ANCHOR = (
    "    module.moe_prefill_fast(\n"
    "        a,\n"
    "        flat_ids.view(m, num_topk),\n"
    "        flat_weights.view(m, num_topk),\n"
    "        planes,\n"
    "        float(swiglu_limit),\n"
    "        n,\n"
    "        route_out=workspace.route_output,\n"
    "        out=scatter_output if fused else None,\n"
    "        topk_sum=fused,\n"
    "    )\n"
)
IMPL_LAUNCH_HEAD = (
    f"    _moet_spg_rows = _moet_sp_gather_take_prequant(a)  # {MARKER}: the rows arrive pre-quantized on the side channel\n"
    "    if _moet_spg_rows is not None and not getattr(module, \"prequant_rows_supported\", lambda: False)():\n"
    "        raise RuntimeError(\"prefill SP gather=mxfp8: the fast prefill MoE extension has no pre-quantized rows entry\")\n"
    "    module.moe_prefill_fast(\n"
    "        None if _moet_spg_rows is not None else a,\n"
    "        flat_ids.view(m, num_topk),\n"
    "        flat_weights.view(m, num_topk),\n"
    "        planes,\n"
    "        float(swiglu_limit),\n"
    "        n,\n"
    "        route_out=workspace.route_output,\n"
)
IMPL_LAUNCH_TAIL = (
    "        **({} if _moet_spg_rows is None else dict(a_values=_moet_spg_rows[0], a_scales=_moet_spg_rows[1])),\n"
    "    )\n"
)
IMPL_LAUNCH_V1_NEW = IMPL_LAUNCH_HEAD + "        topk_sum=False,\n" + IMPL_LAUNCH_TAIL + "    return True\n"
IMPL_LAUNCH_V2_NEW = IMPL_LAUNCH_HEAD + "        out=scatter_output if fused else None,\n        topk_sum=fused,\n" + IMPL_LAUNCH_TAIL
IMPL_LAUNCH_ALTERNATIVES = [(IMPL_LAUNCH_V1_ANCHOR, IMPL_LAUNCH_V1_NEW), (IMPL_LAUNCH_V2_ANCHOR, IMPL_LAUNCH_V2_NEW)]
IMPL_EDITS = [("helpers", IMPL_HELPER_ANCHOR, IMPL_HELPER), ("launch", IMPL_LAUNCH_ALTERNATIVES, None)]

FILES = [
    ("dist", "models/deepseek_v41/nvidia/model.py", MODEL_EDITS, SP_MARKER),
    ("dist", "model_executor/layers/fused_moe/router/gate_linear.py", GATE_EDITS, None),
    ("dist", "model_executor/layers/fused_moe/runner/moe_runner.py", RUNNER_EDITS, "[vllm-moet] step 34"),
    ("dist", "model_executor/layers/fused_moe/router/fused_topk_bias_router.py", TOPK_EDITS, None),
    ("flashinfer", "experimental/b12x/moe/fused_moe/_impl.py", IMPL_EDITS, FAST_MARKER),
]


def patch_text(src: str, edits, check: bool, requires: str | None) -> tuple[str, str]:
    """Apply the edits; returns (new source, status) with status in {applied, already, missing:<edit>, requires:<marker>}."""
    if MARKER in src:
        return src, "already"
    if requires is not None and requires not in src:
        return src, f"requires:{requires}"
    resolved = []
    for label, anchor, new in edits:
        alternatives = anchor if isinstance(anchor, list) else [(anchor, new)]
        hits = [(a, nw) for a, nw in alternatives if src.count(a) == 1]
        if len(hits) != 1:
            return src, f"missing:{label} ({[src.count(a) for a, _ in alternatives]} matches of the alternatives)"
        resolved.append(hits[0])
    if check:
        return src, "applied"
    for anchor, new in resolved:
        src = src.replace(anchor, new, 1)
    compile(src, "<patched>", "exec")
    return src, "applied"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dist", type=Path, default=DEFAULT_DIST, help="the vllm package directory")
    ap.add_argument("--flashinfer", type=Path, default=DEFAULT_FLASHINFER, help="the flashinfer package directory")
    ap.add_argument("--check", action="store_true", help="dry run: verify the anchors, change nothing")
    args = ap.parse_args()
    rc = 0
    for root, rel, edits, requires in FILES:
        path = (args.dist if root == "dist" else args.flashinfer) / rel
        if not path.exists():
            print(f"{rel}: not found under {path.parent}")
            rc = 1
            continue
        src = path.read_text()
        new, status = patch_text(src, edits, args.check, requires)
        if status.startswith("missing") or status.startswith("requires"):
            print(f"{rel}: anchor {status} - not patched")
            rc = 1
            continue
        if status == "already":
            print(f"{rel}: already patched")
            continue
        if args.check:
            print(f"{rel}: all {len(edits)} anchors present (dry run)")
        else:
            path.write_text(new)
            print(f"{rel}: patched ({len(edits)} edits)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
