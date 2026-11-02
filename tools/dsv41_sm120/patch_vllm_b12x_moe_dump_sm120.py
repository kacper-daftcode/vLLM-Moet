#!/usr/bin/env python3
"""Dump hook for the b12x TP MoE of a real prefill (front "cubit + prefill", Phase 1 spec B for MoE FC1/FC2).

The prefill MoE of DeepSeek-V4.1-Flash on sm_120 runs through vLLM's `B12xExperts` (`fused_moe/b12x.py`): the expert
weights are prepared once (`_prepare_experts`: packed FP4 w1/w2 + block scales + per-expert global scales -> FlashInfer
b12x `prepare_weights`), every forward calls `apply` -> `_run_b12x_moe_plan` (hidden_states bf16 [T, 5120], top-k ids /
weights -> output bf16 [T, 5120], the `integration.tp_moe.dynamic` kernels Phase1 / Silu / Phase2 for T > 8). A SASS
replacement must reproduce these bits; this hook records what it has to reproduce, from the served model, on TP rank 0:

  <dir>/weights_layer<ordinal>.pt   once, for the <ordinal>-th B12xExperts instance prepared on this rank: the exact
                                    tensors handed to b12x `prepare_weights` after `_canonicalize_fp4_zero_signs_`
                                    (w1_fp4, w2_fp4 uint8 packed, w1/w2 block scales, global scales, a1/a2 gscale),
                                    plus quant_mode / source_format / w13_layout / params_dtype / activation
  <dir>/call<k>_layer<ordinal>.pt   for the first <max_calls> `apply` calls of that instance with >= <min_rows> rows:
                                    hidden_states, topk_ids (normalized int32), topk_weights (normalized), output after
                                    `fused_moe.run` + synchronize, the plan key and repr

Env: VLLM_MOET_B12X_DUMP_DIR (enables; must be writable from the worker), VLLM_MOET_B12X_DUMP_LAYER_ORDINAL (default 2:
the third MoE layer prepared on the rank), VLLM_MOET_B12X_DUMP_MIN_ROWS (1024), VLLM_MOET_B12X_DUMP_MAX_CALLS (2).
Debug-only; never served. Usage: python3 patch_vllm_b12x_moe_dump_sm120.py [--dist DIR] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "[vllm-moet] b12x MoE dump hook (sm_120)"

IMPORT_ANCHOR = "from vllm.platforms import current_platform\n"
HELPER = f'''

# {MARKER}: records the prepared weights of one MoE layer and the inputs / outputs of its prefill calls (rank 0).
_MOET_DUMP = None  # None = unresolved, False = off, dict = config


def _moet_dump_cfg():
    global _MOET_DUMP
    if _MOET_DUMP is None:
        import os

        d = os.environ.get("VLLM_MOET_B12X_DUMP_DIR", "")
        if not d:
            _MOET_DUMP = False
        else:
            rank = 0
            try:
                from vllm.distributed import get_tensor_model_parallel_rank

                rank = get_tensor_model_parallel_rank()
            except Exception:  # noqa: BLE001
                pass
            _MOET_DUMP = False if rank != 0 else dict(
                dir=d, ordinal=int(os.environ.get("VLLM_MOET_B12X_DUMP_LAYER_ORDINAL", "2")),
                min_rows=int(os.environ.get("VLLM_MOET_B12X_DUMP_MIN_ROWS", "1024")),
                max_calls=int(os.environ.get("VLLM_MOET_B12X_DUMP_MAX_CALLS", "2")), prepared=0, calls=0)
    return _MOET_DUMP


def _moet_dump_weights(experts, **tensors):
    cfg = _moet_dump_cfg()
    if not cfg:
        return
    idx = cfg["prepared"]
    cfg["prepared"] += 1
    if idx != cfg["ordinal"]:
        return
    import os

    os.makedirs(cfg["dir"], exist_ok=True)
    experts._moet_dump_tag = idx
    payload = {{k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}}
    torch.save(payload, os.path.join(cfg["dir"], f"weights_layer{{idx}}.pt"))
    logger.info("[vllm-moet] b12x dump: weights of MoE instance %d -> %s", idx, cfg["dir"])


def _moet_dump_call(experts, plan, hidden_states, topk_ids, topk_weights, output):
    cfg = _moet_dump_cfg()
    if not cfg or getattr(experts, "_moet_dump_tag", None) is None:
        return
    if hidden_states.shape[0] < cfg["min_rows"] or cfg["calls"] >= cfg["max_calls"]:
        return
    import os

    torch.cuda.synchronize()
    k = cfg["calls"]
    cfg["calls"] += 1
    torch.save(dict(hidden_states=hidden_states.detach().cpu(), topk_ids=topk_ids.detach().cpu(),
                    topk_weights=topk_weights.detach().cpu(), output=output.detach().cpu(),
                    plan_repr=repr(plan)[:4000], rows=int(hidden_states.shape[0])),
               os.path.join(cfg["dir"], f"call{{k}}_layer{{experts._moet_dump_tag}}.pt"))
    logger.info("[vllm-moet] b12x dump: call %d of instance %d (%d rows) -> %s", k, experts._moet_dump_tag,
                hidden_states.shape[0], cfg["dir"])
'''

PREP_OLD = (
    "        weight_plan = fused_moe.plan_weights(\n"
    "            quant_modes=quant_mode,\n"
)
PREP_NEW = (
    f"        _moet_dump_weights(  # {MARKER}\n"
    "            self, w1_fp4=w1, w2_fp4=w2, w1_blockscale=self.w1_scale, w2_blockscale=self.w2_scale,\n"
    "            w1_global_scale=w1_global_scale, w2_global_scale=w2_global_scale, a1_gscale=a1_gscale, a2_gscale=a2_gscale,\n"
    "            quant_mode=quant_mode, source_format=self._source_format, w13_layout=self._w13_layout,\n"
    "            params_dtype=str(params_dtype), activation=_b12x_activation_name(activation),\n"
    "            num_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size,\n"
    "        )\n"
    "        weight_plan = fused_moe.plan_weights(\n"
    "            quant_modes=quant_mode,\n"
)
APPLY_OLD = (
    "        _run_b12x_moe_plan(\n"
    "            plan=plan,\n"
    "            scratch=scratch,\n"
    "            hidden_states=hidden_states,\n"
    "            prepared=prepared,\n"
    "            topk_weights=topk_weights,\n"
    "            topk_ids=topk_ids,\n"
    "            output=output,\n"
    "            unit_scale_contract=self._quant_mode == \"w4a16\",\n"
    "        )\n"
)
APPLY_NEW = APPLY_OLD + (
    f"        _moet_dump_call(self, plan, hidden_states, topk_ids, topk_weights, output)  # {MARKER}\n"
)


def _once(src: str, anchor: str, what: str, name: str) -> None:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"{name}: {what} anchor found {n} times (expected 1)")


def patch(src: str) -> str:
    if MARKER in src:
        return src
    for what, anchor in (("import", IMPORT_ANCHOR), ("prepare_weights", PREP_OLD), ("run plan", APPLY_OLD)):
        _once(src, anchor, what, "fused_moe/b12x.py")
    src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + HELPER, 1)
    src = src.replace(PREP_OLD, PREP_NEW, 1)
    return src.replace(APPLY_OLD, APPLY_NEW, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=DIST)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    path = args.dist / "model_executor" / "layers" / "fused_moe" / "b12x.py"
    src = path.read_text()
    if MARKER in src:
        print(f"{path}: already patched")
        return 0
    out = patch(src)
    compile(out, str(path), "exec")
    if args.check:
        print(f"{path}: patch applies cleanly (not written)")
        return 0
    path.write_text(out)
    print(f"{path}: patched ({MARKER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
