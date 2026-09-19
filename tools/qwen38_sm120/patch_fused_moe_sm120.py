#!/usr/bin/env python3
"""Route vLLM's Triton `fused_moe_kernel` launches to the sm_120 FP8 block-scaled MoE GEMV
(tools/qwen38_sm120/moe_gemv/) at decode token counts.

Qwen3.8-Flash-Next-FP8 at TP4 runs its routed experts through vLLM's generic Triton
`fused_moe_kernel` (E=512, per-rank N=160, K=2560, fp8_w8a8, block scales refined to [32, 32]).
With BLOCK_SIZE_K capped at 32 by the block shape, a 4-token decode step is ~200 programs of
2 warps doing 80 serial K-steps each -- roughly half of the HBM bandwidth for the ~40 experts
touched per layer (2 x 28 us per layer in situ vs a ~15 us weight-stream floor, 2026-09-18
profile). The GEMV maps the same math onto ~1600 blocks of 256 threads (see the .cu header).

The patch adds one early dispatch at the top of `invoke_fused_moe_triton_kernel`: when the
launch is fp8_w8a8 with [32, 32] block scales, bf16 output, no bias, and at most
VLLM_MOET_SM120_MOE_GEMV_MAX_PAIRS (160) (token, expert) pairs, it calls the GEMV and returns;
everything else (prefill, large batches, other quantizations) is untouched. The GEMV module is
imported from VLLM_MOET_SM120_MOE_GEMV_DIR (default /opt/vllm-moet/moe_gemv, bind-mounted by the
qwen38 launcher) and JIT-compiled on first use (cached under ~/.cache/torch_extensions).
`VLLM_MOET_SM120_MOE_GEMV=0` disables the dispatch without removing the patch.

The second file, `experts/triton_moe.py` (`TritonExperts.apply`), gets the fused down-GEMM path:
at decode token counts the GEMV computes silu(gate) * up and the per-32-group fp8 quantization
itself (bit-identical to vLLM's act_and_mul + per_token_group_quant + GEMM, two launches less per
layer). It imports the hook from the patched fused_moe.py and is a no-op without it.

The MoE forward is a custom op (`moe_forward`), so the compiled torch graph does not change and
no separate torch_compile_cache is needed (unlike the skinny GEMM patch).

Usage: python3 patch_fused_moe_sm120.py [--file PATH --out PATH] [--experts-file PATH --experts-out PATH]
                                        [--no-experts] [--check]        (idempotent, anchor-based)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py")
MARKER = "[vllm-moet] sm_120 MoE GEMV"

IMPORT_OLD = "logger = init_logger(__name__)\n"
IMPORT_NEW = (
    "logger = init_logger(__name__)\n"
    "\n"
    f"# {MARKER}: FP8 [32,32] block-scaled expert GEMV for decode token counts on RTX PRO 6000\n"
    "# (tools/qwen38_sm120/moe_gemv). Imported lazily from the bind-mounted directory; any failure\n"
    "# leaves the Triton path untouched.\n"
    "_sm120_moe_gemv = None\n"
    "\n"
    "\n"
    "def _sm120_moe_gemv_module():\n"
    "    global _sm120_moe_gemv\n"
    "    if _sm120_moe_gemv is not None:\n"
    "        return _sm120_moe_gemv if _sm120_moe_gemv is not False else None\n"
    "    import os\n"
    "    import sys\n"
    "\n"
    "    _sm120_moe_gemv = False\n"
    "    try:\n"
    '        if os.environ.get("VLLM_MOET_SM120_MOE_GEMV", "1") == "0":\n'
    "            return None\n"
    "        if not current_platform.is_cuda() or not current_platform.is_device_capability_family(120):\n"
    "            return None\n"
    '        path = os.environ.get("VLLM_MOET_SM120_MOE_GEMV_DIR", "/opt/vllm-moet/moe_gemv")\n'
    "        if not os.path.isdir(path):\n"
    '            logger.warning("sm_120 MoE GEMV directory %s not found; Triton fused_moe stays", path)\n'
    "            return None\n"
    "        if path not in sys.path:\n"
    "            sys.path.insert(0, path)\n"
    "        import fused_moe_gemv_sm120 as mod\n"
    "\n"
    "        _sm120_moe_gemv = mod\n"
    '        logger.info("sm_120 MoE GEMV: fp8 [32,32] expert GEMMs with <= %d (token, expert) pairs go to %s",\n'
    "                    mod.max_pairs(), mod.__file__)\n"
    "        return mod\n"
    "    except Exception:  # noqa: BLE001\n"
    '        logger.exception("sm_120 MoE GEMV unavailable; Triton fused_moe stays")\n'
    "        return None\n"
)

DISPATCH_OLD = (
    "    if use_fp8_w8a8:\n"
    '        SWAP_AB = enable_swap_ab(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])\n'
    "    else:\n"
    "        SWAP_AB = False\n"
)
DISPATCH_NEW = (
    f"    # {MARKER}\n"
    "    _gemv = _sm120_moe_gemv_module() if use_fp8_w8a8 else None\n"
    "    if _gemv is not None and _gemv.moe_gemv_applicable(\n"
    "        A, B, C, A_scale, B_scale, B_bias, sorted_token_ids, expert_ids, top_k, config,\n"
    "        use_fp8_w8a8, per_channel_quant, block_shape,\n"
    "    ):\n"
    "        _gemv.fused_moe_gemv(\n"
    "            A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids,\n"
    "            num_tokens_post_padded, mul_routed_weight, top_k, config,\n"
    "        )\n"
    "        return\n"
    "\n"
    "    if use_fp8_w8a8:\n"
    '        SWAP_AB = enable_swap_ab(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])\n'
    "    else:\n"
    "        SWAP_AB = False\n"
)


# --- experts/triton_moe.py: down GEMM with silu(gate)*up + fp8 quant fused (naive path) ---
DEFAULT_EXPERTS_PATH = DEFAULT_PATH.parent / "experts" / "triton_moe.py"

EXPERTS_IMPORT_OLD = (
    "from vllm.model_executor.layers.fused_moe.moe_align_block_size import (\n"
    "    moe_align_block_size,\n"
    ")\n"
)
EXPERTS_IMPORT_NEW = (
    "from vllm.model_executor.layers.fused_moe.moe_align_block_size import (\n"
    "    moe_align_block_size,\n"
    ")\n"
    "\n"
    f"# {MARKER}: the module hook lives in the patched fused_moe.py; without it the fused\n"
    "# activation path below is simply skipped.\n"
    "try:\n"
    "    from vllm.model_executor.layers.fused_moe.fused_moe import _sm120_moe_gemv_module\n"
    "except ImportError:  # unpatched fused_moe.py\n"
    "\n"
    "    def _sm120_moe_gemv_module():\n"
    "        return None\n"
)
# Decision + buffer placement before the first GEMM: the fused down GEMV reads cache1 while
# writing cache3, and vLLM puts both in workspace2 ("done with cache1 by the time we need
# cache3" -- not true any more), so cache1 moves to workspace13 (sized (M, topk, max(N, K)),
# so it fits; the MoE output may alias the start of workspace13 but is only written by the
# final moe_sum, after the fused kernel has consumed cache1).
EXPERTS_DECIDE_OLD = "        # LoRA w13: applied to intermediate_cache1 before activation. When\n"
EXPERTS_DECIDE_NEW = (
    f"        # {MARKER}: at decode token counts (naive expert assignment) the down GEMM runs\n"
    "        # with silu(gate) * up and the per-32-group fp8 quantization fused into the GEMV\n"
    "        # (bit-identical to act_and_mul + per_token_group_quant + the GEMV, two launches\n"
    "        # less per layer); LoRA, static a2 scales and emulation keep the generic path.\n"
    "        # The fused kernel reads cache1 while writing cache3, which share workspace2 in\n"
    "        # vLLM's layout -> cache1 goes to workspace13 instead (the MoE output may alias\n"
    "        # its start, but is written only by the final moe_sum).\n"
    "        _gemv = _sm120_moe_gemv_module() if self.quant_config.use_fp8_w8a8 else None\n"
    "        _gemv_fused = (\n"
    "            _gemv is not None\n"
    "            and self._lora_context is None\n"
    "            and activation == MoEActivation.SILU\n"
    "            and a2_scale is None\n"
    "            and not self.quantization_emulation\n"
    "            and _gemv.fused_act_applicable(\n"
    "                intermediate_cache1.view(-1, N), w2, intermediate_cache3, self.w2_scale,\n"
    "                self.w2_bias, sorted_token_ids, expert_ids, self.quant_config.use_fp8_w8a8,\n"
    "                self.per_act_token_quant, self.block_shape,\n"
    "            )\n"
    "        )\n"
    "        if _gemv_fused:\n"
    "            intermediate_cache1 = _resize_cache(workspace13, (num_tokens, top_k_num, N))\n"
    "\n"
    "        # LoRA w13: applied to intermediate_cache1 before activation. When\n"
)
EXPERTS_DISPATCH_OLD = (
    "        a2q_scale: torch.Tensor | None = None\n"
    "\n"
    "        # Fuse SiLU+Mul + FP8 block quantize into a single kernel\n"
)
EXPERTS_DISPATCH_NEW = (
    "        a2q_scale: torch.Tensor | None = None\n"
    "\n"
    f"        # {MARKER}: fused silu*up + quant + down GEMV (decided above)\n"
    "        if _gemv_fused:\n"
    "            _gemv.fused_moe_gemv_act(\n"
    "                intermediate_cache1.view(-1, N), w2, intermediate_cache3, self.w2_scale,\n"
    "                topk_weights, expert_ids, num_tokens_post_padded,\n"
    "                not apply_router_weight_on_input, config,\n"
    "            )\n"
    "            self.moe_sum(intermediate_cache3, output)\n"
    "            return\n"
    "\n"
    "        # Fuse SiLU+Mul + FP8 block quantize into a single kernel\n"
)


def _apply(src: str, pairs: tuple[tuple[str, str], ...]) -> str:
    if MARKER in src:
        return src
    for old, new in pairs:
        if src.count(old) != 1:
            raise SystemExit(f"anchor found {src.count(old)} times (expected 1):\n{old}")
        src = src.replace(old, new, 1)
    return src


def apply(src: str) -> str:
    return _apply(src, ((IMPORT_OLD, IMPORT_NEW), (DISPATCH_OLD, DISPATCH_NEW)))


def apply_experts(src: str) -> str:
    return _apply(src, ((EXPERTS_IMPORT_OLD, EXPERTS_IMPORT_NEW), (EXPERTS_DECIDE_OLD, EXPERTS_DECIDE_NEW),
                        (EXPERTS_DISPATCH_OLD, EXPERTS_DISPATCH_NEW)))


def _run(path: Path, out: Path | None, check: bool, fn) -> None:
    src = path.read_text()
    if MARKER in src:
        print(f"{path}: already patched")
        return
    res = fn(src)
    compile(res, str(path), "exec")
    if check:
        print(f"{path}: patch applies cleanly (not written)")
        return
    dst = out or path
    dst.write_text(res)
    print(f"{dst}: patched")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH, help="fused_moe.py to patch")
    ap.add_argument("--out", type=Path, default=None, help="write the patched fused_moe.py here instead of in place")
    ap.add_argument("--experts-file", type=Path, default=DEFAULT_EXPERTS_PATH, help="experts/triton_moe.py to patch")
    ap.add_argument("--experts-out", type=Path, default=None, help="write the patched triton_moe.py here")
    ap.add_argument("--no-experts", action="store_true", help="patch fused_moe.py only (no fused activation)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    _run(args.file, args.out, args.check, apply)
    if not args.no_experts:
        _run(args.experts_file, args.experts_out, args.check, apply_experts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
