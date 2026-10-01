#!/usr/bin/env python3
"""DeepSeek-V4.1 sm_120: the small-M BF16 GEMMs of the decode step on tools/dsv41_sm120/bf16_gemm (cuBLAS's bits).

In the served decode graph (RTX PRO 6000, one stream) four BF16 projections run on cuBLAS at decode token counts:
  - the lightning indexer's weights_proj (5120 -> 32, bf16 out) in the 8 index-source layers, on vLLM's second aux
    stream: cuBLAS takes a 16x16 kernel on 2 CTAs, ~35 us next to the main stream's GEMVs, and the main stream waits
    ~26 us for it before the q/kv norm of every one of those layers (~210 us per step);
  - the indexer's wk (512 -> 128, bf16 out) in the 4 kv-source layers, on the main stream;
  - the compressor's fused_wkv_wgate (5120 -> 1024 / 512, fp32 out) on the first aux stream;
  - the MoE router (GateLinear tier 4, 5120 -> 384, the drafter's 5120 -> 128, fp32 out): GEMM + splitKreduce.
bf16_gemm/ reproduces cuBLAS's result for each of them bit for bit in one launch (per-shape calibration against
cuBLAS at load time, see bf16_replica_gemm_sm120.py), so outputs are unchanged.

Patched files:
  vllm/models/deepseek_v41/attention.py: the compressor_kv_score and indexer_weights_proj closures of the attention
      input GEMMs, the indexer's wk call in _produce_k, and the wk shape registered at indexer construction (wk is
      first called inside graph capture: the profile run skips it).
  vllm/model_executor/layers/fused_moe/router/gate_linear.py: tier 4 (cuBLAS bf16 -> fp32).
The indexer / wk layers take the kernels only when they would run F.linear (no bias, default unquantized GEMM, no
VLLM_BATCH_INVARIANT); otherwise, and whenever the module is off or a token count is not calibrated, the original
op runs.
Runtime switches: VLLM_MOET_BF16_GEMM=0 (all sites on cuBLAS), VLLM_MOET_BF16_GEMM_SITES (default
indexer,wk,compressor; add router), VLLM_MOET_BF16_GEMM_SPLITK_MAX_M (8).

Idempotent, anchor-based. Usage:
    python3 patch_vllm_bf16_gemm_sm120.py [--attention-file PATH] [--gate-file PATH] [--kernel-dir DIR] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
DEFAULT_ATTENTION = SITE / "models/deepseek_v41/attention.py"
DEFAULT_GATE = SITE / "model_executor/layers/fused_moe/router/gate_linear.py"
DEFAULT_KERNEL_DIR = "/opt/vllm-moet/dsv41_sm120/bf16_gemm"
MARKER = "# [vllm-moet] sm_120 BF16 small-M GEMM"

HELPER = '''

{marker}: cuBLAS's bits from one launch (tools/dsv41_sm120/bf16_gemm).
_MOET_BF16 = None  # None = not resolved yet, False = off / unavailable, else the module


def _moet_bf16():
    global _MOET_BF16
    if _MOET_BF16 is None:
        import os
        import sys

        from vllm.logger import init_logger as _moet_init_logger

        _MOET_BF16 = False
        if os.environ.get("VLLM_MOET_BF16_GEMM", "1") == "1" and (
            current_platform.is_cuda() and current_platform.is_device_capability_family(120)
        ):
            kernel_dir = os.environ.get("VLLM_MOET_BF16_GEMM_DIR", "{kernel_dir}")
            if kernel_dir not in sys.path:
                sys.path.insert(0, kernel_dir)
            try:
                import bf16_replica_gemm_sm120 as _m

                _m._ext()  # build / load the extension now, not inside the first forward
                _MOET_BF16 = _m
                _moet_init_logger(__name__).info_once(
                    "vllm-moet sm_120 BF16 small-M GEMM: cuBLAS's bits in one launch for %s "
                    "(VLLM_MOET_BF16_GEMM=0 turns it off)",
                    ",".join(sorted(_m.enabled_sites())),
                )
            except Exception as exc:  # noqa: BLE001
                _moet_init_logger(__name__).warning(
                    "vllm-moet sm_120 BF16 small-M GEMM unavailable, cuBLAS stays: %r", exc
                )
    return _MOET_BF16 or None


def _moet_bf16_mm(x, w, out_f32, site):
    """torch.mm(x, w.T, out_dtype=torch.float32) if out_f32 else F.linear(x, w), bit for bit."""
    m = _moet_bf16()
    if m is not None:
        return m.replica_mm(x, w, out_f32, site)
    if out_f32:
        return torch.mm(x, w.T, out_dtype=torch.float32)
    return torch.nn.functional.linear(x, w)
'''

ATTN_HELPER_EXTRA = '''

def _moet_bf16_linear(layer, x, site):
    """layer(x) for an unquantized ReplicatedLinear: the kernels only where it would run F.linear(x, weight)."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    qm = layer.quant_method
    if (
        layer.bias is None
        and type(qm) is UnquantizedLinearMethod
        and getattr(qm, "_gemm_impl", None) is default_unquantized_gemm
        and not envs.VLLM_BATCH_INVARIANT
    ):
        return _moet_bf16_mm(x, layer.weight, False, site)
    out = layer(x)
    return out[0] if isinstance(out, tuple) else out


def _moet_bf16_register(n, k, out_f32, ldx, site):
    m = _moet_bf16()
    if m is not None:
        m.register(n, k, out_f32, ldx, site)
'''

ATTN_ANCHOR_HELPER = "logger = init_logger(__name__)\n"
ATTN_COMPRESSOR_OLD = (
    "            def compressor_kv_score() -> torch.Tensor:\n"
    "                return torch.mm(\n"
    "                    hidden_states,\n"
    "                    compressor.fused_wkv_wgate.weight.T,\n"
    "                    out_dtype=torch.float32,\n"
    "                )\n"
)
ATTN_COMPRESSOR_NEW = (
    "            def compressor_kv_score() -> torch.Tensor:\n"
    "                return _moet_bf16_mm(\n"
    "                    hidden_states, compressor.fused_wkv_wgate.weight, True, \"compressor\"\n"
    "                )\n"
)
ATTN_INDEXER_OLD = (
    "            def indexer_weights_proj() -> torch.Tensor:\n"
    "                # ReplicatedLinear returns (output, bias); bias is None.\n"
    "                weights, _ = indexer.weights_proj(hidden_states)\n"
    "                return weights\n"
)
ATTN_INDEXER_NEW = (
    "            def indexer_weights_proj() -> torch.Tensor:\n"
    "                return _moet_bf16_linear(indexer.weights_proj, hidden_states, \"indexer\")\n"
)
ATTN_WK_INIT_OLD = (
    '                prefix=f"{prefix}.wk",\n'
    "            )\n"
    "            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)\n"
)
ATTN_WK_INIT_NEW = (
    '                prefix=f"{prefix}.wk",\n'
    "            )\n"
    "            # first called inside graph capture (the profile run skips _produce_k)\n"
    '            _moet_bf16_register(self.head_dim, main_head_dim, False, main_head_dim, "wk")\n'
    "            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)\n"
)
ATTN_WK_CALL_OLD = "        k_pre, _ = self.wk(latent)\n"
ATTN_WK_CALL_NEW = '        k_pre = _moet_bf16_linear(self.wk, latent, "wk")\n'

GATE_ANCHOR_HELPER = "from vllm.utils.torch_utils import direct_register_custom_op\n"
GATE_TIER4_OLD = (
    "        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:\n"
    "            output = torch.mm(x, self.weight.T, out_dtype=torch.float32)\n"
    "            return output, None\n"
)
GATE_TIER4_NEW = (
    "        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:\n"
    '            output = _moet_bf16_mm(x, self.weight, True, "router")\n'
    "            return output, None\n"
)


def _replace(src: str, old: str, new: str, name: str) -> str:
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"{name}: anchor found {n} times (expected 1)")
    return src.replace(old, new, 1)


def patch_attention(src: str, kernel_dir: str) -> str:
    if MARKER in src:
        return src
    helper = HELPER.replace("{marker}", MARKER).replace("{kernel_dir}", kernel_dir) + ATTN_HELPER_EXTRA
    out = _replace(src, ATTN_ANCHOR_HELPER, ATTN_ANCHOR_HELPER + helper, "attention helper")
    out = _replace(out, ATTN_COMPRESSOR_OLD, ATTN_COMPRESSOR_NEW, "compressor_kv_score")
    out = _replace(out, ATTN_INDEXER_OLD, ATTN_INDEXER_NEW, "indexer_weights_proj")
    out = _replace(out, ATTN_WK_INIT_OLD, ATTN_WK_INIT_NEW, "indexer wk init")
    out = _replace(out, ATTN_WK_CALL_OLD, ATTN_WK_CALL_NEW, "indexer wk call")
    if "import vllm.envs as envs" not in out:
        raise SystemExit("attention.py: `import vllm.envs as envs` not found (the helper reads VLLM_BATCH_INVARIANT)")
    return out


def patch_gate(src: str, kernel_dir: str) -> str:
    if MARKER in src:
        return src
    helper = HELPER.replace("{marker}", MARKER).replace("{kernel_dir}", kernel_dir)
    out = _replace(src, GATE_ANCHOR_HELPER, GATE_ANCHOR_HELPER + helper, "gate_linear helper")
    return _replace(out, GATE_TIER4_OLD, GATE_TIER4_NEW, "GateLinear tier 4")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention-file", type=Path, default=DEFAULT_ATTENTION)
    ap.add_argument("--gate-file", type=Path, default=DEFAULT_GATE)
    ap.add_argument("--kernel-dir", default=DEFAULT_KERNEL_DIR)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    for path, fn in ((args.attention_file, patch_attention), (args.gate_file, patch_gate)):
        src = path.read_text()
        if MARKER in src:
            print(f"{path}: already patched")
            continue
        patched = fn(src, args.kernel_dir)
        compile(patched, str(path), "exec")
        if args.check:
            print(f"{path}: patch applies cleanly (not written)")
            continue
        path.write_text(patched)
        print(f"{path}: patched (sm_120 BF16 small-M GEMM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
