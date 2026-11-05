#!/usr/bin/env bash
# GPU-side check of a Ripple image (what docker build cannot do without libcuda): the CUDA ops library loads and runs,
# the NVFP4 GEMM op resolves, the model classes are registered, the checkpoint config parses as RicursiveConfig.
# torch.ops namespaces resolve lazily (dir() lists only ops already touched), so ops are called / resolved, not listed.
# One GPU, ~10 s.
# usage: tools/ripple_sm120/image_check.sh IMAGE GPU_UUID_OR_INDEX MODEL_DIR [DRAFT_DIR]
set -euo pipefail
IMAGE=${1:?image}; GPU=${2:?gpu}; MODEL_DIR=${3:?model dir}; DRAFT_DIR=${4:-}
MOUNTS=(-v "$MODEL_DIR:/model:ro")
[[ -n "$DRAFT_DIR" ]] && MOUNTS+=(-v "$DRAFT_DIR:/draft:ro")
docker run --rm -i --gpus "\"device=$GPU\"" "${MOUNTS[@]}" --entrypoint python3 "$IMAGE" - <<'EOF'
import json, os, torch, vllm
import vllm._custom_ops  # noqa: F401  (loads _C_stable_libtorch / _moe_C_stable_libtorch through the platform)
x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16); out = torch.empty(4, 32, device="cuda", dtype=torch.bfloat16)
torch.ops._C.silu_and_mul(out, x)
assert torch.allclose(out, torch.nn.functional.silu(x[:, :32]) * x[:, 32:], atol=1e-2), "silu_and_mul mismatch"
fp4 = torch.ops._C.cutlass_scaled_fp4_mm  # resolves or raises
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert "RicursiveForCausalLM" in archs and "DFlash2RicursiveDraftModel" in archs
from vllm.transformers_utils.config import get_config
cfg = get_config("/model", trust_remote_code=False)
assert type(cfg).__name__ == "RicursiveConfig", type(cfg)
q = getattr(cfg, "quantization_config", None) or {}
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("ops ok (silu_and_mul ran, cutlass_scaled_fp4_mm resolves) | archs ok | config", cfg.model_type, cfg.num_hidden_layers,
      "layers,", sum(t == "full_attention" for t in cfg.layer_types), "full +",
      sum(t == "sliding_attention" for t in cfg.layer_types), "sliding (window", cfg.sliding_window, ") | quant",
      q.get("quant_algo") or q.get("quant_method") or "none")
if os.path.exists("/draft/config.json"):
    d = json.load(open("/draft/config.json"))
    print("draft", d["architectures"], "layers", d["num_hidden_layers"], "block", d["dflash_config"]["block_size"],
          "no_rope_layers", d.get("no_rope_layers"), "taps", d["dflash_config"]["target_layer_ids"])
print("IMAGE_CHECK_OK")
EOF
