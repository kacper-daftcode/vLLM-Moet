#!/usr/bin/env python3
"""The CPU expert tier in vLLM's b12x MoE path (phase 2.3), as an idempotent patcher for the serving image.

    python3 patch_vllm_moe_cpu_tier.py [--install-dir /opt/vllm-moet/cpu_tier] [--arch znver4] [--check]

1. Installs the runtime next to the image's packages: cpu_tier.py, placement_manager.py, vllm_moet_cpu_tier.py, the
   extension sources (cpu_tier_cuda.cu, cpu_tier_abi.h) and builds libcpu_tier.so (cpu_tier.cpp + the b12x-exact
   kernel cpu_moe_b12x.cpp) for this machine's compiler; a .pth file makes them importable.
2. Patches, with markers:
   - vllm/model_executor/layers/quantization/mxfp4.py: Mxfp4MoEMethod.create_weights asks the tier how many expert
     rows a target layer gets (store mode: the resident slots only; the layer's loader maps experts to slots).
   - vllm/model_executor/layers/fused_moe/b12x.py: process_weights_after_loading hands the layer's experts to the
     tier (records into pinned RAM or from the store, the resident rows back for prepare_weights); apply() announces
     the layer to the tier and gets the ids the kernels run with.
   - flashinfer/experimental/b12x/moe/_shared/kernels/w4a8_compact_micro.py: after b12x's rows quant the tier
     publishes the step's missing pairs and remaps the ids (missing -> -1) for the compact kernel (and the grouped /
     fused FC2 of the sm120 image); before the route rows go to TopKSum the tier's rows are injected.
   - vllm/v1/worker/gpu/model_runner.py (the V2 runner; the V1 gpu_model_runner.py likewise when present):
     execute_model wrapped with the placement manager's before/after-step hooks (promotions, the error word, KPIs)
     for every live forward - graph replays included, dummy / profile runs excluded.
Everything is inert unless VLLM_MOET_MOE_CPU_TIER=1 (see vllm_moet_cpu_tier.py for the environment).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = Path("/usr/local/lib/python3.12/dist-packages")
MARK = "[vllm-moet] cpu tier"

FILES_RUNTIME = ["cpu_tier.py", "placement_manager.py", "vllm_moet_cpu_tier.py", "cpu_tier_cuda.cu", "cpu_tier_abi.h",
                 "cpu_tier.cpp", "build.sh"]

# ---- vllm b12x.py -----------------------------------------------------------------------------------------------
B12X_IMPORT_OLD = "import vllm.model_executor.layers.fused_moe.modular_kernel as mk\n"
B12X_IMPORT_NEW = ("import vllm.model_executor.layers.fused_moe.modular_kernel as mk\n"
                   f"# {MARK}\n"
                   "try:\n"
                   "    import vllm_moet_cpu_tier as _moet_cpu_tier\n"
                   "except ImportError:  # the runtime is not installed\n"
                   "    _moet_cpu_tier = None\n")
B12X_PREP_OLD = ("        self._refresh_quant_config(layer)\n"
                 "        prepared = self._prepare_experts(\n"
                 "            w1=layer.w13_weight,\n"
                 "            w2=layer.w2_weight,\n")
B12X_PREP_NEW = ("        self._refresh_quant_config(layer)\n"
                 f"        # {MARK}: the resident subset is what b12x prepares; the rest lives in the tier's pinned store\n"
                 "        _moet_w1, _moet_w2 = layer.w13_weight, layer.w2_weight\n"
                 "        _moet_sub = (_moet_cpu_tier.on_process_weights(self, layer, layer.w13_weight, layer.w2_weight,\n"
                 "                                                       self.w1_scale, self.w2_scale)\n"
                 "                     if _moet_cpu_tier is not None and _moet_cpu_tier.ENABLED else None)\n"
                 "        if _moet_sub is not None:\n"
                 "            _moet_w1, _moet_w2, self.quant_config._w1.scale, self.quant_config._w2.scale = _moet_sub\n"
                 "        prepared = self._prepare_experts(\n"
                 "            w1=_moet_w1,\n"
                 "            w2=_moet_w2,\n")
B12X_APPLY_OLD = ("        prepared = self._prepared()\n"
                  "        topk_ids = _normalize_topk_ids(topk_ids)\n")
B12X_APPLY_NEW = ("        prepared = self._prepared()\n"
                  "        topk_ids = _normalize_topk_ids(topk_ids)\n"
                  f"        if _moet_cpu_tier is not None and _moet_cpu_tier.ENABLED:  # {MARK}\n"
                  "            topk_ids = _moet_cpu_tier.on_apply(self, hidden_states, topk_ids)\n")
B12X_APPLY_END_OLD = ("            output=output,\n"
                      "            unit_scale_contract=self._quant_mode == \"w4a16\",\n"
                      "        )\n"
                      "\n"
                      "    def moe_sum(")
B12X_APPLY_END_NEW = ("            output=output,\n"
                      "            unit_scale_contract=self._quant_mode == \"w4a16\",\n"
                      "        )\n"
                      f"        if _moet_cpu_tier is not None and _moet_cpu_tier.ENABLED:  # {MARK}: the layer context ends here\n"
                      "            _moet_cpu_tier.on_apply_end(self, hidden_states, topk_weights, output, _run_b12x_moe_plan, _b12x_scratch_nbytes)\n"
                      "            _moet_cpu_tier.on_apply_done()\n"
                      "\n"
                      "    def moe_sum(")

# ---- flashinfer w4a8_compact_micro.py ---------------------------------------------------------------------------
CM_IMPORT_OLD = "import cutlass\nimport cutlass.cute as cute\nimport torch\n"
CM_IMPORT_NEW = ("import cutlass\nimport cutlass.cute as cute\nimport torch\n"
                 f"# {MARK}\n"
                 "try:\n"
                 "    import vllm_moet_cpu_tier as _moet_cpu_tier\n"
                 "except ImportError:\n"
                 "    _moet_cpu_tier = None\n")
CM_FUSED_OLD = ("        _moet_moe_fused_module().moe_fused(\n"
                "            values, scale_rows, w13, w13_scales, projections, intermediate, w2, w2_scales, topk_ids.reshape(-1),\n"
                "            topk_weights.reshape(-1), alpha1, input_scale, alpha2, down_scale, route, num_tokens * int(num_topk),\n"
                "            int(num_topk), cap * int(num_topk), k, n, float(swiglu_limit),\n"
                "        )\n"
                "        return route.narrow(0, 0, num_tokens * int(num_topk))\n")
CM_FUSED_NEW = ("        _moet_ids = _moet_cpu_tier.before_compiled(topk_ids, topk_weights, values, scale_rows, num_tokens, w13, w13_scales, w2, w2_scales) if _moet_cpu_tier is not None and _moet_cpu_tier.ENABLED else topk_ids  # " + MARK + "\n"
                "        _moet_moe_fused_module().moe_fused(\n"
                "            values, scale_rows, w13, w13_scales, projections, intermediate, w2, w2_scales, _moet_ids.reshape(-1),\n"
                "            topk_weights.reshape(-1), alpha1, input_scale, alpha2, down_scale, route, num_tokens * int(num_topk),\n"
                "            int(num_topk), cap * int(num_topk), k, n, float(swiglu_limit),\n"
                "        )\n"
                "        if _moet_cpu_tier is not None and _moet_cpu_tier.ENABLED:\n"
                "            _moet_cpu_tier.after_compiled(route)\n"
                "        return route.narrow(0, 0, num_tokens * int(num_topk))\n")
CM_IDS_OLD = ("    ids = topk_ids.reshape(-1)\n"
              "    weights = topk_weights.reshape(-1)\n"
              "    compiled(\n")
CM_IDS_NEW = ("    # " + MARK + ": publish the missing pairs, the compact kernel sees them as -1 (zero rows)\n"
              "    _moet_ids = _moet_cpu_tier.before_compiled(topk_ids, topk_weights, values, scale_rows, num_tokens, w13, w13_scales, w2, w2_scales) if _moet_cpu_tier is not None and _moet_cpu_tier.ENABLED else topk_ids\n"
              "    ids = _moet_ids.reshape(-1)\n"
              "    weights = topk_weights.reshape(-1)\n"
              "    compiled(\n")
CM_RET_OLD = ("            num_tokens * int(num_topk), cap * int(num_topk),\n"
              "        )\n"
              "    return route.narrow(0, 0, num_tokens * int(num_topk))\n")
CM_RET_NEW = ("            num_tokens * int(num_topk), cap * int(num_topk),\n"
              "        )\n"
              "    if _moet_cpu_tier is not None and _moet_cpu_tier.ENABLED:  # " + MARK + ": the host's rows before TopKSum\n"
              "        _moet_cpu_tier.after_compiled(route)\n"
              "    return route.narrow(0, 0, num_tokens * int(num_topk))\n")

# ---- mxfp4.py: Mxfp4MoEMethod.create_weights ------------------------------------------------------------------------
MX_IMPORT_OLD = "from vllm.logger import init_logger\n"
MX_IMPORT_NEW = ("from vllm.logger import init_logger\n"
                 f"# {MARK}\n"
                 "try:\n"
                 "    import vllm_moet_cpu_tier as _moet_cpu_tier\n"
                 "except ImportError:\n"
                 "    _moet_cpu_tier = None\n")
MX_CREATE_OLD = ("        return mxfp4_weight_loader\n"
                 "\n"
                 "    def create_weights(\n"
                 "        self,\n"
                 "        layer: RoutedExperts,\n"
                 "        num_experts: int,\n"
                 "        hidden_size: int,\n"
                 "        intermediate_size_per_partition: int,\n"
                 "        params_dtype: torch.dtype,\n"
                 "        **extra_weight_attrs,\n"
                 "    ):\n"
                 "        self.num_experts = num_experts\n")
MX_CREATE_NEW = ("        return mxfp4_weight_loader\n"
                 "\n"
                 "    def create_weights(\n"
                 "        self,\n"
                 "        layer: RoutedExperts,\n"
                 "        num_experts: int,\n"
                 "        hidden_size: int,\n"
                 "        intermediate_size_per_partition: int,\n"
                 "        params_dtype: torch.dtype,\n"
                 "        **extra_weight_attrs,\n"
                 "    ):\n"
                 f"        if _moet_cpu_tier is not None and _moet_cpu_tier.ENABLED:  # {MARK}: resident slots only\n"
                 "            num_experts = _moet_cpu_tier.on_create_weights(layer, num_experts, hidden_size,\n"
                 "                                                          intermediate_size_per_partition)\n"
                 "        self.num_experts = num_experts\n")

# ---- model_runner.py (V2: vllm/v1/worker/gpu/model_runner.py; V1: vllm/v1/worker/gpu_model_runner.py) -------------
RUNNER_TAIL = ("\n\n# " + MARK + ": the placement manager's hooks around a live step (not dummy / profile runs; graph\n"
               "# capture does not go through execute_model)\n"
               "try:\n"
               "    import vllm_moet_cpu_tier as _moet_cpu_tier\n"
               "except ImportError:\n"
               "    _moet_cpu_tier = None\n"
               "if _moet_cpu_tier is not None and _moet_cpu_tier.ENABLED:\n"
               "    _moet_orig_execute_model = GPUModelRunner.execute_model\n"
               "\n"
               "    def _moet_execute_model(self, *args, **kwargs):\n"
               "        live = not kwargs.get(\"dummy_run\", False) and not kwargs.get(\"is_profile\", False)\n"
               "        if live:\n"
               "            _moet_cpu_tier.runner_before_step()\n"
               "        out = _moet_orig_execute_model(self, *args, **kwargs)\n"
               "        if live:\n"
               "            _moet_cpu_tier.runner_after_step()\n"
               "        return out\n"
               "\n"
               "    GPUModelRunner.execute_model = _moet_execute_model\n")


def patch_text(src: str, pairs: list[tuple[str, str]], name: str) -> str:
    if MARK in src:
        return src
    for old, _ in pairs:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"{name}: anchor found {n} times (expected 1):\n{old}")
    for old, new in pairs:
        src = src.replace(old, new, 1)
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install-dir", default="/opt/vllm-moet/cpu_tier")
    ap.add_argument("--site", default=str(SITE))
    ap.add_argument("--arch", default="znver4")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    site = Path(args.site)
    targets = {
        "mxfp4": (site / "vllm/model_executor/layers/quantization/mxfp4.py",
                  [(MX_IMPORT_OLD, MX_IMPORT_NEW), (MX_CREATE_OLD, MX_CREATE_NEW)]),
        "b12x": (site / "vllm/model_executor/layers/fused_moe/b12x.py",
                 [(B12X_IMPORT_OLD, B12X_IMPORT_NEW), (B12X_PREP_OLD, B12X_PREP_NEW), (B12X_APPLY_OLD, B12X_APPLY_NEW),
                  (B12X_APPLY_END_OLD, B12X_APPLY_END_NEW)]),
        "compact_micro": (site / "flashinfer/experimental/b12x/moe/_shared/kernels/w4a8_compact_micro.py",
                          [(CM_IMPORT_OLD, CM_IMPORT_NEW), (CM_FUSED_OLD, CM_FUSED_NEW), (CM_IDS_OLD, CM_IDS_NEW), (CM_RET_OLD, CM_RET_NEW)]),
    }
    runners = [site / "vllm/v1/worker/gpu/model_runner.py", site / "vllm/v1/worker/gpu_model_runner.py"]
    patched = {}
    for name, (path, pairs) in targets.items():
        src = path.read_text()
        out = patch_text(src, pairs, name)
        compile(out, str(path), "exec")
        patched[path] = out
    if not runners[0].exists():
        raise SystemExit(f"{runners[0]}: the V2 model runner is missing")
    for runner in runners:
        if not runner.exists():
            continue
        rsrc = runner.read_text()
        if MARK not in rsrc:
            if "class GPUModelRunner(" not in rsrc:
                raise SystemExit(f"{runner}: no GPUModelRunner class")
            rsrc = rsrc + RUNNER_TAIL
            compile(rsrc, str(runner), "exec")
        patched[runner] = rsrc
    if args.check:
        print("patches apply cleanly (not written); runtime install skipped")
        return 0
    # the runtime
    inst = Path(args.install_dir)
    inst.mkdir(parents=True, exist_ok=True)
    for f in FILES_RUNTIME:
        shutil.copy2(HERE / f, inst / f)
    (inst / "cpu_moe").mkdir(exist_ok=True)
    shutil.copy2(HERE.parent / "cpu_moe" / "cpu_moe_b12x.cpp", inst / "cpu_moe" / "cpu_moe_b12x.cpp")
    env = dict(os.environ, ARCH=args.arch, OUT=str(inst / "libcpu_tier.so"), KERNEL_SRC=str(inst / "cpu_moe" / "cpu_moe_b12x.cpp"))
    subprocess.run(["bash", str(inst / "build.sh")], check=True, env=env)
    (site / "vllm_moet_cpu_tier.pth").write_text(str(inst) + "\n")
    for path, out in patched.items():
        if path.read_text() != out:
            path.write_text(out)
            print(f"patched {path}")
        else:
            print(f"{path}: already patched")
    print(f"runtime installed in {inst} (libcpu_tier.so for {args.arch}); enable with VLLM_MOET_MOE_CPU_TIER=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
