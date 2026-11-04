#!/usr/bin/env python3
"""Engram v2 (the Engram tables on NVMe + a DRAM row cache) in vLLM's deepseek_v41 Engram module, as an idempotent
patcher for the serving image (the sm_120 line's style; run after patch_vllm_moe_cpu_tier.py or alone).

    python3 patch_vllm_engram_v2.py [--install-dir /opt/vllm-moet/engram_v2] [--check]

1. Installs the runtime next to the image's packages: vllm_moet_engram_v2.py, engram_v2.py, engram_v2_cuda.cu,
   engram_v2_abi.h, uring.h, engram_v2.cpp; builds libengram_v2.so (io_uring on raw syscalls, no liburing needed); a
   .pth file makes them importable.
2. Patches, with markers:
   - vllm/models/deepseek_v41/common/engram.py: ParallelEngramEmbedding._allocate_weights (no pinned table in nvme
     mode), ParallelEngramEmbedding.lookup (the publish / wait / landing path + vLLM's own lookup kernel over the
     landing), Engram._create_embedding (registers the table: file, offsets, this rank's row range).
   - vllm/model_executor/model_loader/ep_weight_filter.py: should_skip_weight also skips the names matching
     VLLM_MOET_SKIP_WEIGHT_RE (the engram.embed tensors: 2 x 98 GB per rank never read from disk in nvme mode).
   - vllm/v1/worker/gpu/model_runner.py (the V2 runner; gpu_model_runner.py likewise when present): execute_model
     wrapped with the lookahead hook before and the error-word check after every live step.
Everything is inert unless VLLM_MOET_ENGRAM_MODE=nvme (see vllm_moet_engram_v2.py for the environment).
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
MARK = "[vllm-moet] engram v2"

FILES_RUNTIME = ["vllm_moet_engram_v2.py", "engram_v2.py", "engram_v2_cuda.cu", "engram_v2_abi.h", "uring.h", "engram_v2.cpp",
                 "build.sh"]

# ---- engram.py ----------------------------------------------------------------------------------------------------
ENG_IMPORT_OLD = "logger = init_logger(__name__)\n"
ENG_IMPORT_NEW = ("logger = init_logger(__name__)\n"
                  f"# {MARK}\n"
                  "try:\n"
                  "    import vllm_moet_engram_v2 as _moet_engram_v2\n"
                  "except ImportError:  # the runtime is not installed\n"
                  "    _moet_engram_v2 = None\n")
ENG_ALLOC_OLD = ("    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:\n"
                 "        if self.dp_shared_memory:\n")
ENG_ALLOC_NEW = ("    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:\n"
                 f"        if _moet_engram_v2 is not None and _moet_engram_v2.ENABLED and self.cpu_offload:  # {MARK}: no pinned table\n"
                 "            _moet_w = _moet_engram_v2.allocate_weights(self)\n"
                 "            if _moet_w is not None:\n"
                 "                return _moet_w\n"
                 "        if self.dp_shared_memory:\n")
ENG_LOOKUP_OLD = ("        rows = indices.shape[0] * self.part_n_hash_cols\n"
                  "        if not rows:\n"
                  "            return\n"
                  "        weight, scales = self._storage()\n")
ENG_LOOKUP_NEW = ("        rows = indices.shape[0] * self.part_n_hash_cols\n"
                  "        if not rows:\n"
                  "            return\n"
                  f"        if _moet_engram_v2 is not None and _moet_engram_v2.ENABLED and self.cpu_offload:  # {MARK}: rows from NVMe / the row cache\n"
                  "            if _moet_engram_v2.lookup(self, indices, out, background):\n"
                  "                return\n"
                  "        weight, scales = self._storage()\n")
ENG_CREATE_OLD = ("        engram_config = get_current_vllm_config().engram_config\n"
                  "        assert engram_config is not None\n"
                  "        return ParallelEngramEmbedding(\n"
                  "            layout.num_embeddings[layer_hash_index],\n"
                  "            layout.head_dim,\n"
                  "            tuple(size for order in layout.primes[layer_hash_index] for size in order),\n"
                  "            cpu_offload=engram_config.cpu_offload,\n"
                  "            dp_shared_memory=bool(engram_config.dp_shared_memory),\n"
                  "            use_thp=engram_config.use_thp,\n"
                  "        )\n")
ENG_CREATE_NEW = ("        engram_config = get_current_vllm_config().engram_config\n"
                  "        assert engram_config is not None\n"
                  "        _moet_emb = ParallelEngramEmbedding(\n"
                  "            layout.num_embeddings[layer_hash_index],\n"
                  "            layout.head_dim,\n"
                  "            tuple(size for order in layout.primes[layer_hash_index] for size in order),\n"
                  "            cpu_offload=engram_config.cpu_offload,\n"
                  "            dp_shared_memory=bool(engram_config.dp_shared_memory),\n"
                  "            use_thp=engram_config.use_thp,\n"
                  "        )\n"
                  f"        if _moet_engram_v2 is not None and _moet_engram_v2.ENABLED and engram_config.cpu_offload:  # {MARK}\n"
                  "            _moet_engram_v2.on_embedding_created(_moet_emb, layout, layer_hash_index)\n"
                  "        return _moet_emb\n")

# ---- ep_weight_filter.py -----------------------------------------------------------------------------------------
EPF_DEF_OLD = ("def should_skip_weight(\n"
               "    weight_name: str,\n"
               "    local_expert_ids: set[int] | None,\n"
               ") -> bool:\n")
EPF_DEF_NEW = ("# " + MARK + ": names matching VLLM_MOET_SKIP_WEIGHT_RE are never read from the checkpoint (the Engram tables in\n"
               "# nvme mode: the fetcher reads their rows from the shards at run time)\n"
               "import os as _moet_os\n"
               "import re as _moet_re\n"
               "_moet_skip_re = _moet_re.compile(_moet_os.environ[\"VLLM_MOET_SKIP_WEIGHT_RE\"]) if _moet_os.environ.get(\"VLLM_MOET_SKIP_WEIGHT_RE\") else None\n"
               "\n"
               "\n"
               "def should_skip_weight(\n"
               "    weight_name: str,\n"
               "    local_expert_ids: set[int] | None,\n"
               ") -> bool:\n"
               "    if _moet_skip_re is not None and _moet_skip_re.search(weight_name):  # " + MARK + "\n"
               "        return True\n")

# ---- model_runner.py ---------------------------------------------------------------------------------------------
RUNNER_TAIL = ("\n\n# " + MARK + ": the lookahead before and the error word after a live step (not dummy / profile runs)\n"
               "try:\n"
               "    import vllm_moet_engram_v2 as _moet_engram_v2\n"
               "except ImportError:\n"
               "    _moet_engram_v2 = None\n"
               "if _moet_engram_v2 is not None and _moet_engram_v2.ENABLED:\n"
               "    _moet_eng_orig_execute_model = GPUModelRunner.execute_model\n"
               "\n"
               "    def _moet_eng_execute_model(self, *args, **kwargs):\n"
               "        live = not kwargs.get(\"dummy_run\", False) and not kwargs.get(\"is_profile\", False)\n"
               "        if live:\n"
               "            _moet_engram_v2.runner_before_step(args[0] if args else kwargs.get(\"scheduler_output\"))\n"
               "        out = _moet_eng_orig_execute_model(self, *args, **kwargs)\n"
               "        if live:\n"
               "            _moet_engram_v2.runner_after_step()\n"
               "        return out\n"
               "\n"
               "    GPUModelRunner.execute_model = _moet_eng_execute_model\n")


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
    ap.add_argument("--install-dir", default="/opt/vllm-moet/engram_v2")
    ap.add_argument("--site", default=str(SITE))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    site = Path(args.site)
    targets = {
        "engram": (site / "vllm/models/deepseek_v41/common/engram.py",
                   [(ENG_IMPORT_OLD, ENG_IMPORT_NEW), (ENG_ALLOC_OLD, ENG_ALLOC_NEW), (ENG_LOOKUP_OLD, ENG_LOOKUP_NEW),
                    (ENG_CREATE_OLD, ENG_CREATE_NEW)]),
        "ep_weight_filter": (site / "vllm/model_executor/model_loader/ep_weight_filter.py", [(EPF_DEF_OLD, EPF_DEF_NEW)]),
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
    inst = Path(args.install_dir)
    inst.mkdir(parents=True, exist_ok=True)
    for f in FILES_RUNTIME:
        shutil.copy2(HERE / f, inst / f)
    env = dict(os.environ, OUT=str(inst / "libengram_v2.so"))
    subprocess.run(["bash", str(inst / "build.sh")], check=True, env=env)
    (site / "vllm_moet_engram_v2.pth").write_text(str(inst) + "\n")
    for path, out in patched.items():
        if path.read_text() != out:
            path.write_text(out)
            print(f"patched {path}")
        else:
            print(f"{path}: already patched")
    print(f"runtime installed in {inst}; enable with VLLM_MOET_ENGRAM_MODE=nvme")
    return 0


if __name__ == "__main__":
    sys.exit(main())
