#!/usr/bin/env python3
"""Ripple step 3: two-microbatch decode for a TP-only dense model - one half's all-reduces under the other half's GEMMs.

Where: vllm/v1/worker/gpu_worker.py, vllm/v1/worker/gpu/{ubatch_utils,dp_utils}.py, vllm/distributed/parallel_state.py
(on top of step 1's hook) - vLLM 03bd3c68, model runner V2; and tools/dsv41_sm120/p2p_ar/p2p_allreduce_sm120.py (an `out`
for `_run`, backward compatible).

Why: at 8-32 requests the decode step of Ripple TP4 on 4 PCIe GPUs is 33-47 % all-reduce (170 per step, on the PCIe
floor: 34-36 GB/s per rank) on one stream, nothing overlapping it. vLLM's dual batch overlap (DBO) has the whole
machinery for the V2 runner - UBatchRunner (two threads, a compute and a comm stream, per-microbatch attention
metadata and forward contexts, FULL CUDA graphs captured with both streams), `_maybe_ubatch_twin` descriptors, the
`vllm.v1.worker.ubatching` hand-off protocol - but gates it on DP > 1 and a DeepEP / NIXL all-to-all backend (its
yield points are inside the MoE all-to-all). For a dense TP model nothing yields, so the threads would just run one
after the other.

What: behind VLLM_MOET_DECODE_UBATCH=2 (default off; DP = 1, TP >= 2, PP = 1; the launcher also sets
VLLM_USE_V2_MODEL_RUNNER=1, because vLLM's V2-runner auto-selection still lists DBO as unsupported and a re-validated
config - the drafter's `replace(vllm_config, ...)` - would fall back to V1 and fail):
  * config/vllm.py: the DBO all-to-all-backend requirement does not apply (the yield is at the TP all-reduce);
  * gpu_worker: before the workspace manager and the runner are built, `enable_dbo = True` with
    dbo_decode_token_threshold = VLLM_MOET_DECODE_UBATCH_MIN_TOKENS (default 128 tokens: 16 x k 7 or more; measured in
    isolation the halves pay back from ~128 rows up) and dbo_prefill_token_threshold = never - so the workspace lanes,
    the attention metadata builders and the CUDA graph twins are sized for two microbatches as they would be with DP;
  * maybe_build_ubatch_runner: builds the UBatchRunner without DP;
  * dispatch_cg_and_sync_dp (DP = 1): a uniform decode above the threshold takes its microbatched FULL graph when one
    was captured, else runs as one batch (never eagerly microbatched: prefill and small decodes are untouched);
  * the TP all-reduce (GroupCoordinator._all_reduce_out_place, step 1's site): inside a microbatched forward the
    output is allocated on the compute stream, the thread yields to the other microbatch and comes back on the comm
    stream once its input is ready, launches the all-reduce there (the P2P push with NCCL's summation order when it
    applies - at 32 requests x k 7 each half is 128 rows, the push's sweet spot - PyNCCL otherwise), yields again and
    resumes on the compute stream once the sum has landed. Per layer: attention(A) | attention(B) while AR(A) |
    MLP(A) while AR(B) | MLP(B) while AR'(A) ... - the DBO schedule of two uniform halves on request boundaries.
Everything else is vLLM's own DBO code path; the drafter runs after the merged step as before. The bits of each
request are those of a batch of its half's size (GEMM / attention shapes of M/2), as any batch composition change.

Usage: python3 patch_vllm_decode_ubatch_tp_sm120.py [--root DIR] [--p2p-dir DIR] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
P2P_DIR = Path("/opt/vllm-moet/dsv41_sm120/p2p_ar")
MARKER = "[vllm-moet] step 3"

GPU_WORKER = [
    (
        "logger = init_logger(__name__)\n\n\ndef _min_across_ranks(value: int) -> int:\n",
        "logger = init_logger(__name__)\n\n\n"
        "def _moet_maybe_enable_decode_ubatching(vllm_config: VllmConfig) -> None:\n"
        f'    """{MARKER}: two-microbatch decode for a TP-only dense model (VLLM_MOET_DECODE_UBATCH=2): the DBO\n'
        "    machinery sized for two microbatches before the workspace, the runner, the attention metadata builders and\n"
        "    the CUDA graph twins are built; the TP all-reduce of one microbatch runs under the GEMMs of the other\n"
        '    (GroupCoordinator._all_reduce_out_place). Decode batches of at least VLLM_MOET_DECODE_UBATCH_MIN_TOKENS\n'
        '    tokens (default 128) with a captured microbatched graph take it; prefill never microbatches."""\n'
        '    if os.environ.get("VLLM_MOET_DECODE_UBATCH", "0") != "2":\n'
        "        return\n"
        "    pc = vllm_config.parallel_config\n"
        "    if (\n"
        "        pc.use_ubatching\n"
        "        or pc.data_parallel_size != 1\n"
        "        or pc.tensor_parallel_size < 2\n"
        "        or pc.pipeline_parallel_size != 1\n"
        "    ):\n"
        "        return\n"
        '    min_tokens = int(os.environ.get("VLLM_MOET_DECODE_UBATCH_MIN_TOKENS", "128"))\n'
        "    pc.enable_dbo = True\n"
        "    pc.dbo_decode_token_threshold = max(min_tokens, 2 * vllm_config.uniform_decode_query_len)\n"
        "    pc.dbo_prefill_token_threshold = 1 << 30\n"
        "    logger.info(\n"
        '        "vllm-moet decode microbatching (TP %d, two microbatches): uniform decodes of >= %d tokens run as two "\n'
        '        "halves, the all-reduces of one under the GEMMs of the other",\n'
        "        pc.tensor_parallel_size,\n"
        "        pc.dbo_decode_token_threshold,\n"
        "    )\n"
        "\n\n"
        "def _min_across_ranks(value: int) -> int:\n",
    ),
    (
        "    def _init_workspace_and_model_runner(self) -> None:\n"
        '        """Set up the workspace manager, build the model runner, report usage."""\n'
        "        # DSpark target and draft CUDA graphs retain workspace views concurrently.\n"
        "        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1\n",
        "    def _init_workspace_and_model_runner(self) -> None:\n"
        '        """Set up the workspace manager, build the model runner, report usage."""\n'
        f"        _moet_maybe_enable_decode_ubatching(self.vllm_config)  # {MARKER}\n"
        "        # DSpark target and draft CUDA graphs retain workspace views concurrently.\n"
        "        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1\n",
    ),
]

UBATCH_UTILS = [
    (
        "import threading\n",
        "import os\n"
        "import threading\n",
    ),
    (
        "    parallel_config = vllm_config.parallel_config\n"
        "    if not parallel_config.use_ubatching or parallel_config.data_parallel_size <= 1:\n"
        "        return None\n",
        "    parallel_config = vllm_config.parallel_config\n"
        "    if not parallel_config.use_ubatching:\n"
        "        return None\n"
        f"    # {MARKER}: a TP-only decode microbatches too (VLLM_MOET_DECODE_UBATCH=2: the TP all-reduces of one\n"
        "    # microbatch under the GEMMs of the other); otherwise the DP handshake is what agrees on microbatching.\n"
        "    if parallel_config.data_parallel_size <= 1 and (\n"
        '        os.environ.get("VLLM_MOET_DECODE_UBATCH", "0") != "2"\n'
        "    ):\n"
        "        return None\n",
    ),
]

UBATCH_UTILS += [
    (
        "            forward_contexts.append(\n"
        "                create_forward_context(\n"
        "                    attn_metadata[i],\n"
        "                    self.vllm_config,\n"
        "                    dp_metadata=DPMetadata.make(\n"
        "                        self.parallel_config, num_tokens, num_tokens_across_dp\n"
        "                    ),\n",
        "            forward_contexts.append(\n"
        "                create_forward_context(\n"
        "                    attn_metadata[i],\n"
        "                    self.vllm_config,\n"
        f"                    # {MARKER}: no DP metadata without DP, as the single-batch forward context\n"
        "                    dp_metadata=(\n"
        "                        DPMetadata.make(\n"
        "                            self.parallel_config, num_tokens, num_tokens_across_dp\n"
        "                        )\n"
        "                        if dp_size > 1\n"
        "                        else None\n"
        "                    ),\n",
    ),
]

UBATCH_UTILS += [
    (
        "        self.comm_stream = torch.cuda.Stream(device=device)\n",
        f"        # {MARKER}: the all-reduce kernels of one microbatch share the SMs with the other's GEMMs; a high-priority\n"
        "        # comm stream gets their blocks scheduled as GEMM blocks retire instead of queueing behind them\n"
        "        # (VLLM_MOET_DECODE_UBATCH_COMM_PRIORITY: -1 = highest (default), 0 = vLLM's default stream priority).\n"
        "        self.comm_stream = torch.cuda.Stream(\n"
        '            device=device, priority=int(os.environ.get("VLLM_MOET_DECODE_UBATCH_COMM_PRIORITY", "-1"))\n'
        "        )\n",
    ),
]

DP_UTILS = [
    (
        "    if dp_size == 1:\n"
        "        return batch_desc, None\n",
        "    if dp_size == 1:\n"
        f"        # {MARKER}: a TP-only uniform decode above the DBO decode threshold takes its microbatched FULL\n"
        "        # graph when one is captured (the all-reduces of one half under the GEMMs of the other); any other\n"
        "        # shape, or none captured, runs as one batch - never eagerly microbatched.\n"
        "        if (\n"
        "            allow_ubatching\n"
        "            and parallel_config is not None\n"
        "            and uniform_token_count is not None\n"
        "            and cudagraph_manager is not None\n"
        "            and batch_desc.cg_mode == CUDAGraphMode.FULL\n"
        "            and check_ubatch_thresholds(parallel_config, num_tokens, uniform_decode=True)\n"
        "        ):\n"
        "            ubatch_desc = cudagraph_manager.dispatch(\n"
        "                num_reqs,\n"
        "                num_tokens,\n"
        "                uniform_token_count,\n"
        "                num_active_loras=num_active_loras,\n"
        "                max_query_len=max_query_len,\n"
        "                num_ubatches=get_num_ubatches(parallel_config),\n"
        "            )\n"
        "            if ubatch_desc.cg_mode == CUDAGraphMode.FULL:\n"
        "                batch_desc = ubatch_desc\n"
        "        return batch_desc, None\n",
    ),
]

PARALLEL_STATE = [
    (
        "\n\nclass GroupCoordinator:\n",
        f"\n\n# {MARKER}: the TP group's all-reduce inside a microbatched decode (two microbatches on two threads,\n"
        "# vllm.v1.worker.ubatching): this microbatch's rows go out on the communication stream while the other\n"
        "# microbatch computes, and this one resumes on the compute stream once its sum has landed - the P2P push\n"
        "# (NCCL's bits) when it applies, PyNCCL otherwise.\n"
        "def _moet_dbo_active() -> bool:\n"
        "    try:\n"
        "        from vllm.v1.worker.ubatching import dbo_enabled\n"
        "    except Exception:  # noqa: BLE001\n"
        "        return False\n"
        "    return dbo_enabled()\n"
        "\n\n"
        "def _moet_dbo_all_reduce(group, x: torch.Tensor):\n"
        '    """The all-reduce of x on the comm stream with the DBO hand-off around it, for the TP group; None otherwise."""\n'
        "    if not _moet_p2p_is_tp(group):\n"
        "        return None\n"
        "    from vllm.v1.worker.ubatching import (\n"
        "        dbo_yield_and_switch_from_comm_to_compute,\n"
        "        dbo_yield_and_switch_from_compute_to_comm,\n"
        "    )\n"
        "\n"
        "    x = x.contiguous()\n"
        "    # The output is the compute stream's (allocated and consumed there); the comm stream only writes it.\n"
        "    out = torch.empty_like(x)\n"
        "    dbo_yield_and_switch_from_compute_to_comm()\n"
        "    done = False\n"
        "    m = _moet_p2p_resolve()\n"
        "    if m and group.world_size == 4:\n"
        "        st = m._init()\n"
        "        if st is not None:\n"
        "            how = m.mode(x)\n"
        '            if how != "nccl":\n'
        '                m._run(st, x, st["orders"][x.shape[0]], how, out=out)\n'
        "                done = True\n"
        "    if not done:\n"
        '        comm = getattr(group.device_communicator, "pynccl_comm", None)\n'
        "        if comm is None or comm.disabled:\n"
        "            out.copy_(group.device_communicator.all_reduce(x))\n"
        "        else:\n"
        "            comm.all_reduce(x, out_tensor=out, stream=torch.cuda.current_stream())\n"
        "    dbo_yield_and_switch_from_comm_to_compute()\n"
        "    return out\n"
        "\n\n"
        "class GroupCoordinator:\n",
    ),
    (
        "        out = _moet_p2p_try_all_reduce(self, input_)  # [vllm-moet] sm_120 P2P all-reduce (TP, generic)\n"
        "        if out is not None:\n"
        "            return out\n"
        "        return self.device_communicator.all_reduce(input_)\n",
        f"        if _moet_dbo_active():  # {MARKER}: inside a microbatched forward\n"
        "            out = _moet_dbo_all_reduce(self, input_)\n"
        "            if out is not None:\n"
        "                return out\n"
        "        out = _moet_p2p_try_all_reduce(self, input_)  # [vllm-moet] sm_120 P2P all-reduce (TP, generic)\n"
        "        if out is not None:\n"
        "            return out\n"
        "        return self.device_communicator.all_reduce(input_)\n",
    ),
]

CONFIG_VLLM = [
    (
        "        if self.parallel_config.use_ubatching:\n"
        "            a2a_backend = self.parallel_config.all2all_backend\n"
        "            if current_platform.is_rocm():\n",
        f"        # {MARKER}: a TP-only two-microbatch decode (VLLM_MOET_DECODE_UBATCH=2) yields at the TP all-reduce,\n"
        "        # not in an expert all-to-all: no all-to-all backend to require (the thresholds are set by the worker).\n"
        "        if self.parallel_config.use_ubatching and (\n"
        '            os.environ.get("VLLM_MOET_DECODE_UBATCH", "0") != "2"\n'
        "        ):\n"
        "            a2a_backend = self.parallel_config.all2all_backend\n"
        "            if current_platform.is_rocm():\n",
    ),
]

P2P_MODULE = [
    (
        "def _run(st: dict, x: torch.Tensor, order: torch.Tensor, how: str) -> torch.Tensor:\n"
        "    out = torch.empty_like(x)\n",
        "def _run(st: dict, x: torch.Tensor, order: torch.Tensor, how: str, out: torch.Tensor | None = None) -> torch.Tensor:\n"
        f"    # {MARKER}: `out` lets a caller on another stream keep the output on the consumer's stream\n"
        "    if out is None:\n"
        "        out = torch.empty_like(x)\n",
    ),
]


def patch(path: Path, src: str, edits: list[tuple[str, str]]) -> str:
    for i, (old, _new) in enumerate(edits):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"{path}: edit {i} anchor found {n} times (expected 1):\n{old}")
    for old, new in edits:
        src = src.replace(old, new, 1)
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT, help="the vllm package directory")
    ap.add_argument("--p2p-dir", type=Path, default=P2P_DIR, help="the p2p_ar module directory")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    files = {
        args.root / "v1/worker/gpu_worker.py": GPU_WORKER,
        args.root / "v1/worker/gpu/ubatch_utils.py": UBATCH_UTILS,
        args.root / "v1/worker/gpu/dp_utils.py": DP_UTILS,
        args.root / "distributed/parallel_state.py": PARALLEL_STATE,
        args.root / "config/vllm.py": CONFIG_VLLM,
        args.p2p_dir / "p2p_allreduce_sm120.py": P2P_MODULE,
    }
    outputs: dict[Path, str] = {}
    for path, edits in files.items():
        src = path.read_text()
        if MARKER in src:
            print(f"{path}: already patched")
            continue
        if path.name == "parallel_state.py" and "_moet_p2p_try_all_reduce" not in src:
            raise SystemExit(f"{path}: step 1 (patch_vllm_p2p_allreduce_tp_sm120.py) must be applied first")
        out = patch(path, src, edits)
        compile(out, str(path), "exec")
        outputs[path] = out
    if args.check:
        print(f"{len(outputs)} file(s) patch cleanly (not written)")
        return 0
    for path, out in outputs.items():
        path.write_text(out)
        print(f"{path}: patched ({MARKER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
