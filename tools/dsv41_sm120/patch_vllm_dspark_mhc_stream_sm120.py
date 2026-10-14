#!/usr/bin/env python3
"""Give the DSpark drafter the same mHC side stream the target decode step already uses.

The served target builds every DeepseekV4DecoderLayer with one mhc_stream (and
fuse_mhc_all_reduce), so a boundary is one kernel on the model stream and the fn
projection plus the Sinkhorn run behind the next sublayer. DSpark's three layers are
ordinary DeepseekV4DecoderLayers too, but dspark.py constructed them without the
stream, so the drafter graph (1,031 us, 157 kernels, rank 0 of run26) keeps the
TileLang pair on the model stream: about 10 us at each of the six sublayer boundaries,
and the first layer's TF32 prenorm GEMM (14 us) in front of them. Nothing else in that
graph is waiting on a side stream, so the pair is entirely on the critical path.

This patch passes the stream through. The kernels, the bit-identity and the switches
are the ones step 10 already installed (patch_vllm_mhc_overlap_sm120.py):
VLLM_MOET_MHC_OVERLAP=0 leaves the stream None, VLLM_MOET_MHC_FUSE_ALLREDUCE=0 keeps
the all-reduce in the linear layers. The drafter gets its own stream; the target's
stream belongs to the target's graph.

With fuse_mhc_all_reduce the layers are built with reduce_results=False on wo_b and the
MoE and every boundary all-reduces the sublayer output it consumes, so the last layer's
FFN output leaves the layer stack unreduced. The target's DeepseekV4Model._collapse
all-reduces it in front of its final mhc_post_tilelang; the drafter's forward calls
mhc_post_tilelang directly, so this patch puts the same all-reduce in front of it
(three anchors: the import block, the layer construction, the final post-mix).

Patched file: vllm/models/deepseek_v41/nvidia/dspark.py
Idempotent, anchor-based. Usage:
    python3 patch_vllm_dspark_mhc_stream_sm120.py [--file PATH] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia/dspark.py"
)
MARKER = "# [vllm-moet] sm_120 drafter mHC stream"

IMPORT_ANCHOR = """from .model import (
    DeepseekV4DecoderLayer,
    DeepseekV4Model,
    _linear_scale_param_name,
    _use_sequence_parallel,
    make_deepseek_v4_expert_params_mapping,
    maybe_init_gemm_rs,
    prepare_mega_gate_routing_metadata,
)
"""
IMPORT_NEW = """from .model import (
    DeepseekV4DecoderLayer,
    DeepseekV4Model,
    _linear_scale_param_name,
    _use_sequence_parallel,
    make_deepseek_v4_expert_params_mapping,
    maybe_init_gemm_rs,
    prepare_mega_gate_routing_metadata,
)
from vllm.distributed import tensor_model_parallel_all_reduce  # [vllm-moet] sm_120 drafter mHC stream
from .ops.mhc import (  # [vllm-moet] sm_120 drafter mHC stream
    init_mhc_all_reduce,
    supports_mhc_all_reduce,
    supports_mhc_overlap,
)
"""

# The last layer's FFN output: with fuse_mhc_all_reduce the layers leave it unreduced (the next
# boundary would reduce it), and the drafter has no next boundary - DeepseekV4Model._collapse does
# exactly this all-reduce in front of its final post-mix.
COLLAPSE_ANCHOR = """        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        # Collapse the hc copies with the pre-mix from the last layer's FFN
"""
COLLAPSE_NEW = """        if self.fuse_mhc_all_reduce:
            # [vllm-moet] sm_120 drafter mHC stream: the last layer has no successor boundary to
            # all-reduce its FFN output (as DeepseekV4Model._collapse does for the target)
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        # Collapse the hc copies with the pre-mix from the last layer's FFN
"""

LAYER_ANCHOR = """        run_gemm_rs = maybe_init_gemm_rs(
            current_vllm_config, self.use_sequence_parallel
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    topk_indices_buffer=self.topk_indices_buffer,
                    run_gemm_rs=run_gemm_rs,
                )
                for i in range(self.num_dspark_layers)
            ]
        )
"""
LAYER_NEW = """        run_gemm_rs = maybe_init_gemm_rs(
            current_vllm_config, self.use_sequence_parallel
        )
        # [vllm-moet] sm_120 drafter mHC stream: the same side stream the target's decoder
        # layers take. The six boundaries stay the TileLang pair on the model stream without
        # it (~10 us each in the drafter graph, plus the first layer's TF32 prenorm GEMM).
        # VLLM_MOET_MHC_OVERLAP=0 leaves the stream None; the drafter does not share the
        # target's stream (that one is captured into the target graph).
        mhc_stream = (
            torch.cuda.Stream() if supports_mhc_overlap(current_vllm_config) else None
        )
        fuse_mhc_all_reduce = mhc_stream is not None and supports_mhc_all_reduce(
            current_vllm_config
        )
        if fuse_mhc_all_reduce:
            init_mhc_all_reduce(current_vllm_config)
        self.mhc_stream = mhc_stream
        self.fuse_mhc_all_reduce = fuse_mhc_all_reduce
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    topk_indices_buffer=self.topk_indices_buffer,
                    run_gemm_rs=run_gemm_rs,
                    mhc_stream=mhc_stream,
                    fuse_mhc_all_reduce=fuse_mhc_all_reduce,
                )
                for i in range(self.num_dspark_layers)
            ]
        )
        if fuse_mhc_all_reduce:
            # As in the target model: a MoE finalize can fold into the next layer's first
            # boundary. DeepGEMM experts (sm_120) do not defer, so this is a no-op there.
            for layer, successor in zip(self.layers, list(self.layers)[1:]):
                if successor.engram is None:
                    layer.ffn.defer_finalize()
"""


def patch_text(src: str) -> str:
    if MARKER in src:
        return src
    for name, anchor in (
        ("import", IMPORT_ANCHOR),
        ("layers", LAYER_ANCHOR),
        ("final post-mix", COLLAPSE_ANCHOR),
    ):
        if src.count(anchor) != 1:
            raise SystemExit(
                f"{name} anchor found {src.count(anchor)} times (expected 1); this patcher "
                "targets vLLM main 21d93d0d's nvidia/dspark.py"
            )
    return (
        src.replace(IMPORT_ANCHOR, IMPORT_NEW, 1)
        .replace(LAYER_ANCHOR, LAYER_NEW, 1)
        .replace(COLLAPSE_ANCHOR, COLLAPSE_NEW, 1)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    if MARKER in src:
        print(f"{args.file}: already patched")
        return 0
    patched = patch_text(src)
    compile(patched, str(args.file), "exec")
    if args.check:
        print(f"{args.file}: patch applies cleanly (not written)")
        return 0
    dst = args.out or args.file
    dst.write_text(patched)
    print(f"{dst}: patched (drafter layers take mhc_stream; the final post-mix all-reduces first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
