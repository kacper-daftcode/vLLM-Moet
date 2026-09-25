#!/usr/bin/env python3
"""Take the mHC projection and Sinkhorn off the decode critical path on sm_120 (DeepSeek-V4.1-Flash).

Between two sublayers vLLM's decode step runs two TileLang kernels back to back on the model stream:
`mhc_fused_tilelang` (post-mix of the four residual streams + the fn projection GEMM, [T, 12, 8] x 128)
and `mhc_pre_big_fuse_with_norm` (split sums, sigmoids, 4x4 Sinkhorn x 20, collapse with the carried
pre-mix, RMSNorm) - 10.5 us per sublayer boundary in the served graph on the RTX PRO 6000, 80 boundaries
per step. Only the post-mix, the collapse and the norm are on the critical path: the next sublayer
consumes the normalized input, while this boundary's coefficients (post mix, residual mix, next pre-mix)
are first read at the *next* boundary. Upstream already has that split for GB200
(`mhc_pre_delayed_overlap`: input collapse on the model stream, projection + coefficients on
`mhc_stream`), gated to SM100 + DeepGEMM's TF32 prenorm GEMM.

This patch enables the side stream on sm_120 and puts tools/dsv41_sm120/mhc_overlap/ on the critical
path: one CUDA kernel per boundary (`mhc_post_norm`: post-mix -> bf16 streams, collapse, RMSNorm, draft
aux; one CTA of H/8 threads per token, 3.3 us on an RTX 5090 against 7.4 us for the TileLang pair) while
the side stream runs the unchanged TileLang kernels behind the sublayer: `mhc_fused_tilelang` for the
projection (tokens <= 32; it recomputes the post-mix in fp32 exactly as served today) or, above 32
tokens, DeepGEMM's TF32 prenorm GEMM on the bf16 streams the kernel wrote (as served today above 32),
then `mhc_pre_big_fuse_with_norm` in its "stats" split mode for the coefficients. Every tensor is
bit-identical to the served path (test_mhc_post_norm_sm120.py; the split modes reproduce the fused
epilogue). The first layer's `mhc_pre` and the Engram layers' `mhc_post` + `mhc_pre` take upstream's
overlap path unchanged (same kernels as today, coefficients on the side stream).

Patched file: vllm/models/deepseek_v41/nvidia/ops/mhc.py
  supports_mhc_overlap: also True on sm_120 (hidden 5120, hc_mult 4, DeepGEMM, no ubatching) when the
      kernel extension loads; MHC_OVERLAP_MAX_TOKENS 16 -> 64 there (VLLM_MOET_MHC_OVERLAP_MAX_TOKENS).
  supports_mhc_all_reduce: True there too (TP4) - upstream's fuse_mhc_all_reduce moves the sublayer's
      all-reduce out of wo_b / the MoE into mhc_shifted_post_pre, where the sm_120 path runs a plain
      all-reduce right before the boundary kernel (no MNNVL kernel): the side stream's join then lands
      on the all-reduce node and the boundary kernel has a single parent in the captured graph.
  mhc_shifted_post_pre: with a stream, the sm_120 path above.
Runtime switches: VLLM_MOET_MHC_OVERLAP=0 (no side stream: the served two-kernel path),
  VLLM_MOET_MHC_FUSE_ALLREDUCE=0 (all-reduce stays in the linear layers), VLLM_MOET_MHC_PROJ,
  VLLM_MOET_MHC_PDL=1 (PDL launch of the boundary kernel).

Idempotent, anchor-based. Usage:
    python3 patch_vllm_mhc_overlap_sm120.py [--file PATH] [--kernel-dir DIR] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia/ops/mhc.py")
DEFAULT_KERNEL_DIR = "/opt/vllm-moet/dsv41_sm120/mhc_overlap"
MARKER = "# [vllm-moet] sm_120 mHC overlap"

CONST_ANCHOR = (
    "# GB200 TP4/FlashInfer improves through 16 tokens; larger screens tie or regress.\n"
    "MHC_OVERLAP_MAX_TOKENS = 16\n"
)
HELPER_TEMPLATE = '''

{marker}: the projection + Sinkhorn on the side stream behind the sublayer, the
# post-mix + collapse + RMSNorm in one kernel on the model stream (tools/dsv41_sm120/mhc_overlap).
from vllm.logger import init_logger as _moet_init_logger

_moet_logger = _moet_init_logger(__name__)
_MOET_MHC = None  # None = not resolved yet, False = off / unavailable, else the module


def _moet_mhc_post_norm():
    """The sm_120 critical-path kernel module (mhc_post_norm_sm120) or None."""
    global _MOET_MHC
    if _MOET_MHC is None:
        import os
        import sys

        _MOET_MHC = False
        if os.environ.get("VLLM_MOET_MHC_OVERLAP", "1") == "1" and (
            current_platform.is_cuda() and current_platform.is_device_capability_family(120)
        ):
            kernel_dir = os.environ.get("VLLM_MOET_MHC_OVERLAP_DIR", "{kernel_dir}")
            if kernel_dir not in sys.path:
                sys.path.insert(0, kernel_dir)
            try:
                import mhc_post_norm_sm120 as _m

                _m._ext()  # build / load the extension now, not inside the first forward
                _MOET_MHC = _m
                _moet_logger.info_once(
                    "vllm-moet sm_120 mHC overlap: post-mix + collapse + RMSNorm in one kernel on "
                    "the model stream, projection + Sinkhorn on a side stream "
                    "(VLLM_MOET_MHC_OVERLAP=0 turns it off)"
                )
            except Exception as exc:  # noqa: BLE001
                _moet_logger.warning(
                    "vllm-moet sm_120 mHC overlap unavailable, using the TileLang pair: %r", exc
                )
    return _MOET_MHC or None


def _moet_mhc_overlap_supported(vllm_config: VllmConfig) -> bool:
    config = vllm_config.model_config.hf_config
    return (
        config.hc_mult == 4
        and config.hidden_size % 1024 == 0
        and 1024 <= config.hidden_size <= 5120
        and not vllm_config.parallel_config.use_ubatching
        and is_deep_gemm_supported()
        and _moet_mhc_post_norm() is not None
    )


def _moet_proj_kind() -> str:
    """Which projection runs on the side stream for <= 32 tokens (VLLM_MOET_MHC_PROJ):
    "ours"  - the served kernel's partials from one CTA per token x split, 64 registers per thread
              (default: bit-identical to the served step and the smallest footprint next to the sublayer),
    "fused" - the served TileLang kernel itself ([T, 12, 8] x 128 CTAs; bit-identical, but it takes
              SMs from the sublayer it runs behind),
    "tf32"  - DeepGEMM's TF32 prenorm GEMM on the bf16 streams (the served path above 32 tokens;
              not bit-identical below 33)."""
    import os

    kind = os.environ.get("VLLM_MOET_MHC_PROJ", "ours")
    return kind if kind in ("fused", "ours", "tf32") else "ours"


def _moet_overlap_max_tokens() -> int:
    """The token limit the decoder layer applies to the side stream: every captured decode batch
    on sm_120 (the kernel takes any count; the projection path switches at 32 like the served one).
    Only cheap checks here - this runs at import; the extension loads in supports_mhc_overlap."""
    import os

    try:
        if os.environ.get("VLLM_MOET_MHC_OVERLAP", "1") == "1" and (
            current_platform.is_cuda() and current_platform.is_device_capability_family(120)
        ):
            return int(os.environ.get("VLLM_MOET_MHC_OVERLAP_MAX_TOKENS", "64"))
    except Exception:  # noqa: BLE001
        pass
    return 16


def _moet_mhc_shifted_post_pre_overlap(
    kernel_module,
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    capture_aux: bool,
    stream: torch.cuda.Stream,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Shifted post/pre on sm_120: the critical path in one kernel, the coefficients on `stream`.

    Only the streams, the layer input and the aux are ready on the caller stream; join `stream`
    before reading post, comb or the next pre-mix (the decoder layer does, after the sublayer).

    The tensors the side stream reads are kept alive on the stream object until the next boundary
    instead of `record_stream`: under graph capture the allocator cannot free a recorded block
    before the capture ends, so every boundary's inputs would stay allocated for the whole graph
    (+0.15 GiB of graph memory per rank, taken from the KV cache); a Python reference released
    at the next boundary - by which point the decoder layer has joined the stream - costs nothing.
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_FUSED_TILELANG_KERNEL,
        mhc_fused_post_pre_split_config,
    )
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    n, hc, hidden = residual.shape
    mix_size = hc * (hc + 2)
    input_size = hc * hidden
    post = torch.empty((n, hc), device=residual.device, dtype=torch.float32)
    comb = torch.empty((n, hc * hc), device=residual.device, dtype=torch.float32)
    next_pre = torch.empty_like(post)
    main = torch.cuda.current_stream()
    # the previous boundary's side-stream inputs: joined by the caller, safe to release now
    stream._moet_pending = None
    fused_config = mhc_fused_post_pre_split_config(n, hidden, hc)
    proj = _moet_proj_kind()
    if proj == "tf32":
        fused_config = None
    # The critical path first, the fork after it. In the captured graph the kernel is then the
    # only child of the all-reduce that precedes it (the caller joined the side stream before the
    # all-reduce, so that node carries the join) and, launched with PDL, starts 0.2 us after it;
    # as a sibling of the side branch or with the join on it, 2.2 us (8 us in a micro-benchmark of
    # the captured pattern). The projection needs only this boundary's inputs, so starting it after
    # the kernel costs nothing.
    residual_out, layer_input, aux = kernel_module.mhc_post_norm(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        pre_mix,
        norm_weight,
        norm_eps,
        capture_aux=capture_aux,
    )
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        if fused_config is not None:
            # The projection of the fp32 post-mix recomputed from the same inputs (what the served
            # fused kernel does): ours (one CTA per split x 4 tokens, the same bits) or the served
            # TileLang kernel ([T, 12, 8] x 128 CTAs).
            if proj == "ours" and kernel_module.mhc_proj_applicable(x, residual):
                mixes, sqrsum = kernel_module.mhc_proj(x, residual, post_layer_mix, comb_res_mix, fn)
            else:
                mixes, sqrsum, _ = _MHC_FUSED_TILELANG_KERNEL(
                    comb_res_mix,
                    residual,
                    post_layer_mix.view(n, hc),
                    x,
                    fn.view(mix_size, hc, hidden),
                    hc,
                    hidden,
                    mix_size,
                )
        else:
            # Above the fused kernel's token range the served path projects the bf16 streams with
            # DeepGEMM's TF32 prenorm GEMM; here it reads the streams the kernel just wrote.
            n_splits = compute_mhc_pre_num_splits(input_size, n)
            mixes = torch.empty(
                (n_splits, n, mix_size), device=residual.device, dtype=torch.float32
            )
            sqrsum = torch.empty((n_splits, n), device=residual.device, dtype=torch.float32)
            tf32_hc_prenorm_gemm(residual_out.view(n, input_size), fn, mixes, sqrsum, n_splits)
        MHC_PRE_NORM_KERNEL(
            mixes,
            sqrsum,
            hc_scale,
            hc_base,
            residual_out,
            post,
            comb,
            layer_input,
            norm_weight,
            pre_mix,
            next_pre,
            layer_input,  # unused aux slot in the stats split mode
            hidden_size=hidden,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            norm_eps=norm_eps,
            hc_mult=hc,
            use_pre_mix_in=True,
            save_pre_mix=True,
            rms_numel=input_size,
            split_mode="stats",
        )
    # everything the side stream reads or writes, alive until the next boundary (see the docstring)
    stream._moet_pending = (x, residual, post_layer_mix, comb_res_mix, residual_out, mixes, sqrsum, pre_mix, post, comb, next_pre)
    return residual_out, post.unsqueeze(-1), comb.view(n, hc, hc), layer_input, next_pre, aux


MHC_OVERLAP_MAX_TOKENS = _moet_overlap_max_tokens()
'''

SUPPORT_ANCHOR = (
    "    mix_size = config.hc_mult * (config.hc_mult + 2)\n"
    "    return (\n"
    "        current_platform.is_device_capability_family(100)\n"
    "        and is_deep_gemm_supported()\n"
)
SUPPORT_NEW = (
    "    mix_size = config.hc_mult * (config.hc_mult + 2)\n"
    f"    {MARKER}: the side stream on sm_120 too\n"
    "    if _moet_mhc_overlap_supported(vllm_config):\n"
    "        return True\n"
    "    return (\n"
    "        current_platform.is_device_capability_family(100)\n"
    "        and is_deep_gemm_supported()\n"
)

ALLREDUCE_ANCHOR = (
    "    comm = cast(\"CudaCommunicator\", get_tp_group().device_communicator).ca_comm\n"
    "    return comm is not None and bool(comm.mnnvl_lamport_ag_multicast_ptr)\n"
)
ALLREDUCE_NEW = (
    f"    {MARKER}: the sublayer's all-reduce moves into mhc_shifted_post_pre, in front of\n"
    "    # the boundary kernel (a plain all-reduce, no MNNVL kernel; VLLM_MOET_MHC_FUSE_ALLREDUCE=0 keeps\n"
    "    # it in the linear layers)\n"
    "    if _moet_mhc_overlap_supported(vllm_config):\n"
    "        import os\n"
    "\n"
    "        return os.environ.get(\"VLLM_MOET_MHC_FUSE_ALLREDUCE\", \"1\") == \"1\"\n"
) + ALLREDUCE_ANCHOR

DISPATCH_ANCHOR = (
    "    layer_input = None\n"
    "    if reduce_results:\n"
    "        tp = get_tp_group()\n"
    "        if stream is not None and 0 < x.shape[0] <= MHC_OVERLAP_MAX_TOKENS:\n"
)
DISPATCH_NEW = (
    f"    {MARKER}: one kernel on the critical path, the coefficients behind the sublayer.\n"
    "    # With fuse_mhc_all_reduce the sublayer's all-reduce happens here, right before that kernel:\n"
    "    # the join of the side stream then lands on the all-reduce node, and the kernel has one\n"
    "    # parent (see _moet_mhc_shifted_post_pre_overlap).\n"
    "    if stream is not None and (_moet_kernels := _moet_mhc_post_norm()) is not None:\n"
    "        if not _moet_kernels.mhc_post_norm_applicable(x, residual, pre_mix, norm_weight):\n"
    "            stream = None  # not the kernel's shape: the served path, coefficients on this stream\n"
    "        else:\n"
    "            assert pre_mix is not None and norm_weight is not None\n"
    "            if reduce_results:\n"
    "                x = get_tp_group().all_reduce(x)\n"
    "            return _moet_mhc_shifted_post_pre_overlap(\n"
    "                _moet_kernels,\n"
    "                x,\n"
    "                residual,\n"
    "                post_layer_mix,\n"
    "                comb_res_mix,\n"
    "                fn,\n"
    "                hc_scale,\n"
    "                hc_base,\n"
    "                rms_eps,\n"
    "                hc_pre_eps,\n"
    "                hc_sinkhorn_eps,\n"
    "                hc_post_mult_value,\n"
    "                sinkhorn_repeat,\n"
    "                pre_mix,\n"
    "                norm_weight,\n"
    "                norm_eps,\n"
    "                capture_aux,\n"
    "                stream,\n"
    "            )\n"
) + DISPATCH_ANCHOR


def patch_text(src: str, kernel_dir: str) -> str:
    if MARKER in src:
        return src
    for name, anchor in (("constant", CONST_ANCHOR), ("supports_mhc_overlap", SUPPORT_ANCHOR),
                         ("supports_mhc_all_reduce", ALLREDUCE_ANCHOR), ("dispatch", DISPATCH_ANCHOR)):
        if src.count(anchor) != 1:
            raise SystemExit(f"{name} anchor found {src.count(anchor)} times (expected 1)")
    out = src.replace(CONST_ANCHOR, CONST_ANCHOR + HELPER_TEMPLATE.format(marker=MARKER, kernel_dir=kernel_dir), 1)
    out = out.replace(SUPPORT_ANCHOR, SUPPORT_NEW, 1)
    out = out.replace(ALLREDUCE_ANCHOR, ALLREDUCE_NEW, 1)
    out = out.replace(DISPATCH_ANCHOR, DISPATCH_NEW, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--kernel-dir", default=DEFAULT_KERNEL_DIR)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    if MARKER in src:
        print(f"{args.file}: already patched")
        return 0
    patched = patch_text(src, args.kernel_dir)
    compile(patched, str(args.file), "exec")
    if args.check:
        print(f"{args.file}: patch applies cleanly (not written)")
        return 0
    dst = args.out or args.file
    dst.write_text(patched)
    print(f"{dst}: patched (mHC overlap kernel dir {args.kernel_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
