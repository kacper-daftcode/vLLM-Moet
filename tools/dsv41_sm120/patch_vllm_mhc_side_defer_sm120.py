#!/usr/bin/env python3
"""The attention sublayer's first GEMV as the first child of the mHC boundary kernel node (step 39, sm_120).

In the served decode graph a kernel that follows its producer on the same hardware stream starts ~0.1 us after it
(sparse MLA chain: +0.10, +0.10, +0.10) or overlaps it when launched with PDL, but a child the graph executor sends
to another hardware stream starts ~1.9 us after the producer's end. The boundary kernel node (mhc_post_norm_sm120,
step 10) has several children: the mHC side stream (projection + Sinkhorn), the attention's two aux-stream GEMMs
(indexer weights_proj, compressor kv_score) and the sublayer's first MXFP8 quantization + GEMV on the model stream.
In capture order the model-stream child comes last (execute_in_parallel enqueues the aux chains first, the overlap
glue launched the side stream right after the kernel), and in the production r21 C1 profile (2026-10-26) the
quantization behind 90 % of the attention boundaries starts +1.9 us after the kernel's end although it is PDL-launched
(behind wo_a the same kernel starts 2.2 us *before* the producer's end; in the few layers where the executor kept it
on the producer's stream it starts 7 us before). At the FFN boundary the router's spread GEMM is the first child and
starts 0.7 us after the boundary kernel *starts*. Cost: ~1.9 us x 40 attention boundaries per step (~1 %).

With VLLM_MOET_MHC_SIDE_DEFER=1 the attention boundary's model-stream chain is enqueued first: the overlap glue
records an event right after the boundary kernel and defers the side-stream launch; the attention runs its default
chain first (execute_in_parallel(default_first=True)) and launches the deferred side work right after the
fused_wqa_wkv GEMV (its dependency is the recorded event - the boundary kernel - exactly as before, so every kernel
reads the same bytes at the same point of the dependency graph; only the capture order changes). FFN boundaries, the
drafter's boundaries and write_kv are untouched (their call sites do not tag the stream). Safety: any deferred work
still pending when the layer joins the side stream, or when the next boundary starts, is launched there (late but
correct); without the tag the glue behaves exactly as today. Bit-exact by construction.

Patched files (after patch_vllm_mhc_overlap_sm120.py and patch_vllm_p2p_allreduce_sm120.py):
  vllm/models/deepseek_v41/nvidia/ops/mhc.py   the glue: event + deferred closure, the queue helpers
  vllm/models/deepseek_v41/nvidia/model.py     the attention call site tags the stream; the joins drain the queue
  vllm/models/deepseek_v41/attention.py        default_first + the deferred launch after fused_wqa_wkv

Idempotent, anchor-based. Usage:
    python3 patch_vllm_mhc_side_defer_sm120.py [--root DIR] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "# [vllm-moet] sm_120 mHC side-stream deferral (step 39)"

# --- ops/mhc.py ---------------------------------------------------------------------------------------------------
OPS_HELPER_ANCHOR = "def _moet_mhc_shifted_post_pre_overlap(\n"
OPS_HELPERS = f'''{MARKER}: the model-stream chain first at the attention boundary
_MOET_SIDE_QUEUE: list = []


def _moet_mhc_side_defer_enabled() -> bool:
    import os

    return os.environ.get("VLLM_MOET_MHC_SIDE_DEFER", "0") == "1"


def _moet_mhc_defer_side(stream) -> None:
    """Tag `stream` (the layer's mHC side stream) so the next boundary defers its side-stream launch."""
    if stream is not None and _moet_mhc_side_defer_enabled():
        stream._moet_defer_side = True


def _moet_mhc_run_deferred(stream=None) -> None:
    """Launch the deferred side-stream work (at most one boundary's) if any is pending."""
    while _MOET_SIDE_QUEUE:
        _MOET_SIDE_QUEUE.pop(0)()


'''
OPS_REGION_START = "    stream.wait_stream(main)\n    with torch.cuda.stream(stream):\n"
OPS_REGION_END_LINE = "    stream._moet_pending = ("
OPS_RETURN = "    return residual_out, post.unsqueeze(-1), comb.view(n, hc, hc), layer_input, next_pre, aux\n"

# --- model.py -----------------------------------------------------------------------------------------------------
MODEL_IMPORT_ANCHOR = "from .ops.mhc import (\n"
MODEL_IMPORT = f"    _moet_mhc_defer_side,  {MARKER}\n    _moet_mhc_run_deferred,\n"
MODEL_ATTN_SITE = "            residual, post_mix, res_mix, x, attn_pre, aux = mhc_shifted_post_pre(\n"
MODEL_ATTN_SITE_NEW = (
    f"            _moet_mhc_defer_side(mhc_stream)  {MARKER}\n" + MODEL_ATTN_SITE
)
MODEL_JOIN = "        if mhc_stream is not None:\n            torch.cuda.current_stream().wait_stream(mhc_stream)\n"
MODEL_JOIN_NEW = (
    "        if mhc_stream is not None:\n"
    f"            _moet_mhc_run_deferred(mhc_stream)  {MARKER}: pending side work, if the attention did not launch it\n"
    "            torch.cuda.current_stream().wait_stream(mhc_stream)\n"
)

# --- attention.py -------------------------------------------------------------------------------------------------
ATTN_GEMM_OLD = "        qr_kv, _ = self.fused_wqa_wkv(hidden_states)\n        return qr_kv\n"
ATTN_GEMM_NEW = (
    "        qr_kv, _ = self.fused_wqa_wkv(hidden_states)\n"
    f"        _moet_mhc_run_deferred_side()  {MARKER}: the mHC side stream, behind this GEMV in capture order\n"
    "        return qr_kv\n"
)
ATTN_CALL_OLD = (
    "            self.ln_events[1:3],\n"
    "            aux_streams,\n"
    "            enable=hidden_states.shape[0]\n"
    "            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,\n"
    "        )\n"
)
ATTN_CALL_NEW = (
    "            self.ln_events[1:3],\n"
    "            aux_streams,\n"
    "            enable=hidden_states.shape[0]\n"
    "            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,\n"
    f"            default_first=_moet_mhc_side_defer_enabled(),  {MARKER}\n"
    "        )\n"
)
ATTN_HELPER_ANCHOR = "class DeepseekV4Attention("
ATTN_HELPERS = f'''{MARKER}
def _moet_mhc_side_defer_enabled() -> bool:
    import os

    return os.environ.get("VLLM_MOET_MHC_SIDE_DEFER", "0") == "1"


def _moet_mhc_run_deferred_side() -> None:
    """Launch the mHC side-stream work the boundary glue deferred (no-op when nothing is pending)."""
    try:
        from vllm.models.deepseek_v41.nvidia.ops.mhc import _moet_mhc_run_deferred
    except ImportError:
        return
    _moet_mhc_run_deferred()


'''


def _once(src: str, anchor: str, what: str, path: str, count: int = 1) -> None:
    if src.count(anchor) != count:
        raise SystemExit(f"{path}: anchor '{what}' found {src.count(anchor)} times (expected {count})")


def patch_ops(src: str) -> str:
    if MARKER in src:
        return src
    _once(src, OPS_HELPER_ANCHOR, "overlap function", "ops/mhc.py")
    _once(src, OPS_RETURN, "overlap return", "ops/mhc.py")
    # the side-stream region of the overlap function only (mhc_pre_delayed_overlap has the same first two lines)
    fn = src.find(OPS_HELPER_ANCHOR)
    a = src.find(OPS_REGION_START, fn)
    b = src.find(OPS_REGION_END_LINE, a)
    ret = src.find(OPS_RETURN, b)
    if not (fn < a < b < ret) or src.find("\ndef ", fn + 1) < ret:
        raise SystemExit("ops/mhc.py: side-stream region of the overlap function not found")
    e = src.find("\n", b) + 1
    body = src[a + len(OPS_REGION_START):e]
    region = (
        f"    {MARKER}: the dependency of the side stream is this event (the boundary kernel); the\n"
        "    # launch itself happens now, or - at a tagged attention boundary - after the sublayer's first GEMV\n"
        "    _moet_side_event = torch.cuda.Event()\n"
        "    _moet_side_event.record(main)\n"
        "\n"
        "    def _moet_side_work():\n"
        "        stream.wait_event(_moet_side_event)\n"
        "        with torch.cuda.stream(stream):\n"
        + "".join(("    " + line) if line.strip() else line for line in body.splitlines(keepends=True))
        + "\n"
        "    _moet_mhc_run_deferred(stream)  # a previous boundary's deferred work, if still pending\n"
        "    if getattr(stream, \"_moet_defer_side\", False):\n"
        "        stream._moet_defer_side = False\n"
        "        _MOET_SIDE_QUEUE.append(_moet_side_work)\n"
        "    else:\n"
        "        _moet_side_work()\n"
    )
    src = src[:a] + region + src[e:]
    src = src.replace(OPS_HELPER_ANCHOR, OPS_HELPERS + OPS_HELPER_ANCHOR, 1)
    return src


def patch_model(src: str) -> str:
    if MARKER in src:
        return src
    _once(src, MODEL_IMPORT_ANCHOR, "ops.mhc import", "model.py")
    _once(src, MODEL_ATTN_SITE, "attention boundary call", "model.py")
    _once(src, MODEL_JOIN, "side-stream join", "model.py", count=2)
    src = src.replace(MODEL_IMPORT_ANCHOR, MODEL_IMPORT_ANCHOR + MODEL_IMPORT, 1)
    src = src.replace(MODEL_ATTN_SITE, MODEL_ATTN_SITE_NEW, 1)
    src = src.replace(MODEL_JOIN, MODEL_JOIN_NEW)
    return src


def patch_attention(src: str) -> str:
    if MARKER in src:
        return src
    _once(src, ATTN_GEMM_OLD, "fused_wqa_wkv GEMM", "attention.py")
    _once(src, ATTN_CALL_OLD, "execute_in_parallel call", "attention.py")
    _once(src, ATTN_HELPER_ANCHOR, "attention class", "attention.py")
    src = src.replace(ATTN_GEMM_OLD, ATTN_GEMM_NEW, 1)
    src = src.replace(ATTN_CALL_OLD, ATTN_CALL_NEW, 1)
    src = src.replace(ATTN_HELPER_ANCHOR, ATTN_HELPERS + ATTN_HELPER_ANCHOR, 1)
    return src


FILES = (
    ("models/deepseek_v41/nvidia/ops/mhc.py", patch_ops),
    ("models/deepseek_v41/nvidia/model.py", patch_model),
    ("models/deepseek_v41/attention.py", patch_attention),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    outs = []
    for rel, fn in FILES:
        p = args.root / rel
        src = p.read_text()
        if MARKER in src:
            print(f"{p}: already patched")
            continue
        new = fn(src)
        compile(new, str(p), "exec")
        outs.append((p, new))
    if args.check:
        print(f"{len(outs)} file(s): patch applies cleanly (not written)")
        return 0
    for p, new in outs:
        p.write_text(new)
        print(f"{p}: patched (mHC side-stream deferral)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
