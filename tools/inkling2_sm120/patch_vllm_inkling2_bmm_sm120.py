#!/usr/bin/env python3
"""Inkling2 step 1: the MoE's batched expert GEMMs on sm_120 (tools/inkling2_sm120/bmm) instead of FlashInfer's
TRT-LLM-gen cubins, which exist for sm_100 only - without it the model does not load on an RTX PRO 6000 / RTX 5090.

Where (vllm/models/inkling2/):
  ops/trtllm_bmm.py   `expert_plan`, `load_kernels` and `bmm` - the three entries `Inkling2Experts` uses. On a
                      capability-12 device they take the sm_120 module: the same token tiles and tile counts, no
                      TRT-LLM kernel lookup, and `bmm` launches inkling2_bmm_sm120's persistent kernel on the same
                      activations, tile metadata and output.
  ops/block_scales.py `_scale_offsets` - the activations' block scales row-major (the kernels read them straight from
                      the quantizers' rows; the R8c4 / R128c4 tiling is the TRT-LLM kernels' operand layout).
  moe.py              `Inkling2Experts._load_weight` / `_load_block_scales` - the expert weights and their block
                      scales into the sm_120 kernel's step-major layout (a byte permutation) instead of the TRT-LLM
                      row shuffle and scale swizzle, a few experts at a time (a rank's experts fill the card: there
                      is no room for a second copy of a layer); with VLLM_MOET_INKLING2_MXFP6=1 the MXFP8 experts
                      (layers 40-47 of the released checkpoint) are packed to e2m3, six bits per value, as they load,
                      and `_register` allocates them in that form from the start. Every chunk's file pages are
                      released as soon as it is on the card (`drop_file_pages`), so a load leaves 13 GiB of the
                      checkpoint in the page cache instead of 82 (VLLM_MOET_INKLING2_DROP_PAGES=0 keeps the pages).
  model.py            `load_weights` - its dtype check accepts a parameter that names the checkpoint's dtype (the
                      packed experts are bytes, the checkpoint's are e4m3).

Switches: VLLM_MOET_INKLING2_BMM=0 leaves vLLM's path (sm_100 and later data-centre parts need nothing from this
step; on sm_120 that path cannot run); =ref computes the GEMM by its float64 definition (eager only, slow: for a
paired run against the kernel). VLLM_MOET_INKLING2_MXFP6=1 (default 0) changes the arithmetic of the MXFP8
layers within the bound documented in docs/inkling2-sm120.md: every weight in the top three binades of its block of
32 is kept bit for bit, smaller ones are rounded on e2m3's subnormal grid; 8.25 -> 6.25 bits per weight.
VLLM_MOET_INKLING2_DIR is the module directory (default /opt/vllm-moet/inkling2_sm120).

Usage: python3 patch_vllm_inkling2_bmm_sm120.py [--root DIR] [--check]   (idempotent; refuses any other file text)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/inkling2")
MARKER = "[vllm-moet] inkling2 expert GEMM on sm_120"

# ------------------------------------------------------------------------------------------- ops/trtllm_bmm.py
BMM_HELPERS_ANCHOR = '''_SOURCE = os.path.join(os.path.dirname(__file__), "trtllm_bmm.cu")
'''
BMM_HELPERS = f'''

# {MARKER}: on capability 12 the expert GEMMs run on tools/inkling2_sm120/bmm (the TRT-LLM-gen cubins below are
# sm_100 only). VLLM_MOET_INKLING2_BMM=0 turns it off.
@functools.cache
def _moet_sm120():
    """The sm_120 expert GEMM module, or None where vLLM's own path applies."""
    if os.environ.get("VLLM_MOET_INKLING2_BMM", "1") == "0":
        return None
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability() if current_platform.is_cuda() else None
    if capability is None or capability.major != 12:
        return None
    import sys

    path = os.path.join(os.environ.get("VLLM_MOET_INKLING2_DIR", "/opt/vllm-moet/inkling2_sm120"), "bmm")
    if path not in sys.path:
        sys.path.insert(0, path)
    import inkling2_bmm_sm120

    inkling2_bmm_sm120._ext()
    return inkling2_bmm_sm120


def _moet_mxfp6(nvfp4: bool) -> bool:
    """Whether MXFP8 expert weights are held as e2m3, six bits per value (VLLM_MOET_INKLING2_MXFP6=1)."""
    return not nvfp4 and os.environ.get("VLLM_MOET_INKLING2_MXFP6", "0") == "1" and _moet_sm120() is not None
'''

PLAN_ANCHOR = '''    max_ctas = min(num_experts, rows) + (rows - min(num_experts, rows)) // tile
    names = configs(nvfp4, tile)
'''
PLAN_NEW = '''    max_ctas = min(num_experts, rows) + (rows - min(num_experts, rows)) // tile
    if _moet_sm120() is not None:
        return ExpertPlan(tile, max_ctas, (0, 0))
    names = configs(nvfp4, tile)
'''

LOAD_ANCHOR = '''    """Loads the kernels of every batch size's plan by running them on no
    tiles: loading is not allowed while a CUDA graph is captured."""
    nvfp4 = weights[0].dtype == torch.uint8
'''
LOAD_NEW = '''    """Loads the kernels of every batch size's plan by running them on no
    tiles: loading is not allowed while a CUDA graph is captured."""
    if _moet_sm120() is not None:
        return
    nvfp4 = weights[0].dtype == torch.uint8
'''

RUN_ANCHOR = '''    expert) is the GEMM's output scale.
    """
    nvfp4 = weight.dtype == torch.uint8
    n = weight.shape[1]
'''
RUN_NEW = '''    expert) is the GEMM's output scale.
    """
    moet = _moet_sm120()
    if moet is not None:
        m = plan.max_ctas
        if act.dtype == torch.uint8:
            fmt, k = moet.NVFP4, act.shape[1] * 2
        else:
            fmt, k = (moet.MXFP6 if weight.dtype == torch.uint8 else moet.MXFP8), act.shape[1]
        return moet.bmm(
            fmt,
            act,
            act_scale,
            row_scale,
            weight,
            weight_scale,
            tiles[:m],
            tiles[m : 2 * m],
            tiles[2 * m : 2 * m + 1],
            plan.tile,
            weight.shape[1],
            k,
        )
    nvfp4 = weight.dtype == torch.uint8
    n = weight.shape[1]
'''

# ------------------------------------------------------------------------------------------ ops/block_scales.py
SCALES_HELPERS_ANCHOR = '''from vllm.triton_utils import tl, triton

'''
SCALES_HELPERS = f'''from vllm.triton_utils import tl, triton


def _moet_row_major() -> bool:
    # {MARKER}: the sm_120 kernels read the activations' block scales row-major
    import os

    if os.environ.get("VLLM_MOET_INKLING2_BMM", "1") == "0":
        return False
    try:
        from vllm.platforms import current_platform

        capability = current_platform.get_device_capability() if current_platform.is_cuda() else None
        return capability is not None and capability.major == 12
    except Exception:  # noqa: BLE001
        return False


_MOET_ROW_MAJOR = tl.constexpr(_moet_row_major())

'''
SCALES_ANCHOR = '''    GEMM token tiles of ``TILE`` rows (0: row-major)."""
    if TILE >= 128:
'''
SCALES_NEW = '''    GEMM token tiles of ``TILE`` rows (0: row-major)."""
    if _MOET_ROW_MAJOR:
        return row * COLS + cols
    if TILE >= 128:
'''

# ------------------------------------------------------------------------------------------------------ moe.py
MOE_IMPORT_ANCHOR = '''from .ops.trtllm_bmm import bmm, expert_plan, load_kernels
'''
MOE_IMPORT_NEW = '''from .ops.trtllm_bmm import _moet_mxfp6, _moet_sm120, bmm, expert_plan, load_kernels
'''
MOE_REGISTER_ANCHOR = '''        if per_expert:
            shape = (self.num_experts, *shape)
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
'''
MOE_REGISTER_NEW = f'''        source_dtype = None
        if name.endswith("_weight") and _moet_mxfp6(self.nvfp4):
            # {MARKER}: the e2m3 form is registered from the start (three bytes per four values), so the
            # e4m3 tensor is never allocated on the card
            shape, dtype, source_dtype = (shape[0], shape[1] * 3 // 4), torch.uint8, dtype
        if per_expert:
            shape = (self.num_experts, *shape)
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        if source_dtype is not None:
            param.moet_source_dtype = source_dtype  # the checkpoint's dtype, for the loader's dtype check
'''
# ---------------------------------------------------------------------------------------------------- model.py
MODEL_DTYPE_ANCHOR = '''            param = params[name]
            if param.dtype != weight.dtype:
                raise ValueError(
'''
MODEL_DTYPE_NEW = f'''            param = params[name]
            # {MARKER}: a parameter held in another form than the checkpoint's names the checkpoint's dtype
            if param.dtype != weight.dtype and getattr(param, "moet_source_dtype", None) != weight.dtype:
                raise ValueError(
'''
MOE_LOAD_ANCHOR = '''    def _load_weight(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load_expert_shard(param, loaded)
        param.data.copy_(shuffle_rows(param.data))

    def _load_block_scales(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load_expert_shard(param, loaded)
        param.data.copy_(swizzle_block_scales(shuffle_rows(param.data)))
'''
MOE_LOAD_NEW = f'''    def _load_weight(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        if _moet_sm120() is not None:
            return self._moet_load(param, loaded, False)
        self._load_expert_shard(param, loaded)
        param.data.copy_(shuffle_rows(param.data))

    def _load_block_scales(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        if _moet_sm120() is not None:
            return self._moet_load(param, loaded, True)
        self._load_expert_shard(param, loaded)
        param.data.copy_(swizzle_block_scales(shuffle_rows(param.data)))

    def _moet_load(self, param: nn.Parameter, loaded: torch.Tensor, is_scale: bool) -> None:
        \"\"\"{MARKER}: this rank's experts into the sm_120 kernel's layout (a byte permutation of the checkpoint's
        row-major tensors), a few experts at a time so that the card holds no second copy of a layer; with
        VLLM_MOET_INKLING2_MXFP6=1 the MXFP8 weights packed to e2m3 (six bits per value) on the way and their
        block scales moved by the packing's per-block exponent change.\"\"\"
        moet = _moet_sm120()
        name = next(n for n, p in self.named_parameters() if p is param)
        projection = name.split("_")[0]
        n, k = self.shapes[0 if projection == "w13" else 1]
        source = (n, k // (16 if self.nvfp4 else 32)) if is_scale else (n, k // (2 if self.nvfp4 else 1))
        assert loaded.shape == (self.num_experts * self.ep_size, *source), f"{{name}}: {{loaded.shape}} {{loaded.dtype}}"
        pack = _moet_mxfp6(self.nvfp4)
        fmt = moet.NVFP4 if self.nvfp4 else (moet.MXFP6 if pack else moet.MXFP8)
        step_bytes = (moet.SCALE_BYTES_PER_STEP if is_scale else moet.WEIGHT_BYTES_PER_STEP)[fmt]
        start = self.ep_rank * self.num_experts
        device = param.device
        out = param.data.view(torch.uint8)
        pending = self.__dict__.setdefault("_moet_pending", {{}})

        def finish_scales(target: torch.Tensor, scales: torch.Tensor, delta: torch.Tensor) -> None:
            moved = (scales.to(torch.int16) + delta.to(torch.int16)).clamp(0, 254).to(torch.uint8)
            target.copy_(moet.prepare(moved, step_bytes))

        if is_scale and pack:
            scales = loaded[start : start + self.num_experts].to(device).view(torch.uint8)
            moet.drop_file_pages(loaded[start : start + self.num_experts])
            delta = pending.pop((projection, "delta"), None)
            if delta is None:  # the weight has not loaded yet: it finishes the scales
                pending[(projection, "scale")] = (out, scales)
            else:
                finish_scales(out, scales, delta)
            return
        deltas = []
        # the packing works in 32- and 64-bit integers (some 40 bytes per value at its peak): two experts at a time,
        # and its temporaries go back to the device after every chunk instead of staying in the allocator's cache
        step = 2 if pack else 8
        for first in range(0, self.num_experts, step):
            chunk = loaded[start + first : start + first + step].to(device).view(torch.uint8)
            moet.drop_file_pages(loaded[start + first : start + first + step])  # the file's pages are not read again
            if pack:
                chunk, delta = moet.pack_e2m3(chunk)
                deltas.append(delta)
            out[first : first + step].copy_(moet.prepare(chunk, step_bytes))
            del chunk
            if pack and device.type == "cuda":
                torch.cuda.empty_cache()
        if not pack:
            return
        delta = torch.cat(deltas)
        waiting = pending.pop((projection, "scale"), None)
        if waiting is None:
            pending[(projection, "delta")] = delta
        else:
            # the scales' own step bytes: this call's step_bytes are the weight's
            target, scales = waiting
            moved = (scales.to(torch.int16) + delta.to(torch.int16)).clamp(0, 254).to(torch.uint8)
            target.copy_(moet.prepare(moved, moet.SCALE_BYTES_PER_STEP[fmt]))
'''

EDITS = {
    "ops/trtllm_bmm.py": [
        (BMM_HELPERS_ANCHOR, BMM_HELPERS_ANCHOR + BMM_HELPERS),
        (PLAN_ANCHOR, PLAN_NEW),
        (LOAD_ANCHOR, LOAD_NEW),
        (RUN_ANCHOR, RUN_NEW),
    ],
    "ops/block_scales.py": [
        (SCALES_HELPERS_ANCHOR, SCALES_HELPERS),
        (SCALES_ANCHOR, SCALES_NEW),
    ],
    "moe.py": [
        (MOE_IMPORT_ANCHOR, MOE_IMPORT_NEW),
        (MOE_REGISTER_ANCHOR, MOE_REGISTER_NEW),
        (MOE_LOAD_ANCHOR, MOE_LOAD_NEW),
    ],
    "model.py": [
        (MODEL_DTYPE_ANCHOR, MODEL_DTYPE_NEW),
    ],
}


def patch_text(rel: str, text: str) -> str:
    if MARKER in text:
        return text
    for old, new in EDITS[rel]:
        if text.count(old) != 1:
            raise SystemExit(f"{rel}: anchor not found exactly once (another vLLM version?): {old.strip().splitlines()[0]!r}")
        text = text.replace(old, new)
    assert MARKER in text
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--check", action="store_true", help="exit 0 if every file is patched, 1 otherwise")
    a = ap.parse_args()
    done = 0
    for rel in EDITS:
        path = a.root / rel
        text = path.read_text()
        if a.check:
            done += MARKER in text
            continue
        new = patch_text(rel, text)
        if new != text:
            path.write_text(new)
            print(f"patched {path}")
        else:
            print(f"already patched {path}")
        compile(new, str(path), "exec")
    if a.check:
        print(f"{done}/{len(EDITS)} files patched")
        return 0 if done == len(EDITS) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
