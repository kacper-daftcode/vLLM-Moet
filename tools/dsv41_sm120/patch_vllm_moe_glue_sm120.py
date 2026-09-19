#!/usr/bin/env python3
"""Latency fixes for vLLM's DeepGEMM MoE glue kernels at decode token counts (DeepSeek-V4.1-Flash,
TP4: 6 tokens x top-6 = 36 (token, expert) pairs per layer, 40 layers).

Both kernels live in `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py` and are pure data
movement, so the changes are bit-exact:

* `_fwd_kernel_ep_scatter_2` (6.4 us/layer in the 2026-09-18 profile): one program per *token*
  that walks its top-k experts with a dependent chain of `atomic_add` -> row store. Now one
  program per (token, expert) pair: 36 independent programs, one atomic each. The slot a pair
  gets inside its expert's block changes with the atomic order, but every row is computed
  independently by the grouped GEMMs and gathered back per token in top-k order, so the MoE
  output is unchanged.
* `_fwd_kernel_ep_gather` (3.6 us/layer): the top-k loop is a `range` loop with a dependent
  load chain per iteration; `tl.static_range` unrolls it so the index/weight/row loads of all
  k are in flight together. The fp32 accumulation order (k = 0..top_k-1) is unchanged.

Usage: python3 patch_vllm_moe_glue_sm120.py [--file PATH] [--out PATH] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py")
MARKER = "[vllm-moet] sm_120 MoE glue"

SCATTER_OLD = (
    "    start_token_id = tl.program_id(0)\n"
    "    grid_num = tl.num_programs(0)\n"
    "\n"
    "    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)\n"
    "    mask = offset_in < HIDDEN_SIZE\n"
    "\n"
    "    output_tensor_stride0 = output_tensor_stride0.to(tl.int64)\n"
    "\n"
    "    if PACK_UE8M0:\n"
    "        # One int32 per 4 consecutive 32-wide UE8M0 groups, stored MN-major.\n"
    "        offs_pk = tl.arange(0, SCALE_PACKED_SIZE_PAD)\n"
    "        mask_pk = offs_pk < SCALE_PACKED_SIZE\n"
    "    else:\n"
    "        offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)\n"
    "        mask_s = offset_in_s < SCALE_HIDDEN_SIZE\n"
    "\n"
    "    for token_id in range(start_token_id, total_token_num, grid_num):\n"
    "        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)\n"
)
SCATTER_NEW = (
    f"    # {MARKER}: one program per (token, expert) pair instead of per token -- at decode\n"
    "    # (6 tokens) the per-token loop was a chain of top_k dependent atomics; the slot order\n"
    "    # inside an expert block changes, the gathered result does not (rows are independent).\n"
    "    start_pair_id = tl.program_id(0)\n"
    "    grid_num = tl.num_programs(0)\n"
    "\n"
    "    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)\n"
    "    mask = offset_in < HIDDEN_SIZE\n"
    "\n"
    "    output_tensor_stride0 = output_tensor_stride0.to(tl.int64)\n"
    "\n"
    "    if PACK_UE8M0:\n"
    "        # One int32 per 4 consecutive 32-wide UE8M0 groups, stored MN-major.\n"
    "        offs_pk = tl.arange(0, SCALE_PACKED_SIZE_PAD)\n"
    "        mask_pk = offs_pk < SCALE_PACKED_SIZE\n"
    "    else:\n"
    "        offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)\n"
    "        mask_s = offset_in_s < SCALE_HIDDEN_SIZE\n"
    "\n"
    "    for pair_id in range(start_pair_id, total_token_num * topk_num, grid_num):\n"
    "        token_id = pair_id // topk_num\n"
    "        topk_index = pair_id % topk_num\n"
    "        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)\n"
)
SCATTER_LOOP_OLD = (
    "        for topk_index in tl.range(0, topk_num, 1, num_stages=4):\n"
    "            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)\n"
    "\n"
    "            if HAS_EXPERT_MAP:\n"
    "                expert_id = apply_expert_map(expert_id, expert_map)\n"
    "\n"
    "            if expert_id >= 0:\n"
    "                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)\n"
    "                dest_token_index_i64 = dest_token_index.to(tl.int64)\n"
    "                tl.store(\n"
    "                    output_index + token_id * output_index_stride0 + topk_index,\n"
    "                    dest_token_index,\n"
    "                )\n"
    "                output_tensor_ptr = (\n"
    "                    output_tensor + dest_token_index_i64 * output_tensor_stride0\n"
    "                )\n"
    "                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)\n"
    "\n"
    "                output_tensor_scale_ptr = (\n"
    "                    output_tensor_scale + dest_token_index * output_tensor_scale_stride0\n"
    "                )\n"
    "                if PACK_UE8M0:\n"
    "                    tl.store(\n"
    "                        output_tensor_scale_ptr + offs_pk * output_tensor_scale_stride1,\n"
    "                        packed_s,\n"
    "                        mask=mask_pk,\n"
    "                    )\n"
    "                else:\n"
    "                    tl.store(\n"
    "                        output_tensor_scale_ptr + offset_in_s, to_copy_s, mask=mask_s\n"
    "                    )\n"
)
SCATTER_LOOP_NEW = (
    "        expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)\n"
    "\n"
    "        if HAS_EXPERT_MAP:\n"
    "            expert_id = apply_expert_map(expert_id, expert_map)\n"
    "\n"
    "        if expert_id >= 0:\n"
    "            dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)\n"
    "            dest_token_index_i64 = dest_token_index.to(tl.int64)\n"
    "            tl.store(\n"
    "                output_index + token_id * output_index_stride0 + topk_index,\n"
    "                dest_token_index,\n"
    "            )\n"
    "            output_tensor_ptr = (\n"
    "                output_tensor + dest_token_index_i64 * output_tensor_stride0\n"
    "            )\n"
    "            tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)\n"
    "\n"
    "            output_tensor_scale_ptr = (\n"
    "                output_tensor_scale + dest_token_index * output_tensor_scale_stride0\n"
    "            )\n"
    "            if PACK_UE8M0:\n"
    "                tl.store(\n"
    "                    output_tensor_scale_ptr + offs_pk * output_tensor_scale_stride1,\n"
    "                    packed_s,\n"
    "                    mask=mask_pk,\n"
    "                )\n"
    "            else:\n"
    "                tl.store(\n"
    "                    output_tensor_scale_ptr + offset_in_s, to_copy_s, mask=mask_s\n"
    "                )\n"
)
SCATTER_GRID_OLD = "    grid = min(recv_topk.shape[0], 1024 * 8)\n\n    _fwd_kernel_ep_scatter_2[(grid,)](\n"
SCATTER_GRID_NEW = (
    f"    # {MARKER}: (token, expert) pairs\n"
    "    grid = min(recv_topk.shape[0] * recv_topk.shape[1], 1024 * 8)\n\n"
    "    _fwd_kernel_ep_scatter_2[(grid,)](\n"
)
GATHER_OLD = (
    "        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)\n"
    "        for topk_index in range(0, topk_num):\n"
    "            expert_id = tl.load(\n"
    "                recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index\n"
    "            )\n"
)
GATHER_NEW = (
    "        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)\n"
    f"        # {MARKER}: unrolled so the top_k index/weight/row loads are issued together\n"
    "        # (same fp32 accumulation order)\n"
    "        for topk_index in tl.static_range(0, topk_num):\n"
    "            expert_id = tl.load(\n"
    "                recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index\n"
    "            )\n"
)


# --- fp8_utils.py: silu*up + UE8M0 quant only on rows that belong to a token ------------------
# DeepGEMM's contiguous grouped layout pads every touched expert to BLOCK_M (64) rows; at decode
# 36 of 2304 rows are real, the kernel processed all of them (5.9 MB read + 1.5 MB written per
# layer). Rows whose m_indices entry is -1 are never gathered, so they are skipped (their fp8 /
# scale slots keep whatever the workspace holds; the grouped GEMM computes those rows into
# outputs nobody reads).
FP8_PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py")
FP8_SIG_OLD = (
    "def silu_mul_quant_fp8_packed_triton(\n"
    "    input: torch.Tensor,\n"
    "    group_size: int = 128,\n"
    "    output_q: torch.Tensor | None = None,\n"
    "    clamp_limit: float | None = None,\n"
    "    alpha: float = 1.0,\n"
    "    beta: float = 0.0,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
)
FP8_SIG_NEW = (
    "def silu_mul_quant_fp8_packed_triton(\n"
    "    input: torch.Tensor,\n"
    "    group_size: int = 128,\n"
    "    output_q: torch.Tensor | None = None,\n"
    "    clamp_limit: float | None = None,\n"
    "    alpha: float = 1.0,\n"
    "    beta: float = 0.0,\n"
    "    m_indices: torch.Tensor | None = None,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
    f"    # {MARKER}: m_indices (DeepGEMM grouped layout, -1 = padding row) skips padding rows\n"
)
FP8_KSIG_OLD = (
    "    BLOCK_M: tl.constexpr,\n"
    "    HAS_CLAMP: tl.constexpr,\n"
    "):\n"
    "    GROUPS_PER_PACK: tl.constexpr = 4\n"
    "    hidden_size: tl.constexpr = N // 2\n"
)
FP8_KSIG_NEW = (
    "    BLOCK_M: tl.constexpr,\n"
    "    HAS_CLAMP: tl.constexpr,\n"
    "    m_indices_ptr,\n"
    "    HAS_M_INDICES: tl.constexpr,\n"
    "):\n"
    "    GROUPS_PER_PACK: tl.constexpr = 4\n"
    "    hidden_size: tl.constexpr = N // 2\n"
)
FP8_MASK_OLD = (
    "    while row_start < M:\n"
    "        rows = row_start + row_offsets\n"
    "        row_mask = rows < M\n"
)
FP8_MASK_NEW = (
    "    while row_start < M:\n"
    "        rows = row_start + row_offsets\n"
    "        row_mask = rows < M\n"
    "        if HAS_M_INDICES:\n"
    "            row_mask = row_mask & (\n"
    "                tl.load(m_indices_ptr + rows, mask=rows < M, other=-1) >= 0\n"
    "            )\n"
)
FP8_LAUNCH_OLD = (
    "        BLOCK_M=BM,\n"
    "        HAS_CLAMP=has_clamp,\n"
    "        num_warps=num_warps,\n"
    "        num_stages=num_stages,\n"
    "    )\n"
    "\n"
    "    return output_q, output_scale_packed\n"
)
FP8_LAUNCH_NEW = (
    "        BLOCK_M=BM,\n"
    "        HAS_CLAMP=has_clamp,\n"
    "        m_indices_ptr=m_indices if m_indices is not None else input,\n"
    "        HAS_M_INDICES=m_indices is not None,\n"
    "        num_warps=num_warps,\n"
    "        num_stages=num_stages,\n"
    "    )\n"
    "\n"
    "    return output_q, output_scale_packed\n"
)

# --- experts/deep_gemm_moe.py (FP4 experts): hand m_indices to the activation kernel ----------
MOE_PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py")
MOE_SIG_OLD = (
    "    def _act_mul_quant(\n"
    "        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
    "        block_k = self._ACT_BLOCK_K\n"
    "        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()\n"
    "\n"
    "        M_sum, N = input.size()\n"
    "        activation_out_dim = self.adjust_N_for_activation(N, activation)\n"
    "\n"
    "        if activation == MoEActivation.SILU:\n"
    "            # Fused gate+mul+quant kernels for the common SILU case.\n"
    "            if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:\n"
    "                return fused_silu_mul_fp8_quant_packed(\n"
    "                    input=input,\n"
    "                    output_q=output,\n"
    "                    group_size=block_k,\n"
    "                    clamp_limit=self.gemm1_clamp_limit,\n"
    "                )\n"
)
MOE_SIG_NEW = (
    "    def _act_mul_quant(\n"
    "        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation,\n"
    "        m_indices: torch.Tensor | None = None,\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
    "        block_k = self._ACT_BLOCK_K\n"
    "        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()\n"
    "\n"
    "        M_sum, N = input.size()\n"
    "        activation_out_dim = self.adjust_N_for_activation(N, activation)\n"
    "\n"
    "        if activation == MoEActivation.SILU:\n"
    "            # Fused gate+mul+quant kernels for the common SILU case.\n"
    "            if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:\n"
    f"                # {MARKER}: padding rows (m_indices == -1) are skipped\n"
    "                return fused_silu_mul_fp8_quant_packed(\n"
    "                    input=input,\n"
    "                    output_q=output,\n"
    "                    group_size=block_k,\n"
    "                    clamp_limit=self.gemm1_clamp_limit,\n"
    "                    m_indices=m_indices,\n"
    "                )\n"
)
MOE_CALL_OLD = (
    "            a2q, a2q_scale = self._act_mul_quant(\n"
    "                input=mm1_out.view(-1, N), output=quant_out, activation=activation\n"
    "            )\n"
    "\n"
    "            # FC2: FP8 activations x FP4 weights\n"
)
MOE_CALL_NEW = (
    "            a2q, a2q_scale = self._act_mul_quant(\n"
    "                input=mm1_out.view(-1, N), output=quant_out, activation=activation,\n"
    "                m_indices=expert_ids,\n"
    "            )\n"
    "\n"
    "            # FC2: FP8 activations x FP4 weights\n"
)


def _apply(src: str, pairs) -> str:
    if MARKER in src:
        return src
    for old, new in pairs:
        if src.count(old) != 1:
            raise SystemExit(f"anchor found {src.count(old)} times (expected 1):\n{old}")
        src = src.replace(old, new, 1)
    return src


def apply(src: str) -> str:  # deep_gemm_utils.py
    return _apply(src, ((SCATTER_OLD, SCATTER_NEW), (SCATTER_LOOP_OLD, SCATTER_LOOP_NEW),
                        (SCATTER_GRID_OLD, SCATTER_GRID_NEW), (GATHER_OLD, GATHER_NEW)))


def apply_fp8_utils(src: str) -> str:
    return _apply(src, ((FP8_SIG_OLD, FP8_SIG_NEW), (FP8_KSIG_OLD, FP8_KSIG_NEW), (FP8_MASK_OLD, FP8_MASK_NEW),
                        (FP8_LAUNCH_OLD, FP8_LAUNCH_NEW)))


def apply_deep_gemm_moe(src: str) -> str:
    return _apply(src, ((MOE_SIG_OLD, MOE_SIG_NEW), (MOE_CALL_OLD, MOE_CALL_NEW)))


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
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH, help="deep_gemm_utils.py")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fp8-file", type=Path, default=FP8_PATH, help="quantization/utils/fp8_utils.py")
    ap.add_argument("--fp8-out", type=Path, default=None)
    ap.add_argument("--moe-file", type=Path, default=MOE_PATH, help="experts/deep_gemm_moe.py")
    ap.add_argument("--moe-out", type=Path, default=None)
    ap.add_argument("--no-act-skip", action="store_true", help="only the scatter/gather changes")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    _run(args.file, args.out, args.check, apply)
    if not args.no_act_skip:
        _run(args.fp8_file, args.fp8_out, args.check, apply_fp8_utils)
        _run(args.moe_file, args.moe_out, args.check, apply_deep_gemm_moe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
