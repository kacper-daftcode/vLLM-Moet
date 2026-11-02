#!/usr/bin/env python3
"""Sequence-parallel mHC boundaries for the eager prefill steps of DeepSeek-V4.1 at TP4 on sm_120 (the AR track).

Switch: VLLM_MOET_PREFILL_SP=1 (default 0 = the served path, untouched). Scope: an eager step of at least
VLLM_MOET_PREFILL_SP_MIN_TOKENS rows (default 2048: block B of 29.10 measured one boundary at +0.1 ms with 1024 rows
and -0.4 / -0.04 ms with 3063, so the break-even lies between; never below 33 x TP rows, because a shard of <= 32 rows
would take the fused post + pre TileLang kernel whose bits differ from the split-k path the full rows take -
measured at T = 128) on a TP group of >= 2 ranks without expert / pipeline / context parallelism or microbatching,
with the boundaries owning the reductions (fuse_mhc_all_reduce, the sm_120 default). Decode steps (CUDA graphs,
<= 64 tokens) and every replay-batch step keep today's code: P2P all-reduce + boundary kernel.

Why. A fresh 163K prefill spends 29.4 % of its GPU time in NCCL all-reduces: two [4096, 5120] bf16 all-reduces per
full layer per 4096-token chunk, 1.34-1.49 ms each (Phase 0, /workspace/tmp/par31/L/phase0/PHASE0_RESULTS.md §2, §5).
On this PCIe ring (NCCL 2.30.7, 4 x RTX PRO 6000): all_reduce 1 341 us, reduce_scatter 728, all_gather 719. Every
sublayer output is a per-rank partial sum that only the mHC boundary (post-mix of the residual streams, the pre-mix
projection + Sinkhorn, RMSNorm of the collapsed input) consumes, and that boundary is row-wise: it can run on this
rank's T/4 rows after a reduce-scatter. What comes after decides whether the rows have to be gathered back:

  boundary 1, MoE output -> attention input (layers 1..19 and layer 20's write_kv in a 163K prefill; layer 0 has no
      all-reduce in front of it): all-reduce -> reduce-scatter; post + pre + norm on the shard; the attention's input
      projections are replicated GEMMs (fused_wqa_wkv 5120 -> 1792 MXFP8 CUTLASS, compressor fused_wkv_wgate
      5120 -> 1024 bf16 at the 4 kv-source layers, indexer weights_proj 5120 -> 32 at the 4 index-source layers among
      the full layers), so each rank projects its rows and the *projections* are all-gathered: [T, 1792] instead of
      [T, 5120] bf16 = 35 % of the bytes. The two Engram layers (1, 14) run Engram on the shard too (its own
      sequence-parallel mode: the hash rows are all-gathered as today, selected per shard; wkv 6144 -> 25600 on T/4
      rows instead of T). Measured per boundary at 4096 rows (block B, 29.10, torchrun x4, medians, max over ranks):
      served AR 1351 + mHC 406 = 1749 us -> RS 743 + mHC on the shard 134 + all-gather [T, 1792] 273 = 1125 us,
      -624 us (3063 rows: -386; 1024: +127). Not in that figure: fused_wqa_wkv on T/4 rows (~-140 us) and, at the
      Engram layers, wkv on T/4 rows (~-600 us).
  boundary 2, attention output -> MoE input (layers 0..19): all-reduce -> reduce-scatter; post + pre + norm on the
      shard; then the MoE needs every row (TP experts: each rank holds 1/4 of every expert's intermediate, routing
      and the shared expert read the full input), so the normed input is all-gathered in bf16: RS 743 + mHC 134 +
      AG [T, 5120] 738 = 1564 us vs 1749, -185 us (3063 rows: -41; 1024: +106). (Follow-up with a real gain:
      all-gather the MXFP8-quantized input the b12x MoE produces first anyway - 1.03 B/element - ~-350 us more.)
  what stays: the decode graphs (P2P all-reduce fused into the boundary kernel), the replay rows of layers 20..39
      (128 rows per request: the states of those rows are gathered from the shards once, the layers run unsharded),
      the final collapse of a step without a replay batch (sharded post, then the hc state all-gathered - rare),
      the DSpark drafter (its own layers and its own sequence-parallel flag), and every collective of the MoE itself.

So in a full layer the two all-reduces become two reduce-scatters, one narrow all-gather (attention input
projections) and one full all-gather (MoE input). Per 4096-token chunk of a 163K prefill (boundary 1 at layers 1..20,
16 of them with the narrow all-gather and 4 - the compressor/indexer layers 2, 8, 14, 20 - with the full one;
boundary 2 at layers 0..19): 16 x 624 + 4 x 185 + 20 x 185 = 14.4 ms of a 202-ms chunk = -7 % from the measured
boundary deltas alone, ~-8.5 % with the GEMMs on the shard - not the -13 % of the Phase 0 estimate, which did not
charge the MoE's all-gather. The rest needs the MoE to take sharded rows (expert parallelism) or a half-width
(MXFP8) all-gather.

Bits. Everything on the shard is the row-wise arithmetic of the served path on the same rows, with these caveats:
  * the mHC pre-mix projection is a split-k TF32 GEMM whose split count follows the token count
    (compute_mhc_pre_num_splits: 4 at 4096 rows, 16 at 1024); the split count is pinned to the unsharded step's
    value around every sharded boundary (warmup.py), so the shard sums in the same k-slices as the full rows. With
    the pin, the boundary on a row subset is bit-identical per row (block B: residual, post, comb, input and the
    next pre-mix EQUAL at T = 4096, 4092, 3063, 1024, 896; without it 98 % of the fp32 mixes move by ~1e-6). Below
    33 rows the shard would take the fused post + pre kernel (another algorithm; T = 128 at TP4: every mix differs by
    ~1e-3) - hence the 33 x TP floor of the gate. The Engram layer's pair (post + pre) and the first layer's pre on
    the embedding shard: EQUAL at every T.
  * the reduce-scatter sums the four partials in NCCL's ring order for *that* collective: ((x[r+1] + x[r+2]) +
    x[r+3]) + x[r] for the rows of rank r, bf16 after every add. NCCL's ring all-reduce ends a row's reduction at the
    rank of its *chunk* (chunks interleave along the buffer, ~2 MB each), so three quarters of the rows are summed in
    a different order there: block B measured 32.0 % of the elements different at every T (identical rows 1017 of
    4096 = the quarter whose chunk ends on the row's owner), max |delta| 0.031 = one bf16 ulp at the largest
    magnitudes present, both collectives deterministic (3 x identical). Everything downstream inherits it: with the
    pinned split count, exactly the rows whose sum differs get different mixes (75.2 % of the rows at 4096), 23 % of
    the next sublayer's input elements differ by 1-2 ulp. A re-association of a 4-term bf16 sum, not a different
    algorithm - but not today's bits: level A/B decision (quality battery), unless a reduce-scatter in the
    all-reduce's own per-element order is written: the P2P push kernels of tools/dsv41_sm120/p2p_ar do exactly that
    for <= 64 rows (nccl_order_tables, two-shot = RS + AG); at 4096 rows it needs the order tables probed per
    distinct M (8 all-reduces of [M, 5120] per M, ~11 ms, cached), 31.5 MB of peer partials pushed per rank per
    boundary (IPC slots of 2 x 3 x 10.5 MB instead of today's 2.6 MB), and a push bandwidth near NCCL's 43 GB/s per
    rank for the time not to be lost again (unmeasured at this size).
  * the replicated GEMMs on the shard: CUTLASS MXFP8 (fused_wqa_wkv, Engram wkv) tiles the rows and runs the full
    k-loop per tile, so a row's bits do not depend on the row count (block B: rows of M = 1024 / 1023 / 766 EQUAL to
    the same rows of M = 4096); cuBLAS picks its split-k by the row count (the bf16 -> fp32 compressor projection:
    96 % of the elements move by ~1e-6; the bf16 indexer weights_proj: 51 % by one ulp), so
    VLLM_MOET_PREFILL_SP_AUX=full (default) all-gathers the input at the layers that have a compressor or an
    indexer and projects every row there (fused_wqa_wkv included: no gain at those 4 of 21 full layers beyond the
    boundary math); =shard projects on the shard everywhere (faster, other bits at those 4 layers).

Patched files (anchor-based, idempotent, after the shipped sm_120 patchers; step 39 of r21-mxfp8 is tolerated):
  vllm/models/deepseek_v41/nvidia/model.py        the step gate (sp_shard of the embedding), the decoder layer's two
                                                  boundaries, Engram in SP mode, write_kv, the replay rows' gather,
                                                  the collapse without a replay batch
  vllm/models/deepseek_v41/attention.py           forward / forward_kv take the row shard: projections + all-gather
  vllm/model_executor/kernels/mhc/warmup.py       compute_mhc_pre_num_splits honours the pinned token count

Usage:
    python3 patch_vllm_prefill_sp_sm120.py [--dist DIR] [--check]
  --dist   the vllm package directory (default /usr/local/lib/python3.12/dist-packages/vllm)
  --check  dry run: report whether every anchor is present (or the patch is already applied), change nothing

Reduce-scatter order (session P, 29.10): VLLM_MOET_PREFILL_SP_RS=nccl (default) uses vLLM's sp_reduce_scatter - NCCL's
ring order, 1 ulp on 32 % of the elements against the served all-reduce; =p2p uses tools/dsv41_sm120/p2p_rs (module
directory VLLM_MOET_P2P_RS_DIR, default /opt/vllm-moet/dsv41_sm120/p2p_rs): a P2P push of the peers' row blocks and
a consumer that adds the four partials of every 8-element group in the order NCCL's all-reduce used for it (tables
probed on the TP communicator per distinct M, cached) - the served all-reduce's bits on the shard. A shape or M the
module does not serve falls back to NCCL's reduce-scatter (the same decision on every rank).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "[vllm-moet] sm_120 prefill SP"

# ---------------------------------------------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------------------------------------------
MODEL_HELPERS_ANCHOR = "logger = init_logger(__name__)\n\n\nclass DeepseekV4MoE(DeepseekV4MoEBase):\n"
MODEL_HELPERS = f'''logger = init_logger(__name__)


# {MARKER} (VLLM_MOET_PREFILL_SP=1): the two mHC boundaries of an eager prefill step reduce-scatter the
# sublayer's partial output and run post + pre + norm on this rank's row shard; the next sublayer's input is
# all-gathered where every row is needed (tools/dsv41_sm120/patch_vllm_prefill_sp_sm120.py has the full account).
import contextlib as _moet_contextlib
import os as _moet_os

import vllm.model_executor.kernels.mhc.warmup as _moet_mhc_warmup
from vllm.distributed import get_tensor_model_parallel_rank as _moet_tp_rank
from vllm.distributed import tensor_model_parallel_all_gather as _moet_all_gather


def _moet_prefill_sp_mode() -> int:
    try:
        return int(_moet_os.environ.get("VLLM_MOET_PREFILL_SP", "0"))
    except ValueError:
        return 0


def _moet_prefill_sp_min_tokens() -> int:
    try:
        return int(_moet_os.environ.get("VLLM_MOET_PREFILL_SP_MIN_TOKENS", "2048"))
    except ValueError:
        return 2048


def _moet_sp_sharded(state: torch.Tensor | None, positions: torch.Tensor) -> bool:
    """Whether a per-token state is this rank's row shard (the sequence-parallel step): its rows != the step's."""
    return isinstance(state, torch.Tensor) and state.shape[0] != positions.shape[0]


_MOET_P2P_RS = None  # None = not resolved, False = NCCL's reduce-scatter, else the P2P module (VLLM_MOET_PREFILL_SP_RS=p2p)


def _moet_p2p_rs_resolve() -> None:
    """VLLM_MOET_PREFILL_SP_RS=p2p: the boundary's reduce-scatter on the P2P push in NCCL's all-reduce order
    (tools/dsv41_sm120/p2p_rs, bit-identical with the served all-reduce); nccl (default) = sp_reduce_scatter."""
    global _MOET_P2P_RS
    if _MOET_P2P_RS is None:
        import sys as _moet_sys

        _MOET_P2P_RS = False
        if _moet_os.environ.get("VLLM_MOET_PREFILL_SP_RS", "nccl") == "p2p":
            d = _moet_os.environ.get("VLLM_MOET_P2P_RS_DIR", "/opt/vllm-moet/dsv41_sm120/p2p_rs")
            if d not in _moet_sys.path:
                _moet_sys.path.insert(0, d)
            try:
                import p2p_reduce_scatter_sm120 as _moet_p2p_rs_module

                _MOET_P2P_RS = _moet_p2p_rs_module
            except Exception as exc:  # noqa: BLE001
                logger.warning("vllm-moet sm_120 prefill SP: P2P reduce-scatter module unavailable, NCCL's stays: %r", exc)


def _moet_sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """The sequence-parallel boundary's reduce-scatter of the sublayer's partial [T, H]: this rank's ceil(T/TP) rows
    of the sum. NCCL's reduce_scatter sums them in another cyclic order than the served all-reduce (1 ulp on 32 % of
    the elements); VLLM_MOET_PREFILL_SP_RS=p2p takes the P2P push that adds in the all-reduce's own order (falls back
    to NCCL's for a shape or row count it does not serve - the same decision on every rank)."""
    _moet_p2p_rs_resolve()
    if _MOET_P2P_RS:
        out = _MOET_P2P_RS.reduce_scatter(x)
        if out is not None:
            return out
    return sp_reduce_scatter(x)


def _moet_sp_pin(num_tokens: int | None) -> None:
    """Pin the mHC pre-mix projection's split-k count to the unsharded step's (None releases it): a boundary
    on T/4 rows then sums the same k-slices as the served boundary on T rows (compute_mhc_pre_num_splits)."""
    _moet_mhc_warmup._MOET_SP_PIN_TOKENS = num_tokens


@_moet_contextlib.contextmanager
def _moet_sp_engram_mode(engram, sharded: bool):
    """Engram's own sequence-parallel mode for one call: the residual is this rank's row shard, the hash ids are
    every row's (Engram.embed all-gathers the staged rows as today and selects the shard; wkv runs on the shard)."""
    if not sharded:
        yield
        return
    # EngramBatch stages every row's wkv input (DP-sharded tables only); the shard would read the wrong rows
    assert getattr(engram, "_batch", None) is None, "prefill SP: Engram with a batched wkv input is not sharded"
    previous = engram.use_sequence_parallel
    engram.use_sequence_parallel = True
    try:
        yield
    finally:
        engram.use_sequence_parallel = previous


def _moet_sp_rows_to_full(shard: torch.Tensor, rows: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """The replay rows of a row-sharded per-token state in a full-row tensor (zeros elsewhere). Each rank picks the
    rows that fall in its shard, an all-gather brings them together: copies only, no arithmetic, the bits stay."""
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = _moet_tp_rank()
    chunk = shard.shape[0]
    owner = torch.div(rows, chunk, rounding_mode="floor")
    picked = shard.index_select(0, (rows - tp_rank * chunk).clamp_(0, chunk - 1))
    keep = (owner == tp_rank).view(-1, *([1] * (shard.ndim - 1)))
    picked = torch.where(keep, picked, torch.zeros((), dtype=shard.dtype, device=shard.device))
    gathered = _moet_all_gather(picked, 0).view(tp_size, rows.shape[0], *shard.shape[1:])
    row_values = gathered[owner, torch.arange(rows.shape[0], device=rows.device)]
    full = shard.new_zeros((num_tokens, *shard.shape[1:]))
    return full.index_copy_(0, rows, row_values)


def _moet_sp_decoder_replay(
    replay,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor | None,
    pre_mix: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """DecoderReplayLayers.__call__ for a sequence-parallel step with a replay batch: the first replay layer's KV for
    every row from the sharded boundary (write_kv reduce-scatters, runs the boundary on this rank's rows and gathers
    the projections), then the replay rows' states brought to full rows and the replay layers run on them unsharded,
    exactly as in the served step."""
    replay_batch = replay.replay_batch
    assert replay_batch is not None
    replay.write_batch_kv(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)
    rows, num_tokens = replay_batch.rows, positions.shape[0]
    pre_mix, post_mix, res_mix, residual = (
        _moet_sp_rows_to_full(state, rows, num_tokens) for state in (pre_mix, post_mix, res_mix, residual)
    )
    return replay._run(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)


class DeepseekV4MoE(DeepseekV4MoEBase):
'''

# the decoder layer's forward: the step's sharding is read off the state, the split-k pin set for the layer
LAYER_FWD_ANCHOR = (
    "        previous_aux: torch.Tensor | None = None\n"
    "        mhc_stream = self.mhc_stream\n"
    "        if mhc_stream is not None and (\n"
)
LAYER_FWD_NEW = (
    "        previous_aux: torch.Tensor | None = None\n"
    f"        # {MARKER}: the hc state (or the embedding, in the first layer) is this rank's row shard\n"
    "        moet_sp = _moet_sp_sharded(residual if residual is not None else x, positions)\n"
    "        _moet_sp_pin(positions.shape[0] if moet_sp else None)\n"
    "        mhc_stream = self.mhc_stream\n"
    "        if mhc_stream is not None and (\n"
)

# the Engram layer's first boundary
LAYER_ENGRAM_ANCHOR = (
    "            assert isinstance(x, torch.Tensor)\n"
    "            if self.fuse_mhc_all_reduce:\n"
    "                x = _moet_p2p_all_reduce(x)  # [vllm-moet] sm_120 P2P all-reduce\n"
    "            previous_post = mhc_post_tilelang(x, residual, post_mix, res_mix)\n"
    "            if capture_previous_aux:\n"
    "                previous_aux = previous_post.mean(dim=1)\n"
    "            residual = self.engram(\n"
    "                previous_post,\n"
    "                engram_hashes[:, self.engram.layer_hash_index],\n"
    "                engram_mask,\n"
    "            )\n"
)
LAYER_ENGRAM_NEW = (
    "            assert isinstance(x, torch.Tensor)\n"
    f"            if moet_sp:  # {MARKER}: this rank's rows of the sum; Engram in its sequence-parallel mode\n"
    "                x = _moet_sp_reduce_scatter(x)\n"
    "            elif self.fuse_mhc_all_reduce:\n"
    "                x = _moet_p2p_all_reduce(x)  # [vllm-moet] sm_120 P2P all-reduce\n"
    "            previous_post = mhc_post_tilelang(x, residual, post_mix, res_mix)\n"
    "            if capture_previous_aux:\n"
    "                previous_aux = previous_post.mean(dim=1)\n"
    "            with _moet_sp_engram_mode(self.engram, moet_sp):\n"
    "                residual = self.engram(\n"
    "                    previous_post,\n"
    "                    engram_hashes[:, self.engram.layer_hash_index],\n"
    "                    engram_mask,\n"
    "                )\n"
)

# the regular first boundary (the reduce-scatter goes in front of the call; the call itself no longer reduces)
LAYER_B1_ANCHOR = (
    "            # The collapse already reads the post-mapped streams, so the mean\n"
    "            # aux consumers want comes out of the same kernel.\n"
)
LAYER_B1_NEW = (
    "            # The collapse already reads the post-mapped streams, so the mean\n"
    "            # aux consumers want comes out of the same kernel.\n"
    f"            if moet_sp:  # {MARKER}: this rank's rows of the sum, the boundary on them\n"
    "                x = _moet_sp_reduce_scatter(x)\n"
)
LAYER_B1_TAIL_ANCHOR = (
    "                reduce_results=self.fuse_mhc_all_reduce,\n"
    "            )\n"
    "            if capture_previous_aux:\n"
    "                previous_aux = aux\n"
)
LAYER_B1_TAIL_NEW = (
    f"                reduce_results=self.fuse_mhc_all_reduce and not moet_sp,  # {MARKER}\n"
    "            )\n"
    "            if capture_previous_aux:\n"
    "                previous_aux = aux\n"
)
# after the attention: the second boundary reduce-scatters too
LAYER_ATTN_RS_ANCHOR = (
    "        if self.use_sequence_parallel and self.attn.gemm_rs is None:\n"
    "            x = sp_reduce_scatter(x)\n"
)
LAYER_ATTN_RS_NEW = (
    f"        if (self.use_sequence_parallel and self.attn.gemm_rs is None) or moet_sp:  # {MARKER}\n"
    "            x = _moet_sp_reduce_scatter(x) if moet_sp else sp_reduce_scatter(x)\n"
)
LAYER_B2_TAIL_ANCHOR = (
    "            reduce_results=self.fuse_mhc_all_reduce,\n"
    "        )\n"
    "        if self.fuse_mhc_all_reduce and self.ffn.defers_finalize(x.shape[0]):\n"
)
LAYER_B2_TAIL_NEW = (
    f"            reduce_results=self.fuse_mhc_all_reduce and not moet_sp,  # {MARKER}\n"
    "        )\n"
    f"        if moet_sp:  # {MARKER}: the TP experts, the router and the shared expert read every row\n"
    "            x = sp_all_gather(x)[: positions.shape[0]]\n"
    "        if self.fuse_mhc_all_reduce and self.ffn.defers_finalize(x.shape[0]):\n"
)
LAYER_RETURN_ANCHOR = "        return x, residual, post_mix, res_mix, ffn_pre, previous_aux\n"
LAYER_RETURN_NEW = (
    f"        _moet_sp_pin(None)  # {MARKER}\n"
    "        return x, residual, post_mix, res_mix, ffn_pre, previous_aux\n"
)

# write_kv (the first replay layer's KV for every row)
WRITE_KV_ANCHOR = (
    "        # The replay batch rules out Engram and sequence parallel here.\n"
    "        assert self.engram is None and not self.use_sequence_parallel\n"
    "        *_, x, _, _ = mhc_shifted_post_pre(\n"
)
WRITE_KV_NEW = (
    "        # The replay batch rules out Engram and sequence parallel here.\n"
    "        assert self.engram is None and not self.use_sequence_parallel\n"
    f"        moet_sp = _moet_sp_sharded(residual, positions)  # {MARKER}: the boundary on this rank's rows\n"
    "        if moet_sp:\n"
    "            x = _moet_sp_reduce_scatter(x)\n"
    "        _moet_sp_pin(positions.shape[0] if moet_sp else None)\n"
    "        *_, x, _, _ = mhc_shifted_post_pre(\n"
)
WRITE_KV_TAIL_ANCHOR = (
    "            reduce_results=self.fuse_mhc_all_reduce,\n"
    "        )\n"
    "        self.attn.forward_kv(positions, x)\n"
)
WRITE_KV_TAIL_NEW = (
    f"            reduce_results=self.fuse_mhc_all_reduce and not moet_sp,  # {MARKER}\n"
    "        )\n"
    "        _moet_sp_pin(None)\n"
    "        self.attn.forward_kv(positions, x)  # the row shard: projections on it, gathered inside\n"
)

# the model: the step gate and the sharding of the embedding
MODEL_GATE_METHOD_ANCHOR = (
    "        else:\n"
    "            self._mtp_hidden_buffer = None\n"
    "\n"
    "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n"
)
MODEL_GATE_METHOD = f'''        else:
            self._mtp_hidden_buffer = None

    def _moet_prefill_sp_step(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> bool:
        """{MARKER}: whether this step runs its mHC boundaries sequence parallel (VLLM_MOET_PREFILL_SP=1):
        an eager step of >= VLLM_MOET_PREFILL_SP_MIN_TOKENS rows, TP >= 2 without expert / pipeline / context
        parallelism or microbatching, the boundaries owning the reductions, the embedding 2-D (first PP rank)."""
        _moet_sp_pin(None)
        if _moet_prefill_sp_mode() == 0 or not self.fuse_mhc_all_reduce or self.use_sequence_parallel:
            return False
        tp_size = get_tensor_model_parallel_world_size()
        # A shard of <= 32 rows would take the fused post + pre mHC kernel (mhc_fused_post_pre_split_config), whose
        # bits differ from the split-k path the full rows take: the shard stays at >= 33 rows whatever the switch says.
        if positions.shape[0] < max(_moet_prefill_sp_min_tokens(), 33 * tp_size) or hidden_states.dim() != 2:
            return False
        parallel_config = self.parallel_config
        if (
            tp_size < 2
            or parallel_config.pipeline_parallel_size != 1
            or parallel_config.prefill_context_parallel_size > 1
            or parallel_config.enable_expert_parallel
            or parallel_config.use_ubatching
            or self.use_native_mega_moe
        ):
            return False
        if torch.cuda.is_current_stream_capturing() or in_piecewise_cudagraph():
            return False
        logger.info_once(
            "vllm-moet sm_120 prefill SP: eager steps of >= %d tokens reduce-scatter the sublayer outputs and run "
            "the mHC boundaries on this rank's row shard (VLLM_MOET_PREFILL_SP=0 turns it off)",
            _moet_prefill_sp_min_tokens(),
        )
        return True

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
'''
MODEL_SHARD_ANCHOR = (
    "        full_num_tokens = positions.shape[0]\n"
    "        if self.use_sequence_parallel:\n"
    "            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():\n"
)
MODEL_SHARD_NEW = (
    "        full_num_tokens = positions.shape[0]\n"
    f"        if self._moet_prefill_sp_step(hidden_states, positions):  # {MARKER}: the hc state is sharded by rows\n"
    "            hidden_states = sp_shard(hidden_states)\n"
    "        if self.use_sequence_parallel:\n"
    "            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():\n"
)
MODEL_REPLAY_ANCHOR = (
    "        late_aux: list[torch.Tensor] = []\n"
    "        if self.decoder_replay_layers is not None:\n"
    "            hidden_states, pre_mix, *late_aux = self.decoder_replay_layers(\n"
    "                hidden_states,\n"
    "                positions,\n"
    "                input_ids,\n"
    "                pre_mix,\n"
    "                post_mix,\n"
    "                res_mix,\n"
    "                residual,\n"
    "            )\n"
    "        else:\n"
    "            hidden_states = self._collapse(\n"
    "                hidden_states,\n"
    "                residual,\n"
    "                post_mix,\n"
    "                res_mix,\n"
    "                aux_hidden_by_layer,\n"
    "                full_num_tokens,\n"
    "            )\n"
)
MODEL_REPLAY_NEW = (
    "        late_aux: list[torch.Tensor] = []\n"
    "        if self.decoder_replay_layers is not None:\n"
    "            if (\n"
    f"                _moet_sp_sharded(residual, positions)  # {MARKER}: KV for every row from the shards, the replay rows gathered\n"
    "                and self.decoder_replay_layers.replay_batch is not None\n"
    "                and not in_piecewise_cudagraph()\n"
    "            ):\n"
    "                hidden_states, pre_mix, *late_aux = _moet_sp_decoder_replay(\n"
    "                    self.decoder_replay_layers,\n"
    "                    hidden_states,\n"
    "                    positions,\n"
    "                    input_ids,\n"
    "                    pre_mix,\n"
    "                    post_mix,\n"
    "                    res_mix,\n"
    "                    residual,\n"
    "                )\n"
    "            else:\n"
    "                hidden_states, pre_mix, *late_aux = self.decoder_replay_layers(\n"
    "                    hidden_states,\n"
    "                    positions,\n"
    "                    input_ids,\n"
    "                    pre_mix,\n"
    "                    post_mix,\n"
    "                    res_mix,\n"
    "                    residual,\n"
    "                )\n"
    "        else:\n"
    "            hidden_states = self._collapse(\n"
    "                hidden_states,\n"
    "                residual,\n"
    "                post_mix,\n"
    "                res_mix,\n"
    "                aux_hidden_by_layer,\n"
    "                full_num_tokens,\n"
    "            )\n"
    f"        if _moet_sp_sharded(hidden_states, positions):  # {MARKER}: a step without a replay batch ends sharded\n"
    "            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]\n"
    "            pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]\n"
)
MODEL_AUX_ANCHOR = (
    "                # idx is the one-based id of the layer whose post this is.\n"
    "                if self.use_sequence_parallel:\n"
    "                    previous_aux = sp_all_gather(previous_aux)[:full_num_tokens]\n"
)
MODEL_AUX_NEW = (
    "                # idx is the one-based id of the layer whose post this is.\n"
    f"                if self.use_sequence_parallel or _moet_sp_sharded(previous_aux, positions):  # {MARKER}\n"
    "                    previous_aux = sp_all_gather(previous_aux)[:full_num_tokens]\n"
)
MODEL_COLLAPSE_ANCHOR = (
    "        if self.fuse_mhc_all_reduce:\n"
    "            hidden_states = _moet_p2p_all_reduce(hidden_states)  # [vllm-moet] sm_120 P2P all-reduce\n"
    "        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)\n"
    "        if self.end_layer in self.aux_hidden_state_layers:\n"
    "            final_aux = hidden_states.mean(dim=1)\n"
    "            if self.use_sequence_parallel:\n"
)
MODEL_COLLAPSE_NEW = (
    f"        moet_sp = hidden_states.shape[0] != residual.shape[0]  # {MARKER}: the hc state is this rank's row shard\n"
    "        if moet_sp:\n"
    "            hidden_states = _moet_sp_reduce_scatter(hidden_states)\n"
    "        elif self.fuse_mhc_all_reduce:\n"
    "            hidden_states = _moet_p2p_all_reduce(hidden_states)  # [vllm-moet] sm_120 P2P all-reduce\n"
    "        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)\n"
    "        if self.end_layer in self.aux_hidden_state_layers:\n"
    "            final_aux = hidden_states.mean(dim=1)\n"
    "            if self.use_sequence_parallel or moet_sp:\n"
)

MODEL_EDITS = [
    ("helpers", MODEL_HELPERS_ANCHOR, MODEL_HELPERS),
    ("layer forward prologue", LAYER_FWD_ANCHOR, LAYER_FWD_NEW),
    ("Engram boundary", LAYER_ENGRAM_ANCHOR, LAYER_ENGRAM_NEW),
    ("boundary 1", LAYER_B1_ANCHOR, LAYER_B1_NEW),
    ("boundary 1 tail", LAYER_B1_TAIL_ANCHOR, LAYER_B1_TAIL_NEW),
    ("boundary 2 reduce-scatter", LAYER_ATTN_RS_ANCHOR, LAYER_ATTN_RS_NEW),
    ("boundary 2 tail", LAYER_B2_TAIL_ANCHOR, LAYER_B2_TAIL_NEW),
    ("layer return", LAYER_RETURN_ANCHOR, LAYER_RETURN_NEW),
    ("write_kv", WRITE_KV_ANCHOR, WRITE_KV_NEW),
    ("write_kv tail", WRITE_KV_TAIL_ANCHOR, WRITE_KV_TAIL_NEW),
    ("step gate", MODEL_GATE_METHOD_ANCHOR, MODEL_GATE_METHOD),
    ("embedding shard", MODEL_SHARD_ANCHOR, MODEL_SHARD_NEW),
    ("replay", MODEL_REPLAY_ANCHOR, MODEL_REPLAY_NEW),
    ("aux gather", MODEL_AUX_ANCHOR, MODEL_AUX_NEW),
    ("collapse", MODEL_COLLAPSE_ANCHOR, MODEL_COLLAPSE_NEW),
]

# ---------------------------------------------------------------------------------------------------------------
# attention.py
# ---------------------------------------------------------------------------------------------------------------
ATTN_HELPER_ANCHOR = (
    "# From this many tokens, a layer with shard_fused_wqa_wkv set projects one token\n"
    "# shard per TP rank and all-gathers the result.\n"
    "_SHARD_FUSED_WQA_WKV_MIN_TOKENS = 2048\n"
)
ATTN_HELPER = f'''# From this many tokens, a layer with shard_fused_wqa_wkv set projects one token
# shard per TP rank and all-gathers the result.
_SHARD_FUSED_WQA_WKV_MIN_TOKENS = 2048


# {MARKER}: the input projections on this rank's row shard of the layer input
def _moet_sp_aux_mode() -> str:
    """VLLM_MOET_PREFILL_SP_AUX: "full" (default) all-gathers the layer input at a layer with a compressor or an
    indexer and projects every row there (cuBLAS picks the bf16 GEMMs' split-k by the row count, so a shard need not
    reproduce the full-row bits); "shard" projects on the shard everywhere and gathers the projections."""
    import os

    mode = os.environ.get("VLLM_MOET_PREFILL_SP_AUX", "full")
    return mode if mode in ("full", "shard") else "full"


def _moet_sp_input_projections(attn, hidden_states: torch.Tensor, num_tokens: int):
    """fused_wqa_wkv, the compressor's kv_score and the indexer's weights for `num_tokens` rows when `hidden_states`
    is this rank's row shard of the layer input (the sequence-parallel prefill step; an unsharded input runs the
    served path). The weights are replicated, so each rank projects its rows and the projections are all-gathered
    ([T, 1792] bf16 for fused_wqa_wkv instead of the [T, 5120] input). Returns (stand-in, projections): the stand-in
    is a [num_tokens, 0] tensor of the input's dtype and device - the attention reads only the row count, dtype and
    device off the layer input after the projections (allocations, indexer workspaces)."""
    if hidden_states.shape[0] == num_tokens:
        return hidden_states, attn._run_parallel_input_projections(hidden_states)
    if _moet_sp_aux_mode() == "full" and (attn.compressor is not None or attn.indexer is not None):
        full = sp_all_gather(hidden_states)[:num_tokens]
        return full, attn._run_parallel_input_projections(full)
    qr_kv, kv_score, indexer_weights = attn._run_parallel_input_projections(hidden_states)
    gathered = tuple(
        None if t is None else sp_all_gather(t.contiguous())[:num_tokens] for t in (qr_kv, kv_score, indexer_weights)
    )
    return hidden_states.new_empty((num_tokens, 0)), gathered
'''
ATTN_FWD_ANCHOR = (
    "        attn_out = self._alloc_attn_out(hidden_states.shape[0], hidden_states)\n"
    "\n"
    "        # Keep the attention input preparation in the captured graph. Only the\n"
    "        # sparse indexer and MLA attention run in the eager break below.\n"
    "        qr_kv, kv_score, indexer_weights = self._run_parallel_input_projections(\n"
    "            hidden_states\n"
    "        )\n"
)
ATTN_FWD_NEW = (
    f"        # {MARKER}: hidden_states may be this rank's row shard of the layer input (positions has every row);\n"
    "        # the projections come back for every row, hidden_states as a row-count stand-in.\n"
    "        # Keep the attention input preparation in the captured graph. Only the\n"
    "        # sparse indexer and MLA attention run in the eager break below.\n"
    "        hidden_states, (qr_kv, kv_score, indexer_weights) = _moet_sp_input_projections(\n"
    "            self, hidden_states, positions.shape[0]\n"
    "        )\n"
    "        attn_out = self._alloc_attn_out(hidden_states.shape[0], hidden_states)\n"
)
ATTN_KV_ANCHOR = (
    "        attn_metadata = get_forward_context().attn_metadata\n"
    "        qr_kv, kv_score, _ = self._run_parallel_input_projections(hidden_states)\n"
)
ATTN_KV_NEW = (
    "        attn_metadata = get_forward_context().attn_metadata\n"
    f"        _, (qr_kv, kv_score, _) = _moet_sp_input_projections(  # {MARKER}: the row shard, projections gathered\n"
    "            self, hidden_states, positions.shape[0]\n"
    "        )\n"
)
ATTN_EDITS = [
    ("helpers", ATTN_HELPER_ANCHOR, ATTN_HELPER),
    ("forward", ATTN_FWD_ANCHOR, ATTN_FWD_NEW),
    ("forward_kv", ATTN_KV_ANCHOR, ATTN_KV_NEW),
]

# ---------------------------------------------------------------------------------------------------------------
# warmup.py
# ---------------------------------------------------------------------------------------------------------------
WARMUP_ANCHOR = (
    "def compute_mhc_pre_num_splits(input_size: int, num_tokens: int) -> int:\n"
    "    splits = compute_num_split(64, input_size, cdiv(num_tokens, 64))\n"
)
WARMUP_NEW = (
    f"# {MARKER}: the split-k count of the unsharded step, set by the decoder layer around a boundary that runs on\n"
    "# this rank's row shard, so the shard's pre-mix projection sums the same k-slices as the full rows' (None = off).\n"
    "_MOET_SP_PIN_TOKENS: int | None = None\n"
    "\n"
    "\n"
    "def compute_mhc_pre_num_splits(input_size: int, num_tokens: int) -> int:\n"
    "    if _MOET_SP_PIN_TOKENS is not None:\n"
    "        num_tokens = _MOET_SP_PIN_TOKENS\n"
    "    splits = compute_num_split(64, input_size, cdiv(num_tokens, 64))\n"
)
WARMUP_EDITS = [("split pin", WARMUP_ANCHOR, WARMUP_NEW)]

FILES = [
    ("models/deepseek_v41/nvidia/model.py", MODEL_EDITS),
    ("models/deepseek_v41/attention.py", ATTN_EDITS),
    ("model_executor/kernels/mhc/warmup.py", WARMUP_EDITS),
]


def patch_text(src: str, edits, name: str, check: bool) -> tuple[str, str]:
    """Apply the edits; returns (new source, status) with status in {applied, already, missing:<edit>}."""
    if MARKER in src:
        return src, "already"
    for label, anchor, new in edits:
        n = src.count(anchor)
        if n != 1:
            return src, f"missing:{label} ({n} matches)"
    if check:
        return src, "applied"
    for _label, anchor, new in edits:
        src = src.replace(anchor, new, 1)
    return src, "applied"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dist", type=Path, default=DEFAULT_DIST, help="the vllm package directory")
    ap.add_argument("--check", action="store_true", help="dry run: verify the anchors, change nothing")
    args = ap.parse_args()
    rc = 0
    for rel, edits in FILES:
        path = args.dist / rel
        if not path.exists():
            print(f"{rel}: not found under {args.dist}")
            rc = 1
            continue
        src = path.read_text()
        new, status = patch_text(src, edits, rel, args.check)
        if status.startswith("missing"):
            print(f"{rel}: anchor {status} - not patched")
            rc = 1
            continue
        if status == "already":
            print(f"{rel}: already patched")
            continue
        if args.check:
            print(f"{rel}: all {len(edits)} anchors present (dry run)")
        else:
            path.write_text(new)
            print(f"{rel}: patched ({len(edits)} edits)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
