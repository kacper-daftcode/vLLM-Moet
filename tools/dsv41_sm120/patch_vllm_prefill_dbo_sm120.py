#!/usr/bin/env python3
"""Dual-batch overlap of the TP all-reduces in the DeepSeek-V4.1 prefill on sm_120 (TP4, no DP / EP).

A 4096-token prefill chunk on 4x RTX PRO 6000 spends a third of its time in NCCL: two [4096, 5120] bf16
all-reduces per non-trimmed layer, 1.6 ms each on the PCIe ring, with 180 SMs idle (docs/dsv41-sm120-port.md,
"Where a 163K-token prefill spends its 9 s"). vLLM's dual-batch overlap (`--enable-dbo`) splits a step into two
microbatches on two threads that hand the GPU to each other at the collectives, so one half's all-reduce runs on a
comm stream behind the other half's GEMMs - measured on this host without the model at 80-97 % of the all-reduce
hidden (nccl_graph/nccl_overlap_bench.py). In vLLM main 21d93d0d that machinery exists for DP + EP on the V2 runner
only: the runner builds it for data_parallel_size > 1, the config refuses it with Engram, speculative decoding and
multimodal models, the model turns the decoder SWA bounded replay off under it, the MoE asserts it is not running
under it when its prepare/finalize does no all-to-all, and the only yield points sit in the EP all-to-all. This patch
set makes it serve the TP4 prefill:

  vllm/v1/worker/ubatching.py           UBatchContext.token_offset (vllm#57773), restore hooks run on every resume
  vllm/v1/worker/gpu/ubatch_utils.py     the runner without DP; a high-priority comm stream; the per-step hook the
                                         model prepares whole-batch state in; aux hidden states merged per element
  vllm/v1/worker/gpu/dp_utils.py         the microbatch decision without DP: eager steps over the token thresholds
  vllm/config/vllm.py                    Engram / DSpark / multimodal / all2all-backend gates admit this model
  .../fused_moe/modular_kernel.py        the no-DP/EP prepare/finalize may run under DBO (it has no collective)
  .../nvidia/ops/mhc.py                  the sublayer all-reduce of mhc_shifted_post_pre in the comm region; the
                                         sm_120 mHC overlap no longer refused with ubatching (its side stream is
                                         only taken inside FULL graphs, which stay one batch)
  .../nvidia/model.py                    the Engram-layer and final all-reduces in the comm region; the top-k and
                                         candidate-block row buffers per microbatch (RowBufferSwitch); the Engram
                                         hashes + table rows prepared once for the whole batch; the decoder SWA
                                         bounded replay kept on under microbatching
  .../deepseek_v41/decoder_replay_layers.py  the replay batch per microbatch, the replay forward context installed
                                         on the microbatch context for the duration (the yields inside see it)
  .../nvidia/model_state.py              the replay batch's buffers, builders and staging per microbatch
  .../common/engram.py                   the table rows at the microbatch's offset; the head all-gather in the
                                         comm region
  vllm/v1/worker/moet_prefill_dbo.py     (new) the comm region, RowBufferSwitch, output merge (prefill_dbo/)

Off by default: nothing changes unless `vllm serve` gets `--enable-dbo`. Serve with
  --enable-dbo --dbo-decode-token-threshold 4096 --dbo-prefill-token-threshold 1024
so decode steps (<= 64 tokens, FULL graphs) stay one batch and prefill steps of >= 1024 tokens split in two.
Runtime: VLLM_MOET_DBO_COMM_PRIORITY (-1). Requires step 10 (the fused all-reduce of the sm_120 mHC overlap,
VLLM_MOET_MHC_FUSE_ALLREDUCE=1): that is where the two all-reduces per layer are.

Not bit-identical to the single-batch step by construction: a half-chunk all-reduce chunks the ring differently
(bf16 sums in another order), so the gate is a quality window.

Base: vLLM main 21d93d0d after Dockerfile steps 10 and 12. Idempotent, anchor-based. Usage:
    python3 patch_vllm_prefill_dbo_sm120.py [--site-packages DIR] [--helper FILE] [--check]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_SITE = Path("/usr/local/lib/python3.12/dist-packages")
DEFAULT_HELPER = Path(__file__).resolve().parent / "prefill_dbo" / "moet_prefill_dbo.py"
HELPER_DST = "vllm/v1/worker/moet_prefill_dbo.py"
MARKER = "# [vllm-moet] prefill DBO"

# ----------------------------------------------------------------------------------------------------------------
# vllm/v1/worker/ubatching.py
# ----------------------------------------------------------------------------------------------------------------
UBATCHING = [
    (
        """        schedule: str = "default",
    ):
        self.id = id
        self.comm_stream = comm_stream
""",
        """        schedule: str = "default",
        token_offset: int = 0,
    ):
        self.id = id
        self.token_offset = token_offset  # [vllm-moet] prefill DBO: this microbatch's first batch row
        self.comm_stream = comm_stream
""",
    ),
    (
        """    def _restore_context(self):
        forward_context._forward_context = self.forward_context
""",
        """    def _restore_context(self):
        forward_context._forward_context = self.forward_context
        # [vllm-moet] prefill DBO: per-microbatch state the model keeps outside the forward context
        # (RowBufferSwitch) follows the thread that resumes.
        for hook in _RESTORE_HOOKS:
            hook(self.id)
""",
    ),
    (
        """def dbo_current_ubatch_id() -> int:
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return 0
    return _THREAD_ID_TO_CONTEXT[threading.get_ident()]
""",
        """def dbo_current_ubatch_id() -> int:
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return 0
    return _THREAD_ID_TO_CONTEXT[threading.get_ident()]


# [vllm-moet] prefill DBO
_RESTORE_HOOKS: list = []


def dbo_register_restore_hook(hook) -> None:
    \"\"\"Run ``hook(ubatch_id)`` whenever a microbatch thread enters or resumes.\"\"\"
    _RESTORE_HOOKS.append(hook)


def dbo_current_ubatch_token_offset() -> int:
    \"\"\"Row offset of the calling microbatch inside the parent batch (0 outside DBO).\"\"\"
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return 0
    ctx = _CURRENT_CONTEXTS[_THREAD_ID_TO_CONTEXT[threading.get_ident()]]
    return ctx.token_offset
""",
    ),
    (
        """    ready_barrier: threading.Barrier,
    schedule: str = "default",
) -> list[UBatchContext]:
""",
        """    ready_barrier: threading.Barrier,
    schedule: str = "default",
    token_offsets: list[int] | None = None,
) -> list[UBatchContext]:
""",
    ),
    (
        """            gpu_compute_done_event=gpu_compute_done_events[i],
            schedule=schedule,
        )
        ctxs.append(ctx)
""",
        """            gpu_compute_done_event=gpu_compute_done_events[i],
            schedule=schedule,
            token_offset=token_offsets[i] if token_offsets else 0,
        )
        ctxs.append(ctx)
""",
    ),
]

# ----------------------------------------------------------------------------------------------------------------
# vllm/v1/worker/gpu/ubatch_utils.py
# ----------------------------------------------------------------------------------------------------------------
UBATCH_UTILS = [
    (
        """from vllm.v1.worker.ubatching import make_ubatch_contexts
from vllm.v1.worker.utils import AttentionGroup
""",
        """from vllm.v1.worker.ubatching import make_ubatch_contexts
from vllm.v1.worker.utils import AttentionGroup
from vllm.v1.worker import moet_prefill_dbo as _moet_dbo  # [vllm-moet] prefill DBO
""",
    ),
    (
        """class UBatchState(NamedTuple):
    \"\"\"Per-microbatch inputs for one dual-batch-overlap step.\"\"\"

    slices: UBatchSlices
    forward_contexts: list[ForwardContext]
""",
        """class UBatchState(NamedTuple):
    \"\"\"Per-microbatch inputs for one dual-batch-overlap step.\"\"\"

    slices: UBatchSlices
    forward_contexts: list[ForwardContext]
    # [vllm-moet] prefill DBO: the whole batch's query offsets, for state the model prepares once per step
    batch_query_start_loc: torch.Tensor | None = None
""",
    ),
    (
        """    first = outputs[0]
    if isinstance(first, IntermediateTensors):
        return IntermediateTensors(
            {
                key: torch.cat([out.tensors[key] for out in outputs], dim=0)
                for key in first.tensors
            }
        )
    if isinstance(first, tuple):
        return tuple(torch.cat(parts, dim=0) for parts in zip(*outputs))
    return torch.cat(outputs, dim=0)
""",
        """    # [vllm-moet] prefill DBO: a tuple's list elements (aux hidden states) merge element-wise
    return _moet_dbo.merge_outputs(outputs)
""",
    ),
    (
        """        self.comm_stream = torch.cuda.Stream(device=device)
        # Threads and graph capture must use the same compute stream.
""",
        """        # [vllm-moet] prefill DBO: high priority, so the all-reduce's blocks get SMs as the other
        # microbatch's GEMM blocks retire (VLLM_MOET_DBO_COMM_PRIORITY).
        self.comm_stream = torch.cuda.Stream(
            device=device, priority=_moet_dbo.comm_stream_priority()
        )
        self._moet_hooked: Any = False  # resolved on first use: the module with the per-step hooks, or None
        # Threads and graph capture must use the same compute stream.
""",
    ),
    (
        """            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i],
                    self.vllm_config,
                    dp_metadata=DPMetadata.make(
                        self.parallel_config, num_tokens, num_tokens_across_dp
                    ),
""",
        """            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i],
                    self.vllm_config,
                    # [vllm-moet] prefill DBO: no DP metadata without DP (DPMetadata.make asserts dp > 1)
                    dp_metadata=(
                        DPMetadata.make(
                            self.parallel_config, num_tokens, num_tokens_across_dp
                        )
                        if dp_size > 1
                        else None
                    ),
""",
    ),
    (
        """        return UBatchState(
            slices=ubatch_slices,
            forward_contexts=self._make_forward_contexts(
                ubatch_slices, attn_metadata, slot_mappings_by_layer, is_padding
            ),
        )
""",
        """        return UBatchState(
            slices=ubatch_slices,
            forward_contexts=self._make_forward_contexts(
                ubatch_slices, attn_metadata, slot_mappings_by_layer, is_padding
            ),
            batch_query_start_loc=input_batch.query_start_loc[: input_batch.num_reqs + 1],
        )
""",
    ),
    (
        """        compute_stream = self.capture_stream if for_capture else current_stream()
        ubatch_contexts = make_ubatch_contexts(
            num_micro_batches=self.num_ubatches,
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=ubatch_state.forward_contexts,
            ready_barrier=self.ready_barrier,
        )
""",
        """        compute_stream = self.capture_stream if for_capture else current_stream()
        ubatch_contexts = make_ubatch_contexts(
            num_micro_batches=self.num_ubatches,
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=ubatch_state.forward_contexts,
            ready_barrier=self.ready_barrier,
            token_offsets=[s.token_slice.start for s in ubatch_slices],
        )
        # [vllm-moet] prefill DBO: state the model prepares once for the whole batch before the threads
        # start (DeepSeek-V4.1: the Engram hashes and table rows), undone when they have joined.
        if self._moet_hooked is False:
            self._moet_hooked = _moet_dbo.find_hooked_module(model)
        hooked = None if for_capture else self._moet_hooked
        if hooked is not None:
            hooked.moet_prepare_ubatched_forward(model_inputs, ubatch_state)
""",
    ),
    (
        """            finally:
                self._pending_finish = None
                stack.close()

            if errors:
""",
        """            finally:
                self._pending_finish = None
                stack.close()
                if hooked is not None:  # [vllm-moet] prefill DBO
                    hooked.moet_finish_ubatched_forward()

            if errors:
""",
    ),
    (
        """    parallel_config = vllm_config.parallel_config
    if not parallel_config.use_ubatching or parallel_config.data_parallel_size <= 1:
        return None

    logger.info_once("Dual batch overlap is enabled.")
""",
        """    parallel_config = vllm_config.parallel_config
    if not parallel_config.use_ubatching:
        return None
    # [vllm-moet] prefill DBO: without DP the decision is this rank's alone (dp_utils.dispatch_cg_and_sync_dp)
    if parallel_config.data_parallel_size <= 1:
        logger.info_once(
            "vllm-moet prefill DBO: eager steps of >= %d tokens (uniform decodes >= %d) run as %d "
            "microbatches, the TP all-reduces of one behind the compute of the other",
            parallel_config.dbo_prefill_token_threshold,
            parallel_config.dbo_decode_token_threshold,
            parallel_config.num_ubatches,
        )
    logger.info_once("Dual batch overlap is enabled.")
""",
    ),
]

# ----------------------------------------------------------------------------------------------------------------
# vllm/v1/worker/gpu/dp_utils.py
# ----------------------------------------------------------------------------------------------------------------
DP_UTILS = [
    (
        """    if dp_size == 1:
        return batch_desc, None

    if dp_sync is not None:
""",
        """    if dp_size == 1:
        # [vllm-moet] prefill DBO: without DP, an eager step over the threshold splits into microbatches here
        # (the DP path below decides the same way across ranks); FULL graphs stay one batch.
        if (
            allow_ubatching
            and parallel_config is not None
            and batch_desc.cg_mode == CUDAGraphMode.NONE
            and check_ubatch_thresholds(parallel_config, num_tokens, uniform_decode)
        ):
            num_ubatches = get_num_ubatches(parallel_config)
            if num_tokens >= 2 * num_ubatches:
                batch_desc = replace(batch_desc, num_ubatches=num_ubatches)
        return batch_desc, None

    if dp_sync is not None:
""",
    ),
]

# ----------------------------------------------------------------------------------------------------------------
# vllm/config/vllm.py
# ----------------------------------------------------------------------------------------------------------------
CONFIG = [
    (
        """        if (
            model_config is not None
            and model_config.architecture == "DeepseekV41ForCausalLM"
            and getattr(model_config.hf_text_config, "engram_layer_ids", None)
            and self.parallel_config.use_ubatching
        ):
            raise ValueError(
                "DeepSeek V4.1 Engram does not support DBO or microbatching. "
                "Disable --enable-dbo and set --ubatch-size to 0."
            )
""",
        """        if (
            model_config is not None
            and model_config.architecture == "DeepseekV41ForCausalLM"
            and getattr(model_config.hf_text_config, "engram_layer_ids", None)
            and self.parallel_config.use_ubatching
            # [vllm-moet] prefill DBO: the patched model hashes and fetches the Engram rows for the whole
            # batch before the microbatch threads start (moet_prepare_ubatched_forward)
            and self.parallel_config.data_parallel_size > 1
        ):
            raise ValueError(
                "DeepSeek V4.1 Engram does not support DBO or microbatching. "
                "Disable --enable-dbo and set --ubatch-size to 0."
            )
""",
    ),
    (
        """        if self.parallel_config.use_ubatching:
            a2a_backend = self.parallel_config.all2all_backend
            assert a2a_backend in [
                "deepep_low_latency",
                "deepep_high_throughput",
                "nixl_ep",
            ], (
""",
        """        if self.parallel_config.use_ubatching:
            a2a_backend = self.parallel_config.all2all_backend
            # [vllm-moet] prefill DBO: without expert parallelism no all-to-all backend is built; the
            # microbatches overlap the TP all-reduces instead
            assert not self.parallel_config.enable_expert_parallel or a2a_backend in [
                "deepep_low_latency",
                "deepep_high_throughput",
                "nixl_ep",
            ], (
""",
    ),
    (
        """        # TODO: DBO with model runner V2 is under development.
        # It should be enabled with explicit VLLM_USE_V2_MODEL_RUNNER environ.
        # Remove it when stable.
        if envs.VLLM_USE_V2_MODEL_RUNNER is None:
            return ["dual batch overlap"]

        unsupported: list[str] = []
        model_config = self.model_config
        parallel_config = self.parallel_config

        if self.lora_config is not None:
            unsupported.append("dual batch overlap with LoRA")
        if self.speculative_config is not None:
            unsupported.append("dual batch overlap with speculative decoding")
""",
        """        # [vllm-moet] prefill DBO: DeepSeek-V4.1 runs on the V2 runner only (DSpark), its DSpark drafter
        # builds its own attention metadata and reads the merged aux hidden states, and its vision encoder
        # runs before the decoder forward that is split - none of the three gates below applies to it.
        _moet_dsv41 = (
            self.model_config is not None
            and self.model_config.architecture == "DeepseekV41ForCausalLM"
            and self.parallel_config.data_parallel_size == 1
        )
        # TODO: DBO with model runner V2 is under development.
        # It should be enabled with explicit VLLM_USE_V2_MODEL_RUNNER environ.
        # Remove it when stable.
        if envs.VLLM_USE_V2_MODEL_RUNNER is None and not _moet_dsv41:
            return ["dual batch overlap"]

        unsupported: list[str] = []
        model_config = self.model_config
        parallel_config = self.parallel_config

        if self.lora_config is not None:
            unsupported.append("dual batch overlap with LoRA")
        if self.speculative_config is not None and not (
            _moet_dsv41 and self.speculative_config.method == "dspark"
        ):
            unsupported.append("dual batch overlap with speculative decoding")
""",
    ),
    (
        """        if model_config is not None and (
            model_config.is_multimodal_model or model_config.is_encoder_decoder
        ):
            unsupported.append("dual batch overlap with multimodal models")
""",
        """        if (
            model_config is not None
            and (model_config.is_multimodal_model or model_config.is_encoder_decoder)
            and not _moet_dsv41
        ):
            unsupported.append("dual batch overlap with multimodal models")
""",
    ),
]

# ----------------------------------------------------------------------------------------------------------------
# vllm/model_executor/layers/fused_moe/modular_kernel.py
# ----------------------------------------------------------------------------------------------------------------
MODULAR_KERNEL = [
    (
        """        if not self.prepare_finalize.supports_async():
            # We shouldn't be running an a2a kernel that doesn't
            # support async prepare/finalize
            # TODO(lucas): enable in follow-up
            assert not dbo_enabled()
""",
        """        if not self.prepare_finalize.supports_async():
            # We shouldn't be running an a2a kernel that doesn't
            # support async prepare/finalize
            # TODO(lucas): enable in follow-up
            # [vllm-moet] prefill DBO: the no-DP/EP prepare has no collective to overlap
            assert not dbo_enabled() or _moet_no_dp_ep(self.prepare_finalize)
""",
    ),
    (
        """        if not self.prepare_finalize.supports_async():
            assert not dbo_enabled()

            self.prepare_finalize.finalize(
""",
        """        if not self.prepare_finalize.supports_async():
            assert not dbo_enabled() or _moet_no_dp_ep(self.prepare_finalize)  # [vllm-moet] prefill DBO

            self.prepare_finalize.finalize(
""",
    ),
    (
        """from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_maybe_run_recv_hook,
    dbo_register_recv_hook,
    dbo_yield,
)
""",
        """from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_maybe_run_recv_hook,
    dbo_register_recv_hook,
    dbo_yield,
)


def _moet_no_dp_ep(prepare_finalize) -> bool:
    \"\"\"[vllm-moet] prefill DBO: a prepare/finalize without DP or EP collectives may run under microbatching.\"\"\"
    return type(prepare_finalize).__name__.startswith("MoEPrepareAndFinalizeNoDPEP")
""",
    ),
]

# ----------------------------------------------------------------------------------------------------------------
# vllm/models/deepseek_v41/nvidia/ops/mhc.py (after step 10)
# ----------------------------------------------------------------------------------------------------------------
MHC = [
    (
        """        else:
            x = get_tp_group().all_reduce(x)
    if stream is not None:
        if layer_input is None:
""",
        """        else:
            # [vllm-moet] prefill DBO: the comm region (a plain all-reduce outside the microbatch threads)
            x = _moet_dbo.all_reduce(x)
    if stream is not None:
        if layer_input is None:
""",
    ),
]
# Optional (only with step 10 applied): the sm_120 overlap support check drops the ubatching clause.
MHC_OPTIONAL = [
    (
        """        and not vllm_config.parallel_config.use_ubatching
        and is_deep_gemm_supported()
        and _moet_mhc_post_norm() is not None
""",
        """        and is_deep_gemm_supported()  # [vllm-moet] prefill DBO: ubatching no longer refused (FULL graphs stay one batch)
        and _moet_mhc_post_norm() is not None
""",
    ),
]
MHC_IMPORT = (
    "from vllm.v1.worker import moet_prefill_dbo as _moet_dbo  # [vllm-moet] prefill DBO\n"
)

# ----------------------------------------------------------------------------------------------------------------
# vllm/models/deepseek_v41/nvidia/model.py
# ----------------------------------------------------------------------------------------------------------------
MODEL = [
    (
        """from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from ..common.engram import (
""",
        """from vllm.v1.worker.ubatching import dbo_current_ubatch_id
from vllm.v1.worker import moet_prefill_dbo as _moet_dbo  # [vllm-moet] prefill DBO

from ..common.engram import (
""",
    ),
    # The Engram layer's pre-boundary all-reduce.
    (
        """            assert isinstance(x, torch.Tensor)
            if self.fuse_mhc_all_reduce:
                x = tensor_model_parallel_all_reduce(x)
            previous_post = mhc_post_tilelang(x, residual, post_mix, res_mix)
""",
        """            assert isinstance(x, torch.Tensor)
            if self.fuse_mhc_all_reduce:
                x = _moet_dbo.all_reduce(x)  # [vllm-moet] prefill DBO: the comm region
            previous_post = mhc_post_tilelang(x, residual, post_mix, res_mix)
""",
    ),
    # The row buffers per microbatch; the per-step hooks' state.
    (
        """        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
""",
        """        else:
            self._mtp_hidden_buffer = None

        # [vllm-moet] prefill DBO: the indexer's top-k indices and the candidate blocks are published for
        # later layers by batch row; each microbatch gets its own storage (swapped in when its thread
        # resumes). The Engram hashes and rows are prepared for the whole batch before the threads start.
        self._moet_dbo_engram: tuple[torch.Tensor, torch.Tensor] | None = None
        self._moet_row_buffers: _moet_dbo.RowBufferSwitch | None = None
        if vllm_config.parallel_config.use_ubatching:
            self._moet_row_buffers = _moet_dbo.RowBufferSwitch(
                [
                    buf
                    for buf in (self.topk_indices_buffer, self.candidate_block_buffer)
                    if buf is not None
                ],
                vllm_config.parallel_config.num_ubatches,
            )

    def moet_prepare_ubatched_forward(self, model_inputs: dict, ubatch_state) -> None:
        \"\"\"[vllm-moet] prefill DBO: on the runner thread, before the microbatch threads start - the
        Engram n-gram hashes of the whole batch and the table rows they fetch (the threads take their
        rows at their token offset), the row buffers on microbatch 0's storage.\"\"\"
        self._moet_dbo_engram = None
        if self._moet_row_buffers is not None:
            self._moet_row_buffers.reset()
        input_ids = model_inputs.get("input_ids")
        query_start_loc = getattr(ubatch_state, "batch_query_start_loc", None)
        if (
            self.engram_hash is None
            or input_ids is None
            or query_start_loc is None
            or not self.engram_hash.ensure_cache()
        ):
            return
        positions = model_inputs["positions"]
        lookback_token_ids = model_inputs.get("lookback_token_ids")
        if lookback_token_ids is None:
            if not self.engram_hash.use_slot_cache:
                raise NotImplementedError(
                    "engram needs `lookback_token_ids` from the model runner"
                )
            num_reqs = query_start_loc.shape[0] - 1
            lookback_token_ids = input_ids.new_full(
                (num_reqs, self.engram_hash.lookback_depth), -1
            )
        image_mask = image_sentinel_mask(input_ids)
        engram_hashes = self.engram_hash(
            input_ids,
            positions,
            query_start_loc,
            image_mask,
            lookback_token_ids,
            image_sentinel_mask(lookback_token_ids),
            None,
            None,
        )
        gathered_hashes = gather_engram_hashes(
            engram_hashes, dp_shared_memory=self.engram_dp_shared_memory
        )
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            engram = getattr(layer, "engram", None)
            if engram is not None:
                engram.prepare_embeddings(gathered_hashes[:, engram.layer_hash_index])
        self._moet_dbo_engram = (engram_hashes, ~image_mask)

    def moet_finish_ubatched_forward(self) -> None:
        \"\"\"[vllm-moet] prefill DBO: after the microbatch threads joined.\"\"\"
        self._moet_dbo_engram = None
        if self._moet_row_buffers is not None:
            self._moet_row_buffers.reset()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
""",
    ),
    # The forward's Engram block: a microbatch takes its slice of the whole-batch preparation.
    (
        """        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if (
            self.engram_hash is not None
            and input_ids is not None
            and is_forward_context_available()
        ):
            attn_metadata = get_forward_context().attn_metadata
""",
        """        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if (
            self.engram_hash is not None
            and input_ids is not None
            and _moet_dbo.enabled()
            and self._moet_dbo_engram is not None
        ):
            # [vllm-moet] prefill DBO: hashed and fetched for the whole batch before the threads started
            engram_hashes = _moet_dbo.slice_rows(self._moet_dbo_engram[0], input_ids.shape[0])
            engram_mask = _moet_dbo.slice_rows(self._moet_dbo_engram[1], input_ids.shape[0])
        elif (
            self.engram_hash is not None
            and input_ids is not None
            and is_forward_context_available()
        ):
            attn_metadata = get_forward_context().attn_metadata
""",
    ),
    # The final all-reduce.
    (
        """        if self.fuse_mhc_all_reduce:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        if self.end_layer in self.aux_hidden_state_layers:
""",
        """        if self.fuse_mhc_all_reduce:
            hidden_states = _moet_dbo.all_reduce(hidden_states)  # [vllm-moet] prefill DBO: the comm region
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        if self.end_layer in self.aux_hidden_state_layers:
""",
    ),
    # The decoder SWA bounded replay stays on under microbatching (model_state prepares it per microbatch).
    (
        """        elif (
            self.use_sequence_parallel
            or parallel_config.prefill_context_parallel_size > 1
            or parallel_config.use_ubatching
        ):
            reason = (
                "the replay-layer batch shrinks per rank, which sequence and "
                "prefill-context parallelism and microbatching cannot follow"
            )
""",
        """        elif (
            self.use_sequence_parallel
            or parallel_config.prefill_context_parallel_size > 1
            # [vllm-moet] prefill DBO: the model state prepares a replay batch per microbatch
        ):
            reason = (
                "the replay-layer batch shrinks per rank, which sequence and "
                "prefill-context parallelism cannot follow"
            )
""",
    ),
]

# ----------------------------------------------------------------------------------------------------------------
# vllm/models/deepseek_v41/decoder_replay_layers.py
# ----------------------------------------------------------------------------------------------------------------
REPLAY_LAYERS = [
    (
        """from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput
""",
        """from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput
from vllm.v1.worker import moet_prefill_dbo as _moet_dbo  # [vllm-moet] prefill DBO
""",
    ),
    (
        """        # The replay batch, set by the model state every step: its rows of the
        # batch and its forward context. None runs the layers on the batch.
        self.rows: torch.Tensor | None = None
        self.forward_context: ForwardContext | None = None

    def __call__(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        rows = self.rows
        if rows is None:
            return self.run_layers(hidden_states, *states)
        # A trimming step holds a prefill longer than the window, more tokens
        # than any step whose MoE leaves its finalize to the next layer.
        assert isinstance(hidden_states, torch.Tensor)
        num_rows = rows.shape[0]
        for buf in self.row_buffers:
            buf[:num_rows].copy_(buf.index_select(0, rows))
        with override_forward_context(self.forward_context):
            row_outputs = self.run_layers(
                hidden_states.index_select(0, rows),
                *(None if t is None else t.index_select(0, rows) for t in states),
            )
""",
        """        # The replay batch, set by the model state every step: its rows of the
        # batch and its forward context, per microbatch ([vllm-moet] prefill DBO:
        # keyed by the microbatch index, 0 for a step that is one batch). A
        # microbatch without an entry runs the layers on its batch.
        self.rows: dict[int, torch.Tensor] = {}
        self.forward_context: dict[int, ForwardContext] = {}

    def __call__(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        ubatch_idx = _moet_dbo.ubatch_id()
        rows = self.rows.get(ubatch_idx)
        if rows is None:
            return self.run_layers(hidden_states, *states)
        # A trimming step holds a prefill longer than the window, more tokens
        # than any step whose MoE leaves its finalize to the next layer.
        assert isinstance(hidden_states, torch.Tensor)
        num_rows = rows.shape[0]
        for buf in self.row_buffers:
            buf[:num_rows].copy_(buf.index_select(0, rows))
        replay_context = self.forward_context[ubatch_idx]
        # [vllm-moet] prefill DBO: a microbatch thread restores its context's forward context at every
        # resume, so the replay context is installed there for the duration of the replay layers.
        ubatch_context = _moet_dbo.context()
        saved_context = None
        if ubatch_context is not None:
            saved_context = ubatch_context.forward_context
            ubatch_context.forward_context = replay_context
        try:
            with override_forward_context(replay_context):
                row_outputs = self.run_layers(
                    hidden_states.index_select(0, rows),
                    *(None if t is None else t.index_select(0, rows) for t in states),
                )
        finally:
            if ubatch_context is not None:
                ubatch_context.forward_context = saved_context
""",
    ),
]

# ----------------------------------------------------------------------------------------------------------------
# vllm/models/deepseek_v41/nvidia/model_state.py
# ----------------------------------------------------------------------------------------------------------------
MODEL_STATE = [
    (
        """        # Per request state index; batches gather from it (see prepare_attn).
        self._replay_start_np = np.zeros(self.max_num_reqs, dtype=np.int32)
        self._replay_start = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self._replay_start_staging = UvaBufferPool(self.max_num_reqs, torch.int32)
        # The replay window and the indices of the prefix-cacheable groups, from
        # the KV cache config on first use; None until then.
        self._replay: tuple[int, torch.Tensor] | None = None

        self.decoder_replay_layers: DecoderReplayLayers | None = getattr(
            model, "decoder_replay_layers", None
        )
        if self.decoder_replay_layers is not None:
            # Requests whose every prompt row is read (prompt logprobs) never trim.
            self._keeps_rows_np = np.zeros(self.max_num_reqs, dtype=np.bool_)
            self._dropped_before = UvaBufferPool(self.max_num_reqs + 1, torch.int32)
            # The replay batch's inputs: its rows of the batch, query offsets,
            # positions, slot mappings (sized with the KV cache groups on first
            # use) and the position from which each request holds replay-layer
            # window KV.
            self._kept_rows = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device
            )
            self._kept_query_start_loc = torch.zeros(
                self.max_num_reqs + 1, dtype=torch.int32, device=device
            )
            self._kept_positions = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device
            )
            self._kept_slot_mappings: torch.Tensor | None = None
            self._kept_kv_start = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=device
            )
            self._replay_attn_groups: list[list[AttentionGroup]] | None = None
""",
        """        # Per request state index; batches gather from it (see prepare_attn).
        # [vllm-moet] prefill DBO: the per-batch buffers, staging and builders below exist once per
        # microbatch (index 0 for a step that is one batch): prepare_attn runs once per microbatch and the
        # threads consume them at the same time.
        parallel_config = vllm_config.parallel_config
        num_ubatches = parallel_config.num_ubatches if parallel_config.use_ubatching else 1
        self._moet_num_ubatches = max(1, num_ubatches)
        self._replay_start_np = np.zeros(self.max_num_reqs, dtype=np.int32)
        self._replay_start = [
            torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
            for _ in range(self._moet_num_ubatches)
        ]
        self._replay_start_staging = [
            UvaBufferPool(self.max_num_reqs, torch.int32)
            for _ in range(self._moet_num_ubatches)
        ]
        # The replay window and the indices of the prefix-cacheable groups, from
        # the KV cache config on first use; None until then.
        self._replay: tuple[int, torch.Tensor] | None = None

        self.decoder_replay_layers: DecoderReplayLayers | None = getattr(
            model, "decoder_replay_layers", None
        )
        if self.decoder_replay_layers is not None:
            # Requests whose every prompt row is read (prompt logprobs) never trim.
            self._keeps_rows_np = np.zeros(self.max_num_reqs, dtype=np.bool_)
            self._dropped_before = [
                UvaBufferPool(self.max_num_reqs + 1, torch.int32)
                for _ in range(self._moet_num_ubatches)
            ]
            # The replay batch's inputs: its rows of the batch, query offsets,
            # positions, slot mappings (sized with the KV cache groups on first
            # use) and the position from which each request holds replay-layer
            # window KV.
            self._kept_rows = [
                torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
                for _ in range(self._moet_num_ubatches)
            ]
            self._kept_query_start_loc = [
                torch.zeros(self.max_num_reqs + 1, dtype=torch.int32, device=device)
                for _ in range(self._moet_num_ubatches)
            ]
            self._kept_positions = [
                torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
                for _ in range(self._moet_num_ubatches)
            ]
            self._kept_slot_mappings: list[torch.Tensor] | None = None
            self._kept_kv_start = [
                torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
                for _ in range(self._moet_num_ubatches)
            ]
            self._replay_attn_groups: list[list[list[AttentionGroup]] | None] = [
                None
            ] * self._moet_num_ubatches
""",
    ),
    (
        """            if self.decoder_replay_layers is not None:
                self._kept_slot_mappings = torch.zeros(
                    len(specs),
                    self.max_num_tokens,
                    dtype=torch.int64,
                    device=self.device,
                )
            self._warm_up_replay_kernels(input_batch, slot_mappings)
        window, cacheable_groups = self._replay
        replay_start: torch.Tensor | None = None
        if window:
            num_reqs = input_batch.num_reqs
            # Decode rows sit above the hit, so only prefills carry a replay
            # start; dummy batches (captures, profiling) carry none.
            replay_start_np = np.where(
                input_batch.is_prefilling_np[:num_reqs],
                self._replay_start_np[input_batch.idx_mapping_np[:num_reqs]],
                0,
            ).astype(np.int32)
            replay_start = self._replay_start_staging.copy_to_gpu(
                replay_start_np, out=self._replay_start[:num_reqs]
            )
""",
        """            if self.decoder_replay_layers is not None:
                self._kept_slot_mappings = [
                    torch.zeros(
                        len(specs),
                        self.max_num_tokens,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    for _ in range(self._moet_num_ubatches)
                ]
            self._warm_up_replay_kernels(input_batch, slot_mappings)
        window, cacheable_groups = self._replay
        replay_start: torch.Tensor | None = None
        if window:
            num_reqs = input_batch.num_reqs
            # Decode rows sit above the hit, so only prefills carry a replay
            # start; dummy batches (captures, profiling) carry none.
            replay_start_np = np.where(
                input_batch.is_prefilling_np[:num_reqs],
                self._replay_start_np[input_batch.idx_mapping_np[:num_reqs]],
                0,
            ).astype(np.int32)
            replay_start = self._replay_start_staging[ubatch_idx].copy_to_gpu(
                replay_start_np, out=self._replay_start[ubatch_idx][:num_reqs]
            )
""",
    ),
    (
        """        if self.decoder_replay_layers is not None:
            assert replay_start is not None
            self._prepare_replay_batch(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                replay_start,
            )
        return attn_metadata
""",
        """        if self.decoder_replay_layers is not None:
            assert replay_start is not None
            self._prepare_replay_batch(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                replay_start,
                ubatch_idx,
            )
        return attn_metadata
""",
    ),
    (
        """        _pad_replayed_slots_kernel.warmup(
            slot_mappings,
            slot_mappings.stride(0),
            cacheable_groups,
            input_batch.query_start_loc,
            input_batch.positions,
            self._replay_start,
            window,
            PAD_SLOT_ID,
            NUM_GROUPS=cacheable_groups.numel(),
            BLOCK=1024,
            grid=(1,),
        )
        if self.decoder_replay_layers is None:
            return
        assert self._kept_slot_mappings is not None
        _gather_replay_batch_kernel.warmup(
            input_batch.query_start_loc,
            self._dropped_before.copy_to_uva(np.zeros(1, dtype=np.int32)),
            input_batch.seq_lens,
            self._replay_start,
            input_batch.positions,
            slot_mappings,
            slot_mappings.stride(0),
            self._kept_rows,
            self._kept_query_start_loc,
            self._kept_positions,
            self._kept_slot_mappings,
            self._kept_slot_mappings.stride(0),
            self._kept_kv_start,
""",
        """        _pad_replayed_slots_kernel.warmup(
            slot_mappings,
            slot_mappings.stride(0),
            cacheable_groups,
            input_batch.query_start_loc,
            input_batch.positions,
            self._replay_start[0],
            window,
            PAD_SLOT_ID,
            NUM_GROUPS=cacheable_groups.numel(),
            BLOCK=1024,
            grid=(1,),
        )
        if self.decoder_replay_layers is None:
            return
        assert self._kept_slot_mappings is not None
        _gather_replay_batch_kernel.warmup(
            input_batch.query_start_loc,
            self._dropped_before[0].copy_to_uva(np.zeros(1, dtype=np.int32)),
            input_batch.seq_lens,
            self._replay_start[0],
            input_batch.positions,
            slot_mappings,
            slot_mappings.stride(0),
            self._kept_rows[0],
            self._kept_query_start_loc[0],
            self._kept_positions[0],
            self._kept_slot_mappings[0],
            self._kept_slot_mappings[0].stride(0),
            self._kept_kv_start[0],
""",
    ),
    (
        """        kv_cache_config: KVCacheConfig,
        replay_start: torch.Tensor,
    ) -> None:
        \"\"\"Set the replay layers' batch for this forward, or none when nothing
        trims. Only eager steps trim: a CUDA graph keeps the layers on the whole
        batch, and ranks share the graph mode. Under data parallelism every rank
        replays if any does.\"\"\"
        layers = self.decoder_replay_layers
        assert layers is not None
        layers.rows = layers.forward_context = None
        if cudagraph_mode != CUDAGraphMode.NONE:
            return
""",
        """        kv_cache_config: KVCacheConfig,
        replay_start: torch.Tensor,
        ubatch_idx: int = 0,
    ) -> None:
        \"\"\"Set the replay layers' batch for this forward, or none when nothing
        trims. Only eager steps trim: a CUDA graph keeps the layers on the whole
        batch, and ranks share the graph mode. Under data parallelism every rank
        replays if any does.\"\"\"
        layers = self.decoder_replay_layers
        assert layers is not None
        # [vllm-moet] prefill DBO: one entry per microbatch (0 for a single-batch step)
        layers.rows.pop(ubatch_idx, None)
        layers.forward_context.pop(ubatch_idx, None)
        if cudagraph_mode != CUDAGraphMode.NONE:
            return
""",
    ),
    (
        """        kept_batch, kept_slot_mappings = self._kept_input_batch(
            input_batch, slot_mappings, replay_start, kept_lens
        )
        attn_metadata = super().prepare_attn(
            kept_batch,
            CUDAGraphMode.NONE,
            block_tables,
            kept_slot_mappings,
            self._replay_groups(attn_groups),
            kv_cache_config,
            model_specific_attn_metadata=ReplayAttnMetadata(
                self._kept_kv_start[:num_reqs]
            ),
        )
        layers.rows = self._kept_rows[:num_tokens]
        layers.forward_context = create_forward_context(
""",
        """        kept_batch, kept_slot_mappings = self._kept_input_batch(
            input_batch, slot_mappings, replay_start, kept_lens, ubatch_idx
        )
        attn_metadata = super().prepare_attn(
            kept_batch,
            CUDAGraphMode.NONE,
            block_tables,
            kept_slot_mappings,
            self._replay_groups(attn_groups, ubatch_idx),
            kv_cache_config,
            model_specific_attn_metadata=ReplayAttnMetadata(
                self._kept_kv_start[ubatch_idx][:num_reqs]
            ),
        )
        layers.rows[ubatch_idx] = self._kept_rows[ubatch_idx][:num_tokens]
        layers.forward_context[ubatch_idx] = create_forward_context(
""",
    ),
    (
        """        replay_start: torch.Tensor,
        kept_lens: np.ndarray,
    ) -> tuple[InputBatch, torch.Tensor]:
""",
        """        replay_start: torch.Tensor,
        kept_lens: np.ndarray,
        ubatch_idx: int = 0,
    ) -> tuple[InputBatch, torch.Tensor]:
""",
    ),
    (
        """        dropped_before = self._dropped_before.copy_to_uva(
            input_batch.query_start_loc_np[: num_reqs + 1] - kept_query_start_loc_np
        )
        assert self._kept_slot_mappings is not None
        kept_slot_mappings = self._kept_slot_mappings[:, :num_tokens]
        _gather_replay_batch_kernel[(num_reqs,)](
            input_batch.query_start_loc,
            dropped_before,
            input_batch.seq_lens,
            replay_start,
            input_batch.positions,
            slot_mappings,
            slot_mappings.stride(0),
            self._kept_rows,
            self._kept_query_start_loc,
            self._kept_positions,
            kept_slot_mappings,
            kept_slot_mappings.stride(0),
            self._kept_kv_start,
""",
        """        dropped_before = self._dropped_before[ubatch_idx].copy_to_uva(
            input_batch.query_start_loc_np[: num_reqs + 1] - kept_query_start_loc_np
        )
        assert self._kept_slot_mappings is not None
        kept_slot_mappings = self._kept_slot_mappings[ubatch_idx][:, :num_tokens]
        _gather_replay_batch_kernel[(num_reqs,)](
            input_batch.query_start_loc,
            dropped_before,
            input_batch.seq_lens,
            replay_start,
            input_batch.positions,
            slot_mappings,
            slot_mappings.stride(0),
            self._kept_rows[ubatch_idx],
            self._kept_query_start_loc[ubatch_idx],
            self._kept_positions[ubatch_idx],
            kept_slot_mappings,
            kept_slot_mappings.stride(0),
            self._kept_kv_start[ubatch_idx],
""",
    ),
    (
        """        kept_batch = replace(
            input_batch,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            query_start_loc=self._kept_query_start_loc[: num_reqs + 1],
            query_start_loc_np=kept_query_start_loc_np,
            max_query_len=max_query_len,
            positions=self._kept_positions[:num_tokens],
            fast_prefill=None,
        )
        return kept_batch, kept_slot_mappings
""",
        """        kept_batch = replace(
            input_batch,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            query_start_loc=self._kept_query_start_loc[ubatch_idx][: num_reqs + 1],
            query_start_loc_np=kept_query_start_loc_np,
            max_query_len=max_query_len,
            positions=self._kept_positions[ubatch_idx][:num_tokens],
            fast_prefill=None,
        )
        return kept_batch, kept_slot_mappings
""",
    ),
    (
        """    def _replay_groups(
        self, attn_groups: list[list[AttentionGroup]]
    ) -> list[list[AttentionGroup]]:
        \"\"\"The attention groups with metadata builders of their own, like each
        microbatch's: a builder keeps the metadata it built, and the runner's
        hold the batch's.\"\"\"
        if self._replay_attn_groups is None:
            self._replay_attn_groups = []
            for groups in attn_groups:
                replay_groups = []
                for group in groups:
                    replay_group = replace(group, metadata_builders=[])
                    replay_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        group.metadata_builders[0].kernel_block_size,
                    )
                    replay_groups.append(replay_group)
                self._replay_attn_groups.append(replay_groups)
        return self._replay_attn_groups
""",
        """    def _replay_groups(
        self, attn_groups: list[list[AttentionGroup]], ubatch_idx: int = 0
    ) -> list[list[AttentionGroup]]:
        \"\"\"The attention groups with metadata builders of their own, like each
        microbatch's: a builder keeps the metadata it built, and the runner's
        hold the batch's. [vllm-moet] prefill DBO: one set per microbatch.\"\"\"
        if self._replay_attn_groups[ubatch_idx] is None:
            replay_attn_groups = []
            for groups in attn_groups:
                replay_groups = []
                for group in groups:
                    replay_group = replace(group, metadata_builders=[])
                    replay_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        group.metadata_builders[0].kernel_block_size,
                    )
                    replay_groups.append(replay_group)
                replay_attn_groups.append(replay_groups)
            self._replay_attn_groups[ubatch_idx] = replay_attn_groups
        return self._replay_attn_groups[ubatch_idx]
""",
    ),
]

# ----------------------------------------------------------------------------------------------------------------
# vllm/models/deepseek_v41/common/engram.py (after step 12)
# ----------------------------------------------------------------------------------------------------------------
ENGRAM = [
    (
        """from vllm.distributed.device_communicators.shm_broadcast import (
    SHM_PATH,
""",
        """from vllm.v1.worker import moet_prefill_dbo as _moet_dbo  # [vllm-moet] prefill DBO
from vllm.distributed.device_communicators.shm_broadcast import (
    SHM_PATH,
""",
    ),
    (
        """            staged = self.staged_rows[: slot * self.embed_tokens.dp_size]
            return _gather_engram_rows(staged, num_tokens)
        return self.staged_rows[:num_tokens]
""",
        """            staged = self.staged_rows[: slot * self.embed_tokens.dp_size]
            return _gather_engram_rows(staged, num_tokens)
        # [vllm-moet] prefill DBO: the rows were fetched for the whole batch; a microbatch reads its own
        return _moet_dbo.slice_rows(self.staged_rows, num_tokens)
""",
    ),
    (
        """        rows = tensor_model_parallel_all_gather(rows, dim=1)
        return rows[:, : self.embed_tokens.n_hash_cols]
""",
        """        rows = _moet_dbo.all_gather(rows, 1)  # [vllm-moet] prefill DBO: the comm region
        return rows[:, : self.embed_tokens.n_hash_cols]
""",
    ),
]

FILES: list[tuple[str, list[tuple[str, str]], list[tuple[str, str]]]] = [
    ("vllm/v1/worker/ubatching.py", UBATCHING, []),
    ("vllm/v1/worker/gpu/ubatch_utils.py", UBATCH_UTILS, []),
    ("vllm/v1/worker/gpu/dp_utils.py", DP_UTILS, []),
    ("vllm/config/vllm.py", CONFIG, []),
    ("vllm/model_executor/layers/fused_moe/modular_kernel.py", MODULAR_KERNEL, []),
    ("vllm/models/deepseek_v41/nvidia/ops/mhc.py", MHC, MHC_OPTIONAL),
    ("vllm/models/deepseek_v41/nvidia/model.py", MODEL, []),
    ("vllm/models/deepseek_v41/decoder_replay_layers.py", REPLAY_LAYERS, []),
    ("vllm/models/deepseek_v41/nvidia/model_state.py", MODEL_STATE, []),
    ("vllm/models/deepseek_v41/common/engram.py", ENGRAM, []),
]


def patch_text(rel: str, src: str, hunks, optional) -> str:
    if MARKER in src:
        return src
    for i, (anchor, _) in enumerate(hunks):
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(
                f"{rel}: anchor {i + 1}/{len(hunks)} found {n} times (expected 1); this patcher targets vLLM "
                "main 21d93d0d after Dockerfile steps 10 and 12"
            )
    for anchor, new in hunks:
        src = src.replace(anchor, new, 1)
    for anchor, new in optional:
        if src.count(anchor) == 1:
            src = src.replace(anchor, new, 1)
    if rel.endswith("ops/mhc.py"):
        # The import goes after the module's last top-level `from vllm...` import block start: the first
        # `from vllm.distributed` line is in every variant of the file.
        key = "from vllm.distributed"
        pos = src.find(key)
        if pos < 0:
            raise SystemExit(f"{rel}: no `from vllm.distributed` import to anchor the helper import on")
        src = src[:pos] + MHC_IMPORT + src[pos:]
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-packages", type=Path, default=DEFAULT_SITE)
    ap.add_argument("--helper", type=Path, default=DEFAULT_HELPER)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    site = args.site_packages
    done = already = 0
    outputs: list[tuple[Path, str]] = []
    for rel, hunks, optional in FILES:
        path = site / rel
        src = path.read_text()
        if MARKER in src:
            already += 1
            continue
        patched = patch_text(rel, src, hunks, optional)
        compile(patched, str(path), "exec")
        outputs.append((path, patched))
        done += 1
    if not args.helper.is_file():
        raise SystemExit(f"helper module missing: {args.helper}")
    compile(args.helper.read_text(), str(args.helper), "exec")
    if args.check:
        print(
            f"{site}: prefill DBO patch applies cleanly to {done} files ({already} already patched; "
            "not written)"
        )
        return 0
    for path, patched in outputs:
        path.write_text(patched)
    helper_dst = site / HELPER_DST
    helper_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.helper, helper_dst)
    print(
        f"{site}: prefill DBO patched {done} files ({already} already patched), helper at {helper_dst}; "
        "serve with --enable-dbo --dbo-decode-token-threshold 4096 --dbo-prefill-token-threshold 1024"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
