#!/usr/bin/env python3
"""The pipelined sequence-parallel prefill of DeepSeek-V4.1 at TP4 on sm_120: two 4 096-row groups of one engine step
one layer apart, the SP collectives of one under the compute of the other (step 47 candidate).

Switch: VLLM_MOET_PREFILL_PIPE=2 (default 0 = today's path, every patched line is a no-op), together with
--max-num-batched-tokens 8192 (two groups per step; a 4 096 step has one group and takes today's path). Runtime module
tools/dsv41_sm120/sp_pipeline/moet_prefill_pipe.py (installed as vllm/v1/worker/moet_prefill_pipe.py) - its docstring has
the design, the dependency graph and the hazards.

Why. On image r25 a 4 096-token SP layer-chunk is 2 reduce-scatters + 2 all-gathers ~ 2.5 ms of NCCL next to 3.3 ms of
compute on one stream with zero overlap: NCCL = 37.9 % of the 19K prefill's kernel time, bandwidth-bound (36 GB/s per
rank over PCIe Gen5 through the root complex), the knobs give nothing. Two whole chunks of the same prefill in flight one
layer apart hide 84 % of it in the microbenchmark (-30.8 % per layer-chunk with the real MoE kernels); slicing a chunk's
rows loses (the MoE's fixed per-call cost). The pipeline keeps every kernel at today's launch shape with today's rows,
so the bits are today's by construction: group B's layer L runs after group A's layer L wrote its KV (stream order on the
one compute stream, asserted on the CPU), the RS/AG per group are the same collectives on the same rows.

Patched files (anchor-based, idempotent, after steps 10/39/41/45; the installed text is checked, nothing else accepted):
  vllm/v1/worker/gpu/model_runner.py                the step's row groups after the whole-step metadata, the model call
                                                    inside the pipeline's step context (eager steps only)
  vllm/models/deepseek_v41/nvidia/model.py          the forward split into head (once) and tail (once per group); the
                                                    layer's two boundary phases; the dump hook of the bits harness
  vllm/models/deepseek_v41/attention.py             the end of phase 1 after the projections' all-gather
  vllm/models/deepseek_v41/nvidia/model_state.py    the replay-batch buffers / staging / builders per slot
  vllm/models/deepseek_v41/decoder_replay_layers.py the replay batch per slot (the thread's group)
  vllm/models/deepseek_v41/common/engram.py         the Engram rows at the group's row offset
  (and tools/dsv41_sm120/sp_moe_gather/sp_moe_gather_sm120.py: one MoE gather in flight per stand-in, not per process)

Usage:
    python3 patch_vllm_prefill_pipe_sm120.py [--dist DIR] [--helper FILE] [--check]
  --dist    the vllm package directory (default /usr/local/lib/python3.12/dist-packages/vllm)
  --helper  the runtime module to install (default sp_pipeline/moet_prefill_pipe.py next to this file)
  --check   dry run: every anchor present (or already patched), nothing written
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm")
DEFAULT_HELPER = Path(__file__).resolve().parent / "sp_pipeline" / "moet_prefill_pipe.py"
HELPER_DST = "v1/worker/moet_prefill_pipe.py"
MARKER = "[vllm-moet] sm_120 prefill pipeline"
IMPORT = f"from vllm.v1.worker import moet_prefill_pipe as _moet_pipe  # {MARKER}\n"

# ---------------------------------------------------------------------------------------------------------------
# vllm/v1/worker/gpu/model_runner.py
# ---------------------------------------------------------------------------------------------------------------
RUNNER_EDITS = [
    (
        "import",
        "from vllm.v1.worker.gpu.ubatch_utils import (\n    UBatchRunner,\n    UBatchState,\n    maybe_build_ubatch_runner,\n)\n",
        "from vllm.v1.worker.gpu.ubatch_utils import (\n    UBatchRunner,\n    UBatchState,\n    maybe_build_ubatch_runner,\n)\n"
        + IMPORT,
    ),
    (
        "plan",
        "        input_ids = input_batch.input_ids\n        inputs_embeds = None\n        ec_connector_output = None\n",
        f"        # {MARKER} (VLLM_MOET_PREFILL_PIPE=2): the step's row groups after the whole step's metadata (None = today's path)\n"
        "        moet_pipe_groups = None\n"
        "        if attn_metadata is not None and not dummy_run and ubatch_state is None:\n"
        "            assert block_tables is not None and slot_mappings is not None\n"
        "            moet_pipe_groups = _moet_pipe.plan_groups(self, input_batch, batch_desc, block_tables, slot_mappings)\n"
        "        input_ids = input_batch.input_ids\n        inputs_embeds = None\n        ec_connector_output = None\n",
    ),
    (
        "eager call",
        "                else:\n                    # Eager (NONE): call the raw model directly.\n                    model_output = self.model(**model_inputs)\n",
        "                else:\n                    # Eager (NONE): call the raw model directly.\n"
        f"                    with _moet_pipe.step_context(moet_pipe_groups):  # {MARKER}\n"
        "                        model_output = self.model(**model_inputs)\n",
    ),
]

# ---------------------------------------------------------------------------------------------------------------
# vllm/models/deepseek_v41/nvidia/model.py
# ---------------------------------------------------------------------------------------------------------------
MODEL_EDITS = [
    (
        "import",
        "from vllm.v1.worker.ubatching import dbo_current_ubatch_id\n",
        "from vllm.v1.worker.ubatching import dbo_current_ubatch_id\n" + IMPORT,
    ),
    (
        "reduce-scatter input",
        "    _moet_p2p_rs_resolve()\n    if _MOET_P2P_RS:\n        out = _MOET_P2P_RS.reduce_scatter(x)\n",
        "    _moet_p2p_rs_resolve()\n"
        f"    _moet_pipe.record_for_comm(x)  # {MARKER}: born on the compute stream, read by the comm stream\n"
        "    if _MOET_P2P_RS:\n        out = _MOET_P2P_RS.reduce_scatter(x)\n",
    ),
    (
        "replay section",
        "    replay.write_batch_kv(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n"
        "    rows, num_tokens = replay_batch.rows, positions.shape[0]\n"
        "    pre_mix, post_mix, res_mix, residual = (\n"
        "        _moet_sp_rows_to_full(state, rows, num_tokens) for state in (pre_mix, post_mix, res_mix, residual)\n"
        "    )\n"
        "    return replay._run(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n",
        "    replay.write_batch_kv(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n"
        f"    # {MARKER}: the replay section (rows gather, layers 20-39 on 128 rows, collapse) on its own stream, handing the\n"
        "    # CPU to the other group once per layer (yield_point in the layer forward)\n"
        "    _moet_pipe.replay_begin(hidden_states, pre_mix, post_mix, res_mix, residual)\n"
        "    rows, num_tokens = replay_batch.rows, positions.shape[0]\n"
        "    pre_mix, post_mix, res_mix, residual = (\n"
        "        _moet_sp_rows_to_full(state, rows, num_tokens) for state in (pre_mix, post_mix, res_mix, residual)\n"
        "    )\n"
        "    out = replay._run(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n"
        "    _moet_pipe.replay_end(out)\n"
        "    return out\n",
    ),
    (
        "layer phase 1",
        "        moet_sp = _moet_sp_sharded(residual if residual is not None else x, positions)\n"
        "        _moet_sp_pin(positions.shape[0] if moet_sp else None)\n"
        "        mhc_stream = self.mhc_stream\n",
        "        moet_sp = _moet_sp_sharded(residual if residual is not None else x, positions)\n"
        "        _moet_sp_pin(positions.shape[0] if moet_sp else None)\n"
        f"        if moet_sp:  # {MARKER}: phase 1 = RS1 -> mHC -> shard projections -> AG1 on the comm stream (ends in attention.py)\n"
        "            _moet_pipe.phase_begin()\n"
        "        else:  # the replay layers (128 rows): hand the CPU to the other group once per layer\n"
        "            _moet_pipe.yield_point()\n"
        "        mhc_stream = self.mhc_stream\n",
    ),
    (
        "layer phase 2",
        "        if (self.use_sequence_parallel and self.attn.gemm_rs is None) or moet_sp:  # [vllm-moet] sm_120 prefill SP\n"
        "            x = _moet_sp_reduce_scatter(x) if moet_sp else sp_reduce_scatter(x)\n",
        f"        if moet_sp:  # {MARKER}: phase 2 = RS2 -> mHC -> quant / route -> AG2 on the comm stream\n"
        "            _moet_pipe.phase_begin()\n"
        "        if (self.use_sequence_parallel and self.attn.gemm_rs is None) or moet_sp:  # [vllm-moet] sm_120 prefill SP\n"
        "            x = _moet_sp_reduce_scatter(x) if moet_sp else sp_reduce_scatter(x)\n",
    ),
    (
        "layer phase 2 end",
        "            x = _moet_sp_moe_gather(self, x, positions, input_ids)  # [vllm-moet] sm_120 prefill SP gather: bf16, or MXFP8 + the shard's top-k\n"
        "            moet_sp_standin = x\n",
        "            x = _moet_sp_moe_gather(self, x, positions, input_ids)  # [vllm-moet] sm_120 prefill SP gather: bf16, or MXFP8 + the shard's top-k\n"
        "            moet_sp_standin = x\n"
        f"            _moet_pipe.phase_end()  # {MARKER}: the MoE on the compute stream, the other group's turn\n",
    ),
    (
        "write_kv phase",
        "        moet_sp = _moet_sp_sharded(residual, positions)  # [vllm-moet] sm_120 prefill SP: the boundary on this rank's rows\n"
        "        if moet_sp:\n            x = _moet_sp_reduce_scatter(x)\n",
        "        moet_sp = _moet_sp_sharded(residual, positions)  # [vllm-moet] sm_120 prefill SP: the boundary on this rank's rows\n"
        "        if moet_sp:\n"
        f"            _moet_pipe.phase_begin()  # {MARKER}: RS -> mHC -> projections -> AG on the comm stream (ends in attention.py)\n"
        "            x = _moet_sp_reduce_scatter(x)\n",
    ),
    (
        "forward head",
        "        lookback_token_ids: torch.Tensor | None = None,\n"
        "    ) -> torch.Tensor | IntermediateTensors:\n"
        "        if get_pp_group().is_first_rank:\n",
        "        lookback_token_ids: torch.Tensor | None = None,\n"
        "    ) -> torch.Tensor | IntermediateTensors:\n"
        f"        _moet_pipe.dump_forward_begin()  # {MARKER}: the bits harness numbers the forwards\n"
        "        if get_pp_group().is_first_rank:\n",
    ),
    (
        "forward split",
        "        full_num_tokens = positions.shape[0]\n"
        "        if self._moet_prefill_sp_step(hidden_states, positions):  # [vllm-moet] sm_120 prefill SP: the hc state is sharded by rows\n"
        "            hidden_states = sp_shard(hidden_states)\n",
        f"        # {MARKER}: the rest of the forward once per row group (two groups one layer apart), or once as today\n"
        "        return self._moet_pipe_dispatch(\n"
        "            hidden_states, positions, input_ids, intermediate_tensors, engram_hashes, engram_mask\n"
        "        )\n"
        "\n"
        "    def _moet_pipe_dispatch(\n"
        "        self,\n"
        "        hidden_states: torch.Tensor,\n"
        "        positions: torch.Tensor,\n"
        "        input_ids: torch.Tensor | None,\n"
        "        intermediate_tensors: IntermediateTensors | None,\n"
        "        engram_hashes: torch.Tensor | None,\n"
        "        engram_mask: torch.Tensor | None,\n"
        "    ) -> torch.Tensor | IntermediateTensors:\n"
        f'        """{MARKER}: the step\'s row groups (set by the runner) run the forward\'s tail each, on two threads\n'
        "        that alternate at the sequence-parallel boundaries when both are sequence parallel (the earlier rows first:\n"
        "        their KV is written before the later rows read it), one after the other otherwise; the outputs are\n"
        '        reassembled in row order. Without groups the tail runs once on the whole step, as today."""\n'
        "        groups = _moet_pipe.pending_groups()\n"
        "        if groups is None:\n"
        "            return self._moet_forward_tail(\n"
        "                hidden_states, positions, input_ids, intermediate_tensors, engram_hashes, engram_mask\n"
        "            )\n"
        "        if (\n"
        "            intermediate_tensors is not None\n"
        "            or self.use_sequence_parallel\n"
        "            or hidden_states.dim() != 2\n"
        "            or hidden_states.shape[0] != positions.shape[0]\n"
        "            or sum(g.num_tokens for g in groups) != positions.shape[0]\n"
        "            or any(g.token_offset != sum(h.num_tokens for h in groups[: g.idx]) for g in groups)\n"
        "        ):\n"
        "            raise RuntimeError(\n"
        "                \"prefill pipeline: the step's groups do not match the forward's inputs \"\n"
        "                f\"(groups {[(g.token_offset, g.num_tokens) for g in groups]}, tokens {positions.shape[0]})\"\n"
        "            )\n"
        "\n"
        "        def rows(t: torch.Tensor | None, g) -> torch.Tensor | None:\n"
        "            return None if t is None else t[g.token_offset : g.token_offset + g.num_tokens]\n"
        "\n"
        "        pipelined = all(self._moet_prefill_sp_step(rows(hidden_states, g), rows(positions, g)) for g in groups)\n"
        "\n"
        "        def run_group(g):\n"
        "            return self._moet_forward_tail(\n"
        "                rows(hidden_states, g),\n"
        "                rows(positions, g),\n"
        "                rows(input_ids, g),\n"
        "                None,\n"
        "                rows(engram_hashes, g),\n"
        "                rows(engram_mask, g),\n"
        "            )\n"
        "\n"
        "        return _moet_pipe.merge_outputs(_moet_pipe.run_groups(groups, run_group, pipelined))\n"
        "\n"
        "    def _moet_forward_tail(\n"
        "        self,\n"
        "        hidden_states: torch.Tensor,\n"
        "        positions: torch.Tensor,\n"
        "        input_ids: torch.Tensor | None,\n"
        "        intermediate_tensors: IntermediateTensors | None,\n"
        "        engram_hashes: torch.Tensor | None,\n"
        "        engram_mask: torch.Tensor | None,\n"
        "    ) -> torch.Tensor | IntermediateTensors:\n"
        f'        """{MARKER}: the forward from the sequence-parallel shard on (the body of forward as before)."""\n'
        "        full_num_tokens = positions.shape[0]\n"
        "        if self._moet_prefill_sp_step(hidden_states, positions):  # [vllm-moet] sm_120 prefill SP: the hc state is sharded by rows\n"
        "            hidden_states = sp_shard(hidden_states)\n",
    ),
    (
        "mtp hidden buffer",
        "            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))\n",
        f"            _moet_off = _moet_pipe.token_offset()  # {MARKER}: the group's rows of the step's buffer\n"
        "            self._mtp_hidden_buffer[_moet_off : _moet_off + num_tokens].copy_(hidden_states.flatten(1))\n",
    ),
    (
        "dump hook",
        "            if previous_aux is not None:\n"
        "                # idx is the one-based id of the layer whose post this is.\n",
        f"            _moet_pipe.dump_layer(  # {MARKER}: the bits harness (VLLM_MOET_PREFILL_PIPE_DUMP), a no-op otherwise\n"
        "                idx, positions, x=hidden_states, residual=residual, post_mix=post_mix, res_mix=res_mix, pre_mix=pre_mix\n"
        "            )\n"
        "            if previous_aux is not None:\n"
        "                # idx is the one-based id of the layer whose post this is.\n",
    ),
]

# ---------------------------------------------------------------------------------------------------------------
# vllm/models/deepseek_v41/attention.py
# ---------------------------------------------------------------------------------------------------------------
ATTN_EDITS = [
    (
        "import",
        "from vllm.utils.multi_stream_utils import (\n    execute_in_parallel,\n    maybe_execute_in_parallel,\n)\n",
        "from vllm.utils.multi_stream_utils import (\n    execute_in_parallel,\n    maybe_execute_in_parallel,\n)\n" + IMPORT,
    ),
    (
        "phase 1 end (full input)",
        "        full = sp_all_gather(hidden_states)[:num_tokens]\n"
        "        return full, attn._run_parallel_input_projections(full)\n",
        "        full = sp_all_gather(hidden_states)[:num_tokens]\n"
        f"        _moet_pipe.phase_end()  # {MARKER}: the full-row projections and the attention on the compute stream\n"
        "        return full, attn._run_parallel_input_projections(full)\n",
    ),
    (
        "phase 1 end (shard projections)",
        "    return hidden_states.new_empty((num_tokens, 0)), gathered\n",
        f"    _moet_pipe.phase_end()  # {MARKER}: the attention on the compute stream\n"
        "    return hidden_states.new_empty((num_tokens, 0)), gathered\n",
    ),
]

# ---------------------------------------------------------------------------------------------------------------
# vllm/models/deepseek_v41/nvidia/model_state.py
# ---------------------------------------------------------------------------------------------------------------
MODEL_STATE_EDITS = [
    (
        "import",
        "from vllm.v1.worker.utils import AttentionGroup\n",
        "from vllm.v1.worker.utils import AttentionGroup\n" + IMPORT,
    ),
    (
        "slots",
        "        layers = self.decoder_replay_layers\n        compilation_config = vllm_config.compilation_config\n",
        f"        # {MARKER}: the replay-batch buffers, staging and builders above exist once per slot (0 = the whole\n"
        "        # step, 1.. = the step's row groups); prepare_attn swaps a slot's set in (_moet_pipe_select_slot).\n"
        "        self._moet_pipe_slots: dict[int, dict] = {}\n"
        "        self._moet_pipe_cur = 0\n"
        "        self._moet_pipe_slot_cudagraphs: dict[int, Any] = {}  # slot -> DecoderReplayCudaGraphManager (capture_inner_cudagraphs)\n"
        "        layers = self.decoder_replay_layers\n        compilation_config = vllm_config.compilation_config\n",
    ),
    (
        "prepare_attn slot",
        "        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,\n"
        "    ) -> dict[str, Any]:\n"
        "        if self._replay is None:\n",
        "        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,\n"
        "    ) -> dict[str, Any]:\n"
        f"        self._moet_pipe_select_slot(_moet_pipe.prepare_slot())  # {MARKER}\n"
        "        if self._replay is None:\n",
    ),
    (
        "replay graphs per slot (dispatch)",
        "        cudagraphs, run_graph = self.replay_cudagraphs, None\n",
        f"        cudagraphs, run_graph = self._moet_pipe_replay_cudagraphs(), None  # {MARKER}: the slot's own graphs\n",
    ),
    (
        "replay graphs per slot (capture)",
        "            self._replay_attn_groups = None\n"
        "            self.replay_cudagraphs.capture_replay_graphs(prepare)\n",
        "            self._replay_attn_groups = None\n"
        "            self.replay_cudagraphs.capture_replay_graphs(prepare)\n"
        f"            # {MARKER}: a row group's replay batch lives in its slot's buffers / builders, so its graphs are captured per slot\n"
        "            # (VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS=1); without them the groups' replay layers run eagerly, never slot 0's graphs.\n"
        "            # A manager is built anew at every capture (K-7): the memory-profiling pass captures into a throwaway graph pool\n"
        "            # that objects created during the pass bind for good (cudagraph_utils: pools are restored only on the wrappers\n"
        "            # that existed before it), and the pass clears every graph - a manager kept from it would capture into a pool\n"
        "            # whose use_count is 0 (c10 'use_count > 0 INTERNAL ASSERT FAILED'), as the speculator's managers would.\n"
        "            self._moet_pipe_slot_cudagraphs.clear()\n"
        "            for _slot in _moet_pipe.graph_slots():\n"
        "                self._moet_pipe_select_slot(_slot)\n"
        "                self._replay_attn_groups = None\n"
        "                self._moet_pipe_slot_cudagraphs[_slot] = DecoderReplayCudaGraphManager(\n"
        "                    self.vllm_config, self.device, self.decoder_replay_layers\n"
        "                )\n"
        "                self._moet_pipe_slot_cudagraphs[_slot].capture_replay_graphs(prepare)\n"
        "            self._moet_pipe_select_slot(0)\n",
    ),
    (
        "slot method",
        "    def _warm_up_replay_kernels(\n",
        "    def _moet_pipe_replay_cudagraphs(self):\n"
        f'        """{MARKER}: the replay CUDA graphs of the slot being prepared: slot 0 (the whole step) = the model state\'s own\n'
        "        manager; a row group's slot = the manager captured for it in capture_inner_cudagraphs, or None (eager) - a group\n"
        '        must never take slot 0\'s graphs, which read slot 0\'s metadata buffers."""\n'
        "        if self._moet_pipe_cur == 0:\n"
        "            return self.replay_cudagraphs\n"
        "        return self._moet_pipe_slot_cudagraphs.get(self._moet_pipe_cur)\n"
        "\n"
        "    _MOET_PIPE_ATTRS = (\n"
        '        "_replay_start",\n'
        '        "_replay_start_staging",\n'
        '        "_dropped_before",\n'
        '        "_kept_rows",\n'
        '        "_kept_query_start_loc",\n'
        '        "_kept_positions",\n'
        '        "_kept_slot_mappings",\n'
        '        "_kept_kv_start",\n'
        '        "_replay_attn_groups",\n'
        "    )\n"
        "\n"
        "    def _moet_pipe_select_slot(self, slot: int) -> None:\n"
        f'        """{MARKER}: swap in the replay-batch buffers / staging / builders of `slot` (0 = the whole step,\n'
        "        1.. = a row group of the step): prepare_attn runs once per slot and the groups consume their batches at the\n"
        '        same time. A new slot starts from zeroed copies of slot 0\'s buffers and builds its own replay groups."""\n'
        "        if slot == self._moet_pipe_cur:\n"
        "            return\n"
        "        attrs = [a for a in self._MOET_PIPE_ATTRS if hasattr(self, a)]\n"
        "        self._moet_pipe_slots[self._moet_pipe_cur] = {a: getattr(self, a) for a in attrs}\n"
        "        if slot not in self._moet_pipe_slots:\n"
        "            base = self._moet_pipe_slots[0]\n"
        "            fresh: dict[str, Any] = {}\n"
        "            for a in attrs:\n"
        "                v = base[a]\n"
        "                if isinstance(v, torch.Tensor):\n"
        "                    fresh[a] = torch.zeros_like(v)\n"
        "                elif isinstance(v, UvaBufferPool):\n"
        "                    fresh[a] = UvaBufferPool(v.size, v.dtype, v.max_concurrency)\n"
        '                elif a == "_replay_attn_groups":\n'
        "                    fresh[a] = None  # built by _copy_attn_groups on the slot's first replay batch\n"
        "                else:\n"
        "                    fresh[a] = v\n"
        "            self._moet_pipe_slots[slot] = fresh\n"
        "        for a, v in self._moet_pipe_slots[slot].items():\n"
        "            setattr(self, a, v)\n"
        "        self._moet_pipe_cur = slot\n"
        "\n"
        "    def _warm_up_replay_kernels(\n",
    ),
]

# ---------------------------------------------------------------------------------------------------------------
# vllm/models/deepseek_v41/decoder_replay_layers.py
# ---------------------------------------------------------------------------------------------------------------
REPLAY_EDITS = [
    (
        "import",
        "from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput\n",
        "from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput\n" + IMPORT,
    ),
    (
        "replay batch per slot",
        "        # Set by the model state every step; None runs the layers on the batch.\n"
        "        self.replay_batch: ReplayBatch | None = None\n",
        "        # Set by the model state every step; None runs the layers on the batch.\n"
        f"        # {MARKER}: one per slot - the model state sets the slot it prepares (0 = the whole step, 1.. = the\n"
        "        # step's row groups), a group's thread reads its own.\n"
        "        self._moet_replay_batches: dict[int, ReplayBatch | None] = {}\n",
    ),
    (
        "replay batch property",
        "    def __call__(\n        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None\n    ) -> tuple[torch.Tensor, ...]:\n        if self.uses_replay_batch():\n",
        "    @property\n"
        "    def replay_batch(self) -> ReplayBatch | None:\n"
        f"        return self._moet_replay_batches.get(_moet_pipe.current_slot())  # {MARKER}\n"
        "\n"
        "    @replay_batch.setter\n"
        "    def replay_batch(self, value: ReplayBatch | None) -> None:\n"
        f"        self._moet_replay_batches[_moet_pipe.prepare_slot()] = value  # {MARKER}\n"
        "\n"
        "    def __call__(\n        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None\n    ) -> tuple[torch.Tensor, ...]:\n        if self.uses_replay_batch():\n",
    ),
]

# ---------------------------------------------------------------------------------------------------------------
# vllm/models/deepseek_v41/common/engram.py
# ---------------------------------------------------------------------------------------------------------------
ENGRAM_EDITS = [
    (
        "import",
        "from vllm.logger import init_logger\n",
        "from vllm.logger import init_logger\n" + IMPORT,
    ),
    (
        "rows at the group's offset",
        "            staged = self.staged_rows[: slot * self.embed_tokens.dp_size]\n"
        "            return _gather_engram_rows(staged, num_tokens)\n"
        "        return self.staged_rows[:num_tokens]\n",
        "            staged = self.staged_rows[: slot * self.embed_tokens.dp_size]\n"
        "            return _gather_engram_rows(staged, num_tokens)\n"
        f"        return _moet_pipe.slice_rows(self.staged_rows, num_tokens)  # {MARKER}: looked up for the whole step\n",
    ),
]

FILES = [
    ("v1/worker/gpu/model_runner.py", RUNNER_EDITS),
    ("models/deepseek_v41/nvidia/model.py", MODEL_EDITS),
    ("models/deepseek_v41/attention.py", ATTN_EDITS),
    ("models/deepseek_v41/nvidia/model_state.py", MODEL_STATE_EDITS),
    ("models/deepseek_v41/decoder_replay_layers.py", REPLAY_EDITS),
    ("models/deepseek_v41/common/engram.py", ENGRAM_EDITS),
]


def patch_text(src: str, edits, check: bool) -> tuple[str, str]:
    """Apply the edits; returns (new source, status) with status in {applied, already, missing:<edit>}."""
    if MARKER in src:
        return src, "already"
    for label, anchor, _new in edits:
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
    ap.add_argument("--helper", type=Path, default=DEFAULT_HELPER, help="the runtime module to install")
    ap.add_argument("--check", action="store_true", help="dry run: verify the anchors, change nothing")
    args = ap.parse_args()
    rc = 0
    if not args.helper.is_file():
        print(f"helper module missing: {args.helper}")
        return 1
    compile(args.helper.read_text(), str(args.helper), "exec")
    outputs = []
    for rel, edits in FILES:
        path = args.dist / rel
        if not path.exists():
            print(f"{rel}: not found under {args.dist}")
            rc = 1
            continue
        src = path.read_text()
        new, status = patch_text(src, edits, args.check)
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
            compile(new, str(path), "exec")
            outputs.append((path, new, len(edits)))
    if rc or args.check:
        return rc
    for path, new, n in outputs:
        path.write_text(new)
        print(f"{path.relative_to(args.dist)}: patched ({n} edits)")
    dst = args.dist / HELPER_DST
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.helper, dst)
    print(f"helper installed at {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
