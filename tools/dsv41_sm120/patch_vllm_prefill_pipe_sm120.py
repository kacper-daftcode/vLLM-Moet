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
                                                    inside the pipeline's step context (eager steps only); K-8: the phase
                                                    graph's capture after the inner cudagraphs (both capture passes)
  vllm/models/deepseek_v41/nvidia/model.py          the forward split into head (once) and tail (once per group); the
                                                    layer's two boundary phases; the dump hook of the bits harness; K-8:
                                                    the tail split into the sequence-parallel section (shard, layers 0-19,
                                                    the first replay layer's KV = the phase graph) and the rest, the SP
                                                    path kept under the phase graph's stream capture
  vllm/models/deepseek_v41/attention.py             the end of phase 1 after the projections' all-gather
  vllm/models/deepseek_v41/nvidia/model_state.py    the replay-batch buffers / staging / builders per slot
  vllm/models/deepseek_v41/decoder_replay_layers.py the replay batch per slot (the thread's group)
  vllm/models/deepseek_v41/common/engram.py         the Engram rows at the group's row offset; K-8: no wait on the prefetch
                                                    event inside the phase graph's capture (the module waits before the launch)
  vllm/v1/attention/backends/mla/indexer.py         K-8 only: a context-independent prefill plan (sub-chunks of the captured
                                                    maximum context, equal TP row shards) for a phase-graph row group
  (and tools/dsv41_sm120/sp_moe_gather/sp_moe_gather_sm120.py: one MoE gather in flight per stand-in, not per process)

The K-8 edits (VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS, marker "... phase graph") form a second stage applied after the first:
a file the first stage already patched (an r26 / r26b image) takes the second stage alone, so the r26b image's files can be
re-patched in place for a bind-mount (K/k8pg/repatch_r26b.sh). The K-9 edits (VLLM_MOET_PREFILL_PIPE_SHARED_TAIL, marker
"... shared tail": one replay tail for both groups - model.py's replay call and the mHC split-k pin, model_state.py's shared
slot graph manager) are a third stage, applied after the second the same way.

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
MARKER2 = "[vllm-moet] sm_120 prefill pipeline phase graph"  # the K-8 stage (VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS)
MARKER3 = "[vllm-moet] sm_120 prefill pipeline shared tail"  # the K-9 stage (VLLM_MOET_PREFILL_PIPE_SHARED_TAIL)
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

# K-8: the phase graph is captured right after the model state's inner graphs, in both of vLLM's capture passes (the
# memory-profiling pass measures it, the real pass keeps it)
RUNNER_EDITS_K8 = [
    (
        "phase graph capture",
        "                    self.model_state.capture_inner_cudagraphs(\n"
        "                        input_buffers,\n"
        "                        self.block_tables,\n"
        "                        self.attn_groups,\n"
        "                        self.kv_cache_config,\n"
        "                    )\n",
        "                    self.model_state.capture_inner_cudagraphs(\n"
        "                        input_buffers,\n"
        "                        self.block_tables,\n"
        "                        self.attn_groups,\n"
        "                        self.kv_cache_config,\n"
        "                    )\n"
        f"                    # {MARKER2} (VLLM_MOET_PREFILL_PIPE_PHASE_GRAPHS): the sequence-parallel section of a full pipelined step\n"
        "                    _moet_pipe.capture_phase_graphs(self, profile_only=profile_only)\n",
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

# K-8 (second stage, on the text the first stage produced): the group's tail split into the sequence-parallel section and
# the rest, the dispatch passing both to the pipeline, the SP path kept under the phase graph's stream capture.
MODEL_EDITS_K8 = [
    (
        "sp step under capture",
        "        if torch.cuda.is_current_stream_capturing() or in_piecewise_cudagraph():\n"
        "            return False\n"
        "        logger.info_once(\n"
        '            "vllm-moet sm_120 prefill SP: eager steps of >= %d tokens reduce-scatter the sublayer outputs and run "\n',
        f"        # {MARKER2}: the step graph is captured on the sequence-parallel path - vLLM's own graphs never carry it\n"
        "        if (torch.cuda.is_current_stream_capturing() and not _moet_pipe.phase_graph_capturing()) or in_piecewise_cudagraph():\n"
        "            return False\n"
        "        logger.info_once(\n"
        '            "vllm-moet sm_120 prefill SP: eager steps of >= %d tokens reduce-scatter the sublayer outputs and run "\n',
    ),
    (
        "replay split",
        "    replay_batch = replay.replay_batch\n"
        "    assert replay_batch is not None\n"
        "    replay.write_batch_kv(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n"
        f"    # {MARKER}: the replay section (rows gather, layers 20-39 on 128 rows, collapse) on its own stream, handing the\n"
        "    # CPU to the other group once per layer (yield_point in the layer forward)\n"
        "    _moet_pipe.replay_begin(hidden_states, pre_mix, post_mix, res_mix, residual)\n",
        "    assert replay.replay_batch is not None\n"
        "    replay.write_batch_kv(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n"
        "    return _moet_sp_decoder_replay_rest(replay, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n"
        "\n"
        "\n"
        "def _moet_sp_decoder_replay_rest(\n"
        "    replay,\n"
        "    hidden_states: torch.Tensor,\n"
        "    positions: torch.Tensor,\n"
        "    input_ids: torch.Tensor | None,\n"
        "    pre_mix: torch.Tensor,\n"
        "    post_mix: torch.Tensor,\n"
        "    res_mix: torch.Tensor,\n"
        "    residual: torch.Tensor,\n"
        ") -> tuple[torch.Tensor, ...]:\n"
        f'    """{MARKER2}: _moet_sp_decoder_replay after the first replay layer\'s KV write (the phase graph ends with that\n'
        '    write): the replay rows\' states brought to full rows, the replay layers, the collapse."""\n'
        "    replay_batch = replay.replay_batch\n"
        "    assert replay_batch is not None\n"
        f"    # {MARKER}: the replay section (rows gather, layers 20-39 on 128 rows, collapse) on its own stream, handing the\n"
        "    # CPU to the other group once per layer (yield_point in the layer forward)\n"
        "    _moet_pipe.replay_begin(hidden_states, pre_mix, post_mix, res_mix, residual)\n",
    ),
    (
        "dispatch hooks",
        "        return _moet_pipe.merge_outputs(_moet_pipe.run_groups(groups, run_group, pipelined))\n",
        f"        def group_inputs(g):  # {MARKER2}: the group's rows of the step's tensors (copied into the graph's static buffers)\n"
        "            return (rows(hidden_states, g), rows(positions, g), rows(input_ids, g), rows(engram_hashes, g), rows(engram_mask, g))\n"
        "\n"
        "        def group_section(g, hs, pos, ids, hashes, mask):\n"
        "            return self._moet_sp_section(hs, pos, ids, None, hashes, mask)\n"
        "\n"
        "        def group_rest(g, pos, ids, state):\n"
        "            return self._moet_tail_rest(pos, ids, None, state)\n"
        "\n"
        "        return _moet_pipe.merge_outputs(\n"
        "            _moet_pipe.run_groups(\n"
        "                groups, run_group, pipelined, phase=_moet_pipe.PhaseHooks(group_inputs, group_section, group_rest)\n"
        "            )\n"
        "        )\n",
    ),
    (
        "tail head",
        "    def _moet_forward_tail(\n"
        "        self,\n"
        "        hidden_states: torch.Tensor,\n"
        "        positions: torch.Tensor,\n"
        "        input_ids: torch.Tensor | None,\n"
        "        intermediate_tensors: IntermediateTensors | None,\n"
        "        engram_hashes: torch.Tensor | None,\n"
        "        engram_mask: torch.Tensor | None,\n"
        "    ) -> torch.Tensor | IntermediateTensors:\n"
        f'        """{MARKER}: the forward from the sequence-parallel shard on (the body of forward as before)."""\n',
        "    def _moet_forward_tail(\n"
        "        self,\n"
        "        hidden_states: torch.Tensor,\n"
        "        positions: torch.Tensor,\n"
        "        input_ids: torch.Tensor | None,\n"
        "        intermediate_tensors: IntermediateTensors | None,\n"
        "        engram_hashes: torch.Tensor | None,\n"
        "        engram_mask: torch.Tensor | None,\n"
        "    ) -> torch.Tensor | IntermediateTensors:\n"
        f'        """{MARKER}: the forward from the sequence-parallel shard on (the body of forward as before) = the\n'
        f'        sequence-parallel section and the rest ({MARKER2}: a pipelined step may run the section as the graph)."""\n'
        "        state = self._moet_sp_section(hidden_states, positions, input_ids, intermediate_tensors, engram_hashes, engram_mask)\n"
        "        return self._moet_tail_rest(positions, input_ids, intermediate_tensors, state)\n"
        "\n"
        "    def _moet_sp_section(\n"
        "        self,\n"
        "        hidden_states: torch.Tensor,\n"
        "        positions: torch.Tensor,\n"
        "        input_ids: torch.Tensor | None,\n"
        "        intermediate_tensors: IntermediateTensors | None,\n"
        "        engram_hashes: torch.Tensor | None,\n"
        "        engram_mask: torch.Tensor | None,\n"
        '    ) -> "_moet_pipe.SectionState":\n'
        f'        """{MARKER2}: the sequence-parallel section of the tail - the shard, the layers before the replay layers,\n'
        '        the first replay layer\'s KV write when a replay batch exists (what the phase graph captures)."""\n',
    ),
    (
        "tail rest",
        "        late_aux: list[torch.Tensor] = []\n"
        "        if self.decoder_replay_layers is not None:\n"
        "            if (\n"
        "                _moet_sp_sharded(residual, positions)  # [vllm-moet] sm_120 prefill SP: KV for every row from the shards, the replay rows gathered\n"
        "                and self.decoder_replay_layers.replay_batch is not None\n"
        "                and not in_piecewise_cudagraph()\n"
        "            ):\n"
        "                hidden_states, pre_mix, *late_aux = _moet_sp_decoder_replay(\n",
        f"        # {MARKER2}: the section ends with the first replay layer's KV write; the rest is _moet_tail_rest\n"
        "        wrote_kv = False\n"
        "        if (\n"
        "            self.decoder_replay_layers is not None\n"
        "            and _moet_sp_sharded(residual, positions)  # [vllm-moet] sm_120 prefill SP: KV for every row from the shards, the replay rows gathered\n"
        "            and self.decoder_replay_layers.replay_batch is not None\n"
        "            and not in_piecewise_cudagraph()\n"
        "        ):\n"
        "            self.decoder_replay_layers.write_batch_kv(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n"
        "            wrote_kv = True\n"
        "        return _moet_pipe.SectionState(hidden_states, residual, post_mix, res_mix, pre_mix, aux_hidden_by_layer, wrote_kv, input_ids)\n"
        "\n"
        "    def _moet_tail_rest(\n"
        "        self,\n"
        "        positions: torch.Tensor,\n"
        "        input_ids: torch.Tensor | None,\n"
        "        intermediate_tensors: IntermediateTensors | None,\n"
        '        state: "_moet_pipe.SectionState",\n'
        "    ) -> torch.Tensor | IntermediateTensors:\n"
        f'        """{MARKER2}: the rest of the tail after the sequence-parallel section - the replay hand-off (the replay rows\n'
        '        gathered to full rows), the replay layers, the collapse and norm (the body of forward as before)."""\n'
        "        full_num_tokens = positions.shape[0]\n"
        "        hidden_states, residual, post_mix, res_mix, pre_mix = (\n"
        "            state.hidden_states, state.residual, state.post_mix, state.res_mix, state.pre_mix\n"
        "        )\n"
        "        aux_hidden_by_layer = state.aux_hidden_by_layer\n"
        "        input_ids = state.input_ids\n"
        "        late_aux: list[torch.Tensor] = []\n"
        "        if self.decoder_replay_layers is not None:\n"
        "            if state.wrote_kv:\n"
        "                hidden_states, pre_mix, *late_aux = _moet_sp_decoder_replay_rest(\n",
    ),
]

# K-9 (third stage): one replay tail for both groups - the replay call goes through the rendezvous (None = this group's own
# tail as today), and a layer that releases the mHC split-k pin inside the shared tail pins it to the group's row count.
MODEL_EDITS_K9 = [
    (
        "shared replay",
        "    out = replay._run(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n"
        "    _moet_pipe.replay_end(out)\n"
        "    return out\n",
        f"    # {MARKER3} (VLLM_MOET_PREFILL_PIPE_SHARED_TAIL): one replay tail for both groups, or None = this group's own as today\n"
        "    out = _moet_pipe.shared_replay(replay, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n"
        "    if out is None:\n"
        "        out = replay._run(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)\n"
        "    _moet_pipe.replay_end(out)\n"
        "    return out\n",
    ),
    (
        "mhc pin",
        "    _moet_mhc_warmup._MOET_SP_PIN_TOKENS = num_tokens\n",
        f"    _moet_mhc_warmup._MOET_SP_PIN_TOKENS = _moet_pipe.pin_tokens(num_tokens)  # {MARKER3}: the group's row count inside the shared tail\n",
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

# K-9 (third stage): the shared slot's graph manager captures the sizes >= 2 x the replay window only (slot_graph_config: a
# config copy; None = the slot gets no graphs); the groups' slots are built as before. A group slot built while the runner
# knows the shared tail will serve the step gets a LIGHT replay batch - its rows only, without the kept batch's two metadata
# builds (shared_tail_light).
MODEL_STATE_EDITS_K9 = [
    (
        "light replay batch",
        "        kept_batch, kept_slot_mappings = self._kept_input_batch(\n"
        "            input_batch, slot_mappings, replay_start, kept_lens, num_padded\n"
        "        )\n"
        "        if self._replay_attn_groups is None:\n",
        "        kept_batch, kept_slot_mappings = self._kept_input_batch(\n"
        "            input_batch, slot_mappings, replay_start, kept_lens, num_padded\n"
        "        )\n"
        f"        if _moet_pipe.light_replay_batch():  # {MARKER3}: the shared tail runs this slot's rows - no kept-batch metadata\n"
        "            layers.replay_batch = ReplayBatch(self._kept_rows[:num_tokens], None, None)  # type: ignore[arg-type]\n"
        "            return\n"
        "        if self._replay_attn_groups is None:\n",
    ),
    (
        "shared slot graph config",
        "            for _slot in _moet_pipe.graph_slots():\n"
        "                self._moet_pipe_select_slot(_slot)\n"
        "                self._replay_attn_groups = None\n"
        "                self._moet_pipe_slot_cudagraphs[_slot] = DecoderReplayCudaGraphManager(\n"
        "                    self.vllm_config, self.device, self.decoder_replay_layers\n"
        "                )\n",
        "            for _slot in _moet_pipe.graph_slots():\n"
        f"                # {MARKER3}: the shared slot's sizes are the capture list's >= 2 x window (None = no graphs for it)\n"
        "                _slot_cfg = _moet_pipe.slot_graph_config(self.vllm_config, _slot, self.decoder_replay_layers.window)\n"
        "                if _slot_cfg is None:\n"
        "                    continue\n"
        "                self._moet_pipe_select_slot(_slot)\n"
        "                self._replay_attn_groups = None\n"
        "                self._moet_pipe_slot_cudagraphs[_slot] = DecoderReplayCudaGraphManager(\n"
        "                    _slot_cfg, self.device, self.decoder_replay_layers\n"
        "                )\n",
    ),
    (
        "shared slot capture scope",
        "                self._moet_pipe_slot_cudagraphs[_slot].capture_replay_graphs(prepare)\n",
        f"                # {MARKER3}: the shared slot's graphs record the per-group split of the all-reduces / cuBLAS GEMMs\n"
        "                with _moet_pipe.shared_capture_scope(_slot, self.decoder_replay_layers.window):\n"
        "                    self._moet_pipe_slot_cudagraphs[_slot].capture_replay_graphs(prepare)\n",
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

# K-8: inside the phase graph's capture the layer must not wait on the prefetch event (recorded outside the capture:
# cudaErrorStreamCaptureIsolation); the module makes the compute stream wait for it before every launch instead.
ENGRAM_EDITS_K8 = [
    (
        "prefetch wait under capture",
        "    def _finish_prefetch(self, event: torch.cuda.Event) -> None:\n"
        "        # Wait for this layer's lookup only, not for later layers on the stream.\n"
        "        torch.cuda.current_stream().wait_event(event)\n",
        "    def _finish_prefetch(self, event: torch.cuda.Event) -> None:\n"
        "        # Wait for this layer's lookup only, not for later layers on the stream.\n"
        f"        if _moet_pipe.phase_graph_capturing():  # {MARKER2}: the event is recorded outside the capture; the launch waits for it\n"
        "            return\n"
        "        torch.cuda.current_stream().wait_event(event)\n",
    ),
]

# ---------------------------------------------------------------------------------------------------------------
# vllm/v1/attention/backends/mla/indexer.py (K-8 only)
# ---------------------------------------------------------------------------------------------------------------
# The indexer's prefill plan depends on the context: the query rows are split into sub-chunks of budget / (4 B x context)
# rows (the number of logits kernels and chunk tensors changes with the context) and the TP row shard is balanced by the
# rows' contexts (the all_gatherv byte counts change). For a phase-graph row group both are made context-independent -
# the chunk boundaries of the captured maximum context, equal shards - so a real step's metadata has the captured
# structure; bits are untouched (rows are scored independently of their grouping and of the rank that scores them).
INDEXER_EDITS_K8 = [
    (
        "import",
        "from vllm.logger import init_logger\n",
        "from vllm.logger import init_logger\n" + IMPORT,
    ),
    (
        "static chunk plan",
        "                chunk_specs = self._split_indexer_prefill_chunks(\n"
        "                    self._prefill_split_seq_lens(compressed_seq_lens_cpu[num_decodes:]),\n"
        "                    prefill_query_lens_cpu,\n"
        "                    self.max_prefill_buffer_size,\n"
        "                    max_logits_bytes,\n"
        "                    request_offset=num_decodes,\n"
        "                )\n",
        "                _moet_split_lens = self._prefill_split_seq_lens(compressed_seq_lens_cpu[num_decodes:])\n"
        "                chunk_specs = self._split_indexer_prefill_chunks(\n"
        "                    _moet_split_lens,\n"
        "                    prefill_query_lens_cpu,\n"
        "                    self.max_prefill_buffer_size,\n"
        f"                    # {MARKER2}: a phase-graph group plans its sub-chunks for the captured maximum context\n"
        "                    _moet_pipe.indexer_logits_budget(max_logits_bytes, _moet_split_lens, self.compress_ratio),\n"
        "                    request_offset=num_decodes,\n"
        "                )\n",
    ),
    (
        "static row shard",
        "                row_shard_sizes = balanced_prefill_row_shard(\n"
        "                    seq_lens_cpu[num_decodes:],\n"
        "                    prefill_query_lens_cpu,\n"
        "                    self.compress_ratio,\n"
        "                    self.tp_size,\n"
        "                    self.vllm_config.attention_config.indexer_tp_shard_min_rows,\n"
        "                )\n",
        f"                row_shard_sizes = _moet_pipe.indexer_row_shard(  # {MARKER2}: equal shards for a phase-graph group\n"
        "                    balanced_prefill_row_shard(\n"
        "                        seq_lens_cpu[num_decodes:],\n"
        "                        prefill_query_lens_cpu,\n"
        "                        self.compress_ratio,\n"
        "                        self.tp_size,\n"
        "                        self.vllm_config.attention_config.indexer_tp_shard_min_rows,\n"
        "                    ),\n"
        "                    int(prefill_query_lens_cpu.sum()),\n"
        "                    self.tp_size,\n"
        "                )\n",
    ),
]

FILES = [
    ("v1/worker/gpu/model_runner.py", RUNNER_EDITS, RUNNER_EDITS_K8, []),
    ("models/deepseek_v41/nvidia/model.py", MODEL_EDITS, MODEL_EDITS_K8, MODEL_EDITS_K9),
    ("models/deepseek_v41/attention.py", ATTN_EDITS, [], []),
    ("models/deepseek_v41/nvidia/model_state.py", MODEL_STATE_EDITS, [], MODEL_STATE_EDITS_K9),
    ("models/deepseek_v41/decoder_replay_layers.py", REPLAY_EDITS, [], []),
    ("models/deepseek_v41/common/engram.py", ENGRAM_EDITS, ENGRAM_EDITS_K8, []),
    ("v1/attention/backends/mla/indexer.py", [], INDEXER_EDITS_K8, []),
]


def _apply(src: str, edits) -> tuple[str, str | None]:
    for label, anchor, _new in edits:
        n = src.count(anchor)
        if n != 1:
            return src, f"missing:{label} ({n} matches)"
    for _label, anchor, new in edits:
        src = src.replace(anchor, new, 1)
    return src, None


def patch_text(src: str, edits, check: bool, edits_k8=(), edits_k9=()) -> tuple[str, str]:
    """Apply the edits in three stages - the pipeline (MARKER), the K-8 phase graph (MARKER2), the K-9 shared tail (MARKER3)
    - each skipped when its marker is in the text already; returns (new source, status) with status in {applied,
    applied:k8, applied:k9, already, missing:<edit>}. A file an earlier stage patched (an r26 / r26b image) takes the later
    stages alone. `check` only names the dry run (the caller writes nothing then); a later stage's anchors are checked on
    the earlier stages' output."""
    stages = []
    if MARKER not in src:
        stages.append(("", edits))
    if edits_k8 and MARKER2 not in src:
        stages.append(("k8", edits_k8))
    if edits_k9 and MARKER3 not in src:
        stages.append(("k9", edits_k9))
    if not stages:
        return src, "already"
    for _tag, stage_edits in stages:
        src, err = _apply(src, stage_edits)
        if err is not None:
            return src, err
    return src, "applied" if stages[0][0] == "" else f"applied:{stages[0][0]}"


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
    for rel, edits, edits_k8, edits_k9 in FILES:
        path = args.dist / rel
        if not path.exists():
            print(f"{rel}: not found under {args.dist}")
            rc = 1
            continue
        src = path.read_text()
        new, status = patch_text(src, edits, args.check, edits_k8, edits_k9)
        if status.startswith("missing"):
            print(f"{rel}: anchor {status} - not patched")
            rc = 1
            continue
        if status == "already":
            print(f"{rel}: already patched")
            continue
        n_edits = {"applied": len(edits) + len(edits_k8) + len(edits_k9), "applied:k8": len(edits_k8) + len(edits_k9),
                   "applied:k9": len(edits_k9)}[status]
        if args.check:
            print(f"{rel}: all {n_edits} anchors present (dry run{', later stages only: ' + status[8:] if status != 'applied' else ''})")
        else:
            compile(new, str(path), "exec")
            outputs.append((path, new, n_edits))
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
