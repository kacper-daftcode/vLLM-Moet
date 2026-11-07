#!/usr/bin/env python3
"""Ripple step 2: the DFlash2 drafter drafts the step's k under a dynamic schedule, not the maximum N.

Where: vllm/v1/worker/gpu/spec_decode/dflash/{speculator,cudagraph}.py, dflash2/speculator.py and
vllm/model_executor/models/qwen3_dflash2.py (vLLM 03bd3c68, model runner V2).

What: with `num_speculative_tokens_per_batch_size` (served: [[1,2,15],[3,8,11],[9,32,7]]) the scheduler hands each step
a k <= N and the target runs 1 + k tokens per request with CUDA graphs per query width (`get_decode_cudagraph_query_lens`).
The DFlash drafter kept its static width: `propose()` received `num_speculative_tokens` but used it only for 0, so every
step ran 1 + N = 16 query rows per request through the 5 drafter layers (intermediate 40960: the drafter is 1/5 of the
target's MLP FLOPs per row), sampled N rows, scored N x top_k candidates, stored 16 x requests context rows and captured
graphs for the 16-wide batch only - at 32 requests 512 drafter rows for 7 drafts each. After the patch a step of k drafts
runs 1 + k query rows per request, samples k rows, scores k x top_k candidates, stores (1 + k) x requests context rows in
its graph and dispatches to graphs captured per query width of the schedule (q = 1 + k up to the range's largest batch:
16 up to 2 requests, 12 up to 8, 8 up to 32 for the served schedule), exactly as the target does. The greedy output is the
target's argmax whatever the drafter proposes (parity vs the unpatched server IDENTICAL; GSM8K-200 at C8 194 vs 193);
only the drafts' count changes (the rows past k were masked to -1 by the runner before and are not computed now).

The DFlash2 grouped convolutions (attention_conv / mlp_conv of every drafter layer: 4 causal taps over the rows of one
request's query block) took each row's position in its block from row % (1 + N), so with 1 + k rows per request the
blocks straddled requests and the drafts collapsed (first arm on rtx: acceptance 0.24 instead of 0.72 at C8). The
speculator now writes each query row's offset in its request (prepare_dflash_inputs -> `block_offsets`, 0 for padding
rows, always <= the row index so the taps never read before row 0) and the convolutions read it instead; a module without
the buffer (any other DFlash2 user) convolves as before. The block's attention is non-causal (Ripple's draft config
`is_causal: false`); a narrower block is what a static `num_speculative_tokens` = k gives the same drafter, and measured
the same acceptance (k7 at 32 requests: 0.81-0.82 either way).

Measured on rtx (4x RTX PRO 6000, TP4, served configuration otherwise; `glm_spec_matrix`, 512 tok, two runs): code
C32 5045 tok/s @ 24.0 steps/s vs 4738 @ 22.4 (+6.5 %, the static-k7 level with k15 kept at C1), prose C32 2074-2089 vs
1948-1970 (+6.5 %); C8 3010-3097 @ 43.5 vs 3036-3112 @ 43.7 (the same; the drafter's 128 vs 96 rows were ~0.4 ms of
22.7); C1 unchanged (k = N: the same graphs and kernels, parity IDENTICAL).

Scope: DFlash2 (and LiLiCorr, which inherits its _generate_draft) opt in through `_dynamic_query_width = True`; plain
DFlash, DSpark and XPress keep the static width (their _generate_draft overrides assume it) - their behaviour is
unchanged. The compiled DFlash2 candidate selector is specialized on its step dimension (dim 1 of every input), so the
other widths run its forward eagerly (a GEMM and a Triton kernel, captured in the FULL graph like everything else).

Usage: python3 patch_vllm_dflash_dynamic_k_sm120.py [--root DIR] [--check]   (idempotent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "[vllm-moet] step 2"

# ----------------------------------------------------------------------------------------------------------------------
# dflash/cudagraph.py: the capture closure tells the context step how many query rows its graph has.
CUDAGRAPH = [
    (
        "        prepare_forward: Callable[[int, dict[str, Any] | None], None],\n",
        "        prepare_forward: Callable[..., None],\n",
    ),
    (
        "            def forward(cg_mode: CUDAGraphMode) -> None:\n"
        "                prepare_forward(num_reqs, attn_metadata)\n",
        "            def forward(cg_mode: CUDAGraphMode) -> None:\n"
        f"                # {MARKER}: the graph's query rows (num_reqs x its query width)\n"
        "                prepare_forward(num_reqs, attn_metadata, num_tokens)\n",
    ),
]

# ----------------------------------------------------------------------------------------------------------------------
# dflash/speculator.py
DFLASH = [
    (
        '    _speculator_name = "DFlash"  # For logging, so we can share methods with subclasses\n',
        '    _speculator_name = "DFlash"  # For logging, so we can share methods with subclasses\n'
        f"    # {MARKER}: under a dynamic schedule (num_speculative_tokens_per_batch_size) a step that drafts k < N\n"
        "    # tokens runs 1 + k query rows per request (and samples k rows) instead of 1 + N, with CUDA graphs per\n"
        "    # query width of the schedule - when the subclass's _generate_draft reads the per-step width\n"
        "    # (_num_query_per_req_step; DFlash2's does, DSpark's and XPress's assume the static one).\n"
        "    _dynamic_query_width = False\n",
    ),
    (
        "        # Each request emits exactly (bonus + N mask) query tokens per step.\n"
        "        self.num_query_per_req = 1 + self.num_speculative_steps\n",
        "        # Each request emits exactly (bonus + N mask) query tokens per step.\n"
        "        self.num_query_per_req = 1 + self.num_speculative_steps\n"
        f"        # {MARKER}: the query width of the step being drafted (1 + k); propose() and the graph capture set it.\n"
        "        self._num_query_per_req_step = self.num_query_per_req\n",
    ),
    (
        "        self.sample_col = torch.arange(\n"
        "            self.num_speculative_steps, dtype=torch.int32, device=device\n"
        "        ).repeat(self.max_num_reqs)\n",
        "        self.sample_col = torch.arange(\n"
        "            self.num_speculative_steps, dtype=torch.int32, device=device\n"
        "        ).repeat(self.max_num_reqs)\n"
        f"        # {MARKER}: the same per k drafted tokens (sample row i -> column i % k).\n"
        "        self._sample_col_by_k = {\n"
        "            k: torch.arange(k, dtype=torch.int32, device=device).repeat(\n"
        "                self.max_num_reqs\n"
        "            )\n"
        "            for k in range(1, self.num_speculative_steps)\n"
        "        }\n"
        "        self._sample_col_by_k[self.num_speculative_steps] = self.sample_col\n",
    ),
    (
        "        self.query_cudagraph_manager = DFlashCudaGraphManager(\n"
        "            self.vllm_config,\n"
        "            self.device,\n"
        "            cudagraph_mode,\n"
        "            decode_query_lens={self.num_query_per_req: self.max_num_reqs},\n"
        "        )\n",
        f"        # {MARKER}: graphs per query width of the dynamic schedule (1 + k of each batch-size range, up to\n"
        "        # the range's largest batch), as the target's; the static width alone without a schedule.\n"
        "        decode_query_lens = (\n"
        "            self.vllm_config.get_decode_cudagraph_query_lens(self.num_query_per_req)\n"
        "            if self._dynamic_query_width\n"
        "            else {self.num_query_per_req: self.max_num_reqs}\n"
        "        )\n"
        "        self.query_cudagraph_manager = DFlashCudaGraphManager(\n"
        "            self.vllm_config,\n"
        "            self.device,\n"
        "            cudagraph_mode,\n"
        "            decode_query_lens=decode_query_lens,\n"
        "        )\n",
    ),
    (
        "    def _prepare_graph_forward(\n"
        "        self, num_reqs: int, attn_metadata: dict[str, Any] | None\n"
        "    ) -> None:\n"
        '        """The FULL graph\'s context step: combine the staged target hidden\n'
        '        states, publish the context anchor and store the context K/V."""\n'
        "        num_context = self._num_graph_context_tokens(num_reqs)\n",
        "    def _prepare_graph_forward(\n"
        "        self,\n"
        "        num_reqs: int,\n"
        "        attn_metadata: dict[str, Any] | None,\n"
        "        num_query_tokens: int | None = None,\n"
        "    ) -> None:\n"
        '        """The FULL graph\'s context step: combine the staged target hidden\n'
        '        states, publish the context anchor and store the context K/V."""\n'
        f"        # {MARKER}: the captured graph's query width (its rows / its requests) for _generate_draft.\n"
        "        self._num_query_per_req_step = (\n"
        "            num_query_tokens // num_reqs\n"
        "            if self._dynamic_query_width and num_query_tokens\n"
        "            else self.num_query_per_req\n"
        "        )\n"
        "        num_context = self._num_graph_context_tokens(num_reqs, num_query_tokens)\n",
    ),
    (
        "        num_rejected: torch.Tensor,\n"
        "        dummy_run: bool,\n"
        "    ) -> None:\n"
        '        """Eager work before a FULL replay: stage the inputs _prepare_graph_forward\n'
        '        reads and store the context rows past the graph\'s num_context rows."""\n'
        "        num_reqs = input_batch.num_reqs\n"
        "        num_target_tokens = input_batch.num_tokens\n"
        "        num_context = self._num_graph_context_tokens(num_graph_reqs)\n",
        "        num_rejected: torch.Tensor,\n"
        "        dummy_run: bool,\n"
        "        num_graph_tokens: int | None = None,\n"
        "    ) -> None:\n"
        '        """Eager work before a FULL replay: stage the inputs _prepare_graph_forward\n'
        '        reads and store the context rows past the graph\'s num_context rows."""\n'
        "        num_reqs = input_batch.num_reqs\n"
        "        num_target_tokens = input_batch.num_tokens\n"
        "        num_context = self._num_graph_context_tokens(num_graph_reqs, num_graph_tokens)\n",
    ),
    (
        "        num_sample = num_reqs * self.num_speculative_steps\n"
        "        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]\n"
        "        # sample_pos is the predicted token's position P. Sampling keys a draw\n"
        "        # by the position before the sampled token, P-1.\n"
        "        draft_tokens = self.sample_draft(\n"
        "            sample_hidden_states,\n"
        "            self.sample_pos[:num_sample] - 1,\n"
        "            self.sample_idx_mapping[:num_sample],\n"
        "            self.temperature,\n"
        "            self.seeds,\n"
        "            self.sample_col[:num_sample],\n"
        "            self.draft_logits,\n"
        "            num_steps=self.num_speculative_steps,\n"
        "        )\n"
        "        self.draft_tokens[:num_reqs] = draft_tokens.view(\n"
        "            num_reqs, self.num_speculative_steps\n"
        "        )\n"
        "\n"
        "    def _num_graph_context_tokens(self, num_reqs: int) -> int:\n"
        "        # Context rows a captured draft step stores: one full verify per request.\n"
        "        return min(num_reqs * (self.num_speculative_steps + 1), self.max_num_tokens)\n",
        f"        # {MARKER}: k = this step's drafted tokens (the width propose() or the graph capture set).\n"
        "        num_steps = self._num_query_per_req_step - 1\n"
        "        num_sample = num_reqs * num_steps\n"
        "        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]\n"
        "        # sample_pos is the predicted token's position P. Sampling keys a draw\n"
        "        # by the position before the sampled token, P-1.\n"
        "        draft_tokens = self.sample_draft(\n"
        "            sample_hidden_states,\n"
        "            self.sample_pos[:num_sample] - 1,\n"
        "            self.sample_idx_mapping[:num_sample],\n"
        "            self.temperature,\n"
        "            self.seeds,\n"
        "            self._sample_col_by_k[num_steps][:num_sample],\n"
        "            self.draft_logits,\n"
        "            num_steps=num_steps,\n"
        "        )\n"
        "        self.draft_tokens[:num_reqs, :num_steps] = draft_tokens.view(\n"
        "            num_reqs, num_steps\n"
        "        )\n"
        "\n"
        "    def _num_graph_context_tokens(\n"
        "        self, num_reqs: int, num_query_tokens: int | None = None\n"
        "    ) -> int:\n"
        "        # Context rows a captured draft step stores: one full verify per request -\n"
        f"        # {MARKER}: under a dynamic schedule the graph's own query rows (a steady\n"
        "        # batch verifies 1 + k target tokens per request; rows past them are\n"
        "        # stored eagerly before the replay).\n"
        "        if num_query_tokens is None or not self._dynamic_query_width:\n"
        "            num_query_tokens = num_reqs * (self.num_speculative_steps + 1)\n"
        "        return min(num_query_tokens, self.max_num_tokens)\n",
    ),
    (
        "        num_target_tokens = input_batch.num_tokens\n"
        "        num_query_tokens = num_reqs * self.num_query_per_req\n"
        "        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()\n"
        "        self.draft_max_seq_len = min(\n"
        "            max_seq_len + self.num_query_per_req, self.max_model_len\n"
        "        )\n",
        "        num_target_tokens = input_batch.num_tokens\n"
        f"        # {MARKER}: this step drafts k tokens (the dynamic schedule's for the batch size; N without one or\n"
        "        # for a dummy run), so each request has 1 + k query rows and k sample rows.\n"
        "        num_steps = self.num_speculative_steps\n"
        "        if self._dynamic_query_width and num_speculative_tokens is not None:\n"
        "            num_steps = max(0, min(num_speculative_tokens, self.num_speculative_steps))\n"
        "        num_query_per_req = self.num_query_per_req - (self.num_speculative_steps - num_steps)\n"
        "        self._num_query_per_req_step = num_query_per_req\n"
        "        num_query_tokens = num_reqs * num_query_per_req\n"
        "        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()\n"
        "        self.draft_max_seq_len = min(\n"
        "            max_seq_len + num_query_per_req, self.max_model_len\n"
        "        )\n",
    ),
    (
        "                self.parallel_drafting_token_id,\n"
        "                self.num_query_per_req,\n"
        "                self.num_speculative_steps,\n"
        "                self.max_num_reqs,\n",
        "                self.parallel_drafting_token_id,\n"
        f"                num_query_per_req,  # {MARKER}\n"
        "                num_steps,\n"
        "                self.max_num_reqs,\n",
    ),
    (
        "            self._build_uniform_batch_dp_sync(\n"
        "                dp_sync_state,\n"
        "                num_reqs,\n"
        "                self.num_query_per_req,\n"
        "            )\n",
        "            self._build_uniform_batch_dp_sync(\n"
        "                dp_sync_state,\n"
        "                num_reqs,\n"
        "                num_query_per_req,\n"
        "            )\n",
    ),
    (
        "            num_batch_tokens,\n"
        "            uniform_token_count=self.num_query_per_req,\n",
        "            num_batch_tokens,\n"
        "            uniform_token_count=num_query_per_req,\n",
    ),
    (
        "            self._prepare_graph_replay(\n"
        "                input_batch,\n"
        "                batch_desc.num_reqs,\n"
        "                last_hidden_states,\n"
        "                aux_hidden_states,\n"
        "                num_rejected,\n"
        "                dummy_run,\n"
        "            )\n",
        "            self._prepare_graph_replay(\n"
        "                input_batch,\n"
        "                batch_desc.num_reqs,\n"
        "                last_hidden_states,\n"
        "                aux_hidden_states,\n"
        "                num_rejected,\n"
        "                dummy_run,\n"
        "                num_graph_tokens=batch_desc.num_tokens,\n"
        "            )\n",
    ),
    (
        "                num_query_per_req=self.num_query_per_req,\n"
        "                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,\n"
        "                step=self.num_query_per_req,\n",
        "                num_query_per_req=num_query_per_req,\n"
        "                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,\n"
        "                step=num_query_per_req,\n",
    ),
]

# ----------------------------------------------------------------------------------------------------------------------
# dflash2/speculator.py
DFLASH2 = [
    (
        "    num_steps: tl.constexpr,\n"
        "    top_k: tl.constexpr,\n"
        "    BLOCK_K: tl.constexpr,\n"
        "    SAMPLE_PROBABILISTIC: tl.constexpr,\n"
        "    USE_FP64: tl.constexpr,\n"
        "):\n"
        "    row = tl.program_id(0)\n",
        "    num_steps: tl.constexpr,\n"
        "    top_k: tl.constexpr,\n"
        "    BLOCK_K: tl.constexpr,\n"
        "    SAMPLE_PROBABILISTIC: tl.constexpr,\n"
        "    USE_FP64: tl.constexpr,\n"
        f"    TOKENS_STRIDE: tl.constexpr,  # {MARKER}: draft_tokens' row stride (N) when num_steps = k < N\n"
        "):\n"
        "    row = tl.program_id(0)\n",
    ),
    (
        "        tl.store(tokens_ptr + flat, token, mask=valid)\n",
        "        tl.store(tokens_ptr + row * TOKENS_STRIDE + step, token, mask=valid)\n",
    ),
    (
        "    draft_logits_stride_1,\n"
        "    num_steps: tl.constexpr,\n"
        "    top_k: tl.constexpr,\n"
        "    BLOCK_K: tl.constexpr,\n"
        "):\n"
        "    flat = tl.program_id(0)\n",
        "    draft_logits_stride_1,\n"
        "    num_steps: tl.constexpr,\n"
        "    top_k: tl.constexpr,\n"
        "    BLOCK_K: tl.constexpr,\n"
        f"    CACHE_STEPS: tl.constexpr,  # {MARKER}: the candidate cache's row stride (N) when num_steps = k < N\n"
        "):\n"
        "    flat = tl.program_id(0)\n",
    ),
    (
        "    cache_base = (row * num_steps + step) * top_k\n",
        "    cache_base = (row * CACHE_STEPS + step) * top_k\n",
    ),
    (
        "        draft_logits: DraftLogitsCache | None,\n"
        "        use_fp64: bool,\n"
        "    ) -> None:\n"
        "        candidate_ids = candidate_ids.contiguous()\n",
        "        draft_logits: DraftLogitsCache | None,\n"
        "        use_fp64: bool,\n"
        "        num_steps: int | None = None,\n"
        "    ) -> None:\n"
        f"        # {MARKER}: num_steps = the k drafted this step (candidate_ids, scores and the sample rows are laid\n"
        "        # out [num_reqs, k]); draft_tokens and the candidate cache keep their [max_num_reqs, N] layout.\n"
        "        if num_steps is None:\n"
        "            num_steps = self.num_steps\n"
        "        candidate_ids = candidate_ids.contiguous()\n",
    ),
    (
        "            self.scores,\n"
        "            num_steps=self.num_steps,\n"
        "            top_k=self.top_k,\n"
        "            BLOCK_K=block_k,\n"
        "            SAMPLE_PROBABILISTIC=draft_logits is not None,\n"
        "            USE_FP64=use_fp64,\n"
        "            num_warps=1,\n"
        "        )\n"
        "\n"
        "        if draft_logits is not None:\n"
        "            self._cache_draft_logits(\n"
        "                candidate_ids,\n"
        "                num_reqs * self.num_steps,\n"
        "                idx_mapping,\n"
        "                draft_logits,\n"
        "            )\n",
        "            self.scores,\n"
        "            num_steps=num_steps,\n"
        "            top_k=self.top_k,\n"
        "            BLOCK_K=block_k,\n"
        "            SAMPLE_PROBABILISTIC=draft_logits is not None,\n"
        "            USE_FP64=use_fp64,\n"
        "            TOKENS_STRIDE=self.num_steps,\n"
        "            num_warps=1,\n"
        "        )\n"
        "\n"
        "        if draft_logits is not None:\n"
        "            self._cache_draft_logits(\n"
        "                candidate_ids,\n"
        "                num_reqs * num_steps,\n"
        "                idx_mapping,\n"
        "                draft_logits,\n"
        "                num_steps,\n"
        "            )\n",
    ),
    (
        "        idx_mapping: torch.Tensor,\n"
        "        draft_logits: DraftLogitsCache,\n"
        "    ) -> None:\n"
        "        block_k = triton.next_power_of_2(self.top_k)\n",
        "        idx_mapping: torch.Tensor,\n"
        "        draft_logits: DraftLogitsCache,\n"
        "        num_steps: int | None = None,\n"
        "    ) -> None:\n"
        "        if num_steps is None:\n"
        "            num_steps = self.num_steps\n"
        "        block_k = triton.next_power_of_2(self.top_k)\n",
    ),
    (
        "            draft_logits.values.stride(1),\n"
        "            num_steps=self.num_steps,\n"
        "            top_k=self.top_k,\n"
        "            BLOCK_K=block_k,\n"
        "            num_warps=1,\n"
        "        )\n",
        "            draft_logits.values.stride(1),\n"
        "            num_steps=num_steps,\n"
        "            top_k=self.top_k,\n"
        "            BLOCK_K=block_k,\n"
        "            CACHE_STEPS=self.num_steps,\n"
        "            num_warps=1,\n"
        "        )\n",
    ),
    (
        '    _speculator_name = "DFlash2"\n'
        '    _candidate_top_k_key = "selector_top_k"\n',
        '    _speculator_name = "DFlash2"\n'
        '    _candidate_top_k_key = "selector_top_k"\n'
        f"    _dynamic_query_width = True  # {MARKER}: drafts the step's k, see DFlashSpeculator\n",
    ),
    (
        "        num_sample = num_reqs * self.num_speculative_steps\n"
        "        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(\n"
        "            num_reqs, self.num_speculative_steps, -1\n"
        "        )\n"
        "        candidate_ids, unary_logits = self.model.compute_candidates(\n"
        "            hidden_states.flatten(0, 1)\n"
        "        )\n"
        "        candidate_ids = candidate_ids.view(\n"
        "            num_reqs, self.num_speculative_steps, self.top_k\n"
        "        )\n",
        f"        # {MARKER}: k = this step's drafted tokens (the width propose() or the graph capture set).\n"
        "        num_steps = self._num_query_per_req_step - 1\n"
        "        num_sample = num_reqs * num_steps\n"
        "        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(\n"
        "            num_reqs, num_steps, -1\n"
        "        )\n"
        "        candidate_ids, unary_logits = self.model.compute_candidates(\n"
        "            hidden_states.flatten(0, 1)\n"
        "        )\n"
        "        candidate_ids = candidate_ids.view(num_reqs, num_steps, self.top_k)\n",
    ),
    (
        "            self.draft_tokens,\n"
        "            self.draft_logits,\n"
        "            self.use_fp64_gumbel,\n"
        "        )\n"
        "        if self.enable_adaptive_verification:\n"
        "            self._maybe_predict_acceptance(\n"
        "                self.candidate_sampler.scores[:num_reqs].flatten(0, 1),\n"
        "                self.sample_idx_mapping[:num_sample],\n"
        "                self.sample_col[:num_sample],\n"
        "            )\n",
        "            self.draft_tokens,\n"
        "            self.draft_logits,\n"
        "            self.use_fp64_gumbel,\n"
        "            num_steps=num_steps,\n"
        "        )\n"
        "        if self.enable_adaptive_verification:\n"
        "            self._maybe_predict_acceptance(\n"
        "                self.candidate_sampler.scores.view(-1, self.top_k)[:num_sample],\n"
        "                self.sample_idx_mapping[:num_sample],\n"
        "                self._sample_col_by_k[num_steps][:num_sample],\n"
        "            )\n",
    ),
    (
        "        num_reqs = candidate_ids.shape[0]\n"
        "        anchor_token_ids = self.input_buffers.input_ids[\n"
        "            : num_reqs * self.num_query_per_req : self.num_query_per_req\n"
        "        ]\n"
        "        return self.model.model.candidate_selector(\n"
        "            candidate_ids, unary_logits, hidden_states, anchor_token_ids\n"
        "        )\n",
        "        num_reqs, num_steps = candidate_ids.shape[:2]\n"
        f"        num_query_per_req = self._num_query_per_req_step  # {MARKER}\n"
        "        anchor_token_ids = self.input_buffers.input_ids[\n"
        "            : num_reqs * num_query_per_req : num_query_per_req\n"
        "        ]\n"
        "        selector = self.model.model.candidate_selector\n"
        "        if num_steps != self.num_speculative_steps:\n"
        f"            # {MARKER}: the compiled selector is specialized on its step dimension (dim 1 of every\n"
        "            # input, static under torch.compile); the schedule's other widths run its forward eagerly\n"
        "            # (one GEMM and one Triton kernel, captured in the FULL graph like the rest).\n"
        "            return selector.forward(\n"
        "                candidate_ids, unary_logits, hidden_states, anchor_token_ids\n"
        "            )\n"
        "        return selector(candidate_ids, unary_logits, hidden_states, anchor_token_ids)\n",
    ),
]

# ----------------------------------------------------------------------------------------------------------------------
# The DFlash2 grouped convolutions (attention_conv / mlp_conv of every drafter layer, 4 taps over the rows of one
# request's query block) take each row's position in its block from row % (1 + N): with 1 + k rows per request the
# blocks would straddle requests. The speculator writes each row's offset in its request (prepare_dflash_inputs) into a
# buffer the convolutions read instead; a module without the buffer convolves as before.
DFLASH += [
    (
        "        # Context positions for the K/V precompute, populated by prepare_dflash_inputs.\n"
        "        self.context_positions = torch.zeros(\n"
        "            self.max_num_tokens, dtype=torch.int64, device=device\n"
        "        )\n",
        "        # Context positions for the K/V precompute, populated by prepare_dflash_inputs.\n"
        "        self.context_positions = torch.zeros(\n"
        "            self.max_num_tokens, dtype=torch.int64, device=device\n"
        "        )\n"
        f"        # {MARKER}: each query row's offset in its request's block (0 for padding rows; always <= the row\n"
        "        # index), populated by prepare_dflash_inputs; the DFlash2 grouped convolutions read it.\n"
        "        self.block_offsets = torch.zeros(\n"
        "            self.max_num_tokens, dtype=torch.int32, device=device\n"
        "        )\n",
    ),
    (
        "                self.sample_from_anchor,\n"
        "                num_reqs,\n"
        "            )\n"
        "\n"
        "        if num_speculative_tokens == 0:\n",
        "                self.sample_from_anchor,\n"
        "                num_reqs,\n"
        "                block_offsets=self.block_offsets,\n"
        "            )\n"
        "\n"
        "        if num_speculative_tokens == 0:\n",
    ),
    (
        "    out_sample_idx_mapping_ptr,\n"
        "    pad_idx,\n"
        "    num_reqs,\n"
        "    num_query_per_req,\n"
        "    num_speculative_steps,\n"
        "    max_num_reqs,\n"
        "    max_num_tokens,\n"
        "    PAD_SLOT_ID: tl.constexpr,\n"
        "    BLOCK_SIZE: tl.constexpr,\n"
        "    NUM_PAD_PROGRAMS: tl.constexpr,\n"
        "):\n",
        "    out_sample_idx_mapping_ptr,\n"
        f"    out_block_offsets_ptr,  # {MARKER} (None: not written)\n"
        "    pad_idx,\n"
        "    num_reqs,\n"
        "    num_query_per_req,\n"
        "    num_speculative_steps,\n"
        "    max_num_reqs,\n"
        "    max_num_tokens,\n"
        "    PAD_SLOT_ID: tl.constexpr,\n"
        "    BLOCK_SIZE: tl.constexpr,\n"
        "    NUM_PAD_PROGRAMS: tl.constexpr,\n"
        "):\n",
    ),
    (
        "        tl.store(out_is_padding_ptr + block, True, mask=mask)\n"
        "        tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)\n",
        "        tl.store(out_is_padding_ptr + block, True, mask=mask)\n"
        "        tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)\n"
        "        if out_block_offsets_ptr is not None:\n"
        "            tl.store(out_block_offsets_ptr + block, 0, mask=mask)\n",
    ),
    (
        "    out_temperature_ptr,\n"
        "    out_seeds_ptr,\n"
        "    # Inputs from target batch\n",
        "    out_temperature_ptr,\n"
        "    out_seeds_ptr,\n"
        f"    out_block_offsets_ptr,  # {MARKER} (None: not written)\n"
        "    # Inputs from target batch\n",
    ),
    (
        "                out_sample_indices_ptr,\n"
        "                out_sample_pos_ptr,\n"
        "                out_sample_idx_mapping_ptr,\n"
        "                req_idx - num_reqs,\n",
        "                out_sample_indices_ptr,\n"
        "                out_sample_pos_ptr,\n"
        "                out_sample_idx_mapping_ptr,\n"
        "                out_block_offsets_ptr,\n"
        "                req_idx - num_reqs,\n",
    ),
    (
        "    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)\n"
        "    tl.store(out_is_padding_ptr + query_idx, False, mask=is_query)\n",
        "    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)\n"
        "    tl.store(out_is_padding_ptr + query_idx, False, mask=is_query)\n"
        "    if out_block_offsets_ptr is not None:\n"
        "        tl.store(out_block_offsets_ptr + query_idx, query_off, mask=is_query)\n",
    ),
    (
        "    sample_from_anchor: bool = False,\n"
        "    num_reqs_override: int | None = None,\n"
        ") -> None:\n"
        "    num_reqs = input_batch.num_reqs if num_reqs_override is None else num_reqs_override\n",
        "    sample_from_anchor: bool = False,\n"
        "    num_reqs_override: int | None = None,\n"
        f"    block_offsets: torch.Tensor | None = None,  # {MARKER}\n"
        ") -> None:\n"
        "    num_reqs = input_batch.num_reqs if num_reqs_override is None else num_reqs_override\n",
    ),
    (
        "        sample_idx_mapping,\n"
        "        temperature,\n"
        "        seeds,\n"
        "        input_batch.positions,\n",
        "        sample_idx_mapping,\n"
        "        temperature,\n"
        "        seeds,\n"
        "        block_offsets,\n"
        "        input_batch.positions,\n",
    ),
]

DFLASH2 += [
    (
        "    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:\n"
        "        # fp32 so the walk and the rejection that checks it read the same\n",
        "    def load_draft_model(\n"
        "        self,\n"
        "        target_model: nn.Module,\n"
        "        target_attn_layer_names: set[str],\n"
        "    ) -> nn.Module:\n"
        "        model = super().load_draft_model(target_model, target_attn_layer_names)\n"
        f"        # {MARKER}: the grouped convolutions take each row's offset in its request's query block from the\n"
        "        # speculator's buffer (prepare_dflash_inputs writes it) instead of row % (1 + N), so a step of k < N\n"
        "        # drafts convolves 1 + k rows per request. Bound before the first forward (the compiled graph reads\n"
        "        # the buffer, a fixed address).\n"
        "        for module in model.modules():\n"
        '            if "block_offsets" in getattr(module, "_buffers", {}):\n'
        "                module.block_offsets = self.block_offsets\n"
        "        return model\n"
        "\n"
        "    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:\n"
        "        # fp32 so the walk and the rejection that checks it read the same\n",
    ),
    (
        "import torch\n"
        "\n"
        "from vllm.config import VllmConfig\n",
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "from vllm.config import VllmConfig\n",
    ),
]

QWEN3_DFLASH2 = [
    (
        "    x_ptr,\n"
        "    delta_ptr,\n"
        "    base_ptr,\n"
        "    output_ptr,\n"
        "    x_stride_row,\n",
        "    x_ptr,\n"
        "    delta_ptr,\n"
        "    base_ptr,\n"
        "    output_ptr,\n"
        f"    offsets_ptr,  # {MARKER}: each row's offset in its request's block (None: row % BLOCK_SIZE)\n"
        "    x_stride_row,\n",
    ),
    (
        "    groups = channels // GROUP_SIZE\n"
        "    position = row % BLOCK_SIZE\n",
        "    groups = channels // GROUP_SIZE\n"
        "    if offsets_ptr is None:\n"
        "        position = row % BLOCK_SIZE\n"
        "    else:\n"
        "        position = tl.load(offsets_ptr + row).to(tl.int32)\n",
    ),
    (
        "def dflash2_grouped_conv_impl(\n"
        "    x: torch.Tensor,\n"
        "    delta: torch.Tensor,\n"
        "    base: torch.Tensor,\n"
        "    block_size: int,\n"
        "    group_size: int,\n"
        ") -> torch.Tensor:\n",
        "def dflash2_grouped_conv_impl(\n"
        "    x: torch.Tensor,\n"
        "    delta: torch.Tensor,\n"
        "    base: torch.Tensor,\n"
        "    block_size: int,\n"
        "    group_size: int,\n"
        f"    offsets: torch.Tensor | None = None,  # {MARKER}\n"
        ") -> torch.Tensor:\n",
    ),
    (
        "    _dflash2_grouped_conv_kernel[grid](\n"
        "        x,\n"
        "        delta,\n"
        "        base,\n"
        "        output,\n"
        "        x.stride(0),\n",
        "    _dflash2_grouped_conv_kernel[grid](\n"
        "        x,\n"
        "        delta,\n"
        "        base,\n"
        "        output,\n"
        "        offsets,\n"
        "        x.stride(0),\n",
    ),
    (
        "def dflash2_grouped_conv_fake(\n"
        "    x: torch.Tensor,\n"
        "    delta: torch.Tensor,\n"
        "    base: torch.Tensor,\n"
        "    block_size: int,\n"
        "    group_size: int,\n"
        ") -> torch.Tensor:\n",
        "def dflash2_grouped_conv_fake(\n"
        "    x: torch.Tensor,\n"
        "    delta: torch.Tensor,\n"
        "    base: torch.Tensor,\n"
        "    block_size: int,\n"
        "    group_size: int,\n"
        "    offsets: torch.Tensor | None = None,\n"
        ") -> torch.Tensor:\n",
    ),
    (
        "def dflash2_grouped_conv(\n"
        "    x: torch.Tensor,\n"
        "    delta: torch.Tensor,\n"
        "    base: torch.Tensor,\n"
        "    block_size: int,\n"
        "    group_size: int,\n"
        ") -> torch.Tensor:\n"
        "    return torch.ops.vllm.dflash2_grouped_conv(x, delta, base, block_size, group_size)\n",
        "def dflash2_grouped_conv(\n"
        "    x: torch.Tensor,\n"
        "    delta: torch.Tensor,\n"
        "    base: torch.Tensor,\n"
        "    block_size: int,\n"
        "    group_size: int,\n"
        "    offsets: torch.Tensor | None = None,\n"
        ") -> torch.Tensor:\n"
        "    return torch.ops.vllm.dflash2_grouped_conv(\n"
        "        x, delta, base, block_size, group_size, offsets\n"
        "    )\n",
    ),
    (
        "    taps: int,\n"
        ") -> torch.Tensor:\n"
        "    if hidden_states.is_cuda:\n"
        "        return dflash2_grouped_conv(hidden_states, delta, base, block_size, group_size)\n"
        "\n"
        "    blocks = hidden_states.unflatten(-1, (num_groups, group_size))\n"
        "    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)\n"
        "    output = coefficients[:, 0] * blocks\n"
        "    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)\n"
        "    if block_size & (block_size - 1) == 0:\n"
        "        position = position & (block_size - 1)\n"
        "    else:\n"
        "        position = position % block_size\n",
        "    taps: int,\n"
        "    offsets: torch.Tensor | None = None,\n"
        ") -> torch.Tensor:\n"
        "    if hidden_states.is_cuda:\n"
        "        return dflash2_grouped_conv(\n"
        "            hidden_states, delta, base, block_size, group_size, offsets\n"
        "        )\n"
        "\n"
        "    blocks = hidden_states.unflatten(-1, (num_groups, group_size))\n"
        "    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)\n"
        "    output = coefficients[:, 0] * blocks\n"
        "    if offsets is not None:\n"
        "        position = offsets[: hidden_states.shape[0]].to(torch.int64)\n"
        "    else:\n"
        "        position = torch.arange(\n"
        "            hidden_states.shape[0], device=hidden_states.device\n"
        "        )\n"
        "        if block_size & (block_size - 1) == 0:\n"
        "            position = position & (block_size - 1)\n"
        "        else:\n"
        "            position = position % block_size\n",
    ),
    (
        "        self.block_size = block_size\n"
        "        self.taps = taps\n"
        "        self.group_size = group_size\n"
        "        self.num_groups = hidden_size // group_size\n",
        "        self.block_size = block_size\n"
        "        self.taps = taps\n"
        "        self.group_size = group_size\n"
        "        self.num_groups = hidden_size // group_size\n"
        f"        # {MARKER}: each row's offset in its request's query block, bound by the speculator (DFlash2's\n"
        "        # load_draft_model) when the block width varies per step; None convolves by row % block_size.\n"
        '        self.register_buffer("block_offsets", None, persistent=False)\n',
    ),
    (
        "    def _convolve(\n"
        "        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int\n"
        "    ) -> torch.Tensor:\n"
        "        return _grouped_conv(\n"
        "            hidden_states,\n"
        "            delta,\n"
        "            self.base_kernel[side],\n"
        "            self.block_size,\n"
        "            self.num_groups,\n"
        "            self.group_size,\n"
        "            self.taps,\n"
        "        )\n",
        "    def _convolve(\n"
        "        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int\n"
        "    ) -> torch.Tensor:\n"
        "        offsets = self.block_offsets\n"
        "        if offsets is not None:\n"
        "            offsets = offsets[: hidden_states.shape[0]]\n"
        "        return _grouped_conv(\n"
        "            hidden_states,\n"
        "            delta,\n"
        "            self.base_kernel[side],\n"
        "            self.block_size,\n"
        "            self.num_groups,\n"
        "            self.group_size,\n"
        "            self.taps,\n"
        "            offsets,\n"
        "        )\n",
    ),
]

FILES = {
    "v1/worker/gpu/spec_decode/dflash/cudagraph.py": CUDAGRAPH,
    "v1/worker/gpu/spec_decode/dflash/speculator.py": DFLASH,
    "v1/worker/gpu/spec_decode/dflash2/speculator.py": DFLASH2,
    "model_executor/models/qwen3_dflash2.py": QWEN3_DFLASH2,
}


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
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    outputs: dict[Path, str] = {}
    for rel, edits in FILES.items():
        path = args.root / rel
        src = path.read_text()
        if MARKER in src:
            print(f"{path}: already patched")
            continue
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
