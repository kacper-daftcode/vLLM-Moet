# DeepSeek-V4.1-Flash on sm_120: patchers, kernels, tests

What `Dockerfile.sm120-dsv41-nightly` installs into the official vLLM nightly image, by step. Each
patcher is anchored on the installed file's text and refuses any other version; each kernel is a
CUDA C++ `torch` extension compiled at image build (`TORCH_EXTENSIONS_DIR=/opt/vllm-moet/torch_extensions`)
with a test against the path it replaces. User-facing description of every step and its runtime
switch: [`docs/dsv41-sm120.md`](../../docs/dsv41-sm120.md); the reasoning and measurements:
[`docs/dsv41-sm120-internals.md`](../../docs/dsv41-sm120-internals.md).

| step | patcher | kernel / module | test (one GPU inside the image unless noted) |
|---|---|---|---|
| 1 | `patch_deepgemm.py` (DeepGEMM host asserts: 128-row paged MQA pages, SM120 per-group BLOCK_M) | DeepGEMM `_C` rebuilt in the builder stage | `test_deepgemm_sm120_paged_mqa.py --packed-stride` |
| 2 | `flashinfer_interleaved/patch_flashinfer_dsv41_interleaved.py` (FlashInfer's DSv4.1 reader learns vLLM's interleaved V4.1 records) | FlashInfer's `sparse_mla_sm120` rebuilt into its AOT cache | `flashinfer_interleaved/test_dsv41_interleaved_sm120.py`, `test_vllm_writers_interleaved.py` |
| 3 | `patch_vllm_mxfp8_gemv.py`, `patch_vllm_wo_a_sm120.py`, `patch_vllm_moe_glue_sm120.py` | `sm120_gemv/` — the MXFP8 tensor-core GEMV (dense decode GEMMs, `wo_a`, grouped) | `sm120_gemv/test_mxfp8_gemv_sm120.py --ms 1,6,16 --shapes decode`, `test_wo_a_gemv_sm120.py`, `test_gemv_pdl_sm120.py`, `test_gemv_v3_rounds_sm120.py`, `test_vllm_integration.py`, `test_wo_a_integration.py`, `test_dequant_fused_sm120.py` (step 3b: the prefill dequant of `wo_a` as one kernel behind `VLLM_MOET_WOA_DEQUANT_FUSED=1` with its scale-code-0 guard, `_CACHE=1`, and the runtime A/B key `WOA_DEQUANT_FUSED` of `/dev/shm/vllm_moet_ab.json`) |
| 4 | `patch_vllm_reasoning_effort.py` | — | `test_reasoning_effort_encoding.py --model-dir /model` (no GPU) |
| 5 | `nvfp4_kv/patch_vllm_nvfp4_sm120_upstream.py`, `patch_vllm_indexer_fp4_sm120.py` (the FP4 KV record and the MXFP4 indexer cache admitted on sm_120) | FlashInfer's DSv4.1 dual cache | `nvfp4_kv/test_flashinfer_dsv41_fp4_extra.py --quick --no-perf`, `test_indexer_fp4_sm120.py` |
| 6 | `patch_vllm_offload_replicated_mla.py` (one host copy of the TP-replicated KV) | — | `test_offload_replicated_mla.py` |
| 8 | `patch_vllm_moe_quant_scatter_sm120.py` | `moe_quant_scatter/` | `moe_quant_scatter/test_moe_quant_scatter_sm120.py`, `test_moe_quant_scatter_integration.py` |
| 10 | `patch_vllm_mhc_overlap_sm120.py`, `patch_vllm_dspark_mhc_stream_sm120.py` | `mhc_overlap/` — the mHC boundary kernel (cluster of five CTAs per token) and its side stream | `mhc_overlap/test_mhc_post_norm_sm120.py`, `test_mhc_post_norm_ar_sm120.py`, `test_mhc_overlap_integration.py`, `test_dspark_mhc_stream.py --bench` |
| 11 | `patch_vllm_bf16_gemm_sm120.py` | `bf16_gemm/` — small-M BF16 GEMMs with cuBLAS's bits | `bf16_gemm/test_bf16_replica_gemm_sm120.py`, `test_bf16_spread_topk_sm120.py`, `test_bf16_gemm_integration.py` |
| 12 | `patch_vllm_engram_wkv_tp.py` | — | `test_engram_wkv_tp.py`; four GPUs: `torchrun --nproc_per_node=4 test_engram_wkv_tp4.py --checkpoint /model` |
| 13 | `nccl_graph/patch_vllm_nccl_graph_records.py` | `nccl_graph/nccl_graph_records.py` | four GPUs: `torchrun --nproc_per_node=4 nccl_graph/test_nccl_graph_records_tp4.py`, `test_nccl_graph_mixing_tp4.py` |
| 14 | `patch_vllm_moe_fc2_discard_sm120.py` | `moe_l2/` — FC2 padding rows out of L2 | `moe_l2/test_moe_l2_discard_sm120.py` |
| 15 | `patch_vllm_prefill_dbo_sm120.py` (build arg 0 by default) | `prefill_dbo/` | four GPUs: `prefill_dbo/test_prefill_dbo_tp4.py` |
| 16 | `patch_vllm_indexer_shard_rows.py` (vllm#61647 backport) | — | — |
| 17 | `patch_flashinfer_b12x_compact_bind.py` | b12x's compact N64 launchers bound at TP4 | `test_b12x_moe_compact_bind.py` |
| 18 | `patch_vllm_dspark_lm_head_mxfp8_sm120.py` | step 3's GEMV | `sm120_gemv/test_vllm_integration.py` |
| 19 | `patch_vllm_dsv4_topk_sm120.py` | `dsv4_topk/` — the router's top-6 | `dsv4_topk/test_dsv4_topk_sm120.py` |
| 20 | `patch_vllm_dspark_fused_markov_sm120.py` | Triton | `fused_markov/test_fused_markov_sm120.py` |
| 21 | `patch_vllm_target_lm_head_mxfp8_sm120.py` | step 3's GEMV | `sm120_gemv/test_vllm_integration.py` |
| 22 | `patch_flashinfer_b12x_fc2_grouped_sm120.py` | `b12x_fc2/` | `b12x_fc2/test_b12x_fc2_grouped_sm120.py` |
| 23 | `patch_vllm_p2p_allreduce_sm120.py` | `p2p_ar/` — the two-shot P2P all-reduce in NCCL's order | four GPUs: `torchrun --nproc_per_node=4 p2p_ar/test_p2p_allreduce_sm120.py` |
| 26 | `patch_vllm_det_topk_sm120.py` | `det_topk/` — the deterministic indexer top-k | `det_topk/test_det_topk_sm120.py` |
| 27 | `patch_flashinfer_b12x_moe_fused_sm120.py` | `b12x_moe_fused/` — the decode MoE in one persistent kernel | `b12x_moe_fused/test_b12x_moe_fused_sm120.py` |
| 28 | `patch_flashinfer_sparse_mla_ldg_sm120.py` | `sparse_mla_ldg/` — the kernel text is generated from the installed FlashInfer header (`gen_spec_kernel.py`) | `sparse_mla_ldg/test_sparse_mla_ldg_sm120.py` |
| 30 | `patch_vllm_dflash_prepare_groups_sm120.py` | Triton | `dflash_prep_groups/test_dflash_prep_groups_sm120.py` |
| 31 | `patch_vllm_candidate_short_sm120.py` | step 26's kernel | `candidate_short/test_candidate_short_sm120.py` |
| 32 | `patch_vllm_swa_decode_build_sm120.py` | Triton | `swa_decode_build/test_swa_decode_build_sm120.py` |
| 33 | `patch_vllm_moe_topksum_add_sm120.py` | Triton | `moe_topksum_add/test_moe_topksum_add_sm120.py` |
| 34 | `patch_vllm_moe_prequant_sm120.py` | — | end-to-end greedy agreement |
| 36 | `patch_vllm_swa_draft_build_sm120.py` | Triton | `swa_draft_build/test_swa_draft_build_sm120.py` |
| 37 | `patch_flashinfer_b12x_moe_mega_sm120.py` (build arg 0: not in the image) | `b12x_moe_mega/` | `b12x_moe_mega/test_b12x_moe_mega_sm120.py` |
| 38 | `patch_vllm_mhc_mxfp8_consumer_sm120.py` | step 10's kernel | `mhc_overlap/test_mhc_post_norm_mxfp8_sm120.py` |
| 39 | `patch_vllm_mhc_side_defer_sm120.py` | — | end-to-end greedy agreement |
| 40 | `patch_vllm_sparse_mla_prefill_bf16_sm120.py` | FlashInfer's full-BF16 SM120 prefill kernel | GSM8K / needle / greedy agreement (a change of the prefill's arithmetic) |
| 41 | `patch_vllm_prefill_sp_sm120.py` | `p2p_rs/` (optional `_RS=p2p`: the reduce-scatter in the all-reduce's order) | four GPUs: `torchrun --nproc_per_node=4 sp_boundary/test_prefill_sp_tp4.py`, `test_sp_boundary_tp4.py`, `p2p_rs/test_p2p_reduce_scatter_sm120.py` |
| 42 | `patch_vllm_b12x_moe_prefill_fast_sm120.py` (runtime: `_WS1=9` FC1 v10, `_WS2=3` FC2 v9, `_SUM_ITEMS=1` + `_SUM_DISCARD=1 _SUM_ROUTE_HINT=1 _SUM_TOKENS=128 _SUM_MIN_ROWS=512` the top-k sum as FC2 work items; `_MT_LOOPS=1` (off by default) the FC1 / FC2 consumers' stage loop specialised on the item's m-tiles; `VLLM_MOET_RUNTIME_AB=1` + `/dev/shm/vllm_moet_ab.json` flips `B12X_MOE_PREFILL_FAST_SUM_ITEMS` / `_WS1` / `_MT_LOOPS` without a restart; `_STATS_EVERY=N` / `touch /dev/shm/vllm_moet_stats` print the activation counters per rank) | `b12x_moe_prefill_fast/` — the prefill MoE FC1 / FC2 kernels; `b12x_moe_prefill_spec/` is the plain-CUDA specification they are tested against | `b12x_moe_prefill_fast/test_b12x_moe_prefill_fast_sm120.py --dumps DIR` (dumps of a real prefill from `patch_vllm_b12x_moe_dump_sm120.py`), `b12x_moe_prefill_fast/test_occupancy_guard_sm120.py --dumps DIR --case served|warn|strict` (the WS kernels' CTAs per SM and registers as launched, the one-time warning when a kernel starts with fewer CTAs per SM than its mode was measured with, the refusal under `_OCC_STRICT=1`), `b12x_moe_prefill_fast/test_mt_loops_sm120.py --dumps DIR --case runtime|env` (the m-tile loops: identical to b12x with the switch off, on, flipped on every call and dropped; the kernels and CTAs per SM as launched; the counters), `b12x_moe_prefill_spec/test_b12x_moe_prefill_spec_sm120.py` |
| 45 | `patch_vllm_prefill_sp_gather_sm120.py` (`VLLM_MOET_PREFILL_SP_GATHER=mxfp8`, default bf16) | `sp_moe_gather/` — the sequence-parallel MoE input all-gathered as e4m3 + UE8M0 scales + the shard's top-6 (0.52× the bytes); needs step 42's kernels with the pre-quantized-rows entry (Phase 1a v8, `468f9ad`) | four GPUs: `torchrun --nproc_per_node=4 sp_moe_gather/test_prefill_sp_gather_tp4.py --dumps DIR` (quantizer identity, router GEMM / top-6 on the shard, boundary times); in situ: bit-identical, fresh prefill −3.8 % at 163K on 4× RTX PRO 6000 SE |
| 46 | `patch_vllm_sparse_mla_prefill_sass_sm120.py` (`VLLM_MOET_SPARSE_MLA_PREFILL_SASS=1`, image default off; the launcher's served configuration sets it; `VLLM_MOET_RUNTIME_AB=1` + `VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN_B=<path>`: the key `SPARSE_MLA_PREFILL_SASS_CUBIN_SEL` of `/dev/shm/vllm_moet_ab.json` selects between two cubins of the kernel per launch, without a restart) | `sparse_mla_sass/` — the step-40 prefill attention chunks (> 64 rows) on the cubit-assembled `kernels/cubins-sm120/sparse_mla_prefill_v41.cubin` (`sass_launcher.py`; the SASS source `kernels/sass/sparse_mla_prefill_v41.sass`, generator `kernels/gen/gen_sparse_mla_prefill_v41.py`, `build_sparse_mla_sass.sh` assembles it with cubit and checks the sha256); FlashInfer's full-BF16 kernel's arithmetic hand-scheduled, 0.66–0.72× its time on the harness | one GPU inside the image: `sparse_mla_sass/test_sparse_mla_sass_sm120.py --cubin X.cubin --synthetic [--time] [dumps]` (`torch.equal` on output + lse against spec B: 8 synthetic cases and dumps of a real prefill from `patch_vllm_sparse_mla_prefill_dump_sm120.py` + `replay_specB_dumps.py`); the runtime A/B key: `sparse_mla_sass/test_sparse_mla_sass_runtime_ab.py` (no GPU: the control file's semantics on a stub launcher; `--gpu --cubin A --cubin-b B [dumps]`: every state and transition with the real cubins); in situ: long_agree / agree ALL IDENTICAL, fresh prefill 163K −3.8…−5.2 % |
| 47 | `patch_vllm_prefill_pipe_sm120.py` (`VLLM_MOET_PREFILL_PIPE=2` together with `--max-num-batched-tokens 8192`, default 0 = the one-group path; `VLLM_MOET_PREFILL_PIPE_REPLAY_GRAPHS=1` with `--compilation-config decoder_replay_cudagraph_capture_sizes [128,256]` = the groups' 128-row replay layers as CUDA graphs per slot, the -r26b hardening against host CPU load: 163K under 100 nice'd CPU hogs +3.3 % instead of +26.6 %, quiet +0.6 %, −1 GiB KV; the other switches of the module's docstring — `_COMM2`, `_ORDER`, `_AFFINITY`, `_KV_CONNECTOR`, `_TAIL`, `_FORCE_SEQUENTIAL`, `_DUMP` — are measured opt-ins and bits oracles; the module's docstrings record each measurement) | `sp_pipeline/` — the pipelined sequence-parallel prefill: an eager step of > 4 096 tokens runs as two 4 096-row groups of the same request on two threads that hand the GPU to each other at the SP boundaries (the DBO protocol): a group's RS → mHC → shard GEMMs → AG chain on a comm stream under the other group's attention / MoE on the one compute stream, group B's layer L after group A's wrote its KV (stream order, asserted); metadata builders, replay batch and the indexer's row buffers per group; needs steps 41 / 45 | one GPU: `sp_pipeline/test_prefill_pipe_unit.py` (the handoff protocol, 21 checks); in situ (4× RTX PRO 6000 SE, TP4): `sp_pipeline/dump_prefill.py` + `compare_dumps.py` — the sha256 of every hc state after every layer of every group, PIPE=2 at 8192 vs PIPE=0 at 4096: 800/800 identical on 4 ranks; long_agree / agree ALL IDENTICAL; fresh prefill 163K 6947 → 5360 ms (−23 %), 19K 819 → 671 |
| 48 | `kv_offload/patch_vllm_kv_swap_async_sm120.py` (`VLLM_MOET_KV_SWAP_ASYNC=1`, default off) | `kv_offload/kv_swap_async.py` — the KV offload connector's host registration under a cross-process lock in rank order (chunks, retries) and a pageable rank's copies through pinned staging on the connector's own threads | CPU: `kv_offload/test_kv_swap_async.py` (37 checks) |
| 49 | `gc_idle_freeze/patch_vllm_gc_idle_freeze_sm120.py` (`VLLM_MOET_GC_IDLE_FREEZE=1`, default off; the slow-collection log line, `VLLM_MOET_GC_WARN_MS=50`, is on either way) | `gc_idle_freeze/moet_gc_idle_freeze.py` — installed as `vllm.utils.moet_gc_idle_freeze` and called by vLLM's `freeze_gc_heap()` in the API server, the EngineCore and every worker: one more collection right after the start-up freeze, then `gc.collect()` + `gc.freeze()` whenever the process has worked and gone idle, so that a full garbage collection walks only what the last burst of requests left | CPU: `gc_idle_freeze/test_gc_idle_freeze.py` (28 checks; 29 inside the image, where the installed file is checked too) |

In the tree, not yet a Dockerfile step (off by default): `patch_vllm_prefill_device_comm_sm120.py` +
`sp_pipeline/moet_device_comm.py` / `moet_device_comm_sm120.cu` (`VLLM_MOET_PREFILL_PIPE_DEVICE_COMM=0|1|verify`: the
pipelined step's reduce-scatters / all-gathers from the device over PCIe P2P in NCCL's summation order —
`sp_pipeline/test_device_comm_unit.py`, CPU, 9 checks; the kernel's 4-rank GPU test is not in the tree yet).

Also off by default, inside step 47's module: `VLLM_MOET_PREFILL_PIPE_GREEN=N` (`sp_pipeline/moet_prefill_pipe.py`,
`green_sms`) — a static split of the GPU's SMs for the pipelined step: its comm stream (NCCL, the mHC kernels, the shard
GEMMs) on a CUDA green context of N SMs, its compute stream and the attention's aux streams on the remaining ones, the
fast prefill MoE's grids capped to the partition, the replay tails on the caller's stream with today's graphs
(`_GREEN_TAIL=compute`: on the compute partition, their graphs captured there), one warm-up step at engine start, and
with `VLLM_MOET_RUNTIME_AB=1` the control-file key `PREFILL_PIPE_GREEN` (0 | N) per forward; the same kernels in the same
order, so the bits are today's. `sp_pipeline/test_prefill_pipe_green_unit.py`: CPU, 31 checks (the control-file reader
and its line formats, the grid caps, the refused configurations); one GPU, 21 checks (the handoff protocol on the
split, the key between forwards, the tails in both modes, graphs captured on the partition; with `--smid-lib` also
which SMs the kernels ran on); `torchrun` × 4 with `--tp`, 15 checks (NCCL's reduce-scatter / all-gather in the comm
phases, a TP all-reduce in a captured graph: split against unsplit on every rank). The time of a served prefill with
it is not measured yet.

Not in the image: `patch_vllm_mxfp8_gemm_b12x_sm120.py` + `mxfp8_gemm_b12x/` (the dense GEMMs on
b12x's block-scaled GEMM: −28 % in the harness, +2.3 % end to end — a negative result),
`patch_vllm_moe_act_skip_padding_sm120.py` (GLM-5.3-Flash line, unmeasured),
`patch_vllm_p2p_allreduce_comm_sm120.py` (the P2P all-reduce as a communicator hook, for other row
widths), `patch_vllm_indexer_sm120.py` (varlen indexer: a net loss), `mhc_pre_norm/`,
`moe_gate_topk/`, `mini_model/` (1/5-scale configs for TP4 tests without the checkpoint). Legacy
of the superseded recipe image (`vllm/vllm-openai:deepseekv41-flash-0909`):
`patch_flashinfer.py` + `sparse_mla_sm120_dsv41.cu` + `test_sparse_mla_sm120_dsv41.py`,
`patch_vllm_decoder_replay.py` + `decoder_replay/`, `patch_vllm_query_quant_gate.py`,
`nvfp4_kv/patch_vllm_packed_kv_sm120.py` and its kernels — vLLM nightly carries their function.

## Running the tests

```bash
IMAGE=vllm-moet-sm120:dsv41-nightly-df3bd094-r23
docker run --rm --gpus '"device=0"' --ipc host --entrypoint bash $IMAGE -c \
  'cd /opt/vllm-moet/dsv41_sm120 && python3 sm120_gemv/test_mxfp8_gemv_sm120.py --ms 1,6,16 --shapes decode \
   && python3 dsv4_topk/test_dsv4_topk_sm120.py && python3 det_topk/test_det_topk_sm120.py \
   && python3 b12x_moe_fused/test_b12x_moe_fused_sm120.py && python3 sparse_mla_ldg/test_sparse_mla_ldg_sm120.py'
docker run --rm --gpus '"device=0,1,2,3"' --ipc host --entrypoint bash $IMAGE -c \
  'cd /opt/vllm-moet/dsv41_sm120 && torchrun --nproc_per_node=4 p2p_ar/test_p2p_allreduce_sm120.py \
   && torchrun --nproc_per_node=4 sp_boundary/test_prefill_sp_tp4.py'
python3 tools/dsv41_sm120/cubit_gate.py $IMAGE     # every cubin the image ships round-trips through cubit
```

Tests that need the checkpoint mount it at `/model` (`-v /srv/models/DeepSeek-V4.1-Flash-1017:/model:ro`);
next to a serving job use `--max-gpu-mem-gib 2.2` for the Engram tests and smaller `--experts` counts for
the MoE tests.
