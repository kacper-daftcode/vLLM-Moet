# DeepSeek-V4.1-Flash on SM120 — kernel gap closure

Kernel-side patches that let the official `vllm/vllm-openai:deepseekv41-flash-0909` image serve
DeepSeek-V4.1-Flash on RTX PRO 6000 / RTX 5090 (sm_120). Port notes, gap inventory and
validation evidence: `docs/dsv41-sm120-port.md`; images: `Dockerfile.sm120-dsv41` (served, vllm-openai:deepseekv41-flash-0909 base) and `Dockerfile.sm120-dsv41-nightly` (vLLM main + FlashInfer nightly, under validation since 2026-09-22).

| file | role |
|---|---|
| `sparse_mla_sm120_dsv41.cu` | new FlashInfer JIT TU: SM120 sparse-MLA instantiations for the V4.1 geometry (SWA page 32, compressed page 128/64, SWA rows 128/192/1152) + `sparse_mla_prefill_dispatch_dsv41` |
| `patch_flashinfer.py` | idempotent, anchored patcher for an installed flashinfer package (installs the TU, orchestrator hook, PBS=32 decode table, python dispatch) |
| `patch_deepgemm.py` | DeepGEMM host asserts: SM120 FP8 **and MXFP4** paged MQA logits on 128-row pages (the device kernels are templated on the page size; only the launchers' asserts stopped at 64). Regex anchors since 2026-09-22, so one patcher fits the 0909 image's pin (`deepseek-ai/DeepGEMM 8b1392b9`) and vLLM main's (`vllm-project/DeepGEMM e1f418c2`, `Dockerfile.sm120-dsv41-nightly`). On the vllm-project pin it also restores nv_dev's **SM120 per-group BLOCK_M** for the contiguous grouped layout (`get_theoretical_mk_alignment_for_contiguous_layout(expected_m, num_groups)` → 64 when the per-expert M is ≤ 64): the port of the SM120 kernels kept upstream's SM100-only heuristic, vLLM's MoE glue got 128 and padded every routed expert to 128 rows — the whole DeepGEMM MoE chain took 2× at decode (RTX 5090: 98 → 186 µs at 6 tokens, 576 → 1101 µs at 48; `tools/sm120_perf/moe_backends_bench.py`); with the patch the chain is back to the 0909 image's numbers |
| `nvfp4_kv/patch_vllm_nvfp4_sm120_upstream.py` | **vLLM main only** (`Dockerfile.sm120-dsv41-nightly`): our upstream PR as a patcher — `--kv-cache-dtype nvfp4_ds_mla` on sm_120 through FlashInfer's DSv4.1 dual-cache sparse MLA (`kv_cache_format="fp8_dsv41_fp4_ca"`, [flashinfer#5197](https://github.com/flashinfer-ai/flashinfer/pull/5197)): capability probe in `utils/flashinfer.py`, the V4.1 records (528 B SWA + 288 B compressed) selected only for `nvfp4_ds_mla`, the format keyword on both kernel calls. Replaces the scratch/pool of `patch_vllm_packed_kv_sm120.py` and the TU/hook of `patch_flashinfer.py` on that line (FlashInfer nightly dispatches page sizes at runtime) |
| `test_sparse_mla_sm120_dsv41.py` | op-level validation: torch reference + bit-exact re-paging parity vs stock PBS=64 kernels |
| `test_deepgemm_sm120_paged_mqa.py` | op-level validation: DeepGEMM reference + bit-exact parity block_kv 64 vs 128; native next_n 1/2/6 and the varlen (`indices=`) mode; `--fmt fp8|mxfp4|both` (default both), `--packed-stride` lays the pages out with vLLM's block-outermost stride/offset for this model (230400 B / 210240 B blocks) |
| `nvfp4_kv/` | the compressed (main) KV in the checkpoint's FP4 record on sm_120 (`KV_RECORD=nvfp4`, default since 2026-09-21): `fp4_kv_quant.py` (bit-exact port of `inference/kernel.py::fp4_act_quant(x, 16, e4m3)`), `nvfp4_kv_kernels.py` (insert into the 288-B / 528-B records, gather + re-quantize into an `fp8_ds_mla` scratch for decode, whole-context dequant into a pool for prefill), `patch_vllm_packed_kv_sm120.py` (installs the merged module into vLLM and patches the spec, the insert dispatch and the SM120 attention's decode / prefill / workspace reservation; inert unless `VLLM_MOET_KV_RECORD` selects a packed record), `patch_vllm_kv_fp4_fake.py` (experiment "A0": the same numerics with today's storage, not applied in the image); tests `test_fp4_kv_quant.py` (vs the checkpoint's TileLang quantizer), `test_kv_fp4_fake_insert.py`, `test_nvfp4_kv_kernels.py` (records vs the reference, scratch == A0 records, FlashInfer dual-cache attention bit-exact on the scratch), `test_packed_kv_glue.py` (sharing across layers, pool layout, fallback) — all need `--model-dir` (the checkpoint's `inference/` for the reference) except the glue test; `test_flashinfer_dsv41_fp4_extra.py` runs FlashInfer `main`'s own reader of this record ([#5197](https://github.com/flashinfer-ai/flashinfer/pull/5197): V4.1 dual cache, `kv_cache_format="fp8_dsv41_fp4_ca"`) on vLLM's V4.1 geometry — our writers vs upstream's byte for byte, attention vs the fp32 reference and vs today's scratch path, CUDA‑graph timings of both paths (needs flashinfer ≥ 0.7.0 built from `main` ≥ eb5f05be; no checkpoint) |
| `patch_vllm_indexer_fp4_sm120.py`, `test_indexer_fp4_sm120.py` | vLLM patch (one gate in `v1/attention/backends/mla/indexer.py`): `--attention-config '{"indexer_kv_dtype":"mxfp4"}'` is accepted on sm_120 (launcher `INDEXER_KV_DTYPE=mxfp4`). The MXFP4 indexer cache stores 64 B of e2m1 pairs + 4 UE8M0 scales per key (68 B instead of 132), the format the indexer was trained with; the packed KV block shrinks 230400 -> 210240 B (+9.6 % KV tokens). The test checks the Q quantizer (CuTe DSL), the K store (Triton, `cvt.rn.satfinite.e2m1x2.f32` on sm_120a) bit-for-bit against DeepSeek's fp4 quantizer (RNE, UE8M0 = 2^ceil(log2(amax/6))), and DeepGEMM's logits on the kernel-written 128-key pages (bit-exact 64/128 re-page) |
| `sm120_gemv/mxfp8_gemv_sm120.cu` | tensor-core (mma.m16n8k32 e4m3) MXFP8 GEMV for decode shapes (M <= 16), F8_128x4 swizzled scales, plus a scalar fallback (`VLLM_MOET_GEMV_IMPL=scalar`) and the v2 experiment (`=v2`, see below; not faster); loaded via `sm120_gemv/mxfp8_gemv_sm120.py` (torch cpp_extension JIT, precompiled in the image) |
| `patch_vllm_mxfp8_gemv.py` | vLLM patch: `FlashInferCutlassMxfp8LinearKernel.apply_weights` routes M <= 16 to the GEMV (`VLLM_MOET_SM120_GEMV=0` reverts) |
| `sm120_gemv/vllm_sm120_gemv_bmm.py`, `patch_vllm_wo_a_sm120.py` | `Sm120GemvMxfp8BmmLinearKernel`: keeps the grouped o-projection `wo_a` (`is_bmm`, 2 head groups × [1024 <- 4096] per TP4 rank) in MXFP8 and runs decode batches (<= 64 tokens) on `mxfp8_gemv_grouped` with the FP8 activations + packed MN-major scales that `fused_inv_rope_fp8_quant` produces for the sm_100 DeepGEMM path; prefill dequantizes the weight on the fly and keeps the bf16 bmm. The patcher installs the module into vLLM, puts the kernel first in `init_mxfp8_linear_kernel()`'s BMM list and adds the dispatch to `deep_gemm_fp8_o_proj` (`VLLM_MOET_SM120_GEMV_BMM=0` reverts to the BF16 emulation) |
| `sm120_gemv/test_mxfp8_gemv_sm120.py`, `sm120_gemv/test_vllm_integration.py` | GEMV validation vs FlashInfer CUTLASS `mm_mxfp8` and an fp32 reference (cold-L2 timing on the served shapes); dispatch check of the patched vLLM kernel class |
| `sm120_gemv/test_wo_a_gemv_sm120.py`, `sm120_gemv/test_wo_a_integration.py` | grouped GEMV vs fp32 reference on the fp8 operands and vs the BF16 bmm path (cold-L2 timing, T = 1..64); end-to-end check of the patched kernel selection + `deep_gemm_fp8_o_proj` dispatch (GEMV for T <= 64, bit-identical bf16 fallback above) |
| `patch_vllm_indexer_sm120.py` | **not applied in the image**: lets the DSA indexer use DeepGEMM's varlen / native multi-row paged MQA on sm_120 (prerequisite for DSpark adaptive verification). Validated at op level, but the varlen decode path cost ~0.5 ms/step and adaptive verification was a net loss on 2026-09-18 (docs/dsv41-sm120-port.md) |
| `patch_vllm_moe_glue_sm120.py` | vLLM patch (three files, bit-exact data movement): DeepGEMM MoE glue at decode token counts — `_fwd_kernel_ep_scatter_2` one program per (token, expert) pair instead of per token (6 dependent atomics → 1), `_fwd_kernel_ep_gather` top-k loop unrolled (`tl.static_range`, same fp32 order), `silu_mul_quant_fp8_packed_triton(m_indices=)` skips the 63 padding rows per expert that DeepGEMM's 64-row contiguous layout adds (2304 rows, 36 real). Per layer in `tools/sm120_perf/dg_moe_blockm_bench.py`: permute 11.8 → 8.5 µs, act+quant 4.6 → 3.9, gather 4.6 → 4.0 (MoE output 0 of 30720 elements differ). Applied at image build (`Dockerfile.sm120-dsv41` step 3) since 2026-09-19; was first deployed as bind-mounts over the previous image |
| `patch_vllm_reasoning_effort.py`, `test_reasoning_effort_encoding.py` | vLLM patch (two files in `vllm/tokenizers/`): the `deepseek_v41` prompt encoder's reasoning-effort tiers follow the released checkpoint — `low 50 / high 75 / max 100`, default `high` — instead of the pre-release table vLLM vendored (`low 25 / high 50 / xhigh 75 / max 100`, still on vLLM main), so a thinking-mode request without an explicit effort renders `Reasoning Effort: 75` like DeepSeek's `encoding/encoding.py` and the tech report's API tiers, not 50 (the tier DeepSeek calls "low"). OpenAI's `minimal` / `medium` / `xhigh` are accepted as interpolated budgets 25 / 62 / 87 instead of failing with HTTP 400; an integer in `chat_template_kwargs` bypasses the table. The test runs at image build (tiers + default) and, with `--model-dir`, requires byte-identical prompts to the checkpoint's encoder on its `encoding/tests` goldens and a 90-case thinking-mode × effort × tools × multi-turn matrix (all identical on 2026-09-20; before the patch only the budget line differed). Not a numerics change; the rest of the prompt grammar was already identical |
| `patch_vllm_offload_replicated_mla.py`, `test_offload_replicated_mla.py` | vLLM patch (one gate in `distributed/kv_transfer/kv_connector/v1/offloading/config.py`, vLLM-main image only): the KV offload keeps **one host copy** of the TP-replicated V4.1 KV instead of one per rank. vLLM's `replicated_layout` (one slot in the shared `/dev/shm` region, rank 0 stores, every rank loads it; the disk tier names it TP-independently) admits only a single bare `MLAAttentionSpec` group; V4.1 has ten KV groups, of which only the compressed one (MLA latent + DSA indexer K) is offloaded, so the patch checks the offloaded groups (every layer an MLA spec with one KV head). vllm#57652 widens the gate to multi-group MLA but still counts V4.1's compressor ring (`CircularBufferSpec`); when that PR is in the file, the patcher narrows its call to the offloaded groups. 18,219 of 18,219 disk block files of the per-rank layout had four byte-identical rank slots (`tools/sm120_perf/tp_copies_check.py`); served: 896 B instead of 3.58 KB of host RAM and disk per token (64 GiB = 76.7M tokens), a 221K-token context back from disk in 1.6 s instead of 2.5 s. The test (no GPU; runs at image build) builds the served KV layout and runs vLLM's own `build_offloading_config`, `TieringOffloadingSpec` sizing and fs-tier `FileMapper` on it: the gate on at TP 2/4/8 and off for non-MLA / HiddenState layers, GQA groups, TP1, PP2, ray; 599,186 instead of 149,796 chunks in 64 GiB; a new TP-independent disk namespace |
| `patch_vllm_decoder_replay.py`, `decoder_replay/` | vLLM patch, **served since 2026-09-23** (`Dockerfile.sm120-dsv41-nightly` step 7, `-ced` tag; `--build-arg VLLM_PR_58132=0` leaves it out): the decoder side of DeepSeek-V4.1's SWA bounded replay — the report's CED prefill, layers 21-39 on each request's last 128 tokens only — from vllm#58132 (head 9a86c2c9) as a GNU patch on the installed package (`decoder_replay/vllm-pr58132-9a86c2c9.patch`; base 496c6472, nothing under `vllm/models/deepseek_v41/` changed since our pin, no overlap with our patchers) plus a kill switch, `VLLM_MOET_DECODER_REPLAY=0`. The PR's unit tests are copied to `decoder_replay/tests/models/` in the image (`cd /opt/vllm-moet/dsv41_sm120/decoder_replay && python3 -m pytest tests/models -q --noconftest`, one GPU; 11/11 on the RTX 5090). Served configuration: fresh prefill 1.63× at 19K tokens, 1.68× at 163K, 1.75× at 391K, quality gate passed, decode unchanged (`docs/dsv41-sm120-port.md`, "Decoder SWA bounded replay") |
| `patch_vllm_moe_quant_scatter_sm120.py`, `moe_quant_scatter/` | vLLM patch (one file, `experts/deep_gemm_moe.py`; `Dockerfile.sm120-dsv41-nightly` step 8, `-moeqs` tag), bit-exact: the MoE input quantization fused with the permutation into DeepGEMM's contiguous grouped layout at decode token counts. vLLM ran seven kernels per MoE layer between the router and FC1 (`per_token_group_quant_8bit` in `prepare()`, two fills, `_count_expert_num_tokens`, `_fwd_kernel_ep_scatter_1/2`, DeepGEMM's `transpose_and_pack_fp32_into_ue8m0` inside the FC1 call); `moe_quant_scatter_sm120.cu` is one launch: one CTA per (token, expert) pair, two barriers, the routing table rebuilt in shared memory (deterministic slots), vLLM's exact scale math (no fast-math), the UE8M0 bytes written straight into DeepGEMM's packed int32 MN-major scale tensor, `m_indices` by each expert's rank-0 pair. `DeepGemmFP4Experts.expects_unquantized_inputs` hands the bf16 rows to `apply()`, which fuses ≤ 1024 pairs (TP without EP, UE8M0) and otherwise runs vLLM's `per_token_group_quant_fp8` + the unchanged permute (prefill: the same kernels as before). `VLLM_MOET_MOE_QUANT_SCATTER=0` restores the quantization in `prepare()`. Tests (one sm_120 GPU): `moe_quant_scatter/test_moe_quant_scatter_sm120.py` — per-pair fp8 rows and scale bytes vs vLLM's kernels, identical `m_indices`, the hardware e4m3 conversion vs c10's on every finite bf16 value × 81 scale exponents, the full DeepGEMM chain bit-identical, `--bench [--profile]` for per-stage CUDA-graph timings; `moe_quant_scatter/test_moe_quant_scatter_integration.py` — the patched class driven like `FusedMoEModularKernel` (fused / deferred fallback / switch off, identical outputs). RTX 5090, per layer in a graph: quant + permute (+ DeepGEMM's pack) 7.9 + 1.6 → 3.8 µs at 6 tokens, 10.0 + 4.3 → 6.8 at 64 (`docs/dsv41-sm120-port.md`, "MoE glue in one launch") |
| `patch_vllm_query_quant_gate.py`, `test_query_quant_gate.py` | vLLM patch (one file, `models/deepseek_v41/common/ops/query_quant.py`; `Dockerfile.sm120-dsv41-nightly` step 9, `--build-arg VLLM_PR_57679=1`, **not served yet**): upstream vllm#57679. vLLM's fused q/kv RMSNorm + MXFP8 quantization of the query (`fused_q_kv_rmsnorm_quant`, one Triton launch, the QuantizedActivation shared by `wq_b` and the indexer's `wq_b`) has been dead since vllm#53793 (2026-09-14) renamed the attribute its gate `can_fuse_query_quant` reads, so every layer ran the BF16 `fused_q_kv_rmsnorm` and a standalone `MXFP8QuantizeSwizzledKernel` (1.6 + 1.6 µs per layer in the served graph). The patch reads the capability through `get_input_quant_key` and passes `launch_pdl`. The test (one sm_120 GPU): the gate flips, the fused kernel's FP8 bytes / F8_128x4 scale bytes / normalized KV are bit-identical to the separate path on the served geometry (1–170 tokens), and the (GEMV-patched) FlashInfer CUTLASS kernel fed the QuantizedActivation returns the same output as fed the bf16 Q |
| `moe_gate_topk/` | **measured, not applied**: a fused MoE router (gate GEMV bf16→fp32 + sqrtsoftplus + correction bias / `bias_vl` for image tokens + top-6 with ties to the smallest id + renormalization) in one launch, replacing cuBLAS's bf16 GEMM + split-K reduce + the Triton `_dsv4_topk_kernel` (4.9 + 2.7 + 3.4 µs in the served graph). `test_moe_gate_topk_sm120.py [--bench]`: bit-exact ids and weights against `dsv4_topk` on exact-logit rows and ties, identical top-6 sets on 330 random tokens (weights within 4e-7), CUDA-graph replay (the arrival counter resets itself) — but with cold gate weights on the RTX 5090 not faster (8.6 vs 8.1 µs at 6 tokens; the GEMV alone 5.4 vs 6.1): the last-CTA hand-off to the top-k serializes what the three kernels overlap. Upstream's `ll_bf16_gemm` (CuTe DSL, gated to sm_90/sm_100) runs on sm_120 but is slower than cuBLAS above 4 tokens (6.7 vs 4.4 µs at 6) |
| `mhc_pre_norm/` | **bit-identical, not applied (slower)**: the second of the two TileLang mHC kernels per sublayer (`mhc_pre_big_fuse_with_norm`: split sums, sigmoids, 4×4 Sinkhorn, collapse of the four residual streams with the carried pre-mix, RMSNorm, draft aux; 80 launches/step, 4.7 µs each in the served graph with a [T,1,1]×96 grid) as a CUDA kernel with 32 + H/8 threads per token. `test_mhc_pre_norm_sm120.py [--bench]`: every output bit-identical to the TileLang kernel for 1–64 tokens (its reduction orders — rows xor 2,1 / columns xor 8,4 / the sum of squares over 64 threads × 16 positions and the 64-wide butterfly — are reproduced from its generated CUDA), but 4.5 µs vs TileLang's 3.5 on the RTX 5090: warp 0's 20 Sinkhorn iterations (~93 ns each) are the critical path in both, with ~1 µs of unexplained overhead in ours. Starting point for a fused sm_120 mHC (DeepGEMM's `mega_mhc` is SM100-only: tcgen05/TMEM/TMA) |

## Dense MXFP8 GEMV: what did *not* help (2026-09-19)

In the server the GEMV streams weights at 0.6–1.2 TB/s: 1792×5120 in 12.3 µs, 1152×5120 in
10.8, 8192×1280 in 10.3, 5120×2048 in 8.4, 25600×5120 in 129 (rank 0 profile, cold weights and
scales). `mxfp8_gemv_sm120.cu` now carries a **v2** kernel (`VLLM_MOET_GEMV_IMPL=v2`, off by
default) that was built to test the obvious hypotheses; none of them moved the served shapes by
more than ~1 µs, so v1 stays:

| hypothesis | v2 change | result (M = 6, cold L2, `test_mxfp8_gemv_sm120.py --shapes decode`) |
|---|---|---|
| too few blocks (144–224 for N = 1152/1792) | split-K across blocks, last-arriving block reduces (deterministic order, per-stream counters, graph-replay safe) | 1152: 10.2 → 8.3 µs, 1792: 11.6 → 10.7; S=2 forced on the large shapes is 30–50 % *slower* (the fence + atomic + reload epilogue costs ~1.4 µs) |
| serial memory round trips per warp | whole 128-wide K tiles per warp, `UKT` up to 5 tiles per round (all loads in flight in one round) | no change (11.5 / 9.8 / 10.1 / 8.3 / 4.8 µs for 1792 / 8192 / 5120·2048 / 1152 / 5120·576) |
| scattered 1-byte scale loads (24 loads per lane per 4 blocks) | one 32-bit word per (row, K tile) in all three scale layouts | ablation: scales cost ~0.2 µs |
| A hot spot (every block re-reads the same 30 KB of A through the same L2 slices) | A slice staged once per block with 16-byte loads (`VLLM_MOET_GEMV_STAGE_A`) | traffic is the same, no gain; without any A loads at all the kernels are only 0.7–2 µs faster |
| DRAM layout (8 rows × 32 B segments per warp load) | ablation reading perfectly contiguous 256 B per warp load | 1792: 11.5 → 10.0, 25600: 96 → 93 µs — layout is not the limiter |

Ablations (`VLLM_MOET_GEMV_ABLATE=1..8`): the kernel skeleton without any loads is 1.0 µs; the
weight stream alone (no A, no scales) runs at ~0.95 TB/s for a 9 MB matrix with 224 blocks in one
round — short bursts on this GPU do not reach the 1.36–1.6 TB/s that the 130 MB `25600×5120`
GEMV or the Qwen MoE GEMV (33 MB, 1600 blocks) see. Realistic remaining upside for the ~250 dense
GEMVs per step is ~0.3 ms (2 %), and it would need the activation quantization to write A in a
replicated / interleaved form rather than kernel work. Register report: v2 UKT=1 uses 80–96
registers (2 blocks/SM), UKT=5 ~180.

## Dense MXFP8 GEMV v3: what the hardware counters said, and the rewrite (2026-09-19/20)

Nsight Compute on v1 (RTX 5090, caches flushed, M = 6; `--clock-control none`): DRAM throughput
**33–50 % of peak**, long-scoreboard stalls in **76–80 %** of the issue slots, 0.3 instructions
issued per scheduler cycle, 3.6 M warp instructions for the 10 MB `5120×2048` weight, 16–58 %
of the warp slots active. The reason is v1's byte movement, not its math: per MX block a lane
issues one 8-byte weight load, two 8-byte A loads and four 1-byte swizzled scale loads, and one
warp load touches eight 32-byte pieces of eight different rows. **v3** (`mxfp8_mma_gemv_v3_kernel`,
the default since 2026-09-20; `VLLM_MOET_GEMV_IMPL=v1` restores the old kernel) keeps the block =
8 columns / 8 K-splitting warps decomposition and changes the transport: K is cut into 128-byte
units (one F8_128x4 scale word), each warp streams its units in rounds — all 16-byte weight and A
loads of a round issued up front (one warp instruction = four rows × 128 contiguous bytes), parked
in registers, written to the warp's smem tile, fragments read back with `LDS.64`, the next round's
loads issued before the current round's mma — and every (row, unit) scale is one 32-bit load.

| shape (N←K), M = 6, cold L2 | v1 | v3 (R = 1) | | ncu, v3 vs v1 |
|---|---|---|---|---|
| RTX 5090 (1.79 TB/s): q_a+kv_a 1792←5120 | 10.4 µs | **7.5 µs** | −28 % | DRAM 45.6 → 53.6 % |
| q_b+indexer 8192←1280 | 9.0 | **8.3** | −8 % | 62 % |
| wo_b 5120←2048 | 8.2 | 8.1 | −1 % | 50 → 65.5 % |
| shared w13 1152←5120 | 8.9 | **5.5** | −38 % | 33 → 47.7 % |
| indexer wq_b 4096←1280 | 5.4 | 5.2 | −4 % | |
| lm_head-like 25600←5120 | 88.0 | 84.0 | −5 % | 1.49 → 1.56 TB/s |
| RTX PRO 6000 (1.6 TB/s, shared with a serving job): 1792←5120 / 1152←5120 | 11.6 / 10.1 | 8.7 / 6.8 | −25 / −33 % | |
| **in the server** (rank-0 trace, all dense shapes, cold weights+scales) | 10.8 µs avg | **10.0 µs avg** | dense GEMM share 20.9 → 19.1 %, **67.0 → 67.7 steps/s**, prose 148.5 → 149.7, code 346 → 347 tok/s | |

What did *not* help, again: **split-K** (last-arriving block reduces; `VLLM_MOET_GEMV_V3_SPLITK_MAX`)
makes the small-N shapes slower (1792←5120: 8.2 → 9.5 µs, 1152←5120: 5.8 → 6.8) — the grid is
not the problem, per-warp stream continuity is; **R = 2 or 4 units per round** (`VLLM_MOET_GEMV_V3_R`)
put more bytes in flight per round but pipeline fewer rounds and cost shared memory (R = 4 →
one block per SM, 16 % warps active): 8.3 / 8.3 / 5.6 µs against R = 1's 7.5 / 8.3 / 5.5. `K = 576`
(shared w2) is not a multiple of 128 and stays on v1 (3.7 µs, 0.8 TB/s).

Where the last 30 % is: a 5–10 µs kernel pays ~2.5 µs of ramp (first DRAM round trip), tail (last
wave) and reduction regardless of its inner loop — the 130 MB shape reaches 87 % of peak with the
same code. The step-level effect (0.35 ms of 15.6) is therefore the ceiling for kernel-body work
on these GEMVs; the remaining lever is launch count (fusing the three `mxfp8_quantize` launches
per layer into the GEMV prologue), see `docs/dsv41-sm120-port.md`.

SASS check with `cubit` (`cubit disassemble --frozen`): nvcc issues all eight 16-byte loads of the
prologue in one burst before the first `STS.128`, the inner loop is `LDS.64 → QMMA.16832.F32.E4M3.E4M3
→ FFMA` with 8-cycle stalls between dependent QMMAs, and nothing in the schedule explains the
DRAM utilisation — consistent with the counters (memory latency + fixed costs, not issue). Two
tool findings: cubit's SM120 table does not decode `LDG.E.128`/`.CONSTANT` (printed as
`__raw__…981`) or `ENDCOLLECTIVE`, and `--clock-control base` (ncu's default) understates
durations by ~30 % on this GPU — compare kernels with `--clock-control none`.

## DeepGEMM grouped MoE: BLOCK_M and padding (2026-09-19)

`tools/sm120_perf/dg_moe_blockm_bench.py` runs vLLM's own permute → FC1 → silu·up+quant → FC2 →
gather chain on the served shape (384 experts, 6 tokens × top-6, K 5120, intermediate 640).
DeepGEMM's sm_120 FP8×FP4 grouped kernel exists only with **BLOCK_M = 64** (the heuristic asserts
`not candidates.empty()` for a 32 or 16 cap), so vLLM pads every touched expert to 64 rows: 2304
rows for 36 pairs. Per layer: FC1 99 µs (118 MB of expert weights, 1.2 TB/s), FC2 41 µs
(59 MB, 1.4 TB/s), permute 11.8 µs, act+quant 4.6 µs, gather 4.6 µs. Letting the heuristic pick
freely (`mk_alignment_scope(128)` while the workspace is padded to 64) halves FC1/FC2 time but
computes the wrong experts for half the tiles (30667 of 30720 outputs differ) — vLLM's cap is
required. The FC1/FC2 padding output (5.9 + 23.6 MB per layer) is written by the GEMM epilogue
(tiles are skipped only when the whole 64-row tile is padding); a smaller BLOCK_M would need a
new DeepGEMM kernel instantiation. The glue kernels above are what was left to take.

## The other sm_120 kernels for the same job: FlashInfer CUTLASS W4A8 and B12x (2026-09-19)

vLLM 0.30 can drive two other MXFP4-expert kernels on sm_120 with a flag: `--moe-backend
flashinfer_cutlass_afp8` (FlashInfer's CUTLASS fused MoE, MXFP8 activations × MXFP4 weights — the
kernel 0xSero's SGLang stack uses here) and `--moe-backend b12x` (`pip install b12x`, the
local-inference-lab CuTe-DSL kernels for Blackwell consumer/pro parts, `w4a8_mx`). Per MoE layer on
the served rank shape (384 experts, K 5120, I 640, top-6), cold L2, RTX 5090
(`tools/sm120_perf/moe_backends_bench.py`):

| tokens per step | DeepGEMM chain (served) | FlashInfer CUTLASS W4A8 | B12x `w4a8_mx` |
|---:|---:|---:|---:|
| 1 | 43 µs | 84 µs | **26 µs** |
| 6 (DSpark k=5, one stream) | **99 µs** | 207 µs | 157 µs |
| 16 | **251 µs** | 455 µs | 350 µs |
| 48 (C8 decode) | **593 µs** | 1 000 µs | 716 µs |
| 64 | **698 µs** | 1 190 µs | 834 µs |

B12x wins only without speculative decoding (one token per step); with DSpark's 6+ verified
tokens the DeepGEMM chain is 1.6× faster than B12x and 2.1× faster than FlashInfer's CUTLASS
MoE, so the served path stays. B12x also ships a dense MXFP8 GEMM (`b12x.gemm.blockscaled.mm`);
against the GEMV above at the six decode shapes it is 1.4–3× slower at M = 6 (e.g. `wo_b`
5120←2048: 6.7 vs 9.7 µs, `kv_a` 576←5120: 5.1 vs 15.1 µs) with identical numerics
(`tools/sm120_perf/dense_b12x_bench.py`).

Run the tests inside the built image on one SM120 GPU:

```bash
docker run --rm --gpus '"device=0"' --entrypoint bash vllm-moet-sm120:dsv41-0909 -c \
  'python3 /opt/vllm-moet/dsv41_sm120/test_sparse_mla_sm120_dsv41.py --quick &&
   python3 /opt/vllm-moet/dsv41_sm120/test_deepgemm_sm120_paged_mqa.py --packed-stride &&
   python3 /opt/vllm-moet/dsv41_sm120/test_indexer_fp4_sm120.py &&
   python3 /opt/vllm-moet/dsv41_sm120/nvfp4_kv/test_packed_kv_glue.py --module /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/ops/moet_packed_kv.py &&
   python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_mxfp8_gemv_sm120.py --ms 1,6,16 &&
   python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_vllm_integration.py &&
   python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_wo_a_gemv_sm120.py &&
   python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_wo_a_integration.py'
```

On the vLLM-main image (`Dockerfile.sm120-dsv41-nightly`, `-moeqs` tag) also the fused MoE quant+scatter:

```bash
docker run --rm --gpus '"device=0"' --entrypoint bash vllm-moet-sm120:dsv41-nightly-20260923-moeqs -c \
  'cd /opt/vllm-moet/dsv41_sm120/moe_quant_scatter && python3 test_moe_quant_scatter_sm120.py --bench &&
   python3 test_moe_quant_scatter_integration.py &&
   python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_query_quant_gate.py && python3 /opt/vllm-moet/dsv41_sm120/test_query_quant_gate.py'
```

The measured-but-not-applied kernels run from the repo checkout mounted into the image:

```bash
docker run --rm --gpus '"device=0"' --entrypoint bash -v "$PWD/tools":/tools vllm-moet-sm120:dsv41-nightly-20260923-moeqs -c \
  'cd /tools/dsv41_sm120/moe_gate_topk && python3 test_moe_gate_topk_sm120.py --bench &&
   cd /tools/dsv41_sm120/mhc_pre_norm && python3 test_mhc_pre_norm_sm120.py --bench'
```

With the checkpoint mounted (`-v /path/to/DeepSeek-V4.1-Flash:/model:ro`) the FP4 KV kernels are checked
against DeepSeek's own TileLang quantizer (`inference/kernel.py`, tilelang is in the image):

```bash
docker run --rm --gpus '"device=0"' --entrypoint bash -v /path/to/DeepSeek-V4.1-Flash:/model:ro vllm-moet-sm120:dsv41-0909 -c \
  'cd /opt/vllm-moet/dsv41_sm120/nvfp4_kv && python3 test_fp4_kv_quant.py --model-dir /model &&
   cp /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py /tmp/fcqc.py &&
   python3 patch_vllm_kv_fp4_fake.py --file /tmp/fcqc.py --out /tmp/fcqc_a0.py &&
   python3 test_kv_fp4_fake_insert.py --model-dir /model --patched /tmp/fcqc_a0.py &&
   python3 test_nvfp4_kv_kernels.py --model-dir /model --a0-patched /tmp/fcqc_a0.py'
```

(the image's `fused_compress_quant_cache.py` already carries the packed-record dispatch; the A0 patcher
adds its switch on top of it.)

FlashInfer `main`'s reader of the FP4 record (not in the image, which pins 0.6.18) is checked from a
source checkout mounted into the image — an editable install replaces the packaged FlashInfer inside
that container only, and the 0.6.18 `flashinfer-cubin` / `flashinfer-jit-cache` must go so the version
check passes and the `sparse_mla_sm120` module is JIT‑built from the checkout (about 30 s):

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git /path/to/flashinfer-main   # >= eb5f05be (#5197)
git -C /path/to/flashinfer-main submodule update --init 3rdparty/cutlass 3rdparty/spdlog 3rdparty/cccl
docker run --rm --gpus '"device=0"' --entrypoint bash \
  -v /path/to/flashinfer-main:/fi -v "$PWD/tools/dsv41_sm120":/dsv41:ro vllm-moet-sm120:dsv41-0909 -c \
  'pip install -q --no-build-isolation --no-deps -e /fi && pip uninstall -q -y flashinfer-cubin flashinfer-jit-cache &&
   cd /dsv41/nvfp4_kv && python3 test_flashinfer_dsv41_fp4_extra.py --quick --precisions'
```

(`--quick` = 24 geometry cases and two timed shapes; the full run is 324 cases and six shapes, `--perf-only`
skips the checks, `--precisions` also times the `default` / `fp8` / `bf16` compute routes through the
wrapper API.) The prompt-encoder check needs no GPU; with the checkpoint mounted it
compares against DeepSeek's own encoder:

```bash
docker run --rm --entrypoint python3 -v /path/to/DeepSeek-V4.1-Flash:/model:ro vllm-moet-sm120:dsv41-0909 \
  /opt/vllm-moet/dsv41_sm120/test_reasoning_effort_encoding.py --model-dir /model
```
