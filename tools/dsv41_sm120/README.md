# DeepSeek-V4.1-Flash on SM120 — kernel gap closure

Kernel-side patches that let the official `vllm/vllm-openai:deepseekv41-flash-0909` image serve
DeepSeek-V4.1-Flash on RTX PRO 6000 / RTX 5090 (sm_120). Port notes, gap inventory and
validation evidence: `docs/dsv41-sm120-port.md`; image: `Dockerfile.sm120-dsv41`.

| file | role |
|---|---|
| `sparse_mla_sm120_dsv41.cu` | new FlashInfer JIT TU: SM120 sparse-MLA instantiations for the V4.1 geometry (SWA page 32, compressed page 128/64, SWA rows 128/192/1152) + `sparse_mla_prefill_dispatch_dsv41` |
| `patch_flashinfer.py` | idempotent, anchored patcher for an installed flashinfer package (installs the TU, orchestrator hook, PBS=32 decode table, python dispatch) |
| `patch_deepgemm.py` | DeepGEMM `8b1392b9` host asserts: SM120 FP8 paged MQA logits on 128-row pages |
| `test_sparse_mla_sm120_dsv41.py` | op-level validation: torch reference + bit-exact re-paging parity vs stock PBS=64 kernels |
| `test_deepgemm_sm120_paged_mqa.py` | op-level validation: DeepGEMM reference + bit-exact parity block_kv 64 vs 128; native next_n 1/2/6 and the varlen (`indices=`) mode |
| `sm120_gemv/mxfp8_gemv_sm120.cu` | tensor-core (mma.m16n8k32 e4m3) MXFP8 GEMV for decode shapes (M <= 16), F8_128x4 swizzled scales, plus a scalar fallback (`VLLM_MOET_GEMV_IMPL=scalar`) and the v2 experiment (`=v2`, see below; not faster); loaded via `sm120_gemv/mxfp8_gemv_sm120.py` (torch cpp_extension JIT, precompiled in the image) |
| `patch_vllm_mxfp8_gemv.py` | vLLM patch: `FlashInferCutlassMxfp8LinearKernel.apply_weights` routes M <= 16 to the GEMV (`VLLM_MOET_SM120_GEMV=0` reverts) |
| `sm120_gemv/vllm_sm120_gemv_bmm.py`, `patch_vllm_wo_a_sm120.py` | `Sm120GemvMxfp8BmmLinearKernel`: keeps the grouped o-projection `wo_a` (`is_bmm`, 2 head groups × [1024 <- 4096] per TP4 rank) in MXFP8 and runs decode batches (<= 64 tokens) on `mxfp8_gemv_grouped` with the FP8 activations + packed MN-major scales that `fused_inv_rope_fp8_quant` produces for the sm_100 DeepGEMM path; prefill dequantizes the weight on the fly and keeps the bf16 bmm. The patcher installs the module into vLLM, puts the kernel first in `init_mxfp8_linear_kernel()`'s BMM list and adds the dispatch to `deep_gemm_fp8_o_proj` (`VLLM_MOET_SM120_GEMV_BMM=0` reverts to the BF16 emulation) |
| `sm120_gemv/test_mxfp8_gemv_sm120.py`, `sm120_gemv/test_vllm_integration.py` | GEMV validation vs FlashInfer CUTLASS `mm_mxfp8` and an fp32 reference (cold-L2 timing on the served shapes); dispatch check of the patched vLLM kernel class |
| `sm120_gemv/test_wo_a_gemv_sm120.py`, `sm120_gemv/test_wo_a_integration.py` | grouped GEMV vs fp32 reference on the fp8 operands and vs the BF16 bmm path (cold-L2 timing, T = 1..64); end-to-end check of the patched kernel selection + `deep_gemm_fp8_o_proj` dispatch (GEMV for T <= 64, bit-identical bf16 fallback above) |
| `patch_vllm_indexer_sm120.py` | **not applied in the image**: lets the DSA indexer use DeepGEMM's varlen / native multi-row paged MQA on sm_120 (prerequisite for DSpark adaptive verification). Validated at op level, but the varlen decode path cost ~0.5 ms/step and adaptive verification was a net loss on 2026-09-18 (docs/dsv41-sm120-port.md) |
| `patch_vllm_moe_glue_sm120.py` | vLLM patch (three files, bit-exact data movement): DeepGEMM MoE glue at decode token counts — `_fwd_kernel_ep_scatter_2` one program per (token, expert) pair instead of per token (6 dependent atomics → 1), `_fwd_kernel_ep_gather` top-k loop unrolled (`tl.static_range`, same fp32 order), `silu_mul_quant_fp8_packed_triton(m_indices=)` skips the 63 padding rows per expert that DeepGEMM's 64-row contiguous layout adds (2304 rows, 36 real). Per layer in `tools/sm120_perf/dg_moe_blockm_bench.py`: permute 11.8 → 8.5 µs, act+quant 4.6 → 3.9, gather 4.6 → 4.0 (MoE output 0 of 30720 elements differ). Applied at image build (`Dockerfile.sm120-dsv41` step 3) since 2026-09-19; was first deployed as bind-mounts over the previous image |

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
   python3 /opt/vllm-moet/dsv41_sm120/test_deepgemm_sm120_paged_mqa.py &&
   python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_mxfp8_gemv_sm120.py --ms 1,6,16 &&
   python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_vllm_integration.py &&
   python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_wo_a_gemv_sm120.py &&
   python3 /opt/vllm-moet/dsv41_sm120/sm120_gemv/test_wo_a_integration.py'
```
