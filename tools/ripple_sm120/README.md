# tools/ripple_sm120 — Ripple-65B Preview on sm_120

The Ripple line's patchers, probes and checks, by Dockerfile step (`Dockerfile.sm120-ripple`; what each step does and the
served state: `docs/ripple-sm120.md`). Shared with the DeepSeek line: `tools/dsv41_sm120/p2p_ar/` (the P2P all-reduce
module step 1 attaches to).

| file | step | what |
|---|---|---|
| `patch_vllm_p2p_allreduce_tp_sm120.py` | 1 | anchored patcher on `vllm/distributed/parallel_state.py`: `GroupCoordinator._all_reduce_out_place` routes the TP group's bf16 `[M, 5120]` all-reduces of `M <= 64` rows through `p2p_allreduce_sm120` (NCCL's summation order, bits = NCCL); `VLLM_MOET_P2P_AR=1` at runtime, default off; idempotent, `--check` |
| `patch_vllm_dflash_dynamic_k_sm120.py` | 2 | anchored patcher on `vllm/v1/worker/gpu/spec_decode/dflash/{speculator,cudagraph}.py`, `dflash2/speculator.py` and `vllm/model_executor/models/qwen3_dflash2.py`: under a dynamic schedule the DFlash2 drafter runs 1 + k query rows per request (k of the step, not 15), samples k rows, stores (1 + k) × requests context rows, captures CUDA graphs per query width like the target; its grouped convolutions read each row's offset in its request's block from the speculator (`block_offsets`) instead of `row % 16` — without that the blocks straddled requests and the drafts collapsed (the first arm: acceptance 0.24 vs 0.72 at C8). +6.5 % at 32 requests, outputs unchanged; DFlash2 opts in (`_dynamic_query_width`), DSpark / XPress / plain DFlash keep the static width; idempotent, `--check`, `--root` |
| `image_check.sh` | — | GPU-side check of a built image (the ops library needs libcuda, which `docker build` has not): ops run, `cutlass_scaled_fp4_mm` resolves, `RicursiveForCausalLM` + `DFlash2RicursiveDraftModel` registered, the checkpoint parses as `RicursiveConfig` |
| `parity_probe.py` | — | greedy parity: 3 prompts × 768 tokens with per-token logprobs through the chat API, saved per run; `--compare a.json b.json` = first diverging token, identical count, max \|Δlogprob\| (run-to-run, or one arm against another) |
| `ripple_smoke.py` | — | functional probes through the OpenAI API: chat (parsers keep reasoning out of `content`), arithmetic, tool call + round trip, strict JSON schema, needle at ~29K / 106K / 400K tokens |
| `step_anatomy.py` | — | per-step anatomy of a decode trace (`PROFILER=1`, `/start_profile` … `/stop_profile`): mean time per class (BF16 / NVFP4 GEMM, all-reduce, norms, attention, drafter glue), GPU idle, kernels per step, `--dump N` for the ordered kernel list of one step |
| `decode_gemm_bench.py` | — | the decode GEMM shapes of Ripple TP4 in isolation on one GPU inside the image: every vLLM NVFP4 / MXFP8 linear kernel for sm_120 and BF16 cuBLAS vs cuBLASLt at M = 16 / 96 / 256, CUDA-graph timed with weight rotation past L2 and FlashInfer's autotuner as in serving, bits compared against the served kernel (a microbench win is not an in-situ win: `flashinfer_cudnn`) |
| `make_mxfp8_attn_checkpoint.py` | candidate | derives `Ripple-65B-Preview-NVFP4-attnMXFP8` from the NVFP4 checkpoint: q/k/g/o_proj (BF16 in ModelOpt's export, 41 % of the bytes a TP rank reads per decode step) to MXFP8 with vLLM's e8m0/32 convention, v_proj / embeddings / lm_head / NVFP4 MLPs unchanged, `quant_algo: MIXED_PRECISION`; loads on stock vLLM (`modelopt_mixed`, FlashInfer-CUTLASS MXFP8 GEMM on sm_120). A numerics change - measured in `docs/ripple-sm120.md`, served only on the owner's decision |

Measurement on the slot uses the GLM-5.3-Flash battery (`glm_spec_matrix.py`, `prefill_sizes.py`, GSM8K-200 through
`llm_decode_bench.py`) so Ripple and GLM-5.3-Flash numbers on the same four cards are comparable.
