# tools/ripple_sm120 — Ripple-65B Preview on sm_120

The Ripple line's patchers, probes and checks, by Dockerfile step (`Dockerfile.sm120-ripple`; what each step does and the
served state: `docs/ripple-sm120.md`). Shared with the DeepSeek line: `tools/dsv41_sm120/p2p_ar/` (the P2P all-reduce
module step 1 attaches to).

| file | step | what |
|---|---|---|
| `patch_vllm_p2p_allreduce_tp_sm120.py` | 1 | anchored patcher on `vllm/distributed/parallel_state.py`: `GroupCoordinator._all_reduce_out_place` routes the TP group's bf16 `[M, 5120]` all-reduces of `M <= 64` rows through `p2p_allreduce_sm120` (NCCL's summation order, bits = NCCL); `VLLM_MOET_P2P_AR=1` at runtime, default off; idempotent, `--check` |
| `image_check.sh` | — | GPU-side check of a built image (the ops library needs libcuda, which `docker build` has not): ops run, `cutlass_scaled_fp4_mm` resolves, `RicursiveForCausalLM` + `DFlash2RicursiveDraftModel` registered, the checkpoint parses as `RicursiveConfig` |
| `parity_probe.py` | — | greedy parity: 3 prompts × 768 tokens with per-token logprobs through the chat API, saved per run; `--compare a.json b.json` = first diverging token, identical count, max \|Δlogprob\| (run-to-run, or one arm against another) |
| `ripple_smoke.py` | — | functional probes through the OpenAI API: chat (parsers keep reasoning out of `content`), arithmetic, tool call + round trip, strict JSON schema, needle at ~29K / 106K / 400K tokens |

Measurement on the slot uses the GLM-5.3-Flash battery (`glm_spec_matrix.py`, `prefill_sizes.py`, GSM8K-200 through
`llm_decode_bench.py`) so Ripple and GLM-5.3-Flash numbers on the same four cards are comparable.
