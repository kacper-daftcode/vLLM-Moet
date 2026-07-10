# Plan: Kimi-K2.7-Code (NVFP4) support

> **Execution log (2026-07-10):**
> - **P0 PASS (after a real finding):** the Kimi NVFP4 export writes **all exact zeros
>   as +0** (13.3% of expert mass; GLM/DS4 are ±0-balanced) — the sign-preserving
>   zero→±1 map would inject a **+0.134** unit-space bias/tensor, 3× the asymmetry bias
>   that degenerates GLM. Fixed in the loader (`_f64_to_codes_scales`): one-signed zeros
>   map ±1 alternating by k-parity (`VLLM_MOE_W2_ZERO_MODE={auto,sign,alt}`); L2
>   identical, net bias −0.000, balanced checkpoints bit-exact. Sweep (162 tensors,
>   9 layers × 6 experts × 3 projections): `{-4,-1,1,4}` wins 162/162, rel-RMS penalty
>   vs per-tensor asym optimum +1.5%. Tool: `tools/sweep_nvfp4_codebook.py`.
> - **P1 DONE:** K=7168 family generated (`gen_moe_w2.py`/`gen_moe_w4.py`), assembled
>   with cubit (public release, `cargo build --release`; `-t tables/sm120.json`), all
>   op-validated on RTX PRO 6000: mc1/mc2/mc4 + w4 worst_rel ≤ 3.0e-3 deterministic,
>   AFRAG bit-identical to mc4. New: `gen/culaunch.py` (standalone driver harness),
>   `gen/moe_w2_afrag_check.py`.
> - **P2 DONE:** `_ensure_ready` probes K=7168; `_layer_cutoff` unwraps VLM configs via
>   `get_text_config()` (KimiK25Config keeps `num_hidden_layers` on `.text_config` —
>   bare lookup silently fell back to 43); zero-balancing in `moe_w2_planes`. Patch
>   regenerated, applies clean on the v0.24.0 tag. `test_moe_w2_forward.py` now
>   shape-parametric (`H=7168` PASS, delta tier sized per-model).
> - **P3 (bring-up) box finding:** GPU **P2P is silently broken** on this host
>   (`can_device_access_peer`=True but D2D copies corrupt) → NCCL's first allreduce
>   hangs at 100% util. **`NCCL_P2P_DISABLE=1` required** for any TP here (SHM
>   transport; PIX topology, RunPod container). TP4 GPU-resident config untested so
>   far — GPUs 0-1 held by the GLM session — bring-up runs TP2 + BASE cache
>   (52 GiB/rank pool ≈ 39% coverage) on GPUs 2-3.
> - **P3 finding 2 (real bug, in-patch):** first TP4 run decoded fine (50 tok/s,
>   coherent) but every prompt past ~96 tokens degenerated into token soup. Root
>   cause: at prefill the tier-less desc path binds `ws["no_slots"]` — a **fixed
>   256-row** `-1` table — as the slot row, while the desc kernel clamps expert ids
>   to `n_experts-1` = **383** and reads past the tensor. OOB int32s ≥ 0 routed pairs
>   to the FP4 tier, whose GEMM never launches at prefill → those rows kept stale
>   workspace values. DS4 (256 experts) could never hit this; GLM (160) neither.
>   Fix: size `no_slots` to the model's expert count. Glue test now runs the prefill
>   tier explicitly (`T=160` PASS at H=7168/I=512; the FP4-mixed check is
>   decode-only by design, skipped at prefill).
> - **P3 finding 3 (SM12x, in-patch):** Triton MLA decode at DeepSeek dims
>   (512+64 tile, num_stages=2) wants 100 KiB smem > SM120's 99 KiB limit →
>   `num_stages=1` fallback on capability 12.x, same pattern as the existing
>   `BLOCK_DMODEL>=1024` case.
> - **P3 finding 5 (long context, ROOT-CAUSED + FIXED in-patch):** needle
>   passed to 65.6K prompt tokens and failed from ~68K — the boundary tracks the MLA
>   **chunked-context workspace cap (64Ki tokens)**: final-chunk context ≤64Ki is
>   single-chunk (PASS), above it `_compute_prefill_context` iterates ≥2 chunks and
>   the output was corrupt. Root cause: **`merge_attn_states` (CUDA + Triton) indexes
>   BOTH prefix and suffix with the single stride `prefix_output.stride(1)`**. On FA2
>   platforms (SM120 — no FA3/FA4) every chunk output is an unpadded slice of the
>   v-padded buffer (head stride 192 at head size 128), while the merged intermediate
>   from the previous merge is allocated contiguous via `empty_like` (stride 128).
>   First merge diverges the strides → the final context⊕suffix merge (≥2 chunks) and
>   the in-loop merges (≥3 chunks) read the suffix at wrong offsets → garbage.
>   The earlier standalone repro missed it by keeping the padded width through the
>   merge; a faithful repro (per-chunk unpad slices, production loop) shows rel err
>   0.22 (2 chunks) / 0.72 (3 chunks) vs 5e-3 single-chunk — and ~8e-3 after the fix.
>   Fix in `vllm/v1/attention/ops/merge_attn_states.py`: contiguize prefix/suffix
>   when their strides differ (no-op otherwise). Upstream master (Jul 2026) has the
>   same bug — FA3/FA4 (Hopper+) never pad v, so strides always match there; affects
>   FA2 fallback platforms only. `VLLM_MLA_CHUNKED_WORKSPACE_TOKENS` stays as a
>   diagnostic knob but is **no longer needed**: default 64Ki workspace + full
>   262144 window serve correctly — needle PASS at 80K/128K/192K/248K post-fix
>   (2-4 context chunks), util back to 0.94 (the 0.94 OOM was a property of the
>   139K-workspace workaround, whose ~1.1 GiB/rank hidden up-projection is gone).
> - **P3 finding 4 (venv-only, not in-patch):** Kimi's MLA rope goes through
>   `DeepseekScalingRotaryEmbedding` → flashinfer JIT rope (rope head 64). On the
>   no-docker venv, flashinfer 0.6.14 detects SM120 but the worker env lacked a CUDA
>   ≥12.9 toolchain → `_normalize_cuda_arch` raised, `TARGET_CUDA_ARCHS` stayed empty,
>   and the JIT died with the misleading "FlashInfer requires GPUs with sm75 or
>   higher". Fix: `FLASHINFER_CUDA_ARCH_LIST=12.0f` + `CUDA_HOME`/`PATH`/
>   `LD_LIBRARY_PATH` pointing at the pip `nvidia/cu13` toolkit (nvcc 13.2). The
>   official docker image ships nvcc and never hits this.

Target checkpoint: [nvidia/Kimi-K2.7-Code-NVFP4](https://huggingface.co/nvidia/Kimi-K2.7-Code-NVFP4)
(595 GB, modelopt NVFP4, local copy at `/root/models/Kimi-K2.7-Code-NVFP4`). Goal: serve it
through the vLLM-Moet 2-bit expert stack on SM120 — TP4 GPU-resident on 4× RTX PRO 6000,
and the BASE-cache (host-resident) path for 2- and 1-card configs. The stock checkpoint
cannot fit this box at all (595 GB weights vs 384 GB total VRAM), so as with GLM-5.2 the
2-bit path is the capacity unlock, not just a speedup.

## 0. Bring-up results (2026-07-10, measured)

**TP4, 4× RTX PRO 6000 (96 GB), GPU-resident 2-bit experts + FP4 delta (auto ≈2.7 GiB/rank),
262144-token window (util 0.94, 292.6K-token fp8 KV pool), CUDA graphs FULL_AND_PIECEWISE:**

| probe | result |
|---|---|
| decode, single stream, 512 tok greedy | **51 tok/s** (median of 3; no MTP — checkpoint ships none) |
| batched decode (256 tok/stream, aggregate) | 1→**52**, 4→**149**, 8→**222 tok/s** (27.7/stream at 8) |
| prefill, 8K unique prompt | **2 448 tok/s** (median of 3; AFRAG on) |
| needle 8K / 32K / 80K / **128K** (depths 0.1-0.5) | **PASS** (128K: TTFT+gen 84.5 s cold) |
| needle post-stride-fix, default 64Ki workspace: 80K / 128K / **192K** / **248K** | **PASS** (2-4 context chunks exercised; 248K: TTFT+gen 629 s cold, reply drifts after the correct passphrase — quality at the YaRN edge, not a retrieval failure) |
| arithmetic (5× multi-digit, chat+reasoning) | 5/5 |
| code-gen (`top_k_frequent`, executed) | tests pass; clean single code block |
| tool calling (`kimi_k2` parser, auto tool choice) | call + round-trip **PASS** (`--enable-auto-tool-choice` required) |
| coherence | greedy outputs coherent across all probes |
| load time (595 GB checkpoint → planes) | ~25 min (4:40 shard read + ~9 min f64 requant/staging + init) |

**TP2 + BASE cache (2× 96 GB, 52 GiB/rank pool ≈ 39% coverage, 16K window):** serves
coherently at **14.4 tok/s** decode with pool still converging (miss replays); needle 8K
PASS. The TP2 path is functional but wants the planes cache + longer warm before real
numbers.

Open items: batched serving numbers, FP8 dense follow-up (§7), upstreaming the
`merge_attn_states` stride fix (§finding 5). Multi-chunk MLA context: root-caused
and fixed (§finding 5); planes cache + Eagle3 drafter: shipped and serving.

## 1. What the checkpoint is

| fact | value | consequence |
|---|---|---|
| architecture | `KimiK25ForConditionalGeneration` (VLM wrapper), text = `DeepseekV3ForCausalLM` | v0.24.0 already registers it (`model_executor/models/kimi_k25.py`); text model instantiated as classic `DeepseekV2ForCausalLM` → standard `FusedMoE` → `ModelOptNvFp4FusedMoE`. **Our existing modelopt hook is on exactly this class.** |
| geometry | 61 layers (layer 0 dense), H=7168, 384 routed experts top-8 + 1 shared, I=2048, MLA (q_lora 1536 / kv_lora 512 / rope 64), vocab 163 840 | **H=7168 needs a new cubin family** — everything else is in the shipped K set. |
| quantization | modelopt NVFP4 (e2m1 + e4m3 block-16 scales + per-tensor `weight_scale_2` + `input_scale`), **routed experts only** | same format as GLM-5.2-NVFP4 → `build_layer_planes_nvfp4` / `nvfp4_to_codes_scales` reuse as-is. |
| everything else | BF16 (attention, shared experts, dense layer 0, embed/lm_head, vision tower); `exclude_modules` lists them, `hf_quant_config.json` | excluded modules resolve to `UnquantizedLinearMethod` — no DeepGEMM/fp8-einsum involvement at all (unlike DS4). BF16 dense is ~3.1× the per-token bytes of the 2-bit experts → main perf lever later (§7). |
| attention | **dense** MLA (no DSA/Lightning indexer), 256K YaRN | SM120 backend = `TRITON_MLA` (supports fp8 KV). The sparse-MLA fixes, cubit MLA-prefill and NVFP4-KV tier in this repo do **not** apply — they are sparse-layout-specific. |
| MTP | `num_nextn_predict_layers: 0`, no drafter weights | no MTP speedup available. Model declares `SupportsEagle3` — external drafter is a possible follow-up. |
| tokenizer / serving | tiktoken custom code, `chat_template.jinja`, `kimi_k2` tool+reasoning parsers exist in v0.24.0 | serve with `--trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2`; **no** `--tokenizer-mode deepseek_v4`, **no** speculative-config. |
| vision | MoonViT 27 blocks BF16, images+video | supported by upstream `kimi_k25_vit.py`; validate late — text-only serving is the primary Code use case. |

Checkpoint naming (`language_model.model.layers.N.mlp.experts.N.{gate,up,down}_proj.*`)
matches the vLLM module tree of the wrapper (prefix `language_model.`), and the
`exclude_modules` wildcards are in the same namespace — the modelopt exclusion matching
and `apply_vllm_mapper` expansion work unchanged. `mlp.gate` (router) is not excluded and
not quantized: `GateLinear` never asks for a quant method, so nothing to do.

## 2. Memory & bandwidth budget (computed, not measured)

2-bit planes: 12.39 MB/expert-layer (codes13 7.34 + sc13 0.92 + codes2 3.67 + sc2 0.46)
× 384 experts × 60 layers = **265.8 GiB** total (vs 531.6 GiB raw NVFP4).

| config | planes/rank | BF16 dense/rank | total/rank | verdict |
|---|---:|---:|---:|---|
| TP4 (4× 96 GB) | 66.4 GiB | ~8 GiB | ~75 GiB | **fits GPU-resident**, ~15-18 GiB left for KV + FP4 delta + graphs |
| TP2 (2× 96 GB) | 132.9 GiB | ~13 GiB | >96 GiB | BASE cache: pool ~40-45 GiB/rank ≈ 30-34% expert coverage |
| TP1 (1× 96 GB) | 265.8 GiB | ~24 GiB | ≫96 GiB | BASE cache: pool ~50-55 GiB ≈ 19-21% coverage (the DS4-on-5090 regime); needs ~266 GiB pinned host (box has 1.5 TB) |

KV (fp8, dense MLA, replicated per TP rank): 34.3 KiB/token → 128K ctx = 4.3 GiB,
256K = 8.6 GiB. TP4 can hold the full 256K window single-seq.

Decode ceiling at TP4 (bytes/token/rank ÷ ~1.6 TB/s): experts-2bit 1.49 GB + shared-BF16
1.32 GB + attn-BF16 3.08 GB + lm_head 0.59 GB ≈ 6.5 GB → **~250 tok/s theoretical, expect
~100-140 single-stream** (no MTP). Batch scaling should mirror DS4/GLM (experts amortize).

## 3. Gap analysis — what actually has to change

Everything below was verified against the patch source and v0.24.0 tree; the patch
applies clean on the tag (checked on this box).

1. **K=7168 cubin family** (the only real kernel work). gate/up GEMM contracts over
   K=H=7168 at every TP degree (H never shards); down-proj contracts over I/TP =
   2048/1024/512 — already shipped. 7168 = 56×128, NWARP=8 → KSLICE=896 ✓ generator
   constraints hold. To build, per `kernels/MANIFEST.md` (cubit pinned @ `5912400`,
   stub `sass/qmma_e4m3.merc.stub` is in-repo):
   - `gen/gen_moe_w2.py` → `moe_w2_mm_k7168` (MC=1 decode), `MOEW2_MC=4` → `mc4_k7168`,
     `MOEW2_MC=4 MOEW2_AFRAG=1` → `mc4afrag_k7168` (prefill default), optionally `mc2`.
   - `gen/gen_moe_w4.py` → `moe_w4_mm_k7168` (FP4 delta/gate tier).
   - op-validate each with `gen/moe_w2_check.py` / `gen/moe_w4_check.py` (rel ~1-3e-3,
     determinism, M ≤ 16), then add rows to `MANIFEST.md` and ship cubins.
2. **Loader probe list** — `moe_w2_cubit.py::_ensure_ready` probes a hardcoded K tuple
   `(6144, 4096, 2048, 1024, 512)` in two loops (w2/w4/mc tiers + afrag). Add 7168.
   `_require_kernels` then enforces presence at weight load. (`_nwarp_for_k(7168)`
   already returns 8 — matches the generator.)
3. **Layer cutoff via text config** — `_layer_cutoff()` reads
   `hf_config.num_hidden_layers`; `KimiK25Config` keeps that on `text_config`, so the
   lookup raises and silently falls back to 43 → layers 43-60 would dodge the w2 path and
   OOM on the stock path. Fix: `hf_config.get_text_config().num_hidden_layers` (HF API;
   returns self for DS4/GLM — no behaviour change there). Workaround until then:
   `VLLM_MOE_W2_NUM_LAYERS=61`.
4. **Nothing else in the hook chain**: `is_w2_layer` regex matches the
   `language_model.model.layers.N.mlp.experts` prefix; `create_weights` host-staging,
   `build_layer_planes_nvfp4` (shape-derived, per-expert `weight_scale_2` [E,2]/[E]
   handling identical to GLM), `_workspaces` (sized from `st["K13"]`/`st["K2"]` = 7168/
   2048-per-rank), delta/base tiers (sized from plane bytes), gate, deterministic
   unpermute — all shape-generic. Shared experts stay a separate BF16 module on CUDA
   (aiter fusion is ROCm-only) and are orchestrated by the runner exactly as on GLM.
5. **Serving flags**, not code: drop MTP speculative-config and `--tokenizer-mode
   deepseek_v4`; add `kimi_k2` parsers. DeepGEMM/flashinfer pins are harmless but
   unnecessary for this model (no sparse MLA, no fp8 dense einsum).

## 4. Phases

**P0 — prep (in progress).** Checkpoint download to `/root/models/Kimi-K2.7-Code-NVFP4`;
integrity check vs HF metadata (67 shards + index + aux files). Codebook pre-validation
on real shards: port the GLM-5.2 sign-asymmetry sweep (repack tooling in
`tools/repack_expert_bits.py`) to ~200 Kimi expert tensors across layers/projections.
Gate: sign-symmetric `{-4,-1,1,4}` wins at equal rel-RMS like on DS4/GLM (asym tail-drop
bias present). If Kimi's expert distributions break the finding, stop and re-evaluate
codebook choice before any serving work.

**P1 — kernels.** Generate + assemble + op-validate the K=7168 family (w2 mc1/mc4/
mc4afrag + w4), ship cubins + MANIFEST rows. No new SASS authoring — generator emits any
K; this is turn-the-crank.

**P2 — glue.** The two patch-side edits from §3 (K tuple, `_layer_cutoff` via
`get_text_config`) + regenerate `patch/vllm-moet-v0.24.0.patch`. Extend
`tools/test_moe_w2_planes.py` shapes to H=7168 and run `tools/test_moe_w2_forward.py`
against the new cubins.

**P3 — TP4 bring-up (4× RTX PRO 6000, GPU-resident).** Load with `VLLM_MOE_W2=1`,
`VLLM_MOE_W2_DELTA_GB=auto`. Watch for: per-rank plane build time over 23 040
expert-layer pairs (f64 dequant path, chunk=8 — expect a slow first load; see P6
planes-cache follow-up), host staging churn (~530 GB transient across ranks — box has
1.5 TB), `moe_w2: layer N planes built` covering layers 1-60.

**Validated serve command** (no-docker venv on this box; in the docker image the
`FLASHINFER_*`/`CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` lines are unnecessary):

```bash
NCCL_P2P_DISABLE=1 \                       # this host: P2P silently corrupt
FLASHINFER_CUDA_ARCH_LIST="12.0f" \        # venv-only: rope JIT arch detect
CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13 \
PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_MOE_W2=1 VLLM_MOE_W2_DELTA_GB=auto VLLM_MOE_W2_DELTA_RESERVE_GB=6 \
VLLM_MOE_W2_CUBIT_DIR=/root/vLLM-Moet/kernels/cubins-sm120 \
VLLM_MOE_W2_PLANES_CACHE=/root/models/.planes-cache-kimi \  # optional: fast restarts
vllm serve /root/models/Kimi-K2.7-Code-NVFP4 \
  --served-model-name kimi-k2.7-code --trust-remote-code \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --kv-cache-dtype fp8 --block-size 256 --max-model-len 262144 \
  --gpu-memory-utilization 0.94 --max-num-batched-tokens 2048 --max-num-seqs 8 \
  --no-scheduler-reserve-full-isl \
  --enable-auto-tool-choice --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
```

(Full native 262144 window since the finding-5 fix: the default 64Ki chunked
workspace serves any context length; fp8 KV pool at util 0.94 = 292.6K tokens,
covers a full-window sequence. Chunked-context up-projection transients are
~0.55 GiB/rank and *outside* the memory profiler's view — with the small default
workspace, util 0.94 + `DELTA_RESERVE_GB=6` survives a cold 248K prefill (the
0.94 OOM only happened with the 139K-token workspace workaround, now obsolete).
`--enable-auto-tool-choice` is required for `"tool_choice": "auto"`.)

Acceptance: coherent greedy output, CUDA graphs captured, `tools/bench_tok.py` decode +
8K unique-prefix prefill numbers, batch scaling 1→32, needle probe to 128K.

**P4 — quality.** QUANT_PROBE-style protocol (`tools/probe_quant_quality.py`,
`docs/quality.md`): greedy agreement of 2-bit vs FP4 reference on a coding-heavy prompt
set, coherence set 12/12, then the FP4 delta + confidence gate (`VLLM_MOE_W2_GATE=1`,
τ sweep 0.60/0.75) — expect the same "gate recovers numeric-heavy quality" shape as
GLM. No MTP acceptance metric exists here (no drafter) — use greedy-agreement % and
task-level spot checks (HumanEval-style snippets, tool-call round trips) instead.

**P5 — BASE cache configs.** TP2 (~43 GiB/rank pool) and TP1×96GB (~53 GiB pool, ~266 GiB
pinned host). First measure routing concentration live (pool hit-rate counters already
logged) — 384 experts top-8 is a different regime than DS4's 256/GLM's landscape; the
19%-coverage≈96%-hit heuristic must be re-established before promising numbers. Three-tier
(host 2-bit → GPU 2-bit cache → gate-filled FP4) as shipped for GLM.

**P6 — follow-ups (separate efforts, not blockers).**
- **FP8 dense + shared experts — ATTEMPTED, PARKED (2026-07-10)**: BF16 attention/
  shared/lm_head is ~77% of per-token bytes at TP4; FP8 would be ~+60% decode ceiling.
  The naive route (override `is_layer_excluded` → `Fp8LinearMethod` online quant,
  `VLLM_MOE_W2_DENSE_FP8={1,attn,l0}`, in-patch but default-off) is **blocked on two
  real issues** found in bring-up: (1) the stock non-serialized Fp8 flow depends on the
  loader's online-quantize hook that only engages when the TOP-LEVEL quantization is
  "fp8" — under the modelopt config `process_weights_after_loading` sees the sentinel
  scales and serves garbage (fixed in-patch with a self-quantizing subclass); (2) even
  then, the ScaledMM kernel wrapper **crashes with an illegal memory access on SM120**
  in an offline single-layer repro (CutlassFP8ScaledMM selected; the raw
  `ops.cutlass_scaled_mm` call with the same tensors computes fine, rel ~3e-2 — the
  fault is in the wrapper's padding/logical-size path, not the GEMM). Serving
  symptom: NaN logits → "!!!" floods in ALL scopes (attn/mlp/l0). Next steps for the
  follow-up: debug `ScaledMMLinearKernel.apply_weights` padding on SM120, or route
  through `skinny_fp8_cubit` (K∈{2048,4096,7168} repack list already covers Kimi), or
  torch._scaled_mm rowwise. Until then the flag stays off; production remains BF16
  dense at 61 tok/s.
- **Planes cache — SHIPPED + VALIDATED (v1)**: `VLLM_MOE_W2_PLANES_CACHE=<dir>` caches
  the built 2-bit planes (+FP4 delta planes when the tier is on) per TP rank
  (`moe_w2_planes_cache.py`; key = checkpoint sha + TP layout + zero mode + codebook
  version; per-layer hit/miss, async best-effort writes, size-validated reads). NVFP4
  builder only for now (mxfp4/fp8 builders: same two hooks). **Measured end-to-end
  (2026-07-10): write pass added no visible load time (739 GiB written async during
  the build); read pass hit 240/240 layers and cut the restart from ~25 min to
  ~14.5 min; needle + code output coherent on cache-loaded planes.** v1.5 follow-up:
  skip the expert shard read too (no-op `weight_loader` on validated hits) — would
  cut restarts to ~5 min.
- **Eagle3 drafter** to replace the missing MTP (model advertises `SupportsEagle3`).
  **Researched + staged (2026-07-10):** a matched community drafter exists —
  [AQ-MedAI/Kimi-K2.7-Code-eagle3](https://huggingface.co/AQ-MedAI/Kimi-K2.7-Code-eagle3)
  (2B, 1 layer, hidden 7168, target vocab 163840 ✓ vLLM's vocab-equality check,
  draft_vocab 96000 + d2t/t2d, `LlamaForCausalLMEagle3` — registered in v0.24;
  acceptance ~2.9 on HumanEval) — downloaded to `/root/models/Kimi-K2.7-Code-eagle3`
  (3.1 GiB, config verified). Expected ~1.3-1.7× single-stream (51 → ~65-85 tok/s).
  **Trialed (2026-07-10):** acceptance on code is excellent (mean acceptance length
  3.4-3.7 of 4, 82-90% draft acceptance — the generic-data drafter transfers fine to
  the 2-bit target), but the net is modest: with `draft_tensor_parallel_size: 1` the
  unsharded 2B drafter (3 sequential passes/step, 96K-vocab head on one GPU) *lost*
  to baseline (median ~42 vs 51 tok/s; GPU0 42% util vs 95% others); with
  `draft_tensor_parallel_size: 4` it nets **~57 tok/s median (+12%)** with high
  run-to-run variance (38-63). Config in production now:
  `--speculative-config '{"model":"/root/models/Kimi-K2.7-Code-eagle3","method":"eagle3","num_speculative_tokens":3,"draft_tensor_parallel_size":4}'`
  (keep 1 or 3 tokens — MLA capture-size issue at other values; drafter decodes run
  FULL_DECODE_ONLY graphs). Tuning ideas: k=2 (cheaper drafting at slightly lower AL),
  suffix-decoding A/B.
  **Variance root cause found (2026-07-10): the FP4 delta tier makes greedy decode
  non-deterministic across runs** — background promotions flip experts between 2-bit
  and FP4 mid-serving, perturbing logits at near-ties; 6/6 identical greedy prompts
  produced 6 distinct outputs (first divergence within ~100 chars). Without
  speculation this only wobbles content, not throughput (baseline was ±1%); with
  speculation, content trajectory drives acceptance → 30-60 tok/s spread. This is
  inherent to the delta tier's design (runtime precision changes), not a bug — but it
  interacts badly with spec decode benchmarking and reproducibility. A/B of spec
  variants therefore runs with `VLLM_MOE_W2_DELTA_GB=0`.

  **Spec-decode A/B (2026-07-10, TP4, 256K window, delta off, greedy code prompt,
  6 runs each):**

  | variant | median decode | spread | deterministic | notes |
  |---|---:|---:|---|---|
  | no speculation | 51 tok/s | 1% | yes | |
  | **eagle3 k=3 dtp=4** | **61.1 tok/s (+20%)** | 6% | **yes (6/6 identical)** | production pick |
  | eagle3 k=2 dtp=4 | 57.4 tok/s | 3% | yes | verify nearly free → deeper drafts win |
  | eagle3 k=3 dtp=1 | ~42 tok/s | high | — | unsharded 2B drafter starves one GPU |
  | eagle3 k=3 + delta auto | 57.2 tok/s | high | no | delta costs ~4 tok/s + determinism |
  | suffix k=32 / k=3, FULL graphs | fails to start | — | — | MLA `build_for_cudagraph_capture` asserts (`max_query_len <= reorder_batch_threshold`); vLLM v0.24 raises the threshold for eagle-family only |
  | suffix k=16, PIECEWISE graphs | 56.6 tok/s | 17% | no (6/6 distinct) | AL 1.86, 35% accept — fresh code-gen has nothing to look up; variable-length verify shapes flip ties via reduction order; loses FULL-graph decode |

  Suffix may still pay on real agent loops (re-emitted files/boilerplate) but is
  structurally handicapped on this backend (no FULL graphs). Production config:
  **eagle3 k=3, draft TP 4, delta off** — 61 tok/s deterministic single-stream. Zero-cost alternative available immediately:
  `{"method":"suffix","num_speculative_tokens":32}` (`arctic-inference` installed in
  the venv; reported ~2× on agentic-code workloads) or plain ngram. Verify the w2
  decode tier under spec verify steps (T = 1+k stays ≤ decode threshold ✓ by design).
  If acceptance on real traffic disappoints (<2.3): train an Eagle3.1-MLA drafter with
  TorchSpec/ModelOpt streaming against THIS NVFP4 deployment (~0.5-1k H200-hours,
  CoreWeave's NVFP4-regenerated-data playbook).
- **Vision path validation** on SM120 (MoonViT BF16); until then serve text-only if it
  misbehaves (`--limit-mm-per-prompt '{"image":0,"video":0}'`).

## 5. Risks

| risk | exposure | mitigation |
|---|---|---|
| sign-sym codebook doesn't transfer to Kimi experts | quality tank, degenerate loops | P0 sweep gate before any kernel/serving work |
| routing concentration weaker at 384 experts | BASE-cache configs (TP2/TP1) throughput | measure hit-rates in P5; TP4 GPU-resident path unaffected |
| `kimi_k2` parsers vs K2.7-Code template drift (`tool_declaration_ts.py` is new) | tool-calling UX | P3 smoke: tool-call round trip; parsers are upstream, fixable in-patch |
| Triton MLA decode perf on SM120 (dense MLA, 61 layers) | TTFT/decode share of attention | acceptable for bring-up; cubit dense-MLA kernel only if profiling says so |
| load time (23 040 experts × f64 requant) | operator pain | P6 planes cache; until then accept slow first load |
| vision tower untested on SM120 | image/video requests | text-only fallback flag; validate in P6 |

## 6. Explicitly out of scope

- MTP (checkpoint ships no drafter head — `num_nextn_predict_layers: 0`).
- NVFP4 KV cache tier (sparse-MLA `fp8_ds_mla`-layout-specific; Kimi is dense MLA).
- DeepGEMM / flashinfer sparse-MLA pins and the DS4 `o_proj`/indexer SM120 fixes (inert
  for this model; keep them in the shared image, they don't activate).
- EP / redundant experts / EPLB.
