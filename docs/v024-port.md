# The v0.24.0 port

Since 2026‑07 the project targets **official vLLM v0.24.0** instead of the old `v0.19.2rc0`
fork. Upstream now ships DeepSeek‑V4 + SM120 natively (new `vllm/models/deepseek_v4/` layout,
FlashInfer SM120 sparse‑MLA, GLM‑5.x `GlmMoeDsaForCausalLM`), so the old 43.5k‑line backport is
gone; what remains is a **~3.4k‑line overlay**: the 2‑bit expert planes, the FP4 delta cache,
the confidence gate, the cubit dispatch — plus the SM120 fixes below.

## Apply

```bash
git clone --branch v0.24.0 https://github.com/vllm-project/vllm && cd vllm
git apply /path/to/vLLM-Moet/patch/vllm-moet-v0.24.0.patch

# python-only overlay: reuse the official precompiled wheel for the C/CUDA artifacts
VLLM_USE_PRECOMPILED=1 pip install -e . --no-deps --no-build-isolation
```

Environment pins that go with the patch (both required on SM120):

| dep | version | why |
|---|---|---|
| DeepGEMM | nv‑dev **`a6b593d2`** (build from source, ~2 min) | the release pin `891d57b4` has no family‑120 host paths — "Unknown SF transformation" (linear), `t.dim()==N` (o_proj einsum), "Unsupported architecture" (indexer paged‑MQA metadata). Same as vLLM issues #47130/#47436; vLLM main already moved its pin. |
| flashinfer‑python | **0.6.14** (+ `flashinfer-jit-cache==0.6.14+cu130`) | the official 0.6.12 pin predates the kwargs (`swa_topk_lens`, `extra_sparse_*`) that v0.24's SM120 DS4 attention passes to `trtllm_batch_decode_sparse_mla_dsv4`. |

## SM120 fixes carried in the patch (base was broken-as-released)

1. `sparse_attn_indexer.py` — don't select `cooperative_topk` on SM12x (thread‑block **cluster
   launch** is SM90/SM100‑only; consumer Blackwell rejects it with "invalid argument").
2. `models/deepseek_v4/nvidia/ops/o_proj.py` — SM12x einsum recipe = SM90‑style
   `(1,128,128)` with **raw row‑major f32 block scales** (matches DeepGEMM nv‑dev's own SM120
   test convention; the SM100 packed/TMA‑aligned int32 layout produces NaN).
3. `fp8_utils.py` — skip the SM100 weight‑scale pre‑packing for the `is_bmm` (einsum) weights
   on family 120.
4. `capture_error_mode="thread_local"` on **all four** CUDA‑graph capture paths
   (`compilation/cuda_graph.py`, `compilation/breakable_cudagraph.py`,
   `v1/worker/gpu/cudagraph_utils.py`, `v1/worker/gpu_ubatch_wrapper.py`) — the FP4 delta
   cache promotes experts from a background thread; without thread_local its side‑stream work
   invalidates capture (`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`).

## Our hooks

- `mxfp4.py` (`Mxfp4MoEMethod`) — FP4‑checkpoint path (DeepSeek‑V4‑Flash): host‑stage experts
  at `create_weights`, build 2‑bit planes at `process_weights_after_loading`, `moe_w2_forward`
  in `apply`.
- `fp8.py` (`Fp8MoEMethod`) — FP8 block‑quant checkpoint path (DS4‑Flash‑Base,
  **GLM‑5.2‑FP8**): same three hooks; the loader re‑quantizes fp8+f32‑block‑128 to the
  sign‑symmetric 2‑bit codebook at load (`build_layer_planes_fp8`, float64 math, golden‑tested
  against the GLM‑5.2 sweep reference).
- `moe_w2_*` utils are shape‑generic now: layer cutoff from `num_hidden_layers` (43 DS4 / 78
  GLM‑5.2), cubins probed for K∈{6144,4096,2048,1024,512}, workspaces sized from the model's
  hidden size (GLM‑5.x H=6144 supported; `kernels/` ships the K=6144 family).

Everything stays **opt‑in** (`VLLM_MOE_W2=1` etc.); with the knobs off the only behavioural
delta vs stock v0.24.0 are the SM120 fixes above.

The **confidence gate is fully wired** (2026‑07‑08 evening): `VLLM_MOE_W2_GATE=1` arms the
FP4 re‑forward — low‑confidence decode steps force‑promote their routed experts to FP4 and
replay the step once (inline on TP/single‑GPU incl. MTP verify steps; a worker‑driven
full‑pipeline replay under PP, pure‑decode only). τ is runtime‑tunable via
`VLLM_MOE_W2_GATE_TAU(_FILE)`. Live‑validated on the official checkpoint (1× PRO 6000, MTP
k=2, graphs): fires/promotes/replays per τ, coherent output; arming the gate costs ~10%
single‑stream (per‑step confidence sync) and the replays at τ=0.60 were throughput‑neutral
on top of that (FP4 re‑decides lift MTP acceptance enough to pay for themselves); τ=0.75
costs ~5% more. Still pending from the fork: the **cubit sparse‑MLA prefill callsites**
(moot — upstream's FlashInfer SM120 prefill outbenches the fork's path, see below).

## Benchmarks vs the old fork (2026‑07‑08)

Same machine, same official FP4 checkpoint, same knobs (`VLLM_MOE_W2=1`, FP4 delta 1 GiB,
MTP k=2, cudagraphs FULL_AND_PIECEWISE, fp8 KV, block 256, `max_num_seqs` 4, mnbt 1024;
PRO 6000 runs at 24576 ctx, 5090 runs at 16384 ctx). Old fork measured on its live legacy
containers the same day, confidence gate disabled for the baseline. Tools: `tools/bench_tok.py`
(single‑stream decode, 512 tok, median of 5) and a unique‑prefix prefill probe (8k tokens,
median of ≥3). MTP acceptance was identical across bases (~2.6 tok/step), so deltas are
step‑time, not speculation luck.

| config | metric | fork v0.19 | port v0.24 | Δ |
|---|---|---:|---:|---|
| 1× RTX PRO 6000 | decode | 127.5 tok/s | **161.2** | **+26%** |
| 1× RTX PRO 6000 | prefill 8k | 2 309 tok/s | **4 847** | **+110%** |
| 1× RTX PRO 6000 | decode conc‑3 | ~254 | ~289 | +14% (noisy) |
| 2× RTX PRO 6000 TP2 | decode | 177.3 tok/s | **209.6** | **+18%** |
| 2× RTX PRO 6000 TP2 | prefill 8k | 3 342 tok/s | **5 791** | **+73%** |
| 2× RTX PRO 6000 TP2 | decode conc‑3 | ~310 | ~380 | +23% |
| 4× RTX 5090 TP4 | decode | 106.7 tok/s | **214.4** | **+101%** |
| 4× RTX 5090 TP4 | prefill 8k | 3 765 tok/s | **5 561** | **+48%** |
| 4× RTX 5090 TP4 | decode conc‑3 | ~206 | ~430 | ~2× |

(Old‑fork TP2 note: the fork disables CUDA graphs on TP unless its
`VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH=1` env is set — with graphs off it decodes at ~12 tok/s.
The number above is the fair graphs‑on config, matching `tools/serve.sh`.)

The prefill gain is upstream's FlashInfer SM120 sparse‑MLA prefill replacing the fork's Triton
path — it also makes the planned cubit MLA‑prefill retarget mostly moot. The w2 GEMM cubins are
identical on both sides.
