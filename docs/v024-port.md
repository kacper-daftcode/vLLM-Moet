# The v0.24.0 port

The project targets **official vLLM v0.24.0**, which ships DeepSeek‑V4 + SM120 natively
(`vllm/models/deepseek_v4/`, FlashInfer SM120 sparse‑MLA, GLM‑5.x `GlmMoeDsaForCausalLM`).
Our overlay is a **3.7k‑line patch**: the 2‑bit expert planes, the FP4 delta cache, the
confidence gate, the cubit dispatch — plus the SM120 fixes below.

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
costs ~5% more.

## Benchmarks (2026‑07‑08)

Official FP4 checkpoint, `VLLM_MOE_W2=1`, FP4 delta 1 GiB, MTP k=2, cudagraphs
FULL_AND_PIECEWISE, fp8 KV, block 256, `max_num_seqs` 4, mnbt 1024; PRO 6000 runs at
24576 ctx, 5090 runs at 16384 ctx. Tools: `tools/bench_tok.py` (single‑stream decode,
512 tok, median of 5) and a unique‑prefix prefill probe (8k tokens, median of ≥3; unique
prefixes defeat the prefix cache). MTP acceptance ~2.6 tok/step in every config.

| config | decode | prefill 8k | decode conc‑3 (aggregate) |
|---|---:|---:|---:|
| 1× RTX PRO 6000 | **161.2 tok/s** | **4 847 tok/s** | ~289 tok/s |
| 2× RTX PRO 6000 TP2 | **209.6 tok/s** | **5 791 tok/s** | ~380 tok/s |
| 4× RTX 5090 TP4 | **214.4 tok/s** | **5 561 tok/s** | ~430 tok/s |

Prefill rides upstream's FlashInfer SM120 sparse‑MLA path — which also makes a custom cubit
MLA‑prefill kernel unnecessary on this base.

**Long context (512K) on one 96 GB card** — validated live (`tools/needle_probe.py`, unique
secret embedded in filler, greedy): PASS at **102 238 / 256 294 / 453 286 prompt tokens**
(depth 0.1) and at 453K with the needle mid‑context (depth 0.5); cold TTFT 27 s / 64 s /
~2 min. Server config for the 512K window on 1× PRO 6000: `--max-model-len 524288
--gpu-memory-utilization 0.97 --max-num-batched-tokens 2048 --max-num-seqs 1` with
`VLLM_MOE_W2=1 VLLM_MOE_W2_DELTA_GB=0` (the FP4 delta pool trades against KV headroom at
extreme context; with delta 1 GiB use ≤256K). The KV fit comes from DS4's compressed KV +
upstream's FP8 Lightning‑Indexer cache; vLLM reports 947K cached tokens in this config.
