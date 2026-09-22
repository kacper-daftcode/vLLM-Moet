# SM120 serving performance tools

Small host-side scripts used in the 2026-09-18 optimisation sessions on the 4× RTX PRO 6000
(TP4) servers. They need only Python 3 (+ `torch`/`vllm` inside the serving image for the
all-reduce benchmark). Findings they produced: `docs/dsv41-sm120-port.md` ("Where a decode
step goes"), `tools/qwen38_sm120/README.md`.

| script | what it does |
|---|---|
| `decode_bench.py BASE MODEL [chat_template_kwargs_json]` | single-stream prose/code decode (tok/s, steps/s, tok/step via streaming chunks) + a ~16K prefill; `API_KEY` env adds the Bearer header |
| `spec_matrix.py --base URL --model M [--concurrencies 1,4,8] [--modes prose,code] [--output-tokens 512] [--repeats 2] [--metrics-url URL/metrics] --out F` | decode matrix for speculative-decoding settings: prose and code prompts at concurrency 1 / 4 / 8 (streaming chat, thinking off, `ignore_eos`): shared-window aggregate tok/s, per-request tok/s, steps/s and tokens per step from the streamed chunks, DSpark accepted-per-step / acceptance from `/metrics` deltas; used for the adaptive-verification and draft-length comparisons (`docs/dsv41-sm120-port.md`, "Tried and not kept") |
| `openai_matrix.py --base URL --model M --model-dir DIR --sizes ... --concurrencies ... --output-tokens N --out F` | burst-serving matrix with 0xSero's `benchmarks/matrix.py` methodology (exact-size token-id prompts, forced output budget, shared decode window) through `/v1/completions`, so vLLM and SGLang are measured by one client; `--metrics-url` reads vLLM's DSpark counters per wave — `docs/dsv41-sm120-vs-0xsero.md` |
| `quality_cmp.py --base URL --model M --out F [--api-key K]` | same-checkpoint fidelity probes: arithmetic (thinking off/on), coherence, 24 greedy answers for cross-stack diff, tool round trip, strict JSON schema, vision smoke, needle at ~29K/106K/400K tokens |
| `compare_outputs.py A.json B.json [--tokenizer tokenizer.json]` | two `quality_cmp.py` results side by side + greedy-output agreement (identical count, first divergence in chars/tokens) |
| `moe_backends_bench.py [--tokens 1,6,16,48,64] [--skip fi,b12x]` | per-layer routed-MoE time at decode shapes: vLLM's DeepGEMM FP8xFP4 chain vs FlashInfer CUTLASS W4A8 fused MoE vs B12x `w4a8_mx` (inside the ds41 image, `pip install b12x`) — `tools/dsv41_sm120/README.md` |
| `dense_b12x_bench.py` | this repo's MXFP8 GEMV vs `b12x.gemm.blockscaled.mm` at the six dense decode shapes, cold L2, identical-numerics check |
| `image_tokens_check.py BASE MODEL [API_KEY]` | image token count of a served DeepSeek-V4.1 stack for six image sizes vs the checkpoint's `inference/image_processor.py` formula |
| `prefill_probe.py BASE MODEL GPUS [ntok] [kwargs]` | prefill on a fresh random prompt (no prefix-cache hit) while sampling `nvidia-smi` for the peak GPU memory |
| `needle_any.py BASE MODEL '{"chat_template_kwargs":{...}}' [targets]` | needle-in-a-haystack at two depths per target length + two coherence prompts |
| `concurrency_probe.py BASE MODEL [C] [prompt_tok] [kwargs]` | C parallel requests (~prompt_tok in, 256 out): per-request times, aggregate tok/s, error count |
| `kv_layout_probe.py --model-dir DIR [--indexer-bytes 132\|68] [--main-bytes 584] [--avail-gib G]` (inside the ds41 image, one GPU) | vLLM's own `get_kv_cache_groups` / `get_kv_cache_config_from_groups` on DeepSeek-V4.1-Flash's KV specs without loading weights: which pages share a physical block, padded page sizes, the packed block stride and its alignments, per-layer view strides, and the token capacity for a given `Available KV cache memory` (reproduces the served 1,490,870 / 2,288,673 tokens to 0.1 %) — the check to run before changing a cache record width |
| `trace_nccl_dist.py label=GLOB ...` | allreduce duration distribution per rank trace (median/p90/p99, long waits and what precedes them, NCCL sum per step) — separates steady-state cost from request-boundary waits |
| `profile_capture.py BASE MODEL [kwargs] [n]` | `/start_profile` → prose + code decode → `/stop_profile` (server started with `--profiler-config.profiler torch`) |
| `trace_agg.py TRACE.json.gz [top_n]` | GPU time by kernel category / top kernels for one rank trace |
| `trace_step_skew.py DIR [gap_us] [min_kernels]` | per rank: steps (kernel runs separated by idle gaps), first-NCCL-of-step vs rest, idle before step; across ranks: step-start skew and first-NCCL duration per rank |
| `trace_gaps.py TRACE.json.gz [gap_us] [n]` | GPU idle gaps: classes by (kernel before → kernel after), whether the gap sits inside one `cudaGraphLaunch` |
| `trace_gap_context.py TRACE.json.gz BEFORE_PREFIX [gap_us] [n] [ctx]` | kernels, CPU runtime calls and ops around gaps of one class |
| `allreduce_bench.py` (torchrun, 4 ranks, inside the vLLM image) | pynccl vs vLLM `CustomAllreduce` one-shot with the "fully connected" gate forced open, eager and in a CUDA graph |
| `skinny_tune.py [shapes] [Ms]` (inside the qwen38 image) | grid-tunes vLLM's CuTe-DSL `SkinnyGemmConfig` per (N, K, M) against cuBLAS, cold L2; prints the plan dict consumed by `tools/qwen38_sm120/patch_low_latency_gemm.py` |
| `moe_tune.py OUT.json [Ms]` (inside the qwen38 image) | small-grid Triton `fused_moe` tuning for E=512,N=160,fp8 [32,32] via `benchmark_moe.benchmark_config` (ray stubbed out); writes the vLLM config JSON with decode entries + default-equivalent large-M entries |
| `trace_graphs.py TRACE.json.gz [--graph N] [--order]` | kernels grouped by `cudaGraphLaunch` (correlation id): clusters of launches by kernel count (decode graph, drafter graph, prefill pieces), per-cluster kernel composition and launch order — how the DSpark drafter graph (167 kernels, 1.18 ms) was broken down |
| `dg_moe_blockm_bench.py` (inside the ds41 image) | vLLM's DeepGEMM FP8×FP4 MoE chain (permute → FC1 → silu·up+quant → FC2 → gather) on the served decode shape under different M alignments / `mk_alignment_scope` values, cold expert weights; `--save-out/--compare-out` for bit-exact comparisons of patched glue kernels |

How the qwen38 PLE finding was made (2026-09-18): `trace_step_skew.py` showed the first NCCL
kernel of every step waiting 0.5–1.1 ms and a 1.2 ms step-start skew between ranks;
`trace_gaps.py` showed the dominant idle gap (1.16 ms median, once per step) *inside* the main
CUDA graph between `_hc_combine_kernel` and the next kernel — i.e. a non-kernel graph node;
`trace_gap_context.py` confirmed the CPU was a full step ahead, which left the PLE CPU-offload
`cuStreamWaitValue32` node as the only candidate. Moving the PLE table onto the GPUs removed
the gap (see `tools/qwen38_sm120/README.md`).

`allreduce_bench.py` result on the KVM host (PCIe P2P, no NVLink): vLLM's one-shot custom
all-reduce is *pull*-based (every rank reads its peers' buffers) and PCIe P2P reads are
latency-bound here — 60 KiB takes 75 µs vs 12.6 µs for NCCL LL over P2P, growing linearly with
size. Not usable on this topology; the FlashInfer PCIe-IPC backend (`PcieIpcAllReduceWorkspace`,
push-based) is what vLLM `>=` dev20904 would prefer, but FlashInfer 0.6.18 does not ship it.
