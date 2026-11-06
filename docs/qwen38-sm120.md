# Qwen3.8-Flash-Next-FP8 on 4× RTX PRO 6000 (sm_120)

The official checkpoint ([`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8),
177B-A6B, ~180 GB) on the official `vllm/vllm-openai` nightly that serves it (`v0.1.dev20073`, pinned
by digest), TP4, MTP k = 3, 256K context, vision on. [`Dockerfile.sm120-qwen38`](../Dockerfile.sm120-qwen38)
keeps the model's numerics and changes where the time goes; the write-up with the measurements is
[`tools/qwen38_sm120/README.md`](../tools/qwen38_sm120/README.md).

| change | switch | effect |
|---|---|---|
| the 51 GB n-gram PLE table TP-sharded on the GPUs instead of the CPU offload worker (whose hand-off stalled every decode graph by ~1.2 ms) | `VLLM_PLE_CPU_OFFLOAD=1` reverts | +10 % decode, +15 % prefill; +12 GB of weights per GPU, so the launcher sets `--kv-cache-memory` explicitly |
| the PLE short-conv metadata copies pinned and non-blocking (the pageable copies synchronized the stream every step) | — | fewer stalls per step |
| vLLM's own CuTe-DSL skinny GEMM (gated to B300 upstream) for the ~480 BF16 decode GEMMs per step, with plans tuned on this card | `VLLM_MOET_SM120_LL_GEMM=0` (use a fresh `CACHE_DIR`) | — |
| an FP8 [32, 32] block-scaled MoE GEMV for the routed experts at decode token counts, silu·up and the quantization fused into the down GEMV; a tuned Triton `fused_moe` config for prefill | `VLLM_MOET_SM120_MOE_GEMV=0` | bit-identical to Triton |

**Measured** (4× RTX PRO 6000, TP4, MTP k = 3): **98.5 steps/s, prose 243 / code 346 tok/s single
stream** against 65 steps/s and 137–160 / 228 tok/s of the unmodified nightly; 2.28M-token KV at 256K
context with `--kv-cache-memory 32GiB`; needle PASS at 27.6K / 91.9K.

## Build and serve

```bash
hf download Qwen/Qwen3.8-Flash-Next-FP8 --local-dir /srv/models/Qwen3.8-Flash-Next-FP8
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-qwen38 -t vllm-moet-sm120:qwen38-20073 .     # ~4 min
MODEL_DIR=/srv/models/Qwen3.8-Flash-Next-FP8 CACHE_DIR=/srv/cache/qwen38 GPUS=4,5,6,7 docker/sm120/run-qwen38.sh
```

The first start compiles for ~10 minutes (torch.compile, FlashInfer JIT) into `CACHE_DIR` — keep one
cache directory per image tag; later starts take ~4 minutes. The API listens on `127.0.0.1:8000`.
Every knob is an environment variable documented in the header of
[`docker/sm120/run-qwen38.sh`](../docker/sm120/run-qwen38.sh). Validate with
`tools/sm120_perf/decode_bench.py http://127.0.0.1:8000 Qwen3.8-Flash-Next '{"enable_thinking": false}'`
(expected 97–99 steps/s) and `tools/sm120_perf/needle_any.py`. In-image kernel test:
`python3 /opt/vllm-moet/qwen38_sm120/moe_gemv/test_fused_moe_integration.py` (one GPU).

Image and video prompts were not stress-tested for memory; drop `KV_CACHE_MEMORY` to 28 GiB if
multimodal traffic OOMs.
