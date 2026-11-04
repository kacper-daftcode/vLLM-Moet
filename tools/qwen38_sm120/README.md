# Qwen3.8-Flash-Next on SM120 — small vLLM patches

Host-side fixes for the Qwen3.8-Flash-Next-FP8 deployment on 4× RTX PRO 6000 (official
`vllm/vllm-openai` nightly `v0.1.dev20073+g8e685d198`, TP4, MTP k=3): runtime patches applied
to the installed vLLM package (bind-mounted by the launcher), a Triton MoE config, and one CUDA
kernel (`moe_gemv/`, JIT-compiled in the container) for the routed experts at decode.

| file | what it does |
|---|---|
| `patch_short_conv_attn.py` | replaces five pageable `tensor.to(device)` copies in `ShortConvAttentionMetadataBuilder.build()` (PLE short-conv metadata) with pinned non-blocking copies. The pageable copies synchronized the stream on every decode step; measured effect: worker main thread no longer stalls in the builder, `nvidia-smi` util 84 % → 94 %, 15.3 → 15.0 ms per decode step (+2–3 %). The remaining step time is GPU kernels. |
| `patch_low_latency_gemm.py`, `test_low_latency_gemm.py` | opens vLLM's own CuTe-DSL **skinny GEMM** for this model (`models/qwen3_8_flash_next/nvidia/low_latency_gemm.py`, upstream gated to sm_103/B300) on sm_120 and adds plans tuned on this host (`tools/sm120_perf/skinny_tune.py`) for the token counts and per-rank shapes upstream lacks. ~480 BF16 decode GEMMs per step (GDN/QSA projections, hyper-connection down+inject, router, shared expert, PLE, LM head) leave cuBLAS's `cutlass_80_wmma` M=4 path: 24×2560 10.6 → 1.7 µs, 2560×1536 11.3 → 6.7, 336×10240 9.2 → 6.2, 320×2560 4.7 → 2.6, 512×2560 4.6 → 3.0, 4096×2560 19.6 → 15.4 (1.36 TB/s). **76.7 → 85.2 steps/s (13.0 → 11.7 ms), prose 192 → 206 tok/s, code 269 → 301 tok/s**; needle 27.6K/91.9K PASS, prefill unchanged (13.6k tok/s). Same numerics as cuBLAS (bf16×bf16→fp32). The compiled torch graph changes, so the launcher gives this mode its own `torch_compile_cache` (vLLM's cache hash does not see the patch). `VLLM_MOET_SM120_LL_GEMM=0` or `LL_GEMM=0` reverts. Still on cuBLAS: `10240×320` HC up-projection ×97/step (skinny ties at 6.5 µs, ~1 TB/s), `2560×160` shared-expert down ×48 (K=160 has no valid skinny tile), two single-call shapes. |

## PLE table on the GPUs instead of the CPU offload worker (+10 % decode, +15 % prefill)

The original `serve.sh` ran with `VLLM_PLE_CPU_OFFLOAD=1`: the model's n-gram PLE embedding
(one layer, 128 shards × [2.5M × 160] fp8 = **51 GB**) lives in a separate CPU process that
gathers the rows for each step and writes them into the GPU workers' buffers over CUDA IPC. The
GPU side is a `cuStreamWaitValue32` node inside the decode CUDA graph, so the graph stalls until
the CPU chain completes: sample on GPU → D2H of the input ids → zmq to the worker → gather →
IPC H2D to four GPUs → four flags, set one after another.

What the traces showed (`tools/sm120_perf/`): the GPU sat idle **1.1–1.6 ms once per step**
(median 1.16 ms) at that node — with the CPU already a full step ahead — and because the flags
arrive staggered, the ranks started each step 1.2 ms apart (median), which the first NCCL
allreduce of the step absorbed as a 0.5–1.1 ms wait. The earlier "PLE worker is 0.5 % busy, not
a bottleneck" reading measured occupancy, not latency.

`VLLM_PLE_CPU_OFFLOAD=0` (vLLM's default) keeps the table on the GPUs as a `VocabParallelEmbedding`
sharded over TP: +12 GB of weights per GPU (33.6 → 45.5 GiB), which this deployment can afford
(KV usage never exceeded 9 % of the previous 3.6M-token pool). Measured:

| | PLE offload (CPU worker) | PLE on GPUs |
|---|---|---|
| decode steps/s | 70 | **77** (14.2 → 13.0 ms/step) |
| prose / code tok/s | 167 / 245 | **192 / 270** |
| prefill (fresh 32K prompt) | 11.5k tok/s | **13.2k tok/s** (120K: 12.5k) |
| weights per GPU | 33.6 GiB | 45.5 GiB |
| KV pool | 3.6M tokens | 2.28M tokens with `--kv-cache-memory 32GiB` (8.7× @262K) |

Needle 27.6K/91.9K at two depths PASS, outputs coherent. One catch: vLLM's start-up memory
profile of the GPU-PLE path reports a **35 GiB "peak activation"** (a compile-time transient — real
32K–120K prefills leave `nvidia-smi` flat at 58.5 GB), which under `--gpu-memory-utilization 0.92`
would have left 5.6 GiB of KV. The launcher therefore passes `--kv-cache-memory` explicitly
(`KV_CACHE_MEMORY`, default 32 GiB when `PLE_OFFLOAD=0`; 84 GB/GPU in use, ~14 GB spare).

| `moe_configs/E=512,N=160,…,block_shape=[32,32].json` | Triton `fused_moe` tile config for the TP4 expert shards (vLLM had none for this device and logged "Using default MoE config"). vLLM caps `BLOCK_SIZE_K` at 32 for [32,32] block scales, so only BLOCK_N / warps / stages can move; tuned with `tools/sm120_perf/moe_tune.py` for decode token counts, default-equivalent entries for M ≥ 96. Warm-L2 per layer: M=4 44.7 → 39.6 µs, M=8 74.7 → 57.0, M=16 288 → 203, M=64 764 → 621. Mounted via `VLLM_TUNED_CONFIG_FOLDER` (launcher `MOE_CONFIG=1`). Still the path for prefill and batches above 32 tokens; decode goes to the GEMV below. |
| `moe_gemv/`, `patch_fused_moe_sm120.py` | **FP8 block-scaled MoE GEMV** for the routed experts at decode token counts (see below). |

Deployment: everything here is baked into `Dockerfile.sm120-qwen38` (image `vllm-moet-sm120:qwen38-20073`)
and served by `docker/sm120/run-qwen38.sh` — see `docs/qwen38-sm120.md`. The patches stay switchable at run
time (`VLLM_MOET_SM120_LL_GEMM=0`, `VLLM_MOET_SM120_MOE_GEMV=0`, `VLLM_PLE_CPU_OFFLOAD=1`).

## FP8 MoE GEMV for the routed experts (`moe_gemv/`)

The experts are the only FP8 weights in the checkpoint and the largest item of the decode step
(Triton `fused_moe_kernel` 23 % after the config tuning). vLLM refines the checkpoint's [128,128]
block scales to **[32, 32]** because the TP4 shard N = 160 is not a multiple of 128, and then caps
the Triton kernel's `BLOCK_SIZE_K` at 32: a 4-token step (MTP k=3, 40 (token, expert) pairs) runs
~200 programs of 2 warps, each doing 80 serial K-steps of `tl.dot` 16×32×64 plus a per-step scale
multiply — about half of the HBM bandwidth for the ~33 MB of gate/up weights it streams.

`moe_gemv/fused_moe_gemv_sm120.cu` computes the same thing (mma.sync m16n8k32 e4m3 → fp32, one
MMA per 32-wide scale block, fp32 accumulation scaled by `a_s[row] · b_s[col block]` exactly like
the Triton kernel, so the outputs are bit-identical up to fp32 summation order) with the mapping of
the sm_120 MXFP8 dense GEMV (`tools/dsv41_sm120/sm120_gemv/`): a block owns one m-block and
8·(8/KSPLIT) output columns, the 8 warps split into KSPLIT K-slices × column groups, every lane
issues UNROLL 8-byte weight loads before the first MMA. Two block shapes match vLLM's two token
assignments:

- **naive** (vLLM skips `moe_align_block_size` when `M·topk·4 ≤ E`, i.e. M ≤ 12 tokens here): one
  pair per block, grid = pairs × N/8 (1600 blocks for M=4), K split over the 8 warps (`8x5`);
- **aligned** (M ≥ 13): one 16-row `sorted_token_ids` block per block, the A tile [16, K] and its
  scales staged once in shared memory (padded strides, conflict-free 64-bit fragment loads), 8
  column groups per block (`1x4`). The K = 160 down GEMM stays on Triton in this mode: Triton's
  2-warp programs keep more weight bytes in flight per SM than 256-thread blocks with 5 K-blocks
  of work (28 vs 23 µs at 16 tokens).

Cold-L2 per layer (`test_fused_moe_gemv_sm120.py`, 12 rotating routings inside a CUDA graph, the
Triton side with the tuned config above), both expert GEMMs:

| tokens | Triton gate/up → GEMV | Triton down → GEMV | per layer |
|---|---|---|---|
| 1 | 14.9 → 6.8 µs | 4.8 → 3.4 µs | 19.8 → 10.2 µs |
| 2 | 25.7 → 11.6 | 5.8 → 4.9 | 31.5 → 16.5 |
| **4** (C1, MTP k=3) | **41.3 → 20.8** | **8.7 → 8.3** | **50.0 → 29.0** |
| 8 | 67.2 → 34.9 | 14.7 → 14.1 | 81.9 → 49.0 |
| 12 | 100.7 → 51.1 | 19.9 → 19.5 | 120.6 → 70.6 |
| 16 (aligned) | 128.2 → 71.5 | 23.4 (Triton kept) | 151.6 → 94.9 |
| 32 (aligned) | 319.8 → 145.1 | 39.6 (Triton kept) | 359.4 → 184.7 |

At 4 tokens the gate/up GEMV streams 33 MB in 20.8 µs (1.6 TB/s) — the weight bytes are the floor;
the down GEMM (16 MB) was already there. Max |GEMV − Triton| ≤ 1 bf16 ulp at every M, error vs an
fp32 reference on the same quantized operands identical to Triton's (3.8e-3).

In situ (`PROFILER=1`, the new default
`MOE_GEMV=1`): **89.6 → 97.2 steps/s (11.2 → 10.3 ms/step), prose 220 → 240 tok/s, code 317 → 341
tok/s**, MTP acceptance unchanged (2.47 / 3.51 tok/step); needle 27.6K/91.9K PASS, coherence and
code prompts fine, fresh 64K prefill 12.9k tok/s, 87.7 GB/GPU. Profile (`profiles/qwen38/`,
rank 0): `fused_moe_gemv_kernel<8,5,false>` 20.7 µs and `<1,5,false>` 10.3 µs per layer (Triton
was 28 + 8.5), MoE share 23 → 16 % of the step, NCCL steady state unchanged (median 11.2 µs,
1.32 ms per step). The old container is kept as `qwen38-moecfg-20260918`; `MOE_GEMV=0 ./run.sh`
is the rollback.

Integration: `patch_fused_moe_sm120.py` adds one early dispatch at the top of vLLM's
`invoke_fused_moe_triton_kernel` (fp8_w8a8, block_shape [32,32], bf16 output, no bias, ≤ 320
pairs → GEMV, else unchanged); the module is imported from `/opt/vllm-moet/moe_gemv` (bind-mount)
and JIT-compiled once into `~/.cache/torch_extensions` (also bind-mounted, so the 4 TP workers
find the build). The MoE forward is already a custom op (`moe_forward`), so the compiled graph
does not change and no separate `torch_compile_cache` is needed. Knobs: `VLLM_MOET_SM120_MOE_GEMV=0`
(runtime off switch), `…_MAX_PAIRS` (320), `…_ALIGNED_MIN_K` (512), `…_CFG` (`K2560:8x5,K2560a:1x4,K160:1x5`),
`…_FUSE_ACT=0` (see below). `test_fused_moe_integration.py` checks the dispatch decisions and the
equality with Triton through the patched module.

**Fused activation in the down GEMM** (same session, second file patched: `experts/triton_moe.py`,
`TritonExperts.apply`): at decode token counts the down GEMV takes the bf16 gate/up output of the
first GEMM directly — every block computes `bf16(silu(gate)) · up`, the per-32-group absmax, the
scale (rounded up to a power of two because vLLM quantizes activations with UE8M0 scales when
DeepGEMM's E8M0 mode is on, `is_deep_gemm_e8m0_used()`, the case on this host), and the fp8 values
into shared memory, then runs the GEMM from there. Bit-identical to `act_and_mul` +
`per_token_group_quant_fp8` + GEMV (0 differing output elements at M = 1…12 in
`test_fused_moe_gemv_sm120.py`; the extension is built without `--use_fast_math` for that), two
launches less per layer: down path 12.7 → 8.5 µs at M=4 (6.9 → 3.9 at M=1, 18.9 → 14.7 at M=8).
The patched `triton_moe.py` imports the hook from the patched `fused_moe.py` and is a no-op
without it; LoRA, static `a2_scale`, emulation and the aligned (≥ 13 tokens) path keep vLLM's
sequence.

In situ (container 5d190863, 22:35Z): **98.5 steps/s (10.15 ms/step), prose 243 / code 346 tok/s**,
acceptance and needle answers identical to the unfused GEMV run (bit-identical path); profile
`profiles/qwen38/`: `fused_moe_gemv_kernel<1,5,false,true>` 11.2 µs per layer replaces
`act_and_mul` 1.4 + `per_token_group_quant` 2.2 + GEMV 10.3 µs and two launch gaps; the previous
profile moved to `qwen38_moegemv1/`. Concurrency probe (4.3K prompts, 256 out) cold 175/248/356,
warm 216/313/410 tok/s aggregate for C2/C4/C8; ~200-token prompts 208/373/559.

Gotcha found the hard way: vLLM keeps `intermediate_cache1` (gate/up output) and
`intermediate_cache3` (down output) in the *same* workspace buffer ("done with cache1 by the time
we need cache3"). The fused kernel reads cache1 while writing cache3, so the first deployment
silently corrupted activations — steps/s looked right but MTP acceptance dropped (2.47 → 2.20
tok/step) and the 92K needle failed while the unit test (separate tensors) was bit-exact. The patch
now places cache1 in `workspace13` when the fused path is taken (it is sized `(M, topk, max(N, K))`;
the MoE output aliases its start but is written only by the final `moe_sum`), and
`test_fused_moe_integration.py` runs the whole `TritonExperts.apply` with vLLM's workspace layout
(fused vs unfused: 0 differing elements).
