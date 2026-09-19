# Qwen3.8-Flash-Next on SM120 — small vLLM patches

Host-side fixes for the Qwen3.8-Flash-Next-FP8 deployment on 4× RTX PRO 6000 (official
`vllm/vllm-openai` nightly `v0.1.dev20073+g8e685d198`, TP4, MTP k=3): runtime patches applied
to the installed vLLM package (bind-mounted by the launcher), a Triton MoE config, and one CUDA
kernel (`moe_gemv/`, JIT-compiled in the container) for the routed experts at decode.

| file | what it does |
|---|---|
| `patch_short_conv_attn.py` | replaces five pageable `tensor.to(device)` copies in `ShortConvAttentionMetadataBuilder.build()` (PLE short-conv metadata) with pinned non-blocking copies. The pageable copies synchronized the stream on every decode step; measured effect on `.206` (2026-09-18): worker main thread no longer stalls in the builder, `nvidia-smi` util 84 % → 94 %, 15.3 → 15.0 ms per decode step (+2–3 %). The remaining step time is GPU kernels. |
| `patch_low_latency_gemm.py`, `test_low_latency_gemm.py` | opens vLLM's own CuTe-DSL **skinny GEMM** for this model (`models/qwen3_8_flash_next/nvidia/low_latency_gemm.py`, upstream gated to sm_103/B300) on sm_120 and adds plans tuned on this host (`tools/sm120_perf/skinny_tune.py`) for the token counts and per-rank shapes upstream lacks. ~480 BF16 decode GEMMs per step (GDN/QSA projections, hyper-connection down+inject, router, shared expert, PLE, LM head) leave cuBLAS's `cutlass_80_wmma` M=4 path: 24×2560 10.6 → 1.7 µs, 2560×1536 11.3 → 6.7, 336×10240 9.2 → 6.2, 320×2560 4.7 → 2.6, 512×2560 4.6 → 3.0, 4096×2560 19.6 → 15.4 (1.36 TB/s). **76.7 → 85.2 steps/s (13.0 → 11.7 ms), prose 192 → 206 tok/s, code 269 → 301 tok/s**; needle 27.6K/91.9K PASS, prefill unchanged (13.6k tok/s). Same numerics as cuBLAS (bf16×bf16→fp32). The compiled torch graph changes, so the launcher gives this mode its own `torch_compile_cache` (vLLM's cache hash does not see the patch). `VLLM_MOET_SM120_LL_GEMM=0` or `LL_GEMM=0` reverts. Still on cuBLAS: `10240×320` HC up-projection ×97/step (skinny ties at 6.5 µs, ~1 TB/s), `2560×160` shared-expert down ×48 (K=160 has no valid skinny tile), two single-call shapes. |

## PLE table on the GPUs instead of the CPU offload worker (2026-09-18, +10 % decode, +15 % prefill)

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
(KV usage never exceeded 9 % of the previous 3.6M-token pool). Measured on `.206`:

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

Deployment: since 2026-09-19 everything below is baked into `Dockerfile.sm120-qwen38` (image
`vllm-moet-sm120:qwen38-20073`) and served by `docker/sm120/run-qwen38.sh` — see `docs/sm120-deploy.md`.
The patches stay switchable at run time (`VLLM_MOET_SM120_LL_GEMM=0`, `VLLM_MOET_SM120_MOE_GEMV=0`,
`VLLM_PLE_CPU_OFFLOAD=1`); before that they were bind-mounted into the official image by a host launcher.

## FP8 MoE GEMV for the routed experts (2026-09-18, `moe_gemv/`)

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

In situ (container 8ba42b92, 2026-09-18 22:00Z, `PROFILER=1 ./run.sh` with the new default
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

## Decode timeline of the 2026-09-18 sessions (TP4, MTP k=3, single stream, T=0)

| state | steps/s | ms/step | prose tok/s | code tok/s |
|---|---|---|---|---|
| original `serve.sh` (2026-09-16) | 65 | 15.3 | 137–160 | 228 |
| + short_conv pinned copies, NCCL P2P | 67–70 | 14.2–15.0 | 160–167 | 242–245 |
| + PLE table on the GPUs | 77 | 13.0 | 192 | 270 |
| + skinny GEMM on sm_120 | 85 | 11.7 | 206 | 301 |
| + tuned Triton `fused_moe` config | 89.5 | 11.2 | 216–220 | 316 |
| + FP8 MoE GEMV for the experts | 97.2 | 10.3 | 240 | 341 |
| + silu·up + quant fused into the down GEMV | **98.5** | **10.15** | **243** | **346** |

Concurrency (`ds41-c4` script adapted, ~4.3K-token prompts, 256 out, aggregate tok/s incl. prefill):
C2 114 → 166, C4 207 → 317, C8 323 → 367 with the tuned MoE config (the M = 8/16/32 entries).
This metric depends on the prefix-cache state of the repeated prompts (the same probe gave
150/257/352 cold and 196/315/481 warm after the MoE GEMV); `concurrency_probe.py` with ~200-token
prompts and 256 out (decode-dominated) after the MoE GEMV: C1 108, C2 220, C4 346, C8 537–563 tok/s
aggregate (the synthetic prompt lowers MTP acceptance, hence the low C1).

Profile after the skinny GEMM + MoE config (`profiles/qwen38_llgemm/`, rank 0): Triton
`fused_moe` 23 %, skinny GEMMs ~19 %, remaining cuBLAS 6 %, NCCL steady-state 1.34 ms/step (median
11.3 µs per allreduce, unchanged), elementwise 9 %, lm_head 5 %. After the MoE GEMV
(`profiles/qwen38/`, 10.3 ms/step): MoE GEMV 1.65 ms (16 %), skinny GEMMs ~2.4 ms (23 %),
NCCL 1.32 ms (13 %), remaining cuBLAS `cutlass_80_wmma` 0.88 ms (8.5 %: the 10240×320 HC up ×97 and
2560×160 ×48), lm_head 3 × 212 µs (6 %), elementwise ~1.5 ms, GDN decode 0.43 ms, topk gating
0.32 ms. Note for future profiles: rank 0 shows a few hundred allreduces > 100 µs that ranks 1–3
do not; they cluster in the steps around request boundaries (prefill, first decode step), not in
steady-state decode, and inflate rank 0's NCCL *average* (16–28 µs) without touching the median.

Apply the short-conv patch inside the container (idempotent), then restart the container:

```bash
docker cp patch_short_conv_attn.py qwen38:/tmp/ && docker exec qwen38 python3 /tmp/patch_short_conv_attn.py
docker stop -t 60 qwen38 && docker start qwen38
```

This was the 2026-09-18 hot-fix procedure on a running official-image container; since 2026-09-19
the patch is applied at image build time (`Dockerfile.sm120-qwen38`) and the container is started
by `docker/sm120/run-qwen38.sh` (`PROFILER=1` for `/start_profile`, `NCCL_P2P_LEVEL=SYS` default,
`EXTRA_ARGS`/`EXTRA_DOCKER_ARGS` pass-through).

## Where a decode step goes (TP4, MTP k=3, 2026-09-18 profile, PLE offload still on)

~15 ms per step, ~2 240 kernel launches, rank 0. Of the 23 % "NCCL", ~1.2 ms per step was the
PLE stall described above showing up as the first allreduce's wait. Re-profiled with PLE on the
GPUs (same prose+code window, `profiles/qwen38_ple_gpu/`): NCCL 303 → 129 ms per
~76 steps (34.4 → 14.7 µs per allreduce), the GPU stream has no idle gaps between decode steps
any more, and the per-step composition is BF16 cuBLAS 42 %, Triton `fused_moe` 23 %, NCCL 12 %,
elementwise 10 %, lm_head GEMV 5 %, QSA indexer 3 %.

| share | what |
|---|---|
| 36 % | BF16 cuBLAS GEMMs — the "FP8" checkpoint quantizes only the routed experts; GDN in/out projections, q/k/v/o, hyper-connection mixers (2 × 320×10240 per layer), router, shared expert are BF16 (~16 GEMMs/layer, `cutlass_80_wmma` kernels at ~10 µs + 140 split-K reduce launches) |
| 23 % | NCCL allreduce, 100/step at 35 µs in situ (12–18 µs in isolation: the rest is rank skew) |
| 19 % | `fused_moe_kernel` (Triton, default config, TP-sharded N=160, block scales refined to [32,32]): 2 × 31 µs per layer for 40 (token, expert) pairs — ~2× off the weight-stream bound |
| 9 % | ~700 elementwise launches: `_hc_*` hyper-connection kernels (3/layer × 2), MoE glue (topk gating, silu, moe_sum, quant), GDN conv/state updates |
| 4 % | lm_head GEMV (248K vocab, 219 µs × 3) |
| 2 % | QSA sparse attention + indexer (Triton) — **not** a bottleneck |

Levers, in order (state after the 2026-09-18 sessions: PLE on GPU, skinny GEMM, MoE config and
MoE GEMV done — see the timeline above): the remaining cuBLAS `cutlass_80_wmma` shapes (10240×320
HC up-projection ×97/step at 6.5 µs ≈ 60 % of bandwidth, 2560×160 ×48), launch-count reduction
(fuse the hyper-connection and MoE glue kernels: ~400 launches of ~2 µs), fusing `silu_and_mul` +
the per-token-group quant + `moe_sum` into the MoE GEMV (3 × 49 launches), weight-only MXFP8 for
the BF16 dense layers (needs an accuracy pass), `--enable-expert-parallel` as a config experiment.
`NCCL_P2P_LEVEL=SYS` alone gave +1–2 % (66–67 → 66–69 steps/s). vLLM's custom one-shot allreduce
is not an option on this PCIe topology (pull-based; 75 µs for 60 KiB vs 12.6 µs NCCL, see
`tools/sm120_perf/README.md`).
