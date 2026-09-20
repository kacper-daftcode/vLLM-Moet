# The DeepSeek-V4.1-Flash SM120 port

**Target:** the official `vllm/vllm-openai:deepseekv41-flash-0909` image — the tag the
[vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml)
pins for DeepSeek‑V4.1‑Flash on NVIDIA — serving the **official checkpoint**
(`deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a`, 510 GB: MXFP8 dense, MXFP4 experts, UE8M0
scales) on **RTX PRO 6000 Blackwell (sm_120)**. The served configuration is **four cards, TP4,
512K** (the default of `docker/sm120/run-dsv41.sh`; measurements from the
[TP4 section](#tp4-4-rtx-pro-6000--fits-thinly) onwards); the gap inventory and the end‑to‑end
validation in the first half of this document are on eight cards (TP8, 1M window), which the same
image also serves. The recipe lists H200/GB200/GB300/MI350X as
verified; on sm_120 the image fails at engine start. This port closes the gaps **kernel‑side**,
in the style of the v0.24.0 port: the official image plus generated/idempotent patches on the
two kernel libraries, precompiled into `Dockerfile.sm120-dsv41`. vLLM code is untouched — the
model's page geometry on sm_120 is identical to the SM100 path; only the kernels learned the
shapes.

Everything in this document was measured on this box (8× RTX PRO 6000 96 GB PCIe, no NVLink,
1× RTX 5090 for op‑level validation, driver 610.43, CUDA 13.3 UMD) on 2026‑09‑16.

## What was missing on sm_120 (the gap inventory)

Method: an instrumented dry run of the stock image (overlay on `flashinfer/mla/_sparse_mla_sm120.py`
and `vllm/utils/deep_gemm.py` that logs every sparse‑MLA / paged‑MQA call and zero‑fills the
shapes the kernels reject) so the forward reaches all 40 layers, DSpark, prefill and the vision
path. 18 shape classes; the inventory is `internal/`‑only
(`/mnt/nvme2/dryrun/calls-attempt3-final.jsonl` on this host).

vLLM serves V4.1 with: SWA cache page **32** tokens (`DeepseekV4SWACache(block_size=32)`),
KV block 128 → compressed cache **128 states/page on compress_ratio=1 layers (20–39)** and 64 on
ratio‑2 layers (2–19), SWA index rows **128** (window) / **192** (DSpark non‑causal, k=5) /
**1152** (prefill: `window_size + max_image_tokens` = 128 + 1024, vision‑padded on every prefill
row), extra (compressed) top‑k 512, 8 query heads at TP8.

| # | gap | stock behaviour | fix |
|---|---|---|---|
| 1 | FlashInfer SM120 DSV4 decode kernels are `PAGE_BLOCK_SIZE=64` only | decode rows with page 32: `SM120 sparse-MLA has no decode kernel for this shape` | PBS=32 decode instantiations, TOPK ∈ {128, 192, 256, 1152}, NH ∈ {8, 16, 32, 64} |
| 2 | FlashInfer SM120 DSV4 prefill (single + dual) is PBS=64 only; dual admits extra pages {64, 2}; SWA rows {128..2048}, dual 128 only | ratio‑1 layers: `Unsupported sparse-MLA prefill configuration … page_block_size=32 … extra_page_block_size=128`; rows 1152 rejected everywhere | new TU `sparse_mla_sm120_dsv41.cu`: PBS=32 × TOPK {128, 192, 1152} single; PBS=32 × PBSX {64, 128} × TOPK {128, 192, 1152} dual (+ full‑tile 128); FP8 compute mode at 1152 like the stock ≥512 table |
| 3 | `dispatch_dsv4_single` (and the dual PBSX=64 table) never looked at `page_block_size` | SWA‑only and ratio‑2 layers **silently ran the 64‑page kernel over 32‑token pages** (wrong addresses, no error) | orchestrator hook: `ModelType::DSV4 && page_block_size != 64` → DSV4.1 table or a loud failure |
| 4 | small prefill batches (≤ 64 tokens) with 1152‑wide rows take the decode entry | `no decode kernel … topk=1152` | decode TOPK=1152 at PBS=32 (item 1) |
| 5 | DeepGEMM SM120 FP8 paged MQA logits (indexer) + its metadata: `block_kv == 64` only; the MXFP4 sibling `block_kv ∈ {32, 64}` | ratio‑1 indexer layers (128 states/page): `Assertion error attention.hpp:262: block_kv == 32 or block_kv == 64`, `:320 … block_kv == 64` | host asserts admit 128 for the FP8 and the MXFP4 cache (`patch_deepgemm.py`); device kernels/scheduler are templated on `BLOCK_KV` (128 rows = one group of eight 16‑row MMA warps, `SPLIT_KV` stays 128, ~71 KB smem for FP8, ~37 KB for MXFP4) |
| 5b | vLLM gates `--attention-config '{"indexer_kv_dtype":"mxfp4"}'` on sm_10x | `indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs` | `patch_vllm_indexer_fp4_sm120.py` admits sm_120; see [the MXFP4 indexer section](#the-indexer-in-its-training-format-mxfp4-k-cache-on-sm_120-20260920) |
| 6 | DSpark `enable_adaptive_verification:true` (recipe's NVIDIA default) | `DeepseekV4IndexerBackend … does not support` adaptive verification on this path | serve with `enable_adaptive_verification:false` (the recipe's AMD override); DSpark still drafts 5 tokens |

Not gaps (verified working on sm_120 in this image): DeepGEMM MXFP4 MoE (`DEEPGEMM_MXFP4` /
`DeepGemmFP4Experts`), MXFP8 dense GEMM (`FlashInferCutlassMxfp8LinearKernel`; a few shapes fall
back to `EmulationMxfp8LinearKernel` — a perf item, not a blocker), `fp8_ds_mla` KV format, FP8
indexer cache (and, with 5/5b, the MXFP4 one), Mega‑mHC TileLang kernels, Engram in pinned host
RAM (2 × 11.8 GiB per rank), ViT with FLASH_ATTN, DSpark drafter, cudagraphs FULL_AND_PIECEWISE.

## The kernels

`tools/dsv41_sm120/sparse_mla_sm120_dsv41.cu` — a new translation unit for FlashInfer's
`sparse_mla_sm120` JIT module. It instantiates the existing `sparse_mla_prefill_mg_kernel`,
`sparse_mla_prefill_mg_dual_kernel` and `sparse_mla_prefill_mg_dual_fulltile_kernel` templates
for the V4.1 geometry and exposes `sparse_mla_prefill_dispatch_dsv41(...)`. Nothing in the kernel
bodies changed: `PAGE_BLOCK_SIZE` / `PAGE_BLOCK_SIZE_EXTRA` only enter the page‑address arithmetic
(`idx / PBS`, `idx % PBS`, scale footer at `PBS * 576`), `TOPK` only sets the index‑row stride and
the tile count (1152 / 64 = 18 tiles). The launchers mirror the anonymous‑namespace launchers of
`sparse_mla_sm120_prefill.cu`. Decode PBS=32 lives in the stock decode TU via the extended
`DSV4_DISPATCH_PBS` table (the extra page was already a runtime argument there).

`tools/dsv41_sm120/patch_flashinfer.py` — idempotent, anchored patcher for the installed
flashinfer package: installs the TU, adds it to `jit/mla.py`, hooks the orchestrator, extends the
decode table, and lets `_sparse_mla_sm120.py` route page‑32 decode rows to the decode kernels.

`tools/dsv41_sm120/patch_deepgemm.py` — the three host‑side asserts in DeepGEMM
`8b1392b978f5a03c828dd1711090d7fb50958b8a` (deepseek‑ai nv_dev tip; byte‑identical to the headers
vendored in the image, checked by sha256 of the SM120 MQA headers). `_C` is rebuilt inside the
official image (Python 3.12 / torch 2.13+cu130 ABI by construction) — the CUDA headers and
`libnvrtc` come from the pip `nvidia/cu13` wheels, no apt CUDA dev packages.

Module compile time for the whole `sparse_mla_sm120` module with the new TU: ~18 s on this box
(`sm_120f`); `_C`: ~40 s.

## Validation

Op level (`tools/dsv41_sm120/`, run on the RTX 5090 inside the image):

- `test_sparse_mla_sm120_dsv41.py` — **372/372**. Every case vs the dense torch reference at the
  upstream tolerances (atol = rtol = 5e‑2; observed max |Δ| ≤ 0.004), and a **re‑paging parity**
  check: the same logical tokens and flat indices with the SWA cache at 32 tokens/page (new
  kernels) vs 64 (stock) — **188 cases bit‑exact**, including decode PBS=32, prefill 128/192,
  1152 (vs stock 2048 FP8 with `topk_length`) and dual PBSX=128 (vs stock PBSX=64 on the extra
  cache re‑paged). Dual 192/1152 have no stock partner and pass the reference.
- `test_deepgemm_sm120_paged_mqa.py` — **8/8**. `block_kv` 64 vs 128 on the same logical KV:
  **bit‑exact**; vs the fp8‑simulated reference of DeepGEMM's own test: rel. diff ≈ 2.5e‑6
  (upstream gate 1e‑3); 20 launch self‑consistency as upstream.

End to end (official checkpoint, TP8 on 8× RTX PRO 6000, `--max-model-len 1048576`, DSpark k=5,
fp8 KV, `--gpu-memory-utilization 0.90`):

| item | result |
|---|---|
| weights | 50.2 GiB/GPU, 200 s cold / 100 s warm page cache; Engram 2 × 11.8 GiB pinned per rank |
| KV pool | 18.9 GiB/GPU after graphs → **7,509,385 tokens**, 7.16× concurrency at 1M |
| `17*19`, thinking off | `323` |
| thinking (`reasoning_effort=10`) | coherent Polish, 521 reasoning tokens |
| tool calling (`deepseek_v41` DSML parser) | `get_weather({"city": "Kraków"})`, `finish_reason=tool_calls` |
| vision (`inference/examples/images`) | carrots / corn / both in one prompt (660 prompt tokens) |
| needle retrieval | **PASS at 30,940 and 116,471 prompt tokens** |
| prefill | ~8.9k tok/s (single request, incl. decode of the answer) |
| decode, single stream | **150.9 tok/s** (512 tokens); DSpark accepted 726 of 2,200 drafted over 440 steps (~2.65 tok/step) |

## TP4 (4× RTX PRO 6000) — fits, thinly

Engram lives in host RAM either way, so the GPU‑resident part is 286 GiB (experts 259.5 + 16.2
scales, dense/attention ~6.6, embeddings/vision 3.7). Measured (same image, GPUs 0–3, 2026‑09‑16):

| config | outcome |
|---|---|
| util 0.95, `--max-num-batched-tokens 8192`, no DSpark, `--language-model-only`, capture 64, ctx 256K | weights **78.9 GiB/GPU**, KV 6.85 GiB (2.03M tokens) allocated — then rank 3 OOM (320 MiB for the mHC post kernel, 217 MiB free) during warmup and an NCCL hang of the other ranks; the profile leaves no headroom for per‑rank asymmetry |
| util **0.92**, `--max-num-batched-tokens 4096`, otherwise as above | **serves**: weights 78.8 GiB/GPU, KV **5.22 GiB → 2,060,792 tokens** (7.86× at 256K), 89.5 GB/GPU used; `17*19 → 323`, needle PASS at 62K prompt tokens, prefill ~9k tok/s, decode **106 tok/s** single stream (no drafter) |

With DSpark (drafter +2.4 GiB/GPU at TP4, same flags otherwise): weights 81.2 GiB/GPU, KV
**2.54 GiB → 1,003,825 tokens** (3.83× at 256K). Synthetic repeated‑text sweep (T=0, 1024 forced
output tokens, `ignore_eos`, unique prefixes): C1 **336 tok/s** (DSpark acceptance 80%), C4 555,
C8 590 total tok/s at 1.5K input; C1 313, C4 425, C8 479 at 24K input; code prompt 194 tok/s
(acceptance 72%), prose 157 tok/s (25%). The same sweep on TP8: C1 211, C4 820, C8 1109 at 1.5K;
C1 173, C4 514, C8 520 at 24K; code 256 tok/s. DSpark acceptance is set by the workload (23–80%),
not by the TP layout.

Reference point — [0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000)
(SGLang image + adapter; SM120 sparse decode through a Triton kernel and 64‑token re‑paging of
the prefill sources instead of native kernels; Engram in a 64 GiB DDR5 row cache with exact NVMe
misses for 128 GB hosts; TP4/EP4; 275 W): C1 197–232 tok/s, C8 700–750 total tok/s on repeated
synthetic text with 8,192 forced output tokens at ~80% acceptance; 4.06M populated KV tokens at
memory fraction 0.95 and 4.2M allocated. Our per‑stream decode on four cards is equal or higher;
our concurrency scaling and KV capacity on four cards are lower (util 0.92 vs 0.95, ~2.7 KB of
KV per token per GPU in vLLM's FP8 layout), and their Engram cache is what makes 128 GB hosts
work at all — vLLM's pinned offload needs ~190 GiB of host RAM.

**Why 0.95 failed, and how to run at 0.94–0.95.** The OOM was not per‑rank asymmetry in the
profile: it hit during the FlashInfer **autotune** step (`kernel_warmup.py: Running FlashInfer
autotune with N tokens`), which runs a full forward at `max_num_batched_tokens` *after* the KV
cache is allocated and sweeps MXFP8 GEMM tactics, each with its own workspace — a transient the
memory profile never saw. The autotune result is cached under
`/root/.cache/vllm/flashinfer_autotune_cache/<flashinfer>/<arch>/<hash>/autotune_configs.json`
(keyed by the kernel shapes, i.e. the TP layout and flags); once it exists the warmup replays the
cached tactics and its peak is an ordinary forward. So: bring the configuration up once at a
conservative utilization to populate the cache (or start with
`--kernel-config '{"enable_flashinfer_autotune": false}'`), keep `/root/.cache/vllm` mounted, then
raise the utilization. Measured with the cache in place (TP4, DSpark k=5, mnbt 4096, capture 64,
text‑only):

| util | KV | idle / C8‑load peak per GPU | notes |
|---|---|---|---|
| 0.92 | 2.54 GiB → 1.00M tokens | 89.6 GB | first start (populates the autotune cache) |
| 0.94 | **4.44 GiB → 1.75M tokens** | 91.7 / 93.8 GB | C8 @17.5K 469 tok/s, 0 errors — recommended |
| 0.95 | **5.39 GiB → 2.13M tokens** | 92.7 / **97.2 GB** | C8 @1.1K **773 tok/s**, C8 @17.5K 578, C1 289; 8×146K concurrent (1.17M populated) OK — 0.7 GB from the wall |

`max_model_len` is a memory input, not only a limit. Measured on a second host (2026‑09‑17; 8× RTX
PRO 6000 **Server Edition** in a VM, ECC on → 94.97 GiB usable per GPU against 95.59 on the
Workstation cards; `NCCL_P2P_DISABLE=1`; same image and flags, GPUs 4–7):

| util | ctx | KV | notes |
|---|---|---|---|
| 0.92 | 512K | 0.95 GiB | **does not start**: one 512K request needs 1.1 GiB; graphs 0.74 GiB (0.52 at 256K) and ~1.4 GiB more non‑KV overhead than at 256K |
| 0.92 | 256K | 2.37 GiB → 934,666 tokens | first start (populates the autotune cache); 0.17 GiB under the Workstation cards = the ECC reserve |
| 0.94 | 512K | **2.85 GiB → 1,352,583 tokens** (2.58× at 512K) | text‑only; idle 91.8 GB, 95.9 GB/GPU peak after a 106K‑token needle and C8 @7.7K (stable); needle PASS @29K/106K, prefill ~10.5k tok/s, decode C1 130 (prose) / 278 (code) tok/s |
| 0.94 | 512K | **2.54 GiB → 1,206,953 tokens** (2.30× at 512K) | **with vision** (current); encoder +0.23 GiB weights per rank (replicated, not TP‑sharded) + encoder cache (4096‑token budget, profiled with 3 max‑size images) ≈ 0.3 GiB of KV; same decode/needle numbers, peak 96.2 GB/GPU; single/two‑image, image+tool and image+thinking prompts OK |

Token counts are vLLM's hybrid‑allocator figures (SWA layers only hold a window per request), so
they are not comparable across `max_model_len` values. With vision on, vLLM forces
`--disable_chunked_mm_input`, and the DSpark drafter does not take multimodal embeddings (image
prompts are drafted from text‑only inputs). Weights load from that host's XFS virtual
disk in ~11–12 min (the 475 GiB checkpoint does not stay in page cache next to ~190 GiB of pinned
Engram), so each restart costs ~17 min.

So four cards work for ≤ ~2M total KV tokens without any offload machinery; the 1M‑token window
is possible only by trading KV (KV/token at TP4 is ~2.6 KB/GPU). Headroom is the limiting factor, not fit: the vLLM‑Moet expert tiers (2‑bit base,
FP4 delta, base cache) would halve the 259.5 GiB of experts and are the route to comfortable TP4
or TP2, but they need a port of the `moe_w2` stack onto this vLLM base plus K=5120 / K=576·1152
cubin families.

## Where a decode step goes (TP4, DSpark k=5, 2026‑09‑18)

Torch profile of the serving container on the 4× RTX PRO 6000 host (rank 0, single stream, prose + code,
~70 target steps, 16.6 ms per step, GPU busy 95 %, **~2 090 kernel launches per step**). The
step is latency‑bound: the weights that a step streams (~2 GB/GPU) would take ~1.2 ms.

| share | ms/step | what |
|---|---|---|
| 27 % | 4.9 | MoE FP4 grouped GEMM (DeepGEMM `sm120_fp8_fp4_gemm_1d1d`): w13 83 µs + w2 36 µs per layer for M ≈ 36 rows — six verified tokens × top‑6 touch up to 36 experts, so the expert stream is ~6× a single token's; already ~75 % of bandwidth |
| 21 % | 3.8 | dense MXFP8 GEMMs (FlashInfer CUTLASS SM120, 229 launches/step, avg 16 µs) + 175 activation‑quantize launches |
| 13 % | 2.4 | BF16 cuBLAS (`cutlass_80_wmma` kernels): `wo_a` emulation bmm 26 µs × 43, lm_heads 234 µs × 3 |
| 11 % | 2.0 | mHC: DeepGEMM TF32 pre‑norm GEMM 14 µs × 85 (4 µs in isolation — PDL overlap inflates it), TileLang pre/post 5 + 4 µs × 40 |
| 10 % | 1.9 | NCCL allreduce 19 µs × 88 + allgather 37 µs × 4 |
| 8 % | 1.4 | ~650 elementwise launches (norms, quant, silu, MoE scatter/gather, topk) |
| 5 % | 0.9 | sparse‑MLA decode (FlashInfer SM120) |

Both servers on that host (this one and Qwen3.8‑Flash‑Next on GPUs 0–3) run at ~60 steps/s
regardless of model or context length; the 4‑rank NCCL allreduce costs 18–30 µs for 5–48 KiB over
NCCL's default SHM transport in the VM, so the ~90 allreduces per step are a fixed ~1.8 ms.
**NCCL refuses P2P by default on the PHB topology the hypervisor exposes, but P2P works**
(`cudaDeviceCanAccessPeer` is true for every pair): with `NCCL_P2P_LEVEL=SYS` NCCL switches to
`P2P/CUMEM` and the small allreduces drop to ~12 µs (16 MiB: 974 → 642 µs). `run.sh` now sets
`NCCL_P2P_LEVEL=SYS` and `NCCL_P2P_DISABLE=0` by default; the earlier `NCCL_P2P_DISABLE=1` was a
no‑op (SHM either way). vLLM's own custom one‑shot allreduce (refused by the ">2 PCIe‑only GPUs"
heuristic in `custom_all_reduce.py`) was measured with the gate forced open
(`tools/sm120_perf/allreduce_bench.py`): it is pull‑based — every rank reads its three peers'
buffers — and PCIe P2P *reads* are latency‑bound in this VM, so 60 KiB costs 75 µs against NCCL's
12.6 µs and the time grows linearly with size. The heuristic is right for this topology. vLLM
dev20904 also lists a push‑based `FLASHINFER_PCIE_IPC` backend, but FlashInfer 0.6.18 does not
ship `PcieIpcAllReduceWorkspace`, so it is not available in this image.

**Small‑M MXFP8 GEMV (`tools/dsv41_sm120/sm120_gemv/`).** The CUTLASS SM120 blockscaled GEMM runs
a 128‑row MMA tile for the 6‑row decode batch. The replacement keeps the exact same inputs
(FlashInfer's activation quantization, F8_128x4 swizzled ue8m0 scales for both operands) and
computes with `mma.sync.m16n8k32` (e4m3 × e4m3 → f32): a block owns 8 output columns, its 8 warps
split the 32‑wide MX blocks, physical k is permuted inside each block so every lane's fragment
bytes are contiguous (A and B identically, so the dot product is unchanged and every mma stays
inside one scale block), and the per‑block result is scaled by 2^(sa+sb−254) before fp32
accumulation. Cold‑L2 timings on RTX PRO 6000 (weights cycled through >128 MB so they stream from
HBM, like in the server), M = 6:

| shape (N←K) | CUTLASS | GEMV | speed‑up |
|---|---|---|---|
| q_a 1280←5120 | 25.6 µs | 11.1 µs | 2.3× |
| q_b / indexer wq_b 4096←1280 | 23.6 µs | 7.3 µs | 3.2× |
| kv_a 576←5120 | 14.4 µs | 7.9 µs | 1.8× |
| wo_b 5120←2048 | 44.7 µs | 13.2 µs | 3.4× |
| shared w13 1152←5120 / w2 5120←576 | 24.3 / 15.8 µs | 10.5 / 5.4 µs | 2.3× / 2.9× |

Output error vs the fp32 reference on dequantized operands is identical to CUTLASS (bf16
rounding, max rel 3.8e‑3). The vLLM hook (`patch_vllm_mxfp8_gemv.py`) is one dispatch branch in
`FlashInferCutlassMxfp8LinearKernel.apply_weights` (M ≤ 16 and bf16 output → GEMV, else CUTLASS);
`VLLM_MOET_SM120_GEMV=0` reverts at run time. A scalar (non‑tensor‑core) variant is kept behind
`VLLM_MOET_GEMV_IMPL=scalar`; it is ~1.5× slower at M = 6 because the fp8→fp32 conversion and
scalar FMAs, not the weight stream, set its pace.

In the serving container (image `dsv41-0909` = 33bf6159, TP4, k=5) the GEMV averages 10.5 µs per
call against CUTLASS's 16 µs (cold weights *and* cold scales, plus the launch tail every kernel
pays inside a graph), the dense‑GEMM share of a step drops 3.8 → 2.8 ms and GPU time per step
17.9 → 16.8 ms in the profiler; at the API the decode step went 16.3–16.6 → 16.0–16.4 ms
(60–61 → 61–63 steps/s; prose 135, code 317 tok/s; needle, vision, tool calls, C8 unchanged).
With NCCL on P2P as well (next paragraph) the step is 15.6–15.8 ms — **63–64 steps/s, prose 140
and code 329 tok/s, +6 % over the 2026‑09‑17 baseline**.
The lesson is the ratio: a kernel 2.6–4.6× faster in isolation buys ~1 ms of a 16.6 ms step,
because with ~2 090 launches and 88 four‑rank allreduces per step the boundaries, not the
kernel bodies, set the pace. The remaining levers are therefore launch‑count reductions —
activation quantization fused into the GEMV (−185 launches), norm/quant fusion (torch.compile
`-O3` is off for this model: `compilation_config.mode = NONE`), and fewer verified tokens when
acceptance is low — rather than faster versions of the existing kernels.

**`wo_a` on the same kernel (2026‑09‑18, second session).** The grouped output projection
(`o_groups` 8, `o_lora_rank` 1024; two 1024←4096 groups per TP4 rank, `is_bmm`) ran the BF16
emulation on sm_120 because `DeepGemmMxfp8BmmLinearKernel` is gated to sm_100: BF16 weights (2×
the bytes) + cuBLAS bmm, 26 µs + 3 µs split‑K reduce per layer, 1.0 ms of the step. The GEMV
kernel gained a grouped entry (`mxfp8_gemv_grouped`, `blockIdx.y` = head group) and two scale
layouts: the checkpoint's row‑major `[N, K/32]` ue8m0 for the weight and DeepGEMM's packed
MN‑major int32 layout that `fused_inv_rope_fp8_quant(tma_aligned_scales=True)` already writes
for the sm_100 path for the activation — so the activation quantization stays inside the
existing fused inverse‑RoPE kernel (1.4 µs vs 1.1 µs unquantized) and nothing new is launched.
Rows are handled in m16 tiles up to 64 tokens; above that (prefill) the weight is dequantized on
the fly and the original bf16 bmm runs. `Sm120GemvMxfp8BmmLinearKernel`
(`sm120_gemv/vllm_sm120_gemv_bmm.py`, installed by `patch_vllm_wo_a_sm120.py`) is put first in
`init_mxfp8_linear_kernel()`'s BMM list and `deep_gemm_fp8_o_proj` dispatches to it.

Cold‑L2, RTX PRO 6000, per layer: T = 1: 13.6 → 7.8 µs; **T = 6: 25.8 → 9.0 µs (2.9×)**; T = 16:
26.2 → 11.6; T = 32: 26.5 → 16.4; T = 48: 26.2 → 18.3; T = 64: 20.6 → 25.3 (the tile loop stops
paying at 64, but dequant+bmm would cost more there, so the GEMV keeps the whole capture range).
Numerics: bf16 rounding vs an fp32 reference on the same fp8 operands (max rel 3.9e‑3); relative
to the emulation path the output differs by 2.7 % (Frobenius) — that is the MXFP8 activation
quantization the sm_100 DeepGEMM path applies by design, i.e. sm_120 now matches the datacenter
numerics instead of running the o‑projection in higher precision.

In the serving container (TP4, k=5, image `dsv41-0909` = 73098d98): the grouped GEMV averages
**9.5 µs per call against the bmm's 24.9 µs**, the BF16‑cuBLAS share drops 13 → 8 % and the
NCCL share 10 → 8 % (13 µs per allreduce, less skew behind the o‑projection); weights per GPU
81.46 → 81.12 GiB, KV 2.54 → 3.14 GiB (1.49M tokens). At the API: **63.6 → 66.3 steps/s (15.7 →
15.1 ms), prose 140 → 146 tok/s, code 329 → 343 tok/s**, DSpark acceptance unchanged (2.21 / 5.17
tok/step), needle PASS @29K/106K, vision + tool + thinking smoke unchanged, C8 @7.7K 242 tok/s
aggregate with a 96.0 GB/GPU peak. The BF16 GEMMs that remain (12 per step in the target, ~40 µs
each in 8 layers next to the Engram/compressor path, plus 5 per step in the DSpark drafter graph)
are weights the checkpoint keeps in BF16 — cuBLAS already runs them at bandwidth, so the only
lever there is weight‑only quantization, a quality decision rather than a kernel gap.

A step of the *drafter* is now visible as its own 167‑kernel graph: 1.17 ms per target step (7 %),
with 15 dense GEMVs, 7 allreduces and 6 mHC blocks.

**MoE glue at decode (2026‑09‑19, `patch_vllm_moe_glue_sm120.py`).** DeepGEMM's sm_120 FP8×FP4
grouped kernel only exists with BLOCK_M = 64, so every touched expert is padded to 64 rows
(2304 rows per layer for 36 real pairs) and the FP4 GEMMs are at the weight‑stream floor
(FC1 118 MB in 85 µs, FC2 59 MB in 36 µs; `tools/sm120_perf/dg_moe_blockm_bench.py`). What
was left were the glue kernels around them: `_fwd_kernel_ep_scatter_2` ran one program per
token with a chain of six dependent atomics (6.4 µs) — now one program per (token, expert)
pair (3.5 µs); `_fwd_kernel_ep_gather`'s top‑k loop is unrolled so its loads overlap (3.6 →
3.3 µs, same fp32 order); the fused silu·up + UE8M0 quant skips the padding rows via
`m_indices` (4.0 → 3.3 µs). Bit‑exact (0 of 30720 MoE outputs differ in the bench), deployed
first as bind‑mounts, now baked into `Dockerfile.sm120-dsv41`: **66.3 → 67.0 steps/s, prose 147 → 148, code
344 → 346 tok/s**, needle @29K/@106K and the vision/tool/thinking smoke unchanged. The dense
GEMV was re‑examined at the same time (v2 kernel: split‑K, one‑round loads, word‑wide scale
loads, staged A) without gain — the findings are in `tools/dsv41_sm120/README.md`.

## What limits the step, with hardware counters (2026‑09‑19/20)

Rank‑0 torch trace of the served image (single stream, prose + code, 66 target steps, 15.6 ms per
step, GPU busy 99 %, ~2 060 kernel launches per step) and Nsight Compute on the kernels in
isolation (RTX 5090, caches flushed, real clocks):

| ms/step | share | what | distance from its floor |
|---|---|---|---|
| 4.9 | 31 % | MoE FP8×FP4 grouped GEMM: FC1 83 µs (118 MB, 1.42 TB/s) + FC2 36 µs (59 MB) × 40 layers | **at the HBM floor** — six verified tokens × top‑6 touch up to 36 experts = 177 MB per layer, 7 GB per step |
| 3.3 | 21 % | ~1 390 launches shorter than 5 µs (quantize, norm, scatter/gather, top‑k, fills, indexer glue, 2.4 µs average) | **at the launch floor** — a kernel inside a CUDA graph cannot take less than ~2 µs |
| 3.3 → 2.9 | 21 → 19 % | dense MXFP8 GEMVs, 222 launches, 10.8 → 10.0 µs average | 50–65 % of peak bandwidth inside the kernel (v3 below); ~2.5 µs of every launch is ramp + tail |
| 2.0 | 13 % | mHC: DeepGEMM TF32 pre‑norm GEMM 14.4 µs × 85 (4 µs of work, the rest is the PDL wait on its predecessor) + TileLang pre/post 5.1 + 3.7 µs | latency, not work |
| 1.3 | 8 % | BF16 cuBLAS: `lm_head` 233 µs × 2 (331 MB at 1.43 TB/s), 17 small `wmma` GEMMs × 26 µs | `lm_head` at the floor |
| 1.2 | 8 % | NCCL: 88 allreduces × 12.9 µs + 4 allgathers × 29 µs | pure latency (5–48 KiB payloads) |
| 0.9 | 6 % | sparse‑MLA decode 14.5 µs × 39 + merge 2.8 µs | ~3× the KV‑byte floor (7 MB per layer) — a gather of scattered 32‑token pages |
| 1.2 | 7 % | the DSpark drafter graph (167 kernels) | same mix as above |

Two things follow. First, the largest item is already at the memory wall: the only way to shrink
the 4.9 ms is to stream fewer expert bytes per step (fewer verified tokens when acceptance is low,
or batching more sequences per step), not a faster kernel. Second, about a third of the step is
made of kernels at or near the launch floor, where the body could be zero and the step would not
notice; that share is addressed by fusion (fewer launches), not by scheduling.

**Where SASS‑level work (`cubit`) would and would not pay.** A hand‑scheduled kernel beats nvcc
when the body is issue‑ or latency‑bound *inside* a memory budget it does not fill. The counters
say that describes none of the large kernels here: the MoE GEMMs are at the HBM floor, the
`lm_head` GEMM is at the floor, and the dense GEMVs were bound by *how many bytes each load
instruction moves and how many are in flight* — a structural property that a rewrite in CUDA
fixed (v3: 16‑byte loads, four rows × 128 contiguous bytes per warp instruction, loads of the next
round issued before the mma of the current one, one 32‑bit scale word per (row, 128 k)), after
which `cubit disassemble --frozen` shows nvcc issuing the whole load burst before the first
`STS.128` and an `LDS.64 → QMMA → FFMA` loop whose stalls are not the limiter. The team's own
record on this GPU points the same way: the hand‑scheduled QMMA GEMV lineage topped out at
745 GB/s while nvcc's k‑loop reached 1 164 GB/s (`cubit-internal/docs/sasstuning.md`). The
candidates where a SASS kernel could still matter are the sparse‑MLA decode (3× above its floor;
the cubit repo already holds a fused decode kernel for the V4 page geometry, 2.5× Triton, which
would need the V4.1 layout — SWA pages of 32, compressed 128/64 — and FlashInfer's dispatch) and
the mHC pre/post pair (SASS versions exist for V4; ~1 ms of latency‑bound TileLang per step).
Both are multi‑day efforts for ≤ 0.5 ms each; the fusion and all‑reduce items above are cheaper
per millisecond.

**Dense GEMV v3 (shipped in `dsv41-0909` since 2026‑09‑20).** ncu on v1: DRAM 33–50 % of peak,
long‑scoreboard stalls in 76–80 % of issue slots, 3.6 M warp instructions per 10 MB weight. v3
cold‑L2 on the RTX 5090, M = 6: 1792←5120 10.4 → 7.5 µs, 1152←5120 8.9 → 5.5, 8192←1280 9.0 →
8.3, 5120←2048 8.2 → 8.1; in the server 10.8 → 10.0 µs per launch, **67.0 → 67.7 steps/s, prose
148.5 → 149.7 and code 346 → 347 tok/s**, needle/vision/tool smoke unchanged (`VLLM_MOET_GEMV_IMPL=v1`
restores the previous kernel). Details and the negative results (split‑K again, larger rounds)
in `tools/dsv41_sm120/README.md`.

**Tried and not kept.** (a) DSpark adaptive verification: DeepGEMM's varlen paged‑MQA logits do
work on sm_120 (`patch_vllm_indexer_sm120.py` only widens vLLM's sm_100 gate; op‑level test
`test_deepgemm_sm120_paged_mqa.py --only varlen`), but the varlen decode path adds ~0.5 ms/step
(topk/indexer glue) and, with adaptive verification on, prose stayed at ~130 tok/s while code fell
313 → 273 tok/s and the varlen decode cudagraphs cost 2 GiB of KV (util had to go to 0.95). Left as
an experiment. (b) An fp32 GEMV for the mHC pre‑norm GEMM: DeepGEMM's TF32 kernel is 4.2 µs in
isolation and the hand‑written one is not faster at M ≥ 6; the 14 µs in the profile is PDL
overlap, not kernel time. (c) vLLM's custom one‑shot allreduce on PCIe P2P (see the NCCL
paragraph): 6× slower than NCCL at 60 KiB, pull‑based reads do not suit this VM. (d) The dense
GEMV above 16 rows: the m16‑tile loop is correct up to 64 rows but slower than CUTLASS from 32
rows on (q_a M = 32: 18.0 vs 14.9 µs), so the dense dispatch keeps `M ≤ 16`.

## The prompt encoder: reasoning‑effort tiers (2026‑09‑20)

Not a kernel gap, but the one place where the served model deviated from DeepSeek's own
deployment. The checkpoint ships its prompt encoder (`encoding/encoding.py`) and the tech report
lists the public API tiers (Table 2): **`low` = 50, `high` = 75, `max` = 100, default `high`**,
rendered as `Reasoning Effort: {budget} (range 1‑100, …)` in front of the system message whenever
thinking is on. vLLM vendored its copy of that encoder from a pre‑release drop
(`ds-code-260903`) whose table reads `low 25 / high 50 / xhigh 75 / max 100` — still so on vLLM
main. Running both encoders in the image on the same messages: a thinking‑mode request without an
explicit effort (the opencode default) rendered **`Reasoning Effort: 50`** — the tier DeepSeek
calls "low" — where the reference renders **75**; `low` rendered 25, a budget the API does not
expose; the OpenAI vocabulary `medium` / `minimal` failed with HTTP 400. Everything else in the
prompt (tools, multi‑turn, chat mode, images) was already byte‑identical to the checkpoint's
`encoding/tests` goldens; only the budget line differed. The report's Figure 9 puts most of the
accuracy gain between effort 25 and 75 (eight reasoning benchmarks 67.1 → 76.3 %, DeepSWE
66.0 → 74.2 % from 25 to 100), so the default request was leaving quality on the table. The
quality numbers in these docs (GSM8K‑200, needle) were measured with thinking off and are not
affected.

`tools/dsv41_sm120/patch_vllm_reasoning_effort.py` (applied in `Dockerfile.sm120-dsv41`) puts the
official table into `vllm/tokenizers/deepseek_v41_encoding.py` and keeps `minimal` / `medium` /
`xhigh` as interpolated budgets (25 / 62 / 87 — not DeepSeek tiers; the report says intermediate
values interpolate) so such requests do not fail; the wrapper's error message lists the accepted
names. An integer in `chat_template_kwargs` (`{"reasoning_effort": 75}`) bypasses the table, as
before. `test_reasoning_effort_encoding.py` runs at image build (tiers + default budget) and, with
the checkpoint mounted, requires byte‑identical prompts to DeepSeek's encoder on its five goldens
and a 90‑case matrix (thinking/chat × 9 efforts × 5 conversations incl. tools, reasoning turns and a
tool‑call round trip) — all identical after the patch. Worth a vLLM PR (the fix is the same three
lines there).

## The indexer in its training format: MXFP4 K cache on sm_120 (2026‑09‑20)

The tech report quantizes the Lightning Indexer's Q and K to FP4 with UE8M0 block scales (32
values per scale) — the format its top‑k selection was trained with (V4's QAT). vLLM has that path
(`--attention-config '{"indexer_kv_dtype":"mxfp4"}'`, the recipe's Blackwell setting) but gates it
on sm_10x with a message that names sm_120 unsupported. It is not: the gate was written before
DeepGEMM `8b1392b9` shipped `sm120_fp4_paged_mqa_logits.cuh` / `sm120_fp4_mqa_logits.cuh`
(head_dim 128, the indexer's), and everything else on the path is arch‑generic — the Q quantizer
(CuTe DSL), the K store (Triton, `cvt.rn.satfinite.e2m1x2.f32`, which sm_120a has) and the prefill
gather op (byte‑width generic). What stood in the way on sm_120 was the same thing as for the FP8
cache: the DeepGEMM launchers' host asserts stop at 64‑key pages and V4.1 pages 128 keys on the
compress_ratio‑1 layers. `patch_deepgemm.py` now admits 128 for the MXFP4 launcher too;
`patch_vllm_indexer_fp4_sm120.py` widens the gate; the launcher takes `INDEXER_KV_DTYPE=mxfp4`.

Per indexer key this is 68 B (64 B of e2m1 pairs + 4 UE8M0 bytes) instead of 132 (128 fp8 + one
fp32 scale). Because vLLM packs every kv source's compressed page and indexer page into one
physical block (block‑outermost layout `BLHNC`; `tools/sm120_perf/kv_layout_probe.py` runs
vLLM's own `get_kv_cache_groups` on the model's specs offline and reproduces the served
1,490,870‑token figure to 0.1 %), the block shrinks from 230 400 to 210 240 B: 3 × (37 440 + 4 608)
+ 74 880 + 9 216, i.e. +9.6 % KV tokens for the same bytes (the 43 SWA pages are packed into four
groups of 11/11/11/10 × 19 008 B, just under that). The 210 240‑byte stride is 64‑B‑ but not
512‑B‑aligned; DeepGEMM's TMA descriptors need 16 B, and the op‑level test lays its pages out at
exactly these strides and offsets (`--packed-stride`).

Validation, op level (RTX 5090, `test_deepgemm_sm120_paged_mqa.py --fmt mxfp4 --packed-stride`
and `test_indexer_fp4_sm120.py`): DeepGEMM MXFP4 logits on 128‑key pages vs the fp4‑simulated
reference 1.2e‑6, bit‑exact between 64‑ and 128‑key pagings of the same keys, native next_n 1/2/6
and varlen (40/40 cases with FP8); the CuTe DSL Q quantizer and the Triton K store reproduce
DeepSeek's quantizer (RNE e2m1, UE8M0 = 2^ceil(log2(amax/6))) on 16.8 M + 0.8 M values with zero
off‑tie differences (the hardware keeps the sign of values that round to zero; exact midpoints round
to even as expected), and DeepGEMM's logits on the kernel‑written cache match the dequantized
reference. On Gaussian data the FP4 indexer's scores sit 1.4e‑2 (cosine `calc_diff`) from the
fp32 ideal where the FP8 path sits at 7e‑4 — twenty times coarser, which is why the check that
matters is the served one below.

Served (4× RTX PRO 6000, TP4, `INDEXER_KV_DTYPE=mxfp4`, otherwise the production configuration):
the greedy outputs are **identical** to the FP8‑indexer baseline on every probe of
`quality_cmp.py` (24/24 agreement prompts, 12/12 raw coherence texts, arithmetic 5/5 with and
without thinking, tools / JSON / vision PASS) and the needle answers at 26.7K, 97.4K and 367.5K
tokens (both depths) are the same six numbers; GSM8K‑200 greedy C4 **193/200 = 96.5 %, the same
as both FP8 runs, paired exact McNemar p = 1, flips 2/2** (the two FP8 runs flip 1/1 against each
other — batching noise), completion tokens +0.5 %. Speed is unchanged (prose 149.9 tok/s at
67.7 steps/s, code 347.8 at 67.9, fresh prefill 11.7k tok/s at 17K and 10.9k at 139K).

KV: `Available KV cache memory` **4.39 GiB = 2,288,673 tokens (4.37× at 512K)** against 3.14 GiB
= 1,490,870 (2.84×) — +53 %, of which the smaller indexer page explains +9.6 %; the rest comes from
vLLM's start‑up accounting reporting **83.07 GiB consumed (weights + non‑torch) instead of the
84.32–84.92 GiB of every FP8‑indexer start** (peak activation 1.81 GiB and CUDA graphs 0.42 GiB
unchanged), i.e. ~1.25 GiB less non‑torch memory during the profile run, which vLLM hands to the
KV cache. That memory is not gone at run time: under load the cards level off at **97 001 of
97 887 MiB (886 MiB from the wall)** where the FP8 configuration peaked at ~96.0 GB, and stay
there through the full battery — GSM8K C4, eight concurrent 126K‑token fresh prefills (0 errors),
a fresh 139K prefill, the 367K needle and vision at the same time. The margin is therefore closer
to the opt‑in 0.95/2048 profile's (716 MiB) than to the old default's (1.2–1.8 GB);
`GPU_MEM_UTIL=0.93` with the MXFP4 indexer buys the old margin back at ~3.4 GiB of KV (~1.8M
tokens, still +20 %). `INDEXER_KV_DTYPE=mxfp4` is the launcher default since 2026‑09‑20;
`INDEXER_KV_DTYPE=fp8` restores the previous cache (no rebuild).

## Apply / build / run

```bash
# repo root; official image + the two patches + precompiled JIT
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41 -t vllm-moet-sm120:dsv41-0909 .
```

Serve exactly as the vLLM recipe, minus adaptive verification (see the Dockerfile header for the
full command). Everything else (`--tokenizer-mode/--reasoning-parser/--tool-call-parser
deepseek_v41`, `--kv-cache-dtype fp8`) is the recipe's own configuration.

Without an image rebuild (development loop): copy the installed `flashinfer` package out of the
official image, run `patch_flashinfer.py --flashinfer-dir <copy>`, bind‑mount the copy over
`/usr/local/lib/python3.12/dist-packages/flashinfer`, an empty file over
`flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so` (FlashInfer falls back to
JIT when the AOT artifact fails to load) and the rebuilt `_C.so` over
`vllm/third_party/deep_gemm/_C.cpython-312-x86_64-linux-gnu.so`.

## Known follow‑ups

- Upstream: [vllm#56837](https://github.com/vllm-project/vllm/issues/56837),
  [vllm#56509](https://github.com/vllm-project/vllm/pull/56509),
  [vllm#57028](https://github.com/vllm-project/vllm/pull/57028) change the vLLM‑side page geometry
  instead; this port keeps vLLM stock and would be superseded by FlashInfer/DeepGEMM shipping the
  instantiations. Worth proposing there.
- `EmulationMxfp8LinearKernel` is no longer selected in the serving log (dense → FlashInfer
  CUTLASS + GEMV, `wo_a` → grouped GEMV); the remaining BF16 GEMMs are BF16 checkpoint weights.
- Bench recipe (`bench/recipes/`) for `deepseek-v4.1-flash/pro6000x8-tp8-dspark` not yet
  registered; the numbers above are single‑shot smoke measurements, not a release row.
