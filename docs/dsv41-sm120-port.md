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
12.6 µs and the time grows linearly with size. The heuristic is right for this topology. The
push‑based `FLASHINFER_PCIE_IPC` backend (FlashInfer's `PcieIpcAllReduceWorkspace`, shipped by the
nightly wheels of the served image, opt‑in through `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`) was
measured on 2026‑09‑23 with the same bench, tuned on this host: 5.3 µs against NCCL's 10.4 at one
token, but **16.5 against 12.8 at the 6 tokens of a DSpark k=5 decode step** and a tie from 24 to
48 tokens — it stays off. NCCL LL over P2P is the floor for this `rootcplx-noswitch` topology;
what is left on the all‑reduce line (1.3 ms per step at one stream, 5.7 ms at eight) is the
number of all‑reduces, not their speed.

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
an experiment; **re‑measured on the served image on 2026‑09‑22** (same client, same day, k = 5,
`tools/sm120_perf/spec_matrix.py`: prose / code × concurrency 1 / 4 / 8, 512 tokens each): the
FULL cudagraphs of the varlen path take 2.46 GiB instead of 0.41 (**KV 3.87 → 1.92 GiB, 3.06M →
1.52M tokens**); the verifier trims the drafts (prose 2.3 → 2.0 accepted tokens per step, 67 → 75
steps/s), which is a wash for a single prose stream (153 → 149 tok/s) and **+15 % at eight prose
streams (555 → 640 tok/s)**, but **code loses 14 % single‑stream (360 → 308 tok/s) and 8 % at four
streams (950 → 872)** — the trimmed drafts would have been accepted. Against the plan's criterion
(prose C1 ≥ +5 %, code ≥ −2 %, KV ≥ −5 %) it fails on all three; the topic is closed for this
image. The cheaper knob, a shorter draft (`SPEC_TOKENS=3`, measured the same way): prose
unchanged single‑stream and **+13 % at four and eight streams (366 → 413, 555 → 625 tok/s)**, KV
+6 % (3.06M → 3.23M tokens; smaller drafter graphs), but **code −24 % single‑stream (360 → 276)
and −16 % at eight streams** — code accepts 4.2 of 5 drafts, so the cap at 3 costs directly. k = 5
stays the default for the mixed agent workload; a prose‑only deployment would take k = 3.
(b) An fp32 GEMV for the mHC pre‑norm GEMM: DeepGEMM's TF32 kernel is 4.2 µs in
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

## The compressed KV in its training format: FP4 records on sm_120 (2026‑09‑21)

DeepSeek's own inference (`inference/model.py::_compress_kv`) keeps the compressed (main) KV as
`fp4_act_quant(latent, 16, inplace=True, scale_dtype=e4m3)` of the RoPE'd latent: FP4 e2m1 values
with one E4M3 scale per 16 dims, 288 B per state — the format the model was trained with (V4's
QAT). vLLM 0909 stores the bf16 latent as `fp8_ds_mla` instead (448 fp8 with a UE8M0 scale per
64 + 64 bf16 RoPE dims, 584 B), which is *more* precise than the reference but twice the bytes,
and it is the biggest item of the KV budget: every kv source's compressed page and indexer page
share one physical block, so the block goes 210 240 → 115 200 B with the FP4 record. FlashInfer's
SM120 sparse‑MLA kernels read `fp8_ds_mla` only (there is no NVFP4 sparse MLA for SM120 in
0.6.18; vLLM main's `nvfp4_ds_mla` decodes through FlashMLA on SM100), so this port keeps the
kernels and feeds them a scratch:

- **Record.** `tools/dsv41_sm120/nvfp4_kv/fp4_kv_quant.py` reproduces the checkpoint's quantizer
  bit for bit (values, e2m1 codes, e4m3 scales, the 6·2⁻⁹ amax floor, −0; the two details that
  matter are IEEE `div.rn` instead of Triton's `div.full` or `x·(1/s)` — either flips 0.04 % of
  the codes at e2m1 ties — and keeping the sign of values that round to zero). The insert kernel
  (`rope_quant_insert_packed`) writes upstream's `nvfp4_ds_mla` page layout (256 B of e2m1 pairs
  per state, then 32 e4m3 bytes) after the same GPT‑J RoPE the fp8 kernel applies; the 528‑byte
  V4.1 fp8 record (`act_quant(kv, 32, "ue8m0")`) is implemented alongside as the fallback format.
- **Read side (variant A of the plan).** Before each SM120 sparse‑MLA call the attended states are
  dequantized and re‑quantized into an `fp8_ds_mla` scratch — a 128‑state paged cache whose
  indices replace the compressed indices — with vLLM's own fp8 recipe, so the kernel sees
  `fp8(dequant(fp4(latent)))`. Decode gathers the rows' top‑512 records once per (kv source,
  index set): the index‑source layer of each group computes it and its consumers reuse it (8
  gathers per step instead of 38; 19 MB scratch). Prefill would gather 4096 × 512 records per
  layer (1.8 GB per layer per chunk, −15 % prefill in the first version); instead the whole
  compressed context of the step's prefill requests is dequantized in logical order into a pool
  once per kv source (state *i* of request *k* → slot base_k + *i*) and addressed with the
  request‑local top‑k indices plus the base — O(context) instead of O(rows × 512), 306 MB pool at
  512K (`VLLM_MOET_KV_PREFILL_POOL_STATES`), per‑request dequant as the fallback when the step's
  requests do not fit together. The pool and the scratch are reserved in vLLM's profile run
  (`_reserve_empty_forward_workspace`), so they come out of the accounted budget, not the margin.
- **Numerics, isolated first.** `patch_vllm_kv_fp4_fake.py` ("A0") applies the reference FP4
  quantizer inside today's fp8 insert kernel — the same double quantization with today's storage
  — so the quality of variant A could be measured before the storage side existed; the packed path
  then reproduced A0's greedy outputs token for token (24/24, 12/12), and `test_nvfp4_kv_kernels.py`
  proves it byte for byte (gather / context‑dequant scratch == A0 records; FlashInfer dual‑cache
  attention on the scratch + remapped indices bit‑exact with the same kernel on a real cache).

Served (4× RTX PRO 6000, TP4, `KV_RECORD=nvfp4`, MXFP4 indexer): GSM8K‑200 greedy C4 **194–195
/200 in three runs against 193/200 with the fp8 record** (paired McNemar p = 1 / 0.5, flips
0–1 / 1–2), needle 6/6 to 367K with the same answers, arithmetic / tools / JSON / vision PASS,
coherence 0 degenerate. The greedy outputs themselves change — 8/24 agreement prompts and 5/12
coherence texts identical to the fp8 record, where the fp8 record and 0xSero's SGLang stack
(different kernels, same checkpoint) agree on 7/24 and 1/12 — i.e. the FP4 keys move the greedy
path about as much as switching serving stacks does, with no measurable effect on correctness.
Speed: 67.0–67.1 steps/s (fp8 record 67.7), prose 152.6 / code 343 tok/s, prefill 10.0k (16K) /
11.6k (17K fresh) / 10.8k (139K fresh) tok/s against 10.3 / 11.7 / 10.9k — the gathers cost
~1 % of the step and ~3 % of prefill. KV: **4.37 GiB = 3,453,349 tokens (6.6× at 512K)** before
the profile‑run reservation, **3.87 GiB = 3,057,484 tokens (5.8×)** with it, against 2,288,673 with
the fp8 record and 1,490,870 before the MXFP4 indexer; peak under the stress battery (8 concurrent
fresh 126K prefills, fresh 139K, GSM8K C4, 367K needle, vision) **95 897 MiB of 97 887 — 1.99 GB
of margin**, more than the fp8 record left at either indexer format, because the pool is now part
of the accounted budget. `KV_RECORD=nvfp4`
is the launcher default since 2026‑09‑21; `KV_RECORD=fp8_ds_mla` restores the previous record and
`fp8_v41` selects the 528‑byte one (op‑level validated only), both without a rebuild.
`tools/sm120_perf/kv_layout_probe.py --main-bytes 288` predicted the block and the capacity
(115 200 B, 3.47M tokens) before any code ran.

**Upstream reads the record now (checked 2026‑09‑21).** FlashInfer `main` since
[#5197](https://github.com/flashinfer-ai/flashinfer/pull/5197) (merged 2026‑09‑18, not in 0.7.0rc3
or `release-v0.7.0`) has a DeepSeek‑V4.1 dual‑cache sparse MLA for SM120/121 that takes the 528‑byte
V4.1 fp8 sliding‑window record as the main cache and exactly this 288‑byte FP4 record as the extra
cache (`kv_cache_format="fp8_dsv41_fp4_ca"` on `trtllm_batch_decode_sparse_mla_dsv4`; the FP4 rows
are converted to fp8/UE8M0‑32 tiles on chip, or dequantized to bf16 on the `compute_precision="bf16"`
route), with runtime page sizes (32‑token SWA pages included) and cache writers
(`dsv41_fp4_quantize_pack/append_sparse_mla_cache`). It is the pairing vLLM `main` calls
`nvfp4_ds_mla` (SM100‑only there today) — so the scratch/pool above becomes unnecessary once the
port moves to a FlashInfer release carrying #5197 and vLLM lets sm_120 pick that format.
`tools/dsv41_sm120/nvfp4_kv/test_flashinfer_dsv41_fp4_extra.py` runs upstream's kernel on this
port's geometry against our writers and references (RTX 5090, FlashInfer main 6870e3ff, CUDA 13.0):
our 288‑byte writer produces upstream's bytes exactly (both page sizes, floor/ceiling groups), the
528‑byte writer matches the V4.1 reference; 324/324 cases pass FlashInfer's tolerance — SWA page 32,
compressed page 128 / 64, 8 / 16 / 64 heads, SWA top‑k 128 / 192 / 1152 with −1 padding and per‑row
lengths, sinks, decode and prefill entries, vLLM's block‑strided cache views — with the same distance
from the fp32 reference as today's scratch path (max 2.4e‑3 both). Per call, CUDA‑graph timed, cold
256K‑token pools, 16 heads: decode 6 / 24 / 48 rows × (192 + 512) 14.5 / 18.8 / 31.0 µs against
today's DSV4 kernel + gather 21.5 / 29.2 / 43.7 µs (**−30 %**, the `bf16` route is as fast at 6 rows
and 8× closer to the reference); prefill 4096 rows × (128 + 512) **1408 µs against 1151** and
× (1152 + 512) 3221 against 2671 (**+21 %**): the V4.1 family only has the single‑group FP8 prefill
kernel, while the DSV4 dual cache runs the multi‑group BF16‑QK kernel (the all‑fp8 V4.1 cache is
already +12 % on the same shapes, the FP4 conversion adds +9 %). Net for this deployment: +325 MB
of KV (no pool / scratch, ≈ +8 % tokens), ≈ +1 % decode, ≈ −3 % prefill until upstream's V4.1 dual
prefill gets the MG path — a scoped kernel contribution with a measured target.

## The vLLM main line as a candidate image (2026‑09‑22)

`Dockerfile.sm120-dsv41-nightly` builds the same idea on `vllm/vllm-openai:nightly` (vLLM main
0961bbae, pinned digest; the recipe marks 0909 "superseded") with FlashInfer's nightly wheels
(0.7.0.dev20260922, #5197 included: the DSv4.1 dual cache reads the FP4 record — no TU / dispatch
hook, no scratch / pool) and DeepGEMM `_C` rebuilt at vLLM main's pin (`vllm-project/DeepGEMM`
e1f418c2). The patchers above apply unchanged (all anchors hold on main); our vLLM PR rides along as
`nvfp4_kv/patch_vllm_nvfp4_sm120_upstream.py` (`--kv-cache-dtype nvfp4_ds_mla` on sm_120 through
`kv_cache_format="fp8_dsv41_fp4_ca"`), and the launcher reads the image label to pick that plumbing.

First serve (4× RTX PRO 6000, TP4, 512K, DSpark k=5, MXFP4 indexer): quality identical to the 0909
image (GSM8K‑200 195/200, needle 6/6 to 367K with the same answers), prefill as predicted (−1…−3 %),
but decode **−3 % single‑stream to −10 % at eight streams**. The op‑level MoE bench
(`tools/sm120_perf/moe_backends_bench.py`, RTX 5090, this model's per‑rank geometry) found the
cause without a server: vLLM's DeepGEMM chain took **2×** in the main‑line image — 98 → 186 µs at
6 tokens, 576 → 1101 µs at 48 — and printed `align 128` where the 0909 image prints `align 64`.
The vllm‑project fork's port of the SM120 kernels (its #4) kept upstream's
`get_theoretical_mk_alignment_for_contiguous_layout`, which knows SM100 (256) and answers the
legacy 128 for every other arch and has no `num_groups` argument; nv_dev's version (the 0909 pin)
shrinks BLOCK_M on SM120 to the smallest tile covering the per‑expert M (BLOCK_M ∈ {64, 128}:
kMWarps(4) × MMA_M(16)). vLLM's MoE glue calls it with `(M·top_k, local_num_experts)`, falls back
to the one‑argument form on `TypeError`, gets 128, pads every routed expert to 128 rows and caps
the kernel at BLOCK_M=128 — tiles of 128 rows on 1–6 real rows, and the grouped GEMM no longer
reaches the HBM floor. `patch_deepgemm.py` (patches 5–6, skipped on the nv_dev pin) restores the
policy before `_C` is rebuilt; the chain is back to 97 / 575 µs and the `_C` still passes the paged
MQA (40/40) and MXFP4 indexer tests. The same two‑file change is the upstream PR to the fork
(`internal/upstream-deepgemm-sm120-mk-alignment-*`). Served numbers of the fixed image
(`dsv41-nightly-20260923`): the table below. **It is the served image since 2026‑09‑23**; the 0909
image stays as the rollback (`IMAGE=vllm-moet-sm120:dsv41-0909`).

Served on the same 4× RTX PRO 6000 (TP4, 512K, DSpark k=5, MXFP4 indexer, FP4 compressed KV,
`GPU_MEM_UTIL=0.94`, warm caches; `tools/sm120_perf/spec_matrix.py`, 512 output tokens, two waves
per cell, thinking off; steps/s = engine steps per second of one stream):

| | 0909 image (served) | main line, DeepGEMM fork as shipped | main line + BLOCK_M fix (`dsv41-nightly-20260923`) |
|---|---|---|---|
| KV capacity at 0.94 | 3.87 GiB = 3,057,484 tokens | (0.92: 2.5 GiB = 1,983,987) | **4.35 GiB = 3,454,536 tokens (+13 %)** |
| prose, 1 stream | 152–154 tok/s, 67 steps/s | 150–161, 64–65 | **163–164, 73.5–73.9 (+10 %)** |
| prose, 4 streams | 365–367 tok/s, 41 steps/s | 340–351, 39 | 371–380, 43.7 |
| prose, 8 streams | 553–557 tok/s, 31 steps/s | 502–511, 28 | 554–568, 31.5–32.0 |
| code, 1 stream | 349–371 tok/s, 68.8 steps/s | 346–350, 67 | **383–387, 75.6** |
| code, 4 streams | 943–961 tok/s, 45–47 steps/s | 826–849, 40–41 | 904–977, 43–46 |
| code, 8 streams | 1377–1397 tok/s, 34 steps/s | 1237–1263, 30 | 1372–1428, 33.4–34.6 |
| fresh prefill 19K / 163K | 11.4k / 10.5k tok/s | 11.1k / 10.4k | 11.4–11.5k / 10.6k |
| GSM8K‑200, thinking off | 195/200 | 195/200 | 194/200 |
| needle (27K, 92K, 184K, 367K × 2 depths) | 6/6 to 367K | 6/6, same answers | 8/8, same answers at 367K |
| probes (arith ×2, tools, JSON, vision, coherence) | PASS | — | PASS; greedy agreement 8/24 vs 0909 (the FP4 route's own numerics, as between two stacks) |
| 8 × 126K‑token requests, peak GPU memory | 97,001 / 97,887 MiB | — | 96,491 / 97,887 MiB |

Where the +10 % of a single stream comes from (rank 0 traces, `profile_capture.py` + `trace_agg.py`,
per decode step; 0909 15.9 ms of GPU time, main line 13.9 ms): the MoE grouped GEMM is identical
again (4.94 → 4.91 ms; FC1 83.5 → 83.1 µs per launch), **mHC 2.0 → 1.0 ms** (main runs one
`mhc_fused_tilelang_kernel` of 6 µs where 0909 ran the TF32 prenorm GEMM 14.4 µs + `mhc_post`
3.7 µs), **NCCL 1.8 → 1.3 ms** (13.4 vs 19.0 µs per bf16 ring all‑reduce at 6 tokens; 58.8 vs
87.5 µs at 48), elementwise 1.3 → 1.0 ms (−160 launches per step), sparse MLA 0.9 → 0.8 ms (the
DSv4.1 mixed‑cache decode kernel, no gather). At eight streams the per‑step GPU time is 33.3 →
30.7 ms with the same composition (the MoE at 48 tokens +2 %: 213 vs 208 µs per FC1 launch).

## KV outside HBM: host‑RAM and disk tiers (2026‑09‑23)

vLLM main's native `OffloadingConnector` (`--kv-offloading-size N`, GiB of pinned host RAM shared by the
TP ranks as one `/dev/shm` region) takes the DeepSeek‑V4.1 hybrid KV layout as it is: of the ten KV
cache groups it offloads only the block‑128 group (the compressed MLA states and the indexer K cache —
the ones that are prefix‑cacheable), skips the eight 32‑token SWA groups and the compressor ring
(`prefix_cacheable=False` under the default `swa_bounded_replay=True`), and on a hit the scheduler
replays the 128‑token SWA window exactly as it does for a GPU prefix‑cache hit. The packed
block‑outermost layout is handled by copying whole packed blocks (`offloading/worker.py`, "Packed
layouts (e.g. DSv4)"); the connector's preferred `LBHNC` layout is dropped with a warning. vLLM
main's de‑duplication of TP‑replicated MLA KV (`replicated_layout`) admits only a single MLA group,
so on the plain main image the host holds four copies at TP4 — and so does the disk tier below,
which writes what the RAM tier holds; the served image keeps one ("One copy instead of four"
below). `canonical_layout` (topology‑free pages) refuses packed layouts.

Measured on the served image (TP4, `--kv-offloading-size 64`, GPU KV limited to 1.25 GiB = 992K
tokens so that eviction is quick; `tools/sm120_perf/kv_offload_probe.py`, greedy, thinking off):

| step | wall | what happened |
|---|---:|---|
| 179K‑token prompt with a needle, fresh | 17.28 s (10.4k tok/s) | 641 MB stored GPU → host |
| same prompt again | 0.56 s | GPU prefix‑cache hit (178,944 of 179,037 tokens) |
| 8 distinct 179–189K‑token prompts | 17.0–18.0 s each (10.5k tok/s) | stores of 640–680 MB each, no prefill slowdown |
| the first prompt again (evicted from the GPU) | **0.49 s** | **offload hit**: 641 MB loaded host → GPU, `external_prefix_cache_hits` 178,944, needle answered correctly |

Host cost 3.58 KB per token (four TP copies of 896 B/token/rank; 64 GiB ≈ 19M tokens, 128 GiB ≈
38M; one copy since the same evening: 896 B, 64 GiB ≈ 76.7M). Decode and prefill with the connector on and 6.5 GB stored: prose 165–180 tok/s at 74–75
steps/s, eight streams 559–572, code 386–402 / 1409–1442, fresh prefill 11.5k / 10.6k tok/s — the same
as without it. Startup +0 s (the region is pre‑faulted in a few seconds). The launcher exposes it as
`KV_OFFLOAD_GIB` (default 0).

**Disk tier.** `spec_name: TieringOffloadingSpec` with `secondary_tiers: [{type: fs, root_dir: DIR}]`
puts a filesystem tier behind the RAM tier: every block stored to RAM is also written to DIR (one
file per 128‑token block, named by the block's content hash, 448 KiB = the four TP copies (112 KiB with one), O_DIRECT
on XFS); a lookup that misses RAM checks the files, and a hit is read into RAM and loaded to the GPU
from there. Measured with the GPU pool at 992K tokens and the RAM tier at 4 GiB (1.2M tokens), so both
evict within a few long prompts; the disk is a virtio volume of the KVM guest (the report's deployment
has NVMe):

| step | wall | what happened |
|---|---:|---|
| 209K‑token prompt with a needle, fresh | 20.25 s (10.3k tok/s) | 749 MB stored to RAM and written to disk in the background (3.9 s of write time) |
| 8 distinct 209–227K‑token prompts | 20.3–22.2 s each (10.2–10.3k tok/s) | 750–810 MB each stored and written; the GPU pool and the RAM tier both evict the first prompt |
| the first prompt again | **2.53 s** | **disk hit**: 1,633 chunks, 749 MB read in 1.9 s, promoted to RAM, loaded to the GPU; needle answered |
| turn 2 of a conversation evicted the same way (turn 1: 104K‑token prompt + 1,347 generated) | 1.29 s | 105,344 of its 105,394 tokens from disk (377 MB) |

The writes do not slow the prefill down (10.2–10.3k tok/s is the no‑offload rate at 209–227K tokens).
The files are named by the content hash of the token chain (sha256 with a fixed seed), so they stay
valid for any later server with the same model, TP size, KV dtype and group layout: **after a
restart** (new process, empty GPU pool and RAM tier) the first request for a 174K‑token prompt stored
by the previous process took **3.20 s instead of 16.7 s** — 1,359 chunks read from disk (623 MB in
1.5 s), needle answered. vLLM never deletes these files; `docker/sm120/kvcache-ttl.sh DIR` (hourly from
cron) removes the ones not read for 72 h — the retention the V4.1 report gives its SSD tier — and then
the least recently read ones above 200 GB. A deleted file is a miss for the server, so the script is
safe while it serves. The launcher exposes the tier as `KV_OFFLOAD_FS_DIR`.

**Generated tokens.** `offload_prompt_only` (default true) keeps a request's generated tokens out of
the tiers, while the GPU prefix cache keeps them. In the conversation above, turn 2 repeats turn 1's
prompt *and* its 1,347 generated ids token for token (thinking off; the V4.1 encoder also keeps earlier
reasoning and tool calls in the history whenever the request has tools, as agents' requests do, and
drops earlier reasoning otherwise). With `offload_prompt_only=false` the disk hit covered all of it;
the default would have stopped at the 104,029 prompt tokens. Launcher: `KV_OFFLOAD_PROMPT_ONLY=0`.

**What a prefix hit costs in fidelity.** Every hit — GPU or offloaded — recomputes the hit's last 128
tokens with the SWA window clamped to them (`swa_bounded_replay`, vLLM main's default with model
runner V2; it is also what lets the tiers skip the SWA cache), so what follows a hit attends to an
approximate window. `tools/sm120_perf/prefix_hit_probe.py` compared, on 12 prompts of 3K–96K tokens
(code and docs, a 6‑digit note right before the question; greedy, thinking off, 128 tokens, served
image), a fresh prefill against the same prompt served as a GPU hit, the same fresh prefill repeated,
and a fresh prefill whose chunk boundaries a queued request moves, plus a second turn fresh vs hit
(first‑token TV = total variation between the two top‑5 distributions):

| against the fresh prefill | identical 128 tokens | first token identical | first‑token TV (mean) |
|---|---:|---:|---:|
| GPU prefix hit (with the replay) | 0/12 | 7/12 | 0.31 |
| the same fresh prefill again | 1/12 | 7/12 | 0.27 |
| fresh, chunk boundaries moved | 1/12 | 6/12 | 0.30 |
| turn 2: hit vs fresh | 0/12 | 9/12 | 0.14 |
| turn 2: moved vs fresh | 1/12 | 10/12 | 0.15 |

A hit is as far from the fresh prefill as the fresh prefill is from itself. The served stack is not
deterministic from run to run once a prompt is longer than a few hundred tokens (identical requests
on an idle server: 12–17‑token prompts identical 3/3, 494 tokens 1/3, 2K tokens 0/3), and the replay's
approximation disappears in that spread; the note was answered correctly in all 36 second turns.
`swa_bounded_replay` stays on. Where the run‑to‑run spread comes from is not localized yet
(candidates: split reductions with atomics; top‑k tie‑breaking over the MXFP4 indexer logits once the
context exceeds the top‑k).

**One copy instead of four.** vLLM main de‑duplicates TP‑replicated MLA KV (vllm#48906, #50301: one
slot per chunk in the shared region, rank 0 stores, every rank loads the same bytes, and the disk tier
names that copy TP‑independently), but the gate admits only a single bare `MLAAttentionSpec` group,
and V4.1 has ten: eight bounded‑replay `SlidingWindowMLASpec` groups, the compressed group and the
compressor ring (a `CircularBufferSpec`). Only the compressed group is offloaded, and it is all MLA —
one latent KV head per layer that every rank computes in full; all 18,219 block files the per‑rank
layout had written (three server processes, prompts up to 227K tokens, generated tokens included)
held four byte‑identical rank slots (`tools/sm120_perf/tp_copies_check.py`).
`tools/dsv41_sm120/patch_vllm_offload_replicated_mla.py` gates the de‑duplication on the offloaded
groups (vllm#57652 widens the gate to multi‑group MLA but still counts the ring, which V4.0 keeps as a
`SlidingWindowMLASpec`; the one‑line narrowing and a V4.1 test case go to its review). Same probe and
GPU pool (992K tokens) as the disk‑tier table above, RAM tier 2 GiB (2.4M tokens with one copy), image
`dsv41-nightly-20260923-kvdedup`:

| | one copy per rank | one copy |
|---|---:|---:|
| host RAM and disk per token | 3.58 KB | 896 B |
| 64 GiB RAM tier | 149,796 chunks = 19.2M tokens | 599,186 chunks = 76.7M tokens |
| disk file per 128‑token block | 448 KiB | 112 KiB |
| stored per 209–221K‑token prompt | 749 MB | 198 MB (1.8 s of disk writes) |
| fresh prefill of the evicting prompts, stores running | 10.2–10.3k tok/s | 10.2–10.3k tok/s |
| hit from host RAM, GPU pool evicted | 0.49 s (179K tokens) | 0.63 s (221K tokens, 1,726 chunks) |
| hit from disk, GPU pool and RAM tier evicted | 2.53 s (209K; 749 MB read in 1.9 s) | **1.62 s** (221K; 198 MB read in 0.97 s) |
| turn 2 from disk (104K‑token prompt + ~1.4K generated) | 1.29 s (377 MB) | **0.77 s** (94 MB; 105,461 of 105,479 tokens) |
| 174K‑token prompt stored by the previous server process | 3.20 s (623 MB) | **1.47 s** (156 MB read in 0.78 s; the test server wrote it, the production server read it) |

The needle was answered after every hit. With one copy, rank 0 alone copies GPU → host (the same
bytes per link as before) and each rank loads the shared slot; the disk tier moves to a new
directory on its own (`FileMapper` adds `replicated_layout` to the namespace and drops the TP size),
so a file of the four‑copy layout is never read as a one‑copy row — the old tree ages out through
the retention script.

Two things the tier does not guard against. vLLM's directory name covers the model path, TP, KV dtype
and the offloaded layer names, not the indexer format, and a block file is read up to the block size
whatever its length (`tiering/fs/io.py`, `csrc/fs_io.cpp`): a file written with the other
`INDEXER_KV_DTYPE` would load as garbage, so the launcher puts `INDEXER_KV_DTYPE=fp8` into
`DIR/indexer-fp8`. And the host‑RAM region outlives the server: vLLM's default shutdown
(`--shutdown-timeout 0`, "mode=abort") kills EngineCore right after SIGTERM, before the creating worker
unlinks `/dev/shm/vllm_offload_<engine id>.mmap`, and `TieringOffloadingSpec` — unlike the plain
CPU spec — does not unlink it early, because the scheduler maps it after the workers. Every
`docker stop` (exit code 0) left the whole region behind (64 GiB after the production server,
2 GiB after the test server); the launcher deletes regions no process maps before it starts one
(upstream: vllm#57303, fixes in review #57313 / #57422).

**Served** since 2026‑09‑23: the 64 GiB RAM tier (one copy since 18:57Z: 76.7M tokens), the disk tier
with the hourly retention script, `KV_OFFLOAD_PROMPT_ONLY=0`; decode unchanged (prose 164–170 /
570–579 tok/s at one / eight streams, code 387–395 / 1416–1451, 73.6–75.7 steps/s single stream; with
one copy 166 / 559–564 and 380–387 / 1397–1410 at the same 73.7 / 75.7 steps/s — the tok/s spread is
the drafts' acceptance).

## Decoder SWA bounded replay: the CED prefill, measured (2026‑09‑23)

The report's Causal Encoder‑Decoder (§2.2, §3.2.2) projects the decoder layers' global KV from the
encoder output, so the only thing a prefill needs the upper layers for is their own sliding‑window KV —
and "Decoder SWA Bounded Replay" builds that from the prompt's last `n_win` = 128 tokens only, with the
window clamped to that segment (an approximation the model was post‑trained with): nearly half of the
prefill compute goes away. vLLM main does not do it yet; vllm#58132 (open, head 9a86c2c9) does: layers
21–39 (those after the last KV‑source layer, 20) run on each prefill's last 128 rows as a sub‑batch with
metadata of its own, eagerly inside the breakable piecewise graphs, while decode steps keep their FULL
graphs untouched. Its base has no change under `vllm/models/deepseek_v41/` after our pin and none of our
patchers touches its files, so `tools/dsv41_sm120/patch_vllm_decoder_replay.py` applies it as a patch
(step 7 of `Dockerfile.sm120-dsv41-nightly`, tag `…-ced`; runtime switch `VLLM_MOET_DECODER_REPLAY=0`). Its gates (no sequence‑parallel MoE, no Engram layer after the cut,
breakable graphs, a drafter window no wider than the target's) all pass on the served configuration:
the log says `Decoder SWA bounded replay: layers 21-39 prefill only each request's last 128 tokens`.
Its unit tests pass on the RTX 5090 (11/11).

Served configuration (TP4, DSpark k=5, FP4 KV, MXFP4 indexer, vision, KV offload) with and without it,
same image otherwise, same day (GSM8K and the fresh needles without it: the same image line on
2026‑09‑22):

| | without | with the decoder replay |
|---|---:|---:|
| fresh prefill, 19.4K tokens | 1.71 s (11.3k tok/s) | 1.05 s (18.4k tok/s), **1.63×** |
| fresh prefill, 163K tokens | 15.4 s (10.5k tok/s) | 9.2 s (17.6k tok/s), **1.68×** |
| fresh prefill, 391K tokens | 43.5 s (9.0k tok/s) | 24.8 s (15.8k tok/s), **1.75×** |
| 8 × 122K‑token fresh prompts at once | — | 53.8 s, 0 errors, peak 96,437 / 97,887 MiB (96,491 without, PERF10) |
| GPU KV pool | 3,454,536 tokens | 3,412,407 (−1.2 %: the sub‑batch buffers) |
| GSM8K‑200, thinking off | 194/200 | 193/200 (McNemar p = 1; 2 / 1 flips) |
| greedy agreement (24 short chat prompts), raw 128‑token completions | — | 24/24 and 12/12 identical |
| needle 27K–367K (8 fresh), 29K–400K (6, from the disk tier) | 8/8, 6/6 | 8/8, 6/6, the same answers |
| 12 prompts of 3K–96K (code, docs; `prefix_hit_probe.py`), note in turn 2 | 36/36 | 36/36 |
| decode, prose / code, one stream | 166 / 380–387 tok/s, 73.7 / 75.7 steps/s | 166–171 / 395–399, 75.3–75.5 / 75.5–75.6 |
| decode, eight streams | 559–564 / 1397–1410 | 559–561 / 1377–1403 |

On the long prompts the replay's deviation is the stack's own: its fresh greedy output against the
fresh output without it (the same 12 prompts as in "What a prefix hit costs in fidelity") first
diverges after 8.5 tokens (median; |Δlogprob| over the common prefix 0.039), where two fresh prefills
without it diverge after 14 (0.031), a prefix hit after 8.5 (0.030) and moved chunk boundaries give
0.054; the first token differs in 5 of 12 either way. Prompts of ≤ 128 tokens are exact by construction.
It changes what a prefill computes and the code is an open upstream PR, so it went through the gate
above before it was switched on: **served since 2026‑09‑23 21:19Z** (image
`vllm-moet-sm120:dsv41-nightly-20260923-ced`, the launcher's default). Without it, same image:
`EXTRA_DOCKER_ARGS="-e VLLM_MOET_DECODER_REPLAY=0"`; without it in the image: the `-kvdedup` tag
(`--build-arg VLLM_PR_58132=0`).

## MoE glue in one launch: the input quantization fused with the permutation (2026‑09‑23/24)

The first of the launch fusions from the 2026‑09‑22 inventory ("Where a decode step goes": the
inductor fusions are off for this model, so they are done by hand). Between the router and FC1 every
MoE layer ran seven kernels at decode: `per_token_group_quant_8bit` (bf16 → fp8 e4m3 per 128‑group with
UE8M0 scales, in `prepare()`), two fills (`m_indices = −1`, scales = 0), `_count_expert_num_tokens`,
`_fwd_kernel_ep_scatter_1` (expert offsets, m_indices), `_fwd_kernel_ep_scatter_2` (the row copy) and,
inside the DeepGEMM call, `transpose_and_pack_fp32_into_ue8m0` (the fp32 scales into DeepGEMM's packed
int32 MN‑major layout) — 8.6–9.4 µs per layer, 11.9 µs of kernel time in the 2026‑09‑22 profile, for a
few hundred bytes of routing and 36 rows of 5120.

`tools/dsv41_sm120/moe_quant_scatter/moe_quant_scatter_sm120.cu` does it in one launch. One CTA per
(token, expert) pair, two barriers: the CTA issues its token's row loads first (they do not depend on
the routing), rebuilds the routing table from `topk_ids` in shared memory (per‑expert counts — thread
*t* counts expert *t* itself up to 64 pairs, a shared‑memory histogram above that), and one block
reduction yields the rows of the experts before its expert (BLOCK_M‑aligned regions in expert order),
its rank among the earlier pairs of the same expert, the expert's count and the rows in use. The slot is
a pure function of `topk_ids` (vLLM's scatter assigns slots in atomic order; the gather undoes either).
It then quantizes the row exactly as vLLM's kernel does (same absmax with the same eps, same
`exp2f(ceilf(log2f()))` scale — compiled without fast‑math like vLLM's — round‑to‑nearest‑even e4m3
with the clamp before the conversion) into its slot and writes the UE8M0 exponent byte straight into
the packed scale tensor DeepGEMM accepts as is; the expert's rank‑0 pair writes the expert's
`m_indices` region, all CTAs fill the tail with −1. vLLM side (`patch_vllm_moe_quant_scatter_sm120.py`,
one file): `DeepGemmFP4Experts.expects_unquantized_inputs` → the bf16 rows reach `apply()`, where the
fused kernel takes ≤ 1024 pairs (TP without EP, UE8M0 format); anything else — prefill, mixed steps
above 170 tokens — runs vLLM's `per_token_group_quant_fp8` there and the unchanged permute, i.e. the
same kernels as before, only moved. `VLLM_MOET_MOE_QUANT_SCATTER=0` restores the quantization in
`prepare()`.

Bit‑exact by construction and by test (`test_moe_quant_scatter_sm120.py`, RTX 5090): per pair the same
fp8 row bytes and scale bytes as vLLM's kernels through each path's own inverse permutation, identical
`m_indices`, the hardware e4m3 conversion identical to c10's software one (what vLLM instantiates) on
every finite bf16 value under 81 scale exponents, and the MoE output of the full DeepGEMM chain
bit‑identical at 1–64 tokens; the integration test drives the patched class the way
`FusedMoEModularKernel` does (fused ≤ 1024 pairs, the deferred fallback above, the switch off) with
identical outputs. Per layer in a CUDA graph on the RTX 5090 (E = 384, K = 5120, top‑6; DeepGEMM's pack
kernel counted with FC1): quant + permute **7.9 + 1.6 → 3.8 µs at 6 tokens**, 6.6 + 1.0 → 3.1 at 1,
7.9 + 1.9 → 3.5 at 12, 9.6 + 3.7 → 5.5 at 48, 10.0 + 4.3 → 6.8 at 64; on the RTX PRO 6000 (next to a
serving Qwen) 11.7 + 1.7 → 5.2 at 6 tokens, 14.2 + 4.8 → 8.0 at 64. Step 8 of
`Dockerfile.sm120-dsv41-nightly` (tag `…-moeqs`).

Served configuration (TP4, DSpark k=5, FP4 KV, MXFP4 indexer, vision, KV offload, CED), the `-ced` image
against the same plus this step, same day:

| | `-ced` (23.09 21:37Z–22:55Z) | `-moeqs` |
|---|---:|---:|
| decode, prose / code, one stream | 157–172 / 377–391 tok/s, 73.8–74.1 / 74.4–74.8 steps/s | 158–193 / 394–417 tok/s, **75.7–77.9 / 77.4–77.8 steps/s** |
| decode, eight streams | 559–564 / 1397–1410 tok/s, 31.3–31.9 / 33.8 steps/s | 567–573 / 1421–1422, 31.7–32.0 / 34.6–34.7 |
| fresh prefill 19.4K / 163K / 391K | 1.05 / 9.2 / 24.8 s | 1.03 / 9.15 / 24.7 s |
| greedy agreement (24 short prompts), raw 128‑token completions | — | 24/24 and 12/12 identical to the `-ced` window |
| GSM8K‑200, thinking off | 193/200 | 193/200 (McNemar p = 1; 1 / 1 flips) |
| needle 27K–367K (6) | 6/6 | 6/6, the same answers |
| 8 × 122K fresh prompts at once | 53.8 s, 0 errors | 52.2 s, 0 errors |
| GPU KV pool | 3,412,407 tokens | 3,414,187 |
| a C1 decode step's trace (rank 0) | 7 glue kernels × 43 MoE layers | `moe_quant_scatter_kernel` × 43 (40 target + 3 drafter layers), 3.9 µs each; none of the seven |

**Served since 2026‑09‑23 23:15Z** (the launcher's default image). `EXTRA_DOCKER_ARGS="-e
VLLM_MOET_MOE_QUANT_SCATTER=0"` turns it off on the same image, `IMAGE=…-ced` is the image without it.

### Where the served step goes now, and what the next fusions measured (2026‑09‑24)

Rank‑0 trace of a one‑stream decode step on the `-moeqs` image (torch profiler, 2026‑09‑24;
40 layers, 254 µs each on the critical path, ~12.9 ms per step with the DSpark drafter's 1.04 ms):
FC1 + FC2 of the routed experts 110 µs per layer (118 + 59 MB of FP4 weights — the bandwidth floor),
the two all‑reduces 24, the two mHC TileLang kernel pairs 21 (5.3–6.4 + 4.7 µs per sublayer), the four
dense MXFP8 GEMVs 36 (38.6 MB at ~1.07 TB/s), sparse MLA + merge 15, the router 11 (cuBLAS bf16 GEMM
4.9 + split‑K reduce 2.7 + `_dsv4_topk` 3.4), the activation quantizers on the critical path ~6, the
remaining small kernels ~20, and ~36 launch gaps of ~0.35 µs. The shared expert runs on the aux
stream behind FC1. cuBLAS BF16 small‑M GEMMs (12 × 25.6 µs, 41 × 5.2 µs and 44 split‑K reduces per
step, 0.65 ms; the drafter's single `s16816gemm_relu` 229 µs) are the largest not yet examined item.

Three fusions were then built and measured on the RTX 5090 (all in `tools/dsv41_sm120/`):

- **vllm#57679** (`patch_vllm_query_quant_gate.py`): vLLM's own fused q/kv RMSNorm + MXFP8 quantization
  of the query has been unreachable since vllm#53793 renamed the attribute its gate reads; the
  one‑line fix restores it. Bit‑identical to the separate norm + quantize (FP8 bytes, swizzled scale
  bytes, normalized KV; 1–170 tokens) and the GEMV consumer gives the same output. One launch less
  per layer (≈ 1.6 µs + a gap). Step 9 of the Dockerfile behind `--build-arg VLLM_PR_57679=1`,
  **not served yet** (to be bundled with the next image).
- **Fused router** (`moe_gate_topk/`): gate GEMV + sqrtsoftplus + bias + top‑6 + renorm in one launch,
  bit‑exact against `dsv4_topk` — and not faster with cold gate weights (8.6 vs 8.1 µs at 6 tokens):
  the last‑CTA hand‑off serializes ~2 µs the three kernels overlap; upstream's `ll_bf16_gemm` (CuTe
  DSL, gated off sm_120) runs here but loses to cuBLAS above 4 tokens. Not applied.
- **mHC pre epilogue** (`mhc_pre_norm/`): the TileLang `mhc_pre_big_fuse_with_norm` (split sums,
  Sinkhorn, collapse, RMSNorm, aux) as one CUDA kernel with 32 + H/8 threads per token, bit‑identical
  on every output (its reduction orders reproduced from the generated CUDA) — but 4.5 µs vs 3.5:
  both are bound by warp 0's 20 Sinkhorn iterations (~93 ns each), ours carries ~1 µs of unexplained
  overhead. Not applied; the starting point for a fused sm_120 mHC (DeepGEMM's `mega_mhc` needs
  tcgen05/TMEM/TMA, i.e. sm_100).

## The mHC boundary off the critical path (2026‑09‑25)

Between two sublayers the served step runs two TileLang kernels back to back on the model stream:
`mhc_fused_tilelang` (the post‑mapping of the four residual streams fused with the fn projection,
grid [T, 12, 8] × 128, 5.3–6.4 µs) and `mhc_pre_big_fuse_with_norm` (split sums, sigmoids, the 4×4
Sinkhorn × 20, the collapse with the carried pre‑mix, RMSNorm, the draft aux; [T] × 96, 4.7 µs) —
10.5 µs per boundary, 80 boundaries per step. Reading what the next sublayer actually needs changes
the picture: it consumes the collapsed, normalized input, which depends on the post‑mapped streams
and on the pre‑mix the *previous* boundary produced (shifted mHC). This boundary's own projection
and Sinkhorn produce the post mix, the residual mix and the next pre‑mix, and those are first read
at the *next* boundary — after the whole sublayer. Upstream has exactly that split for GB200
(`mhc_pre_delayed_overlap` in `models/deepseek_v41/nvidia/ops/mhc.py`: the input collapse on the
model stream, the projection + coefficients on `mhc_stream`, joined after the sublayer), gated to
SM100 + DeepGEMM's TF32 prenorm GEMM.

`tools/dsv41_sm120/mhc_overlap/` puts that design on sm_120 with two kernels
(`mhc_post_norm_sm120.cu`) and one patcher (`patch_vllm_mhc_overlap_sm120.py`, one file:
`ops/mhc.py`):

- **`mhc_post_norm`** — the critical path in one launch, one CTA of H/8 threads per token: the
  post‑mix in fp32, the bf16 streams, the collapse of the rounded streams with the carried pre‑mix,
  TileLang's sum of squares (64 threads × 16 positions per 1024‑block, the 0,8,1,9,… summation, the
  64‑wide butterfly), the RMSNorm, the stream mean for the drafter. Bit‑identical to the TileLang
  pair on every output for 1–64 tokens — the post‑mix reproduces the contraction nvcc chose for
  TileLang's `pm * x + Σ cm · r` (the first product rounded, `pm * x` fused into it, one fma per
  remaining stream; the test's `--modes` shows the two other candidates off by a bit), and the
  streams equal `mhc_post_tilelang`'s (the >32‑token path) too. 3.3 µs on the RTX 5090 against the
  pair's 7.4 (upstream's own split, `mhc_post` + `pre_norm[input]`, would be 4.9). Launched with
  PDL optionally (`pdl=`): no gain in a graph here (+0.5 µs behind a small kernel), off by default.
- **`mhc_proj`** — the projection for the side stream: `mhc_fused_tilelang`'s split partials
  (`mixes[8, T, 24]`, `sqrsum[8, T]`, the fp32 post‑mix recomputed from the same inputs) from one
  CTA per (split, block of 4 tokens) holding all 24 outputs, the fn slice read once per block: 8 ×
  ⌈T/4⌉ CTAs instead of 96 T. Bit‑identical (same position‑to‑thread mapping, same fma chains per
  token and output, TileLang's warp butterfly 16..1, the cross‑warp sum 0..3 in order). Alone it is
  slower than the TileLang kernel (7 vs 4 µs at 6 tokens) — it runs behind the sublayer, where what
  counts is how few SM slots it holds: with the served TileLang kernel on the side stream the
  boundary still cost +8.7 µs next to a weight‑streaming stand‑in (its 576 CTAs at 6 tokens take the
  SMs the GEMVs need), with this kernel +4.8.
- The coefficients themselves stay TileLang's: `mhc_pre_big_fuse_with_norm` in its `"stats"` split
  mode (3.2 µs, [T] × 96), bit‑identical to the fused epilogue. Above 32 tokens the side stream
  runs the served path's DeepGEMM TF32 prenorm GEMM on the bf16 streams the kernel wrote (the same
  bits as today above 32), then the same stats kernel.
- The patch makes `supports_mhc_overlap` true on sm_120 (hidden 5120, hc 4, DeepGEMM, no ubatching,
  the extension loads), raises `MHC_OVERLAP_MAX_TOKENS` from 16 to 64 there (every captured decode
  batch; `VLLM_MOET_MHC_OVERLAP_MAX_TOKENS`), and routes `mhc_shifted_post_pre(stream=…)` to the
  path above; the first layer's `mhc_pre` and the two Engram layers' `mhc_post` + `mhc_pre` take
  upstream's overlap code unchanged (its TF32 GEMM + `input`/`stats` epilogues are the same kernels
  as today's, so the same bits). Decode only: the decoder layer disables the side stream outside
  FULL‑graph capture and above the token limit. `VLLM_MOET_MHC_OVERLAP=0` restores the pair;
  `VLLM_MOET_MHC_PROJ=fused|tf32` swaps the side‑stream projection for the TileLang kernel or the
  TF32 GEMM (the latter not bit‑identical below 33 tokens).

Measured on the RTX 5090 (`test_mhc_overlap_integration.py --bench`: one boundary followed by a
stand‑in for the sublayer, the join after it as in the decoder layer; the served path is the
TileLang pair in front of the same stand‑in):

| tokens | stand‑in | served boundary | overlapped (`ours`) | with the TileLang projection on the side stream |
|---:|---|---:|---:|---:|
| 1 | weight‑streaming GEMVs | +5.9 µs | **+2.7** | +8.2 |
| 6 | weight‑streaming GEMVs | +10.2 | **+4.8** | +8.8 |
| 6 | compute‑bound GEMM | +9.8 | **+4.2** | +8.7 |
| 16 | weight‑streaming GEMVs | +16.1 | **+9.0** | +15.2 |
| 32 | weight‑streaming GEMVs | +20.7 | **+11.8** | +20.6 |
| 48 | weight‑streaming GEMVs | +23.8 | **+7.6** (TF32 path) | — |

At the served shape (6 tokens) the boundary costs the step ~5 µs less than the pair; 78 boundaries
take the shifted path (the first layer and the two Engram layers keep upstream's), so the expected
gain was ~0.4 ms of a 12.9 ms step on the 5090 numbers. The drafter's three layers are built without
`mhc_stream` and keep the pair (6 boundaries, ~30 µs).

**In the served graph (4× RTX PRO 6000, TP4, window 5, 2026‑09‑25).** Three server starts, each
bit‑identical to the `-moeqs` window (greedy agreement 24/24, raw completions 12/12, needle 6/6 the
same answers, GSM8K‑200 193 = 193 with 0 flips), prefill unchanged (1.03 / 9.2 / 24.7 s at 19K /
163K / 391K), 8 × 122K stress 0 errors, KV 3,414,865 tokens:

| variant | decode graph span (rank 0, 6 tokens) | C1 prose / code steps/s | C8 prose / code |
|---|---:|---|---|
| `-moeqs` (window 4) | 11,152 µs | 75.7–77.9 / 77.4–77.8 | 31.7–32.0 / 34.6–34.7 |
| a. as measured on the 5090: `record_stream`, PDL launch, fork before the kernel | 10,883 | 77.3–78.9 / 79.1–79.2 | 33.2 / 36.4 |
| b. references instead of `record_stream`, PDL off | 10,899 | 78.0–79.6 / 79.3–79.7 | 33.0–33.1 / 36.3–36.4 |
| c. + fork after the kernel | 10,803 | 78.1–80.2 / 79.8–80.0 | 33.6 / 36.1–36.2 |
| **d. + the all‑reduce in front of the kernel (`fuse_mhc_all_reduce`), PDL on — served** | **10,681** | **79.0–80.6 / 80.7–80.9** | **33.4–33.7 / 36.9–37.4** |

What the rank‑0 traces (`run19..22_*` in the profiles directory) showed, boundary by boundary:

- The boundary kernel takes 3.8 µs (4.4 with PDL) where the pair took 5.9 + 4.7; the projection runs
  12–16 µs on the side stream (7 alone) behind the first GEMVs, the stats kernel 5 µs behind the
  second, both done ~30 µs into a 87 µs attention sublayer. Their bytes are not free: the first
  GEMV and the quantizer next to them run ~0.5–1 µs longer, i.e. of the 6.9 µs the pair leaves the
  critical path, ~1.5 come back as contention (the fn read, 1.97 MB per boundary, has to happen
  either way; behind a bandwidth‑bound sublayer it costs its bandwidth).
- **The kernel starts 2.2 µs after the all‑reduce ends, where the TileLang kernel started 0.2 µs
  after it** — 77 × 2 µs ≈ 160 µs per step. Not PDL (the same gap with it on and off) and not the
  fork order (the same after moving the fork behind the kernel — although a micro‑benchmark of the
  captured pattern does show a kernel captured as a sibling of the side branch starting 8 µs late
  on the RTX PRO 6000, so the order is kept). What the kernel has that the TileLang kernel had not
  is the second parent: the decoder layer joins the side stream right before it, so in the graph
  the kernel depends on the all‑reduce *and* on the stats kernel of the previous boundary, and a
  node behind an NCCL node loses its fast path when it has another parent. Variant d moves the
  join onto the all‑reduce itself through upstream's `fuse_mhc_all_reduce` (the all‑reduce leaves
  `wo_b` / the MoE and runs in `mhc_shifted_post_pre`, right before the kernel; on sm_120 a plain
  `all_reduce`, not the MNNVL kernel the GB200 path uses): the kernel's gap drops to 0.19 µs, the
  all‑reduce's own gap stays at its 2.8 µs (unchanged from `-moeqs`), the attention sublayer goes
  from 91.4 to 83.6 µs and the step from 11,152 to 10,681 µs (−4.2 %). The same all‑reduce kernel
  on the same operands, so the bits are the same (agreement 24/24, raw 12/12).
- `record_stream` under graph capture keeps every recorded block allocated until the capture ends:
  +0.15 GiB of graph memory and +0.14 GiB of "peak activation" per rank, −3.2 % KV tokens
  (3,303,226). Holding Python references on the stream object until the next boundary — by which
  point the decoder layer has joined the stream — costs nothing and gives the KV back (3,414,865;
  3,412,830 with PDL's slightly larger graphs).

Served since 2026‑09‑25 20:50Z (`vllm-moet-sm120:dsv41-nightly-20260923-mhc`, also carrying
vllm#57679 — step 9 — whose `_q_kv_norm_quant_kernel` replaces the norm + quantize pair in every
attention sublayer). Left on the table: the drafter's six boundaries (its layers have no
`mhc_stream`), an A/B of the PDL launch inside variant d (`VLLM_MOET_MHC_PDL=0`: the kernel would
be 0.5 µs shorter if its gap stayed at 0.2 µs), and the projection's bandwidth behind the GEMVs
(fn in fp32 is 157 MB per step; the checkpoint's fn is fp32, so bf16 storage would change bits).

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
  instantiations. FlashInfer `main` ships them since
  [#5197](https://github.com/flashinfer-ai/flashinfer/pull/5197) (runtime page sizes, V4.1 dual
  cache with the FP4 record — measured above); what remains upstream is the vLLM side (let sm_120
  select `nvfp4_ds_mla` through FlashInfer's `fp8_dsv41_fp4_ca` once the pin carries #5197) and,
  in FlashInfer, a multi‑group BF16‑QK prefill for the V4.1 dual cache (today's SG kernel is +21 %
  on the 4096‑row chunk).
- Upstream, DeepGEMM: the vllm‑project fork's SM120 port needs nv_dev's per‑group BLOCK_M policy
  back (`get_theoretical_mk_alignment_for_contiguous_layout(expected_m, num_groups)` → 64 on
  SM120 when the per‑expert M is ≤ 64); until then every vLLM main build pads MoE decode batches to
  128 rows per expert on sm_120 and the grouped GEMM takes 2×. `patch_deepgemm.py` carries the fix.
- `EmulationMxfp8LinearKernel` is no longer selected in the serving log (dense → FlashInfer
  CUTLASS + GEMV, `wo_a` → grouped GEMV); the remaining BF16 GEMMs are BF16 checkpoint weights.
- Bench recipe (`bench/recipes/`) for `deepseek-v4.1-flash/pro6000x8-tp8-dspark` not yet
  registered; the numbers above are single‑shot smoke measurements, not a release row.
