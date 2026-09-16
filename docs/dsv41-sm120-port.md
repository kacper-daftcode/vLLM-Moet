# The DeepSeek-V4.1-Flash SM120 port

**Target:** the official `vllm/vllm-openai:deepseekv41-flash-0909` image — the tag the
[vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml)
pins for DeepSeek‑V4.1‑Flash on NVIDIA — serving the **official checkpoint**
(`deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a`, 510 GB: MXFP8 dense, MXFP4 experts, UE8M0
scales) on **8× RTX PRO 6000 Blackwell (sm_120)**. The recipe lists H200/GB200/GB300/MI350X as
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
| 5 | DeepGEMM SM120 FP8 paged MQA logits (indexer) + its metadata: `block_kv == 64` only | ratio‑1 indexer layers (128 states/page): `Assertion error attention.hpp:262: block_kv == 32 or block_kv == 64`, `:320 … block_kv == 64` | three host asserts admit 128 for the FP8 cache (`patch_deepgemm.py`); device kernel/scheduler are templated on `BLOCK_KV` (128 rows = one group of eight 16‑row MMA warps, `SPLIT_KV` stays 128, ~71 KB smem) |
| 6 | DSpark `enable_adaptive_verification:true` (recipe's NVIDIA default) | `DeepseekV4IndexerBackend … does not support` adaptive verification on this path | serve with `enable_adaptive_verification:false` (the recipe's AMD override); DSpark still drafts 5 tokens |

Not gaps (verified working on sm_120 in this image): DeepGEMM MXFP4 MoE (`DEEPGEMM_MXFP4` /
`DeepGemmFP4Experts`), MXFP8 dense GEMM (`FlashInferCutlassMxfp8LinearKernel`; a few shapes fall
back to `EmulationMxfp8LinearKernel` — a perf item, not a blocker), `fp8_ds_mla` KV format, FP8
indexer cache, Mega‑mHC TileLang kernels, Engram in pinned host RAM (2 × 11.8 GiB per rank),
ViT with FLASH_ATTN, DSpark drafter, cudagraphs FULL_AND_PIECEWISE.

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

So four cards work for ≤ ~2M total KV tokens without any offload machinery; the 1M‑token window
is possible only by trading KV (KV/token at TP4 is ~2.6 KB/GPU). Headroom is the limiting factor, not fit: the vLLM‑Moet expert tiers (2‑bit base,
FP4 delta, base cache) would halve the 259.5 GiB of experts and are the route to comfortable TP4
or TP2, but they need a port of the `moe_w2` stack onto this vLLM base plus K=5120 / K=576·1152
cubin families.

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
- `EmulationMxfp8LinearKernel` still serves a few dense shapes (perf, not correctness).
- Bench recipe (`bench/recipes/`) for `deepseek-v4.1-flash/pro6000x8-tp8-dspark` not yet
  registered; the numbers above are single‑shot smoke measurements, not a release row.
