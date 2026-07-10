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
- `modelopt.py` (`ModelOptNvFp4FusedMoE`) — NVFP4 checkpoint path (**GLM‑5.2‑NVFP4**): same
  three hooks; the loader dequantizes modelopt NVFP4 (e2m1 codes + f8e4m3 block‑16 scales +
  per‑tensor `weight_scale_2`) to f64 — exact, all three factors representable — and
  re‑quantizes to the sign‑symmetric codebook (`nvfp4_to_codes_scales`; the UE8M0 block‑32
  output scales absorb `scale_2`). Golden‑tested EXACT on real checkpoint shards; forward
  op‑validated through the K=6144/K=2048 cubins on real weights.
- **BASE cache** (`VLLM_MOE_W2_BASE_CACHE_GB=N`, the "159B on one 5090" path): the packed
  2‑bit base planes (codes + UE8M0 scales, four sections per expert slot) live in pinned host
  RAM; the GPU pool caches hot experts through the same slot‑table/manager/eviction machinery
  as the delta tier. Decode misses zero the pair's contribution, bump an in‑graph miss
  counter, and the runner fetches all missing routed experts synchronously (batched pinned
  H2D, 51.6 GiB/s measured) and replays the step's graph once — replay bit‑identical to a
  resident forward (unit‑tested, `internal` test_base_cache). Prefill prefetches per layer
  via `ensure_resident`. TP MAX‑reduces the miss decision. Under **PP** a miss is local to
  its stage (per‑stage counter, inputs still held in the stage's static buffers), so each
  stage re‑runs only its own **segment** before activations flow downstream — no cross‑stage
  collective; TP ranks within a stage replay together. The replay iterates toward a **fixed
  point, adaptively**: corrected early layers can re‑route later layers onto experts the
  first pass never fetched (second‑order misses, otherwise zeroed *inside* the replay —
  measured as cross‑request greedy nondeterminism). The first‑order restore pass is
  mandatory; further passes run only while the step is within `VLLM_MOE_W2_FP_THRESH` of
  miss‑free (default 0 = mandatory pass only; file‑tunable via `…_FP_THRESH_FILE`;
  `VLLM_MOE_W2_FP_MAX` bounds ping‑pong). An unconditional loop collapsed decode at low
  coverage (GLM TP2 29→16 tok/s, DS4‑14 GiB 43→15 — chasing a moving target at up to 8
  forwards/step); the live A/B behind the default: thresh 16 → 19.2, thresh 0 → 28.3 tok/s
  at identical quality probes. Slots touched by any pass of a step are pinned against
  eviction until the next step (the passes must not cannibalize each other's fetches); on
  tight pools an emergency eviction pass (synchronous callers only) relaxes the 2‑tick
  coldness bound rather than leave a miss UNRESTORED. Coexists with the FP4 need‑pool (explicit
  `VLLM_MOE_W2_DELTA_GB`, three‑tier stack). Opt‑in `VLLM_MOE_W2_PREFETCH=1` adds a
  draft‑affinity prefetcher: an in‑graph routing log feeds a token→experts table, and each
  decode step's input ids (under MTP: last step's sampled+draft tokens) prefetch predicted
  experts on the side stream, overlapping the forward.

  **Pool sizing is the dominant knob — the engine reports it as a KPI.** The mandatory
  replay is per‑STEP, so decode tracks the fraction of zero‑miss steps, which is brutally
  non‑linear in coverage while the token hit‑rate barely moves. Controlled A/B (DS4 1×5090,
  MTP k=2, idle box, fox 512, median of 10+, 2026‑07‑10):

  | pool | coverage | token hit | decode (pre‑fix / shipped stack) |
  |---|---:|---:|---:|
  | 11 GiB (util 0.90) | 15.2% | 96.5–97.7% | 32.7 / **27–28 tok/s** |
  | 14 GiB (util 0.95) | 19.3% | 98.7–98.9% | 43.4 / **~31 tok/s** |

  The pre‑fix column is the single‑replay stack (silently kept second‑order zeros — the
  nondeterminism the fixed‑point restore later closed); the shipped column is the final
  adaptive‑replay + NVMe‑store stack (2026‑07‑10 evening, same box/bench idiom). The pool
  slope survives the stack change (+12–15% for 3 GiB). 14 GiB does NOT fit at util 0.90 (KV
  needs 0.46 GiB after graphs) — raise `--gpu-memory-utilization` alongside the pool. The
  engine logs pool
  coverage at startup and a windowed **`[base] KPI: replay X% of last N steps (avg Y missing
  pairs/step…)`** line every `VLLM_MOE_W2_KPI_EVERY` steps (default 500, always on) — the
  replay % is the number to watch when sizing a config; if it is high, grow the pool first.
  Earlier notes conflated configs here ("~38 tok/s", "~32 no‑MTP @ 19%"): those were mixed
  11–14 GiB runs on a loaded box. Bench‑prompt caveat: the fox loop flatters both locality
  and MTP acceptance (2.70); varied prompts at pool 11 land at 21–25 tok/s (acc 2.5–2.6).
  Also: the "51.6 GiB/s" pinned‑H2D figure is the PRO 6000; **an RTX 5090 does ~26.6 GiB/s**
  (measured under load), so 5090 miss fetches cost 2× the notes' assumption.
- **NVMe expert stores** (`VLLM_MOE_W2_STORE_DIR=<dir>`, `moe_w2_store.py`): the host stores
  behind the base cache and the FP4 need‑pool move to per‑rank **pack files** (row offset =
  `(li·E+ei)·stride`, stride 4 KiB‑aligned, JSON sidecar; sparse holes for unwritten
  layers). One store interface, three backends — pinned (default, byte‑identical to the
  pre‑store code), pack (buffered `preadv` thread pool into a pinned stage; page cache = RAM
  tier), and **tiered** (`VLLM_MOE_W2_BASE_RAM_GB=<GiB|auto>`): a pinned MRU arena whose
  hits are zero‑copy pinned views (H2D DMAs straight from the arena — recovers the pack
  backend's −9%) and whose misses read into the arena slot (buffered by default;
  `VLLM_MOE_W2_TIER_DIRECT=1` for O_DIRECT). Same‑night A/B, DS4 1×5090 @ 11 GiB pool:
  pinned 33.0 / pack 25.5 / tiered+20 GiB arena **32.8 tok/s (parity)** at RSS 26–33 vs
  42–44 GiB; GLM TP2 both stores on NVMe: expert‑store RAM ~568 → ~136 GiB, 28–32 tok/s
  (tol 0–8, adaptive replay), needle 4/4 to **121K prompt tokens** at a served 128K window
  (KV 157K tokens measured). Prefill batches are scan‑flagged (fill free arena slots,
  never evict the decode hot set — a caller flag, NOT a batch‑size heuristic: GLM's
  100+‑row decode replay fetches misclassified and froze the arena at −66%); the arena hot
  set persists to `<pack>.heat.json` and preheats on boot (57 GiB ≈ 35 s). **Boot‑from‑pack**
  (`_try_skip_requant`, all three loaders): a layer present in every serving pack skips
  dequant→re‑quant entirely — GLM TP2 second boot 408 s vs ~11 min, no ~405 GiB transient;
  the pack is a persistent quantization cache keyed by shape/config sidecar match. Also
  fixed here, exposed by long‑prefill testing but **pre‑existing**: the manager tick and
  forward‑thread paths ran concurrent seen‑snapshots into one shared pinned `_seen_host`
  (torch's two‑pass `nonzero` overruns its output when the input mutates mid‑call →
  TensorAdvancedIndexing.cpp:3008 assert → glibc heap corruption → dead worker on ≥16K
  prefills; a torn snapshot could also evict an in‑flight expert's slot). Snapshots are now
  serialized under a dedicated lock. Backends unit‑tested byte‑identical
  (`tools/test_store_backends.py`: 3 backends × cold/warm/reboot/evict/overflow/scan/
  preheat, both IO modes). Ops: packs on a bind‑mounted real FS (not overlayfs); ~1 TB NVMe
  for the full GLM stack; parity holds even on a Gen3‑x4 drive (3.7 GB/s) — steady‑state
  misses ride the page cache, cold shifts pay drive speed.
- **Deterministic unpermute**: the MoE output scatter used atomic `index_add_`, so identical
  runs wobbled (~1.6e‑2 on prefill) and greedy decode was not reproducible (surfaced by the
  PP determinism investigation; never PP‑specific). Valid `sorted_ids` form a permutation of
  `token*top_k+j`, so a bijective `index_copy_` + fixed‑order `sum(dim=1)` replaces it —
  6/6 bit‑identical repeats on prefill and decode, capture‑safe.
- MTP under **pipeline parallelism** (ported from the fork, inert off‑PP/off‑spec): draft‑token
  broadcast to rank 0 under async scheduling, `output_token_ids` trim on all ranks, and the
  drafter `embed_tokens` share across PP ranks (the NVFP4/DS4 MTP head ships no embedding of
  its own; upstream's share is gated to `pp_world_size == 1`). Validated on DS4‑Flash PP4
  (4× RTX 5090): acceptance to 2.81, 184 vs 93 tok/s (~2× MTP speedup).
- **NVFP4 KV cache** (`--kv-cache-dtype nvfp4`) for the SM120 sparse‑MLA path — a packed
  **352 B/token** layout (512× E2M1 + 32× E4M3 block‑16 scales at a fixed 2⁻⁶ global scale +
  64× FP8 rope) replacing the 656 B `fp8_ds_mla` layout, **1.86× less KV/token**. The write
  kernel is a standalone SM120 torch extension (`csrc/nvfp4_ds_mla/`); FlashInfer's sparse‑MLA
  JIT sources are patched with `ModelType::GLM_NSA_NVFP4` (`tools/nvfp4_flashinfer_sm120/`) so
  the packed bulk expands in place before QK and the FP8‑MMA pipeline is unchanged. Live on
  GLM‑5.2 TP4 (128K ctx): KV pool **+38%** (415K → 571K tokens), decode parity (104 tok/s —
  sparse reads only top‑2048), needle PASS to 126K, arithmetic + coherence intact. (Follow‑up:
  move the write kernel into `vllm._C`; FlashInfer 0.6.14 ships an AOT `sparse_mla_sm120.so`
  that must be removed so the patched JIT sources rebuild.)
- **Deterministic MoE unpermute** — the routed‑expert scatter‑add uses a bijective
  `index_copy` instead of `index_add`, removing the atomic‑accumulation non‑determinism in
  free‑running decode (matters under PP where physical KV‑block assignment varies run to run).
- **BASE cache (inverted delta)** — an opt‑in mirror of the delta tier: the 2‑bit base lives in
  host RAM and a GPU pool caches the hot experts, so a model whose 2‑bit base does not fit VRAM
  (e.g. GLM‑5.2 on 2 cards) can still serve at cache‑hit speed. GLM routing is concentrated
  enough (≈89% of token→expert routings served from ~20% of experts) to make this practical.
- **AFRAG prefill** — fragment‑major activation repack (single‑pass Triton into dedicated
  buffers) so each QMMA A‑fragment loads in one `LDG.128`; ~1.3× on the prefill GEMM,
  bit‑identical to the mc4 path, default‑on where the `mc4afrag` cubins ship.

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

**2026‑07‑09 — AFRAG prefill kernels ship and default on** (fragment‑major activations;
bit‑identical outputs, 1.30×/1.27× on the K=4096/K=2048 prefill GEMMs). Same 8k‑unique‑prompt
probe, median of 5: 1× PRO 6000 **4 777 → 5 340 tok/s (+11.8%)**; 4× 5090 TP4 (median of 3)
5 987 → 6 101 tok/s (+1.9% — the 1024‑token chunks shard per‑rank GEMM work too thin for the
full kernel win). TP2 has **not** been re‑measured with AFRAG yet — the tables carry its
2026‑07‑08 pre‑AFRAG figure (5 791), which is conservative. Opt out with
`VLLM_MOE_W2_AFRAG=0`.

Prefill rides upstream's FlashInfer SM120 sparse‑MLA path — which also makes a custom cubit
MLA‑prefill kernel unnecessary on this base.

**Batch scaling** (same knobs but `--max-num-seqs 32 --max-num-batched-tokens 2048`;
N identical‑length greedy requests, 384 tok each, aggregate decode tok/s, median of 5):

| concurrency | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| 1× RTX PRO 6000 | 156 | 290 | 493 | 659 | **933** |
| 4× RTX 5090 (TP4) | 198 | 460 | 762 | 1 006 | **1 560** |

6–8× aggregate from batch 1→32 — the 2‑bit expert reads amortize well across the batch
(decode stays HBM‑bound; the per‑step expert working set grows sublinearly with batch).
At 32 streams each request still gets ~29 tok/s (PRO 6000) / ~49 tok/s (TP4).

**Long context (512K) on one 96 GB card** — validated live (`tools/needle_probe.py`, unique
secret embedded in filler, greedy): PASS at **102 238 / 256 294 / 453 286 prompt tokens**
(depth 0.1) and at 453K with the needle mid‑context (depth 0.5); cold TTFT 27 s / 64 s /
~2 min. Server config for the 512K window on 1× PRO 6000: `--max-model-len 524288
--gpu-memory-utilization 0.97 --max-num-batched-tokens 2048 --max-num-seqs 1` with
`VLLM_MOE_W2=1 VLLM_MOE_W2_DELTA_GB=0` (the FP4 delta pool trades against KV headroom at
extreme context; with delta 1 GiB use ≤256K). The KV fit comes from DS4's compressed KV +
upstream's FP8 Lightning‑Indexer cache; vLLM reports 947K cached tokens in this config.

**Delta pool auto‑sizing.** `VLLM_MOE_W2_DELTA_GB=auto` resolves the delta‑vs‑KV trade
automatically: the pool allocation is deferred until after the KV cache is allocated
(and before cudagraph capture — the graphs bake the pool pointer), then sized as
`free VRAM − VLLM_MOE_W2_DELTA_RESERVE_GB` (default 3, capture/workspace headroom),
optionally capped by `VLLM_MOE_W2_DELTA_MAX_GB`. At extreme context it lands at 0 slots
(pure 2‑bit, pinned host store released — the manual `DELTA_GB=0` rule, without the manual
step); at 24K ctx / util 0.95 it recovers a ~1.6 GiB pool (133 slots) and benches at
166 tok/s single‑stream (1× PRO 6000, MTP k=2, graphs).
