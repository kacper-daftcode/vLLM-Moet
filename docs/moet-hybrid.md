# The MoET hybrid: DeepSeek-V4.1-Flash on 4× RTX 5090 with CPU-computed experts

The official `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint (552B MoE, MXFP4 experts, UE8M0 scales) served at native
quality on four 32 GB GeForce RTX 5090 cards — a box the model does not fit. The cards hold the dense weights, the
KV cache and **110–120 of the 384 experts per MoE layer per TP rank**; every other routed expert of a decode step is
**computed on the CPU** from the checkpoint's own bytes in pinned host RAM, in parallel with the GPU, and its rows are
injected into the GPU's own MoE kernel before the top-k sum. No re-quantization, no replay: the hybrid's output is
**bit-identical to the all-resident b12x path** on every step. The reference box: a Threadripper PRO 9975WX (32 cores,
4 CCDs), 498 GB of DDR5, 4× RTX 5090 on PCIe 5.0 x16, NVMe for the stores.

## Measurements

The launcher's defaults, TP4, no DSpark, 512K context. The full record of 2026-11-03/04 — the context curve, the 2- and
4-stream soaks, the energy per token, Fraser Price's BRRR bench and the other published ways to run this model below the
datacenter — is [`benchmarks/moet-hybrid-4x5090-2026-11-04.md`](benchmarks/moet-hybrid-4x5090-2026-11-04.md).

| | |
|---|---|
| decode, single stream, short prompts | **145 / 125 tok/s** ('Hello' / 'def fib(n):', greedy; eleven boots of the same configuration span 134–149 / 110–129). The step is 7.5 ms, of which ~2.2 ms is the GPU waiting for the CPU tier on the ~10 layers per step that miss (one expert's full CPU latency each, ~130 µs in situ) |
| decode on the BRRR script's sampled code generation | **93–94 tok/s on one stream at 2K / 16K / 64K of prompt** (temperature 1.0, to EOS ≤ 2,048 tokens: more distinct experts per step than the greedy short prompts), 135–155 tok/s aggregate at 4 streams, 136–166 at 8 |
| decode behind context | **83 / 84 / 73 / 70 / 87 / 85 / 82 tok/s behind 2K / 8K / 32K / 64K / 128K / 256K / 512K** — the sparse attention keeps the step independent of the prompt length; the spread is the expert hit rate of the texts |
| prefill (streamed experts, 4096-token chunks) | **2.4–2.7K tok/s at 2K, 4.4K at 14K, 4.7–4.8K at 30K**, 4.3–4.6K cold at 512K — PCIe-bound: 0.09–0.10 ms per missing expert per rank (~50 GB/s per link); the missing set of a chunk grows sublinearly, so a bigger chunk streams fewer bytes per token |
| long context | needle-in-a-haystack **32K / 64K / 256K / ~512K: 3/3 each** (depths 0.1 / 0.5 / 0.9; TTFT 8.5 s at 32K, 14.5–15 s at 64K, 54–60 s at 256K, 117–118 s at 512K = 4.3–4.4K tok/s cold); 128K prompts 27–35 s TTFT |
| concurrency | 10-min soaks of mixed prompts: **2 streams 106.9 tok/s aggregate (53.5 each), 4 streams 123.4 (30.9 each)**, 0 errors; 60 min × 4 streams: **2,197 requests, 0 errors, no leak**, 114.6 aggregate — every stream brings its own expert misses (~136 CPU experts per 4-token step), so the hybrid serves one stream fast and several correctly but slowly; `MAX_NUM_SEQS=1` is the serving default. A 30-min single-stream soak of mixed prompts holds 84.4 tok/s flat with the 110-slot pool (87.6 with 120 slots at a 64K context) |
| power | 1.16 kW under one stream (socket ~320 W + the four cards ~850 W at the 400 W cap) → **13.8 J per generated token**; 10.2 J at 2 streams, 8.3 J at 4; 366 W idle with the model loaded |
| quality, paired against the native FP4 reference (same prompts, greedy, thinking off) | **HumanEval 160/164 vs 160/164** (0 discordant, 127 identical programs, McNemar p = 1.0); **MATH-500 level 5: 55/60 vs 55/60** (1 / 1 discordant, 56 identical answers, p = 1.0) — 2026-11-04 on the served defaults; 2026-10-26 on the 120-slot configuration: 159/164 vs 160/164, 55/60 vs 55/60 |
| identity | bit-exact against the all-resident b12x path within one boot: 21,040 compact calls per rank, 98,220 CPU-computed rows, 0 differ (`DEBUG_COMPACT=1`; re-run PASS at the 110-slot pool); the CPU kernel bit-exact against b12x's GPU kernels on four parity sets (0 of 890,880 `route_output` elements differ) |
| expert hit rate (pairs resident) | 0.83–0.85 on short prompts, 0.87 on HumanEval, 0.91 on MATH-500 |
| host RAM while up | ~450 GB (the expert store 65 GB per rank + the Engram tables 2 × 23.6 GiB per rank, pinned), MemAvailable ~45 GB |
| VRAM | ~29.3 GB per card at `GPU_MEM_UTIL` 0.90 with 110 expert slots; KV cache 1.28 GiB (nvfp4 MLA + mxfp4 indexer KV = 1.98 requests of 512K; a sequence's KV cost is sublinear in its length) |
| boot | ~6 min (weights 160–250 s + 65 GB of records pinned per rank in ~60 s + warm-up) |

## How it works

Per TP rank (the four ranks are symmetric and need no CPU-side coordination):

1. **Resident pool.** Each MoE layer's expert parameters are created with R = 110 rows instead of 384; vLLM's weight
   loader maps a resident expert to its slot and skips the rest (its own path for non-local EP experts). The pool is
   b12x's prepared storage; a per-layer device table `slot[e]` (−1 = not resident) is read in-graph. 40 of the slots
   hold a **static placement** from routing frequencies (`routing/placement.py`); the other 70 are **LRU slots** filled
   by promotions (below). At equal VRAM, dynamic slots beat frequency-placed static ones (`routing/promo_sim.py` on 237K
   traced positions: 107 + 13 → hit rate 0.777, 40 + 80 → 0.859, all dynamic → 0.88).
2. **The split, inside the captured decode graph.** After b12x's activation quant, a publish kernel remaps the router's
   ids through `slot[]`; resident pairs run on the GPU (b12x's compact W4A8 micro path), missing pairs get id −1 (b12x
   writes zero rows for them) and are written — `(token, expert, row, weight)` plus the step's quantized tokens — into
   the layer's mapped-memory mailbox with a system-scope release. Before b12x's TopKSum a wait kernel spins (bounded) on
   the layer's ack and copies the host's bf16 rows into `route_output` by zero-copy, so the final sum has the GPU's own
   order. Timeout → an error word → the step fails closed (never a silent zero row).
3. **The CPU tier.** A dispatcher thread per rank walks the mailboxes, groups a request's pairs by expert and runs a
   6-thread pool pinned to the rank's CCD with the CPU expert kernel (`cpu_moe/cpu_moe_b12x.cpp`): exact integer block
   dots on the MXFP4 nibbles × MXFP8 activations (AVX-512 VNNI), with the sm_120 block-scaled MMA's rounding and
   truncation rules reproduced (`qmma_probe.cu` measured them) — bit-identical to b12x's compact W4A8 path. ~111 µs per
   expert per token from DRAM (4.7 MB per expert slice at 45 GB/s per CCD), 130–165 µs in situ. The pool spins through
   a decode step's gaps and sleeps on a futex when the API is idle.
4. **The expert store.** Every rank's slices of the experts that can be missing live in pinned host RAM in the
   checkpoint's own layout (`cpu_moe/build_store.py` writes them once to NVMe, 269 GB; the loader pins 344 × 40 records
   per rank). A promotion is a 4.7 MB H2D of the record plus a small repack kernel into b12x's tiled layout.
5. **Promote-on-miss.** After each step the host schedules promotions on a copy stream: a missed expert is admitted
   only on its **second miss within 12 steps** (`PROMOTE_WINDOW`; promote-on-every-miss moved 27–53 records per step and
   its DMA slowed the CPU kernel ~15 %), the victim is the least recently used LRU slot not routed in this step, the
   slot table is written on the copy stream and the next steps wait on events — no recapture.
6. **Prefill.** Prompts longer than 8 tokens take b12x's non-compact kernels on vLLM's eager path; the chunk's missing
   experts are streamed from the pinned store into a scratch pool (`SCRATCH` slots, rounds of ≤ 128; the next round
   staged on a side stream during the current run) and a second b12x run on the ids remapped to scratch slots is added —
   the output is within the non-compact kernel's own run-to-run noise (1–2 bf16 ulps) of the all-in-one reference.
7. **Engram** (the model's n-gram tables, 2 × 23.6 GiB per rank) stay vLLM's own pinned tables, allocated exact-size
   (`--engram-config '{"use_thp": true}'`; torch's pinned allocator would round each to 32 GiB).

Rules the implementation keeps ("graph or death"): the decode step is one captured graph including the handoff; no
steady-state allocation (pools, mailboxes, staging, the prefill scratch are sized at startup); a miss is served inside
the step or the step fails closed; placement changes are stream-ordered table writes; prefill stays on vLLM's eager
path; identity against the all-resident path is the quality gate.

## Running it

```bash
# the image: the DeepSeek-V4.1 image with Dockerfile step 43 (the CPU tier) built in; -march of the serving host's CPU
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-dsv41-nightly \
  --build-arg VLLM_MOET_CPU_TIER_PATCH=1 --build-arg VLLM_MOET_CPU_TIER_ARCH=znver4 \
  -t vllm-moet-sm120:dsv41-nightly-df3bd094-r23-hyb .
# one-time: the expert store (269 GB on NVMe, ~1 min; verified against the checkpoint)
DSV41_MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash python tools/dsv41_5090/cpu_moe/build_store.py --out /srv/packs/dsv41-hybrid --check 8
# one-time per GPU model: the kernel tables (b12x's activation / scale tables and its weight layout permutation;
# one GPU, seconds; qmma_probe is built from cpu_moe/qmma_probe.cu, the layout probe runs in the serving image)
mkdir -p /srv/cache/ds41-hyb/cpu-tier-tables
CUDA_VISIBLE_DEVICES=1 tools/dsv41_5090/cpu_moe/qmma_probe --tables /srv/cache/ds41-hyb/cpu-tier-tables
python tools/dsv41_5090/cpu_tier/b12x_layout_probe.py --out /srv/cache/ds41-hyb/cpu-tier-tables
# serve
MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash STORE=/srv/packs/dsv41-hybrid CACHE_DIR=/srv/cache/ds41-hyb docker/sm120/run-dsv41-5090.sh
# the dev flow instead: the checkout mounted at /repo, the patcher applied at container start (~60 s), the plain -r23 image
MODEL_DIR=... STORE=... CACHE_DIR=... BAKED=0 docker/sm120/run-dsv41-5090.sh
```

`docker/sm120/run-dsv41-5090.sh` boots the measured serving configuration by default; its header lists every knob
(`TP`, `PLACEMENT`, `SPARE`, `PROMOTE_WINDOW`, `SCRATCH`, `PREFILL_OVERLAP`, `POOL_THREADS`, `POOL_SPIN_*`, `GPU_MEM_UTIL`,
`MAX_MODEL_LEN`, `SPEC_TOKENS`, `BAKED`, …). The static placement ships in `tools/dsv41_5090/routing/placements/`
(`routing/placement.py` derives one from routing traces, `routing/collect_routes.py`). The hybrid's own switches arrive
in the container as `VLLM_MOET_CPU_TIER_*` (documented at the top of `tools/dsv41_5090/cpu_tier/vllm_moet_cpu_tier.py`);
everything is inert unless `VLLM_MOET_MOE_CPU_TIER=1`, which the launcher sets. `tools/dsv41_5090/cpu_tier/tp4_window.sh
TAG [KEY=VALUE …]` runs one boot with the measurements (identity prompts, decode, memory, optional `TTFT_LENGTHS=…`,
`NEEDLE_LENGTHS=…`, `SOAK_MINUTES=… SOAK_CONC=…`, `HARD_PROBE=…` with `PROBE_DIR`) and keeps the artefacts under
`$CACHE_DIR/boots/` (`KEEP_DIR` overrides).

Host-side requirements: ~450 GB of RAM free at boot (the window script refuses below 465 GB available and drops the page
cache before the boot and again after the weights are read — the store pinning otherwise reclaims the loader's shard
pages under pressure); the pool cores (8r+2..8r+7 per rank) and their SMT siblings idle — a foreign busy thread on a
sibling halves the kernel on that core and the job's barrier waits for it. The GPUs' power cap (`docker/sm120/host/`,
400 W on the reference box after a card fell off the bus under full power) is re-applied by the launcher.

## Verification

- `cpu_moe/test_cpu_moe_b12x.py`: the CPU kernel against an independent scalar reference (synthetic + checkpoint slices).
- `cpu_moe/gpu_parity_b12x_{prepare,run,compare}.py`: against b12x's GPU kernels on recorded inputs (four sets, layers
  0/1/20/39: 0 of 890,880 `route_output` elements differ); `parity_sweep_{ranks,layer}.sh` (`PARITY_DIR=`) over ranks and
  a whole layer.
- `cpu_tier/test_cpu_tier.py`, `test_promotion.py`: the tier end to end in captured graphs (rows checked, fail-closed, the
  U/M promotion protocol); `cpu_tier/mini_identity_runs.sh` on the mini model (`make_mini_config.py`, `make_mini_real.py`).
- `DEBUG_COMPACT=1` with `EXTRA_ARGS=--enforce-eager`: every compact call recomputed with all its routed experts in the
  scratch pool and compared bit for bit — the within-boot identity check (TP4 boots are not bit-reproducible among
  themselves unless `NCCL_CHANNELS` is pinned, so cross-boot comparisons need the pin).
- HumanEval / MATH-500 paired against the native references (`tp4_window.sh HARD_PROBE=humaneval,math500`).

## Benches and tools

`cpu_moe/bench_real_records_b12x.py` (the kernel on real records — the synthetic ones read 1.3× too fast),
`cpu_moe/bench_pattern_b12x.py` (the decode's job pattern per CCD with the cores' clock), `cpu_moe/bench_l3_vs_dram_b12x.py`
and `ccd_bw.cpp` (the expert from DRAM vs L3, one CCD's DRAM ceiling), `routing/cache_sim.py` and `routing/promo_sim.py`
(placement and promotion policies on the routing traces; `--streams N` models concurrency), `routing/route_predict_eval.py`
(layer L+1's router on layer L's input: top-6 precision 0.33 — the prefetch-by-prediction lever is closed),
`cpu_tier/ttft_probe.py` (streamed prefill TTFT on disjoint prompts), `cpu_tier/profile_anatomy.py` (the decode step's
GPU busy / idle anatomy from a torch profile), `cpu_tier/cpu_placement_probe.sh` (who runs on the pool cores).

## One or two GPUs: TP2 / TP1 through virtual ranks (branch `moet-hybrid-tp12`, 2026-11-04)

The same hybrid runs on 2× or 1× GPU with `TP=2` / `TP=1` in the launcher. b12x's compact W4A8 path — where the publish / inject
hooks live and whose numerics the CPU kernel reproduces — exists only at the 576-row TP4 slice, so on one GPU every expert stays its
4 / TP slices of 576 rows ("virtual ranks"): the store's per-rank records are the slices, b12x prepares them as R × V slice-experts
and runs the same compact kernels at N = 576, the router's top-6 becomes 6 × V (token, slice) pairs, one CPU pool (CCD) + mailbox
set per slice computes the misses with the unchanged kernel, and the compact TopKSum adds a token's 6V rows in a fixed order
(k-major, slice-minor; fp32, one bf16 rounding — the role NCCL's bf16 ring plays at TP4). Bit-identity with the all-resident path
holds within each configuration (`DEBUG_COMPACT`: 0 mismatches over 15,160 compact calls per rank at TP2 and 19,040 at TP1 on the
full model, 11-04); the TP4 code paths are untouched (V = 1). Measured on 32 GB cards (the slots per layer follow the VRAM: 46 at
TP2, 16 at TP1): **TP2 on 2× 5090: 45.9 / 47.1 tok/s** on 'Hello' / 'def fib(n):', TTFT 1.9 s at 2K and 7.4 s at 14K; **TP1 on 1× 5090:
39.8 / 33.5 tok/s**, TTFT 3.8 / 15.2 s (chunk 4096) — the step is the CPU tier (hit rate 0.54 / 0.44 at those slot counts: 3–4 misses
per layer-step); host RAM is the same ~450 GB (+23 GB at TP1 for the 9-static placement's records). The design, budgets and every
measurement: the private note `tp12-virtual-ranks-2026-11-04.md` (the hybrid's handoff directory).

Running it: the launcher's `TP=2` / `TP=1` set the 32 GB-card defaults (GPUS 0,1 / 0; `SPARE` 6 / 7 with the 40-static / 9-static
placement — `routing/placements/tp12-placement-0.026-uniform.json` for TP1 — `MAX_MODEL_LEN` 65536 / 32768, `PROMOTIONS` 4 / 1); `VRANKS=1`
forces plain ranks; `VRANK_CPUS` / `VRANK_DISPATCHERS` lay the pools out on another CPU geometry (default: TP rank r takes CCDs rV..rV+V−1,
dispatcher 8g+1 and pool 8g+2..8g+7 per CCD g). `SCRATCH` counts slices (an expert = 4 / TP slots); `DEBUG_COMPACT` needs `SCRATCH ≥ 48 × 4 / TP`.
`tp4_window.sh` takes `TP` / `GPUS` (the KPI lines of every rank carry their virtual ranks' in-kernel µs). The same image serves every TP
(`vllm-moet-sm120:dsv41-nightly-df3bd094-r23-hyb-tp12` = this branch's step 43 on the merged `-r23` layers): on it the TP4 path is the
record's (identity 0 mismatches over 19,080 compact calls per rank; 146.6 / 126.2 tok/s, TTFT 0.97 / 3.27 / 6.50 s at 2K / 14K / 30K — a
sample with the box's `kcompactd0` on a pool core's SMT sibling gave 143 / 121) and TP2 repeats the numbers above (0 mismatches over
15,160 calls per rank; 46.1 / 47.2 tok/s; TTFT 1.91 / 7.34 s; the compact-path outputs bit-identical to the pre-merge boots). The 96 GB cards (2× / 1× RTX PRO 6000: ~200–235 / ~90–100 slots per layer) are the target of this
support and were not measured here: the run book — host requirements, the launcher lines at 96 GB, the window sequence, the expected
numbers from the routing traces (~105–125 tok/s mixed / ~120–145 short at TP2, ~62–70 / ~80–90 at TP1) — is §13 of that note.

## Limits and open items

- DSpark does not pay for the hybrid: a verify step routes ~5× more distinct experts for 3.6× more tokens and the CPU
  kernel's cost per extra token (~73 µs) keeps it CPU-bound; k = 0 is the serving mode.
- Prefill is PCIe-bound; a CPU tail for experts with few tokens is the next lever (its numerics budget is 1 ulp of bf16:
  the streamed path sums two b12x runs).
- Concurrency: the CPU tier's cost is the number of distinct missing experts a step pulls from DRAM, and concurrent
  sequences almost never share their misses (34 → 82 → 197 distinct CPU experts per step at 1 / 2 / 4 streams) — a
  single-stream (at most two-stream) design on this box's memory bandwidth.
- `GPU_MEM_UTIL` 0.90 with 110 expert slots is the 512K default: the sparse-attention indexer's prefill peak needs ~1.3 GiB
  outside the budget (at 0.94 it OOMed on the 4th chunk of a 128K prompt); 0.94 with 120 slots fits a 64K context.
- The CPU kernel's idle spin in mode 2 (a DRAM stream per worker) is faster per expert but heats the memory side until it
  throttles after ~18 min of load; mode 1 (ALU spin) is the default.
