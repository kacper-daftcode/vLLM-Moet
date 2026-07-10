# Frontier MoE on consumer Blackwell (SM120)

**Official vLLM v0.24.0 + a 7.4k‑line patch** that serves frontier Mixture‑of‑Experts models —
**GLM‑5.2 (753B)**, **DeepSeek‑V4‑Flash (159B)** and **Kimi‑K2.7‑Code (1T)** — on
consumer/workstation Blackwell (RTX PRO 6000, RTX 5090), hardware their official checkpoints
cannot even fit on. Three ideas carry it:

1. **2‑bit experts with FP4 recovery** — routed experts compress to a sign‑symmetric 2‑bit
   codebook on hand‑written SM120 SASS kernels; a runtime FP4 tier (delta cache + confidence
   gate) restores precision exactly where it matters.
2. **Tiered expert residency** — when even the 2‑bit base outgrows VRAM, it moves to pinned
   host RAM — and, one tier further, to an **NVMe pack file with a pinned‑RAM arena** — and
   the GPU becomes an **expert cache** (miss → batched fetch + bit‑identical graph replay).
   That puts 753B on two 96 GB cards and 159B on a single RTX 5090, and the packs double as
   a **persistent quantization cache** (reboots skip the re‑quant).
3. **A rebuilt serving base** — vLLM v0.24.0 actually working on SM120 (the release is
   broken‑as‑shipped), plus MTP speculative decoding (incl. under pipeline parallelism,
   bit‑deterministic), an **NVFP4 KV cache** (352 B/token), and agent‑ready tool/reasoning
   parsing.

---

## GLM‑5.2 (753B) — the headline model

Served from the official [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4)
checkpoint (433 GB): the loader re‑quantizes modelopt NVFP4 experts (e2m1 × e4m3 block‑16 ×
per‑tensor scale_2) to the sign‑symmetric 2‑bit planes at load — f64‑exact vs the reference
pipeline on real shards. Single‑stream, greedy, CUDA graphs:

| hardware | config | decode | max context window (served, needle‑validated) | host RAM |
|---|---|---:|---:|---:|
| **4× RTX PRO 6000 (TP4)** | 2‑bit + **MTP k=2** | **105 tok/s** | **256K** | — |
| 4× RTX PRO 6000 (TP4) | 2‑bit + MTP k=2 + **FP4 delta + confidence gate** | **83–85 tok/s** | 128K | — |
| **2× RTX PRO 6000 (TP2)** | **three‑tier + NVMe stores** + MTP + gate | **28–32 tok/s** | **128K** | **~140 GiB** |

- **4 cards:** prefill ~2.5k tok/s; MTP acceptance 2.3–2.8; needle retrieval **PASS to 126K**
  on the nvfp4 KV cache and **to 276K** on fp8 (331K window fits at util 0.95). GLM's nominal
  1M window is KV‑bound on 4 cards. Tool calling (`glm47`) + reasoning (`glm45`) parsers work
  out of the box — the endpoint drives coding agents (opencode) directly.
- **2 cards — a model that doesn't fit, running anyway:** the 2‑bit planes alone (~190 GiB)
  match the entire 2‑GPU VRAM budget. **Three tiers** make it work: NVMe pack + 57 GiB/rank
  pinned arena for the 2‑bit base → 46 GiB/rank GPU expert cache → a small gate‑filled FP4
  pool for precision (expert stores need ~136 GiB host RAM instead of ~568 pinned). The
  single‑user window is real: `--max-model-len 131072` with 8 GiB/rank KV = **157K tokens of
  KV measured**, needle retrieval **4/4 PASS at 36K / 86K / 121K prompt tokens** (fp8 KV,
  tol=0). Decode (MTP k=2, acceptance ~2.9): **28.3 tok/s** strict (tol=0), **31.7** at
  miss‑tolerance 8 — arithmetic and retrieval probes clean at both. Bare‑2‑bit quality
  artifacts ("capital of Poland: Krakow", garbled Polish) are corrected by the FP4 tier.
  Booting from existing quantization packs takes **~7 min** (vs ~11 for a full
  re‑quantizing load, which also stages ~405 GiB of transients).
- **NVFP4 KV cache** (`--kv-cache-dtype nvfp4`): packed 352 B/token vs 656 B `fp8_ds_mla` —
  **+38% KV pool** (415K → 571K tokens at equal settings) at decode parity, or the freed VRAM
  goes to the FP4 pool (the standing 4‑card config runs a 19.6 GiB/GPU pool + 175K‑token KV).

## DeepSeek‑V4‑Flash (159B)

Official checkpoint, 2‑bit experts + FP4 delta cache, MTP k=2, CUDA graphs (single‑stream
medians; prefill = 8k‑token prompt, uncached):

| hardware | decode | prefill 8k | max context window (served, needle‑validated) | host RAM |
|---|---:|---:|---:|---:|
| **1× RTX PRO 6000 (96 GB)** | **161 tok/s** | **5 340 tok/s** | **512K** | — |
| 2× RTX PRO 6000 (TP2) | 210 tok/s | 5 790 tok/s | 512K | — |
| 4× RTX 5090 (TP4) | 214 tok/s | 6 100 tok/s | 16K | — |
| **1× RTX 5090 (32 GB)** | **~31 tok/s** (14 GiB pool + NVMe stores) | ~400–540 tok/s | **32K** | **~30 GiB** |

Retrieval behind the window column: needle PASS at 453K on the PRO 6000 (947K‑token KV
measured) and at 29.7K on the single 5090 (131K‑token KV). "—" in host RAM = all‑VRAM
config, no host expert store.

**Batched serving** (aggregate decode tok/s at N concurrent streams; per‑stream in
parentheses at N=32):

| concurrency | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| 1× RTX PRO 6000 | 156 | 290 | 493 | 659 | **933** (29/stream) |
| 4× RTX 5090 (TP4) | 198 | 460 | 762 | 1 006 | **1 560** (49/stream) |

Four consumer 5090s match two PRO 6000s on decode. MTP acceptance ~2.6 tok/step across
configs. MTP also runs under **pipeline parallelism** (draft propagation + drafter embedding
share across ranks): DS4 on 4× RTX 5090 **PP4** does 184 tok/s vs 93 without (~2×), and greedy
decode under PP is **bit‑deterministic** (6/6 identical runs, with and without MTP).
Methodology: **[docs/v024-port.md](docs/v024-port.md)**.

---

## How it fits — 2‑bit experts at FP4 quality

We compress **only the routed experts** to 2 bits (the dense stack keeps the checkpoint's
precision — FP8 on DS4, NVFP4 on GLM) and recover FP4 precision adaptively:

- **2‑bit expert planes — the sign‑bias finding.** Naive 2‑bit *destroys* these models
  (degenerate loops). The cause is **sign asymmetry**, not error magnitude — the optimal‑L2
  codebook drops one sign's tail and the per‑expert bias compounds over dozens of layers.
  Forcing a **sign‑symmetric** `{−4,−1,1,4}` codebook at the same L2 error fixes it entirely
  (33,023 of 33,024 DS4 tensors pick it), landing MTP acceptance **at/above** the FP4 experts
  (2.73 ≥ 2.68 in the QUANT_PROBE study). The finding reproduces on **GLM‑5.2** (180‑tensor
  sweep: asym bias −0.042, 99% negative; symmetric 392× smaller at equal rel‑RMS).
- **FP4 recovery — used surgically.** Decode is HBM‑bound and an FP4 read is 2× the bytes, so
  2‑bit is the *fast* default: a **delta cache** keeps the hot experts at FP4 (background
  promote/evict, CUDA‑graph‑safe, `VLLM_MOE_W2_DELTA_GB=auto` sizes it from post‑KV VRAM),
  and a **confidence gate** (`VLLM_MOE_W2_GATE=1`) re‑runs low‑confidence steps at FP4 —
  force‑promote the step's routed experts, replay the graph once, re‑decide. Works inline on
  TP/single‑GPU (incl. MTP verify steps) and as a full‑pipeline replay under PP; τ tunable at
  runtime.
- **The kernels.** `moe_w2_mm` (2‑bit MoE GEMM: PRMT‑LUT in‑register decode → `QMMA.SF`
  block‑scaled tensor cores, 4 CTA/SM) and `moe_w4_mm` (FP4 delta GEMM) — hand‑written SASS,
  shipped as sources + prebuilt cubins for every sharding (K = 6144/4096/2048/1024/512), so
  TP2/TP4 work out of the box. Op‑validated (rel ~1–3e‑3, deterministic), graph‑capture‑exact.
  Prefill runs the **AFRAG** variant (fragment‑major activations → one `LDG.128` per QMMA
  A‑fragment; the prefill GEMM is load‑issue‑bound, not DRAM‑bound): bit‑identical outputs,
  1.3× on the GEMM, **+12% e2e prefill** on one card — default on (`VLLM_MOE_W2_AFRAG=0`
  opts out).

All three checkpoint flavors load: **FP4 experts** (DeepSeek‑V4‑Flash — codes remap),
**FP8 block‑quant** (Flash‑Base, GLM‑5.2‑FP8) and **modelopt NVFP4** (GLM‑5.2‑NVFP4) — the
latter two re‑quantized to the sign‑symmetric codebook at load, float64‑exact vs the
reference pipeline.

---

## When the model doesn't fit at all — the GPU as an expert cache

`VLLM_MOE_W2_BASE_CACHE_GB=N` inverts residency: the **whole 2‑bit base lives in pinned host
RAM**, and the GPU holds only the dense stack, KV, and an N‑GiB **cache of hot experts** (the
delta‑tier slot machinery, read inside CUDA graphs; background prefetch converges it to the
routed working set). MoE routing is concentrated enough to make this practical: **~19%
coverage serves ~96% of token→expert routings** on DS4, **~51% serves ~91%** on GLM —
measured live, not simulated.

Misses stay correct through the gate's replay trick: the desc kernel zeroes a missing
expert's contribution and bumps an in‑graph miss counter; the runner fetches **all** missing
routed experts in one batched pinned‑H2D transfer (51.6 GiB/s here; a 64‑expert fetch ≈ 3 ms)
and replays the step's graph once — **bit‑identical** to a fully resident forward
(unit‑tested). A **miss‑tolerance knob** (`VLLM_MOE_W2_BASE_MISS_TOL=k`, runtime‑tunable)
skips the replay when ≤ k of the step's ~600 routings miss — +12% decode on GLM TP2 at
tol 8 (28.3 → 31.7 tok/s) with clean quality probes (arithmetic, PL coherence, needle
retrieval; quantitative eval pending).

**Pool size is the dominant knob — treat it as a config KPI.** The mandatory replay is paid
*per step*, so decode tracks the fraction of **zero‑miss steps**, which falls off a cliff
with coverage while the token hit‑rate barely moves. On DS4 (1× 5090, NVMe‑store stack,
same box, same bench): 11 GiB pool / util 0.90 = **27–28 tok/s**, 14 GiB / util 0.95 =
**~31 tok/s**. The engine reports the KPI directly: pool coverage at startup and a periodic
**`[base] KPI: replay X% of last N steps…`** line (cadence `VLLM_MOE_W2_KPI_EVERY`, default
500 steps). If replay % runs high, grow `VLLM_MOE_W2_BASE_CACHE_GB` (and free VRAM for it,
e.g. `--gpu-memory-utilization 0.95`) before touching any other knob.

**Misses are restored adaptively.** A replayed step can re‑route onto experts the first
pass never fetched (second‑order misses); the runner re‑checks after each replay and keeps
replaying **only while the step is within `VLLM_MOE_W2_FP_THRESH` of miss‑free** (default 0
= the mandatory first‑order restore only, the throughput‑optimal setting; raising it buys
bit‑deterministic fixed points on converged working sets at a decode cost — runtime‑tunable
via `VLLM_MOE_W2_FP_THRESH_FILE`). The accepted residue is second‑order only and
KPI‑visible (`fp-residue`).

Results: **DeepSeek‑V4‑Flash 159B on one RTX 5090** (72.7 GiB of 2‑bit planes vs 32 GB of
VRAM): ~31 tok/s steady with MTP, 32K window served, coherent — and ~30 GiB of host RAM
with the NVMe stores (below, RSS‑measured) instead of ~80 GiB pinned.
**GLM‑5.2 753B on two RTX PRO 6000**: 28–32 tok/s with the full three‑tier stack (NVMe 2‑bit
base + pinned arena → GPU 2‑bit cache → GPU FP4) at a 128K single‑user window — see the GLM
table above. Neither model can otherwise run on that hardware at any precision.

---

## One tier further — expert stores on NVMe

`VLLM_MOE_W2_STORE_DIR=/path/on/real/fs` moves the host expert stores — the 73–190 GiB of
2‑bit base planes, and the FP4 need‑pool sections when that tier is enabled — out of RAM
into per‑rank **pack files**: raw rows at `(layer·E + expert) · stride`, 4 KiB‑aligned, JSON
sidecar with shapes and the layers written. Three things fall out:

- **The RAM wall falls.** Single‑5090 DS4 no longer needs ~80 GiB of free host RAM
  (measured host RAM of the serving process: 42–44 → 26–33 GiB), and GLM TP2's expert
  stores go **~568 → ~136 GiB** host RAM — the config fits hosts that could never hold the
  pinned stores.
- **The pack is a persistent quantization cache.** The first boot writes it while
  quantizing; every later boot **skips the dequant→re‑quant entirely** and serves experts
  straight from the pack — GLM TP2 boots in **~7 min instead of ~11** and skips the
  ~405 GiB staging transient. Stale packs (shape/config mismatch) rebuild automatically;
  layers absent from a pack (e.g. the MTP drafter) quantize as before.
- **Decode stays at pinned parity — give it an arena.** `VLLM_MOE_W2_BASE_RAM_GB=<GiB|auto>`
  pins an MRU **arena** over the base pack: arena hits are zero‑copy pinned views (H2D DMAs
  straight from the arena — no syscall, no memcpy), misses read through the page cache into
  the arena slot. The arena behaves as a victim cache of the GPU pool (85% of fetches served
  from RAM at 27% arena coverage on DS4).

The three selectable host‑store backends, benched head‑to‑head (DS4 1× 5090, 11 GiB GPU
pool, MTP k=2, same box and bench):

| host store | decode | host RAM (process RSS) |
|---|---:|---:|
| pinned — all 73 GiB in RAM (default) | 33.0 tok/s | 42–44 GiB |
| pack only — page cache as the RAM tier | 25.5 tok/s | 15 GiB |
| pack + **20 GiB pinned arena** | **32.8 tok/s (parity)** | 26–33 GiB |

GLM‑5.2 TP2 three‑tier with both stores on NVMe (57 GiB/rank arena): 28–32 tok/s steady,
needle retrieval 4/4 to **121K prompt tokens** at a served 128K window. Enabling it is two
env lines on any base‑cache config (DS4 shown; same two lines serve GLM TP2):

```bash
  -e VLLM_MOE_W2_STORE_DIR=/serve/packs \  # pack dir on a bind-mounted ext4/xfs
  -e VLLM_MOE_W2_BASE_RAM_GB=20 \          # pinned MRU arena; "auto" = 25% of the pack
```

Operational notes:

- **Disk budget:** DS4 pack 75 GB; GLM TP2 2×100 GB (base) + 2×189 GB (fp4) — plan ~1 TB of
  NVMe for the full GLM stack including the checkpoint. Packs are read‑only after the first
  boot (SSD wear is a non‑issue). The dir must be a real filesystem via bind mount — **not
  overlayfs** (the container filesystem).
- **You don't need a fast drive for steady decode.** Misses are buffered reads, so the page
  cache acts as an opportunistic L3 under the arena — the parity numbers above come from a
  drive on a PCIe **Gen3 x4** link (3.7 GB/s). Cold working‑set shifts and first‑touch
  prefills do pay drive speed. `VLLM_MOE_W2_TIER_DIRECT=1` switches misses to O_DIRECT
  (hard RAM budget, page cache stays flat; raw drive latency on every miss).
- **Prefill can't wipe the arena** (scan discipline: prefill working sets fill free slots
  but never evict the decode hot set), and the arena's hot set persists to
  `<pack>.heat.json` and **preheats on boot** (57 GiB ≈ 35 s; `VLLM_MOE_W2_TIER_PREHEAT=0`
  opts out). Reader pool: `VLLM_MOE_W2_STORE_THREADS` (default 8).
- **Observability:** with `VLLM_MOE_W2_DELTA_TRACE=1` the summary carries a
  `[base] tiered store: arena N/M | fetch rows X ram + Y nvme (Z% ram) | … p50/p99` line —
  the ram‑hit % *is* the arena‑coverage curve; grow `BASE_RAM_GB` if it sags.
- Replays stay **bit‑identical** across backends (bytes are bytes; only the copy source
  changes) — unit‑tested per backend × cold/warm/reboot/evict/overflow/scan/preheat in
  `tools/test_store_backends.py`.

---

## The base: vLLM v0.24.0 on SM120

Upstream v0.24.0 ships DeepSeek‑V4 + GLM‑5.x + SM120 natively — but the release cannot
actually serve them on SM120. The patch carries the fixes (details in
[docs/v024-port.md](docs/v024-port.md)):

- **DeepGEMM**: release pin has no family‑120 host paths ("Unknown SF transformation",
  einsum/indexer asserts) → pin **nv‑dev `a6b593d2`** (as vLLM main did).
- **flashinfer**: official 0.6.12 pin predates the SM120 DS4 attention API → **0.6.14**.
- `cooperative_topk` uses thread‑block **cluster launch** (SM90/100‑only) → gated off on SM12x.
- o_proj fp8 einsum: SM100 packed scale layout NaNs on SM120 → SM90‑style raw f32 scales.
- CUDA‑graph capture: `thread_local` error mode on **all four** capture paths (the expert
  caches' background threads must not invalidate capture).

With the 2‑bit knobs off, the patch is exactly these base fixes — stock behaviour otherwise.

## Quickstart

```bash
git clone https://github.com/kacper-daftcode/vLLM-Moet && cd vLLM-Moet

# official vllm-openai:v0.24.0 image + patch + pins + SM120 cubins
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-v024 -t vllm-moet-sm120:v024 .
```

**Easiest path — run a benchmarked recipe.** The recipes image downloads the
checkpoint from HuggingFace on first run and starts the exact configuration
the benchmark table below was measured with (one recipe per supported
model×hardware combo, see `bench/recipes/`):

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.recipes -t vllm-moet-recipes:v024 .

docker run --rm vllm-moet-recipes:v024 --list          # supported configs
docker run --rm --gpus all --network host --ipc host --shm-size 64g \
  -v /srv/models:/models -e HF_TOKEN=... \
  vllm-moet-recipes:v024  glm-5.2-nvfp4/pro6000x4-tp4-mtp
```

(`-e KNOB=...` overrides any recipe knob, `<recipe> --print` shows what would
run without serving, args after `--` go to vllm serve; details in
[bench/README.md](bench/README.md).)

Or hand‑roll the serve. **GLM‑5.2 on 4× PRO 6000** (the standing agent‑serving config:
128K window, MTP, FP4 pool + gate, tool/reasoning parsers):

```bash
docker run --rm --gpus '"device=0,1,2,3"' --network host --ipc host --shm-size 64g \
  -v /path/to/GLM-5.2-NVFP4:/model:ro \
  -e VLLM_MOE_W2=1 -e VLLM_MOE_W2_DELTA_GB=auto -e VLLM_MOE_W2_GATE=1 \
  vllm-moet-sm120:v024 \
  --model /model --served-model-name glm-5.2 --trust-remote-code \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --kv-cache-dtype fp8 --max-model-len 131072 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port 8000
```

(`--kv-cache-dtype nvfp4` enables the 352 B/token KV cache; it requires the FlashInfer JIT
patch from `tools/nvfp4_flashinfer_sm120/` baked into the image and is currently validated to
128K windows.)

**GLM‑5.2 on 2× PRO 6000** (three‑tier + NVMe stores, the 128K/needle‑121K config from the
table; ~140 GiB host RAM + ~580 GB NVMe for the packs, first boot writes them):

```bash
docker run --rm --gpus '"device=0,1"' --network host --ipc host --shm-size 64g \
  -v /path/to/GLM-5.2-NVFP4:/model:ro -v /nvme/packs-glm:/packs \
  -e VLLM_MOE_W2=1 -e VLLM_MOE_W2_BASE_CACHE_GB=46 -e VLLM_MOE_W2_DELTA_GB=2 \
  -e VLLM_MOE_W2_GATE=1 -e VLLM_MOE_W2_BASE_MISS_TOL=8 \
  -e VLLM_MOE_W2_STORE_DIR=/packs -e VLLM_MOE_W2_BASE_RAM_GB=57 \
  vllm-moet-sm120:v024 \
  --model /model --served-model-name glm-5.2 --trust-remote-code \
  --tensor-parallel-size 2 --disable-custom-all-reduce \
  --kv-cache-dtype fp8 --max-model-len 131072 --kv-cache-memory-bytes 8589934592 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 --max-num-seqs 2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port 8000
```

(31.7 tok/s at `MISS_TOL=8` as shown, 28.3 at strict `0` — both probe‑clean on this stack;
drop the two `STORE_DIR`/`BASE_RAM_GB` lines for the all‑RAM variant, which then needs
~200 GiB free host RAM and re‑quantizes on every boot.)

**DeepSeek‑V4‑Flash on one PRO 6000** (161 tok/s, 512K window):

```bash
docker run --rm --gpus '"device=0"' --network host --ipc host --shm-size 64g \
  -v /path/to/DeepSeek-V4-Flash:/model:ro \
  -e VLLM_MOE_W2=1 -e VLLM_MOE_W2_DELTA_GB=1 \
  vllm-moet-sm120:v024 \
  --model /model --served-model-name deepseek-v4-flash --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 --max-model-len 24576 \
  --gpu-memory-utilization 0.95 --max-num-batched-tokens 1024 --max-num-seqs 4 \
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl \
  --speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port 8000
```

`VLLM_MOE_W2=0` = stock FP4 path (needs ≥2 cards for DS4; GLM's stock NVFP4 does not fit this
box at all). TP: `--tensor-parallel-size 2|4` + `--disable-custom-all-reduce`. Single‑5090
DS4 (host‑resident base): `-e VLLM_MOE_W2_BASE_CACHE_GB=14 -e VLLM_MOE_W2_DELTA_GB=0` with
`--max-model-len 8192 --gpu-memory-utilization 0.95 --max-num-seqs 2` (~80 GiB free host
RAM — **or add the two NVMe‑store lines from the section above and run in ~30 GiB**; MTP
works). Do not shrink the pool below 14 GiB to "play it safe" — pool size is the dominant
perf knob (see the KPI note above; 11 GiB costs ~12% decode and doubles the missing pairs
per step) and 14 GiB needs util 0.95 to leave room for KV.

## Quality

Method: baseline is the untouched official checkpoint; our variant changes only the expert
codes (same stack, byte‑identical dense/scales/headers), so any delta is the quantization
alone — see [docs/quality.md](docs/quality.md). The QUANT_PROBE study (identical quant scheme
and cubins): MTP acceptance 2.73 vs 2.68 FP4 reference, draft accept 86.3% vs 84.1%, 12/12
coherent greedy outputs; bare 2‑bit agrees with FP4 on 89% of next‑token picks — the delta
cache + gate close that gap. Live serving reproduces the acceptance (~2.6 tok/step on DS4,
2.3–3.0 on GLM).

## The SM120 toolchain we built

These kernels exist only because we first built the assembler and the ISA data they need.
Consumer Blackwell (sm_120) has **no public SASS toolchain**. Current CUDA does expose the
block‑scaled MMA *instruction* itself (PTX `kind::mxf8f6f4` compiles to `QMMA.SF` — DeepGEMM's
SM120 port uses it), but everything these kernels are actually made of — hand scheduling
against measured latencies and control words, the PRMT‑LUT decode interleaved into the QMMA
stream, register‑bank and occupancy shaping (regcount 64 → 4 CTA/SM) — is decided by ptxas
and unreachable from CUDA/PTX. So the stack underneath this repo is end‑to‑end ours:

- **[`blackwell-isa`](https://github.com/kacper-daftcode/blackwell-isa)** — a machine‑readable
  **SM120 SASS ISA database**: 1,994 instruction forms, 128‑bit encoding templates + operand/
  bitfield maps, and per‑opcode scheduling metadata (pipeline/latency/throughput, control‑word
  classes). Reverse‑engineered and hardware‑validated on RTX 5090 (47,244 instructions decoded
  across 178 cubins at 100% coverage; 5,014/5,014 roundtrip‑fuzz). It documents what the CUDA
  toolchain hides — e.g. `QMMA.SF` block‑scaled FP4 MMA and an undocumented `E3M4` type code.
  Ships a [searchable HTML reference](https://kacper-daftcode.github.io/blackwell-isa/SM120_ISA_REFERENCE.html).
- **[`cubit`](https://github.com/kacper-daftcode/cubit)** — an **SM120 SASS assembler/disassembler**
  built on that database. It turns the hand‑written `.sass` sources in `kernels/sass/` into the
  cubins this server loads, and is the only tool needed to rebuild or audit them.

**ISA ([`blackwell-isa`](https://github.com/kacper-daftcode/blackwell-isa)) → assembler
([`cubit`](https://github.com/kacper-daftcode/cubit)) → SASS kernels → this vLLM.** None of the
kernels here can be *written* through stock CUDA on sm_120 — the instructions compile, the
kernels don't; this toolchain is what makes them possible.

## Kimi-K2.7-Code (1T MoE) on 4× RTX PRO 6000

**[nvidia/Kimi-K2.7-Code-NVFP4](https://huggingface.co/nvidia/Kimi-K2.7-Code-NVFP4)
(1T params, 384 experts top‑8, H=7168, dense MLA, 256K) serves on 4× RTX PRO 6000 (TP4)**
— a checkpoint whose 595 GB of weights cannot even load on this box (384 GB total VRAM):
the 2‑bit planes total **265.8 GiB** (~66 GiB/rank + ~8 GiB BF16 dense/vision), leaving
room for a 337K‑token fp8 KV pool and the FP4 delta tier in 96 GB/card. The loader path
is the same modelopt‑NVFP4 requant as GLM‑5.2‑NVFP4 (f64‑exact dequant → sign‑symmetric
2‑bit), through a new **K=7168** cubin family (`kernels/`, generated + op‑validated like
the rest).

Measured (TP4, 131072‑token window, greedy, CUDA graphs, no MTP — the checkpoint ships
no drafter head; 2026‑07‑10): **51 tok/s** single‑stream decode (**222 tok/s** aggregate
at 8 streams), **2 448 tok/s** 8K‑unique prefill, needle retrieval **PASS at
8K/32K/80K/128K**, arithmetic 5/5, generated code executes, `kimi_k2` tool‑calling
round‑trip works. Serve recipe and
the bring‑up findings (a checkpoint‑specific **zero‑sign balancing** fix the sweep gate
caught — the INT4→NVFP4 export writes all exact zeros as +0, which would inject 3× the
bias that degenerates GLM; an SM12x smem fix for dense‑MLA triton decode; a
single‑chunk‑context workaround for >64K prompts):
**[docs/kimi-k27-code-plan.md](docs/kimi-k27-code-plan.md)**.

## Benchmark results

<!-- bench:table:begin (generated by bench/runner/render.py - do not edit) -->

Release **`baseline-2026-07-10`** — one row per supported recipe (`bench/recipes/`), measured by `bench/runner/bench.py`; full report: [`docs/benchmarks/baseline-2026-07-10.md`](docs/benchmarks/baseline-2026-07-10.md). Single-stream decode and prefill are medians; batch is aggregate tok/s at the noted concurrency.

| model | hardware | config | ctx | decode tok/s | batch | prefill 8K | needle | notes |
|---|---|---|---:|---:|---:|---:|---|---|
| deepseek-v4-flash | 1x RTX 5090 (32 GB) | host-resident 2-bit base, GPU as expert cache | 8K | **38** | — | — | — | acc 2.83 † |
| deepseek-v4-flash | 4x RTX 5090 TP4 | consumer-card throughput | 16K | **214.4** | 1 560 @32 | 6 101 | — | acc 2.6 † |
| deepseek-v4-flash | 1x RTX PRO 6000 | throughput (24K ctx, FP4 delta auto, MTP k=2) | 24K | **161.2** | 933 @32 | 5 340 | — | acc 2.6 † |
| deepseek-v4-flash | 1x RTX PRO 6000 | 512K window (delta pool traded for KV) | 512K | — | — | — | PASS ≤453K tok | † |
| deepseek-v4-flash | 2x RTX PRO 6000 TP2 | throughput | 24K | **209.6** | 380 @3 | 5 791 | — | acc 2.6 † |
| glm-5.2-nvfp4 | 2x RTX PRO 6000 TP2 | host-resident base, 44 GiB/rank expert cache | 32K | **33** | — | — | PASS ≤27K tok | acc 3 † |
| glm-5.2-nvfp4 | 4x RTX PRO 6000 TP4 | 2-bit base + MTP k=2, 128K window | 128K | **105** | — | 2 500 | PASS ≤276K tok | † |
| glm-5.2-nvfp4 | 4x RTX PRO 6000 TP4 | + FP4 delta (auto) + confidence gate tau=0.60 | 128K | **84** | — | — | — | † |
| kimi-k2.7-code-nvfp4 | 2x RTX PRO 6000 TP2 | host-resident base, 52 GiB/rank cache (~39% coverage) | 16K | **14.4** | — | — | PASS ≤8K tok | † |
| kimi-k2.7-code-nvfp4 | 4x RTX PRO 6000 TP4 | GPU-resident 2-bit + FP4 delta, 256K window | 256K | **51** | 222 @8 | 2 448 | PASS ≤248K tok | † |
| kimi-k2.7-code-nvfp4 | 4x RTX PRO 6000 TP4 | + Eagle3 drafter (k=3, drafter TP4) | 256K | **57** (±44%) | — | — | — | acc 3.5 † |

† imported from pre-harness measurements (README/docs history) — re-measured on the next release.

<!-- bench:table:end -->

## Repository layout
- **`patch/vllm-moet-v0.24.0.patch`** — the delta vs official vLLM `v0.24.0` (37 files,
  +7.4k lines; applies clean on the tag). Goes with the pins above.
- **`Dockerfile.sm120-v024`** — the image: official `vllm/vllm-openai:v0.24.0` + patch + pins +
  cubins.
- **`kernels/`** — SASS (`sass/`) + prebuilt SM120 cubins (`cubins-sm120/`, incl. the K=6144
  GLM‑5.x family) + generators (`gen/`) + `MANIFEST.md`.
- **`docs/v024-port.md`** — the port: pins, SM120 fixes, apply recipe, benchmark methodology.
- **`docs/quality.md`** — quality methodology.
- **`bench/`** — the release benchmark system: recipes (the tested serve configs, one YAML per
  supported model×hardware combo), the runner that stands up each config and measures it, and
  committed results per release. The table above is rendered from `bench/results/` by
  `bench/runner/render.py` (CI keeps them in sync); process: **[bench/README.md](bench/README.md)**,
  per‑release detail: `docs/benchmarks/`.
- **`Dockerfile.recipes`** — the user‑facing image: picks a recipe by id, downloads the
  checkpoint into the `/models` volume, and serves the exact benchmarked configuration
  (`docker/serve_recipe.py` is the entrypoint; the bench's docker runtime measures this same
  image).
