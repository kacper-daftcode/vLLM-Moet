# The MoET hybrid: DeepSeek‑V4.1‑Flash on 4× RTX 5090 with CPU‑computed experts

**What it is.** The official `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint (552B MoE, MXFP4 experts, UE8M0 scales)
served at native quality on four 32 GB GeForce RTX 5090 cards — a box the model does not fit: the four cards hold
the dense weights, the KV cache and **120 of the 384 experts per MoE layer per TP rank**; every other routed expert
of a decode step is **computed on the CPU** from the checkpoint's own bytes in pinned host RAM, in parallel with the
GPU, and its rows are injected into the GPU's own MoE kernel before the top‑k sum. No re‑quantization, no replay:
the hybrid's output is **bit‑identical to the all‑resident b12x path** on every step. The box: a Threadripper PRO
9975WX (32 cores, 4 CCDs), 498 GB of DDR5, 4× RTX 5090 on PCIe 5.0 x16, NVMe RAID for the stores.

Measured (2026‑10‑26, the launcher's defaults, TP4, no DSpark):

| | |
|---|---|
| decode, single stream, short prompts | **143 / 120 tok/s** ('Hello' / 'def fib(n):'; the degenerate 'Hello' run has a hit rate → 1 and is the stack's GPU floor at 7.0 ms per step — the hybrid's miss handling costs ~1.3 ms per step on a real prompt) |
| decode behind 2K / 14K / 30K of context | 50–64 / 43–46 / 47–48 tok/s (the attention side) |
| prefill with the streamed experts (chunk 4096, the default) | **2.1–2.4K tok/s at 2K, 3.9K tok/s at 14K** (TTFT 0.9 s / 3.7 s; PCIe‑bound: 0.12 ms per missing expert per rank ≈ 39 GB/s per link; the missing set of a chunk grows sublinearly, so a bigger chunk streams fewer bytes per token — chunk 2048 gave 2.0–2.3K at 2K / 14K / 30K) |
| quality, paired against the native FP4 reference | **HumanEval 159/164 vs 160/164** (McNemar p = 1.0, 134 identical programs); **MATH‑500 level 5: 55/60 vs 55/60** (0 discordant items, 59 identical answers) |
| identity | bit‑exact against the all‑resident b12x path within one boot: 21,040 compact calls per rank, 98,220 CPU‑computed rows, 0 differ (`DEBUG_COMPACT`); the CPU kernel bit‑exact against b12x's GPU kernels on four parity sets |
| expert hit rate (pairs resident) | 0.83–0.85 on short prompts, **0.87 on HumanEval, 0.91 on MATH‑500** |
| host RAM while up | ~450 GB used (the expert store 64.7 GB per rank + the Engram tables 2 × 23.6 GiB per rank, pinned), MemAvailable ~45 GB |
| VRAM | 29.9 GB per card at util 0.92; KV cache 0.8–1.1 GiB (nvfp4 MLA: 300K+ tokens) |
| boot | ~6 min (weights 160–250 s + 65 GB of records pinned per rank in ~90 s + warmup) |

## How it works

Per TP rank (the four ranks are symmetric and need no CPU‑side coordination):

1. **Resident pool.** Each MoE layer's expert parameters are created with R = 120 rows instead of 384; vLLM's
   weight loader maps a resident expert to its slot and skips the rest (its own path for non‑local EP experts). The
   pool is b12x's prepared storage; a per‑layer device table `slot[e]` (−1 = not resident) is read in‑graph. 40 of the
   120 slots hold a **static placement** from routing frequencies (`routing/placement.py`); the other 80 are **LRU
   slots** filled by promotions (below). At equal VRAM, dynamic slots beat frequency‑placed static ones
   (`routing/promo_sim.py` on 237K traced positions: 107 + 13 → 0.777, 40 + 80 → 0.859, all 120 dynamic → 0.88).
2. **The split, inside the captured decode graph.** After b12x's activation quant, a publish kernel remaps the
   router's ids through `slot[]`; resident pairs run on the GPU (b12x's compact W4A8 micro path), missing pairs get
   id −1 (b12x writes zero rows for them) and are written — `(token, expert, row, weight)` plus the step's quantized
   tokens — into the layer's mapped‑memory mailbox with a system‑scope release. Before b12x's TopKSum a wait kernel
   spins (bounded) on the layer's ack and copies the host's bf16 rows into `route_output` by zero‑copy, so the final
   sum has the GPU's own order. Timeout → an error word → the step fails closed (never a silent zero row).
3. **The CPU tier.** A dispatcher thread per rank walks the mailboxes, groups a request's pairs by expert and runs a
   6‑thread pool pinned to the rank's CCD with **kernel v2** (`tools/dsv41_5090/cpu_moe/cpu_moe_b12x.cpp`): exact
   integer block dots on the MXFP4 nibbles × MXFP8 activations (AVX‑512 VNNI), with the sm_120 block‑scaled MMA's
   rounding and truncation rules reproduced (`qmma_probe.cu` measured them) — bit‑identical to b12x's compact W4A8
   path. ~111 µs per expert per token from DRAM (4.7 MB per expert slice at 45 GB/s per CCD), 145–165 µs in situ.
   The pool spins through a decode step's gaps and sleeps on a futex when the API is idle.
4. **The expert store.** Every rank's slices of the experts that can be missing live in pinned host RAM in the
   checkpoint's own layout (`cpu_moe/build_store.py` writes them once to NVMe, 269 GB; the loader pins 344 × 40
   records per rank). A promotion is a 4.7 MB H2D of the record plus a small repack kernel into b12x's tiled layout.
5. **Promote‑on‑miss.** After each step the host schedules promotions on a copy stream: a missed expert is admitted
   only on its **second miss within 12 steps** (`PROMOTE_WINDOW`; promote‑on‑every‑miss moved 27–53 records per
   step and its DMA slowed the CPU kernel ~15 %), the victim is the least recently used LRU slot not routed in this
   step, the slot table is written on the copy stream and the next steps wait on events — no recapture.
6. **Prefill.** Prompts longer than 8 tokens take b12x's non‑compact kernels on vLLM's eager path; the chunk's
   missing experts are streamed from the pinned store into a scratch pool (`SCRATCH` slots, rounds of ≤ 128) and a
   second b12x run on the ids remapped to scratch slots is added — the output is within the non‑compact kernel's own
   run‑to‑run noise (1–2 bf16 ulps) of the all‑in‑one reference.
7. **Engram** (the model's n‑gram tables, 2 × 23.6 GiB per rank) stay vLLM's own pinned tables, allocated exact‑size
   (`--engram-config '{"use_thp": true}'`; torch's pinned allocator would round each to 32 GiB).

Rules the implementation keeps ("graph or death"): the decode step is one captured graph including the handoff; no
steady‑state allocation (pools, mailboxes, staging, the prefill scratch are sized at startup); a miss is served inside
the step or the step fails closed; placement changes are stream‑ordered table writes; prefill stays on vLLM's eager
path; identity against the all‑resident path is the quality gate.

## Running it

```bash
# one-time: the expert store (269 GB on NVMe, ~1 min; verified against the checkpoint)
DSV41_MODEL_DIR=/mnt/nvme/models/DeepSeek-V4.1-Flash python tools/dsv41_5090/cpu_moe/build_store.py --out /mnt/nvme/packs/dsv41-hybrid --check 8
# the kernel's GPU tables (b12x's activation / scale tables and its weight layout permutation; one GPU, seconds;
# qmma_probe is built from qmma_probe.cu, the layout probe runs in the serving image)
CUDA_VISIBLE_DEVICES=1 tools/dsv41_5090/cpu_moe/qmma_probe --tables internal/dsv41-5090-cpu/parity
python tools/dsv41_5090/cpu_tier/b12x_layout_probe.py --out internal/dsv41-5090-cpu/parity
# serve (the image = this repo's sm_120 line on the nightly base; the patcher is applied at container start)
MODEL_DIR=/mnt/nvme/models/DeepSeek-V4.1-Flash docker/sm120/run-dsv41-5090.sh
```

`docker/sm120/run-dsv41-5090.sh` boots the measured serving configuration by default; its header lists every knob
(`PLACEMENT`, `SPARE`, `PROMOTE_WINDOW`, `SCRATCH`, `ENGRAM_THP`, `POOL_SPIN_HOT`, `SPEC_TOKENS`, `PREFIX_CACHE`,
`GPU_MEM_UTIL`, …). The hybrid's own switches arrive in the container as `VLLM_MOET_CPU_TIER_*` (documented at the
top of `tools/dsv41_5090/cpu_tier/vllm_moet_cpu_tier.py`); everything is inert unless `VLLM_MOET_MOE_CPU_TIER=1`.
`tools/dsv41_5090/cpu_tier/tp4_window.sh TAG [KEY=VALUE …]` runs one boot with the measurements (identity prompts,
decode, memory, optional `TTFT_LENGTHS=…` and `HARD_PROBE=humaneval,math500`) under the box's GPU locks and keeps
the artefacts under `internal/dsv41-5090-cpu/handoff/boots/`.

Host‑side requirements: ~450 GB of RAM free at boot (the window script refuses below 465 GB available and drops the
page cache before the boot and again after the weights are read — the store pinning otherwise reclaims the loader's
shard pages under pressure); the pool cores (8r+2..8r+7 per rank) and their SMT siblings idle — a foreign busy
thread on a sibling halves the kernel on that core and the job's barrier waits for it.

## Verification

- `cpu_moe/test_cpu_moe_b12x.py`: kernel v2 against an independent scalar reference (synthetic + checkpoint slices).
- `cpu_moe/gpu_parity_b12x_{prepare,run,compare}.py`: against b12x's GPU kernels on recorded inputs (four sets, layers
  0/1/20/39: 0 of 890,880 route_output elements differ); `parity_sweep_{ranks,layer}.sh` over ranks and a whole layer.
- `cpu_tier/test_cpu_tier.py`, `test_promotion.py`: the tier end to end in captured graphs (rows checked, fail‑closed,
  the U/M promotion protocol).
- `VLLM_MOET_CPU_TIER_DEBUG_COMPACT=1` with `--enforce-eager`: every compact call recomputed with all its routed
  experts in the scratch pool and compared bit for bit — the within‑boot identity check (TP4 boots of this box are
  not bit‑reproducible among themselves, so cross‑boot comparisons are void).
- `internal/dsv41-5090-cpu/probe/hard_probe.py`: HumanEval / MATH‑500 paired against the native references.

## Benches and tools

`cpu_moe/bench_real_records_b12x.py` (the kernel on real records — the synthetic ones read 1.3× too fast),
`cpu_moe/bench_pattern_b12x.py` (the decode's job pattern per CCD with the cores' clock), `routing/cache_sim.py` and
`routing/promo_sim.py` (placement and promotion policies on the routing traces), `cpu_tier/ttft_probe.py` (streamed
prefill TTFT on disjoint prompts), `cpu_tier/cpu_placement_probe.sh` (who runs on the pool cores).

## Limits and open items

- DSpark (k = 5) does not pay for the hybrid: a verify step routes ~5× more distinct experts for 3.6× more tokens and
  the CPU kernel's cost per extra token (~73 µs) keeps it CPU‑bound; k = 0 is the serving mode.
- Prefill is PCIe‑bound (0.12 ms per missing expert per rank): 2.1–2.4K tok/s on 2K prompts, 3.9K on 14K with the
  4096‑token chunk; a CPU tail for experts with few tokens and double‑buffered staging are the next levers.
- Phase 3's needle, concurrency and soak runs are pending; the GPU 0 of this box dropped off the PCIe bus twice under
  sustained full‑power four‑GPU prefill (hardware), so the cards run capped at 400 W.
- The working notes with every measurement and its command: `internal/dsv41-5090-cpu/handoff/*.md` and
  `/root/work/HANDOFF-moet-hybrid.md`.
