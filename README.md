# DeepSeek‑V4‑Flash on Blackwell (SM120)

**Official vLLM v0.24.0 + a 3.7k‑line patch** that (a) makes DeepSeek‑V4‑Flash (159B MoE)
actually run on consumer/workstation Blackwell — the release is broken‑as‑shipped for DS4 on
SM120 — and (b) fits it on hardware the FP4 checkpoint can't reach, by compressing the experts
to **2 bits** with **FP4 recovery**, on hand‑written SM120 SASS kernels.

## What you get

Official DeepSeek‑V4‑Flash checkpoint, 2‑bit experts + FP4 delta cache, MTP k=2, CUDA graphs
(single‑stream medians; prefill = 8k‑token prompt, uncached; 2026‑07‑08):

| hardware | decode | prefill 8k | context |
|---|---:|---:|---:|
| **1× RTX PRO 6000 (96 GB)** | **161 tok/s** | **4 850 tok/s** | **512K** |
| **2× RTX PRO 6000 (TP2)** | **210 tok/s** | **5 790 tok/s** | 512K |
| **4× RTX 5090 (TP4)** | **214 tok/s** | **5 560 tok/s** | 16K+ |

Four consumer 5090s match two PRO 6000s on decode. MTP acceptance ~2.6 tok/step across all
configs. **512K on one card is live-validated**: needle retrieval PASS at 102K / 256K / 453K
prompt tokens (depths 0.1 and 0.5; cold TTFT at 453K ≈ 2 min). Methodology:
**[docs/v024-port.md](docs/v024-port.md)**.

A 96 GB card cannot even load the official checkpoint (FP4 experts + FP8 dense ≈ 149 GiB);
here it serves it at 161 tok/s with the full 512K window.

---

## How it fits — 2‑bit experts at FP4 quality

We compress **only the routed experts** to 2 bits (dense stays FP8) and recover FP4 precision
adaptively:

- **2‑bit expert planes — the sign‑bias finding.** Naive 2‑bit *destroys* this model
  (degenerate loops). The cause is **sign asymmetry**, not error magnitude — the optimal‑L2
  codebook drops one sign's tail and the per‑expert bias compounds over 43 layers. Forcing a
  **sign‑symmetric** `{−4,−1,1,4}` codebook at the same L2 error fixes it entirely (33,023 of
  33,024 tensors pick it), landing MTP acceptance **at/above** the FP4 experts (2.73 ≥ 2.68 in
  the original QUANT_PROBE study; ~2.6 reproduced on the v0.24 base). The finding also
  reproduces on **GLM‑5.2** (180‑tensor sweep: asym bias −0.042, 99% negative; symmetric 392×
  smaller at equal rel‑RMS) — see below.
- **FP4 recovery — used surgically.** Decode is HBM‑bound and an FP4 read is 2× the bytes, so
  2‑bit is the *fast* default: a **delta cache** keeps the hot experts at FP4 (background
  promote/evict thread, CUDA‑graph‑safe), and a **confidence gate** (`VLLM_MOE_W2_GATE=1`)
  re‑runs low‑confidence tokens at FP4 — force‑promote the step's routed experts, replay the
  graph once, re‑decide. Works inline on TP/single‑GPU (incl. MTP verify steps) and as a
  full‑pipeline replay under PP; τ tunable at runtime. Arming it costs ~10% single‑stream;
  the τ=0.60 replays were throughput‑neutral on top (FP4 re‑decides lift MTP acceptance).
- **The kernels.** `moe_w2_mm` (2‑bit MoE GEMM: PRMT‑LUT in‑register decode → `QMMA.SF`
  block‑scaled tensor cores, 4 CTA/SM) and `moe_w4_mm` (FP4 delta GEMM) — hand‑written SASS,
  shipped as sources + prebuilt cubins for every sharding (K = 6144/4096/2048/1024/512), so
  TP2/TP4 work out of the box. Op‑validated (rel ~1–3e‑3, deterministic), graph‑capture‑exact.

Both checkpoint flavors are supported: **FP4 experts** (DeepSeek‑V4‑Flash — codes remap at
load) and **FP8 block‑quant experts** (Flash‑Base, **GLM‑5.2‑FP8** — re‑quantized to the
sign‑symmetric codebook at load, float64‑exact vs the reference pipeline).

---

## The base: vLLM v0.24.0 on SM120

Upstream v0.24.0 ships DeepSeek‑V4 + SM120 natively — but the release cannot actually serve
DS4 on SM120. The patch carries the fixes (details in
[docs/v024-port.md](docs/v024-port.md)):

- **DeepGEMM**: release pin has no family‑120 host paths ("Unknown SF transformation",
  einsum/indexer asserts) → pin **nv‑dev `a6b593d2`** (as vLLM main did).
- **flashinfer**: official 0.6.12 pin predates the SM120 DS4 attention API → **0.6.14**.
- `cooperative_topk` uses thread‑block **cluster launch** (SM90/100‑only) → gated off on SM12x.
- o_proj fp8 einsum: SM100 packed scale layout NaNs on SM120 → SM90‑style raw f32 scales.
- CUDA‑graph capture: `thread_local` error mode on **all four** capture paths (the delta
  cache's background thread must not invalidate capture).

With the 2‑bit knobs off, the patch is exactly these base fixes — stock behaviour otherwise.

## Quickstart

```bash
git clone https://github.com/kacper-daftcode/vLLM-Moet && cd vLLM-Moet

# official vllm-openai:v0.24.0 image + patch + pins + SM120 cubins
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-v024 -t vllm-moet-sm120:v024 .
```

Serve (single 96 GB card, production knobs — same as the benchmark config):

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

`VLLM_MOE_W2=0` = stock FP4 path (needs ≥2 cards). TP: add `--tensor-parallel-size 2|4` and
`--disable-custom-all-reduce`. Confidence gate: `-e VLLM_MOE_W2_GATE=1`
(+ optional `VLLM_MOE_W2_GATE_TAU`).

## Quality

Method: baseline is the untouched official checkpoint; our variant changes only the expert
codes (same stack, byte‑identical dense/scales/headers), so any delta is the quantization
alone — see [docs/quality.md](docs/quality.md). The QUANT_PROBE study (identical quant scheme
and cubins): MTP acceptance 2.73 vs 2.68 FP4 reference, draft accept 86.3% vs 84.1%, 12/12
coherent greedy outputs; bare 2‑bit agrees with FP4 on 89% of next‑token picks — the delta
cache + gate close that gap. Live serving reproduces the acceptance (~2.6 tok/step).

## Next: GLM‑5.2

The port is GLM‑ready: the FP8→2‑bit load path is golden‑tested against the GLM‑5.2 sweep
reference, the K=6144 GEMM family (GLM's hidden size) ships in `kernels/`, and the layer
cutoff follows the model config (78 layers). The sign‑symmetric codebook finding reproduces on
GLM‑5.2's weights (`internal` sweep, 180 tensors / 16 layers). Bring‑up starts when the
checkpoint lands (~350 GB).

## The SM120 toolchain we built

These kernels exist only because we first built the assembler and the ISA data they need.
Consumer Blackwell (sm_120) has **no public SASS toolchain**, and CUDA's `sm_120` path doesn't
expose the block‑scaled MMA forms (`QMMA.SF`, the FP4/FP6 type codes) these kernels are built on.
So the stack underneath this repo is end‑to‑end ours:

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
kernels here are reachable through stock CUDA on sm_120; this toolchain is what makes them possible.

## Repository layout
- **`patch/vllm-moet-v0.24.0.patch`** — the delta vs official vLLM `v0.24.0` (18 files,
  +3.4k lines; applies clean on the tag). Goes with the pins above.
- **`Dockerfile.sm120-v024`** — the image: official `vllm/vllm-openai:v0.24.0` + patch + pins +
  cubins.
- **`kernels/`** — SASS (`sass/`) + prebuilt SM120 cubins (`cubins-sm120/`, incl. the K=6144
  GLM‑5.x family) + generators (`gen/`) + `MANIFEST.md`.
- **`docs/v024-port.md`** — the port: pins, SM120 fixes, apply recipe, benchmark methodology.
- **`docs/quality.md`** — quality methodology.
