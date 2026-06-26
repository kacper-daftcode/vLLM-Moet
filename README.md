# DeepSeek‑V4‑Flash on Blackwell (SM120)

A hard‑tuned **vLLM** that makes **DeepSeek‑V4‑Flash** (159B MoE) **fast** on Blackwell — and
fits it on hardware the FP4 checkpoint can't. Stock vLLM has no SM120 path for DS4 (its FP8
route needs DeepGEMM, which has no SM120 kernels) — other stacks do run it on SM120 (e.g.
sglang); this is a **vLLM** path, tuned to be fast, with hand‑written SM120 kernels.

## What you get

| | hardware | tok/s | context |
|---|---|---:|---|
| **Full quality, fast** | 2× RTX PRO 6000 | **177** | **512K** |
| **On a single card** | 1× RTX PRO 6000 (96 GB) | **109** | 512K |
| **On consumer GPUs** | 4× RTX 5090 (TP4) | **89** | 512K |

Single‑stream, MTP k=2, warmed coding corpus. Quality matches the official checkpoint
(**MTP acceptance 2.73 ≥ 2.68**, 12/12 coherent greedy outputs). Full tables below.

---

## How it's fast — the SM120 tuning (any precision)

The speed comes from hand‑written SASS kernels (assembled by our own
[`cubit`](https://github.com/kacper-daftcode/cubit) SM120 assembler — see
[the toolchain](#the-sm120-toolchain-we-built)) for the paths stock libraries don't serve on
SM120. They accelerate the **unmodified FP4/FP8 checkpoint** — you don't touch the weights to go
faster:

| kernel | speedup vs SM120 fallback | effect |
|---|---:|---|
| sparse‑MLA prefill (`cubit`) | **2.21×** (over Triton) | cold‑prefill TTFT −15–17% |
| fp8 MQA‑logits indexer | **~28×** (over f32) | removes the indexer n² → **512K context** on one card |
| cache‑direct prefill | — | workspace O(query chunk), not O(context) |

Net: the full FP4 checkpoint runs at **177 tok/s on 2× PRO 6000 with the full 512K context
window** (needle retrieval validated @ 100K/250K/440K).

---

## How it fits where FP4 can't — at FP4 quality

The official checkpoint (FP4 experts + FP8 dense, ~149 GiB) needs two 96 GB cards. To run the
*same model* on **one** card — or **four RTX 5090 gaming cards** — we compress **only the
experts** to 2 bits (dense stays FP8) and **recover FP4 precision adaptively**, so the
delivered quality stays at the FP4 level:

- **2‑bit expert planes — the sign‑bias finding.** Naive 2‑bit *destroys* this model
  (degenerate loops). The cause is **sign asymmetry**, not error magnitude — the optimal‑L2
  codebook drops one sign's tail and the per‑expert bias compounds over 43 layers. Forcing a
  **sign‑symmetric** `{−4,−1,1,4}` codebook at the same L2 error fixes it entirely (33,023 of
  33,024 tensors pick it), landing MTP acceptance **at/above** the FP4 experts.
- **FP4 recovery — used surgically.** Decode is HBM‑bound and an FP4 read is 2× the bytes, so
  2‑bit is the *fast* default and FP4 is spent only where it's needed: a **delta cache** keeps
  the hot experts at FP4, and a **confidence gate** re‑runs the low‑confidence tokens at FP4
  (it fires where the 2‑bit and FP4 picks actually differ — 4.2× the base rate at τ=0.60).
  Result: MTP acceptance and coherence match FP4, multi‑step arithmetic is recovered.

So on a single 96 GB card — **with the FP4 recovery on** — the 159B model runs at **109 tok/s**
with 512K context, matching FP4 on MTP acceptance + coherence (the gate adds per‑token recovery
at 84). FP4‑level quality comes *from the recovery*, not from the bare 2‑bit base — that's the
89% floor. Either way it's a model the FP4 checkpoint can't even load on one card.

---

## Quickstart

```bash
git clone https://github.com/kacper-daftcode/vLLM-Moet && cd vLLM-Moet

# one self-contained build: clones official vLLM v0.19.2rc0, applies our patch,
# builds for sm_120, and bakes in the prebuilt SM120 cubins (~15-25 min)
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120 -t vllm-moet-sm120:base .

# serve — cubins are baked in, so only the checkpoint is needed
MODEL=/path/to/DeepSeek-V4-Flash bash tools/serve.sh 0 8000   # GPUs "0" / "0,1"; PP= for pipeline
```
`MOE_W2=0` runs native FP4 (full quality, ≥2 cards); default is the 2‑bit fit. Knobs: `GATE`,
`DELTA_GB`, `MAX_LEN`, `UTIL`, `SEQS`. All details → **[BUILD.md](BUILD.md)**.

---

## Full benchmarks

**Throughput** (single‑stream, MTP k=2, warmed coding corpus; continuous decode runs higher):

| run | hardware | experts | tok/s |
|---|---|---|---:|
| full FP4 (reference) | 2× PRO 6000 | FP4 | 177 |
| 2‑bit | 1× PRO 6000 | 2‑bit | 109 |
| 2‑bit + confidence gate (τ0.60) | 1× PRO 6000 | 2‑bit + FP4 recovery | 84 |
| 2‑bit | 4× RTX 5090 (TP4) | 2‑bit | 89 |
| 2‑bit + confidence gate (τ0.60) | 4× RTX 5090 (TP4) | 2‑bit + FP4 recovery | 47 |

The gate's cost depends on interconnect: −23% on one card (109→84) vs −47% on 4×5090 (89→47) —
no NVLink, so each re‑run re‑forwards across all four cards over PCIe.

**Quality** (vs official FP4 checkpoint): MTP acceptance 2.73 (≥ 2.68), draft accept 86.3% (vs
84.1%), 12/12 coherent, arithmetic recovered, 512K validated. The bare 2‑bit base agrees with
FP4 on **89%** of next‑token picks (teacher‑forced; same on 1× and 4×5090 ±0.1%) — that's the
*floor before recovery*; closing it is what the delta cache and gate do.

---

## How it works (details)

- **`cubit` SASS kernels** — `moe_w2_mm` (2‑bit MoE GEMM, PRMT‑LUT decode → `QMMA.SF`, tuned to
  4 CTA/SM), `moe_w4_mm` (FP4 delta GEMM), fused sparse‑MLA (decode+prefill), fp8 MQA‑logits.
  Both SASS sources (`kernels/sass/`) and prebuilt SM120 cubins (`kernels/cubins-sm120/`) ship;
  running needs no `cubit`. Op‑validated vs reference (rel ~1–3e‑3), graph‑capture‑exact.
- **Multi‑GPU.** Under TP each expert's down‑proj splits across GPUs (contraction K 2048 → 1024
  @ TP2 → 512 @ TP4); each K is a different GEMM with no library kernel, so we generate the
  cubins from one SASS source. The 4×5090 (K=512) run measures the same 89% fidelity as one
  card — confirming the shard kernels are numerically equivalent. PP keeps whole layers per rank.
- **FP4 delta cache** — full FP4 planes pinned on the host (when enabled); a GPU pool **sized to
  free VRAM** caches the hot experts at FP4 via a background promote/evict thread inside the
  CUDA graph. (A `need`‑driven pool was tested and rejected: low‑confidence steps route ~96% of
  experts, so 2‑bit difficulty doesn't concentrate — the useful signal is per‑token, the gate.)
- **Quality method** — baseline is the untouched official checkpoint; our variant changes only
  the expert codes (same stack, byte‑identical dense/scales/headers), so any delta is the
  quantization alone. See [docs/quality.md](docs/quality.md). Standardized functional evals
  (HumanEval/GSM8K) are the next step.

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
- **`kernels/`** — SASS (`sass/`) + prebuilt SM120 cubins (`cubins-sm120/`) + generators (`gen/`) + `MANIFEST.md`.
- **`tools/serve.sh`** — single launcher (TP/PP, `MOE_W2`/`GATE`/`DELTA_GB` knobs) + probes.
- **`patch/vllm-moet.patch`** — runtime delta vs official vLLM `v0.19.2rc0` (verified to apply clean).
- **`BUILD.md`** / **`Dockerfile.sm120`** — pinned base, SM120 build recipe, run instructions.
