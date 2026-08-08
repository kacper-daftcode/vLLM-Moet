# SM120 ALU subunits — measured throughput map and kernel-design rules

Hand-written SASS kernels in this project (the `cubit` GEMV family) are
**ALU-issue-throughput-bound** at M=1, and the real constraint is invisible in
ncu: per-subpipe counters stay below ~36% while the shared execution resource
saturates. This page is the silicon-measured map of that resource and the
design rules that follow. If you are writing a new kernel for this stack,
start here — the mix decisions below are worth more than any amount of
instruction scheduling.

Everything measured on RTX PRO 6000 Blackwell (SM120), clocks locked
1680 MHz, CUDA 13.0, deterministic ×3, fresh process per variant. Anchor
cells reproduce the earlier ALU-cap campaign to 0.1 cyc/instr.

## Probe method (self-contained, reproduce at will)

One CTA of 16 warps on one SM; warp classes paired **on schedulers** via
`warpid & 0x4` (each of the 4 schedulers gets 2+2 warps of the two classes —
pairing on the scheduler is what makes the verdict about the *unit*, not
about arbitration). Each warp issues 512 back-to-back ops of its class with
an 8-register rotation (8-way ILP hides ≤6-cycle latency; stall S01), timed
with `SR_CLOCKLO` deltas. Positive control: IMAD|HMMA shows full separation
(4.69/32.0 — tensor is its own resource), so the probe distinguishes sharing
from separation.

## The map

Issue throughput per scheduler (instructions/cycle, both classes combined;
saturation = 4 warps/scheduler):

| pairing | combined instr/cyc/sched |
|---|---|
| any single class alone (IMAD, IDP.4A, HFMA2, HADD2, LOP3, PRMT, SHF, SGXT) | **0.50** |
| IMAD \| IDP, IMAD \| HFMA2, IMAD \| HADD2 (within class A) | 0.50 |
| PRMT \| LOP3, SHF \| LOP3, PRMT \| SHF, SGXT \| LOP3, SGXT \| PRMT (within port B) | 0.50 |
| **A \| B** (IMAD\|LOP3, IMAD\|PRMT, HADD2\|LOP3) | **0.66** |
| **A \| S** (IMAD\|SHF, IMAD\|SGXT) | **0.94 — issue-limited, near dual rate** |
| IMAD \| HMMA (control) | separate resources |

Three classes emerge:

- **A ("math"):** `IMAD ≡ IDP.4A ≡ HFMA2 ≡ HADD2` — one shared resource.
  The ISA-table pipe classes (INT_ARITH vs FP_ARITH) do **not** match this
  physical sharing; fp16 ALU ops live on the same unit as integer multiply.
- **B ("logic port"):** `LOP3 ≡ PRMT` — one shared port. Swapping work
  between LOP3 and PRMT changes nothing.
- **S ("shift"):** `SHF ≡ SGXT` — shares the B port (B|S = 0.50), but is
  **nearly free alongside A** (0.94 combined, the only pairing that
  approaches dual rate).

The cap is visible even without contention: a single warp of any class with
8-way ILP runs at 2.13 cyc/instr (≈0.47/cyc).

## Three-class mixes beat the pair table; budget the B-port share

Real decode streams mix all three classes, and the measured ceilings sit
**above** a pairwise interpolation:

| stream (single warp-stream, 16 warps) | class shares A:S:B | instr/cyc/sched |
|---|---|---|
| 2:1 B-heavy (3INST-GEMV profile) | 33:0:67 | 0.576 |
| projected MUL1-decode GEMV (probe `ucb2`) | 56:19:25 | **0.694** |
| nvcc wave K1 (masks on SGXT, packing on IMAD) | ~50:~21:29 | 0.65 |

Two working rules from kernels built against this map (sessions [G]/[J]):

- **Budget the LOP3+PRMT share of the stream to ≤ ~33%**, not just the
  total op count. A hand kernel with 11% fewer instructions but 57% of
  them on port B measured *slower* than nvcc's stream at 29%.
- Moves that shift pair-packing work from PRMT (B) to IMAD/HADD2 (A) are
  real wins; `HADD2.H1H1A Rd, w, w` (h0 = lo+hi, bit-exact for fp16 adds)
  replaces a PRMT in every lo+hi fold of the 3INST decode.

## Tensor pipe: accumulator width is free; QMMA halves instructions, not rate

Saturated rates, 16 warps, same probe pattern (session [J]):

| form | cyc/instr/warp |
|---|---:|
| HMMA.16816.F16 | 63.83 |
| **HMMA.16816.F32** | **63.83** |
| QMMA.16832.F32.E4M3.E4M3 | 63.83 |
| HMMA.F32 \| QMMA.F32 mixed on a scheduler | 63.90 / 61.33 |

FP32 accumulation is **not** half-rate on this part (unlike GeForce-class
silicon), and HMMA/QMMA remain one throughput domain. QMMA's only value is
2× k-depth per instruction — it halves the tensor *instruction count*, not
the cycle cost. Before designing for it, check the tensor budget against
the ALU floor of your mix: at M=8 the decode/extraction ALU work dominates
the GEMV classes measured here (tensor ≈ 50-60% of the ALU floor with
HMMA.F32, ≈ 25-30% with QMMA), so the halving does not move wall time.
Note the numerics: e4m3 on **both** QMMA operands measured a 3.7e-2
rel-RMS band on the real delta streams — a class change versus the
lossless fp16 decode path, to be re-certified before any use.

## Mix sets the ceiling; arrangement does not

- A 2:1 B:A stream (the profile of the 3INST-decode GEMV) measures
  **0.576/sched** regardless of arrangement.
- **The hardware integrates the class mix over a ~8–32 op window**: 8-op
  class blocks ≈ perfect op-level interleave (6.17 vs 6.11 cyc/instr),
  32-op blocks partial (6.54), 128-op blocks mostly lose the relief (7.40).

Confirmed end-to-end on the production unified GEMV (K2-cb0, 1558 instr,
31.2–31.5 µs solo): three bit-exact interventions — op-level class
interleave (+0.89 µs: dependency padding costs more than balance gains),
warp start-skew (neutral), LOP3→SGXT extraction swap (neutral: the B/S port
stays binding) — all failed to move wall time, exactly as the integration
window predicts. That kernel runs at ~92% of its mix ceiling: **it sits at
the ALU floor of its fixed instruction mix.**

## Design rules for new kernels

1. **Budget the mix before writing any SASS.** Count port-B ops
   (LOP3 + PRMT + SHF + SGXT) and A-ops per weight; the B port is usually
   binding. Ceiling for your mix interpolates the table above (1:1 → 0.66,
   2:1 B-heavy → 0.576, pure → 0.50 per scheduler).
2. **Move work to class A when semantics allow.** A|B co-runs at 0.66;
   B|B at 0.50. An op that can be IMAD/IADD-shaped instead of LOP3/PRMT-shaped
   is a throughput win even at equal op count.
3. **Route extraction through SHF/SGXT in A-heavy streams.** A|S = 0.94 —
   an A-dominated decode chain (e.g. the MUL1 codebook:
   IMAD → IDP.4A → HFMA2, pure class A) gets its shifts/extractions almost
   for free. Note `SGXT.U32 Rd, Ra, imm` (zero-extend = AND mask) is
   bit-equivalent to the LOP3 AND and lives on S.
4. **Do not shuffle instructions to "balance units."** The hardware
   integrates the mix over ~8–32 ops; phase-level clustering is already
   smoothed, and interleaving tightens dependency chains (padding costs
   real cycles). Spend scheduling effort on latency coverage, not class
   alternation.
5. **Do not trade LOP3 ↔ PRMT** — same port, zero effect.
6. **Tensor is separate.** HMMA/QMMA co-issue freely with ALU classes
   (warp-heterogeneous or within a warp). ALU→tensor offload (e.g. QMMA
   for the delta codebook at M≥8, where the tensor pipe stops idling) is
   real headroom; ALU-side rearrangement is not.
7. **Distrust per-pipe ncu counters for this diagnosis.** The shared-unit
   saturation shows up as `math_pipe_throttle` warp stalls and sub-40%
   per-pipe utilization simultaneously. Measure with the probe pattern
   above when in doubt.

## Provenance

Session [H] 2026-08-07/08: 25 probe variants + three bit-exact kernel
interventions, each with full parity battery (bit-equality vs the certified
build ×3, synthetic + real-pack gates, A/B cold-rotation benches ×2). The
`SGXT.U32` imm-width encoding used by rule 3 was missing from the ISA table
and shipped from the same ground truth (blackwell-isa `4e8a589`, cubit
`dd97d6e`).

Session [J] 2026-08-08: tensor-form rates (HMMA.F16/.F32, QMMA.F32, mixed),
the `ucb2` three-class mix ceiling, and the e4m3 delta-band measurement on
real packs; same probe harness, anchors reproduce [T]/[H] to 0.05 cyc/instr.
Session [G] 2026-08-08: the ≤33% B-port budget rule and the
`HADD2.H1H1A` pack replacement, validated on the K1-cb0 port (full parity
matrix, 1.20× vs the nvcc wave kernel measured under the same contract).

Measurement hygiene on this box: the bench GPUs run with a reduced power
limit (300 W vs the 600 W default) — sustained loads throttle SM clocks
below the lock after a few tens of milliseconds, while the canonical
500-iteration burst protocol holds the locked clock. Certified numbers
reproduce to 0.01 µs under that protocol; profiler (ncu) replay runs in a
different clock regime and is used for attribution only, never for wall
time.
