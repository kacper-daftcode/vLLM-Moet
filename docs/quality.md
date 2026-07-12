# Quality — methodology, ablation, and delta recovery

The headline tables are in the [README](../README.md). This is the detailed companion.

## What we read and compare to
Baseline = the **untouched official DeepSeek‑V4‑Flash checkpoint** (~149 GiB), verified from its
safetensors headers: MoE experts are **FP4** (`I8`, e2m1 packed 2/byte) with **`F8_E8M0`**
block‑32 scales (`layers.0..42.ffn.experts.E.{w1,w2,w3}.weight`, 33,024 tensors); dense and
attention are **FP8** (`F8_E4M3`, block‑128, per `config.json`). This project changes **only the
expert codes** (FP4 → 2‑bit symmetric) — scales, dense FP8, and headers stay byte‑identical — and
runs on the *same* serving stack (only `MODEL_DIR` differs).

## Probe protocol (`tools/probe_quant_quality.py`)
1. **MTP acceptance** — mean accepted draft length + avg draft‑acceptance rate over 3×512‑token
   greedy generations (`ignore_eos`). Primary fidelity proxy (drafter+target share the model, so a
   degraded target lowers acceptance). Noise band ≈ ±0.05.
2. **Coherence** — 12 fixed prompts, exact‑match vs the official‑checkpoint outputs + manual verdict.
3. **Arithmetic** — 5 multi‑step problems (the official model itself tops out at 3/5).
4. **Long context** — passphrase needle at 100K/250K/440K depths (`tools/needle_probe.py`).

## DS4-W2 reliability methodology (P1/P2 release gate)

The release gate uses `tools/ds4_eval/`; it is stricter than a transcript spot check. Every warm
run must first pass the fixed ten-prompt `ds4-w2-prewarm-v4` suite (temperature 0, top-p 1, seed
`20260711`, independent of the eval seed). There are no hidden retries: the first bad response
aborts and its receipt remains on disk. The run manifest binds the result to immutable server
provenance, including the boot, container and image identities, complete source-diff hash,
checkpoint and pack fingerprints, launcher hash, runtime argv, W2 environment, and engine settings.

Before looking at quality results, register one pool policy and retain its SHA-256. The same frozen
policy evaluates the configured live metrics before and after scoring. The release policy requires
full FP4 occupancy, zero unrestored BASE experts, and measured eviction progress so a stale
saturated pool cannot pass; replay and second-order residue remain recorded in every snapshot even
when they are not rejection thresholds. A failed pre- or post-gate invalidates the run regardless
of its answers. At the default continuation threshold (`FP_THRESH=0`), second-order experts are
fetched for later steps but their current-step contributions remain zero, so `UNRESTORED=0` is a
mechanism-liveness gate rather than a miss-free or bit-deterministic claim. The three-seed quality
receipts are the empirical acceptance test for that explicit throughput approximation.

P1 scores the fixed 40-item set separately at eval seeds 42, 43, and 44 (120 responses total).
Exact clean correctness is primary: the normalized terminal answer must match and the response
must not be a sink. The answer-anywhere `lenient` score is diagnostic only and never rescues a
sink. Sink rules catch max-token non-completions, repeated 3/4-grams, repeated lines, collapsed
vocabulary, and special-token spew; duplicate IDs, missing rows, or conflicting token-count aliases
fail closed. The corrected historical tau-0.67 baseline is **16/120 sinks (13.3%)** across those
three seeds. It is context only, not a tau-0.75 control: its warmups and pool evidence do not meet
the new comparability contract.
The corrected baseline summary is published under
[`evidence/public/ds4-w2-2026-07-11/baseline/`](../evidence/public/ds4-w2-2026-07-11/baseline/).

Use repository-relative inputs and a localhost endpoint; write each seed to a new output directory:

```bash
PORT="${PORT:-18001}"
python3 tools/ds4_eval/eval_rig.py \
  --items tools/ds4_eval/items.json \
  --server-provenance evidence/example-p1/server.json \
  --pool-gate-policy evidence/example-p1/pool-gate.json \
  --pool-command-json '["docker","logs","ds4-w2-candidate"]' \
  --output-dir evidence/example-p1/seed-42 \
  --run-label example-p1-s42 \
  --url "http://localhost:${PORT}/v1/chat/completions" \
  --model deepseek-v4-flash-w2 --mode warm --eval-seed 42 \
  --eval-temperature 0.6 --eval-top-p 0.95 --eval-max-tokens 700 \
  --expected-count 40
```

Repeat with seeds 43 and 44 and distinct labels/directories. P2 is a two-part contract on the same
ready server: first run this quality gate, then run `tools/ds4_eval/context_probe.py` with MTP off,
`--max-num-seqs 1`, a validated 131,072-token window, and 120,000-token needles at depths 0.1,
0.5, and 0.9. The probe requires exact tokenizer/usage token counts and an exact terminal
passphrase. A successful 128K boot without retrieval, or retrieval without the quality gate, is
not a P2 verdict.

The guarded single-RTX-5090 application of this protocol is recorded in
[`ds4-w2-5090-2026-07-11.md`](benchmarks/ds4-w2-5090-2026-07-11.md), with sanitized machine
receipts under [`evidence/public/ds4-w2-2026-07-11/`](../evidence/public/ds4-w2-2026-07-11/).
The historical predecessor P1 series at 32K produced 119/120 machine-exact and 120/120
semantically correct answers with 0/120 frozen-rule sink detections. The current integrated
artifact's independent P2 series at 128K also produced 119/120 machine-exact, 120/120
semantically correct, and 0/120 sink detections, thereby satisfying the P1 stability gate at the
larger window. In each series, the sole machine mismatch was the equivalent `3/8` LaTeX wrapper.
P2 then passed exact 120,000-token needles at depths 0.1, 0.5, and 0.9. Calibration-only seed
selection happened before inference; the public composite contains only fresh zero-tolerance
receipts whose tokenizer and response-usage counts both equal 120,000.

## Bits‑vs‑quality ablation (same stack, only the expert codes change)
| codebook | bits | MTP acc. length | draft accept % | arithmetic | coherence |
|---|---|---|---|---|---|
| FP4 (16‑level) — official | 4.0 | 2.68 | 84.1 | 3/5 | 12/12 |
| K=8 (3‑bit) | 3.0 | 2.60 | 79.8 | 3/5 | 12/12 |
| K=6 (~2.58‑bit) | 2.58 | 2.68 | 84.0 | 2/5 | 11/12 |
| **K=4 naive 2‑bit (asymmetric)** | 2.0 | **1.00** | **0.0** | 0/5 | **0/12 — broken** |
| **K=4 tensor‑sym `{−4,−1,1,4}` (ours)** | 2.0 | **2.73** | **86.3** | 2/5 | **12/12** |

## The finding: it was sign bias, not magnitude
Naive 2‑bit collapses to a degenerate loop (`}<?}<?…`). The optimal‑L2 2‑bit codebook goes
**sign‑asymmetric** (drops one sign's tail, e.g. `{−3,−1,0.5,4}`); accumulated per‑expert sign
bias over 43 layers destroys the model. Forcing a **sign‑symmetric** codebook at the *same* L2
error restores MTP acceptance to **at/above** the official FP4 experts (2.73/86.3 vs 2.68/84.1),
12/12 coherent. Symmetry — not granularity — is the lever (33,023/33,024 tensors pick the same
`{−4,−1,1,4}`).

## Delta recovery (FP4 hot‑expert tier)
Symmetric 2‑bit already matches the official experts on acceptance + coherence. The residual is
**numerical precision** (multi‑step arithmetic) — the 2‑bit base is "coherent but numerically
sloppy." Caching the hot experts at FP4 closes it:

| | 2‑bit base only | **2‑bit + FP4 delta** |
|---|---|---|
| arithmetic, e.g. `17×23−100` | ✗ | **✓ (=291)** |
| single‑stream decode | 151 tok/s | **143 tok/s** (≈ −0.6 ms/step) |
| extra VRAM | 0 | **~2 GiB** (170 hot‑expert slots) |

Production 1‑GPU end‑to‑end A/B (warmed): MTP acceptance with the delta tier **2.725 / 2.744**
(mean/median, σ 0.136) — at/above the same‑run two‑card FP4 control **2.546 / 2.496** (σ 0.172).

## Studying the precision tiering

The delta tier is a *workload‑adaptive* mixed‑precision scheme — the set of FP4 experts is
chosen at runtime from routing. The manager is instrumented so this can be measured directly
(default off; logging only, no effect on the serving path):

```bash
VLLM_MOE_W2_DELTA_TRACE=1                 # coverage + churn summary + per-layer FP4 histogram
VLLM_MOE_W2_DELTA_TRACE=2                 # + one line per promote/evict event
VLLM_MOE_W2_DELTA_TRACE_EVERY=64          # ticks between summaries
VLLM_MOE_W2_DELTA_DUMP=/tmp/delta.json    # atomic precision-map snapshots (tail-able)
```

Each summary reports filled slots, coverage (`cached / layers×experts`), per‑window and
cumulative promote/evict counts, and FP4 experts per layer; the JSON dump is the full
`{layer: [fp4 expert ids]}` map plus counters. `DeltaTier.precision_of(layer, expert)` and
`precision_map()` expose the same state programmatically. `tools/delta_trace_demo.py` drives
the real manager with synthetic Zipfian routing so you can see this output (and validate the
instrumentation) in ~1 s without loading the model. Questions this makes answerable:

- **Coverage vs. pool size** — sweep `VLLM_MOE_W2_DELTA_GB` and watch how coverage % and the
  arithmetic/recall probes move; find the knee where extra VRAM stops buying quality.
- **Working‑set stability** — cumulative evictions / promotions over a long run is a churn
  rate; low churn means routing is concentrated and a small pool suffices.
- **Per‑layer concentration** — the histogram shows whether some layers route to far more
  distinct experts than others (where precision matters most).

These are open questions the design invites; the README tables are the headline, this is the
instrumentation to push further.

## Reproduce
```bash
python3 tools/probe_quant_quality.py <port>           # acceptance + coherence + arithmetic
python3 tools/needle_probe.py <port> 48000 0.1        # long-context retrieval
# ablation variants are produced offline by tools/repack_expert_bits.py (quality probe only)
```
These are engineering‑grade fidelity checks, not a standardized benchmark sweep (MMLU/GSM8K‑style
evals are a natural next step). Discipline: every surprising delta is re‑measured against a
same‑GPU placebo control before it's believed.
