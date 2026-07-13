# MTP acceptance vs context depth — the "decay" was a probe artifact

**Verdict (2026-07-13): GLM-5.2 MTP drafter acceptance does NOT decay with
context depth on real data.** The 2.9 -> 1.8 "decay" at 891K observed in the
2026-07-12 DCP session was an artifact of the random-word filler used by the
original `decode_at_depth` probe. No mitigation is needed; the 1M recipe keeps
MTP k=2 unchanged.

## Setup

- Server: recipe `glm-5.2-nvfp4/pro6000x4-tp4-dcp4-1m` (TP4+DCP4, nvfp4 KV
  8.5 GiB/rank, FP4 delta `auto` + gate tau 0.60, MTP k=2, FULL_AND_PIECEWISE
  graphs), image `vllm-moet-sm120:v024-r5` (repo `e45746b`, vllm `20228ee9a`,
  DCP exact merge = default), 4x RTX PRO 6000.
- Probe: `tools/dcp/decode_at_depth_real.py` — greedy 512-token completions,
  1 discarded warm + 3 measured runs per depth; acceptance/steps/decode-time
  from per-request Prometheus counter deltas (server otherwise idle, guarded
  by `request_success_total` delta == 1); per-position acceptance recorded;
  output-degeneration ratio (unique/total 4-grams over the output tail)
  guards against loop-inflated acceptance.
- Real filler: deterministic concatenation of the `vllm-moet-src` tree at
  `20228ee9a` (408 text files, 899K GLM tokens, never looped). Junk filler:
  the legacy 20-word random vocabulary, seed 7 (as `needle_full.py`).
- Raw data: `/root/bench-results/20260713-0038-mtpdecay-phase0/` (jsonl+log
  per probe, `server.json` fingerprint).

## Results (medians of 3)

Real corpus, code task ("refactor", the `decode_ab.py` prompt):

| depth | decode tok/s | acceptance len | steps/s | degen |
|---:|---:|---:|---:|---:|
| 8K   | 66.5 | 2.942 | 22.6 | 0.67 |
| 200K | 65.7 | 2.942 | 22.2 | 0.45 |
| 500K | 66.2 | 2.959 | 22.4 | 0.57 |
| 700K | 66.1 | 2.954 | 22.3 | 0.57 |
| 891K | 60.5 | 2.942 | 20.5 | 0.57 |

Real corpus, prose task ("lighthouse story"):

| depth | decode tok/s | acceptance len | steps/s | degen |
|---:|---:|---:|---:|---:|
| 8K   | 43.8 | 2.004 | 22.4 | 0.92 |
| 200K | 44.8 | 2.000 | 22.4 | 0.99 |
| 500K | 42.4 | 1.914 | 22.3 | 1.00 |
| 700K | 43.9 | 1.969 | 22.3 | 0.96 |
| 891K | 44.9 | 2.020 | 22.2 | 0.98 |

Random-word filler, same boot, prose task:

| depth | decode tok/s | acceptance len | steps/s | degen |
|---:|---:|---:|---:|---:|
| 8K   | 43.9 | 1.943 | 22.6 | 1.00 |
| 200K | 66.8 | 2.937 | 22.9 | **0.01** |
| 500K | 45.0 | 2.028 | 22.3 | **0.01** |
| 700K | 40.6 | 1.862 | 22.1 | **0.02** |
| 891K | 42.6 | 1.928 | 22.1 | 0.99 |

## Reading

1. **On real context, acceptance is depth-flat** for both tasks (2.94-2.96
   for code, 1.91-2.02 for prose, 8K through 891K). Acceptance is a property
   of the *generated text*, not of the context depth.
2. **Junk context induces degenerate outputs, and those poison acceptance
   both ways.** With random-word filler the greedy continuation falls into
   repetition attractors at 200K-700K (degen 0.01-0.02). A short loop is
   trivially draftable (2.94 @200K); other loops are not (2.03 @500K, 1.86
   @700K). The 2026-07-12 "baseline 2.9 @202K/500K" was loop-inflated; the
   "decayed 1.8 @891K" was simply a non-degenerate outcome at the true
   prose-task level (~1.9-2.0). Nothing decays — the artifact does.
3. **Which attractor you get is a lottery.** Identical greedy requests
   produce different outputs run-to-run (3 distinct shas per depth): the FP4
   delta/gate tier perturbs logits as its pool converges, tipping the
   trajectory into or out of loops. Any acceptance comparison on this stack
   needs the degeneration guard + median-of-N, or it measures noise.
4. **The only real depth effect is on the step rate**: steps/s 22.3 -> 20.5
   (-8%) at 891K (code task), i.e. step *time*, not acceptance — that is the
   DCP indexer/attention-collectives line of work, tracked separately.

## Methodology rules going forward

- Never benchmark decode/acceptance over random-word filler; use a real
  corpus (`--source-tree`) — junk context measures loop-attractor luck.
- Always record the degeneration ratio next to acceptance; discard or flag
  runs with degen < ~0.5.
- Derive acceptance from Prometheus counter deltas per request (the
  `SpecDecoding metrics` log line aggregates cumulative windows across
  requests and cannot be attributed to a depth point).
- Acceptance depends on the task: fox-loop ~3.0 (ceiling), refactor ~2.94,
  prose ~2.0. Compare like with like; never mix tasks across depth points.
