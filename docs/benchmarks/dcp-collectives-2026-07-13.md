# DCP decode collectives: a2a combine + MTP k=5 ship (2026-07-13)

**Outcome: the 1M recipe (`glm-5.2-nvfp4/pro6000x4-tp4-dcp4-1m`) moves to
`--dcp-comm-backend a2a` + MTP k=5.** Empty-KV decode on the full recipe:
fox 99-102 tok/s (stable), refactor 78.8-91.5 tok/s in the good text
attractor (see "Greedy attractor lottery"), steps/s 16.8-17.5 at k=5 /
25.2-25.7 at k=2 (vs 24.2-24.3 ag_rs k=2 baseline). Needle PASS at
**1,038,717 tokens** (nvfp4 KV), arithmetic 5/5, coherence 0/12,
steps/s depth-flat to 500K (-2.3%).

Setup: 4x RTX PRO 6000 (PCIe, all pairs PHB, no NVLink), image
`vllm-moet-sm120:v024-r5` (repo `e45746b`, vllm `20228ee9a`), PYNCCL only
(`--disable-custom-all-reduce`), FULL_AND_PIECEWISE graphs. Probes:
`tools/dcp/decode_2prompts.py`, `tools/dcp/decode_at_depth.py` (tok/s +
steps/s + acceptance from per-request Prometheus counter deltas). Raw
data: `/root/bench-results/20260713-dcpperf-collectives/`.

## Where the 16 ms/step DCP4 tax actually is (torch profiler, 68 steps)

Baseline ag_rs k=2, 41.3 ms/step wall, GPU ~96% busy. Per-step NCCL
device-kernel time (rank0):

| collective | calls/step | time/step | avg |
|---|---:|---:|---:|
| AllGather RING_LL (AG-Q 78 + AG-LSE 78 + indexer ~21 + drafter) | 186 | 4.20 ms | 22.6 us |
| AllReduce RING_LL (TP o_proj/MoE - exists at DCP1 too) | 163 | 3.21 ms | 19.7 us |
| ReduceScatter RING_LL (RS-out 78 + drafter 2) | 80 | 2.42 ms | 30.3 us |

DCP-specific collectives are **~6.6 ms/step of kernel time** - not 16.
The task sheet's "~63 us/collective effective" folded structural per-step
overhead (drafter forwards, verify batching, graph serialization) into
the collective count. Consequences:

- NCCL was already on RING+LL: `NCCL_PROTO=LL` is a no-op, `NCCL_ALGO`
  irrelevant at 4 ranks for AG/RS. Measured: `NCCL_P2P_DISABLE=1`
  neutral (+0.3 tok/s; PHB topology - keep it in the *box* yaml, not the
  recipe), `NCCL_MAX_NCHANNELS=1` a -10 tok/s REGRESSION (rejected).
- The index-conversion kernel (`_convert_req_index_to_global_index`)
  costs ~2 us/layer, not 20-30: phase 2 (reorder before AG-Q) would hide
  ~0.16 ms/step, not 2-3. **Dropped** - not worth the custom-op signature
  change (compile-cache invalidation + autotune lottery).
- Even zeroing ALL remaining DCP comms (4.75 ms after a2a) lands at
  ~28.6 steps/s: the "steps/s >= 30 at k=2" bar is unreachable from the
  attention-collectives side alone. The step-rate gap to DCP1 is mostly
  structural, not comms.

## Phase 1: a2a combine (zero code, `--dcp-comm-backend a2a`)

One packed `all_to_all_single` (out + fp32 LSE in 2 bf16 lanes) replaces
AG-LSE + RS-out per attention layer; exact LSE-weighted reduction,
indexer top-k merge untouched. Measured (k=2, delta=0, same GPUs/cache):

| | refactor | fox | steps/s |
|---|---:|---:|---:|
| ag_rs | 62.4 (acc 2.71) | 71.1 (acc 3.00) | 24.2 / 24.3 |
| a2a | 72.8 (acc 2.97) | 76.0 (acc 3.00) | 25.2 / 25.6 |

Profile confirms: AllGather calls 12648 -> 7208 (AG-LSE gone), RS 5440 -> 0,
+80 SendRecv/step at 21.4 us = 1.71 ms; DCP comm kernel time 6.6 -> 4.75
ms/step; GPU busy -1.8 ms/step. Boot with FULL_AND_PIECEWISE captures the
a2a cleanly (buffers intentionally outside WorkspaceManager - unchanged).

## Phase 3: MTP k=5 (zero code)

Step cost grows 39.5 -> 54.9 ms (5 sequential drafter forwards, +39% -
more than the sheet's 10-15% guess), but acceptance on code/easy prompts
more than repays it. k=7 was tried and LOSES on refactor (acc saturates
~4.3-4.9, the extra 2 drafts are dead weight: 63.0 tok/s): k=5 ships.

Full recipe (delta=auto, gate tau 0.60), three identical boots:

| boot | refactor | fox |
|---|---:|---:|
| 1 | 78.8 (acc 4.89) | 99.4 (acc 5.94) |
| 2 | 50.6 (acc 3.02) | 101.9 (acc 5.94) |
| 3 | 91.5 (acc 5.65) | 100.7 (acc 5.99) |

tau runtime-swept on boot 2: 0.60/0.75/0.85 all ~50-51 - the attractor,
not tau, owns the refactor number. delta=0 comparison: 83.3 (acc 4.73) -
the FP4 tier costs ~4-5 tok/s at k=5 (verify batch of 6 gates more).

### Greedy attractor lottery (why refactor spreads 50-92)

The refactor prompt greedy-bifurcates between "write the pathlib
version" (drafter nails it, acc ~4.9-5.7) and a self-critique loop
("Let me fix that", acc ~3.0). Which attractor a boot lands in is
decided by ULP-level logit noise (autotune lottery, FP4 pool state, a2a
vs ag_rs reduction order); k=5 multiplies the acceptance delta into
+-25 tok/s (at k=2 the same fork was 62 vs 73). steps/s is the stable
metric: 16.8-17.5 across all k=5 boots/taus. Both attractor texts are
coherent code-with-prose; quality gates pass either way. Same class of
noise as DCP1-vs-DCP4 tie-breaks (task sheet) - now with a bigger lever.

## Validation (full recipe + a2a + k=5, boot 2 - the WORST attractor)

- needle (GLACIER secret, depth 0.5): PASS at 127,167 / 244,776 /
  519,153 / **1,038,717** prompt tokens.
- arithmetic 5/5, coherence 0/12 degenerate.
- depth flatness: steps/s 17.6 @8K -> 17.2 @497K (-2.3%, criterion +-10%)
  at k=5; 26.2 -> 25.7 (-1.9%) at k=2. (Session B measured -8% at 891K on
  real corpus - indexer local-shard scoring grows with depth; still
  inside +-10%.)
- A/B greedy a2a vs ag_rs (k=2, delta=0): fox byte-identical; refactor
  diverges at char 313 into the two known attractors (both valid).
- DCP1 regression: N/A - zero shared-code changes (a2a is a per-recipe
  CLI flag; the config validator rejects it at dcp=1), DCP1 recipes
  untouched.
- No patch changes this session: FILES.txt and vllm-moet-v0.24.0.patch
  are byte-identical to `e45746b`; the serving image stays v024-r5.
  Only the recipes image needs a rebuild (recipe yaml changed).

## What did NOT ship

- NCCL env knobs (all neutral-or-worse on PHB; box yaml already carries
  NCCL_P2P_DISABLE=1 for host quirks).
- Phase 2 reorder (measured upper bound ~0.16 ms/step - see above).
- Indexer all-gather: untouched per task sheet (local-quota mode stays
  a documented opt-in; kills MTP acceptance + fails 1M needle).

## Follow-ups

- README GLM table has no 1M row yet ("1M window is KV-bound on 4
  cards" is stale since a664675) - update at next release pass with the
  boot-median numbers above.
- The refactor attractor lottery deserves its own probe (N-boot
  distribution, or a fixed-seed multi-prompt panel) before headline
  numbers are quoted from single boots at k>=5.
- steps/s recovery beyond ~25.7 at k=2 needs structural work (drafter
  cost, verify path), not attention collectives.
