# DS4 W2 guarded validation evidence — 2026-07-11

This directory contains the sanitized, publishable receipts for the single-RTX-5090
DeepSeek-V4-Flash W2 safety, quality, and context run. The measured narrative and evidence
boundaries are in [`docs/benchmarks/ds4-w2-5090-2026-07-11.md`](../../../docs/benchmarks/ds4-w2-5090-2026-07-11.md).

## Artifact lineage

| Evidence role | Artifact identity |
|---|---|
| P0 all-layer cold canary | mechanism-equivalent predecessor `e7417054a6e8`; measured memory belongs only to this canary |
| Historical P1 32K receipts | patch `55d30bb9cf9bef45e7130fd5afe0090c5b540be22f6de3abfc51b782f738a6f7`; image `sha256:7385d21d26b665884e97a97dc67a100db328ed7b00b634e4d18f8aedd9f29eab` |
| Comprehensive integrated P2 predecessor | official vLLM `ee0da84ab9e04ac7610e28580af62c365e898389`; patch `41d7b2f96ca3b966cac1b7ed5cff37bc03c27c616ed254aca961cc75a9ffe31d`; image `sha256:66abc2f145244e03ff0f0fcca088be813ad311f2963c62b281e4bb888ac605e9`; wrapper `5e69228e1e8e6f7345ae96657ab33b50e5805be7` |
| Frozen release candidate and focused current-head supplement | official vLLM `ee0da84ab9e04ac7610e28580af62c365e898389`; origin/main cutoff `94be3aa3d7a8b82c7fc9687990a7edb6035f69f3`; patch `241ba984b1c56f5dc7adbc8d7f519d60b5746024bf7dfeb875e3546a668e79a7`; image `sha256:fc6e1244d60855fe45ccc0236daaaa722abcb8d354200eb92aca104bd954d3f2`; wrapper `0a73c2bcf262ba0aa53560d6290ad2af350e34fc` |

[`exact-image-verification.json`](exact-image-verification.json) records the clean-apply shape,
baked source hashes, exact-image test matrix, and synthetic safety-probe hashes for the frozen
release candidate. The earlier `41d7b2f` artifact remains the comprehensive three-seed,
three-depth quality record; the frozen candidate adds a focused exact-head release sentinel.

## P0 — bounded cold restage

[`p0/`](p0/) contains the all-43-layer cold-restage memory trace and guarded cleanup receipt.
That measured canary ran on mechanism-equivalent predecessor patch `e7417054a6e8`; its memory
maxima are not represented as integrated-image measurements. The checkpoint-residency,
next-shard retry, and constrained-pack probes in the same directory were rerun on the frozen
release-candidate image and bind to its exact image ID.

The canary kept cgroup swap, pressure, hard-limit, and OOM events at zero and recorded no host
swap-out. Host swap-in increased by 160 pages; that boundary is explicit in the summary and trace.

The frozen candidate also received a fresh full cold restage. [`p0/frozen-rc-summary.json`](p0/frozen-rc-summary.json)
records 46/46 checkpoint shards, all 43 expert-pack layers, zero swap and OOM activity, zero
memory-limit event deltas, preserved production and immutable-seed fingerprints, restored service
health, and disposable-store removal. Its 218-sample cold-only trace is
[`p0/frozen-rc-memory-trace.tsv`](p0/frozen-rc-memory-trace.tsv). These measurements bind to the
frozen candidate and supersede the historical limitation only for that candidate; the earlier
canary files remain historical receipts.

## P1 — 32K quality and near-edge context

The files under [`p1/`](p1/) are retained byte-for-byte as historical predecessor evidence.
[`p1/aggregate.json`](p1/aggregate.json) reports 119/120 machine-exact, 120/120 semantically
correct, and 0/120 frozen-rule sink detections across seeds 42–44, plus 30/30 prewarm prompts and
all six pool-lifecycle gates. Per-seed manifests, raw responses, and warmup receipts are retained
beside the aggregate. Published receipt hashes are recomputed after sanitization.

[`p1/context-30k.manifest.json`](p1/context-30k.manifest.json) and
[`p1/context-30k.receipts.jsonl`](p1/context-30k.receipts.jsonl) record 3/3 exact needles at
30,000 prompt tokens. [`p1/runtime-clean.json`](p1/runtime-clean.json) records the complete lane
through context, including soft `memory.high` reclaim and nonzero PSI; [`p1/cleanup.json`](p1/cleanup.json)
records fingerprint preservation, service restoration, and disposable-store removal.

## P2 — comprehensive 128K series and frozen-RC sentinel

The comprehensive P2 receipts bind to the `41d7b2f` integrated predecessor above. Its three-seed
128K series also satisfies the P1 stability requirement.
[`p2/quality-aggregate.json`](p2/quality-aggregate.json)
reports 120/120 machine-exact and semantically correct, with 0/120 frozen-rule sink detections at
a 131,072-token window.

[`p2/context-120k.manifest.json`](p2/context-120k.manifest.json) and
[`p2/context-120k.receipts.jsonl`](p2/context-120k.receipts.jsonl) contain the admitted exact
120,000-token receipts at depths 0.1, 0.5, and 0.9. All three published receipts use zero token
tolerance. Each case calibrated to exactly 120,000 tokens with tokenizer-only requests before its
sole inference; no rejected calibration contributed a chat response in this run.
Readiness, complete runtime/memory, and guarded cleanup are recorded in
[`p2/readiness.json`](p2/readiness.json), [`p2/runtime-clean.json`](p2/runtime-clean.json), and
[`p2/cleanup.json`](p2/cleanup.json).

[`p2/frozen-rc-sentinel.json`](p2/frozen-rc-sentinel.json) is the focused supplement for the
frozen candidate. It records two quality seeds (80 responses): 79/80 strict machine-exact,
80/80 semantically correct, 0/80 sink detections, all 20 prewarms without retry, and all four
pool gates. The sole strict mismatch is an explicitly recorded equivalent LaTeX rendering of
`3/8`. It also records one exact 120,000-token retrieval at depth 0.9 with zero token tolerance,
plus complete-run memory and cleanup results. This supplement did not repeat seed 44 or depths
0.1 and 0.5, so those broader claims remain attached only to the immediately preceding
comprehensive P2 artifact.

The frozen source supports split-FP4 refinement over a base cache only behind the explicit split
flag, with coupled base/refinement residency. The serving sentinel kept split mode disabled and
used full-nibble FP4 delta planes. Split mode was exercised separately by the exact-image E=8 GPU
three-tier fixture; serving-sentinel results are not presented as live split-mode evidence.

## Interpretation and sanitization

These receipts establish bounded restaging, observed quality, pool liveness, exact selected
retrieval, and guarded cleanup for the recorded artifact and policy. They do not establish
bit-deterministic output, a miss-free forward, numerical equivalence to fully resident FP4, or
128K throughput. Second-order residue remains explicit.

Host topology, opaque runtime IDs, host-specific paths, host- and address-specific endpoint
details, checkpoint/pack
fingerprints, launcher identity, and response IDs remain in private source receipts. Public run
labels are descriptive. The lane started on July 11 Pacific time, while UTC timestamps in the
integrated receipts fall on July 12. Verify the final public tree from this directory with:

```bash
shasum -a 256 -c SHA256SUMS
```
