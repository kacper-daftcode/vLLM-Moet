# DS4 W2 guarded validation evidence — 2026-07-11

This directory contains the sanitized, publishable receipts for the single-RTX-5090
DeepSeek-V4-Flash W2 safety, quality, and context run. The measured narrative and evidence
boundaries are in [`docs/benchmarks/ds4-w2-5090-2026-07-11.md`](../../../docs/benchmarks/ds4-w2-5090-2026-07-11.md).

## Canonical identity

- Official vLLM source: `ee0da84ab9e04ac7610e28580af62c365e898389` (`v0.24.0`)
- Canonical source patch: `55d30bb9cf9bef45e7130fd5afe0090c5b540be22f6de3abfc51b782f738a6f7`
- Runtime image: `sha256:7385d21d26b665884e97a97dc67a100db328ed7b00b634e4d18f8aedd9f29eab`
- Image patch label: exact match to the full source-patch SHA-256

[`exact-image-verification.json`](exact-image-verification.json) records the clean-apply shape,
baked source hashes, exact-image test matrix, and canonical synthetic safety-probe hashes.

## P0 — bounded cold restage

[`p0/`](p0/) contains the all-43-layer cold-restage memory trace and guarded cleanup receipt.
That measured canary ran on mechanism-equivalent predecessor patch `e7417054a6e8`; its memory
maxima are not represented as canonical-image measurements. The checkpoint-residency,
next-shard retry, and constrained-pack probes in the same directory were rerun on the canonical
image and bind to its exact image ID.

The canary kept cgroup swap, pressure, hard-limit, and OOM events at zero and recorded no host
swap-out. Host swap-in increased by 160 pages; that boundary is explicit in the summary and trace.

## P1 — 32K quality and near-edge context

[`p1/aggregate.json`](p1/aggregate.json) reports 119/120 machine-exact, 120/120 semantically
correct, and 0/120 frozen-rule sink detections across seeds 42–44, plus 30/30 prewarm prompts and
all six pool-lifecycle gates. Per-seed manifests, raw responses, and warmup receipts are retained
beside the aggregate. Published receipt hashes are recomputed after sanitization.

[`p1/context-30k.manifest.json`](p1/context-30k.manifest.json) and
[`p1/context-30k.receipts.jsonl`](p1/context-30k.receipts.jsonl) record 3/3 exact needles at
30,000 prompt tokens. [`p1/runtime-clean.json`](p1/runtime-clean.json) records the complete lane
through context, including soft `memory.high` reclaim and nonzero PSI; [`p1/cleanup.json`](p1/cleanup.json)
records fingerprint preservation, service restoration, and disposable-store removal.

## P2 — 128K quality and exact 120K retrieval

[`p2/quality-aggregate.json`](p2/quality-aggregate.json) reports 117/120 machine-exact,
120/120 semantically correct, and 0/120 frozen-rule sink detections at a 131,072-token window.
All three machine mismatches are the same mathematically equivalent `3/8` LaTeX wrapper.

[`p2/context-120k.manifest.json`](p2/context-120k.manifest.json) and
[`p2/context-120k.receipts.jsonl`](p2/context-120k.receipts.jsonl) contain the admitted exact
120,000-token receipts at depths 0.1, 0.5, and 0.9. The original depth-0.9 calibration landed at
119,999 and is excluded; the published composite selects its zero-tolerance corrective receipt.
Readiness, complete runtime/memory, and guarded cleanup are recorded in
[`p2/readiness.json`](p2/readiness.json), [`p2/runtime-clean.json`](p2/runtime-clean.json), and
[`p2/cleanup.json`](p2/cleanup.json).

## Interpretation and sanitization

These receipts establish bounded restaging, observed quality, pool liveness, exact selected
retrieval, and guarded cleanup for the recorded artifact and policy. They do not establish
bit-deterministic output, a miss-free forward, numerical equivalence to fully resident FP4, or
128K throughput. Second-order residue remains explicit.

Host topology, opaque runtime IDs, host-specific paths, host- and address-specific endpoint
details, checkpoint/pack
fingerprints, launcher identity, and response IDs remain in private source receipts. Public run
labels are descriptive. Verify the final public tree from this directory with:

```bash
shasum -a 256 -c SHA256SUMS
```
