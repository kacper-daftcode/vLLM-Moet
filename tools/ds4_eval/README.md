# DS4-W2 reliability evaluation harness

This repository-integrated harness produces fail-closed, provenance-bound
quality and long-context receipts for a running DS4-W2 OpenAI-compatible server.
It never boots, stops, or reconfigures the server itself.

## What changed

- `rescore_robust.py` reads the emitted `ct` field and the canonical
  `completion_tokens` field through one alias boundary. If aliases disagree,
  scoring stops instead of choosing one silently. It also rejects duplicate
  IDs and incomplete row counts.
- `eval_rig.py` emits `completion_tokens` canonically, uses no hidden request
  retries, and writes an immutable run manifest, raw rows, and prewarm receipts.
  Exact clean correctness remains primary; the original lenient answer-anywhere
  diagnostic is retained separately and never overrides a sink.
- Prewarm suite `ds4-w2-prewarm-v4` is a fixed ten-prompt suite at temperature
  `0`, top-p `1`, and seed `20260711`. The eval seed never changes prewarm.
  Every warmup must return a non-empty, correctly terminated `FINAL:` answer,
  or an exact terminal `<answer>...</answer>` wrapper at the response start or
  immediately after DeepSeek's `</think>` boundary, with token usage;
  surrounding or trailing junk fails closed. The first bad receipt aborts the
  run and remains on disk.
- Warm mode requires a measured pool gate. The gate can consume the current
  `[fp4] tick ...` and `[base] KPI ...` log lines, the current flat delta JSON
  dump, a future combined JSON KPI endpoint, or an explicit argv command. The
  same frozen policy must pass both before and after scoring. A relational
  `min_fp4_total_evicted_delta` check proves that a saturated LRU pool kept
  evicting during the scored requests instead of passing on a stale cumulative
  counter.
- Server provenance is mandatory and includes the host boot, exact container,
  image, source, checkpoint, pack, launcher, full runtime argv, W2 environment,
  and key engine settings. The complete source-diff SHA-256 is required. W2
  environment values, structured runtime fields, and corresponding argv values
  must agree; placeholders, missing keys, and contradictions are rejected.

## Verification

```bash
python3 -m unittest discover -s tools/ds4_eval/tests -v
python3 tools/ds4_eval/rescore_robust.py --expected-count 40 \
  tools/ds4_eval/tests/fixtures/raw-rig-32k-base8-delta6-lru-s42.jsonl \
  tools/ds4_eval/tests/fixtures/raw-rig-32k-b8d6-lru-s43.jsonl \
  tools/ds4_eval/tests/fixtures/raw-rig-32k-b8d6-lru-s44.jsonl
```

The corrected tau-0.67 baseline is 5, 7, and 4 sinks for seeds 42, 43, and 44,
respectively: 16/120 (13.3%) across the three runs. Tau 0.75 has only one old
seed-42 result, also 5/40 after correction. See the
[published baseline summary](../../evidence/public/ds4-w2-2026-07-11/baseline/README.md)
for file hashes and sink IDs.

## Running a future evaluation

1. Copy `server-provenance.schema-example.json` and replace every angle-bracket
   value with evidence collected from the exact live container. The example is
   intentionally invalid until completed.
2. Pre-register a pool gate policy. `pool-gate.schema-example.json` shows a
   representative field set. The `min_fp4_total_evicted_delta` value of `16` is
   the recommended minimum live-churn proof for this 40-item P1 regression;
   tune other numeric thresholds from operational limits before seeing the new
   quality score. A future combined JSON endpoint can additionally gate on
   `min_gate_steps`, `min_gate_fire_rate`, and `max_gate_fire_rate`.
3. Supply one live KPI source. A log source is currently the most complete because
   it can contain both `[fp4] tick` occupancy/churn and `[base] KPI` replay and
   residue counters. The flat delta dump supports occupancy/churn only; checks
   for unavailable base metrics correctly fail. A static `--pool-log-file`
   cannot satisfy a nonzero pre/post eviction delta; use a live command, JSON
   file, or endpoint for that policy.
4. Run `eval_rig.py`. It refuses to overwrite artifacts.

```bash
python3 tools/ds4_eval/eval_rig.py \
  --items tools/ds4_eval/items.json \
  --server-provenance /path/to/server-provenance.json \
  --pool-gate-policy /path/to/frozen-pool-policy.json \
  --pool-command-json '["docker","logs","ds4-w2-rig"]' \
  --output-dir /path/to/new-run \
  --run-label p1-candidate-32k-b8d6-lru-tau075-s42 \
  --url http://127.0.0.1:18001/v1/chat/completions \
  --model deepseek-v4-flash-w2 \
  --mode warm \
  --eval-seed 42 \
  --eval-temperature 0.6 \
  --eval-top-p 0.95 \
  --eval-max-tokens 700 \
  --expected-count 40
```

For live collection, `--pool-command-json` accepts a JSON argv array and never
invokes a local shell. `--pool-json-file` and `--pool-kpi-url` accept the current
flat dump or a future object with `fp4`, `base`, and `gate` sections. The policy
file's SHA-256 is recorded before scoring and rechecked afterward.

## Comparability boundary

This correction makes the existing transcripts scoreable and future runs
auditable. It does not prove that the sink detector is a complete semantic
quality metric, nor does it retroactively make the old warmups comparable:
those warmups used eval-seeded temperature-0.6 calls, ignored failures, and did
not record a passing pool-state gate. New results should be compared only when
their manifests show matching checkpoint, pack, launcher, runtime, W2
environment, harness, and item hashes; the expected image/source-patch identity
is the only implementation difference; and both pre- and post-eval pool gates
passed. For the tau-0.75 P1 verdict, collect fresh control and candidate runs for
all three seeds rather than comparing the candidate to the tau-0.67 historical
16/120 aggregate.

## Repository contents

The runnable harness lives under `tools/ds4_eval/`:

- runtime: `eval_rig.py`, `harness.py`, and `rescore_robust.py`;
- contract: this `README.md`, `pool-gate.schema-example.json`, and
  `server-provenance.schema-example.json`;
- regression: `tests/test_harness.py`;
- input: `items.json` is the fixed 40-item reasoning/coding set;
- historical fixtures: the three tau-0.67 raw JSONL files and the separate
  tau-0.75 seed-42 raw JSONL file are under `tools/ds4_eval/tests/fixtures/`;
- evidence: the corrected historical baseline is under
  `evidence/public/ds4-w2-2026-07-11/baseline/`.

The test suite uses only repository-local fixtures. Do not integrate
`.ruff_cache` or `__pycache__`.

## P2 long-context receipts

`context_probe.py` is the fail-closed retrieval half of a P2 verdict. It binds
each run to a validated server-provenance file, requires no MTP and
`--max-num-seqs 1`, verifies the requested window, fp8 KV dtype, base/FP4 pool,
policy, and gate threshold, and refuses to overwrite artifacts. Each requested
prompt length is calibrated with the live server's `/tokenize` endpoint. The
completion's `usage.prompt_tokens` must exactly match that tokenizer receipt,
the finish reason must be `stop`, and the deterministic passphrase must be the
exact terminal answer.

Run the quality harness first on the ready server, then use three needle depths
near the usable edge of the window. This keeps the robust 40-item sink score and
actual long-range retrieval as separate, reviewable claims.

```bash
python3 tools/ds4_eval/context_probe.py \
  --server-provenance /path/to/server.json \
  --output-dir /path/to/new-context-run \
  --run-label p2-128k-context \
  --url http://127.0.0.1:18001/v1/chat/completions \
  --tokenize-url http://127.0.0.1:18001/tokenize \
  --model deepseek-v4-flash-w2 \
  --expected-window 131072 --expected-kv-dtype fp8 \
  --expected-base-gb 8 --expected-delta-gb 6 \
  --expected-policy lru --expected-tau 0.75 \
  --case 120000:0.1 --case 120000:0.5 --case 120000:0.9
```

The result is complete only when the manifest has `context_validated: true`
and all JSONL receipts have `accepted: true`. A 128K boot alone is capacity
evidence, not a context-quality result; a short-prompt 40-item score alone is
decode-quality evidence, not proof that attention retrieves at length.
