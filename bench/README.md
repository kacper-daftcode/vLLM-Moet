# Bench — the release performance process

Structured, repeatable benchmarking for every supported model/hardware config.
Replaces free-text README numbers: **recipes are code, results are data, the
README table is rendered from them** — and CI fails if they drift apart.

## Layout

```
bench/
  matrix.yaml          # the release matrix: which recipes a release must cover
  recipes/<model>/<config>.yaml   # one file per supported (model, hardware, knobs) combo
  boxes/<id>.yaml      # per-host quirks (env, paths, runtime) — recipes stay portable
  suites/{standard,quick}.yaml    # what gets measured and how many times
  runner/
    bench.py           # run one recipe or the whole matrix
    sweep.py           # empirical knob search ("find the optimal setup")
    render.py          # results -> README block + docs/benchmarks/<release>.md
    lint.py            # schema checks + release-gate checks (CI)
  results/<release>/<box>/<recipe>.json   # committed, append-only history
```

Concepts:

- **Recipe** — the "receptura": the exact env knobs + `vllm serve` args for one
  supported configuration, plus its hardware requirements, the probe parameters
  (needle sizes, batch levels) and optional tuning axes. A recipe is the thing
  we ship to users; the benchmark proves it.
- **Box** — a physical host. Holds host-specific env (`NCCL_P2P_DISABLE=1`,
  CUDA paths for venv runs), local model paths, and the runtime
  (`venv` or `docker`). Recipes never contain host paths.
  - `runtime: docker` benches through the **recipes image**
    (`Dockerfile.recipes`) — the exact artifact customers run: the runner
    `docker run`s the image with local weights mounted over `/models` (the
    launcher's download no-ops) and the checkout's recipe YAMLs mounted over
    the baked-in copies (recipe edits and sweeps without a rebuild). Prefer
    it for release numbers.
  - `runtime: venv` drives `vllm serve` from the box's venv — the dev loop.
- **Suite** — the probe list: single-stream decode, batched decode, cold
  prefill, needle retrieval, arithmetic, coherence. `standard` for releases,
  `quick` for smoke tests. Recipes override per-probe params via `suite_params`.
- **Release** — an identifier (use the git tag being cut). All results for a
  release live under `results/<release>/`, keyed by box.
- **Result** — one JSON per (release, box, recipe): environment fingerprint
  (GPU/driver/package versions/git SHAs/patch hash), the exact serve command,
  load time, and every probe's raw samples + aggregates. `provenance: live`
  for harness runs; `imported` for pre-harness numbers carried over.

## The release process

1. **Freeze the candidate.** Pick the release id (the tag you are about to
   cut, e.g. `v2026.07.15`). Set it as `current_release` in `matrix.yaml`.
2. **Stand up the test environment** on each bench box:
   - docker (preferred):
     ```bash
     DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-v024 -t vllm-moet-sm120:v024 .
     DOCKER_BUILDKIT=1 docker build -f Dockerfile.recipes   -t vllm-moet-recipes:v024 .
     ```
     and point the box file's `docker_image` at the recipes image.
   - venv: apply `patch/` on the v0.24.0 tag per `docs/v024-port.md`.
   The runner records what it actually ran against (SHAs, versions, patch
   hash, image id), so a mismatched env is visible in the result, not silent.
3. **Run the matrix** on every box you have hardware for:
   ```bash
   python3 bench/runner/bench.py matrix --box bench/boxes/<box>.yaml --release <release>
   ```
   Incompatible recipes (not enough GPUs, model not on disk) are skipped with
   a reason. Failures write a `status: failed` result with the log tail —
   commit those too; a release that can't serve a config should say so.
4. **Re-tune what the matrix flags.** Entries with `retune: true` should get a
   sweep before their headline run:
   ```bash
   python3 bench/runner/sweep.py --recipe <model>/<config> --box bench/boxes/<box>.yaml --release <release>
   ```
   The sweep boots the server once per restart-group, scores each combo on the
   recipe's `tuning.objective` subject to `tuning.constraints`, writes the full
   sweep data under `results/<release>/<box>/sweeps/`, and prints the winning
   knobs. **A human applies the winner to the recipe and commits it** — the
   recipe diff *is* the documented decision.
5. **Render.**
   ```bash
   python3 bench/runner/render.py
   ```
   Updates the generated block in `README.md` and writes
   `docs/benchmarks/<release>.md` (full detail: serve commands, fingerprints,
   raw samples, regressions vs `previous_release`).
6. **Open the release PR** with: recipe changes, `results/<release>/`, the
   rendered README block and report. CI (`bench-lint`) validates schemas,
   re-renders, and fails on drift. `lint.py --release <release>` additionally
   fails if any `blocking: true` matrix entry has no result.
7. Tag.

## Commands

```bash
python3 bench/runner/bench.py list --box bench/boxes/<box>.yaml   # recipes + compatibility
python3 bench/runner/bench.py run --recipe kimi-k2.7-code-nvfp4/pro6000x4-tp4-256k \
    --box bench/boxes/runpod-4xpro6000.yaml --release v2026.07.15 [--suite quick] [--gpus 2,3]
python3 bench/runner/bench.py matrix --box ... --release ... [--only-blocking]
python3 bench/runner/sweep.py --recipe ... --box ... --release ...
python3 bench/runner/render.py [--check] [--release <id>]
python3 bench/runner/lint.py [--release <id>]
```

Only dependency beyond the stdlib is PyYAML (present in the serving venv; CI
installs it).

## Methodology (what the numbers mean)

- **decode** — short prompt, `ignore_eos`, greedy; 1 warmup + N runs; median,
  min/max spread, and distinct-output count (the FP4 delta tier makes greedy
  non-deterministic by design — the spread and distinct counts make that
  visible instead of hiding it).
- **batch_decode** — N concurrent distinct prompts of identical length,
  384 tok each; aggregate tok/s over the window from first send to last done.
- **prefill** — unique random prompts (defeats the prefix cache), `max_tokens=1`,
  tok/s from *measured* `prompt_tokens`; median of N.
- **needle** — passphrase at depth 0.5 inside random filler, thinking off;
  PASS = exact secret in the reply. Sizes are per-recipe (`suite_params`).
- **arithmetic / coherence** — the fixed 5-question and 12-prompt sets from
  `docs/quality.md`; coherence additionally gets a cheap automatic
  degeneration check, and all texts land in the result JSON for review.
- **spec decode** — acceptance length / draft-acceptance are scraped from the
  server log right after the decode probe. When A/B-ing speculative variants,
  bench with `VLLM_MOE_W2_DELTA_GB=0` (see `docs/kimi-k27-code.md`:
  the delta tier's runtime precision changes perturb acceptance).
- Cache-tier configs (BASE cache) converge over traffic: recipes set
  `warmup.decode_requests` so the pool is warm before measurement, and pool
  hit-rate lines from the log are attached to the result.

## Quality process (parity vs native)

Perf tells you the config is fast; the **quality suite** proves it serves
the model's full capability. The KPI is **parity with the native serving
path** (stock `VLLM_MOE_W2=0` on the same checkpoint), not absolute scores:
paired accuracy flips + McNemar p + completion-token inflation. Token
inflation is a first-class damage signal — the +8–11% inflation this
process was born from was a real quality bug (2-bit prefill KV) that
accuracy alone did not resolve (see `internal/PREFILL_KV_INFLATION_
FINDINGS.md` history in the repo notes).

Pieces:

- `bench/suites/quality.yaml` — GSM8K-200, GPQA-diamond (non-think) and
  GPQA-diamond THINK (official sampling, `request_overrides` carries the
  chat-template kwargs). Probes are keyed by `id`; recipes override via
  `suite_params`.
- `bench/baselines/` — committed NATIVE reference results + `registry.yaml`
  (checkpoint, mode, hardware, provenance). Re-measure on checkpoint
  revision or native-path changes; the eval tool fails loudly on dataset
  hash mismatch.
- The probe shells out to **llm-inference-bench** (external checkout; path
  from the box yaml `quality_tool` or `LLM_BENCH`; its git SHA lands in the
  result). Raw tool JSONs are committed as artifacts next to the result
  (`results/<release>/<box>/artifacts/`) so flips stay reviewable per item.
- **Quality releases have their own cadence** (`quality_release` in
  `matrix.yaml`): a full quality campaign is hours of GPU (the think probe
  alone is ~7 h on 2x PRO 6000), so it runs when the serving numerics
  change, not per perf release. The README "Quality vs native" table and
  `docs/benchmarks/<quality_release>.md` render from it; `render.py
  --check` guards drift like the perf table.

Running it:

```bash
python3 bench/runner/bench.py run --recipe deepseek-v4-flash/pro6000x2-tp2-maxq \
    --box bench/boxes/rtx-pro6000x4.yaml --release v2026.07.17-quality --suite quality
```

## Regression policy

`render.py` compares each (recipe, box) against `previous_release` in
`matrix.yaml`: decode deltas beyond `regression_threshold_pct` are flagged
`⚠` in the README table and listed in the release report. A flagged release
either gets a fix, a sweep that re-tunes the recipe, or an explicit note in
the recipe's `notes:` — never a silently worse number.

## Shipping recipes to users (`Dockerfile.recipes`)

Recipes are not only bench inputs — they are the deliverable. The recipes
image bakes `bench/recipes/` + `bench/models.yaml` + `docker/serve_recipe.py`
on top of the serving image; a user picks a recipe id and the launcher
downloads the checkpoint from HuggingFace into the `/models` volume (if
missing), applies the recipe's knobs, and execs `vllm serve` with exactly the
benchmarked flags:

```bash
docker run --rm --gpus all --network host --ipc host --shm-size 64g \
  -v /srv/models:/models -e HF_TOKEN=... \
  vllm-moet-recipes:v024  glm-5.2-nvfp4/pro6000x4-tp4-mtp
```

`--list` enumerates recipes, `<recipe> --print` shows what would run,
container env overrides recipe knobs (`-e VLLM_MOE_W2_DELTA_GB=0`), host
quirks pass through (`-e NCCL_P2P_DISABLE=1`), `-v ...:/planes -e
PLANES_CACHE=/planes` enables the planes cache, and args after `--` reach
vllm serve. Because the bench's docker runtime drives this same image and
entrypoint, the published numbers describe the exact thing users launch.

## Adding a model / config

1. Write `recipes/<model>/<config>.yaml` (copy a neighbour; keep JSON args
   space-free so they stay single shell tokens).
2. Register the checkpoint in `models.yaml` (name → `hf_repo`) — lint
   enforces this — and add local paths to the boxes that have the weights.
3. Add a `matrix.yaml` entry (`blocking: true` once the config is a shipping
   claim).
4. `bench.py run` it, then `render.py`.
