# AGENTS.md — the contract for agents and contributors

This repository is a publication: the kernels, patchers, images, launchers and documentation of
several lines of work. More than one agent may edit it at a time. The rules below keep the published
tree correct and readable; the maintainers' operational notes live outside the repository.

## The lines of work and where their code lives

| line | source of truth | how a change ships |
|---|---|---|
| DeepSeek-V4.1-Flash on 4× RTX PRO 6000 | `tools/dsv41_sm120/` + `Dockerfile.sm120-dsv41-nightly` | a numbered Dockerfile step: an anchored patcher, a kernel with its op-level test, a build arg and a `com.vllm-moet.*` label, a runtime switch (`VLLM_MOET_*`, off by default until validated); `cubit_gate.py` on the built image; the step's row in `docs/dsv41-sm120.md` and `tools/dsv41_sm120/README.md` |
| MoET hybrid on 1–4× RTX 5090 | `tools/dsv41_5090/` (index in its README) + the Dockerfile's steps 43 (CPU tier) and 44 (Engram v2), `docker/sm120/run-dsv41-5090.sh` | the same; identity against the all-resident path (`DEBUG_COMPACT=1` in a `tp4_window.sh` boot) is the gate, at every TP and RAM tier a change touches |
| 2-bit experts on vLLM v0.24.0 (frozen) | the fork branch `moet-v0.24.0` of [`kacper-daftcode/vllm-tmp`](https://github.com/kacper-daftcode/vllm-tmp); `patch/` is **generated** from it | the procedure below — never the patch file |
| Qwen3.8-Flash-Next-FP8 | `tools/qwen38_sm120/` + `Dockerfile.sm120-qwen38` | as the first line |
| Ripple-65B Preview (NVFP4) on 4× RTX PRO 6000 | `tools/ripple_sm120/` + `Dockerfile.sm120-ripple` (+ `tools/dsv41_sm120/p2p_ar/`, shared) | as the first line; steps numbered from 0 (the vLLM wheel) |

## The iron rule

`patch/vllm-moet-v0.24.0.patch` is a **generated artifact**: byte-for-byte `git diff v0.24.0
moet-v0.24.0` from the fork branch. It is **never edited by hand**, never patched incrementally,
never regenerated from anything but the fork branch. The only sanctioned way to change it:

```bash
python3 tools/check_patch_files.py --update
```

If your change touches vLLM code of the 2-bit line, it goes to the **fork branch first**; the patch
is derived afterwards. A change that exists only inside the patch file will be erased by the next
regeneration.

## Where a change goes

| change | commit where | must also update |
|---|---|---|
| vLLM runtime code of the 2-bit line | fork branch `moet-v0.24.0` | regenerate the patch here — procedure below |
| a kernel-side change of the other lines | the line's `tools/<line>/` + a Dockerfile step | the switch row in the line's doc, the tools README, `CHANGELOG.md` when an image ships |
| SASS kernels / cubins | `kernels/` | a `kernels/MANIFEST.md` row (generator + validation status) |
| serve configs | `bench/recipes/` | `bench/models.yaml`, `bench/matrix.yaml`; run `bench/runner/lint.py` |
| bench results | `bench/results/<release>/` | `bench/runner/render.py` — the tables in `docs/v024-2bit.md` and the per-release report are **generated**; never hand-edit the marked blocks |
| docs | `docs/` (one doc per line + `docs/benchmarks/` records), `README.md` | — |
| session notes, experiment logs, hand-offs, coordination, deployment specifics | **not in this repository** | — |

## Documentation is reference, not a journal

Public docs describe what the code does, how to run it and what was measured — in the present
tense, with the method next to every number. They do not carry: dates or times of deployments,
decision narratives ("decided", "approved", who asked for what), session letters or names, machine
names, host paths, deployment addresses, references to private notes, or "next session" plans.
A measurement worth keeping is a record in `docs/benchmarks/<topic>-<date>.md`; the reasoning behind
a design is a paragraph in the line's internals doc, not a dated section. One place per fact: a switch
is described in the switch table of its line's doc and nowhere else. The same applies to comments and
docstrings in code and to commit messages.

`python3 tools/docs_lint.py` enforces the mechanical part (host paths, session jargon, deployment
narrative, private references, broken links, `py_compile`, `bash -n`); CI runs it on every push and
it must pass before one. It also holds the form: in `README.md` and `docs/*.md` no table cell longer than
400 characters and no line longer than 1 200, and in `docs/dsv41-sm120.md` the switch table's rows in step
order with a `served` cell that is `on`, `off`, `not built` or a value plus at most one parenthesis of 40
characters — a step's mechanism, its measurements and its image history go to the internals doc, a record
and `CHANGELOG.md`, not into its row.

Never commit: wheels (`patch/*.whl` is ignored), checkpoints, expert packs, smoke results
(`bench/results/smoke/`).

## Shipping a vLLM code change on the 2-bit line

1. **Commit on the fork branch** `moet-v0.24.0`. If the remote may have moved, fetch and merge first —
   the regen tool refuses to run when the local branch is missing pushed commits.
2. **Regenerate** from this repo: `python3 tools/check_patch_files.py --update`. This rewrites the patch
   from the branch tip and updates the two committed fingerprints, `patch/FILES.txt` (file list) and
   `patch/SOURCE.txt` (the fork SHA the patch was generated from). It refuses to move `SOURCE.txt`
   backwards along the branch, so a regeneration can never roll back work that already shipped.
3. **Review `git diff patch/`.** An entry *vanishing* from `FILES.txt` means the patch carried work that
   never reached the fork branch. Stop and merge that work into the branch; never ship the loss, never
   `--update` a second time to silence it.
4. **Validate what the change class requires** (`tools/test_moe_w2_forward.py`,
   `tools/test_store_backends.py`, the three-tier tests for GPU-relevant changes) and put the verdicts in
   the commit body.
5. **Commit both repos, cross-referenced**: the commit that ships a regeneration names the fork SHA.
6. **Push both together** once the pre-push checklist passes — or leave both unpushed.

## Several agents, one checkout

- `git status` before you start. Dirty paths you did not create belong to another live session: leave
  them alone. No `git add -A`, no `git commit -a`, no `git stash`, no `git checkout --` / `git reset`
  over someone else's files.
- Stage **explicit paths only**: `git add <file> <file> …`.
- `main` and `moet-v0.24.0` are shared trunks: no amending commits you did not just create, no rebase,
  no force-push, no history rewrite.
- The `patch/` trio (patch, `FILES.txt`, `SOURCE.txt`) changes **only** via `--update`. The pre-commit
  hook runs the patch guard whenever `patch/` is staged; do not bypass it with `--no-verify`.
- Shared GPUs: re-check `docker ps` and `nvidia-smi` in the same command that launches a container, name
  containers with a per-session prefix, and never remove another session's container.

## Pre-push checklist (mirrors CI)

```bash
python3 tools/check_patch_files.py       # patch <-> FILES.txt <-> SOURCE.txt
python3 bench/runner/lint.py             # recipes/boxes/suites/results schemas
python3 bench/runner/lint.py --release current  # strict live release gate
python3 -m unittest discover -s bench/tests -p 'test_*.py' -v
python3 bench/runner/render.py --check   # docs/v024-2bit.md tables == committed results
python3 tools/docs_lint.py               # public-tree hygiene
```

Plus `python3 docker/serve_recipe.py <recipe> --print` if you touched recipes or the launcher.

## Commit style

Subject ≤ 72 characters, declarative, what changed (`Ripple: two-microbatch decode for TP (step 3)`).
The body carries the why and the evidence — measured numbers with their configuration, test verdicts,
the fork SHA for a regeneration. No timestamps, no deployment narrative, no session letters.
