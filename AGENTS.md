# AGENTS.md — the commit contract

Multiple agents and humans work on this project concurrently, often in the
**same checkout**. Work has been lost here three separate times by treating
the generated patch as a source file. Read this before your first commit;
every rule below traces back to a real incident.

## The iron rule

`patch/vllm-moet-v0.24.0.patch` and
`patch/vllm-moet-v0.25.0.patch` are **generated artifacts**. Each is the
canonicalized `git diff` from its official release tag to the matching
production-fork ship branch. They are **never edited by hand**, never patched
incrementally, and never regenerated from anything but that branch. The only
sanctioned commands are:

```bash
python3 tools/check_patch_files.py --version 0.24.0 --update
python3 tools/check_patch_files.py --version 0.25.0 --update
```

If your change touches vLLM code, it goes to the **fork branch first**; the
patch is derived afterwards. A change that exists only inside the patch file
WILL be erased by somebody's next regeneration.

## The two repos on this box

| repo | path | branch | role |
|---|---|---|---|
| **vLLM-Moet** (this one, public) | `/workspace/vllm-moet` | `main` | publication: generated patch, kernels + cubins, bench system, docs |
| **vllm v0.24 fork clone** | `/workspace/vllm-v0.24.0` | `moet-v0.24.0` | legacy production source for the v0.24 overlay; remotes remain as documented in that clone |
| **vllm v0.25 fork clone** | `/workspace/vllm-v0.25.0` | `moet-v0.25.0` | production source for the v0.25 overlay; `origin` = `vllm-project/vllm`, `fork` = `OmarB97/vllm` |

`/root/workspace` is a symlink to `/workspace`. Upstream-PR branches and
experiments live in worktrees off the same clone
(`git -C /workspace/vllm-v0.24.0 worktree list`).

For v0.25, the production fork branch and its recorded SHA gate rollout; an
optional PR to `kacper-daftcode/vllm` or `vllm-project/vllm` never does. This
settlement does not migrate or redefine the legacy v0.24 remote contract.

`moet-v0.24.0` and `moet-v0.25.0` are ship lineages: everything committed
there is meant to ship in that release overlay. Park half-done work on a side
branch or worktree.

## Where a change goes

| change | commit where | must also update |
|---|---|---|
| vLLM runtime code (`moe_w2_*`, loaders, runner, attention, …) | matching fork branch `moet-v0.24.0` or `moet-v0.25.0` | regen that release patch here — procedure below |
| SASS kernels / cubins | `kernels/` | a `kernels/MANIFEST.md` row (generator + validation status) |
| serve configs | `bench/recipes/` | `bench/models.yaml`, `bench/matrix.yaml`; run `bench/runner/lint.py` |
| bench results | `bench/results/<release>/` | `bench/runner/render.py` — the README table and per-release report are **generated**; never hand-edit the marked README block |
| docs | `docs/`, README outside the generated block | — |
| session notes / handoffs / experiment scraps | `internal/` (gitignored, stays local) | never into `docs/` |

Never commit: wheels (`patch/*.whl` is ignored), checkpoints, expert packs,
smoke results (`bench/results/smoke/`).

## Shipping a vLLM code change — the procedure

1. **Commit and push on the matching production fork branch**
   (`moet-v0.24.0` or `moet-v0.25.0`). If the remote may have moved, fetch
   and merge first. Strict releases refuse to regenerate from unpublished
   source.
2. **Regenerate** from this repo:

   ```bash
   VLLM_MOET_FORK=/workspace/vllm-v0.24.0 \
     python3 tools/check_patch_files.py --version 0.24.0 --update
   VLLM_MOET_FORK=/workspace/vllm-v0.25.0 \
     python3 tools/check_patch_files.py --version 0.25.0 --update
   ```

   This rewrites the patch from the published branch tip and updates that
   release's committed file list and source fingerprint
   (`FILES.txt`+`SOURCE.txt` or
   `FILES-v025.txt`+`SOURCE-v025.txt`). It refuses to move a source
   fingerprint backwards, so regeneration cannot roll back shipped work.
3. **Review `git diff patch/`.** An entry *vanishing* from `FILES.txt`
   means the patch carried work that never reached the fork branch —
   someone skipped step 1. **Stop and merge that work into the branch**;
   never ship the loss, never `--update` a second time to silence it.
4. **Validate what the change class requires.** Byte-exact generation from
   the branch already guarantees `git apply --check` on the tag. For
   GPU-relevant changes run the relevant suites
   (`tools/test_moe_w2_forward.py`, `tools/test_store_backends.py`, the
   three-tier tests) and put the results in the commit message, as the
   existing history does.
5. **Commit both repos, cross-referenced.** The vLLM-Moet commit that ships
   a regen names the fork SHA, following the established style:

   > `Three-tier starvation fix ships: step-scoped seen windows (vllm 9736e4d34)`

6. **Publish source first, then distribution.** The matching production fork
   branch must contain the recorded source SHA before the distribution guard
   can pass. Push the distribution branch only after every guard is green.
   Optional contribution PRs are follow-up evidence, not rollout gates.

## Concurrency — several agents, one checkout

- `git status` before you start. Dirty paths you did not create belong to
  another live session: **leave them alone**. No `git add -A`, no
  `git commit -a`, no `git stash`, no `git checkout --` / `git reset` over
  someone else's files, ever.
- Stage **explicit paths only**: `git add <file> <file> …`.
- `main`, `moet-v0.24.0`, and `moet-v0.25.0` are shared trunks: no
  amending commits you did not just create, no rebase, no force-push, no
  history rewrite.
- Each release's patch, FILES, and SOURCE artifacts change **only** via the
  matching versioned `--update`. If your commit would touch them another way,
  you are doing something wrong.
- A pre-commit hook in this checkout runs the patch guard whenever `patch/`
  is staged. Do not bypass it with `--no-verify`.
- Commit identity: no global git identity is configured on this box —
  pass the session identity per command, matching the existing history:

  ```bash
  git -c user.name=vllm-moet -c user.email=moet@local commit ...
  ```

  Upstream PRs use the user's public identity + DCO instead — see
  `internal/UPSTREAM_PRS.md`. Never edit git config.

## Pre-push checklist (mirrors CI `bench-lint`)

```bash
python3 tools/check_patch_files.py       # patch <-> FILES.txt <-> SOURCE.txt
python3 tools/check_patch_files.py --version 0.25.0
python3 -m unittest discover -s tests -p "test_check_patch_files.py"
python3 bench/runner/lint.py             # recipes/boxes/suites/results schemas
python3 bench/runner/render.py --check   # README table == committed results
```

Plus `python3 docker/serve_recipe.py <recipe> --print` if you touched
recipes or the launcher.

## The incidents these rules come from

- **Kimi-K2.7 bring-up** — landed in the patch without its commits reaching
  the fork branch; the next regeneration erased it; caught by hand, folded
  back in `81c1b34`.
- **DSpark backport** — `ad7f29a` regenerated from a branch that lacked the
  DSpark line and silently dropped it; repaired by merging DSpark *into the
  generating branch* and regenerating the union (`8aff1b7`).
- **Stream-build hunks** — lost to a merge-side resolution inside the patch
  file; restored by hand in `8f50e57`. Hunk-level losses inside an
  unchanged file set are invisible to `FILES.txt` — that class is what the
  `SOURCE.txt` byte-verification catches.

**A failing guard means work would be lost.** The fix is always to move the
work onto the fork branch — never to edit the patch, `FILES.txt` or
`SOURCE.txt` into agreement.

## Commit style

Match `git log`. Subject: declarative, what ships / what changed, no
prefixes. Body: the why, the evidence (measured numbers, test verdicts),
and for regens the fork SHA. On the fork branch, upstream-style component
prefixes are fine (`DCP DSA indexer: …`).
