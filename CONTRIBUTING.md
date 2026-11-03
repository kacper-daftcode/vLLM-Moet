# Contributing

## Which line is it?

| line | status | where a change goes |
|---|---|---|
| DeepSeek-V4.1-Flash on 4× RTX PRO 6000 | active | `tools/dsv41_sm120/` + a step in `Dockerfile.sm120-dsv41-nightly` |
| MoET hybrid on 4× RTX 5090 | merging | `tools/dsv41_5090/` (branch `moet-hybrid` until merged) |
| 2-bit experts on vLLM v0.24.0 | frozen | the fork branch `moet-v0.24.0` of [`kacper-daftcode/vllm-tmp`](https://github.com/kacper-daftcode/vllm-tmp) — **not** `patch/` |
| Qwen3.8-Flash-Next-FP8 | as built | `tools/qwen38_sm120/` + `Dockerfile.sm120-qwen38` |

The frozen line takes fixes, not features: open the pull request against the fork branch; the patch
file in `patch/` is generated from it (`tools/check_patch_files.py --update`) and a PR that edits
the patch directly cannot be merged — the next regeneration would erase it.

## A change on the DeepSeek-V4.1 line

1. A patcher anchored on the installed file's text (it must refuse any other version), a kernel as a
   `torch` CUDA extension compiled at image build, and an op-level test against the path it replaces.
   Bit-identical (`torch.equal`) unless the change is a documented, quality-gated change of the arithmetic.
2. A numbered Dockerfile step with a build argument (`VLLM_MOET_<STEP>=1`), a `com.vllm-moet.*` label,
   and a runtime switch `VLLM_MOET_*` that defaults to off until the change is validated in production.
3. Measurements at three levels — the op-level test (cold L2), a profile of the served step or a fresh
   prefill, and end to end (fresh prefill 19K / 163K, the Fraser bench as a median of 40 reps). Long-prompt
   bit comparisons need `VLLM_MOET_DET_TOPK=1` in both arms.
4. `python3 tools/dsv41_sm120/cubit_gate.py IMAGE` on the built image (every cubin round-trips through cubit).
5. The step's row in `docs/dsv41-sm120.md` (switches) and `tools/dsv41_sm120/README.md` (patcher, kernel, test);
   the reasoning in `docs/dsv41-sm120-internals.md`; a line in `CHANGELOG.md` when an image ships.

## Documentation rules

Present tense; no dates outside `CHANGELOG.md` and `docs/benchmarks/`; no host paths; one place per
fact. Measurements worth keeping are records in `docs/benchmarks/<topic>-<date>.md`. Session notes
and experiment logs do not belong in the repository.

## Before you push

```bash
python3 tools/check_patch_files.py                    # patch <-> FILES.txt <-> SOURCE.txt
python3 bench/runner/lint.py && python3 bench/runner/lint.py --release current
python3 -m unittest discover -s bench/tests -p 'test_*.py'
python3 bench/runner/render.py --check               # generated tables == committed results
```

CI (`bench-lint`) runs the same. Commit subjects are declarative and state what ships; bodies carry
the evidence (numbers, test verdicts). `AGENTS.md` has the full contract for agents working in a
shared checkout.

## Reporting

Bugs: the issue template asks for the image tag, the line, the GPUs, the launcher command and the log.
Validation reports on other hardware or models go to
[Discussions](https://github.com/kacper-daftcode/vLLM-Moet/discussions).
