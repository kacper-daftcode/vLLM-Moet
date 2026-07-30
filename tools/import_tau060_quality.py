#!/usr/bin/env python3
"""One-shot import of the 2026-07-30 maxq tau-0.60 certification into
bench/results/v2026.07.30-quality/ as a schema-2 `imported` quality result.

The raw driver JSONs are annotated and compacted through the SAME code path
the harness quality probe uses (paired_token_stats + harness_invocation
binding + the exact compact vs_baseline structure), so the schema-2 strict
validator re-derives identical values. Provenance stays `imported` because
the serve was launched by hand (session I's sweep server), not by bench.py;
the env deltas vs the recipe are documented in `source`.

Runtime pins (byte-verified): image vllm-moet-sm120:v024-db13b2e9c =
vllm db13b2e9cd94 + patch sha256 234b97c9f715... — identical to the
v2026.07.27-quality manifest. Idempotent via reserve_result."""
import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench",
                                "runner"))
import common      # noqa: E402
import integrity   # noqa: E402
import probes as probes_mod  # noqa: E402

SRC = "/root/logs/bench/i-audit"
RELEASE = "v2026.07.30-quality"
BOX_ID = "rtx-pro6000x8"
RECIPE = "deepseek-v4-flash/pro6000x2-tp2-maxq"
MODEL = "DeepSeek-V4-Flash"


def build_probe(src_path, configured, recipe, registry, adir, key):
    """Annotate the raw tool JSON and compact it exactly like the harness
    quality probe does (bench/runner/probes.py), then stamp status."""
    with open(src_path) as f:
        data = json.load(f)
    meta = data.get("metadata") or {}
    acc = data.get("accuracy") or {}
    profile = configured["profile"]
    runs = configured.get("runs", 200)
    concurrency = configured.get("concurrency", 2)
    max_tokens = configured.get("max_tokens", 6000)
    baseline = configured.get("baseline")
    profile_rows = [r for r in data.get("runs", [])
                    if r.get("phase") == "profile"]
    assert meta.get("interrupted") is False
    assert meta.get("test_profile") == profile
    assert meta.get("requested_runs") == runs
    assert meta.get("fixed_concurrency") == concurrency
    assert meta.get("max_tokens") == max_tokens
    assert acc.get("scored") == runs and len(profile_rows) == runs
    assert not any(r.get("error") or r.get("cancelled")
                   for r in profile_rows)
    cmp_ = data.get("comparison") or {}
    assert cmp_.get("paired_items") == runs
    assert cmp_.get("profile_match") and cmp_.get("dataset_match")

    registry_spec = common.load_baseline_registry()[baseline]
    baseline_file = common.baseline_path(baseline)
    with open(baseline_file) as f:
        baseline_data = json.load(f)
    baseline_sha256 = registry_spec["artifact_sha256"]
    assert integrity.sha256_file(baseline_file) == baseline_sha256

    data["comparison"]["paired_token_robust"] = probes_mod.paired_token_stats(
        baseline_data, data)
    model_revision = registry[MODEL]["revision"]
    serve_args_sha256 = integrity.canonical_sha256(
        recipe.get("serve_args", []))
    box = common.load_box(os.path.join(common.BENCH_DIR, "boxes",
                                       BOX_ID + ".yaml"))
    tool_sha256 = box["quality_tool_sha256"]
    assert integrity.sha256_file(box["quality_tool"]) == tool_sha256
    data["harness_invocation"] = {
        "schema": 1,
        "candidate_model": MODEL,
        "model_revision": model_revision,
        "serve_args_sha256": serve_args_sha256,
        "served_model": recipe["served_name"],
        "context": recipe.get("context"),
        "profile": profile,
        "runs": runs,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "request_overrides": configured.get("request_overrides") or {},
        "baseline": baseline,
        "baseline_artifact_sha256": baseline_sha256,
        "tool_sha256": tool_sha256,
    }

    tag = f"{RECIPE.replace('/', '__')}__{key}"
    out_json = os.path.join(adir, f"quality__{tag}.json")
    fd, tmp = tempfile.mkstemp(dir=adir)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_json)
    artifact_sha = integrity.sha256_file(out_json)

    toks = sorted(r.get("completion_tokens") or 0 for r in profile_rows)
    res = {
        "profile": profile, "runs": runs, "concurrency": concurrency,
        "max_tokens": max_tokens,
        "request_overrides": configured.get("request_overrides") or {},
        "accuracy_pct": round(100.0 * acc.get("accuracy", 0.0), 2),
        "correct": acc.get("correct"), "of": acc.get("scored"),
        "truncated_no_answer": acc.get("truncated_no_answer"),
        "hit_max_tokens": acc.get("hit_max_tokens"),
        "tokens_avg": round(sum(toks) / len(toks), 1) if toks else None,
        "tokens_p50": toks[len(toks) // 2] if toks else None,
        "tokens_p90": toks[int(0.9 * len(toks))] if toks else None,
        "tool_sha": None,
        "tool_sha256": tool_sha256,
        "duration_s": None,
        "artifact": os.path.basename(out_json),
        "artifact_sha256": artifact_sha,
        "dataset_sha256": meta.get("dataset_sha256"),
        "historical_baseline": False,
        "baseline_artifact_sha256": baseline_sha256,
        "baseline": baseline,
    }
    ct = cmp_.get("completion_tokens") or {}
    bm, cm = ct.get("baseline_mean"), ct.get("candidate_mean")
    bp50, cp50 = ct.get("baseline_p50"), ct.get("candidate_p50")
    bp90, cp90 = ct.get("baseline_p90"), ct.get("candidate_p90")
    truncated = cmp_.get("truncated_no_answer") or {}
    hit_max = cmp_.get("hit_max_tokens") or {}
    robust = data["comparison"]["paired_token_robust"]

    def pct_delta(candidate, reference):
        return (round((candidate - reference) / reference * 100, 1)
                if reference and candidate is not None else None)

    res["vs_baseline"] = {
        "acc_delta_pp": round(cmp_["delta_pp"], 2)
        if cmp_.get("delta_pp") is not None else None,
        "flips_only_baseline": cmp_.get("flips_baseline_only_correct"),
        "flips_only_candidate": cmp_.get("flips_candidate_only_correct"),
        "mcnemar_p": cmp_.get("mcnemar_exact_p"),
        "token_inflation_pct": pct_delta(cm, bm),
        "token_p50_delta_pct": pct_delta(cp50, bp50),
        "token_p90_delta_pct": pct_delta(cp90, bp90),
        "token_ratio_median_delta_pct": pct_delta(
            robust.get("ratio_median"), 1.0),
        "token_ratio_geomean_delta_pct": pct_delta(
            robust.get("ratio_geomean"), 1.0),
        "token_sign_p": robust.get("sign_p"),
        "token_longer": robust.get("longer"),
        "token_shorter": robust.get("shorter"),
        "baseline_runaways": robust.get("baseline_runaways"),
        "candidate_runaways": robust.get("candidate_runaways"),
        "extra_truncated_no_answer": (
            truncated.get("candidate", 0) - truncated.get("baseline", 0)),
        "extra_hit_max_tokens": (
            hit_max.get("candidate", 0) - hit_max.get("baseline", 0)),
    }
    ok = probes_mod.probe_succeeded("quality", res, configured)
    res["status"] = "ok" if ok else "failed"
    assert ok, f"probe {key} fails its own acceptance predicate"
    return res


def main():
    recipe = common.load_recipe(RECIPE)
    suite = common.load_suite("quality")
    probes_cfg = common.merge_suite(suite, recipe)
    registry = common.load_model_registry()
    box = common.load_box(os.path.join(common.BENCH_DIR, "boxes",
                                       BOX_ID + ".yaml"))
    bindings = integrity.result_bindings(
        recipe, "quality", probes_cfg, registry, release=RELEASE, box=box)
    bindings["model_manifests"] = {
        MODEL:
            "72248d76cccb7c8b2d5f05ba8206ba9345337c8af9d3b58ac43a672a5cb3d3f5",
    }

    adir = common.artifacts_dir(RELEASE, BOX_ID)
    by_key = {p.get("id") or p["kind"]: p for p in probes_cfg}
    probes = {
        "gsm8k": build_probe(os.path.join(SRC, "i-maxq-tau060-gsm8k200.json"),
                             by_key["gsm8k"], recipe, registry, adir,
                             "gsm8k"),
        "gpqa": build_probe(os.path.join(SRC, "i-maxq-tau060-gpqa198.json"),
                            by_key["gpqa"], recipe, registry, adir, "gpqa"),
        "gpqa-think": {"status": "disabled"},
    }

    result = {
        "schema": 2,
        "complete": True,
        "release": RELEASE, "box": BOX_ID, "recipe": RECIPE,
        "suite": "quality", "provenance": "imported",
        "source": "2026-07-30 tau-0.60 certification (session I): live TAU "
                  "sweep, then full paired runs at the recipe geometry "
                  "(131k window, C=2) on GPUs 0+1 of the upgraded "
                  "8xPRO6000 box; driver llm-inference-bench with "
                  "--compare-baseline against the committed v2 native "
                  "baselines. Serve env == recipe except tau via "
                  "GATE_TAU_FILE (content 0.60) and observability knobs "
                  "(KPI_EVERY=100, GATE_TRACE=1). THINK not re-measured: "
                  "advisory under the 2026-07-28 outcome criterion. Raw "
                  "artifacts + SHA256SUMS: /root/logs/bench/i-audit/.",
        "runtime": {
            "docker_image": "vllm-moet-sm120:v024-db13b2e9c",
            "vllm_sha": "db13b2e9cd942cf015f3137c8b51644bbe67fb06",
            "patch_sha256": "234b97c9f7153f9d1eb0964a4d9b0271dcbe31bbfb"
                            "6cad2e29d9f7a686fa4fcb",
            "note": "pin-identical to the v2026.07.27-quality manifest; "
                    "the unified-repair commit 76c9997e5 is inert on this "
                    "resident cell (no base tier).",
        },
        "summary": recipe.get("summary", ""),
        "context": recipe.get("context"),
        "gpus_used": [0, 1],
        "started": "2026-07-30T05:17:48+0000",
        "finished": "2026-07-30T05:34:00+0000",
        "status": "ok",
        "serve_env": {k: str(v) for k, v in (recipe.get("env") or {}).items()},
        "inputs": bindings,
        "probes": probes,
        "notes": "decode C=1 sweep on this serve: tau 1.0 = 116.9 tok/s, "
                 "0.70 = 140.3, 0.60 = 149.6, gate-off = 150.5 "
                 "(quality-excluded). GPQA aggregate 61.6 tok/s C2, wall "
                 "13 min. Operational audit (B, 2026-07-29): needle PASS "
                 "at 130,535 prompt +512 output; 57.2 tok/s C=1 sustained "
                 "on the pre-upgrade box state.",
    }
    path = common.result_path(RELEASE, BOX_ID, RECIPE)
    common.reserve_result(path)
    common.write_result(path, result)
    print(f"imported -> {path}")


if __name__ == "__main__":
    main()
