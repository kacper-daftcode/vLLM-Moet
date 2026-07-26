#!/usr/bin/env python3
"""Structural checks for bench/ (CI + release gate). Stdlib+PyYAML only.

  lint.py                      # recipes, suites, boxes, matrix, all results
  lint.py --release <id>       # + release gate: every blocking matrix entry
                               #   must have a non-failed result for <id>

Exit 1 on any error. Warnings don't fail the lint."""

import argparse
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import common
import integrity
import probes as probes_mod

ERRORS, WARNINGS = [], []


def err(msg):
    ERRORS.append(msg)
    print(f"ERROR: {msg}")


def warn(msg):
    WARNINGS.append(msg)
    print(f"warn:  {msg}")


def lint_recipe(rid):
    try:
        r = common.load_recipe(rid)
    except Exception as e:  # noqa: BLE001
        err(f"recipe {rid}: unloadable ({e})")
        return None
    for field in ("model", "served_name", "summary", "context"):
        if not r.get(field):
            err(f"recipe {rid}: missing `{field}`")
    if " - " not in (r.get("summary") or ""):
        warn(f"recipe {rid}: summary should be 'hardware - description' "
             "(the README table splits on ' - ')")
    if not r["serve_args"]:
        err(f"recipe {rid}: empty serve_args")
    for a in r["serve_args"]:
        if not isinstance(a, str) or not a.startswith("--"):
            err(f"recipe {rid}: serve_arg {a!r} must be a '--flag[ value]' string")
            continue
        _flag, _, rest = a.partition(" ")
        if rest.startswith("{") and "{model" not in rest and "{planes" not in rest:
            try:
                json.loads(rest)
            except ValueError:
                err(f"recipe {rid}: unparseable JSON in {a!r}")
    if not r["requires"].get("gpus"):
        err(f"recipe {rid}: requires.gpus missing")
    t = r.get("tuning")
    if t:
        if not t.get("objective"):
            err(f"recipe {rid}: tuning without objective")
        for ax in t.get("axes", []):
            if ax.get("kind") not in ("env", "serve_json"):
                err(f"recipe {rid}: unknown tuning axis kind {ax.get('kind')!r}")
            if ax.get("restart") is False and ax.get("kind") == "env" \
                    and not ax.get("file_env"):
                warn(f"recipe {rid}: axis {ax.get('name')} restart:false "
                     "needs file_env to actually avoid reboots")
            if not ax.get("values"):
                err(f"recipe {rid}: tuning axis without values")
    return r


def lint_results(release):
    n = 0
    historical = set(common.load_matrix().get("historical_releases", []))
    for box_id, result_path, res in common.iter_result_files(release):
        n += 1
        rid = res.get("recipe", "?")
        where = f"results/{release}/{box_id}/{rid}"
        if res.get("release") != release:
            err(f"{where}: JSON release does not match directory")
        if res.get("box") != box_id:
            err(f"{where}: JSON box does not match directory")
        expected_name = str(rid).replace("/", "__") + ".json"
        if os.path.basename(result_path) != expected_name:
            err(f"{where}: filename does not match recipe id")
        schema = res.get("schema")
        if schema not in (1, 2):
            err(f"{where}: unsupported schema {schema!r}")
        if schema == 1 and release not in historical:
            warn(f"{where}: schema-1 result cannot satisfy a fresh release "
                 "gate unless the release is explicitly historical")
        if res.get("status") not in ("ok", "partial", "failed"):
            err(f"{where}: bad status {res.get('status')!r}")
        if res.get("provenance") not in ("live", "imported"):
            err(f"{where}: bad provenance {res.get('provenance')!r}")
        if not os.path.exists(common.recipe_path(rid)):
            (warn if release in historical else err)(
                f"{where}: references retired/unknown recipe {rid}")
        if res.get("provenance") == "live":
            required = ["serve_cmd", "env_fingerprint"]
            if res.get("status") != "failed":
                required.append("load_time_s")
            for field in required:
                if not res.get(field):
                    (err if schema == 2 else warn)(
                        f"{where}: live result missing `{field}`")
        if res.get("status") == "ok" and not res.get("probes"):
            err(f"{where}: ok result with no probes")
        if schema == 2:
            if res.get("complete") is not True:
                err(f"{where}: schema-2 result is not complete")
            try:
                inputs = res.get("inputs") or {}
                for message in integrity.self_binding_errors(inputs):
                    err(f"{where}: {message}")
                if inputs.get("release") != release:
                    err(f"{where}: input binding release mismatch")
                if inputs.get("box") != box_id:
                    err(f"{where}: input binding box mismatch")
                probes = (inputs.get("suite_snapshot") or {}).get(
                    "probes", [])
                got_probes = res.get("probes") or {}
                for probe in probes:
                    key = probe.get("id") or probe["kind"]
                    got = got_probes.get(key)
                    if probe["enabled"]:
                        if not got and res.get("status") != "failed":
                            err(f"{where}: enabled probe {key} missing")
                        elif (got and got.get("status") == "disabled"
                              and res.get("status") != "failed"):
                            err(f"{where}: enabled probe {key} marked disabled")
                        elif (got and res.get("status") == "ok"
                              and got.get("error")):
                            err(f"{where}: enabled probe {key} contains error")
                        elif (got and res.get("status") == "ok"
                              and got.get("status") != "ok"):
                            err(f"{where}: enabled probe {key} status is "
                                f"{got.get('status')!r}, expected 'ok'")
                        if (got and res.get("status") == "ok"
                                and not probes_mod.probe_succeeded(
                                    probe["kind"], got, probe)):
                            err(f"{where}: enabled probe {key} fails "
                                "acceptance predicate")
                        if (got and res.get("status") == "ok"
                                and probe.get("kind") == "quality"
                                and not got.get("artifact")):
                            err(f"{where}: quality probe {key} has no "
                                "raw artifact")
                    elif (res.get("status") != "failed"
                          and got != {"status": "disabled"}):
                        err(f"{where}: disabled probe {key} not recorded "
                            "as disabled")
                for key, probe in got_probes.items():
                    artifact = probe.get("artifact") if isinstance(
                        probe, dict) else None
                    digest = probe.get("artifact_sha256") if isinstance(
                        probe, dict) else None
                    if artifact:
                        path = integrity.artifact_path(
                            result_path, artifact)
                        if not os.path.exists(path):
                            err(f"{where}: probe {key} artifact missing")
                        elif not digest or integrity.sha256_file(path) != digest:
                            err(f"{where}: probe {key} artifact hash mismatch")
                        else:
                            with open(path) as artifact_file:
                                raw = json.load(artifact_file)
                            configured_probe = next(
                                (
                                    configured
                                    for configured in probes
                                    if (configured.get("id")
                                        or configured["kind"]) == key
                                ),
                                {},
                            )
                            expected_runs = configured_probe.get("runs", 200)
                            raw_meta = raw.get("metadata") or {}
                            profile_rows = [
                                row for row in raw.get("runs", [])
                                if row.get("phase") == "profile"
                            ]
                            if (
                                raw_meta.get("interrupted") is not False
                                or raw_meta.get("test_profile")
                                != configured_probe.get("profile")
                                or raw_meta.get("requested_runs")
                                != expected_runs
                                or raw_meta.get("fixed_concurrency")
                                != configured_probe.get("concurrency", 2)
                                or raw_meta.get("max_tokens")
                                != configured_probe.get("max_tokens", 6000)
                                or len(profile_rows) != expected_runs
                                or any(
                                    row.get("error") or row.get("cancelled")
                                    for row in profile_rows)
                            ):
                                err(f"{where}: probe {key} raw artifact "
                                    "metadata/counts are incomplete or "
                                    "mismatched")
                            accuracy = raw.get("accuracy") or {}
                            raw_pct = round(
                                100.0 * accuracy.get("accuracy", 0.0), 2)
                            if (
                                probe.get("accuracy_pct") != raw_pct
                                or probe.get("correct")
                                != accuracy.get("correct")
                                or probe.get("of") != accuracy.get("scored")
                                or accuracy.get("scored") != expected_runs
                                or probe.get("dataset_sha256")
                                != raw_meta.get("dataset_sha256")
                            ):
                                err(f"{where}: probe {key} compact accuracy "
                                    "does not match raw artifact")
                            tokens = sorted(
                                row.get("completion_tokens") or 0
                                for row in raw.get("runs", [])
                                if row.get("phase") == "profile")
                            token_checks = {
                                "tokens_avg": (
                                    round(sum(tokens) / len(tokens), 1)
                                    if tokens else None),
                                "tokens_p50": (
                                    tokens[len(tokens) // 2]
                                    if tokens else None),
                                "tokens_p90": (
                                    tokens[int(0.9 * len(tokens))]
                                    if tokens else None),
                            }
                            if any(
                                probe.get(field) != value
                                for field, value in token_checks.items()
                            ):
                                err(f"{where}: probe {key} compact token "
                                    "statistics do not match raw artifact")
                            comparison = raw.get("comparison") or {}
                            if (configured_probe.get("baseline")
                                    and not comparison):
                                err(f"{where}: probe {key} raw artifact "
                                    "has no required baseline comparison")
                            if configured_probe.get("baseline") and comparison:
                                ct = comparison.get(
                                    "completion_tokens") or {}
                                bm = ct.get("baseline_mean")
                                cm = ct.get("candidate_mean")
                                bp50 = ct.get("baseline_p50")
                                cp50 = ct.get("candidate_p50")
                                bp90 = ct.get("baseline_p90")
                                cp90 = ct.get("candidate_p90")
                                truncated = comparison.get(
                                    "truncated_no_answer") or {}
                                hit_max = comparison.get(
                                    "hit_max_tokens") or {}
                                baseline_id = configured_probe.get(
                                    "baseline")
                                with open(common.baseline_path(
                                        baseline_id)) as baseline_file:
                                    baseline_raw = json.load(baseline_file)
                                expected_robust = (
                                    probes_mod.paired_token_stats(
                                        baseline_raw, raw))
                                if comparison.get(
                                        "paired_token_robust") != (
                                            expected_robust):
                                    err(f"{where}: probe {key} paired robust "
                                        "token statistics do not match raw "
                                        "artifacts")
                                robust = expected_robust

                                def pct_delta(candidate, reference):
                                    return (
                                        round(
                                            (candidate - reference)
                                            / reference * 100, 1)
                                        if reference and candidate is not None
                                        else None)

                                expected_vs = {
                                    "acc_delta_pp": (
                                        round(comparison["delta_pp"], 2)
                                        if comparison.get("delta_pp")
                                        is not None else None),
                                    "flips_only_baseline": comparison.get(
                                        "flips_baseline_only_correct"),
                                    "flips_only_candidate": comparison.get(
                                        "flips_candidate_only_correct"),
                                    "mcnemar_p": comparison.get(
                                        "mcnemar_exact_p"),
                                    "token_inflation_pct": pct_delta(cm, bm),
                                    "token_p50_delta_pct": pct_delta(
                                        cp50, bp50),
                                    "token_p90_delta_pct": pct_delta(
                                        cp90, bp90),
                                    "token_ratio_median_delta_pct": pct_delta(
                                        robust.get("ratio_median"), 1.0),
                                    "token_ratio_geomean_delta_pct": pct_delta(
                                        robust.get("ratio_geomean"), 1.0),
                                    "token_sign_p": robust.get("sign_p"),
                                    "token_longer": robust.get("longer"),
                                    "token_shorter": robust.get("shorter"),
                                    "baseline_runaways": robust.get(
                                        "baseline_runaways"),
                                    "candidate_runaways": robust.get(
                                        "candidate_runaways"),
                                    "extra_truncated_no_answer": (
                                        truncated.get("candidate", 0)
                                        - truncated.get("baseline", 0)),
                                    "extra_hit_max_tokens": (
                                        hit_max.get("candidate", 0)
                                        - hit_max.get("baseline", 0)),
                                }
                                if probe.get("vs_baseline") != expected_vs:
                                    err(f"{where}: probe {key} compact "
                                        "baseline comparison does not "
                                        "match raw artifact")
                            invocation = raw.get("harness_invocation") or {}
                            if (not invocation
                                    or invocation.get("tool_sha256")
                                    != probe.get("tool_sha256")):
                                err(f"{where}: probe {key} invocation/tool "
                                    "binding missing or mismatched")
                            recipe_snapshot = inputs.get(
                                "recipe_snapshot") or {}
                            model_name = recipe_snapshot.get("model")
                            model_revision = (
                                (inputs.get("models") or {})
                                .get(model_name, {}).get("revision"))
                            expected_invocation = {
                                "candidate_model": model_name,
                                "model_revision": model_revision,
                                "serve_args_sha256": (
                                    integrity.canonical_sha256(
                                        recipe_snapshot.get(
                                            "serve_args", []))),
                                "served_model": recipe_snapshot.get(
                                    "served_name"),
                                "context": recipe_snapshot.get("context"),
                                "profile": configured_probe.get("profile"),
                                "runs": configured_probe.get("runs", 200),
                                "concurrency": configured_probe.get(
                                    "concurrency", 2),
                                "max_tokens": configured_probe.get(
                                    "max_tokens", 6000),
                                "request_overrides": (
                                    configured_probe.get(
                                        "request_overrides") or {}),
                                "baseline": configured_probe.get("baseline"),
                                "baseline_artifact_sha256": (
                                    (
                                        common.load_baseline_registry()
                                        .get(
                                            configured_probe.get("baseline"),
                                            {},
                                        )
                                        .get("artifact_sha256")
                                    )
                                    if configured_probe.get("baseline")
                                    else None
                                ),
                            }
                            if any(
                                invocation.get(field) != value
                                for field, value
                                in expected_invocation.items()
                            ):
                                err(f"{where}: probe {key} harness "
                                    "invocation differs from bound suite")
            except Exception as e:  # noqa: BLE001
                err(f"{where}: cannot verify schema-2 bindings ({e})")
    return n


def lint_model_registry(recipes):
    """Every model a recipe references must be in models.yaml — otherwise the
    customer image cannot download it."""
    path = os.path.join(common.BENCH_DIR, "models.yaml")
    try:
        import yaml
        with open(path) as f:
            registry = yaml.safe_load(f)["models"]
    except Exception as e:  # noqa: BLE001
        err(f"models.yaml: unloadable ({e})")
        return
    for name, spec in registry.items():
        if not (spec or {}).get("hf_repo"):
            err(f"models.yaml: {name} missing hf_repo")
        revision = (spec or {}).get("revision")
        if (not isinstance(revision, str) or len(revision) != 40
                or any(c not in "0123456789abcdef" for c in revision.lower())):
            err(f"models.yaml: {name} revision must be a full 40-hex commit")
    referenced = set()
    for r in recipes:
        if not r:
            continue
        referenced.add(r["model"])
        referenced.update(r["requires"].get("extra_models", []))
    for name in sorted(referenced):
        if name not in registry:
            err(f"models.yaml: {name} referenced by a recipe but not "
                "registered (the recipes image cannot download it)")
    for name in sorted(set(registry) - referenced):
        warn(f"models.yaml: {name} registered but unused by any recipe")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default=None)
    args = ap.parse_args()

    recipe_ids = common.list_recipe_ids()
    if not recipe_ids:
        err("no recipes found")
    recipes = [lint_recipe(rid) for rid in recipe_ids]
    lint_model_registry(recipes)

    baselines = common.load_baseline_registry()
    for bid, spec in baselines.items():
        f = os.path.join(common.baselines_dir(), (spec or {}).get("file", ""))
        if not (spec or {}).get("file") or not os.path.exists(f):
            err(f"baseline {bid}: reference file missing ({f})")
        elif (not (spec or {}).get("artifact_sha256")
              or integrity.sha256_file(f) != spec["artifact_sha256"]):
            err(f"baseline {bid}: artifact hash missing or mismatched")
        for field in ("checkpoint", "mode", "provenance"):
            if not (spec or {}).get(field):
                warn(f"baseline {bid}: missing `{field}`")
        if (spec or {}).get("strict"):
            for field in (
                "model",
                "model_revision",
                "serve_args_sha256",
                "context",
                "protocol",
            ):
                if spec.get(field) is None:
                    err(f"baseline {bid}: strict baseline missing `{field}`")
            try:
                with open(f) as artifact_file:
                    artifact = json.load(artifact_file)
                invocation = artifact.get("harness_invocation") or {}
                protocol = spec.get("protocol") or {}
                checks = {
                    "candidate_model": spec.get("model"),
                    "model_revision": spec.get("model_revision"),
                    "serve_args_sha256": spec.get(
                        "serve_args_sha256"),
                    "context": spec.get("context"),
                    "profile": protocol.get("profile"),
                    "runs": protocol.get("runs"),
                    "concurrency": protocol.get("concurrency"),
                    "max_tokens": protocol.get("max_tokens"),
                    "request_overrides": (
                        protocol.get("request_overrides") or {}),
                }
                if any(invocation.get(k) != v for k, v in checks.items()):
                    err(f"baseline {bid}: artifact invocation does not "
                        "match registry protocol")
            except Exception as e:  # noqa: BLE001
                err(f"baseline {bid}: cannot validate strict artifact ({e})")

    all_probe_keys = set()
    suites_dir = os.path.join(common.BENCH_DIR, "suites")
    suite_names = sorted(
        fn[:-5] for fn in os.listdir(suites_dir) if fn.endswith(".yaml"))
    for required in ("standard", "quick", "quality"):
        if required not in suite_names:
            err(f"suite {required}: missing")
    for suite in suite_names:
        try:
            s = common.load_suite(suite)
            seen_keys = set()
            for p in s["probes"]:
                k = p["kind"]
                if "enabled" in p and not isinstance(p["enabled"], bool):
                    err(f"suite {suite}: probe enabled must be boolean")
                if k not in probes_mod.PROBES:
                    err(f"suite {suite}: unknown probe kind {k!r}")
                key = p.get("id") or k
                all_probe_keys.add(key)
                if key in seen_keys:
                    err(f"suite {suite}: duplicate probe key {key!r} "
                        "(give multi-instance kinds unique `id`s)")
                seen_keys.add(key)
                b = p.get("baseline")
                if b and b not in baselines:
                    err(f"suite {suite}: probe {key} references unknown "
                        f"baseline {b!r}")
        except FileNotFoundError:
            err(f"suite {suite}: missing")
        except Exception as e:  # noqa: BLE001
            err(f"suite {suite}: unloadable ({e})")
    for recipe in recipes:
        if not recipe:
            continue
        unknown = sorted(
            set(recipe.get("suite_params", {})) - all_probe_keys)
        if unknown:
            err(f"recipe {recipe['id']}: suite_params has unknown probe "
                f"keys {unknown}")

    boxes_dir = os.path.join(common.BENCH_DIR, "boxes")
    for fn in sorted(os.listdir(boxes_dir)):
        if not fn.endswith(".yaml"):
            continue
        try:
            b = common.load_box(os.path.join(boxes_dir, fn))
            for field in ("id", "gpus", "runtime"):
                if not b.get(field):
                    err(f"box {fn}: missing `{field}`")
            if b.get("runtime") == "docker" and not b.get("docker_image"):
                err(f"box {fn}: runtime docker without docker_image")
            if b.get("runtime") == "venv" and not b.get("venv"):
                err(f"box {fn}: runtime venv without venv path")
            if b.get("quality_tool"):
                digest = b.get("quality_tool_sha256")
                if (not isinstance(digest, str) or len(digest) != 64):
                    err(f"box {fn}: quality_tool requires full sha256")
            if b.get("runtime") == "docker":
                leaked = set(b.get("env", {})) & set(common._VENV_ONLY_ENV)
                if leaked:
                    err(f"box {fn}: venv-only env in a docker box: "
                        f"{sorted(leaked)}")
        except Exception as e:  # noqa: BLE001
            err(f"box {fn}: unloadable ({e})")

    mx = common.load_matrix()
    if not mx.get("current_release"):
        err("matrix.yaml: current_release unset")
    for entry in mx["entries"]:
        rid = entry.get("recipe")
        if rid not in recipe_ids:
            err(f"matrix entry {rid!r}: no such recipe")

    results_root = os.path.join(common.BENCH_DIR, "results")
    if os.path.isdir(results_root):
        for release in sorted(os.listdir(results_root)):
            lint_results(release)

    if args.release:
        release = (mx["current_release"]
                   if args.release == "current" else args.release)
        historical = set(mx.get("historical_releases", []))
        by_recipe: dict[str, list[tuple[str, dict]]] = {}
        for result_box_id, res in common.iter_results(release):
            by_recipe.setdefault(res.get("recipe"), []).append(
                (result_box_id, res))
        if release in historical:
            warn(f"release {release}: legacy schema-1/imported evidence is "
                 "allowlisted as historical; it does not certify a fresh "
                 "live release")
            for entry in mx["entries"]:
                if not entry.get("blocking"):
                    continue
                rid = entry["recipe"]
                statuses = {
                    r.get("status") for _box_id, r
                    in by_recipe.get(rid, [])
                }
                if not statuses & {"ok", "partial"}:
                    err(f"release {release}: blocking recipe {rid} has no "
                        "historical successful result")
        else:
            for entry in mx["entries"]:
                if not entry.get("blocking"):
                    continue
                rid = entry["recipe"]
                expected_suite = entry.get("suite", "standard")
                valid = []
                for result_box_id, result in by_recipe.get(rid, []):
                    try:
                        recipe = common.load_recipe(rid)
                        suite = common.load_suite(expected_suite)
                        probes = common.merge_suite(suite, recipe)
                        box_path = next(
                            os.path.join(boxes_dir, fn)
                            for fn in os.listdir(boxes_dir)
                            if fn.endswith(".yaml")
                            and common.load_box(
                                os.path.join(boxes_dir, fn)
                            )["id"] == result_box_id
                        )
                        box = common.load_box(box_path)
                        expected = integrity.result_bindings(
                            recipe, expected_suite, probes,
                            common.load_model_registry(),
                            release=release, box=box)
                        runtime_manifest = (
                            (mx.get("runtime_manifests") or {})
                            .get(release, {})
                            .get(result_box_id)
                        )
                        if runtime_manifest:
                            expected["model_manifests"] = (
                                runtime_manifest.get("model_manifests"))
                        if not integrity.strict_gate_errors(
                                result, expected_suite, expected,
                                runtime_manifest):
                            valid.append(result)
                    except Exception:
                        continue
                if not valid:
                    err(f"release {release}: blocking recipe {rid} needs a "
                        f"complete schema-2 live+ok {expected_suite} result")

    print(f"\nlint: {len(ERRORS)} error(s), {len(WARNINGS)} warning(s)")
    sys.exit(1 if ERRORS else 0)


if __name__ == "__main__":
    main()
