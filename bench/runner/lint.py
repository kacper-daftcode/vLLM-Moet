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
    for box_id, res in common.iter_results(release):
        n += 1
        rid = res.get("recipe", "?")
        where = f"results/{release}/{box_id}/{rid}"
        if res.get("schema") != 1:
            err(f"{where}: schema != 1")
        if res.get("status") not in ("ok", "partial", "failed"):
            err(f"{where}: bad status {res.get('status')!r}")
        if res.get("provenance") not in ("live", "imported"):
            err(f"{where}: bad provenance {res.get('provenance')!r}")
        if not os.path.exists(common.recipe_path(rid)):
            err(f"{where}: references unknown recipe {rid}")
        if res.get("provenance") == "live":
            for field in ("serve_cmd", "env_fingerprint", "load_time_s"):
                if not res.get(field):
                    warn(f"{where}: live result missing `{field}`")
        if res.get("status") == "ok" and not res.get("probes"):
            err(f"{where}: ok result with no probes")
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

    for suite in ("standard", "quick"):
        try:
            s = common.load_suite(suite)
            kinds = [p["kind"] for p in s["probes"]]
            import probes as probes_mod
            for k in kinds:
                if k not in probes_mod.PROBES:
                    err(f"suite {suite}: unknown probe kind {k!r}")
        except Exception as e:  # noqa: BLE001
            err(f"suite {suite}: unloadable ({e})")

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
        rank = {"ok": 2, "partial": 1}
        best: dict = {}
        for _box_id, res in common.iter_results(args.release):
            rid = res.get("recipe")
            st = res.get("status")
            if rank.get(st, 0) > rank.get(best.get(rid), 0):
                best[rid] = st
        for entry in mx["entries"]:
            if not entry.get("blocking"):
                continue
            rid = entry["recipe"]
            st = best.get(rid)
            if st not in ("ok", "partial"):
                err(f"release {args.release}: blocking recipe {rid} has no "
                    "successful result")
            elif st != "ok":
                warn(f"release {args.release}: {rid} best status is {st}")

    print(f"\nlint: {len(ERRORS)} error(s), {len(WARNINGS)} warning(s)")
    sys.exit(1 if ERRORS else 0)


if __name__ == "__main__":
    main()
