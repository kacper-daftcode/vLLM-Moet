#!/usr/bin/env python3
"""Empirical knob search for one recipe: enumerate the recipe's tuning.axes,
bench each combo, score by tuning.objective subject to tuning.constraints,
and print the winner. The full sweep lands in
results/<release>/<box>/sweeps/<recipe>__sweep.json.

The winner is NOT auto-applied: a human edits the recipe (that diff is the
documented decision) and re-runs bench.py for the headline numbers.

Server boots are the expensive part (14-25 min for the big checkpoints), so
axes marked `restart: false` + `file_env: <VAR>` are changed at runtime: the
server is started with <VAR> pointing at a temp file and the sweep rewrites
that file between combos (the VLLM_MOE_W2_GATE_TAU_FILE mechanism).

  sweep.py --recipe glm-5.2-nvfp4/pro6000x4-tp4-quality \
           --box bench/boxes/runpod-4xpro6000.yaml --release v2026.07.15
"""

import argparse
import itertools
import json
import os
import re
import sys
import tempfile
import time

import yaml

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import common
import probes as probes_mod
from serverctl import Server, ServerFailed


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _dig(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


_CMP = re.compile(r"^([\w.]+)\s*(>=|<=|==|>|<)\s*([\d.]+)$")


def check_constraint(results, expr):
    m = _CMP.match(expr.strip())
    if m:
        path, op, num = m.groups()
        v = _dig(results, path)
        if v is None:
            return False
        num = float(num)
        return {"<": v < num, "<=": v <= num, ">": v > num,
                ">=": v >= num, "==": v == num}[op]
    return bool(_dig(results, expr.strip()))


def needed_probes(tuning):
    refs = [tuning["objective"], *tuning.get("constraints", [])]
    kinds = []
    for r in refs:
        k = re.split(r"[.\s<>=]", r.strip())[0]
        if k in probes_mod.PROBES and k not in kinds:
            kinds.append(k)
    return kinds


def sweep_probe_params(recipe, suite, kinds):
    """Suite+recipe params, thinned for ranking speed (headline run comes
    after, on the winning knobs, with the full suite)."""
    merged = {p["kind"]: p for p in common.merge_suite(suite, recipe)}
    out = []
    for k in kinds:
        p = dict(merged.get(k, {"kind": k}))
        p["kind"] = k
        if k == "decode":
            p["runs"] = min(int(p.get("runs", 5)), 3)
        if k == "needle":
            if "sizes_words" in p:
                p["sizes_words"] = [min(p["sizes_words"])]
        if k == "batch_decode":
            p["runs"] = 1
        out.append(p)
    return out


def apply_serve_json(args_list, arg_name, key, value):
    out = []
    for a in args_list:
        flag, _, rest = a.partition(" ")
        if flag == arg_name and rest:
            d = json.loads(rest)
            d[key] = value
            out.append(f"{flag} {json.dumps(d, separators=(',', ':'))}")
        else:
            out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--box", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--port", type=int, default=common.DEFAULT_PORT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    box = common.load_box(args.box)
    recipe = common.load_recipe(args.recipe)
    tuning = recipe.get("tuning")
    if not tuning or not tuning.get("axes"):
        raise SystemExit(f"{args.recipe} declares no tuning.axes")
    reason = common.compat(recipe, box)
    if reason:
        raise SystemExit(f"box cannot run {args.recipe}: {reason}")

    suite = common.load_suite("standard")
    kinds = needed_probes(tuning)
    probe_params = sweep_probe_params(recipe, suite, kinds)

    axes = tuning["axes"]
    runtime_axes = [a for a in axes
                    if a.get("restart") is False and a.get("file_env")]
    restart_axes = [a for a in axes if a not in runtime_axes]
    combos = list(itertools.product(*[a["values"] for a in restart_axes])) or [()]
    rt_combos = list(itertools.product(*[a["values"] for a in runtime_axes])) or [()]
    log(f"{len(combos)} server boot(s) x {len(rt_combos)} runtime combo(s); "
        f"probes per combo: {kinds}")
    if args.dry_run:
        for c in combos:
            for rc in rt_combos:
                print(dict(zip([a.get('name') or a.get('key') for a in restart_axes], c)) |
                      dict(zip([a['name'] for a in runtime_axes], rc)))
        return

    n_gpus = recipe["requires"].get("gpus", 1)
    gpus = ([int(x) for x in args.gpus.split(",")] if args.gpus
            else list(range(n_gpus)))
    rows = []

    runtime = box.get("runtime", "venv")
    for combo in combos:
        # materialize this combo into a copy of the recipe
        rc = json.loads(json.dumps({k: recipe[k] for k in
                                    ("env", "serve_args")}))
        knobs = {}
        for axis, val in zip(restart_axes, combo):
            if axis["kind"] == "env":
                rc["env"][axis["name"]] = str(val)
                knobs[axis["name"]] = val
            elif axis["kind"] == "serve_json":
                rc["serve_args"] = apply_serve_json(
                    rc["serve_args"], axis["arg"], axis["key"], val)
                knobs[f"{axis['arg']}:{axis['key']}"] = val
        tau_files = {}
        for axis in runtime_axes:
            f = tempfile.NamedTemporaryFile(  # noqa: SIM115 — lives past loop
                mode="w", prefix="sweep-", suffix=".knob", delete=False)
            f.write(str(rt_combos[0][runtime_axes.index(axis)]))
            f.close()
            rc["env"][axis["file_env"]] = f.name
            tau_files[axis["name"]] = f.name

        var = dict(recipe)
        var["env"], var["serve_args"] = rc["env"], rc["serve_args"]
        if runtime == "docker":
            # the container reads the variant recipe + knob files via binds
            rf = tempfile.NamedTemporaryFile(  # noqa: SIM115
                mode="w", prefix="sweep-recipe-", suffix=".yaml", delete=False)
            yaml.safe_dump(var, rf)
            rf.close()
            binds = [f"{p}:{p}" for p in tau_files.values()]
            env = dict(os.environ)
            cmd = common.build_serve_cmd(var, box, args.port, gpus,
                                         recipe_file=rf.name,
                                         extra_binds=binds)
        else:
            env = common.build_env(var, box, gpus)
            cmd = common.build_serve_cmd(var, box, args.port, gpus)
        log_path = (f"{box['log_dir']}/{args.release}/sweep-"
                    f"{args.recipe.replace('/', '__')}-{len(rows)}.log")
        srv = Server(cmd, env, log_path, args.port, gpus,
                     container=(common.container_name(args.port)
                                if runtime == "docker" else None))
        log(f"boot {knobs} (log: {log_path})")
        try:
            srv.start(recipe.get("load_timeout_s",
                                 common.DEFAULT_LOAD_TIMEOUT_S))
        except ServerFailed as e:
            for rt in rt_combos:
                rows.append({"knobs": {**knobs, **dict(zip(
                    [a["name"] for a in runtime_axes], rt))},
                    "status": "failed", "error": str(e)})
            log(f"boot FAILED: {e}")
            continue

        try:
            model = srv.model_id()
            nw = recipe.get("warmup", {}).get("decode_requests", 0)
            if nw:
                probes_mod.warmup(srv.base, model, nw, log)
            for rt in rt_combos:
                rt_knobs = dict(zip([a["name"] for a in runtime_axes], rt))
                for name, val in rt_knobs.items():
                    with open(tau_files[name], "w") as f:
                        f.write(str(val))
                if rt_knobs:
                    log(f"runtime knobs -> {rt_knobs}")
                    time.sleep(2)
                res = {}
                for p in probe_params:
                    p = dict(p)
                    kind = p.pop("kind")
                    res[kind] = probes_mod.PROBES[kind](
                        srv.base, model, log=log,
                        context=recipe.get("context"), **p)
                cons = {c: check_constraint(res, c)
                        for c in tuning.get("constraints", [])}
                rows.append({
                    "knobs": {**knobs, **rt_knobs},
                    "status": "ok",
                    "objective": _dig(res, tuning["objective"]),
                    "constraints": cons,
                    "valid": all(cons.values()),
                    "probes": res,
                })
                log(f"combo {rows[-1]['knobs']}: "
                    f"{tuning['objective']}={rows[-1]['objective']} "
                    f"valid={rows[-1]['valid']}")
        finally:
            srv.stop()

    out_dir = os.path.join(common.BENCH_DIR, "results", args.release,
                           box["id"], "sweeps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                            args.recipe.replace("/", "__") + "__sweep.json")
    common.write_result(out_path, {
        "schema": 1, "release": args.release, "box": box["id"],
        "recipe": args.recipe, "objective": tuning["objective"],
        "constraints": tuning.get("constraints", []),
        "finished": common.now_iso(), "rows": rows,
    })

    ranked = sorted((r for r in rows if r.get("valid")),
                    key=lambda r: -(r["objective"] or 0))
    print("\n=== sweep ranking (valid combos) ===")
    for r in ranked:
        print(f"{r['objective']!s:>10}  {r['knobs']}")
    if ranked:
        print(f"\nWINNER: {ranked[0]['knobs']} "
              f"({tuning['objective']}={ranked[0]['objective']})")
        print("Apply it to the recipe, commit, then re-run bench.py for the "
              "headline numbers.")
    else:
        print("no combo satisfied the constraints — inspect "
              f"{out_path}")
    print(f"sweep data: {out_path}")


if __name__ == "__main__":
    main()
