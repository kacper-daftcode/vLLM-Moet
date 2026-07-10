#!/usr/bin/env python3
"""Run one recipe (or the whole matrix) against a box and write result JSONs.

  bench.py list   --box bench/boxes/<id>.yaml
  bench.py run    --recipe <model>/<config> --box ... --release <id>
                  [--suite standard] [--gpus 0,1] [--port 8123] [--dry-run]
  bench.py matrix --box ... --release <id> [--only-blocking] [--suite standard]

Results land in bench/results/<release>/<box>/<recipe>.json (committed).
A failed serve still writes a result (status: failed, log tail attached)."""

import argparse
import os
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import common
import envinfo
import probes as probes_mod
from serverctl import Server, ServerFailed, gpu_mem_used_mib, tail


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def preflight_gpus(gpus):
    """Refuse to boot onto GPUs something else is using — a stale server would
    otherwise fail the util-based memory request minutes into the load."""
    used = gpu_mem_used_mib(gpus)
    busy = {g: m for g, m in used.items() if m > 2048}
    if busy:
        raise SystemExit(
            f"GPUs busy before start: {busy} (MiB used). Kill the stale "
            "process or pick other GPUs with --gpus.")


def run_recipe(recipe_id, box, release, suite_name, gpus=None, port=None,
               dry_run=False):
    recipe = common.load_recipe(recipe_id)
    suite = common.load_suite(suite_name)
    reason = common.compat(recipe, box)
    if reason:
        log(f"SKIP {recipe_id}: {reason}")
        return "skipped"

    n_gpus = recipe["requires"].get("gpus", 1)
    gpus = gpus if gpus is not None else list(range(n_gpus))
    if len(gpus) != n_gpus:
        raise SystemExit(f"{recipe_id} needs {n_gpus} GPUs, got --gpus {gpus}")
    port = port or common.DEFAULT_PORT

    runtime = box.get("runtime", "venv")
    env = (dict(os.environ) if runtime == "docker"
           else common.build_env(recipe, box, gpus))
    knobs = common.declared_env(recipe, box)
    cmd = common.build_serve_cmd(recipe, box, port, gpus)
    display = common.serve_cmd_display(cmd, knobs if runtime == "venv" else {})

    if dry_run:
        print(f"# {recipe_id} on {box['id']} (GPUs {gpus})")
        prefix = (f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))} "
                  if runtime == "venv" else "")
        print(prefix + display)
        for p in common.merge_suite(suite, recipe):
            print(f"probe: {p}")
        return "dry-run"

    log_path = f"{box['log_dir']}/{release}/{recipe_id.replace('/', '__')}.log"
    result = {
        "schema": 1,
        "release": release, "box": box["id"], "recipe": recipe_id,
        "suite": suite_name, "provenance": "live",
        "summary": recipe.get("summary", ""),
        "context": recipe.get("context"),
        "gpus_used": gpus,
        "started": common.now_iso(),
        "serve_env": knobs,
        "serve_cmd": display,
        "env_fingerprint": envinfo.collect(box, knobs),
        "probes": {},
    }

    preflight_gpus(gpus)
    srv = Server(cmd, env, log_path, port, gpus,
                 container=(common.container_name(port)
                            if runtime == "docker" else None))
    log(f"starting {recipe_id} (load timeout "
        f"{recipe.get('load_timeout_s', common.DEFAULT_LOAD_TIMEOUT_S)}s), "
        f"log: {log_path}")
    try:
        srv.start(recipe.get("load_timeout_s", common.DEFAULT_LOAD_TIMEOUT_S))
    except ServerFailed as e:
        result.update(status="failed", error=str(e), log_tail=e.log_tail,
                      finished=common.now_iso())
        path = common.result_path(release, box["id"], recipe_id)
        common.write_result(path, result)
        log(f"FAILED {recipe_id}: {e} -> {path}")
        return "failed"

    result["load_time_s"] = srv.load_time_s
    log(f"healthy after {srv.load_time_s}s")
    failures = []
    try:
        model = srv.model_id()
        nw = recipe.get("warmup", {}).get("decode_requests", 0)
        if nw:
            probes_mod.warmup(srv.base, model, nw, log)
        for p in common.merge_suite(suite, recipe):
            kind = p.pop("kind")
            fn = probes_mod.PROBES[kind]
            log(f"probe {kind} {p}")
            try:
                result["probes"][kind] = fn(srv.base, model, log=log,
                                            context=recipe.get("context"), **p)
            except Exception as e:  # noqa: BLE001 — keep benching, record it
                failures.append(kind)
                result["probes"][kind] = {"error": f"{type(e).__name__}: {e}"}
                log(f"probe {kind} FAILED: {e}")
            if kind == "decode":
                # acceptance counters scraped immediately, before other traffic
                result["server_metrics"] = probes_mod.scrape_server_log(log_path)
        result["server_metrics_final"] = probes_mod.scrape_server_log(log_path)
    finally:
        log("stopping server")
        srv.stop()

    result["status"] = "partial" if failures else "ok"
    if failures:
        result["failed_probes"] = failures
        result["log_tail"] = tail(log_path)
    result["finished"] = common.now_iso()
    path = common.result_path(release, box["id"], recipe_id)
    common.write_result(path, result)
    log(f"{result['status'].upper()} {recipe_id} -> {path}")
    return result["status"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--box", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--recipe", required=True)
    p_run.add_argument("--box", required=True)
    p_run.add_argument("--release", required=True)
    p_run.add_argument("--suite", default="standard")
    p_run.add_argument("--gpus", default=None,
                       help="comma-separated GPU indices (default 0..n-1)")
    p_run.add_argument("--port", type=int, default=None)
    p_run.add_argument("--dry-run", action="store_true")

    p_mx = sub.add_parser("matrix")
    p_mx.add_argument("--box", required=True)
    p_mx.add_argument("--release", required=True)
    p_mx.add_argument("--suite", default="standard")
    p_mx.add_argument("--only-blocking", action="store_true")
    p_mx.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    box = common.load_box(args.box)

    if args.cmd == "list":
        for rid in common.list_recipe_ids():
            r = common.load_recipe(rid)
            reason = common.compat(r, box)
            mark = "ok  " if reason is None else "SKIP"
            print(f"{mark} {rid:55s} {reason or r.get('summary', '')}")
        return

    if args.cmd == "run":
        gpus = ([int(x) for x in args.gpus.split(",")]
                if args.gpus else None)
        status = run_recipe(args.recipe, box, args.release, args.suite,
                            gpus, args.port, args.dry_run)
        sys.exit(0 if status in ("ok", "dry-run", "skipped") else 1)

    if args.cmd == "matrix":
        mx = common.load_matrix()
        statuses = {}
        for entry in mx["entries"]:
            if args.only_blocking and not entry.get("blocking"):
                continue
            rid = entry["recipe"]
            if entry.get("retune"):
                log(f"note: {rid} is marked retune — consider sweep.py first")
            statuses[rid] = run_recipe(rid, box, args.release, args.suite,
                                       dry_run=args.dry_run)
        print("\n=== matrix summary ===")
        for rid, st in statuses.items():
            print(f"{st:8s} {rid}")
        bad = [s for s in statuses.values() if s in ("failed", "partial")]
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
