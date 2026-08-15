#!/usr/bin/env python3
"""Manifest-first DS4-W2 evaluation runner.

This runner intentionally has no default host command. It talks only to the
explicit endpoint and KPI source supplied by the operator.  Warm runs require
a pool gate policy and KPI source; otherwise "warm" would remain a request
count rather than a measured state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

from harness import (
    Completion,
    HarnessError,
    PoolGateError,
    PoolGatePolicy,
    RequestSpec,
    SCHEMA_VERSION,
    WARMUP_CASES,
    WARMUP_MAX_TOKENS,
    WARMUP_SEED,
    WARMUP_SUITE_VERSION,
    WARMUP_TEMPERATURE,
    WARMUP_TOP_P,
    append_jsonl,
    evaluate_pool_gate,
    interleaved_items,
    load_server_provenance,
    parse_pool_json,
    parse_pool_log,
    run_prewarm,
    sha256_file,
    utc_now,
    write_json,
)
from rescore_robust import extract_final, lenient_matches, matches, score_rows, sink


FINAL_INSTRUCTION = (
    "\n\nThink briefly if needed, then end your reply with a single line of "
    "the form:\nFINAL: <answer>"
)


class HttpCompletionClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def __call__(self, spec: RequestSpec) -> Completion:
        body = {
            "model": spec.model,
            "stream": False,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "seed": spec.seed,
            "max_tokens": spec.max_tokens,
            "messages": [{"role": "user", "content": spec.prompt}],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(
                request, timeout=spec.timeout_seconds
            ) as response:
                status = response.status
                payload = json.loads(response.read().decode("utf-8"))
            choice = payload["choices"][0]
            usage = payload.get("usage", {})
            return Completion(
                ok=True,
                content=str(choice["message"].get("content", "")),
                status_code=status,
                completion_tokens=usage.get("completion_tokens"),
                wall_seconds=round(time.monotonic() - started, 3),
                finish_reason=choice.get("finish_reason"),
                response_id=payload.get("id"),
            )
        except urllib.error.HTTPError as error:
            return Completion(
                ok=False,
                status_code=error.code,
                wall_seconds=round(time.monotonic() - started, 3),
                error=f"HTTPError: {error.reason}",
            )
        except Exception as error:  # noqa: BLE001 - serialized into the receipt
            return Completion(
                ok=False,
                wall_seconds=round(time.monotonic() - started, 3),
                error=f"{type(error).__name__}: {error}",
            )


def _read_json(path: str | Path) -> Any:
    with Path(path).open() as stream:
        return json.load(stream)


def _load_pool_snapshot(args: argparse.Namespace):
    if args.pool_log_file:
        return parse_pool_log(Path(args.pool_log_file).read_text())
    if args.pool_json_file:
        text = Path(args.pool_json_file).read_text()
        return parse_pool_json(json.loads(text), source_text=text)
    if args.pool_kpi_url:
        with urllib.request.urlopen(
            args.pool_kpi_url, timeout=args.pool_kpi_timeout
        ) as response:
            text = response.read().decode("utf-8")
        try:
            return parse_pool_json(json.loads(text), source_text=text)
        except json.JSONDecodeError:
            return parse_pool_log(text)
    if args.pool_command_json:
        command = json.loads(args.pool_command_json)
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) for item in command)
        ):
            raise PoolGateError(
                "--pool-command-json must be a non-empty JSON argv array"
            )
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=args.pool_kpi_timeout,
        )
        text = process.stdout
        if process.stderr:
            text += ("\n" if text else "") + process.stderr
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return parse_pool_log(text)
        return parse_pool_json(value, source_text=text)
    raise PoolGateError("warm mode requires one explicit pool KPI source")


def _load_items(path: Path, expected_count: int) -> list[dict[str, Any]]:
    value = _read_json(path)
    if not isinstance(value, list):
        raise HarnessError("items file must contain a JSON list")
    order = interleaved_items(value)
    if len(order) != expected_count:
        raise HarnessError(f"expected {expected_count} items, found {len(order)}")
    for item in order:
        if "prompt" not in item or "answer" not in item:
            raise HarnessError(f"item {item['id']} requires prompt and answer")
    return order


def _build_manifest(
    args: argparse.Namespace, items_path: Path, server: dict[str, Any]
) -> dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    code_paths = [
        script_dir / name for name in ("eval_rig.py", "harness.py", "rescore_robust.py")
    ]
    run_id = str(uuid.uuid4())
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_label": args.run_label,
        "status": "initializing",
        "quality_comparable": False,
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "server": server,
        "server_provenance_sha256": sha256_file(args.server_provenance),
        "inputs": {
            "items_path": str(items_path.resolve()),
            "items_sha256": sha256_file(items_path),
            "expected_count": args.expected_count,
        },
        "harness": {
            "python": sys.version,
            "argv": sys.argv,
            "files": {path.name: sha256_file(path) for path in code_paths},
        },
        "eval": {
            "seed": args.eval_seed,
            "temperature": args.eval_temperature,
            "top_p": args.eval_top_p,
            "max_tokens": args.eval_max_tokens,
            "timeout_seconds": args.eval_timeout,
            "order": "reasoning-coding-interleaved-fixed",
            "retry_count": 0,
        },
        "prewarm": {
            "enabled": args.mode == "warm",
            "suite_version": WARMUP_SUITE_VERSION,
            "cases": len(WARMUP_CASES),
            "seed": WARMUP_SEED,
            "temperature": WARMUP_TEMPERATURE,
            "top_p": WARMUP_TOP_P,
            "max_tokens": WARMUP_MAX_TOKENS,
            "retry_count": 0,
            "status": "pending" if args.mode == "warm" else "not_requested",
        },
        "pool_gate": {
            "configured": args.mode == "warm",
            "status": "pending" if args.mode == "warm" else "not_requested",
        },
        "artifacts": {},
        "summary": None,
        "error": None,
    }


def _validate_provenance_matches(
    server: dict[str, Any], args: argparse.Namespace
) -> None:
    mismatches = []
    if server["served_model"] != args.model:
        mismatches.append(
            f"served_model={server['served_model']!r} != --model={args.model!r}"
        )
    if server["endpoint"].rstrip("/") != args.url.rstrip("/"):
        mismatches.append(f"endpoint={server['endpoint']!r} != --url={args.url!r}")
    if mismatches:
        raise HarnessError("server provenance mismatch: " + "; ".join(mismatches))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--server-provenance", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash-w2")
    parser.add_argument("--mode", choices=("cold", "warm"), default="warm")
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--eval-temperature", type=float, default=0.6)
    parser.add_argument("--eval-top-p", type=float, default=0.95)
    parser.add_argument("--eval-max-tokens", type=int, default=700)
    parser.add_argument("--eval-timeout", type=int, default=300)
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument(
        "--pool-gate-policy", help="JSON policy file; required in warm mode"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pool-log-file")
    source.add_argument("--pool-json-file")
    source.add_argument("--pool-kpi-url")
    source.add_argument(
        "--pool-command-json", help="JSON argv array; no shell evaluation"
    )
    parser.add_argument("--pool-kpi-timeout", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "warm" and not args.pool_gate_policy:
        raise PoolGateError("warm mode requires --pool-gate-policy")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.run_label
    manifest_path = output_dir / f"{stem}.manifest.json"
    raw_path = output_dir / f"{stem}.raw.jsonl"
    warmup_path = output_dir / f"{stem}.warmup.jsonl"
    for path in (manifest_path, raw_path, warmup_path):
        if path.exists():
            raise HarnessError(f"refusing to overwrite existing artifact: {path}")

    items_path = Path(args.items)
    server = load_server_provenance(args.server_provenance)
    _validate_provenance_matches(server, args)
    items = _load_items(items_path, args.expected_count)
    manifest = _build_manifest(args, items_path, server)
    manifest["artifacts"] = {
        "manifest": str(manifest_path.resolve()),
        "raw_jsonl": str(raw_path.resolve()),
        "warmup_jsonl": str(warmup_path.resolve()) if args.mode == "warm" else None,
    }
    write_json(manifest_path, manifest)
    client = HttpCompletionClient(args.url)
    pool_policy: PoolGatePolicy | None = None
    pre_eval_snapshot = None

    try:
        if args.mode == "warm":
            run_prewarm(
                client, args.model, lambda receipt: append_jsonl(warmup_path, receipt)
            )
            manifest["prewarm"]["status"] = "passed"
            manifest["prewarm"]["receipts_sha256"] = sha256_file(warmup_path)
            snapshot = _load_pool_snapshot(args)
            policy_value = _read_json(args.pool_gate_policy)
            if not isinstance(policy_value, dict):
                raise PoolGateError("pool gate policy root must be an object")
            pool_policy = PoolGatePolicy.from_mapping(policy_value)
            gate = evaluate_pool_gate(snapshot, pool_policy)
            manifest["pool_gate"] = {
                "status": "passed" if gate["passed"] else "failed",
                "policy_path": str(Path(args.pool_gate_policy).resolve()),
                "policy_sha256": sha256_file(args.pool_gate_policy),
                **gate,
            }
            write_json(manifest_path, manifest)
            if not gate["passed"]:
                failed = [check for check in gate["checks"] if not check["passed"]]
                raise PoolGateError(f"pool gate failed: {failed}")
            pre_eval_snapshot = snapshot
        manifest["status"] = "scoring"
        write_json(manifest_path, manifest)

        scored_rows: list[dict[str, Any]] = []
        for position, item in enumerate(items, start=1):
            spec = RequestSpec(
                model=args.model,
                prompt=str(item["prompt"]) + FINAL_INSTRUCTION,
                temperature=args.eval_temperature,
                top_p=args.eval_top_p,
                seed=args.eval_seed,
                max_tokens=args.eval_max_tokens,
                timeout_seconds=args.eval_timeout,
            )
            result = client(spec)
            row = {
                "schema_version": SCHEMA_VERSION,
                "run_id": manifest["run_id"],
                "position": position,
                "id": item["id"],
                "cat": item["cat"],
                "expected": item["answer"],
                "request_sha256": spec.request_hash(),
                "response_id": result.response_id,
                "status_code": result.status_code,
                "finish_reason": result.finish_reason,
                "completion_tokens": result.completion_tokens,
                "wall_seconds": result.wall_seconds,
                "content": result.content,
                "error": result.error,
            }
            if (
                result.ok
                and result.content.strip()
                and result.completion_tokens is not None
            ):
                is_sink, reason = sink(result.content, result.completion_tokens)
                final = extract_final(result.content)
                row.update(
                    {
                        "final": final,
                        "sink": is_sink,
                        "why": reason,
                        "exact": (not is_sink) and matches(item["answer"], final),
                        "lenient": (not is_sink)
                        and lenient_matches(item["answer"], result.content),
                    }
                )
            else:
                row.update(
                    {
                        "final": None,
                        "sink": None,
                        "why": "request-failed",
                        "exact": False,
                        "lenient": False,
                    }
                )
            append_jsonl(raw_path, row)
            scored_rows.append(row)
            if (
                not result.ok
                or not result.content.strip()
                or result.completion_tokens is None
            ):
                raise HarnessError(
                    f"scored request {position}/{len(items)} ({item['id']}) failed; partial raw preserved"
                )

        summary = score_rows(scored_rows, expected_count=args.expected_count)
        manifest["summary"] = {
            key: value for key, value in summary.items() if key != "rows"
        }
        manifest["artifacts"]["raw_sha256"] = sha256_file(raw_path)
        if args.mode == "warm":
            assert pool_policy is not None
            assert pre_eval_snapshot is not None
            final_snapshot = _load_pool_snapshot(args)
            post_eval_gate = evaluate_pool_gate(
                final_snapshot, pool_policy, baseline=pre_eval_snapshot
            )
            manifest["pool_gate"]["post_eval"] = post_eval_gate
            current_policy_sha256 = sha256_file(args.pool_gate_policy)
            if current_policy_sha256 != manifest["pool_gate"]["policy_sha256"]:
                raise PoolGateError("pool gate policy changed while scoring")
            if not post_eval_gate["passed"]:
                failed = [
                    check for check in post_eval_gate["checks"] if not check["passed"]
                ]
                raise PoolGateError(f"post-eval pool gate failed: {failed}")
        manifest["status"] = "complete"
        manifest["quality_comparable"] = args.mode == "cold" or bool(
            manifest["pool_gate"].get("passed")
        )
        manifest["completed_at_utc"] = utc_now()
        write_json(manifest_path, manifest)
        print(json.dumps(manifest["summary"], sort_keys=True))
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["quality_comparable"] = False
        manifest["completed_at_utc"] = utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        if args.mode == "warm" and manifest["prewarm"]["status"] == "pending":
            manifest["prewarm"]["status"] = "failed"
            if warmup_path.exists():
                manifest["prewarm"]["receipts_sha256"] = sha256_file(warmup_path)
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
