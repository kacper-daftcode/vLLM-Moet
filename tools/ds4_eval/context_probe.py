#!/usr/bin/env python3
"""Fail-closed, provenance-bound long-context retrieval probe for DS4-W2.

The probe calibrates each deterministic haystack with the live server's
``/tokenize`` endpoint.  A requested 120,000-token case therefore means the
rendered chat prompt observed by the serving tokenizer, not a words/token
estimate.  It writes an immutable manifest and one JSONL receipt per case.

Example::

    python3 context_probe.py \
      --server-provenance /evidence/server.json \
      --output-dir /evidence/p2-128k-context \
      --run-label p2-128k-context-s42 \
      --url http://127.0.0.1:18001/v1/chat/completions \
      --tokenize-url http://127.0.0.1:18001/tokenize \
      --model deepseek-v4-flash-w2 \
      --expected-window 131072 --expected-kv-dtype fp8 \
      --expected-base-gb 8 --expected-delta-gb 6 \
      --expected-policy lru --expected-tau 0.75 \
      --case 120000:0.1 --case 120000:0.5 --case 120000:0.9
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.request
from urllib.parse import urlparse
import uuid

from harness import (
    HarnessError,
    append_jsonl,
    load_server_provenance,
    sha256_file,
    sha256_text,
    utc_now,
    write_json,
)


SCHEMA_VERSION = "ds4-w2-context-v1"
GENERATOR_VERSION = "deterministic-word-haystack-v1"
DEFAULT_SEED = 20_260_711
DEFAULT_TOLERANCE = 64
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT_SECONDS = 3600

WORDS = (
    "alpha quantum river matrix ember glacier syntax violet nimbus cobalt "
    "tangent fjord lantern zephyr cipher marble thunder willow plasma onyx "
    "harbor crimson vector lattice meadow falcon pixel saffron tundra orbit"
).split()

_RUN_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ContextProbeError(HarnessError):
    """The context receipt cannot be accepted as evidence."""


@dataclass(frozen=True)
class CaseSpec:
    target_prompt_tokens: int
    depth: float


@dataclass(frozen=True)
class ProbeConfig:
    server_provenance: str
    output_dir: str
    run_label: str
    url: str
    tokenize_url: str
    model: str
    expected_window: int
    expected_kv_dtype: str
    expected_base_gb: float
    expected_delta_gb: float
    expected_policy: str
    expected_tau: float
    cases: tuple[CaseSpec, ...]
    seed: int = DEFAULT_SEED
    prompt_token_tolerance: int = DEFAULT_TOLERANCE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


Transport = Callable[
    [str, Mapping[str, Any], int], tuple[int, Mapping[str, Any], float]
]


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _http_post(
    url: str, body: Mapping[str, Any], timeout_seconds: int
) -> tuple[int, Mapping[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise ContextProbeError(
            f"HTTP {error.code} from {url}: {detail or error.reason}"
        ) from error
    except Exception as error:  # noqa: BLE001 - serialized into a failed receipt
        raise ContextProbeError(
            f"{type(error).__name__} from {url}: {error}"
        ) from error
    wall = round(time.monotonic() - started, 3)
    if not 200 <= status < 300:
        raise ContextProbeError(f"non-2xx HTTP {status} from {url}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextProbeError(f"invalid JSON from {url}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ContextProbeError(f"JSON response from {url} is not an object")
    return status, payload, wall


def _one_argv_value(argv: Sequence[str], name: str) -> str:
    values: list[str] = []
    for index, item in enumerate(argv):
        if item == name:
            if index + 1 >= len(argv):
                raise ContextProbeError(f"runtime option {name} has no value")
            values.append(argv[index + 1])
        elif item.startswith(name + "="):
            values.append(item.split("=", 1)[1])
    if len(values) != 1:
        raise ContextProbeError(
            f"runtime must contain exactly one {name}; observed {values}"
        )
    return values[0]


def _runtime_has_option(argv: Sequence[str], name: str) -> bool:
    return any(item == name or item.startswith(name + "=") for item in argv)


def _same_number(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return False


def validate_p2_server(server: Mapping[str, Any], config: ProbeConfig) -> None:
    """Bind a receipt to the intended single-stream, no-MTP P2 runtime."""

    if server["endpoint"].rstrip("/") != config.url.rstrip("/"):
        raise ContextProbeError("server provenance endpoint does not match --url")
    if server["served_model"] != config.model:
        raise ContextProbeError("server provenance model does not match --model")

    endpoint = urlparse(config.url)
    tokenize_endpoint = urlparse(config.tokenize_url)
    if (
        endpoint.scheme not in {"http", "https"}
        or endpoint.netloc != tokenize_endpoint.netloc
        or endpoint.scheme != tokenize_endpoint.scheme
        or tokenize_endpoint.path.rstrip("/") != "/tokenize"
    ):
        raise ContextProbeError(
            "--tokenize-url must be the /tokenize endpoint on the provenance host"
        )

    runtime = server["runtime"]
    expected_runtime = {
        "max_model_len": config.expected_window,
        "kv_cache_dtype": config.expected_kv_dtype,
        "base_cache_gb": config.expected_base_gb,
        "delta_gb": config.expected_delta_gb,
        "delta_policy": config.expected_policy,
        "gate_tau": config.expected_tau,
    }
    for key, expected in expected_runtime.items():
        observed = runtime[key]
        matches = (
            _same_number(observed, expected)
            if isinstance(expected, (int, float))
            else str(observed) == str(expected)
        )
        if not matches:
            raise ContextProbeError(
                f"server runtime {key}={observed!r}, expected {expected!r}"
            )
    if int(runtime["speculative_tokens"]) != 0:
        raise ContextProbeError("P2 context proof requires speculative_tokens=0")

    argv = server["runtime_argv"]
    if _runtime_has_option(argv, "--speculative-config"):
        raise ContextProbeError(
            "P2 context proof requires --speculative-config to be absent"
        )
    if int(_one_argv_value(argv, "--max-num-seqs")) != 1:
        raise ContextProbeError("P2 context proof requires --max-num-seqs 1")


def _case_secret(config: ProbeConfig, ordinal: int, case: CaseSpec) -> str:
    material = (
        f"{config.run_label}\0{config.seed}\0{ordinal}\0"
        f"{case.target_prompt_tokens}\0{case.depth:.12g}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16].upper()
    return f"DS4-{digest}"


def _filler(max_words: int, seed: int) -> list[str]:
    randomizer = random.Random(seed)
    return [randomizer.choice(WORDS) for _ in range(max_words)]


def _build_prompt(words: Sequence[str], n_words: int, depth: float, secret: str) -> str:
    at = max(0, min(n_words, int(n_words * depth)))
    before = " ".join(words[:at])
    after = " ".join(words[at:n_words])
    needle = f"IMPORTANT FACT: the vault passphrase is {secret}. Remember it exactly."
    document = f"{before}\n\n{needle}\n\n{after}"
    return (
        "Read the complete document and answer the question at the end.\n\n"
        f"<document>\n{document}\n</document>\n\n"
        "Question: What is the vault passphrase stated in the document? "
        "Reply with only the exact passphrase."
    )


def _chat_body(model: str, prompt: str, max_tokens: int, seed: int) -> dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"thinking": False},
    }


def _tokenize_body(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"thinking": False},
    }


def _parse_tokenize_payload(
    payload: Mapping[str, Any], expected_window: int
) -> tuple[int, int]:
    count = payload.get("count")
    max_model_len = payload.get("max_model_len")
    tokens = payload.get("tokens")
    if not isinstance(count, int) or count <= 0:
        raise ContextProbeError("/tokenize response has invalid count")
    if not isinstance(max_model_len, int) or max_model_len != expected_window:
        raise ContextProbeError(
            f"/tokenize max_model_len={max_model_len!r}, expected {expected_window}"
        )
    if not isinstance(tokens, list) or len(tokens) != count:
        raise ContextProbeError("/tokenize tokens length does not match count")
    return count, max_model_len


def _calibrate_prompt(
    config: ProbeConfig,
    case: CaseSpec,
    secret: str,
    case_seed: int,
    transport: Transport,
) -> tuple[str, int, int, list[dict[str, Any]]]:
    """Binary-search word count against the server-rendered chat token count."""

    target = case.target_prompt_tokens
    max_words = max(2048, target * 2)
    words = _filler(max_words, case_seed)
    low, high = 0, max_words
    best: tuple[int, int, str] | None = None  # absolute error, nwords, prompt
    trace: list[dict[str, Any]] = []
    selected_count = 0

    while low <= high:
        n_words = (low + high) // 2
        prompt = _build_prompt(words, n_words, case.depth, secret)
        body = _tokenize_body(config.model, prompt)
        status, payload, wall = transport(
            config.tokenize_url, body, config.timeout_seconds
        )
        count, max_model_len = _parse_tokenize_payload(payload, config.expected_window)
        trace.append(
            {
                "n_words": n_words,
                "status_code": status,
                "prompt_tokens": count,
                "max_model_len": max_model_len,
                "wall_seconds": wall,
                "request_sha256": _canonical_hash(body),
            }
        )
        error = abs(count - target)
        if best is None or (error, n_words) < (best[0], best[1]):
            best = (error, n_words, prompt)
            selected_count = count
        if count < target:
            low = n_words + 1
        elif count > target:
            high = n_words - 1
        else:
            break

    if best is None:
        raise ContextProbeError("prompt calibration produced no tokenizer receipt")
    error, n_words, prompt = best
    if error > config.prompt_token_tolerance:
        raise ContextProbeError(
            f"could not calibrate target {target} within +/-"
            f"{config.prompt_token_tolerance} tokens; closest was {selected_count}"
        )
    if selected_count + config.max_tokens > config.expected_window:
        raise ContextProbeError(
            f"prompt {selected_count} + max_tokens {config.max_tokens} exceeds "
            f"window {config.expected_window}"
        )
    return prompt, selected_count, n_words, trace


def _extract_exact_answer(content: str) -> str:
    value = (content or "").strip()
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[1].strip()
    tagged = re.fullmatch(r"<answer>\s*([^<>\r\n]+?)\s*</answer>", value)
    if tagged is not None:
        return tagged.group(1).strip()
    final = re.fullmatch(r"(?:FINAL\s*:\s*)?([^\r\n]+?)", value, re.IGNORECASE)
    return final.group(1).strip() if final is not None else ""


def _completion_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    usage = payload.get("usage")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ContextProbeError("completion response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping) or not isinstance(
        choice.get("message"), Mapping
    ):
        raise ContextProbeError("completion choice has no message object")
    if not isinstance(usage, Mapping):
        raise ContextProbeError("completion response has no usage object")
    message = choice["message"]
    content = message.get("content")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(content, str) or not content.strip():
        raise ContextProbeError("completion content is empty")
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise ContextProbeError("completion usage.prompt_tokens is missing or invalid")
    if not isinstance(completion_tokens, int) or completion_tokens <= 0:
        raise ContextProbeError(
            "completion usage.completion_tokens is missing or invalid"
        )
    return {
        "response_id": payload.get("id"),
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "reasoning": message.get("reasoning") or message.get("reasoning_content") or "",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _run_case(
    config: ProbeConfig,
    ordinal: int,
    case: CaseSpec,
    transport: Transport,
) -> dict[str, Any]:
    secret = _case_secret(config, ordinal, case)
    case_seed = config.seed + ordinal - 1
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ordinal": ordinal,
        "started_at_utc": utc_now(),
        "generator_version": GENERATOR_VERSION,
        "seed": case_seed,
        "target_prompt_tokens": case.target_prompt_tokens,
        "prompt_token_tolerance": config.prompt_token_tolerance,
        "depth": case.depth,
        "expected_answer": secret,
        "accepted": False,
        "error": None,
    }
    try:
        prompt, tokenized_count, n_words, calibration = _calibrate_prompt(
            config, case, secret, case_seed, transport
        )
        receipt["calibration"] = {
            "selected_words": n_words,
            "observed_prompt_tokens": tokenized_count,
            "absolute_error_tokens": abs(tokenized_count - case.target_prompt_tokens),
            "iterations": calibration,
        }
        body = _chat_body(config.model, prompt, config.max_tokens, case_seed)
        status, payload, wall = transport(config.url, body, config.timeout_seconds)
        fields = _completion_fields(payload)
        extracted = _extract_exact_answer(fields["content"])
        receipt["request"] = {
            "request_sha256": _canonical_hash(body),
            "prompt_sha256": sha256_text(prompt),
            "prompt_bytes": len(prompt.encode("utf-8")),
            "max_tokens": config.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "thinking": False,
        }
        receipt["response"] = {
            **fields,
            "status_code": status,
            "wall_seconds": wall,
            "content_sha256": sha256_text(fields["content"]),
            "reasoning_sha256": sha256_text(fields["reasoning"]),
            "extracted_answer": extracted,
        }
        if fields["finish_reason"] != "stop":
            raise ContextProbeError(
                f"finish_reason={fields['finish_reason']!r}, expected 'stop'"
            )
        if fields["prompt_tokens"] != tokenized_count:
            raise ContextProbeError(
                "completion usage.prompt_tokens does not match the accepted "
                f"/tokenize count ({fields['prompt_tokens']} != {tokenized_count})"
            )
        if extracted != secret:
            raise ContextProbeError(
                f"expected exact terminal answer {secret!r}, got {extracted!r}"
            )
        receipt["accepted"] = True
    except Exception as error:  # noqa: BLE001 - receipt is the failure boundary
        receipt["error"] = f"{type(error).__name__}: {error}"
    receipt["completed_at_utc"] = utc_now()
    return receipt


def _build_manifest(
    config: ProbeConfig, server: Mapping[str, Any], manifest_path: Path, raw_path: Path
) -> dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "run_label": config.run_label,
        "status": "probing",
        "context_validated": False,
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "server": server,
        "server_provenance_sha256": sha256_file(config.server_provenance),
        "harness": {
            "python": sys.version,
            "argv": sys.argv,
            "files": {
                "context_probe.py": sha256_file(__file__),
                "harness.py": sha256_file(script_dir / "harness.py"),
            },
        },
        "probe": {
            "generator_version": GENERATOR_VERSION,
            "model": config.model,
            "url": config.url,
            "tokenize_url": config.tokenize_url,
            "seed": config.seed,
            "max_tokens": config.max_tokens,
            "timeout_seconds": config.timeout_seconds,
            "prompt_token_tolerance": config.prompt_token_tolerance,
            "expected_runtime": {
                "max_model_len": config.expected_window,
                "kv_cache_dtype": config.expected_kv_dtype,
                "base_cache_gb": config.expected_base_gb,
                "delta_gb": config.expected_delta_gb,
                "delta_policy": config.expected_policy,
                "gate_tau": config.expected_tau,
                "speculative_tokens": 0,
                "max_num_seqs": 1,
            },
            "cases": [asdict(case) for case in config.cases],
        },
        "artifacts": {
            "manifest": str(manifest_path.resolve()),
            "receipts_jsonl": str(raw_path.resolve()),
        },
        "summary": None,
        "error": None,
    }


def run(config: ProbeConfig, transport: Transport = _http_post) -> int:
    if not _RUN_LABEL_RE.fullmatch(config.run_label):
        raise ContextProbeError(f"unsafe run label: {config.run_label!r}")
    if not config.cases:
        raise ContextProbeError("at least one --case is required")
    if len(set(config.cases)) != len(config.cases):
        raise ContextProbeError("duplicate context cases are not allowed")
    if config.prompt_token_tolerance < 0:
        raise ContextProbeError("prompt-token tolerance must be non-negative")
    if config.max_tokens <= 0 or config.timeout_seconds <= 0:
        raise ContextProbeError("max tokens and timeout must be positive")
    for case in config.cases:
        if case.target_prompt_tokens <= 0 or not 0.0 <= case.depth <= 1.0:
            raise ContextProbeError(f"invalid context case: {case}")
        if case.target_prompt_tokens + config.max_tokens > config.expected_window:
            raise ContextProbeError(
                f"requested case {case.target_prompt_tokens} + max tokens "
                f"{config.max_tokens} exceeds window {config.expected_window}"
            )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{config.run_label}.manifest.json"
    raw_path = output_dir / f"{config.run_label}.context.jsonl"
    for path in (manifest_path, raw_path):
        if path.exists():
            raise ContextProbeError(f"refusing to overwrite artifact: {path}")

    server = load_server_provenance(config.server_provenance)
    validate_p2_server(server, config)
    manifest = _build_manifest(config, server, manifest_path, raw_path)
    write_json(manifest_path, manifest)

    accepted = 0
    try:
        for ordinal, case in enumerate(config.cases, start=1):
            receipt = _run_case(config, ordinal, case, transport)
            append_jsonl(raw_path, receipt)
            if not receipt["accepted"]:
                raise ContextProbeError(str(receipt["error"]))
            accepted += 1
        manifest["status"] = "complete"
        manifest["context_validated"] = True
        manifest["summary"] = {
            "accepted": accepted,
            "of": len(config.cases),
            "max_observed_prompt_tokens": max(
                json.loads(line)["calibration"]["observed_prompt_tokens"]
                for line in raw_path.read_text().splitlines()
            ),
            "receipts_sha256": sha256_file(raw_path),
        }
        return_code = 0
    except Exception as error:  # noqa: BLE001 - preserve an auditable manifest
        manifest["status"] = "failed"
        manifest["context_validated"] = False
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["summary"] = {
            "accepted": accepted,
            "of": len(config.cases),
            "receipts_sha256": sha256_file(raw_path) if raw_path.exists() else None,
        }
        return_code = 1
    manifest["completed_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    return return_code


def _parse_case(raw: str) -> CaseSpec:
    try:
        target_raw, depth_raw = raw.split(":", 1)
        value = CaseSpec(int(target_raw), float(depth_raw))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "case must be TARGET_PROMPT_TOKENS:DEPTH, e.g. 120000:0.5"
        ) from error
    if value.target_prompt_tokens <= 0 or not 0.0 <= value.depth <= 1.0:
        raise argparse.ArgumentTypeError(
            "case target must be positive and depth in [0,1]"
        )
    return value


def parse_args() -> ProbeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-provenance", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--tokenize-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-window", type=int, required=True)
    parser.add_argument("--expected-kv-dtype", required=True)
    parser.add_argument("--expected-base-gb", type=float, required=True)
    parser.add_argument("--expected-delta-gb", type=float, required=True)
    parser.add_argument("--expected-policy", required=True)
    parser.add_argument("--expected-tau", type=float, required=True)
    parser.add_argument("--case", action="append", type=_parse_case, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt-token-tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    return ProbeConfig(
        server_provenance=args.server_provenance,
        output_dir=args.output_dir,
        run_label=args.run_label,
        url=args.url,
        tokenize_url=args.tokenize_url,
        model=args.model,
        expected_window=args.expected_window,
        expected_kv_dtype=args.expected_kv_dtype,
        expected_base_gb=args.expected_base_gb,
        expected_delta_gb=args.expected_delta_gb,
        expected_policy=args.expected_policy,
        expected_tau=args.expected_tau,
        cases=tuple(args.case),
        seed=args.seed,
        prompt_token_tolerance=args.prompt_token_tolerance,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
    )


def main() -> int:
    try:
        return run(parse_args())
    except HarnessError as error:
        print(f"context probe failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
