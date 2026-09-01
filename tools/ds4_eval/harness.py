#!/usr/bin/env python3
"""Shared, dependency-free primitives for reproducible DS4-W2 evaluation.

The module deliberately contains no host-specific command. A caller may feed
it a log command, JSON snapshot, or future HTTP KPI endpoint, but tests and
offline scoring remain local and side-effect free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse


SCHEMA_VERSION = "ds4-w2-eval-v1"
WARMUP_SUITE_VERSION = "ds4-w2-prewarm-v4"
WARMUP_SEED = 20_260_711
WARMUP_TEMPERATURE = 0.0
WARMUP_TOP_P = 1.0
WARMUP_MAX_TOKENS = 150


@dataclass(frozen=True)
class WarmupCase:
    name: str
    prompt: str
    expected: str


# Fixed prompts and answers: the warm state must not depend on the eval seed.
# Spell out the output contract instead of using ``<answer>`` as a placeholder;
# some chat templates preserve those brackets literally in the completion.
_FINAL_INSTRUCTION = (
    " On the final line, write FINAL: followed by only the answer; "
    "do not include angle brackets."
)
WARMUP_CASES: tuple[WarmupCase, ...] = (
    WarmupCase("add", "Compute 2+2." + _FINAL_INSTRUCTION, "4"),
    WarmupCase("reverse", "Reverse the string abc." + _FINAL_INSTRUCTION, "cba"),
    WarmupCase("multiply", "Compute 5*6." + _FINAL_INSTRUCTION, "30"),
    WarmupCase("capital", "Name the capital of France." + _FINAL_INSTRUCTION, "Paris"),
    WarmupCase("prime", "Is 7 prime?" + _FINAL_INSTRUCTION, "yes"),
    WarmupCase("subtract", "Compute 31-9." + _FINAL_INSTRUCTION, "22"),
    WarmupCase("weekday", "What day follows Monday?" + _FINAL_INSTRUCTION, "Tuesday"),
    WarmupCase("power", "Compute 2^5." + _FINAL_INSTRUCTION, "32"),
    WarmupCase("division", "Compute 100/4." + _FINAL_INSTRUCTION, "25"),
    WarmupCase("reverse2", "Reverse the string hello." + _FINAL_INSTRUCTION, "olleh"),
)


class HarnessError(RuntimeError):
    """Base class for fail-closed harness errors."""


class ProvenanceError(HarnessError):
    """Server provenance is absent, ambiguous, or uses a placeholder."""


class WarmupError(HarnessError):
    """At least one deterministic prewarm request failed validation."""

    def __init__(self, message: str, receipts: Sequence[Mapping[str, Any]]):
        super().__init__(message)
        self.receipts = list(receipts)


class PoolGateError(HarnessError):
    """The requested pool-state readiness gate did not pass."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class RequestSpec:
    model: str
    prompt: str
    temperature: float
    top_p: float
    seed: int
    max_tokens: int
    timeout_seconds: int

    def request_hash(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256_text(body)


@dataclass(frozen=True)
class Completion:
    ok: bool
    content: str = ""
    status_code: int | None = None
    completion_tokens: int | None = None
    wall_seconds: float | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    error: str | None = None


def build_warmup_specs(model: str) -> list[tuple[WarmupCase, RequestSpec]]:
    return [
        (
            case,
            RequestSpec(
                model=model,
                prompt=case.prompt,
                temperature=WARMUP_TEMPERATURE,
                top_p=WARMUP_TOP_P,
                seed=WARMUP_SEED,
                max_tokens=WARMUP_MAX_TOKENS,
                timeout_seconds=200,
            ),
        )
        for case in WARMUP_CASES
    ]


_FINAL_RE = re.compile(
    r"(?:^|\n|</think>)\s*FINAL\s*:\s*([^\r\n]+?)\s*\Z", re.IGNORECASE
)
_ANSWER_TAG_RE = re.compile(
    r"(?:^|\n|</think>)\s*<answer>\s*([^\r\n<>]+?)\s*</answer>\s*\Z",
    re.IGNORECASE,
)


def _normalise_warm_answer(value: str) -> str:
    value = value.strip().strip("`*_ ").rstrip(".").strip()
    return re.sub(r"[^a-z0-9.+-]+", "", value.lower())


def _warmup_answer(content: str) -> str:
    value = content or ""
    final = _FINAL_RE.search(value)
    if final is not None:
        return final.group(1)
    # DeepSeek's reasoning chat template can place the final answer directly
    # after its ``</think>`` boundary or normalize an explicit ``FINAL:`` to
    # the native answer tag. The final marker must still be terminal and either
    # start the response or immediately follow that canonical boundary, so
    # arbitrary prefixes, suffixes, and duplicate tags remain fail-closed.
    tagged = _ANSWER_TAG_RE.search(value.strip())
    return tagged.group(1) if tagged is not None else ""


def _validate_warmup(case: WarmupCase, result: Completion) -> tuple[bool, str]:
    if not result.ok:
        return False, result.error or "request failed"
    if result.status_code is not None and not 200 <= result.status_code < 300:
        return False, f"http status {result.status_code}"
    if not result.content.strip():
        return False, "empty completion"
    if result.completion_tokens is None or result.completion_tokens <= 0:
        return False, "missing or zero completion_tokens"
    if result.finish_reason != "stop":
        return False, f"finish_reason={result.finish_reason!r}"
    got = _normalise_warm_answer(_warmup_answer(result.content))
    expected = _normalise_warm_answer(case.expected)
    if got != expected:
        return (
            False,
            f"expected FINAL: {case.expected!r}, got {_warmup_answer(result.content)!r}",
        )
    return True, "accepted"


ReceiptSink = Callable[[Mapping[str, Any]], None]
CompletionClient = Callable[[RequestSpec], Completion]


def run_prewarm(
    client: CompletionClient,
    model: str,
    receipt_sink: ReceiptSink | None = None,
) -> list[dict[str, Any]]:
    """Execute the fixed temp-0 prewarm and stop at the first bad receipt.

    Every completed attempt is emitted before an exception, so a failed run
    remains auditable instead of disappearing behind a retry or list
    comprehension.
    """

    receipts: list[dict[str, Any]] = []
    for ordinal, (case, spec) in enumerate(build_warmup_specs(model), start=1):
        started_at = utc_now()
        try:
            result = client(spec)
        except Exception as error:  # noqa: BLE001 - the failure still needs a receipt
            result = Completion(ok=False, error=f"{type(error).__name__}: {error}")
        accepted, reason = _validate_warmup(case, result)
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "suite_version": WARMUP_SUITE_VERSION,
            "ordinal": ordinal,
            "case": case.name,
            "prompt_sha256": sha256_text(case.prompt),
            "request_sha256": spec.request_hash(),
            "request": asdict(spec),
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "response": {
                "ok": result.ok,
                "status_code": result.status_code,
                "completion_tokens": result.completion_tokens,
                "wall_seconds": result.wall_seconds,
                "finish_reason": result.finish_reason,
                "response_id": result.response_id,
                "content": result.content,
                "content_sha256": sha256_text(result.content),
                "error": result.error,
            },
            "expected_final": case.expected,
            "extracted_final": _warmup_answer(result.content),
            "accepted": accepted,
            "reason": reason,
        }
        receipts.append(receipt)
        if receipt_sink is not None:
            receipt_sink(receipt)
        if not accepted:
            raise WarmupError(
                f"prewarm case {ordinal}/{len(WARMUP_CASES)} ({case.name}) failed: {reason}",
                receipts,
            )
    return receipts


REQUIRED_SERVER_PROVENANCE = {
    "target_host",
    "host_boot_id",
    "container_name",
    "container_id",
    "container_inspect_sha256",
    "server_started_at_utc",
    "served_model",
    "endpoint",
    "source_commit",
    "source_patch_sha256",
    "image_ref",
    "image_id",
    "checkpoint_fingerprint",
    "pack_fingerprint",
    "launcher_sha256",
    "runtime_argv",
    "w2_environment",
    "runtime",
}

REQUIRED_W2_ENVIRONMENT = {
    "VLLM_MOE_W2",
    "VLLM_MOE_W2_BASE_CACHE_GB",
    "VLLM_MOE_W2_DELTA_GB",
    "VLLM_MOE_W2_DELTA_POLICY",
    "VLLM_MOE_W2_GATE",
    "VLLM_MOE_W2_GATE_TAU",
}

REQUIRED_RUNTIME_PROVENANCE = {
    "max_model_len",
    "base_cache_gb",
    "delta_gb",
    "delta_policy",
    "gate_tau",
    "kv_cache_dtype",
    "speculative_tokens",
    "gpu_memory_utilization",
}

_PLACEHOLDERS = {"", "unknown", "unset", "none", "null", "todo", "tbd", "n/a"}
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$", re.IGNORECASE)


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normal = value.strip().lower()
        return normal in _PLACEHOLDERS or "<" in normal or ">" in normal
    return False


def _argv_option(
    argv: Sequence[str], name: str, *, required: bool = True
) -> str | None:
    """Return one unambiguous option value from a captured server argv."""

    values: list[str] = []
    for index, item in enumerate(argv):
        if item == name:
            if index + 1 >= len(argv):
                raise ProvenanceError(f"runtime_argv option {name} has no value")
            values.append(argv[index + 1])
        elif item.startswith(name + "="):
            values.append(item.split("=", 1)[1])
    if not values:
        if required:
            raise ProvenanceError(f"runtime_argv is missing required option {name}")
        return None
    if len(set(values)) != 1:
        raise ProvenanceError(f"runtime_argv has conflicting {name} values: {values}")
    if not values[0].strip():
        raise ProvenanceError(f"runtime_argv option {name} has an empty value")
    return values[0]


def _numbers_match(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return str(left) == str(right)


def _require_match(label: str, observed: Any, expected: Any) -> None:
    if not _numbers_match(observed, expected):
        raise ProvenanceError(
            f"server provenance mismatch: {label}={observed!r} != {expected!r}"
        )


def _validate_runtime_consistency(value: Mapping[str, Any]) -> None:
    runtime = value["runtime"]
    argv = value["runtime_argv"]
    environment = value["w2_environment"]

    env_runtime = {
        "VLLM_MOE_W2_BASE_CACHE_GB": "base_cache_gb",
        "VLLM_MOE_W2_DELTA_GB": "delta_gb",
        "VLLM_MOE_W2_DELTA_POLICY": "delta_policy",
        "VLLM_MOE_W2_GATE_TAU": "gate_tau",
    }
    for env_name, runtime_name in env_runtime.items():
        _require_match(
            f"w2_environment.{env_name}",
            environment[env_name],
            runtime[runtime_name],
        )
    for enabled_name in ("VLLM_MOE_W2", "VLLM_MOE_W2_GATE"):
        if str(environment[enabled_name]).strip() != "1":
            raise ProvenanceError(
                f"w2_environment.{enabled_name} must be explicitly set to '1'"
            )

    argv_runtime = {
        "--max-model-len": "max_model_len",
        "--gpu-memory-utilization": "gpu_memory_utilization",
        "--kv-cache-dtype": "kv_cache_dtype",
    }
    for option, runtime_name in argv_runtime.items():
        _require_match(
            f"runtime_argv.{option}",
            _argv_option(argv, option),
            runtime[runtime_name],
        )

    _require_match(
        "runtime_argv.--served-model-name",
        _argv_option(argv, "--served-model-name"),
        value["served_model"],
    )
    endpoint_port = urlparse(str(value["endpoint"])).port
    if endpoint_port is None:
        raise ProvenanceError(
            "server provenance endpoint must include an explicit port"
        )
    _require_match("runtime_argv.--port", _argv_option(argv, "--port"), endpoint_port)

    speculative_tokens = int(runtime["speculative_tokens"])
    speculative_json = _argv_option(
        argv, "--speculative-config", required=speculative_tokens != 0
    )
    if speculative_json is not None:
        try:
            speculative = json.loads(speculative_json)
        except json.JSONDecodeError as error:
            raise ProvenanceError(
                "runtime_argv --speculative-config must be valid JSON"
            ) from error
        if not isinstance(speculative, Mapping):
            raise ProvenanceError(
                "runtime_argv --speculative-config must contain a JSON object"
            )
        _require_match(
            "runtime_argv.--speculative-config.num_speculative_tokens",
            speculative.get("num_speculative_tokens"),
            speculative_tokens,
        )


def validate_server_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_SERVER_PROVENANCE - set(value))
    if missing:
        raise ProvenanceError(f"missing server provenance fields: {', '.join(missing)}")
    bad = sorted(
        key for key in REQUIRED_SERVER_PROVENANCE if _is_placeholder(value[key])
    )
    if bad:
        raise ProvenanceError(f"placeholder server provenance fields: {', '.join(bad)}")
    if not _SHA256_RE.fullmatch(str(value["source_patch_sha256"])):
        raise ProvenanceError(
            "source_patch_sha256 must be a full SHA-256 digest of the applied source diff"
        )
    runtime = value.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ProvenanceError("server provenance runtime must be an object")
    missing_runtime = sorted(REQUIRED_RUNTIME_PROVENANCE - set(runtime))
    if missing_runtime:
        raise ProvenanceError(
            f"missing runtime provenance fields: {', '.join(missing_runtime)}"
        )
    bad_runtime = sorted(
        key for key in REQUIRED_RUNTIME_PROVENANCE if _is_placeholder(runtime[key])
    )
    if bad_runtime:
        raise ProvenanceError(
            f"placeholder runtime provenance fields: {', '.join(bad_runtime)}"
        )
    runtime_argv = value.get("runtime_argv")
    if (
        not isinstance(runtime_argv, list)
        or not runtime_argv
        or not all(isinstance(item, str) and item.strip() for item in runtime_argv)
    ):
        raise ProvenanceError("runtime_argv must be a non-empty string array")
    w2_environment = value.get("w2_environment")
    if not isinstance(w2_environment, Mapping) or not w2_environment:
        raise ProvenanceError("w2_environment must be a non-empty object")
    missing_environment = sorted(REQUIRED_W2_ENVIRONMENT - set(w2_environment))
    if missing_environment:
        raise ProvenanceError(
            "missing w2_environment fields: " + ", ".join(missing_environment)
        )
    placeholder_env = sorted(
        key for key, item in w2_environment.items() if _is_placeholder(item)
    )
    if placeholder_env:
        raise ProvenanceError(
            f"placeholder w2_environment fields: {', '.join(placeholder_env)}"
        )
    _validate_runtime_consistency(value)
    return json.loads(json.dumps(value, sort_keys=True))


def load_server_provenance(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ProvenanceError("server provenance root must be an object")
    return validate_server_provenance(value)


@dataclass(frozen=True)
class PoolSnapshot:
    captured_at_utc: str
    source_sha256: str
    fp4_tick: int | None = None
    fp4_cached: int | None = None
    fp4_slots: int | None = None
    fp4_token_hit_pct: float | None = None
    fp4_expert_hit_pct: float | None = None
    fp4_window_promoted: int | None = None
    fp4_window_evicted: int | None = None
    fp4_total_promoted: int | None = None
    fp4_total_evicted: int | None = None
    base_replay_pct: float | None = None
    base_replay_steps: int | None = None
    base_cumulative_replay_pct: float | None = None
    base_cumulative_steps: int | None = None
    base_unrestored_experts: int | None = None
    base_fp_residue_steps: int | None = None
    gate_steps: int | None = None
    gate_fired: int | None = None

    @property
    def fp4_occupancy(self) -> float | None:
        if self.fp4_cached is None or not self.fp4_slots:
            return None
        return self.fp4_cached / self.fp4_slots

    @property
    def gate_fire_rate(self) -> float | None:
        if self.gate_fired is None or not self.gate_steps:
            return None
        return self.gate_fired / self.gate_steps

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["fp4_occupancy"] = self.fp4_occupancy
        value["gate_fire_rate"] = self.gate_fire_rate
        return value


_FP4_TICK_RE = re.compile(
    r"\[fp4\]\s+tick\s+(?P<tick>\d+):\s+"
    r"(?P<cached>\d+)/(?P<slots>\d+)\s+slots.*?"
    r"hit-rate\s+(?P<token_hit>[\d.]+)%\s+tokens\s+/\s+"
    r"(?P<expert_hit>[\d.]+)%\s+experts;\s+"
    r"window\s+\+(?P<win_promoted>\d+)/-(?P<win_evicted>\d+),\s+"
    r"cumulative\s+\+(?P<promoted>\d+)/-(?P<evicted>\d+)",
    re.IGNORECASE,
)

_BASE_KPI_RE = re.compile(
    r"\[base\]\s+KPI:\s+replay\s+(?P<replay>[\d.]+)%\s+of\s+last\s+"
    r"(?P<steps>\d+)\s+steps.*?cumulative\s+"
    r"(?P<cumulative>[\d.]+)%\s+of\s+(?P<cumulative_steps>\d+)",
    re.IGNORECASE,
)

_UNRESTORED_RE = re.compile(r"UNRESTORED\s+experts:\s*(\d+)", re.IGNORECASE)
_FP_RESIDUE_RE = re.compile(r"fp-residue:\s*(\d+)\s+steps", re.IGNORECASE)


def parse_pool_log(text: str) -> PoolSnapshot:
    """Parse the latest FP4 tick and base replay KPI from existing logs."""

    fp4 = list(_FP4_TICK_RE.finditer(text))
    base = list(_BASE_KPI_RE.finditer(text))
    f = fp4[-1].groupdict() if fp4 else {}
    b = base[-1].groupdict() if base else {}
    tail = ""
    # The qualifiers are appended to the KPI line after the core regex. Search
    # the physical line so a previous window cannot leak into this snapshot.
    if base:
        start = base[-1].start()
        tail = text[
            start : text.find("\n", start) if "\n" in text[start:] else len(text)
        ]
    unrestored = _UNRESTORED_RE.search(tail)
    fp_residue = _FP_RESIDUE_RE.search(tail)
    return PoolSnapshot(
        captured_at_utc=utc_now(),
        source_sha256=sha256_text(text),
        fp4_tick=int(f["tick"]) if f else None,
        fp4_cached=int(f["cached"]) if f else None,
        fp4_slots=int(f["slots"]) if f else None,
        fp4_token_hit_pct=float(f["token_hit"]) if f else None,
        fp4_expert_hit_pct=float(f["expert_hit"]) if f else None,
        fp4_window_promoted=int(f["win_promoted"]) if f else None,
        fp4_window_evicted=int(f["win_evicted"]) if f else None,
        fp4_total_promoted=int(f["promoted"]) if f else None,
        fp4_total_evicted=int(f["evicted"]) if f else None,
        base_replay_pct=float(b["replay"]) if b else None,
        base_replay_steps=int(b["steps"]) if b else None,
        base_cumulative_replay_pct=float(b["cumulative"]) if b else None,
        base_cumulative_steps=int(b["cumulative_steps"]) if b else None,
        base_unrestored_experts=int(unrestored.group(1))
        if unrestored
        else 0
        if b
        else None,
        base_fp_residue_steps=int(fp_residue.group(1))
        if fp_residue
        else 0
        if b
        else None,
    )


def parse_pool_json(
    value: Mapping[str, Any], source_text: str | None = None
) -> PoolSnapshot:
    """Parse the current delta dump, or a future combined KPI JSON endpoint.

    Current ``moe_w2_delta._dump`` files are flat (tick/n_slots/cached/etc.).
    A future endpoint may provide ``fp4``, ``base``, and ``gate`` objects; this
    parser intentionally accepts both so the gate does not need redesign.
    """

    fp4 = value.get("fp4", value)
    base = value.get("base", {})
    gate = value.get("gate", {})
    if (
        not isinstance(fp4, Mapping)
        or not isinstance(base, Mapping)
        or not isinstance(gate, Mapping)
    ):
        raise PoolGateError("KPI JSON sections must be objects")
    raw = source_text if source_text is not None else json.dumps(value, sort_keys=True)
    return PoolSnapshot(
        captured_at_utc=utc_now(),
        source_sha256=sha256_text(raw),
        fp4_tick=_optional_int(fp4.get("tick")),
        fp4_cached=_optional_int(fp4.get("cached")),
        fp4_slots=_optional_int(fp4.get("slots", fp4.get("n_slots"))),
        fp4_token_hit_pct=_optional_float(fp4.get("token_hit_pct")),
        fp4_expert_hit_pct=_optional_float(fp4.get("expert_hit_pct")),
        fp4_total_promoted=_optional_int(fp4.get("promoted_total")),
        fp4_total_evicted=_optional_int(fp4.get("evicted_total")),
        base_replay_pct=_optional_float(base.get("replay_pct")),
        base_replay_steps=_optional_int(base.get("replay_steps")),
        base_cumulative_replay_pct=_optional_float(base.get("cumulative_replay_pct")),
        base_cumulative_steps=_optional_int(base.get("cumulative_steps")),
        base_unrestored_experts=_optional_int(base.get("unrestored_experts")),
        base_fp_residue_steps=_optional_int(base.get("fp_residue_steps")),
        gate_steps=_optional_int(gate.get("steps")),
        gate_fired=_optional_int(gate.get("fired")),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class PoolGatePolicy:
    min_fp4_tick: int | None = None
    min_fp4_cached: int | None = None
    min_fp4_occupancy: float | None = None
    min_fp4_total_promoted: int | None = None
    min_fp4_total_evicted: int | None = None
    min_fp4_total_evicted_delta: int | None = None
    max_base_replay_pct: float | None = None
    max_base_cumulative_replay_pct: float | None = None
    max_base_unrestored_experts: int | None = None
    max_base_fp_residue_steps: int | None = None
    min_gate_steps: int | None = None
    min_gate_fire_rate: float | None = None
    max_gate_fire_rate: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PoolGatePolicy":
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise PoolGateError(
                f"unknown pool gate policy fields: {', '.join(unknown)}"
            )
        policy = cls(**value)
        if all(v is None for v in asdict(policy).values()):
            raise PoolGateError("pool gate policy has no checks")
        if (
            policy.min_fp4_occupancy is not None
            and not 0 <= policy.min_fp4_occupancy <= 1
        ):
            raise PoolGateError("min_fp4_occupancy must be in [0, 1]")
        for name in ("min_gate_fire_rate", "max_gate_fire_rate"):
            threshold = getattr(policy, name)
            if threshold is not None and not 0 <= threshold <= 1:
                raise PoolGateError(f"{name} must be in [0, 1]")
        for name in (
            "min_fp4_tick",
            "min_fp4_cached",
            "min_fp4_total_promoted",
            "min_fp4_total_evicted",
            "min_fp4_total_evicted_delta",
            "min_gate_steps",
        ):
            threshold = getattr(policy, name)
            if threshold is not None and threshold < 0:
                raise PoolGateError(f"{name} must be non-negative")
        for name in (
            "max_base_replay_pct",
            "max_base_cumulative_replay_pct",
        ):
            threshold = getattr(policy, name)
            if threshold is not None and not 0 <= threshold <= 100:
                raise PoolGateError(f"{name} must be in [0, 100]")
        if (
            policy.min_gate_fire_rate is not None
            and policy.max_gate_fire_rate is not None
            and policy.min_gate_fire_rate > policy.max_gate_fire_rate
        ):
            raise PoolGateError("min_gate_fire_rate cannot exceed max_gate_fire_rate")
        return policy


def evaluate_pool_gate(
    snapshot: PoolSnapshot,
    policy: PoolGatePolicy,
    *,
    baseline: PoolSnapshot | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def at_least(name: str, observed: Any, threshold: Any) -> None:
        if threshold is None:
            return
        passed = observed is not None and observed >= threshold
        checks.append(
            {
                "metric": name,
                "operator": ">=",
                "threshold": threshold,
                "observed": observed,
                "passed": passed,
            }
        )

    def at_most(name: str, observed: Any, threshold: Any) -> None:
        if threshold is None:
            return
        passed = observed is not None and observed <= threshold
        checks.append(
            {
                "metric": name,
                "operator": "<=",
                "threshold": threshold,
                "observed": observed,
                "passed": passed,
            }
        )

    at_least("fp4_tick", snapshot.fp4_tick, policy.min_fp4_tick)
    at_least("fp4_cached", snapshot.fp4_cached, policy.min_fp4_cached)
    at_least("fp4_occupancy", snapshot.fp4_occupancy, policy.min_fp4_occupancy)
    at_least(
        "fp4_total_promoted", snapshot.fp4_total_promoted, policy.min_fp4_total_promoted
    )
    at_least(
        "fp4_total_evicted", snapshot.fp4_total_evicted, policy.min_fp4_total_evicted
    )
    if policy.min_fp4_total_evicted_delta is not None and baseline is not None:
        evicted_delta = (
            snapshot.fp4_total_evicted - baseline.fp4_total_evicted
            if snapshot.fp4_total_evicted is not None
            and baseline.fp4_total_evicted is not None
            else None
        )
        at_least(
            "fp4_total_evicted_delta",
            evicted_delta,
            policy.min_fp4_total_evicted_delta,
        )
    at_most("base_replay_pct", snapshot.base_replay_pct, policy.max_base_replay_pct)
    at_most(
        "base_cumulative_replay_pct",
        snapshot.base_cumulative_replay_pct,
        policy.max_base_cumulative_replay_pct,
    )
    at_most(
        "base_unrestored_experts",
        snapshot.base_unrestored_experts,
        policy.max_base_unrestored_experts,
    )
    at_most(
        "base_fp_residue_steps",
        snapshot.base_fp_residue_steps,
        policy.max_base_fp_residue_steps,
    )
    at_least("gate_steps", snapshot.gate_steps, policy.min_gate_steps)
    at_least("gate_fire_rate", snapshot.gate_fire_rate, policy.min_gate_fire_rate)
    at_most("gate_fire_rate", snapshot.gate_fire_rate, policy.max_gate_fire_rate)
    return {
        "configured": True,
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "policy": asdict(policy),
        "snapshot": snapshot.to_dict(),
        "baseline_snapshot": baseline.to_dict() if baseline is not None else None,
        "deferred_checks": ["fp4_total_evicted_delta"]
        if baseline is None and policy.min_fp4_total_evicted_delta is not None
        else [],
        "checks": checks,
    }


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(target)


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def interleaved_items(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(item) for item in items]
    ids = [str(row.get("id", "")) for row in rows]
    if any(not item_id for item_id in ids):
        raise HarnessError("every item requires a non-empty id")
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise HarnessError(f"duplicate item ids: {', '.join(duplicates)}")
    reasoning = [row for row in rows if row.get("cat") == "reasoning"]
    coding = [row for row in rows if row.get("cat") == "coding"]
    if len(reasoning) != len(coding) or len(reasoning) + len(coding) != len(rows):
        raise HarnessError("items must contain equal reasoning and coding groups only")
    order: list[dict[str, Any]] = []
    for r_item, c_item in zip(reasoning, coding):
        order.extend((r_item, c_item))
    return order
