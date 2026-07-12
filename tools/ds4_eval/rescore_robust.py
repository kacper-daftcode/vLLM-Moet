#!/usr/bin/env python3
"""Robust, fail-closed re-scorer for DS4-W2 raw JSONL.

The original runner emitted ``ct`` while the standalone scorer looked only at
``completion_tokens``.  ``completion_token_count`` is now the single alias
boundary used by both online and offline scoring.
"""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


class ScoreDataError(ValueError):
    pass


def completion_token_count(row: Mapping[str, Any]) -> int | None:
    values: dict[str, int] = {}
    for key in ("completion_tokens", "ct"):
        if row.get(key) is not None:
            values[key] = int(row[key])
    usage = row.get("usage")
    if isinstance(usage, Mapping) and usage.get("completion_tokens") is not None:
        values["usage.completion_tokens"] = int(usage["completion_tokens"])
    if not values:
        return None
    if len(set(values.values())) != 1:
        raise ScoreDataError(f"conflicting completion token fields: {values}")
    return next(iter(values.values()))


def norm(value: Any) -> str:
    text = (
        str(value or "").strip().replace("$", "").strip("`").strip().rstrip(".").strip()
    )
    text = re.sub(r"\\d?frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\*\*", "", text)
    for quote in ('"', "'"):
        if text.startswith(quote) and text.endswith(quote) and len(text) > 1:
            text = text[1:-1]
    return text.strip()


def extract_final(content: str) -> str:
    if not content:
        return ""
    matches = re.findall(r"FINAL\s*:\s*([^\n]+)", content, re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def matches(expected: Any, got: Any) -> bool:
    e, g = norm(expected), norm(got)
    if not g:
        return False
    if (
        e.lower() == g.lower()
        or re.sub(r"\s+", "", e).lower() == re.sub(r"\s+", "", g).lower()
    ):
        return True
    try:
        expected_number = float(e)
        match = re.search(r"-?\d+(?:\.\d+)?", g.replace(",", ""))
        if match:
            return abs(float(match.group(0)) - expected_number) < 1e-6
    except ValueError:
        pass
    expected_fraction = re.fullmatch(r"(-?\d+)/(\d+)", e)
    if expected_fraction:
        try:
            expected_number = int(expected_fraction.group(1)) / int(
                expected_fraction.group(2)
            )
            match = re.search(r"-?\d+(?:\.\d+)?", g)
            if match:
                return abs(expected_number - float(match.group(0))) < 1e-9
            got_fraction = re.fullmatch(r"(-?\d+)/(\d+)", g)
            if got_fraction:
                got_number = int(got_fraction.group(1)) / int(got_fraction.group(2))
                return abs(expected_number - got_number) < 1e-9
        except (ValueError, ZeroDivisionError):
            pass
    return False


def lenient_matches(expected: Any, content: str) -> bool:
    """Diagnostic only: the expected answer appears somewhere in a non-sink.

    This preserves the original extraction-gap diagnostic without treating it
    as clean correctness. Numeric expectations are compared against every
    number in the response, rather than only the first incidental number.
    """

    expected_text = norm(expected)
    if expected_text.lower() in (content or "").lower():
        return True
    try:
        expected_number = float(expected_text)
    except ValueError:
        return False
    numbers = re.findall(r"-?\d+(?:\.\d+)?", (content or "").replace(",", ""))
    return any(abs(float(number) - expected_number) < 1e-6 for number in numbers)


def sink(content: str, completion_tokens: int | None = None) -> tuple[bool, str]:
    if completion_tokens is not None and completion_tokens >= 690:
        return True, "MAXTOK-noncompletion"
    words = (content or "").split()
    if len(words) < 12:
        return False, ""
    for ngram_size in (4, 3):
        grams = [
            " ".join(words[index : index + ngram_size])
            for index in range(len(words) - ngram_size)
        ]
        if grams:
            top, count = Counter(grams).most_common(1)[0]
            if count >= 6:
                return True, f"{ngram_size}gram x{count}: {top[:40]!r}"
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    if len(lines) >= 6:
        top, count = Counter(lines).most_common(1)[0]
        if count / len(lines) > 0.35 and count >= 4:
            return True, f"line x{count}/{len(lines)}: {top[:40]!r}"
    if len(words) >= 60 and len(set(words)) < len(words) / 6:
        return True, f"vocab {len(set(words))}/{len(words)}"
    if re.search(
        r"(?:<answer>\s*){3,}|(?:</think>\s*){3,}|(?:\bh\?){4,}|"
        r"(?P<spew>[|~?])(?P=spew){6,}",
        content or "",
    ):
        return True, "special-token spew"
    return False, ""


def score_rows(
    rows: Iterable[Mapping[str, Any]], expected_count: int | None = None
) -> dict[str, Any]:
    source_rows = [dict(row) for row in rows]
    if expected_count is not None and len(source_rows) != expected_count:
        raise ScoreDataError(
            f"expected {expected_count} rows, found {len(source_rows)}"
        )
    ids = [str(row.get("id", "")) for row in source_rows]
    if any(not item_id for item_id in ids):
        raise ScoreDataError("every row requires a non-empty id")
    duplicate_ids = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicate_ids:
        raise ScoreDataError(f"duplicate ids: {', '.join(duplicate_ids)}")
    out: list[dict[str, Any]] = []
    for row in source_rows:
        if "expected" not in row:
            raise ScoreDataError(f"row {row['id']} is missing expected")
        content = str(row.get("content", ""))
        token_count = completion_token_count(row)
        is_sink, reason = sink(content, token_count)
        final = extract_final(content)
        correct = (not is_sink) and matches(row["expected"], final)
        lenient = (not is_sink) and lenient_matches(row["expected"], content)
        out.append(
            {
                "id": row["id"],
                "cat": row.get("cat"),
                "correct": correct,
                "lenient": lenient,
                "sink": is_sink,
                "why": reason,
                "final": final,
                "expected": row["expected"],
                "completion_tokens": token_count,
            }
        )
    return {
        "n": len(out),
        "clean": sum(row["correct"] for row in out),
        "sinks": sum(row["sink"] for row in out),
        "lenient": sum(row["lenient"] for row in out),
        "reasoning_clean": sum(
            row["correct"] and row["cat"] == "reasoning" for row in out
        ),
        "coding_clean": sum(row["correct"] and row["cat"] == "coding" for row in out),
        "rows": out,
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ScoreDataError(f"{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ScoreDataError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def score_file(path: str | Path, expected_count: int | None = None) -> dict[str, Any]:
    return score_rows(load_jsonl(path), expected_count=expected_count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    for raw_path in args.paths:
        result = score_file(raw_path, expected_count=args.expected_count)
        if args.json:
            print(json.dumps({"path": raw_path, **result}, sort_keys=True))
            continue
        print(
            f"{Path(raw_path).name}: n={result['n']} CLEAN {result['clean']}/{result['n']} "
            f"(R {result['reasoning_clean']} / C {result['coding_clean']}) "
            f"SINKS {result['sinks']}/{result['n']}"
        )
        flagged = [(row["id"], row["why"]) for row in result["rows"] if row["sink"]]
        print(f"  sinks: {flagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
