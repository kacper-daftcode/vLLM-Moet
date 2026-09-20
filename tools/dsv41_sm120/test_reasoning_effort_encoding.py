#!/usr/bin/env python3
"""Verify the DeepSeek-V4.1 prompt encoder vLLM serves against the checkpoint's.

Run inside the image (the installed vLLM is the subject). Two levels:

  1. Always: the reasoning-effort table vLLM renders must be the official one
     (low 50 / high 75 / max 100, default high) and a thinking-mode request
     without an explicit effort must render "Reasoning Effort: 75".
  2. With --model-dir (checkpoint directory that carries encoding/): load the
     reference encoder deepseek ships (encoding/encoding.py) and require
     byte-identical prompts for (a) the encoding/tests goldens, encoded through
     the same message normalisation the vLLM tokenizer wrapper applies, and
     (b) a matrix of thinking-mode x reasoning_effort x tools x multi-turn
     conversations. OpenAI aliases (minimal / medium / xhigh) are not DeepSeek
     tiers, so they are only checked for the budget they render.

Exit status 1 on any mismatch. Usage:
    python3 test_reasoning_effort_encoding.py [--model-dir /model]
"""

from __future__ import annotations

import argparse
import copy
import difflib
import importlib.util
import json
import sys
from pathlib import Path

OFFICIAL = {"low": 50, "high": 75, "max": 100}
BUDGET_PREFIX = "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: "

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

CONVERSATIONS: dict[str, list[dict]] = {
    "system+user": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Ile to 17*19?"},
    ],
    "user-only": [{"role": "user", "content": "Napisz haiku o jesieni."}],
    "system+user+tools": [
        {"role": "system", "content": "You can call tools.", "tools": [TOOL]},
        {"role": "user", "content": "Jaka jest pogoda w Krakowie?"},
    ],
    "multi-turn with reasoning": [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "2+2?"},
        {
            "role": "assistant",
            "reasoning_content": "Two plus two is four.",
            "content": "4",
        },
        {"role": "user", "content": "And times 3?"},
    ],
    "tool call round-trip": [
        {"role": "system", "content": "You can call tools.", "tools": [TOOL]},
        {"role": "user", "content": "Pogoda w Gdańsku?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Gdańsk", "unit": "celsius"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 14}"},
    ],
}

EFFORTS: list[str | int | None] = [None, "low", "high", "max", 1, 10, 62, 75, 100]


def load_reference(model_dir: Path):
    path = model_dir / "encoding" / "encoding.py"
    if not path.exists():
        raise SystemExit(f"reference encoder not found: {path}")
    spec = importlib.util.spec_from_file_location("dsv41_reference_encoding", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalize_like_vllm(messages: list[dict]) -> list[dict]:
    """The vLLM tokenizer wrapper's normalisation (content blocks -> text with
    image placeholders, `reasoning` -> `reasoning_content`). Falls back to a local
    copy of that logic for goldens using roles the OpenAI API does not carry
    (e.g. latest_reminder), which the wrapper rejects before encoding."""
    from vllm.tokenizers.deepseek_v41 import _normalize_messages
    from vllm.tokenizers.deepseek_v41_encoding import IMAGE_PLACEHOLDER

    try:
        return _normalize_messages(copy.deepcopy(messages))
    except ValueError as exc:
        if "Invalid role" not in str(exc):
            raise
    result = [dict(m) for m in copy.deepcopy(messages)]
    for message in result:
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") in ("image_url", "input_image", "image_pil", "image"):
                    parts.append(IMAGE_PLACEHOLDER)
                else:
                    raise ValueError(f"unsupported content block {block.get('type')!r}")
            message["content"] = "\n\n".join(parts)
    return result


def budget_of(prompt: str) -> int | None:
    if not prompt.startswith(BUDGET_PREFIX):
        return None
    rest = prompt[len(BUDGET_PREFIX) :]
    return int(rest.split(" ", 1)[0])


def show_diff(gold: str, got: str, limit: int = 8) -> str:
    lines = list(
        difflib.unified_diff(
            gold.splitlines(), got.splitlines(), "reference", "vllm", lineterm="", n=0
        )
    )
    return "\n".join("        " + line[:200] for line in lines[:limit])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=None)
    args = ap.parse_args()

    from vllm.tokenizers import deepseek_v41_encoding as venc

    failures: list[str] = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        print(("PASS " if ok else "FAIL ") + what)
        if not ok:
            if detail:
                print(detail)
            failures.append(what)

    # ---- 1. tiers -----------------------------------------------------------
    table = dict(venc.REASONING_EFFORT_MAPPINGS)
    check(
        all(table.get(k) == v for k, v in OFFICIAL.items()),
        f"official tiers low 50 / high 75 / max 100 (vLLM table: {table})",
    )
    check(venc.DEFAULT_REASONING_EFFORT == "high", "default tier is high")
    default_prompt = venc.encode_messages(
        normalize_like_vllm(copy.deepcopy(CONVERSATIONS["system+user"])),
        thinking_mode="thinking",
        reasoning_effort="high",  # the tokenizer wrapper substitutes this for None
    )
    check(
        budget_of(default_prompt) == 75,
        f"thinking-mode default renders budget 75 (got {budget_of(default_prompt)})",
    )
    for alias in ("minimal", "medium", "xhigh"):
        if alias in table:
            prompt = venc.encode_messages(
                normalize_like_vllm(copy.deepcopy(CONVERSATIONS["user-only"])),
                thinking_mode="thinking",
                reasoning_effort=alias,
            )
            check(budget_of(prompt) == table[alias], f"alias {alias} renders {table[alias]}")
    chat_prompt = venc.encode_messages(
        normalize_like_vllm(copy.deepcopy(CONVERSATIONS["system+user"])),
        thinking_mode="chat",
        reasoning_effort="high",
    )
    check("Reasoning Effort" not in chat_prompt, "chat mode renders no effort prefix")

    if args.model_dir is None:
        print("(no --model-dir: reference comparison skipped)")
        return 1 if failures else 0

    # ---- 2. reference encoder ----------------------------------------------
    ref = load_reference(args.model_dir)
    ref_table = dict(ref.REASONING_EFFORT_MAPPINGS)
    check(
        all(table.get(k) == v for k, v in ref_table.items()),
        f"vLLM table is a superset of the checkpoint table {ref_table}",
    )

    # (a) goldens shipped with the checkpoint
    tests_dir = args.model_dir / "encoding" / "tests"
    for input_file in sorted(tests_dir.glob("test_input_*.json")):
        case_id = input_file.stem.split("_")[-1]
        gold = (tests_dir / f"test_output_{case_id}.txt").read_text()
        case = ref.load_cases(str(input_file))[0]
        ref_prompt, _ = ref.encode_case(case, thinking_mode="chat")
        check(ref_prompt == gold, f"golden {case_id}: reference encoder reproduces the golden")
        got = venc.encode_messages(
            normalize_like_vllm(copy.deepcopy(case["messages"])),
            thinking_mode=case.get("thinking_mode") or "chat",
            context=case.get("context"),
            reasoning_effort=case.get("reasoning_effort"),
        )
        check(
            got == gold,
            f"golden {case_id}: vLLM encoder == golden "
            f"(thinking_mode={case.get('thinking_mode')}, effort={case.get('reasoning_effort')})",
            show_diff(gold, got),
        )

    # (b) matrix
    n_cases = 0
    for name, messages in CONVERSATIONS.items():
        for thinking_mode in ("chat", "thinking"):
            for effort in EFFORTS:
                n_cases += 1
                ref_prompt = ref.encode_messages(
                    copy.deepcopy(messages),
                    thinking_mode=thinking_mode,
                    reasoning_effort=effort,
                )
                got = venc.encode_messages(
                    normalize_like_vllm(copy.deepcopy(messages)),
                    thinking_mode=thinking_mode,
                    reasoning_effort=effort,
                )
                if got != ref_prompt:
                    check(
                        False,
                        f"matrix: {name} / {thinking_mode} / effort={effort!r}",
                        show_diff(ref_prompt, got),
                    )
    check(
        not any(f.startswith("matrix:") for f in failures),
        f"matrix: {n_cases} conversation x mode x effort cases identical to the reference",
    )

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
