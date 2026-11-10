#!/usr/bin/env python3
"""A content-free skeleton of a /v1/responses request body: the structure that decides how the request is rendered
(item types, roles, names, namespaces, which output answers which call, how many content parts) with every text,
argument, output, instruction, schema and identifier replaced.

    python3 make_request_skeleton.py REQUEST.json > fixtures/NAME.json

What is kept: `type`, `role`, `name`, `namespace`, `status`, `phase` of items and content parts, the pairing of
`call_id`s (renumbered call_1, call_2, ...), tool and namespace names, the request switches (`tool_choice`,
`parallel_tool_calls`, `reasoning`, `include`, `stream`, `store`). Everything else is "x", "{}" or dropped; item ids
are renumbered. A skeleton in fixtures/ is a regression case of test_k31_api_patch.py, which checks that it still
behaves as the request did: rejected by the unpatched code, rendered by the patched.
"""
import json
import sys

ITEM_KEYS = ("type", "role", "name", "namespace", "status", "phase")
SWITCHES = ("tool_choice", "parallel_tool_calls", "reasoning", "include", "stream", "store")


def parts(value, default_type):
    """Content parts: the part types, the text gone. A part that is not text becomes a text part."""
    if isinstance(value, str):
        return "x"
    if not isinstance(value, list):
        return value if value is None else "x"
    out = []
    for part in value:
        kind = part.get("type", default_type) if isinstance(part, dict) else default_type
        if kind not in ("input_text", "output_text", "reasoning_text", "summary_text", "text"):
            kind = default_type
        new = {"type": kind, "text": "x"}
        if kind == "output_text":
            new["annotations"] = []
        out.append(new)
    return out


def tool(entry):
    kind = entry.get("type")
    if kind == "function":
        return {"type": "function", "name": entry.get("name"), "description": "x",
                "parameters": {"type": "object", "properties": {}}}
    if kind == "namespace":
        return {"type": "namespace", "name": entry.get("name"), "description": "x",
                "tools": [tool(inner) for inner in entry.get("tools", [])]}
    if kind == "custom":
        return {"type": "custom", "name": entry.get("name"), "description": "x"}
    return {"type": kind}


def main() -> int:
    body = json.load(open(sys.argv[1]))
    calls: dict[str, str] = {}
    counts: dict[str, int] = {}

    def call_id(value):
        if value is None:
            return None
        return calls.setdefault(str(value), f"call_{len(calls) + 1}")

    def item_id(kind):
        counts[kind] = counts.get(kind, 0) + 1
        return f"{kind}_{counts[kind]}"

    items = []
    source = body.get("input")
    for item in source if isinstance(source, list) else []:
        kind = item.get("type", "message")
        new = {k: item[k] for k in ITEM_KEYS if item.get(k) is not None}
        new.setdefault("type", kind)
        if "id" in item:
            new["id"] = item_id({"message": "msg", "reasoning": "rs", "function_call": "fc",
                                 "function_call_output": "fco"}.get(kind, "item"))
        if "call_id" in item:
            new["call_id"] = call_id(item["call_id"])
        if kind == "message":
            new["content"] = parts(item.get("content"), "output_text" if item.get("role") == "assistant" else "input_text")
        elif kind == "reasoning":
            new["summary"] = parts(item.get("summary") or [], "summary_text")
            if item.get("content") is not None:
                new["content"] = parts(item["content"], "reasoning_text")
        elif kind in ("function_call", "custom_tool_call"):
            new["arguments" if kind == "function_call" else "input"] = "{}" if kind == "function_call" else "x"
        elif kind in ("function_call_output", "custom_tool_call_output"):
            new["output"] = parts(item.get("output"), "input_text")
        items.append(new)
    skeleton = {"input": items if isinstance(source, list) else "x"}
    if isinstance(body.get("instructions"), str):
        skeleton["instructions"] = "x"
    if body.get("tools"):
        skeleton["tools"] = [tool(entry) for entry in body["tools"]]
    for key in SWITCHES:
        if key in body:
            skeleton[key] = body[key]
    json.dump(skeleton, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
