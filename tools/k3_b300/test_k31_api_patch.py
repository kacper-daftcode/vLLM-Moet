#!/usr/bin/env python3
"""CPU test of step 4 of Dockerfile.b300-k31 (tools/k3_b300/k31/api.patch): what API clients send that Kimi K3's
encoding answers with HTTP 400 - the `developer` role and tool results it cannot name. Runs inside an image built
with the step, without a GPU; MODEL is a Kimi-K3.1-Pro checkpoint directory (its tokenizer and encoding_k3.py):

    docker run --rm --network none --entrypoint python3 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
      -v "$PWD/tools/k3_b300:/t:ro" -v "$MODEL:/model:ro" [-v "$STOCK:/stock:ro"] IMAGE \
      /t/test_k31_api_patch.py [--stock /stock] [--requests DIR] [--rollouts DIR]

    --stock DIR     the base image's two files as DIR/vllm/renderers/kimi_k3.py and
                    DIR/vllm/entrypoints/openai/responses/utils.py (tools/k3_b300/README.md shows how to copy them
                    out): adds every comparison with the unpatched code - what it rejects, and that the
                    conversations it accepts render to the same tokens
    --requests DIR  every *.json in DIR is a /v1/responses request body the unpatched server rejects
    --rollouts DIR  every rollout-*.jsonl under DIR (a Codex thread) must render to the same tokens as unpatched

The served path is reproduced with the server's own code: Responses `input` -> construct_input_messages ->
KimiK3Renderer.render_messages / render_messages_async (the real method bodies on a stand-in for `self`: the engine
config cannot be built without a GPU) -> parse_chat_messages -> the model's tokenizer (encoding_k3).

Checks: (R) the developer role renders as system; (A) conversations the unpatched code accepts; (B) a turn split into
several assistant messages and a result of an earlier turn render to the structure written next to each case;
(G) results that name their own tool and answer no call (Codex Desktop's companion page) render under that name, also
in front of an ordinary result; (C) a result with neither a call nor a name is still rejected; (D) 3000 random
conversations: no change where the encoding already resolves every block, a render with the right tool name on every
result everywhere else, nothing lost, idempotent; (E) the switch VLLM_MOET_K3_REJOIN_TOOL_RESULTS=0; (F) the
content-free skeletons of rejected requests in fixtures/.
Prints RESULT: PASS/FAIL, exit 0 only on PASS."""
import argparse
import asyncio
import copy
import glob
import importlib.util
import json
import os
import random
import re
import sys
import traceback
import types

NEEDLE = "resolvable tool name"
HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(cond, what):
    print(("OK   " if cond else "FAIL ") + what)
    if not cond:
        fails.append(what)


def load(name, path, env=None):
    saved = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return module


# ---------------------------------------------------------------- Responses items and chat messages
def user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def system(text, role="system"):
    return {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}


def msg(text):
    return {"type": "message", "role": "assistant", "status": "completed", "id": "msg_1",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def reasoning(text):
    return {"type": "reasoning", "id": "rs_1", "summary": [], "content": [{"type": "reasoning_text", "text": text}]}


def call(cid, name="exec_command"):
    return {"type": "function_call", "call_id": cid, "name": name, "arguments": json.dumps({"cmd": "run " + cid}),
            "status": "completed", "id": "fc_" + cid}


def out(cid):
    return {"type": "function_call_output", "call_id": cid, "output": "result-of-" + cid}


def named(tag, name="codex_apps_open_page", namespace="agent", **extra):
    """A result that names its own tool and answers no call (what Codex Desktop sends for its companion page)."""
    item = {"type": "function_call_output", "id": "fco_" + tag, "name": name, "output": "result-of-" + tag, **extra}
    if namespace:
        item["namespace"] = namespace
    return item


TOOLS = [{"type": "function", "name": n, "description": "d",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}
         for n in ("exec_command", "apply_patch", "view_image")]


def chat_call(cid, name="exec_command"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps({"cmd": "run " + cid})}}


def chat_tool(cid):
    return {"role": "tool", "tool_call_id": cid, "content": "result-of-" + cid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--stock")
    ap.add_argument("--requests")
    ap.add_argument("--rollouts")
    args = ap.parse_args()

    import vllm.entrypoints.openai.responses.utils as new_u
    import vllm.renderers.kimi_k3 as new_r
    from transformers import AutoTokenizer
    from vllm.config import ModelConfig
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.renderers.params import ChatParams

    patched_image = (hasattr(new_r, "_rejoin_k3_tool_results") and hasattr(new_r, "_normalize_k3_roles")
                     and "An output item may name its tool itself" in open(new_u.__file__).read())
    check(patched_image, "(0) the image carries step 4 (renderers/kimi_k3.py and entrypoints/openai/responses/utils.py)")
    if not patched_image:
        return
    off_r = load("vllm.renderers.kimi_k3_off", new_r.__file__, {"VLLM_MOET_K3_REJOIN_TOOL_RESULTS": "0"})
    stock_r = stock_u = None
    if args.stock:
        stock_r = load("vllm.renderers.kimi_k3_stock", os.path.join(args.stock, "vllm", "renderers", "kimi_k3.py"))
        stock_u = load("vllm.entrypoints.openai.responses.utils_stock",
                       os.path.join(args.stock, "vllm", "entrypoints", "openai", "responses", "utils.py"))
        check(not hasattr(stock_r, "_rejoin_k3_tool_results") and not hasattr(stock_r, "_normalize_k3_roles")
              and "An output item may name its tool itself" not in open(stock_u.__file__).read(),
              f"(0) --stock {args.stock}: the unpatched renderer and conversion")
    else:
        print("     no --stock: the comparisons with the unpatched code are skipped")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_config = ModelConfig(model=args.model, trust_remote_code=True)

    def stand_in(mod):
        me = types.SimpleNamespace(model_config=model_config, _thinking_efforts=mod._K31_THINKING_EFFORTS,
                                   get_tokenizer=lambda: tok)
        me._apply_chat_template = lambda conv, params: mod.KimiK3Renderer._apply_chat_template(me, conv, params)

        async def apply_async(conv, params):
            return me._apply_chat_template(conv, params)

        me._apply_chat_template_async = apply_async
        return me

    def render(mod, messages, tools=None, use_async=False):
        """(rendered conversation, token ids) through the renderer's own render_messages[_async]."""
        params = ChatParams(chat_template_content_format="string",
                            chat_template_kwargs={"add_generation_prompt": True, "tools": tools})
        me = stand_in(mod)
        if use_async:
            conv, prompt = asyncio.run(mod.KimiK3Renderer.render_messages_async(me, messages, params))
        else:
            conv, prompt = mod.KimiK3Renderer.render_messages(me, messages, params)
        return conv, prompt["prompt_token_ids"]

    def convert(conversion, body):
        """(chat messages, chat tool dicts) the Responses endpoint builds from a request body."""
        req = ResponsesRequest.model_validate({"model": "k", "stream": True, "store": False, **body})
        messages = conversion.construct_input_messages(request_instructions=req.instructions, request_input=req.input)
        return messages, (conversion.construct_tool_dicts(req.tools, req.tool_choice) if req.tools else None)

    def body_of(items):
        return {"instructions": "You are a coding agent.", "input": items, "tools": TOOLS}

    def outcome(conversion, renderer, body, use_async=False):
        """(conversation, tokens), or the error text of a rejected request."""
        try:
            messages, tools = body if isinstance(body, tuple) else convert(conversion, body)
            return render(renderer, messages, tools, use_async)
        except ValueError as exc:
            return str(exc)

    def unpatched(body):
        """The outcome with the base image's code, or None without --stock."""
        return outcome(stock_u, stock_r, body) if stock_r else None

    def patched(body, use_async=False):
        return outcome(new_u, new_r, body, use_async)

    def shape(ids):
        """[(message header, body)] of the user / assistant / tool messages of a prompt."""
        found = []
        for part in tok.decode(ids).split("<|open|>message ")[1:]:
            header, _, body = part.partition("<|sep|>")
            if not header.startswith('role="system"'):
                found.append((header, body.split("<|close|>message")[0]))
        return found

    def plain(body):
        return re.sub(r"<\|[a-z_]+\|>", "|", body)

    def rejected(result):
        return isinstance(result, str) and NEEDLE in result

    def rejected_before(result, label):
        if stock_r:
            check(rejected(result), f"{label}: rejected by the unpatched code")

    chat_tools = new_u.construct_tool_dicts(ResponsesRequest.model_validate({"model": "k", "input": "x", "tools": TOOLS}).tools, "auto")

    # ------------------------------------------------------------ (R) the developer role
    dev = [{"role": "developer", "content": "Follow the repository's instructions."}, {"role": "user", "content": "List the files."}]
    as_system = [{"role": "system", "content": dev[0]["content"]}, dev[1]]
    before = copy.deepcopy(dev)
    a, b = outcome(None, new_r, (dev, None)), outcome(None, new_r, (as_system, None))
    check(not isinstance(a, str) and a[1] == b[1], "(R) a developer message renders to the tokens of the same message as system")
    check(dev == before and [m["role"] for m in new_r._normalize_k3_roles(dev)] == ["system", "user"]
          and new_r._normalize_k3_roles(as_system) == as_system, "(R) the caller's messages are not mutated, other roles pass through")
    if stock_r:
        s = outcome(None, stock_r, (dev, None))
        check(isinstance(s, str) and "Unknown message role 'developer'" in s, "(R) the unpatched renderer rejects the developer role")
    codex = {"input": [system("You are a coding agent.", "developer"), user("Show the status."), reasoning("t1"), call("a"), out("a"),
                       system("Answer briefly.", "developer"), user("Summarize.")], "tools": TOOLS}
    c = patched(codex)
    text = tok.decode(c[1]) if not isinstance(c, str) else c
    check(not isinstance(c, str) and text.count('role="system"<|sep|>') == 2 and "developer" not in text.lower()
          and [h for h, _ in shape(c[1])] == ['role="user"', 'role="assistant"', 'role="tool" tool="exec_command" index="1"', 'role="user"', 'role="assistant"'],
          "(R) a Responses input with two developer messages around a tool call: two system messages, no developer in the prompt")

    # ------------------------------------------------------------ (A) accepted by the unpatched code: token-identical
    same = {
        "R M F O": [user("u1"), reasoning("t1"), msg("m1"), call("a"), out("a"), user("u2")],
        "R F F F O O O": [user("u1"), reasoning("t1"), call("a"), call("b"), call("c"), out("a"), out("b"), out("c")],
        "R F F O O, outputs in the other order": [user("u1"), reasoning("t1"), call("a"), call("b"), out("b"), out("a")],
        "R F M O (text only after the call)": [user("u1"), reasoning("t1"), call("a"), msg("m2"), out("a"), user("u2")],
        "R F user O (a user message before the output)": [user("u1"), reasoning("t1"), call("a"), user("steer"), out("a")],
        "R F O O (an output twice)": [user("u1"), reasoning("t1"), call("a"), out("a"), out("a"), user("u2")],
        "three turns": [user("u1"), reasoning("t1"), msg("m1"), call("a"), out("a"), reasoning("t2"), call("b"),
                        call("c", "apply_patch"), out("b"), out("c"), reasoning("t3"), msg("done"), user("u2"),
                        reasoning("t4"), msg("m4"), call("d", "view_image"), out("d")],
        "R M R M (two assistant messages, no tools)": [user("u1"), reasoning("t1"), msg("m1"), reasoning("t2"), msg("m2"), user("u2")],
        "R F R F O (the result of the last call only)": [user("u1"), reasoning("t1"), call("a"), reasoning("t2"), call("c"), out("c")],
    }
    chat_same = {
        "chat: two calls, two results": [
            {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": None, "tool_calls": [chat_call("a"), chat_call("b", "apply_patch")]},
            chat_tool("a"), chat_tool("b"), {"role": "assistant", "content": "done"}, {"role": "user", "content": "u2"}],
        "chat: results named by the name:position alias": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [chat_call("a"), chat_call("b", "apply_patch")]},
            {"role": "tool", "tool_call_id": "apply_patch:1", "content": "r1"},
            {"role": "tool", "tool_call_id": "exec_command:0", "content": "r0"}],
        "chat: assistant, assistant, no tools": [
            {"role": "user", "content": "u"}, {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"}, {"role": "user", "content": "u2"}],
        "chat: calls without results": [
            {"role": "user", "content": "u"}, {"role": "assistant", "content": "", "tool_calls": [chat_call("a")]},
            {"role": "user", "content": "u2"}],
        "chat: a system message between call and result": [
            {"role": "user", "content": "u"}, {"role": "assistant", "content": "", "tool_calls": [chat_call("a")]},
            {"role": "system", "content": "note"}, chat_tool("a")],
        "chat: a tool message with its own name and no id": [
            {"role": "user", "content": "u"}, {"role": "tool", "tool_call_id": None, "name": "probe", "content": "r"},
            {"role": "user", "content": "u2"}],
    }
    n_ok, n_same, n_converted = 0, 0, 0
    for name, items in same.items():
        a, b = unpatched(body_of(items)), patched(body_of(items))
        n_ok += not isinstance(b, str)
        if stock_r:
            n_same += not isinstance(a, str) and a == b
            n_converted += convert(stock_u, body_of(items)) == convert(new_u, body_of(items))
    for name, messages in chat_same.items():
        b = outcome(None, new_r, (messages, chat_tools))
        n_ok += not isinstance(b, str)
        if stock_r:
            a = outcome(None, stock_r, (messages, chat_tools))
            n_same += not isinstance(a, str) and a == b
    total = len(same) + len(chat_same)
    check(n_ok == total, f"(A) {n_ok} of {total} ordinary conversations render")
    if stock_r:
        check(n_same == total, f"(A) {n_same} of {total}: accepted by the unpatched code, the same conversation and tokens")
        check(n_converted == len(same), f"(A) {n_converted} of {len(same)} Responses inputs: the same chat messages from both conversions")

    # ------------------------------------------------------------ (B) a split turn, a result of an earlier turn
    A, T = 'role="assistant"', 'role="tool" tool="{}" index="{}"'
    split = {
        # name: (items, expected headers after the first user message, checks on the bodies)
        "R M F M O (text before and after the call)": (
            [user("u1"), reasoning("t1"), msg("before"), call("a"), msg("after"), out("a"), user("u2")],
            [A, T.format("exec_command", 1), 'role="user"', A],
            lambda s: "|t1|" in plain(s[1][1]) and "|before\n\nafter|" in plain(s[1][1]) and plain(s[1][1]).count("|call ") == 1
            and s[2][1] == "result-of-a"),
        "R F R O (a second reasoning item after the call)": (
            [user("u1"), reasoning("t1"), call("a"), reasoning("t2"), out("a"), user("u2")],
            [A, T.format("exec_command", 1), 'role="user"', A],
            lambda s: "|t1\n\nt2|" in plain(s[1][1]) and s[2][1] == "result-of-a"),
        "R F R F O O (reasoning between two calls)": (
            [user("u1"), reasoning("t1"), call("a"), reasoning("t2"), call("c", "apply_patch"), out("a"), out("c")],
            [A, T.format("exec_command", 1), T.format("apply_patch", 2), A],
            lambda s: plain(s[1][1]).count("|call ") == 2 and (s[2][1], s[3][1]) == ("result-of-a", "result-of-c")),
        "R F R F O O, outputs in the other order": (
            [user("u1"), reasoning("t1"), call("a"), reasoning("t2"), call("c", "apply_patch"), out("c"), out("a")],
            [A, T.format("exec_command", 1), T.format("apply_patch", 2), A],
            lambda s: (s[2][1], s[3][1]) == ("result-of-a", "result-of-c")),
        "R M F F M O O, outputs in the other order": (
            [user("u1"), reasoning("t1"), msg("before"), call("a"), call("b", "view_image"), msg("after"), out("b"), out("a")],
            [A, T.format("exec_command", 1), T.format("view_image", 2), A],
            lambda s: "|before\n\nafter|" in plain(s[1][1]) and (s[2][1], s[3][1]) == ("result-of-a", "result-of-b")),
        "R M | R M F M O (an earlier assistant message of the run stays separate)": (
            [user("u1"), reasoning("t0"), msg("m0"), reasoning("t1"), msg("before"), call("a"), msg("after"), out("a")],
            [A, A, T.format("exec_command", 1), A],
            lambda s: "|m0|" in plain(s[1][1]) and "before" not in s[1][1] and "|before\n\nafter|" in plain(s[2][1])),
        "R F user R M O (the call is in the turn before a user message)": (
            [user("u1"), reasoning("t1"), call("a", "apply_patch"), user("steer"), reasoning("t2"), msg("m2"), out("a"), user("u2")],
            [A, 'role="user"', A, T.format("apply_patch", 1), 'role="user"', A],
            lambda s: "|m2|" in plain(s[3][1]) and s[4][1] == "result-of-a"),
        "R F user R F O O (one result of the turn before, one of this turn)": (
            [user("u1"), reasoning("t1"), call("a", "apply_patch"), user("steer"), reasoning("t2"), call("c"), out("a"), out("c")],
            [A, 'role="user"', A, T.format("apply_patch", 1), T.format("exec_command", 2), A],
            lambda s: (s[4][1], s[5][1]) == ("result-of-a", "result-of-c")),
    }
    for name, (items, headers, body_ok) in split.items():
        body = body_of(items)
        b = patched(body)
        rejected_before(unpatched(body), f"(B) {name}")
        if isinstance(b, str):
            check(False, f"(B) {name}: the patched code raises {b[:90]}")
            continue
        s = shape(b[1])
        got = [h for h, _ in s][1:]
        check(got == headers, f"(B) {name}: messages {got}")
        check(bool(body_ok(s)), f"(B) {name}: text, reasoning, calls and results where expected")
        check(b == patched(body, use_async=True) and (not stock_r or b == outcome(stock_u, new_r, body)),
              f"(B) {name}: the same through render_messages_async" + (", and with the patched renderer alone" if stock_r else ""))

    if stock_r:  # a result named by position with the WRONG call is named by its id now (the one accepted shape that changes)
        body = body_of([user("u1"), reasoning("t1"), call("a", "apply_patch"), reasoning("t2"), call("c", "exec_command"), out("a")])
        a, b = unpatched(body), patched(body)
        was = [h for h, _ in shape(a[1])][-2] if not isinstance(a, str) else a
        now = [h for h, _ in shape(b[1])][-2] if not isinstance(b, str) else b
        check(was == T.format("exec_command", 1) and now == T.format("apply_patch", 1),
              f"(B) R F R F O with the FIRST call's result: unpatched names it {was!r}, patched {now!r}")

    conv = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "", "tool_calls": [chat_call("a", "apply_patch")]},
            {"role": "user", "content": "steer"}, {"role": "assistant", "content": "m"},
            {"role": "tool", "tool_call_id": "a", "tool": "explicit", "content": "r"}]
    check(new_r._rejoin_k3_tool_results(conv)[-1].get("tool") == "explicit", "(B) an explicit tool name on a result is kept")

    # ------------------------------------------------------------ (G) results that name their own tool
    own = {
        # name: (items, expected headers after the first user message, the patched conversion alone is enough)
        "a named result between system messages (Codex Desktop, thread start)": (
            [system("s1"), user("env"), system("s2"), named("p1"), system("s3"),
             named("p2", "codex_apps_selected_page_part_1_of_1"), user("u1")],
            [T.format("agent__codex_apps_open_page", 1), T.format("agent__codex_apps_selected_page_part_1_of_1", 2), 'role="user"', A], True),
        "a named result without a namespace": (
            [user("u1"), named("p1", "probe", None), user("u2")], [T.format("probe", 1), 'role="user"', A], True),
        "a named result after an ordinary one": (
            [user("u1"), reasoning("t1"), call("a"), out("a"), named("p1"), user("u2")],
            [A, T.format("exec_command", 1), T.format("agent__codex_apps_open_page", 2), 'role="user"', A], True),
        "a named result in front of an ordinary one": (
            [user("u1"), reasoning("t1"), call("a", "apply_patch"), named("p1"), out("a"), user("u2")],
            [A, T.format("agent__codex_apps_open_page", 1), T.format("apply_patch", 2), 'role="user"', A], False),
        "a named result between two ordinary ones": (
            [user("u1"), reasoning("t1"), call("a", "apply_patch"), call("b"), out("a"), named("p1"), out("b"), user("u2")],
            [A, T.format("apply_patch", 1), T.format("agent__codex_apps_open_page", 2), T.format("exec_command", 3), 'role="user"', A], False),
        "a named result after a turn without calls": (
            [user("u1"), reasoning("t1"), msg("m1"), named("p1"), user("u2")],
            [A, T.format("agent__codex_apps_open_page", 1), 'role="user"', A], True),
    }
    for name, (items, headers, conversion_alone) in own.items():
        body = body_of(items)
        b = patched(body)
        rejected_before(unpatched(body), f"(G) {name}")
        if isinstance(b, str):
            check(False, f"(G) {name}: the patched code raises {b[:90]}")
            continue
        s = shape(b[1])
        got = [h for h, _ in s][1:]
        check(got == headers, f"(G) {name}: messages {got}")
        truth = {i["call_id"]: i["name"] for i in items if i["type"] == "function_call"}
        truth.update({i["id"][4:]: "__".join(filter(None, (i.get("namespace"), i["name"])))
                      for i in items if i["type"] == "function_call_output" and "call_id" not in i})
        results = [(h, body[len("result-of-"):]) for h, body in s if h.startswith('role="tool"')]
        check(sorted(tag for _, tag in results) == sorted(i.get("call_id") or i["id"][4:] for i in items if i["type"] == "function_call_output")
              and all(f'tool="{truth[tag]}"' in h for h, tag in results), f"(G) {name}: every result under its own tool name")
        if stock_r:
            c = outcome(new_u, stock_r, body)
            check((not isinstance(c, str) and c[1] == b[1]) if conversion_alone else rejected(c),
                  f"(G) {name}: the patched conversion with the unpatched renderer: "
                  + ("the same tokens" if conversion_alone else "still rejected (needs both files)"))
    body = body_of([user("u1"), reasoning("t1"), call("a", "apply_patch"), named("z", "other_name", None, call_id="a"), user("u2")])
    b = patched(body)
    check(not isinstance(b, str) and [h for h, _ in shape(b[1])][2] == T.format("apply_patch", 1),
          "(G) a result with a call_id AND a name of its own: the call it answers names it")
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "", "tool_calls": [chat_call("a", "apply_patch")]},
                {"role": "tool", "tool_call_id": None, "name": "probe", "content": "r0"},
                {"role": "tool", "tool_call_id": "apply_patch:0", "content": "r1"}]
    b = outcome(None, new_r, (messages, chat_tools))
    check(not isinstance(b, str) and [h for h, _ in shape(b[1])][2:4] == [T.format("probe", 1), T.format("apply_patch", 2)],
          "(G) chat: a named result in front of one that answers by the name:position alias")

    # ------------------------------------------------------------ (C) nothing to name a result by: still rejected
    for name, items in {
        "an output whose call is not in the input": [user("u1"), reasoning("t1"), call("a"), out("a"), out("zzz"), user("u2")],
        "the input starts with an output": [user("u1"), out("zzz"), user("u2")],
        "an output after a turn without calls": [user("u1"), reasoning("t1"), msg("m1"), out("zzz"), user("u2")],
    }.items():
        a, b = unpatched(body_of(items)), patched(body_of(items))
        check(rejected(b) and (not stock_r or a == b), f"(C) {name}: rejected" + (" as before" if stock_r else ""))

    # ------------------------------------------------------------ (D) random conversations
    rnd = random.Random(20261110)
    names = ("exec_command", "apply_patch", "view_image")

    def random_conversation():
        conv, pending, truth, n = [{"role": "system", "content": "s"}, {"role": "user", "content": "u0"}], [], {}, 0
        for step in range(rnd.randint(3, 16)):
            kind = rnd.random()
            if kind < 0.46:
                m = {"role": "assistant", "content": rnd.choice(["", f"text{step}"])}
                if rnd.random() < 0.6:
                    m["reasoning"] = m["reasoning_content"] = f"think{step}"
                calls = []
                for _ in range(rnd.choice([0, 0, 1, 1, 2, 3])):
                    n += 1
                    truth[f"call_{n}"] = rnd.choice(names)
                    calls.append({"id": f"call_{n}", "type": "function",
                                  "function": {"name": truth[f"call_{n}"], "arguments": {"cmd": "ls"}}})
                    pending.append(f"call_{n}")
                if calls:
                    m["tool_calls"] = calls
                conv.append(m)
            elif kind < 0.8 and pending:
                for cid in rnd.sample(pending, rnd.randint(1, min(3, len(pending)))):
                    conv.append({"role": "tool", "tool_call_id": cid, "content": f"result-of-{cid}"})
                    if rnd.random() < 0.85:
                        pending.remove(cid)
            elif kind < 0.88:
                n += 1
                truth[f"ctx_{n}"] = "agent__page"
                conv.append({"role": "tool", "tool_call_id": None, "name": "agent__page", "content": f"result-of-ctx_{n}"})
            else:
                conv.append({"role": "user", "content": f"u{step}"})
        return conv, truth

    def resolved_already(conv):
        """Every block of results is one the encoding resolves as a whole: all by id in the most recent assistant
        message, or no id in the block."""
        latest, block = set(), []
        for m in conv + [{"role": "user"}]:
            if m["role"] == "tool":
                block.append(m.get("tool_call_id"))
                continue
            if block and any(block) and not all(cid in latest for cid in block):
                return False
            block = []
            if m["role"] == "assistant":
                latest = {c["id"] for c in m.get("tool_calls", [])}
        return True

    def texts(conv, field):
        return [t for m in conv if m["role"] == "assistant" for t in (m.get(field) or "").split("\n\n") if t]

    def misnamed(ids, truth):
        return [h for h, body in shape(ids) if h.startswith('role="tool"') and f'tool="{truth[body[len("result-of-"):]]}"' not in h]

    stats = {"resolved already": 0, "other": 0}
    before_stats = {"rejected": 0, "misnamed": 0}
    bad = {"changed": 0, "raised": 0, "misnamed": 0, "lost": 0, "idempotent": 0, "mutated": 0}
    template = dict(tools=None, tokenize=True, add_generation_prompt=True)
    for _ in range(3000):
        conv, truth = random_conversation()
        before = copy.deepcopy(conv)
        rejoined = new_r._rejoin_k3_tool_results(conv)
        bad["mutated"] += conv != before
        bad["idempotent"] += new_r._rejoin_k3_tool_results(rejoined) != rejoined
        is_resolved = resolved_already(conv)
        stats["resolved already" if is_resolved else "other"] += 1
        if is_resolved:
            bad["changed"] += rejoined != conv
        kept = (sorted(m["content"] for m in conv if m["role"] == "tool") == sorted(m["content"] for m in rejoined if m["role"] == "tool")
                and [m for m in conv if m["role"] in ("user", "system")] == [m for m in rejoined if m["role"] in ("user", "system")]
                and [c["id"] for m in conv for c in m.get("tool_calls", [])] == [c["id"] for m in rejoined for c in m.get("tool_calls", [])]
                and all(texts(conv, f) == texts(rejoined, f) for f in ("content", "reasoning", "reasoning_content")))
        bad["lost"] += not kept
        ids = None
        try:
            ids = tok.apply_chat_template(new_r._normalize_k3_tool_messages(rejoined), **template)
            bad["misnamed"] += bool(misnamed(ids, truth))
        except ValueError:
            bad["raised"] += 1
        if not stock_r:
            continue
        try:
            old_ids = tok.apply_chat_template(stock_r._normalize_k3_tool_messages(conv), **template)
            if is_resolved:
                bad["changed"] += old_ids != ids
            else:
                before_stats["misnamed"] += bool(misnamed(old_ids, truth))
        except ValueError:
            before_stats["rejected"] += 1
            bad["changed"] += is_resolved
    check(stats["resolved already"] > 300 and stats["other"] > 300,
          f"(D) 3000 random conversations: {stats}" + (f", of the others unpatched: {before_stats}" if stock_r else ""))
    check(bad["changed"] == 0, "(D) every block resolved already: unchanged" + (", same tokens as unpatched" if stock_r else "") + f" ({bad['changed']} differ)")
    check(bad["raised"] == 0 and bad["misnamed"] == 0,
          f"(D) all 3000 render, every result under its own tool name ({bad['raised']} raised, {bad['misnamed']} misnamed)")
    check(bad["lost"] == 0, f"(D) results, user / system messages, calls, text and reasoning all kept in order ({bad['lost']} differ)")
    check(bad["idempotent"] == 0 and bad["mutated"] == 0,
          f"(D) idempotent, the caller's messages not mutated ({bad['idempotent']}, {bad['mutated']})")

    # ------------------------------------------------------------ (E) the switch
    body = body_of(split["R M F M O (text before and after the call)"][0])
    check(rejected(outcome(new_u, off_r, body)), "(E) VLLM_MOET_K3_REJOIN_TOOL_RESULTS=0: the split turn is rejected")
    body = body_of(same["three turns"])
    e = outcome(new_u, off_r, body)
    check(not isinstance(e, str) and e == patched(body) and (not stock_r or e == unpatched(body)),
          "(E) =0: an ordinary conversation renders the same" + (", as unpatched" if stock_r else ""))

    # ------------------------------------------------------------ (F) skeletons of rejected requests, extra inputs
    def run_request(label, body, expect_rejected):
        try:
            n = len(convert(new_u, body)[0])
        except Exception as exc:  # noqa: BLE001
            check(False, f"(F) {label}: not a Responses request vLLM accepts: {type(exc).__name__}: {str(exc)[:120]}")
            return
        a, b = unpatched(body), patched(body)
        if expect_rejected:
            rejected_before(a, f"(F) {label}")
            check(not isinstance(b, str), f"(F) {label}: rendered ({n} messages" + (f", {len(b[1])} tokens)" if not isinstance(b, str) else f"): {b[:90]}"))
            if not isinstance(b, str):
                print(f"     {label}: tool messages " + str([h for h, _ in shape(b[1]) if h.startswith('role="tool"')]))
        elif not stock_r:
            check(not isinstance(b, str), f"(F) {label}: rendered ({n} messages)")
        elif isinstance(a, str):
            check(not isinstance(b, str) or a == b, f"(F) {label}: unpatched rejects ({a[:50]}), patched: {'rendered' if not isinstance(b, str) else 'the same error'}")
        else:
            check(a == b, f"(F) {label}: {n} messages, {len(a[1])} tokens, the same as unpatched")

    for path in sorted(glob.glob(os.path.join(HERE, "fixtures", "*.json"))):
        run_request("fixture " + os.path.basename(path), json.load(open(path)), expect_rejected=True)
    if args.requests:
        for path in sorted(glob.glob(os.path.join(args.requests, "*.json"))):
            run_request("request " + os.path.basename(path), json.load(open(path)), expect_rejected=True)
    if args.rollouts:
        for path in sorted(glob.glob(os.path.join(args.rollouts, "**", "rollout-*.jsonl"), recursive=True)):
            items = []
            for line in open(path, encoding="utf-8", errors="replace"):
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("type") == "response_item" and isinstance(record.get("payload"), dict):
                    item = {k: v for k, v in record["payload"].items() if v is not None}
                    if item.get("type") == "reasoning":
                        item.pop("encrypted_content", None)
                    if item.get("role") == "developer":   # the unpatched renderer has no developer role
                        item["role"] = "system"
                    items.append(item)
            run_request(f"rollout {os.path.basename(path)[8:27]} ({len(items)} items)", {"input": items}, expect_rejected=False)


try:
    main()
except Exception:
    traceback.print_exc()
    fails.append("exception")

verdict = "PASS" if not fails else "FAIL"
print(f"RESULT: {verdict} ({len(fails)} failure(s))")
sys.exit(0 if verdict == "PASS" else 1)
