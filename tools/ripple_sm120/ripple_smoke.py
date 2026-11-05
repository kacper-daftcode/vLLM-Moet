#!/usr/bin/env python3
"""Functional probes for Ripple-65B Preview behind vLLM (OpenAI API); greedy. Ripple decides per turn whether to reason
(no effort switch in its template), so the probes check the answers and that the `ripple` parsers keep reasoning out of
`content` - not that reasoning happened.

  chat      capital question (one word) + a counting question that usually reasons: answers right, no `<think>` in content
  arith     5 multi-step products/sums (the DS4.1 battery set)
  tools     get_weather tool call (ripple tool parser) and the round trip with the tool result
  json      strict json_schema answer=42
  needle    6-digit code at depth 0.2/0.7 in ~29K / ~106K / ~400K token prompts
usage: ripple_smoke.py --base http://127.0.0.1:8002 [--model ricursiveAI/Ripple-65B-Preview] --out FILE [--skip needle]
       [--needle-sizes 29000,106000,400000]
"""
import argparse
import json
import random
import re
import time
import urllib.request

ARITH = [
    ("What is 347 * 28? Reply with only the final number.", 9716),
    ("What is 86 * 74? Reply with only the final number.", 6364),
    ("What is 512 * 943? Reply with only the final number.", 482816),
    ("What is 38 + 277 + 4019 + 86? Reply with only the final number.", 4420),
    ("What is 1234 + 5678 + 9101 + 234 + 87? Reply with only the final number.", 16334),
]


class Client:
    def __init__(self, base, model, timeout=1800):
        self.base, self.model, self.timeout = base.rstrip("/"), model, timeout

    def chat(self, messages, max_tokens, **extra):
        body = dict(model=self.model, messages=messages, max_tokens=max_tokens, temperature=0, **extra)
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.load(r)
        d["_secs"] = time.time() - t0
        return d


def parts(d):
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning") or m.get("reasoning_content") or ""), m.get("tool_calls") or []


def t_chat(c):
    d = c.chat([{"role": "user", "content": "What is the capital of Australia? Answer in one word."}], 2048)
    content, reasoning, _ = parts(d)
    d2 = c.chat([{"role": "user", "content": "How many prime numbers are there below 30? Answer with the number."}], 8192)
    content2, reasoning2, _ = parts(d2)
    leak = any(tag in content + content2 for tag in ("<think>", "</think>", "<tool_call>"))
    return dict(ok="canberra" in content.lower() and "10" in content2 and not leak,
                content=content[:200], content2=content2[:120], reasoning_chars=(len(reasoning), len(reasoning2)),
                usage=d["usage"], usage2=d2["usage"], finish=d["choices"][0]["finish_reason"])


def t_arith(c):
    out = []
    for q, want in ARITH:
        d = c.chat([{"role": "user", "content": q}], 8192)
        content, reasoning, _ = parts(d)
        nums = [int(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*", content)]
        out.append(dict(q=q, want=want, got=content.strip()[:80], ok=bool(nums) and nums[-1] == want,
                        reasoning_chars=len(reasoning), completion_tokens=d["usage"]["completion_tokens"],
                        secs=round(d["_secs"], 2)))
    return dict(ok=all(x["ok"] for x in out), cases=out)


def t_tools(c):
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    msgs = [{"role": "user", "content": "What's the weather in Warsaw right now? Use the tool."}]
    d = c.chat(msgs, 4096, tools=tools, tool_choice="auto")
    content, reasoning, calls = parts(d)
    call_ok = bool(calls) and calls[0]["function"]["name"] == "get_weather" and \
        "warsaw" in json.loads(calls[0]["function"]["arguments"] or "{}").get("city", "").lower()
    res = dict(call_ok=call_ok, calls=[x["function"] for x in calls], finish=d["choices"][0]["finish_reason"],
               content=content[:120], reasoning_chars=len(reasoning))
    if call_ok:
        assistant = {"role": "assistant", "content": content or None, "tool_calls": calls}
        if reasoning:
            assistant["reasoning_content"] = reasoning
        msgs += [assistant,
                 {"role": "tool", "tool_call_id": calls[0]["id"],
                  "content": json.dumps({"city": "Warsaw", "temperature_c": 17, "sky": "light rain"})}]
        d2 = c.chat(msgs, 4096, tools=tools)
        content2, _, _ = parts(d2)
        res["round_trip"] = content2[:200]
        res["round_trip_ok"] = "17" in content2
    res["ok"] = call_ok and res.get("round_trip_ok", False)
    return res


def t_json(c):
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"],
              "additionalProperties": False}
    d = c.chat([{"role": "user", "content": "What is 6 times 7? Respond as JSON."}], 2048,
               response_format={"type": "json_schema", "json_schema": {"name": "a", "schema": schema, "strict": True}})
    content, _, _ = parts(d)
    try:
        ok = json.loads(content).get("answer") == 42
    except Exception:  # noqa: BLE001
        ok = False
    return dict(ok=ok, content=content[:200])


def t_needle(c, sizes):
    rnd = random.Random(1234)
    filler = ("The committee reviewed the quarterly logistics report, noting that warehouse throughput rose while "
              "shipping delays fell across the northern routes. ")
    cases = []
    for target in sizes:
        for depth in (0.2, 0.7):
            code = rnd.randint(100000, 999999)
            n = max(1, target // 26)
            body = [f"[{i}] {filler}" for i in range(n)]
            body.insert(int(n * depth), f"\n\nThe secret access code for the vault is {code}. Remember it.\n\n")
            q = "".join(body) + "\n\nWhat is the secret access code for the vault? Reply with only the code."
            try:
                d = c.chat([{"role": "user", "content": q}], 2048)
                content, _, _ = parts(d)
                cases.append(dict(target=target, depth=depth, prompt_tokens=d["usage"]["prompt_tokens"],
                                  ok=str(code) in content, got=content.strip()[:40], secs=round(d["_secs"], 1)))
            except Exception as e:  # noqa: BLE001
                cases.append(dict(target=target, depth=depth, ok=False, error=repr(e)[:200]))
            print("  needle", cases[-1], flush=True)
    return dict(ok=all(x["ok"] for x in cases), cases=cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8002")
    ap.add_argument("--model", default="ricursiveAI/Ripple-65B-Preview")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default="")
    ap.add_argument("--needle-sizes", default="29000,106000,400000")
    a = ap.parse_args()
    c = Client(a.base, a.model)
    skip = set(filter(None, a.skip.split(",")))
    tests = [("chat", lambda: t_chat(c)), ("arith", lambda: t_arith(c)), ("tools", lambda: t_tools(c)),
             ("json", lambda: t_json(c)),
             ("needle", lambda: t_needle(c, [int(x) for x in a.needle_sizes.split(",")]))]
    res = {}
    for name, fn in tests:
        if name in skip:
            continue
        t0 = time.time()
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001
            res[name] = dict(ok=False, error=repr(e)[:500])
        res[name]["secs"] = round(time.time() - t0, 1)
        print(f"{name:7s} ok={res[name].get('ok')}  ({res[name]['secs']} s)", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("ALL OK" if all(v.get("ok") for v in res.values()) else "SOME FAILED", flush=True)


if __name__ == "__main__":
    main()
