#!/usr/bin/env python3
"""Same-checkpoint quality/fidelity probes against one OpenAI-compatible endpoint; JSON out.

Greedy everywhere. Run once per serving stack, then compare_outputs.py on the two JSON files.
  arith        5 multi-step products/sums (vLLM-Moet bench probe set), thinking off and on
  coherence    12 raw-completion prompts, 128 tokens, loop detector (bench probe set)
  agreement    24 chat prompts, 192 tokens, thinking off -> texts for cross-stack diff
  tools        lookup_fixture round trip (0xSero boot.py smoke)
  json         strict json_schema answer=42 (0xSero boot.py smoke)
  vision       red circle + blue square 3024x588 PNG (0xSero boot.py smoke)
  needle       6-digit code at depth 0.2/0.7 in ~29K / ~106K / ~400K token prompts (vLLM-Moet needle_any)
usage: quality_cmp.py --base URL --model NAME --out FILE [--api-key KEY] [--skip needle,vision] [--needle-sizes 29000,106000,400000]
"""
import argparse
import base64
import io
import json
import os
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
COHERENCE_PROMPTS = [
    "Q: What is the capital of Australia?\nA:",
    "Q: A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?\nA:",
    "Write a Python function that returns the n-th Fibonacci number iteratively.\n\n```python\n",
    "Translate to French: 'The weather is beautiful today, let's go for a walk in the park.'\nFrench:",
    "Q: If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Explain step by step.\nA:",
    ("Summarize the following paragraph in one sentence:\n\n"
     "The industrial revolution, which began in Britain in the late eighteenth century, "
     "transformed economies that had been based on agriculture and handicrafts into economies "
     "based on large-scale industry, mechanized manufacturing, and the factory system. New "
     "machines, new power sources, and new ways of organizing work made existing industries "
     "more productive and efficient.\n\nSummary:"),
    "Q: Which planet in our solar system has the most moons?\nA:",
    "Q: What is 847 + 256? Show your work.\nA:",
    "Write a one-line Python list comprehension that squares the even numbers in a list called xs.\n\n```python\n",
    "Q: Name three primary colors.\nA:",
    "Once upon a time, in a small village by the sea,",
    "The main differences between TCP and UDP are",
]
AGREEMENT_PROMPTS = [
    "Explain in 3 short sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively. Code only.",
    "List the planets of the solar system in order from the sun, one per line.",
    "Write a haiku about a GPU.",
    "What is the derivative of x^3 * sin(x)? Show the steps briefly.",
    "Translate to German: 'The library closes at eight on weekdays and at noon on Saturdays.'",
    "Write a bash one-liner that counts the lines in all .py files under the current directory.",
    "Summarize the plot of Romeo and Juliet in four sentences.",
    "Give three differences between a process and a thread.",
    "Write a SQL query that returns the ten customers with the highest total order value from tables customers(id, name) and orders(id, customer_id, amount).",
    "What is 19 * 23 + 7? Reply with the number only.",
    "Explain what a Bloom filter is and when you would use one, in one paragraph.",
    "Write a limerick about a cat who learned to code.",
    "Convert 72 degrees Fahrenheit to Celsius. Show the formula.",
    "Write a Python class Stack with push, pop and peek, with type hints. Code only.",
    "Name the capital cities of Canada, Australia and Brazil.",
    "Explain the difference between TCP and UDP in three bullet points.",
    "Write a regular expression that matches an IPv4 address and explain it briefly.",
    "What happens when you mix baking soda and vinegar? Two sentences.",
    "Write a JSON object describing a book with title, author, year and a list of three tags.",
    "Give a mnemonic for the order of operations in arithmetic.",
    "Explain recursion to a ten-year-old in three sentences.",
    "Write a C function that reverses a null-terminated string in place. Code only.",
    "Describe the water cycle in four steps.",
]
FILLER = ("The quick brown fox jumps over the lazy dog. A journey of a thousand miles begins with a single step. "
          "All that glitters is not gold. Fortune favours the bold. Actions speak louder than words. ")


class Client:
    def __init__(self, base, model, api_key, timeout=3600):
        self.base, self.model, self.api_key, self.timeout = base.rstrip("/"), model, api_key, timeout

    def post(self, path, payload, timeout=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(), headers=headers)
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            d = json.loads(r.read())
        d["_elapsed_s"] = round(time.monotonic() - t0, 2)
        return d

    def chat(self, messages, max_tokens, thinking=False, **extra):
        body = dict(model=self.model, messages=messages, max_tokens=max_tokens, temperature=0,
                    chat_template_kwargs={"thinking": thinking})
        body.update(extra)
        return self.post("/v1/chat/completions", body)

    def completion(self, prompt, max_tokens):
        return self.post("/v1/completions", dict(model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=0))


def msg_text(d):
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning") or m.get("reasoning_content") or "")


def degenerate(text):
    if len(text) < 64:
        return False
    for w in (4, 8, 16):
        chunk = text[-w:]
        if chunk.strip() and text.endswith(chunk * min(8, len(text) // w)):
            return True
    return False


def t_arith(c, thinking):
    rows = []
    for q, gold in ARITH:
        d = c.chat([{"role": "user", "content": q}], 4096 if thinking else 64, thinking=thinking)
        content, reasoning = msg_text(d)
        nums = re.findall(r"-?[\d,]*\d", content.replace(",", ""))
        got = int(nums[-1]) if nums else None
        rows.append(dict(gold=gold, got=got, ok=got == gold, completion_tokens=d["usage"]["completion_tokens"],
                         s=d["_elapsed_s"]))
    return dict(score=sum(r["ok"] for r in rows), of=len(rows), rows=rows)


def t_coherence(c):
    texts, degen = [], 0
    for p in COHERENCE_PROMPTS:
        d = c.completion(p, 128)
        t = d["choices"][0]["text"]
        texts.append(t)
        degen += degenerate(t)
    return dict(degenerate=degen, of=len(texts), texts=texts)


def t_agreement(c):
    out = []
    for p in AGREEMENT_PROMPTS:
        d = c.chat([{"role": "user", "content": p}], 192)
        content, reasoning = msg_text(d)
        out.append(dict(prompt=p, text=content, reasoning=reasoning, completion_tokens=d["usage"]["completion_tokens"],
                        finish=d["choices"][0].get("finish_reason")))
    return out


def t_tools(c):
    tool = {"type": "function", "function": {"name": "lookup_fixture", "description": "Retrieve a stored test value.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"],
                           "additionalProperties": False}}}
    messages = [{"role": "user", "content": "Use lookup_fixture to retrieve the value for key alpha. Do not guess."}]
    called = c.chat(messages, 512, tools=[tool])
    assistant = called["choices"][0]["message"]
    calls = assistant.get("tool_calls") or []
    ok1 = len(calls) == 1 and calls[0]["function"]["name"] == "lookup_fixture"
    args_ok = False
    if ok1:
        try:
            args_ok = json.loads(calls[0]["function"]["arguments"]) == {"key": "alpha"}
        except Exception:  # noqa: BLE001
            pass
    result = dict(call_ok=ok1, args_ok=args_ok, call=assistant)
    if ok1:
        messages += [assistant, {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"value":42}'}]
        cont = c.chat(messages, 256, tools=[tool])
        text, _ = msg_text(cont)
        result.update(continuation=text, continuation_ok="42" in text)
    return result


def t_json(c):
    d = c.chat([{"role": "user", "content": "Return an object whose answer is the integer 42."}], 64,
               response_format={"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
                   "type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"],
                   "additionalProperties": False}}})
    text, _ = msg_text(d)
    try:
        ok = json.loads(text) == {"answer": 42}
    except Exception:  # noqa: BLE001
        ok = False
    return dict(ok=ok, text=text)


def t_vision(c):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (3024, 588), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((200, 100, 588, 488), fill="red")
    draw.rectangle((2400, 100, 2788, 488), fill="blue")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    d = c.chat([{"role": "user", "content": [
        {"type": "text", "text": "Describe the two colored shapes and their left-to-right order. Be concise."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}]}], 256)
    text, _ = msg_text(d)
    low = text.lower()
    return dict(ok=all(w in low for w in ("red", "circle", "blue")) and ("square" in low or "rectangle" in low),
                text=text, usage=d.get("usage"))


def t_needle(c, sizes):
    cases = []
    for target in sizes:
        for depth in (0.2, 0.7):
            random.seed(target + int(depth * 100))
            code = f"{random.randint(100000, 999999)}"
            needle_txt = f"\n\nThe secret access code for the vault is {code}. Remember it.\n\n"
            parts = [FILLER] * int(target * 4.2 / len(FILLER))
            parts.insert(int(len(parts) * depth), needle_txt)
            doc = "".join(parts)
            try:
                d = c.chat([{"role": "user", "content": doc + "\n\nWhat is the secret access code for the vault? Answer with the number only."}], 24)
                ans, _ = msg_text(d)
                cases.append(dict(target=target, depth=depth, prompt_tokens=d["usage"]["prompt_tokens"], ok=code in (ans or ""),
                                  answer=ans, s=d["_elapsed_s"]))
            except Exception as e:  # noqa: BLE001
                cases.append(dict(target=target, depth=depth, ok=False, error=repr(e)))
            print("  needle", cases[-1], flush=True)
    return dict(all_pass=all(x["ok"] for x in cases), cases=cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default="")
    ap.add_argument("--needle-sizes", default="29000,106000,400000")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    c = Client(a.base, a.model, a.api_key)
    res = dict(label=a.label, base=a.base, model=a.model, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    tests = [("arith_nothink", lambda: t_arith(c, False)), ("arith_think", lambda: t_arith(c, True)),
             ("coherence", lambda: t_coherence(c)), ("agreement", lambda: t_agreement(c)),
             ("tools", lambda: t_tools(c)), ("json", lambda: t_json(c)), ("vision", lambda: t_vision(c)),
             ("needle", lambda: t_needle(c, [int(x) for x in a.needle_sizes.split(",")]))]
    for name, fn in tests:
        if name in skip or name.split("_")[0] in skip:
            continue
        t0 = time.monotonic()
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001
            res[name] = dict(error=repr(e))
        res[name + "_s"] = round(time.monotonic() - t0, 1)
        brief = {k: v for k, v in (res[name].items() if isinstance(res[name], dict) else []) if k in ("score", "of", "degenerate", "ok", "all_pass", "call_ok", "args_ok", "continuation_ok", "error")}
        print(f"[{name}] {brief if brief else len(res[name])} ({res[name + '_s']}s)", flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
