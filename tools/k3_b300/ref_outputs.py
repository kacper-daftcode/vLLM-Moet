#!/usr/bin/env python3
"""Greedy reference outputs for A/B parity checks: fixed prompts, T=0, saved as JSON.
usage: ref_outputs.py --base URL --out FILE [--max-tokens 400] [--api-key KEY] [--compare FILE]
"""
import argparse, json, os, sys, time, urllib.request

PROMPTS = [
    ("code", "Write a Python function that parses an ISO-8601 duration string (e.g. 'P3Y6M4DT12H30M5S') into a "
             "datetime.timedelta-compatible dict of components, with full validation and unit tests using pytest. "
             "Return only code."),
    ("reason", "A farmer has 17 sheep. All but 9 run away. Then he buys twice as many as remain and sells a third of "
               "the total. How many sheep does he have now? Explain step by step, then give the final number."),
    ("prose", "Explain in about 250 words how a paged KV cache works in an LLM inference engine and why hybrid "
              "linear-attention models complicate it."),
    ("long", "Here is a list of 40 numbers: " + ", ".join(str((i * 7919) % 1000) for i in range(40)) +
             ". Sort them ascending, then compute the median and the mean, showing your work briefly."),
]


def chat(base, key, model, prompt, max_tokens, effort):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "top_p": 1.0, "seed": 0}
    if effort:
        body["reasoning_effort"] = effort
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    dt = time.monotonic() - t0
    m = d["choices"][0]["message"]
    return {"reasoning": m.get("reasoning_content") or m.get("reasoning") or "", "content": m.get("content") or "",
            "usage": d.get("usage"), "seconds": dt, "finish": d["choices"][0].get("finish_reason")}


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default="")
    a = ap.parse_args()
    res = {}
    for name, p in PROMPTS:
        r = chat(a.base, a.api_key, a.model, p, a.max_tokens, a.reasoning_effort)
        res[name] = r
        u = r["usage"] or {}
        print(f"{name:7s} completion_tokens={u.get('completion_tokens')} finish={r['finish']} {r['seconds']:.1f}s",
              flush=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    if a.compare:
        ref = json.load(open(a.compare))
        print("--- parity vs", a.compare)
        for name in res:
            x = (res[name]["reasoning"] + "\n<<<>>>\n" + res[name]["content"])
            y = (ref[name]["reasoning"] + "\n<<<>>>\n" + ref[name]["content"])
            same = x == y
            cp = common_prefix_len(x, y)
            print(f"{name:7s} identical={same} common_prefix_chars={cp}/{len(y)} new_len={len(x)}")


if __name__ == "__main__":
    main()
