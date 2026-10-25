#!/usr/bin/env python3
"""GSM8K quick accuracy probe over the chat API (greedy, concurrent). Deterministic subset: the first N test items.
usage: gsm8k_probe.py --base URL [--n 100] [--concurrency 16] [--max-tokens 2048] [--reasoning-effort low] --out F
"""
import argparse, concurrent.futures, json, os, re, time, urllib.request

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
CACHE = "/root/work/k3-test/gsm8k_test.jsonl"


def load(n):
    if not os.path.exists(CACHE):
        urllib.request.urlretrieve(URL, CACHE)
    rows = [json.loads(l) for l in open(CACHE)]
    return rows[:n]


def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def pred(text):
    m = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    if not m:
        return None
    return m[-1].replace(",", "").rstrip(".")


def ask(base, key, model, q, max_tokens, effort):
    body = {"model": model, "messages": [{"role": "user", "content": q + "\nGive the final numeric answer on the "
                                                                          "last line as 'Answer: <number>'."}],
            "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0, "seed": 0}
    if effort:
        body["reasoning_effort"] = effort
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    return m.get("content") or "", d.get("usage") or {}, d["choices"][0].get("finish_reason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    rows = load(a.n)
    t0 = time.time()
    results = [None] * len(rows)

    def work(i):
        content, usage, fin = ask(a.base, a.api_key, a.model, rows[i]["question"], a.max_tokens, a.reasoning_effort)
        return i, content, usage, fin

    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
        for i, content, usage, fin in ex.map(work, range(len(rows))):
            g = gold(rows[i]["answer"])
            p = pred(content)
            try:
                ok = p is not None and abs(float(p) - float(g)) < 1e-6
            except ValueError:
                ok = False
            results[i] = {"i": i, "gold": g, "pred": p, "ok": ok, "finish": fin,
                          "completion_tokens": usage.get("completion_tokens"), "content_tail": (content or "")[-200:]}
    dt = time.time() - t0
    acc = sum(r["ok"] for r in results) / len(results)
    toks = sum((r["completion_tokens"] or 0) for r in results)
    trunc = sum(r["finish"] == "length" for r in results)
    summary = {"label": a.label, "n": len(rows), "accuracy": acc, "correct": sum(r["ok"] for r in results),
               "completion_tokens": toks, "truncated": trunc, "seconds": dt, "reasoning_effort": a.reasoning_effort}
    json.dump({"summary": summary, "results": results}, open(a.out, "w"), indent=1)
    print(f"{a.label:24s} GSM8K-{len(rows)}: {summary['correct']}/{len(rows)} = {acc:.1%}  tokens={toks} "
          f"truncated={trunc} {dt:.0f}s")


if __name__ == "__main__":
    main()
