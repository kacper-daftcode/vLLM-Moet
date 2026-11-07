#!/usr/bin/env python3
"""Greedy parity probe: the same prompts, one at a time (fresh prefill via cache_salt), temperature 0, with the chosen
token's logprob per position - saved to a JSON file; `--compare A.json B.json` reports per prompt the first diverging
token, the number of identical generated tokens and the max |delta logprob| over the common prefix. Two runs on the same
server show the stack's own run-to-run determinism; a run per arm (e.g. VLLM_MOET_P2P_AR=1 vs 0) shows the arm's bits.

usage: parity_probe.py --base http://127.0.0.1:8002 --model NAME --out run.json [--max-tokens 768]
       parity_probe.py --compare a.json b.json
"""
import argparse
import json
import sys
import urllib.request
import uuid

PROMPTS = {
    "code": "Write a complete Python module implementing an LRU cache, a trie and Dijkstra's algorithm with type hints "
            "and docstrings. Code only, no explanations.",
    "prose": "Write a detailed essay about the history of GPU computing, from fixed-function pipelines to general-purpose "
             "accelerators. Be thorough.",
    "math": "A rectangular garden has a perimeter of 56 meters and its length is 4 meters more than twice its width. "
            "Find its area, then find the side of a square with the same area, then the diagonal of that square. "
            "Show every step.",
}


def run(a):
    out = {"base": a.base, "model": a.model, "max_tokens": a.max_tokens, "prompts": {}}
    for name, text in PROMPTS.items():
        body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "logprobs": True, "top_logprobs": 1,
                "cache_salt": uuid.uuid4().hex, "messages": [{"role": "user", "content": text}]}
        req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        ch = d["choices"][0]
        lp = ch.get("logprobs") or {}
        content = lp.get("content") or []
        tokens = [c["token"] for c in content]
        logprobs = [c["logprob"] for c in content]
        m = ch["message"]
        out["prompts"][name] = {"tokens": tokens, "logprobs": logprobs, "finish": ch.get("finish_reason"),
                                "text": (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or ""),
                                "usage": d.get("usage")}
        print(f"{name:6s} {len(tokens)} tokens, finish={ch.get('finish_reason')}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


def compare(pa, pb):
    a, b = json.load(open(pa)), json.load(open(pb))
    worst = 0
    for name in a["prompts"]:
        ta, tb = a["prompts"][name]["tokens"], b["prompts"][name]["tokens"]
        la, lb = a["prompts"][name]["logprobs"], b["prompts"][name]["logprobs"]
        n = 0
        while n < min(len(ta), len(tb)) and ta[n] == tb[n]:
            n += 1
        dl = max((abs(x - y) for x, y in zip(la[:n], lb[:n])), default=0.0)
        same = n == len(ta) == len(tb)
        worst = max(worst, dl)
        print(f"{name:6s} {'IDENTICAL' if same else 'DIVERGE'}: {n} identical tokens of {len(ta)} / {len(tb)}; "
              f"max |dlogprob| over the common prefix {dl:.3e}" + ("" if same else f"; first differing token: {ta[n:n+1]} vs {tb[n:n+1]}"))
    print("ALL IDENTICAL" if all(a["prompts"][k]["tokens"] == b["prompts"][k]["tokens"] for k in a["prompts"]) and worst == 0
          else "DIFFERENCES")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8002")
    ap.add_argument("--model", default="ricursiveAI/Ripple-65B-Preview")
    ap.add_argument("--out")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    elif a.out:
        run(a)
    else:
        ap.error("--out or --compare")


if __name__ == "__main__":
    sys.exit(main())
