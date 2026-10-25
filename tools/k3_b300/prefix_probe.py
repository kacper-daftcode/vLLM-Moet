#!/usr/bin/env python3
"""Prefix-cache probe for hybrid K3: cold vs warm identical greedy requests, then a multi-turn continuation.
Reports TTFT, prefix hit counters and whether the warm output equals the cold one.
usage: prefix_probe.py --base URL [--prompt-tokens 30000] [--api-key KEY]
"""
import argparse, json, os, time, urllib.request, uuid

FILLER = ("Reference notes: the cache stores recently accessed entries. "
          "An implementation should maintain ordering, handle replacement and validate its invariants.\n")


def counters(base):
    out = {}
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            for key in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                        "vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total"):
                if line.startswith(key):
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def chat(base, key, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
            "seed": 0, "reasoning_effort": "low", "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0 = time.monotonic()
    first = None
    content, reasoning, usage = [], [], {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    first = first or time.monotonic()
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    reasoning.append(delta.get("reasoning_content") or delta.get("reasoning"))
    return {"ttft": (first or time.monotonic()) - t0, "content": "".join(content), "reasoning": "".join(reasoning),
            "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--prompt-tokens", type=int, default=30000)
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    doc = uuid.uuid4().hex + "\n" + FILLER * int(a.prompt_tokens * 4.3 / len(FILLER))
    q1 = doc + "\nIn two sentences, what do the notes above describe?"
    c0 = counters(a.base)
    cold = chat(a.base, a.api_key, a.model, [{"role": "user", "content": q1}], 120)
    c1 = counters(a.base)
    warm = chat(a.base, a.api_key, a.model, [{"role": "user", "content": q1}], 120)
    c2 = counters(a.base)
    same = (cold["content"] == warm["content"]) and (cold["reasoning"] == warm["reasoning"])
    print(f"{a.label} cold: prompt={cold['usage'].get('prompt_tokens')} ttft={cold['ttft']:.2f}s | "
          f"warm: ttft={warm['ttft']:.2f}s identical={same} | prefix hits +{c2.get('vllm:prefix_cache_hits_total', 0) - c1.get('vllm:prefix_cache_hits_total', 0):.0f} "
          f"queries +{c2.get('vllm:prefix_cache_queries_total', 0) - c1.get('vllm:prefix_cache_queries_total', 0):.0f}")
    # multi-turn continuation: previous turn (with reasoning_content sent back, as K3 requires) + new question
    msgs = [{"role": "user", "content": q1},
            {"role": "assistant", "content": cold["content"], "reasoning_content": cold["reasoning"]},
            {"role": "user", "content": "Now list three invariants such a cache should validate. Be brief."}]
    c3 = counters(a.base)
    turn2 = chat(a.base, a.api_key, a.model, msgs, 200)
    c4 = counters(a.base)
    turn2b = chat(a.base, a.api_key, a.model, msgs, 200)
    print(f"{a.label} turn2: prompt={turn2['usage'].get('prompt_tokens')} ttft={turn2['ttft']:.2f}s "
          f"prefix hits +{c4.get('vllm:prefix_cache_hits_total', 0) - c3.get('vllm:prefix_cache_hits_total', 0):.0f} | "
          f"repeat ttft={turn2b['ttft']:.2f}s identical={(turn2['content'], turn2['reasoning']) == (turn2b['content'], turn2b['reasoning'])}")
    print("turn2 content:", repr(turn2["content"][:300]))


if __name__ == "__main__":
    main()
