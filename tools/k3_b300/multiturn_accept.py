#!/usr/bin/env python3
"""Multi-turn acceptance: a 20K-token first turn, then follow-up turns that reuse the prefix (as an agent does).
Reports per turn: prefix hits, TTFT, accepted length and tok/s - i.e. whether the draft keeps its acceptance after a
prefix hit (a sliding-window draft must get its window rebuilt).
usage: multiturn_accept.py --base URL [--prompt-tokens 20000] [--turns 3] [--max-tokens 600] [--label L]
"""
import argparse, json, os, time, urllib.request, uuid

FILLER = ("Design note: the service keeps a bounded cache of parsed configuration documents keyed by path and version. "
          "Entries expire after a configurable TTL; a background sweeper evicts the least recently used entries when the "
          "size limit is exceeded, and every mutation is recorded in an append-only audit log with a monotonic sequence number.\n")
TASKS = [
    "Based on the design notes above, write a Python ConfigCache class with get/put/invalidate and TTL expiry. Code only.",
    "Now add LRU eviction with a size limit to the class. Return the full updated code only.",
    "Now add the append-only audit log with monotonic sequence numbers and thread safety. Return the full updated code only.",
    "Now write a pytest test suite for the class. Code only.",
]


def counters(base):
    out = {"pos": {}}
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:spec_decode_num_") and "per_pos" not in line:
                key = line.split("{")[0].removeprefix("vllm:spec_decode_num_").removesuffix("_total")
                out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:prefix_cache_hits_total") or line.startswith("vllm:prefix_cache_queries_total"):
                key = line.split("{")[0]
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
    first = last = None
    content, reasoning, usage = [], [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                dl = ch.get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):
                    last = time.monotonic()
                    first = first or last
                if dl.get("content"):
                    content.append(dl["content"])
                if dl.get("reasoning_content") or dl.get("reasoning"):
                    reasoning.append(dl.get("reasoning_content") or dl.get("reasoning"))
    return {"ttft": (first or t0) - t0, "gen_s": (last - first) if first and last and last > first else 0.0,
            "content": "".join(content), "reasoning": "".join(reasoning), "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--prompt-tokens", type=int, default=20000)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    doc = uuid.uuid4().hex + "\n" + FILLER * int(a.prompt_tokens * 4.3 / len(FILLER))
    msgs = [{"role": "user", "content": doc + "\n\n" + TASKS[0]}]
    for t in range(min(a.turns, len(TASKS))):
        c0 = counters(a.base)
        r = chat(a.base, a.api_key, a.model, msgs, a.max_tokens)
        c1 = counters(a.base)
        drafts = c1.get("drafts", 0) - c0.get("drafts", 0)
        acc = c1.get("accepted_tokens", 0) - c0.get("accepted_tokens", 0)
        hits = c1.get("vllm:prefix_cache_hits_total", 0) - c0.get("vllm:prefix_cache_hits_total", 0)
        q = c1.get("vllm:prefix_cache_queries_total", 0) - c0.get("vllm:prefix_cache_queries_total", 0)
        toks = r["usage"].get("completion_tokens", 0)
        print(f"{a.label} turn {t + 1}: prompt={r['usage'].get('prompt_tokens')} prefix hits {hits:.0f}/{q:.0f} "
              f"ttft={r['ttft']:.2f}s | {toks} tok, accepted length {1 + acc / drafts if drafts else 0:.2f}, "
              f"{toks / r['gen_s'] if r['gen_s'] else 0:.0f} tok/s", flush=True)
        msgs.append({"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"]})
        if t + 1 < len(TASKS):
            msgs.append({"role": "user", "content": TASKS[t + 1]})


if __name__ == "__main__":
    main()
