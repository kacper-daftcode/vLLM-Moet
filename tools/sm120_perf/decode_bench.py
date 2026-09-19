#!/usr/bin/env python3
"""Single-stream decode + prefill timing via OpenAI API (streaming to get TTFT).

usage: decode_bench.py BASE MODEL ['{"thinking": false}']   (API_KEY env -> Bearer header)
"""
import json
import os
import sys
import time
import urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
THINK_KW = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
HEADERS = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    HEADERS["Authorization"] = "Bearer " + os.environ["API_KEY"]


def run(prompt, max_tokens, label, ignore_eos=True):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if THINK_KW:
        body["chat_template_kwargs"] = THINK_KW
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(), headers=HEADERS)
    t0 = time.perf_counter()
    ttft = None
    usage = None
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                n_chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    out = usage["completion_tokens"]
    pin = usage["prompt_tokens"]
    dec = total - ttft
    print(f"{label:28s} in={pin:6d} out={out:4d} TTFT={ttft*1000:7.0f} ms ({pin/ttft:7.0f} tok/s prefill)  "
          f"decode={out/dec:6.1f} tok/s  chunks={n_chunks} ({n_chunks/dec:5.1f} steps/s, {out/max(n_chunks,1):.2f} tok/step)")


filler = ("def f_%d(x):\n    return x * %d + 1\n\n" * 1)
run("Write a detailed essay about the history of GPU computing. Be thorough and long.", 512, "prose 512")
run("Write a complete Python module implementing an LRU cache, a trie and Dijkstra with type hints. Code only.", 512, "code 512")
long_prompt = "".join(filler % (i, i) for i in range(900)) + "\nSummarize what these functions have in common in one sentence."
run(long_prompt, 128, "prefill ~16K + 128 out")
