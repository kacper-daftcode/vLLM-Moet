#!/usr/bin/env python3
"""Light concurrency check: C parallel requests with ~PROMPT_TOK-token prompts, 256 output tokens.

usage: concurrency_probe.py BASE MODEL [C=4] [PROMPT_TOK=8000] [chat_template_kwargs_json]
Prints per-request in/out/time and the aggregate output tok/s (wall includes the prefills).
"""
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek-ai/DeepSeek-V4.1-Flash"
C = int(sys.argv[3]) if len(sys.argv) > 3 else 4
PROMPT_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
KW = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {"thinking": False}

filler = (
    "Line %d: the scheduler assigns blocks to requests and reclaims them when sequences finish; "
    "prefix caching reuses identical leading blocks across requests.\n"
)
results = []


def worker(i):
    n = PROMPT_TOK // 30
    text = "".join(filler % j for j in range(i * 100000, i * 100000 + n))
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text + "\nSummarize the text above in 3 sentences, then count the lines."}],
        "max_tokens": 256,
        "temperature": 0,
        "ignore_eos": True,
        "chat_template_kwargs": KW,
    }
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        u = d["usage"]
        results.append((i, u["prompt_tokens"], u["completion_tokens"], time.time() - t0, None))
    except Exception as e:  # noqa: BLE001
        results.append((i, 0, 0, time.time() - t0, repr(e)[:200]))


t0 = time.time()
ths = [threading.Thread(target=worker, args=(i,)) for i in range(C)]
[t.start() for t in ths]
[t.join() for t in ths]
wall = time.time() - t0
tot_out = sum(r[2] for r in results)
tot_in = sum(r[1] for r in results)
errs = [r for r in results if r[4]]
for r in sorted(results):
    print(f"  req{r[0]}: in={r[1]:,} out={r[2]} t={r[3]:.1f}s err={r[4]}")
print(f"C{C} @~{tot_in//max(C,1):,} in: wall={wall:.1f}s aggregate out={tot_out/wall:.0f} tok/s errors={len(errs)}")
sys.exit(1 if errs else 0)
