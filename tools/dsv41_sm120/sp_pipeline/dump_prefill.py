#!/usr/bin/env python3
"""One fresh prefill of a *deterministic* prompt (the bits harness): the same text in every run, so the per-layer dumps
of two servers (VLLM_MOET_PREFILL_PIPE_DUMP) describe the same chunks of the same request.

usage: dump_prefill.py BASE MODEL LINES SEED [DUMP_DIR]
  LINES 560 ~ 19K tokens (prefill_time.py's recipe with seeded ids instead of uuid4); DUMP_DIR: touch DUMP_DIR/ENABLE
  before the request and remove it after, so only this prefill is dumped.
"""
import json
import os
import random
import sys
import time
import urllib.request

BASE, MODEL, LINES, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
DUMP_DIR = sys.argv[5] if len(sys.argv) > 5 else ""
rng = random.Random(SEED)
text = "".join(
    f"[{rng.getrandbits(32):08x}] Line {j}: the scheduler assigns blocks to requests and reclaims them when sequences "
    "finish; prefix caching reuses identical leading blocks across requests.\n"
    for j in range(LINES)
)
body = {
    "model": MODEL,
    "messages": [{"role": "user", "content": text + "\nReply with one word."}],
    "max_tokens": 1,
    "temperature": 0,
    "chat_template_kwargs": {"thinking": False},
}
flag = os.path.join(DUMP_DIR, "ENABLE") if DUMP_DIR else ""
if flag:
    os.makedirs(DUMP_DIR, exist_ok=True)
    open(flag, "w").close()
try:
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    dt = time.perf_counter() - t0
finally:
    if flag and os.path.exists(flag):
        time.sleep(2)  # the last step's dump lines
        os.remove(flag)
pin = d["usage"]["prompt_tokens"]
print(f"dump prefill seed {SEED}: {pin} tok in {dt*1000:.0f} ms; answer {d['choices'][0]['message']['content']!r}")
