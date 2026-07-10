#!/usr/bin/env python3
"""Long-context needle retrieval over the chat endpoint (thinking off).

Embeds a unique secret token at a given depth inside ~N words of filler and asks
the model to echo it. Confirms the sparse-MLA attention path is not just
non-crashing but CORRECT at long range. Pass = the secret appears in the reply.

Usage: python3 tools/needle_probe.py <port> <filler_words> [depth_frac]
  e.g. python3 tools/needle_probe.py 8000 48000 0.1
"""
import json
import random
import sys
import time
import urllib.request

PORT = sys.argv[1]
NWORDS = int(sys.argv[2])
DEPTH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
SECRET = "GLACIER-7741-ORYX"
WORDS = ("alpha quantum river matrix ember glacier syntax violet nimbus cobalt "
         "tangent fjord lantern zephyr cipher marble thunder willow plasma onyx").split()

random.seed(7)
filler = [random.choice(WORDS) for _ in range(NWORDS)]
at = max(0, min(len(filler), int(len(filler) * DEPTH)))
needle = (f"IMPORTANT FACT: the vault passphrase is {SECRET}. "
          f"Remember it exactly.")
ctx = " ".join(filler[:at]) + "\n\n" + needle + "\n\n" + " ".join(filler[at:])
user = (ctx + "\n\nQuestion: What is the vault passphrase? "
        "Reply with ONLY the passphrase, nothing else.")

import os
MODEL = os.environ.get("BENCH_MODEL")
if not MODEL:
    with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/v1/models", timeout=30) as _r:
        MODEL = json.load(_r)["data"][0]["id"]
body = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": user}],
    # reasoning models (Kimi-K2.x) burn budget on thinking before the
    # answer: give headroom and scan reasoning+content for the secret.
    "max_tokens": int(os.environ.get("NEEDLE_MAX_TOKENS", "512")),
    "temperature": 0,
    "chat_template_kwargs": {"thinking": False},
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                             data=body, headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
dt = time.perf_counter() - t0
m = r["choices"][0]["message"]
msg = (m.get("content") or "") + " || " + (m.get("reasoning") or
                                           m.get("reasoning_content") or "")
pt = r.get("usage", {}).get("prompt_tokens", 0)
ok = SECRET in msg
print(f"prompt_tokens={pt} depth={DEPTH} ttft+gen={dt:.1f}s")
print(f"reply: {msg!r}")
print("NEEDLE PASS" if ok else "NEEDLE FAIL")
sys.exit(0 if ok else 4)
