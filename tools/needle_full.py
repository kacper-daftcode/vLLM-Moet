#!/usr/bin/env python3
"""needle_probe with a reasoning budget: PASS when the secret appears in the
final answer (after </think>) or anywhere in the content when finish=stop.

Usage: needle_full.py <port> <n_words> [depth] [model]
(model: 4th arg > NEEDLE_MODEL env > auto-detect from /v1/models)"""
import json
import os
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

MODEL = (sys.argv[4] if len(sys.argv) > 4 else "") or os.environ.get(
    "NEEDLE_MODEL", "")
if not MODEL:
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/v1/models", timeout=30) as mr:
            MODEL = json.loads(mr.read())["data"][0]["id"]
    except Exception:
        MODEL = "glm-5.2"      # old hardcoded default
body = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": user}],
    "max_tokens": 2000, "temperature": 0,
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                             data=body, headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
dt = time.perf_counter() - t0
ch = r["choices"][0]
msg = ch["message"]["content"] or ""
final = msg.split("</think>")[-1]
pt = r.get("usage", {}).get("prompt_tokens", 0)
ok = SECRET in final or (SECRET in msg and ch["finish_reason"] == "stop")
print(f"prompt_tokens={pt} ttft+gen={dt:.1f}s finish={ch['finish_reason']}")
print(f"final: {' '.join(final.split())[:100]!r}")
print("NEEDLE PASS" if ok else f"NEEDLE FAIL (tail: {' '.join(msg.split())[-120:]!r})")
sys.exit(0 if ok else 4)
