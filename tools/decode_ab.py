#!/usr/bin/env python3
"""Standardized decode A/B probe for spec-decode / quantization variants.

Runs N identical greedy code-prompt generations against a vLLM server and
reports per-run tok/s, median/spread, and output determinism (distinct
outputs — expected 1 unless a runtime precision tier perturbs logits).
Optionally greps the newest SpecDecoding metrics line from a server log.

Usage: decode_ab.py [port] [runs] [max_tokens] [server_log]
"""
import hashlib
import json
import statistics
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8100"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 400
LOG = sys.argv[4] if len(sys.argv) > 4 else None

PROMPT = '''Refactor this function to use pathlib and add type hints:

import os

def find_files(directory, extension):
    results = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(extension):
                results.append(os.path.join(root, f))
    return results

Refactored version:
'''

with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",
                            timeout=30) as r:
    model = json.load(r)["data"][0]["id"]

texts, speeds = [], []
for run in range(RUNS + 1):          # +1 warmup, dropped
    body = json.dumps({"model": model, "prompt": PROMPT,
                       "max_tokens": MAXTOK, "temperature": 0,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.perf_counter() - t0
    ct = d["usage"]["completion_tokens"]
    if run == 0:
        continue
    text = d["choices"][0]["text"]
    texts.append(text)
    speeds.append(ct / dt)
    print(f"run {run}: {ct / dt:6.1f} tok/s  "
          f"sha={hashlib.sha1(text.encode()).hexdigest()[:10]}")

speeds.sort()
med = statistics.median(speeds)
print(f"\nMEDIAN {med:.1f} tok/s  min {speeds[0]:.1f}  max {speeds[-1]:.1f}  "
      f"spread {(speeds[-1] - speeds[0]) / med * 100:.0f}%")
print(f"distinct outputs: {len(set(texts))}/{len(texts)}")

if LOG:
    try:
        lines = [ln for ln in open(LOG, errors="ignore")
                 if "SpecDecoding metrics" in ln]
        if lines:
            print("last acceptance:", lines[-1].split("SpecDecoding metrics:")[1].strip())
    except OSError:
        pass
