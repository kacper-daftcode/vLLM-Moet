#!/usr/bin/env python3
"""Decode tok/s: short prompt, long ignore_eos generation; median of N runs."""
import json
import sys
import time
import urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "8091"
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 512
url = f"http://127.0.0.1:{port}/v1/completions"
body = {
    "model": "deepseek-v4-flash",
    "prompt": "The quick brown fox",
    "max_tokens": max_tokens,
    "temperature": 0.0,
    "ignore_eos": True,
}


def one(tag):
    t0 = time.perf_counter()
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    dt = time.perf_counter() - t0
    ct = out["usage"]["completion_tokens"]
    tps = ct / dt
    print(f"  {tag}: {ct} tok in {dt:.2f}s = {tps:.1f} tok/s")
    return tps


one("warmup")
vals = sorted(one(f"run {i+1}") for i in range(runs))
print(f"MEDIAN: {vals[len(vals)//2]:.1f} tok/s")
