#!/usr/bin/env python3
"""Escalating cold-prefill probe / crash-localizer for the sparse-MLA prefill path.

Sends a unique random prompt of N words (max_tokens=1, temp 0) for each N in a
ramp, reporting prompt_tokens + TTFT and STOPPING on the first failure (non-200,
connection refused, timeout, engine-dead). Used to (a) reproduce the 512k cubit
prefill crash and localize the prompt-size trigger, and (b) validate a fix.

Usage:  python3 tools/prefill_probe.py <port> <word_counts_csv> [max_tokens]
  e.g.  python3 tools/prefill_probe.py 8000 16000,100000,300000
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request

PORT = sys.argv[1]
COUNTS = [int(x) for x in sys.argv[2].split(",")]
MAX_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 1
WORDS = ("alpha quantum river matrix ember glacier syntax violet nimbus cobalt "
         "tangent fjord lantern zephyr cipher marble thunder willow plasma onyx "
         "harbor crimson vector lattice meadow falcon pixel saffron tundra orbit").split()


def probe(nwords: int) -> None:
    random.seed(0xC0FFEE ^ nwords)
    prompt = " ".join(random.choice(WORDS) for _ in range(nwords))
    body = json.dumps({
        "model": "deepseek-v4-flash", "prompt": prompt,
        "max_tokens": MAX_TOK, "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=1800).read()
    dt = time.perf_counter() - t0
    d = json.loads(r)
    pt = d.get("usage", {}).get("prompt_tokens", 0)
    print(f"  OK  ~{nwords} words -> {pt} prompt_tokens : {dt:.1f} s "
          f"({pt / dt:,.0f} tok/s)", flush=True)


for n in COUNTS:
    print(f"[probe] {n} words ...", flush=True)
    try:
        probe(n)
    except urllib.error.HTTPError as e:
        print(f"  FAIL HTTP {e.code}: {e.read()[:300]!r}", flush=True)
        sys.exit(2)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL ({type(e).__name__}): {e}", flush=True)
        sys.exit(3)
print("[probe] all sizes completed without failure", flush=True)
