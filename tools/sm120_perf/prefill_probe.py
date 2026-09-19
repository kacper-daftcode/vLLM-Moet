#!/usr/bin/env python3
"""Fresh-prompt prefill timing + GPU memory peak sampling (nvidia-smi) during the request."""
import json
import random
import subprocess
import sys
import threading
import time
import urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
GPUS = sys.argv[3] if len(sys.argv) > 3 else "0,1,2,3"
NTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 16000
KW = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}

peak = {"mem": 0}
stop = threading.Event()


def sampler():
    while not stop.is_set():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", GPUS],
                             capture_output=True, text=True).stdout
        vals = [int(v) for v in out.split() if v.strip().isdigit()]
        if vals:
            peak["mem"] = max(peak["mem"], max(vals))
        time.sleep(0.05)


random.seed(time.time())
words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa", "lambda", "zeta"]
prompt = " ".join(f"{random.choice(words)}{random.randint(0, 99999)}" for _ in range(NTOK // 3)) + "\nSummarize the above in one sentence."
body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True}}
if KW:
    body["chat_template_kwargs"] = KW
th = threading.Thread(target=sampler, daemon=True)
th.start()
req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
ttft = None
usage = None
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        if d.get("usage"):
            usage = d["usage"]
        if ttft is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
            ttft = time.perf_counter() - t0
stop.set()
th.join()
pin = usage["prompt_tokens"]
print(f"prefill {pin} tok: TTFT {ttft*1000:.0f} ms = {pin/ttft:.0f} tok/s; peak GPU mem during request {peak['mem']} MiB")
