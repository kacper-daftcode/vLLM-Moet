#!/usr/bin/env python3
"""Start torch profiler on a vLLM server, run prose+code decode, stop."""
import json
import sys
import time
import urllib.request

B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
M = sys.argv[2] if len(sys.argv) > 2 else "deepseek-ai/DeepSeek-V4.1-Flash"
KW = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {"thinking": False}
N = int(sys.argv[4]) if len(sys.argv) > 4 else 96


def post(path, body=None, timeout=900):
    req = urllib.request.Request(B + path, data=json.dumps(body).encode() if body is not None else b"",
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, raw


def gen(prompt, n):
    body = {"model": M, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0,
            "ignore_eos": True, "chat_template_kwargs": KW}
    t0 = time.time()
    st, raw = post("/v1/chat/completions", body)
    d = json.loads(raw)
    dt = time.time() - t0
    print(f"  {d['usage']['completion_tokens']} tok in {dt:.2f}s = {d['usage']['completion_tokens']/dt:.0f} tok/s")


print("warmup"); gen("Say hello in five words.", 8)
st, _ = post("/start_profile"); print("start_profile ->", st)
t0 = time.time()
print("prose"); gen("Write a detailed essay about the history of GPU computing. Be thorough and long.", N)
print("code"); gen("Write a complete Python module implementing an LRU cache, a trie and Dijkstra with type hints. Code only.", N)
print(f"profiled window {time.time()-t0:.1f}s")
st, _ = post("/stop_profile", timeout=1800); print("stop_profile ->", st)
