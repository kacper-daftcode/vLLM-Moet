#!/usr/bin/env python3
"""Start torch profiler on a vLLM server, run prose+code decode, stop.

usage: profile_capture.py [BASE] [MODEL] [CHAT_TEMPLATE_KWARGS_JSON] [N_TOKENS] [CONCURRENCY]
  CONCURRENCY > 1 runs that many concurrent prose streams (then code streams) inside the
  profiled window, e.g. 8 for the 48-tokens-per-step DSpark k=5 decode shape.
"""
import concurrent.futures
import json
import sys
import time
import urllib.request

B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
M = sys.argv[2] if len(sys.argv) > 2 else "deepseek-ai/DeepSeek-V4.1-Flash"
KW = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {"thinking": False}
N = int(sys.argv[4]) if len(sys.argv) > 4 else 96
C = int(sys.argv[5]) if len(sys.argv) > 5 else 1


def post(path, body=None, timeout=900):
    req = urllib.request.Request(B + path, data=json.dumps(body).encode() if body is not None else b"",
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, raw


def gen(prompt, n, nonce=""):
    body = {"model": M, "messages": [{"role": "user", "content": prompt + nonce}], "max_tokens": n,
            "temperature": 0, "ignore_eos": True, "chat_template_kwargs": KW}
    t0 = time.time()
    st, raw = post("/v1/chat/completions", body)
    d = json.loads(raw)
    dt = time.time() - t0
    return d["usage"]["completion_tokens"], dt


def wave(prompt, n, c):
    t0 = time.time()
    if c == 1:
        toks, dt = gen(prompt, n)
        print(f"  {toks} tok in {dt:.2f}s = {toks/dt:.0f} tok/s")
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        res = list(ex.map(lambda i: gen(prompt, n, f" (request {i})"), range(c)))
    dt = time.time() - t0
    toks = sum(r[0] for r in res)
    print(f"  {c} streams x {n} tok in {dt:.2f}s = {toks/dt:.0f} tok/s aggregate")


print("warmup"); wave("Say hello in five words.", 8, C)
st, _ = post("/start_profile"); print("start_profile ->", st)
t0 = time.time()
print("prose"); wave("Write a detailed essay about the history of GPU computing. Be thorough and long.", N, C)
print("code"); wave("Write a complete Python module implementing an LRU cache, a trie and Dijkstra with type hints. Code only.", N, C)
print(f"profiled window {time.time()-t0:.1f}s")
st, _ = post("/stop_profile", timeout=1800); print("stop_profile ->", st)
