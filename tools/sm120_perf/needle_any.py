#!/usr/bin/env python3
"""Needle-in-a-haystack + coherence check for any vLLM chat model.

usage: needle_any.py BASE MODEL '{"chat_template_kwargs": {...}}' [target_tokens,...]
"""
import json
import random
import sys
import time
import urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
EXTRA = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
TARGETS = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [30000, 100000]


def chat(messages, max_tokens, timeout=1800):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    body.update(EXTRA)
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


FILLER = ("The quick brown fox jumps over the lazy dog. A journey of a thousand miles begins with a single step. "
          "All that glitters is not gold. Fortune favours the bold. Actions speak louder than words. ")


def needle(target_tokens, depth):
    random.seed(target_tokens + int(depth * 100))
    code = f"{random.randint(100000, 999999)}"
    needle_txt = f"\n\nThe secret access code for the vault is {code}. Remember it.\n\n"
    n_fill = int(target_tokens * 4.2 / len(FILLER))  # ~4.2 chars/token for this filler
    parts = [FILLER] * n_fill
    parts.insert(int(len(parts) * depth), needle_txt)
    doc = "".join(parts)
    msgs = [{"role": "user", "content": doc + "\n\nWhat is the secret access code for the vault? Answer with the number only."}]
    d, dt = chat(msgs, 24)
    ans = d["choices"][0]["message"]["content"]
    ok = code in (ans or "")
    print(f"needle ~{d['usage']['prompt_tokens']} tok depth {depth:.1f}: {'PASS' if ok else 'FAIL'} (answer={ans!r}, {dt:.1f}s)")
    return ok


ok_all = True
for t in TARGETS:
    for depth in (0.2, 0.7):
        ok_all &= needle(t, depth)
d, dt = chat([{"role": "user", "content": "Explain in 3 short sentences why the sky is blue."}], 120)
print("coherence:", repr(d["choices"][0]["message"]["content"][:400]))
d, dt = chat([{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively. Code only."}], 160)
print("code:", repr(d["choices"][0]["message"]["content"][:400]))
print("ALL PASS" if ok_all else "SOME FAIL")
