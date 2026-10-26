"""Soak of the hybrid serve (phase 3): CONC concurrent streams for MINUTES, each a loop of chat completions with
mixed prompts (HumanEval tasks, MATH problems and plain questions from the probe data in /tmp/probe, or a built-in
list), greedy, thinking off; counts requests, tokens, errors (HTTP / stream failures), per-minute aggregate tok/s.

    python soak_probe.py --port 8041 --model deepseek-ai/DeepSeek-V4.1-Flash --minutes 120 --conc 4 --max-tokens 256
The server's `errors` KPI (the tier's fail-closed counter) is read from the container log by tp4_window.sh.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import threading
import time
import urllib.request

FALLBACK = ["Explain the difference between a process and a thread.", "Write a Python function that merges two sorted lists.",
            "What is the capital of Australia and why is it not Sydney?", "Summarize the plot of Hamlet in three sentences.",
            "Give three uses of the Fourier transform.", "How does TCP congestion control work?",
            "Write a SQL query that finds duplicate emails in a users table.", "What causes the seasons on Earth?"]


def load_prompts() -> list[str]:
    prompts = list(FALLBACK)
    he = "/tmp/probe/HumanEval.jsonl.gz"
    if os.path.exists(he):
        with gzip.open(he, "rt") as f:
            for line in f:
                d = json.loads(line)
                prompts.append("Complete this Python function. Return only code.\n\n" + d["prompt"])
    ma = "/tmp/probe/math500.jsonl"
    if os.path.exists(ma):
        for line in open(ma):
            d = json.loads(line)
            prompts.append("Solve the math problem. Reason step by step, concisely, and put the final answer in \\boxed{}.\n\n" + d["problem"])
    return prompts


def one(port: int, model: str, prompt: str, max_tokens: int) -> tuple[int, int, float]:
    body = dict(model=model, temperature=0, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}],
                chat_template_kwargs={"thinking": False})
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    u = d.get("usage", {})
    return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8041)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    prompts = load_prompts()
    lock = threading.Lock()
    stats = dict(requests=0, prompt_tokens=0, completion_tokens=0, errors=0, last_error="", minute_tokens=0)
    stop = time.time() + args.minutes * 60
    per_minute = []

    def worker(seed: int) -> None:
        rng = random.Random(seed)
        while time.time() < stop:
            p = rng.choice(prompts)
            try:
                pt, ct, dt = one(args.port, args.model, p, args.max_tokens)
                with lock:
                    stats["requests"] += 1
                    stats["prompt_tokens"] += pt
                    stats["completion_tokens"] += ct
                    stats["minute_tokens"] += ct
            except Exception as ex:
                with lock:
                    stats["errors"] += 1
                    stats["last_error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
                time.sleep(2)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.conc)]
    for t in threads:
        t.start()
    t0 = time.time()
    while time.time() < stop and any(t.is_alive() for t in threads):
        time.sleep(60)
        with lock:
            per_minute.append(stats["minute_tokens"])
            print(f"  [{(time.time() - t0) / 60:5.1f} min] requests {stats['requests']}, completion tokens {stats['completion_tokens']}, "
                  f"errors {stats['errors']}, last minute {stats['minute_tokens'] / 60:.1f} tok/s aggregate"
                  + (f"; last error {stats['last_error']}" if stats["last_error"] else ""), flush=True)
            stats["minute_tokens"] = 0
    for t in threads:
        t.join(timeout=1800)
    wall = time.time() - t0
    print(f"soak {args.minutes:.0f} min x {args.conc} streams: {stats['requests']} requests, {stats['completion_tokens']} completion tokens "
          f"({stats['completion_tokens'] / wall:.1f} tok/s aggregate), {stats['prompt_tokens']} prompt tokens, errors {stats['errors']}")
    if args.out:
        json.dump(dict(stats=stats, per_minute=per_minute, minutes=args.minutes, conc=args.conc), open(args.out, "w"), indent=1)
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
