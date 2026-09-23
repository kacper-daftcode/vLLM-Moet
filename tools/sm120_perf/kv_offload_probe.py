#!/usr/bin/env python3
"""KV offload hit probe: is a long prompt's KV served from host RAM after it was evicted from the GPU?

Sequence (greedy, thinking off, max_tokens small so the wall time is the prefill / load time):
  1. prompt A (~--tokens tokens, a 6-digit needle inside, question at the end)  -> fresh prefill
  2. prompt A again                                                            -> GPU prefix-cache hit
  3. --evict distinct prompts of the same size                                 -> push A out of the GPU KV pool
  4. prompt A again                                                            -> offload hit (or a fresh prefill if there is none)
Every step prints wall time, prompt/completion tokens, the answer (must equal the needle each time) and the
deltas of the server's kv_offload_* / prefix-cache Prometheus counters.

usage: kv_offload_probe.py --base http://127.0.0.1:8001 --model deepseek-ai/DeepSeek-V4.1-Flash [--tokens 150000]
                           [--evict 6] [--metrics-url http://127.0.0.1:8001/metrics] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
import uuid

LINE = "[{tag}] Line {j}: the scheduler assigns blocks to requests and reclaims them when sequences finish; prefix caching reuses identical leading blocks across requests.\n"


def scrape(url: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not url:
        return out
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                name = line.split("{", 1)[0].split(" ", 1)[0]
                if "kv_offload" in name or "prefix_cache" in name or "external" in name:
                    try:
                        out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
                    except ValueError:
                        pass
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)  # type: ignore[assignment]
    return out


def delta(a: dict, b: dict) -> dict:
    return {k: round(b[k] - a.get(k, 0.0), 3) for k in b if isinstance(b[k], float) and b[k] != a.get(k, 0.0)}


def make_prompt(tag: str, n_lines: int, needle: str | None, depth: float = 0.5) -> str:
    lines = [LINE.format(tag=f"{tag}-{j % 977:03d}", j=j) for j in range(n_lines)]
    if needle:
        lines.insert(int(len(lines) * depth), f"\nThe secret access code for the vault is {needle}. Remember it.\n\n")
    text = "".join(lines)
    q = "\nWhat is the secret access code for the vault? Answer with the 6 digits only." if needle else "\nReply with one word."
    return text + q


def chat(base: str, model: str, prompt: str, max_tokens: int, timeout: int = 1800) -> tuple[dict, float]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--tokens", type=int, default=150000, help="approximate prompt length in tokens")
    ap.add_argument("--evict", type=int, default=6, help="distinct prompts sent between the two hits")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8001/metrics")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    n_lines = a.tokens // 30  # ~30 tokens per line
    random.seed(a.tokens)
    needle = f"{random.randint(100000, 999999)}"
    tag_a = uuid.uuid4().hex[:6]
    prompt_a = make_prompt(tag_a, n_lines, needle)
    results = []

    def step(name: str, prompt: str, max_tokens: int = 8) -> dict:
        m0 = scrape(a.metrics_url)
        d, dt = chat(a.base, a.model, prompt, max_tokens)
        m1 = scrape(a.metrics_url)
        ans = d["choices"][0]["message"]["content"].strip()
        u = d["usage"]
        row = {"step": name, "wall_s": round(dt, 3), "prompt_tokens": u["prompt_tokens"],
               "completion_tokens": u["completion_tokens"], "answer": ans[:40],
               "answer_ok": (needle in ans) if prompt is prompt_a else None,
               "tok_s": round(u["prompt_tokens"] / dt), "metrics_delta": delta(m0, m1)}
        print(f"{name:>14}: {dt:7.2f} s  {u['prompt_tokens']:>7} tok  ({row['tok_s']:>6} tok/s)  answer={ans[:24]!r}"
              f"{'' if row['answer_ok'] is None else ('  OK' if row['answer_ok'] else '  WRONG')}", flush=True)
        shown = {k.split(":")[-1]: v for k, v in row["metrics_delta"].items()
                 if ("bytes" in k or "hits" in k or "queries" in k or "failure" in k) and "bucket" not in k}
        if shown:
            print(f"                {shown}", flush=True)
        results.append(row)
        return row

    print(f"prompt A: ~{a.tokens} tokens, needle {needle}, {a.evict} eviction prompts of the same size", flush=True)
    step("A fresh", prompt_a)
    step("A gpu-hit", prompt_a)
    for i in range(a.evict):
        step(f"evict {i+1}/{a.evict}", make_prompt(uuid.uuid4().hex[:6], n_lines, None), max_tokens=1)
    step("A after evict", prompt_a)
    step("A again", prompt_a)
    if a.out:
        json.dump({"tokens": a.tokens, "evict": a.evict, "needle": needle, "results": results}, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
