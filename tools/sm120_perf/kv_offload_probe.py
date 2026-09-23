#!/usr/bin/env python3
"""KV offload hit probe: is a long prompt's KV served from host RAM after it was evicted from the GPU?

Sequence (greedy, thinking off, max_tokens small so the wall time is the prefill / load time):
  1. prompt A (~--tokens tokens, a 6-digit needle inside, question at the end)  -> fresh prefill
  2. prompt A again                                                            -> GPU prefix-cache hit
  3. --evict distinct prompts of the same size                                 -> push A out of the GPU KV pool
  4. prompt A again                                                            -> offload hit (or a fresh prefill if there is none)
Every step prints wall time, prompt/completion tokens, the answer (must equal the needle each time) and the
deltas of the server's kv_offload_* / prefix-cache Prometheus counters.

--turn2-gen N adds a two-turn conversation around it (on the chat-rendered token ids, through /v1/completions):
turn 1 (a context of ~tokens/2, N generated tokens) runs before step 1, turn 2 (turn 1 + its answer + a follow-up)
after step 4, when both are out of the GPU pool. The report gives the part of turn 2 that repeats turn 1's prompt
and generated ids token for token; the offload hit covers the generated part only with offload_prompt_only=false.

--seed N makes prompt A the same text in every run and --only-a sends it once: run it before a restart and again
after it to see whether a disk tier (TieringOffloadingSpec, secondary tier "fs") serves the KV of the previous
server process.

usage: kv_offload_probe.py --base http://127.0.0.1:8001 --model deepseek-ai/DeepSeek-V4.1-Flash [--tokens 150000]
                           [--evict 6] [--turn2-gen 0] [--seed N [--only-a]]
                           [--metrics-url http://127.0.0.1:8001/metrics] [--out FILE]
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
                if 'tier="' in line:
                    name += "[" + line.split('tier="', 1)[1].split('"', 1)[0] + "]"
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


def post(base: str, path: str, body: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(base: str, model: str, prompt: str, max_tokens: int, timeout: int = 1800) -> tuple[dict, float]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    t0 = time.perf_counter()
    d = post(base, "/v1/chat/completions", body, timeout)
    return d, time.perf_counter() - t0


def render(base: str, model: str, messages: list[dict]) -> list[int]:
    return post(base, "/tokenize", {"model": model, "messages": messages, "add_generation_prompt": True,
                                    "chat_template_kwargs": {"thinking": False}})["tokens"]


def complete_ids(base: str, model: str, ids: list[int], max_tokens: int) -> tuple[str, list[int], float]:
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0, "logprobs": 0,
            "return_tokens_as_token_ids": True}
    t0 = time.perf_counter()
    c = post(base, "/v1/completions", body)["choices"][0]
    return c["text"], [int(t.split(":", 1)[1]) for t in c["logprobs"]["tokens"]], time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--tokens", type=int, default=150000, help="approximate prompt length in tokens")
    ap.add_argument("--evict", type=int, default=6, help="distinct prompts sent between the two hits")
    ap.add_argument("--turn2-gen", type=int, default=0, help="generated tokens of the two-turn check's turn 1 (0 = off)")
    ap.add_argument("--seed", type=int, default=None, help="the same prompt A in every run (restart persistence)")
    ap.add_argument("--only-a", action="store_true", help="send prompt A once and stop")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8001/metrics")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    n_lines = a.tokens // 30  # ~30 tokens per line
    random.seed(a.tokens if a.seed is None else a.seed)
    needle = f"{random.randint(100000, 999999)}"
    tag_a = uuid.uuid4().hex[:6] if a.seed is None else f"s{a.seed:05d}"[-6:]
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
        shown = {k.removeprefix("vllm:"): v for k, v in row["metrics_delta"].items()
                 if ("bytes" in k or "hits" in k or "queries" in k or "failure" in k or "_time" in k)
                 and "bucket" not in k and "created" not in k}
        if shown:
            print(f"                {shown}", flush=True)
        results.append(row)
        return row

    if a.only_a:
        print(f"prompt A: ~{a.tokens} tokens, needle {needle}, seed {a.seed}", flush=True)
        step("A", prompt_a)
        if a.out:
            json.dump({"tokens": a.tokens, "seed": a.seed, "needle": needle, "results": results}, open(a.out, "w"),
                      indent=1)
            print("saved", a.out)
        return

    turn2 = None
    if a.turn2_gen:
        user1 = make_prompt(uuid.uuid4().hex[:6], n_lines // 2, None).rsplit("\n", 1)[0] + (
            f"\nWrite a detailed essay of about {int(a.turn2_gen * 0.7)} words on how such a scheduler should share"
            " the KV cache between long agent sessions.")
        t1_ids = render(a.base, a.model, [{"role": "user", "content": user1}])
        m0 = scrape(a.metrics_url)
        text1, gen1, dt = complete_ids(a.base, a.model, t1_ids, a.turn2_gen)
        turn2 = {"turn1_prompt_tokens": len(t1_ids), "turn1_generated": len(gen1), "turn1_wall_s": round(dt, 3),
                 "turn1_metrics_delta": delta(m0, scrape(a.metrics_url))}
        print(f"{'T turn 1':>14}: {dt:7.2f} s  {len(t1_ids):>7} tok  generated {len(gen1)}", flush=True)

    print(f"prompt A: ~{a.tokens} tokens, needle {needle}, {a.evict} eviction prompts of the same size", flush=True)
    step("A fresh", prompt_a)
    step("A gpu-hit", prompt_a)
    for i in range(a.evict):
        step(f"evict {i+1}/{a.evict}", make_prompt(uuid.uuid4().hex[:6], n_lines, None), max_tokens=1)
    step("A after evict", prompt_a)
    step("A again", prompt_a)
    if turn2 is not None:
        t2_ids = render(a.base, a.model, [{"role": "user", "content": user1},
                                          {"role": "assistant", "content": text1.strip()},
                                          {"role": "user", "content": "Now list the three most important points of"
                                                                      " your essay in one line each."}])
        seq1 = t1_ids + gen1
        common = next((i for i in range(min(len(seq1), len(t2_ids))) if seq1[i] != t2_ids[i]),
                      min(len(seq1), len(t2_ids)))
        m0 = scrape(a.metrics_url)
        _, gen2, dt = complete_ids(a.base, a.model, t2_ids, 32)
        md = delta(m0, scrape(a.metrics_url))
        turn2.update({"turn2_prompt_tokens": len(t2_ids), "turn2_repeats_turn1_ids": common, "turn2_wall_s": round(dt, 3),
                      "turn2_metrics_delta": md})
        shown = {k.removeprefix("vllm:"): v for k, v in md.items()
                 if ("bytes" in k or "hits" in k or "queries" in k) and "bucket" not in k and "created" not in k}
        print(f"{'T turn 2':>14}: {dt:7.2f} s  {len(t2_ids):>7} tok  repeats turn 1's prompt+generated ids for {common}"
              f" tokens (turn 1 prompt {len(t1_ids)}, +{len(gen1)} generated)\n                {shown}", flush=True)
    if a.out:
        json.dump({"tokens": a.tokens, "evict": a.evict, "needle": needle, "results": results, "turn2": turn2},
                  open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
