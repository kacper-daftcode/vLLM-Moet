#!/usr/bin/env python3
"""Single-stream code decode with DSpark counters, measured like vLLM-Moet's tools/sm120_perf/fraser_bench.py:
tok/s = completion tokens / (last token - first token), mean accepted length = 1 + accepted / drafts (the target's
own token included), a cache-bust nonce line first. K3 always thinks, so reasoning_effort stands in for fraser_bench's
"thinking off", and the per-position acceptance and the reasoning share of the completion are reported as well.

usage: k3_code_bench.py --base http://127.0.0.1:8000 --model moonshotai/Kimi-K3 --prompt-file F [--reps 3]
           [--temperature 0] [--top-p 1.0] [--reasoning-effort low] [--api-key KEY]
"""
import argparse
import json
import os
import statistics
import time
import urllib.request
import uuid


def counters(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        out = {"pos": {}}
        for line in r.read().decode().splitlines():
            if not line.startswith("vllm:spec_decode_num_"):
                continue
            name, value = line.rsplit(" ", 1)
            if "per_pos" in name:
                pos = int(name.split('position="')[1].split('"')[0])
                out["pos"][pos] = out["pos"].get(pos, 0.0) + float(value)
            else:
                key = name.split("{")[0].removeprefix("vllm:spec_decode_num_").removesuffix("_total")
                out[key] = out.get(key, 0.0) + float(value)
        return out


def run_once(a, prompt):
    body = {"model": a.model, "messages": [{"role": "user", "content": uuid.uuid4().hex + "\n" + prompt}],
            "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p, "stream": True,
            "stream_options": {"include_usage": True}}
    if a.reasoning_effort:
        body["reasoning_effort"] = a.reasoning_effort
    headers = {"Content-Type": "application/json"}
    if a.api_key:
        headers["Authorization"] = "Bearer " + a.api_key
    before = counters(a.base)
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    first = last = None
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    last = time.monotonic()
                    first = first or last
    after = counters(a.base)
    drafts = after.get("drafts", 0) - before.get("drafts", 0)
    accepted = after.get("accepted_tokens", 0) - before.get("accepted_tokens", 0)
    drafted = after.get("draft_tokens", 0) - before.get("draft_tokens", 0)
    tokens = usage.get("completion_tokens", 0)
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return {"tok_s": tokens / (last - first) if last and last > first else 0.0,
            "completion_tokens": tokens, "reasoning_share": reasoning / tokens if tokens else 0.0,
            "mean_accepted_length": 1 + accepted / drafts if drafts else None,
            "acceptance": accepted / drafted if drafted else None,
            "per_position": [(after["pos"].get(p, 0) - before["pos"].get(p, 0)) / drafts if drafts else None
                             for p in sorted(after["pos"])],
            "steps_s": drafts / (last - first) if drafts and last and last > first else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--reasoning-effort", default="low", help='"low", "high", "max" or "" for the server default')
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    prompt = open(a.prompt_file).read()
    runs = [run_once(a, prompt) for _ in range(a.reps)]
    med = lambda k: statistics.median(r[k] for r in runs if r[k] is not None)  # noqa: E731
    pos = [statistics.median(r["per_position"][i] for r in runs) for i in range(len(runs[0]["per_position"]))]
    print(f"{a.label:<28} tok/s {med('tok_s'):6.1f} | accepted length {med('mean_accepted_length'):.2f} | "
          f"steps/s {med('steps_s'):5.1f} | reasoning {100 * med('reasoning_share'):3.0f}% of "
          f"{med('completion_tokens'):.0f} tok | per position " + " ".join(f"{p:.2f}" for p in pos), flush=True)


if __name__ == "__main__":
    main()
