#!/usr/bin/env python3
"""The Fraser bench: single-stream code decode on one fixed prompt, measured the way LocalMaxxing reports a
speed test: tokSOut = completion tokens / (last token - first token), TTFT from the first streamed token, tokSTotal =
(prompt + completion) / total time, DSpark acceptance from the server's spec-decode counters. Temperature 0,
max_tokens 2048, thinking off, a cache-bust nonce line first (as in LocalMaxxing prompts). The prompt next to this
file (fraser_prompt.txt) is the exact input of the runs reported in the README's Fraser bench section.

Report the median over the reps, not the best run: with nondeterministic serving paths the per-run acceptance -
hence tok/s - varies by several percent on identical hardware, and a best-of-N number mostly measures that spread.

usage: fraser_bench.py --base http://127.0.0.1:8001 --model NAME --prompt-file F [--reps 10] [--chat-kwargs JSON]
"""
import argparse
import json
import statistics
import time
import urllib.request
import uuid


def scrape(base):
    out = {}
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            for key in ("spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total",
                        "spec_decode_num_drafts_total"):
                if line.startswith("vllm:" + key):
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def one(a, text):
    body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": json.loads(a.chat_kwargs),
            **({"cache_salt": uuid.uuid4().hex} if a.fresh_prefill else {}),
            "messages": [{"role": "user", "content": f"[LocalMaxxing cache-bust nonce: {a.nonce or uuid.uuid4().hex}]\n{text}"}]}
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    before = scrape(a.base)
    t0 = time.perf_counter(); first = last = None; count = 0; usage = {}; pieces = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                pieces.append((ch.get("delta") or {}).get("content") or "")
            n = usage.get("completion_tokens", 0) if usage else 0
            if n > count:
                now = time.perf_counter()
                first = first or now
                last = now
                count = n
    after = scrape(a.base)
    if a.save_output:
        with open(a.save_output, "a") as f:
            f.write(json.dumps({"nonce": a.nonce, "text": "".join(pieces)}) + "\n")
    drafted = after["spec_decode_num_draft_tokens_total"] - before["spec_decode_num_draft_tokens_total"]
    accepted = after["spec_decode_num_accepted_tokens_total"] - before["spec_decode_num_accepted_tokens_total"]
    drafts = after["spec_decode_num_drafts_total"] - before["spec_decode_num_drafts_total"]
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": count,
            "ttft_ms": round((first - t0) * 1000, 1), "tok_s_out": round((count - 1) / (last - first), 1) if count > 1 else None,
            "tok_s_out_lm": round(count / (last - t0 - (first - t0)), 1) if count else None,
            "tok_s_total": round((usage.get("prompt_tokens", 0) + count) / (last - t0), 1),
            "acceptance": round(accepted / drafted, 3) if drafted else None,
            "mean_accepted_length": round(1 + accepted / drafts, 2) if drafts else None,
            "steps_s": round(drafts / (last - first), 1) if drafts and last > first else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--chat-kwargs", default='{"thinking": false}')
    ap.add_argument("--nonce", default=None, help="fixed cache-bust nonce (hex) instead of a random one per request")
    ap.add_argument("--fresh-prefill", action="store_true", help="a random cache_salt per request (no prefix-cache hit)")
    ap.add_argument("--save-output", default=None, help="append each request's generated text (JSON lines)")
    a = ap.parse_args()
    text = open(a.prompt_file).read()
    runs = [one(a, text) for _ in range(a.reps)]
    for r in runs:
        print(json.dumps(r), flush=True)
    print("median tok/s out:", statistics.median(r["tok_s_out_lm"] for r in runs),
          "| acceptance:", statistics.median(r["acceptance"] for r in runs),
          "| steps/s:", statistics.median(r["steps_s"] for r in runs), flush=True)


if __name__ == "__main__":
    main()
