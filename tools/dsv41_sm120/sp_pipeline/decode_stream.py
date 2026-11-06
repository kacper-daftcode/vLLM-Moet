#!/usr/bin/env python3
"""One decode stream (temperature 0, 2048 tokens, thinking off, a cache-bust nonce line first, streaming - the shape of
a LocalMaxxing-style speed test) that also asks for per-token logprobs and saves every token (text + top-1 logprob)
with the speed metrics, so the same prompt run alone / batched / under a concurrent 163K prefill / on another image
can be compared token by token (mixed_compare.py).

usage: decode_stream.py --base URL --model NAME --prompt-file F --nonce HEX --out run.jsonl [--reps 1] [--fresh-prefill]
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


def one(a, text, rep):
    body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": json.loads(a.chat_kwargs),
            "logprobs": True, "top_logprobs": 1,
            **({"cache_salt": uuid.uuid4().hex} if a.fresh_prefill else {}),
            "messages": [{"role": "user", "content": f"[LocalMaxxing cache-bust nonce: {a.nonce}]\n{text}"}]}
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    before = scrape(a.base)
    t0 = time.perf_counter(); first = last = None; count = 0; usage = {}; pieces = []; toks = []; finish = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                pieces.append((ch.get("delta") or {}).get("content") or "")
                for lp in ((ch.get("logprobs") or {}).get("content") or []):
                    toks.append([lp.get("token"), lp.get("logprob")])
                finish = ch.get("finish_reason") or finish
            n = usage.get("completion_tokens", 0) if usage else 0
            if n > count:
                now = time.perf_counter()
                first = first or now
                last = now
                count = n
    after = scrape(a.base)
    drafted = after["spec_decode_num_draft_tokens_total"] - before["spec_decode_num_draft_tokens_total"]
    accepted = after["spec_decode_num_accepted_tokens_total"] - before["spec_decode_num_accepted_tokens_total"]
    drafts = after["spec_decode_num_drafts_total"] - before["spec_decode_num_drafts_total"]
    res = {"nonce": a.nonce, "rep": rep, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": count,
           "finish_reason": finish, "n_logprob_tokens": len(toks),
           "ttft_ms": round((first - t0) * 1000, 1) if first else None,
           "tok_s_out": round((count - 1) / (last - first), 1) if count > 1 and last > first else None,
           "wall_s": round(time.perf_counter() - t0, 2),
           "acceptance": round(accepted / drafted, 3) if drafted else None,
           "steps_s": round(drafts / (last - first), 1) if drafts and last and last > first else None,
           "text": "".join(pieces), "tokens": toks}
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8066")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--nonce", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--chat-kwargs", default='{"thinking": false}')
    ap.add_argument("--fresh-prefill", action="store_true")
    a = ap.parse_args()
    text = open(a.prompt_file).read()
    runs = [one(a, text, rep) for rep in range(a.reps)]
    for r in runs:
        print(json.dumps({k: v for k, v in r.items() if k not in ("text", "tokens")}), flush=True)
    sp = [r["steps_s"] for r in runs if r["steps_s"]]
    print(f"nonce {a.nonce}: completion tokens {[r['completion_tokens'] for r in runs]}, median tok/s out "
          f"{statistics.median(r['tok_s_out'] for r in runs if r['tok_s_out']) if any(r['tok_s_out'] for r in runs) else None}, "
          f"steps/s {statistics.median(sp) if sp else None}", flush=True)


if __name__ == "__main__":
    main()
