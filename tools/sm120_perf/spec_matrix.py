#!/usr/bin/env python3
"""Decode throughput matrix for speculative-decoding settings: prose and code at concurrency 1 / 4 / 8.

One wave per (mode, concurrency): C concurrent streaming chat requests (thinking off, temperature 0,
``ignore_eos``, ``--output-tokens`` each; a nonce per request keeps the prompts distinct). Per wave:

  decode tok/s (window)   tokens all streams deliver inside the shared window
                          [max(first-token times), min(finish times)] / window length
                          (per-stream tokens in the window taken pro rata from its own rate)
  tok/s per request       median over streams of completion tokens / (finish - first token)
  steps/s, tok/step       median over streams of streamed chunks / decode time (one chunk = one
                          engine step = the tokens DSpark accepted) and tokens per chunk
  accepted/step, acceptance   from vLLM /metrics deltas (spec_decode_* counters) when --metrics-url
                          is given: draft tokens accepted per drafting step, accepted / drafted

usage:
  spec_matrix.py --base http://127.0.0.1:8001 --model deepseek-ai/DeepSeek-V4.1-Flash \
      [--concurrencies 1,4,8] [--modes prose,code] [--output-tokens 512] [--repeats 2] \
      [--metrics-url http://127.0.0.1:8001/metrics] [--label k5] --out results.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.request
import uuid

PROMPTS = {
    "prose": "Write a detailed essay about the history of GPU computing. Be thorough and long.",
    "code": (
        "Write a complete Python module implementing an LRU cache, a trie and Dijkstra with type hints. "
        "Code only."
    ),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrencies", default="1,4,8")
    ap.add_argument("--modes", default="prose,code")
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=2, help="waves per (mode, concurrency); the JSON keeps all")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--metrics-url", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def scrape(url: str) -> dict:
    if not url:
        return {}
    keys = (
        "spec_decode_num_accepted_tokens_total",
        "spec_decode_num_draft_tokens_total",
        "spec_decode_num_drafts_total",
        "generation_tokens_total",
    )
    out: dict = {}
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#"):
                    continue
                for key in keys:
                    if key in line:
                        try:
                            out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                        except ValueError:
                            pass
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    return out


def one_request(args, prompt: str, wave_start: float) -> dict:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex}] {prompt}"}],
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
    req = urllib.request.Request(
        args.base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers
    )
    first = None
    chunks = 0
    usage = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                d = json.loads(payload)
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        if first is None:
                            first = time.monotonic()
                        chunks += 1
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    end = time.monotonic()
    completion = usage["completion_tokens"] if usage else 0
    return dict(
        first=(first - wave_start) if first else None,
        end=end - wave_start,
        chunks=chunks,
        completion=completion,
        error=err,
    )


def wave(args, mode: str, conc: int) -> dict:
    before = scrape(args.metrics_url)
    wave_start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        reqs = list(ex.map(lambda _i: one_request(args, PROMPTS[mode], wave_start), range(conc)))
    after = scrape(args.metrics_url)
    ok = [r for r in reqs if r["error"] is None and r["first"] is not None and r["completion"] > 0]
    res = dict(mode=mode, concurrency=conc, errors=[r["error"] for r in reqs if r["error"]], n_ok=len(ok))
    if ok:
        dec = [r["end"] - r["first"] for r in ok]
        rates = [r["completion"] / d for r, d in zip(ok, dec)]
        res["tok_s_per_request_median"] = statistics.median(rates)
        res["steps_s_median"] = statistics.median(r["chunks"] / d for r, d in zip(ok, dec))
        res["tok_per_step_median"] = statistics.median(r["completion"] / max(r["chunks"], 1) for r in ok)
        w0, w1 = max(r["first"] for r in ok), min(r["end"] for r in ok)
        if w1 > w0:
            res["window_s"] = w1 - w0
            res["decode_tok_s_window"] = sum(rate * (w1 - w0) for rate in rates) / (w1 - w0)
        res["ttft_median_s"] = statistics.median(r["first"] for r in ok)
    if "spec_decode_num_drafts_total" in after and "spec_decode_num_drafts_total" in before:
        drafts = after["spec_decode_num_drafts_total"] - before["spec_decode_num_drafts_total"]
        acc = after.get("spec_decode_num_accepted_tokens_total", 0) - before.get(
            "spec_decode_num_accepted_tokens_total", 0
        )
        drafted = after.get("spec_decode_num_draft_tokens_total", 0) - before.get(
            "spec_decode_num_draft_tokens_total", 0
        )
        if drafts > 0:
            res["dspark_accepted_per_step"] = acc / drafts
            res["dspark_draft_tokens_per_step"] = drafted / drafts
            res["dspark_acceptance"] = acc / drafted if drafted else None
    return res


def main() -> int:
    args = parse_args()
    concs = [int(c) for c in args.concurrencies.split(",") if c]
    modes = [m for m in args.modes.split(",") if m]
    report = dict(base=args.base, model=args.model, label=args.label, output_tokens=args.output_tokens, waves=[])
    print(f"{'mode':<6} {'C':>2} {'tok/s win':>10} {'tok/s/req':>10} {'steps/s':>8} {'tok/step':>8} "
          f"{'acc/step':>8} {'drafted':>8} {'accept':>7}  err")
    for mode in modes:
        for conc in concs:
            for _ in range(args.repeats):
                r = wave(args, mode, conc)
                report["waves"].append(r)
                print(f"{mode:<6} {conc:>2} {r.get('decode_tok_s_window', float('nan')):>10.1f} "
                      f"{r.get('tok_s_per_request_median', float('nan')):>10.1f} "
                      f"{r.get('steps_s_median', float('nan')):>8.1f} {r.get('tok_per_step_median', float('nan')):>8.2f} "
                      f"{r.get('dspark_accepted_per_step', float('nan')):>8.2f} "
                      f"{r.get('dspark_draft_tokens_per_step', float('nan')):>8.2f} "
                      f"{(r.get('dspark_acceptance') or float('nan')):>7.3f}  {len(r['errors'])}", flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
