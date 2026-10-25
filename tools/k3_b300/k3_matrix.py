#!/usr/bin/env python3
"""Burst-serving matrix for Kimi K3 over the OpenAI chat API, to compare two configurations of the same server.

Same waves and metric definitions as vLLM-Moet's tools/sm120_perf/openai_matrix.py (one wave per (input size,
concurrency), a unique nonce opening every prompt, repeated filler text, a code-writing instruction, a forced output
budget with ignore_eos, decode measured in the window all streams share). Requests go through /v1/chat/completions so
the server's K3 chat template and reasoning parser are in the path; prompt sizes are calibrated from the server's
usage counts, chunk token counts are approximated by characters and scaled to the reported completion tokens, and
sampling defaults to K3's recommended temperature 1.0 / top_p 0.95.

Metrics per wave:
  prefill tok/s     = sum(input tokens) / (last first-token time - wave start), queueing included
  total decode      = tokens delivered by all streams inside the shared window
                      [max(first-token times), min(finish times)] / window length
  decode/request    = median over requests of (its tokens in the window / window length)
  TTFT              = median time to first token
  gap p95           = 95th percentile of the gaps between streamed chunks after the first token

usage:
  k3_matrix.py --base http://127.0.0.1:8000 --model moonshotai/Kimi-K3 --sizes 8192,32768 \
      --concurrencies 1,8,16,32,64 --output-tokens 1024 --out results.json [--api-key KEY] \
      [--metrics-url http://127.0.0.1:8000/metrics]
"""
import argparse
import concurrent.futures
import json
import os
import statistics
import time
import urllib.request
import uuid

FILLER = ("Reference notes: the cache stores recently accessed entries. "
          "An implementation should maintain ordering, handle replacement and validate its invariants.\n")
INSTRUCTION = ("\nNow write a complete Python LRU cache module with a doubly linked list and dictionary, "
               "including get, put, delete, iteration, resize, clear, invariant validation, detailed docstrings "
               "and ten usage examples. Return code only. Implement all methods fully.\n")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", default="8192,32768")
    ap.add_argument("--concurrencies", default="1,8,16,32")
    ap.add_argument("--output-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--no-ignore-eos", action="store_true")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--metrics-url", default="", help="vLLM /metrics to read DSpark acceptance per wave")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--timeout", type=int, default=3600)
    return ap.parse_args()


def post(args, body):
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
    req = urllib.request.Request(args.base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    return urllib.request.urlopen(req, timeout=args.timeout)


def messages(n_filler):
    return [{"role": "user", "content": uuid.uuid4().hex + "\nRead these notes.\n" + FILLER * n_filler + INSTRUCTION}]


def calibrate(args):
    def prompt_tokens(n_filler):
        with post(args, {"model": args.model, "messages": messages(n_filler), "max_tokens": 1}) as r:
            return json.load(r)["usage"]["prompt_tokens"]
    p0, p1 = prompt_tokens(0), prompt_tokens(200)
    per_filler = (p1 - p0) / 200
    return lambda size: max(0, round((size - p0) / per_filler))


def scrape(url):
    if not url:
        return {}
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            out = {}
            for line in r.read().decode().splitlines():
                if line.startswith("#"):
                    continue
                for key in ("spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total",
                            "spec_decode_num_drafts_total", "generation_tokens_total"):
                    if key in line:
                        try:
                            out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                        except ValueError:
                            pass
            return out
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def one_request(args, msgs, wave_start):
    body = {"model": args.model, "messages": msgs, "max_tokens": args.output_tokens, "temperature": args.temperature,
            "top_p": args.top_p, "stream": True, "stream_options": {"include_usage": True}}
    if not args.no_ignore_eos:
        body["ignore_eos"] = True
    t0 = time.monotonic()
    events = []  # (t_rel, approx_tokens)
    usage = None
    text_parts = []
    first = None
    err = None
    try:
        with post(args, body) as r:
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
                    delta = ch.get("delta") or {}
                    text = (delta.get("reasoning_content") or delta.get("reasoning") or "") + (delta.get("content") or "")
                    if text:
                        now = time.monotonic()
                        if first is None:
                            first = now
                        events.append((now - wave_start, len(text)))
                        text_parts.append(text)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    end = time.monotonic()
    approx = sum(n for _, n in events) or 1
    completion = usage["completion_tokens"] if usage else approx
    scale = completion / approx
    return dict(start=t0 - wave_start, first=(first - wave_start) if first else None, end=end - wave_start,
                prompt_tokens=(usage or {}).get("prompt_tokens", 0), completion_tokens=completion,
                events=[(t, n * scale) for t, n in events], error=err, text_head=("".join(text_parts))[:200])


def wave(args, fillers_for, size, conc):
    msgs_list = [messages(fillers_for(size)) for _ in range(conc)]
    before = scrape(args.metrics_url)
    wave_start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(conc) as pool:
        results = list(pool.map(lambda m: one_request(args, m, wave_start), msgs_list))
    after = scrape(args.metrics_url)
    ok = [r for r in results if r["error"] is None and r["first"] is not None]
    summary = dict(size=size, concurrency=conc, ok=len(ok), errors=[r["error"] for r in results if r["error"]])
    if ok:
        last_first = max(r["first"] for r in ok)
        summary["prompt_tokens_median"] = statistics.median(r["prompt_tokens"] for r in ok)
        summary["prefill_tok_s"] = sum(r["prompt_tokens"] for r in ok) / last_first
        summary["ttft_median_s"] = statistics.median(r["first"] for r in ok)
        w0, w1 = last_first, min(r["end"] for r in ok)
        summary["window_s"] = w1 - w0
        if w1 > w0:
            per = [sum(n for t, n in r["events"] if w0 < t <= w1) / (w1 - w0) for r in ok]
            summary["total_decode_tok_s"] = sum(per)
            summary["decode_per_request_tok_s"] = statistics.median(per)
        gaps = [b[0] - a[0] for r in ok for a, b in zip(r["events"], r["events"][1:])]
        if len(gaps) >= 20:
            summary["gap_p95_ms"] = 1000 * statistics.quantiles(gaps, n=20)[-1]
        summary["completion_tokens"] = [r["completion_tokens"] for r in ok]
        summary["wall_s"] = max(r["end"] for r in ok)
        summary["end_to_end_output_tok_s"] = sum(r["completion_tokens"] for r in ok) / summary["wall_s"]
    if before and after and "spec_decode_num_drafts_total" in after:
        drafts = after.get("spec_decode_num_drafts_total", 0) - before.get("spec_decode_num_drafts_total", 0)
        acc = after.get("spec_decode_num_accepted_tokens_total", 0) - before.get("spec_decode_num_accepted_tokens_total", 0)
        dr = after.get("spec_decode_num_draft_tokens_total", 0) - before.get("spec_decode_num_draft_tokens_total", 0)
        if drafts > 0:
            summary["dspark_accepted_per_step"] = acc / drafts
            summary["dspark_acceptance"] = acc / dr if dr else None
    summary["heads"] = [r["text_head"][:60] for r in ok[:2]]
    return summary, results


def main():
    args = parse_args()
    fillers_for = calibrate(args)
    sizes = [int(x) for x in args.sizes.split(",")]
    concs = [int(x) for x in args.concurrencies.split(",")]
    out = dict(label=args.label, base=args.base, model=args.model, output_tokens=args.output_tokens,
               temperature=args.temperature, top_p=args.top_p, ignore_eos=not args.no_ignore_eos,
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), waves=[])
    print(f"{'input':>8} {'C':>3} {'prefill':>9} {'total dec':>10} {'dec/req':>8} {'TTFT s':>7} {'gap p95':>8} "
          f"{'window':>7} {'acc/step':>8}")
    for size in sizes:
        for conc in concs:
            summary, raw = wave(args, fillers_for, size, conc)
            out["waves"].append(dict(summary=summary, requests=[{k: v for k, v in r.items() if k != "events"} for r in raw]))
            s = summary
            print(f"{size:>8} {conc:>3} {s.get('prefill_tok_s', 0):>9.1f} {s.get('total_decode_tok_s', 0):>10.1f} "
                  f"{s.get('decode_per_request_tok_s', 0):>8.1f} {s.get('ttft_median_s', 0):>7.2f} "
                  f"{s.get('gap_p95_ms', 0):>7.0f}ms {s.get('window_s', 0):>7.1f} "
                  f"{s.get('dspark_accepted_per_step', float('nan')):>8.2f}  err={len(s['errors'])}", flush=True)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
