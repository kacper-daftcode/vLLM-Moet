#!/usr/bin/env python3
"""Burst-serving matrix over an OpenAI-compatible endpoint (vLLM or SGLang), one client for both.

Reproduces the methodology of 0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000 `benchmarks/matrix.py`
(exact-size token-id prompts: chat-template prefix with a unique nonce, repeated filler text, an
LRU-cache instruction, forced output budget with ignore_eos, one wave per (input size, concurrency),
shared decode window across all streams) but through `/v1/completions`, so that two engines are
measured by the same client with identical token ids.

Metrics per wave (0xSero's definitions):
  prefill tok/s     = sum(input tokens) / (last first-token time - wave start), queueing included
  total decode      = tokens delivered by all streams inside the shared window
                      [max(first-token times), min(finish times)] / window length
  decode/request    = median over requests of (its tokens in the window / window length)
  TTFT              = median time to first token

usage:
  openai_matrix.py --base http://127.0.0.1:8001 --model deepseek-ai/DeepSeek-V4.1-Flash \
      --model-dir /path/to/DeepSeek-V4.1-Flash --sizes 512,2048,8192,32768 --concurrencies 1,4,8 \
      --output-tokens 1024 --out results.json [--api-key KEY] [--metrics-url http://127.0.0.1:8001/metrics]
"""
import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.request
import uuid


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-dir", required=True, help="checkpoint dir (tokenizer.json + encoding/)")
    ap.add_argument("--sizes", default="512,2048,8192,32768")
    ap.add_argument("--concurrencies", default="1,4,8")
    ap.add_argument("--output-tokens", type=int, default=1024)
    ap.add_argument("--no-ignore-eos", action="store_true")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--metrics-url", default="", help="vLLM /metrics to read DSpark acceptance per wave")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--timeout", type=int, default=3600)
    return ap.parse_args()


def build_prompt_ids(model_dir):
    sys.path.insert(0, os.path.join(model_dir, "encoding"))
    from encoding import encode_messages  # the checkpoint's own chat encoder
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

    def encode(text):
        return tok.encode(text, add_special_tokens=False).ids

    filler = encode("Reference notes: the cache stores recently accessed entries. "
                    "An implementation should maintain ordering, handle replacement and validate its invariants.\n")
    instruction = ("\nNow write a complete Python LRU cache module with a doubly linked list and dictionary, "
                   "including get, put, delete, iteration, resize, clear, invariant validation, detailed docstrings "
                   "and ten usage examples. Return code only. Implement all methods fully.\n<｜Assistant｜></think>")
    suffix = encode(instruction)

    def make(size):
        nonce = uuid.uuid4().hex
        prefix = encode(encode_messages([dict(role="user", content=nonce + "\nRead these notes.\n")],
                                        thinking_mode="chat").split("<｜Assistant｜>")[0])
        room = size - len(prefix) - len(suffix)
        assert room >= 0, f"size {size} too small"
        ids = prefix + (filler * (room // len(filler) + 1))[:room] + suffix
        assert len(ids) == size
        return ids

    return make, tok


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


def one_request(args, ids, wave_start, tok):
    body = {"model": args.model, "prompt": ids, "max_tokens": args.output_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    if not args.no_ignore_eos:
        body["ignore_eos"] = True
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
    req = urllib.request.Request(args.base.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    t0 = time.monotonic()
    events = []  # (t_rel, approx_tokens)
    usage = None
    text_parts = []
    first = None
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
                    text = ch.get("text") or ""
                    if text:
                        now = time.monotonic()
                        if first is None:
                            first = now
                        n = len(tok.encode(text, add_special_tokens=False).ids)
                        events.append((now - wave_start, max(n, 1)))
                        text_parts.append(text)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    end = time.monotonic()
    completion = usage["completion_tokens"] if usage else sum(n for _, n in events)
    approx = sum(n for _, n in events) or 1
    scale = completion / approx
    return dict(start=t0 - wave_start, first=(first - wave_start) if first else None, end=end - wave_start,
                prompt_tokens=(usage or {}).get("prompt_tokens", len(ids)), completion_tokens=completion,
                events=[(t, n * scale) for t, n in events], error=err, text_head=("".join(text_parts))[:200])


def wave(args, make, tok, size, conc):
    ids_list = [make(size) for _ in range(conc)]
    before = scrape(args.metrics_url)
    wave_start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(conc) as pool:
        results = list(pool.map(lambda ids: one_request(args, ids, wave_start, tok), ids_list))
    after = scrape(args.metrics_url)
    ok = [r for r in results if r["error"] is None and r["first"] is not None]
    summary = dict(size=size, concurrency=conc, ok=len(ok), errors=[r["error"] for r in results if r["error"]])
    if ok:
        last_first = max(r["first"] for r in ok)
        summary["prefill_tok_s"] = sum(r["prompt_tokens"] for r in ok) / last_first
        summary["ttft_median_s"] = statistics.median(r["first"] for r in ok)
        w0, w1 = last_first, min(r["end"] for r in ok)
        summary["window_s"] = w1 - w0
        if w1 > w0:
            per = [sum(n for t, n in r["events"] if w0 < t <= w1) / (w1 - w0) for r in ok]
            summary["total_decode_tok_s"] = sum(per)
            summary["decode_per_request_tok_s"] = statistics.median(per)
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
    make, tok = build_prompt_ids(args.model_dir)
    sizes = [int(x) for x in args.sizes.split(",")]
    concs = [int(x) for x in args.concurrencies.split(",")]
    out = dict(label=args.label, base=args.base, model=args.model, output_tokens=args.output_tokens,
               ignore_eos=not args.no_ignore_eos, started=time.strftime("%Y-%m-%dT%H:%M:%S"), waves=[])
    print(f"{'input':>8} {'C':>2} {'prefill':>9} {'total dec':>10} {'dec/req':>8} {'TTFT s':>7} {'window':>7} {'acc/step':>8}")
    for size in sizes:
        for conc in concs:
            summary, raw = wave(args, make, tok, size, conc)
            out["waves"].append(dict(summary=summary, requests=[{k: v for k, v in r.items() if k != "events"} for r in raw]))
            s = summary
            print(f"{size:>8} {conc:>2} {s.get('prefill_tok_s', 0):>9.1f} {s.get('total_decode_tok_s', 0):>10.1f} "
                  f"{s.get('decode_per_request_tok_s', 0):>8.1f} {s.get('ttft_median_s', 0):>7.2f} {s.get('window_s', 0):>7.1f} "
                  f"{s.get('dspark_accepted_per_step', float('nan')):>8.2f}  err={len(s['errors'])}", flush=True)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
