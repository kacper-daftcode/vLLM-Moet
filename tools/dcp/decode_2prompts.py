#!/usr/bin/env python3
"""Two-prompt decode probe: tok/s + steps/s + MTP acceptance, per prompt.

Reconstruction of the 2026-07-12 DCP-session probe. Runs two fixed greedy
prompts (the decode_ab.py "refactor" code prompt and the bench_tok.py
"fox" prompt) N times each against a vLLM server and reports per-run
tok/s plus engine-step rate and speculative acceptance derived from
Prometheus counter deltas (exact per-request isolation - requires the
server to be otherwise idle):

    steps/s    = delta(spec_decode_num_drafts) / delta(request_decode_time_seconds_sum)
    acceptance = 1 + delta(spec_decode_num_accepted_tokens) / delta(num_drafts)

Without speculative decoding the step rate falls back to
delta(generation_tokens) / delta(decode_time) (1 token per step).

Usage: decode_2prompts.py [--port 8123] [--runs 5] [--max-tokens 512]
                          [--output out.jsonl]
"""

import argparse
import hashlib
import json
import statistics
import time
import urllib.request

REFACTOR = """Refactor this function to use pathlib and add type hints:

import os

def find_files(directory, extension):
    results = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(extension):
                results.append(os.path.join(root, f))
    return results

Refactored version:
"""

FOX = "The quick brown fox"

METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:request_decode_time_seconds_sum",
    "vllm:generation_tokens_total",
)


def scrape(port: int) -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",
                                timeout=30) as r:
        text = r.read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for name in METRICS:
            if line.startswith(name):
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def complete(port: int, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", body,
        {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.load(r)
    d["_wall"] = time.perf_counter() - t0
    return d


def probe(port: int, model: str, tag: str, prompt: str, runs: int,
          max_tokens: int, records: list[dict]) -> dict:
    complete(port, model, prompt, max_tokens)  # warmup, dropped
    rows = []
    for run in range(1, runs + 1):
        m0 = scrape(port)
        d = complete(port, model, prompt, max_tokens)
        m1 = scrape(port)
        ct = d["usage"]["completion_tokens"]
        text = d["choices"][0]["text"]
        dd = {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in METRICS}
        drafts = dd["vllm:spec_decode_num_drafts_total"]
        dec_s = dd["vllm:request_decode_time_seconds_sum"]
        steps = drafts if drafts > 0 else dd["vllm:generation_tokens_total"]
        row = {
            "probe": tag, "run": run,
            "tok_s": ct / d["_wall"],
            "completion_tokens": ct,
            "steps_s": steps / dec_s if dec_s > 0 else None,
            "acceptance": 1.0 + dd["vllm:spec_decode_num_accepted_tokens_total"]
            / drafts if drafts > 0 else None,
            "sha": hashlib.sha1(text.encode()).hexdigest()[:10],
        }
        rows.append(row)
        records.append(row)
        acc = f"{row['acceptance']:.2f}" if row["acceptance"] else "-"
        st = f"{row['steps_s']:5.1f}" if row["steps_s"] else "    -"
        print(f"  {tag} run {run}: {row['tok_s']:6.1f} tok/s  "
              f"steps/s {st}  acc {acc}  sha={row['sha']}")
    med = statistics.median(r["tok_s"] for r in rows)
    med_steps = statistics.median(r["steps_s"] for r in rows
                                  if r["steps_s"] is not None)
    accs = [r["acceptance"] for r in rows if r["acceptance"] is not None]
    med_acc = statistics.median(accs) if accs else None
    shas = {r["sha"] for r in rows}
    print(f"  {tag} MEDIAN: {med:.1f} tok/s  steps/s {med_steps:.1f}  "
          f"acc {med_acc:.2f}" if med_acc else
          f"  {tag} MEDIAN: {med:.1f} tok/s  steps/s {med_steps:.1f}",
          end="")
    print(f"  distinct outputs {len(shas)}/{len(rows)}")
    return {"probe": tag, "median_tok_s": med, "median_steps_s": med_steps,
            "median_acceptance": med_acc, "distinct": len(shas)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--output")
    args = ap.parse_args()

    with urllib.request.urlopen(
            f"http://127.0.0.1:{args.port}/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]

    records: list[dict] = []
    summaries = [
        probe(args.port, model, "refactor", REFACTOR, args.runs,
              args.max_tokens, records),
        probe(args.port, model, "fox", FOX, args.runs,
              args.max_tokens, records),
    ]
    print("\nSUMMARY: " + "  |  ".join(
        f"{s['probe']} {s['median_tok_s']:.1f} tok/s, "
        f"{s['median_steps_s']:.1f} steps/s" for s in summaries))
    if args.output:
        with open(args.output, "w") as f:
            for row in records:
                f.write(json.dumps(row) + "\n")
            f.write(json.dumps({"summaries": summaries}) + "\n")


if __name__ == "__main__":
    main()
