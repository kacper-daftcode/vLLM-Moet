#!/usr/bin/env python3
"""Decode rate vs context depth: tok/s + steps/s + acceptance per depth.

Reconstruction of the 2026-07-12 DCP-session probe. Builds a prompt of
deterministic random-word filler (seed 7, same style as needle_full.py)
up to a target token depth, appends a fixed story task, and measures
greedy decode at that depth. Engine-step rate and MTP acceptance come
from Prometheus counter deltas (see decode_2prompts.py; server must be
otherwise idle).

The filler token/word ratio is calibrated with one tiny request, then
each depth reports its *actual* prompt_tokens. The flatness criterion of
the DCP work is judged on steps/s across depths (tok/s decays with the
drafter's acceptance, which is a separate concern).

Usage: decode_at_depth.py [--port 8123] [--depths 8192,200000,500000]
                          [--runs 3] [--max-tokens 256] [--output out.jsonl]
"""

import argparse
import json
import random
import statistics
import time
import urllib.request

WORDS = [
    "amber", "breeze", "cobalt", "dune", "ember", "fjord", "grove",
    "harbor", "inlet", "juniper", "kelp", "lagoon", "meadow", "nectar",
    "orchid", "pebble", "quartz", "ridge", "summit", "tundra",
]

TASK = ("\n\nNow ignore the word list above and write a short story about "
        "a lighthouse keeper who discovers something unusual at low tide. "
        "Story:\n")

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


def complete(port: int, model: str, prompt: str, max_tokens: int,
             timeout: int = 3600) -> dict:
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", body,
        {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    d["_wall"] = time.perf_counter() - t0
    return d


def filler(n_words: int) -> str:
    rng = random.Random(7)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--depths", default="8192,200000,500000",
                    help="comma-separated prompt-token targets")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--output")
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]

    with urllib.request.urlopen(
            f"http://127.0.0.1:{args.port}/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]

    # Calibrate filler tokens/word on a small sample.
    cal_words = 2000
    d = complete(args.port, model, filler(cal_words) + TASK, 1)
    base_tokens = d["usage"]["prompt_tokens"]
    d0 = complete(args.port, model, filler(cal_words // 2) + TASK, 1)
    tok_per_word = (base_tokens - d0["usage"]["prompt_tokens"]) / (cal_words / 2)
    print(f"calibration: {tok_per_word:.3f} tokens/word")

    records, summary = [], []
    for depth in depths:
        n_words = max(10, int((depth - 60) / tok_per_word))
        prompt = filler(n_words) + TASK
        d = complete(args.port, model, prompt, args.max_tokens)  # warmup
        actual = d["usage"]["prompt_tokens"]
        print(f"depth target {depth}: actual prompt_tokens={actual} "
              f"(prefill+warmup {d['_wall']:.0f}s)")
        rows = []
        for run in range(1, args.runs + 1):
            m0 = scrape(args.port)
            d = complete(args.port, model, prompt, args.max_tokens)
            m1 = scrape(args.port)
            ct = d["usage"]["completion_tokens"]
            dd = {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in METRICS}
            drafts = dd["vllm:spec_decode_num_drafts_total"]
            dec_s = dd["vllm:request_decode_time_seconds_sum"]
            steps = drafts if drafts > 0 else dd["vllm:generation_tokens_total"]
            row = {
                "depth_target": depth, "prompt_tokens": actual, "run": run,
                "tok_s": ct / d["_wall"],
                "decode_tok_s": ct / dec_s if dec_s > 0 else None,
                "steps_s": steps / dec_s if dec_s > 0 else None,
                "acceptance": 1.0
                + dd["vllm:spec_decode_num_accepted_tokens_total"] / drafts
                if drafts > 0 else None,
            }
            rows.append(row)
            records.append(row)
            acc = f"{row['acceptance']:.2f}" if row["acceptance"] else "-"
            print(f"  run {run}: {row['decode_tok_s']:6.1f} tok/s (decode)  "
                  f"steps/s {row['steps_s']:5.1f}  acc {acc}")
        med = {
            "depth_target": depth, "prompt_tokens": actual,
            "median_decode_tok_s": statistics.median(
                r["decode_tok_s"] for r in rows),
            "median_steps_s": statistics.median(r["steps_s"] for r in rows),
        }
        accs = [r["acceptance"] for r in rows if r["acceptance"] is not None]
        med["median_acceptance"] = statistics.median(accs) if accs else None
        summary.append(med)

    print("\nDEPTH SUMMARY")
    base = summary[0]["median_steps_s"]
    for s in summary:
        rel = s["median_steps_s"] / base * 100
        acc = (f"{s['median_acceptance']:.2f}"
               if s["median_acceptance"] else "-")
        print(f"  @{s['prompt_tokens']:>7} tok: "
              f"{s['median_decode_tok_s']:6.1f} tok/s  "
              f"{s['median_steps_s']:5.1f} steps/s ({rel:5.1f}% of first)  "
              f"acc {acc}")
    if args.output:
        with open(args.output, "w") as f:
            for row in records:
                f.write(json.dumps(row) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")


if __name__ == "__main__":
    main()
