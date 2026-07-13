#!/usr/bin/env python3
"""Decode tok/s + MTP acceptance vs prompt depth, on a REAL corpus.

Phase-0 probe of the MTP-decay task (sesja 2026-07-12): the original
decode_at_depth probe filled the context with random words; a junk
context may confuse the drafter more than a real document, so the
observed acceptance decay (2.9 -> 1.8 @891K) must be re-validated on
real text before it is treated as a production problem.

Differences vs decode_at_depth.py (sesja A, random-word reconstruction):
  * filler = deterministic concatenation of source files from a git
    tree (non-repeating, real code/prose), sliced to EXACT token depths
    with the model's own tokenizer (no live calibration);
  * --random-words switch reproduces the legacy junk filler (same
    20-word vocab + seed 7 as needle_full.py) for a same-boot A/B;
  * per-position acceptance rates from
    spec_decode_num_accepted_tokens_per_pos (distinguishes "first draft
    token degrades" vs "tail degrades" without rebooting at other k);
  * metrics scrape settles (re-reads until counters stop moving) so the
    delta cannot race the engine's async logger;
  * degeneration ratio of the output tail (unique 4-grams / total): a
    looping output inflates acceptance and would poison the comparison.

The server must be OTHERWISE IDLE. Run under the probe lock:
  flock /tmp/moet-dcp1m-probe.lock python3 decode_at_depth_real.py ...

Usage:
  decode_at_depth_real.py --port 8123 \
    --source-tree /root/vllm-moet-src \
    --tokenizer /root/models/GLM-5.2-NVFP4 \
    --depths 8192,200000,500000,700000,891000 \
    --runs 3 --max-tokens 512 --task story \
    --output /root/bench-results/mtpdecay-phase0.jsonl
"""
import argparse
import hashlib
import json
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TEXT_EXT = {
    ".py", ".md", ".rst", ".txt", ".cu", ".cuh", ".h", ".hpp", ".c", ".cpp",
    ".cc", ".cmake", ".yaml", ".yml", ".toml", ".sh", ".cfg", ".ini", ".json",
}
MAX_FILE_BYTES = 2 * 1024 * 1024

TASKS = {
    # Same character as the original 2026-07-12 probe task (story after
    # filler); phrasing pinned here for reproducibility.
    "story": (
        "\n\nNow set the material above aside and write a long, vivid short "
        "story about a lighthouse keeper on a remote northern island who "
        "discovers something unexpected during a winter storm. Do not stop "
        "early; keep narrating.\n\nStory:\n"
    ),
    # decode_ab.py's refactor prompt: a code task, natural after a code corpus.
    "refactor": (
        "\n\nRefactor this function to use pathlib and add type hints:\n\n"
        "import os\n\ndef find_files(directory, extension):\n"
        "    results = []\n    for root, dirs, files in os.walk(directory):\n"
        "        for f in files:\n            if f.endswith(extension):\n"
        "                results.append(os.path.join(root, f))\n"
        "    return results\n\nRefactored version:\n"
    ),
}

# needle_full.py vocabulary, seed 7 - the legacy junk filler.
WORDS = ("alpha quantum river matrix ember glacier syntax violet nimbus cobalt "
         "tangent fjord lantern zephyr cipher marble thunder willow plasma onyx"
         ).split()

COUNTERS = (
    "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
    "request_prefill_time_seconds_sum",
    "request_decode_time_seconds_sum",
    "generation_tokens_total",
    # Contamination guard: exactly one request (ours) may finish per
    # measurement window, else the counter deltas are polluted by foreign
    # traffic (e.g. the parallel DCP-perf session).
    "request_success_total",
)


def log(msg):
    print(msg, flush=True)


def http_json(url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    out["_wall"] = time.perf_counter() - t0
    return out


def scrape(port):
    with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        text = r.read().decode()
    out = {}
    pos_pat = re.compile(
        r"^vllm:spec_decode_num_accepted_tokens_per_pos_total"
        r"\{[^}]*position=\"(\d+)\"[^}]*\}\s+([0-9.e+-]+)$")
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = pos_pat.match(line)
        if m:
            key = f"acc_pos{m.group(1)}"
            out[key] = out.get(key, 0.0) + float(m.group(2))
            continue
        for name in COUNTERS:
            if line.startswith(f"vllm:{name}"):
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def scrape_settled(port, probe_keys=("spec_decode_num_drafts_total",
                                     "generation_tokens_total")):
    """Scrape until the async logger stops moving the counters."""
    prev = scrape(port)
    for _ in range(20):
        time.sleep(0.5)
        cur = scrape(port)
        if all(cur.get(k, 0.0) == prev.get(k, 0.0) for k in probe_keys):
            return cur
        prev = cur
    return prev


def build_corpus_ids(tok, tree, budget):
    """Deterministic token-id stream from a source tree; refuses to loop."""
    tree = Path(tree)
    try:
        files = sorted(subprocess.run(
            ["git", "-C", str(tree), "ls-files"],
            capture_output=True, text=True, check=True).stdout.splitlines())
        rev = subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError:
        files = sorted(str(p.relative_to(tree))
                       for p in tree.rglob("*") if p.is_file())
        rev = "no-git"

    ids, nfiles, seen = [], 0, set()
    for rel in files:
        p = tree / rel
        if p.suffix.lower() not in TEXT_EXT or not p.is_file():
            continue
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        if not raw or len(raw) > MAX_FILE_BYTES:
            continue
        h = hashlib.sha1(raw).digest()
        if h in seen:
            continue
        seen.add(h)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        ids.extend(tok.encode(f"\n\n===== FILE: {rel} =====\n\n{text}",
                              add_special_tokens=False))
        nfiles += 1
        if len(ids) >= budget:
            break
    if len(ids) < budget:
        raise SystemExit(f"corpus too small: {len(ids)} tokens < {budget} "
                         f"(refusing to loop text)")
    return ids, f"source-tree:{tree}@{rev} ({nfiles} files)"


def build_random_ids(tok, budget):
    rng = random.Random(7)
    words = [rng.choice(WORDS) for _ in range(int(budget * 1.2) + 2048)]
    ids = tok.encode(" ".join(words), add_special_tokens=False)
    if len(ids) < budget:
        raise SystemExit(f"random filler too small: {len(ids)} < {budget}")
    return ids, "random-words(needle_full vocab, seed=7)"


def degeneration_ratio(text, tail_words=100):
    """unique-4gram / total-4gram over the output tail; ~1.0 = no looping."""
    ws = text.split()[-tail_words:]
    if len(ws) < 8:
        return 1.0
    grams = [tuple(ws[i:i + 4]) for i in range(len(ws) - 3)]
    return len(set(grams)) / len(grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tokenizer", default="/root/models/GLM-5.2-NVFP4")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-tree")
    src.add_argument("--random-words", action="store_true")
    ap.add_argument("--depths", default="8192,200000,500000,700000,891000")
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--task", choices=sorted(TASKS), default="story")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    model = args.model or http_json(
        f"http://127.0.0.1:{args.port}/v1/models")["data"][0]["id"]

    from transformers import AutoTokenizer  # deferred: slow import
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    task_text = TASKS[args.task]
    task_len = len(tok.encode(task_text, add_special_tokens=False))
    budget = max(depths)  # filler upper bound (slice is per-depth)

    t0 = time.perf_counter()
    if args.source_tree:
        corpus_ids, filler_kind = build_corpus_ids(tok, args.source_tree,
                                                   budget)
    else:
        corpus_ids, filler_kind = build_random_ids(tok, budget)
    log(f"[corpus] {filler_kind}: {len(corpus_ids)} tokens in "
        f"{time.perf_counter() - t0:.1f}s; task={args.task} (+{task_len} tok)")

    out_f = open(args.output, "a") if args.output else None

    def emit(rec):
        if out_f:
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

    url = f"http://127.0.0.1:{args.port}/v1/completions"
    emit({"kind": "meta", "argv": sys.argv[1:], "model": model,
          "filler": filler_kind, "task": args.task,
          "max_tokens": args.max_tokens, "depths": depths,
          "runs": args.runs, "warmups": args.warmups,
          "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    summaries = []
    for depth in depths:
        n_fill = depth - task_len
        prompt = tok.decode(corpus_ids[:n_fill]) + task_text
        body = {"model": model, "prompt": prompt,
                "max_tokens": args.max_tokens, "temperature": 0.0,
                "ignore_eos": True}

        log(f"\n=== depth {depth} (filler {n_fill} tok) ===")
        rows = []
        for i in range(args.warmups + args.runs):
            warm = i < args.warmups
            m0 = scrape_settled(args.port)
            resp = http_json(url, body, timeout=args.timeout)
            m1 = scrape_settled(args.port)

            d = {k: m1.get(k, 0.0) - m0.get(k, 0.0)
                 for k in set(m0) | set(m1)}
            finished = d.get("request_success_total", 0.0)
            if finished != 1.0:
                log(f"  WARNING: {finished:.0f} requests finished in this "
                    f"window (expected 1) - foreign traffic, deltas tainted")
            ct = resp["usage"]["completion_tokens"]
            pt = resp["usage"]["prompt_tokens"]
            text = resp["choices"][0]["text"]
            drafts = d.get("spec_decode_num_drafts_total", 0.0)
            acc = d.get("spec_decode_num_accepted_tokens_total", 0.0)
            dec_s = d.get("request_decode_time_seconds_sum", 0.0)
            pre_s = d.get("request_prefill_time_seconds_sum", 0.0)
            steps = drafts if drafts else d.get("generation_tokens_total", 0.0)
            acc_len = 1.0 + acc / drafts if drafts else None
            steps_s = steps / dec_s if dec_s else None
            dtoks = ct / dec_s if dec_s else None
            pos_rates = {k: v / drafts for k, v in sorted(d.items())
                         if k.startswith("acc_pos") and drafts}
            degen = degeneration_ratio(text)

            tag = "warm" if warm else f"run{i - args.warmups + 1}"
            acc_s = f"{acc_len:.3f}" if acc_len else "-"
            log(f"  {tag}: prompt={pt} gen={ct} wall={resp['_wall']:.1f}s "
                f"prefill={pre_s:.1f}s decode={dec_s:.1f}s | "
                f"{dtoks or 0:.1f} tok/s, acc_len={acc_s}, "
                f"steps/s={steps_s or 0:.1f}, "
                f"pos={[round(v, 3) for v in pos_rates.values()]}, "
                f"degen={degen:.2f} "
                f"sha={hashlib.sha1(text.encode()).hexdigest()[:10]}")

            rec = {
                "kind": "run", "depth": depth, "tag": tag, "warm": warm,
                "tainted": finished != 1.0,
                "prompt_tokens": pt, "completion_tokens": ct,
                "wall_s": round(resp["_wall"], 3),
                "prefill_s": round(pre_s, 3), "decode_s": round(dec_s, 3),
                "decode_tok_s": round(dtoks, 2) if dtoks else None,
                "acceptance_len": round(acc_len, 4) if acc_len else None,
                "steps_per_s": round(steps_s, 2) if steps_s else None,
                "drafts": drafts, "accepted": acc,
                "draft_tokens": d.get("spec_decode_num_draft_tokens_total", 0),
                "pos_rates": {k: round(v, 4) for k, v in pos_rates.items()},
                "degen_ratio": round(degen, 3),
                "output_sha": hashlib.sha1(text.encode()).hexdigest()[:16],
                "output_tail": " ".join(text.split()[-40:]),
            }
            emit(rec)
            if not warm:
                rows.append(rec)

        def med(k):
            vals = [r[k] for r in rows if r[k] is not None]
            return statistics.median(vals) if vals else None

        summ = {
            "kind": "summary", "depth": depth,
            "prompt_tokens": rows[0]["prompt_tokens"],
            "decode_tok_s": med("decode_tok_s"),
            "acceptance_len": med("acceptance_len"),
            "steps_per_s": med("steps_per_s"),
            "degen_ratio": min(r["degen_ratio"] for r in rows),
            "distinct_outputs": len({r["output_sha"] for r in rows}),
        }
        summaries.append(summ)
        emit(summ)

    log("\n=== SUMMARY (medians) ===")
    log(f"{'depth':>8} {'prompt':>8} {'tok/s':>8} {'acc_len':>8} "
        f"{'steps/s':>8} {'degen':>6} {'distinct':>8}")
    for s in summaries:
        acc_s = f"{s['acceptance_len']:.3f}" if s['acceptance_len'] else "-"
        log(f"{s['depth']:>8} {s['prompt_tokens']:>8} "
            f"{s['decode_tok_s'] or 0:>8.1f} {acc_s:>8} "
            f"{s['steps_per_s'] or 0:>8.1f} {s['degen_ratio']:>6.2f} "
            f"{s['distinct_outputs']:>8}")
    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
