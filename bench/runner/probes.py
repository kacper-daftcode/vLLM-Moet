"""The measurement probes. Same methodology as the historical tools/ scripts
(bench_tok, prefill_probe, needle_probe, probe_quant_quality, decode_ab), but
returning structured dicts instead of prints.

All greedy (temperature 0). Raw samples are kept in the result — aggregates
are derived, never the only record."""

import hashlib
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.request

WORDS = ("alpha quantum river matrix ember glacier syntax violet nimbus cobalt "
         "tangent fjord lantern zephyr cipher marble thunder willow plasma onyx "
         "harbor crimson vector lattice meadow falcon pixel saffron tundra orbit"
         ).split()

NEEDLE_SECRET = "GLACIER-7741-ORYX"

DECODE_PROMPT = '''Refactor this function to use pathlib and add type hints:

import os

def find_files(directory, extension):
    results = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(extension):
                results.append(os.path.join(root, f))
    return results

Refactored version:
'''

COHERENCE_PROMPTS = [
    "Q: What is the capital of Australia?\nA:",
    "Q: A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?\nA:",
    "Write a Python function that returns the n-th Fibonacci number iteratively.\n\n```python\n",
    "Translate to French: 'The weather is beautiful today, let's go for a walk in the park.'\nFrench:",
    "Q: If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Explain step by step.\nA:",
    ("Summarize the following paragraph in one sentence:\n\n"
     "The industrial revolution, which began in Britain in the late eighteenth century, "
     "transformed economies that had been based on agriculture and handicrafts into economies "
     "based on large-scale industry, mechanized manufacturing, and the factory system. New "
     "machines, new power sources, and new ways of organizing work made existing industries "
     "more productive and efficient.\n\nSummary:"),
    "Q: Which planet in our solar system has the most moons?\nA:",
    "Q: What is 847 + 256? Show your work.\nA:",
    "Write a one-line Python list comprehension that squares the even numbers in a list called xs.\n\n```python\n",
    "Explain in two sentences why the sky is blue.\nAnswer:",
    "Q: What is the next number in the sequence 2, 6, 18, 54?\nA:",
    "The three primary colors of light are",
]

ARITH = [
    ("What is 347 * 28? Reply with only the final number.", 9716),
    ("What is 86 * 74? Reply with only the final number.", 6364),
    ("What is 512 * 943? Reply with only the final number.", 482816),
    ("What is 38 + 277 + 4019 + 86? Reply with only the final number.", 4420),
    ("What is 1234 + 5678 + 9101 + 234 + 87? Reply with only the final number.", 16334),
]


def _post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _completion(base, model, prompt, max_tokens, timeout=900, ignore_eos=True):
    t0 = time.perf_counter()
    d = _post(base + "/v1/completions", {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": ignore_eos}, timeout)
    dt = time.perf_counter() - t0
    return d, dt


def warmup(base, model, n, log=print):
    for i in range(n):
        random.seed(0xBEEF + i)
        p = " ".join(random.choice(WORDS) for _ in range(24))
        _completion(base, model, p, 128)
        log(f"[warmup] {i + 1}/{n}")


def decode(base, model, runs=5, max_tokens=512, log=print, **_):
    _completion(base, model, DECODE_PROMPT, max_tokens)   # warmup, dropped
    samples, texts = [], []
    for i in range(runs):
        d, dt = _completion(base, model, DECODE_PROMPT, max_tokens)
        ct = d["usage"]["completion_tokens"]
        text = d["choices"][0]["text"]
        samples.append(round(ct / dt, 1))
        texts.append(hashlib.sha1(text.encode()).hexdigest()[:10])
        log(f"[decode] run {i + 1}/{runs}: {samples[-1]} tok/s")
    s = sorted(samples)
    return {
        "samples_tok_s": samples,
        "median_tok_s": round(statistics.median(s), 1),
        "min_tok_s": s[0], "max_tok_s": s[-1],
        "spread_pct": round((s[-1] - s[0]) / statistics.median(s) * 100, 1),
        "distinct_outputs": f"{len(set(texts))}/{len(texts)}",
        "max_tokens": max_tokens,
    }


def batch_decode(base, model, concurrency=(1, 4, 8), max_tokens=384, runs=3,
                 log=print, **_):
    def stream_prompt(i):
        random.seed(0xD00D ^ i)
        return " ".join(random.choice(WORDS) for _ in range(32))

    levels = {}
    for n in concurrency:
        aggs = []
        for r in range(runs):
            done = [None] * n
            def worker(i):
                d, _dt = _completion(base, model, stream_prompt(i + r * 100),
                                     max_tokens, timeout=1800)
                done[i] = d["usage"]["completion_tokens"]
            t0 = time.perf_counter()
            ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            wall = time.perf_counter() - t0
            aggs.append(round(sum(done) / wall, 1))
        med = statistics.median(sorted(aggs))
        levels[str(n)] = {
            "aggregate_tok_s": round(med, 1),
            "per_stream_tok_s": round(med / n, 1),
            "samples": aggs,
        }
        log(f"[batch] {n} streams: {med:.1f} tok/s aggregate")
    return {"levels": levels, "max_tokens": max_tokens}


def prefill(base, model, words=8000, runs=3, log=print, **_):
    samples = []
    for i in range(runs):
        random.seed(0xC0FFEE ^ (words + i))
        p = " ".join(random.choice(WORDS) for _ in range(words))
        d, dt = _completion(base, model, p, 1, timeout=3600, ignore_eos=False)
        pt = d["usage"]["prompt_tokens"]
        samples.append({"prompt_tokens": pt, "wall_s": round(dt, 2),
                        "tok_s": round(pt / dt, 0)})
        log(f"[prefill] run {i + 1}/{runs}: {pt} tok in {dt:.1f}s "
            f"= {pt / dt:,.0f} tok/s")
    med = statistics.median(sorted(s["tok_s"] for s in samples))
    return {"words": words, "samples": samples, "median_tok_s": med}


def needle(base, model, sizes_words=None, sizes_frac=None, context=None,
           depth=0.5, max_tokens=512, log=print, **_):
    if not sizes_words:
        # conservative words-per-token guess; the result records the real count
        sizes_words = [max(1000, int(f * (context or 0) / 1.35 / 1000) * 1000)
                       for f in (sizes_frac or [0.03, 0.5])]
    cases = []
    for nwords in sizes_words:
        random.seed(7)
        filler = [random.choice(WORDS) for _ in range(nwords)]
        at = int(len(filler) * depth)
        needle_txt = (f"IMPORTANT FACT: the vault passphrase is "
                      f"{NEEDLE_SECRET}. Remember it exactly.")
        ctx = (" ".join(filler[:at]) + "\n\n" + needle_txt + "\n\n"
               + " ".join(filler[at:]))
        user = (ctx + "\n\nQuestion: What is the vault passphrase? "
                "Reply with ONLY the passphrase, nothing else.")
        t0 = time.perf_counter()
        try:
            r = _post(base + "/v1/chat/completions", {
                "model": model,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": max_tokens, "temperature": 0,
                "chat_template_kwargs": {"thinking": False},
            }, timeout=3600)
        except Exception as e:  # noqa: BLE001 — a FAIL case, not a crash
            cases.append({"words": nwords, "error": f"{type(e).__name__}: {e}",
                          "pass": False})
            log(f"[needle] {nwords} words: ERROR {e}")
            continue
        dt = time.perf_counter() - t0
        m = r["choices"][0]["message"]
        msg = ((m.get("content") or "") + " "
               + (m.get("reasoning") or m.get("reasoning_content") or ""))
        ok = NEEDLE_SECRET in msg
        pt = r.get("usage", {}).get("prompt_tokens", 0)
        cases.append({"words": nwords, "prompt_tokens": pt, "depth": depth,
                      "ttft_gen_s": round(dt, 1), "pass": ok})
        log(f"[needle] {pt} tok @ depth {depth}: "
            f"{'PASS' if ok else 'FAIL'} ({dt:.0f}s)")
    return {"cases": cases, "all_pass": all(c["pass"] for c in cases)}


def arithmetic(base, model, log=print, **_):
    rows = []
    for q, gold in ARITH:
        r = _post(base + "/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": q}],
            "max_tokens": 4096, "temperature": 0}, timeout=900)
        content = (r["choices"][0]["message"].get("content") or "").strip()
        nums = re.findall(r"-?[\d,]*\d", content.replace(",", ""))
        got = int(nums[-1]) if nums else None
        rows.append({"gold": gold, "got": got, "ok": got == gold})
    score = sum(r["ok"] for r in rows)
    log(f"[arith] {score}/{len(rows)}")
    return {"score": score, "of": len(rows), "rows": rows}


def _degenerate(text):
    """Cheap loop detector: a short chunk repeating many times back-to-back."""
    if len(text) < 64:
        return False
    for w in (4, 8, 16):
        chunk = text[-w:]
        if chunk.strip() and text.endswith(chunk * min(8, len(text) // w)):
            return True
    return False


def coherence(base, model, log=print, **_):
    texts, degen = [], 0
    for i, p in enumerate(COHERENCE_PROMPTS):
        d, _dt = _completion(base, model, p, 128, ignore_eos=False)
        t = d["choices"][0]["text"]
        texts.append(t)
        if _degenerate(t):
            degen += 1
            log(f"[coherence] prompt {i}: DEGENERATE tail")
    log(f"[coherence] degenerate {degen}/{len(texts)}")
    return {"degenerate": degen, "of": len(texts), "pass": degen == 0,
            "texts": texts}


def quality(base, model, log=print, *, profile, runs=200, concurrency=2,
            max_tokens=6000, baseline=None, request_overrides=None,
            tool=None, artifacts_dir=None, artifact_tag=None, **_):
    """Dataset-eval probe (GSM8K / GPQA-diamond / …) via llm-inference-bench.

    Runs the pinned external tool against the recipe's server and keeps BOTH
    representations: the raw tool JSON (append-only artifact next to the
    result — flips and per-item data stay reviewable) and compact aggregates
    in the result itself. `baseline` names an entry in bench/baselines/
    (a NATIVE reference measured by the same tool): the tool then computes
    the paired comparison (accuracy flips, completion-token inflation) that
    is THE quality KPI of this project — parity with native, not absolute
    scores. `request_overrides` is merged into every request payload (e.g.
    {"chat_template_kwargs": {"thinking": true}, "temperature": 1.0} for
    think-mode evals)."""
    import json as _json
    import subprocess
    import sys as _sys

    tool = tool or os.environ.get(
        "LLM_BENCH", "/root/workspace/llm-inference-bench/llm_decode_bench.py")
    if not os.path.exists(tool):
        raise RuntimeError(
            f"quality probe needs llm-inference-bench (looked at {tool}; "
            "set `quality_tool` in the box yaml or LLM_BENCH)")
    port = base.rsplit(":", 1)[1].split("/", 1)[0]
    tag = artifact_tag or profile
    out_json = os.path.join(artifacts_dir or "/tmp",
                            f"quality__{tag}.json")
    cmd = [_sys.executable, tool, "--port", port, "--model", model,
           "--test-profile", profile,
           "--profile-runs", str(runs),
           "--profile-concurrency", str(concurrency),
           "--max-tokens", str(max_tokens),
           "--display-mode", "plain", "--no-hw-monitor",
           "--output", out_json]
    from common import baseline_path  # late import: probes stays standalone
    if baseline:
        cmd += ["--compare-baseline", baseline_path(baseline)]
    if request_overrides:
        cmd += ["--request-overrides-json", _json.dumps(request_overrides)]
    log(f"[quality] {profile} runs={runs} c={concurrency}"
        + (f" baseline={baseline}" if baseline else "")
        + (" +overrides" if request_overrides else ""))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=8 * 3600)
    if r.returncode != 0 or not os.path.exists(out_json):
        raise RuntimeError(
            f"tool exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    with open(out_json) as f:
        data = _json.load(f)
    acc = (data.get("accuracy") or {})
    toks = [x.get("completion_tokens") or 0 for x in data.get("runs", [])
            if x.get("phase") == "profile"]
    toks.sort()
    res = {
        "profile": profile, "runs": runs, "concurrency": concurrency,
        "max_tokens": max_tokens,
        "request_overrides": request_overrides or {},
        "accuracy_pct": round(100.0 * acc.get("accuracy", 0.0), 2),
        "correct": acc.get("correct"), "of": acc.get("scored"),
        "truncated_no_answer": acc.get("truncated_no_answer"),
        "hit_max_tokens": acc.get("hit_max_tokens"),
        "tokens_avg": round(sum(toks) / len(toks), 1) if toks else None,
        "tokens_p50": toks[len(toks) // 2] if toks else None,
        "tokens_p90": toks[int(0.9 * len(toks))] if toks else None,
        "tool_sha": _tool_sha(tool),
        "duration_s": round(time.time() - t0, 1),
        "artifact": os.path.basename(out_json),
    }
    if baseline:
        res["baseline"] = baseline
        cmp_ = data.get("comparison") or {}
        ct = cmp_.get("completion_tokens") or {}
        if cmp_:
            bm, cm = ct.get("baseline_mean"), ct.get("candidate_mean")
            res["vs_baseline"] = {
                "acc_delta_pp": round(cmp_["delta_pp"], 2)
                if cmp_.get("delta_pp") is not None else None,
                "flips_only_baseline": cmp_.get("flips_baseline_only_correct"),
                "flips_only_candidate": cmp_.get("flips_candidate_only_correct"),
                "mcnemar_p": cmp_.get("mcnemar_exact_p"),
                "token_inflation_pct": round((cm - bm) / bm * 100, 1)
                if bm and cm else None,
            }
    log(f"[quality] {profile}: {res['accuracy_pct']}% "
        f"({res['correct']}/{res['of']}), tokens avg {res['tokens_avg']}"
        + (f", vs {baseline}: {res.get('vs_baseline')}" if baseline else ""))
    return res


def _tool_sha(tool):
    try:
        import subprocess
        d = os.path.dirname(os.path.abspath(tool))
        sha = subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", d, "status", "--short"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "") if sha else None
    except Exception:  # noqa: BLE001
        return None


PROBES = {
    "decode": decode,
    "batch_decode": batch_decode,
    "prefill": prefill,
    "needle": needle,
    "arithmetic": arithmetic,
    "coherence": coherence,
    "quality": quality,
}


# --- server-log scraping ------------------------------------------------------

def scrape_server_log(log_path):
    """Spec-decode acceptance + cache-tier hit-rate lines from the serve log."""
    out = {"spec_acceptance_length": None, "draft_acceptance_pct": None,
           "pool_lines": [], "planes_cache_lines": []}
    try:
        with open(log_path, errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return out
    acc_len = acc_rate = None
    for ln in lines:
        if "Mean acceptance length:" in ln:
            m = re.search(r"Mean acceptance length:\s*([0-9.]+)", ln)
            acc_len = float(m.group(1)) if m else acc_len
        if "Avg Draft acceptance rate:" in ln:
            m = re.search(r"Avg Draft acceptance rate:\s*([0-9.]+)", ln)
            acc_rate = float(m.group(1)) if m else acc_rate
        if "SpecDecoding metrics" in ln:
            m = re.search(r"Mean acceptance length:\s*([0-9.]+)", ln)
            if m:
                acc_len = float(m.group(1))
        if "hit-rate" in ln:
            out["pool_lines"].append(ln.strip()[-300:])
        if "planes cache" in ln or "planes from cache" in ln:
            out["planes_cache_lines"].append(ln.strip()[-300:])
    out["spec_acceptance_length"] = acc_len
    out["draft_acceptance_pct"] = acc_rate
    out["pool_lines"] = out["pool_lines"][-8:]
    out["planes_cache_lines"] = out["planes_cache_lines"][-4:]
    return out
