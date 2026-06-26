#!/usr/bin/env python3
"""Quality probe for expert-quantization variants served by vllm-probe.

Per variant (server already running on --port):
  1. MTP acceptance: 3 greedy generations (prompt "The quick brown fox",
     max_tokens 512, ignore_eos), tok/s per gen; then the LAST
     "Mean acceptance length:" / "Avg Draft acceptance rate:" lines from
     `docker logs vllm-probe` (captured immediately, before other traffic).
  2. Greedy drift: 12 fixed prompts, temperature 0, max_tokens 128 via
     /v1/completions; exact-match vs a reference run (--ref).
  3. Arithmetic sanity: 5 chat questions (temperature 0), score parsed
     final numbers.

Writes results.json (+ greedy texts) into --out.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

GREEDY_PROMPTS = [
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

MTP_PROMPT = "The quick brown fox"


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(base, timeout_s=720):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    print(f"[health] up after {time.time() - t0:.0f}s")
                    return True
        except Exception:
            pass
        time.sleep(10)
    return False


def mtp_probe(base, model, container):
    gens = []
    for i in range(3):
        t0 = time.time()
        r = post(base + "/v1/completions", {
            "model": model, "prompt": MTP_PROMPT, "max_tokens": 512,
            "temperature": 0, "ignore_eos": True})
        dt = time.time() - t0
        ct = r["usage"]["completion_tokens"]
        gens.append({"completion_tokens": ct, "wall_s": round(dt, 2),
                     "tok_s": round(ct / dt, 1)})
        print(f"[mtp] gen{i}: {ct} tok in {dt:.2f}s = {ct/dt:.1f} tok/s")
    # capture spec-decode counters immediately, before any other traffic
    logs = subprocess.run(["docker", "logs", container],
                          capture_output=True, text=True).stdout
    acc_len = re.findall(r"Mean acceptance length:\s*([0-9.]+)", logs)
    acc_rate = re.findall(r"Avg Draft acceptance rate:\s*([0-9.]+)", logs)
    out = {
        "gens": gens,
        "mean_tok_s": round(sum(g["tok_s"] for g in gens) / len(gens), 1),
        "mean_acceptance_length": float(acc_len[-1]) if acc_len else None,
        "draft_acceptance_rate_pct": float(acc_rate[-1]) if acc_rate else None,
    }
    print(f"[mtp] acceptance_length={out['mean_acceptance_length']} "
          f"draft_acceptance={out['draft_acceptance_rate_pct']}% "
          f"mean tok/s={out['mean_tok_s']}")
    return out


def greedy_drift(base, model):
    outs = []
    for i, p in enumerate(GREEDY_PROMPTS):
        r = post(base + "/v1/completions", {
            "model": model, "prompt": p, "max_tokens": 128, "temperature": 0})
        txt = r["choices"][0]["text"]
        outs.append(txt)
        print(f"[greedy] {i}: {len(txt)} chars")
    return outs


def arithmetic(base, model):
    rows = []
    for q, gold in ARITH:
        r = post(base + "/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": q}],
            "max_tokens": 4096, "temperature": 0})
        msg = r["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        nums = re.findall(r"-?[\d,]*\d", content.replace(",", ""))
        got = int(nums[-1]) if nums else None
        ok = got == gold
        rows.append({"q": q, "gold": gold, "got": got, "ok": ok,
                     "content": content[:200],
                     "reasoning_len": len(msg.get("reasoning_content") or "")})
        print(f"[arith] gold={gold} got={got} {'OK' if ok else 'WRONG'}")
    score = sum(r["ok"] for r in rows)
    print(f"[arith] score {score}/{len(rows)}")
    return {"rows": rows, "score": score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None,
                    help="reference results dir (baseline) for exact-match")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--container", default="vllm-probe")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    os.makedirs(args.out, exist_ok=True)

    if not wait_health(base, 720):
        json.dump({"variant": args.variant, "status": "FAILED_TO_SERVE"},
                  open(os.path.join(args.out, "results.json"), "w"), indent=1)
        print("[fatal] server never became healthy")
        return 2

    res = {"variant": args.variant, "status": "ok", "ts": time.strftime("%F %T")}
    res["mtp"] = mtp_probe(base, args.model, args.container)
    greedy = greedy_drift(base, args.model)
    json.dump(greedy, open(os.path.join(args.out, "greedy.json"), "w"), indent=1)
    res["arith"] = arithmetic(base, args.model)

    if args.ref:
        ref = json.load(open(os.path.join(args.ref, "greedy.json")))
        matches = [a == b for a, b in zip(greedy, ref)]
        res["greedy_exact_match"] = {
            "matches": matches,
            "rate": f"{sum(matches)}/{len(matches)}",
        }
        print(f"[drift] exact-match vs ref: {sum(matches)}/{len(matches)} "
              f"(per-prompt: {''.join('Y' if m else 'n' for m in matches)})")

    json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    print(f"[done] wrote {args.out}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
