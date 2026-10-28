"""Client of the mini-model identity test (phase 2.3): greedy completions with logprobs for short prompts (<= 8
tokens, so that prefill takes b12x's compact path too), saved as JSON; two files compared token by token and
logprob by logprob (the hybrid must equal the all-resident run bit for bit -> identical logprobs).

    python compare_mini_outputs.py --port 8031 --out run_r8.json          # query a server
    python compare_mini_outputs.py --compare run_r16.json run_r8.json      # compare two runs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

PROMPTS = ["The capital of France is", "1, 2, 3, 4,", "def fib(n):", "Once upon a time", "import numpy as np",
           "Hello", "The quick brown fox", "x = [1, 2, 3]", "Paris is", "Water boils at", "A list:", "The end"]
# prefill-sized prompts (> 8 tokens: b12x's non-compact path, phase 2.5)
PROMPTS_LONG = [
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n",
    "The Industrial Revolution, which took place from the 18th to 19th centuries, was a period during which predominantly agrarian, rural societies in Europe and America became industrial and urban. Prior to the Industrial Revolution,",
    "import torch\nimport torch.nn as nn\n\nclass MLP(nn.Module):\n    def __init__(self, d_in, d_hidden, d_out):\n        super().__init__()\n        self.fc1 = nn.Linear(d_in, d_hidden)\n",
    "Q: A train travels 60 miles per hour for 2.5 hours. How far does it travel?\nA: Let's think step by step.",
]


def post(port: int, path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_ready(port: int, timeout: float = 1800.0, container: str = "") -> None:
    import subprocess
    t0 = time.time()
    gone, seen = 0, False
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        if container:
            up = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{container}$"], capture_output=True, text=True).stdout.strip()
            seen = seen or bool(up)
            gone = gone + 1 if (seen and not up) else 0
            if gone >= 6:   # 30 s without the container after it had been up: it died
                raise SystemExit(f"server not ready: container {container} is gone")
        time.sleep(5)
    raise SystemExit("server not ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--model", default="mini")
    ap.add_argument("--prompts", type=int, default=0, help="only the first N prompts (0 = all)")
    ap.add_argument("--container", default="", help="give up early when this docker container is gone")
    ap.add_argument("--long", action="store_true", help="the prefill-sized prompts (PROMPTS_LONG) instead of the short ones")
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        a, b = (json.load(open(p)) for p in args.compare)
        bad = 0
        compact_bad, compact_n = 0, 0
        for pa, pb in zip(a["results"], b["results"]):
            same_tok = pa["tokens"] == pb["tokens"]
            same_lp = pa["logprobs"] == pb["logprobs"]
            n = min(len(pa["tokens"]), len(pb["tokens"]))
            first = next((i for i in range(n) if pa["tokens"][i] != pb["tokens"][i] or pa["logprobs"][i] != pb["logprobs"][i]), None)
            print(f"  {pa['prompt']!r:28s} ({pa.get('prompt_tokens', '?')} prompt tokens): tokens {'same' if same_tok else 'DIFFER'} ({len(pa['tokens'])} vs {len(pb['tokens'])}), "
                  f"logprobs {'same' if same_lp else 'DIFFER'}" + (f" (first difference at {first})" if first is not None else ""))
            bad += not (same_tok and same_lp)
            # the identity rule (R5) is about the compact path: prompts of <= 8 tokens prefill on it (padded to the capture
            # sizes); the 11-token prompts take b12x's non-compact kernels, which are nondeterministic at the ulp level
            # (two runs of ONE server differ on them: 10-25 dc2, 10-28 se) - reported, but not part of the verdict
            if int(pa.get("prompt_tokens", 99)) <= 8:
                compact_n += 1
                compact_bad += not (same_tok and same_lp)
        print(f"{bad} of {len(a['results'])} prompts differ (all paths); compact-path prompts (<= 8 tokens): "
              f"{compact_bad} of {compact_n} differ -> {'PASS' if compact_bad == 0 else 'FAIL'}")
        return 0 if compact_bad == 0 else 1
    wait_ready(args.port, container=args.container)
    results = []
    for p in (PROMPTS_LONG if args.long else PROMPTS)[: args.prompts or None]:
        t0 = time.time()
        r = post(args.port, "/v1/completions", dict(model=args.model, prompt=p, max_tokens=args.max_tokens, temperature=0,
                                                     logprobs=1, seed=0))
        ch = r["choices"][0]
        toks = ch["logprobs"]["tokens"]
        lps = ch["logprobs"]["token_logprobs"]
        dt = time.time() - t0
        n_prompt = int(r.get("usage", {}).get("prompt_tokens", -1))
        results.append(dict(prompt=p, prompt_tokens=n_prompt, tokens=toks, logprobs=lps, text=ch["text"], seconds=dt))
        print(f"  {p!r:28s} ({n_prompt} prompt tokens): {len(toks)} tokens in {dt:.1f} s ({len(toks) / max(dt, 1e-6):.1f} tok/s): {ch['text'][:50]!r}", flush=True)
    out = dict(port=args.port, results=results)
    if args.out:
        json.dump(out, open(args.out, "w"))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
