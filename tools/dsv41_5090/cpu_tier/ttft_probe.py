"""TTFT / prefill probe of the hybrid serve (phase 2.5): prompts of given token lengths built from local source text
(trimmed to the length with the server's /tokenize), streamed completions, the time to the first token and the decode
rate behind the long context.

    python ttft_probe.py --port 8041 --model deepseek-ai/DeepSeek-V4.1-Flash --lengths 2048,14336,30720 --gen 64 --repeat 2 --out ttft.json

The first request of a length pays b12x's plans / compiles for the chunk shapes it meets (the tail chunk of a chunked
prefill is a new token count); --repeat 2 shows the steady state. Prefix caching must be off on the server (it is in
the hybrid launcher's identity settings) or the repeats would skip the prefill.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.request


def post(port: int, path: str, body: dict, timeout: float = 1800.0):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def count_tokens(port: int, model: str, text: str) -> int:
    with post(port, "/tokenize", dict(model=model, prompt=text)) as r:
        return int(json.load(r)["count"])


def corpus(source: str, min_chars: int) -> str:
    """Natural text: source files under `source` (default: this Python's stdlib), concatenated in a fixed order."""
    paths = sorted(glob.glob(os.path.join(source, "**", "*.py"), recursive=True))
    parts, n = [], 0
    for p in paths:
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if len(t) < 2000:
            continue
        parts.append(f"# file: {os.path.basename(p)}\n{t}\n")
        n += len(parts[-1])
        if n >= min_chars:
            break
    if n < min_chars:
        raise SystemExit(f"corpus under {source}: {n} chars < {min_chars} needed")
    return "".join(parts)


def prompt_of_length(port: int, model: str, text: str, target: int) -> tuple[str, int]:
    """Trim `text` (chars) so that the server counts ~target tokens (within 0.5 %); a few /tokenize rounds."""
    lo, hi = 0, len(text)
    cut = min(len(text), target * 4)
    best = None
    for _ in range(12):
        n = count_tokens(port, model, text[:cut])
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (cut, n)
        if abs(n - target) <= max(4, target // 200):
            break
        if n < target:
            lo = cut
        else:
            hi = cut
        # secant step on chars-per-token, kept inside (lo, hi)
        cpt = cut / max(1, n)
        cut = int(target * cpt)
        if not (lo < cut < hi):
            cut = (lo + hi) // 2
        if hi - lo < 8:
            break
    return text[: best[0]], best[1]


def stream_once(port: int, model: str, prompt: str, gen: int) -> dict:
    body = dict(model=model, prompt=prompt, max_tokens=gen, temperature=0, seed=0, ignore_eos=True, stream=True,
                stream_options={"include_usage": True})
    t0 = time.perf_counter()
    t_first = t_last = None
    n_chunks = 0
    usage = None
    text = []
    with post(port, "/v1/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("text") or ch[0].get("finish_reason")):
                now = time.perf_counter()
                if t_first is None and ch[0].get("text"):
                    t_first = now
                t_last = now
                n_chunks += 1
                text.append(ch[0].get("text") or "")
    t_end = time.perf_counter()
    ttft = (t_first or t_end) - t0
    completion = (usage or {}).get("completion_tokens", n_chunks)
    decode_s = (t_last - t_first) if (t_first is not None and t_last is not None and t_last > t_first) else None
    return dict(ttft_s=ttft, total_s=t_end - t0, prompt_tokens=(usage or {}).get("prompt_tokens"),
                completion_tokens=completion, decode_tok_s=((completion - 1) / decode_s) if decode_s else None,
                text=("".join(text))[:80])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8041)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--lengths", default="2048,14336,30720")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--source", default=os.path.dirname(os.__file__))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x]
    text = corpus(args.source, max(lengths) * 6)
    results = []
    for L in lengths:
        prompt, n = prompt_of_length(args.port, args.model, text, L)
        for rep in range(args.repeat):
            r = stream_once(args.port, args.model, prompt, args.gen)
            r.update(target=L, rep=rep, tokenized=n)
            results.append(r)
            pt = r["prompt_tokens"] or n
            dec = f"{r['decode_tok_s']:.1f} tok/s" if r["decode_tok_s"] else "n/a"
            print(f"  prompt {pt} tokens (rep {rep}): TTFT {r['ttft_s']:.2f} s = {pt / r['ttft_s']:.0f} tok/s prefill; "
                  f"decode {dec} over {r['completion_tokens']} tokens; total {r['total_s']:.1f} s; text {r['text'][:40]!r}", flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
