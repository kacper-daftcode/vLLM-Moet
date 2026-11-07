#!/usr/bin/env python3
"""Compare the decode streams of mixed_steps.sh token by token: for every stream, alone vs together vs under
(the same prompt + nonce), and optionally against another run's files (another image / arm). Reports the completion
length, the first differing token position, how many of the common-prefix tokens have a different top-1 logprob, and
the largest |delta logprob| over the identical prefix - so "identical" means the same tokens with the same logprobs.
usage: mixed_compare.py OUT_PREFIX [--ref OTHER_OUT_PREFIX] [--streams 3]"""
import argparse
import glob
import json
import os


def load(path):
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[0] if rows else None


def cmp(x, y):
    if x is None or y is None:
        return "missing"
    tx, ty = x["tokens"], y["tokens"]
    if not tx and not ty:  # no logprobs returned: fall back to the texts
        if x["text"] == y["text"]:
            return f"IDENTICAL text ({x['completion_tokens']} tokens; no per-token logprobs in the stream)"
        n = next((i for i, (a, b) in enumerate(zip(x["text"], y["text"])) if a != b), min(len(x["text"]), len(y["text"])))
        return f"DIFFER text at char {n} of {len(x['text'])}/{len(y['text'])} (no per-token logprobs in the stream)"
    n = min(len(tx), len(ty))
    first_tok = next((i for i in range(n) if tx[i][0] != ty[i][0]), None)
    same_len = len(tx) == len(ty)
    pre = n if first_tok is None else first_tok
    lp_diff = sum(1 for i in range(pre) if tx[i][1] != ty[i][1])
    max_d = max((abs((tx[i][1] or 0.0) - (ty[i][1] or 0.0)) for i in range(pre)), default=0.0)
    if first_tok is None and same_len and lp_diff == 0:
        return f"IDENTICAL ({len(tx)} tokens, logprobs identical)"
    if first_tok is None and same_len:
        return f"same tokens ({len(tx)}), {lp_diff} logprobs differ (max |d| {max_d:.3g})"
    return (f"DIFFER at token {first_tok} of {len(tx)}/{len(ty)} ('{tx[first_tok][0] if first_tok is not None and first_tok < len(tx) else ''}' vs "
            f"'{ty[first_tok][0] if first_tok is not None and first_tok < len(ty) else ''}'); before it {lp_diff} logprobs differ (max |d| {max_d:.3g})")


ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--ref", default=None, help="another run's out-prefix (e.g. the r25 arm): alone vs alone, under vs under")
ap.add_argument("--streams", type=int, default=3)
a = ap.parse_args()
for i in range(1, a.streams + 1):
    alone = load(f"{a.prefix}.alone.s{i}.jsonl")
    together = load(f"{a.prefix}.together.s{i}.jsonl")
    under = load(f"{a.prefix}.under.s{i}.jsonl")
    def spd(r):
        return "-" if r is None else f"{r['completion_tokens']} tok, {r['tok_s_out']} tok/s, steps/s {r['steps_s']}, acc {r['acceptance']}, ttft {r['ttft_ms']} ms, prompt {r['prompt_tokens']}"
    print(f"stream {i}: alone [{spd(alone)}] | together [{spd(together)}] | under [{spd(under)}]")
    print(f"   together vs alone: {cmp(together, alone)}")
    print(f"   under    vs alone: {cmp(under, alone)}")
    print(f"   under    vs together: {cmp(under, together)}")
    if a.ref:
        for kind in ("alone", "together", "under"):
            print(f"   {kind:8s} vs ref {kind}: {cmp(load(f'{a.prefix}.{kind}.s{i}.jsonl'), load(f'{a.ref}.{kind}.s{i}.jsonl'))}")
