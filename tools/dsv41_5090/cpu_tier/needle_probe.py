"""Needle-in-a-haystack probe of the hybrid serve (phase 3): a code word planted at several depths of a long filler text
(disjoint source slices, as ttft_probe.py), the model asked for it (chat, greedy, thinking off); pass = the word in
the answer. Also reports the TTFT of each long prompt.

    python needle_probe.py --port 8041 --model deepseek-ai/DeepSeek-V4.1-Flash --lengths 32768,65536 --depths 0.1,0.5,0.9
Prefix caching off on the server (`PREFIX_CACHE=0`) or the repeats of a filler would skip the prefill.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttft_probe import corpus, count_tokens, post  # noqa: E402

WORDS = ["lantern", "quartz", "meadow", "copper", "harbor", "velvet", "falcon", "ember", "saffron", "glacier", "orchid", "basalt"]


def chat(port: int, model: str, prompt: str, max_tokens: int = 32) -> tuple[str, float, int]:
    body = dict(model=model, temperature=0, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}],
                chat_template_kwargs={"thinking": False}, stream=True, stream_options={"include_usage": True})
    t0 = time.perf_counter()
    t_first = None
    text, prompt_tokens = [], 0
    with post(port, "/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                prompt_tokens = d["usage"].get("prompt_tokens", 0)
            for ch in d.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    text.append(piece)
    return "".join(text), (t_first or time.perf_counter()) - t0, prompt_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8041)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--lengths", default="32768,65536")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--source", default=os.path.dirname(os.__file__))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x]
    depths = [float(x) for x in args.depths.split(",") if x]
    text = corpus(args.source, sum(lengths) * 5 * len(depths) + 20000)
    rng = random.Random(7)
    results, passed, pos = [], 0, 0
    for L in lengths:
        for depth in depths:
            # a filler of ~L tokens cut from a fresh part of the corpus; the needle sentence goes at `depth` of its chars
            body = text[pos:]
            cut = min(len(body), L * 4)
            n = count_tokens(args.port, args.model, body[:cut])
            cut = int(cut * (L - 1000) / max(1, n))   # the needle, the question and the chat template on top; L must stay under --max-model-len
            filler = body[:cut]
            pos += cut
            word = rng.choice(WORDS) + str(rng.randint(100, 999))
            needle = f"\n\nNote for the reader: the secret code word is '{word}'. Remember it.\n\n"
            at = filler.rfind("\n", 0, int(len(filler) * depth)) + 1
            prompt = (filler[:at] + needle + filler[at:] +
                      "\n\nQuestion: what is the secret code word mentioned in the text above? Answer with the code word only.")
            answer, ttft, ptok = chat(args.port, args.model, prompt)
            ok = word in answer
            passed += ok
            results.append(dict(length=L, depth=depth, prompt_tokens=ptok, ttft_s=ttft, ok=ok, word=word, answer=answer[:60]))
            print(f"  {ptok:6d} tokens, needle at {depth:.1f}: {'PASS' if ok else 'FAIL'} ({word} -> {answer.strip()[:40]!r}); "
                  f"TTFT {ttft:.2f} s = {ptok / max(ttft, 1e-6):.0f} tok/s", flush=True)
    print(f"needle: {passed} of {len(results)} passed")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
