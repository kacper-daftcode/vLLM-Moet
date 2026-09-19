#!/usr/bin/env python3
"""Cross-stack comparison of two quality_cmp.py JSON files: scores side by side, greedy-output agreement.
usage: compare_outputs.py A.json B.json [--tokenizer /path/tokenizer.json]
"""
import json
import sys


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def main():
    fa, fb = sys.argv[1], sys.argv[2]
    tok = None
    if "--tokenizer" in sys.argv:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(sys.argv[sys.argv.index("--tokenizer") + 1])
    A, B = json.load(open(fa)), json.load(open(fb))
    la, lb = A.get("label") or fa, B.get("label") or fb
    print(f"{'test':<16} {la:<28} {lb:<28}")
    for name in ("arith_nothink", "arith_think"):
        if name in A and name in B and "score" in A[name]:
            print(f"{name:<16} {A[name]['score']}/{A[name]['of']:<26} {B[name]['score']}/{B[name]['of']}")
    if "coherence" in A and "coherence" in B and "degenerate" in A["coherence"]:
        print(f"{'coherence':<16} {'degenerate ' + str(A['coherence']['degenerate']):<28} degenerate {B['coherence']['degenerate']}")
        same = sum(x == y for x, y in zip(A["coherence"]["texts"], B["coherence"]["texts"]))
        print(f"{'  raw 128-tok':<16} identical texts {same}/{len(A['coherence']['texts'])}")
    for name in ("tools", "json", "vision"):
        if name in A and name in B:
            ka = {k: v for k, v in A[name].items() if k.endswith("ok")}
            kb = {k: v for k, v in B[name].items() if k.endswith("ok")}
            print(f"{name:<16} {str(ka):<28} {kb}")
    if "needle" in A and "needle" in B and "cases" in A["needle"]:
        for ca, cb in zip(A["needle"]["cases"], B["needle"]["cases"]):
            print(f"{'needle':<16} {str(ca.get('prompt_tokens')) + ' @' + str(ca['depth']) + ' ' + ('PASS' if ca['ok'] else 'FAIL') + f' {ca.get(chr(115), 0):.0f}s':<28} "
                  f"{str(cb.get('prompt_tokens')) + ' @' + str(cb['depth']) + ' ' + ('PASS' if cb['ok'] else 'FAIL') + f' {cb.get(chr(115), 0):.0f}s'}")
    if "agreement" in A and "agreement" in B and isinstance(A["agreement"], list):
        ident, divs, toks = 0, [], []
        for x, y in zip(A["agreement"], B["agreement"]):
            if x["text"] == y["text"]:
                ident += 1
            else:
                divs.append(first_div(x["text"], y["text"]))
                if tok:
                    ta, tb = tok.encode(x["text"], add_special_tokens=False).ids, tok.encode(y["text"], add_special_tokens=False).ids
                    toks.append((first_div(ta, tb), len(ta), len(tb)))
        n = len(A["agreement"])
        print(f"{'agreement':<16} identical {ident}/{n}; first divergence (chars) of the rest: {sorted(divs)}")
        if toks:
            print(f"{'':<16} first divergence (tokens, lenA, lenB): {toks}")
        for x, y in zip(A["agreement"], B["agreement"]):
            if x["text"] != y["text"]:
                i = first_div(x["text"], y["text"])
                print(f"  - {x['prompt'][:50]!r}\n      A: {x['text'][max(0, i - 40):i + 60]!r}\n      B: {y['text'][max(0, i - 40):i + 60]!r}")


if __name__ == "__main__":
    main()
