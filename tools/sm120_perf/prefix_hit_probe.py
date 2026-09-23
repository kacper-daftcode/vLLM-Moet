#!/usr/bin/env python3
"""Prefix-cache hit fidelity: does a prompt served from the prefix cache continue the way a fresh prefill does?

DeepSeek-V4.1 on vLLM main keeps the sliding-window KV out of prefix caching (CacheConfig.swa_bounded_replay, on by
default with model runner V2): a hit recomputes the hit's last window (128 tokens) with the SWA window clamped to it,
so everything after a hit attends to an approximate window. The same holds for hits served by the OffloadingConnector.
This probe puts a number on it next to the stack's own numerical floor.

Per case (a context of ~N tokens built from a corpus, a note with a 6-digit id right before the question; greedy,
thinking off; everything through /v1/completions on the chat-rendered token ids, one request at a time):
  fresh          salt S1                     full prefill: the reference
  hit            salt S1 again               prefix-cache hit (+ the replay when bounded replay is on)
  fresh2         salt S5                     the same full prefill again: is the stack deterministic at all?
  shifted        salt S2, queued behind a filler of 3*CHUNK+D tokens (sent first, max_tokens 1): a full prefill whose
                 chunk boundaries move by D tokens and whose last chunk has a different length -- the floor a
                 concurrent request already puts on the same prompt (and what an exact prefix hit would see)
  turn2-fresh    P2 = P + fresh answer + a follow-up that first asks for the note's id, salt S3
  turn2-hit      P2 with salt S1 (hits P's blocks and the generated ones)
  turn2-shifted  P2 with salt S4 behind a filler
Per variant against its reference (fresh, turn2-fresh): identical generated ids, first diverging token, the reference's
top-1/top-2 logprob gap at that token (a near-tie diverges under any noise), max / mean |delta logprob| of the reference
tokens before the divergence, the first token's top-5 total variation distance (both sides see exactly the prompt),
the note's id answered (turn 2), prefix-cache hit tokens and wall time. The logits are bf16 (top-1/top-2 gaps come in
multiples of 1/8 at the usual logit magnitudes), so exact ties are common and greedy text diverges at the first
near-tie under any perturbation: compare the variants' statistics with each other, not with zero.

usage: prefix_hit_probe.py --base http://127.0.0.1:8001 --corpus code=DIR:GLOB --corpus docs=DIR:GLOB
                           [--lengths 6000,24000,96000] [--gen 128] [--chunk 4096] [--shift 1000] [--out FILE]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import threading
import time
import urllib.request
import uuid

QUESTIONS = {
    "code": ("Explain what the last file above does, then point out one thing in it that could be simplified.",
             "What is the build identifier from the note in my first message? Give the digits first, then explain in "
             "two sentences how the last file relates to the first one."),
    "docs": ("Summarize the last document above in five bullet points, then name one question it leaves open.",
             "What is the build identifier from the note in my first message? Give the digits first, then explain in "
             "two sentences how the last document relates to the first one."),
}
FILLER = "[{tag}] Line {j}: the scheduler assigns blocks to requests and reclaims them when sequences finish.\n"


def post(base: str, path: str, body: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def scrape(url: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not url:
        return out
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                name = line.split("{", 1)[0].split(" ", 1)[0]
                if ("prefix_cache" in name or "kv_offload" in name) and "bucket" not in name:
                    try:
                        out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
                    except ValueError:
                        pass
    except Exception:  # noqa: BLE001
        pass
    return out


def delta(a: dict, b: dict) -> dict:
    return {k.split(":")[-1]: round(b[k] - a.get(k, 0.0)) for k in b if b[k] != a.get(k, 0.0)}


def corpus_files(spec: str) -> list[str]:
    root, _, pattern = spec.partition(":")
    files = sorted(f for f in glob.glob(os.path.join(root, pattern or "**/*"), recursive=True) if os.path.isfile(f))
    return [f for f in files if 200 < os.path.getsize(f) < 200_000]


def build_context(files: list[str], root: str, n_chars: int, seed: str) -> str:
    order = list(range(len(files)))
    random.Random(seed).shuffle(order)
    parts, total = [], 0
    for i in order:
        try:
            text = open(files[i], encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        part = f"### {os.path.relpath(files[i], root)}\n```\n{text}\n```\n\n"
        parts.append(part)
        total += len(part)
        if total >= n_chars:
            break
    ctx = "".join(parts)
    return ctx if len(ctx) <= n_chars else ctx[:n_chars] + "\n```\n\n"


def render(base: str, model: str, messages: list[dict]) -> list[int]:
    d = post(base, "/tokenize", {"model": model, "messages": messages, "add_generation_prompt": True,
                                 "chat_template_kwargs": {"thinking": False}})
    return d["tokens"]


def complete(base: str, model: str, ids: list[int], gen: int, salt: str) -> tuple[dict, float]:
    body = {"model": model, "prompt": ids, "max_tokens": gen, "temperature": 0, "logprobs": 5,
            "return_tokens_as_token_ids": True, "cache_salt": salt}
    t0 = time.perf_counter()
    d = post(base, "/v1/completions", body)
    return d, time.perf_counter() - t0


def generation(d: dict) -> dict:
    c = d["choices"][0]
    lp = c["logprobs"]
    return {"text": c["text"], "ids": [int(t.split(":", 1)[1]) for t in lp["tokens"]],
            "lp": lp["token_logprobs"], "top": lp["top_logprobs"]}


def compare(ref: dict, var: dict) -> dict:
    a, b = ref["ids"], var["ids"]
    n = min(len(a), len(b))
    div = next((i for i in range(n) if a[i] != b[i]), None)
    if div is None and len(a) != len(b):
        div = n
    upto = n if div is None else div
    d_lp = [abs(ref["lp"][i] - var["lp"][i]) for i in range(upto)]
    gap = None
    if div is not None and div < len(ref["top"]) and ref["top"][div]:
        top = sorted(ref["top"][div].values(), reverse=True)
        gap = round(top[0] - top[1], 4) if len(top) > 1 else None
    # First token: both see exactly the prompt, so the top-5 distributions compare like for like (total variation
    # over the union of the two top-5 sets, a token missing from one side counted as probability 0).
    p_ref, p_var = ref["top"][0] or {}, var["top"][0] or {}
    tv0 = 0.5 * sum(abs(2.718281828 ** p_ref.get(t, -1e9) - 2.718281828 ** p_var.get(t, -1e9))
                    for t in set(p_ref) | set(p_var))
    return {"identical": div is None, "first_div": div, "ref_gap_at_div": gap,
            "dlp_max": round(max(d_lp), 5) if d_lp else None, "dlp_mean": round(statistics.fmean(d_lp), 6) if d_lp else None,
            "tok0_same": a[:1] == b[:1], "tok0_tv": round(tv0, 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8001/metrics")
    ap.add_argument("--corpus", action="append", required=True, help="KIND=DIR:GLOB, KIND in code/docs")
    ap.add_argument("--lengths", default="6000,24000,96000")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=4096, help="the server's --max-num-batched-tokens")
    ap.add_argument("--shift", type=int, default=1000, help="D: chunk-boundary shift of the 'shifted' variants")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    run = uuid.uuid4().hex[:8]
    corpora = {}
    for spec in a.corpus:
        kind, _, path = spec.partition("=")
        corpora[kind] = (path.partition(":")[0], corpus_files(path))
        print(f"corpus {kind}: {len(corpora[kind][1])} files under {corpora[kind][0]}", flush=True)
    filler_ids = render(a.base, a.model, [{"role": "user", "content": "".join(
        FILLER.format(tag=f"{run}-{j % 997:03d}", j=j) for j in range((3 * a.chunk + a.shift) // 20))}])
    # Exactly 3 chunks + D: the filler takes the whole budget for three steps, the probed prompt then shares the
    # fourth with the filler's last D tokens whatever the (async) scheduler timing.
    filler_ids = filler_ids[: 3 * a.chunk + a.shift]

    def salt(tag: str) -> str:
        return f"php-{run}-{tag}-{uuid.uuid4().hex[:8]}"

    def request(name: str, ids: list[int], s: str, behind_filler: bool = False) -> dict:
        m0 = scrape(a.metrics_url)
        th = None
        if behind_filler:
            th = threading.Thread(target=complete, args=(a.base, a.model, filler_ids, 1, salt("filler")))
            th.start()
            time.sleep(0.05)
        d, dt = complete(a.base, a.model, ids, a.gen, s)
        if th:
            th.join()
        m1 = scrape(a.metrics_url)
        g = generation(d)
        g.update({"variant": name, "prompt_tokens": len(ids), "wall_s": round(dt, 3), "metrics": delta(m0, m1)})
        return g

    cases = []
    for n_tok in [int(x) for x in a.lengths.split(",")]:
        for kind, (root, files) in corpora.items():
            rng = random.Random(f"{kind}-{n_tok}")
            note_id = f"{rng.randint(100000, 999999)}"
            q1, q2 = QUESTIONS[kind]
            ratio = 3.6 if kind == "docs" else 3.1
            for _ in range(2):  # one refinement of the chars-per-token estimate
                ctx = build_context(files, root, int(n_tok * ratio), f"{kind}-{n_tok}")
                user1 = f"{ctx}Note: the build identifier of this snapshot is {note_id}.\n\n{q1}"
                p1 = render(a.base, a.model, [{"role": "user", "content": user1}])
                ratio *= n_tok / len(p1)
            s1 = salt("s1")
            rows = [request("fresh", p1, s1), request("hit", p1, s1), request("fresh2", p1, salt("s5")),
                    request("shifted", p1, salt("s2"), True)]
            answer = rows[0]["text"].strip()
            p2 = render(a.base, a.model, [{"role": "user", "content": user1}, {"role": "assistant", "content": answer},
                                          {"role": "user", "content": q2}])
            rows += [request("turn2-fresh", p2, salt("s3")), request("turn2-hit", p2, s1),
                     request("turn2-shifted", p2, salt("s4"), True)]
            ref = {"fresh": rows[0], "turn2-fresh": rows[4]}
            for r in rows:
                base_row = ref["turn2-fresh" if r["variant"].startswith("turn2") else "fresh"]
                r["vs_ref"] = compare(base_row, r) if r is not base_row else None
                r["note_ok"] = (note_id in r["text"][:40]) if r["variant"].startswith("turn2") else None
            case = {"kind": kind, "target_tokens": n_tok, "p1_tokens": len(p1), "p2_tokens": len(p2),
                    "note_id": note_id, "rows": [{k: v for k, v in r.items() if k not in ("top",)} for r in rows]}
            cases.append(case)
            print(f"\n{kind} ~{n_tok}: P1 {len(p1)} tok, P2 {len(p2)} tok, note {note_id}", flush=True)
            for r in rows:
                v = r["vs_ref"]
                hits = r["metrics"].get("prefix_cache_hits_total", 0)
                cmp_s = "reference" if v is None else (
                    "identical" if v["identical"] else
                    f"diverges @{v['first_div']} (ref gap {v['ref_gap_at_div']})")
                lp_s = "" if v is None or v["dlp_mean"] is None else f"  |dlp| mean {v['dlp_mean']:.2e} max {v['dlp_max']:.2e}"
                tv_s = "" if v is None else f"  tok0 TV {v['tok0_tv']:.4f}"
                note_s = "" if r["note_ok"] is None else ("  note OK" if r["note_ok"] else "  note WRONG")
                print(f"  {r['variant']:<14} {r['wall_s']:6.2f} s  hit {hits:>7}  {cmp_s}{tv_s}{lp_s}{note_s}", flush=True)

    summary = {}
    for variant in ("hit", "fresh2", "shifted", "turn2-hit", "turn2-shifted"):
        vs = [r["vs_ref"] for c in cases for r in c["rows"] if r["variant"] == variant]
        divs = [v["first_div"] for v in vs if not v["identical"]]
        gaps = [v["ref_gap_at_div"] for v in vs if v["ref_gap_at_div"] is not None]
        means = [v["dlp_mean"] for v in vs if v["dlp_mean"] is not None]
        summary[variant] = {"cases": len(vs), "identical": sum(v["identical"] for v in vs),
                            "first_div_median": statistics.median(divs) if divs else None,
                            "ref_gap_at_div_max": max(gaps) if gaps else None,
                            "dlp_mean_median": statistics.median(means) if means else None,
                            "tok0_same": sum(v["tok0_same"] for v in vs),
                            "tok0_tv_mean": round(statistics.fmean(v["tok0_tv"] for v in vs), 5) if vs else None}
    note = [r["note_ok"] for c in cases for r in c["rows"] if r["note_ok"] is not None]
    summary["turn2_note_ok"] = f"{sum(note)}/{len(note)}"
    print("\nsummary:", json.dumps(summary, indent=1), flush=True)
    if a.out:
        json.dump({"label": a.label, "run": run, "gen": a.gen, "chunk": a.chunk, "shift": a.shift,
                   "filler_tokens": len(filler_ids), "summary": summary, "cases": cases},
                  open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
