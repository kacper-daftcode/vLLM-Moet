"""Sequences for the native routing traces (PLAN-dsv41-5090-cpu 1.2): text + token counts, one JSONL.

    python build_sequences.py --out sequences.jsonl [--sessions 12] [--session-tokens 16384] [--code-files 8]

Kinds:
  humaneval  164 x (V4.1 chat prompt of hard_probe.py + the native completion recorded on fin-01)  ~450 tokens
  math500    60 x (chat prompt + the recorded tail of the native solution)                          ~350 tokens
  session    the largest Cursor agent transcripts rendered as V4.1 chat (user / assistant turns,
             tool calls as text), cut to --session-tokens                                          8-16K tokens
  code       raw source files from github-code-clean, >= --code-tokens each                        ~8K tokens
Every sequence is sent as one /v1/completions prompt with max_tokens = 1 (teacher forcing): all positions are
prefill-routed, which under VLLM_MOE_W2_PREFILL_FP4=ensure is the native model's routing.
Decode-position masks (where the model would have been generating) are recorded as character spans: for
humaneval / math500 the completion, for sessions the assistant turns, for code everything.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

MODEL_DIR = os.environ.get("DSV41_MODEL_DIR", "/mnt/nvme/models/DeepSeek-V4.1-Flash")
INTERNAL = "/mnt/nvme/vLLM-Moet/internal"
sys.path.insert(0, INTERNAL)

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER, ASSISTANT, SYSTEM = "<｜User｜>", "<｜Assistant｜>", "<｜System｜>"
THINK_END = "</think>"


def render_chat(turns: list[tuple[str, str]], system: str | None = None) -> tuple[str, list[tuple[int, int]]]:
    """[(role, text)] -> prompt text in the V4.1 chat format (thinking off) + assistant char spans."""
    s = BOS
    if system:
        s += SYSTEM + system
    spans = []
    for role, text in turns:
        if role == "user":
            s += USER + text
        else:
            s += ASSISTANT + THINK_END
            a0 = len(s)
            s += text
            spans.append((a0, len(s)))
            s += EOS
    return s, spans


def humaneval_sequences(tok) -> list[dict]:
    import hard_probe as HP  # noqa: E402

    items = [json.loads(l) for l in gzip.open(HP.fetch(HP.HE_URL, "HumanEval.jsonl.gz"), "rt") if l.strip()]
    native = {}
    with open(os.path.join(INTERNAL, "hard_humaneval_native_fin01.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            native[r["i"]] = r
    out = []
    for i, it in enumerate(items):
        comp = native[i]["code"] if i in native else ""
        text, spans = render_chat([("user", HP.HE_USER.format(prompt=it["prompt"])),
                                   ("assistant", "```python\n" + comp + "\n```")])
        out.append(dict(id=f"he/{i}", kind="humaneval", text=text, decode_spans=spans))
    return out


def math_sequences(tok) -> list[dict]:
    import hard_probe as HP  # noqa: E402

    items = [json.loads(l) for l in open(HP.fetch(HP.MATH_URL, "math500.jsonl"))]
    items = [it for it in items if it["level"] == 5][:60]
    native = {}
    with open(os.path.join(INTERNAL, "hard_math500_native_fin01.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            native[r["i"]] = r
    out = []
    for i, it in enumerate(items):
        tail = native[i]["text"] if i in native else ""
        text, spans = render_chat([("user", it["problem"]), ("assistant", tail)], system=HP.MATH_SYS)
        out.append(dict(id=f"math/{i}", kind="math500", text=text, decode_spans=spans))
    return out


def transcript_turns(path: str) -> list[tuple[str, str]]:
    turns = []
    with open(path) as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except Exception:
                continue
            msg = o.get("message") if isinstance(o, dict) else None
            if not isinstance(msg, dict) or o.get("role") not in ("user", "assistant"):
                continue
            parts = []
            for part in msg.get("content", []) or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and part.get("text"):
                    parts.append(part["text"])
                elif part.get("type") == "tool_use":
                    parts.append("```tool_call\n" + json.dumps({"name": part.get("name"), "input": part.get("input", {})})[:6000] + "\n```")
                elif part.get("type") == "tool_result":
                    c = part.get("content")
                    if isinstance(c, list):
                        c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                    parts.append("```tool_result\n" + str(c)[:6000] + "\n```")
            if parts:
                text = "\n".join(parts)
                if turns and turns[-1][0] == o["role"]:
                    turns[-1] = (o["role"], turns[-1][1] + "\n" + text)
                else:
                    turns.append((o["role"], text))
    return turns


def session_sequences(tok, root: str, n: int, max_tokens: int) -> list[dict]:
    files = sorted(glob.glob(os.path.join(root, "*", "*.jsonl")), key=os.path.getsize, reverse=True)
    out = []
    for f in files:
        if len(out) >= n:
            break
        turns = transcript_turns(f)
        if not turns:
            continue
        # grow the rendered prefix turn by turn until max_tokens
        kept = []
        for t in turns:
            text, _ = render_chat(kept + [t])
            if len(tok.encode(text, add_special_tokens=False)) > max_tokens:
                break
            kept.append(t)
        if len(kept) < 2:
            continue
        text, spans = render_chat(kept)
        ntok = len(tok.encode(text, add_special_tokens=False))
        if ntok < max_tokens // 2:
            continue
        out.append(dict(id=f"session/{os.path.basename(f)[:8]}", kind="session", text=text, decode_spans=spans))
    return out


def code_sequences(tok, parquet: str, n: int, min_tokens: int) -> list[dict]:
    import pyarrow.parquet as pq

    out = []
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(batch_size=256, columns=["code", "path"]):
        for code, path in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if len(code) < min_tokens * 2:
                continue
            ids = tok.encode(code, add_special_tokens=False)
            if len(ids) < min_tokens:
                continue
            text = BOS + tok.decode(ids[: min_tokens + 256])
            out.append(dict(id=f"code/{len(out)}:{path}", kind="code", text=text, decode_spans=[(len(BOS), len(text))]))
            if len(out) >= n:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--session-tokens", type=int, default=16384)
    ap.add_argument("--transcripts", default="/root/.cursor/projects/root/agent-transcripts")
    ap.add_argument("--code-files", type=int, default=8)
    ap.add_argument("--code-tokens", type=int, default=8192)
    ap.add_argument("--parquet", default="/mnt/nvme/data/github-code-clean/train-00000-of-00880.parquet")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    seqs = humaneval_sequences(tok) + math_sequences(tok)
    seqs += session_sequences(tok, args.transcripts, args.sessions, args.session_tokens)
    seqs += code_sequences(tok, args.parquet, args.code_files, args.code_tokens)
    total = 0
    with open(args.out, "w") as f:
        for s in seqs:
            s["tokens"] = len(tok.encode(s["text"], add_special_tokens=False))
            total += s["tokens"]
            f.write(json.dumps(s) + "\n")
    by_kind = {}
    for s in seqs:
        k = by_kind.setdefault(s["kind"], [0, 0])
        k[0] += 1
        k[1] += s["tokens"]
    print(f"{len(seqs)} sequences, {total} tokens -> {args.out}")
    for k, (n, t) in by_kind.items():
        print(f"  {k:10s} {n:4d} sequences  {t:8d} tokens  (mean {t // max(n, 1)})")


if __name__ == "__main__":
    main()
