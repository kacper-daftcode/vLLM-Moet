"""Engram row-id statistics of DeepSeek-V4.1-Flash on real text: what a DRAM row cache in front of NVMe buys.

    python engram_hash_probe.py [--code-tokens 20000000] [--parquet .../train-00000-of-00880.parquet]
                                [--transcripts /root/.cursor/projects/root/agent-transcripts] [--json out.json]

Ports vLLM's hashing (models/deepseek_v41/common/engram.py: compressed token map, per-layer multipliers,
prime bucket ranges, rolling XOR hash) to numpy, tokenizes a corpus (code + agent transcripts), and reports per
Engram layer: lookups per token (24: 3 n-gram sizes x 8 heads), distinct rows, the share of lookups served by
the most frequent rows (a static cache sized in rows / GiB at 264 B per row = 256 B e4m3 + 8 scale bytes),
estimated honestly on a held-out tail of the stream, and the working set per 2048-token prefill chunk (the
NVMe reads a chunk needs on a cold cache). Document boundaries are ignored (one stream).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

MODEL_DIR = os.environ.get("DSV41_MODEL_DIR", "/mnt/nvme/models/DeepSeek-V4.1-Flash")
ROW_BYTES = 256 + 8


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    c = start + 1
    while not is_prime(c) or c in seen:
        c += 1
    return c


def build_compressed_token_map(tokenizer) -> tuple[np.ndarray, int]:
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = np.zeros(len(tokenizer), dtype=np.int64)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids, max_ngram_size, compressed_vocab_size) -> np.ndarray:
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        rows.append(gen.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64) * 2 + 1)
    return np.stack(rows)


def layout(cfg: dict):
    layer_ids = tuple(cfg["engram_layer_ids"])
    max_ngram, n_heads = cfg["engram_max_ngram_size"], cfg["engram_n_heads"]
    primes, seen = [], set()
    for _ in layer_ids:
        per = []
        for _ in range(max_ngram - 1):
            sizes, cur = [], cfg["engram_vocab_size"] - 1
            for _ in range(n_heads):
                cur = find_next_prime(cur, seen)
                seen.add(cur)
                sizes.append(cur)
            per.append(sizes)
        primes.append(per)
    flat = [[p for per in layer for p in per] for layer in primes]
    offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
    return layer_ids, max_ngram, n_heads, np.array(flat, dtype=np.int64), np.array(offsets, dtype=np.int64)


def hash_rows(comp: np.ndarray, layer: int, mult: np.ndarray, primes: np.ndarray, offsets: np.ndarray,
              max_ngram: int, n_heads: int, pad_id: int) -> np.ndarray:
    """[T] compressed ids -> [T, (max_ngram-1)*n_heads] row ids of one layer (stream start padded)."""
    T = comp.shape[0]
    out = np.empty((T, (max_ngram - 1) * n_heads), dtype=np.int64)
    rolling = np.zeros(T, dtype=np.int64)
    for shift in range(max_ngram):
        value = np.full(T, pad_id, dtype=np.int64)
        value[shift:] = comp[: T - shift] if shift else comp
        rolling ^= value * mult[layer, shift]
        if shift > 0:
            for h in range(n_heads):
                col = (shift - 1) * n_heads + h
                out[:, col] = rolling % primes[layer, col] + offsets[layer, col]
    return out


def corpus_tokens(tok, parquet: str, code_tokens: int, transcripts: str) -> tuple[np.ndarray, dict]:
    texts, kinds = [], {}
    if transcripts:
        n = 0
        for f in sorted(glob.glob(os.path.join(transcripts, "**", "*.jsonl"), recursive=True)):
            with open(f) as fh:
                for line in fh:
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    msg = o.get("message") if isinstance(o, dict) else None
                    if not isinstance(msg, dict):
                        continue
                    for part in msg.get("content", []) or []:
                        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                            texts.append(part["text"])
                            n += len(part["text"])
                        elif isinstance(part, dict) and part.get("type") == "tool_use":
                            texts.append(json.dumps(part.get("input", {}))[:20000])
        kinds["transcript_chars"] = n
    ids = []
    t0 = time.time()
    for t in texts:
        ids.extend(tok.encode(t, add_special_tokens=False))
    kinds["transcript_tokens"] = len(ids)
    if parquet and code_tokens > 0:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet)
        got = 0
        for batch in pf.iter_batches(batch_size=512, columns=["code"]):
            for code in batch.column(0).to_pylist():
                e = tok.encode(code, add_special_tokens=False)
                ids.extend(e)
                got += len(e)
                if got >= code_tokens:
                    break
            if got >= code_tokens:
                break
        kinds["code_tokens"] = got
    kinds["tokenize_s"] = time.time() - t0
    return np.array(ids, dtype=np.int64), kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/mnt/nvme/data/github-code-clean/train-00000-of-00880.parquet")
    ap.add_argument("--code-tokens", type=int, default=20_000_000)
    ap.add_argument("--transcripts", default="/root/.cursor/projects/root/agent-transcripts")
    ap.add_argument("--holdout", type=float, default=0.3, help="tail fraction of the stream scored against the head")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))["text_config"]
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    t0 = time.time()
    token_map, csize = build_compressed_token_map(tok)
    assert csize == cfg["engram_compressed_vocab_size"], (csize, cfg["engram_compressed_vocab_size"])
    layer_ids, max_ngram, n_heads, primes, offsets = layout(cfg)
    mult = compute_hash_multipliers(layer_ids, max_ngram, csize)
    print(f"token map {len(tok)} -> {csize} ids, layers {layer_ids}, {max_ngram - 1} n-gram sizes x {n_heads} heads, "
          f"bucket ranges ~{primes[0, 0] / 1e6:.1f}M rows each; setup {time.time() - t0:.1f}s")
    ids, info = corpus_tokens(tok, args.parquet, args.code_tokens, args.transcripts)
    T = ids.shape[0]
    print(f"corpus: {T / 1e6:.1f}M tokens ({info})")
    comp = token_map[ids]
    pad_id = cfg["engram_pad_token_id"]
    split = int(T * (1 - args.holdout))
    results = {"tokens": int(T), "row_bytes": ROW_BYTES, "layers": {}}
    sizes_rows = [1 << 20, 4 << 20, 16 << 20, 32 << 20, 64 << 20, 128 << 20]
    for li, layer in enumerate(layer_ids):
        rows = hash_rows(comp, li, mult, primes, offsets, max_ngram, n_heads, pad_id)   # [T, 24]
        n_cols = rows.shape[1]
        table_rows = cfg["engram_num_embeddings"][li]
        flat_train = rows[:split].ravel()
        flat_test = rows[split:].ravel()
        uniq_all = np.unique(rows.ravel()).size
        print(f"layer {layer}: table {table_rows / 1e6:.0f}M rows ({table_rows * ROW_BYTES / 2**30:.1f} GiB); "
              f"{n_cols} lookups/token; distinct rows touched by {T / 1e6:.1f}M tokens: {uniq_all / 1e6:.2f}M "
              f"({uniq_all * ROW_BYTES / 2**30:.2f} GiB)")
        # frequency-ranked static cache from the head, scored on the tail
        tr_ids, tr_cnt = np.unique(flat_train, return_counts=True)
        order = np.argsort(-tr_cnt, kind="stable")
        ranked = tr_ids[order]
        te_ids, te_cnt = np.unique(flat_test, return_counts=True)
        te_total = te_cnt.sum()
        lay = {"table_rows": int(table_rows), "distinct_rows": int(uniq_all), "static_cache": {}, "per_ngram": {}}
        line = []
        for s in sizes_rows:
            top = ranked[:s]
            hit = te_cnt[np.isin(te_ids, top, assume_unique=True)].sum() / te_total
            lay["static_cache"][str(s)] = float(hit)
            line.append(f"{s >> 20}M rows ({s * ROW_BYTES / 2**30:.1f} GiB): {hit * 100:5.1f} %")
        print("  static top-frequency cache (trained on the head of the stream, scored on the tail): " + " | ".join(line))
        # per n-gram size: distinct share and self-coverage
        for g in range(max_ngram - 1):
            cols = rows[:, g * n_heads:(g + 1) * n_heads].ravel()
            u, c = np.unique(cols, return_counts=True)
            c.sort()
            top16 = c[-(16 << 20) // n_heads * 1:].sum() / c.sum() if c.size else 0.0  # rows per head range
            lay["per_ngram"][str(g + 2)] = {"distinct": int(u.size), "cov_top_2M_rows_per_head": float(top16)}
            print(f"  {g + 2}-grams: distinct rows {u.size / 1e6:.2f}M of {n_heads * primes[li, g * n_heads] / 1e6:.0f}M; "
                  f"top 2M rows per head cover {top16 * 100:.1f} % of their lookups")
        # working set per 2048-token chunk and per 16K window (cold-cache NVMe reads)
        for win in (2048, 16384):
            n_win = T // win
            ws = np.array([np.unique(rows[i * win:(i + 1) * win].ravel()).size for i in range(min(n_win, 200))])
            lay[f"unique_rows_per_{win}"] = float(ws.mean())
            print(f"  distinct rows per {win}-token window: mean {ws.mean():.0f} of {win * n_cols} lookups "
                  f"({ws.mean() / (win * n_cols) * 100:.0f} % distinct)")
        results["layers"][str(layer)] = lay
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
