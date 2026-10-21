"""Record the native routing of the sequences through the W2 server (PLAN-dsv41-5090-cpu 1.2).

Server: ds41-w2 relaunched on the dev-loop overlay with the recorder + native prefill, e.g.
  cd /mnt/nvme/vLLM-Moet && IMAGE=vllm-moet-sm120:dsv41-0909 W2_OVERLAY_DIR=/mnt/nvme/w2-overlay \
    CUBIT_DIR=/mnt/nvme/vLLM-Moet/kernels/cubins-sm120 \
    EXTRA_DOCKER_ARGS="-v /mnt/nvme/vLLM-Moet-cpu/internal/dsv41-5090-cpu/routes:/dump" \
    EXTRA_ARGS="--no-enable-prefix-caching" \
    internal/relaunch_w2.sh 46-routes GPU_MEM_UTIL=0.93 W2_BASE_CACHE_GB=8 W2_DELTA_GB=12 SPEC_TOKENS=0 \
      MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=2048 \
      W2_EXTRA_ENV="VLLM_MOE_W2_PREFILL_FP4=ensure VLLM_MOE_W2_DELTA_POLICY=need VLLM_MOE_W2_ROUTE_DUMP=/dump/route.bin VLLM_MOE_W2_KPI_EVERY=100"
Then:
  python collect_routes.py send  --sequences sequences.jsonl --manifest manifest.json [--base http://127.0.0.1:8001]
  python collect_routes.py parse --sequences sequences.jsonl --manifest manifest.json --dump route.bin --out routes.npz

`send` posts every sequence as one /v1/completions prompt (max_tokens 1, temperature 0, one at a time) and records
the server's prompt_tokens. `parse` splits the recorder's records ([i32 layer_key, n_rows, k] + int16 ids) into
per-sequence arrays [T, L, k] using the manifest (prefix caching must be off so every position is routed), skips
decode steps, and stores routes.npz: ids_<i> int16 [T, 40, 6], plus kinds / ids / decode masks (token-level, from
the character spans tokenized with the model's tokenizer).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

MODEL_DIR = os.environ.get("DSV41_MODEL_DIR", "/mnt/nvme/models/DeepSeek-V4.1-Flash")
N_LAYERS = 40
TOPK = 6


def select(seqs, args):
    """--kinds a,b restricts kinds; --every N --phase P keeps items with index-within-kind % N == P
    (applied to the kinds in --every-kinds, default humaneval,math500)."""
    kinds = set(args.kinds.split(",")) if args.kinds else None
    every_kinds = set(args.every_kinds.split(","))
    idx_in_kind = {}
    out = []
    for i, s in enumerate(seqs):
        j = idx_in_kind.get(s["kind"], 0)
        idx_in_kind[s["kind"]] = j + 1
        if kinds and s["kind"] not in kinds:
            continue
        if args.every > 1 and s["kind"] in every_kinds and j % args.every != args.phase:
            continue
        out.append(i)
    return out


def send(args):
    seqs = [json.loads(l) for l in open(args.sequences)]
    chosen = select(seqs, args)
    print(f"sending {len(chosen)} of {len(seqs)} sequences, {sum(seqs[i]['tokens'] for i in chosen)} tokens", flush=True)
    manifest = []
    t_all = time.time()
    for n_done, i in enumerate(chosen):
        s = seqs[i]
        body = {"model": args.model, "prompt": s["text"], "max_tokens": 1, "temperature": 0}
        req = urllib.request.Request(f"{args.base}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.load(r)
        dt = time.time() - t0
        pt = d["usage"]["prompt_tokens"]
        manifest.append(dict(i=i, id=s["id"], kind=s["kind"], prompt_tokens=pt, local_tokens=s["tokens"],
                             secs=round(dt, 2)))
        print(f"[{n_done + 1}/{len(chosen)}] {s['id'][:40]:40s} {pt:6d} tokens  {dt:6.1f}s  "
              f"({pt / max(dt, 1e-9):.0f} tok/s)", flush=True)
        with open(args.manifest, "w") as f:
            json.dump(manifest, f, indent=1)
    print(f"done: {sum(m['prompt_tokens'] for m in manifest)} tokens in {time.time() - t_all:.0f}s -> {args.manifest}")


def read_records(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    pos = 0
    while pos + 12 <= data.size:
        layer, n, k = np.frombuffer(data[pos:pos + 12].tobytes(), dtype=np.int32)
        pos += 12
        ids = np.frombuffer(data[pos:pos + 2 * n * k].tobytes(), dtype=np.int16).reshape(n, k)
        pos += 2 * n * k
        yield int(layer), ids


def parse(args):
    seqs = [json.loads(l) for l in open(args.sequences)]
    manifest = []
    for mf in args.manifest.split(","):
        manifest += json.load(open(mf))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    recs = list(read_records(args.dump))
    print(f"{len(recs)} records in {args.dump}")
    out = {}
    kinds, ids_list, ntoks = [], [], []
    r = 0
    for m in manifest:
        T = m["prompt_tokens"]
        s = seqs[m["i"]]
        # the server may prepend a BOS to the raw prompt: shift the character-span masks by the difference
        shift = T - m.get("local_tokens", T)
        if shift not in (0, 1):
            print(f"  warning: {m['id']} server tokens {T} vs local {m.get('local_tokens')} (shift {shift})")
        rows = [[] for _ in range(N_LAYERS)]
        got = [0] * N_LAYERS
        # skip stray 1-row (decode) steps between sequences
        while r < len(recs) and recs[r][1].shape[0] == 1 and T != 1:
            r += 1
        while min(got) < T:
            if r >= len(recs):
                raise RuntimeError(f"dump ended inside sequence {m['id']} (got {got})")
            layer, ids = recs[r]
            r += 1
            if got[layer] + ids.shape[0] > T:
                raise RuntimeError(f"sequence {m['id']}: layer {layer} overflow {got[layer]} + {ids.shape[0]} > {T}")
            rows[layer].append(ids)
            got[layer] += ids.shape[0]
        arr = np.stack([np.concatenate(rows[l], axis=0) for l in range(N_LAYERS)], axis=1)  # [T, L, k]
        assert arr.shape == (T, N_LAYERS, TOPK), arr.shape
        # token-level decode mask from the character spans (tokenize the prefix up to each span boundary)
        mask = np.zeros(T, dtype=bool)
        for a, b in s.get("decode_spans", []):
            ta = len(tok.encode(s["text"][:a], add_special_tokens=False)) + max(shift, 0)
            tb = len(tok.encode(s["text"][:b], add_special_tokens=False)) + max(shift, 0)
            mask[min(ta, T):min(tb, T)] = True
        out[f"ids_{m['i']}"] = arr.astype(np.int16)
        out[f"decode_{m['i']}"] = mask
        kinds.append(s["kind"])
        ids_list.append(s["id"])
        ntoks.append(T)
    out["kinds"] = np.array(kinds)
    out["seq_ids"] = np.array(ids_list)
    out["tokens"] = np.array(ntoks)
    out["seq_index"] = np.array([m["i"] for m in manifest])   # keys are ids_<seq_index>, decode_<seq_index>
    np.savez_compressed(args.out, **out)
    print(f"{len(manifest)} sequences, {sum(ntoks)} positions -> {args.out}; {len(recs) - r} records left over")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("send")
    a.add_argument("--sequences", required=True)
    a.add_argument("--manifest", required=True)
    a.add_argument("--base", default="http://127.0.0.1:8001")
    a.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    a.add_argument("--kinds", default="", help="comma list of kinds to send (default all)")
    a.add_argument("--every", type=int, default=1, help="keep every N-th item of --every-kinds")
    a.add_argument("--phase", type=int, default=0)
    a.add_argument("--every-kinds", default="humaneval,math500")
    a.set_defaults(fn=send)
    p = sub.add_parser("parse")
    p.add_argument("--sequences", required=True)
    p.add_argument("--manifest", required=True, help="one or more manifest files, comma-separated, in send order")
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=parse)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
