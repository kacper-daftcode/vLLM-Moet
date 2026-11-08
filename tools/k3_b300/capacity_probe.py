#!/usr/bin/env python3
"""KV capacity probe: N concurrent long decodes with ~P-token prompts; reports the peak number of requests the
engine keeps running at once, preemptions, KV usage and aggregate decode rate.
usage: capacity_probe.py --base URL --n 128 --prompt-tokens 32768 --max-tokens 3000 --out F
"""
import argparse, concurrent.futures, json, os, threading, time, urllib.request, uuid

FILLER = ("Reference notes: the cache stores recently accessed entries. "
          "An implementation should maintain ordering, handle replacement and validate its invariants.\n")


def metrics(base):
    out = {}
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            for key in ("vllm:num_requests_running", "vllm:num_requests_waiting{", "vllm:num_preemptions_total",
                        "vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"):
                if line.startswith(key):
                    out[key.rstrip("{")] = float(line.rsplit(" ", 1)[1])
    return out


def run_one(base, key, model, prompt_chars, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": uuid.uuid4().hex + "\n" + FILLER * (prompt_chars // len(FILLER)) +
                                          "\nSummarize the notes above in one paragraph."}],
            "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0 = time.monotonic()
    first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            if first is None and d.get("choices"):
                first = time.monotonic()
    return {"ttft": (first or t0) - t0, "total": time.monotonic() - t0, "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--prompt-tokens", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    prompt_chars = int(a.prompt_tokens * 4.3)
    m0 = metrics(a.base)
    samples = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                m = metrics(a.base)
                m["t"] = time.monotonic()
                samples.append(m)
            except Exception:
                pass
            stop.wait(2.0)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(a.n) as ex:
        res = list(ex.map(lambda _: run_one(a.base, a.api_key, a.model, prompt_chars, a.max_tokens), range(a.n)))
    dt = time.monotonic() - t0
    stop.set()
    th.join()
    m1 = metrics(a.base)
    peak_running = max(s.get("vllm:num_requests_running", 0) for s in samples)
    peak_kv = max(s.get("vllm:kv_cache_usage_perc", s.get("vllm:gpu_cache_usage_perc", 0)) for s in samples)
    preempt = m1.get("vllm:num_preemptions_total", 0) - m0.get("vllm:num_preemptions_total", 0)
    toks = sum((r["usage"] or {}).get("completion_tokens", 0) for r in res)
    ptoks = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in res)
    summary = {"label": a.label, "n": a.n, "prompt_tokens_each": ptoks / max(len(res), 1), "max_tokens": a.max_tokens,
               "peak_running": peak_running, "peak_kv_usage": peak_kv, "preemptions": preempt, "seconds": dt,
               "completion_tokens": toks, "agg_decode_tok_s_over_run": toks / dt,
               "ttft_median": sorted(r["ttft"] for r in res)[len(res) // 2]}
    json.dump({"summary": summary, "samples": samples, "results": res}, open(a.out, "w"), indent=1)
    print(f"{a.label:22s} n={a.n} prompt~{summary['prompt_tokens_each']:.0f} peak_running={peak_running:.0f} "
          f"peak_kv={peak_kv:.1%} preemptions={preempt:.0f} agg {toks / dt:.0f} tok/s over {dt:.0f}s "
          f"ttft_med={summary['ttft_median']:.1f}s")


if __name__ == "__main__":
    main()
