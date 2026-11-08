#!/usr/bin/env python3
"""KV offload round-trip probe for hybrid K3: a greedy prompt is answered cold, then GPU-warm, then evicted from the
GPU by a stream of large distinct prompts, then answered again from the connector (disk / page cache). Reports the
external prefix-hit counters, TTFTs and whether the restored answer (and a multi-turn continuation on top of it)
equals the GPU-warm one. Run against a server with a small --kv-cache-memory so the eviction is quick.
usage: evict_probe.py --base URL [--prompt-tokens 30000] [--evict-prompts 30] [--evict-tokens 100000] [--api-key KEY]
"""
import argparse, concurrent.futures, json, os, time, urllib.request, uuid

FILLER = ("Reference notes: the cache stores recently accessed entries. "
          "An implementation should maintain ordering, handle replacement and validate its invariants.\n")
FILLER2 = ("Field log: the sensor array reported nominal readings; calibration drift stayed within tolerance and the "
           "operator logged no anomalies during the shift.\n")


def counters(base):
    out = {}
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            for key in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                        "vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
                        "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total"):
                if line.startswith(key):
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def delta(a, b, key):
    return b.get(key, 0.0) - a.get(key, 0.0)


def chat(base, key, model, messages, max_tokens, temperature=0.0):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 1.0, "seed": 0, "reasoning_effort": "low", "stream": True,
            "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0 = time.monotonic()
    first = None
    content, reasoning, usage = [], [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                dl = ch.get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):
                    first = first or time.monotonic()
                if dl.get("content"):
                    content.append(dl["content"])
                if dl.get("reasoning_content") or dl.get("reasoning"):
                    reasoning.append(dl.get("reasoning_content") or dl.get("reasoning"))
    return {"ttft": (first or time.monotonic()) - t0, "content": "".join(content), "reasoning": "".join(reasoning),
            "usage": usage}


def same(x, y):
    return (x["content"], x["reasoning"]) == (y["content"], y["reasoning"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--prompt-tokens", type=int, default=30000)
    ap.add_argument("--evict-prompts", type=int, default=30)
    ap.add_argument("--evict-tokens", type=int, default=100000)
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    B, K, M = a.base, a.api_key, a.model
    doc = uuid.uuid4().hex + "\n" + FILLER * int(a.prompt_tokens * 4.3 / len(FILLER))
    q1 = [{"role": "user", "content": doc + "\nIn two sentences, what do the notes above describe?"}]

    c0 = counters(B)
    cold = chat(B, K, M, q1, 160)
    c1 = counters(B)
    warm = chat(B, K, M, q1, 160)
    c2 = counters(B)
    follow = lambda prev: q1 + [{"role": "assistant", "content": prev["content"], "reasoning_content": prev["reasoning"]},
                                {"role": "user", "content": "Now list three invariants such a cache should validate. Be brief."}]
    turn2_warm = chat(B, K, M, follow(warm), 200)
    c3 = counters(B)
    print(f"{a.label} cold  : prompt={cold['usage'].get('prompt_tokens')} ttft={cold['ttft']:.2f}s | "
          f"gpu hits +{delta(c0, c1, 'vllm:prefix_cache_hits_total'):.0f} ext hits +{delta(c0, c1, 'vllm:external_prefix_cache_hits_total'):.0f}")
    print(f"{a.label} warm  : ttft={warm['ttft']:.2f}s identical_to_cold={same(cold, warm)} | "
          f"gpu hits +{delta(c1, c2, 'vllm:prefix_cache_hits_total'):.0f} ext hits +{delta(c1, c2, 'vllm:external_prefix_cache_hits_total'):.0f}")
    print(f"{a.label} turn2 : ttft={turn2_warm['ttft']:.2f}s gpu hits +{delta(c2, c3, 'vllm:prefix_cache_hits_total'):.0f}")

    # Evict: distinct large prompts, enough to cycle the whole GPU pool several times over.
    t0 = time.monotonic()
    n_fill = int(a.evict_tokens * 4.3 / len(FILLER2))

    def one(i):
        p = [{"role": "user", "content": uuid.uuid4().hex + "\n" + FILLER2 * n_fill + "\nReply with the single word OK."}]
        return chat(B, K, M, p, 8, temperature=1.0)["usage"].get("prompt_tokens", 0)

    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        toks = sum(ex.map(one, range(a.evict_prompts)))
    c4 = counters(B)
    print(f"{a.label} evict : {a.evict_prompts} prompts, {toks} tokens in {time.monotonic() - t0:.0f}s, "
          f"kv usage now {c4.get('vllm:kv_cache_usage_perc', 0):.0%}, preemptions +{delta(c3, c4, 'vllm:num_preemptions_total'):.0f}")

    restored = chat(B, K, M, q1, 160)
    c5 = counters(B)
    print(f"{a.label} disk  : ttft={restored['ttft']:.2f}s identical_to_warm={same(warm, restored)} | "
          f"gpu hits +{delta(c4, c5, 'vllm:prefix_cache_hits_total'):.0f} ext queries +{delta(c4, c5, 'vllm:external_prefix_cache_queries_total'):.0f} "
          f"ext hits +{delta(c4, c5, 'vllm:external_prefix_cache_hits_total'):.0f}")
    turn2_disk = chat(B, K, M, follow(warm), 200)
    c6 = counters(B)
    print(f"{a.label} turn2': ttft={turn2_disk['ttft']:.2f}s identical_to_turn2={same(turn2_warm, turn2_disk)} | "
          f"gpu hits +{delta(c5, c6, 'vllm:prefix_cache_hits_total'):.0f} ext hits +{delta(c5, c6, 'vllm:external_prefix_cache_hits_total'):.0f}")
    if not same(warm, restored):
        print("warm   :", repr(warm["content"][:300]))
        print("disk   :", repr(restored["content"][:300]))
    if not same(turn2_warm, turn2_disk):
        print("turn2  :", repr(turn2_warm["content"][:300]))
        print("turn2' :", repr(turn2_disk["content"][:300]))
    if a.out:
        json.dump({"cold": cold, "warm": warm, "turn2_warm": turn2_warm, "restored": restored, "turn2_disk": turn2_disk,
                   "counters": [c0, c1, c2, c3, c4, c5, c6]}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
