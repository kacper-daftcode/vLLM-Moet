#!/usr/bin/env python3
"""Fresh-prefill throughput of a running K3.1 server (the prefill counterpart of k3_code_bench's steps/s).

Every request carries a unique nonce in front of a random-word body, so nothing hits the prefix cache (GPU or the
disk tier) and the measured time is the prefill of the whole prompt. max_tokens=1, so the request time is the TTFT
plus ~1 decode step; the server's usage.prompt_tokens gives the exact token count (the word count is calibrated with
one pilot request, see --calib-words).

Phases (default): single stream at 1K / 2K / 8K / 16K / 64K tokens, --reps repetitions each (median reported); then C=4 and
C=8 streams of 16K tokens launched together (aggregate = sum of prompt tokens / time from launch to the last first
token; TTFT median). One summary line on stdout for exp-*-summary.txt (exp.sh takes `tail -1`), everything in --out.

usage: prefill_bench.py --base URL --model M [--sizes 1024,2048,8192,16384,65536] [--reps 2] [--conc-size 16384]
                        [--concurrencies 4,8] [--label L] [--out prefill-exp-L.json]
API_KEY in the environment if the server needs one.
"""
import argparse, concurrent.futures, json, os, random, statistics, sys, time, urllib.request, uuid

WORDS = ("alpha quantum river matrix ember glacier syntax violet nimbus cobalt tangent fjord lantern zephyr cipher "
         "marble thunder willow plasma onyx harbor crimson vector lattice meadow falcon pixel saffron tundra orbit "
         "kernel ledger prism summit canyon beacon cobble dynamo ferrite granite helium iodine jasper kelvin lumen "
         "magnet nickel oxide pylon quartz radius sulfur titan uranium velvet walnut xenon yttrium zircon argon "
         "basalt copper delta epoch fathom gamma hinge index joule krypton lithium mantle neutron octave photon "
         "quasar resin sigma torque umbra vortex wafer yield zenith anchor bridge circuit dial engine filter gear").split()


def make_prompt(nwords, seed):
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(nwords))
    return f"{uuid.uuid4().hex}\n{body}\nReply with the single word OK."


def post(base, key, body, timeout):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def one(base, key, model, nwords, seed, timeout=1800):
    """One fresh prefill: returns (seconds, prompt_tokens, error)."""
    body = {"model": model, "messages": [{"role": "user", "content": make_prompt(nwords, seed)}],
            "max_tokens": 1, "temperature": 0, "stream": False}
    t0 = time.monotonic()
    try:
        d = post(base, key, body, timeout)
    except Exception as e:  # noqa: BLE001
        return time.monotonic() - t0, 0, f"{type(e).__name__}: {str(e)[:120]}"
    dt = time.monotonic() - t0
    return dt, int((d.get("usage") or {}).get("prompt_tokens", 0)), None


def fmt_k(x):
    return f"{x / 1000:.1f}K"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="moonshotai/Kimi-K3.1-Pro")
    ap.add_argument("--sizes", default="1024,2048,8192,16384,65536", help="single-stream prompt sizes in tokens")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--conc-size", type=int, default=16384, help="prompt size of the concurrent phase (0 = skip)")
    ap.add_argument("--concurrencies", default="4,8")
    ap.add_argument("--calib-words", type=int, default=1500, help="pilot request size for the tokens/word calibration")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between requests (lets the server drain)")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",") if s]
    concs = [int(c) for c in a.concurrencies.split(",") if c]
    out = {"label": a.label, "base": a.base, "model": a.model, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "single": {}, "concurrent": {}, "errors": []}

    # calibration: tokens per word on this tokenizer (random words of this vocabulary)
    dt, toks, err = one(a.base, a.api_key, a.model, a.calib_words, seed=1)
    if err or toks <= 0:
        print(f"{a.label} prefill: calibration failed ({err})")
        return 1
    tpw = toks / a.calib_words
    out["tokens_per_word"] = tpw
    parts = []
    seed = 1000
    for size in sizes:
        nwords = max(8, int(size / tpw))
        runs = []
        for _ in range(a.reps):
            time.sleep(a.pause)
            seed += 1
            dt, toks, err = one(a.base, a.api_key, a.model, nwords, seed)
            if err:
                out["errors"].append({"phase": f"single-{size}", "error": err})
                continue
            runs.append({"seconds": dt, "prompt_tokens": toks, "tok_s": toks / dt})
        out["single"][str(size)] = runs
        if runs:
            med = statistics.median(r["seconds"] for r in runs)
            med_tps = statistics.median(r["tok_s"] for r in runs)
            ptoks = int(statistics.median(r["prompt_tokens"] for r in runs))
            parts.append(f"{fmt_k(ptoks)} x1 {med:.2f}s {med_tps / 1000:.1f}K tok/s")
        else:
            parts.append(f"{fmt_k(size)} x1 FAIL")
    if a.conc_size > 0:
        nwords = max(8, int(a.conc_size / tpw))
        for c in concs:
            time.sleep(a.pause)
            seeds = [seed + 1 + i for i in range(c)]
            seed += c
            t0 = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(c) as ex:
                res = list(ex.map(lambda s: one(a.base, a.api_key, a.model, nwords, s), seeds))
            # every request starts at ~t0; its own duration is its TTFT (+1 step); the aggregate uses the last one
            last = max(r[0] for r in res)
            toks = sum(r[1] for r in res)
            errs = [r[2] for r in res if r[2]]
            out["errors"].extend({"phase": f"conc-{c}", "error": e} for e in errs)
            ttfts = sorted(r[0] for r in res if not r[2])
            rec = {"concurrency": c, "prompt_tokens_total": toks, "launch_to_last_s": last,
                   "agg_tok_s": toks / last if last > 0 else 0,
                   "ttft_median_s": ttfts[len(ttfts) // 2] if ttfts else None, "errors": len(errs),
                   "wall_s": time.monotonic() - t0}
            out["concurrent"][str(c)] = rec
            if ttfts:
                parts.append(f"{fmt_k(a.conc_size)} x{c} agg {rec['agg_tok_s'] / 1000:.1f}K tok/s "
                             f"ttft_p50 {rec['ttft_median_s']:.2f}s")
            else:
                parts.append(f"{fmt_k(a.conc_size)} x{c} FAIL")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    errtxt = f" | errors {len(out['errors'])}" if out["errors"] else ""
    print(f"{a.label} prefill: " + " | ".join(parts) + errtxt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
