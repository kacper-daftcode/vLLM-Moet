#!/usr/bin/env python3
"""Torch-profiler capture of a fresh K3.1 prefill on a running vLLM server started with
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/.cache/prof",...}', and a per-kernel breakdown.

For every prompt size (default 16K and 64K tokens): /start_profile, one fresh prompt (unique nonce + random words,
no prefix hits, max_tokens=1), /stop_profile, then the newest rank-0 worker trace is parsed: GPU time per category
(MoE / dense GEMM / sparse MLA / indexer / KDA / collectives / norm+elementwise / draft / other), GPU busy vs idle over
the span, kernels per prefill step (= per chunk of --max-num-batched-tokens), the top kernels, and the two numbers
to compare against floors: us of GPU kernel time per prompt token and wall us per token (= 1 / single-stream tok/s).

usage: profile_prefill.py --base URL --model M --trace-dir HOSTDIR [--sizes 16384,65536] [--label L] [--out report.txt]
       profile_prefill.py --analyze-only TRACE.json.gz [--tokens N] [--label L]
API_KEY in the environment if the server needs one.
"""
import argparse, glob, gzip, json, os, re, sys, time, urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefill_bench import make_prompt  # noqa: E402

# First match wins. The indexer and the sparse attention come before the MoE row because "topk"/"routing" words
# occur in both; the MoE row before the GEMM row because trtllm-gen MoE kernels carry "gemm"/"bmm" in their names.
CATEGORIES = [
    ("draft (dspark)", re.compile(r"dspark|draft|speculat|eagle", re.I)),
    ("dsa indexer (fp8 logits + top-k)", re.compile(r"indexer|mqa_logits|fp8_mqa|topk_per_row|fast_topk|dsa_topk|"
                                                    r"sparse_attn_indexer|dcp_topk|topk_merge|index_k|index_q|k_norm|"
                                                    r"lightning|weights_proj|wq_b|wk_", re.I)),
    ("sparse mla attention", re.compile(r"prims_ts|sparse_mla|sparsemla|SparseMLA|mla|flash.*attn|fmha|attention|"
                                        r"attn|paged|concat_kv|kv_cache|rope|rotary|shard_q|gather_kv|mqa", re.I)),
    ("kda (flashkda / chunk)", re.compile(r"flashkda|kda|chunk_|gated_delta|wy_|fla_|conv1d|causal_conv|cumsum|"
                                          r"recurrent|ssm|recoverssm|mamba|l2norm|fused_recurrent|solve_tril", re.I)),
    ("collectives (nccl / custom ar)", re.compile(r"nccl|allreduce|all_reduce|reduce_scatter|all_gather|allgather|a2a|"
                                                 r"alltoall|symm|oneshot|twoshot|cross_device|multimem|mailbox|lamport|"
                                                 r"barrier", re.I)),
    ("moe (trtllm-gen fc1/fc2 + routing + quant)", re.compile(r"trtllm|bmm_|moe|Moe|expert|fc1|fc2|situ|swiglu|silu_and_mul|"
                                                               r"routing|topk_gating|permute|finalize|MXFP8Quant|mxfp8|"
                                                               r"fp4_quant|block_scale|activation", re.I)),
    ("dense gemm (bf16)", re.compile(r"gemm|cutlass|nvjet|sm100|sm_100|cublas|matmul|xmma|skinny|splitK|dotprod|"
                                     r"ampere_|hopper_|blackwell_", re.I)),
    ("norm / elementwise / copies", re.compile(r"rmsnorm|rms_norm|layer_norm|norm|elementwise|vectorized|cast|copy|"
                                               r"memcpy|memset|fill|index_select|gather|scatter|arange|where|cat|"
                                               r"embedding|add_|mul_|sigmoid|reduce_kernel|unrolled|transpose|"
                                               r"contiguous|clone|slice|view|pad", re.I)),
    ("sampling / lm_head", re.compile(r"sampl|argmax|softmax|logit|penalt|gumbel|rejection|topp|top_p|top_k", re.I)),
]


def post(base, path, body=None, key="", timeout=600):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + path, data=json.dumps(body or {}).encode(), headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def prefill_once(base, model, key, nwords, seed):
    body = {"model": model, "messages": [{"role": "user", "content": make_prompt(nwords, seed)}],
            "max_tokens": 1, "temperature": 0, "stream": False}
    t0 = time.monotonic()
    d = json.loads(post(base, "/v1/chat/completions", body, key, timeout=1800))
    return time.monotonic() - t0, int((d.get("usage") or {}).get("prompt_tokens", 0))


def wait_new_traces(trace_dir, before, timeout=240):
    deadline = time.time() + timeout
    new = []
    while time.time() < deadline:
        after = set(glob.glob(os.path.join(trace_dir, "**", "*.pt.trace.json*"), recursive=True))
        new = sorted(after - before, key=os.path.getmtime)
        if new:
            # the exporter writes the files one by one; give the last ones a moment to finish
            time.sleep(10)
            after = set(glob.glob(os.path.join(trace_dir, "**", "*.pt.trace.json*"), recursive=True))
            return sorted(after - before, key=os.path.getmtime)
        time.sleep(3)
    return new


def capture(base, model, key, trace_dir, nwords, seed, label):
    before = set(glob.glob(os.path.join(trace_dir, "**", "*.pt.trace.json*"), recursive=True))
    post(base, "/start_profile", key=key)
    time.sleep(1.0)
    dt, toks = prefill_once(base, model, key, nwords, seed)
    time.sleep(1.0)
    post(base, "/stop_profile", key=key, timeout=1800)
    print(f"[{label}] fresh prefill of {toks} tokens in {dt:.2f} s ({toks / dt:.0f} tok/s single stream) captured")
    files = wait_new_traces(trace_dir, before)
    return dt, toks, files


def load_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def union_busy(intervals):
    intervals.sort()
    busy, (cs, ce) = 0.0, intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return busy + (ce - cs)


def analyze(path, label, out, tokens=0, wall_s=0.0):
    data = load_trace(path)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    gpu = [e for e in events if e.get("cat") in ("kernel", "Kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
    if not gpu:
        print(f"[{label}] no GPU kernel events in {path}", file=out)
        return
    t_start = min(e["ts"] for e in gpu)
    t_end = max(e["ts"] + e["dur"] for e in gpu)
    span_ms = (t_end - t_start) / 1000.0
    busy_ms = union_busy([(e["ts"], e["ts"] + e["dur"]) for e in gpu]) / 1000.0
    by_name, n_by_name = defaultdict(float), defaultdict(int)
    for e in gpu:
        by_name[e["name"]] += e["dur"]
        n_by_name[e["name"]] += 1
    total_us = sum(by_name.values())
    # steps = the runner's per-step annotations (one per prefill chunk, plus one if a decode step slipped in)
    ann = defaultdict(int)
    for e in events:
        if e.get("cat") == "user_annotation" and e.get("name", "").startswith("execute_"):
            ann[e["name"]] += 1
    steps = max(ann.values()) if ann else 0
    cat_us, cat_n = defaultdict(float), defaultdict(int)
    for name, us in by_name.items():
        for cat, rx in CATEGORIES:
            if rx.search(name):
                cat_us[cat] += us
                cat_n[cat] += n_by_name[name]
                break
        else:
            cat_us["other"] += us
            cat_n["other"] += n_by_name[name]
    # per-stream busy: which streams carry the prefill, how much overlaps
    by_stream = defaultdict(list)
    for e in gpu:
        by_stream[e.get("tid")].append((e["ts"], e["ts"] + e["dur"]))
    print(f"\n=== {label}: {os.path.basename(path)}", file=out)
    tok_txt = f", {tokens} prompt tokens" if tokens else ""
    print(f"GPU span {span_ms:.1f} ms, GPU busy {busy_ms:.1f} ms ({100 * busy_ms / span_ms:.0f}%, idle {span_ms - busy_ms:.1f} ms), "
          f"kernel time summed over streams {total_us / 1000:.1f} ms, {len(gpu)} kernels"
          f"{', ' + str(steps) + ' runner steps (chunks)' if steps else ''}{tok_txt}", file=out)
    if tokens:
        print(f"per prompt token: {1000 * busy_ms / tokens:.1f} us GPU busy, {1000 * total_us / 1000 / tokens:.1f} us of kernels, "
              f"{1000 * span_ms / tokens:.1f} us of span" + (f"; request wall {wall_s:.2f} s = {1e6 * wall_s / tokens:.1f} us/token "
                                                          f"({tokens / wall_s:.0f} tok/s)" if wall_s else ""), file=out)
    if steps:
        print(f"per step (chunk): {span_ms / steps:.1f} ms span, {total_us / 1000 / steps:.1f} ms of kernels, {len(gpu) / steps:.0f} kernels", file=out)
    for sid, iv in sorted(by_stream.items(), key=lambda kv: -len(kv[1]))[:5]:
        b = union_busy(list(iv)) / 1000.0
        print(f"  stream {sid}: {len(iv):6d} kernels, busy {b:7.1f} ms ({100 * b / span_ms:3.0f}% of the span)", file=out)
    print("by category (ms | % of kernel time | kernels" + (" | us per token" if tokens else "") + "):", file=out)
    for cat, us in sorted(cat_us.items(), key=lambda kv: -kv[1]):
        extra = f" | {us / tokens:6.2f}" if tokens else ""
        print(f"  {cat:44s} {us / 1000:8.2f}  ({100 * us / total_us:4.1f}%) | {cat_n[cat]:6d}{extra}", file=out)
    print("top kernels (ms total | calls | us per call | name):", file=out)
    for name, us in sorted(by_name.items(), key=lambda kv: -kv[1])[:40]:
        print(f"  {us / 1000:8.2f} | {n_by_name[name]:6d} | {us / n_by_name[name]:8.1f} | {name[:120]}", file=out)
    out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="moonshotai/Kimi-K3.1-Pro")
    ap.add_argument("--trace-dir", default="")
    ap.add_argument("--sizes", default="16384,65536")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--analyze-only", default="", help="an existing trace file to analyze")
    ap.add_argument("--tokens", type=int, default=0, help="prompt tokens of an --analyze-only trace")
    a = ap.parse_args()
    key = os.environ.get("API_KEY", "")
    out = open(a.out, "a") if a.out else sys.stdout
    if a.analyze_only:
        analyze(a.analyze_only, a.label or "trace", out, a.tokens)
        return 0
    # calibration (also the warm-up request)
    dt, toks = prefill_once(a.base, a.model, key, 1500, seed=1)
    tpw = toks / 1500.0
    print(f"[{a.label}] calibration: {toks} tokens for 1500 words ({tpw:.2f} tok/word), {dt:.2f} s")
    seed = 5000
    for size in [int(s) for s in a.sizes.split(",") if s]:
        seed += 1
        time.sleep(3)
        nwords = max(8, int(size / tpw))
        wall, toks, files = capture(a.base, a.model, key, a.trace_dir, nwords, seed, f"{a.label} {size // 1024}K")
        worker = [f for f in files if "async" not in os.path.basename(f).lower()]
        rank0 = sorted(worker, key=lambda f: (0 if re.search(r"rank[_-]?0\b|TP0|_0\.", f) else 1, f))
        print(f"[{a.label} {size // 1024}K] traces: {len(files)} new file(s)")
        if rank0:
            analyze(rank0[0], f"{a.label} prefill {toks} tokens", out, toks, wall)
        else:
            print(f"[{a.label} {size // 1024}K] no worker trace found under {a.trace_dir}", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
