#!/usr/bin/env python3
"""One-shot import of the 2026-07-15/16 quality campaign (pre-harness JSONs
from /root/logs/bench/ds4-2x6000) into bench/results/<release>/ in the
harness schema, with the raw tool JSONs as artifacts. Idempotent."""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench",
                                "runner"))
import common  # noqa: E402

SRC = "/root/logs/bench/ds4-2x6000"
RELEASE = "v2026.07.17-quality"
BOX = "rtx-pro6000x4"
RECIPE = "deepseek-v4-flash/pro6000x2-tp2-maxq"


def _items(path):
    with open(path) as f:
        d = json.load(f)
    return d, {r["item_id"]: r for r in d.get("runs", [])
               if r.get("phase") == "profile"}


def _mcnemar_p(b, c):
    """Two-sided exact McNemar (binomial on the discordant pairs)."""
    import math
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def agg(path, baseline_file=None):
    """Aggregates for the result JSON; the paired comparison is RECOMPUTED
    against the declared baseline (the campaign files carry embedded
    comparisons against OTHER references — e.g. think vs its own non-think
    — which must not leak into the harness result)."""
    d, items = _items(path)
    acc = d.get("accuracy") or {}
    toks = sorted(x.get("completion_tokens") or 0 for x in items.values())
    out = {
        "accuracy_pct": round(100.0 * acc.get("accuracy", 0.0), 2),
        "correct": acc.get("correct"), "of": acc.get("scored"),
        "truncated_no_answer": acc.get("truncated_no_answer"),
        "hit_max_tokens": acc.get("hit_max_tokens"),
        "tokens_avg": round(sum(toks) / len(toks), 1) if toks else None,
        "tokens_p50": toks[len(toks) // 2] if toks else None,
        "tokens_p90": toks[int(0.9 * len(toks))] if toks else None,
    }
    if baseline_file:
        _, base = _items(baseline_file)
        common_ids = [k for k in base if k in items]
        ob = sum(1 for k in common_ids
                 if base[k]["correct"] and not items[k]["correct"])
        oc = sum(1 for k in common_ids
                 if items[k]["correct"] and not base[k]["correct"])
        bt = [base[k].get("completion_tokens") or 0 for k in common_ids]
        ct = [items[k].get("completion_tokens") or 0 for k in common_ids]
        bm = sum(bt) / len(bt) if bt else None
        cm = sum(ct) / len(ct) if ct else None
        acc_b = sum(1 for k in common_ids if base[k]["correct"]) / len(common_ids)
        acc_c = sum(1 for k in common_ids if items[k]["correct"]) / len(common_ids)
        out["vs_baseline"] = {
            "acc_delta_pp": round(100 * (acc_c - acc_b), 2),
            "flips_only_baseline": ob,
            "flips_only_candidate": oc,
            "mcnemar_p": round(_mcnemar_p(ob, oc), 4),
            "token_inflation_pct": round((cm - bm) / bm * 100, 1)
            if bm and cm else None,
        }
    return out


def main():
    adir = common.artifacts_dir(RELEASE, BOX)
    probes = {}

    for key, src, profile, runs, conc, mt, baseline, overrides in [
        ("gsm8k", "gsm8k_maxq_a.json", "gsm8k", 200, 2, 6000,
         "ds4-flash/native-tp2-gsm8k-c2", {}),
        ("gsm8k-run2", "gsm8k_maxq_b.json", "gsm8k", 200, 2, 6000,
         "ds4-flash/native-tp2-gsm8k-c2", {}),
        ("gpqa", "gpqa_maxq.json", "gpqa-diamond", 198, 2, 30000,
         "ds4-flash/native-tp2-gpqa-nt", {}),
        ("gpqa-think", "gpqa_maxq_think.json", "gpqa-diamond", 198, 2, 65536,
         "ds4-flash/native-tp2-gpqa-think-high",
         {"chat_template_kwargs": {"thinking": True,
                                   "reasoning_effort": "high"},
          "temperature": 1.0, "top_p": 1.0}),
    ]:
        tag = f"{RECIPE.replace('/', '__')}__{key}"
        shutil.copy(os.path.join(SRC, src),
                    os.path.join(adir, f"quality__{tag}.json"))
        bfile = os.path.join(common.baselines_dir(),
                             common.load_baseline_registry()[baseline]["file"])
        p = agg(os.path.join(SRC, src), baseline_file=bfile)
        p.update(profile=profile, runs=runs, concurrency=conc,
                 max_tokens=mt, baseline=baseline,
                 request_overrides=overrides,
                 artifact=f"quality__{tag}.json")
        probes[key] = p

    # needle 121k from the campaign log (needle_maxq.out): PASS at 121152 tok
    probes["needle"] = {
        "cases": [{"words": 90000, "prompt_tokens": 121152, "depth": 0.5,
                   "pass": True}],
        "all_pass": True,
    }

    result = {
        "schema": 1,
        "release": RELEASE, "box": BOX, "recipe": RECIPE,
        "suite": "quality", "provenance": "imported",
        "source": "2026-07-15/16 campaign (logs ds4-2x6000: maxq.log, "
                  "gpqa-think.log; fork c2c066e3a via worktree "
                  "prefill-fp4-tier; internal/PREFILL_KV_INFLATION_"
                  "FINDINGS.md)",
        "summary": common.load_recipe(RECIPE).get("summary", ""),
        "context": 131072,
        "gpus_used": [0, 1],
        "started": "2026-07-15T19:03:00+0000",
        "finished": "2026-07-16T22:30:00+0000",
        "status": "ok",
        "probes": probes,
        "notes": "decode agg @c2: 98-110 tok/s short-form, ~97 think "
                 "long-form (native 122.5) - the -21% think-mode decode "
                 "cost is a standing follow-up.",
    }
    path = common.result_path(RELEASE, BOX, RECIPE)
    common.write_result(path, result)
    print(f"imported -> {path}")


if __name__ == "__main__":
    main()
