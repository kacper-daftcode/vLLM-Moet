#!/usr/bin/env python3
"""The two plan dictionaries of the b12x MoE path, held to ONE rule - no GPU, no FlashInfer, no vLLM.

vLLM's B12xExperts keeps one plan per distinct row count of a layer in `_plans` and never evicts one. The compact-bind patch
(patch_flashinfer_b12x_compact_bind.py) keeps one entry per plan that was ever bound in `_MOET_COMPACT_LAUNCHES`, keyed by
id(plan); the entry holds the plan and, for the plans of the compact path, its launchers; nothing evicts it either. Both only
grow, and they stop together, because the row counts of a layer are finite.

THE RULE: whoever bounds one of the two bounds the other in the same change. An eviction from `_plans` alone frees nothing -
the second dictionary still holds every plan - and an evicted plan's id can be handed to a new plan, which is what the helper's
`hit[0] is not plan` test is for.

What it holds:
  (1) the helper exactly as the patch writes it, executed against stubs: one entry per bound plan; a second bind of the same
      plan compiles nothing; the entry keeps the plan alive after its caller dropped it; a plan under a reused id gets its
      own launchers; a first bind inside a CUDA graph capture is refused;
  (2) the rule, over the tracked tools and Dockerfiles: the b12x experts' `_plans` and `_MOET_COMPACT_LAUNCHES` are evicted
      together or not at all. Today neither is. A change that evicts one of them fails here until it evicts the other.
      (vLLM's own file clears `_plans` wholesale in one place, when the packed weights' storage is replaced; that file is
      not in this tree and outside this check.)

usage: python3 test_b12x_plan_dicts.py
"""
from __future__ import annotations

import gc
import re
import subprocess
import sys
import types
import weakref
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import patch_flashinfer_b12x_compact_bind as patch  # noqa: E402

fails = 0


def check(cond: bool, text: str) -> None:
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {text}")
    fails += 0 if cond else 1


# ---- (1) the helper's text, as the patch inserts it, against stubs
compiled = []
pkg = types.ModuleType("moet_stub_pkg")
pkg.__path__ = []
prep = types.ModuleType("moet_stub_pkg._preparation")


def _compact_launches(launch_plan, caps):
    compiled.append((launch_plan, caps))
    return ("launchers", len(compiled))


prep._compact_launches = _compact_launches
sys.modules["moet_stub_pkg"], sys.modules["moet_stub_pkg._preparation"] = pkg, prep
capturing = [False]
torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_current_stream_capturing=lambda: capturing[0]))
ns = {"__name__": "moet_stub_pkg._impl", "__package__": "moet_stub_pkg", "torch": torch}
exec(compile(patch.HELPER, "<the helper of the compact-bind patch>", "exec"), ns)
cache, bind = ns["_MOET_COMPACT_LAUNCHES"], ns["_moet_compact_launches"]


class Plan:
    def __init__(self, n):
        self.launch_plan, self.caps = ("launch plan", n), ("caps", n)


print("the helper, as patched in")
plans = [Plan(n) for n in range(5)]
got = [bind(p) for p in plans]
check(len(cache) == 5 and len(compiled) == 5 and all(cache[id(p)][0] is p for p in plans),
      "(1) one entry per bound plan, keyed by its id, holding the plan")
again = [bind(p) for p in plans]
check(again == got and len(compiled) == 5 and len(cache) == 5, "(1) a second bind of the same plans compiles nothing and adds nothing")
alive = weakref.ref(plans[0])
key0 = id(plans[0])
del plans[0], got, again
gc.collect()
check(alive() is not None and key0 in cache, "(1) the entry keeps a plan alive after its caller dropped it: nothing is ever evicted")
fresh = Plan(99)
cache[id(fresh)] = (Plan(98), "the launchers of a plan that is gone")   # a stale entry under the id a new plan was given
check(bind(fresh) == ("launchers", 6) and cache[id(fresh)][0] is fresh, "(1) a plan under a reused id gets its own launchers, not the stale entry's")
capturing[0] = True
try:
    bind(Plan(100))
    refused = False
except RuntimeError as e:
    refused = "before CUDA graph capture" in str(e)
capturing[0] = False
check(refused and len(compiled) == 6, "(1) a first bind inside a CUDA graph capture is refused; nothing is compiled there")

# ---- (2) the rule: evicted together or not at all
print("the rule over the tracked tools")
PLANS = re.compile(r"(?<![A-Za-z0-9_])_plans\.(?:pop|popitem|clear)\(|\bdel\s+[\w.]*(?<![A-Za-z0-9_])_plans\[")
COMPACT = re.compile(r"_MOET_COMPACT_LAUNCHES\.(?:pop|popitem|clear)\(|\bdel\s+_MOET_COMPACT_LAUNCHES\[")
try:
    names = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "tools", "docker", "Dockerfile*"], check=True,
                           capture_output=True).stdout.decode().split("\0")
except (OSError, subprocess.CalledProcessError):
    names = [str(p.relative_to(ROOT)) for p in list(ROOT.glob("tools/**/*")) + list(ROOT.glob("Dockerfile*")) if p.is_file()]
me = str(Path(__file__).resolve().relative_to(ROOT))
hits = {"plans": [], "compact": []}
scanned = 0
for name in names:
    if not name or name == me or not (name.endswith((".py", ".sh")) or Path(name).name.startswith("Dockerfile")):
        continue
    try:
        text = (ROOT / name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    scanned += 1
    if PLANS.search(text):
        hits["plans"].append(name)
    if COMPACT.search(text):
        hits["compact"].append(name)
check(scanned > 100 and str(Path("tools/dsv41_sm120/patch_flashinfer_b12x_compact_bind.py")) in [n for n in names if n],
      f"(2) {scanned} tracked sources scanned, the compact-bind patch among them")
check(bool(hits["plans"]) == bool(hits["compact"]),
      "(2) `_plans` and `_MOET_COMPACT_LAUNCHES` are evicted together or not at all"
      + (f" - `_plans` evicted in {hits['plans']}, `_MOET_COMPACT_LAUNCHES` in {hits['compact']}: bound the other one in the same change"
         if bool(hits["plans"]) != bool(hits["compact"]) else f" (today: {'both' if hits['plans'] else 'neither'})"))
# the check has teeth: the same patterns on the two ways such a bound would be written
check(PLANS.search("        self._plans.popitem(last=False)\n") and PLANS.search("    del experts._plans[key]\n")
      and not PLANS.search("    _prepared._functional_plans.clear()\n") and not PLANS.search("plans = [p for key, p in experts._plans.items()]")
      and COMPACT.search("    _MOET_COMPACT_LAUNCHES.pop(id(plan), None)\n") and not COMPACT.search("hit = _MOET_COMPACT_LAUNCHES.get(id(plan))"),
      "(2) the patterns see an eviction of either dictionary and no reading of them, nor another object's dictionary")

print(f"{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
