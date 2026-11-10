#!/usr/bin/env python3
"""CPU test of the idle freeze of the interpreter's heap (`moet_gc_idle_freeze.py`) and of its patcher.

No GPU, no vLLM needed: every case that touches the collector runs in a fresh interpreter (the collector's state is
per process). With vLLM installed and patched (the image), the last case also checks the installed file.

    python3 test_gc_idle_freeze.py            # prints one line per check, exit status 0 = all passed
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import patch_vllm_gc_idle_freeze_sm120 as patcher  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}", flush=True)


# vLLM's freeze_gc_heap() as the patcher's anchor expects it (vllm/utils/gc_utils.py of the patched version)
GC_UTILS = '''\
import gc


def freeze_gc_heap() -> None:
    """Freeze all objects tracked by the garbage collector. It should be invoked
    after server init / warmup, to reduce GC overhead from static objects
    during serving time.
    """
    # Ensure all static objects are pushed down to the oldest generation for
    # freeze
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    # Freeze all GC tracked objects
    gc.freeze()


def maybe_attach_gc_debug_callback() -> None:
    pass
'''

# the body every collector case starts with: a start-up heap of cyclic objects, frozen by the patched freeze_gc_heap()
PRELUDE = '''\
import gc, json, os, sys, time, threading, logging
sys.path.insert(0, {pkg!r})
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
from vllm.utils import gc_utils
from vllm.utils import moet_gc_idle_freeze as m


class Node:                      # a long-lived object in a reference cycle, like most of a server's state
    def __init__(self):
        self.me = self
        self.d = {{"a": [1, 2, 3]}}


def full_ms():
    t0 = time.perf_counter(); gc.collect(); return (time.perf_counter() - t0) * 1000.0


def burn(seconds):               # CPU work that leaves nothing behind
    t_end = time.process_time() + seconds
    while time.process_time() < t_end:
        pass


def wait_for(cond, seconds):
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


heap = [Node() for _ in range(200_000)]
out = {{}}
'''


def run_case(pkg: str, body: str, env: dict[str, str]) -> tuple[dict, str]:
    code = PRELUDE.format(pkg=pkg) + textwrap.dedent(body) + "\nprint('RESULT ' + json.dumps(out))\n"
    e = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_MOET_GC")}
    e.update(env)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e, timeout=180)
    res = {}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            import json

            res = json.loads(line[7:])
    if p.returncode != 0 and not res:
        res = {"error": p.stderr[-400:]}
    return res, p.stderr


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        # ---------------------------------------------------------------------------------------------- the patcher
        dist = Path(tmp) / "vllm"
        (dist / "utils").mkdir(parents=True)
        (dist / "__init__.py").write_text("")
        (dist / "utils" / "__init__.py").write_text("")
        target = dist / "utils" / "gc_utils.py"
        target.write_text(GC_UTILS)
        argv = sys.argv
        try:
            sys.argv = ["patch", "--dist", str(dist), "--check"]
            check("patcher --check on the unpatched file: not applied (status 1)", patcher.main() == 1)
            sys.argv = ["patch", "--dist", str(dist)]
            check("patcher applies (status 0)", patcher.main() == 0)
            once = target.read_text()
            check("the hand-over is in freeze_gc_heap(), after gc.freeze()",
                  once.index("gc.freeze()") < once.index("_moet_gc_idle_freeze.after_startup_freeze()") < once.index("def maybe_attach_gc_debug_callback"))
            check("the rest of the file is untouched", once.replace(patcher.EDITS[0][2], patcher.EDITS[0][1]) == GC_UTILS)
            check("the module is installed next to it, byte-identical",
                  (dist / "utils" / "moet_gc_idle_freeze.py").read_bytes() == (HERE / "moet_gc_idle_freeze.py").read_bytes())
            check("patcher is idempotent", patcher.main() == 0 and target.read_text() == once)
            sys.argv = ["patch", "--dist", str(dist), "--check"]
            check("patcher --check on the patched file: applied (status 0)", patcher.main() == 0)
            (dist / "utils" / "moet_gc_idle_freeze.py").write_text("# another version\n")
            check("patcher --check notices a different module (status 1)", patcher.main() == 1)
            sys.argv = ["patch", "--dist", str(dist)]
            patcher.main()
            other = Path(tmp) / "other" / "vllm"
            (other / "utils").mkdir(parents=True)
            (other / "utils" / "gc_utils.py").write_text(GC_UTILS.replace("# Freeze all GC tracked objects", "# freeze"))
            sys.argv = ["patch", "--dist", str(other)]
            try:
                patcher.main()
                refused = False
            except SystemExit as exc:
                refused = "anchor" in str(exc)
            check("a file whose anchor does not match is refused, not half-patched",
                  refused and "moet" not in (other / "utils" / "gc_utils.py").read_text() and not (other / "utils" / "moet_gc_idle_freeze.py").exists())
        finally:
            sys.argv = argv
        pkg = str(Path(tmp))

        # ------------------------------------------------------------------------------- switch off: only the log line
        res, err = run_case(pkg, '''
            before = len(gc.callbacks)
            gc_utils.freeze_gc_heap()
            out["callbacks_added"] = len(gc.callbacks) - before
            out["threads"] = [t.name for t in threading.enumerate() if t.name.startswith("moet-gc")]
            n2 = gc.get_stats()[2]["collections"]
            small = []
            for _ in range(340):                              # 102 K new long-lived objects, in small steps
                small.append([Node() for _ in range(100)])
            out["auto_full_collections_early"] = gc.get_stats()[2]["collections"] - n2
            keep = [Node() for _ in range(300_000)]          # accumulates after the freeze
            gc.collect()
            out["full_ms_grown"] = full_ms()
            burn(0.7); time.sleep(2.0)                        # worked, then idle: nothing may happen
            out["full_ms_after_idle"] = full_ms()
            out["stats"] = m.stats()
        ''', {"VLLM_MOET_GC_WARN_MS": "20"})
        check("off: freeze_gc_heap() still works, the stopwatch is attached, no thread", res.get("callbacks_added") == 1 and res.get("threads") == [], str(res.get("error", ""))[:200])
        check("off: the interpreter still remembers the pre-freeze heap (600 K objects): no automatic full collection within 102 K new objects",
              res.get("auto_full_collections_early") == 0, f"{res.get('auto_full_collections_early')} automatic full collection(s)")
        check("off: no idle freeze (a full collection stays as long as before the idle moment)",
              res.get("stats", {}).get("freezes") == 0 and res.get("full_ms_after_idle", 0) > 0.5 * res.get("full_ms_grown", 1e9),
              f"{res.get('full_ms_grown', -1):.1f} ms -> {res.get('full_ms_after_idle', -1):.1f} ms")
        check("off: a slow collection is logged, with the hint to the switch",
              "gc: a generation-2 collection took" in err and "VLLM_MOET_GC_IDLE_FREEZE=1" in err and res.get("stats", {}).get("slow", 0) >= 1,
              err.strip().splitlines()[0][:150] if err.strip() else "no log line")
        check("off: at most one line per 5 s, the rest counted", err.count("gc: a generation-") == 1 and res.get("stats", {}).get("slow", 0) >= 2,
              f"{err.count('gc: a generation-')} line(s), {res.get('stats', {}).get('slow')} slow collections")

        res, err = run_case(pkg, '''
            gc_utils.freeze_gc_heap()
            keep = [Node() for _ in range(300_000)]
            gc.collect(); full_ms()
            out["callbacks"] = len(gc.callbacks)
            out["stats"] = m.stats()
        ''', {"VLLM_MOET_GC_WARN_MS": "0"})
        check("VLLM_MOET_GC_WARN_MS=0: no stopwatch, no line", res.get("callbacks") == 0 and "gc: a generation-" not in err, str(res.get("error", ""))[:200])

        # -------------------------------------------------------------------------------------------------- switch on
        res, err = run_case(pkg, '''
            # the remembered old-generation size: without the extra collection the interpreter waits for a quarter of the
            # PRE-freeze heap (600 K objects here -> 150 K promotions) before its first full collection - the "off" case
            gc_utils.freeze_gc_heap()
            out["threads"] = [t.name for t in threading.enumerate() if t.name.startswith("moet-gc")]
            n2 = gc.get_stats()[2]["collections"]
            keep = []
            for _ in range(340):                              # 102 K new long-lived objects, in small steps
                keep.append([Node() for _ in range(100)])
            out["auto_full_collections_early"] = gc.get_stats()[2]["collections"] - n2
            keep.append([Node() for _ in range(300_000)])     # one burst leaves 900 K objects behind
            gc.collect()
            out["full_ms_grown"] = full_ms()
            burn(0.7)                                         # the process works ...
            out["freezes_while_busy"] = m.stats()["freezes"]
            ok = wait_for(lambda: m.stats()["freezes"] >= 1, 6.0)   # ... and goes idle
            out["froze_when_idle"] = ok
            out["full_ms_after_idle"] = full_ms()
            out["freeze_ms"] = m.stats()["last_freeze_ms"]
            n = m.stats()["freezes"]
            time.sleep(2.5)                                   # idle without work in between: no second freeze
            out["freezes_idle_again"] = m.stats()["freezes"] - n
            keep.append([Node() for _ in range(20_000)])      # the next burst: a collection walks only this
            gc.collect()
            out["full_ms_next_burst"] = full_ms()
            burn(0.7)
            out["froze_again"] = wait_for(lambda: m.stats()["freezes"] >= n + 1, 6.0)
            out["alive"] = len(heap) == 200_000 and sum(len(k) for k in keep) == 340 * 100 + 300_000 + 20_000 and keep[-1][-1].d["a"] == [1, 2, 3]
            out["stats"] = m.stats()
        ''', {"VLLM_MOET_GC_IDLE_FREEZE": "1", "VLLM_MOET_GC_WARN_MS": "0"})
        check("on: the idle thread runs", res.get("threads") == ["moet-gc-idle-freeze"], str(res.get("error", ""))[:300])
        check("on: the extra collection after the start-up freeze resets the remembered size (automatic full collections come early)",
              res.get("auto_full_collections_early", 0) >= 1, f"{res.get('auto_full_collections_early')} automatic full collection(s) within 102 K new objects")
        check("on: no freeze while the process works", res.get("freezes_while_busy") == 0)
        check("on: the heap is frozen again once the process has worked and gone idle", res.get("froze_when_idle") is True)
        check("on: a full collection is short again after it",
              res.get("full_ms_after_idle", 1e9) < 0.1 * res.get("full_ms_grown", 0) and res.get("full_ms_after_idle", 1e9) < 5.0,
              f"{res.get('full_ms_grown', -1):.1f} ms -> {res.get('full_ms_after_idle', -1):.2f} ms (the freeze itself {res.get('freeze_ms', -1):.1f} ms)")
        check("on: an idle process that has not worked is not frozen again", res.get("freezes_idle_again") == 0)
        check("on: the next burst's collection walks only that burst",
              res.get("full_ms_next_burst", 1e9) < 0.25 * res.get("full_ms_grown", 0), f"{res.get('full_ms_next_burst', -1):.1f} ms for 60 K objects")
        check("on: frozen again after the next burst", res.get("froze_again") is True)
        check("on: every object is still there and usable", res.get("alive") is True)

        res, err = run_case(pkg, '''
            gc_utils.freeze_gc_heap()
            gc_utils.freeze_gc_heap()                          # a second call (another component freezing again)
            out["threads"] = len([t for t in threading.enumerate() if t.name.startswith("moet-gc")])
            out["callbacks"] = len(gc.callbacks)
            # garbage of a burst is collected by the idle freeze, not frozen
            class Big:
                def __init__(self): self.me = self; self.blob = bytearray(1 << 20)
            import weakref
            g = Big(); w = weakref.ref(g); del g
            burn(0.7)
            wait_for(lambda: m.stats()["freezes"] >= 1, 6.0)
            out["garbage_collected_before_freeze"] = w() is None
            # a child forked from a serving process does not inherit a thread; it gets its own when it freezes its heap
            r, wfd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r)
                ok0 = m.stats()["running"] is False
                gc_utils.freeze_gc_heap()
                ok1 = m.stats()["running"] is True
                os.write(wfd, b"1" if (ok0 and ok1) else b"0"); os._exit(0)
            os.close(wfd); out["fork_ok"] = os.read(r, 1) == b"1"; os.waitpid(pid, 0)
            # an exception inside must never reach the caller
            real = m.threading.Thread
            m._state["thread"] = None
            m.threading.Thread = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no thread for you"))
            try:
                m.after_startup_freeze(); out["never_raises"] = True
            except Exception:
                out["never_raises"] = False
            m.threading.Thread = real
        ''', {"VLLM_MOET_GC_IDLE_FREEZE": "1", "VLLM_MOET_GC_IDLE_FREEZE_IDLE_S": "0.6"})
        check("on: a second freeze_gc_heap() adds no second thread and no second stopwatch", res.get("threads") == 1 and res.get("callbacks") == 1, str(res.get("error", ""))[:300])
        check("on: the garbage of a burst is collected by the idle freeze, not frozen", res.get("garbage_collected_before_freeze") is True)
        check("on: a forked child starts without the thread and gets its own at its freeze", res.get("fork_ok") is True)
        check("a failure inside the hook never reaches freeze_gc_heap()'s caller, and is logged", res.get("never_raises") is True and "not installed" in err)

        # ------------------------------------------------------------------------------- the installed file (the image)
        spec = importlib.util.find_spec("vllm")
        if spec is not None and spec.origin:
            vdist = Path(spec.origin).parent
            text = (vdist / patcher.REL).read_text()
            inst = vdist / "utils" / patcher.HELPER
            check("installed vLLM: gc_utils.py carries the hand-over and the module is the tree's",
                  patcher.EDITS[0][2] in text and inst.exists() and inst.read_bytes() == (HERE / patcher.HELPER).read_bytes())
        else:
            print("skip  installed vLLM: not importable here (the stand-alone cases above cover the code)")

    bad = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
