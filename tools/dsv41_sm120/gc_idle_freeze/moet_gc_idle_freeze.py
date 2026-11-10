"""The interpreter's full garbage collections kept short in a long-running serving process (`VLLM_MOET_GC_IDLE_FREEZE=1`,
default off), and one log line for every collection that is slow (always on).

Installed as `vllm.utils.moet_gc_idle_freeze` by `patch_vllm_gc_idle_freeze_sm120.py`; vLLM's `freeze_gc_heap()` calls
`after_startup_freeze()` in each serving process (the API server, the EngineCore, every TP worker).

Why. vLLM freezes a serving process's heap once, when its start-up is done (`freeze_gc_heap()`: three collections, then
`gc.freeze()`), so that the collector never walks the model, the compiled kernels and the captured graphs again - 1.25 M
objects in a TP worker of the DeepSeek-V4.1 server. Whatever becomes long-lived after that moment is not frozen. Measured in
a serving 4-rank server (`gc.get_objects` by type between requests): every worker gains about 800 objects for each step
token count it has not seen before - b12x's fused-MoE plans, 20 plan sets of about 40 objects; a fresh prompt brings one new
count (its last step), a fresh 19K + 163K prefill pair 1 450-1 600 objects - and the EngineCore one KV-offload index entry
per stored chunk (about 1 450 per pair); the API server's old generation does not grow. A full (generation 2) collection
walks all of them, on the thread that happened to allocate and with the interpreter lock held - 0.13-0.36 us per such object
in a serving worker (10-19 ms 34 pairs after a start; 550-600 ms for the whole heap of a worker when it is un-frozen) - and
CPython runs one whenever the objects promoted since the last full collection exceed a quarter of the old generation's size
at that collection. In a pipelined prefill step that thread is one of a rank's two launch threads: the rank stops, the other
ranks wait for it in the collectives. Nine hours and 160 M prompt tokens after its start a 4-rank server stops for 0.3-0.4 s
in this way every few minutes of load per rank, and the pause grows with every request served (+40 % in 2.7 hours of
benchmark traffic); a restart resets it.

What, with the switch on:

  1. right after the start-up freeze: one more `gc.collect()`. It walks only the handful of objects created since the freeze
     and resets the old-generation size the interpreter remembers for its rule (after `gc.freeze()` that size is still the
     whole pre-freeze heap, so the first automatic full collection would come only after about 312 K promotions and then meet
     everything accumulated until then);
  2. whenever the process has worked and then been idle for `VLLM_MOET_GC_IDLE_FREEZE_IDLE_S` seconds (default 1.0; idle =
     all its threads together used less than 2 % of one core in that time): `gc.collect()` followed by `gc.freeze()`. The
     collection is short (it walks what the requests since the last idle moment left; their garbage is collected before the
     freeze) and no request waits for it. In a worker and in the EngineCore no request is in flight at that moment, so
     nothing a request holds is frozen. The API server is different: it is idle while the engine works on a long prompt, so
     its freezes fall inside such requests (about 1 ms, off the request's path); what the request holds in that process is
     frozen then and is released by reference count when the request ends, unless it sits in a reference cycle. Every later
     automatic full collection walks only what has become long-lived since the last idle moment.

In a serving server (the collector's callbacks logged in every process, one server lifetime): with 4 s between fresh 19K +
163K pairs every worker and the EngineCore freeze once per gap - the workers 2 s after the pair, their main threads spin for
the first second - 1.5-2.1 ms per freeze, no full collection runs inside a request, and a forced one afterwards takes
0.01 ms; with the pairs back to back (no idle second) full collections of 1.3-3.0 ms run inside requests, growing until the
next idle second - as without the switch, but from the last idle moment instead of from the start of the server. Un-freezing
a worker after 13 idle freezes and collecting its 1.32 M objects found no dead one.

Always, switch on or off: a warning line for every collection that takes `VLLM_MOET_GC_WARN_MS` milliseconds or more
(default 50; 0 turns it off; at most one line per 5 s, the others are counted) - two clock reads per collection.

It touches no tensor, no kernel and no order of operations: the served bits cannot change. What it costs: one thread per
serving process that wakes four times a second; an idle freeze holds the interpreter lock for the milliseconds of its
collection (a request that arrives in exactly that moment starts that much later); an object that is alive at an idle moment
and later dies inside a reference cycle is not reclaimed by the collector any more - the same as for vLLM's own start-up
freeze.

Never call `gc.get_freeze_count()` on a serving path: it walks the whole frozen list (about 100 ms at 1.25 M objects, with the
interpreter lock held).
"""
from __future__ import annotations

import gc
import os
import threading
import time

try:  # inside vLLM: its logger (and format); stand-alone (the unit test): the standard library's
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger("vllm.utils.moet_gc_idle_freeze")

POLL_S = 0.25          # the idle thread's period
IDLE_SHARE = 0.02      # of one core: below this the process counts as idle
MIN_WORK_S = 0.5       # CPU seconds the process must have used since the last freeze before another one is worth doing
WARN_EVERY_S = 5.0     # at most one slow-collection line in this time

_state = {
    "pid": None,            # the process the idle thread belongs to (a forked child starts its own when it freezes its heap)
    "thread": None,
    "stopwatch": False,
    "t0": 0.0,
    "freezes": 0,           # idle freezes done
    "last_freeze_ms": 0.0,
    "slow": 0,              # collections at or above the warning threshold
    "slow_max_ms": 0.0,
    "last_warn": 0.0,
    "suppressed": 0,
}


def enabled() -> bool:
    return os.environ.get("VLLM_MOET_GC_IDLE_FREEZE", "0") == "1"


def idle_seconds() -> float:
    try:
        return max(float(os.environ.get("VLLM_MOET_GC_IDLE_FREEZE_IDLE_S", "1.0")), 2 * POLL_S)
    except ValueError:
        return 1.0


def warn_ms() -> float:
    try:
        return float(os.environ.get("VLLM_MOET_GC_WARN_MS", "50"))
    except ValueError:
        return 50.0


def stats() -> dict:
    return {k: _state[k] for k in ("freezes", "last_freeze_ms", "slow", "slow_max_ms")} | {"enabled": enabled(), "running": _running()}


def _running() -> bool:
    t = _state["thread"]
    return t is not None and _state["pid"] == os.getpid() and t.is_alive()


def _stopwatch(phase: str, info: dict) -> None:
    """gc.callbacks entry: two clock reads per collection; a line when one takes warn_ms() or longer."""
    try:
        if phase == "start":
            _state["t0"] = time.perf_counter()
            return
        ms = (time.perf_counter() - _state["t0"]) * 1000.0
        limit = warn_ms()
        if limit <= 0 or ms < limit:
            return
        _state["slow"] += 1
        _state["slow_max_ms"] = max(_state["slow_max_ms"], ms)
        now = time.monotonic()
        if now - _state["last_warn"] < WARN_EVERY_S:
            _state["suppressed"] += 1
            return
        more = f"; {_state['suppressed']} more since the last line" if _state["suppressed"] else ""
        _state["last_warn"], _state["suppressed"] = now, 0
        hint = "" if enabled() else " - VLLM_MOET_GC_IDLE_FREEZE=1 keeps full collections short"
        logger.warning(
            "gc: a generation-%s collection took %.0f ms on thread %s with the interpreter lock held (collected %s objects%s)%s",
            info.get("generation"), ms, threading.current_thread().name, info.get("collected"), more, hint,
        )
    except Exception:  # noqa: BLE001 - never raise into the collector
        pass


def freeze_now(reason: str = "idle") -> float:
    """`gc.collect()` then `gc.freeze()`; returns the milliseconds both took."""
    t0 = time.perf_counter()
    gc.collect()
    gc.freeze()
    ms = (time.perf_counter() - t0) * 1000.0
    _state["freezes"] += 1
    _state["last_freeze_ms"] = ms
    if ms >= 20.0:
        logger.info("gc idle freeze: the %s freeze took %.0f ms", reason, ms)
    return ms


def _idle_loop(idle_s: float) -> None:
    cpu_at_freeze = time.process_time()
    hist: list[tuple[float, float]] = []          # (monotonic, process CPU seconds) over the last idle_s
    while True:
        time.sleep(POLL_S)
        try:
            if _state["pid"] != os.getpid():      # a forked child: this is not its thread
                return
            now, cpu = time.monotonic(), time.process_time()
            hist.append((now, cpu))
            while len(hist) > 2 and now - hist[1][0] >= idle_s:
                hist.pop(0)
            span = now - hist[0][0]
            if span < idle_s:
                continue
            if cpu - hist[0][1] <= IDLE_SHARE * span and cpu - cpu_at_freeze >= MIN_WORK_S:
                freeze_now()
                cpu_at_freeze = time.process_time()
        except Exception:  # noqa: BLE001 - the thread must outlive any surprise
            pass


def after_startup_freeze() -> None:
    """Called by vLLM's `freeze_gc_heap()` right after its `gc.freeze()`. Never raises."""
    try:
        if warn_ms() > 0 and not _state["stopwatch"]:
            gc.callbacks.append(_stopwatch)
            _state["stopwatch"] = True
        if not enabled():
            return
        gc.collect()                              # step 1 of the docstring: resets the remembered old-generation size
        if _running():
            return
        idle_s = idle_seconds()
        t = threading.Thread(target=_idle_loop, args=(idle_s,), name="moet-gc-idle-freeze", daemon=True)
        _state["pid"], _state["thread"] = os.getpid(), t
        t.start()
        logger.info("gc idle freeze: on - the heap is frozen again whenever this process has worked and been idle for %.1f s", idle_s)
    except Exception as exc:  # noqa: BLE001 - never break a server's start
        try:
            logger.warning("gc idle freeze: not installed (%s: %s)", type(exc).__name__, exc)
        except Exception:  # noqa: BLE001
            pass
