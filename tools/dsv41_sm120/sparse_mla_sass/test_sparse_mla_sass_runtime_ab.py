#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The runtime A/B key of the sparse-MLA prefill SASS route (`SPARSE_MLA_PREFILL_SASS_CUBIN_SEL`), tested on the helper
that `patch_vllm_sparse_mla_prefill_sass_sm120.py` embeds into vLLM.

  default (no GPU, no vLLM): the helper is executed with a stub launcher and a fake clock; every row of the control
      file's semantics is checked — master switch off = no stat at all, the 100-ms stat gate, key present / absent /
      bad value / unknown key / other namespaces, file missing / removed / malformed, the change line and the on-demand
      line (exact text), both cubins launched once at the first eager launch and never during a CUDA-graph capture, a
      second cubin that fails to load or whose output differs is refused.
  --gpu --cubin A --cubin-b B [dumps ...] (one GPU inside the image): the same helper with the real cubins on vLLM-shaped
      arguments — every state and every transition (0 -> 1 -> 0 -> 1 -> {} -> 1 -> file removed), output `torch.equal`
      to spec B on every launch (synthetic chunk + the given dumps of a real prefill), counters exact, the launch
      captured into a CUDA graph replays the cubin it was captured with, and with the master switch off the control
      file is never looked at.

usage: python3 test_sparse_mla_sass_runtime_ab.py            (CPU)
       python3 test_sparse_mla_sass_runtime_ab.py --gpu --cubin X.cubin --cubin-b Y.cubin [attn_*.pt ...]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import tempfile
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import patch_vllm_sparse_mla_prefill_sass_sm120 as P  # noqa: E402

KEY = "SPARSE_MLA_PREFILL_SASS_CUBIN_SEL"
TAG = "[vllm-moet sparse-MLA prefill SASS, device"
ENV = ("VLLM_MOET_SPARSE_MLA_PREFILL_SASS", "VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN", "VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN_B",
       "VLLM_MOET_RUNTIME_AB", "VLLM_MOET_RUNTIME_AB_FILE", "VLLM_MOET_RUNTIME_STATS_FILE")
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail and not ok else ''}", flush=True)


def stub_vllm_logger() -> None:
    vllm = types.ModuleType("vllm")
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda name: logging.getLogger(name)
    sys.modules.setdefault("vllm", vllm)
    sys.modules["vllm.logger"] = logger


def fresh_helper() -> dict:
    """A new copy of the helper's namespace = a new worker process (its state lives in module globals)."""
    ns: dict = {"__name__": "moet_sass_helper_under_test"}
    exec(P.HELPER, ns)
    return ns


def write_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(obj if isinstance(obj, str) else json.dumps(obj))
    os.replace(tmp, path)


class StatCounter:
    """Counts os.stat calls on the two runtime files (the helper stats nothing else)."""

    def __init__(self, paths):
        self.paths = {os.fspath(p) for p in paths}
        self.n = 0
        self._orig = os.stat

    def __enter__(self):
        def stat(p, *a, **k):
            if os.fspath(p) in self.paths:
                self.n += 1
            return self._orig(p, *a, **k)

        os.stat = stat
        return self

    def __exit__(self, *exc):
        os.stat = self._orig


# ------------------------------------------------------------------------------------------------------------ CPU part
class FakeLauncher:
    """Stands in for _MoetSparseMlaSass: output = a fixed function of q (a cubin whose path contains DIFF is off by one
    bit in one element; a path containing NOLOAD raises on load)."""

    calls: list[tuple[str, int]] = []

    def __init__(self, cubin: str):
        if "NOLOAD" in cubin:
            raise RuntimeError("CUDA 301: b'CUDA_ERROR_FILE_NOT_FOUND'")
        self.cubin = cubin

    def run(self, q, kv_cache, indices, output, sm_scale, topk_length, attn_sink, extra_kv_cache, extra_indices, extra_topk_length):
        FakeLauncher.calls.append((self.cubin, q.shape[0]))
        output.copy_((q.float() * 0.5).to(output.dtype))
        if "DIFF" in self.cubin:
            v = output.view(torch.int16)
            v[0, 0, 0] ^= 1


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def cpu_tests() -> None:
    stub_vllm_logger()
    capturing = [False]
    torch.cuda.device = lambda dev: contextlib.nullcontext()            # no GPU here: the helper's device scope is a no-op
    torch.cuda.is_current_stream_capturing = lambda: capturing[0]
    clock = Clock()
    real_monotonic = time.monotonic
    time.monotonic = clock
    tmp = tempfile.mkdtemp(prefix="moet_sass_ab_")
    ctl, stats = os.path.join(tmp, "ab.json"), os.path.join(tmp, "stats")
    q = (torch.randn(100, 16, 512) / 10).to(torch.bfloat16)
    idx = torch.zeros(100, 1, 8, dtype=torch.int32)
    lens = torch.full((100,), 8, dtype=torch.int32)
    kv = torch.zeros(1, 128, 1, 528, dtype=torch.uint8)
    self_ = types.SimpleNamespace(scale=0.04, attn_sink=None)
    want = (q.float() * 0.5).to(torch.bfloat16)

    def env(ab="1", b="B.cubin", a="A.cubin"):
        for k in ENV:
            os.environ.pop(k, None)
        os.environ["VLLM_MOET_SPARSE_MLA_PREFILL_SASS"] = "1"
        os.environ["VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN"] = a
        if b is not None:
            os.environ["VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN_B"] = b
        if ab is not None:
            os.environ["VLLM_MOET_RUNTIME_AB"] = ab
        os.environ["VLLM_MOET_RUNTIME_AB_FILE"] = ctl
        os.environ["VLLM_MOET_RUNTIME_STATS_FILE"] = stats
        for p in (ctl, stats):
            if os.path.exists(p):
                os.remove(p)
        FakeLauncher.calls = []
        ns = fresh_helper()
        ns["_MoetSparseMlaSass"] = FakeLauncher
        return ns

    def unit(ns, advance=0.11, rows=100):
        """One launch (the A/B's unit); returns (taken, stderr text, output)."""
        clock.t += advance
        out = torch.zeros(rows, 16, 512, dtype=torch.bfloat16)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            taken = ns["_moet_prefill_sass_run"](self_, q[:rows], kv, idx[:rows], lens[:rows], None, None, None, out)
        return taken, err.getvalue(), out

    def cubins(calls):
        return [c for c, _ in calls]

    # 1. master switch off / no second cubin: nothing of the A/B runs
    for label, kw in (("RUNTIME_AB unset", dict(ab=None)), ("RUNTIME_AB=0", dict(ab="0")), ("no CUBIN_B", dict(b=None)),
                      ("CUBIN_B empty", dict(b=""))):
        ns = env(**kw)
        write_atomic(ctl, {KEY: 1})
        with StatCounter((ctl, stats)) as sc:
            res = [unit(ns) for _ in range(40)]
        ok = (all(t for t, _, _ in res) and all(e == "" for _, e, _ in res) and all(torch.equal(o, want) for _, _, o in res)
              and sc.n == 0 and cubins(FakeLauncher.calls) == ["A.cubin"] * 40 and "json" not in ns
              and ns["_MOET_SASS_AB"]["launchers_b"] == {} and ns["_MOET_SASS_AB"]["n"] == 0)
        check(f"off ({label}): 40 launches on the start-up cubin, 0 stats of the control / stats files, no output", ok,
              f"stats {sc.n}, calls {cubins(FakeLauncher.calls)[:4]}")

    # 2. on, no file: start-up state, one WARNING; first eager launch runs both cubins once and compares
    ns = env()
    t, e, o = unit(ns)
    check("on, no file: the first launch is taken, served by cubin 0, and launches both cubins once",
          t and torch.equal(o, want) and cubins(FakeLauncher.calls) == ["A.cubin", "B.cubin"], str(FakeLauncher.calls))
    check("on, no file: one WARNING 'missing or unreadable - the current state is kept until it appears'",
          e.count("WARNING: runtime A/B: control file") == 1 and "missing or unreadable - the current state is kept until it appears" in e
          and e.count(TAG) == 2 and "both cubins launched once on the first chunk (100 rows), outputs bit-identical" in e, e)
    t, e, o = unit(ns)
    check("on, no file: later launches use cubin 0 only, no further line", e == "" and cubins(FakeLauncher.calls)[2:] == ["A.cubin"], e)

    # 3. the 100-ms stat gate and the (mtime, size) re-read gate
    write_atomic(ctl, {KEY: 0})
    with StatCounter((ctl,)) as sc:
        for _ in range(50):
            unit(ns, advance=0.01)          # 0.5 s of launches 10 ms apart
    check("stat gate: 50 launches in 0.5 s -> at most 6 stats of the control file", 4 <= sc.n <= 6, f"{sc.n} stats")
    reads = [0]
    real_open = io.open

    def counting_open(p, *a, **k):
        if os.fspath(p) == ctl:
            reads[0] += 1
        return real_open(p, *a, **k)

    import builtins
    builtins.open = counting_open
    try:
        for _ in range(20):
            unit(ns)
        check("re-read gate: an unchanged file is not opened again", reads[0] == 0, f"{reads[0]} opens")
        # 4. key 1 -> cubin B from the unit that polls; exact change line; no line when the value is the current one
        write_atomic(ctl, {KEY: 1, "B12X_MOE_PREFILL_FAST_WS1": 0, "_seq": 7})
        FakeLauncher.calls = []
        t, e, o = unit(ns)
        check("key 1: the launch that polls runs cubin 1; one change line with the full key; other namespaces silent",
              cubins(FakeLauncher.calls) == ["B.cubin"] and torch.equal(o, want)
              and e == f"{TAG} None] runtime A/B: {KEY} 0 -> 1\n", repr(e))
        check("key 1: the file is opened once for the change", reads[0] == 1, f"{reads[0]} opens")
        write_atomic(ctl, {KEY: 1, "_seq": 8})
        t, e, o = unit(ns)
        check("same value written again: no line, still cubin 1", e == "" and cubins(FakeLauncher.calls)[-1] == "B.cubin", repr(e))
        # 4b. two atomic writes of the same size back to back (one clock tick apart at most): both are seen
        write_atomic(ctl, {KEY: 0, "_seq": 8})
        t, e, o = unit(ns)
        write_atomic(ctl, {KEY: 1, "_seq": 8})
        t2, e2, o2 = unit(ns)
        check("equal-size atomic rewrites in quick succession are each seen (the inode is part of the change test)",
              f"{KEY} 1 -> 0" in e and f"{KEY} 0 -> 1" in e2 and cubins(FakeLauncher.calls)[-2:] == ["A.cubin", "B.cubin"], repr(e + e2))
        # 5. a flip landing less than 100 ms after the previous poll waits for the gate (never mid-unit)
        write_atomic(ctl, {KEY: 0})
        t, e, o = unit(ns, advance=0.05)
        check("flip inside the 100-ms gate: this launch keeps cubin 1", e == "" and cubins(FakeLauncher.calls)[-1] == "B.cubin", repr(e))
        t, e, o = unit(ns, advance=0.06)
        check("... and the next launch past the gate applies it", cubins(FakeLauncher.calls)[-1] == "A.cubin" and f"{KEY} 1 -> 0" in e, repr(e))
        # 6. key absent ({} or only other keys) = the start-up cubin
        write_atomic(ctl, {KEY: 1}); unit(ns)
        write_atomic(ctl, {})
        t, e, o = unit(ns)
        check("{}: back to the start-up cubin with a change line", cubins(FakeLauncher.calls)[-1] == "A.cubin" and f"{KEY} 1 -> 0" in e, repr(e))
        write_atomic(ctl, {KEY: 1}); unit(ns)
        write_atomic(ctl, {"B12X_MOE_PREFILL_FAST_SUM_ITEMS": 0, "WOA_DEQUANT_FUSED": 1})
        t, e, o = unit(ns)
        check("only other modules' keys: back to the start-up cubin", cubins(FakeLauncher.calls)[-1] == "A.cubin" and e.count("\n") == 1
              and f"{KEY} 1 -> 0" in e, repr(e))
        # 7. file removed after a state was applied: the state stays (missing = keep), one WARNING
        write_atomic(ctl, {KEY: 1}); unit(ns)
        os.remove(ctl)
        t, e, o = unit(ns)
        t2, e2, o2 = unit(ns)
        check("file removed in state 1: state 1 stays, one WARNING", cubins(FakeLauncher.calls)[-2:] == ["B.cubin"] * 2
              and e.count("WARNING") == 1 and "missing or unreadable" in e and e2 == "", repr(e + e2))
        # 8. malformed / not an object: state kept, one WARNING per cause, cleared by a good file
        write_atomic(ctl, '{"SPARSE_MLA_PREFILL_SASS_CUBIN_SEL": ')
        t, e, o = unit(ns)
        write_atomic(ctl, "[1, 2]")
        t2, e2, o2 = unit(ns)
        check("malformed file: state kept (cubin 1), one WARNING for the cause", cubins(FakeLauncher.calls)[-2:] == ["B.cubin"] * 2
              and e.count("WARNING") == 1 and "malformed" in e and "the current state is kept" in e and e2 == "", repr(e + e2))
        write_atomic(ctl, {KEY: 1, "_ok": 1}); unit(ns)
        write_atomic(ctl, "not json")
        t, e, o = unit(ns)
        check("malformed again after a good file: the WARNING is printed again", e.count("WARNING") == 1 and "malformed" in e, repr(e))
        # 9. bad values keep the current effective value (one WARNING per key); an unknown key of the namespace is ignored
        for bad in (2, -1, "1", 1.0, True, None, [1]):
            ns2 = env()
            unit(ns2)
            write_atomic(ctl, {KEY: 1}); unit(ns2)
            write_atomic(ctl, {KEY: bad})
            t, e, o = unit(ns2)
            write_atomic(ctl, {KEY: bad, "_again": 1})
            t2, e2, o2 = unit(ns2)
            check(f"bad value {bad!r}: state 1 kept, one WARNING", cubins(FakeLauncher.calls)[-2:] == ["B.cubin"] * 2
                  and e.count("WARNING") == 1 and "is not the integer 0 / 1 - kept 1" in e and e2 == "", repr(e + e2))
        ns = env()
        unit(ns)
        write_atomic(ctl, {KEY: 1, "SPARSE_MLA_PREFILL_SASS_FOO": 1})
        t, e, o = unit(ns)
        check("unknown key of the namespace: ignored with one WARNING, the rest of the file applies",
              cubins(FakeLauncher.calls)[-1] == "B.cubin" and e.count("WARNING") == 1
              and "unknown key SPARSE_MLA_PREFILL_SASS_FOO of this module - ignored" in e and f"{KEY} 0 -> 1" in e, repr(e))
    finally:
        builtins.open = real_open

    # 10. counters and the on-demand line (every 16th launch looks at the stats file; one line per touch)
    ns = env()
    for _ in range(10):
        unit(ns)                                         # 10 launches on cubin 0 (the first also warms cubin 1)
    write_atomic(ctl, {KEY: 1})
    for _ in range(5):
        unit(ns, rows=80)                                # 5 on cubin 1 -> 15 launches
    open(stats, "w").close()
    t, e, o = unit(ns, rows=80)                          # the 16th launch
    line = f"{TAG} None] activation counters (on demand, call 16; calls / rows): cubin 0 10 / 1000, cubin 1 6 / 480; switches now {KEY}=1\n"
    check("on-demand line at the 16th launch: exact counters and 'switches now'", e == line, repr(e))
    res = "".join(unit(ns)[1] for _ in range(32))
    check("the same touch prints nothing again", res == "", repr(res))
    os.utime(stats, ns=(time.time_ns(), time.time_ns() + 1_000_000))
    res = "".join(unit(ns)[1] for _ in range(16))
    check("a new touch prints one line", res.count("activation counters") == 1 and "cubin 1 54 / " in res, repr(res))

    # 11. CUDA-graph capture: nothing but the selected launch is recorded; the two-cubin warm-up waits for an eager launch
    ns = env()
    capturing[0] = True
    t, e, o = unit(ns)
    check("first launch inside a capture: one launch of cubin 0, no warm-up", cubins(FakeLauncher.calls) == ["A.cubin"]
          and "both cubins" not in e, repr(e) + str(FakeLauncher.calls))
    capturing[0] = False
    t, e, o = unit(ns)
    check("first eager launch after it: both cubins launched once", cubins(FakeLauncher.calls)[1:] == ["A.cubin", "B.cubin"]
          and "outputs bit-identical" in e, repr(e))

    # 12. the second cubin does not load / differs: the start-up cubin keeps serving, the key is refused
    ns = env(b="NOLOAD.cubin")
    t, e, o = unit(ns)
    write_atomic(ctl, {KEY: 1})
    t2, e2, o2 = unit(ns)
    t3, e3, o3 = unit(ns)
    check("second cubin fails to load: route still taken on cubin 0, WARNING, key 1 refused once",
          t and t2 and torch.equal(o2, want) and cubins(FakeLauncher.calls) == ["A.cubin"] * 3 and "did not load" in e
          and "refused" in e2 and e3 == "" and not ns["_MOET_SASS_STATE"]["disabled"], repr(e + e2 + e3))
    ns = env(b="DIFF.cubin")
    write_atomic(ctl, {KEY: 1})
    t, e, o = unit(ns)
    t2, e2, o2 = unit(ns)
    check("second cubin's output differs on the first chunk: served by cubin 0, DIFFERS warning, state forced to 0",
          t and torch.equal(o, want) and "DIFFERS" in e and f"{KEY} 1 -> 0" in e and cubins(FakeLauncher.calls)[-1] == "A.cubin"
          and torch.equal(o2, want) and ns["_MOET_SASS_AB"]["sel"] == 0, repr(e + e2))
    write_atomic(ctl, {KEY: 1, "_retry": 1})
    t, e, o = unit(ns)
    check("... and a later key 1 is refused (one WARNING), cubin 0 stays", cubins(FakeLauncher.calls)[-1] == "A.cubin" and "refused" in e, repr(e))

    # 13. chunks the route does not take are not units of the A/B (no poll, no count)
    ns = env()
    with StatCounter((ctl, stats)) as sc:
        t, e, o = unit(ns, rows=64)
    check("<= 64 rows: declined, no stat, no counter", (not t) and sc.n == 0 and ns["_MOET_SASS_AB"]["n"] == 0)
    time.monotonic = real_monotonic


# ------------------------------------------------------------------------------------------------------------ GPU part
def gpu_tests(cubin_a: str, cubin_b: str, dumps: list[str]) -> None:
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.environ.get("NVFP4_KV_DIR", os.path.join(HERE, "..", "nvfp4_kv")))
    from test_flashinfer_dsv41_fp4_extra import our_insert
    from test_sparse_mla_sass_sm120 import D_QK, D_V, HEADS, make_wrapper, run_specB, to_interleaved

    stub_vllm_logger()
    logging.basicConfig(level=logging.INFO)
    dev = torch.device("cuda")
    tmp = tempfile.mkdtemp(prefix="moet_sass_ab_")
    ctl, stats = os.path.join(tmp, "ab.json"), os.path.join(tmp, "stats")

    def env(ab: bool, b: bool = True):
        for k in ENV:
            os.environ.pop(k, None)
        os.environ["VLLM_MOET_SPARSE_MLA_PREFILL_SASS"] = "1"
        os.environ["VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN"] = cubin_a
        if b:
            os.environ["VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN_B"] = cubin_b
        if ab:
            os.environ["VLLM_MOET_RUNTIME_AB"] = "1"
        os.environ["VLLM_MOET_RUNTIME_AB_FILE"] = ctl
        os.environ["VLLM_MOET_RUNTIME_STATS_FILE"] = stats
        for p in (ctl, stats):
            if os.path.exists(p):
                os.remove(p)
        return fresh_helper()

    # ---- chunks: one synthetic (vLLM-shaped: [rows, 1, topk] indices, a slice of a bigger output) + the real dumps
    torch.manual_seed(11)
    n = 1 << 14
    swa = to_interleaved(our_insert((torch.randn(n, D_QK, device=dev, dtype=torch.bfloat16) / 10).clamp(-1, 1), 128, 528).view(-1, 128, 1, 528), 128, 512, 16)
    ext = to_interleaved(our_insert((torch.randn(n, D_QK, device=dev, dtype=torch.bfloat16) / 10).clamp(-1, 1), 128, 288).view(-1, 128, 1, 288), 128, 256, 32)
    rows = 1500
    q = (torch.randn(rows, HEADS, D_QK, device=dev, dtype=torch.bfloat16) / 10).clamp(-1, 1)
    idx = torch.randint(0, n, (rows, 1, 128), device=dev, dtype=torch.int32)
    eidx = torch.randint(0, n, (rows, 1, 512), device=dev, dtype=torch.int32)
    lens = torch.randint(1, 129, (rows,), device=dev, dtype=torch.int32)
    elens = torch.randint(0, 513, (rows,), device=dev, dtype=torch.int32)
    sink = torch.randn(HEADS, device=dev, dtype=torch.float32)
    sm = D_QK ** -0.5
    ref, _ = run_specB(make_wrapper(True), q, swa, idx.view(rows, -1), lens, sink, sm, ext, eidx.view(rows, -1), elens)
    chunks = [dict(name=f"synthetic {rows} rows 128+512", self_=types.SimpleNamespace(scale=sm, attn_sink=sink),
                   args=(q, swa, idx, lens, ext, eidx, elens), ref=ref, rows=rows)]
    for p in dumps:
        if p.endswith(".ref.pt"):
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        has_extra = "extra_cache" in d
        dq = d["q"].to(dev)
        r = dq.shape[0]
        dsink = None if d["attn_sink"] is None else d["attn_sink"].to(dev)
        sidx, slens = d["swa_indices"].to(dev), d["swa_lens"].to(dev)
        ex = (d["extra_cache"].to(dev), d["extra_indices"].to(dev), d["extra_lens"].to(dev)) if has_extra else (None, None, None)
        dref = d["output"].to(dev)                      # the output of the served kernel at dump time (== spec B, checked there)
        chunks.append(dict(name=f"dump {os.path.basename(p)} ({r} rows)", self_=types.SimpleNamespace(scale=d["sm_scale"], attn_sink=dsink),
                           args=(dq, d["swa_cache"].to(dev), sidx.view(r, 1, -1), slens, ex[0], None if ex[1] is None else ex[1].view(r, 1, -1), ex[2]),
                           ref=dref, rows=r))
    print(f"[gpu] {len(chunks)} chunks ({sum(c['rows'] for c in chunks)} rows); A = {cubin_a}; B = {cubin_b}", flush=True)

    def launch_all(ns, expect_sel=None):
        """Every chunk once; returns (#taken, #equal to the reference, stderr text)."""
        err = io.StringIO()
        taken = equal = 0
        for c in chunks:
            out = torch.zeros(c["rows"], HEADS, D_V, device=dev, dtype=torch.bfloat16)
            with contextlib.redirect_stderr(err):
                t = ns["_moet_prefill_sass_run"](c["self_"], *c["args"], out)
            torch.cuda.synchronize()
            taken += bool(t)
            equal += bool(torch.equal(out, c["ref"]))
        return taken, equal, err.getvalue()

    nc = len(chunks)
    # ---- 1. master switch off (and: on but no second cubin): the control file is never looked at, bits as before
    for label, kw in (("RUNTIME_AB unset", dict(ab=False)), ("RUNTIME_AB=1 without CUBIN_B", dict(ab=True, b=False))):
        ns = env(**kw)
        write_atomic(ctl, {KEY: 1})
        with StatCounter((ctl, stats)) as sc:
            t, eq, e = launch_all(ns)
            t2, eq2, e2 = launch_all(ns)
        check(f"[gpu] off ({label}): {2 * nc} launches identical to the reference, 0 stats, no A/B line",
              t == nc and eq == nc and t2 == nc and eq2 == nc and sc.n == 0 and "runtime A/B" not in e + e2
              and ns["_MOET_SASS_AB"]["launchers_b"] == {}, f"taken {t}/{t2} equal {eq}/{eq2} stats {sc.n}")
    # ---- 2. every state and transition, bits on every launch, counters exact
    ns = env(ab=True)
    seq = [("no file", None, 0), ("1", {KEY: 1}, 1), ("0", {KEY: 0}, 0), ("1 again", {KEY: 1}, 1), ("{}", {}, 0),
           ("1 + other keys", {KEY: 1, "B12X_MOE_PREFILL_FAST_WS1": 9, "_seq": 3}, 1), ("file removed (state kept)", "rm", 1),
           ("0 after the removal", {KEY: 0}, 0), ("bad value 2 (state kept)", {KEY: 2}, 0), ("1 last", {KEY: 1}, 1), ("{} at the end", {}, 0)]
    exp_calls, exp_rows = [0, 0], [0, 0]
    cur = 0
    all_ok = True
    for label, state, want_sel in seq:
        if state == "rm":
            os.remove(ctl)
        elif state is not None:
            write_atomic(ctl, state)
        time.sleep(0.12)                                  # past the 100-ms stat gate: the next launch polls
        t, eq, e = launch_all(ns)
        t2, eq2, e2 = launch_all(ns)
        exp_calls[want_sel] += 2 * nc
        exp_rows[want_sel] += 2 * sum(c["rows"] for c in chunks)
        change = f"runtime A/B: {KEY} {cur} -> {want_sel}"
        ok = (t == nc and eq == nc and t2 == nc and eq2 == nc and ns["_MOET_SASS_AB"]["sel"] == want_sel
              and ((e + e2).count(change) == 1 if want_sel != cur else f"{KEY} " not in (e + e2).replace("switches now", "")))
        all_ok &= ok
        check(f"[gpu] state '{label}': {2 * nc} launches identical to the reference on cubin {want_sel}"
              f"{', one change line' if want_sel != cur else ', no change line'}", ok, f"taken {t}/{t2} equal {eq}/{eq2} sel "
              f"{ns['_MOET_SASS_AB']['sel']} lines {e + e2!r}"[:600])
        cur = want_sel
    st = ns["_MOET_SASS_AB"]
    check("[gpu] counters exact over the whole sequence (launches / rows per cubin)", st["calls"] == exp_calls and st["rows"] == exp_rows,
          f"{st['calls']} {st['rows']} vs {exp_calls} {exp_rows}")
    check("[gpu] both cubins loaded once per device, nothing loaded at a flip", len(ns["_MOET_SASS_STATE"]["launchers"]) == 1
          and len(st["launchers_b"]) == 1 and st["b_ok"] and not ns["_MOET_SASS_STATE"]["disabled"])
    # ---- 3. the on-demand line
    open(stats, "w").close()
    lines = ""
    for _ in range(3):
        lines += launch_all(ns)[2]
    check("[gpu] touch of the stats file: one on-demand line with 'switches now'", lines.count("activation counters (on demand") == 1
          and f"switches now {KEY}=0" in lines, repr(lines)[:400])
    # ---- 4. a launch captured into a CUDA graph keeps its cubin; eager launches follow the key
    ns = env(ab=True)
    c = chunks[0]
    launch_all(ns)                                        # eager first: both cubins loaded and launched once
    out_g = torch.zeros(c["rows"], HEADS, D_V, device=dev, dtype=torch.bfloat16)
    g = torch.cuda.CUDAGraph()
    before = list(ns["_MOET_SASS_AB"]["calls"])
    with torch.cuda.graph(g):
        ns["_moet_prefill_sass_run"](c["self_"], *c["args"], out_g)
    after = list(ns["_MOET_SASS_AB"]["calls"])
    write_atomic(ctl, {KEY: 1})
    time.sleep(0.12)
    t, eq, e = launch_all(ns)                             # eager launches flip to cubin 1
    out_g.zero_()
    g.replay()
    torch.cuda.synchronize()
    check("[gpu] CUDA graph: captured in state 0 (one launch recorded on cubin 0), replayed in state 1 = identical to the reference",
          after[0] - before[0] == 1 and after[1] == before[1] and torch.equal(out_g, c["ref"]) and eq == nc
          and ns["_MOET_SASS_AB"]["sel"] == 1, f"calls {before} -> {after}, eager equal {eq}/{nc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--cubin")
    ap.add_argument("--cubin-b")
    ap.add_argument("dumps", nargs="*")
    args = ap.parse_args()
    saved = {k: os.environ.get(k) for k in ENV}
    try:
        if args.gpu:
            if not (args.cubin and args.cubin_b):
                ap.error("--gpu needs --cubin and --cubin-b")
            gpu_tests(args.cubin, args.cubin_b, args.dumps)
        else:
            cpu_tests()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"{'PASS' if not bad else 'FAIL'} runtime A/B ({'GPU' if args.gpu else 'CPU'}): {len(RESULTS) - len(bad)}/{len(RESULTS)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
